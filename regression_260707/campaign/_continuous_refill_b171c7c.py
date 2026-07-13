"""Maintain exactly 300 logical MFT active tasks with pinned SHA-b171 work.

The default command is a scheduler-read-free static audit.  ``--execute`` is
the only mutation path.  Each execute cycle re-reads the logical MFT project
inside the shared campaign mutation lock and asks :mod:`feeder` to fill only
the current ``queued + attaching + running`` deficit to 300.  Existing SHA754
work is capacity, not a cancellation target; IPMSM belongs to another project.

The first 300 candidate identities are checked against the root-reviewed b171
plan.  The durable cursor starts immediately before that plan (cursor 2795,
raw 2794, serial 17611) and continues deterministically beyond it without
reusing names, parameter payloads, or dedupe keys.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, REPO_ROOT, VERIFY_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import _adopted_refill_sha688c6f9 as durable
import _rolling_recycle_prebinding_260712 as rolling_recycle
import _submit_production300_b171c7c as production
import feeder
import pinned_pilot
import rapid_campaign
import scheduler_client
from training.checkpoint_contract import (
    checkpoint_status_revision_identity_matches,
)


SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PLAN_SHA256 = "b24e2a9b00caa22bbec8793f4dbd99de51362fac87f9e9509358610abe9982d0"
SEED = 260710
LEGACY_TARGET_ACTIVE = 250
PREVIOUS_TARGET_ACTIVE = 400
TARGET_ACTIVE = 300
TARGET_ACTIVE_MIN = 1
TARGET_ACTIVE_MAX = 300
TARGET_POLICY_FIXED = "fixed_pool300_v1"
TARGET_POLICY_DYNAMIC = "scheduler_project_max_active_tasks_v1"
TARGET_400_TRANSITION_CYCLE = 334
TARGET_300_TRANSITION_CYCLE = 336
TARGET_STRICT_ROWS = 3_000
MAX_SAMPLES = 12_000
CPUS = 4
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
EVIDENCE_MODE = "dynamic_project_cap_v1"
REFILL_ACTION = f"refill_{TARGET_ACTIVE}"
PRODUCTION_HEALTH_COHORT_CUTOFF = "2026-07-12T06:52:07Z"
_PRODUCTION_HEALTH_COHORT_CUTOFF_AT = datetime(
    2026, 7, 12, 6, 52, 7, tzinfo=timezone.utc)
_SCHEDULER_CPU_POLICY_CUTOVER_AT = datetime(
    2026, 7, 12, 11, 0, 54, tzinfo=timezone.utc)
INITIAL_SERIAL = 17_611
INITIAL_CURSOR = 2_795
INITIAL_RAW_INDEX = 2_794
PLAN_TASK_COUNT = 300
PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
ACTIVE_STATUSES = ("queued", "attaching", "running")
STATE_PATH = HERE / "continuous_refill_b171c7c_state.json"
FEEDER_STATE_PATH = HERE / "continuous_refill_b171c7c_feeder_state.json"
CYCLE_ROOT = HERE / "pilot_manifests" / "continuous-refill-sb171c7c-le6b9b9d"
TARGET_TRANSITION_ROOT = CYCLE_ROOT / "target-transitions"
STRICT_STATUS_PATH = REGRESSION_ROOT / "training" / "strict_data_status.json"
LOCAL_RECOVERY_SHA = "7873ddddcf7ac7412d14c9e3ae216ed73b82fffe"
LOCAL_RECOVERY_EVIDENCE_PATH = (
    HERE / "evidence" / "b171_local_recovery_7873.json")
LOCAL_RECOVERY_EVIDENCE_SHA256 = (
    "849e0b7c02a42313ddb1567c8458e290492a0d91440573fd4ec3714f8624f7b7")
LOCAL_RECOVERY_SOURCE_SHA256 = (
    "a873a53d49f9678684599300a0af7a4ac9a864f494251b5ca9dd0cca5b1cc7ef")
REJECTED_TASK_ID = 28_101
REJECTED_TASK_NAME = "mft-camp-sb171c7c-le6b9b9d-17612"
REJECTED_TASK_DEDUPE = (
    "mft-al:mft-camp-sb171c7c-le6b9b9d-17612:"
    f"{SOLVER}:{LIBRARY}:ae6d03b35d7cfefc")
REJECTED_EXPECTED_DEDUPE = (
    "mft-al:mft-camp-sb171c7c-le6b9b9d-17612:"
    f"{SOLVER}:{LIBRARY}:6f2e0101a9a8878c")
REJECTED_CANCELLATION_EVIDENCE_PATH = (
    HERE / "evidence" / "b171_rejected_submission_28101.json")
REJECTED_CANCELLATION_EVIDENCE_SHA256 = (
    "739a7fa81ccd4d5a145cf06d0c095c4d51f4eee3846860a1faa6f176a75c2bd4")
REJECTED_CANCELLATION_SHA256 = (
    "ef28db0da228478b79a6919cec6deadc456f425012559ebe89e2169776fa0ecd")
EXTERNAL_STALE_PIN_CANCELLED_IDS = (
    28_138, 28_141, 28_144, 28_147, 28_150, 28_153, 28_170,
    28_174, 28_175, 28_179, 28_180, 28_184, 28_185, 28_187,
    28_188, 28_190, 28_191, 28_193, 28_194, 28_196, 28_197,
)
# Canonical SHA256 of the exact live identity fields captured after the single
# 2026-07-12 10:39:55Z external cancellation request. This is deliberately an
# incident-specific allow-list; unrelated cancelled work must still fail the
# production-health contract.
EXTERNAL_STALE_PIN_CANCELLATION_IDENTITY_SHA256 = (
    "fe7ef7c14ea47c5416b155d4b9069571d85048571970ea11056b2aab78327987")
_EXTERNAL_CANCELLATION_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "project", "status", "account_name",
    "created_at", "attached_at", "launch_started_at", "started_at",
    "finished_at",
)
REMOTE_STEP_CANCELLED_IDS = (
    28_417, 28_423, 28_424, 28_425, 28_427, 28_433,
    28_437, 28_444, 28_446, 28_453, 28_456, 28_457,
)
# Canonical SHA256 of the exact twelve-task cancellation burst observed from
# 2026-07-12 11:12:12Z through 11:12:17Z. The tasks had no exit code and their
# eight parent allocations stayed active, so this operational cancellation is
# not a solver-validity outcome. The exact incident is excluded; generic or
# future cancellations remain production-health failures.
REMOTE_STEP_CANCELLATION_IDENTITY_SHA256 = (
    "9f70d6b199f054cbd0f663679d02d5ca63668301ead0f518971e471f44c8b6aa")
_REMOTE_STEP_CANCELLATION_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "project", "status", "account_name",
    "requested_account_name", "allocation_id", "slurm_job_id",
    "allocation_node_name", "created_at", "attached_at", "launch_started_at",
    "started_at", "finished_at", "exit_code", "failure_message",
)
OPERATOR_CANCELLED_STALE_PREPOLICY_IDS = (28_746, 28_747, 28_748)
OPERATOR_CANCELLED_STALE_PREPOLICY_IDENTITY_SHA256 = (
    "2b0b5283c895bc3fb19c58dc7b83cf3adaf66cc58c8216a5340e966c629ed883")
_OPERATOR_CANCELLED_STALE_PREPOLICY_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "project", "status", "account_name",
    "requested_account_name", "allocation_id", "slurm_job_id",
    "allocation_node_name", "created_at", "attached_at", "launch_started_at",
    "started_at", "finished_at", "exit_code", "failure_message",
)
_OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT = {
    "source_address": "192.168.56.1",
    "request_path": "/api/tasks/cancel?task_ids=28748,28747,28746",
    "web_log": {
        "path": r"C:\Users\peets\NEC\slurm_scheduler\logs\web.log",
        "line": 972135,
        "local_time": "2026-07-12 23:24:22",
    },
    "recovery_held_events": [
        {
            "event_id": 30008, "task_id": 28746,
            "created_at": "2026-07-12 11:00:54",
            "message": "task attach launch may have started; claim preserved "
                       "to prevent duplicate execution",
        },
        {
            "event_id": 30007, "task_id": 28747,
            "created_at": "2026-07-12 11:00:54",
            "message": "task attach launch may have started; claim preserved "
                       "to prevent duplicate execution",
        },
        {
            "event_id": 30006, "task_id": 28748,
            "created_at": "2026-07-12 11:00:54",
            "message": "task attach launch may have started; claim preserved "
                       "to prevent duplicate execution",
        },
    ],
    "operator_confirmation": "direct_web_ui_cancel_of_abnormal_tasks",
}
OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT_SHA256 = (
    "798e93648290c7ee8ec111857fadb3cc0e4169698bb4538a3f9b59fb1d8f93db")
RESOLVED_SCHEDULER_PARENT_CANCEL_IDS = (
    29_026, 29_037, 29_038, 29_047, 29_054, 29_055, 29_061,
    29_069, 29_075, 29_095, 29_096, 29_100, 29_106, 29_117,
    29_187, 29_188, 29_191, 29_193, 29_194, 29_197, 29_264,
    29_266, 29_270, 29_273, 29_290, 29_295,
)
_RESOLVED_SCHEDULER_PARENT_CANCEL_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "allocation_id", "slurm_job_id",
    "allocation_node_name", "started_at", "finished_at", "exit_code",
)
RESOLVED_SCHEDULER_PARENT_CANCEL_IDENTITY_SHA256 = (
    "8fc13554eaa88ff5e16a211f8077774e4c01417114e08859953e72035279b875")
_RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT = {
    "incident": {
        "parent_cancelled_at_utc": "2026-07-12 14:39:33",
        "child_steps_cancelled_at_kst": "2026-07-12 23:39:34-36",
        "cancel_uid": 1641,
        "cancel_account": "dw16",
        "parents": [
            {
                "allocation_id": 8345, "slurm_job_id": "732195",
                "node": "n014", "state": "CANCELLED by 1641",
                "reason": None, "time_limit": "2-00:00:00",
                "elapsed": "02:42:01",
            },
            {
                "allocation_id": 8346, "slurm_job_id": "732196",
                "node": "n007", "state": "CANCELLED by 1641",
                "reason": None, "time_limit": "2-00:00:00",
                "elapsed": "02:42:01",
            },
            {
                "allocation_id": 8347, "slurm_job_id": "732197",
                "node": "n003", "state": "CANCELLED by 1641",
                "reason": None, "time_limit": "2-00:00:00",
                "elapsed": "02:42:01",
            },
            {
                "allocation_id": 8349, "slurm_job_id": "732200",
                "node": "n002", "state": "CANCELLED by 1641",
                "reason": None, "time_limit": "2-00:00:00",
                "elapsed": "02:41:00",
            },
        ],
    },
    "root_cause": (
        "Demand scale-in could call parent scancel from a stale "
        "WARM/no-queued-demand view without an allocation-scoped live-claim "
        "check. active_task_ids used a global latest-5000 scan, and "
        "close_allocation had neither a DB claim guard nor Slurm child-step "
        "guard, so DB/Slurm disagreement could cancel a parent allocation "
        "that still owned numeric task steps."),
    "fix_invariant": [
        "automatic close shares the assignment lock",
        "unlimited allocation-scoped ATTACHING/RUNNING claim query",
        "squeue --steps numeric child check",
        "live-step probe error fails closed",
        "stale WARM with live ownership reconciles to ACTIVE",
        "only explicit force bypasses the guards",
    ],
    "scheduler_runtime_sha256": {
        "slurm_scheduler/app.py": (
            "81a4ba2aeec6d6e1e72412a6fab3eb8510b3adcdc3a9203c99ecbb3e1ba20bae"),
        "slurm_scheduler/db.py": (
            "5b79ab562932a95b705ebb0698c5a47e8702ded57dfb05dbcc56fff2b953bfc5"),
        "slurm_scheduler/scheduler.py": (
            "8117321683f168fc01e9ee8c5be9f8aa80a4fb591dd837616fefa582e02e2be4"),
        "slurm_scheduler/slurm.py": (
            "610c19e58b495a8d8aa056628f500971bbe529bd1b4b05e5667612435f01a367"),
        "tests/test_core.py": (
            "5d1430415725b55d08c244c6b77e12eadd96ec1d67c810e34d84440c624f0e57"),
    },
    "dynamic_project_cap_route": {
        "method": "PATCH",
        "path": "/api/projects/{name}/max-active-tasks",
        "bounds_inclusive": [1, 300],
        "body_fields": ["max_active_tasks"],
        "cap_only_mutation": True,
        "exact_live_count_readback": True,
        "project_api_tests_passed": 13,
    },
    "deployment_evidence": {
        "old_child_pid": 40544,
        "new_child_pid": 193480,
        "watchdog_parent_pid": 84024,
        "first_full_tick_utc": "2026-07-12 15:25:42",
        "first_full_tick_failures": 0,
        "first_full_tick_stalled": False,
        "live_allocation_ids_pre_post_equal": True,
        "live_allocation_count": 17,
        "active_pre_post": [140, 128],
        "active_drop_events": "task_completed_only",
        "allocation_close_fail_cancel_events": 0,
        "mft_project_cap_change": [300, 400],
        "mft_project_cap_readback_utc": "2026-07-12 15:29:08",
        "ipmsm_project_cap": 50,
    },
    "regression_tests": [
        "SlurmParsingTests.test_live_allocation_task_steps_keeps_only_numeric_children",
        "SlurmParsingTests.test_live_allocation_task_steps_probe_failure_raises",
        "SchedulerTests.test_request_close_allocation_force_fails_active_tasks",
        "SchedulerTests.test_warm_demand_allocation_keeps_live_db_claims_then_closes_after_release",
        "SchedulerTests.test_warm_demand_allocation_keeps_untracked_live_slurm_step",
        "SchedulerTests.test_warm_demand_allocation_close_probe_failure_is_fail_closed",
        "ProjectApiTests.test_project_cap_patch_accepts_inclusive_bounds",
        "ProjectApiTests.test_project_cap_patch_changes_only_cap_and_returns_exact_live_counts",
        "ProjectApiTests.test_project_cap_patch_rejects_non_cap_bodies_and_out_of_range_values",
        "ProjectApiTests.test_project_cap_patch_returns_not_found_without_creating_project",
    ],
    "unrelated_user_cancel": {
        "task_ids": [28746, 28747, 28748],
        "allocation_id": 8019,
        "account": "r1jae262",
    },
}
RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT_SHA256 = (
    "c9096b2dbf0097295dad644bd099d8b5f85b2ccdbb5cc5fdd4d9032c53968834")
SEALED_OLD_TIMEOUT_CONTRACT_IDS = (
    28_749, 28_773, 28_789, 28_819, 28_830, 28_842,
)
_SEALED_OLD_TIMEOUT_CONTRACT_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "allocation_id", "slurm_job_id",
    "allocation_node_name", "started_at", "finished_at", "exit_code",
)
SEALED_OLD_TIMEOUT_CONTRACT_IDENTITY_SHA256 = (
    "369f5e3a6891fd3653338f01ce2ff2682eb82c36455aeeb548bc22266d679e22")
_SEALED_OLD_TIMEOUT_CONTRACT_AUDIT = {
    "classification": "sealed_old_timeout_contract_incident",
    "timeout_seconds": 14_400,
    "exit_code": 124,
    "allocation_policy_cutover_utc": "2026-07-12 11:00:54",
    "allocations": [
        {"allocation_id": 8034, "slurm_job_id": "731400", "node": "n046",
         "account": "jji0930", "parent_started_at_utc": None},
        {"allocation_id": 8293, "slurm_job_id": "731975", "node": "n116",
         "account": "dhj02", "parent_started_at_utc": "2026-07-12 07:14:38"},
        {"allocation_id": 8004, "slurm_job_id": "731339", "node": "n041",
         "account": "jji0930", "parent_started_at_utc": "2026-07-11 17:58:06"},
        {"allocation_id": 8212, "slurm_job_id": "731735", "node": "n090",
         "account": "jji0930", "parent_started_at_utc": "2026-07-12 01:44:03"},
        {"allocation_id": 8311, "slurm_job_id": "732022", "node": "n111",
         "account": "jji0930", "parent_started_at_utc": "2026-07-12 08:11:40"},
        {"allocation_id": 8289, "slurm_job_id": "731970", "node": "n082",
         "account": "jji0930", "parent_started_at_utc": "2026-07-12 07:07:37"},
    ],
    "scope": (
        "exact immutable task and old allocation identities only; future "
        "exit124 remains fail-closed"),
}
SEALED_OLD_TIMEOUT_CONTRACT_AUDIT_SHA256 = (
    "18a80c16b4c6ccdbce6c59dde5ce03546a5b89eea73668a1adbf517534d42988")
TARGET300_ROLLBACK_AUDIT_PATH = (
    HERE / "pilot_manifests"
    / "target-rollback-pool400-to300-20260713.json")
TARGET300_ROLLBACK_AUDIT_FILE_SHA256 = (
    "bad9f6e7d7e9579bf9cac8a4b81ca3790b17c3272d132696d115d3df9965a237")
TARGET300_ROLLBACK_CANCELLED_IDS = tuple(range(29_572, 29_649))
TARGET300_ROLLBACK_CANCELLED_IDENTITY_SHA256 = (
    "1aecaf7dad32f84f842f0546d622e7fae58e5af5ca774abb431d702de288ff48")
TARGET300_ROLLBACK_ELIGIBLE_IDENTITY_SHA256 = (
    "c8e8a03de668b487f3dab5f6a7e992feac8619c92a135d7056096a4f53b4d5bd")
_TARGET300_ROLLBACK_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "project", "status", "created_at",
    "attached_at", "launch_started_at", "started_at", "finished_at",
    "allocation_id", "assigned_allocation", "slurm_job_id",
    "allocation_node_name", "account_name", "requested_account_name",
    "exit_code", "failure_message", "cpus", "memory_mb",
    "timeout_seconds", "scheduling_profile", "required_capability",
    "env_profile", "gpus",
)



def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trace(stage: str) -> None:
    """Emit a flushed controller-stage marker for mounted-drive diagnosis."""
    print(f"[continuous-refill] {_now()} {stage}", file=sys.stderr, flush=True)


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _candidate_contract(params: dict, label: str) -> None:
    try:
        turns = int(params["N1_main"]) + int(params["N1_side"])
        cw1 = float(params["cw1"])
        plates = (float(params["wcp_t"]), float(params["core_plate_t"]))
        pads = (float(params["wcp_pad_t"]), float(params["core_plate_pad_t"]))
        wcp_len = float(params["wcp_len_x"])
        n2_main = int(params["N2_main"])
        nwl2_main = n2_main * float(params["cw2"]) + max(n2_main - 1, 0) * float(params["gap2"])
        sl2_main_x = 2.0 * float(params["l1"]) + 2.0 * float(params["cc_w2c_space_x"])
        ref = sl2_main_x + 2.0 * nwl2_main + 2.0 * float(params["w2c_w1c_space_x"])
        if int(params["round_corner"]) != 0:
            ref -= 2.0 * float(params["corner_radius"])
        pct = 100.0 * wcp_len / ref
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise RuntimeError(f"{label} candidate contract is unreadable") from exc
    if not (1 <= turns <= 8 and 0 < cw1 <= 10.0):
        raise RuntimeError(f"{label} primary winding cap drifted")
    if not all(10.0 <= value <= 30.0 for value in plates):
        raise RuntimeError(f"{label} cold-plate thickness drifted")
    if pads != (2.0, 2.0):
        raise RuntimeError(f"{label} thermal-pad thickness drifted")
    if not 20.0 - 0.05 <= pct <= 80.0 + 0.05:
        raise RuntimeError(
            f"{label} winding plate length drifted: {wcp_len}mm/{ref}mm={pct}%")


def _local_recovery_evidence() -> dict:
    """Authenticate the reviewed local Icepak recovery evidence.

    The original solver log lived in a historical mounted worktree.  Static
    audit must be reproducible from a clean clone and must never fall back to
    that mutable/mounted path, so this release seals the reviewed facts and
    original byte digest in a repository-owned immutable artifact.
    """
    try:
        payload = json.loads(
            LOCAL_RECOVERY_EVIDENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"sealed local recovery evidence is unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("sealed local recovery evidence is not an object")
    unsigned = dict(payload)
    stored_seal = unsigned.pop("evidence_sha256", None)
    computed_seal = _sha(unsigned)
    source_value = payload.get("source")
    reviewed_value = payload.get("reviewed_contract")
    provenance_value = payload.get("provenance")
    source = source_value if isinstance(source_value, dict) else {}
    result = reviewed_value if isinstance(reviewed_value, dict) else {}
    provenance = provenance_value if isinstance(provenance_value, dict) else {}
    checks = {
        "schema": payload.get("schema")
            == "b171-local-recovery-evidence-v1",
        "seal": stored_seal == LOCAL_RECOVERY_EVIDENCE_SHA256
            and computed_seal == LOCAL_RECOVERY_EVIDENCE_SHA256,
        "source_object": isinstance(source_value, dict),
        "source_kind": source.get("kind") == "reviewed_local_solver_log",
        "source_path": source.get("original_path")
            == (
                r"Y:\git\MFT_1MW_2026_local_recovery_7873"
                r"\regression_260707\logs\local_recovery_7873\stdout.log"),
        "source_sha": source.get("sha256") == LOCAL_RECOVERY_SOURCE_SHA256,
        "source_rows": source.get("result_json_rows") == 1,
        "reviewed_object": isinstance(reviewed_value, dict),
        "strict_valid": result.get("strict_valid") is True,
        "solver": result.get("solver_revision") == LOCAL_RECOVERY_SHA,
        "solver_clean": result.get("solver_clean") is True,
        "library": result.get("library_revision") == LIBRARY,
        "library_clean": result.get("library_clean") is True,
        "em_valid": result.get("result_valid_em") == 1,
        "thermal_valid": result.get("result_valid_thermal") == 1,
        "thermal_solved": result.get("thermal_solved") == 1,
        "thermal_extracted": result.get("thermal_extraction_complete") == 1,
        "missing_zero": result.get("thermal_required_missing_count") == 0,
        "one_attempt": result.get("thermal_solve_attempts") == 1,
        "analyze_ok": result.get("thermal_analyze_call_ok") == 1,
        "dispatch_success": result.get("thermal_dispatch_status") == "success",
        "solution_available": result.get("thermal_solution_data_available") == 1,
        "converged": result.get("thermal_converged") == 1,
        "iterations": result.get("thermal_iterations") == 143,
        "monitor": result.get("thermal_monitor_present") is True,
        "forensic_one_attempt": result.get("forensic_attempt_count") == 1,
        "forensic_schema":
            result.get("forensic_schema") == "thermal-dispatch-forensic-v1",
        "forensic_dispatch":
            result.get("forensic_dispatch_status") == "success",
        "forensic_entrypoint":
            result.get("forensic_design") == "icepak_thermal"
            and result.get("forensic_design_type") == "Icepak"
            and result.get("forensic_setups") == ["ThermalSetup"]
            and result.get("forensic_wrapper_setups") == ["ThermalSetup"],
        "forensic_fresh_monitor":
            result.get("forensic_monitor_reason") == "converged"
            and result.get("forensic_monitor_identity_matched") is True,
        "forensic_monitor": result.get("forensic_final_converged") == 1,
        "provenance": isinstance(provenance_value, dict)
            and provenance.get("source_bytes_republished") is False
            and isinstance(provenance.get("review_scope"), str)
            and bool(provenance["review_scope"].strip()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"exact local recovery result contract drifted: {checks}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCAL_RECOVERY_SHA, SOLVER],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, timeout=15, check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            "local recovery SHA is not an ancestor of b171: "
            + ancestry.stderr.strip())
    return {
        "evidence": str(LOCAL_RECOVERY_EVIDENCE_PATH),
        "evidence_sha256": computed_seal,
        "source_log": source["original_path"],
        "source_log_sha256": source["sha256"],
        "solver_revision": LOCAL_RECOVERY_SHA,
        "library_revision": LIBRARY,
        "b171_descendant": True,
        "thermal_dispatch_status": result["thermal_dispatch_status"],
        "thermal_iterations": int(result["thermal_iterations"]),
        "result_valid_em": int(result["result_valid_em"]),
        "result_valid_thermal": int(result["result_valid_thermal"]),
    }


def _rejected_submission_seal() -> dict:
    """Authenticate reviewed never-started cancellation evidence.

    The original cycle artifact is represented by its reviewed canonical seal;
    clean-clone static audit never consults a runtime campaign directory.
    """
    try:
        payload = json.loads(REJECTED_CANCELLATION_EVIDENCE_PATH.read_text(
            encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError("rejected submission cancellation evidence is unavailable") from exc
    unsigned = dict(payload)
    stored = unsigned.pop("evidence_sha256", None)
    source_value = payload.get("source")
    reviewed_value = payload.get("reviewed_contract")
    provenance_value = payload.get("provenance")
    source = source_value if isinstance(source_value, dict) else {}
    reviewed = reviewed_value if isinstance(reviewed_value, dict) else {}
    provenance = provenance_value if isinstance(provenance_value, dict) else {}
    checks = {
        "seal": stored == REJECTED_CANCELLATION_EVIDENCE_SHA256
            and _sha(unsigned) == REJECTED_CANCELLATION_EVIDENCE_SHA256,
        "schema": payload.get("schema")
            == "b171-rejected-submission-evidence-v1",
        "source": isinstance(source_value, dict)
            and source.get("kind") == "sealed_controller_cancellation_artifact"
            and source.get("original_relative_path")
                == (
                    "pilot_manifests/continuous-refill-sb171c7c-le6b9b9d/"
                    "cycle-000001-cancellation.json")
            and source.get("cancellation_sha256")
                == REJECTED_CANCELLATION_SHA256,
        "reviewed": isinstance(reviewed_value, dict),
        "cycle": reviewed.get("cycle_serial") == 1,
        "task_id": reviewed.get("task_id") == REJECTED_TASK_ID,
        "name": reviewed.get("name") == REJECTED_TASK_NAME,
        "project": reviewed.get("project") == scheduler_client.MFT_PROJECT,
        "dedupe": reviewed.get("actual_dedupe_key") == REJECTED_TASK_DEDUPE,
        "expected_dedupe": reviewed.get("expected_plan_dedupe_key")
            == REJECTED_EXPECTED_DEDUPE,
        "status_before": reviewed.get("status_before") == "queued",
        "status_after": reviewed.get("status_after") == "cancelled",
        "never_started": reviewed.get("attached_at_before") is None
            and reviewed.get("started_at_before") is None,
        "ack": reviewed.get("acknowledgement")
            == {"cancelled": [REJECTED_TASK_ID], "count": 1},
        "provenance": isinstance(provenance_value, dict)
            and provenance.get("source_bytes_republished") is False
            and isinstance(provenance.get("review_scope"), str)
            and bool(provenance["review_scope"].strip()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"rejected submission cancellation evidence drifted: {checks}")
    return {
        **reviewed,
        "cancellation_sha256": source["cancellation_sha256"],
        "evidence_sha256": stored,
    }


def _rejected_submission_evidence() -> dict:
    """Add live cancelled/never-started proof to the exact local seal."""
    payload = _rejected_submission_seal()
    live = production._task_detail(REJECTED_TASK_ID)
    live_checks = {
        "id": live.get("id", live.get("task_id")) == REJECTED_TASK_ID,
        "name": live.get("name") == REJECTED_TASK_NAME,
        "dedupe": live.get("dedupe_key") == REJECTED_TASK_DEDUPE,
        "project": live.get("project") == scheduler_client.MFT_PROJECT,
        "cancelled": str(live.get("status") or "").lower() == "cancelled",
        "never_started": live.get("attached_at") is None
            and live.get("started_at") is None
            and live.get("allocation_id") is None,
    }
    if not all(live_checks.values()):
        raise RuntimeError(f"rejected submission live identity drifted: {live_checks}")
    return {
        "task_id": REJECTED_TASK_ID,
        "name": REJECTED_TASK_NAME,
        "dedupe_key": REJECTED_TASK_DEDUPE,
        "status": "cancelled",
        "never_started": True,
        "cancellation_sha256": REJECTED_CANCELLATION_SHA256,
    }


def _external_stale_pin_cancellation_evidence(
        inventory: list[dict]) -> dict:
    """Authenticate the exact unlaunched-current-retry cancellation incident.

    Each task had an older pressure-killed attempt, but its current retry was
    requeued and unattached when the external cancellation made it terminal.
    Excluding a generic cancelled class would hide future accidental
    cancellations, so every identity/timestamp is bound to one reviewed digest
    and any partial/drifted match fails closed.
    """
    expected = set(EXTERNAL_STALE_PIN_CANCELLED_IDS)
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory
        if isinstance(row, dict)
    }
    present = expected.intersection(by_id)
    if not present:
        return {
            "task_ids": [],
            "identity_sha256": None,
            "excluded_from_production_health": True,
        }
    if present != expected:
        raise RuntimeError(
            "external stale-pin cancellation evidence is incomplete: "
            f"missing={sorted(expected - present)}")

    identities = []
    for task_id in EXTERNAL_STALE_PIN_CANCELLED_IDS:
        row = by_id[task_id]
        name = str(row.get("name") or "")
        dedupe = str(row.get("dedupe_key") or "")
        checks = {
            "id": row.get("id", row.get("task_id")) == task_id,
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "cancelled": row.get("status") == "cancelled",
            "assigned_dw16": row.get("account_name") == "dw16",
            "current_attempt_unlaunched": row.get("attached_at") is None
                and row.get("launch_started_at") is None
                and row.get("started_at") is None
                and row.get("allocation_id") is None,
            "exact_identity": name.startswith(PREFIX)
                and dedupe.startswith(f"mft-al:{name}:")
                and f":{SOLVER}:{LIBRARY}:" in dedupe,
            "finished_at": row.get("finished_at")
                in {"2026-07-12 10:39:55", "2026-07-12 10:39:56"},
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"external stale-pin cancellation {task_id} drifted: {checks}")
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _EXTERNAL_CANCELLATION_IDENTITY_FIELDS
        })
    digest = _sha(identities)
    if digest != EXTERNAL_STALE_PIN_CANCELLATION_IDENTITY_SHA256:
        raise RuntimeError(
            "external stale-pin cancellation identity seal drifted: "
            f"{digest}")
    return {
        "task_ids": list(EXTERNAL_STALE_PIN_CANCELLED_IDS),
        "identity_sha256": digest,
        "source_address": "192.168.56.1",
        "finished_at_utc": "2026-07-12 10:39:55-56",
        "current_attempt_never_started": True,
        "excluded_from_production_health": True,
    }


def _remote_step_cancellation_evidence(inventory: list[dict]) -> dict:
    """Authenticate only the exact multi-allocation cancellation burst."""
    expected = set(REMOTE_STEP_CANCELLED_IDS)
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory
        if isinstance(row, dict)
    }
    present = expected.intersection(by_id)
    if not present:
        return {
            "task_ids": [],
            "identity_sha256": None,
            "excluded_from_production_health": True,
        }
    if present != expected:
        raise RuntimeError(
            "remote-step cancellation evidence is incomplete: "
            f"missing={sorted(expected - present)}")

    incident_ids = {
        row.get("id", row.get("task_id"))
        for row in inventory
        if (isinstance(row, dict)
            and row.get("project") == scheduler_client.MFT_PROJECT
            and row.get("status") == "cancelled"
            and isinstance(row.get("finished_at"), str)
            and "2026-07-12 11:12:12" <= row["finished_at"]
            <= "2026-07-12 11:12:17")
    }
    if incident_ids != expected:
        raise RuntimeError(
            "remote-step cancellation interval drifted: "
            f"expected={sorted(expected)}, actual={sorted(incident_ids)}")

    identities = []
    allocation_ids = set()
    for task_id in REMOTE_STEP_CANCELLED_IDS:
        row = by_id[task_id]
        name = str(row.get("name") or "")
        dedupe = str(row.get("dedupe_key") or "")
        allocation_id = row.get("allocation_id")
        checks = {
            "id": row.get("id", row.get("task_id")) == task_id,
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "cancelled": row.get("status") == "cancelled",
            "legacy_unpinned": row.get("requested_account_name") == "",
            "active_attempt_existed": all(
                isinstance(row.get(field), str) and bool(row[field].strip())
                for field in ("attached_at", "launch_started_at", "started_at")),
            "allocation": type(allocation_id) is int and allocation_id > 0,
            "slurm_job": str(row.get("slurm_job_id") or "").isdigit(),
            "allocation_node": re.fullmatch(
                r"n\d+", str(row.get("allocation_node_name") or "")) is not None,
            "no_solver_exit": row.get("exit_code") is None,
            "no_failure_message": row.get("failure_message") in (None, ""),
            "exact_identity": name.startswith(PREFIX)
                and dedupe.startswith(f"mft-al:{name}:")
                and f":{SOLVER}:{LIBRARY}:" in dedupe,
            "finished_at": isinstance(row.get("finished_at"), str)
                and "2026-07-12 11:12:12" <= row["finished_at"]
                <= "2026-07-12 11:12:17",
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"remote-step cancellation {task_id} drifted: {checks}")
        allocation_ids.add(allocation_id)
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _REMOTE_STEP_CANCELLATION_IDENTITY_FIELDS
        })
    if len(allocation_ids) != 8:
        raise RuntimeError(
            "remote-step cancellation allocation spread drifted: "
            f"{sorted(allocation_ids)}")
    digest = _sha(identities)
    if digest != REMOTE_STEP_CANCELLATION_IDENTITY_SHA256:
        raise RuntimeError(
            "remote-step cancellation identity seal drifted: "
            f"{digest}")
    return {
        "task_ids": list(REMOTE_STEP_CANCELLED_IDS),
        "identity_sha256": digest,
        "finished_at_utc": "2026-07-12 11:12:12-17",
        "allocation_count": len(allocation_ids),
        "solver_exit_codes": [],
        "scheduler_web_log": {
            "path": r"C:\Users\peets\NEC\slurm_scheduler\logs\web.log",
            "line_window": "950908-950999",
            "post_count": 0,
        },
        "excluded_from_production_health": True,
    }


def _operator_cancelled_stale_prepolicy_evidence(
        inventory: list[dict]) -> dict:
    """Seal the exact operator-cancelled, pre-policy stale launch incident."""
    expected = set(OPERATOR_CANCELLED_STALE_PREPOLICY_IDS)
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory
        if isinstance(row, dict)
    }
    present = expected.intersection(by_id)
    if present != expected:
        raise RuntimeError(
            "operator-cancelled stale prepolicy evidence is incomplete: "
            f"missing={sorted(expected - present)}")

    identities = []
    for task_id in OPERATOR_CANCELLED_STALE_PREPOLICY_IDS:
        row = by_id[task_id]
        if not rapid_campaign._is_operator_cancelled_stale_prepolicy_launch(row):
            raise RuntimeError(
                f"operator-cancelled stale prepolicy task {task_id} drifted")
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _OPERATOR_CANCELLED_STALE_PREPOLICY_IDENTITY_FIELDS
        })
    identity_digest = _sha(identities)
    if identity_digest != OPERATOR_CANCELLED_STALE_PREPOLICY_IDENTITY_SHA256:
        raise RuntimeError(
            "operator-cancelled stale prepolicy identity seal drifted: "
            f"{identity_digest}")
    audit_digest = _sha(_OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT)
    if audit_digest != OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT_SHA256:
        raise RuntimeError(
            "operator-cancelled stale prepolicy audit seal drifted: "
            f"{audit_digest}")
    return {
        "task_ids": list(OPERATOR_CANCELLED_STALE_PREPOLICY_IDS),
        "classification": "operator_cancelled_stale_prepolicy_launch",
        "identity_sha256": identity_digest,
        "audit_sha256": audit_digest,
        "source_address": _OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT[
            "source_address"],
        "request_path": _OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT[
            "request_path"],
        "web_log": dict(_OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT["web_log"]),
        "recovery_held_events": copy.deepcopy(
            _OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT[
                "recovery_held_events"]),
        "operator_confirmation": _OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT[
            "operator_confirmation"],
        "excluded_from_current_simulation_valid_rate": True,
        "retained_in_lifetime_invalid_accounting": True,
    }


def _sealed_old_timeout_contract_evidence(inventory: list[dict]) -> dict:
    """Authenticate only six reviewed timeouts from old parent allocations."""
    expected = set(SEALED_OLD_TIMEOUT_CONTRACT_IDS)
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory if isinstance(row, dict)
    }
    present = expected.intersection(by_id)
    if present != expected:
        raise RuntimeError(
            "sealed old-timeout evidence is incomplete: "
            f"missing={sorted(expected - present)}")
    identities = []
    for task_id in SEALED_OLD_TIMEOUT_CONTRACT_IDS:
        row = by_id[task_id]
        message = row.get("failure_message")
        if not rapid_campaign._is_sealed_old_timeout_contract_incident(
                row, message):
            raise RuntimeError(f"sealed old-timeout task {task_id} drifted")
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _SEALED_OLD_TIMEOUT_CONTRACT_IDENTITY_FIELDS
        })
    identity_digest = _sha(identities)
    if identity_digest != SEALED_OLD_TIMEOUT_CONTRACT_IDENTITY_SHA256:
        raise RuntimeError(
            f"sealed old-timeout identity seal drifted: {identity_digest}")
    audit_digest = _sha(_SEALED_OLD_TIMEOUT_CONTRACT_AUDIT)
    if audit_digest != SEALED_OLD_TIMEOUT_CONTRACT_AUDIT_SHA256:
        raise RuntimeError(
            f"sealed old-timeout audit seal drifted: {audit_digest}")
    return {
        "task_ids": list(SEALED_OLD_TIMEOUT_CONTRACT_IDS),
        "classification": "sealed_old_timeout_contract_incident",
        "identity_sha256": identity_digest,
        "audit_sha256": audit_digest,
        "allocations": copy.deepcopy(
            _SEALED_OLD_TIMEOUT_CONTRACT_AUDIT["allocations"]),
        "excluded_from_current_runtime_health": True,
        "retained_in_lifetime_invalid_accounting": True,
    }


def _target300_rollback_artifact() -> dict:
    """Load only the immutable operator-authorized pool400->300 audit."""
    try:
        raw = TARGET300_ROLLBACK_AUDIT_PATH.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError("target300 rollback audit is unavailable/unreadable") from exc
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != TARGET300_ROLLBACK_AUDIT_FILE_SHA256:
        raise RuntimeError(
            f"target300 rollback audit file seal drifted: {file_sha}")
    expected_ids = list(TARGET300_ROLLBACK_CANCELLED_IDS)
    eligible = payload.get("eligible_snapshot")
    batches = payload.get("batches")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "revision": payload.get("state_revision") == 4,
        "type": payload.get("artifact_type") == "mft_pool_target_rollback",
        "status": payload.get("status") == "completed",
        "project": payload.get("project") == scheduler_client.MFT_PROJECT,
        "targets": payload.get("from_target") == 400
            and payload.get("to_target") == 300,
        "cap": payload.get("project_cap") == 300,
        "cycle": payload.get("cycle_serial") == 335,
        "plan": payload.get("plan_sha256") == PLAN_SHA256,
        "initial": payload.get("initial_active") == 377
            and payload.get("initial_statuses") == {"queued": 77, "running": 300},
        "final": payload.get("final_active") == 300
            and payload.get("final_statuses") == {"running": 300},
        "cancelled_ids": payload.get("cancelled_ids") == expected_ids,
        "cancelled_identity": payload.get("cancelled_identity_sha256")
            == TARGET300_ROLLBACK_CANCELLED_IDENTITY_SHA256,
        "eligible": isinstance(eligible, list) and len(eligible) == 77,
        "eligible_sha": payload.get("eligible_snapshot_sha256")
            == TARGET300_ROLLBACK_ELIGIBLE_IDENTITY_SHA256,
        "batches": isinstance(batches, list) and len(batches) == 1,
        "error": payload.get("error") is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"target300 rollback audit contract drifted: {checks}")
    if (_sha(eligible) != TARGET300_ROLLBACK_ELIGIBLE_IDENTITY_SHA256
            or batches[0].get("requested") != eligible
            or batches[0].get("requested_sha256")
            != TARGET300_ROLLBACK_ELIGIBLE_IDENTITY_SHA256
            or batches[0].get("active_before") != 377
            or batches[0].get("excess_before") != 77):
        raise RuntimeError("target300 rollback prepared batch seal drifted")
    eligible_ids = [row.get("id") for row in eligible if isinstance(row, dict)]
    if sorted(eligible_ids) != expected_ids or len(set(eligible_ids)) != 77:
        raise RuntimeError("target300 rollback eligible task identity drifted")
    return payload


def _target300_rollback_cancelled_evidence(inventory: list[dict]) -> dict:
    """Authenticate exactly the 77 queued-only cancellations used for 300."""
    artifact = _target300_rollback_artifact()
    expected_ids = set(TARGET300_ROLLBACK_CANCELLED_IDS)
    rows = [row for row in inventory if isinstance(row, dict)]
    by_id = {row.get("id", row.get("task_id")): row for row in rows}
    if expected_ids.intersection(by_id) != expected_ids:
        raise RuntimeError(
            "target300 rollback task evidence is incomplete: missing="
            f"{sorted(expected_ids - set(by_id))}")
    prepared = {row["id"]: row for row in artifact["eligible_snapshot"]}
    identities = []
    for task_id in TARGET300_ROLLBACK_CANCELLED_IDS:
        row = by_id[task_id]
        prior = prepared[task_id]
        safe = (
            row.get("id", row.get("task_id")) == task_id
            and row.get("name") == prior.get("name")
            and row.get("dedupe_key") == prior.get("dedupe_key")
            and row.get("project") == scheduler_client.MFT_PROJECT
            and row.get("status") == "cancelled"
            and row.get("attached_at") in (None, "")
            and row.get("launch_started_at") in (None, "")
            and row.get("started_at") in (None, "")
            and row.get("allocation_id") is None
            and row.get("assigned_allocation") is None
            and row.get("slurm_job_id") in (None, "")
            and row.get("exit_code") is None
            and row.get("failure_message") in (None, "")
            and row.get("cpus") == 4
            and row.get("memory_mb") == 65_536
            and row.get("timeout_seconds") == 14_400
            and row.get("scheduling_profile") == "fea_bursty"
            and row.get("required_capability") == "conda:pyaedt2026v1"
            and row.get("env_profile") == "pyaedt2026v1"
            and row.get("gpus") == 0
        )
        if not safe:
            raise RuntimeError(f"target300 rollback task {task_id} drifted")
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _TARGET300_ROLLBACK_IDENTITY_FIELDS
        })
    identity_sha = _sha(identities)
    if identity_sha != TARGET300_ROLLBACK_CANCELLED_IDENTITY_SHA256:
        raise RuntimeError(
            f"target300 rollback live identity seal drifted: {identity_sha}")
    return {
        "task_ids": list(TARGET300_ROLLBACK_CANCELLED_IDS),
        "classification": "user_target_rollback_cancelled",
        "artifact": str(TARGET300_ROLLBACK_AUDIT_PATH.resolve()),
        "artifact_sha256": TARGET300_ROLLBACK_AUDIT_FILE_SHA256,
        "eligible_identity_sha256": TARGET300_ROLLBACK_ELIGIBLE_IDENTITY_SHA256,
        "cancelled_identity_sha256": identity_sha,
        "from_target": 400,
        "to_target": 300,
        "excluded_from_current_runtime_health": True,
        "retained_in_lifetime_invalid_accounting": True,
    }


def _classify_target300_rollback_outcomes(
        production_state: dict, evidence: dict) -> None:
    """Reclassify only the exact artifact-authenticated terminal outcomes."""
    expected = set(evidence.get("task_ids") or ())
    if expected != set(TARGET300_ROLLBACK_CANCELLED_IDS):
        raise RuntimeError("target300 rollback outcome authorization drifted")
    seen = set()
    for outcome in production_state.get("outcomes", []):
        task_id = outcome.get("task_id") if isinstance(outcome, dict) else None
        if task_id not in expected:
            continue
        if (outcome.get("status") != "cancelled"
                or outcome.get("state") != "invalid"):
            raise RuntimeError(
                f"target300 rollback outcome {task_id} is not cancelled-invalid")
        outcome["expected_failure_reason"] = "user_target_rollback_cancelled"
        outcome["error_fingerprint"] = None
        production_state["cache"][str(task_id)] = dict(outcome)
        seen.add(task_id)
    if seen != expected:
        raise RuntimeError(
            "target300 rollback terminal outcomes are incomplete: missing="
            f"{sorted(expected - seen)}")


def _dynamic_target_cancelled_evidence(
        state: dict, inventory: list[dict]) -> dict:
    """Authenticate every cancellation from immutable target transitions."""
    sealed = state.get("target_cancelled_tasks", {})
    if not isinstance(sealed, dict):
        raise RuntimeError("dynamic target cancellation state is invalid")
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory if isinstance(row, dict)
    }
    task_ids = []
    transition_serials = set()
    identities = []
    for key, expected in sorted(
            sealed.items(), key=lambda item: int(item[0])):
        task_id = int(key)
        if expected.get("task_id") != task_id:
            raise RuntimeError(
                f"dynamic target cancellation state {task_id} drifted")
        transition_serial = expected.get("transition_serial")
        transition = _load_target_transition(
            _target_transition_path(transition_serial))
        if (transition is None
                or transition.get("status") not in _TARGET_TRANSITION_TERMINAL
                or task_id not in transition.get("cancelled_ids", [])):
            raise RuntimeError(
                f"dynamic target cancellation journal {task_id} is missing")
        row = by_id.get(task_id)
        if row is None:
            raise RuntimeError(
                f"dynamic target cancelled task {task_id} is missing")
        checks = {
            "name": row.get("name") == expected.get("name"),
            "dedupe": row.get("dedupe_key") == expected.get("dedupe_key"),
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "cancelled": row.get("status") == "cancelled",
            "attached": row.get("attached_at") in (None, ""),
            "launch": row.get("launch_started_at") in (None, ""),
            "started": row.get("started_at") in (None, ""),
            "allocation": row.get("allocation_id") is None
                and row.get("assigned_allocation") is None,
            "slurm": row.get("slurm_job_id") in (None, ""),
            "exit": row.get("exit_code") is None,
            "failure": row.get("failure_message") in (None, ""),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"dynamic target cancelled task {task_id} drifted: {checks}")
        identity = {
            field: (
                task_id if field == "id"
                else transition_serial if field == "controller_cycle_serial"
                else row.get(field))
            for field in _TARGET_CANCEL_IDENTITY_FIELDS
        }
        # ``controller_cycle_serial`` identifies submission ownership, not the
        # target-transition serial; retain the exact prepared proof.
        identity["controller_cycle_serial"] = expected.get(
            "controller_cycle_serial")
        identities.append(identity)
        task_ids.append(task_id)
        transition_serials.add(transition_serial)
    return {
        "task_ids": task_ids,
        "classification": "operator_target_reduction_cancelled",
        "transition_serials": sorted(transition_serials),
        "identity_sha256": _sha(identities),
        "excluded_from_current_runtime_health": True,
        "retained_in_lifetime_invalid_accounting": True,
    }


def _classify_dynamic_target_cancelled_outcomes(
        production_state: dict, evidence: dict) -> None:
    expected = set(evidence.get("task_ids") or ())
    if not expected:
        return
    seen = set()
    for outcome in production_state.get("outcomes", []):
        task_id = outcome.get("task_id") if isinstance(outcome, dict) else None
        if task_id not in expected:
            continue
        if (outcome.get("status") != "cancelled"
                or outcome.get("state") != "invalid"):
            raise RuntimeError(
                f"dynamic target outcome {task_id} is not cancelled-invalid")
        outcome["expected_failure_reason"] = (
            "operator_target_reduction_cancelled")
        outcome["error_fingerprint"] = None
        production_state["cache"][str(task_id)] = dict(outcome)
        seen.add(task_id)
    if seen != expected:
        raise RuntimeError(
            "dynamic target cancelled outcomes are incomplete: missing="
            f"{sorted(expected - seen)}")



def _resolved_scheduler_parent_cancel_evidence(
        inventory: list[dict]) -> dict:
    """Authenticate the fixed 14:39:33Z four-parent cancellation incident."""
    expected = set(RESOLVED_SCHEDULER_PARENT_CANCEL_IDS)
    by_id = {
        row.get("id", row.get("task_id")): row
        for row in inventory
        if isinstance(row, dict)
    }
    present = expected.intersection(by_id)
    if present != expected:
        raise RuntimeError(
            "resolved scheduler parent-cancel evidence is incomplete: "
            f"missing={sorted(expected - present)}")

    parent_allocations = {8345, 8346, 8347, 8349}
    incident_ids = {
        row.get("id", row.get("task_id"))
        for row in inventory
        if (isinstance(row, dict)
            and row.get("project") == scheduler_client.MFT_PROJECT
            and row.get("status") == "failed"
            and row.get("exit_code") in (143, "143")
            and row.get("allocation_id") in parent_allocations
            and isinstance(row.get("finished_at"), str)
            and "2026-07-12 14:40:19" <= row["finished_at"]
            <= "2026-07-12 14:46:28")
    }
    if incident_ids != expected:
        raise RuntimeError(
            "resolved scheduler parent-cancel task set drifted: "
            f"expected={sorted(expected)}, actual={sorted(incident_ids)}")

    identities = []
    for task_id in RESOLVED_SCHEDULER_PARENT_CANCEL_IDS:
        row = by_id[task_id]
        if not rapid_campaign._is_resolved_scheduler_parent_cancel_incident(
                row):
            raise RuntimeError(
                f"resolved scheduler parent-cancel task {task_id} drifted")
        identities.append({
            field: (task_id if field == "id" else row.get(field))
            for field in _RESOLVED_SCHEDULER_PARENT_CANCEL_IDENTITY_FIELDS
        })
    identity_digest = _sha(identities)
    if identity_digest != RESOLVED_SCHEDULER_PARENT_CANCEL_IDENTITY_SHA256:
        raise RuntimeError(
            "resolved scheduler parent-cancel identity seal drifted: "
            f"{identity_digest}")

    audit_digest = _sha(_RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT)
    if audit_digest != RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT_SHA256:
        raise RuntimeError(
            "resolved scheduler parent-cancel audit seal drifted: "
            f"{audit_digest}")

    # These hashes describe the exact scheduler deployment that resolved this
    # historical incident. They are already covered by the audit digest above.
    # Re-reading a mutable local scheduler checkout would make clean-clone
    # audit non-reproducible and would turn later legitimate scheduler changes
    # into false controller failures.
    runtime_hashes = copy.deepcopy(
        _RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT[
            "scheduler_runtime_sha256"])

    return {
        "task_ids": list(RESOLVED_SCHEDULER_PARENT_CANCEL_IDS),
        "classification": "resolved_scheduler_parent_cancel_incident",
        "identity_sha256": identity_digest,
        "audit_sha256": audit_digest,
        "parent_cancelled_at_utc": "2026-07-12 14:39:33",
        "parents": copy.deepcopy(
            _RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT[
                "incident"]["parents"]),
        "scheduler_runtime_sha256": runtime_hashes,
        "dynamic_project_cap_route": copy.deepcopy(
            _RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT[
                "dynamic_project_cap_route"]),
        "deployment_evidence": copy.deepcopy(
            _RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT[
                "deployment_evidence"]),
        "excluded_from_current_runtime_health": True,
        "retained_in_lifetime_invalid_accounting": True,
    }



def _static_bundle() -> dict:
    bundle = production.static_audit()
    plan = bundle["plan"]
    checks = {
        "seal": plan.get("plan_sha256") == PLAN_SHA256,
        "solver": plan.get("solver_revision") == SOLVER,
        "library": plan.get("library_revision") == LIBRARY,
        "seed": plan.get("seed") == SEED,
        "cursor_start": plan.get("candidate_cursor_start") == INITIAL_CURSOR,
        "cursor_end": plan.get("candidate_cursor_end") == 3_917,
        "first_serial": plan.get("first_serial") == INITIAL_SERIAL + 1,
        "last_serial": plan.get("last_serial") == INITIAL_SERIAL + PLAN_TASK_COUNT,
        "count": plan.get("task_count") == PLAN_TASK_COUNT,
        "prefix": plan.get("task_prefix") == PREFIX,
    }
    if not all(checks.values()):
        raise RuntimeError(f"b171 candidate plan identity drifted: {checks}")
    names, params_digests, dedupes = set(), set(), set()
    for index, task in enumerate(plan["tasks"]):
        _candidate_contract(task["effective_params"], f"plan[{index}]")
        names.add(task["name"])
        params_digests.add(task["params_sha256"])
        dedupes.add(task["dedupe_key"])
    if tuple(map(len, (names, params_digests, dedupes))) != (
            PLAN_TASK_COUNT, PLAN_TASK_COUNT, PLAN_TASK_COUNT):
        raise RuntimeError("b171 plan names/params/dedupes are not unique")
    return bundle


def _strict_snapshot() -> dict:
    try:
        candidates = [STRICT_STATUS_PATH, *STRICT_STATUS_PATH.parent.glob(
            STRICT_STATUS_PATH.name + ".gen-*.json")]
        snapshots = []
        errors = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                stamp = datetime.fromisoformat(
                    str(payload["time"]).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.astimezone()
                snapshots.append((stamp.astimezone(timezone.utc), path, payload))
            except Exception as exc:
                errors.append(f"{path.name}:{type(exc).__name__}")
        if not snapshots:
            raise RuntimeError("no valid strict status snapshot: " + ",".join(errors))
        stamp, source, payload = max(snapshots, key=lambda item: item[0])
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        pinned = (
            checkpoint_status_revision_identity_matches(
                payload, SOLVER, LIBRARY
            )
            and age <= 20 * 60
        )
        rows = int(payload.get("strict_full_rows") or 0) if pinned else 0
        return {
            "pinned": pinned, "rows": rows, "age_seconds": age,
            "source": str(source),
        }
    except Exception as exc:
        return {"pinned": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}"}


def _new_state() -> dict:
    return {
        "schema_version": 1,
        "state_revision": 0,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "plan_sha256": PLAN_SHA256,
        "target_active": TARGET_ACTIVE,
        "target_policy": TARGET_POLICY_FIXED,
        "target_transition_serial": 0,
        "target_transition_highwater": 0,
        "target_cancelled_tasks": {},
        "dynamic_policy_adopted_cycle": None,
        "target_strict_rows": TARGET_STRICT_ROWS,
        "cycle_serial": 0,
        # Highest cycle whose terminal journal was already authenticated
        # before a later controller-state generation committed.  Legacy
        # states omit this key; their old full-prefix-audit invariant safely
        # authenticates every cycle before the latest one.
        "terminal_cycle_highwater": 0,
        "paused": False,
        "pause_reasons": [],
        "task_outcomes": {},
        "last_evidence": None,
        "updated_at": None,
    }


def _validate_state(state: dict) -> dict:
    expected = _new_state()
    immutable = (
        "schema_version", "solver_revision", "library_revision", "plan_sha256",
        "target_strict_rows",
    )
    mismatch = {key: (state.get(key), expected[key]) for key in immutable
                if state.get(key) != expected[key]}
    if mismatch:
        raise RuntimeError(f"continuous controller state identity drifted: {mismatch}")
    durable._state_revision(state, "continuous controller")
    if type(state.get("cycle_serial")) is not int or state["cycle_serial"] < 0:
        raise RuntimeError("continuous controller cycle serial is invalid")
    target_active = state.get("target_active")
    target_policy = state.get("target_policy", TARGET_POLICY_FIXED)
    if target_policy == TARGET_POLICY_FIXED:
        if target_active == LEGACY_TARGET_ACTIVE:
            if state["cycle_serial"] > TARGET_400_TRANSITION_CYCLE:
                raise RuntimeError(
                    "legacy target persisted beyond the pool400 transition cycle")
        elif target_active == PREVIOUS_TARGET_ACTIVE:
            if not (TARGET_400_TRANSITION_CYCLE
                    <= state["cycle_serial"] <= TARGET_300_TRANSITION_CYCLE):
                raise RuntimeError(
                    "pool400 target persisted outside its authenticated window")
        elif target_active == TARGET_ACTIVE:
            if (state["cycle_serial"] != 0
                    and state["cycle_serial"] < TARGET_300_TRANSITION_CYCLE):
                raise RuntimeError(
                    "pool300 target predates the target-reduction boundary")
        else:
            raise RuntimeError(
                "fixed controller target must be legacy250, prior400, or current300")
    elif target_policy == TARGET_POLICY_DYNAMIC:
        if (isinstance(target_active, bool) or not isinstance(target_active, int)
                or not TARGET_ACTIVE_MIN <= target_active <= TARGET_ACTIVE_MAX):
            raise RuntimeError("dynamic controller target is outside 1..300")
        adopted_cycle = state.get("dynamic_policy_adopted_cycle")
        if (isinstance(adopted_cycle, bool) or not isinstance(adopted_cycle, int)
                or not 0 <= adopted_cycle <= state["cycle_serial"]):
            raise RuntimeError("dynamic policy adoption cycle is invalid")
    else:
        raise RuntimeError("continuous controller target policy is invalid")
    transition_serial = state.get("target_transition_serial", 0)
    transition_highwater = state.get("target_transition_highwater", 0)
    if (isinstance(transition_serial, bool)
            or not isinstance(transition_serial, int)
            or isinstance(transition_highwater, bool)
            or not isinstance(transition_highwater, int)
            or not 0 <= transition_highwater <= transition_serial):
        raise RuntimeError("target transition serial/high-water is invalid")
    if target_policy == TARGET_POLICY_FIXED and (
            transition_serial != 0 or transition_highwater != 0):
        raise RuntimeError("fixed target policy cannot own dynamic transitions")
    if (target_policy == TARGET_POLICY_DYNAMIC
            and (transition_serial < 1
                 or transition_highwater != transition_serial)):
        raise RuntimeError("dynamic target transition history is incomplete")
    cancelled_tasks = state.get("target_cancelled_tasks", {})
    if not isinstance(cancelled_tasks, dict):
        raise RuntimeError("target cancellation evidence must be an object")
    for key, item in cancelled_tasks.items():
        if (not isinstance(item, dict)
                or isinstance(item.get("task_id"), bool)
                or not isinstance(item.get("task_id"), int)
                or str(item.get("task_id")) != str(key)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("dedupe_key"), str)
                or type(item.get("transition_serial")) is not int
                or item["transition_serial"] < 1):
            raise RuntimeError(f"target cancellation evidence {key!r} is invalid")
    # Every legacy controller incremented ``cycle_serial`` only after a full
    # prefix audit.  A crash could leave the newest serial missing or
    # interrupted, so only ``cycle_serial - 1`` is an authenticated implicit
    # high-water for a state written before this field existed.
    legacy_terminal_highwater = max(0, state["cycle_serial"] - 1)
    terminal_highwater = state.get(
        "terminal_cycle_highwater", legacy_terminal_highwater)
    if (type(terminal_highwater) is not int
            or not 0 <= terminal_highwater <= state["cycle_serial"]):
        raise RuntimeError("continuous controller terminal cycle high-water is invalid")
    if type(state.get("paused")) is not bool or not isinstance(state.get("pause_reasons"), list):
        raise RuntimeError("continuous controller pause state is invalid")
    if not isinstance(state.get("task_outcomes"), dict):
        raise RuntimeError("continuous controller outcome cache is invalid")
    return state


def _validate_state_transition(before: dict, after: dict) -> None:
    before_policy = before.get("target_policy", TARGET_POLICY_FIXED)
    after_policy = after.get("target_policy", TARGET_POLICY_FIXED)
    before_target = before.get("target_active")
    after_target = after.get("target_active")
    before_transition = int(before.get("target_transition_serial", 0))
    after_transition = int(after.get("target_transition_serial", 0))
    before_cancelled = before.get("target_cancelled_tasks", {})
    after_cancelled = after.get("target_cancelled_tasks", {})
    if before_policy == TARGET_POLICY_DYNAMIC:
        if after_policy != TARGET_POLICY_DYNAMIC:
            raise RuntimeError("dynamic target policy cannot regress")
        changed = (
            before_target != after_target
            or before_transition != after_transition
            or before_cancelled != after_cancelled
        )
        if not changed:
            return
        checks = {
            "serial": after_transition == before_transition + 1,
            "highwater": after.get("target_transition_highwater")
                == after_transition,
            "cycle": after.get("cycle_serial") == before.get("cycle_serial"),
            "cancelled_append_only": isinstance(before_cancelled, dict)
                and isinstance(after_cancelled, dict)
                and all(after_cancelled.get(key) == value
                        for key, value in before_cancelled.items()),
            "new_cancelled_bound": all(
                key in before_cancelled
                or (isinstance(value, dict)
                    and value.get("transition_serial") == after_transition)
                for key, value in after_cancelled.items()),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"dynamic target state transition is invalid: {checks}")
        return
    if before_policy != TARGET_POLICY_FIXED:
        raise RuntimeError("continuous controller source target policy is invalid")
    if after_policy == TARGET_POLICY_DYNAMIC:
        checks = {
            "same_target": before_target == after_target == TARGET_ACTIVE,
            "serial": before_transition == 0 and after_transition == 1,
            "highwater": after.get("target_transition_highwater") == 1,
            "cycle": after.get("cycle_serial") == before.get("cycle_serial"),
            "adopted_cycle": after.get("dynamic_policy_adopted_cycle")
                == before.get("cycle_serial"),
            "no_cancelled": after_cancelled == {},
        }
        if not all(checks.values()):
            raise RuntimeError(f"dynamic policy adoption is invalid: {checks}")
        return
    if after_policy != TARGET_POLICY_FIXED:
        raise RuntimeError("continuous controller target policy is invalid")
    if before_target == after_target:
        return
    transition = (before_target, after_target)
    if transition == (LEGACY_TARGET_ACTIVE, PREVIOUS_TARGET_ACTIVE):
        boundary = TARGET_400_TRANSITION_CYCLE
        label = "pool400"
    elif transition == (PREVIOUS_TARGET_ACTIVE, TARGET_ACTIVE):
        boundary = TARGET_300_TRANSITION_CYCLE
        label = "pool300"
    else:
        raise RuntimeError(
            f"continuous controller target transition is invalid: "
            f"{before_target}->{after_target}")
    if (before.get("cycle_serial") != boundary
            or after.get("cycle_serial") != boundary
            or before.get("terminal_cycle_highwater") != boundary
            or after.get("terminal_cycle_highwater") != boundary):
        raise RuntimeError(
            f"{label} transition requires exact authenticated cycle{boundary}")


def _promote_target400_state(state: dict) -> bool:
    """One-way, boundary-authenticated migration from the former pool250."""
    if state.get("target_active") in (PREVIOUS_TARGET_ACTIVE, TARGET_ACTIVE):
        return False
    if (state.get("target_active") != LEGACY_TARGET_ACTIVE
            or state.get("cycle_serial") != TARGET_400_TRANSITION_CYCLE
            or state.get("terminal_cycle_highwater")
            != TARGET_400_TRANSITION_CYCLE):
        raise RuntimeError(
            "pool400 migration requires legacy target250 at terminal cycle334")
    state["target_active"] = PREVIOUS_TARGET_ACTIVE
    return True


def _migrate_target300_state(state: dict) -> bool:
    """Reduce the maintenance target only after terminal cycle336 proof."""
    if state.get("target_active") == TARGET_ACTIVE:
        return False
    if (state.get("target_active") != PREVIOUS_TARGET_ACTIVE
            or state.get("cycle_serial") != TARGET_300_TRANSITION_CYCLE
            or state.get("terminal_cycle_highwater")
            != TARGET_300_TRANSITION_CYCLE):
        raise RuntimeError(
            "pool300 migration requires prior target400 at terminal cycle336")
    state["target_active"] = TARGET_ACTIVE
    return True


def _load_state(create: bool) -> dict:
    return durable._load_durable_state(
        STATE_PATH, _validate_state, _new_state, create=create)


def _save_state(state: dict) -> None:
    durable._save_durable_state(
        STATE_PATH, state, _validate_state,
        transition_validator=_validate_state_transition)


def _initial_feeder_state(bundle: dict) -> dict:
    recovery = bundle["recovery_plan"].get("tasks", [])
    generation = f"{SOLVER}:{LIBRARY}:seed{SEED}"
    return {
        "state_revision": 0,
        "serial": INITIAL_SERIAL,
        "submitted_samples": 0,
        "outstanding": [],
        "candidate_generation": generation,
        "candidate_cursor": INITIAL_CURSOR,
        "candidate_cursors": {generation: INITIAL_CURSOR},
        "candidate_raw_index": INITIAL_RAW_INDEX,
        "task_ids_by_generation": {generation: []},
        "task_expected_rows": {},
        "adoption_sha256": PLAN_SHA256,
        "adoption_manifest": str(production.PLAN_PATH.resolve()),
        "used_names": sorted({str(row["name"]) for row in recovery}),
        "used_params_sha256": sorted({str(row["source_params_sha256"]) for row in recovery}),
        "used_dedupe_keys": sorted({str(row["dedupe_key"]) for row in recovery}),
    }


def _validate_feeder_state(state: dict) -> dict:
    durable._state_revision(state, "continuous feeder")
    generation = f"{SOLVER}:{LIBRARY}:seed{SEED}"
    if (state.get("candidate_generation") != generation
            or state.get("adoption_sha256") != PLAN_SHA256
            or int(state.get("serial", -1)) < INITIAL_SERIAL
            or int(state.get("candidate_cursor", -1)) < INITIAL_CURSOR
            or int((state.get("candidate_cursors") or {}).get(generation, -1))
            < INITIAL_CURSOR):
        raise RuntimeError("continuous feeder state would replay cursor/serial")
    for key in ("used_names", "used_params_sha256", "used_dedupe_keys"):
        values = state.get(key)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise RuntimeError(f"continuous feeder {key} is invalid")
    return state


def _validate_feeder_transition(before: dict, after: dict) -> None:
    for key in ("serial", "submitted_samples", "candidate_cursor", "candidate_raw_index"):
        if int(after[key]) < int(before[key]):
            raise RuntimeError(f"continuous feeder {key} regressed")
    for key in ("used_names", "used_params_sha256", "used_dedupe_keys"):
        if not set(before[key]).issubset(set(after[key])):
            raise RuntimeError(f"continuous feeder {key} lost committed identities")


def _load_feeder_state(bundle: dict, create: bool) -> dict:
    return durable._load_durable_state(
        FEEDER_STATE_PATH, _validate_feeder_state,
        lambda: _initial_feeder_state(bundle), create=create)


def _save_feeder_state(state: dict) -> None:
    durable._save_durable_state(
        FEEDER_STATE_PATH, state, _validate_feeder_state,
        transition_validator=_validate_feeder_transition)


def _cycle_path(serial: int) -> Path:
    return CYCLE_ROOT / f"cycle-{serial:06d}.json"


_CYCLE_NAME = re.compile(r"^cycle-(\d{6})\.json$")
_CYCLE_ARTIFACT = re.compile(
    r"^cycle-(\d{6})\.json(?:\.gen-\d{20}-[0-9a-f]{64}\.json"
    r"|\.bak|\.tmp|\.\d+\.tmp)?$")
_CYCLE_TERMINAL_STATUSES = frozenset({
    "completed", "reconciled_no_mutation", "reconciled_committed",
})
_CYCLE_STATUS_TRANSITIONS = {
    "authorized_pending": frozenset({
        "mutation_about_to_submit", "completed", "failed_closed",
        "reconciled_no_mutation",
    }),
    "mutation_about_to_submit": frozenset({
        "accepted_readback_pending_commit", "ledger_committed", "failed_closed",
        "reconciled_no_mutation",
    }),
    "accepted_readback_pending_commit": frozenset({
        "ledger_committed", "failed_closed", "reconciled_committed",
    }),
    "ledger_committed": frozenset({
        "mutation_about_to_submit", "completed", "failed_closed",
        "reconciled_committed",
    }),
    "failed_closed": frozenset({
        "reconciled_no_mutation", "reconciled_committed",
    }),
}
RECONCILIATION_EVIDENCE_SCHEMA = (
    "continuous-refill-cycle-reconciliation-evidence-v1")


def _cycle_serial(path: Path) -> int:
    match = _CYCLE_NAME.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"invalid continuous refill cycle path: {path}")
    return int(match.group(1))


def _validate_cycle(payload: dict, path: Path) -> dict:
    """Validate one canonical or immutable cycle-journal payload.

    Cycles 1--36 predate immutable journal generations.  They are interpreted
    as revision zero, but every subsequent write has an explicit monotonically
    increasing ``state_revision`` and is committed to a checksummed immutable
    generation before the canonical convenience view is touched.
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"cycle journal is not an object: {path}")
    serial = _cycle_serial(path)
    schema = payload.get("schema_version")
    if schema not in (1, 2) or payload.get("cycle_serial") != serial:
        raise RuntimeError(f"cycle journal identity drifted: {path}")
    if schema == 1 and "state_revision" not in payload:
        payload["state_revision"] = 0
    durable._state_revision(payload, f"cycle {serial}")
    cycle_policy = payload.get("target_policy", TARGET_POLICY_FIXED)
    if cycle_policy == TARGET_POLICY_DYNAMIC:
        target = payload.get("target_active")
        target_ok = bool(
            serial > TARGET_300_TRANSITION_CYCLE
            and type(target) is int
            and TARGET_ACTIVE_MIN <= target <= TARGET_ACTIVE_MAX
            and payload.get("project_cap_observed") == target)
    else:
        expected_target = (
            LEGACY_TARGET_ACTIVE
            if serial <= TARGET_400_TRANSITION_CYCLE
            else PREVIOUS_TARGET_ACTIVE
            if serial <= TARGET_300_TRANSITION_CYCLE
            else TARGET_ACTIVE)
        target_ok = bool(
            cycle_policy == TARGET_POLICY_FIXED
            and payload.get("target_active") == expected_target)
    if payload.get("plan_sha256") != PLAN_SHA256 or not target_ok:
        raise RuntimeError(f"cycle journal campaign identity drifted: {path}")
    status = payload.get("status")
    known = set(_CYCLE_STATUS_TRANSITIONS) | set(_CYCLE_TERMINAL_STATUSES)
    if not isinstance(status, str) or status not in known:
        raise RuntimeError(f"cycle journal status is invalid: {path}: {status!r}")
    journal = payload.get("formal_journal")
    if not isinstance(journal, dict) or not isinstance(journal.get("events"), list):
        raise RuntimeError(f"cycle formal journal is invalid: {path}")
    if status.startswith("reconciled_"):
        reconciliation = payload.get("reconciliation")
        if (not isinstance(reconciliation, dict)
                or reconciliation.get("action") != status
                or not isinstance(reconciliation.get("evidence_sha256"), str)):
            raise RuntimeError(f"cycle reconciliation seal is invalid: {path}")
    return payload


def _load_cycle(path: Path) -> dict | None:
    return durable._authoritative_state(
        path, lambda payload: _validate_cycle(payload, path), repair=False)


def _cycle_history_exists(path: Path) -> bool:
    return bool(
        path.exists()
        or list(path.parent.glob(f"{path.name}.gen-*.json"))
        or durable._recovery_artifact_paths(path)
    )


def _initialize_cycle(path: Path, cycle: dict) -> None:
    """Commit a new cycle authorization without relying on ``os.replace``."""
    if _cycle_history_exists(path):
        raise RuntimeError(f"continuous refill cycle already exists: {path}")
    committed = copy.deepcopy(cycle)
    committed["schema_version"] = 2
    committed["state_revision"] = 0
    _validate_cycle(committed, path)
    durable._write_immutable_generation(path, committed)
    cycle.clear()
    cycle.update(copy.deepcopy(committed))
    # Canonical JSON is only a convenience view.  The verified immutable
    # generation above is the commit point, so a RaiDrive WinError 5 here can
    # never erase authorization or permit a submit without a durable record.
    durable._best_effort_canonical(path, committed)


def _validate_cycle_transition(before: dict, after: dict, path: Path) -> None:
    immutable = (
        "cycle_serial", "plan_sha256", "target_active", "target_policy",
        "project_cap_observed",
    )
    drift = {
        key: (before.get(key), after.get(key)) for key in immutable
        if before.get(key) != after.get(key)
    }
    if drift:
        raise RuntimeError(f"cycle journal immutable identity drifted: {path}: {drift}")
    if not (after.get("schema_version") == before.get("schema_version")
            or (before.get("schema_version") == 1
                and after.get("schema_version") == 2)):
        raise RuntimeError(f"cycle journal schema regressed: {path}")
    before_status = before["status"]
    after_status = after["status"]
    if after_status not in _CYCLE_STATUS_TRANSITIONS.get(before_status, frozenset()):
        raise RuntimeError(
            f"invalid cycle journal transition for {path}: "
            f"{before_status}->{after_status}")
    before_events = before["formal_journal"]["events"]
    after_events = after["formal_journal"]["events"]
    if len(after_events) < len(before_events):
        raise RuntimeError(f"cycle journal events regressed: {path}")
    for index, prior in enumerate(before_events):
        current = after_events[index]
        if not isinstance(prior, dict) or not isinstance(current, dict):
            raise RuntimeError(f"cycle journal event is invalid: {path}")
        for key in ("name", "dedupe_key", "candidate_raw_index", "params_sha256"):
            if key in prior and current.get(key) != prior[key]:
                raise RuntimeError(
                    f"cycle journal event {index} {key} drifted: {path}")
        for key in ("accepted_or_reconciled", "ledger_committed"):
            if prior.get(key) is True and current.get(key) is not True:
                raise RuntimeError(
                    f"cycle journal event {index} {key} regressed: {path}")


def _save_cycle(path: Path, cycle: dict, status: str) -> None:
    """Append and verify a journal generation before returning to its caller."""
    disk = _load_cycle(path)
    if disk is None:
        raise RuntimeError(f"refusing to update missing cycle history: {path}")
    memory_revision = durable._state_revision(cycle, path.name)
    disk_revision = durable._state_revision(disk, path.name)
    if memory_revision != disk_revision:
        raise RuntimeError(
            f"stale cycle journal update refused for {path}: memory revision "
            f"{memory_revision}, durable revision {disk_revision}")
    # Materialize a legacy canonical/recovery payload as generation zero
    # before appending its successor.  This preserves the exact authoritative
    # interrupted state even when the later canonical convenience repair
    # succeeds and overwrites the legacy file.
    durable._write_immutable_generation(path, disk)
    committed = copy.deepcopy(cycle)
    committed["schema_version"] = 2
    committed["status"] = status
    committed["updated_at"] = _now()
    committed["state_revision"] = memory_revision + 1
    _validate_cycle(committed, path)
    _validate_cycle_transition(disk, committed, path)
    durable._write_immutable_generation(path, committed)
    cycle.clear()
    cycle.update(copy.deepcopy(committed))
    durable._best_effort_canonical(path, committed)


_TARGET_TRANSITION_NAME = re.compile(r"^transition-(\d{6})\.json$")
_TARGET_TRANSITION_ARTIFACT = re.compile(
    r"^transition-(\d{6})\.json(?:\.gen-\d{20}-[0-9a-f]{64}\.json"
    r"|\.bak|\.tmp|\.\d+\.tmp)?$")
_TARGET_TRANSITION_TERMINAL = frozenset({"completed", "superseded"})
_TARGET_TRANSITION_STATUS_TRANSITIONS = {
    "prepared": frozenset({"cancelling", "completed", "superseded"}),
    "cancelling": frozenset({"completed", "superseded"}),
}
_TARGET_CANCEL_IDENTITY_FIELDS = (
    "id", "name", "dedupe_key", "project", "status", "created_at",
    "attached_at", "launch_started_at", "started_at", "finished_at",
    "allocation_id", "assigned_allocation", "slurm_job_id",
    "allocation_node_name", "account_name", "requested_account_name",
    "exit_code", "failure_message", "cpus", "memory_mb",
    "timeout_seconds", "scheduling_profile", "required_capability",
    "env_profile", "gpus", "controller_cycle_serial",
)


def _target_transition_path(serial: int) -> Path:
    return (CYCLE_ROOT / "target-transitions"
            / f"transition-{serial:06d}.json")


def _target_transition_serial(path: Path) -> int:
    match = _TARGET_TRANSITION_NAME.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"invalid target transition path: {path}")
    return int(match.group(1))


def _validate_target_transition(payload: dict, path: Path) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError(f"target transition is not an object: {path}")
    serial = _target_transition_serial(path)
    if (payload.get("schema_version") != 1
            or payload.get("artifact_type")
            != "continuous-refill-dynamic-target-transition-v1"
            or payload.get("transition_serial") != serial
            or payload.get("project") != scheduler_client.MFT_PROJECT
            or payload.get("plan_sha256") != PLAN_SHA256
            or payload.get("target_policy") != TARGET_POLICY_DYNAMIC):
        raise RuntimeError(f"target transition identity drifted: {path}")
    durable._state_revision(payload, f"target transition {serial}")
    from_target = payload.get("from_target")
    to_target = payload.get("to_target")
    if (type(from_target) is not int or type(to_target) is not int
            or not TARGET_ACTIVE_MIN <= from_target <= TARGET_ACTIVE_MAX
            or not TARGET_ACTIVE_MIN <= to_target <= TARGET_ACTIVE_MAX):
        raise RuntimeError(f"target transition bounds are invalid: {path}")
    action = payload.get("action")
    if action == "adopt_dynamic_policy":
        action_ok = serial == 1 and from_target == to_target == TARGET_ACTIVE
    elif action == "increase_target":
        action_ok = to_target > from_target
    elif action == "decrease_target":
        action_ok = to_target < from_target
    else:
        action_ok = False
    if not action_ok:
        raise RuntimeError(f"target transition action is invalid: {path}")
    active_snapshot = _transition_active_snapshot(payload.get("active_snapshot"))
    if active_snapshot["project_max_active_tasks"] != to_target:
        raise RuntimeError(f"target transition project snapshot cap drifted: {path}")
    status = payload.get("status")
    known_statuses = (
        set(_TARGET_TRANSITION_STATUS_TRANSITIONS)
        | set(_TARGET_TRANSITION_TERMINAL))
    if status not in known_statuses:
        raise RuntimeError(f"target transition status is invalid: {path}")
    eligible = payload.get("eligible_tasks")
    selected = payload.get("selected_tasks")
    if not isinstance(eligible, list) or not isinstance(selected, list):
        raise RuntimeError(f"target transition selected tasks are invalid: {path}")
    eligible_ids = [
        item.get("id") for item in eligible if isinstance(item, dict)]
    if (len(eligible_ids) != len(eligible)
            or len(eligible_ids) != len(set(eligible_ids))
            or payload.get("eligible_identity_sha256") != _sha(eligible)):
        raise RuntimeError(f"target transition eligible-task seal drifted: {path}")
    selected_ids = []
    for item in selected:
        task_id = item.get("id") if isinstance(item, dict) else None
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or task_id in selected_ids
                or item.get("status") != "queued"
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("dedupe_key"), str)
                or not isinstance(item.get("controller_cycle_serial"), int)):
            raise RuntimeError(
                f"target transition selected task is invalid: {path}")
        selected_ids.append(task_id)
    if (payload.get("selected_identity_sha256") != _sha(selected)
            or (action != "decrease_target" and (eligible or selected))
            or not set(selected_ids).issubset(eligible_ids)
            or any(item not in eligible for item in selected)):
        raise RuntimeError(f"target transition selected-task seal drifted: {path}")
    cancelled_ids = payload.get("cancelled_ids")
    skipped_ids = payload.get("skipped_ids")
    readback = payload.get("readback")
    if (not isinstance(cancelled_ids, list)
            or not isinstance(skipped_ids, list)
            or not isinstance(readback, list)
            or len(cancelled_ids) != len(set(cancelled_ids))
            or len(skipped_ids) != len(set(skipped_ids))
            or set(cancelled_ids).intersection(skipped_ids)
            or not set(cancelled_ids).union(skipped_ids).issubset(selected_ids)):
        raise RuntimeError(f"target transition readback sets are invalid: {path}")
    if status in _TARGET_TRANSITION_TERMINAL:
        if (set(cancelled_ids).union(skipped_ids) != set(selected_ids)
                or {item.get("id") for item in readback
                    if isinstance(item, dict)} != set(selected_ids)
                or payload.get("readback_identity_sha256") != _sha(readback)):
            raise RuntimeError(
                f"terminal target transition readback is incomplete: {path}")
    return payload


def _load_target_transition(path: Path) -> dict | None:
    return durable._authoritative_state(
        path, lambda payload: _validate_target_transition(payload, path),
        repair=False)


def _target_transition_history_exists(path: Path) -> bool:
    return bool(
        path.exists()
        or list(path.parent.glob(f"{path.name}.gen-*.json"))
        or durable._recovery_artifact_paths(path))


def _initialize_target_transition(path: Path, payload: dict) -> None:
    if _target_transition_history_exists(path):
        raise RuntimeError(f"target transition already exists: {path}")
    committed = copy.deepcopy(payload)
    committed["schema_version"] = 1
    committed["state_revision"] = 0
    _validate_target_transition(committed, path)
    durable._write_immutable_generation(path, committed)
    payload.clear()
    payload.update(copy.deepcopy(committed))
    durable._best_effort_canonical(path, committed)


def _validate_target_transition_update(
        before: dict, after: dict, path: Path) -> None:
    immutable = (
        "artifact_type", "transition_serial", "project", "plan_sha256",
        "target_policy", "action", "from_target", "to_target",
        "project_updated_at", "active_snapshot", "eligible_tasks",
        "eligible_identity_sha256", "selected_tasks",
        "selected_identity_sha256", "created_at",
    )
    drift = {
        key: (before.get(key), after.get(key)) for key in immutable
        if before.get(key) != after.get(key)
    }
    if drift:
        raise RuntimeError(f"target transition immutable fields drifted: {drift}")
    if after.get("status") not in _TARGET_TRANSITION_STATUS_TRANSITIONS.get(
            before.get("status"), frozenset()):
        raise RuntimeError(
            f"invalid target transition status change: "
            f"{before.get('status')}->{after.get('status')}")
    if not set(before.get("cancelled_ids") or ()).issubset(
            after.get("cancelled_ids") or ()):
        raise RuntimeError("target transition cancelled IDs regressed")
    if not set(before.get("skipped_ids") or ()).issubset(
            after.get("skipped_ids") or ()):
        raise RuntimeError("target transition skipped IDs regressed")


def _save_target_transition(path: Path, payload: dict, status: str) -> None:
    disk = _load_target_transition(path)
    if disk is None:
        raise RuntimeError(f"target transition is missing: {path}")
    memory_revision = durable._state_revision(payload, path.name)
    disk_revision = durable._state_revision(disk, path.name)
    if memory_revision != disk_revision:
        raise RuntimeError(f"stale target transition update refused: {path}")
    committed = copy.deepcopy(payload)
    committed["status"] = status
    committed["updated_at"] = _now()
    committed["state_revision"] = memory_revision + 1
    _validate_target_transition(committed, path)
    _validate_target_transition_update(disk, committed, path)
    durable._write_immutable_generation(path, committed)
    payload.clear()
    payload.update(copy.deepcopy(committed))
    durable._best_effort_canonical(path, committed)


def _target_transition_serials_on_disk() -> set[int]:
    root = CYCLE_ROOT / "target-transitions"
    if not root.exists():
        return set()
    serials = set()
    for item in root.iterdir():
        match = _TARGET_TRANSITION_ARTIFACT.fullmatch(item.name)
        if match is not None:
            serials.add(int(match.group(1)))
    return serials


def _safe_controller_queued_identity(
        row: dict, controller_cycle_serial: int) -> dict:
    task_id = row.get("id", row.get("task_id"))
    checks = {
        "id": type(task_id) is int and task_id > 0,
        "name": isinstance(row.get("name"), str)
            and row["name"].startswith(PREFIX),
        "dedupe": isinstance(row.get("dedupe_key"), str)
            and f":{SOLVER}:{LIBRARY}:" in row["dedupe_key"],
        "project": row.get("project") == scheduler_client.MFT_PROJECT,
        "queued": row.get("status") == "queued",
        "attached": row.get("attached_at") in (None, ""),
        "launch": row.get("launch_started_at") in (None, ""),
        "started": row.get("started_at") in (None, ""),
        "finished": row.get("finished_at") in (None, ""),
        "allocation": row.get("allocation_id") is None
            and row.get("assigned_allocation") is None,
        "slurm": row.get("slurm_job_id") in (None, ""),
        "node": row.get("allocation_node_name") in (None, ""),
        "account": row.get("account_name") in (None, "")
            and row.get("requested_account_name") in (None, ""),
        "exit": row.get("exit_code") is None,
        "failure": row.get("failure_message") in (None, ""),
        "resources": row.get("cpus") == CPUS
            and row.get("memory_mb") == MEMORY_MB
            and row.get("timeout_seconds") == TIMEOUT_SECONDS
            and row.get("scheduling_profile") == "fea_bursty"
            and row.get("required_capability") == "conda:pyaedt2026v1"
            and row.get("env_profile") == "pyaedt2026v1"
            and row.get("gpus") == 0,
        "cycle": type(controller_cycle_serial) is int
            and controller_cycle_serial > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"controller-owned queued task {task_id} is unsafe: {checks}")
    return {
        field: (
            task_id if field == "id"
            else controller_cycle_serial if field == "controller_cycle_serial"
            else row.get(field))
        for field in _TARGET_CANCEL_IDENTITY_FIELDS
    }


def _controller_owned_queued_candidates(
        inventory: list[dict], feeder_state: dict,
        last_cycle_serial: int) -> list[dict]:
    """Return only queued rows proven by both feeder ledger and cycle journal."""
    rows = [row for row in inventory if isinstance(row, dict)]
    by_id = {}
    for row in rows:
        task_id = row.get("id", row.get("task_id"))
        if type(task_id) is not int or task_id <= 0 or task_id in by_id:
            raise RuntimeError("campaign inventory has invalid/duplicate task IDs")
        by_id[task_id] = row
    outstanding = feeder_state.get("outstanding")
    expected_rows = feeder_state.get("task_expected_rows")
    if not isinstance(outstanding, list) or not isinstance(expected_rows, dict):
        raise RuntimeError("continuous feeder ownership ledger is invalid")
    ledger_ids = {
        task_id for task_id in outstanding
        if type(task_id) is int and task_id > 0
    }
    wanted = {
        task_id for task_id, row in by_id.items()
        if row.get("status") == "queued" and task_id in ledger_ids
    }
    for task_id in wanted:
        if expected_rows.get(str(task_id)) != 1:
            raise RuntimeError(
                f"queued task {task_id} lacks one-row feeder ownership")
    proven = {}
    for serial in range(int(last_cycle_serial), 0, -1):
        if not wanted.difference(proven):
            break
        cycle = _load_cycle(_cycle_path(serial))
        if cycle is None or cycle.get("status") not in _CYCLE_TERMINAL_STATUSES:
            continue
        for event in cycle["formal_journal"]["events"]:
            task_id = event.get("task_id") if isinstance(event, dict) else None
            if task_id not in wanted:
                continue
            metadata = event.get("scheduler_metadata")
            live = by_id[task_id]
            checks = {
                "accepted": event.get("accepted_or_reconciled") is True,
                "ledger": event.get("ledger_committed") is True,
                "name": event.get("name") == live.get("name"),
                "dedupe": event.get("dedupe_key") == live.get("dedupe_key"),
                "metadata": isinstance(metadata, dict)
                    and metadata.get("id") == task_id
                    and metadata.get("name") == live.get("name")
                    and metadata.get("dedupe_key") == live.get("dedupe_key"),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"queued controller ownership {task_id} drifted: {checks}")
            if task_id in proven and proven[task_id] != serial:
                raise RuntimeError(
                    f"queued task {task_id} appears in multiple cycles")
            proven[task_id] = serial
    missing = wanted.difference(proven)
    if missing:
        raise RuntimeError(
            f"feeder-owned queued tasks lack terminal cycle proof: "
            f"{sorted(missing)}")
    candidates = [
        _safe_controller_queued_identity(by_id[task_id], proven[task_id])
        for task_id in wanted
    ]
    return sorted(candidates, key=lambda item: item["id"], reverse=True)


def _target_transition_readback(
        selected: list[dict], inventory: list[dict]) -> tuple[list[dict], list[int], list[int], list[int]]:
    by_id = {}
    for row in inventory:
        if not isinstance(row, dict):
            raise RuntimeError("target transition readback contains a non-object")
        task_id = row.get("id", row.get("task_id"))
        if type(task_id) is not int or task_id <= 0 or task_id in by_id:
            raise RuntimeError(
                "target transition readback has invalid/duplicate task IDs")
        by_id[task_id] = row
    readback = []
    cancelled = []
    skipped = []
    still_queued = []
    for prepared in selected:
        task_id = prepared["id"]
        row = by_id.get(task_id)
        if row is None:
            raise RuntimeError(
                f"target transition task {task_id} disappeared from inventory")
        status = str(row.get("status") or "").strip().lower()
        checks = {
            "name": row.get("name") == prepared.get("name"),
            "dedupe": row.get("dedupe_key") == prepared.get("dedupe_key"),
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "status": status in {
                "queued", "attaching", "running", "completed", "failed",
                "cancelled",
            },
            "resources": row.get("cpus") == CPUS
                and row.get("memory_mb") == MEMORY_MB
                and row.get("timeout_seconds") == TIMEOUT_SECONDS
                and row.get("scheduling_profile") == "fea_bursty"
                and row.get("required_capability") == "conda:pyaedt2026v1"
                and row.get("env_profile") == "pyaedt2026v1"
                and row.get("gpus") == 0,
        }
        if status in {"queued", "cancelled"}:
            checks.update({
                "attached": row.get("attached_at") in (None, ""),
                "launch": row.get("launch_started_at") in (None, ""),
                "started": row.get("started_at") in (None, ""),
                "allocation": row.get("allocation_id") is None
                    and row.get("assigned_allocation") is None,
                "slurm": row.get("slurm_job_id") in (None, ""),
                "exit": row.get("exit_code") is None,
                "failure": row.get("failure_message") in (None, ""),
            })
        if not all(checks.values()):
            raise RuntimeError(
                f"target transition readback task {task_id} drifted: {checks}")
        identity = {
            field: (
                task_id if field == "id"
                else prepared["controller_cycle_serial"]
                if field == "controller_cycle_serial"
                else row.get(field))
            for field in _TARGET_CANCEL_IDENTITY_FIELDS
        }
        readback.append(identity)
        if status == "cancelled":
            cancelled.append(task_id)
        else:
            skipped.append(task_id)
            if status == "queued":
                still_queued.append(task_id)
    readback.sort(key=lambda item: item["id"])
    return readback, sorted(cancelled), sorted(skipped), sorted(still_queued)


def _seal_target_transition_readback(
        transition: dict, inventory: list[dict], *,
        acknowledgement: dict | None, terminal_status: str,
        allow_still_queued: bool) -> None:
    readback, cancelled, skipped, still_queued = _target_transition_readback(
        transition["selected_tasks"], inventory)
    if still_queued and not allow_still_queued:
        raise RuntimeError(
            "queued-only cancellation left selected tasks queued: "
            f"{still_queued}")
    if acknowledgement is not None:
        acknowledged = acknowledgement.get("cancelled")
        if (not isinstance(acknowledged, list)
                or not set(acknowledged).issubset(cancelled)):
            raise RuntimeError(
                "queued-only cancellation acknowledgement/readback mismatch: "
                f"ack={acknowledged}, readback={cancelled}")
    transition["cancellation_acknowledgement"] = copy.deepcopy(acknowledgement)
    transition["readback"] = readback
    transition["readback_identity_sha256"] = _sha(readback)
    transition["cancelled_ids"] = cancelled
    transition["skipped_ids"] = skipped
    transition["settled_at"] = _now()
    _save_target_transition(
        _target_transition_path(transition["transition_serial"]),
        transition, terminal_status)


def _transition_active_snapshot(snapshot: dict) -> dict:
    required = (
        "project", "project_max_active_tasks", "project_counts",
        "project_active", "project_tagged_active", "legacy_active",
    )
    if not isinstance(snapshot, dict) or any(key not in snapshot for key in required):
        raise RuntimeError("dynamic target active snapshot is incomplete")
    cap = snapshot.get("project_max_active_tasks")
    active = snapshot.get("project_active")
    counts = snapshot.get("project_counts")
    if (type(cap) is not int or not TARGET_ACTIVE_MIN <= cap <= TARGET_ACTIVE_MAX
            or type(active) is not int or active < 0
            or not isinstance(counts, dict)
            or set(counts) != set(ACTIVE_STATUSES)
            or any(type(value) is not int or value < 0
                   for value in counts.values())
            or sum(counts.values()) != active):
        raise RuntimeError("dynamic target active snapshot is inconsistent")
    return {key: copy.deepcopy(snapshot[key]) for key in required}


def _new_target_transition(
        state: dict, project_contract: dict, snapshot: dict, *,
        action: str, eligible: list[dict], selected: list[dict]) -> dict:
    serial = int(state.get("target_transition_serial", 0)) + 1
    return {
        "schema_version": 1,
        "state_revision": 0,
        "artifact_type": "continuous-refill-dynamic-target-transition-v1",
        "transition_serial": serial,
        "project": scheduler_client.MFT_PROJECT,
        "plan_sha256": PLAN_SHA256,
        "target_policy": TARGET_POLICY_DYNAMIC,
        "action": action,
        "from_target": int(state["target_active"]),
        "to_target": int(project_contract["max_active_tasks"]),
        "project_updated_at": project_contract.get("updated_at"),
        "active_snapshot": _transition_active_snapshot(snapshot),
        "eligible_tasks": copy.deepcopy(eligible),
        "eligible_identity_sha256": _sha(eligible),
        "selected_tasks": copy.deepcopy(selected),
        "selected_identity_sha256": _sha(selected),
        "status": "prepared",
        "created_at": _now(),
        "updated_at": _now(),
        "cancellation_acknowledgement": None,
        "cancelled_ids": [],
        "skipped_ids": [],
        "readback": [],
        "readback_identity_sha256": None,
        "settled_at": None,
        "observed_superseding_cap": None,
        "error": None,
    }


def _state_cancelled_entry(readback: dict, transition_serial: int) -> dict:
    return {
        **copy.deepcopy(readback),
        "task_id": int(readback["id"]),
        "transition_serial": int(transition_serial),
    }


def _apply_terminal_target_transition(state: dict, transition: dict) -> None:
    if transition.get("status") not in _TARGET_TRANSITION_TERMINAL:
        raise RuntimeError("cannot apply a nonterminal target transition")
    expected_serial = int(state.get("target_transition_serial", 0)) + 1
    if (transition.get("transition_serial") != expected_serial
            or transition.get("from_target") != state.get("target_active")):
        raise RuntimeError("target transition is not the next state transition")
    before = copy.deepcopy(state)
    if transition["status"] == "completed":
        state["target_active"] = int(transition["to_target"])
    state["target_transition_serial"] = expected_serial
    state["target_transition_highwater"] = expected_serial
    if transition["action"] == "adopt_dynamic_policy":
        state["target_policy"] = TARGET_POLICY_DYNAMIC
        state["dynamic_policy_adopted_cycle"] = int(state["cycle_serial"])
    cancelled_by_id = {
        row["id"]: row for row in transition.get("readback", [])
        if row.get("id") in set(transition.get("cancelled_ids", []))
    }
    target_cancelled = copy.deepcopy(state.get("target_cancelled_tasks", {}))
    for task_id in transition.get("cancelled_ids", []):
        target_cancelled[str(task_id)] = _state_cancelled_entry(
            cancelled_by_id[task_id], expected_serial)
    state["target_cancelled_tasks"] = target_cancelled
    _validate_state_transition(before, state)
    _save_state(state)


def _strict_live_project_contract(*, expected_cap=None) -> dict:
    return scheduler_client.require_live_project_mutation_contract(
        expected_cap=expected_cap, require_full=True)


def _strict_live_project_snapshot(target: int) -> dict:
    return scheduler_client.live_project_submission_snapshot(
        target,
        require_exact_project_cap=True,
        require_full_project=True,
    )


def _settle_nonterminal_target_transition(
        transition: dict, live_cap: int) -> dict:
    """Reconcile a prepared/uncertain transition without broad cancellation."""
    path = _target_transition_path(transition["transition_serial"])
    selected = transition["selected_tasks"]
    acknowledgement = None
    if transition["status"] == "prepared" and selected and live_cap == transition["to_target"]:
        _save_target_transition(path, transition, "cancelling")
    inventory = feeder.campaign_inventory()
    _, _, _, queued = _target_transition_readback(selected, inventory)
    if queued and live_cap == transition["to_target"]:
        if transition["status"] != "cancelling":
            raise RuntimeError("target transition cancellation was not pre-sealed")
        acknowledgement = scheduler_client.cancel_queued_tasks_cas(queued)
        inventory = feeder.campaign_inventory()
    terminal = (
        "completed" if live_cap == transition["to_target"]
        else "superseded")
    if terminal == "superseded":
        transition["observed_superseding_cap"] = int(live_cap)
    _seal_target_transition_readback(
        transition, inventory,
        acknowledgement=acknowledgement,
        terminal_status=terminal,
        allow_still_queued=(terminal == "superseded"),
    )
    return transition


def _reconcile_target_transition_suffix(
        state: dict, live_cap: int) -> None:
    serials = _target_transition_serials_on_disk()
    applied = int(state.get("target_transition_highwater", 0))
    if applied:
        latest = _load_target_transition(_target_transition_path(applied))
        if latest is None or latest.get("status") not in _TARGET_TRANSITION_TERMINAL:
            raise RuntimeError(
                f"applied target transition {applied} is missing/nonterminal")
        expected_state_target = (
            latest["to_target"] if latest["status"] == "completed"
            else latest["from_target"])
        if state.get("target_active") != expected_state_target:
            raise RuntimeError(
                "controller target disagrees with latest transition journal")
    ahead = sorted(serial for serial in serials if serial > applied)
    if ahead and ahead != list(range(applied + 1, max(ahead) + 1)):
        raise RuntimeError(
            f"target transition history has a gap after {applied}: {ahead}")
    for serial in ahead:
        transition = _load_target_transition(_target_transition_path(serial))
        if transition is None:
            raise RuntimeError(f"target transition {serial} is missing")
        if transition["status"] not in _TARGET_TRANSITION_TERMINAL:
            transition = _settle_nonterminal_target_transition(
                transition, live_cap)
        _apply_terminal_target_transition(state, transition)


def _execute_new_target_transition(
        state: dict, bundle: dict, project_contract: dict,
        snapshot: dict) -> dict:
    from_target = int(state["target_active"])
    to_target = int(project_contract["max_active_tasks"])
    if state.get("target_policy", TARGET_POLICY_FIXED) == TARGET_POLICY_FIXED:
        if from_target != TARGET_ACTIVE or to_target != TARGET_ACTIVE:
            raise RuntimeError(
                "dynamic target adoption requires unchanged live cap300")
        action = "adopt_dynamic_policy"
        eligible = []
        selected = []
    elif to_target > from_target:
        action = "increase_target"
        eligible = []
        selected = []
    elif to_target < from_target:
        action = "decrease_target"
        inventory = feeder.campaign_inventory()
        feeder_state = _load_feeder_state(bundle, create=False)
        eligible = _controller_owned_queued_candidates(
            inventory, feeder_state, int(state["cycle_serial"]))
        excess = max(0, int(snapshot["project_active"]) - to_target)
        selected = eligible[:excess]
    else:
        raise RuntimeError("new target transition does not change/adopt policy")
    transition = _new_target_transition(
        state, project_contract, snapshot,
        action=action, eligible=eligible, selected=selected)
    path = _target_transition_path(transition["transition_serial"])
    _initialize_target_transition(path, transition)
    if selected:
        _save_target_transition(path, transition, "cancelling")
        acknowledgement = scheduler_client.cancel_queued_tasks_cas(
            [item["id"] for item in selected])
        inventory = feeder.campaign_inventory()
    else:
        acknowledgement = None
        inventory = []
    current_contract = _strict_live_project_contract()
    terminal = (
        "completed"
        if current_contract["max_active_tasks"] == to_target
        else "superseded")
    if terminal == "superseded":
        transition["observed_superseding_cap"] = int(
            current_contract["max_active_tasks"])
    _seal_target_transition_readback(
        transition, inventory,
        acknowledgement=acknowledgement,
        terminal_status=terminal,
        allow_still_queued=(terminal == "superseded"),
    )
    _apply_terminal_target_transition(state, transition)
    return transition


def _synchronize_dynamic_target(
        state: dict, bundle: dict, project_contract: dict,
        snapshot: dict) -> dict | None:
    """Reconcile prior journals, then apply the exact current project cap."""
    live_cap = int(project_contract["max_active_tasks"])
    _reconcile_target_transition_suffix(state, live_cap)
    policy = state.get("target_policy", TARGET_POLICY_FIXED)
    needs_adoption = policy == TARGET_POLICY_FIXED
    if policy not in {TARGET_POLICY_FIXED, TARGET_POLICY_DYNAMIC}:
        raise RuntimeError("target policy is invalid before synchronization")
    if needs_adoption or int(state["target_active"]) != live_cap:
        return _execute_new_target_transition(
            state, bundle, project_contract, snapshot)
    return None


def _cycle_one_cancelled_submission_is_reconciled(cycle: dict) -> bool:
    """Recognize the separately sealed/cancelled, never-started cycle 1."""
    if cycle.get("cycle_serial") != 1 or cycle.get("status") != "failed_closed":
        return False
    try:
        sealed = _rejected_submission_seal()
        events = cycle["formal_journal"]["events"]
        return bool(
            len(events) == 1
            and events[0].get("name") == sealed["name"]
            and events[0].get("dedupe_key") == sealed["actual_dedupe_key"]
            and sealed.get("status_after") == "cancelled"
            and sealed.get("attached_at_before") is None
            and sealed.get("started_at_before") is None
        )
    except Exception:
        return False


def _cycle_presubmit_get_timeout_is_terminal(cycle: dict) -> bool:
    """Accept only a durable batch cycle that failed before any mutation."""
    if not isinstance(cycle, dict) or cycle.get("status") != "failed_closed":
        return False
    error = str(cycle.get("error") or "")
    journal = cycle.get("formal_journal")
    return bool(
        error.startswith(
            "SchedulerError: scheduler request failed for /api/tasks: ")
        and "Read timed out." in error
        and error.endswith("(read timeout=30)")
        and isinstance(journal, dict)
        and journal.get("batch_commit") is True
        and journal.get("entered") is True
        and journal.get("completed") is False
        and journal.get("events") == []
        and journal.get("submitted_count") == 0
    )


def _latest_cycle_is_safe_presubmit_get_timeout() -> bool:
    """Return true only when durable controller state points at that proof."""
    try:
        state = _load_state(create=False)
        serial = state.get("cycle_serial")
        if isinstance(serial, bool) or not isinstance(serial, int) or serial <= 0:
            return False
        cycle = _load_cycle(_cycle_path(serial))
        return bool(
            isinstance(cycle, dict)
            and cycle.get("cycle_serial") == serial
            and _cycle_presubmit_get_timeout_is_terminal(cycle))
    except Exception:
        return False



def _cycle_serials_on_disk() -> set[int]:
    serials = set()
    if not CYCLE_ROOT.exists():
        return serials
    for item in CYCLE_ROOT.iterdir():
        match = _CYCLE_ARTIFACT.fullmatch(item.name)
        if match is not None:
            serials.add(int(match.group(1)))
    return serials


def _assert_no_unreconciled_cycles(
        last_controller_serial: int, terminal_highwater: int = 0) -> int:
    """Fail closed on every unaudited or controller-ahead cycle journal.

    A successfully authenticated terminal prefix is sealed into the durable
    controller state before another scheduler mutation is allowed.  Re-reading
    that immutable prefix on every 60-second cycle made one refill pass take
    roughly ten minutes on RaiDrive once hundreds of journals existed.  Only
    the suffix after that sealed high-water mark can contain a newly
    interrupted submission; artifacts ahead of controller state are still
    discovered from the directory and fail closed.
    """
    last_controller_serial = int(last_controller_serial)
    terminal_highwater = int(terminal_highwater)
    if not 0 <= terminal_highwater <= last_controller_serial:
        raise RuntimeError(
            "terminal cycle high-water is outside controller state: "
            f"{terminal_highwater}>{last_controller_serial}")
    serials = set(range(terminal_highwater + 1, last_controller_serial + 1))
    serials.update(
        serial for serial in _cycle_serials_on_disk()
        if serial > terminal_highwater)
    unresolved = []
    for serial in sorted(serials):
        path = _cycle_path(serial)
        try:
            cycle = _load_cycle(path)
        except Exception as exc:
            unresolved.append(
                f"cycle-{serial:06d}:unreadable:{type(exc).__name__}:{exc}")
            continue
        if cycle is None:
            if serial <= last_controller_serial:
                unresolved.append(f"cycle-{serial:06d}:missing")
            continue
        status = cycle["status"]
        if serial > last_controller_serial:
            unresolved.append(
                f"cycle-{serial:06d}:ahead_of_controller_state:{status}")
        elif (status not in _CYCLE_TERMINAL_STATUSES
                and not _cycle_one_cancelled_submission_is_reconciled(cycle)
                and not _cycle_presubmit_get_timeout_is_terminal(cycle)):
            unresolved.append(f"cycle-{serial:06d}:{status}")
    if unresolved:
        raise RuntimeError(
            "unreconciled continuous refill cycle journal(s); explicit reviewed "
            "reconciliation is required before any scheduler mutation: "
            + "; ".join(unresolved))
    return last_controller_serial


def _validated_no_mutation_evidence(
        serial: int, evidence: dict, reviewed_evidence_sha256: str) -> dict:
    if not isinstance(evidence, dict):
        raise RuntimeError("cycle reconciliation evidence is not an object")
    unsigned = copy.deepcopy(evidence)
    stored_sha = unsigned.pop("evidence_sha256", None)
    computed_sha = _sha(unsigned)
    feeder_before = evidence.get("feeder_before")
    feeder_after = evidence.get("feeder_after")
    scheduler = evidence.get("scheduler")
    required_feeder = {
        "state_revision", "serial", "candidate_cursor", "submitted_samples",
    }
    checks = {
        "schema": evidence.get("schema") == RECONCILIATION_EVIDENCE_SCHEMA,
        "cycle": evidence.get("cycle_serial") == serial,
        "action": evidence.get("action") == "reconciled_no_mutation",
        "observed": bool(str(evidence.get("observed_at") or "").strip()),
        "controller_stopped": evidence.get("controller_stopped") is True,
        "feeder_before": isinstance(feeder_before, dict)
            and required_feeder.issubset(feeder_before),
        "feeder_unchanged": isinstance(feeder_after, dict)
            and feeder_after == feeder_before,
        "scheduler": isinstance(scheduler, dict),
        "no_matching_tasks": isinstance(scheduler, dict)
            and scheduler.get("matching_task_ids") == [],
        "no_later_names": isinstance(scheduler, dict)
            and scheduler.get("production_names_above_feeder_serial") == [],
        "max_serial": isinstance(scheduler, dict)
            and isinstance(feeder_before, dict)
            and scheduler.get("max_production_serial") == feeder_before.get("serial"),
        "artifacts": isinstance(evidence.get("interrupted_artifacts"), list),
        "reviewed_sha": isinstance(reviewed_evidence_sha256, str)
            and len(reviewed_evidence_sha256) == 64
            and stored_sha == reviewed_evidence_sha256
            and computed_sha == reviewed_evidence_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"cycle no-mutation reconciliation evidence failed: {checks}")
    return copy.deepcopy(evidence)


def _publish_reconciled_no_mutation(
        serial: int, evidence: dict, reviewed_evidence_sha256: str) -> dict:
    """Append one reviewed no-mutation resolution; never touch campaign state."""
    if type(serial) is not int or serial <= 0:
        raise RuntimeError("reconciliation cycle serial is invalid")
    reviewed = _validated_no_mutation_evidence(
        serial, evidence, reviewed_evidence_sha256)
    path = _cycle_path(serial)
    with FileLock(str(STATE_PATH) + ".lock", timeout=30):
        cycle = _load_cycle(path)
        if cycle is None:
            raise RuntimeError(f"cannot reconcile a missing cycle journal: {path}")
        if (cycle["status"] == "reconciled_no_mutation"
                and (cycle.get("reconciliation") or {}).get("evidence_sha256")
                == reviewed_evidence_sha256):
            # Safe process-restart idempotency: return the already-sealed
            # generation, but never append or rewrite it.
            return copy.deepcopy(cycle)
        if cycle["status"] in _CYCLE_TERMINAL_STATUSES:
            raise RuntimeError(
                f"cycle journal is already terminal and cannot be reconciled again: {path}")
        if cycle["status"] in (
                "accepted_readback_pending_commit", "ledger_committed"):
            raise RuntimeError(
                f"cycle {serial} records an accepted task; no-mutation "
                "reconciliation is forbidden")
        updated = copy.deepcopy(cycle)
        journal = updated["formal_journal"]
        if int(journal.get("submitted_count") or 0) != 0:
            raise RuntimeError(
                f"cycle {serial} records submitted work; no-mutation "
                "reconciliation is forbidden")
        if any(event.get("accepted_or_reconciled") is True
               or event.get("ledger_committed") is True
               or event.get("task_id") is not None
               for event in journal["events"]):
            raise RuntimeError(
                f"cycle {serial} contains an accepted event; no-mutation "
                "reconciliation is forbidden")
        journal["submitted_count"] = 0
        journal["completed"] = True
        journal["stop_reason"] = "reviewed_reconciled_no_mutation"
        updated["reconciliation"] = {
            "action": "reconciled_no_mutation",
            "published_at": _now(),
            "evidence_sha256": reviewed_evidence_sha256,
            "evidence": reviewed,
        }
        _save_cycle(path, updated, "reconciled_no_mutation")
        return copy.deepcopy(updated)


@contextmanager
def _feeder_io(bundle: dict, cycle_path: Path, cycle: dict, journal: dict):
    original = {
        "STATE": feeder.STATE,
        "load_state": feeder.load_state,
        "save_state": feeder.save_state,
        "next": feeder.next_valid_candidate,
        "submit": feeder.submit,
    }
    snapshot = _load_feeder_state(bundle, create=True)
    plan_tasks = bundle["plan"]["tasks"]
    used_names = set(snapshot["used_names"])
    used_params = set(snapshot["used_params_sha256"])
    used_dedupes = set(snapshot["used_dedupe_keys"])
    candidate_index = int(snapshot["serial"]) - INITIAL_SERIAL
    pending = []
    submission_index = 0

    def load_state():
        return copy.deepcopy(_load_feeder_state(bundle, create=True))

    def next_candidate(cursor=0, seed=SEED, max_attempts=1000):
        nonlocal candidate_index
        for _ in range(max_attempts):
            next_cursor, raw_index, params = original["next"](
                cursor, seed=seed, max_attempts=max_attempts)
            _candidate_contract(params, f"refill_raw[{raw_index}]")
            params_sha = pinned_pilot.candidate_digest(params)
            if (params_sha in used_params
                    or any(item["params_sha256"] == params_sha
                           for item in pending)):
                cursor = next_cursor
                continue
            planned_index = candidate_index + len(pending)
            if planned_index < len(plan_tasks):
                expected = plan_tasks[planned_index]
                checks = {
                    "cursor": expected["candidate_cursor_after"] == next_cursor,
                    "raw": expected["candidate_raw_index"] == raw_index,
                    "params": expected["params"] == params,
                    "params_sha": expected["params_sha256"] == params_sha,
                }
                if not all(checks.values()):
                    raise RuntimeError(
                        f"sealed plan candidate {planned_index} drifted: {checks}")
                # Scheduler dedupe hashes the submitted JSON byte ordering.
                # Preserve the sealed plan's exact key order, not merely dict
                # value equality, so readback dedupe is byte-for-byte exact.
                submission_params = copy.deepcopy(expected["params"])
            else:
                # Beyond the sealed 300, impose one stable ordering rather
                # than inheriting generator implementation insertion order.
                submission_params = {
                    key: copy.deepcopy(params[key]) for key in sorted(params)}
            pending.append({"params_sha256": params_sha, "index": planned_index})
            return next_cursor, raw_index, submission_params
        raise RuntimeError("unable to find a unique strict b171 candidate")

    def persist_cycle(status: str):
        cycle["formal_journal"] = copy.deepcopy(journal)
        _save_cycle(cycle_path, cycle, status)

    def submit(name, workdir, params, solver_revision, library_revision, **kwargs):
        nonlocal submission_index
        if not journal["events"] or not pending:
            raise RuntimeError("submission has no pre-mutation journal identity")
        batch_commit = journal.get("batch_commit") is True
        item_index = submission_index if batch_commit else len(pending) - 1
        if item_index >= len(pending) or item_index >= len(journal["events"]):
            raise RuntimeError("submission index exceeds the durable batch plan")
        event = journal["events"][item_index]
        pending_item = pending[item_index]
        if batch_commit and submission_index == 0:
            for planned_event, planned_item in zip(journal["events"], pending):
                planned_event["params_sha256"] = planned_item["params_sha256"]
        else:
            event["params_sha256"] = pending_item["params_sha256"]
        event["mutation_about_to_start_at"] = _now()
        if not batch_commit or submission_index == 0:
            persist_cycle("mutation_about_to_submit")
        task_id = original["submit"](
            name, workdir, params, solver_revision, library_revision, **kwargs)
        expected = (
            plan_tasks[pending_item["index"]]
            if pending_item["index"] < len(plan_tasks)
            else {"name": name, "dedupe_key": event["dedupe_key"]}
        )
        event["scheduler_metadata"] = production._task_metadata(task_id, expected)
        event["readback_audited_at"] = _now()
        if batch_commit:
            submission_index += 1
        else:
            persist_cycle("accepted_readback_pending_commit")
        return task_id

    def save_state(state):
        nonlocal candidate_index
        if not pending or not journal["events"]:
            raise RuntimeError("durable feeder commit has no pending identity")
        batch_commit = journal.get("batch_commit") is True
        items = list(pending) if batch_commit else [pending[-1]]
        events = (
            journal["events"][:len(items)]
            if batch_commit else [journal["events"][-1]])
        if batch_commit and submission_index != len(items):
            raise RuntimeError("batch ledger commit precedes scheduler readback completion")
        for item, event in zip(items, events):
            name = str(event["name"])
            dedupe = str(event["dedupe_key"])
            params_sha = str(item["params_sha256"])
            if (event.get("accepted_or_reconciled") is not True
                    or event.get("task_id") is None):
                raise RuntimeError("batch ledger commit lacks an accepted task ID")
            if name in used_names or dedupe in used_dedupes or params_sha in used_params:
                raise RuntimeError("continuous refill identity would be duplicated")
            used_names.add(name)
            used_dedupes.add(dedupe)
            used_params.add(params_sha)
        state["used_names"] = sorted(used_names)
        state["used_dedupe_keys"] = sorted(used_dedupes)
        state["used_params_sha256"] = sorted(used_params)
        _save_feeder_state(state)
        candidate_index += len(items)
        if batch_commit:
            pending.clear()
        else:
            pending.pop()
        for event in events:
            event["ledger_committed"] = True
        if batch_commit:
            journal["submitted_count"] = len(events)
        persist_cycle("ledger_committed")

    feeder.STATE = str(FEEDER_STATE_PATH)
    feeder.load_state = load_state
    feeder.save_state = save_state
    feeder.next_valid_candidate = next_candidate
    feeder.submit = submit
    try:
        yield
    finally:
        feeder.STATE = original["STATE"]
        feeder.load_state = original["load_state"]
        feeder.save_state = original["save_state"]
        feeder.next_valid_candidate = original["next"]
        feeder.submit = original["submit"]


def _strict_recovery_result(result: dict, planned: dict) -> bool:
    if (not scheduler_client.is_valid_result(
            result, expected_revision=SOLVER,
            expected_library_revision=LIBRARY)
            or not scheduler_client.result_matches_params(
                result, planned["effective_params"])
            or rapid_campaign.thermal_saturation_columns(result)):
        return False
    try:
        forensic = json.loads(result["thermal_dispatch_forensic_json"])
        attempts = forensic["attempts"]
        final = forensic["final_convergence"]
        if (forensic.get("schema") != "thermal-dispatch-forensic-v1"
                or not isinstance(attempts, list)
                or not 1 <= len(attempts) <= 2):
            return False
        for index, attempt in enumerate(attempts, start=1):
            identity = attempt.get("identity")
            if (not isinstance(identity, dict)
                    or attempt.get("attempt") != index
                    or identity.get("design") != "icepak_thermal"
                    or identity.get("design_type") != "Icepak"
                    or identity.get("setups") != ["ThermalSetup"]
                    or identity.get("wrapper_setups") != ["ThermalSetup"]):
                return False
            if index < len(attempts) and (
                    attempt.get("dispatch_status") not in ("false", "exception")
                    or attempt.get("monitor_reason") != "monitor_missing"
                    or attempt.get("native_running") is not False):
                return False
        last = attempts[-1]
        monitor_file = str(result.get("thermal_monitor_file") or "").strip()
        return bool(
            int(result.get("thermal_solve_attempts")) == len(attempts)
            and last.get("dispatch_status") == "success"
            and monitor_file
            and last.get("monitor_file") == monitor_file
            and last.get("monitor_reason") == "converged"
            and final.get("converged") == 1
            and final.get("reason") == "converged"
            and final.get("monitor_file") == monitor_file
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _sealed_recovery_outcomes(gate: dict, tasks: list[dict]) -> str:
    """Apply only the root-reviewed, durable terminal-gate verdicts.

    The terminal-gate builder is the sole recovery stdout consumer.  Its
    sealed rows are therefore the restart-safe recovery outcome cache; this
    controller must never reconstruct a verdict from an old mutable
    ``last_evidence`` entry (notably the legacy task-28080 ``valid`` row).
    """
    gate_sha = gate.get("gate_sha256")
    rows = gate.get("tasks")
    if (not isinstance(gate_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", gate_sha) is None
            or not isinstance(rows, list)
            or len(rows) != len(tasks)):
        raise RuntimeError("sealed recovery4 outcome cache is incomplete")
    for index, (outcome, row) in enumerate(zip(tasks, rows), start=1):
        result_sha = row.get("result_sha256") if isinstance(row, dict) else None
        checks = {
            "task_id": isinstance(row, dict)
                and row.get("task_id") == outcome["task_id"],
            "name": isinstance(row, dict)
                and row.get("name") == outcome["name"],
            "status": isinstance(row, dict)
                and row.get("status") == "completed",
            "result_state": isinstance(row, dict)
                and row.get("result_state") == scheduler_client.RESULT_VALID,
            "strict_valid": isinstance(row, dict)
                and row.get("strict_valid") is True,
            "result_sha256": isinstance(result_sha, str)
                and re.fullmatch(r"[0-9a-f]{64}", result_sha) is not None,
            "effective_params_match": isinstance(row, dict)
                and row.get("effective_params_match") is True,
            "saturation_columns": isinstance(row, dict)
                and row.get("saturation_columns") == [],
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"sealed recovery4 outcome cache row {index} drifted: {checks}")
        outcome.update({
            "state": "valid",
            "result_state": row["result_state"],
            "strict_valid": True,
            "result_sha256": result_sha,
            "effective_params_match": True,
            "saturation_columns": [],
            "terminal_gate_sha256": gate_sha,
        })
    return gate_sha


def _recovery_live_evidence(
        bundle: dict, reviewed_recovery_gate_sha: str | None = None) -> dict:
    if (reviewed_recovery_gate_sha is not None
            and (not isinstance(reviewed_recovery_gate_sha, str)
                 or re.fullmatch(
                     r"[0-9a-f]{64}", reviewed_recovery_gate_sha) is None)):
        raise RuntimeError("reviewed recovery4 terminal-gate SHA is invalid")
    plan_tasks = bundle["recovery_plan"].get("tasks", [])
    submitted = bundle["recovery_submission"].get("tasks", [])
    if len(plan_tasks) != 4 or len(submitted) != 4:
        raise RuntimeError("recovery4 sealed evidence does not contain four tasks")
    tasks = []
    reasons = []
    for planned, expected in zip(plan_tasks, submitted):
        task_id = int(expected["task_id"])
        if (planned.get("name") != expected.get("name")
                or planned.get("dedupe_key") != expected.get("dedupe_key")):
            raise RuntimeError(f"recovery4 sealed identity {task_id} drifted")
        row = production._task_detail(task_id)
        status = str(row.get("status") or "").strip().lower()
        checks = {
            "id": row.get("id", row.get("task_id")) == task_id,
            "name": row.get("name") == expected["name"],
            "dedupe": row.get("dedupe_key") == expected["dedupe_key"],
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "cpus": row.get("cpus") == CPUS,
            "memory_mb": row.get("memory_mb") == MEMORY_MB,
            "timeout_seconds": row.get("timeout_seconds") == TIMEOUT_SECONDS,
            "env_profile": row.get("env_profile") == "pyaedt2026v1",
            "required_capability": row.get("required_capability") == "conda:pyaedt2026v1",
            "scheduling_profile": row.get("scheduling_profile") == "fea_bursty",
            "gpus": row.get("gpus") == 0,
            "remote_cwd": row.get("remote_cwd")
                == "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        }
        if not all(checks.values()):
            raise RuntimeError(f"live recovery4 task {task_id} identity drifted: {checks}")
        outcome = {"task_id": task_id, "name": expected["name"], "status": status}
        if status in ACTIVE_STATUSES:
            outcome["state"] = "active"
        elif status == "completed":
            # Metadata alone cannot authenticate a result.  In particular,
            # never reuse the legacy mutable-state verdict for task 28080.
            # The all-four terminal gate below is the only durable authority.
            outcome["state"] = "terminal_unreviewed"
        elif status in ("failed", "cancelled"):
            outcome["state"] = "invalid"
            exit_code = row.get("exit_code")
            try:
                normalized_exit_code = int(exit_code) if exit_code is not None else None
            except (TypeError, ValueError):
                normalized_exit_code = None
            outcome["exit_code"] = normalized_exit_code
            if status == "failed" and normalized_exit_code == 124:
                outcome["failure_class"] = "timeout"
                reasons.append(
                    f"recovery4_terminal_timeout:{task_id}:exit124")
            else:
                outcome["failure_class"] = status
                reasons.append(f"recovery4_terminal_failure:{task_id}:{status}")
        else:
            raise RuntimeError(
                f"live recovery4 task {task_id} has unknown status {status!r}")
        tasks.append(outcome)
    wait_reasons = []
    gate_sha = None
    if all(row["status"] == "completed" for row in tasks) and not reasons:
        if reviewed_recovery_gate_sha is None:
            wait_reasons.append("recovery4_terminal_gate_not_root_reviewed")
        else:
            gate = production._load_gate(
                production.GATE_PATH, reviewed_recovery_gate_sha,
                bundle["recovery_submission"], required=True)
            gate_sha = _sealed_recovery_outcomes(gate, tasks)
            if gate_sha != reviewed_recovery_gate_sha:
                raise RuntimeError("reviewed recovery4 terminal-gate SHA drifted")
    return {
        "tasks": tasks,
        "active": sum(row["state"] == "active" for row in tasks),
        "completed_valid": sum(row["state"] == "valid" for row in tasks),
        "reasons": sorted(set(reasons)),
        "wait_reasons": wait_reasons,
        "terminal_gate_sha256": gate_sha,
    }


def _inspect_production_with_retry(
        inventory: list[dict], solver: str, library: str,
        cached_outcomes: dict | None = None, *, attempts: int = 8,
        retry_seconds: float = 5, sleeper=time.sleep) -> dict:
    """Retry only a transient HTTP-429 while terminal output is unavailable."""
    attempts = int(attempts)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return rapid_campaign.inspect_production_tasks(
                inventory, solver, library, cached_outcomes=cached_outcomes)
        except Exception as exc:
            message = str(exc).lower()
            retryable = (
                "result is unavailable" in message
                and ("429" in message or "too many requests" in message)
            )
            if not retryable or attempt + 1 >= attempts:
                raise
            sleeper(float(retry_seconds))
    raise AssertionError("production inspection retry loop exhausted")


def _production_health_cohort(
        production_state: dict,
        authenticated_target300_rollback_ids: set[int] | None = None,
        authenticated_dynamic_target_cancelled_ids: set[int] | None = None) -> dict:
    """Select terminal health evidence by scheduler task start time.

    Lifetime outcomes remain authoritative for quarantine/accounting.  Only
    operational health gates use work started after the scheduler memory fix.
    A terminal without durable scheduler start metadata is ambiguous and must
    fail closed instead of being silently placed before the cutoff.
    """
    if not isinstance(production_state, dict):
        raise RuntimeError("production health state is not an object")
    tasks = production_state.get("tasks", [])
    outcomes = production_state.get("outcomes", [])
    if not isinstance(tasks, list) or not isinstance(outcomes, list):
        raise RuntimeError("production health tasks/outcomes are invalid")
    metadata = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise RuntimeError("production health task metadata is not an object")
        task_id = task.get("id", task.get("task_id"))
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or task_id in metadata):
            raise RuntimeError("production health task metadata has an invalid/duplicate ID")
        metadata[task_id] = task

    rollback_ids = set(authenticated_target300_rollback_ids or ())
    if rollback_ids and rollback_ids != set(TARGET300_ROLLBACK_CANCELLED_IDS):
        raise RuntimeError("target300 rollback health authorization is incomplete")
    dynamic_cancelled_ids = set(
        authenticated_dynamic_target_cancelled_ids or ())
    selected = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise RuntimeError("production terminal outcome is not an object")
        task_id = outcome.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("production terminal outcome has an invalid task ID")
        task = metadata.get(task_id)
        if task is None:
            raise RuntimeError(
                f"production terminal task {task_id} has no scheduler metadata")
        if outcome.get("expected_failure_reason") == (
                "operator_cancelled_stale_prepolicy_launch"):
            if not rapid_campaign._is_operator_cancelled_stale_prepolicy_launch(
                    task):
                raise RuntimeError(
                    "operator-cancelled stale prepolicy classification drifted "
                    f"for task {task_id}")
            raw_launch_started_at = str(task.get("launch_started_at") or "").strip()
            try:
                launch_started_at = datetime.fromisoformat(
                    raw_launch_started_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise RuntimeError(
                    "operator-cancelled stale prepolicy task "
                    f"{task_id} has malformed launch_started_at") from exc
            if launch_started_at.tzinfo is None:
                launch_started_at = launch_started_at.replace(
                    tzinfo=timezone.utc)
            else:
                launch_started_at = launch_started_at.astimezone(timezone.utc)
            if launch_started_at >= _SCHEDULER_CPU_POLICY_CUTOVER_AT:
                raise RuntimeError(
                    "operator-cancelled stale launch is not prepolicy: "
                    f"task {task_id}")
            # Keep the cancellation in lifetime invalid accounting, but it is
            # an operator-reviewed scheduler launch incident rather than a
            # completed simulation attempt, so it contributes to neither side
            # of the current simulation valid-rate fraction.
            continue
        if outcome.get("expected_failure_reason") == (
                "resolved_scheduler_parent_cancel_incident"):
            if not rapid_campaign._is_resolved_scheduler_parent_cancel_incident(
                    task):
                raise RuntimeError(
                    "resolved scheduler parent-cancel classification drifted "
                    f"for task {task_id}")
            # The failed task remains lifetime-invalid.  The exact parent
            # allocation incident is operational evidence from the old
            # scheduler, not a verdict on the current fixed runtime.
            continue
        if outcome.get("expected_failure_reason") == (
                "sealed_old_timeout_contract_incident"):
            if not rapid_campaign._is_sealed_old_timeout_contract_incident(
                    task, outcome.get("error_message")):
                raise RuntimeError(
                    "sealed old-timeout classification drifted for task "
                    f"{task_id}")
            continue
        if outcome.get("expected_failure_reason") == (
                "user_target_rollback_cancelled"):
            if (task_id not in rollback_ids
                    or task.get("status") != "cancelled"
                    or task.get("attached_at") not in (None, "")
                    or task.get("launch_started_at") not in (None, "")
                    or task.get("started_at") not in (None, "")
                    or task.get("allocation_id") is not None
                    or task.get("assigned_allocation") is not None):
                raise RuntimeError(
                    "target300 rollback health classification drifted for "
                    f"task {task_id}")
            continue
        if outcome.get("expected_failure_reason") == (
                "operator_target_reduction_cancelled"):
            if (task_id not in dynamic_cancelled_ids
                    or task.get("status") != "cancelled"
                    or task.get("attached_at") not in (None, "")
                    or task.get("launch_started_at") not in (None, "")
                    or task.get("started_at") not in (None, "")
                    or task.get("allocation_id") is not None
                    or task.get("assigned_allocation") is not None):
                raise RuntimeError(
                    "dynamic target health classification drifted for "
                    f"task {task_id}")
            continue
        raw_started_at = task.get("started_at")
        if not isinstance(raw_started_at, str) or not raw_started_at.strip():
            raise RuntimeError(
                f"production terminal task {task_id} has no started_at")
        raw_started_at = raw_started_at.strip()
        if re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
                r"(?:Z|[+-]\d{2}:\d{2})?",
                raw_started_at) is None:
            raise RuntimeError(
                f"production terminal task {task_id} has malformed started_at")
        try:
            started_at = datetime.fromisoformat(
                raw_started_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(
                f"production terminal task {task_id} has malformed started_at"
            ) from exc
        # Scheduler DB timestamps without an offset are canonical UTC.
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        else:
            started_at = started_at.astimezone(timezone.utc)
        if started_at >= _PRODUCTION_HEALTH_COHORT_CUTOFF_AT:
            selected.append(outcome)

    terminal = len(selected)
    valid = sum(row.get("state") == "valid" for row in selected)
    return {
        "cutoff_started_at": PRODUCTION_HEALTH_COHORT_CUTOFF,
        "terminal": terminal,
        "valid": valid,
        "valid_rate": valid / terminal if terminal else None,
        "outcomes": selected,
    }


def _maintained_pool_action(
        *, target_active: int, logical_active: int,
        target_reached: bool, reasons: list[str],
        wait_reasons: list[str]) -> str:
    if target_reached:
        return "target_reached_stop"
    if reasons:
        return "manual_intervention"
    if wait_reasons:
        return "wait_health_evidence"
    if logical_active > target_active:
        return f"wait_natural_drain_to_{target_active}"
    return f"refill_{target_active}"


def _mature_production_policy(
        health_state: dict, outcomes: list[dict], health: dict,
        strict: dict) -> tuple[list[str], list[str], list[str]]:
    """Classify mature-campaign health without stopping replacement work.

    Strict collection remains the data-quality boundary. Sample failures,
    rolling-rate degradation, revision/saturation findings and collector lag
    are therefore observable alerts, not automatic fleet pauses. Structural
    errors are deliberately not caught here: malformed evidence and identity
    drift still raise and fail closed.
    """
    alerts = list(rapid_campaign._production_gate_reasons(health_state))
    strict_invalid = [
        row for row in outcomes
        if row["status"] == "completed" and row["state"] != "valid"
    ]
    if strict_invalid:
        alerts.append(
            "completed_strict_invalid:" + ",".join(
                str(row["task_id"]) for row in strict_invalid))
    if (health["terminal"] >= 20 and health["valid_rate"] is not None
            and health["valid_rate"] < 0.90):
        alerts.append(
            f"fleet_valid_rate_below_90pct:{health['valid_rate']:.3f}")
    if health["terminal"] >= 20 and not strict["pinned"]:
        alerts.append("strict_collector_not_pinned_to_b171")
    return sorted(set(alerts)), [], []


def _evidence(
        state: dict, bundle: dict,
        reviewed_recovery_gate_sha: str | None = None, *,
        target_active: int | None = None,
        project_snapshot: dict | None = None) -> dict:
    if target_active is None:
        target_active = int(state.get("target_active", TARGET_ACTIVE))
    if (isinstance(target_active, bool) or not isinstance(target_active, int)
            or not TARGET_ACTIVE_MIN <= target_active <= TARGET_ACTIVE_MAX):
        raise RuntimeError("evidence target is outside 1..300")
    rejected = _rejected_submission_evidence()
    inventory_all = [
        task for task in feeder.campaign_inventory()
        if task.get("id", task.get("task_id")) != rejected["task_id"]
    ]
    # A sealed rolling entry is only omitted after the scheduler itself reports
    # ``cancelled``.  Its preauthorization therefore cannot reduce live active
    # accounting, while unrelated cancelled/failed work remains in health.
    rolling_cancelled = rolling_recycle.authorized_cancelled_task_ids(
        inventory_all)
    external_cancellation = _external_stale_pin_cancellation_evidence(
        inventory_all)
    remote_step_cancellation = _remote_step_cancellation_evidence(inventory_all)
    operator_cancelled_stale_prepolicy = (
        _operator_cancelled_stale_prepolicy_evidence(inventory_all))
    resolved_scheduler_parent_cancel = (
        _resolved_scheduler_parent_cancel_evidence(inventory_all))
    sealed_old_timeout_contract = (
        _sealed_old_timeout_contract_evidence(inventory_all))
    target300_rollback = _target300_rollback_cancelled_evidence(inventory_all)
    dynamic_target_cancelled = _dynamic_target_cancelled_evidence(
        state, inventory_all)
    external_cancelled = {
        *external_cancellation["task_ids"],
        *remote_step_cancellation["task_ids"],
    }
    health_exclusions = rolling_cancelled | external_cancelled
    inventory = [
        task for task in inventory_all
        if task.get("id", task.get("task_id")) not in health_exclusions
    ]
    health_exclusion_keys = {str(task_id) for task_id in health_exclusions}
    cached_outcomes = {
        key: value for key, value in (state.get("task_outcomes") or {}).items()
        if (str(key) != str(rejected["task_id"])
            and str(key) not in health_exclusion_keys)
    }
    production_state = _inspect_production_with_retry(
        inventory, SOLVER, LIBRARY, cached_outcomes=cached_outcomes)
    _classify_target300_rollback_outcomes(production_state, target300_rollback)
    _classify_dynamic_target_cancelled_outcomes(
        production_state, dynamic_target_cancelled)
    recovery = _recovery_live_evidence(bundle, reviewed_recovery_gate_sha)
    outcomes = production_state["outcomes"]
    terminal = len(outcomes)
    valid = sum(row["state"] == "valid" for row in outcomes)
    valid_rate = valid / terminal if terminal else None
    health = _production_health_cohort(
        production_state,
        set(target300_rollback["task_ids"]),
        set(dynamic_target_cancelled["task_ids"]),
    )
    health_state = {**production_state, "outcomes": health["outcomes"]}
    strict = _strict_snapshot()
    # This is a mature, revision-pinned production campaign.  A bad sample is
    # quarantined by the strict collector and must not stop replacement work.
    # Keep health signals observable, but reserve controller termination for
    # hard execution/invariant failures rather than rolling-rate policy.
    recovery_alerts = sorted(set([
        *recovery["reasons"], *recovery["wait_reasons"],
    ]))
    production_alerts, reasons, wait_reasons = _mature_production_policy(
        health_state, outcomes, health, strict)
    target_reached = strict["pinned"] and strict["rows"] >= TARGET_STRICT_ROWS
    if project_snapshot is None:
        logical_active = int(production_state["active"])
        logical_counts = None
    else:
        normalized_snapshot = _transition_active_snapshot(project_snapshot)
        if normalized_snapshot["project_max_active_tasks"] != target_active:
            raise RuntimeError("evidence project cap/target mismatch")
        logical_active = int(normalized_snapshot["project_active"])
        logical_counts = copy.deepcopy(normalized_snapshot["project_counts"])
    action = _maintained_pool_action(
        target_active=target_active,
        logical_active=logical_active,
        target_reached=target_reached,
        reasons=reasons,
        wait_reasons=wait_reasons,
    )
    return {
        "time": _now(),
        "action": action,
        "paused": bool(reasons),
        "pause_reasons": reasons,
        "production_active_b171": int(production_state["active"]),
        "maintained_target_active": target_active,
        "logical_mft_active": logical_active,
        "logical_mft_active_counts": logical_counts,
        "production_terminal_b171": terminal,
        "production_valid_b171": valid,
        "production_valid_rate_b171": valid_rate,
        "production_lifetime_terminal_b171": terminal,
        "production_lifetime_valid_b171": valid,
        "production_lifetime_valid_rate_b171": valid_rate,
        "production_health_cohort_cutoff": health["cutoff_started_at"],
        "production_health_terminal_b171": health["terminal"],
        "production_health_valid_b171": health["valid"],
        "production_health_valid_rate_b171": health["valid_rate"],
        "rolling_recycle_cancelled_exclusions": sorted(rolling_cancelled),
        "external_stale_pin_cancellation": external_cancellation,
        "remote_step_cancellation": remote_step_cancellation,
        "operator_cancelled_stale_prepolicy_launch": (
            operator_cancelled_stale_prepolicy),
        "resolved_scheduler_parent_cancel_incident": (
            resolved_scheduler_parent_cancel),
        "sealed_old_timeout_contract_incident": (
            sealed_old_timeout_contract),
        "user_target_rollback_cancelled": target300_rollback,
        "operator_target_reduction_cancelled": dynamic_target_cancelled,
        "production_lifetime": {
            "terminal": terminal,
            "valid": valid,
            "valid_rate": valid_rate,
        },
        "production_health_cohort": {
            key: value for key, value in health.items() if key != "outcomes"
        },
        "recovery4_active": recovery["active"],
        "recovery4_completed_valid": recovery["completed_valid"],
        "recovery4_tasks": recovery["tasks"],
        "recovery4_wait_reasons": recovery["wait_reasons"],
        "recovery4_nonblocking_alerts": recovery_alerts,
        "production_nonblocking_alerts": production_alerts,
        "recovery4_terminal_gate_sha256": recovery.get("terminal_gate_sha256"),
        "wait_reasons": wait_reasons,
        "excluded_rejected_submission": rejected,
        "strict": strict,
        "task_outcomes": production_state["cache"],
    }


def _apply_evidence(state: dict, evidence: dict) -> None:
    state["paused"] = evidence["paused"]
    state["pause_reasons"] = list(evidence["pause_reasons"])
    state["task_outcomes"] = copy.deepcopy(evidence["task_outcomes"])
    state["last_evidence"] = {
        key: value for key, value in evidence.items() if key != "task_outcomes"
    }


def static_audit() -> dict:
    bundle = _static_bundle()
    local_recovery = _local_recovery_evidence()
    rejected = _rejected_submission_seal()
    feeder_state = _load_feeder_state(bundle, create=False)
    next_cursor, raw_index, params = pinned_pilot.next_valid_candidate(
        INITIAL_CURSOR, seed=SEED)
    first = bundle["plan"]["tasks"][0]
    if (next_cursor != first["candidate_cursor_after"]
            or raw_index != first["candidate_raw_index"]
            or params != first["params"]):
        raise RuntimeError("first concurrent300 candidate drifted from sealed plan")
    return {
        "mode": "static_audit_only",
        "scheduler_query_count": 0,
        "scheduler_mutation_count": 0,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "plan_sha256": PLAN_SHA256,
        "target_logical_mft_active": int(feeder_state.get(
            "last_dynamic_target", TARGET_ACTIVE)),
        "target_source": "scheduler project MFT_1MW_2026v1 max_active_tasks",
        "target_bounds": [TARGET_ACTIVE_MIN, TARGET_ACTIVE_MAX],
        "local_recovery": local_recovery,
        "rejected_identity_cancellation": {
            "task_id": rejected["task_id"],
            "cancellation_sha256": rejected["cancellation_sha256"],
            "never_started": rejected["attached_at_before"] is None
                and rejected["started_at_before"] is None,
            "excluded_from_production_health": True,
        },
        "active_statuses": list(ACTIVE_STATUSES),
        "active_definition": "MFT_1MW_2026v1 queued+attaching+running; all revisions; IPMSM excluded",
        "first_candidate": {
            "serial": INITIAL_SERIAL + 1,
            "cursor_before": INITIAL_CURSOR,
            "cursor_after": next_cursor,
            "raw_index": raw_index,
            "name": first["name"],
            "params_sha256": first["params_sha256"],
            "dedupe_key": first["dedupe_key"],
        },
        "durable_serial": int(feeder_state["serial"]),
        "durable_cursor": int(feeder_state["candidate_cursor"]),
        "execute_requires": {
            "reviewed_plan_sha": PLAN_SHA256,
            "explicit_dynamic_project_cap_authorization": True,
            "shared_mutation_lock": True,
            "live_deficit_recheck_inside_lock": True,
        },
    }


def run_once(*, execute: bool, reviewed_plan_sha: str | None,
             authorize_dynamic_project_cap: bool,
             reviewed_recovery_gate_sha: str | None = None) -> dict:
    _trace("static_audit:start")
    audit = static_audit()
    _trace("static_audit:complete")
    if not execute:
        return audit
    if (reviewed_plan_sha != PLAN_SHA256
            or authorize_dynamic_project_cap is not True):
        raise RuntimeError(
            "execute requires the reviewed plan and explicit dynamic-cap authorization")
    bundle = _static_bundle()
    _trace("static_bundle:complete")
    with FileLock(str(STATE_PATH) + ".lock", timeout=30):
        _trace("state_lock:acquired")
        state = _load_state(create=True)
        _trace(f"state_load:cycle={state['cycle_serial']}")
        # A nonzero cycle serial is durable proof that this same sealed
        # controller already passed ``deployment_audit`` before its first
        # scheduler mutation.  The submitted task payload remains pinned to
        # the exact solver/library SHAs, so repeating full local git status
        # scans every 60 seconds adds no mutation safety and can block for
        # minutes on RaiDrive.  Fresh campaigns still fail closed here.
        if state["cycle_serial"] == 0:
            production.deployment_audit()
        # A previous process may have stopped after durable authorization, an
        # accepted scheduler mutation, or a feeder-ledger commit.  Never infer
        # safety from the controller serial alone and never start another
        # cycle until the authoritative append-only journal is terminal.
        legacy_terminal_highwater = max(0, state["cycle_serial"] - 1)
        stored_highwater = state.get(
            "terminal_cycle_highwater", legacy_terminal_highwater)
        audited_highwater = _assert_no_unreconciled_cycles(
            state["cycle_serial"], stored_highwater)
        _trace(f"cycle_tail_audit:highwater={audited_highwater}")
        if state.get("terminal_cycle_highwater") != audited_highwater:
            # The immutable controller-state generation is the durable proof
            # that this terminal prefix was checked.  A crash before this save
            # merely causes the same suffix to be checked again.
            state["terminal_cycle_highwater"] = audited_highwater
            _save_state(state)
        if _promote_target400_state(state):
            _trace("state_migration:target250_to_target400:start")
            _save_state(state)
            _trace("state_migration:target250_to_target400:complete")
        if _migrate_target300_state(state):
            _trace("state_migration:target400_to_target300:start")
            _save_state(state)
            _trace("state_migration:target400_to_target300:complete")
        with scheduler_client.campaign_mutation_lock():
            _trace("campaign_mutation_lock:acquired")
            project_contract = _strict_live_project_contract()
            observed_target = int(project_contract["max_active_tasks"])
            project_snapshot = _strict_live_project_snapshot(observed_target)
            transition = _synchronize_dynamic_target(
                state, bundle, project_contract, project_snapshot)
            if transition is not None:
                _trace(
                    "target_transition:complete:"
                    f"{transition['from_target']}->{transition['to_target']}:"
                    f"{transition['status']}")
            target_active = int(state["target_active"])
            # Re-read after transition/cancellation. A direct API change that
            # bypasses the shared operator lock fails closed here.
            project_contract = _strict_live_project_contract(
                expected_cap=target_active)
            project_snapshot = _strict_live_project_snapshot(target_active)
            evidence = _evidence(
                state, bundle, reviewed_recovery_gate_sha,
                target_active=target_active,
                project_snapshot=project_snapshot,
            )
            _trace(
                "evidence:complete:active="
                f"{evidence.get('production_active_b171')}")
            _apply_evidence(state, evidence)
            _trace(f"evidence:applied:action={evidence.get('action')}")
            refill_action = f"refill_{target_active}"
            if evidence["action"] != refill_action:
                _trace("state_save:non_refill:start")
                _save_state(state)
                _trace("state_save:non_refill:complete")
                return {**audit, **state["last_evidence"], "mode": "execute", "mutation": None}
            state["cycle_serial"] += 1
            _trace(f"state_save:cycle_authorized:{state['cycle_serial']}:start")
            _save_state(state)
            _trace(f"state_save:cycle_authorized:{state['cycle_serial']}:complete")
            cycle_path = _cycle_path(state["cycle_serial"])
            if _cycle_history_exists(cycle_path):
                raise RuntimeError(f"continuous refill cycle already exists: {cycle_path}")
            journal = {"events": []}
            cycle = {
                "schema_version": 1,
                "cycle_serial": state["cycle_serial"],
                "created_at": _now(),
                "updated_at": _now(),
                "status": "authorized_pending",
                "plan_sha256": PLAN_SHA256,
                "target_active": target_active,
                "target_policy": TARGET_POLICY_DYNAMIC,
                "project_cap_observed": target_active,
                "evidence": state["last_evidence"],
                "formal_journal": journal,
                "error": None,
            }
            _initialize_cycle(cycle_path, cycle)
            decision = {
                "paused": False,
                "target_active": target_active,
                "action": refill_action,
                "production": {
                    "terminal": evidence["production_health_cohort"]["terminal"],
                    "valid": evidence["production_health_cohort"]["valid"],
                    "valid_rate": evidence["production_health_cohort"]["valid_rate"],
                },
            }
            strict_rows = int(evidence["strict"]["rows"])
            authorization = feeder._authorize_adopted_refill(
                decision,
                max_samples=MAX_SAMPLES,
                solver_revision=SOLVER,
                library_revision=LIBRARY,
                candidate_seed=SEED,
                local_passed=True,
                adoption_sha256=PLAN_SHA256,
                initial_count=0,
                cpus=CPUS,
                memory_mb=MEMORY_MB,
                timeout_seconds=TIMEOUT_SECONDS,
                evidence_mode=EVIDENCE_MODE,
                strict_rows=strict_rows,
                target_strict_rows=TARGET_STRICT_ROWS,
            )
            try:
                with _feeder_io(bundle, cycle_path, cycle, journal):
                    changed = feeder._step_from_adopted_controller(
                        MAX_SAMPLES,
                        authorization=authorization,
                        target=target_active,
                        buffer=0,
                        solver_revision=SOLVER,
                        library_revision=LIBRARY,
                        candidate_seed=SEED,
                        adoption_sha256=PLAN_SHA256,
                        initial_count=0,
                        cpus=CPUS,
                        memory_mb=MEMORY_MB,
                        timeout_seconds=TIMEOUT_SECONDS,
                        evidence_mode=EVIDENCE_MODE,
                        strict_rows=strict_rows,
                        target_strict_rows=TARGET_STRICT_ROWS,
                        journal=journal,
                    )
                cycle["formal_journal"] = copy.deepcopy(journal)
                _save_cycle(cycle_path, cycle, "completed")
            except BaseException as exc:
                cycle["error"] = f"{type(exc).__name__}: {exc}"
                cycle["formal_journal"] = copy.deepcopy(journal)
                try:
                    _save_cycle(cycle_path, cycle, "failed_closed")
                except BaseException as journal_exc:
                    raise RuntimeError(
                        "cycle failed and its failed-closed journal generation "
                        f"also failed: original={type(exc).__name__}: {exc}; "
                        f"journal={type(journal_exc).__name__}: {journal_exc}"
                    ) from exc
                raise
            return {
                **audit,
                **state["last_evidence"],
                "mode": "execute",
                "mutation": {
                    "cycle": state["cycle_serial"],
                    "journal": str(cycle_path.resolve()),
                    "submitted_count": int(journal.get("submitted_count") or 0),
                    "stop_reason": journal.get("stop_reason"),
                    "feeder_result": bool(changed),
                },
            }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewed-plan-sha")
    parser.add_argument("--authorize-dynamic-project-cap", action="store_true")
    parser.add_argument("--reviewed-recovery-gate-sha")
    parser.add_argument("--loop", type=int)
    args = parser.parse_args(argv)
    if args.loop is not None and (not args.execute or args.loop < 60):
        parser.error("--loop requires --execute and at least 60 seconds")
    if args.execute and (
            args.reviewed_plan_sha != PLAN_SHA256
            or not args.authorize_dynamic_project_cap):
        parser.error(
            "--execute requires exact --reviewed-plan-sha and "
            "--authorize-dynamic-project-cap")
    while True:
        cycle_started = time.monotonic()
        try:
            result = run_once(
                execute=args.execute,
                reviewed_plan_sha=args.reviewed_plan_sha,
                authorize_dynamic_project_cap=args.authorize_dynamic_project_cap,
                reviewed_recovery_gate_sha=args.reviewed_recovery_gate_sha,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            safe_presubmit_retry = bool(
                args.execute and args.loop is not None
                and _latest_cycle_is_safe_presubmit_get_timeout())
            failed = {
                "time": _now(),
                "mode": "execute" if args.execute else "static_audit_only",
                "action": "failed_closed",
                "error": f"{type(exc).__name__}: {exc}",
                "scheduler_mutation_count": None if args.execute else 0,
            }
            if args.execute:
                if safe_presubmit_retry:
                    failed.update({
                        "mutation_may_have_occurred": False,
                        "cycle_journal_root": str(CYCLE_ROOT.resolve()),
                        "reconcile_before_retry": False,
                        "automatic_retry": "durable_presubmit_get_timeout",
                    })
                else:
                    failed.update({
                        "mutation_may_have_occurred": True,
                        "cycle_journal_root": str(CYCLE_ROOT.resolve()),
                        "reconcile_before_retry": True,
                    })
            print(json.dumps(
                failed, ensure_ascii=False, sort_keys=True), flush=True)
            # Only an exact durable batch proof with no planned/submitted
            # events can continue. Every uncertain mutation still exits.
            if not safe_presubmit_retry:
                return 2
        if args.loop is None:
            return 0
        # ``--loop`` is a start-to-start cadence.  A live inventory/health
        # pass can itself take tens of seconds; sleeping the full interval
        # after that work left a completed-task deficit unfilled for roughly
        # one work duration plus one interval.  Keep the requested logical
        # pool close to 300 without overlapping mutation epochs.
        elapsed = max(0.0, time.monotonic() - cycle_started)
        time.sleep(max(0.0, float(args.loop) - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
