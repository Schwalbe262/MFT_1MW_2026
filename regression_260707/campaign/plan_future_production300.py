"""Build a sealed, scheduler-read-only SHA-b171 production-300 plan.

The previously reviewed SHA7873 plan remains the authoritative source for the
300 candidate draws.  This generator preserves those exact parameters and
cursor positions, then derives fresh names and dedupe identities for the
sealed SHA-b171 recovery deployment.  There is deliberately no scheduler
query or mutation path in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import pinned_pilot
import scheduler_client


OLD_SOLVER = "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
STALE_SOLVER = "7873ddddcf7ac7412d14c9e3ae216ed73b82fffe"
SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SEED = 260710
CURSOR_START = 2795
COUNT = 300
FIRST_SERIAL = 17612
PLAN_CREATED_AT = "2026-07-12T01:43:32+00:00"
OLD_FIRST_ID = 27755
OLD_LAST_ID = 28004
OLD_MANIFEST_SHA256 = "f1490f2cda497c9475fe079fb0a04e5adb7686c6f4c99ae28a0f946a918319a8"
OLD_MANIFEST_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.json"
)
STALE_PLAN_SHA256 = "b67027d4e3ac22fd79079b690ff01cdc626ddd9c7a0fc60ed62ce60afbeae026"
STALE_PLAN_PATH = HERE / "pilot_manifests" / (
    "production300-s7873ddd-le6b9b9d-seed260710-cursor2795.json"
)
RECOVERY_PLAN_SHA256 = "3e453deb61137c2d29c13bbbe8d5117b4c4111e5ea7e255d37dfd0d5e4444af5"
RECOVERY_PLAN_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.json"
)
RECOVERY_SUBMISSION_SHA256 = "fa951faa0cd29c3502e511f827ff0fc2573facc1413c76fb1ad3db0f689d5abc"
RECOVERY_SUBMISSION_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.submission.json"
)
RECOVERY_TASK_IDS = (28077, 28078, 28079, 28080)
PROJECT_HARD_CAP = 300
SOLVER_DEPLOYMENT_REFS = (
    "refs/heads/fix/mft-rx-block-fastpath-260712",
    "refs/heads/stabilize/mft-sim-260710",
)
LIBRARY_DEPLOYMENT_REFS = ("refs/heads/pyaedt_022",)
LIBRARY_REPO_ROOT = REPO_ROOT.parent / "pyaedt_library_mft_clean"
PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
RESOURCES = {
    "project": scheduler_client.MFT_PROJECT,
    "cpus": 4,
    "memory_mb": 65_536,
    "gpus": 0,
    "timeout_seconds": 14_400,
    "required_capability": "conda:pyaedt2026v1",
    "env_profile": "pyaedt2026v1",
    "scheduling_profile": "fea_bursty",
    "priority": 0,
    "remote_cwd": scheduler_client.GPFS_RUNS_REMOTE_CWD,
}


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sealed_json(path, expected_sha, label, seal_key):
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    stored = unsigned.pop(seal_key, None)
    if stored != expected_sha or _sha(unsigned) != expected_sha:
        raise RuntimeError(f"{label} seal mismatch")
    return payload


def _validate_recovery_evidence(recovery, recovery_plan):
    expected_deployment = {
        "solver": {
            "repo_root": str(REPO_ROOT.resolve()),
            "revision": SOLVER,
            "refs": list(SOLVER_DEPLOYMENT_REFS),
        },
        "library": {
            "repo_root": str(LIBRARY_REPO_ROOT.resolve()),
            "revision": LIBRARY,
            "refs": list(LIBRARY_DEPLOYMENT_REFS),
        },
    }
    if recovery.get("deployment") != expected_deployment:
        raise RuntimeError("recovery4 deployment evidence drifted")
    if recovery.get("resources") != RESOURCES:
        raise RuntimeError("recovery4 resource contract drifted")
    if recovery.get("scheduler_mutation_count") != 4 \
            or recovery.get("submission_policy") \
            != "all-four-in-one-mutation-lock-no-cancellation":
        raise RuntimeError("recovery4 submission policy drifted")
    if recovery.get("plan") != str(RECOVERY_PLAN_PATH.resolve()) \
            or recovery.get("root_reviewed_plan_sha256") != RECOVERY_PLAN_SHA256:
        raise RuntimeError("recovery4 reviewed-plan reference drifted")

    before = recovery.get("capacity_before")
    after = recovery.get("capacity_after")
    for label, snapshot in (("before", before), ("after", after)):
        if not isinstance(snapshot, dict) \
                or snapshot.get("project") != RESOURCES["project"] \
                or snapshot.get("project_max_active_tasks") != PROJECT_HARD_CAP \
                or snapshot.get("project_required_hard_cap") != PROJECT_HARD_CAP:
            raise RuntimeError(f"recovery4 {label} capacity contract drifted")
        active = int(snapshot.get("project_active", -1))
        if active < 0 or active > PROJECT_HARD_CAP:
            raise RuntimeError(f"recovery4 {label} active count exceeds cap")
    if int(after["project_active"]) - int(before["project_active"]) != 4:
        raise RuntimeError("recovery4 capacity delta does not match four submissions")

    journal_tasks = recovery.get("tasks", [])
    plan_tasks = recovery_plan.get("tasks", [])
    if len(journal_tasks) != 4 or len(plan_tasks) != 4:
        raise RuntimeError("recovery4 task evidence is incomplete")
    scheduler_resource_keys = (
        "project", "cpus", "memory_mb", "gpus", "timeout_seconds",
        "required_capability", "env_profile", "scheduling_profile",
        "remote_cwd",
    )
    for plan_task, journal_task in zip(plan_tasks, journal_tasks):
        if any(
            journal_task.get(key) != plan_task.get(key)
            for key in ("ordinal", "source_task_id", "name", "dedupe_key")
        ):
            raise RuntimeError("recovery4 task identity drifted")
        metadata = journal_task.get("scheduler_metadata", {})
        if metadata.get("id") != journal_task.get("task_id") \
                or metadata.get("name") != journal_task.get("name") \
                or metadata.get("dedupe_key") != journal_task.get("dedupe_key"):
            raise RuntimeError("recovery4 scheduler identity drifted")
        for key in scheduler_resource_keys:
            if metadata.get(key) != RESOURCES[key]:
                raise RuntimeError(f"recovery4 scheduler resource {key} drifted")
        expected_identity = f":{SOLVER}:{LIBRARY}:"
        if expected_identity not in str(journal_task.get("dedupe_key", "")):
            raise RuntimeError("recovery4 dedupe revision identity drifted")


def _load_predecessors():
    old = _sealed_json(
        OLD_MANIFEST_PATH, OLD_MANIFEST_SHA256, "SHA754 manifest",
        "manifest_sha256",
    )
    if old.get("solver_revision") != OLD_SOLVER \
            or old.get("library_revision") != LIBRARY \
            or old.get("task_count") != 250 \
            or old.get("candidate_cursor_end") != CURSOR_START \
            or old.get("last_serial") != FIRST_SERIAL - 1:
        raise RuntimeError("SHA754 predecessor identity drifted")
    stale = _sealed_json(
        STALE_PLAN_PATH, STALE_PLAN_SHA256, "SHA7873 production300 plan",
        "plan_sha256",
    )
    if stale.get("solver_revision") != STALE_SOLVER \
            or stale.get("library_revision") != LIBRARY \
            or stale.get("task_count") != COUNT \
            or stale.get("candidate_cursor_start") != CURSOR_START \
            or stale.get("first_serial") != FIRST_SERIAL:
        raise RuntimeError("SHA7873 production300 identity drifted")
    recovery_plan = _sealed_json(
        RECOVERY_PLAN_PATH, RECOVERY_PLAN_SHA256, "SHA-b171 recovery4 plan",
        "plan_sha256",
    )
    recovery = _sealed_json(
        RECOVERY_SUBMISSION_PATH, RECOVERY_SUBMISSION_SHA256,
        "SHA-b171 recovery4 submission journal", "submission_sha256",
    )
    if recovery.get("root_reviewed_plan_sha256") != RECOVERY_PLAN_SHA256 \
            or recovery.get("solver_revision") != SOLVER \
            or recovery.get("library_revision") != LIBRARY \
            or recovery.get("task_count") != 4:
        raise RuntimeError("recovery4 predecessor identity drifted")
    recovery_ids = [int(row["task_id"]) for row in recovery.get("tasks", [])]
    if recovery_ids != list(RECOVERY_TASK_IDS):
        raise RuntimeError("recovery4 task IDs drifted")
    if recovery_plan.get("solver_revision") != SOLVER \
            or recovery_plan.get("library_revision") != LIBRARY \
            or recovery_plan.get("task_count") != 4 \
            or recovery_plan.get("resources") != RESOURCES:
        raise RuntimeError("recovery4 plan identity drifted")
    _validate_recovery_evidence(recovery, recovery_plan)
    return old, stale, recovery_plan, recovery


def _profile():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT \
            or profile.get("cli_flags") != "--thermal --headless":
        raise RuntimeError("standard full thermal profile drifted")
    profile["timeout_seconds"] = RESOURCES["timeout_seconds"]
    return profile


def _validate_candidate(params, raw_index):
    primary_turns = int(params["N1_main"]) + int(params["N1_side"])
    if primary_turns > 8:
        raise RuntimeError(f"candidate primary turns exceed 8: {raw_index}")
    cw1 = float(params["cw1"])
    if not math.isfinite(cw1) or cw1 > 10.0:
        raise RuntimeError(f"candidate cw1 exceeds 10mm: {raw_index}/{cw1}")
    for key in ("wcp_t", "core_plate_t"):
        value = float(params[key])
        if not math.isfinite(value) or not 10.0 <= value <= 30.0:
            raise RuntimeError(f"candidate {key} outside [10,30]: {raw_index}/{value}")
    if tuple(float(params[key]) for key in ("wcp_pad_t", "core_plate_pad_t")) \
            != (2.0, 2.0):
        raise RuntimeError(f"candidate pad thickness drifted: {raw_index}")
    n2_main = int(params["N2_main"])
    nwl2_main = (
        n2_main * float(params["cw2"])
        + max(n2_main - 1, 0) * float(params["gap2"])
    ) if n2_main > 0 else 0.0
    sl2_main_x = 2.0 * float(params["l1"]) \
        + 2.0 * float(params["cc_w2c_space_x"])
    reference_mm = sl2_main_x + 2.0 * nwl2_main \
        + 2.0 * float(params["w2c_w1c_space_x"])
    if int(params["round_corner"]):
        reference_mm -= 2.0 * float(params["corner_radius"])
    length_mm = float(params["wcp_len_x"])
    pct = 100.0 * length_mm / reference_mm if reference_mm > 0 else float("nan")
    if not (math.isfinite(length_mm) and length_mm > 0 \
            and math.isfinite(pct) and 20.0 <= pct <= 80.0):
        raise RuntimeError(
            f"candidate winding plate length drifted: {raw_index}/"
            f"{length_mm}mm/{pct}%"
        )


def _validate_candidate_stability(records, stale, duplicate_candidates_skipped, cursor):
    stale_records = stale.get("tasks", [])
    if len(stale_records) != COUNT or len(records) != COUNT:
        raise RuntimeError("production300 candidate count drifted")
    stable_keys = (
        "index", "serial", "workdir", "candidate_cursor_before",
        "candidate_cursor_after", "candidate_raw_index", "params_sha256",
        "parameter_digest", "params", "effective_params",
    )
    for expected, actual in zip(stale_records, records):
        for key in stable_keys:
            if actual.get(key) != expected.get(key):
                raise RuntimeError(
                    f"production300 candidate drifted at index {actual.get('index')}: {key}"
                )
    if duplicate_candidates_skipped != stale.get("duplicate_candidates_skipped", []):
        raise RuntimeError("production300 duplicate-skip history drifted")
    if int(cursor) != int(stale.get("candidate_cursor_end", -1)):
        raise RuntimeError("production300 final candidate cursor drifted")


def build_plan():
    old, stale, recovery_plan, recovery = _load_predecessors()
    profile = _profile()
    if scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS != PROJECT_HARD_CAP:
        raise RuntimeError("local MFT project cap is not 300")
    predecessor_digests = {
        str(row.get("params_sha256")) for row in old.get("tasks", [])
    }
    predecessor_digests.update(
        str(row.get("source_params_sha256"))
        for row in recovery_plan.get("tasks", [])
    )
    prior_digests = set(predecessor_digests)
    cursor = CURSOR_START
    records = []
    duplicate_candidates_skipped = []
    while len(records) < COUNT:
        cursor_before = cursor
        cursor, raw_index, params = pinned_pilot.next_valid_candidate(
            cursor, seed=SEED,
        )
        params = json.loads(_canonical(params))
        _validate_candidate(params, raw_index)
        params_sha = pinned_pilot.candidate_digest(params)
        if params_sha in prior_digests:
            duplicate_candidates_skipped.append({
                "candidate_raw_index": int(raw_index),
                "params_sha256": params_sha,
            })
            continue
        prior_digests.add(params_sha)
        index = len(records)
        serial = FIRST_SERIAL + index
        name = f"{PREFIX}{serial:05d}"
        identity = scheduler_client.verification_submission_identity(
            name, params, profile, SOLVER, LIBRARY,
        )
        records.append({
            "index": index,
            "serial": serial,
            "name": name,
            "workdir": f"mft_p300_t{serial % 500:03d}",
            "candidate_cursor_before": int(cursor_before),
            "candidate_cursor_after": int(cursor),
            "candidate_raw_index": int(raw_index),
            "params_sha256": params_sha,
            "parameter_digest": identity["parameter_digest"],
            "dedupe_key": identity["dedupe_key"],
            "params": params,
            "effective_params": identity["merged"],
        })

    if len({row["name"] for row in records}) != COUNT \
            or len({row["dedupe_key"] for row in records}) != COUNT \
            or len({row["params_sha256"] for row in records}) != COUNT:
        raise RuntimeError("production300 identities are not unique")
    _validate_candidate_stability(
        records, stale, duplicate_candidates_skipped, cursor,
    )
    generated_digests = {row["params_sha256"] for row in records}
    if generated_digests & predecessor_digests:
        raise RuntimeError("production300 parameters overlap predecessor tasks")
    predecessor_names = {
        str(row.get("name"))
        for source in (old, stale, recovery_plan, recovery)
        for row in source.get("tasks", [])
    }
    predecessor_dedupe = {
        str(row.get("dedupe_key"))
        for source in (old, stale, recovery_plan, recovery)
        for row in source.get("tasks", [])
    }
    if {row["name"] for row in records} & predecessor_names:
        raise RuntimeError("production300 task names overlap predecessor plans")
    if {row["dedupe_key"] for row in records} & predecessor_dedupe:
        raise RuntimeError("production300 dedupe keys overlap predecessor plans")
    revision_identity = f":{SOLVER}:{LIBRARY}:"
    if any(
        not row["name"].startswith(PREFIX)
        or revision_identity not in row["dedupe_key"]
        for row in records
    ):
        raise RuntimeError("production300 revision identity drifted")
    plan = {
        "schema_version": 1,
        "created_at": PLAN_CREATED_AT,
        "mode": "plan_only",
        "submission_enabled": False,
        "scheduler_mutation_count": 0,
        "authorization": (
            "after recovery4 4/4 strict gate, replace the exact remaining "
            "SHA754 cohort and fill the logical MFT project to 300 SHA-b171 tasks"
        ),
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "candidate_cursor_start": CURSOR_START,
        "candidate_cursor_end": int(cursor),
        "task_count": COUNT,
        "first_serial": FIRST_SERIAL,
        "last_serial": FIRST_SERIAL + COUNT - 1,
        "task_prefix": PREFIX,
        "resources": RESOURCES,
        "capacity_contract": {
            "project": RESOURCES["project"],
            "project_max_active_tasks": PROJECT_HARD_CAP,
            "project_required_hard_cap": PROJECT_HARD_CAP,
            "planned_task_count": COUNT,
            "live_capacity_recheck_required_inside_mutation_lock": True,
        },
        "profile": {
            "path": str(PROFILE_PATH.resolve()),
            "file_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
            "effective_sha256": _sha(profile),
            "cli_flags": profile["cli_flags"],
            "param_overrides": profile["param_overrides"],
        },
        "predecessor": {
            "old_solver_revision": OLD_SOLVER,
            "old_manifest": str(OLD_MANIFEST_PATH.resolve()),
            "old_manifest_sha256": OLD_MANIFEST_SHA256,
            "old_task_id_range": [OLD_FIRST_ID, OLD_LAST_ID],
            "old_task_count": 250,
            "stale_production_plan": str(STALE_PLAN_PATH.resolve()),
            "stale_production_plan_sha256": STALE_PLAN_SHA256,
            "stale_solver_revision": STALE_SOLVER,
            "exact_candidate_fields_preserved": True,
            "recovery_plan": str(RECOVERY_PLAN_PATH.resolve()),
            "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
            "recovery_submission": str(RECOVERY_SUBMISSION_PATH.resolve()),
            "recovery_submission_sha256": RECOVERY_SUBMISSION_SHA256,
            "recovery_submission_policy": recovery["submission_policy"],
            "recovery_scheduler_mutation_count": recovery["scheduler_mutation_count"],
            "recovery_task_ids": [
                int(row["task_id"]) for row in recovery["tasks"]
            ],
            "recovery_deployment": recovery["deployment"],
            "recovery_capacity_after": recovery["capacity_after"],
        },
        "static_attestations": {
            "old_parameter_overlap_count": 0,
            "predecessor_name_overlap_count": 0,
            "predecessor_dedupe_overlap_count": 0,
            "candidate_identity_matches_stale_plan": True,
            "authoritative_recovery_submission_sealed": True,
            "authoritative_recovery_plan_sealed": True,
            "recovery_deployment_refs_match": True,
            "recovery_resource_contract_matches": True,
            "project_cap_300_attested": True,
        },
        "activation_requirements": {
            "recovery4_all_strict_valid": False,
            "failed_sources_exact_thermal_setup": False,
            "failed_sources_analyze_all_absent": False,
            "failed_sources_fresh_monitor": False,
            "known_good_nonregression": False,
            "deployment_gate_passed_inside_mutation_lock": False,
            "root_reviewed_plan_sha256": None,
            "cancel_only_exact_remaining_old_ids": True,
            "submit_under_one_campaign_mutation_lock": True,
        },
        "duplicate_candidates_skipped": duplicate_candidates_skipped,
        "tasks": records,
    }
    plan["plan_sha256"] = _sha(plan)
    return plan


def _write(path, plan):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    last_error = None
    for attempt in range(20):
        try:
            os.replace(staged, path)
            break
        except PermissionError as exc:
            last_error = exc
            if attempt == 19:
                raise
            time.sleep(0.25)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded != plan:
        raise RuntimeError(f"production300 plan readback mismatch: {path}") from last_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = build_plan()
    if args.output is not None:
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing != plan:
                raise RuntimeError(f"refusing to overwrite different plan: {args.output}")
        else:
            _write(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
