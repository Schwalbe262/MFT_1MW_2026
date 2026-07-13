"""Audit first; explicitly replace SHA754 with the sealed SHA-b171 production300.

The default mode is scheduler-read-free and has no mutation path.  Execution
requires the exact root-reviewed production plan and a separately sealed,
root-reviewed terminal recovery gate.  All scheduler mutations then happen
under one shared campaign mutation lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import deployment_gate
import pinned_pilot
import scheduler_client


SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
OLD_SOLVER = "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"

PLAN_SHA256 = "b24e2a9b00caa22bbec8793f4dbd99de51362fac87f9e9509358610abe9982d0"
PLAN_FILE_SHA256 = "1e7af20277cfd18a3e6a692e6a6c2d43810475a9eef27fb1c1746671d7e69d85"
PLAN_PATH = HERE / "pilot_manifests" / (
    "production300-sb171c7c-le6b9b9d-seed260710-cursor2795.json"
)
PLAN_PREFIX = "mft-camp-sb171c7c-le6b9b9d-"

OLD_MANIFEST_SHA256 = "f1490f2cda497c9475fe079fb0a04e5adb7686c6f4c99ae28a0f946a918319a8"
OLD_MANIFEST_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.json"
)
OLD_JOURNAL_FILE_SHA256 = (
    "a856792b167f0d50bc83e8a85066ba0ee2a87031fff1ac8006466e9e681abc2e"
)
OLD_JOURNAL_PATH = OLD_MANIFEST_PATH.with_name(OLD_MANIFEST_PATH.stem + ".journal.json")
OLD_PREFIX = "mft-camp-s754923c-le6b9b9d-"
OLD_FIRST_ID = 27755
OLD_LAST_ID = 28004
OLD_COUNT = 250

RECOVERY_PLAN_SHA256 = "3e453deb61137c2d29c13bbbe8d5117b4c4111e5ea7e255d37dfd0d5e4444af5"
RECOVERY_PLAN_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.json"
)
RECOVERY_SUBMISSION_SHA256 = (
    "fa951faa0cd29c3502e511f827ff0fc2573facc1413c76fb1ad3db0f689d5abc"
)
RECOVERY_SUBMISSION_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.submission.json"
)
RECOVERY_IDS = (28077, 28078, 28079, 28080)
GATE_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-sb171c7c-le6b9b9d.terminal-gate.json"
)
GATE_SCHEMA = "thermal-recovery4-terminal-gate-v1"

PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
SEALED_PROFILE_PROVENANCE_PATH = (
    r"\\RaiDrive-peets\ANSYS\git\MFT_1MW_2026"
    r"\regression_260707\verify\profiles\standard.json")
SEALED_OLD_MANIFEST_PROVENANCE_PATH = (
    r"\\RaiDrive-peets\ANSYS\git\MFT_1MW_2026"
    r"\regression_260707\campaign\pilot_manifests"
    r"\replacement-s754923c-le6b9b9d-seed260710-cursor1843.json")
LIBRARY_ROOT = REPO_ROOT.parent / "pyaedt_library_mft_clean"
SOLVER_REFS = (
    "refs/heads/fix/mft-rx-block-fastpath-260712",
    "refs/heads/stabilize/mft-sim-260710",
)
LIBRARY_REFS = ("refs/heads/pyaedt_022",)

COUNT = 300
PROJECT_CAP = 300
ACTIVE_STATUSES = frozenset(("queued", "attaching", "running"))
TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))
FULL_SHA = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_RESOURCES = {
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

PARTIAL_PATH = PLAN_PATH.with_name(PLAN_PATH.stem + ".submission.partial.json")
FINAL_PATH = PLAN_PATH.with_name(PLAN_PATH.stem + ".submission.json")
CANCEL_SETTLE_SECONDS = 300
CANCEL_POLL_SECONDS = 2


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _sealed_json(path, expected_sha, label, seal_key):
    payload = _read_json(path, label)
    unsigned = dict(payload)
    stored = unsigned.pop(seal_key, None)
    if stored != expected_sha or _sha(unsigned) != expected_sha:
        raise RuntimeError(f"{label} seal mismatch")
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _write_sealed(path, payload, seal_key):
    sealed = dict(payload)
    sealed.pop(seal_key, None)
    sealed[seal_key] = _sha(sealed)
    _atomic_json(path, sealed)
    readback = _read_json(path, f"{seal_key} readback")
    unsigned = dict(readback)
    stored = unsigned.pop(seal_key, None)
    if readback != sealed or stored != _sha(unsigned):
        raise RuntimeError(f"sealed journal readback mismatch: {path}")
    return sealed


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True, stderr=subprocess.STDOUT,
    ).strip()


def _clean_solver_deployment_root():
    if _git(REPO_ROOT, "rev-parse", "HEAD") != SOLVER:
        raise RuntimeError("local solver HEAD is not exact SHA-b171")
    override = os.environ.get("MFT_SOLVER_DEPLOYMENT_ROOT", "").strip()
    candidates = [Path(override).resolve()] if override else []
    if not override:
        records = _git(REPO_ROOT, "worktree", "list", "--porcelain")
        for block in re.split(r"\r?\n\r?\n", records.strip()):
            fields = {}
            for line in block.splitlines():
                key, _, value = line.partition(" ")
                if value:
                    fields[key] = value
            if fields.get("HEAD") == SOLVER and fields.get("worktree"):
                candidates.append(Path(fields["worktree"]).resolve())
    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            exact = _git(candidate, "rev-parse", "HEAD") == SOLVER
            clean = not _git(
                candidate, "status", "--porcelain", "--untracked-files=no",
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        if exact and clean:
            return candidate
    raise RuntimeError("no clean exact-SHA-b171 solver deployment worktree")


def _load_profile(plan):
    profile = _read_json(PROFILE_PATH, "standard profile")
    profile["timeout_seconds"] = EXPECTED_RESOURCES["timeout_seconds"]
    evidence = plan.get("profile")
    if not isinstance(evidence, dict):
        raise RuntimeError("production300 profile evidence is missing")
    if evidence.get("path") != SEALED_PROFILE_PROVENANCE_PATH:
        raise RuntimeError("production300 profile path drifted")
    if hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() \
            != evidence.get("file_sha256"):
        raise RuntimeError("standard profile file drifted")
    if _sha(profile) != evidence.get("effective_sha256"):
        raise RuntimeError("effective standard profile drifted")
    if profile.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT \
            or profile.get("cli_flags") != "--thermal --headless":
        raise RuntimeError("standard full-thermal profile contract drifted")
    return profile


def _validate_plan_task(task, index, profile):
    serial = 17612 + index
    expected_name = f"{PLAN_PREFIX}{serial:05d}"
    checks = {
        "index": task.get("index") == index,
        "serial": task.get("serial") == serial,
        "name": task.get("name") == expected_name,
        "workdir": task.get("workdir") == f"mft_p300_t{serial % 500:03d}",
        "params_object": isinstance(task.get("params"), dict),
    }
    if not all(checks.values()):
        raise RuntimeError(f"production300 task {index} structural identity drifted: {checks}")
    params = task["params"]
    if pinned_pilot.candidate_digest(params) != task.get("params_sha256"):
        raise RuntimeError(f"production300 task {index} params digest drifted")
    identity = scheduler_client.verification_submission_identity(
        task["name"], params, profile, SOLVER, LIBRARY,
    )
    for key in ("dedupe_key", "parameter_digest"):
        if task.get(key) != identity[key]:
            raise RuntimeError(f"production300 task {index} {key} drifted")
    if task.get("effective_params") != identity["merged"]:
        raise RuntimeError(f"production300 task {index} effective params drifted")
    effective = identity["merged"]
    if int(effective.get("N1_main", -1)) + int(effective.get("N1_side", -1)) > 8:
        raise RuntimeError(f"production300 task {index} exceeds N1=8")
    if not 0 < float(effective.get("cw1", 0)) <= 10.0:
        raise RuntimeError(f"production300 task {index} exceeds cw1=10mm")
    if effective.get("matrix_on") != 1 or effective.get("loss_on") != 1 \
            or effective.get("thermal_on") != 1 \
            or float(effective.get("P_target", 0)) != 1_000_000.0:
        raise RuntimeError(f"production300 task {index} is not full thermal")
    return {
        "index": index,
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "params_sha256": task["params_sha256"],
    }


def _load_plan():
    if _file_sha(PLAN_PATH) != PLAN_FILE_SHA256:
        raise RuntimeError("root-reviewed production300 plan file bytes drifted")
    plan = _sealed_json(PLAN_PATH, PLAN_SHA256, "production300 plan", "plan_sha256")
    if plan.get("mode") != "plan_only" or plan.get("submission_enabled") is not False \
            or plan.get("scheduler_mutation_count") != 0:
        raise RuntimeError("production300 source is not immutable plan-only evidence")
    if plan.get("solver_revision") != SOLVER \
            or plan.get("library_revision") != LIBRARY:
        raise RuntimeError("production300 revision identity drifted")
    if plan.get("task_count") != COUNT or plan.get("task_prefix") != PLAN_PREFIX:
        raise RuntimeError("production300 count/prefix drifted")
    if plan.get("resources") != EXPECTED_RESOURCES:
        raise RuntimeError("production300 resource contract drifted")
    capacity = plan.get("capacity_contract")
    if not isinstance(capacity, dict) \
            or capacity.get("project") != scheduler_client.MFT_PROJECT \
            or capacity.get("project_max_active_tasks") != PROJECT_CAP \
            or capacity.get("project_required_hard_cap") != PROJECT_CAP \
            or capacity.get("planned_task_count") != COUNT \
            or capacity.get("live_capacity_recheck_required_inside_mutation_lock") is not True:
        raise RuntimeError("production300 capacity contract drifted")
    predecessor = plan.get("predecessor")
    if not isinstance(predecessor, dict) \
            or predecessor.get("old_manifest") != SEALED_OLD_MANIFEST_PROVENANCE_PATH \
            or predecessor.get("old_manifest_sha256") != OLD_MANIFEST_SHA256 \
            or predecessor.get("old_task_id_range") != [OLD_FIRST_ID, OLD_LAST_ID] \
            or predecessor.get("old_task_count") != OLD_COUNT \
            or predecessor.get("recovery_plan_sha256") != RECOVERY_PLAN_SHA256 \
            or predecessor.get("recovery_submission_sha256") != RECOVERY_SUBMISSION_SHA256 \
            or predecessor.get("recovery_task_ids") != list(RECOVERY_IDS):
        raise RuntimeError("production300 predecessor contract drifted")
    activation = plan.get("activation_requirements")
    required_false = (
        "recovery4_all_strict_valid",
        "failed_sources_exact_thermal_setup",
        "failed_sources_analyze_all_absent",
        "failed_sources_fresh_monitor",
        "known_good_nonregression",
        "deployment_gate_passed_inside_mutation_lock",
    )
    if not isinstance(activation, dict) \
            or any(activation.get(key) is not False for key in required_false) \
            or activation.get("root_reviewed_plan_sha256") is not None \
            or activation.get("cancel_only_exact_remaining_old_ids") is not True \
            or activation.get("submit_under_one_campaign_mutation_lock") is not True:
        raise RuntimeError("production300 activation requirements drifted")
    profile = _load_profile(plan)
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != COUNT:
        raise RuntimeError("production300 task list is incomplete")
    compact = [_validate_plan_task(task, index, profile)
               for index, task in enumerate(tasks)]
    for key in ("name", "dedupe_key", "params_sha256"):
        if len({row[key] for row in compact}) != COUNT:
            raise RuntimeError(f"production300 {key} values are not unique")
    return plan, profile, compact


def _load_old_cohort(profile):
    old = _sealed_json(
        OLD_MANIFEST_PATH, OLD_MANIFEST_SHA256,
        "SHA754 predecessor manifest", "manifest_sha256",
    )
    if old.get("solver_revision") != OLD_SOLVER \
            or old.get("library_revision") != LIBRARY \
            or old.get("task_count") != OLD_COUNT \
            or old.get("task_prefix") != OLD_PREFIX:
        raise RuntimeError("SHA754 predecessor identity drifted")
    tasks = old.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != OLD_COUNT:
        raise RuntimeError("SHA754 predecessor task list is incomplete")
    for index, task in enumerate(tasks):
        if task.get("index") != index or not isinstance(task.get("params"), dict):
            raise RuntimeError(f"SHA754 manifest index {index} drifted")
        if pinned_pilot.candidate_digest(task["params"]) != task.get("params_sha256"):
            raise RuntimeError(f"SHA754 manifest params {index} drifted")
        identity = scheduler_client.verification_submission_identity(
            task.get("name"), task["params"], profile, OLD_SOLVER, LIBRARY,
        )
        if task.get("dedupe_key") != identity["dedupe_key"] \
                or task.get("parameter_digest") != identity["parameter_digest"] \
                or task.get("effective_params") != identity["merged"]:
            raise RuntimeError(f"SHA754 manifest identity {index} drifted")
    for key in ("name", "dedupe_key", "params_sha256"):
        if len({task.get(key) for task in tasks}) != OLD_COUNT:
            raise RuntimeError(f"SHA754 manifest {key} values are not unique")

    if _file_sha(OLD_JOURNAL_PATH) != OLD_JOURNAL_FILE_SHA256:
        raise RuntimeError("SHA754 ID journal file bytes drifted")
    journal = _read_json(OLD_JOURNAL_PATH, "SHA754 ID journal")
    submissions = journal.get("submissions")
    if journal.get("completed") is not True \
            or journal.get("manifest_sha256") != OLD_MANIFEST_SHA256 \
            or not isinstance(submissions, dict) \
            or set(submissions) != {str(index) for index in range(OLD_COUNT)}:
        raise RuntimeError("SHA754 ID journal header/index set drifted")
    records = []
    for index, manifest_task in enumerate(tasks):
        row = submissions[str(index)]
        expected_id = OLD_FIRST_ID + index
        if not isinstance(row, dict) \
                or row.get("index") != index \
                or row.get("task_id") != expected_id \
                or row.get("name") != manifest_task.get("name") \
                or row.get("dedupe_key") != manifest_task.get("dedupe_key"):
            raise RuntimeError(f"SHA754 ID/name/dedupe mapping drifted at {index}")
        records.append({
            "index": index,
            "task_id": expected_id,
            "name": row["name"],
            "dedupe_key": row["dedupe_key"],
        })
    if [row["task_id"] for row in records] != list(range(OLD_FIRST_ID, OLD_LAST_ID + 1)):
        raise RuntimeError("SHA754 task IDs are not the exact sealed range")
    for key in ("task_id", "name", "dedupe_key"):
        if len({row[key] for row in records}) != OLD_COUNT:
            raise RuntimeError(f"SHA754 journal {key} values are not unique")
    return old, records


def _load_recovery():
    recovery_plan = _sealed_json(
        RECOVERY_PLAN_PATH, RECOVERY_PLAN_SHA256,
        "recovery4 plan", "plan_sha256",
    )
    submission = _sealed_json(
        RECOVERY_SUBMISSION_PATH, RECOVERY_SUBMISSION_SHA256,
        "recovery4 submission", "submission_sha256",
    )
    if recovery_plan.get("solver_revision") != SOLVER \
            or recovery_plan.get("library_revision") != LIBRARY \
            or recovery_plan.get("task_count") != 4 \
            or recovery_plan.get("resources") != EXPECTED_RESOURCES:
        raise RuntimeError("recovery4 plan identity drifted")
    if submission.get("solver_revision") != SOLVER \
            or submission.get("library_revision") != LIBRARY \
            or submission.get("root_reviewed_plan_sha256") != RECOVERY_PLAN_SHA256 \
            or submission.get("task_count") != 4:
        raise RuntimeError("recovery4 submission identity drifted")
    plan_tasks = recovery_plan.get("tasks")
    rows = submission.get("tasks")
    if not isinstance(plan_tasks, list) or not isinstance(rows, list) \
            or len(plan_tasks) != 4 or len(rows) != 4:
        raise RuntimeError("recovery4 task evidence is incomplete")
    if tuple(row.get("task_id") for row in rows) != RECOVERY_IDS:
        raise RuntimeError("recovery4 task IDs drifted")
    for index, (expected, row) in enumerate(zip(plan_tasks, rows)):
        for key in ("ordinal", "source_task_id", "name", "dedupe_key"):
            if row.get(key) != expected.get(key):
                raise RuntimeError(f"recovery4 submission {key} drifted")
        acceptance = expected.get("acceptance")
        if not isinstance(acceptance, dict) \
                or acceptance.get("strict_valid_required") is not True:
            raise RuntimeError("recovery4 strict acceptance contract drifted")
        if index < 3:
            if acceptance.get("thermal_entrypoint_exact") != "ThermalSetup" \
                    or acceptance.get("analyze_all_forbidden") is not True \
                    or acceptance.get("fresh_monitor_required") is not True \
                    or acceptance.get("startup_retry_max") != 1:
                raise RuntimeError("recovery4 failed-source acceptance drifted")
        elif acceptance.get("known_good_nonregression") is not True:
            raise RuntimeError("recovery4 known-good acceptance drifted")
        metadata = row.get("scheduler_metadata")
        if not isinstance(metadata, dict) \
                or metadata.get("id") != row.get("task_id") \
                or metadata.get("name") != row.get("name") \
                or metadata.get("dedupe_key") != row.get("dedupe_key") \
                or metadata.get("project") != scheduler_client.MFT_PROJECT:
            raise RuntimeError("recovery4 scheduler identity drifted")
    return recovery_plan, submission


def static_audit():
    plan, profile, plan_records = _load_plan()
    old, old_records = _load_old_cohort(profile)
    recovery_plan, recovery_submission = _load_recovery()
    plan_names = {row["name"] for row in plan_records}
    plan_dedupes = {row["dedupe_key"] for row in plan_records}
    prior_names = {row["name"] for row in old_records}
    prior_dedupes = {row["dedupe_key"] for row in old_records}
    for source in (recovery_plan.get("tasks", []), recovery_submission.get("tasks", [])):
        prior_names.update(str(row.get("name")) for row in source)
        prior_dedupes.update(str(row.get("dedupe_key")) for row in source)
    if plan_names & prior_names or plan_dedupes & prior_dedupes:
        raise RuntimeError("production300 names/dedupe overlap predecessor identities")
    return {
        "plan": plan,
        "profile": profile,
        "plan_records": plan_records,
        "old_manifest": old,
        "old_records": old_records,
        "recovery_plan": recovery_plan,
        "recovery_submission": recovery_submission,
    }


def _validate_gate(payload, reviewed_sha, recovery_submission):
    if not isinstance(reviewed_sha, str) or not FULL_SHA.fullmatch(reviewed_sha):
        raise RuntimeError("terminal recovery gate requires a reviewed 64-character SHA256")
    unsigned = dict(payload)
    stored = unsigned.pop("gate_sha256", None)
    if stored != reviewed_sha or _sha(unsigned) != reviewed_sha:
        raise RuntimeError("terminal recovery gate seal mismatch")
    exact_four = (
        type(payload.get("task_count")) is int
        and payload.get("task_count") == 4
        and type(payload.get("strict_valid_count")) is int
        and payload.get("strict_valid_count") == 4
    )
    if payload.get("schema") != GATE_SCHEMA \
            or payload.get("gate_decision") != "pass" \
            or payload.get("solver_revision") != SOLVER \
            or payload.get("library_revision") != LIBRARY \
            or payload.get("recovery_plan_sha256") != RECOVERY_PLAN_SHA256 \
            or payload.get("recovery_submission_sha256") != RECOVERY_SUBMISSION_SHA256 \
            or not exact_four \
            or payload.get("all_strict_valid") is not True \
            or payload.get("partial_pass_allowed") is not False:
        raise RuntimeError("terminal recovery gate header/decision drifted")
    rows = payload.get("tasks")
    submitted = recovery_submission.get("tasks", [])
    if not isinstance(rows, list) or len(rows) != 4 or len(submitted) != 4:
        raise RuntimeError("terminal recovery gate task evidence is incomplete")
    for index, (row, expected) in enumerate(zip(rows, submitted)):
        for key in ("ordinal", "task_id", "source_task_id", "name", "dedupe_key"):
            if row.get(key) != expected.get(key):
                raise RuntimeError(f"terminal recovery gate task {index + 1} {key} drifted")
        if row.get("status") != "completed" \
                or row.get("result_state") != "valid" \
                or row.get("strict_valid") is not True \
                or not isinstance(row.get("result_sha256"), str) \
                or not FULL_SHA.fullmatch(row["result_sha256"]):
            raise RuntimeError(f"terminal recovery gate task {index + 1} is not strict-valid")
        if index < 3:
            dispatch = row.get("thermal_dispatch")
            if not isinstance(dispatch, dict) \
                    or dispatch.get("entrypoint") != "ThermalSetup" \
                    or type(dispatch.get("analyze_all_call_count")) is not int \
                    or dispatch.get("analyze_all_call_count") != 0 \
                    or dispatch.get("fresh_monitor") is not True \
                    or isinstance(dispatch.get("startup_retry_count"), bool) \
                    or not isinstance(dispatch.get("startup_retry_count"), int) \
                    or not 0 <= dispatch["startup_retry_count"] <= 1:
                raise RuntimeError(
                    f"terminal recovery failed-source dispatch evidence {index + 1} drifted"
                )
        elif row.get("known_good_nonregression") is not True:
            raise RuntimeError("terminal recovery known-good nonregression is absent")
    return payload


def _load_gate(path, reviewed_sha, recovery_submission, required):
    path = Path(path)
    if not path.exists():
        if required:
            raise RuntimeError(f"terminal recovery gate evidence is missing: {path}")
        return None
    payload = _read_json(path, "terminal recovery gate")
    if reviewed_sha is None:
        reviewed_sha = payload.get("gate_sha256")
    return _validate_gate(payload, reviewed_sha, recovery_submission)


def deployment_audit():
    solver_root = _clean_solver_deployment_root()
    if _git(LIBRARY_ROOT, "rev-parse", "HEAD") != LIBRARY:
        raise RuntimeError("local library HEAD is not exact SHA-e6b9")
    if _git(LIBRARY_ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("local library worktree is dirty")
    deployed = deployment_gate.validate_deployment(
        solver_root, SOLVER, LIBRARY_ROOT, LIBRARY,
    )
    if tuple(deployed.get("solver", {}).get("refs", ())) != SOLVER_REFS:
        raise RuntimeError("advertised solver refs drifted")
    if tuple(deployed.get("library", {}).get("refs", ())) != LIBRARY_REFS:
        raise RuntimeError("advertised library refs drifted")
    return deployed


def _task_rows(response, label):
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None
    )
    if not isinstance(rows, list):
        raise RuntimeError(f"scheduler {label} inventory is not a list")
    return rows


def _inventory(prefix):
    try:
        response = requests.get(
            f"{scheduler_client.SCHEDULER}/api/tasks",
            params={"limit": 10000, "name_prefix": prefix}, timeout=30,
        )
        return _task_rows(response, prefix)
    except Exception as exc:
        raise RuntimeError(f"failed to read scheduler prefix {prefix!r}: {exc}") from exc


def _task_detail(task_id):
    try:
        response = requests.get(
            f"{scheduler_client.SCHEDULER}/api/tasks/{int(task_id)}", timeout=20,
        )
        response.raise_for_status()
        row = response.json()
    except Exception as exc:
        raise RuntimeError(f"failed to read scheduler task {task_id}: {exc}") from exc
    if not isinstance(row, dict):
        raise RuntimeError(f"scheduler task {task_id} is not an object")
    return row


def _validate_live_old(inventory, expected_records):
    by_id = {}
    for row in inventory:
        if not isinstance(row, dict):
            raise RuntimeError("scheduler SHA754 inventory contains a non-object")
        task_id = row.get("id", row.get("task_id"))
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise RuntimeError("scheduler SHA754 inventory contains an invalid ID")
        if task_id in by_id:
            raise RuntimeError(f"scheduler SHA754 inventory duplicates task ID {task_id}")
        by_id[task_id] = row
    audited = []
    for expected in expected_records:
        task_id = expected["task_id"]
        row = by_id.get(task_id)
        if row is None:
            raise RuntimeError(f"exact SHA754 scheduler task {task_id} is missing")
        status = str(row.get("status") or "").strip().lower()
        checks = {
            "name": row.get("name") == expected["name"],
            "dedupe": row.get("dedupe_key") == expected["dedupe_key"],
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "status": status in ACTIVE_STATUSES | TERMINAL_STATUSES,
        }
        if not all(checks.values()):
            raise RuntimeError(f"exact SHA754 scheduler task {task_id} drifted: {checks}")
        audited.append({
            **expected,
            "status": status,
            "active": status in ACTIVE_STATUSES,
        })
    if len({row["name"] for row in audited}) != len(audited) \
            or len({row["dedupe_key"] for row in audited}) != len(audited):
        raise RuntimeError("live SHA754 identity values are duplicated")
    return audited


def _validate_live_recovery(submission):
    audited = []
    for expected in submission.get("tasks", []):
        row = _task_detail(expected["task_id"])
        checks = {
            "id": row.get("id", row.get("task_id")) == expected["task_id"],
            "name": row.get("name") == expected["name"],
            "dedupe": row.get("dedupe_key") == expected["dedupe_key"],
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "completed": str(row.get("status") or "").lower() == "completed",
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"live recovery4 task {expected['task_id']} is not terminal gate evidence: {checks}"
            )
        audited.append({
            "task_id": expected["task_id"],
            "name": expected["name"],
            "dedupe_key": expected["dedupe_key"],
            "status": "completed",
        })
    return audited


def _assert_no_new_duplicates(plan):
    inventory = _inventory(PLAN_PREFIX)
    if inventory:
        summary = [
            (row.get("id", row.get("task_id")), row.get("name"), row.get("status"))
            for row in inventory[:10] if isinstance(row, dict)
        ]
        raise RuntimeError(
            f"new production300 scheduler namespace is not empty: {summary}"
        )
    # The prefix is unique to this exact solver/library campaign.  An empty
    # namespace therefore proves both exact-name and exact-dedupe count zero.
    return {"prefix": PLAN_PREFIX, "existing_name_count": 0, "existing_dedupe_count": 0}


def _capacity(required_slots):
    snapshot = scheduler_client.live_project_submission_snapshot(PROJECT_CAP)
    if snapshot.get("project") != scheduler_client.MFT_PROJECT \
            or snapshot.get("project_max_active_tasks") != PROJECT_CAP \
            or snapshot.get("project_required_hard_cap") != PROJECT_CAP:
        raise RuntimeError("live MFT capacity contract drifted")
    if int(snapshot.get("project_submission_slots", -1)) < int(required_slots):
        raise RuntimeError(
            f"MFT project needs {required_slots} exact slots, capacity is {snapshot}"
        )
    return snapshot


def _cancel_exact_active(active_rows):
    if not active_rows:
        return {"requested_ids": [], "acknowledgement": None, "after": []}
    ids = [row["task_id"] for row in active_rows]
    response = requests.post(
        f"{scheduler_client.SCHEDULER}/api/tasks/cancel",
        params={
            "statuses": ",".join(sorted(ACTIVE_STATUSES)),
            "task_ids": ",".join(map(str, ids)),
        },
        timeout=60,
    )
    response.raise_for_status()
    acknowledgement = response.json()
    if not isinstance(acknowledgement, dict):
        raise RuntimeError("SHA754 cancellation acknowledgement is not an object")
    acknowledged = None
    for key in ("cancelled", "acknowledged_ids", "task_ids"):
        if isinstance(acknowledgement.get(key), list):
            acknowledged = acknowledgement[key]
            break
    try:
        acknowledged_ids = sorted(int(value) for value in acknowledged)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SHA754 cancellation acknowledgement has no exact ID list") from exc
    if acknowledged_ids != sorted(ids):
        raise RuntimeError(
            f"SHA754 cancellation acknowledgement drifted: {acknowledged_ids} != {sorted(ids)}"
        )

    expected = {row["task_id"]: row for row in active_rows}
    deadline = time.monotonic() + CANCEL_SETTLE_SECONDS
    after = []
    while time.monotonic() < deadline:
        after = []
        for task_id in ids:
            row = _task_detail(task_id)
            source = expected[task_id]
            checks = {
                "name": row.get("name") == source["name"],
                "dedupe": row.get("dedupe_key") == source["dedupe_key"],
                "project": row.get("project") == scheduler_client.MFT_PROJECT,
            }
            if not all(checks.values()):
                raise RuntimeError(f"SHA754 task {task_id} drifted during cancellation: {checks}")
            after.append({
                "task_id": task_id,
                "name": source["name"],
                "dedupe_key": source["dedupe_key"],
                "status": str(row.get("status") or "").lower(),
            })
        if all(row["status"] == "cancelled" for row in after):
            break
        time.sleep(CANCEL_POLL_SECONDS)
    if not after or not all(row["status"] == "cancelled" for row in after):
        raise RuntimeError(
            "exact SHA754 cancellation did not settle: "
            + repr([(row["task_id"], row["status"]) for row in after])
        )
    return {
        "requested_ids": ids,
        "acknowledgement": acknowledgement,
        "after": after,
    }


def _task_metadata(task_id, expected):
    row = _task_detail(task_id)
    checks = {
        "id": row.get("id", row.get("task_id")) == int(task_id),
        "name": row.get("name") == expected["name"],
        "project": row.get("project") == EXPECTED_RESOURCES["project"],
        "dedupe": row.get("dedupe_key") == expected["dedupe_key"],
        "cpus": row.get("cpus") == EXPECTED_RESOURCES["cpus"],
        "memory": row.get("memory_mb") == EXPECTED_RESOURCES["memory_mb"],
        "gpus": row.get("gpus") == EXPECTED_RESOURCES["gpus"],
        "timeout": row.get("timeout_seconds") == EXPECTED_RESOURCES["timeout_seconds"],
        "capability": row.get("required_capability") == EXPECTED_RESOURCES["required_capability"],
        "env": row.get("env_profile") == EXPECTED_RESOURCES["env_profile"],
        "profile": row.get("scheduling_profile") == EXPECTED_RESOURCES["scheduling_profile"],
        "remote_cwd": row.get("remote_cwd") == EXPECTED_RESOURCES["remote_cwd"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"submitted production300 task {task_id} metadata drifted: {checks}")
    return {key: row.get(key) for key in (
        "id", "name", "status", "project", "dedupe_key", "cpus", "memory_mb",
        "gpus", "timeout_seconds", "required_capability", "env_profile",
        "scheduling_profile", "remote_cwd", "created_at",
    )}


def _assert_no_existing_output():
    if FINAL_PATH.exists():
        raise RuntimeError(f"sealed final journal already exists: {FINAL_PATH}")
    if PARTIAL_PATH.exists():
        raise RuntimeError(
            f"partial execution journal exists and requires separate root review: {PARTIAL_PATH}"
        )


def _execute_locked(bundle, gate, deployed):
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise RuntimeError("production300 execution requires the campaign mutation lock")
    _assert_no_existing_output()
    duplicate_audit = _assert_no_new_duplicates(bundle["plan"])
    old_live = _validate_live_old(
        _inventory(OLD_PREFIX), bundle["old_records"],
    )
    recovery_live = _validate_live_recovery(bundle["recovery_submission"])
    active_old = [row for row in old_live if row["active"]]
    capacity_before = _capacity(0)

    ledger = {
        "schema": "production300-b171-submission-v1",
        "created_at": _now(),
        "execution_state": "pre-mutation-audited",
        "plan": str(PLAN_PATH.resolve()),
        "root_reviewed_plan_sha256": PLAN_SHA256,
        "terminal_gate": str(GATE_PATH.resolve()),
        "root_reviewed_terminal_gate_sha256": gate["gate_sha256"],
        "old_manifest": str(OLD_MANIFEST_PATH.resolve()),
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "old_id_journal": str(OLD_JOURNAL_PATH.resolve()),
        "old_id_journal_file_sha256": OLD_JOURNAL_FILE_SHA256,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "deployment": deployed,
        "resources": EXPECTED_RESOURCES,
        "project_hard_cap": PROJECT_CAP,
        "task_count": len(bundle["plan"]["tasks"]),
        "duplicate_audit": duplicate_audit,
        "recovery_live_terminal": recovery_live,
        "old_status_counts_before": dict(Counter(row["status"] for row in old_live)),
        "old_exact_active_ids": [row["task_id"] for row in active_old],
        "capacity_before": capacity_before,
        "cancellation": None,
        "tasks": [],
        "scheduler_mutation_attempt_count": 0,
        "scheduler_mutation_count": 0,
    }
    ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")

    try:
        if active_old:
            ledger["execution_state"] = "cancellation-request-about-to-mutate"
            ledger["scheduler_mutation_attempt_count"] += 1
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")
            ledger["cancellation"] = _cancel_exact_active(active_old)
            ledger["scheduler_mutation_count"] += 1
            ledger["execution_state"] = "old-cancellation-settled"
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")
        else:
            ledger["cancellation"] = {
                "requested_ids": [], "acknowledgement": None, "after": [],
            }
            ledger["execution_state"] = "no-active-old-tasks"
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")

        # Re-read both the exact namespace and absolute logical-project cap
        # after cancellation has settled, before the first new POST.
        ledger["duplicate_audit_after_cancellation"] = _assert_no_new_duplicates(
            bundle["plan"]
        )
        ledger["capacity_after_cancellation"] = _capacity(len(bundle["plan"]["tasks"]))
        ledger["execution_state"] = "submitting-sequentially"
        ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")

        for task in bundle["plan"]["tasks"]:
            attempt = {
                "index": task["index"],
                "name": task["name"],
                "dedupe_key": task["dedupe_key"],
                "params_sha256": task["params_sha256"],
                "submission_attempted_at": _now(),
            }
            ledger["tasks"].append(attempt)
            ledger["scheduler_mutation_attempt_count"] += 1
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")
            task_id = scheduler_client.submit_verification(
                name=task["name"],
                workdir=task["workdir"],
                params=task["params"],
                profile=bundle["profile"],
                mem_mb=EXPECTED_RESOURCES["memory_mb"],
                cpus=EXPECTED_RESOURCES["cpus"],
                solver_revision=SOLVER,
                library_revision=LIBRARY,
            )
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
                raise RuntimeError(f"scheduler returned no durable ID for {task['name']}")
            attempt["task_id"] = task_id
            attempt["accepted_at"] = _now()
            ledger["scheduler_mutation_count"] += 1
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")
            attempt["scheduler_metadata"] = _task_metadata(task_id, task)
            attempt["readback_audited_at"] = _now()
            ledger = _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")

        if len(ledger["tasks"]) != len(bundle["plan"]["tasks"]):
            raise RuntimeError("production300 sequential submission count drifted")
        ledger["capacity_after_submission"] = _capacity(0)
        ledger["execution_state"] = "complete"
        ledger["completed_at"] = _now()
        ledger.pop("partial_sha256", None)
        final = _write_sealed(FINAL_PATH, ledger, "submission_sha256")
        PARTIAL_PATH.unlink(missing_ok=True)
        return final
    except Exception as exc:
        ledger["execution_state"] = "failed-closed"
        ledger["failed_at"] = _now()
        ledger["failure"] = f"{type(exc).__name__}: {exc}"
        _write_sealed(PARTIAL_PATH, ledger, "partial_sha256")
        raise


def execute(plan_path, reviewed_plan_sha, gate_path, reviewed_gate_sha):
    if Path(plan_path).resolve() != PLAN_PATH.resolve() \
            or reviewed_plan_sha != PLAN_SHA256:
        raise RuntimeError("execution requires the exact root-reviewed production300 plan seal")
    if Path(gate_path).resolve() != GATE_PATH.resolve():
        raise RuntimeError("execution requires the exact terminal recovery gate path")
    # All filesystem, gate and deployment failures are detected before the
    # mutation lock is acquired, and are repeated inside it to close TOCTOU.
    preflight = static_audit()
    _load_gate(gate_path, reviewed_gate_sha, preflight["recovery_submission"], required=True)
    deployment_audit()
    _assert_no_existing_output()
    with scheduler_client.campaign_mutation_lock():
        locked = static_audit()
        gate = _load_gate(
            gate_path, reviewed_gate_sha, locked["recovery_submission"], required=True,
        )
        deployed = deployment_audit()
        return _execute_locked(locked, gate, deployed)


def _live_read_only_audit(bundle):
    duplicate_audit = _assert_no_new_duplicates(bundle["plan"])
    old_live = _validate_live_old(_inventory(OLD_PREFIX), bundle["old_records"])
    recovery_live = _validate_live_recovery(bundle["recovery_submission"])
    return {
        "authoritative_for_mutation": False,
        "duplicate_audit": duplicate_audit,
        "old_status_counts": dict(Counter(row["status"] for row in old_live)),
        "exact_active_old_ids": [row["task_id"] for row in old_live if row["active"]],
        "recovery_live_terminal": recovery_live,
    }


def audit(gate_path=GATE_PATH, reviewed_gate_sha=None, live=False):
    bundle = static_audit()
    blockers = []
    deployment = None
    try:
        deployment = deployment_audit()
    except Exception as exc:
        blockers.append(f"deployment: {exc}")
    gate = None
    if not Path(gate_path).exists():
        blockers.append(f"terminal_gate_missing: {Path(gate_path)}")
    else:
        gate = _load_gate(
            gate_path, reviewed_gate_sha,
            bundle["recovery_submission"], required=True,
        )
        if reviewed_gate_sha is None:
            blockers.append("terminal_gate_not_explicitly_root_reviewed")
    live_result = None
    if live:
        if blockers:
            blockers.append("live_scheduler_audit_skipped_until_static_gates_pass")
        else:
            live_result = _live_read_only_audit(bundle)
    return {
        "mode": "audit-only",
        "scheduler_query_count": 0 if live_result is None else "read-only",
        "scheduler_mutation_count": 0,
        "execution_ready": not blockers and (not live or live_result is not None),
        "blockers": blockers,
        "plan": str(PLAN_PATH.resolve()),
        "root_reviewed_plan_sha256": PLAN_SHA256,
        "plan_task_count": len(bundle["plan_records"]),
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "old_exact_task_id_range": [OLD_FIRST_ID, OLD_LAST_ID],
        "old_exact_task_count": len(bundle["old_records"]),
        "recovery_submission_sha256": RECOVERY_SUBMISSION_SHA256,
        "terminal_gate_sha256": None if gate is None else gate["gate_sha256"],
        "deployment": deployment,
        "live_read_only": live_result,
        "would_cancel": "only exact active SHA754 IDs, unknown until locked execution audit",
        "would_submit": COUNT,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--live-audit", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reviewed-plan-sha")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--reviewed-gate-sha")
    args = parser.parse_args(argv)
    audit_plan = PLAN_PATH if args.plan is None else args.plan
    audit_gate = GATE_PATH if args.gate is None else args.gate
    if audit_plan.resolve() != PLAN_PATH.resolve():
        parser.error(f"only exact plan is allowed: {PLAN_PATH}")
    if args.execute:
        if args.live_audit:
            parser.error("--execute and --live-audit are mutually exclusive")
        if args.plan is None:
            parser.error(f"--execute requires --plan {PLAN_PATH}")
        if args.reviewed_plan_sha != PLAN_SHA256:
            parser.error(f"--execute requires --reviewed-plan-sha {PLAN_SHA256}")
        if args.gate is None or args.gate.resolve() != GATE_PATH.resolve():
            parser.error(f"--execute requires --gate {GATE_PATH}")
        if not isinstance(args.reviewed_gate_sha, str) \
                or not FULL_SHA.fullmatch(args.reviewed_gate_sha):
            parser.error("--execute requires an exact --reviewed-gate-sha")
        result = execute(
            args.plan, args.reviewed_plan_sha,
            args.gate, args.reviewed_gate_sha,
        )
    else:
        result = audit(
            gate_path=audit_gate,
            reviewed_gate_sha=args.reviewed_gate_sha,
            live=args.live_audit,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
