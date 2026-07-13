"""Audit and idempotently submit the sealed four-case SHA7873 recovery pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
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


SOLVER = "7873ddddcf7ac7412d14c9e3ae216ed73b82fffe"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PLAN_SHA256 = "ece474e96603139769897693455fe3fedd465da578aa54207e1e7e7104bf7d0f"
PLAN_PATH = HERE / "pilot_manifests" / "thermal-recovery4-s7873ddd-le6b9b9d.json"
PARTIAL_PATH = PLAN_PATH.with_name(PLAN_PATH.stem + ".submission.partial.json")
FINAL_PATH = PLAN_PATH.with_name(PLAN_PATH.stem + ".submission.json")
PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
LIBRARY_ROOT = REPO_ROOT.parent / "pyaedt_library_mft_clean"
PREFIX = "mft-recovery4-s7873ddd-le6b9b9d-"
EXPECTED_SOURCES = (27794, 27928, 27880, 27758)
EXPECTED_RESOURCES = {
    "cpus": 4,
    "memory_mb": 65_536,
    "gpus": 0,
    "timeout_seconds": 14_400,
    "project": scheduler_client.MFT_PROJECT,
    "required_capability": "conda:pyaedt2026v1",
    "env_profile": "pyaedt2026v1",
    "scheduling_profile": "fea_bursty",
    "remote_cwd": scheduler_client.GPFS_RUNS_REMOTE_CWD,
    "priority": 0,
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT,
    ).strip()


def _atomic_json(path, payload):
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


def _load_and_validate_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    stored = unsigned.pop("plan_sha256", None)
    if stored != PLAN_SHA256 or _sha(unsigned) != PLAN_SHA256:
        raise RuntimeError("recovery4 plan seal mismatch")
    if plan.get("solver_revision") != SOLVER or plan.get("library_revision") != LIBRARY:
        raise RuntimeError("recovery4 plan revision mismatch")
    if plan.get("task_count") != 4 or plan.get("concurrency") != 4:
        raise RuntimeError("recovery4 plan task/concurrency mismatch")
    if plan.get("submission_enabled") is not False \
            or plan.get("scheduler_mutation_count") != 0:
        raise RuntimeError("recovery4 source plan is not immutable plan-only evidence")
    if plan.get("resources") != EXPECTED_RESOURCES:
        raise RuntimeError("recovery4 resource contract drifted")
    if plan.get("pilot_gate", {}).get("strict_valid_required") != 4 \
            or plan.get("pilot_gate", {}).get("partial_pass_allowed") is not False:
        raise RuntimeError("recovery4 all-four gate drifted")

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile["timeout_seconds"] = EXPECTED_RESOURCES["timeout_seconds"]
    profile_evidence = plan.get("profile", {})
    if hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest() \
            != profile_evidence.get("file_sha256"):
        raise RuntimeError("standard profile file drifted")
    if _sha(profile) != profile_evidence.get("effective_sha256"):
        raise RuntimeError("effective standard profile drifted")
    if profile.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT \
            or profile.get("cli_flags") != "--thermal --headless":
        raise RuntimeError("standard full-thermal profile contract drifted")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise RuntimeError("recovery4 task list mismatch")
    if tuple(task.get("source_task_id") for task in tasks) != EXPECTED_SOURCES:
        raise RuntimeError("recovery4 source ordering drifted")
    for ordinal, task in enumerate(tasks, start=1):
        expected_name = f"{PREFIX}{ordinal:02d}-src{EXPECTED_SOURCES[ordinal - 1]}"
        if task.get("ordinal") != ordinal or task.get("name") != expected_name:
            raise RuntimeError(f"recovery4 task {ordinal} identity drifted")
        if task.get("resources") != EXPECTED_RESOURCES:
            raise RuntimeError(f"recovery4 task {ordinal} resources drifted")
        if pinned_pilot.candidate_digest(task.get("params")) \
                != task.get("source_params_sha256"):
            raise RuntimeError(f"recovery4 task {ordinal} params drifted")
        identity = scheduler_client.verification_submission_identity(
            task["name"], task["params"], profile, SOLVER, LIBRARY,
        )
        for key in ("dedupe_key", "parameter_digest"):
            if task.get(key) != identity[key]:
                raise RuntimeError(f"recovery4 task {ordinal} {key} drifted")
        if task.get("effective_params") != identity["merged"]:
            raise RuntimeError(f"recovery4 task {ordinal} effective params drifted")
    return plan, profile


def _inventory():
    response = requests.get(
        f"{scheduler_client.SCHEDULER}/api/tasks",
        params={"limit": 10000, "name_prefix": PREFIX}, timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise RuntimeError("scheduler recovery4 inventory is not a list")
    return tasks


def _audit_scheduler(plan):
    inventory = _inventory()
    records = []
    for task in plan["tasks"]:
        exact = [row for row in inventory if row.get("name") == task["name"]]
        if len(exact) > 1:
            raise RuntimeError(f"duplicate exact recovery task name: {task['name']}")
        if exact:
            row = exact[0]
            if row.get("dedupe_key") != task["dedupe_key"] \
                    or str(row.get("project") or "") != scheduler_client.MFT_PROJECT:
                raise RuntimeError(f"conflicting recovery task identity: {task['name']}")
            task_id = int(row["id"])
        else:
            task_id = scheduler_client.reconcile_task_id(
                task["name"], task["dedupe_key"],
            )
        records.append({
            "ordinal": task["ordinal"],
            "name": task["name"],
            "source_task_id": task["source_task_id"],
            "dedupe_key": task["dedupe_key"],
            "task_id": task_id,
            "recovered": task_id is not None,
        })
    capacity = scheduler_client.live_project_submission_snapshot(
        scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
    )
    missing = sum(record["task_id"] is None for record in records)
    if capacity["project_submission_slots"] < missing:
        raise RuntimeError(
            f"recovery4 needs {missing} slots, only "
            f"{capacity['project_submission_slots']} available: {capacity}"
        )
    return records, capacity


def _task_metadata(task_id, expected):
    response = requests.get(
        f"{scheduler_client.SCHEDULER}/api/tasks/{int(task_id)}", timeout=20,
    )
    response.raise_for_status()
    row = response.json()
    checks = {
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
        raise RuntimeError(f"submitted task {task_id} metadata mismatch: {checks}")
    return {key: row.get(key) for key in (
        "id", "name", "status", "project", "dedupe_key", "cpus", "memory_mb",
        "gpus", "timeout_seconds", "required_capability", "env_profile",
        "scheduling_profile", "remote_cwd", "created_at",
    )}


def audit():
    plan, profile = _load_and_validate_plan()
    if _git(REPO_ROOT, "rev-parse", "HEAD") != SOLVER:
        raise RuntimeError("local solver HEAD is not the reviewed recovery SHA")
    if _git(LIBRARY_ROOT, "rev-parse", "HEAD") != LIBRARY:
        raise RuntimeError("clean library HEAD mismatch")
    if _git(LIBRARY_ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("recovery library worktree is dirty")
    deployed = deployment_gate.validate_deployment(
        REPO_ROOT, SOLVER, LIBRARY_ROOT, LIBRARY,
    )
    if scheduler_client.campaign_mutation_lock_is_held():
        records, capacity = _audit_scheduler(plan)
    else:
        with scheduler_client.campaign_mutation_lock():
            records, capacity = _audit_scheduler(plan)
    return plan, profile, deployed, records, capacity


def _load_existing_final(plan):
    if not FINAL_PATH.exists():
        return None
    ledger = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
    unsigned = dict(ledger)
    stored = unsigned.pop("submission_sha256", None)
    if not stored or _sha(unsigned) != stored:
        raise RuntimeError("existing recovery4 submission journal seal mismatch")
    if ledger.get("root_reviewed_plan_sha256") != PLAN_SHA256 \
            or ledger.get("solver_revision") != SOLVER \
            or ledger.get("library_revision") != LIBRARY:
        raise RuntimeError("existing recovery4 submission journal identity mismatch")
    expected = {task["name"]: task for task in plan["tasks"]}
    records = ledger.get("tasks")
    if not isinstance(records, list) or {row.get("name") for row in records} != set(expected):
        raise RuntimeError("existing recovery4 submission task set mismatch")
    for record in records:
        if record.get("dedupe_key") != expected[record["name"]]["dedupe_key"]:
            raise RuntimeError("existing recovery4 submission dedupe mismatch")
        _task_metadata(record.get("task_id"), expected[record["name"]])
    return ledger


def execute():
    plan, profile, deployed, _records, _capacity = audit()
    existing = _load_existing_final(plan)
    if existing is not None:
        return existing
    with scheduler_client.campaign_mutation_lock():
        if _git(REPO_ROOT, "rev-parse", "HEAD") != SOLVER \
                or _git(LIBRARY_ROOT, "rev-parse", "HEAD") != LIBRARY \
                or _git(LIBRARY_ROOT, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("locked recovery deployment identity drifted")
        deployed = deployment_gate.validate_deployment(
            REPO_ROOT, SOLVER, LIBRARY_ROOT, LIBRARY,
        )
        records, capacity = _audit_scheduler(plan)
        ledger = {
            "schema": "thermal-recovery4-submission-v1",
            "created_at": _now(),
            "plan": str(PLAN_PATH.resolve()),
            "root_reviewed_plan_sha256": PLAN_SHA256,
            "solver_revision": SOLVER,
            "library_revision": LIBRARY,
            "deployment": deployed,
            "resources": EXPECTED_RESOURCES,
            "capacity_before": capacity,
            "task_count": 4,
            "submission_policy": "all-four-in-one-mutation-lock-no-cancellation",
            "tasks": records,
        }
        _atomic_json(PARTIAL_PATH, ledger)
        plan_by_name = {task["name"]: task for task in plan["tasks"]}
        mutation_count = 0
        for record in records:
            if record["task_id"] is not None:
                continue
            task = plan_by_name[record["name"]]
            record["task_id"] = scheduler_client.submit_verification(
                name=task["name"],
                workdir=task["workdir"],
                params=task["params"],
                profile=profile,
                mem_mb=EXPECTED_RESOURCES["memory_mb"],
                cpus=EXPECTED_RESOURCES["cpus"],
                solver_revision=SOLVER,
                library_revision=LIBRARY,
            )
            if record["task_id"] is None:
                raise RuntimeError(f"scheduler returned no ID for {record['name']}")
            record["submitted_at"] = _now()
            mutation_count += 1
            _atomic_json(PARTIAL_PATH, ledger)

        for record in records:
            task = plan_by_name[record["name"]]
            record["scheduler_metadata"] = _task_metadata(record["task_id"], task)
        ledger["scheduler_mutation_count"] = mutation_count
        ledger["completed_at"] = _now()
        ledger["capacity_after"] = scheduler_client.live_project_submission_snapshot(
            scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
        )
        ledger["submission_sha256"] = _sha(ledger)
        _atomic_json(FINAL_PATH, ledger)
        PARTIAL_PATH.unlink(missing_ok=True)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reviewed-plan-sha")
    args = parser.parse_args(argv)
    if args.execute:
        if args.plan is None or args.plan.resolve() != PLAN_PATH.resolve():
            parser.error(f"--execute requires --plan {PLAN_PATH}")
        if args.reviewed_plan_sha != PLAN_SHA256:
            parser.error(
                f"--execute requires --reviewed-plan-sha {PLAN_SHA256}"
            )
        result = execute()
    else:
        plan, _profile, deployed, records, capacity = audit()
        result = {
            "mode": "audit-only",
            "plan_sha256": plan["plan_sha256"],
            "deployment": deployed,
            "tasks": records,
            "capacity": capacity,
            "would_mutate": sum(row["task_id"] is None for row in records),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
