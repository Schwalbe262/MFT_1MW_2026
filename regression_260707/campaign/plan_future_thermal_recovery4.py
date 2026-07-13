"""Build a sealed, plan-only four-case thermal recovery pilot.

This module has no scheduler mutation path.  It copies three exact
``monitor_missing`` candidates and one exact known-good candidate from the
sealed SHA754 replacement manifest, then derives new names and dedupe keys for
an explicitly supplied future solver SHA.  Submission remains a separate,
root-reviewed operation after push/deployment validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
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


SOURCE_SOLVER = "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SOURCE_MANIFEST_SHA256 = (
    "f1490f2cda497c9475fe079fb0a04e5adb7686c6f4c99ae28a0f946a918319a8"
)
SOURCE_MANIFEST_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.json"
)
PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
SOURCE_FIRST_TASK_ID = 27755

# Ordered as requested: three exact thermal failures, then one exact good case.
SOURCE_CASES = (
    {
        "role": "monitor_missing",
        "task_id": 27794,
        "manifest_index": 39,
        "candidate_raw_index": 1978,
        "params_sha256": (
            "9073e4460c2a09ac63f446a012ec8dd89b3b1fe7bfcc42cca2e2643e89b46edb"
        ),
        "source_parameter_digest": "65bde1bb0b4f91d5",
    },
    {
        "role": "monitor_missing",
        "task_id": 27928,
        "manifest_index": 173,
        "candidate_raw_index": 2481,
        "params_sha256": (
            "cc4d8492809b400a182253fca8b3c46ec042ddbf1d466e53364e171397e20ede"
        ),
        "source_parameter_digest": "8295238ff8f2b594",
    },
    {
        "role": "monitor_missing",
        "task_id": 27880,
        "manifest_index": 125,
        "candidate_raw_index": 2313,
        "params_sha256": (
            "4f28bf1aefeeea4480ab5b8543b277e412019cd06bd7e3d969dc250c1b7b2635"
        ),
        "source_parameter_digest": "5e9faf4159f56675",
    },
    {
        "role": "known_good",
        "task_id": 27758,
        "manifest_index": 3,
        "candidate_raw_index": 1855,
        "params_sha256": (
            "0dd09f313753ed0bb487e09a595a29f300fb78b12de999fd994f7e8830262c9a"
        ),
        "source_parameter_digest": "864554c785d42d74",
    },
)

RESOURCES = {
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _full_sha(value: str, label: str) -> str:
    text = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise RuntimeError(f"{label} must be a full 40-character git SHA")
    return text


def _load_source_manifest() -> dict:
    payload = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    seal = unsigned.pop("manifest_sha256", None)
    if seal != SOURCE_MANIFEST_SHA256 or _sha(unsigned) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("SHA754 source manifest seal mismatch")
    if (payload.get("solver_revision") != SOURCE_SOLVER
            or payload.get("library_revision") != LIBRARY
            or payload.get("task_count") != 250):
        raise RuntimeError("SHA754 source manifest identity mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 250:
        raise RuntimeError("SHA754 source manifest task list mismatch")
    return payload


def _load_profile() -> dict:
    raw = PROFILE_PATH.read_bytes()
    profile = json.loads(raw.decode("utf-8"))
    if profile.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT:
        raise RuntimeError("standard profile parameter contract drifted")
    if profile.get("cli_flags") != "--thermal --headless":
        raise RuntimeError("standard profile is not a full headless thermal run")
    profile = copy.deepcopy(profile)
    profile["timeout_seconds"] = RESOURCES["timeout_seconds"]
    return profile


def _source_record(manifest: dict, source: dict) -> dict:
    index = int(source["manifest_index"])
    record = manifest["tasks"][index]
    checks = {
        "index": record.get("index") == index,
        "task_id": SOURCE_FIRST_TASK_ID + index == source["task_id"],
        "raw_index": (
            record.get("candidate_raw_index") == source["candidate_raw_index"]
        ),
        "params_sha": record.get("params_sha256") == source["params_sha256"],
        "parameter_digest": (
            record.get("parameter_digest") == source["source_parameter_digest"]
        ),
        "recomputed_params_sha": (
            pinned_pilot.candidate_digest(record.get("params"))
            == source["params_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"source case {source['task_id']} drifted from sealed manifest: {checks}"
        )
    return record


def build_plan(solver_revision: str, library_revision: str = LIBRARY) -> dict:
    solver_revision = _full_sha(solver_revision, "solver_revision")
    library_revision = _full_sha(library_revision, "library_revision")
    if solver_revision == SOURCE_SOLVER:
        raise RuntimeError("recovery pilot requires a future solver revision")
    if library_revision != LIBRARY:
        raise RuntimeError("recovery pilot library revision is not root-reviewed")

    manifest = _load_source_manifest()
    profile = _load_profile()
    profile_raw_sha = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    tasks = []
    for ordinal, source in enumerate(SOURCE_CASES, start=1):
        record = _source_record(manifest, source)
        params = json.loads(_canonical(record["params"]))
        name = (
            f"mft-recovery4-s{solver_revision[:7]}-l{library_revision[:7]}-"
            f"{ordinal:02d}-src{source['task_id']}"
        )
        identity = scheduler_client.verification_submission_identity(
            name, params, profile, solver_revision, library_revision,
        )
        is_failed_source = source["role"] == "monitor_missing"
        tasks.append({
            "ordinal": ordinal,
            "role": source["role"],
            "name": name,
            "workdir": f"mft_recovery4_{ordinal:02d}_src{source['task_id']}",
            "source_task_id": source["task_id"],
            "source_manifest_index": source["manifest_index"],
            "source_candidate_raw_index": source["candidate_raw_index"],
            "source_params_sha256": source["params_sha256"],
            "source_parameter_digest": source["source_parameter_digest"],
            "params": params,
            "effective_params": identity["merged"],
            "parameter_digest": identity["parameter_digest"],
            "dedupe_key": identity["dedupe_key"],
            "resources": copy.deepcopy(RESOURCES),
            "acceptance": {
                "strict_valid_required": True,
                "failed_source_recovery_case": is_failed_source,
                "thermal_entrypoint_exact": (
                    "ThermalSetup" if is_failed_source else None
                ),
                "analyze_all_forbidden": is_failed_source,
                "fresh_monitor_required": is_failed_source,
                "startup_retry_max": 1 if is_failed_source else None,
                "known_good_nonregression": source["role"] == "known_good",
            },
        })

    names = {task["name"] for task in tasks}
    dedupes = {task["dedupe_key"] for task in tasks}
    parameter_payloads = {task["source_params_sha256"] for task in tasks}
    if len(names) != 4 or len(dedupes) != 4 or len(parameter_payloads) != 4:
        raise RuntimeError("future recovery pilot names/dedupes/parameters are not unique")

    plan = {
        "schema_version": 1,
        "created_at": _now(),
        "mode": "plan_only",
        "submission_enabled": False,
        "scheduler_mutation_count": 0,
        "task_count": 4,
        "concurrency": 4,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "source_solver_revision": SOURCE_SOLVER,
        "source_manifest": str(SOURCE_MANIFEST_PATH.resolve()),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "profile": {
            "path": str(PROFILE_PATH.resolve()),
            "file_sha256": profile_raw_sha,
            "effective_sha256": _sha(profile),
            "cli_flags": profile["cli_flags"],
            "param_overrides": profile["param_overrides"],
        },
        "resources": copy.deepcopy(RESOURCES),
        "activation_requirements": {
            "solver_revision_pushed": False,
            "deployment_gate_passed": False,
            "root_reviewed_plan_sha256": None,
            "all_four_submit_concurrently": True,
            "submission_implementation_included": False,
        },
        "pilot_gate": {
            "policy": "all_four_strict_valid",
            "strict_valid_required": 4,
            "task_count": 4,
            "partial_pass_allowed": False,
            "failed_source_cases": {
                "count": 3,
                "each_requires_exact_thermal_entrypoint": "ThermalSetup",
                "analyze_all_forbidden": True,
                "fresh_monitor_required": True,
                "startup_retry_max": 1,
            },
            "known_good_cases": {
                "count": 1,
                "nonregression_required": True,
            },
        },
        "tasks": tasks,
    }
    plan["plan_sha256"] = _sha(plan)
    return plan


def _write_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        plan, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")
    try:
        with staged.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded != plan:
        raise RuntimeError(f"future recovery plan readback mismatch: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", default=LIBRARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = build_plan(args.solver_revision, args.library_revision)
    if args.output is not None:
        _write_plan(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
