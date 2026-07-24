"""Stage and submit the isolated 2026-07-26 MFT NSGA-II campaign.

The 1MW_MFT repository owns the scientific bundle and worker command.  The
separate Scheduler service owns only placement and task lifecycle.  This tool
therefore consumes a fully prepared goal bundle, publishes one immutable GPFS
tree, and POSTs one already sealed goal task per Slurm task.  It never edits
or imports Scheduler application code into the MFT source tree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shlex
import sys
import time
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_20260726_launch as goal  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402


PLAN_SCHEMA = "mft-goal-20260726-slurm-offload-plan-v1"
DEPLOYMENT_SCHEMA = "mft-goal-20260726-slurm-deployment-v1"
SUBMISSION_SCHEMA = "mft-goal-20260726-slurm-submissions-v1"
RELOCATION_SCHEMA = "mft-goal-20260726-worker-relocation-v1"
HARVEST_SCHEMA = "mft-goal-20260726-slurm-harvest-v1"
HARVEST_READINESS_SCHEMA = (
    "mft-goal-20260726-slurm-harvest-readiness-v1"
)
TASK_HARVEST_RECEIPT_SCHEMA = (
    "mft-goal-20260726-slurm-task-harvest-receipt-v1"
)
DEFAULT_LOCAL_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "slurm_offload"
)
DEFAULT_REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_goal_20260726"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
DEFAULT_STAGING_ACCOUNT = "harry261"
CPUS_PER_TASK = 8
MEMORY_MB_PER_TASK = 65_536
TIMEOUT_SECONDS = 14_400
MAX_WORKERS_PER_NODE = 8
EXPECTED_CAMPAIGN_TASKS = 512
MAX_STATUS_WORKERS = 4
MAX_ACCOUNT_WORKERS = 4
DEFAULT_HARVEST_RETRIES = 3
RESULT_JSON_MAX_BYTES = 1024 * 1024
RESULT_JSON_SFTP_MAX_BYTES = 8 * 1024 * 1024
TERMINAL_ARTIFACT_ROLES = (
    "terminal_physical_candidates",
    "terminal_physical_candidates_manifest",
)
TERMINAL_ARTIFACT_NAMES = {
    "terminal_physical_candidates": "terminal_physical_candidates.csv",
    "terminal_physical_candidates_manifest": (
        "terminal_physical_candidates.manifest.json"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = dict(value)
    observed = result.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != canonical_sha256(result):
        raise RuntimeError(f"{schema} seal mismatch")
    return dict(value)


def _safe_relative(value: str, label: str) -> str:
    if "\\" in str(value):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _add_file(
    files: dict[str, dict[str, Any]],
    sources: dict[str, str],
    relative: str,
    source: Path,
    kind: str,
    *,
    expected_sha256: str | None = None,
) -> None:
    relative = _safe_relative(relative, "deployment path")
    path = source.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"regular file required: {path}")
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"source fingerprint mismatch: {path}")
    if relative in files:
        raise RuntimeError(f"duplicate deployment path: {relative}")
    files[relative] = {
        "sha256": digest,
        "size": int(path.stat().st_size),
        "kind": kind,
    }
    sources[relative] = str(path)


def _task_paths(bundle_root: Path, bundle: Mapping[str, Any]) -> list[Path]:
    relatives = bundle.get("task_relative_paths")
    digests = bundle.get("task_payload_sha256")
    if (
        not isinstance(relatives, list)
        or not isinstance(digests, list)
        or len(relatives) != len(digests)
        or len(relatives) != int(bundle.get("task_count", -1))
    ):
        raise RuntimeError("goal bundle task inventory mismatch")
    result: list[Path] = []
    for relative, expected in zip(relatives, digests):
        safe = _safe_relative(str(relative), "goal task path")
        path = (bundle_root / safe).resolve(strict=True)
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise RuntimeError("goal task escaped bundle") from exc
        task = goal.validate_task_payload(_read_json(path))
        if task["payload_sha256"] != expected:
            raise RuntimeError("goal task payload identity mismatch")
        result.append(path)
    return result


def _deployment_identity(
    *,
    bundle: Mapping[str, Any],
    code_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    candidate_sha256: str,
    quality_sha256: str,
    dataset_sha256: str,
    profile_sha256: str,
) -> str:
    stable = {
        "goal_bundle_payload_sha256": bundle["payload_sha256"],
        "code_manifest_payload_sha256": code_manifest["payload_sha256"],
        "training_run_id": report["training_run_id"],
        "generation_report_sha256": hashlib.sha256(
            _canonical_bytes(report)
        ).hexdigest(),
        "candidate_sha256": candidate_sha256,
        "quality_status_sha256": quality_sha256,
        "dataset_sha256": dataset_sha256,
        "profile_sha256": profile_sha256,
    }
    return canonical_sha256(stable)


def build_plan(
    *,
    goal_bundle_root: Path,
    generation: Path,
    candidate: Path,
    quality_status: Path,
    dataset: Path,
    profile: Path,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_root = goal_bundle_root.resolve(strict=True)
    if not bundle_root.is_dir():
        raise RuntimeError("goal bundle root must be a directory")
    bundle_path = bundle_root / "bundle_manifest.json"
    local_preflight_path = bundle_root / "local_preflight.json"
    scheduler_manifest_path = bundle_root / "scheduler_manifest.json"
    code_manifest_path = bundle_root / "code_manifest.json"
    bundle = _validate_seal(
        _read_json(bundle_path), goal.BUNDLE_SCHEMA
    )
    local_preflight = _validate_seal(
        _read_json(local_preflight_path), goal.LOCAL_PREFLIGHT_SCHEMA
    )
    scheduler_manifest = _validate_seal(
        _read_json(scheduler_manifest_path), goal.SCHEDULER_MANIFEST_SCHEMA
    )
    code_manifest = _read_json(code_manifest_path)
    code_schema = "mft-goal-20260726-code-inventory-v1"
    _validate_seal(code_manifest, code_schema)
    if (
        bundle.get("local_preflight_sha256")
        != local_preflight["payload_sha256"]
        or scheduler_manifest.get("bundle_sha256")
        != bundle["payload_sha256"]
        or bundle.get("task_count") != scheduler_manifest.get(
            "maximum_parallel_tasks"
        )
        or code_manifest.get("campaign_id") != "mft-goal-20260726"
        or code_manifest.get("scheduler_project_code_included") is not False
        or code_manifest.get("remote_git_checkout_required") is not False
    ):
        raise RuntimeError("goal campaign manifests are not mutually bound")

    generation = generation.resolve(strict=True)
    candidate = candidate.resolve(strict=True)
    quality_status = quality_status.resolve(strict=True)
    dataset = dataset.resolve(strict=True)
    profile = profile.resolve(strict=True)
    report_path = generation / "train_report.json"
    report = _read_json(report_path)
    artifacts = report.get("artifacts")
    if (
        generation.parent.name != "generations"
        or report.get("training_run_id") != generation.name
        or not isinstance(artifacts, dict)
        or not artifacts
    ):
        raise RuntimeError("G0 generation contract mismatch")

    task_paths = _task_paths(bundle_root, bundle)
    first_task = goal.validate_task_payload(_read_json(task_paths[0]))
    identity = first_task["source_identity"]
    candidate_sha = _sha256_file(candidate)
    quality_sha = _sha256_file(quality_status)
    dataset_sha = _sha256_file(dataset)
    profile_canonical_sha = goal.adapter.training_profile_sha256(
        _read_json(profile)
    )
    if (
        _sha256_file(report_path) != identity["train_report_sha256"]
        or candidate_sha != identity["candidate_sha256"]
        or quality_sha != identity["quality_status_sha256"]
        or dataset_sha != identity["dataset_sha256"]
        or profile_canonical_sha != identity["profile_sha256"]
        or canonical_sha256(artifacts)
        != identity["evaluation_model_sha256"]
        or code_manifest["payload_sha256"]
        != identity["code_manifest_payload_sha256"]
        or code_manifest["code_inventory_sha256"]
        != identity["code_inventory_sha256"]
        or code_manifest["code_revision"] != identity["code_revision"]
    ):
        raise RuntimeError("goal G0/task source identity mismatch")
    for task_path in task_paths[1:]:
        task = goal.validate_task_payload(_read_json(task_path))
        if task["source_identity"] != identity:
            raise RuntimeError("goal tasks mix source identities")

    deployment_sha = _deployment_identity(
        bundle=bundle,
        code_manifest=code_manifest,
        report=report,
        candidate_sha256=candidate_sha,
        quality_sha256=quality_sha,
        dataset_sha256=dataset_sha,
        profile_sha256=profile_canonical_sha,
    )
    bundle_id = f"mft-goal-{deployment_sha[:24]}"
    remote_base = remote_root.rstrip("/")
    remote_bundle = f"{remote_base}/{bundle_id}"
    plan_dir = local_root.resolve() / bundle_id
    if plan_dir.exists():
        raise RuntimeError(f"offload plan already exists: {plan_dir}")
    plan_dir.mkdir(parents=True)

    files: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    code_inventory = code_manifest.get("code_inventory")
    if (
        not isinstance(code_inventory, dict)
        or code_manifest.get("files") != code_inventory
        or code_manifest.get("code_inventory_sha256")
        != canonical_sha256(code_inventory)
    ):
        raise RuntimeError("goal code inventory mismatch")
    for relative, record in sorted(code_inventory.items()):
        relative = _safe_relative(relative, "goal code inventory path")
        local = (bundle_root / relative).resolve(strict=True)
        try:
            local.relative_to(bundle_root)
        except ValueError as exc:
            raise RuntimeError("goal code inventory escaped bundle") from exc
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise RuntimeError("goal code inventory record mismatch")
        _add_file(
            files,
            sources,
            relative,
            local,
            "goal_code",
            expected_sha256=str(record["sha256"]),
        )
        if int(record["size"]) != files[relative]["size"]:
            raise RuntimeError("goal code inventory size mismatch")

    campaign_files = {
        "bundle_manifest.json": bundle_path,
        "local_preflight.json": local_preflight_path,
        "scheduler_manifest.json": scheduler_manifest_path,
        "code_manifest.json": code_manifest_path,
    }
    for name, source in campaign_files.items():
        _add_file(
            files,
            sources,
            f"artifacts/campaign/{name}",
            source,
            "goal_campaign_manifest",
        )
    for task_path in task_paths:
        _add_file(
            files,
            sources,
            f"artifacts/campaign/tasks/{task_path.name}",
            task_path,
            "goal_task",
        )

    generation_remote = (
        f"artifacts/g0/registry/generations/{generation.name}"
    )
    _add_file(
        files,
        sources,
        f"{generation_remote}/train_report.json",
        report_path,
        "generation_report",
        expected_sha256=identity["train_report_sha256"],
    )
    actual_generation_files = {
        path.relative_to(generation).as_posix()
        for path in generation.rglob("*")
        if path.is_file() and path.name != "train_report.json"
    }
    if actual_generation_files != set(artifacts):
        raise RuntimeError("G0 generation file inventory mismatch")
    for relative, expected in sorted(artifacts.items()):
        safe = _safe_relative(relative, "G0 artifact")
        _add_file(
            files,
            sources,
            f"{generation_remote}/{safe}",
            generation / safe,
            "model_artifact",
            expected_sha256=str(expected),
        )
    remote_roles = {
        "generation": f"{remote_bundle}/{generation_remote}",
        "candidate": f"{remote_bundle}/artifacts/g0/candidate.json",
        "quality_status": (
            f"{remote_bundle}/artifacts/g0/quality_status.json"
        ),
        "code_root": f"{remote_bundle}/artifacts/code",
        "dataset": f"{remote_bundle}/artifacts/g0/dataset/train.parquet",
        "profile": f"{remote_bundle}/artifacts/g0/profile.json",
    }
    _add_file(
        files,
        sources,
        "artifacts/g0/candidate.json",
        candidate,
        "generation_candidate",
        expected_sha256=identity["candidate_sha256"],
    )
    _add_file(
        files,
        sources,
        "artifacts/g0/quality_status.json",
        quality_status,
        "quality_status",
        expected_sha256=identity["quality_status_sha256"],
    )
    _add_file(
        files,
        sources,
        "artifacts/g0/dataset/train.parquet",
        dataset,
        "training_dataset",
        expected_sha256=identity["dataset_sha256"],
    )
    _add_file(
        files,
        sources,
        "artifacts/g0/profile.json",
        profile,
        "training_profile",
    )

    relocation_sources: dict[int, str] = {}
    for task_path in task_paths:
        task = goal.validate_task_payload(_read_json(task_path))
        relocation = _sealed(
            {
                "schema_version": RELOCATION_SCHEMA,
                "task_payload_sha256": task["payload_sha256"],
                "paths": dict(remote_roles),
                "code_manifest_path": (
                    f"{remote_bundle}/artifacts/campaign/code_manifest.json"
                ),
                "source_absolute_paths_are_documentary_only": True,
                "remote_git_checkout_required": False,
            }
        )
        relocation_path = (
            plan_dir
            / "relocations"
            / f"seed-{task['seed']}-n1-{task['fixed_primary_turns']}.json"
        )
        _atomic_json(relocation_path, relocation)
        relative = (
            "artifacts/campaign/relocations/" + relocation_path.name
        )
        _add_file(
            files,
            sources,
            relative,
            relocation_path,
            "worker_relocation",
        )
        relocation_sources[int(task["seed"])] = relative

    requirements_path = plan_dir / "requirements.lock"
    requirements_path.write_bytes(transport.REQUIREMENTS_LOCK.encode("utf-8"))
    _add_file(
        files,
        sources,
        "artifacts/runtime/requirements.lock",
        requirements_path,
        "runtime_lock",
    )

    deployment_stable = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "campaign_id": "mft-goal-20260726",
        "deployment_identity_sha256": deployment_sha,
        "goal_bundle_payload_sha256": bundle["payload_sha256"],
        "code_manifest_payload_sha256": code_manifest["payload_sha256"],
        "code_inventory_sha256": code_manifest["code_inventory_sha256"],
        "source_identity": identity,
        "source_paths": {
            **remote_roles,
            "code_manifest": (
                f"{remote_bundle}/artifacts/campaign/code_manifest.json"
            ),
        },
        "relocations": relocation_sources,
        "files": files,
        "runtime": {
            "python_major_minor": "3.11",
            "critical_packages": dict(transport.CRITICAL_PACKAGES),
        },
        "execution_contract": {
            "population": goal.POPULATION,
            "generations": goal.GENERATIONS,
            "inference_threads": goal.INFERENCE_THREADS,
            "one_seed_per_scheduler_task": True,
            "aedt_used": False,
            "gpus": 0,
        },
        "scheduler_project_source_included": False,
        "scheduler_service_modified": False,
    }
    deployment_contract_sha = canonical_sha256(deployment_stable)
    deployment = {
        **deployment_stable,
        "bundle_id": bundle_id,
        "contract_sha256": deployment_contract_sha,
    }
    deployment_path = plan_dir / "bundle_manifest.json"
    source_map_path = plan_dir / "local_sources.json"
    plan_path = plan_dir / "offload_plan.json"
    _atomic_json(deployment_path, deployment)
    _atomic_json(source_map_path, sources)
    plan = {
        "schema_version": transport.PLAN_SCHEMA,
        "goal_schema_version": PLAN_SCHEMA,
        "created_at": _now(),
        "bundle_id": bundle_id,
        "contract_sha256": deployment_contract_sha,
        "bundle_manifest_sha256": _sha256_file(deployment_path),
        "bundle_manifest": str(deployment_path),
        "local_sources": str(source_map_path),
        "local_plan_dir": str(plan_dir),
        "remote_root": remote_base,
        "remote_bundle": remote_bundle,
        "runtime_requirements": str(requirements_path),
        "goal_bundle_root": str(bundle_root),
        "goal_bundle_manifest": str(bundle_path),
        "goal_bundle_payload_sha256": bundle["payload_sha256"],
        "task_count": len(task_paths),
        "task_paths": [str(path) for path in task_paths],
        "relocation_sources": relocation_sources,
        "recommended_resources": {
            "cpus": CPUS_PER_TASK,
            "memory_mb": MEMORY_MB_PER_TASK,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
            "maximum_parallel_tasks": len(task_paths),
        },
    }
    _atomic_json(plan_path, plan)
    return plan, deployment


def load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(path.resolve(strict=True))
    if (
        plan.get("schema_version") != transport.PLAN_SCHEMA
        or plan.get("goal_schema_version") != PLAN_SCHEMA
    ):
        raise RuntimeError("goal offload plan schema mismatch")
    deployment_path = Path(plan["bundle_manifest"]).resolve(strict=True)
    deployment = _read_json(deployment_path)
    if (
        deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("bundle_id") != plan.get("bundle_id")
        or _sha256_file(deployment_path)
        != plan.get("bundle_manifest_sha256")
    ):
        raise RuntimeError("goal deployment identity mismatch")
    stable = {
        key: value
        for key, value in deployment.items()
        if key not in {"bundle_id", "contract_sha256"}
    }
    if deployment.get("contract_sha256") != canonical_sha256(stable):
        raise RuntimeError("goal deployment contract seal mismatch")
    return plan, deployment


def _task_selection(
    plan: Mapping[str, Any],
    *,
    phase: str,
    wave: int | None = None,
    seeds: set[int] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    root = Path(plan["goal_bundle_root"]).resolve(strict=True)
    scheduler = _validate_seal(
        _read_json(root / "scheduler_manifest.json"),
        goal.SCHEDULER_MANIFEST_SCHEMA,
    )
    canary = {int(value) for value in scheduler.get("canary_seeds") or []}
    waves = {
        int(key): {int(value) for value in values}
        for key, values in (scheduler.get("rolling_waves") or {}).items()
    }
    if phase == "canary":
        selected = canary
    elif phase == "wave":
        if wave is None or wave not in waves:
            raise RuntimeError("requested rolling wave is unavailable")
        selected = waves[wave]
    elif phase == "all":
        selected = None
    elif phase == "seeds":
        if not seeds:
            raise RuntimeError("explicit seed selection is empty")
        selected = set(seeds)
    else:
        raise RuntimeError("unknown submission phase")
    result = []
    for raw in plan["task_paths"]:
        path = Path(raw).resolve(strict=True)
        task = goal.validate_task_payload(_read_json(path))
        if selected is None or int(task["seed"]) in selected:
            result.append((path, task))
    if selected is not None and {int(task["seed"]) for _, task in result} != selected:
        raise RuntimeError("submission seed selection is incomplete")
    return result


def scheduler_payload(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    priority: int = 10,
) -> dict[str, Any]:
    seed = int(task["seed"])
    turns = int(task["fixed_primary_turns"])
    relocation = plan["relocation_sources"].get(str(seed))
    if relocation is None:
        relocation = plan["relocation_sources"].get(seed)
    if not isinstance(relocation, str):
        raise RuntimeError("task relocation is unavailable")
    command = "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:'
            '?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) '
            'payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path" >&2; exit 66 ;; esac',
            'output="runs/task-${SLURM_SCHED_TASK_ID:?missing task id}"',
            'test ! -e "$output"',
            "exec python artifacts/code/tools/mft_goal_20260726_launch.py "
            'execute-seed --payload "$payload_path" '
            f"--relocation {shlex.quote(relocation)} "
            '--output "$output"',
        ]
    )
    dedupe = canonical_sha256(
        {
            "deployment": plan["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "cpus": CPUS_PER_TASK,
            "memory_mb": MEMORY_MB_PER_TASK,
            "timeout_seconds": TIMEOUT_SECONDS,
        }
    )
    return {
        "name": f"mft-goal-nsga-s{seed}-n1-{turns}",
        "remote_cwd": plan["remote_bundle"],
        "command": command,
        "payload_json": dict(task),
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": CPUS_PER_TASK,
        "memory_mb": MEMORY_MB_PER_TASK,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": int(priority),
        "timeout_seconds": TIMEOUT_SECONDS,
        "dedupe_key": f"mft-goal-20260726-nsga:{dedupe}",
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
    }


def submit(
    *,
    plan_path: Path,
    phase: str,
    wave: int | None = None,
    seeds: set[int] | None = None,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = DEFAULT_STAGING_ACCOUNT,
    priority: int = 10,
    apply: bool = False,
) -> dict[str, Any]:
    plan, _deployment = load_plan(plan_path)
    selected = _task_selection(
        plan, phase=phase, wave=wave, seeds=seeds
    )
    payloads = [
        scheduler_payload(plan=plan, task=task, priority=priority)
        for _path, task in selected
    ]
    result = {
        "schema_version": SUBMISSION_SCHEMA,
        "apply": bool(apply),
        "bundle_id": plan["bundle_id"],
        "phase": phase,
        "wave": wave,
        "task_count": len(payloads),
        "payloads": payloads,
        "submissions": [],
    }
    if not apply:
        return result
    account, ssh_session = transport._account(
        accounts_path, scheduler_source, staging_account
    )
    with ssh_session(account, default_timeout=60) as session:
        if not transport._remote_ready(
            session,
            plan["remote_bundle"],
            plan["bundle_manifest_sha256"],
        ):
            raise RuntimeError("remote goal deployment is not ready")
    for payload in payloads:
        response = transport._api_json(
            scheduler_url.rstrip("/") + "/api/tasks",
            method="POST",
            payload=payload,
            timeout=30,
        )
        result["submissions"].append(
            {
                "task_id": int(response["task_id"]),
                "deduped": bool(response.get("deduped")),
                "name": payload["name"],
                "task_payload_sha256": payload["payload_json"][
                    "payload_sha256"
                ],
                "seed": int(payload["payload_json"]["seed"]),
                "fixed_primary_turns": int(
                    payload["payload_json"]["fixed_primary_turns"]
                ),
                "submitted_at": _now(),
            }
        )
    ledger_path = Path(plan["local_plan_dir"]) / "submissions.json"
    previous: list[dict[str, Any]] = []
    if ledger_path.is_file():
        ledger = _read_json(ledger_path)
        if (
            ledger.get("schema_version") != SUBMISSION_SCHEMA
            or ledger.get("bundle_id") != plan["bundle_id"]
        ):
            raise RuntimeError("existing submission ledger drifted")
        previous = list(ledger.get("submissions") or [])
    combined = previous + result["submissions"]
    task_ids = [int(item["task_id"]) for item in combined]
    identities = [str(item["task_payload_sha256"]) for item in combined]
    if len(task_ids) != len(set(task_ids)) or len(identities) != len(set(identities)):
        raise RuntimeError("submission ledger contains duplicate task identity")
    _atomic_json(
        ledger_path,
        {
            "schema_version": SUBMISSION_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "scheduler_url": scheduler_url.rstrip("/"),
            "submissions": combined,
            "updated_at": _now(),
        },
    )
    result["ledger"] = str(ledger_path)
    return result


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.resolve(strict=True).read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value, payload


def _submission_identity(
    entries: Iterable[Mapping[str, Any]],
) -> str:
    stable = [
        {
            "task_id": int(item["task_id"]),
            "seed": int(item["seed"]),
            "fixed_primary_turns": int(item["fixed_primary_turns"]),
            "name": str(item["name"]),
            "task_payload_sha256": str(item["task_payload_sha256"]),
        }
        for item in entries
    ]
    stable.sort(key=lambda item: (item["seed"], item["task_id"]))
    return canonical_sha256(stable)


def _authenticate_submission_entries(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    expected_tasks: Mapping[str, Mapping[str, Any]],
    scheduler_url: str,
) -> list[dict[str, Any]]:
    """Bind all 512 Scheduler IDs to the original sealed task ledger."""

    if (
        ledger.get("schema_version") != SUBMISSION_SCHEMA
        or ledger.get("bundle_id") != plan.get("bundle_id")
        or str(ledger.get("scheduler_url") or "").rstrip("/")
        != scheduler_url.rstrip("/")
    ):
        raise RuntimeError("submission ledger header mismatch")
    raw = ledger.get("submissions")
    if not isinstance(raw, list) or len(raw) != EXPECTED_CAMPAIGN_TASKS:
        raise RuntimeError(
            f"submission ledger must contain exactly "
            f"{EXPECTED_CAMPAIGN_TASKS} entries"
        )
    if len(expected_tasks) != EXPECTED_CAMPAIGN_TASKS:
        raise RuntimeError(
            f"prepared bundle must contain exactly "
            f"{EXPECTED_CAMPAIGN_TASKS} tasks"
        )

    authenticated: list[dict[str, Any]] = []
    task_ids: set[int] = set()
    seeds: set[int] = set()
    payloads: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("submission ledger entry must be an object")
        try:
            task_id = int(item["task_id"])
            seed = int(item["seed"])
            turns = int(item["fixed_primary_turns"])
            payload_sha = str(item["task_payload_sha256"])
            name = str(item["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("submission ledger entry is incomplete") from exc
        if (
            isinstance(item.get("task_id"), bool)
            or isinstance(item.get("seed"), bool)
            or isinstance(item.get("fixed_primary_turns"), bool)
            or task_id <= 0
            or item.get("deduped") is not False
        ):
            raise RuntimeError("submission ledger entry identity is invalid")
        if task_id in task_ids or seed in seeds or payload_sha in payloads:
            raise RuntimeError(
                "submission ledger contains duplicate task/seed/payload identity"
            )
        expected = expected_tasks.get(payload_sha)
        if expected is None:
            raise RuntimeError(
                "submission payload is not in the prepared bundle"
            )
        expected_name = scheduler_payload(plan=plan, task=expected)["name"]
        if (
            seed != int(expected["seed"])
            or turns != int(expected["fixed_primary_turns"])
            or name != expected_name
            or not str(item.get("submitted_at") or "")
        ):
            raise RuntimeError("submission entry does not match its sealed task")
        task_ids.add(task_id)
        seeds.add(seed)
        payloads.add(payload_sha)
        authenticated.append(
            {
                "task_id": task_id,
                "seed": seed,
                "fixed_primary_turns": turns,
                "name": name,
                "task_payload_sha256": payload_sha,
                "task": expected,
            }
        )
    if payloads != set(expected_tasks):
        raise RuntimeError(
            "submission ledger is not an exact cover of the prepared bundle"
        )
    authenticated.sort(key=lambda item: item["task_id"])
    return authenticated


def authenticate_harvest_context(
    *,
    plan_path: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Authenticate local plan, source bundle and complete submission ledger."""

    plan, deployment = load_plan(plan_path)
    manifest_path = Path(plan["goal_bundle_manifest"]).resolve(strict=True)
    bundle, expected_tasks = goal.load_bundle_task_ledger(manifest_path)
    planned_tasks = _task_paths(
        Path(plan["goal_bundle_root"]).resolve(strict=True), bundle
    )
    planned_payloads = {
        goal.validate_task_payload(_read_json(path))["payload_sha256"]
        for path in planned_tasks
    }
    remote = PurePosixPath(str(plan.get("remote_bundle") or ""))
    source_code_manifest = str(
        (deployment.get("source_paths") or {}).get("code_manifest") or ""
    )
    code_manifest_suffix = "/artifacts/campaign/code_manifest.json"
    expected_remote = (
        source_code_manifest[: -len(code_manifest_suffix)]
        if source_code_manifest.endswith(code_manifest_suffix)
        else ""
    )
    planned_path_set = {
        str(path.resolve(strict=True)) for path in planned_tasks
    }
    recorded_path_set = {
        str(Path(path).resolve(strict=True))
        for path in plan.get("task_paths") or []
    }
    if (
        not remote.is_absolute()
        or ".." in remote.parts
        or remote.as_posix() != expected_remote
        or int(plan.get("task_count", -1)) != EXPECTED_CAMPAIGN_TASKS
        or int(bundle.get("task_count", -1)) != EXPECTED_CAMPAIGN_TASKS
        or set(expected_tasks) != planned_payloads
        or planned_path_set != recorded_path_set
        or Path(plan["goal_bundle_root"]).resolve(strict=True)
        != manifest_path.parent
        or plan.get("goal_bundle_payload_sha256")
        != bundle.get("payload_sha256")
        or deployment.get("goal_bundle_payload_sha256")
        != bundle.get("payload_sha256")
        or plan.get("relocation_sources") != deployment.get("relocations")
    ):
        raise RuntimeError("offload plan and prepared bundle identity mismatch")

    ledger_path = Path(plan["local_plan_dir"]) / "submissions.json"
    ledger, ledger_bytes = _read_json_bytes(ledger_path)
    submissions = _authenticate_submission_entries(
        plan=plan,
        ledger=ledger,
        expected_tasks=expected_tasks,
        scheduler_url=scheduler_url,
    )
    root = (
        results_root.resolve()
        if results_root is not None
        else (Path(plan["local_plan_dir"]) / "harvest").resolve()
    )
    return {
        "plan": plan,
        "deployment": deployment,
        "bundle": bundle,
        "expected_tasks": expected_tasks,
        "submissions": submissions,
        "submission_ledger_path": str(ledger_path.resolve(strict=True)),
        "submission_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "submission_identity_sha256": _submission_identity(submissions),
        "scheduler_url": scheduler_url.rstrip("/"),
        "results_root": root,
    }


def _api_json_with_retry(url: str, retries: int) -> dict[str, Any]:
    return transport._api_json_with_retry(url, retries)


def _validate_scheduler_task(
    *,
    status: Mapping[str, Any],
    submission: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    task_id = int(submission["task_id"])
    if int(status.get("task_id", status.get("id", -1))) != task_id:
        raise RuntimeError("Scheduler returned a different task ID")
    expected = scheduler_payload(
        plan=plan,
        task=submission["task"],
        priority=10,
    )
    exact = {
        "name": expected["name"],
        "remote_cwd": expected["remote_cwd"],
        "required_capability": expected["required_capability"],
        "env_profile": expected["env_profile"],
        "cpus": expected["cpus"],
        "memory_mb": expected["memory_mb"],
        "scheduling_profile": expected["scheduling_profile"],
        "aedt_backend": expected["aedt_backend"],
        "gpus": expected["gpus"],
        "priority": expected["priority"],
        "timeout_seconds": expected["timeout_seconds"],
        "dedupe_key": expected["dedupe_key"],
        "max_workers_per_node": expected["max_workers_per_node"],
    }
    for key, value in exact.items():
        observed = status.get(key)
        if isinstance(value, int):
            if isinstance(observed, bool) or int(observed or 0) != value:
                raise RuntimeError(f"Scheduler task {key} mismatch")
        elif str(observed or "") != str(value):
            raise RuntimeError(f"Scheduler task {key} mismatch")
    if str(status.get("project") or ""):
        raise RuntimeError("Scheduler task unexpectedly belongs to a project")
    stable = {
        "task_id": task_id,
        "task_payload_sha256": submission["task_payload_sha256"],
        **exact,
    }
    return canonical_sha256(stable)


def _query_one_status(
    *,
    submission: Mapping[str, Any],
    plan: Mapping[str, Any],
    scheduler_url: str,
    retries: int,
) -> dict[str, Any]:
    task_id = int(submission["task_id"])
    row = {
        "task_id": task_id,
        "seed": int(submission["seed"]),
        "fixed_primary_turns": int(submission["fixed_primary_turns"]),
        "task_payload_sha256": str(submission["task_payload_sha256"]),
        "scheduler_status": None,
        "exit_code": None,
        "account_name": None,
        "actual_node_name": None,
        "scheduler_task_identity_sha256": None,
        "state_class": "query_error",
        "harvested": False,
        "error": None,
    }
    try:
        status = _api_json_with_retry(
            scheduler_url.rstrip("/") + f"/api/tasks/{task_id}",
            retries,
        )
        identity = _validate_scheduler_task(
            status=status,
            submission=submission,
            plan=plan,
        )
        raw_exit = status.get("exit_code")
        if isinstance(raw_exit, bool):
            raise RuntimeError("Scheduler exit code has invalid type")
        exit_code = None if raw_exit is None else int(raw_exit)
        scheduler_status = str(status.get("status") or "")
        row.update(
            {
                "scheduler_status": scheduler_status,
                "exit_code": exit_code,
                "account_name": str(status.get("account_name") or "") or None,
                "actual_node_name": (
                    str(
                        status.get("actual_node_name")
                        or status.get("allocation_node_name")
                        or ""
                    )
                    or None
                ),
                "scheduler_task_identity_sha256": identity,
            }
        )
        if scheduler_status == "completed" and exit_code == 0:
            if not row["account_name"]:
                raise RuntimeError("completed task has no owning account")
            row["state_class"] = "completed_exit_zero"
        elif scheduler_status in {"failed", "cancelled"} or (
            scheduler_status == "completed" and exit_code != 0
        ):
            row["state_class"] = "terminal_failure"
            row["error"] = (
                f"scheduler_terminal_failure:{scheduler_status}:"
                f"exit_code={exit_code}"
            )
        else:
            row["state_class"] = "incomplete"
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        row["error"] = (
            f"scheduler_status_authentication_failed:"
            f"{type(exc).__name__}:{exc}"
        )
    return row


def _query_all_statuses(
    *,
    context: Mapping[str, Any],
    max_workers: int,
    retries: int,
) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    workers = min(
        int(max_workers),
        MAX_STATUS_WORKERS,
        len(context["submissions"]),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_one_status,
                submission=submission,
                plan=context["plan"],
                scheduler_url=context["scheduler_url"],
                retries=retries,
            ): int(submission["task_id"])
            for submission in context["submissions"]
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                rows[task_id] = future.result()
            except Exception as exc:
                submission = next(
                    item
                    for item in context["submissions"]
                    if int(item["task_id"]) == task_id
                )
                rows[task_id] = {
                    "task_id": task_id,
                    "seed": int(submission["seed"]),
                    "fixed_primary_turns": int(
                        submission["fixed_primary_turns"]
                    ),
                    "task_payload_sha256": str(
                        submission["task_payload_sha256"]
                    ),
                    "scheduler_status": None,
                    "exit_code": None,
                    "account_name": None,
                    "actual_node_name": None,
                    "scheduler_task_identity_sha256": None,
                    "state_class": "query_error",
                    "harvested": False,
                    "error": (
                        "unexpected_scheduler_query_failure:"
                        f"{type(exc).__name__}:{exc}"
                    ),
                }
    return [rows[task_id] for task_id in sorted(rows)]


def _fetch_result_bytes(
    *,
    scheduler_url: str,
    task_id: int,
    retries: int,
) -> bytes:
    relative = f"runs/task-{int(task_id)}/result.json"
    query = urllib.parse.urlencode(
        {
            "path": relative,
            "base": "remote_cwd",
            "max_bytes": RESULT_JSON_MAX_BYTES,
        }
    )
    url = (
        scheduler_url.rstrip("/")
        + f"/api/tasks/{int(task_id)}/remote-file?{query}"
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read(RESULT_JSON_MAX_BYTES + 1)
            if not payload:
                raise RuntimeError("remote result.json is unavailable")
            if len(payload) > RESULT_JSON_MAX_BYTES:
                raise RuntimeError("remote result.json exceeds safety limit")
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(0.5, 0.05 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise RuntimeError(
        f"result_fetch_failed_after_{retries}_attempts:"
        f"{type(last_error).__name__}:{last_error}"
    ) from last_error


def _sha256_record(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} inventory record is missing")
    try:
        size = int(value["size_bytes"])
        digest = str(value["sha256"]).lower()
        relative = _safe_relative(str(value["path"]), label)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} inventory record is invalid") from exc
    if (
        isinstance(value.get("size_bytes"), bool)
        or size < 0
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise RuntimeError(f"{label} inventory identity is invalid")
    return {"path": relative, "sha256": digest, "size_bytes": size}


def _validate_result_header(
    *,
    payload: bytes,
    submission: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("remote result.json is not valid UTF-8 JSON") from exc
    result = goal._validate_seal(value, schema=goal.SEARCH_RESULT_SCHEMA)
    task = submission["task"]
    if (
        result.get("campaign_id") != "mft-goal-20260726"
        or result.get("goal_contract_schema") != goal.GOAL_CONTRACT_SCHEMA
        or result.get("task_payload_sha256") != task["payload_sha256"]
        or int(result.get("seed", -1)) != int(task["seed"])
        or int(result.get("fixed_primary_turns", -1))
        != int(task["fixed_primary_turns"])
        or result.get("population") != goal.POPULATION
        or result.get("generations") != goal.GENERATIONS
        or result.get("evaluated_generations") != goal.GENERATIONS
        or result.get("completed_generations")
        != goal.EXPECTED_ALGORITHM_N_GEN_COUNTER
        or result.get("terminal_population_count") != goal.POPULATION
        or result.get("stage_spec_sha256")
        != task["stage_spec_sha256"]
        or result.get("hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
        or result.get("temperature_contract_sha256")
        != goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        or result.get("dataset_sha256") != task["dataset_sha256"]
        or result.get("evaluation_model_sha256")
        != task["evaluation_model_sha256"]
        or result.get("cooling_contract_sha256")
        != goal.FIXED_COOLING_IDENTITY_SHA256
        or result.get("operating_point_sha256")
        != goal.FIXED_OPERATING_IDENTITY_SHA256
        or result.get("search_only_proposal")
        != task["search_only_proposal"]
        or result.get("production_eligible") is not False
        or result.get("automatic_promotion_allowed") is not False
        or result.get("fea_submission_performed") is not False
    ):
        raise RuntimeError("goal result identity mismatch")

    inventory = result.get("artifact_inventory")
    if (
        not isinstance(inventory, dict)
        or result.get("artifact_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise RuntimeError("goal result artifact inventory seal mismatch")
    paths: set[str] = set()
    for role, raw_record in inventory.items():
        record = _sha256_record(raw_record, f"result artifact {role}")
        if record["path"] in paths:
            raise RuntimeError("goal result artifact inventory has duplicate paths")
        paths.add(record["path"])

    selected: dict[str, dict[str, Any]] = {}
    for role in TERMINAL_ARTIFACT_ROLES:
        record = _sha256_record(inventory.get(role), role)
        if record["path"] != TERMINAL_ARTIFACT_NAMES[role]:
            raise RuntimeError(f"{role} path does not match the goal contract")
        selected[role] = record

    embedded = goal._validate_seal(
        result.get("terminal_physical_candidates_manifest"),
        schema=goal.preflight.GOAL_TERMINAL_TABLE_SCHEMA,
    )
    source = embedded.get("source_identity") or {}
    csv_identity = embedded.get("csv") or {}
    if (
        embedded.get("row_count") != goal.POPULATION
        or embedded.get("terminal_population_index_min") != 0
        or embedded.get("terminal_population_index_max")
        != goal.POPULATION - 1
        or embedded.get("physical_deduplication_key")
        != "physical_geometry_sha256"
        or embedded.get("global_pareto_provenance_ready") is not True
        or csv_identity.get("path")
        != selected["terminal_physical_candidates"]["path"]
        or csv_identity.get("sha256")
        != selected["terminal_physical_candidates"]["sha256"]
        or int(csv_identity.get("size_bytes", -1))
        != selected["terminal_physical_candidates"]["size_bytes"]
        or int(source.get("seed", -1)) != int(submission["seed"])
        or str(source.get("task_id") or "") != str(submission["task_id"])
        or source.get("bundle_id") != task["payload_sha256"]
        or source.get("island_id")
        != f"n1-{int(submission['fixed_primary_turns'])}"
    ):
        raise RuntimeError("goal terminal manifest identity mismatch")
    return result, selected


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(
        f"{path.name}.part.{os.getpid()}.{uuid.uuid4().hex[:10]}"
    )
    try:
        with staged.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _safe_task_directory(results_root: Path, task_id: int) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    if results_root.is_symlink() or not results_root.is_dir():
        raise RuntimeError("harvest results root is unsafe")
    root = results_root.resolve(strict=True)
    task_dir = root / f"task-{int(task_id)}"
    if task_dir.exists() and (task_dir.is_symlink() or not task_dir.is_dir()):
        raise RuntimeError("local task harvest path is unsafe")
    task_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.resolve(strict=True).parent != root:
        raise RuntimeError("local task harvest path escaped results root")
    return task_dir


def _contained_remote_identity(
    session: Any,
    *,
    remote_task_root: str,
    remote_file: str,
) -> dict[str, Any]:
    script = "\n".join(
        [
            "set -e",
            "root_input=" + shlex.quote(remote_task_root),
            "file_input=" + shlex.quote(remote_file),
            'test -d "$root_input" || exit 43',
            'test -e "$file_input" || exit 44',
            'root=$(realpath -e -- "$root_input")',
            'file=$(realpath -e -- "$file_input")',
            'case "$file" in "$root"/*) ;; *) exit 45 ;; esac',
            'test -f "$file" || exit 46',
            'size=$(stat -c %s -- "$file")',
            'sha=$(sha256sum -- "$file")',
            "sha=${sha%% *}",
            'printf "%s\\t%s\\n" "$size" "$sha"',
        ]
    )
    response = session.run(
        "bash -lc " + shlex.quote(script),
        timeout=300,
    )
    if response.exit_code == 44:
        raise transport.RemoteArtifactMissing(remote_file)
    if response.exit_code == 45:
        raise RuntimeError("declared artifact resolves outside task directory")
    if response.exit_code != 0:
        raise RuntimeError(
            (
                response.stderr
                or response.stdout
                or "remote artifact identity failed"
            ).strip()
        )
    line = next(
        (
            item.strip()
            for item in reversed(response.stdout.splitlines())
            if item.strip()
        ),
        "",
    )
    size_text, separator, digest = line.partition("\t")
    digest = digest.strip().lower()
    if (
        not separator
        or not size_text.isdigit()
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise RuntimeError("invalid contained remote identity response")
    return {"bytes": int(size_text), "sha256": digest}


def _ensure_declared_artifact(
    *,
    connection: Any,
    remote_task_root: str,
    task_dir: Path,
    record: Mapping[str, Any],
    retries: int,
) -> dict[str, Any]:
    relative = _safe_relative(str(record["path"]), "terminal artifact")
    destination = task_dir.joinpath(*PurePosixPath(relative).parts)
    if destination.parent.resolve(strict=True) != task_dir.resolve(strict=True):
        raise RuntimeError("terminal artifact escaped local task directory")
    expected = {
        "bytes": int(record["size_bytes"]),
        "sha256": str(record["sha256"]),
    }
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError("existing local terminal artifact is unsafe")
        if (
            destination.stat().st_size == expected["bytes"]
            and _sha256_file(destination) == expected["sha256"]
        ):
            return {
                "path": relative,
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "transport": "local_verified_resume",
                "downloaded": False,
                "verified": True,
            }

    remote_file = posixpath.join(remote_task_root.rstrip("/"), relative)
    before = _contained_remote_identity(
        connection.get(),
        remote_task_root=remote_task_root,
        remote_file=remote_file,
    )
    if before != expected:
        raise RuntimeError("declared remote artifact identity mismatch")
    downloaded = transport._download_verified_sftp(
        connection,
        remote_file,
        destination,
        retries,
        expected_identity=before,
    )
    after = _contained_remote_identity(
        connection.get(),
        remote_task_root=remote_task_root,
        remote_file=remote_file,
    )
    if after != expected:
        raise RuntimeError("declared remote artifact changed after download")
    return {
        "path": relative,
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
        "transport": downloaded["transport"],
        "downloaded": True,
        "verified": True,
        "attempts": downloaded["attempts"],
    }


def _fetch_contained_full_result(
    *,
    connection: Any,
    remote_task_root: str,
    task_dir: Path,
    retries: int,
) -> tuple[bytes, dict[str, Any]]:
    """Read a result larger than the Scheduler text endpoint's hard cap."""

    remote_file = posixpath.join(remote_task_root.rstrip("/"), "result.json")
    identity = _contained_remote_identity(
        connection.get(),
        remote_task_root=remote_task_root,
        remote_file=remote_file,
    )
    if int(identity["bytes"]) > RESULT_JSON_SFTP_MAX_BYTES:
        raise RuntimeError("remote result.json exceeds SFTP safety limit")
    cache = task_dir / ".result.json.remote-cache"
    reused = False
    if cache.exists():
        if cache.is_symlink() or not cache.is_file():
            raise RuntimeError("local result cache is unsafe")
        reused = bool(
            cache.stat().st_size == int(identity["bytes"])
            and _sha256_file(cache) == identity["sha256"]
        )
    if not reused:
        transport._download_verified_sftp(
            connection,
            remote_file,
            cache,
            retries,
            expected_identity=identity,
            max_bytes=RESULT_JSON_SFTP_MAX_BYTES,
        )
    after = _contained_remote_identity(
        connection.get(),
        remote_task_root=remote_task_root,
        remote_file=remote_file,
    )
    if after != identity:
        raise RuntimeError("remote result.json changed during fallback read")
    payload = cache.read_bytes()
    if (
        len(payload) != int(identity["bytes"])
        or hashlib.sha256(payload).hexdigest() != identity["sha256"]
    ):
        raise RuntimeError("local result cache identity mismatch")
    return payload, {
        "transport": (
            "scheduler_remote_file_probe_then_contained_sftp"
        ),
        "remote_sha256": identity["sha256"],
        "remote_size_bytes": int(identity["bytes"]),
        "local_cache_reused": reused,
    }


def _harvest_one_completed_task(
    *,
    context: Mapping[str, Any],
    submission: Mapping[str, Any],
    row: Mapping[str, Any],
    connection: Any,
    retries: int,
) -> dict[str, Any]:
    result_row = copy.deepcopy(dict(row))
    if (
        result_row.get("state_class") != "completed_exit_zero"
        or result_row.get("scheduler_status") != "completed"
        or result_row.get("exit_code") != 0
    ):
        raise RuntimeError("only completed exit-0 tasks may be harvested")
    task_id = int(submission["task_id"])
    try:
        task_dir = _safe_task_directory(
            Path(context["results_root"]),
            task_id,
        )
        remote_task_root = posixpath.join(
            str(context["plan"]["remote_bundle"]).rstrip("/"),
            "runs",
            f"task-{task_id}",
        )
        result_bytes = _fetch_result_bytes(
            scheduler_url=context["scheduler_url"],
            task_id=task_id,
            retries=retries,
        )
        result_transport = {
            "transport": "scheduler_remote_file_base_remote_cwd",
            "remote_sha256": None,
            "remote_size_bytes": None,
            "local_cache_reused": False,
        }
        try:
            result, selected = _validate_result_header(
                payload=result_bytes,
                submission=submission,
            )
        except RuntimeError:
            if len(result_bytes) != RESULT_JSON_MAX_BYTES:
                raise
            result_bytes, result_transport = (
                _fetch_contained_full_result(
                    connection=connection,
                    remote_task_root=remote_task_root,
                    task_dir=task_dir,
                    retries=retries,
                )
            )
            result, selected = _validate_result_header(
                payload=result_bytes,
                submission=submission,
            )
        pending_result = task_dir / ".result.json.pending"
        _atomic_bytes(pending_result, result_bytes)
        records: dict[str, dict[str, Any]] = {}
        for role in TERMINAL_ARTIFACT_ROLES:
            records[role] = _ensure_declared_artifact(
                connection=connection,
                remote_task_root=remote_task_root,
                task_dir=task_dir,
                record=selected[role],
                retries=retries,
            )
        goal._validated_seed_table(
            pending_result,
            expected_task=submission["task"],
        )
        final_result = task_dir / "result.json"
        if final_result.exists() and (
            final_result.is_symlink() or not final_result.is_file()
        ):
            raise RuntimeError("existing local result.json is unsafe")
        os.replace(pending_result, final_result)
        result_sha = _sha256_file(final_result)
        receipt = _sealed(
            {
                "schema_version": TASK_HARVEST_RECEIPT_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "bundle_id": context["plan"]["bundle_id"],
                "task_id": task_id,
                "seed": int(submission["seed"]),
                "fixed_primary_turns": int(
                    submission["fixed_primary_turns"]
                ),
                "task_payload_sha256": submission[
                    "task_payload_sha256"
                ],
                "scheduler_task_identity_sha256": result_row[
                    "scheduler_task_identity_sha256"
                ],
                "scheduler_status": "completed",
                "exit_code": 0,
                "result": {
                    "path": "result.json",
                    "sha256": result_sha,
                    "size_bytes": final_result.stat().st_size,
                    "payload_sha256": result["payload_sha256"],
                    **result_transport,
                },
                "terminal_artifacts": records,
                "full_goal_seed_validation_passed": True,
                "aggregate_result_relative_path": (
                    f"task-{task_id}/result.json"
                ),
                "harvested_at": _now(),
            }
        )
        receipt_path = task_dir / "harvest_receipt.json"
        _atomic_json(receipt_path, receipt)
        result_row.update(
            {
                "harvested": True,
                "error": None,
                "result_relative_path": f"task-{task_id}/result.json",
                "result_sha256": result_sha,
                "result_payload_sha256": result["payload_sha256"],
                "receipt_relative_path": (
                    f"task-{task_id}/harvest_receipt.json"
                ),
                "receipt_payload_sha256": receipt["payload_sha256"],
                "terminal_artifacts": records,
            }
        )
    except Exception as exc:
        result_row["harvested"] = False
        result_row["error"] = (
            f"harvest_validation_failed:{type(exc).__name__}:{exc}"
        )
    return result_row


def _summary_values(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(rows)
    states = Counter(str(row.get("state_class") or "") for row in materialized)
    statuses = Counter(
        str(row.get("scheduler_status") or "query_error")
        for row in materialized
    )
    harvested = sum(row.get("harvested") is True for row in materialized)
    errors = sum(bool(row.get("error")) for row in materialized)
    return {
        "status_counts": dict(sorted(statuses.items())),
        "state_class_counts": dict(sorted(states.items())),
        "harvested_task_count": harvested,
        "error_task_count": errors,
        "completed_exit_zero_count": states.get("completed_exit_zero", 0),
        "terminal_failure_count": states.get("terminal_failure", 0),
        "incomplete_count": states.get("incomplete", 0),
    }


def _write_harvest_summaries(
    *,
    context: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    polls_performed: int,
    max_polls: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(
        (copy.deepcopy(dict(row)) for row in rows),
        key=lambda row: int(row["task_id"]),
    )
    values = _summary_values(ordered)
    ready = bool(
        len(ordered) == EXPECTED_CAMPAIGN_TASKS
        and values["harvested_task_count"] == EXPECTED_CAMPAIGN_TASKS
        and values["completed_exit_zero_count"]
        == EXPECTED_CAMPAIGN_TASKS
        and values["error_task_count"] == 0
    )
    ledger = _sealed(
        {
            "schema_version": HARVEST_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "bundle_id": context["plan"]["bundle_id"],
            "goal_bundle_payload_sha256": context["bundle"][
                "payload_sha256"
            ],
            "submission_ledger_path": context[
                "submission_ledger_path"
            ],
            "submission_ledger_sha256": context[
                "submission_ledger_sha256"
            ],
            "submission_identity_sha256": context[
                "submission_identity_sha256"
            ],
            "scheduler_url": context["scheduler_url"],
            "scheduler_access_mode": (
                "read_only_get_plus_contained_sftp_reads"
            ),
            "expected_task_count": EXPECTED_CAMPAIGN_TASKS,
            "observed_task_count": len(ordered),
            "polls_performed": int(polls_performed),
            "max_polls": int(max_polls),
            "aggregate_ready": ready,
            **values,
            "tasks": ordered,
            "updated_at": _now(),
        }
    )
    root = Path(context["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "harvest_ledger.json"
    _atomic_json(ledger_path, ledger)
    result_paths = [
        str(row["result_relative_path"])
        for row in ordered
        if row.get("harvested") is True
    ]
    readiness = _sealed(
        {
            "schema_version": HARVEST_READINESS_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "bundle_id": context["plan"]["bundle_id"],
            "harvest_ledger": ledger_path.name,
            "harvest_ledger_payload_sha256": ledger[
                "payload_sha256"
            ],
            "expected_task_count": EXPECTED_CAMPAIGN_TASKS,
            **values,
            "aggregate_ready": ready,
            "aggregate_results_root": str(root),
            "aggregate_bundle_manifest": context["plan"][
                "goal_bundle_manifest"
            ],
            "authenticated_result_relative_paths": result_paths,
            "updated_at": _now(),
        }
    )
    _atomic_json(root / "readiness.json", readiness)
    return ledger, readiness


def _harvest_completed_rows(
    *,
    context: Mapping[str, Any],
    submissions_by_id: Mapping[int, Mapping[str, Any]],
    rows: list[dict[str, Any]],
    max_workers: int,
    retries: int,
) -> list[dict[str, Any]]:
    by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: dict[int, dict[str, Any]] = {}
    for row in rows:
        task_id = int(row["task_id"])
        if row.get("state_class") == "completed_exit_zero":
            account_name = str(row.get("account_name") or "")
            if not account_name:
                failed = copy.deepcopy(row)
                failed["error"] = "completed_task_has_no_account"
                passthrough[task_id] = failed
            else:
                by_account[account_name].append(row)
        else:
            passthrough[task_id] = row

    def harvest_account(
        account_name: str,
        account_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            account, ssh_session = transport._account(
                Path(context["accounts_path"]),
                Path(context["scheduler_source"]),
                account_name,
            )
        except Exception as exc:
            failed = []
            for row in account_rows:
                value = copy.deepcopy(row)
                value["error"] = (
                    f"account_lookup_failed:{type(exc).__name__}:{exc}"
                )
                failed.append(value)
            return failed
        connection = transport._PersistentAccountConnection(
            account,
            ssh_session,
        )
        harvested = []
        try:
            for row in account_rows:
                task_id = int(row["task_id"])
                harvested.append(
                    _harvest_one_completed_task(
                        context=context,
                        submission=submissions_by_id[task_id],
                        row=row,
                        connection=connection,
                        retries=retries,
                    )
                )
        finally:
            connection.close()
        return harvested

    if by_account:
        workers = min(int(max_workers), MAX_ACCOUNT_WORKERS, len(by_account))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    harvest_account,
                    account_name,
                    account_rows,
                ): account_name
                for account_name, account_rows in sorted(by_account.items())
            }
            for future in as_completed(futures):
                account_name = futures[future]
                try:
                    account_rows = future.result()
                except Exception as exc:
                    account_rows = []
                    for row in by_account[account_name]:
                        value = copy.deepcopy(row)
                        value["error"] = (
                            "unexpected_account_harvest_failure:"
                            f"{type(exc).__name__}:{exc}"
                        )
                        account_rows.append(value)
                for row in account_rows:
                    passthrough[int(row["task_id"])] = row
    return [passthrough[task_id] for task_id in sorted(passthrough)]


def monitor_and_harvest(
    *,
    plan_path: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    results_root: Path | None = None,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    max_workers: int = MAX_STATUS_WORKERS,
    retries: int = DEFAULT_HARVEST_RETRIES,
    max_polls: int = 1,
    poll_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Boundedly poll 512 tasks and harvest only authenticated successes."""

    if not 1 <= int(max_workers) <= MAX_STATUS_WORKERS:
        raise ValueError(
            f"max_workers must be 1..{MAX_STATUS_WORKERS}"
        )
    if not 1 <= int(retries) <= 10:
        raise ValueError("retries must be 1..10")
    if not 1 <= int(max_polls) <= 120:
        raise ValueError("max_polls must be 1..120")
    interval = float(poll_interval_seconds)
    if not 0.0 <= interval <= 60.0:
        raise ValueError("poll_interval_seconds must be 0..60")
    context = authenticate_harvest_context(
        plan_path=plan_path,
        scheduler_url=scheduler_url,
        results_root=results_root,
    )
    context["accounts_path"] = str(accounts_path.resolve(strict=True))
    context["scheduler_source"] = str(
        scheduler_source.resolve(strict=True)
    )
    submissions_by_id = {
        int(item["task_id"]): item for item in context["submissions"]
    }
    prior_success: dict[int, dict[str, Any]] = {}
    ledger: dict[str, Any] = {}
    readiness: dict[str, Any] = {}
    for poll in range(1, int(max_polls) + 1):
        rows = _query_all_statuses(
            context=context,
            max_workers=int(max_workers),
            retries=int(retries),
        )
        for index, row in enumerate(rows):
            task_id = int(row["task_id"])
            previous = prior_success.get(task_id)
            if (
                previous is not None
                and row.get("state_class") == "completed_exit_zero"
                and row.get("scheduler_task_identity_sha256")
                == previous.get("scheduler_task_identity_sha256")
            ):
                rows[index] = copy.deepcopy(previous)
        ledger, readiness = _write_harvest_summaries(
            context=context,
            rows=rows,
            polls_performed=poll,
            max_polls=int(max_polls),
        )
        pending_harvest = [
            row
            for row in rows
            if row.get("state_class") == "completed_exit_zero"
            and row.get("harvested") is not True
        ]
        if pending_harvest:
            harvested_by_id = {
                int(row["task_id"]): row
                for row in _harvest_completed_rows(
                    context=context,
                    submissions_by_id=submissions_by_id,
                    rows=pending_harvest,
                    max_workers=int(max_workers),
                    retries=int(retries),
                )
            }
            rows = [
                harvested_by_id.get(int(row["task_id"]), row)
                for row in rows
            ]
            prior_success.update(
                {
                    int(row["task_id"]): copy.deepcopy(row)
                    for row in rows
                    if row.get("harvested") is True
                }
            )
            ledger, readiness = _write_harvest_summaries(
                context=context,
                rows=rows,
                polls_performed=poll,
                max_polls=int(max_polls),
            )
        if readiness["aggregate_ready"]:
            break
        if poll < int(max_polls):
            time.sleep(interval)
    return {
        "harvest_ledger": str(
            Path(context["results_root"]) / "harvest_ledger.json"
        ),
        "readiness": str(
            Path(context["results_root"]) / "readiness.json"
        ),
        "aggregate_ready": bool(readiness["aggregate_ready"]),
        "expected_task_count": EXPECTED_CAMPAIGN_TASKS,
        "harvested_task_count": int(
            readiness["harvested_task_count"]
        ),
        "completed_exit_zero_count": int(
            readiness["completed_exit_zero_count"]
        ),
        "terminal_failure_count": int(
            readiness["terminal_failure_count"]
        ),
        "incomplete_count": int(readiness["incomplete_count"]),
        "error_task_count": int(readiness["error_task_count"]),
        "polls_performed": int(ledger["polls_performed"]),
    }


def _comma_ints(value: str) -> set[int]:
    try:
        result = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comma-separated integers required") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage, submit, and fail-closed harvest the isolated "
            "2026-07-26 MFT goal"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--goal-bundle-root", type=Path, required=True)
    plan.add_argument("--generation", type=Path, required=True)
    plan.add_argument("--candidate", type=Path, required=True)
    plan.add_argument("--quality-status", type=Path, required=True)
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    plan.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)

    stage = commands.add_parser("stage")
    stage.add_argument("--plan", type=Path, required=True)
    stage.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    stage.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    stage.add_argument("--staging-account", default=DEFAULT_STAGING_ACCOUNT)
    stage.add_argument("--resume-incoming")
    stage.add_argument("--apply", action="store_true")

    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--phase", choices=("canary", "wave", "all", "seeds"), required=True
    )
    submit_parser.add_argument("--wave", type=int)
    submit_parser.add_argument("--seeds", type=_comma_ints)
    submit_parser.add_argument(
        "--scheduler-url", default=DEFAULT_SCHEDULER_URL
    )
    submit_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    submit_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    submit_parser.add_argument(
        "--staging-account", default=DEFAULT_STAGING_ACCOUNT
    )
    submit_parser.add_argument("--priority", type=int, default=10)
    submit_parser.add_argument("--apply", action="store_true")

    harvest = commands.add_parser(
        "harvest",
        help=(
            "read-only bounded status polling and authenticated 512-task "
            "result harvest"
        ),
    )
    harvest.add_argument("--plan", type=Path, required=True)
    harvest.add_argument(
        "--scheduler-url",
        default=DEFAULT_SCHEDULER_URL,
        help="must match the URL sealed by the local submission ledger",
    )
    harvest.add_argument("--results-root", type=Path)
    harvest.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    harvest.add_argument(
        "--scheduler-source",
        type=Path,
        default=DEFAULT_SCHEDULER_SOURCE,
    )
    harvest.add_argument(
        "--max-workers",
        type=int,
        default=MAX_STATUS_WORKERS,
        help="bounded status/SFTP concurrency (1..4)",
    )
    harvest.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_HARVEST_RETRIES,
    )
    harvest.add_argument(
        "--max-polls",
        type=int,
        default=1,
        help="bounded poll count; one gives a read-only snapshot",
    )
    harvest.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
    )
    harvest.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit status 2 until all 512 results authenticate",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan, deployment = build_plan(
            goal_bundle_root=args.goal_bundle_root,
            generation=args.generation,
            candidate=args.candidate,
            quality_status=args.quality_status,
            dataset=args.dataset,
            profile=args.profile,
            local_root=args.local_root,
            remote_root=args.remote_root,
        )
        output = {"plan": plan, "deployment": deployment}
    elif args.command == "stage":
        output = transport.stage_bundle(
            args.plan,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            resume_incoming=args.resume_incoming,
            apply=args.apply,
        )
    elif args.command == "submit":
        output = submit(
            plan_path=args.plan,
            phase=args.phase,
            wave=args.wave,
            seeds=args.seeds,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            priority=args.priority,
            apply=args.apply,
        )
    else:
        output = monitor_and_harvest(
            plan_path=args.plan,
            scheduler_url=args.scheduler_url,
            results_root=args.results_root,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            max_workers=args.max_workers,
            retries=args.retries,
            max_polls=args.max_polls,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    if (
        args.command == "harvest"
        and args.require_ready
        and not output["aggregate_ready"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
