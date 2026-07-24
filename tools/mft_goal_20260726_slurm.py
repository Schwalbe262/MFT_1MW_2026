"""Stage and submit the isolated 2026-07-26 MFT NSGA-II campaign.

The 1MW_MFT repository owns the scientific bundle and worker command.  The
separate Scheduler service owns only placement and task lifecycle.  This tool
therefore consumes a fully prepared goal bundle, publishes one immutable GPFS
tree, and POSTs one already sealed goal task per Slurm task.  It never edits
or imports Scheduler application code into the MFT source tree.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any, Iterable, Mapping


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
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
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
        description="Stage and submit the isolated 2026-07-26 MFT goal"
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
    else:
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
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
