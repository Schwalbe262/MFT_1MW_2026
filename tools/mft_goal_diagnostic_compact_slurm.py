"""Authenticate, stage, and exactly-once submit the N1=6 compact scout.

This adapter is intentionally separate from the reserved fresh512 offload
tool.  It accepts exactly 32 diagnostic screening tasks with seeds
2607264000..2607264031 and N1=6.  Planning, staging, authentication, and
submission never confer production or final-design status.  Scheduler POSTs
are possible only when ``submit --apply`` is supplied; every other command is
read-only with respect to Scheduler task state.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_20260726_launch as goal_launch  # noqa: E402
from tools import mft_goal_20260726_slurm as main_slurm  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402
from tools import tier1_corrected_generation_adapter as adapter  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402
from tools.tier1_final1000_multiseed_contract import (  # noqa: E402
    SCHEDULER_TASK_STATUSES,
    scheduler_task_identity_matches,
    scheduler_task_observation,
)


PLAN_SCHEMA = "mft-goal-diagnostic-compact-offload-plan-v1"
DEPLOYMENT_SCHEMA = "mft-goal-diagnostic-compact-deployment-v1"
AUTHENTICATION_SCHEMA = "mft-goal-diagnostic-compact-offload-auth-v1"
RECEIPT_SCHEMA = "mft-goal-diagnostic-compact-submission-receipt-v1"
DRY_RUN_SCHEMA = "mft-goal-diagnostic-compact-submission-dry-run-v1"
CLAIM_ROOT_SCHEMA = "mft-goal-diagnostic-compact-claim-root-v1"
PENDING_CLAIM_SCHEMA = "mft-goal-diagnostic-compact-pending-claim-v1"
FINALIZED_CLAIM_SCHEMA = "mft-goal-diagnostic-compact-finalized-claim-v1"
EXACT_SEED_START = scout.PROFILE_SEED_STARTS[20]
EXACT_TASK_COUNT = scout.PROFILE_SEED_COUNT
EXACT_SEEDS = tuple(range(EXACT_SEED_START, EXACT_SEED_START + EXACT_TASK_COUNT))
FIXED_PRIMARY_TURNS = 6
TASK_NAME_PREFIX = "mft-goal-diag-compact-"
DEDUPE_PREFIX = "mft-goal-20260726-diag-compact:"
DEFAULT_LOCAL_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "diagnostic_compact_offload"
)
DEFAULT_REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_goal_20260726/diagnostic_compact"
DEFAULT_SCHEDULER_URL = main_slurm.DEFAULT_SCHEDULER_URL
DEFAULT_ACCOUNTS = main_slurm.DEFAULT_ACCOUNTS
DEFAULT_SCHEDULER_SOURCE = main_slurm.DEFAULT_SCHEDULER_SOURCE
DEFAULT_STAGING_ACCOUNT = main_slurm.DEFAULT_STAGING_ACCOUNT
CPUS_PER_TASK = scout.INFERENCE_THREADS
MEMORY_MB_PER_TASK = 65_536
TIMEOUT_SECONDS = 14_400
MAX_WORKERS_PER_NODE = 8
SCHEDULER_PRIORITY = 10

if (
    EXACT_TASK_COUNT != 100
    or scout.DEFAULT_SEED_COUNT != EXACT_TASK_COUNT
):  # pragma: no cover
    raise RuntimeError("diagnostic compact scout seed authority drifted")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete JSON receipt without an overwrite race."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError as exc:
            raise RuntimeError(
                "immutable diagnostic receipt already exists"
            ) from exc
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise RuntimeError("diagnostic offload value is already sealed")
    result["sha256"] = canonical_sha256(result)
    return result


def _validate_seal(
    value: Mapping[str, Any],
    *,
    schema: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("diagnostic offload seal must be an object")
    result = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in result.items() if key != "sha256"}
    if (
        result.get("schema_version") != schema
        or result.get("sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} seal mismatch")
    return result


def _safe_relative(value: str, label: str) -> str:
    path = PurePosixPath(str(value))
    if (
        "\\" in str(value)
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _authorized_activation_seeds(
    activation: Mapping[str, Any],
) -> tuple[int, ...]:
    profile = activation.get("manufacturing_search_profile")
    if profile is None:
        return EXACT_SEEDS
    normalized = scout._validate_search_profile(profile)
    if (
        activation.get("manufacturing_search_profile_payload_sha256")
        != normalized["payload_sha256"]
    ):
        raise RuntimeError("diagnostic profile seed authority is unbound")
    start = int(normalized["authorized_seed_start"])
    count = int(normalized["authorized_seed_count"])
    seeds = tuple(range(start, start + count))
    if (
        count != EXACT_TASK_COUNT
        or seeds[-1] != normalized["authorized_seed_end_inclusive"]
    ):
        raise RuntimeError("diagnostic profile seed authority is not exact100")
    return seeds


def _authorized_plan_seeds(plan: Mapping[str, Any]) -> tuple[int, ...]:
    raw = plan.get("authorized_seeds")
    if raw is None:
        return EXACT_SEEDS
    if (
        not isinstance(raw, list)
        or len(raw) != EXACT_TASK_COUNT
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in raw
        )
    ):
        raise RuntimeError("diagnostic plan seed authority is invalid")
    seeds = tuple(raw)
    if (
        len(set(seeds)) != EXACT_TASK_COUNT
        or seeds != tuple(range(seeds[0], seeds[0] + EXACT_TASK_COUNT))
        or plan.get("seed_start") != seeds[0]
        or plan.get("seed_end_inclusive") != seeds[-1]
    ):
        raise RuntimeError("diagnostic plan seed authority is not contiguous exact100")
    return seeds


def _exact_task_inventory(
    task_values: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = [scout._validate_task(dict(value)) for value in task_values]
    if not tasks:
        raise RuntimeError("diagnostic offload task inventory is empty")
    first = tasks[0]
    activation = scout._validate_activation(first["activation"])
    authorized_seeds = _authorized_activation_seeds(activation)
    seeds = [int(task["seed"]) for task in tasks]
    if (
        len(tasks) != EXACT_TASK_COUNT
        or sorted(seeds) != list(authorized_seeds)
        or len(set(seeds)) != EXACT_TASK_COUNT
        or any(
            task.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
            or task.get("screening_only") is not True
            or task.get("production_eligible") is not False
            or task.get("final_design_claim_allowed") is not False
            or task.get("fresh512_activation_evidence") is not False
            for task in tasks
        )
    ):
        raise RuntimeError("diagnostic offload requires exact100 N1=6 seed inventory")
    source_identity = first["source_identity"]
    source = first["source"]
    code_manifest_sha = first["code_manifest_payload_sha256"]
    code_inventory_sha = first["code_inventory_sha256"]
    activation_sha = activation["payload_sha256"]
    contract_sha = activation["compact_search_contract_sha256"]
    bank_sha = activation["compact_coordinate_bank_sha256"]
    search_profile = activation.get("manufacturing_search_profile")
    secondary_gap_mode = (
        scout.SECONDARY_GAP_MODE_FIXED
        if search_profile is None
        else scout._secondary_gap_mode_from_profile(search_profile)
    )
    scheduler_priority = (
        SCHEDULER_PRIORITY
        if secondary_gap_mode == scout.SECONDARY_GAP_MODE_FIXED
        else scout.VARIABLE_SECONDARY_PROFILE_SCHEDULER_PRIORITY
    )
    model_sha = source_identity["evaluation_model_sha256"]
    dataset_sha = source_identity["dataset_sha256"]
    quality_sha = source_identity["quality_status_sha256"]
    for task in tasks:
        authorization = (
            preflight.validate_goal_compact_run_authorization_seal(
                task["compact_run_authorization"]
            )
        )
        if (
            task["activation"]["payload_sha256"] != activation_sha
            or task["source_identity"] != source_identity
            or task["source"] != source
            or task["code_manifest_payload_sha256"] != code_manifest_sha
            or task["code_inventory_sha256"] != code_inventory_sha
            or authorization["authorization_mode"]
            != preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
            or authorization["source_activation_payload_sha256"]
            != activation_sha
            or authorization["seed"] != task["seed"]
            or authorization["fixed_primary_turns"] != FIXED_PRIMARY_TURNS
            or authorization["compact_search_contract_sha256"] != contract_sha
            or authorization["compact_coordinate_bank_sha256"] != bank_sha
            or authorization["dataset_sha256"] != dataset_sha
            or authorization["evaluation_model_sha256"] != model_sha
            or authorization["quality_status_sha256"] != quality_sha
            or authorization["source_quality_passed"]
            is not activation["source_quality_passed"]
            or authorization["screening_only"] is not True
            or authorization["production_eligible"] is not False
            or authorization["final_design_claim_allowed"] is not False
        ):
            raise RuntimeError(
                "diagnostic task authority/source/bank identity is mixed"
            )
    ordered = sorted(tasks, key=lambda task: int(task["seed"]))
    common = {
        "activation_payload_sha256": activation_sha,
        "source_identity": copy.deepcopy(source_identity),
        "source": copy.deepcopy(source),
        "code_manifest_payload_sha256": code_manifest_sha,
        "code_inventory_sha256": code_inventory_sha,
        "compact_search_contract_sha256": contract_sha,
        "compact_coordinate_bank_sha256": bank_sha,
        "source_quality_passed": activation["source_quality_passed"],
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "automatic_promotion_allowed": False,
        "fresh512_activation_evidence": False,
        "authorized_seeds": list(authorized_seeds),
        "seed_start": authorized_seeds[0],
        "seed_end_inclusive": authorized_seeds[-1],
        "manufacturing_search_profile_payload_sha256": activation.get(
            "manufacturing_search_profile_payload_sha256"
        ),
        "geometry_constraint_profile_sha256": activation.get(
            "geometry_constraint_profile_sha256"
        ),
        "secondary_gap_mode": secondary_gap_mode,
        "scheduler_priority": scheduler_priority,
    }
    return ordered, common


def _bundle_tasks(
    bundle_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[Path],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = bundle_root.resolve(strict=True)
    bundle = goal_launch._validate_seal(
        _read_json(root / "bundle_manifest.json"),
        schema=scout.BUNDLE_SCHEMA,
    )
    scheduler = goal_launch._validate_seal(
        _read_json(root / "scheduler_manifest.json"),
        schema=scout.SCHEDULER_SCHEMA,
    )
    activation = scout._validate_activation(
        _read_json(root / "activation.json")
    )
    code_manifest = goal_launch._validate_seal(
        _read_json(root / "code_manifest.json"),
        schema=goal_launch.CODE_MANIFEST_SCHEMA,
    )
    relatives = bundle.get("task_relative_paths")
    payloads = bundle.get("task_payload_sha256")
    if (
        bundle.get("campaign_id") != scout.CAMPAIGN_ID
        or bundle.get("task_count") != EXACT_TASK_COUNT
        or scheduler.get("task_count") != EXACT_TASK_COUNT
        or scheduler.get("maximum_parallel_tasks") != EXACT_TASK_COUNT
        or scheduler.get("bundle_payload_sha256")
        != bundle["payload_sha256"]
        or bundle.get("activation_payload_sha256")
        != activation["payload_sha256"]
        or bundle.get("screening_only") is not True
        or bundle.get("production_eligible") is not False
        or bundle.get("final_design_claim_allowed") is not False
        or bundle.get("fresh512_activation_evidence") is not False
        or scheduler.get("screening_only") is not True
        or scheduler.get("production_eligible") is not False
        or scheduler.get("final_design_claim_allowed") is not False
        or scheduler.get("fresh512_activation_evidence") is not False
        or scheduler.get("scheduler_write_performed") is not False
        or scheduler.get("scheduler_submission_performed") is not False
        or code_manifest.get("scheduler_project_code_included") is not False
        or code_manifest.get("remote_git_checkout_required") is not False
        or not isinstance(relatives, list)
        or not isinstance(payloads, list)
        or len(relatives) != EXACT_TASK_COUNT
        or len(payloads) != EXACT_TASK_COUNT
    ):
        raise RuntimeError("diagnostic bundle/scheduler authority mismatch")
    task_paths: list[Path] = []
    task_values: list[dict[str, Any]] = []
    for relative, payload_sha in zip(relatives, payloads):
        safe = _safe_relative(str(relative), "diagnostic task path")
        path = (root / safe).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("diagnostic task escaped bundle") from exc
        task = scout._validate_task(_read_json(path))
        if task["payload_sha256"] != payload_sha:
            raise RuntimeError("diagnostic task payload identity mismatch")
        task_paths.append(path)
        task_values.append(task)
    tasks, common = _exact_task_inventory(task_values)
    paths_by_sha = {
        task["payload_sha256"]: path
        for path, task in zip(task_paths, task_values)
    }
    ordered_paths = [paths_by_sha[task["payload_sha256"]] for task in tasks]
    if (
        common["activation_payload_sha256"] != activation["payload_sha256"]
        or common["code_manifest_payload_sha256"]
        != code_manifest["payload_sha256"]
        or common["code_inventory_sha256"]
        != code_manifest["code_inventory_sha256"]
        or bundle.get("code_manifest_payload_sha256")
        != code_manifest["payload_sha256"]
    ):
        raise RuntimeError("diagnostic bundle code/activation identity mismatch")
    if common["manufacturing_search_profile_payload_sha256"] is not None and (
        bundle.get("manufacturing_search_profile_payload_sha256")
        != common["manufacturing_search_profile_payload_sha256"]
        or scheduler.get("manufacturing_search_profile_payload_sha256")
        != common["manufacturing_search_profile_payload_sha256"]
        or bundle.get("geometry_constraint_profile_sha256")
        != common["geometry_constraint_profile_sha256"]
        or scheduler.get("geometry_constraint_profile_sha256")
        != common["geometry_constraint_profile_sha256"]
        or bundle.get("authorized_seed_start") != common["seed_start"]
        or bundle.get("authorized_seed_count") != EXACT_TASK_COUNT
        or bundle.get("authorized_seed_end_inclusive")
        != common["seed_end_inclusive"]
    ):
        raise RuntimeError("diagnostic bundle search profile identity mismatch")
    return (
        bundle,
        scheduler,
        activation,
        code_manifest,
        ordered_paths,
        tasks,
        common,
    )


def _authenticate_detached_source(
    *,
    code_root: Path,
    code_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = code_root.resolve(strict=True)
    revision = str(code_manifest["code_revision"])
    identity = adapter.authenticate_code_root(root, revision)
    detached = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        env=goal_launch._git_environment(root),
    )
    if detached.returncode != 1 or detached.stdout.strip():
        raise RuntimeError(
            "diagnostic source must be a clean detached Git checkout"
        )
    rebuilt = goal_launch._build_goal_code_manifest(
        goal_launch._collect_goal_code_sources(root),
        code_revision=revision,
    )
    if (
        rebuilt["payload_sha256"] != code_manifest["payload_sha256"]
        or rebuilt["code_inventory_sha256"]
        != code_manifest["code_inventory_sha256"]
    ):
        raise RuntimeError(
            "detached diagnostic source differs from staged code inventory"
        )
    return {
        "revision": identity["revision"],
        "clean": identity["clean"],
        "detached_head": True,
        "code_manifest_payload_sha256": code_manifest["payload_sha256"],
        "code_inventory_sha256": code_manifest["code_inventory_sha256"],
    }


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
    (
        bundle,
        _scheduler,
        activation,
        code_manifest,
        task_paths,
        tasks,
        common,
    ) = _bundle_tasks(goal_bundle_root)
    source_checkout = _authenticate_detached_source(
        code_root=Path(common["source"]["code_root"]),
        code_manifest=code_manifest,
    )
    bundle_root = goal_bundle_root.resolve(strict=True)
    generation = generation.resolve(strict=True)
    candidate = candidate.resolve(strict=True)
    quality_status = quality_status.resolve(strict=True)
    dataset = dataset.resolve(strict=True)
    profile = profile.resolve(strict=True)
    report_path = generation / "train_report.json"
    report = _read_json(report_path)
    artifacts = report.get("artifacts")
    identity = common["source_identity"]
    if (
        generation.parent.name != "generations"
        or report.get("training_run_id") != generation.name
        or not isinstance(artifacts, dict)
        or not artifacts
        or adapter.sha256_file(report_path)
        != identity["train_report_sha256"]
        or adapter.sha256_file(candidate) != identity["candidate_sha256"]
        or adapter.sha256_file(quality_status)
        != identity["quality_status_sha256"]
        or adapter.sha256_file(dataset) != identity["dataset_sha256"]
        or adapter.training_profile_sha256(_read_json(profile))
        != identity["profile_sha256"]
        or canonical_sha256(artifacts)
        != identity["evaluation_model_sha256"]
        or code_manifest["code_revision"] != identity["code_revision"]
    ):
        raise RuntimeError("diagnostic source/model identity mismatch")
    deployment_identity = canonical_sha256(
        {
            "bundle_payload_sha256": bundle["payload_sha256"],
            "activation_payload_sha256": activation["payload_sha256"],
            "code_manifest_payload_sha256": code_manifest["payload_sha256"],
            "source_identity": identity,
            "task_payload_sha256": [
                task["payload_sha256"] for task in tasks
            ],
        }
    )
    bundle_id = f"mft-goal-diag-compact-{deployment_identity[:24]}"
    remote_base = str(remote_root).rstrip("/")
    remote_bundle = f"{remote_base}/{bundle_id}"
    plan_dir = local_root.resolve() / bundle_id
    if plan_dir.exists():
        raise RuntimeError(f"diagnostic offload plan already exists: {plan_dir}")
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
        raise RuntimeError("diagnostic code inventory mismatch")
    for relative, record in sorted(code_inventory.items()):
        safe = _safe_relative(relative, "diagnostic code path")
        local = (bundle_root / safe).resolve(strict=True)
        main_slurm._add_file(
            files,
            sources,
            safe,
            local,
            "diagnostic_code",
            expected_sha256=str(record["sha256"]),
        )
        if files[safe]["size"] != int(record["size"]):
            raise RuntimeError("diagnostic code inventory size mismatch")

    campaign_files = {
        "activation.json": bundle_root / "activation.json",
        "bundle_manifest.json": bundle_root / "bundle_manifest.json",
        "scheduler_manifest.json": bundle_root / "scheduler_manifest.json",
        "code_manifest.json": bundle_root / "code_manifest.json",
    }
    for name, source in campaign_files.items():
        main_slurm._add_file(
            files,
            sources,
            f"artifacts/campaign/{name}",
            source,
            "diagnostic_campaign_manifest",
        )
    for path in task_paths:
        main_slurm._add_file(
            files,
            sources,
            f"artifacts/campaign/tasks/{path.name}",
            path,
            "diagnostic_task",
        )

    generation_remote = (
        f"artifacts/g0/registry/generations/{generation.name}"
    )
    main_slurm._add_file(
        files,
        sources,
        f"{generation_remote}/train_report.json",
        report_path,
        "generation_report",
        expected_sha256=identity["train_report_sha256"],
    )
    actual_artifacts = {
        path.relative_to(generation).as_posix()
        for path in generation.rglob("*")
        if path.is_file() and path.name != "train_report.json"
    }
    if actual_artifacts != set(artifacts):
        raise RuntimeError("diagnostic model artifact inventory mismatch")
    for relative, expected_sha in sorted(artifacts.items()):
        safe = _safe_relative(relative, "diagnostic model artifact")
        main_slurm._add_file(
            files,
            sources,
            f"{generation_remote}/{safe}",
            generation / safe,
            "model_artifact",
            expected_sha256=str(expected_sha),
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
    for relative, source, kind, expected_sha in (
        (
            "artifacts/g0/candidate.json",
            candidate,
            "generation_candidate",
            identity["candidate_sha256"],
        ),
        (
            "artifacts/g0/quality_status.json",
            quality_status,
            "quality_status",
            identity["quality_status_sha256"],
        ),
        (
            "artifacts/g0/dataset/train.parquet",
            dataset,
            "training_dataset",
            identity["dataset_sha256"],
        ),
    ):
        main_slurm._add_file(
            files,
            sources,
            relative,
            source,
            kind,
            expected_sha256=expected_sha,
        )
    main_slurm._add_file(
        files,
        sources,
        "artifacts/g0/profile.json",
        profile,
        "training_profile",
    )

    relocation_sources: dict[int, str] = {}
    for task in tasks:
        relocation = goal_launch._seal(
            {
                "schema_version": goal_launch.RELOCATION_SCHEMA,
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
            / f"seed-{task['seed']}-n1-{FIXED_PRIMARY_TURNS}.json"
        )
        _atomic_json(relocation_path, relocation)
        relative = (
            "artifacts/campaign/relocations/" + relocation_path.name
        )
        main_slurm._add_file(
            files,
            sources,
            relative,
            relocation_path,
            "worker_relocation",
        )
        relocation_sources[int(task["seed"])] = relative

    requirements_path = plan_dir / "requirements.lock"
    requirements_path.write_bytes(
        transport.REQUIREMENTS_LOCK.encode("utf-8")
    )
    main_slurm._add_file(
        files,
        sources,
        "artifacts/runtime/requirements.lock",
        requirements_path,
        "runtime_lock",
    )
    deployment_stable = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "campaign_id": scout.CAMPAIGN_ID,
        "deployment_identity_sha256": deployment_identity,
        "diagnostic_bundle_payload_sha256": bundle["payload_sha256"],
        "activation_payload_sha256": activation["payload_sha256"],
        "code_manifest_payload_sha256": code_manifest["payload_sha256"],
        "code_inventory_sha256": code_manifest["code_inventory_sha256"],
        "code_revision": code_manifest["code_revision"],
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
            "task_count": EXACT_TASK_COUNT,
            "seed_start": common["seed_start"],
            "seed_end_inclusive": common["seed_end_inclusive"],
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "population": scout.POPULATION,
            "generations": scout.GENERATIONS,
            "inference_threads": scout.INFERENCE_THREADS,
            "one_seed_per_scheduler_task": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "fresh512_activation_evidence": False,
            "aedt_used": False,
            "gpus": 0,
            **(
                {}
                if common[
                    "manufacturing_search_profile_payload_sha256"
                ]
                is None
                else {
                    "authorized_seeds": common["authorized_seeds"],
                    "manufacturing_search_profile_payload_sha256": common[
                        "manufacturing_search_profile_payload_sha256"
                    ],
                    "geometry_constraint_profile_sha256": common[
                        "geometry_constraint_profile_sha256"
                    ],
                    "exact_equal_winding_height_initialization_and_repair_required": (
                        True
                    ),
                    "fixed_core_plate_thickness_mm": (
                        scout.FIXED_CORE_PLATE_THICKNESS_MM
                    ),
                    "fixed_winding_cold_plate_thickness_mm": (
                        scout.FIXED_WINDING_COLD_PLATE_THICKNESS_MM
                    ),
                    **(
                        {
                            "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
                        }
                        if common["secondary_gap_mode"]
                        == scout.SECONDARY_GAP_MODE_FIXED
                        else {
                            "raw_same_metric_C_rx_rx_F_UCB_gate_active": False,
                            "provisional_turn_graded_C_acquisition_gate_active": (
                                True
                            ),
                            "authenticated_turn_graded_transfer_ratio": (
                                scout.AUTHENTICATED_TURN_GRADED_TRANSFER_RATIO
                            ),
                            "secondary_interturn_gap_search_mm": {
                                "minimum": (
                                    scout.VARIABLE_SECONDARY_INTERTURN_GAP_MINIMUM_MM
                                ),
                                "maximum": (
                                    scout.VARIABLE_SECONDARY_INTERTURN_GAP_MAXIMUM_MM
                                ),
                                "step": (
                                    scout.VARIABLE_SECONDARY_INTERTURN_GAP_STEP_MM
                                ),
                            },
                            "secondary_conductor_thickness_search_mm": {
                                "minimum": (
                                    scout.SECONDARY_CONDUCTOR_THICKNESS_MINIMUM_MM
                                ),
                                "maximum": (
                                    scout.SECONDARY_CONDUCTOR_THICKNESS_MAXIMUM_MM
                                ),
                            },
                            "final_turn_graded_symmetric_FEA_required": True,
                        }
                    ),
                }
            ),
        },
        "source_checkout_authentication": source_checkout,
        "clean_source_authenticated_during_prepare": True,
        "detached_source_authenticated_before_plan": True,
        "full_model_artifact_hashes_authenticated": True,
        "scheduler_project_source_included": False,
        "scheduler_service_modified": False,
    }
    deployment = {
        **deployment_stable,
        "bundle_id": bundle_id,
        "contract_sha256": canonical_sha256(deployment_stable),
    }
    deployment_path = plan_dir / "bundle_manifest.json"
    source_map_path = plan_dir / "local_sources.json"
    plan_path = plan_dir / "offload_plan.json"
    _atomic_json(deployment_path, deployment)
    _atomic_json(source_map_path, sources)
    plan = {
        "schema_version": transport.PLAN_SCHEMA,
        "diagnostic_schema_version": PLAN_SCHEMA,
        "created_at": _now(),
        "bundle_id": bundle_id,
        "contract_sha256": deployment["contract_sha256"],
        "bundle_manifest_sha256": adapter.sha256_file(deployment_path),
        "bundle_manifest": str(deployment_path),
        "local_sources": str(source_map_path),
        "local_plan_dir": str(plan_dir),
        "remote_root": remote_base,
        "remote_bundle": remote_bundle,
        "runtime_requirements": str(requirements_path),
        "goal_bundle_root": str(bundle_root),
        "goal_bundle_payload_sha256": bundle["payload_sha256"],
        "task_count": EXACT_TASK_COUNT,
        "authorized_seeds": common["authorized_seeds"],
        "seed_start": common["seed_start"],
        "seed_end_inclusive": common["seed_end_inclusive"],
        "manufacturing_search_profile_payload_sha256": common[
            "manufacturing_search_profile_payload_sha256"
        ],
        "geometry_constraint_profile_sha256": common[
            "geometry_constraint_profile_sha256"
        ],
        "task_paths": [str(path) for path in task_paths],
        "relocation_sources": relocation_sources,
        "scheduler_claim_root": str(plan_dir / "scheduler-claims"),
        "secondary_gap_mode": common["secondary_gap_mode"],
        "scheduler_priority": common["scheduler_priority"],
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "automatic_promotion_allowed": False,
        "fresh512_activation_evidence": False,
        "recommended_resources": {
            "cpus": CPUS_PER_TASK,
            "memory_mb": MEMORY_MB_PER_TASK,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
            "maximum_parallel_tasks": EXACT_TASK_COUNT,
        },
    }
    plan["diagnostic_plan_sha256"] = canonical_sha256(plan)
    _atomic_json(plan_path, plan)
    _initialize_claim_root(plan=plan, tasks=tasks)
    return plan, deployment


def _validate_deployment_inventory(
    *,
    plan: Mapping[str, Any],
    deployment: Mapping[str, Any],
    files: Mapping[str, Any],
) -> None:
    authorized_seeds = _authorized_plan_seeds(plan)
    expected_execution = {
        "task_count": EXACT_TASK_COUNT,
        "seed_start": authorized_seeds[0],
        "seed_end_inclusive": authorized_seeds[-1],
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "population": scout.POPULATION,
        "generations": scout.GENERATIONS,
        "inference_threads": scout.INFERENCE_THREADS,
        "one_seed_per_scheduler_task": True,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "automatic_promotion_allowed": False,
        "fresh512_activation_evidence": False,
        "aedt_used": False,
        "gpus": 0,
    }
    if plan.get("manufacturing_search_profile_payload_sha256") is not None:
        expected_execution.update(
            {
                "authorized_seeds": list(authorized_seeds),
                "manufacturing_search_profile_payload_sha256": plan[
                    "manufacturing_search_profile_payload_sha256"
                ],
                "geometry_constraint_profile_sha256": plan[
                    "geometry_constraint_profile_sha256"
                ],
                "exact_equal_winding_height_initialization_and_repair_required": (
                    True
                ),
                "fixed_core_plate_thickness_mm": (
                    scout.FIXED_CORE_PLATE_THICKNESS_MM
                ),
                "fixed_winding_cold_plate_thickness_mm": (
                    scout.FIXED_WINDING_COLD_PLATE_THICKNESS_MM
                ),
                **(
                    {
                        "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
                    }
                    if plan.get(
                        "secondary_gap_mode",
                        scout.SECONDARY_GAP_MODE_FIXED,
                    )
                    == scout.SECONDARY_GAP_MODE_FIXED
                    else {
                        "raw_same_metric_C_rx_rx_F_UCB_gate_active": False,
                        "provisional_turn_graded_C_acquisition_gate_active": (
                            True
                        ),
                        "authenticated_turn_graded_transfer_ratio": (
                            scout.AUTHENTICATED_TURN_GRADED_TRANSFER_RATIO
                        ),
                        "secondary_interturn_gap_search_mm": {
                            "minimum": (
                                scout.VARIABLE_SECONDARY_INTERTURN_GAP_MINIMUM_MM
                            ),
                            "maximum": (
                                scout.VARIABLE_SECONDARY_INTERTURN_GAP_MAXIMUM_MM
                            ),
                            "step": (
                                scout.VARIABLE_SECONDARY_INTERTURN_GAP_STEP_MM
                            ),
                        },
                        "secondary_conductor_thickness_search_mm": {
                            "minimum": (
                                scout.SECONDARY_CONDUCTOR_THICKNESS_MINIMUM_MM
                            ),
                            "maximum": (
                                scout.SECONDARY_CONDUCTOR_THICKNESS_MAXIMUM_MM
                            ),
                        },
                        "final_turn_graded_symmetric_FEA_required": True,
                    }
                ),
            }
        )
    expected_resources = {
        "cpus": CPUS_PER_TASK,
        "memory_mb": MEMORY_MB_PER_TASK,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "maximum_parallel_tasks": EXACT_TASK_COUNT,
    }
    bundle_id = plan.get("bundle_id")
    remote_root = str(plan.get("remote_root") or "").rstrip("/")
    remote_bundle = str(plan.get("remote_bundle") or "")
    local_plan_dir = Path(str(plan.get("local_plan_dir") or ""))
    claim_root = Path(str(plan.get("scheduler_claim_root") or ""))
    source_paths = deployment.get("source_paths")
    identity = deployment.get("source_identity")
    source_checkout = deployment.get("source_checkout_authentication")
    if (
        deployment.get("execution_contract") != expected_execution
        or plan.get("recommended_resources") != expected_resources
        or plan.get(
            "secondary_gap_mode",
            scout.SECONDARY_GAP_MODE_FIXED,
        )
        not in scout.SECONDARY_GAP_MODES
        or plan.get("scheduler_priority")
        != (
            SCHEDULER_PRIORITY
            if plan.get(
                "secondary_gap_mode",
                scout.SECONDARY_GAP_MODE_FIXED,
            )
            == scout.SECONDARY_GAP_MODE_FIXED
            else scout.VARIABLE_SECONDARY_PROFILE_SCHEDULER_PRIORITY
        )
        or not local_plan_dir.is_absolute()
        or claim_root != local_plan_dir / "scheduler-claims"
        or not isinstance(bundle_id, str)
        or not bundle_id.startswith("mft-goal-diag-compact-")
        or not remote_root.startswith("/")
        or remote_bundle != f"{remote_root}/{bundle_id}"
        or not isinstance(source_paths, Mapping)
        or set(source_paths)
        != {
            "generation",
            "candidate",
            "quality_status",
            "code_root",
            "dataset",
            "profile",
            "code_manifest",
        }
        or not isinstance(identity, Mapping)
        or source_checkout
        != {
            "revision": deployment.get("code_revision"),
            "clean": True,
            "detached_head": True,
            "code_manifest_payload_sha256": deployment.get(
                "code_manifest_payload_sha256"
            ),
            "code_inventory_sha256": deployment.get(
                "code_inventory_sha256"
            ),
        }
        or deployment.get("relocations") != plan.get("relocation_sources")
    ):
        raise RuntimeError("diagnostic deployment execution authority mismatch")

    expected_roles = {
        "candidate": "artifacts/g0/candidate.json",
        "quality_status": "artifacts/g0/quality_status.json",
        "code_root": "artifacts/code",
        "dataset": "artifacts/g0/dataset/train.parquet",
        "profile": "artifacts/g0/profile.json",
        "code_manifest": "artifacts/campaign/code_manifest.json",
    }
    if any(
        source_paths.get(role) != f"{remote_bundle}/{relative}"
        for role, relative in expected_roles.items()
    ):
        raise RuntimeError("diagnostic deployment source role mismatch")
    generation_prefix = (
        f"{remote_bundle}/artifacts/g0/registry/generations/"
    )
    generation_path = str(source_paths.get("generation") or "")
    generation_name = generation_path.removeprefix(generation_prefix)
    if (
        not generation_path.startswith(generation_prefix)
        or not generation_name
        or "/" in generation_name
        or "\\" in generation_name
    ):
        raise RuntimeError("diagnostic deployment generation role mismatch")
    generation_relative = (
        f"artifacts/g0/registry/generations/{generation_name}"
    )

    model_hashes: dict[str, str] = {}
    code_inventory: dict[str, dict[str, Any]] = {}
    for relative, raw_record in files.items():
        safe = _safe_relative(str(relative), "deployment file path")
        if safe != relative or not isinstance(raw_record, Mapping):
            raise RuntimeError("diagnostic deployment file record mismatch")
        record = dict(raw_record)
        if (
            set(record) != {"sha256", "size", "kind"}
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or isinstance(record["size"], bool)
            or not isinstance(record["size"], int)
            or record["size"] < 0
            or not isinstance(record["kind"], str)
        ):
            raise RuntimeError("diagnostic deployment file record mismatch")
        if record["kind"] == "model_artifact":
            prefix = generation_relative + "/"
            if not relative.startswith(prefix):
                raise RuntimeError("diagnostic model artifact escaped generation")
            model_hashes[relative.removeprefix(prefix)] = record["sha256"]
        elif record["kind"] == "diagnostic_code":
            code_inventory[relative] = {
                "sha256": record["sha256"],
                "size": record["size"],
            }
    source_bindings = (
        (
            f"{generation_relative}/train_report.json",
            "generation_report",
            "train_report_sha256",
        ),
        (
            "artifacts/g0/candidate.json",
            "generation_candidate",
            "candidate_sha256",
        ),
        (
            "artifacts/g0/quality_status.json",
            "quality_status",
            "quality_status_sha256",
        ),
        (
            "artifacts/g0/dataset/train.parquet",
            "training_dataset",
            "dataset_sha256",
        ),
    )
    if (
        not model_hashes
        or canonical_sha256(model_hashes)
        != identity.get("evaluation_model_sha256")
        or not code_inventory
        or canonical_sha256(code_inventory)
        != deployment.get("code_inventory_sha256")
        or any(
            relative not in files
            or files[relative].get("kind") != kind
            or files[relative].get("sha256") != identity.get(identity_field)
            for relative, kind, identity_field in source_bindings
        )
    ):
        raise RuntimeError("diagnostic deployment model/source inventory mismatch")


def authenticate_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan, deployment, source_map = transport.load_plan(
        plan_path.resolve(strict=True)
    )
    authorized_seeds = _authorized_plan_seeds(plan)
    unsigned_plan = {
        key: item
        for key, item in plan.items()
        if key != "diagnostic_plan_sha256"
    }
    if (
        plan.get("diagnostic_plan_sha256")
        != canonical_sha256(unsigned_plan)
        or plan.get("diagnostic_schema_version") != PLAN_SCHEMA
        or plan.get("task_count") != EXACT_TASK_COUNT
        or plan.get("screening_only") is not True
        or plan.get("production_eligible") is not False
        or plan.get("final_design_claim_allowed") is not False
        or plan.get("automatic_promotion_allowed") is not False
        or plan.get("fresh512_activation_evidence") is not False
        or deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("campaign_id") != scout.CAMPAIGN_ID
        or deployment.get("bundle_id") != plan.get("bundle_id")
        or deployment.get("contract_sha256") != plan.get("contract_sha256")
        or deployment.get("full_model_artifact_hashes_authenticated") is not True
        or deployment.get("clean_source_authenticated_during_prepare")
        is not True
        or deployment.get("detached_source_authenticated_before_plan")
        is not True
        or deployment.get("scheduler_project_source_included") is not False
        or deployment.get("scheduler_service_modified") is not False
    ):
        raise RuntimeError("diagnostic offload plan authority mismatch")
    stable = {
        key: item
        for key, item in deployment.items()
        if key not in {"bundle_id", "contract_sha256"}
    }
    if deployment["contract_sha256"] != canonical_sha256(stable):
        raise RuntimeError("diagnostic deployment contract seal mismatch")
    files = deployment.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != set(source_map)
        or not files
    ):
        raise RuntimeError("diagnostic deployment source inventory mismatch")
    _validate_deployment_inventory(
        plan=plan,
        deployment=deployment,
        files=files,
    )
    for relative, record in files.items():
        local = Path(source_map[relative]).resolve(strict=True)
        if (
            local.stat().st_size != int(record["size"])
            or adapter.sha256_file(local) != record["sha256"]
        ):
            raise RuntimeError(
                f"diagnostic deployment source changed: {relative}"
            )
    (
        bundle,
        _scheduler,
        activation,
        code_manifest,
        _task_paths,
        tasks,
        common,
    ) = _bundle_tasks(Path(plan["goal_bundle_root"]))
    if (
        bundle["payload_sha256"] != plan["goal_bundle_payload_sha256"]
        or activation["payload_sha256"]
        != deployment["activation_payload_sha256"]
        or code_manifest["payload_sha256"]
        != deployment["code_manifest_payload_sha256"]
        or code_manifest["code_revision"] != deployment["code_revision"]
        or common["source_identity"] != deployment["source_identity"]
        or (
            set(plan["relocation_sources"]) != set(authorized_seeds)
            and {
                str(key) for key in plan["relocation_sources"]
            }
            != {str(seed) for seed in authorized_seeds}
        )
        or common.get("authorized_seeds", list(EXACT_SEEDS))
        != list(authorized_seeds)
        or common.get("manufacturing_search_profile_payload_sha256")
        != plan.get("manufacturing_search_profile_payload_sha256")
        or common.get("geometry_constraint_profile_sha256")
        != plan.get("geometry_constraint_profile_sha256")
        or common.get(
            "secondary_gap_mode", scout.SECONDARY_GAP_MODE_FIXED
        )
        != plan.get(
            "secondary_gap_mode", scout.SECONDARY_GAP_MODE_FIXED
        )
        or common.get("scheduler_priority", SCHEDULER_PRIORITY)
        != plan.get("scheduler_priority", SCHEDULER_PRIORITY)
    ):
        raise RuntimeError("diagnostic plan/bundle source binding mismatch")
    claim_authority = _load_claim_root(plan=plan, tasks=tasks)
    evidence = _seal(
        {
            "schema_version": AUTHENTICATION_SCHEMA,
            "plan_contract_sha256": plan["contract_sha256"],
            "diagnostic_plan_sha256": plan["diagnostic_plan_sha256"],
            "scheduler_claim_root_sha256": claim_authority["sha256"],
            "deployment_manifest_sha256": plan[
                "bundle_manifest_sha256"
            ],
            "diagnostic_bundle_payload_sha256": bundle["payload_sha256"],
            "activation_payload_sha256": activation["payload_sha256"],
            "code_manifest_payload_sha256": code_manifest["payload_sha256"],
            "code_revision": code_manifest["code_revision"],
            "task_count": EXACT_TASK_COUNT,
            "seed_start": authorized_seeds[0],
            "seed_end_inclusive": authorized_seeds[-1],
            "authorized_seeds": list(authorized_seeds),
            "manufacturing_search_profile_payload_sha256": plan.get(
                "manufacturing_search_profile_payload_sha256"
            ),
            "geometry_constraint_profile_sha256": plan.get(
                "geometry_constraint_profile_sha256"
            ),
            "secondary_gap_mode": plan.get(
                "secondary_gap_mode",
                scout.SECONDARY_GAP_MODE_FIXED,
            ),
            "scheduler_priority": plan.get(
                "scheduler_priority", SCHEDULER_PRIORITY
            ),
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "compact_search_contract_sha256": common[
                "compact_search_contract_sha256"
            ],
            "compact_coordinate_bank_sha256": common[
                "compact_coordinate_bank_sha256"
            ],
            "dataset_sha256": common["source_identity"]["dataset_sha256"],
            "evaluation_model_sha256": common["source_identity"][
                "evaluation_model_sha256"
            ],
            "quality_status_sha256": common["source_identity"][
                "quality_status_sha256"
            ],
            "source_quality_passed": common["source_quality_passed"],
            "local_file_count": len(files),
            "all_local_file_hashes_reauthenticated": True,
            "full_model_artifact_hashes_reauthenticated": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "fresh512_activation_evidence": False,
            "scheduler_post_count": 0,
        }
    )
    return plan, deployment, tasks, evidence


def _task_name_prefix(plan: Mapping[str, Any]) -> str:
    bundle_id = str(plan.get("bundle_id") or "")
    identity = bundle_id.removeprefix("mft-goal-diag-compact-")
    if (
        len(identity) != 24
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise RuntimeError("diagnostic bundle task namespace is invalid")
    return f"{TASK_NAME_PREFIX}{identity}-"


def scheduler_payload(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    priority: int | None = None,
) -> dict[str, Any]:
    expected_priority = int(
        plan.get("scheduler_priority", SCHEDULER_PRIORITY)
    )
    effective_priority = (
        expected_priority if priority is None else priority
    )
    if (
        isinstance(effective_priority, bool)
        or not isinstance(effective_priority, int)
        or effective_priority != expected_priority
        or expected_priority
        not in {
            SCHEDULER_PRIORITY,
            scout.VARIABLE_SECONDARY_PROFILE_SCHEDULER_PRIORITY,
        }
    ):
        raise RuntimeError("diagnostic Scheduler priority authority mismatch")
    seed = int(task["seed"])
    turns = int(task["fixed_primary_turns"])
    relocation = plan["relocation_sources"].get(seed)
    if relocation is None:
        relocation = plan["relocation_sources"].get(str(seed))
    if not isinstance(relocation, str):
        raise RuntimeError("diagnostic task relocation is unavailable")
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
            "exec python "
            "artifacts/code/tools/mft_goal_diagnostic_compact_scout.py "
            'execute --payload "$payload_path" '
            f"--relocation {shlex.quote(relocation)} "
            '--output "$output"',
        ]
    )
    envelope = {
        "name": (
            f"{_task_name_prefix(plan)}"
            f"s{seed}-n1-{turns}"
        ),
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
        "priority": int(effective_priority),
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
    }
    dedupe = canonical_sha256(
        {
            "schema_version": "mft-goal-diagnostic-scheduler-dedupe-v1",
            "plan_contract_sha256": plan["contract_sha256"],
            "diagnostic_plan_sha256": plan["diagnostic_plan_sha256"],
            "immutable_scheduler_envelope": envelope,
        }
    )
    return {
        **envelope,
        "dedupe_key": DEDUPE_PREFIX + dedupe,
    }


def _link_like(path: Path) -> bool:
    status = os.lstat(path)
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(status, "st_file_attributes", 0) & reparse)


def _plain_directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"{label} must be an absolute path")
    resolved = raw.resolve(strict=True)
    if (
        os.path.normcase(str(raw)) != os.path.normcase(str(resolved))
        or _link_like(resolved)
        or not resolved.is_dir()
    ):
        raise RuntimeError(f"{label} is aliased or not a plain directory")
    return resolved


def _plain_json(path: Path, label: str) -> dict[str, Any]:
    raw = Path(path)
    if _link_like(raw) or not raw.is_file():
        raise RuntimeError(f"{label} is not a plain file")
    return _read_json(raw)


def _plain_child(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path))
    if raw.name in {"", ".", ".."} or ".." in raw.parts:
        raise RuntimeError(f"{label} is unsafe")
    parent = _plain_directory(raw.parent, f"{label} parent")
    expected = parent / raw.name
    if os.path.normcase(str(raw)) != os.path.normcase(str(expected)):
        raise RuntimeError(f"{label} is aliased")
    if expected.exists() and _link_like(expected):
        raise RuntimeError(f"{label} uses a link or reparse point")
    return expected


def _claim_root_authority(
    *,
    plan: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    authorized_seeds = _authorized_plan_seeds(plan)
    inventory = []
    for task in sorted(tasks, key=lambda item: int(item["seed"])):
        payload = scheduler_payload(
            plan=plan,
            task=task,
            priority=int(
                plan.get("scheduler_priority", SCHEDULER_PRIORITY)
            ),
        )
        inventory.append(
            {
                "seed": int(task["seed"]),
                "task_payload_sha256": task["payload_sha256"],
                "name": payload["name"],
                "dedupe_key": payload["dedupe_key"],
                "scheduler_payload_sha256": canonical_sha256(payload),
            }
        )
    if (
        len(inventory) != EXACT_TASK_COUNT
        or [item["seed"] for item in inventory] != list(authorized_seeds)
    ):
        raise RuntimeError("diagnostic claim inventory is not exact100")
    return _seal(
        {
            "schema_version": CLAIM_ROOT_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "plan_contract_sha256": plan["contract_sha256"],
            "diagnostic_plan_sha256": plan["diagnostic_plan_sha256"],
            "task_count": EXACT_TASK_COUNT,
            "authorized_seeds": list(authorized_seeds),
            "manufacturing_search_profile_payload_sha256": plan.get(
                "manufacturing_search_profile_payload_sha256"
            ),
            "task_inventory": inventory,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "fresh512_activation_evidence": False,
            "scheduler_post_authority_per_seed": 1,
        }
    )


def _initialize_claim_root(
    *,
    plan: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(plan["scheduler_claim_root"])
    parent = _plain_directory(root.parent, "diagnostic claim-root parent")
    if root.parent != parent:
        raise RuntimeError("diagnostic claim root escaped its plan directory")
    try:
        os.mkdir(root)
        os.mkdir(root / "claims")
    except FileExistsError as exc:
        raise RuntimeError("diagnostic claim root already exists") from exc
    authority = _claim_root_authority(plan=plan, tasks=tasks)
    _immutable_json(root / "authority.json", authority)
    return _load_claim_root(plan=plan, tasks=tasks)


def _load_claim_root(
    *,
    plan: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _plain_directory(
        Path(plan["scheduler_claim_root"]),
        "diagnostic claim root",
    )
    expected_parent = Path(plan["local_plan_dir"]).resolve(strict=True)
    if root.parent != expected_parent:
        raise RuntimeError("diagnostic claim root is outside the plan")
    _plain_directory(root / "claims", "diagnostic claims directory")
    observed = _validate_seal(
        _plain_json(root / "authority.json", "diagnostic claim authority"),
        schema=CLAIM_ROOT_SCHEMA,
    )
    expected = _claim_root_authority(plan=plan, tasks=tasks)
    if observed != expected:
        raise RuntimeError("diagnostic claim-root authority mismatch")
    return observed


def _pending_claim(
    *,
    authority: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": PENDING_CLAIM_SCHEMA,
            "state": "pending",
            "claim_root_sha256": authority["sha256"],
            "seed": int(payload["payload_json"]["seed"]),
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "task_payload_sha256": payload["payload_json"]["payload_sha256"],
            "scheduler_payload_sha256": canonical_sha256(payload),
            "scheduler_post_authority_count": 1,
        }
    )


def _claim_directory(
    *,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Path:
    seed = int(payload["payload_json"]["seed"])
    if seed not in _authorized_plan_seeds(plan):
        raise RuntimeError("diagnostic claim seed is outside exact100")
    claims = _plain_directory(
        Path(plan["scheduler_claim_root"]) / "claims",
        "diagnostic claims directory",
    )
    return claims / f"seed-{seed}"


def _validate_pending_claim(
    *,
    path: Path,
    authority: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _validate_seal(
        _plain_json(path, "diagnostic pending claim"),
        schema=PENDING_CLAIM_SCHEMA,
    )
    expected = _pending_claim(authority=authority, payload=payload)
    if observed != expected:
        raise RuntimeError("diagnostic pending claim authority mismatch")
    return observed


def _validate_finalized_claim(
    *,
    path: Path,
    pending: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(
        _plain_json(path, "diagnostic finalized claim"),
        schema=FINALIZED_CLAIM_SCHEMA,
    )
    task_id = value.get("task_id")
    if (
        set(value)
        != {
            "schema_version",
            "state",
            "pending_claim_sha256",
            "seed",
            "task_id",
            "scheduler_payload_sha256",
            "scheduler_identity_verified_by_GET",
            "scheduler_post_authority_count",
            "sha256",
        }
        or value.get("state") != "finalized"
        or value.get("pending_claim_sha256") != pending["sha256"]
        or value.get("seed") != payload["payload_json"]["seed"]
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or value.get("scheduler_payload_sha256")
        != canonical_sha256(payload)
        or value.get("scheduler_identity_verified_by_GET") is not True
        or value.get("scheduler_post_authority_count") != 1
    ):
        raise RuntimeError("diagnostic finalized claim mismatch")
    return value


def _acquire_task_claim(
    *,
    plan: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    authority = _load_claim_root(plan=plan, tasks=tasks)
    claim_dir = _claim_directory(plan=plan, payload=payload)
    pending_path = claim_dir / "pending.json"
    finalized_path = claim_dir / "finalized.json"
    fresh = False
    try:
        os.mkdir(claim_dir)
        fresh = True
    except FileExistsError:
        _plain_directory(claim_dir, "diagnostic task claim directory")
    if fresh:
        pending = _pending_claim(authority=authority, payload=payload)
        _immutable_json(pending_path, pending)
    else:
        deadline = time.monotonic() + 5.0
        while not pending_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not pending_path.exists():
            raise RuntimeError(
                "diagnostic claim has no pending authority; manual audit required"
            )
        pending = _validate_pending_claim(
            path=pending_path,
            authority=authority,
            payload=payload,
        )
    if finalized_path.exists():
        finalized = _validate_finalized_claim(
            path=finalized_path,
            pending=pending,
            payload=payload,
        )
        return "existing_finalized", pending, finalized
    return ("fresh_pending" if fresh else "existing_pending"), pending, None


def _finalize_task_claim(
    *,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
    pending: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    finalized = _seal(
        {
            "schema_version": FINALIZED_CLAIM_SCHEMA,
            "state": "finalized",
            "pending_claim_sha256": pending["sha256"],
            "seed": payload["payload_json"]["seed"],
            "task_id": int(task_id),
            "scheduler_payload_sha256": canonical_sha256(payload),
            "scheduler_identity_verified_by_GET": True,
            "scheduler_post_authority_count": 1,
        }
    )
    path = _claim_directory(plan=plan, payload=payload) / "finalized.json"
    if not path.exists():
        try:
            _immutable_json(path, finalized)
        except RuntimeError:
            if not path.exists():
                raise
    durable = _validate_finalized_claim(
        path=path,
        pending=pending,
        payload=payload,
    )
    if durable != finalized:
        raise RuntimeError(
            "diagnostic task claim finalized with a different task"
        )
    return durable


class SchedulerClient:
    """Small namespace-scoped client whose only mutation is submit_task."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.post_count = 0
        self.get_count = 0

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        self.get_count += method == "GET"
        self.post_count += method == "POST"
        return transport._api_json(
            self.base_url + path,
            method=method,
            payload=None if payload is None else dict(payload),
            timeout=self.timeout,
        )

    def list_namespace_tasks(
        self,
        name_prefix: str,
    ) -> list[dict[str, Any]]:
        if (
            not name_prefix.startswith(TASK_NAME_PREFIX)
            or not name_prefix.endswith("-")
            or len(name_prefix) <= len(TASK_NAME_PREFIX) + 1
        ):
            raise RuntimeError("diagnostic Scheduler namespace is invalid")
        result: list[dict[str, Any]] = []
        page = 1
        filtered_total: int | None = None
        while True:
            query = urllib.parse.urlencode(
                {
                    "name_prefix": name_prefix,
                    "sort_by": "id",
                    "sort_order": "asc",
                    "paged": "true",
                    "page": page,
                    "page_size": 10_000,
                }
            )
            value = self._request(f"/api/tasks?{query}")
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    "diagnostic Scheduler inventory is not paged"
                )
            items = value.get("items")
            total = value.get("filtered_total")
            observed_page = value.get("page")
            has_next = value.get("has_next")
            if (
                not isinstance(items, list)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total < 0
                or observed_page != page
                or not isinstance(has_next, bool)
                or value.get("page_size") != 10_000
                or value.get("sort_by") != "id"
                or value.get("sort_order") != "asc"
            ):
                raise RuntimeError(
                    "diagnostic Scheduler page metadata mismatch"
                )
            if filtered_total is None:
                filtered_total = total
            elif total != filtered_total:
                raise RuntimeError(
                    "diagnostic Scheduler inventory changed during pagination"
                )
            for row in items:
                if (
                    not isinstance(row, dict)
                    or not str(row.get("name") or "").startswith(name_prefix)
                    or not str(row.get("dedupe_key") or "").startswith(
                        DEDUPE_PREFIX
                    )
                ):
                    raise RuntimeError(
                        "diagnostic Scheduler inventory escaped its namespace"
                    )
                result.append(dict(row))
            if not has_next:
                break
            page += 1
        if filtered_total != len(result):
            raise RuntimeError(
                "diagnostic Scheduler inventory pagination is incomplete"
            )
        return result

    def submit_task(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if (
            not str(payload.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not str(payload.get("dedupe_key") or "").startswith(
                DEDUPE_PREFIX
            )
        ):
            raise RuntimeError("foreign diagnostic Scheduler payload")
        value = self._request("/api/tasks", method="POST", payload=payload)
        if not isinstance(value, dict):
            raise RuntimeError("diagnostic Scheduler POST response is invalid")
        return value

    def get_task(self, task_id: int) -> dict[str, Any]:
        value = self._request(f"/api/tasks/{int(task_id)}")
        if not isinstance(value, dict):
            raise RuntimeError("diagnostic Scheduler task detail is invalid")
        return value


def _task_authentication(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, str]:
    observation = scheduler_task_observation(row)
    if (
        observation is None
        or not scheduler_task_identity_matches(row, expected)
    ):
        raise RuntimeError(
            f"{label} changed the diagnostic Scheduler task identity"
        )
    return observation


def _remote_ready_authenticated(
    *,
    plan: Mapping[str, Any],
    accounts_path: Path,
    scheduler_source: Path,
    staging_account: str,
) -> dict[str, Any]:
    account, ssh_session = transport._account(
        accounts_path, scheduler_source, staging_account
    )
    with ssh_session(account, default_timeout=60) as session:
        command = (
            f"cat {shlex.quote(plan['remote_bundle'] + '/READY.json')}"
        )
        result = session.run(command, timeout=60)
    if result.exit_code != 0:
        raise RuntimeError("diagnostic remote READY is unavailable")
    try:
        ready = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("diagnostic remote READY is invalid") from exc
    if (
        not isinstance(ready, dict)
        or ready.get("bundle_id") != plan["bundle_id"]
        or ready.get("bundle_manifest_sha256")
        != plan["bundle_manifest_sha256"]
        or ready.get("runtime_verified") is not True
    ):
        raise RuntimeError("diagnostic remote READY identity mismatch")
    return ready


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
    authentication: Mapping[str, Any],
    ready: Mapping[str, Any],
    scheduler_url: str,
) -> dict[str, Any]:
    authorized_seeds = _authorized_plan_seeds(plan)
    receipt = _validate_seal(value, schema=RECEIPT_SCHEMA)
    rows = receipt.get("tasks")
    required = {
        "schema_version",
        "observed_at",
        "apply",
        "plan_contract_sha256",
        "diagnostic_plan_sha256",
        "bundle_id",
        "authentication_sha256",
        "scheduler_claim_root_sha256",
        "remote_ready",
        "task_count",
        "tasks",
        "submitted_count",
        "existing_count",
        "absent_count",
        "campaign_authorized_post_count",
        "first_clean_run_exact100_scheduler_posts",
        "scheduler_url",
        "scheduler_endpoint",
        "scheduler_post_count",
        "scheduler_get_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "screening_only",
        "production_eligible",
        "final_design_claim_allowed",
        "automatic_promotion_allowed",
        "fresh512_activation_evidence",
        "fea_submission_performed",
        "sha256",
    }
    expected_rows = {
        dedupe: {
            "seed": payload["payload_json"]["seed"],
            "name": payload["name"],
        }
        for dedupe, payload in expected.items()
    }
    if (
        len(expected_rows) != EXACT_TASK_COUNT
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise RuntimeError("diagnostic submission receipt task inventory mismatch")
    count_fields = (
        "submitted_count",
        "existing_count",
        "absent_count",
        "campaign_authorized_post_count",
        "scheduler_post_count",
        "scheduler_get_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
    )
    if any(
        isinstance(receipt.get(field), bool)
        or not isinstance(receipt.get(field), int)
        or receipt[field] < 0
        for field in count_fields
    ):
        raise RuntimeError("diagnostic submission receipt count mismatch")
    if (
        set(receipt) != required
        or receipt.get("apply") is not True
        or receipt.get("plan_contract_sha256") != plan["contract_sha256"]
        or receipt.get("diagnostic_plan_sha256")
        != plan["diagnostic_plan_sha256"]
        or receipt.get("bundle_id") != plan["bundle_id"]
        or receipt.get("authentication_sha256") != authentication["sha256"]
        or receipt.get("scheduler_claim_root_sha256")
        != authentication["scheduler_claim_root_sha256"]
        or receipt.get("remote_ready") != ready
        or receipt.get("task_count") != EXACT_TASK_COUNT
        or receipt.get("submitted_count")
        + receipt.get("existing_count")
        != EXACT_TASK_COUNT
        or receipt.get("absent_count") != 0
        or receipt.get("campaign_authorized_post_count")
        != EXACT_TASK_COUNT
        or receipt.get("scheduler_post_count")
        != receipt.get("submitted_count")
        or receipt.get("first_clean_run_exact100_scheduler_posts")
        is not (
            receipt.get("submitted_count") == EXACT_TASK_COUNT
            and receipt.get("existing_count") == 0
        )
        or receipt.get("scheduler_url") != scheduler_url.rstrip("/")
        or receipt.get("scheduler_endpoint") != "POST /api/tasks"
        or isinstance(receipt.get("scheduler_get_count"), bool)
        or not isinstance(receipt.get("scheduler_get_count"), int)
        or receipt.get("scheduler_get_count") < EXACT_TASK_COUNT + 1
        or receipt.get("screening_only") is not True
        or receipt.get("production_eligible") is not False
        or receipt.get("final_design_claim_allowed") is not False
        or receipt.get("automatic_promotion_allowed") is not False
        or receipt.get("fresh512_activation_evidence") is not False
        or receipt.get("fea_submission_performed") is not False
        or receipt.get("scheduler_cancel_count") != 0
        or receipt.get("scheduler_preempt_count") != 0
        or len(rows) != EXACT_TASK_COUNT
        or {row.get("seed") for row in rows} != set(authorized_seeds)
        or len({row.get("task_id") for row in rows}) != EXACT_TASK_COUNT
        or len({row.get("dedupe_key") for row in rows}) != EXACT_TASK_COUNT
    ):
        raise RuntimeError("diagnostic submission receipt mismatch")
    for row in rows:
        dedupe = row.get("dedupe_key")
        task_id = row.get("task_id")
        claim_sha = row.get("claim_finalized_sha256")
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "seed",
                "dedupe_key",
                "name",
                "source",
                "status",
                "task_id",
                "claim_finalized_sha256",
                "scheduler_post_authority_count",
            }
            or dedupe not in expected_rows
            or row.get("seed") != expected_rows[dedupe]["seed"]
            or row.get("name") != expected_rows[dedupe]["name"]
            or row.get("source")
            not in {"submitted", "recovered_pending", "finalized_claim"}
            or row.get("status") not in SCHEDULER_TASK_STATUSES
            or isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not isinstance(claim_sha, str)
            or len(claim_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in claim_sha
            )
            or row.get("scheduler_post_authority_count") != 1
        ):
            raise RuntimeError("diagnostic submission receipt task mismatch")
    return receipt


def _scheduler_inventory(
    *,
    scheduler: Any,
    name_prefix: str,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    inventory: dict[str, Mapping[str, Any]] = {}
    names: dict[str, str] = {}
    for row in scheduler.list_namespace_tasks(name_prefix):
        if not isinstance(row, Mapping):
            raise RuntimeError("diagnostic inventory row is invalid")
        dedupe = str(row.get("dedupe_key") or "")
        name = str(row.get("name") or "")
        if scheduler_task_observation(row) is None:
            raise RuntimeError("diagnostic inventory task observation is invalid")
        if (
            dedupe not in expected
            or name != expected[dedupe]["name"]
            or dedupe in inventory
            or (name in names and names[name] != dedupe)
        ):
            raise RuntimeError(
                "diagnostic Scheduler inventory identity is foreign or duplicated"
            )
        inventory[dedupe] = dict(row)
        names[name] = dedupe
    return inventory


def submit(
    *,
    plan_path: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = DEFAULT_STAGING_ACCOUNT,
    priority: int | None = None,
    apply: bool = False,
    receipt_out: Path | None = None,
    client: SchedulerClient | Any | None = None,
    remote_ready: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan, _deployment, tasks, authentication = authenticate_plan(plan_path)
    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            scheduler_payload(plan=plan, task=task, priority=priority)
            for task in tasks
        )
    }
    if len(expected) != EXACT_TASK_COUNT:
        raise RuntimeError("diagnostic Scheduler dedupe identity is not unique")
    if apply and receipt_out is None:
        raise RuntimeError("--apply requires --receipt-out")
    if apply:
        assert receipt_out is not None
        receipt_out = _plain_child(
            receipt_out,
            "diagnostic submission receipt",
        )
    ready = (
        dict(remote_ready)
        if remote_ready is not None
        else _remote_ready_authenticated(
            plan=plan,
            accounts_path=accounts_path,
            scheduler_source=scheduler_source,
            staging_account=staging_account,
        )
    )
    if (
        ready.get("bundle_id") != plan["bundle_id"]
        or ready.get("bundle_manifest_sha256")
        != plan["bundle_manifest_sha256"]
        or ready.get("runtime_verified") is not True
    ):
        raise RuntimeError("diagnostic remote READY precondition failed")
    scheduler = client or SchedulerClient(scheduler_url)
    name_prefix = _task_name_prefix(plan)
    inventory = _scheduler_inventory(
        scheduler=scheduler,
        name_prefix=name_prefix,
        expected=expected,
    )

    existing_receipt = None
    if apply and receipt_out is not None and receipt_out.is_file():
        existing_receipt = _validate_receipt(
            _plain_json(receipt_out, "diagnostic submission receipt"),
            plan=plan,
            expected=expected,
            authentication=authentication,
            ready=ready,
            scheduler_url=scheduler_url,
        )

    ordered_payloads = sorted(
        expected.values(),
        key=lambda payload: int(payload["payload_json"]["seed"]),
    )
    if not apply:
        preview_rows = []
        for payload in ordered_payloads:
            dedupe = payload["dedupe_key"]
            row = inventory.get(dedupe)
            if row is None:
                preview_rows.append(
                    {
                        "seed": payload["payload_json"]["seed"],
                        "dedupe_key": dedupe,
                        "name": payload["name"],
                        "status": "absent",
                        "task_id": None,
                    }
                )
                continue
            observation = scheduler_task_observation(row)
            assert observation is not None
            detail = scheduler.get_task(observation[0])
            task_id, status = _task_authentication(
                detail,
                payload,
                label="dry-run existing",
            )
            preview_rows.append(
                {
                    "seed": payload["payload_json"]["seed"],
                    "dedupe_key": dedupe,
                    "name": payload["name"],
                    "status": status,
                    "task_id": task_id,
                }
            )
        if scheduler.post_count != 0:
            raise RuntimeError("diagnostic dry-run performed a Scheduler POST")
        return _seal(
            {
                "schema_version": DRY_RUN_SCHEMA,
                "observed_at": _now(),
                "apply": False,
                "plan_contract_sha256": plan["contract_sha256"],
                "diagnostic_plan_sha256": plan[
                    "diagnostic_plan_sha256"
                ],
                "authentication_sha256": authentication["sha256"],
                "remote_ready": ready,
                "task_count": EXACT_TASK_COUNT,
                "tasks": preview_rows,
                "existing_count": len(inventory),
                "absent_count": EXACT_TASK_COUNT - len(inventory),
                "scheduler_post_count": 0,
                "scheduler_get_count": scheduler.get_count,
                "screening_only": True,
                "production_eligible": False,
                "final_design_claim_allowed": False,
                "automatic_promotion_allowed": False,
                "fresh512_activation_evidence": False,
            }
        )

    rows: list[dict[str, Any]] = []
    submitted_count = 0
    existing_count = 0
    for payload in ordered_payloads:
        dedupe = payload["dedupe_key"]
        claim_state, pending, finalized = _acquire_task_claim(
            plan=plan,
            tasks=tasks,
            payload=payload,
        )
        row = inventory.get(dedupe)
        if claim_state == "fresh_pending":
            refreshed = _scheduler_inventory(
                scheduler=scheduler,
                name_prefix=name_prefix,
                expected=expected,
            )
            if dedupe in refreshed:
                raise RuntimeError(
                    "unclaimed diagnostic task appeared before authorized POST"
                )
            response = scheduler.submit_task(payload)
            if response.get("deduped") is not False:
                raise RuntimeError(
                    "fresh diagnostic claim received a deduplicated POST"
                )
            task_id = response.get("task_id", response.get("id"))
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
            ):
                raise RuntimeError("diagnostic Scheduler POST lacks task id")
            row = scheduler.get_task(task_id)
            submitted_count += 1
            source = "submitted"
        elif claim_state == "existing_pending":
            if row is None:
                refreshed = _scheduler_inventory(
                    scheduler=scheduler,
                    name_prefix=name_prefix,
                    expected=expected,
                )
                row = refreshed.get(dedupe)
            if row is None:
                raise RuntimeError(
                    "pending diagnostic claim has no Scheduler task; "
                    "re-POST is forbidden and manual audit is required"
                )
            observation = scheduler_task_observation(row)
            assert observation is not None
            row = scheduler.get_task(observation[0])
            existing_count += 1
            source = "recovered_pending"
        else:
            assert finalized is not None
            finalized_id = int(finalized["task_id"])
            if row is not None:
                observation = scheduler_task_observation(row)
                if observation is None or observation[0] != finalized_id:
                    raise RuntimeError(
                        "finalized diagnostic claim/inventory task mismatch"
                    )
            row = scheduler.get_task(finalized_id)
            existing_count += 1
            source = "finalized_claim"
        task_id, status = _task_authentication(
            row,
            payload,
            label=source,
        )
        if finalized is None:
            finalized = _finalize_task_claim(
                plan=plan,
                payload=payload,
                pending=pending,
                task_id=task_id,
            )
        elif finalized["task_id"] != task_id:
            raise RuntimeError("diagnostic finalized claim task ID changed")
        rows.append(
            {
                "seed": payload["payload_json"]["seed"],
                "dedupe_key": dedupe,
                "name": payload["name"],
                "source": source,
                "status": status,
                "task_id": task_id,
                "claim_finalized_sha256": finalized["sha256"],
                "scheduler_post_authority_count": finalized[
                    "scheduler_post_authority_count"
                ],
            }
        )
    if existing_receipt is not None:
        sealed = {
            row["dedupe_key"]: (
                row["task_id"],
                row["claim_finalized_sha256"],
            )
            for row in existing_receipt["tasks"]
        }
        observed = {
            row["dedupe_key"]: (
                row["task_id"],
                row["claim_finalized_sha256"],
            )
            for row in rows
        }
        if (
            sealed != observed
            or scheduler.post_count != 0
            or submitted_count != 0
        ):
            raise RuntimeError(
                "diagnostic receipt/live mapping or exactly-once state changed"
            )
        return existing_receipt

    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "observed_at": _now(),
        "apply": True,
        "plan_contract_sha256": plan["contract_sha256"],
        "diagnostic_plan_sha256": plan["diagnostic_plan_sha256"],
        "bundle_id": plan["bundle_id"],
        "authentication_sha256": authentication["sha256"],
        "scheduler_claim_root_sha256": authentication[
            "scheduler_claim_root_sha256"
        ],
        "remote_ready": ready,
        "task_count": EXACT_TASK_COUNT,
        "tasks": rows,
        "submitted_count": submitted_count,
        "existing_count": existing_count,
        "absent_count": 0,
        "campaign_authorized_post_count": sum(
            row["scheduler_post_authority_count"] for row in rows
        ),
        "first_clean_run_exact100_scheduler_posts": (
            submitted_count == EXACT_TASK_COUNT and existing_count == 0
        ),
        "scheduler_url": scheduler_url.rstrip("/"),
        "scheduler_endpoint": "POST /api/tasks",
        "scheduler_post_count": scheduler.post_count,
        "scheduler_get_count": scheduler.get_count,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "automatic_promotion_allowed": False,
        "fresh512_activation_evidence": False,
        "fea_submission_performed": False,
    }
    receipt = _seal(unsigned)
    if (
        len(rows) != EXACT_TASK_COUNT
        or len({row["task_id"] for row in rows}) != EXACT_TASK_COUNT
        or scheduler.post_count != submitted_count
        or submitted_count + existing_count != EXACT_TASK_COUNT
        or unsigned["campaign_authorized_post_count"] != EXACT_TASK_COUNT
        or scheduler.get_count < EXACT_TASK_COUNT + 1
    ):
        raise RuntimeError(
            "diagnostic exactly-once submission accounting mismatch"
        )
    _validate_receipt(
        receipt,
        plan=plan,
        expected=expected,
        authentication=authentication,
        ready=ready,
        scheduler_url=scheduler_url,
    )
    assert receipt_out is not None
    try:
        _immutable_json(receipt_out, receipt)
    except RuntimeError:
        if not receipt_out.is_file():
            raise
        concurrent = _validate_receipt(
            _plain_json(receipt_out, "diagnostic submission receipt"),
            plan=plan,
            expected=expected,
            authentication=authentication,
            ready=ready,
            scheduler_url=scheduler_url,
        )
        sealed = {
            row["dedupe_key"]: (
                row["task_id"],
                row["claim_finalized_sha256"],
            )
            for row in concurrent["tasks"]
        }
        observed = {
            row["dedupe_key"]: (
                row["task_id"],
                row["claim_finalized_sha256"],
            )
            for row in rows
        }
        if sealed != observed:
            raise RuntimeError(
                "concurrent diagnostic receipt finalized differently"
            )
        return concurrent
    return receipt


def stage(
    *,
    plan_path: Path,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = DEFAULT_STAGING_ACCOUNT,
    resume_incoming: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    authenticate_plan(plan_path)
    return transport.stage_bundle(
        plan_path,
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        staging_account=staging_account,
        resume_incoming=resume_incoming,
        apply=apply,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--plan", type=Path, required=True)
    stage_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    stage_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    stage_parser.add_argument(
        "--staging-account", default=DEFAULT_STAGING_ACCOUNT
    )
    stage_parser.add_argument("--resume-incoming")
    stage_parser.add_argument("--apply", action="store_true")
    auth = commands.add_parser("auth")
    auth.add_argument("--plan", type=Path, required=True)
    auth.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    auth.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    auth.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    auth.add_argument("--staging-account", default=DEFAULT_STAGING_ACCOUNT)
    auth.add_argument("--priority", type=int)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
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
    submit_parser.add_argument("--priority", type=int)
    submit_parser.add_argument("--receipt-out", type=Path)
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
        output = stage(
            plan_path=args.plan,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            resume_incoming=args.resume_incoming,
            apply=args.apply,
        )
    else:
        output = submit(
            plan_path=args.plan,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            priority=args.priority,
            apply=(args.command == "submit" and args.apply),
            receipt_out=(
                args.receipt_out if args.command == "submit" else None
            ),
        )
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
