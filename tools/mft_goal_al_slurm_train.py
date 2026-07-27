"""Fail-closed Slurm training lane for the MFT goal AL contingency.

The scientific repository owns the admitted dataset, exact 25-target training
contract, quality gate, and collected model evidence.  The separate Scheduler
project owns placement and task lifecycle only.  This module therefore builds
one immutable GPFS deployment, submits exactly one eight-CPU training task
through the existing Scheduler API, and collects its generation through the
existing verified SFTP transport.

No command mutates the canonical 6,151-row dataset.  ``plan`` is local-only;
``stage`` and ``submit`` require an explicit ``--apply``; ``collect`` is
read-only with respect to Scheduler and the remote task.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_G0_MODEL_TARGETS,
    canonical_sha256,
)
from tools import mft_goal_20260726_launch as goal_launch  # noqa: E402
from tools import mft_goal_strict_al_ingest as strict_al  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402
from tools import tier1_corrected_generation_adapter as adapter  # noqa: E402


GOAL_PLAN_SCHEMA = "mft-goal-al-slurm-training-plan-v1"
DEPLOYMENT_SCHEMA = "mft-goal-al-slurm-training-deployment-v1"
TASK_SCHEMA = "mft-goal-al-slurm-training-task-v1"
SUBMISSION_SCHEMA = "mft-goal-al-slurm-training-submission-v1"
RESULT_SCHEMA = "mft-goal-al-slurm-training-result-v1"
COLLECTION_SCHEMA = "mft-goal-al-slurm-training-collection-v1"
RESUME_SCHEMA = "mft-goal-al-slurm-training-resume-v1"

CAMPAIGN_ID = "mft-goal-20260726"
DEFAULT_LOCAL_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "al_training_offload"
)
DEFAULT_REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_goal_20260726/al_training"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
DEFAULT_STAGING_ACCOUNT = "harry261"

CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 7_200
MAX_WORKERS_PER_NODE = 1
MODEL_THREADS = 2
TARGET_WORKERS = 4
MAX_MODEL_THREAD_BUDGET = 8
TARGETS = tuple(GOAL_G0_MODEL_TARGETS)
TARGETS_SHA256 = canonical_sha256(list(TARGETS))
PROFILE_RELATIVE = (
    "regression_260707/verify/profiles/goal_standard.json"
)
THRESHOLDS_RELATIVE = (
    "regression_260707/training/model_quality_thresholds.json"
)
TRAIN_RELATIVE = "regression_260707/training/train_models.py"
QUALITY_RELATIVE = (
    "regression_260707/training/model_quality_gate.py"
)
TOOL_RELATIVE = "tools/mft_goal_al_slurm_train.py"
EXTRA_CODE_RELATIVES = (
    TOOL_RELATIVE,
    "tools/slurm_nsga_offload.py",
    "tools/mft_goal_strict_al_ingest.py",
)
RESULT_MAX_BYTES = 4 * 1024 * 1024
SMALL_JSON_MAX_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024 * 1024
MAX_MODEL_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_GENERATION_BYTES = 64 * 1024 * 1024 * 1024
DEFAULT_RETRIES = 3
TRAINING_PACKAGES = {
    **transport.CRITICAL_PACKAGES,
    "filelock": "3.20.3",
}
TRAINING_REQUIREMENTS_LOCK = "".join(
    f"{name}=={version}\n"
    for name, version in TRAINING_PACKAGES.items()
)


class ALTrainingError(RuntimeError):
    """Raised when the AL training boundary cannot authenticate an artifact."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or path.is_symlink()
        ):
            raise OSError("not a regular file")
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ALTrainingError(f"JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise ALTrainingError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    if "payload_sha256" in output:
        raise ALTrainingError("payload is already sealed")
    output["payload_sha256"] = canonical_sha256(output)
    return output


def _validate_seal(
    value: Mapping[str, Any],
    schema: str,
) -> dict[str, Any]:
    output = dict(value)
    observed = output.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or observed != canonical_sha256(output)
    ):
        raise ALTrainingError(f"{schema} payload seal mismatch")
    return dict(value)


def _safe_relative(value: str, label: str) -> str:
    if "\\" in str(value):
        raise ALTrainingError(f"unsafe {label}: {value!r}")
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ALTrainingError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _file_record(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or path.is_symlink()
    ):
        raise ALTrainingError(f"regular file required: {path}")
    output = {
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if relative is not None:
        output["path"] = _safe_relative(relative, "file record path")
    else:
        output["path"] = str(resolved)
    return output


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ALTrainingError(f"{label} is not a SHA-256")
    return digest


def _record_matches(path: Path, record: Mapping[str, Any]) -> bool:
    try:
        size = int(record["size_bytes"])
        digest = _require_sha256(record["sha256"], "file record")
    except (KeyError, TypeError, ValueError, ALTrainingError):
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == size
        and _sha256_file(path) == digest
    )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ALTrainingError(f"{label} is not an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ALTrainingError(f"{label} is not an integer") from exc
    if number != value or number < minimum:
        raise ALTrainingError(f"{label} is outside its allowed range")
    return number


def _validate_dataset_manifest(
    path: Path,
    *,
    expected_code_revision: str | None = None,
) -> tuple[dict[str, Any], Path]:
    manifest_path = path.resolve(strict=True)
    manifest = _validate_seal(
        _read_json(manifest_path),
        strict_al.MANIFEST_SCHEMA,
    )
    repository = manifest.get("repository")
    admission = manifest.get("retraining_admission")
    output = manifest.get("output_dataset")
    next_campaign = manifest.get("next_campaign_contract")
    collection_facts = manifest.get("authenticated_standard_collections")
    thermal_truth = manifest.get("authenticated_thermal_mesh_truth")
    if (
        not isinstance(repository, Mapping)
        or not isinstance(admission, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(next_campaign, Mapping)
        or not isinstance(collection_facts, list)
        or not isinstance(thermal_truth, Mapping)
    ):
        raise ALTrainingError("strict AL dataset manifest is incomplete")
    revision = str(repository.get("revision") or "").lower()
    if expected_code_revision is not None and revision != (
        str(expected_code_revision).lower()
    ):
        raise ALTrainingError(
            "strict AL dataset was not built by the exact training code revision"
        )
    relative = _safe_relative(
        str(output.get("path") or ""),
        "strict AL dataset path",
    )
    if "/" in relative:
        raise ALTrainingError("strict AL dataset must be beside its manifest")
    dataset = (manifest_path.parent / relative).resolve(strict=True)
    try:
        dataset.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ALTrainingError("strict AL dataset escaped its bundle") from exc
    if (
        not dataset.is_file()
        or dataset.is_symlink()
        or int(output.get("size_bytes", -1)) != dataset.stat().st_size
        or _require_sha256(
            output.get("sha256"),
            "strict AL dataset SHA",
        )
        != _sha256_file(dataset)
    ):
        raise ALTrainingError("strict AL dataset bytes drifted")
    if (
        manifest.get("goal_targets") != list(TARGETS)
        or admission.get("allowed") is not True
        or admission.get("reasons") != []
        or int(admission.get("strict_new_rows", -1))
        < strict_al.DEFAULT_MINIMUM_USEFUL_ROWS
        or int(admission.get("unique_complete_geometries", -1))
        < strict_al.DEFAULT_MINIMUM_UNIQUE_GEOMETRIES
        or int(admission.get("unique_source_tasks", -1))
        < strict_al.DEFAULT_MINIMUM_SOURCE_TASKS
        or admission.get("targeted_strata")
        != [strict_al.TARGETED_PRIMARY_TURNS]
        or admission.get("all_25_targets_required_per_new_row") is not True
        or manifest.get("canonical_source_mutated") is not False
        or manifest.get("scheduler_mutation_performed") is not False
        or manifest.get("physics_override_performed") is not False
        or manifest.get("quality_threshold_relaxation_performed") is not False
        or manifest.get("new_model_generation_required") is not True
        or manifest.get("old_generation_result_mixing_allowed") is not False
    ):
        raise ALTrainingError("strict AL retraining admission is not satisfied")
    if (
        len(collection_facts) < strict_al.DEFAULT_MINIMUM_USEFUL_ROWS
        or any(
            not isinstance(fact, Mapping)
            or fact.get("thermal_mesh_policy")
            != strict_al.REQUIRED_THERMAL_MESH_POLICY
            or fact.get("thermal_mesh_plan_contract_version")
            != strict_al.REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
            for fact in collection_facts
        )
        or thermal_truth.get("thermal_mesh_policy")
        != strict_al.REQUIRED_THERMAL_MESH_POLICY
        or thermal_truth.get("thermal_mesh_plan_contract_version")
        != strict_al.REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
        or thermal_truth.get("authenticated_row_count") != len(collection_facts)
        or thermal_truth.get("every_authenticated_row_exact_B7_v8") is not True
    ):
        raise ALTrainingError(
            "strict AL dataset does not contain only exact B7/v8 thermal truth"
        )
    if (
        int(next_campaign.get("seed_start", -1))
        != strict_al.NEXT_CAMPAIGN_SEED_START
        or int(next_campaign.get("seed_end_inclusive", -1))
        != strict_al.NEXT_CAMPAIGN_SEED_END
        or int(next_campaign.get("seed_count", -1))
        != strict_al.NEXT_CAMPAIGN_SEED_COUNT
        or next_campaign.get("all_four_N1_strata_required") is not True
        or next_campaign.get("single_dataset_sha256_required")
        != output["sha256"]
        or next_campaign.get("single_model_generation_required") is not True
        or next_campaign.get("old_generation_result_mixing_allowed") is not False
    ):
        raise ALTrainingError("strict AL next-campaign contract drifted")
    return manifest, dataset


def _add_deployment_file(
    files: dict[str, dict[str, Any]],
    sources: dict[str, str],
    relative: str,
    source: Path,
    kind: str,
) -> None:
    relative = _safe_relative(relative, "deployment path")
    if relative in files:
        raise ALTrainingError(f"duplicate deployment path: {relative}")
    record = _file_record(source)
    files[relative] = {
        "sha256": record["sha256"],
        "size": record["size_bytes"],
        "kind": kind,
    }
    sources[relative] = record["path"]


def _code_sources(code_root: Path) -> dict[str, Path]:
    sources = dict(goal_launch._collect_goal_code_sources(code_root))
    for relative in EXTRA_CODE_RELATIVES:
        path = (code_root / relative).resolve(strict=True)
        sources[f"artifacts/code/{relative}"] = path
    return dict(sorted(sources.items()))


def _validate_deployment(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ALTrainingError("AL training deployment schema mismatch")
    stable = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"bundle_id", "contract_sha256"}
    }
    if value.get("contract_sha256") != canonical_sha256(stable):
        raise ALTrainingError("AL training deployment contract drifted")
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("scheduler_project_source_included") is not False
        or value.get("scheduler_service_modified") is not False
        or (value.get("resources") or {}).get("cpus") != CPUS
        or (value.get("training") or {}).get("targets") != list(TARGETS)
    ):
        raise ALTrainingError("AL training deployment safety contract drifted")
    return dict(value)


def _validate_task(value: Mapping[str, Any]) -> dict[str, Any]:
    task = _validate_seal(value, TASK_SCHEMA)
    resources = task.get("resources")
    training = task.get("training")
    if (
        task.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(resources, Mapping)
        or not isinstance(training, Mapping)
        or resources
        != {
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
            "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1",
        }
        or training.get("targets") != list(TARGETS)
        or training.get("targets_sha256") != TARGETS_SHA256
        or training.get("model_threads") != MODEL_THREADS
        or training.get("target_workers") != TARGET_WORKERS
        or training.get("max_model_thread_budget")
        != MAX_MODEL_THREAD_BUDGET
        or task.get("canonical_dataset_mutation_allowed") is not False
        or task.get("quality_threshold_relaxation_allowed") is not False
        or task.get("automatic_promotion_allowed") is not False
        or task.get("scheduler_project_code_included") is not False
    ):
        raise ALTrainingError("AL training task contract drifted")
    return task


def build_plan(
    *,
    dataset_manifest_path: Path,
    code_root: Path,
    expected_code_revision: str,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    code_root = code_root.resolve(strict=True)
    code_identity = adapter.authenticate_code_root(
        code_root,
        expected_code_revision,
    )
    dataset_manifest, dataset = _validate_dataset_manifest(
        dataset_manifest_path,
        expected_code_revision=code_identity["revision"],
    )
    profile = (code_root / PROFILE_RELATIVE).resolve(strict=True)
    thresholds = (code_root / THRESHOLDS_RELATIVE).resolve(strict=True)
    profile_record = _file_record(profile)
    thresholds_record = _file_record(thresholds)
    profile_canonical = adapter.training_profile_sha256(
        _read_json(profile)
    )
    dataset_record = _file_record(dataset)
    dataset_manifest_record = _file_record(
        dataset_manifest_path.resolve(strict=True)
    )

    science_identity = {
        "campaign_id": CAMPAIGN_ID,
        "code_revision": code_identity["revision"],
        "dataset_manifest_payload_sha256": dataset_manifest["payload_sha256"],
        "dataset_manifest_sha256": dataset_manifest_record["sha256"],
        "dataset_sha256": dataset_record["sha256"],
        "profile_sha256": profile_canonical,
        "quality_thresholds_sha256": thresholds_record["sha256"],
        "targets_sha256": TARGETS_SHA256,
        "training": {
            "model_threads": MODEL_THREADS,
            "target_workers": TARGET_WORKERS,
            "max_model_thread_budget": MAX_MODEL_THREAD_BUDGET,
        },
    }
    deployment_identity = canonical_sha256(science_identity)
    bundle_id = f"mft-goal-al-train-{deployment_identity[:24]}"
    remote_base = str(remote_root).rstrip("/")
    if not remote_base.startswith("/"):
        raise ALTrainingError("remote root must be absolute")
    remote_bundle = f"{remote_base}/{bundle_id}"
    plan_dir = local_root.resolve() / bundle_id
    if plan_dir.exists():
        raise ALTrainingError(f"immutable AL training plan exists: {plan_dir}")
    plan_dir.mkdir(parents=True)

    files: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for relative, source in _code_sources(code_root).items():
        _add_deployment_file(
            files,
            sources,
            relative,
            source,
            "mft_scientific_code",
        )
    marker = plan_dir / ".source-revision"
    marker.write_bytes(f"{code_identity['revision']}\n".encode("ascii"))
    _add_deployment_file(
        files,
        sources,
        "artifacts/code/.source-revision",
        marker,
        "code_revision_marker",
    )
    _add_deployment_file(
        files,
        sources,
        "artifacts/input/strict_al.parquet",
        dataset,
        "admitted_strict_al_dataset",
    )
    _add_deployment_file(
        files,
        sources,
        "artifacts/input/manifest.json",
        dataset_manifest_path.resolve(strict=True),
        "strict_al_dataset_manifest",
    )
    requirements = plan_dir / "requirements.lock"
    requirements.write_bytes(TRAINING_REQUIREMENTS_LOCK.encode("utf-8"))
    _add_deployment_file(
        files,
        sources,
        "artifacts/runtime/requirements.lock",
        requirements,
        "runtime_lock",
    )

    code_inventory = {
        relative: {
            "sha256": record["sha256"],
            "size": record["size"],
        }
        for relative, record in files.items()
        if relative.startswith("artifacts/code/")
    }
    task = _sealed(
        {
            "schema_version": TASK_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "deployment_identity_sha256": deployment_identity,
            "bundle_id": bundle_id,
            "code": {
                "revision": code_identity["revision"],
                "inventory_sha256": canonical_sha256(code_inventory),
                "source_marker_sha256": files[
                    "artifacts/code/.source-revision"
                ]["sha256"],
            },
            "dataset": {
                **dataset_record,
                "path": "artifacts/input/strict_al.parquet",
                "manifest_path": "artifacts/input/manifest.json",
                "manifest_sha256": dataset_manifest_record["sha256"],
                "manifest_payload_sha256": dataset_manifest["payload_sha256"],
                "row_count": int(
                    dataset_manifest["output_dataset"]["row_count"]
                ),
                "source_dataset_generation": (
                    "goal-al-"
                    + dataset_manifest["payload_sha256"][:16]
                ),
            },
            "profile": {
                **profile_record,
                "path": f"artifacts/code/{PROFILE_RELATIVE}",
                "canonical_sha256": profile_canonical,
            },
            "quality_thresholds": {
                **thresholds_record,
                "path": f"artifacts/code/{THRESHOLDS_RELATIVE}",
            },
            "training": {
                "targets": list(TARGETS),
                "targets_sha256": TARGETS_SHA256,
                "model_threads": MODEL_THREADS,
                "target_workers": TARGET_WORKERS,
                "max_model_thread_budget": MAX_MODEL_THREAD_BUDGET,
                "hyperparameter_tuning_performed": False,
                "training_invocation_count": 1,
            },
            "resources": {
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "timeout_seconds": TIMEOUT_SECONDS,
                "max_workers_per_node": MAX_WORKERS_PER_NODE,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
            },
            "next_campaign": copy.deepcopy(
                dataset_manifest["next_campaign_contract"]
            ),
            "fixed_cooling_identity_sha256": (
                FIXED_COOLING_IDENTITY_SHA256
            ),
            "fixed_operating_identity_sha256": (
                FIXED_OPERATING_IDENTITY_SHA256
            ),
            "canonical_dataset_mutation_allowed": False,
            "quality_threshold_relaxation_allowed": False,
            "automatic_promotion_allowed": False,
            "scheduler_project_code_included": False,
        }
    )
    task_path = plan_dir / "task_payload.json"
    _atomic_json(task_path, task)
    _add_deployment_file(
        files,
        sources,
        "artifacts/training/task_payload.json",
        task_path,
        "training_task_payload",
    )

    stable = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "deployment_identity_sha256": deployment_identity,
        "science_identity": science_identity,
        "task_payload_sha256": task["payload_sha256"],
        "code_inventory_sha256": canonical_sha256(code_inventory),
        "files": files,
        "runtime": {
            "python_major_minor": "3.11",
            "critical_packages": dict(TRAINING_PACKAGES),
        },
        "training": {
            "targets": list(TARGETS),
            "targets_sha256": TARGETS_SHA256,
            "single_scheduler_task": True,
            "training_invocation_count": 1,
            "quality_gate_after_training": True,
        },
        "resources": copy.deepcopy(task["resources"]),
        "scheduler_project_source_included": False,
        "scheduler_service_modified": False,
    }
    deployment = {
        **stable,
        "bundle_id": bundle_id,
        "contract_sha256": canonical_sha256(stable),
    }
    deployment_path = plan_dir / "bundle_manifest.json"
    sources_path = plan_dir / "local_sources.json"
    plan_path = plan_dir / "offload_plan.json"
    _atomic_json(deployment_path, deployment)
    _atomic_json(sources_path, sources)
    plan = {
        "schema_version": transport.PLAN_SCHEMA,
        "goal_schema_version": GOAL_PLAN_SCHEMA,
        "created_at": _now(),
        "bundle_id": bundle_id,
        "contract_sha256": deployment["contract_sha256"],
        "bundle_manifest_sha256": _sha256_file(deployment_path),
        "bundle_manifest": str(deployment_path),
        "local_sources": str(sources_path),
        "local_plan_dir": str(plan_dir),
        "remote_root": remote_base,
        "remote_bundle": remote_bundle,
        "runtime_requirements": str(requirements),
        "task_payload": str(task_path),
        "task_payload_sha256": task["payload_sha256"],
        "dataset_manifest": str(dataset_manifest_path.resolve(strict=True)),
        "dataset": str(dataset),
        "profile": str(profile),
        "quality_thresholds": str(thresholds),
        "code_root": str(code_root),
        "expected_code_revision": code_identity["revision"],
        "recommended_resources": copy.deepcopy(task["resources"]),
    }
    _atomic_json(plan_path, plan)
    return plan, deployment, task


def load_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path = path.resolve(strict=True)
    plan = _read_json(plan_path)
    if (
        plan.get("schema_version") != transport.PLAN_SCHEMA
        or plan.get("goal_schema_version") != GOAL_PLAN_SCHEMA
    ):
        raise ALTrainingError("AL training plan schema mismatch")
    deployment_path = Path(plan["bundle_manifest"]).resolve(strict=True)
    deployment = _validate_deployment(_read_json(deployment_path))
    task_path = Path(plan["task_payload"]).resolve(strict=True)
    task = _validate_task(_read_json(task_path))
    if (
        _sha256_file(deployment_path)
        != plan.get("bundle_manifest_sha256")
        or deployment.get("bundle_id") != plan.get("bundle_id")
        or deployment.get("contract_sha256") != plan.get("contract_sha256")
        or deployment.get("task_payload_sha256")
        != task["payload_sha256"]
        or task.get("payload_sha256") != plan.get("task_payload_sha256")
        or task.get("bundle_id") != plan.get("bundle_id")
        or task.get("deployment_identity_sha256")
        != deployment.get("deployment_identity_sha256")
    ):
        raise ALTrainingError("AL training plan identity drifted")
    _validate_dataset_manifest(
        Path(plan["dataset_manifest"]),
        expected_code_revision=plan["expected_code_revision"],
    )
    adapter.authenticate_code_root(
        Path(plan["code_root"]),
        plan["expected_code_revision"],
    )
    return plan, deployment, task


def scheduler_payload(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    priority: int = 100,
) -> dict[str, Any]:
    task = _validate_task(task)
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or priority < 0
    ):
        raise ALTrainingError(
            "Scheduler priority must be a non-negative integer"
        )
    if (
        task.get("bundle_id") != plan.get("bundle_id")
        or task.get("payload_sha256") != plan.get("task_payload_sha256")
    ):
        raise ALTrainingError("Scheduler payload task/plan identity mismatch")
    command = "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site:'
            '$PWD/artifacts/code:${PYTHONPATH:-}"',
            f"export OMP_NUM_THREADS={MODEL_THREADS}",
            f"export OPENBLAS_NUM_THREADS={MODEL_THREADS}",
            f"export MKL_NUM_THREADS={MODEL_THREADS}",
            f"export NUMEXPR_NUM_THREADS={MODEL_THREADS}",
            f"export VECLIB_MAXIMUM_THREADS={MODEL_THREADS}",
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:'
            '?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) '
            'payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path" >&2; exit 66 ;; esac',
            'output="runs/task-${SLURM_SCHED_TASK_ID:'
            '?missing Scheduler task id}"',
            'test ! -e "$output"',
            "exec python artifacts/code/"
            f"{TOOL_RELATIVE} execute "
            '--payload "$payload_path" --output "$output"',
        ]
    )
    dedupe = canonical_sha256(
        {
            "bundle_id": plan["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
        }
    )
    return {
        "name": f"mft-goal-al-train-{task['payload_sha256'][:12]}",
        "remote_cwd": plan["remote_bundle"],
        "command": command,
        "payload_json": dict(task),
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": priority,
        "timeout_seconds": TIMEOUT_SECONDS,
        "dedupe_key": f"mft-goal-20260726-al-train:{dedupe}",
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
    }


def stage(
    *,
    plan_path: Path,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = DEFAULT_STAGING_ACCOUNT,
    resume_incoming: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    load_plan(plan_path)
    return transport.stage_bundle(
        plan_path,
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        staging_account=staging_account,
        resume_incoming=resume_incoming,
        apply=apply,
    )


def _submission_path(plan: Mapping[str, Any]) -> Path:
    return Path(plan["local_plan_dir"]) / "submission.json"


def submit(
    *,
    plan_path: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = DEFAULT_STAGING_ACCOUNT,
    priority: int = 100,
    apply: bool = False,
) -> dict[str, Any]:
    plan, _deployment, task = load_plan(plan_path)
    payload = scheduler_payload(
        plan=plan,
        task=task,
        priority=priority,
    )
    output = {
        "schema_version": SUBMISSION_SCHEMA,
        "apply": bool(apply),
        "bundle_id": plan["bundle_id"],
        "task_payload_sha256": task["payload_sha256"],
        "scheduler_url": scheduler_url.rstrip("/"),
        "scheduler_request": payload,
        "scheduler_request_sha256": canonical_sha256(payload),
        "submission": None,
    }
    if not apply:
        return output
    submission_path = _submission_path(plan)
    if submission_path.exists():
        raise ALTrainingError(
            "exactly-one training submission ledger already exists"
        )
    account, ssh_session = transport._account(
        accounts_path,
        scheduler_source,
        staging_account,
    )
    with ssh_session(account, default_timeout=60) as session:
        if not transport._remote_ready(
            session,
            plan["remote_bundle"],
            plan["bundle_manifest_sha256"],
        ):
            raise ALTrainingError("remote AL training deployment is not ready")
    response = transport._api_json(
        scheduler_url.rstrip("/") + "/api/tasks",
        method="POST",
        payload=payload,
        timeout=30,
    )
    task_id = int(response.get("task_id", -1))
    if task_id <= 0:
        raise ALTrainingError("Scheduler did not return a valid task ID")
    submission = _sealed(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "scheduler_url": scheduler_url.rstrip("/"),
            "scheduler_request_sha256": canonical_sha256(payload),
            "priority": int(priority),
            "task_id": task_id,
            "deduped": bool(response.get("deduped")),
            "submitted_at": _now(),
            "single_training_task": True,
            "scheduler_project_modified": False,
        }
    )
    _atomic_json(submission_path, submission)
    output["submission"] = submission
    output["submission_path"] = str(submission_path)
    return output


def _load_submission(
    path: Path,
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    submission = _validate_seal(
        _read_json(path.resolve(strict=True)),
        SUBMISSION_SCHEMA,
    )
    expected_payload = scheduler_payload(
        plan=plan,
        task=task,
        priority=int(submission.get("priority", -1)),
    )
    if (
        submission.get("bundle_id") != plan["bundle_id"]
        or submission.get("task_payload_sha256")
        != task["payload_sha256"]
        or not str(submission.get("scheduler_url") or "").startswith("http")
        or not isinstance(submission.get("task_id"), int)
        or int(submission["task_id"]) <= 0
        or isinstance(submission.get("priority"), bool)
        or not isinstance(submission.get("priority"), int)
        or submission.get("single_training_task") is not True
        or submission.get("scheduler_project_modified") is not False
        or submission.get("scheduler_request_sha256")
        != canonical_sha256(expected_payload)
    ):
        raise ALTrainingError("AL training submission identity drifted")
    return submission


def _positive_env(name: str) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        raise ALTrainingError(f"{name} is absent or invalid")
    return int(raw)


def _resolve_deployment_file(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    safe = _safe_relative(relative, "runtime artifact")
    path = (root / PurePosixPath(safe)).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ALTrainingError("runtime artifact escaped deployment") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(expected_size)
        or _sha256_file(path)
        != _require_sha256(expected_sha256, "runtime artifact SHA")
    ):
        raise ALTrainingError(f"runtime artifact identity drifted: {relative}")
    return path


def _run_logged(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(completed.returncode)


def execute(
    *,
    payload_path: Path,
    output: Path,
) -> Path:
    task = _validate_task(_read_json(payload_path.resolve(strict=True)))
    if _positive_env("SLURM_CPUS_PER_TASK") != CPUS:
        raise ALTrainingError("Slurm CPU task contract is not exactly eight")
    if _positive_env("SLURM_SCHEDULER_TASK_CPUS") != CPUS:
        raise ALTrainingError(
            "Scheduler task CPU contract is not exactly eight"
        )
    scheduler_task_id = _positive_env("SLURM_SCHED_TASK_ID")
    deployment_root = Path.cwd().resolve(strict=True)
    if deployment_root.name != task["bundle_id"]:
        raise ALTrainingError("runtime deployment/bundle identity mismatch")
    staged_task = _validate_task(
        _read_json(
            deployment_root
            / "artifacts"
            / "training"
            / "task_payload.json"
        )
    )
    if staged_task != task:
        raise ALTrainingError(
            "Scheduler payload differs from immutable staged task"
        )
    deployment = _validate_deployment(
        _read_json(deployment_root / "bundle_manifest.json")
    )
    if (
        deployment.get("bundle_id") != task["bundle_id"]
        or deployment.get("task_payload_sha256")
        != task["payload_sha256"]
        or deployment.get("deployment_identity_sha256")
        != task["deployment_identity_sha256"]
    ):
        raise ALTrainingError("runtime deployment/task binding drifted")
    code_root = deployment_root / "artifacts" / "code"

    def deployment_file(relative: str) -> Path:
        record = (deployment.get("files") or {}).get(relative)
        if not isinstance(record, Mapping):
            raise ALTrainingError(
                f"runtime deployment file is unsealed: {relative}"
            )
        return _resolve_deployment_file(
            deployment_root,
            relative,
            expected_sha256=str(record.get("sha256") or ""),
            expected_size=int(record.get("size", -1)),
        )

    for critical in (
        f"artifacts/code/{TOOL_RELATIVE}",
        f"artifacts/code/{TRAIN_RELATIVE}",
        f"artifacts/code/{QUALITY_RELATIVE}",
        "artifacts/training/task_payload.json",
    ):
        deployment_file(critical)
    marker = _resolve_deployment_file(
        deployment_root,
        "artifacts/code/.source-revision",
        expected_sha256=task["code"]["source_marker_sha256"],
        expected_size=41,
    )
    if marker.read_bytes() != (
        f"{task['code']['revision']}\n".encode("ascii")
    ):
        raise ALTrainingError("runtime code revision marker drifted")
    dataset = _resolve_deployment_file(
        deployment_root,
        task["dataset"]["path"],
        expected_sha256=task["dataset"]["sha256"],
        expected_size=int(task["dataset"]["size_bytes"]),
    )
    dataset_manifest_path = _resolve_deployment_file(
        deployment_root,
        task["dataset"]["manifest_path"],
        expected_sha256=task["dataset"]["manifest_sha256"],
        expected_size=int(
            deployment["files"][
                task["dataset"]["manifest_path"]
            ]["size"]
        ),
    )
    manifest, manifest_dataset = _validate_dataset_manifest(
        dataset_manifest_path,
        expected_code_revision=task["code"]["revision"],
    )
    if (
        manifest["payload_sha256"]
        != task["dataset"]["manifest_payload_sha256"]
        or manifest_dataset != dataset
    ):
        raise ALTrainingError("runtime strict AL dataset binding drifted")
    profile = _resolve_deployment_file(
        deployment_root,
        task["profile"]["path"],
        expected_sha256=task["profile"]["sha256"],
        expected_size=int(task["profile"]["size_bytes"]),
    )
    if (
        adapter.training_profile_sha256(_read_json(profile))
        != task["profile"]["canonical_sha256"]
    ):
        raise ALTrainingError("runtime training profile canonical SHA drifted")
    thresholds = _resolve_deployment_file(
        deployment_root,
        task["quality_thresholds"]["path"],
        expected_sha256=task["quality_thresholds"]["sha256"],
        expected_size=int(task["quality_thresholds"]["size_bytes"]),
    )
    train_script = deployment_file(f"artifacts/code/{TRAIN_RELATIVE}")
    quality_script = deployment_file(
        f"artifacts/code/{QUALITY_RELATIVE}"
    )

    output = output.resolve()
    if output.exists():
        raise ALTrainingError(f"immutable training output exists: {output}")
    output.mkdir(parents=True)
    registry = output / "registry"
    candidate_path = output / "candidate.json"
    quality_path = output / "quality_status.json"
    train_stdout = output / "train_stdout.log"
    train_stderr = output / "train_stderr.log"
    quality_stdout = output / "quality_stdout.log"
    quality_stderr = output / "quality_stderr.log"
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[key] = str(MODEL_THREADS)
    train_command = [
        sys.executable,
        str(train_script),
        "--dataset",
        str(dataset),
        "--source-dataset-path",
        str(dataset),
        "--source-dataset-generation",
        task["dataset"]["source_dataset_generation"],
        "--targets",
        *TARGETS,
        "--registry",
        str(registry),
        "--profile",
        str(profile),
        "--model-threads",
        str(MODEL_THREADS),
        "--target-workers",
        str(TARGET_WORKERS),
        "--max-model-thread-budget",
        str(MAX_MODEL_THREAD_BUDGET),
        "--result-json",
        str(candidate_path),
    ]
    train_exit = _run_logged(
        train_command,
        stdout_path=train_stdout,
        stderr_path=train_stderr,
        cwd=code_root,
        environment=environment,
    )
    if train_exit != 0:
        raise ALTrainingError(
            f"single 25-target training invocation failed: {train_exit}"
        )
    candidate = _read_json(candidate_path)
    generation = Path(str(candidate.get("generation_path") or "")).resolve(
        strict=True
    )
    quality_command = [
        sys.executable,
        str(quality_script),
        "--registry",
        str(registry),
        "--generation",
        str(generation),
        "--dataset",
        str(dataset),
        "--thresholds",
        str(thresholds),
        "--status",
        str(quality_path),
    ]
    quality_exit = _run_logged(
        quality_command,
        stdout_path=quality_stdout,
        stderr_path=quality_stderr,
        cwd=code_root,
        environment=environment,
    )
    if quality_exit not in {0, 2}:
        raise ALTrainingError(
            f"quality gate did not complete: {quality_exit}"
        )
    authenticated = adapter.authenticate_corrected_generation(
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
        goal_campaign=True,
    )
    if authenticated.report.get("targets") != list(TARGETS):
        raise ALTrainingError("generated target inventory drifted")
    generation_relative = generation.relative_to(output).as_posix()
    artifact_inventory: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(
        authenticated.report["artifacts"].items()
    ):
        safe = _safe_relative(relative, "trained model artifact")
        artifact = (generation / safe).resolve(strict=True)
        try:
            artifact.relative_to(generation)
        except ValueError as exc:
            raise ALTrainingError("trained model artifact escaped") from exc
        record = _file_record(
            artifact,
            relative=f"{generation_relative}/{safe}",
        )
        if record["sha256"] != expected:
            raise ALTrainingError(
                f"trained model artifact SHA drifted: {relative}"
            )
        artifact_inventory[record["path"]] = record

    named_files = {
        "candidate": candidate_path,
        "quality_status": quality_path,
        "train_report": authenticated.report_path,
        "train_stdout": train_stdout,
        "train_stderr": train_stderr,
        "quality_stdout": quality_stdout,
        "quality_stderr": quality_stderr,
    }
    output_inventory = {
        role: _file_record(
            path,
            relative=path.relative_to(output).as_posix(),
        )
        for role, path in named_files.items()
    }
    result = _sealed(
        {
            "schema_version": RESULT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": task["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "scheduler_task_id": scheduler_task_id,
            "completed_at": _now(),
            "training_invocation_count": 1,
            "training_exit_code": train_exit,
            "quality_exit_code": quality_exit,
            "quality_passed": authenticated.quality["passed"],
            "search_only_proposal": not authenticated.quality["passed"],
            "generation_relative": generation_relative,
            "documentary_generation_path": str(generation),
            "dataset_sha256": task["dataset"]["sha256"],
            "dataset_manifest_payload_sha256": task["dataset"][
                "manifest_payload_sha256"
            ],
            "code_revision": task["code"]["revision"],
            "targets": list(TARGETS),
            "targets_sha256": TARGETS_SHA256,
            "output_inventory": output_inventory,
            "generation_artifacts": artifact_inventory,
            "slurm_cpu_contract": {
                "SLURM_CPUS_PER_TASK": CPUS,
                "SLURM_SCHEDULER_TASK_CPUS": CPUS,
            },
            "canonical_dataset_mutated": False,
            "quality_threshold_relaxation_performed": False,
            "scheduler_mutation_performed": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    result_path = output / "result.json"
    _atomic_json(result_path, result)
    return result_path


def _validate_result(
    value: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    scheduler_task_id: int,
) -> dict[str, Any]:
    result = _validate_seal(value, RESULT_SCHEMA)
    output_inventory = result.get("output_inventory")
    artifacts = result.get("generation_artifacts")
    required_roles = {
        "candidate",
        "quality_status",
        "train_report",
        "train_stdout",
        "train_stderr",
        "quality_stdout",
        "quality_stderr",
    }
    if (
        result.get("campaign_id") != CAMPAIGN_ID
        or result.get("bundle_id") != task["bundle_id"]
        or result.get("task_payload_sha256") != task["payload_sha256"]
        or result.get("scheduler_task_id") != int(scheduler_task_id)
        or result.get("training_invocation_count") != 1
        or result.get("training_exit_code") != 0
        or result.get("quality_exit_code") not in {0, 2}
        or not isinstance(result.get("quality_passed"), bool)
        or (
            result.get("quality_exit_code") == 0
            and result.get("quality_passed") is not True
        )
        or (
            result.get("quality_exit_code") == 2
            and result.get("quality_passed") is not False
        )
        or result.get("search_only_proposal")
        is not (not result.get("quality_passed"))
        or result.get("dataset_sha256") != task["dataset"]["sha256"]
        or result.get("dataset_manifest_payload_sha256")
        != task["dataset"]["manifest_payload_sha256"]
        or result.get("code_revision") != task["code"]["revision"]
        or result.get("targets") != list(TARGETS)
        or result.get("targets_sha256") != TARGETS_SHA256
        or result.get("slurm_cpu_contract")
        != {
            "SLURM_CPUS_PER_TASK": CPUS,
            "SLURM_SCHEDULER_TASK_CPUS": CPUS,
        }
        or result.get("canonical_dataset_mutated") is not False
        or result.get("quality_threshold_relaxation_performed") is not False
        or result.get("scheduler_mutation_performed") is not False
        or result.get("production_eligible") is not False
        or result.get("automatic_promotion_allowed") is not False
        or not isinstance(output_inventory, Mapping)
        or set(output_inventory) != required_roles
        or not isinstance(artifacts, Mapping)
        or len(artifacts) != len(TARGETS) * 2
    ):
        raise ALTrainingError("AL training result contract drifted")
    generation_relative = _safe_relative(
        str(result.get("generation_relative") or ""),
        "generation relative path",
    )
    if (
        not generation_relative.startswith("registry/generations/")
        or len(PurePosixPath(generation_relative).parts) != 3
        or not str(result.get("documentary_generation_path") or "").startswith(
            "/"
        )
    ):
        raise ALTrainingError("AL training generation identity is invalid")
    seen: set[str] = set()
    total_bytes = 0
    for label, inventory in (
        ("output inventory", output_inventory),
        ("generation artifacts", artifacts),
    ):
        for key, raw in inventory.items():
            if not isinstance(raw, Mapping):
                raise ALTrainingError(f"{label} record is malformed")
            path = _safe_relative(
                str(raw.get("path") or ""),
                f"{label} path",
            )
            if path in seen or (
                label == "generation artifacts"
                and not path.startswith(generation_relative + "/")
            ):
                raise ALTrainingError(f"{label} path identity drifted")
            seen.add(path)
            _require_sha256(raw.get("sha256"), f"{label} SHA")
            size = raw.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ALTrainingError(f"{label} size is invalid")
            if label == "generation artifacts" and (
                size > MAX_MODEL_ARTIFACT_BYTES
            ):
                raise ALTrainingError("trained model artifact exceeds safety cap")
            if (
                label == "output inventory"
                and str(key).endswith(("stdout", "stderr"))
                and size > MAX_LOG_BYTES
            ):
                raise ALTrainingError("training log exceeds safety cap")
            if (
                label == "output inventory"
                and not str(key).endswith(("stdout", "stderr"))
                and size > SMALL_JSON_MAX_BYTES
            ):
                raise ALTrainingError("training JSON exceeds safety cap")
            total_bytes += size
            if key != path and label == "generation artifacts":
                raise ALTrainingError(
                    "generation artifact key/path identity drifted"
                )
    if total_bytes > MAX_GENERATION_BYTES:
        raise ALTrainingError("training generation exceeds collection safety cap")
    for role in ("candidate", "quality_status", "train_report"):
        path = str(output_inventory[role]["path"])
        if role == "train_report":
            expected = generation_relative + "/train_report.json"
        else:
            expected = f"{role}.json"
        if path != expected:
            raise ALTrainingError(f"{role} output path drifted")
    return result


def _validated_scheduler_task(
    scheduler_task: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    expected = scheduler_payload(
        plan=plan,
        task=task,
        priority=int(submission["priority"]),
    )
    if scheduler_task.get("status") != "completed":
        raise ALTrainingError(
            "Scheduler task is not the completed exact training task"
        )
    task_id = _integer(
        submission.get("task_id"),
        "submitted Scheduler task ID",
        minimum=1,
    )
    scheduler_id = _integer(
        scheduler_task.get(
            "task_id",
            scheduler_task.get("id"),
        ),
        "observed Scheduler task ID",
        minimum=1,
    )
    exit_code = _integer(
        scheduler_task.get("exit_code"),
        "Scheduler exit code",
    )
    if (
        scheduler_id != task_id
        or exit_code != 0
        or scheduler_task.get("name") != expected["name"]
        or scheduler_task.get("remote_cwd") != expected["remote_cwd"]
        or _integer(scheduler_task.get("cpus"), "Scheduler CPUs") != CPUS
        or _integer(
            scheduler_task.get("memory_mb"),
            "Scheduler memory",
        )
        != MEMORY_MB
        or scheduler_task.get("scheduling_profile") != "standard"
        or scheduler_task.get("aedt_backend") != "standalone"
        or _integer(scheduler_task.get("gpus"), "Scheduler GPUs") != 0
        or _integer(
            scheduler_task.get("priority"),
            "Scheduler priority",
        )
        != int(submission["priority"])
        or _integer(
            scheduler_task.get("timeout_seconds"),
            "Scheduler timeout",
        )
        != TIMEOUT_SECONDS
        or scheduler_task.get("dedupe_key") != expected["dedupe_key"]
        or _integer(
            scheduler_task.get("max_workers_per_node"),
            "Scheduler per-node worker cap",
        )
        != MAX_WORKERS_PER_NODE
        or scheduler_task.get("required_capability")
        != "conda:pyaedt2026v1"
        or scheduler_task.get("env_profile") != "pyaedt2026v1"
        or not str(scheduler_task.get("account_name") or "").strip()
    ):
        raise ALTrainingError(
            "Scheduler task is not the completed exact training task"
        )
    return dict(scheduler_task)


def _download_expected(
    connection: transport._PersistentAccountConnection,
    *,
    remote: str,
    destination: Path,
    expected: Mapping[str, Any],
    retries: int,
) -> dict[str, Any]:
    identity, identity_attempts = transport._retry_remote_identity(
        connection,
        remote,
        retries,
    )
    expected_identity = {
        "bytes": int(expected["size_bytes"]),
        "sha256": _require_sha256(
            expected["sha256"],
            "expected remote artifact SHA",
        ),
    }
    if identity != expected_identity:
        raise ALTrainingError(
            f"remote artifact identity differs from result: {remote}"
        )
    if _record_matches(destination, expected):
        return {
            "path": str(destination),
            "bytes": int(identity["bytes"]),
            "sha256": identity["sha256"],
            "remote_sha256": identity["sha256"],
            "remote_bytes": int(identity["bytes"]),
            "verified": True,
            "transport": "verified_sftp_cached",
            "identity_attempts": identity_attempts,
        }
    record = transport._download_verified_sftp(
        connection,
        remote,
        destination,
        retries,
        expected_identity=identity,
    )
    record["path"] = str(destination)
    record["identity_attempts"] = identity_attempts
    return record


def _copy_verified(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    source_record = _file_record(source)
    if not _record_matches(destination, source_record):
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        try:
            shutil.copy2(source, staged)
            if not _record_matches(staged, source_record):
                raise ALTrainingError(
                    f"local evidence copy drifted: {source}"
                )
            os.replace(staged, destination)
        finally:
            if staged.exists():
                staged.unlink()
    return {
        "path": str(destination),
        "sha256": source_record["sha256"],
        "size_bytes": source_record["size_bytes"],
    }


def _collection_file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"collection_manifest.json", ".resume.json"}:
            continue
        records[relative] = _file_record(path, relative=relative)
    return records


def validate_collection(path: Path) -> dict[str, Any]:
    manifest_path = path.resolve(strict=True)
    collection = _validate_seal(
        _read_json(manifest_path),
        COLLECTION_SCHEMA,
    )
    root = manifest_path.parent
    if (
        manifest_path.name != "collection_manifest.json"
        or collection.get("campaign_id") != CAMPAIGN_ID
        or collection.get("scheduler_get_only_collection") is not True
        or collection.get("scheduler_mutation_performed") is not False
        or collection.get("canonical_dataset_mutated") is not False
        or collection.get("quality_threshold_relaxation_performed") is not False
        or collection.get("production_eligible") is not False
        or collection.get("automatic_promotion_allowed") is not False
    ):
        raise ALTrainingError("AL training collection safety flags drifted")
    expected_files = collection.get("files")
    if not isinstance(expected_files, Mapping):
        raise ALTrainingError("AL training collection inventory is absent")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"collection_manifest.json", ".resume.json"}
    }
    if set(expected_files) != actual_paths:
        raise ALTrainingError("AL training collection file set drifted")
    for relative, raw in expected_files.items():
        safe = _safe_relative(relative, "collection file")
        if not isinstance(raw, Mapping) or raw.get("path") != safe:
            raise ALTrainingError("AL training collection record is malformed")
        target = (root / safe).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ALTrainingError("collection file escaped") from exc
        if not _record_matches(target, raw):
            raise ALTrainingError(f"collection file bytes drifted: {safe}")

    deployment = _validate_deployment(
        _read_json(root / "evidence" / "bundle_manifest.json")
    )
    task = _validate_task(
        _read_json(root / "evidence" / "task_payload.json")
    )
    submission = _validate_seal(
        _read_json(root / "evidence" / "submission.json"),
        SUBMISSION_SCHEMA,
    )
    scheduler_task = _read_json(
        root / "evidence" / "scheduler_task.json"
    )
    if (
        task["bundle_id"] != deployment["bundle_id"]
        or task["payload_sha256"]
        != deployment["task_payload_sha256"]
        or submission.get("bundle_id") != task["bundle_id"]
        or submission.get("task_payload_sha256")
        != task["payload_sha256"]
        or int(submission.get("task_id", -1))
        != int(scheduler_task.get("task_id", scheduler_task.get("id", -2)))
        or scheduler_task.get("status") != "completed"
        or int(scheduler_task.get("exit_code", -1)) != 0
    ):
        raise ALTrainingError("collection task/deployment evidence drifted")
    inferred_plan = {
        "bundle_id": task["bundle_id"],
        "task_payload_sha256": task["payload_sha256"],
        "remote_bundle": scheduler_task.get("remote_cwd"),
    }
    expected_request = scheduler_payload(
        plan=inferred_plan,
        task=task,
        priority=_integer(
            submission.get("priority"),
            "submission priority",
        ),
    )
    if (
        submission.get("scheduler_request_sha256")
        != canonical_sha256(expected_request)
        or submission.get("single_training_task") is not True
        or submission.get("scheduler_project_modified") is not False
    ):
        raise ALTrainingError("collection Scheduler request evidence drifted")
    _validated_scheduler_task(
        scheduler_task,
        plan=inferred_plan,
        task=task,
        submission=submission,
    )
    result = _validate_result(
        _read_json(root / "result.json"),
        task=task,
        scheduler_task_id=int(submission["task_id"]),
    )
    dataset_manifest, dataset = _validate_dataset_manifest(
        root / "inputs" / "manifest.json",
        expected_code_revision=task["code"]["revision"],
    )
    profile = root / "inputs" / "goal_standard.json"
    thresholds = root / "inputs" / "model_quality_thresholds.json"
    if (
        dataset != (root / "inputs" / "strict_al.parquet").resolve(strict=True)
        or dataset_manifest["payload_sha256"]
        != task["dataset"]["manifest_payload_sha256"]
        or _sha256_file(dataset) != task["dataset"]["sha256"]
        or adapter.training_profile_sha256(_read_json(profile))
        != task["profile"]["canonical_sha256"]
        or _sha256_file(thresholds)
        != task["quality_thresholds"]["sha256"]
    ):
        raise ALTrainingError("collection input identity drifted")
    generation = (root / result["generation_relative"]).resolve(strict=True)
    authenticated = adapter.authenticate_corrected_generation(
        generation=generation,
        candidate_path=root / "candidate.json",
        quality_path=root / "quality_status.json",
        goal_campaign=True,
        dataset_path_override=dataset,
        profile_path_override=profile,
        expected_documentary_generation_path=result[
            "documentary_generation_path"
        ],
    )
    if (
        authenticated.report.get("targets") != list(TARGETS)
        or authenticated.quality.get("passed")
        is not result["quality_passed"]
        or authenticated.quality.get("quality_thresholds_sha256")
        != task["quality_thresholds"]["sha256"]
        or authenticated.evidence["train_report"]["sha256"]
        != result["output_inventory"]["train_report"]["sha256"]
        or authenticated.evidence["candidate"]["sha256"]
        != result["output_inventory"]["candidate"]["sha256"]
        or authenticated.evidence["quality_status"]["sha256"]
        != result["output_inventory"]["quality_status"]["sha256"]
    ):
        raise ALTrainingError("collected generation authentication drifted")
    return collection


def collect(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    retries: int = DEFAULT_RETRIES,
) -> Path:
    if not 1 <= int(retries) <= 10:
        raise ALTrainingError("collection retries must be from 1 through 10")
    plan, deployment, task = load_plan(plan_path)
    submission = _load_submission(
        submission_path,
        plan=plan,
        task=task,
    )
    if scheduler_url.rstrip("/") != submission["scheduler_url"]:
        raise ALTrainingError("collection Scheduler URL differs from submission")
    scheduler_task = transport._api_json_with_retry(
        scheduler_url.rstrip("/")
        + f"/api/tasks/{int(submission['task_id'])}",
        int(retries),
    )
    scheduler_task = _validated_scheduler_task(
        scheduler_task,
        plan=plan,
        task=task,
        submission=submission,
    )
    output = output.resolve()
    if output.exists():
        raise ALTrainingError(f"immutable collection exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = output.with_name(f".{output.name}.incoming")
    resume = _sealed(
        {
            "schema_version": RESUME_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "task_id": int(submission["task_id"]),
            "remote_output": (
                plan["remote_bundle"]
                + f"/runs/task-{int(submission['task_id'])}"
            ),
        }
    )
    resume_path = incoming / ".resume.json"
    if incoming.exists():
        if (
            not incoming.is_dir()
            or not resume_path.is_file()
            or _validate_seal(_read_json(resume_path), RESUME_SCHEMA)
            != resume
        ):
            raise ALTrainingError(
                "incomplete collection belongs to another task"
            )
    else:
        incoming.mkdir()
        _atomic_json(resume_path, resume)
    remote_output = resume["remote_output"]
    account, ssh_session = transport._account(
        accounts_path,
        scheduler_source,
        str(scheduler_task["account_name"]),
    )
    connection = transport._PersistentAccountConnection(
        account,
        ssh_session,
    )
    transfers: dict[str, dict[str, Any]] = {}
    try:
        result_remote = remote_output + "/result.json"
        result_local = incoming / "result.json"
        result_transfer = transport._download_verified_sftp(
            connection,
            result_remote,
            result_local,
            int(retries),
            max_bytes=RESULT_MAX_BYTES,
        )
        result = _validate_result(
            _read_json(result_local),
            task=task,
            scheduler_task_id=int(submission["task_id"]),
        )
        transfers["result.json"] = result_transfer
        records = {
            str(record["path"]): record
            for record in result["output_inventory"].values()
        }
        records.update(result["generation_artifacts"])
        for relative, expected in sorted(records.items()):
            safe = _safe_relative(relative, "remote training output")
            transfer = _download_expected(
                connection,
                remote=remote_output + "/" + safe,
                destination=incoming / safe,
                expected=expected,
                retries=int(retries),
            )
            transfers[safe] = transfer
    finally:
        connection.close()

    input_sources = {
        "inputs/manifest.json": Path(plan["dataset_manifest"]),
        "inputs/strict_al.parquet": Path(plan["dataset"]),
        "inputs/goal_standard.json": Path(plan["profile"]),
        "inputs/model_quality_thresholds.json": Path(
            plan["quality_thresholds"]
        ),
        "evidence/bundle_manifest.json": Path(plan["bundle_manifest"]),
        "evidence/task_payload.json": Path(plan["task_payload"]),
        "evidence/submission.json": submission_path,
    }
    for relative, source in input_sources.items():
        _copy_verified(source.resolve(strict=True), incoming / relative)
    _atomic_json(
        incoming / "evidence" / "scheduler_task.json",
        scheduler_task,
    )
    result = _validate_result(
        _read_json(incoming / "result.json"),
        task=task,
        scheduler_task_id=int(submission["task_id"]),
    )
    generation = (incoming / result["generation_relative"]).resolve(
        strict=True
    )
    authenticated = adapter.authenticate_corrected_generation(
        generation=generation,
        candidate_path=incoming / "candidate.json",
        quality_path=incoming / "quality_status.json",
        goal_campaign=True,
        dataset_path_override=incoming / "inputs" / "strict_al.parquet",
        profile_path_override=incoming / "inputs" / "goal_standard.json",
        expected_documentary_generation_path=result[
            "documentary_generation_path"
        ],
    )
    expected_paths = {
        "result.json",
        *records,
        *input_sources,
        "evidence/scheduler_task.json",
        ".resume.json",
    }
    actual_paths = {
        path.relative_to(incoming).as_posix()
        for path in incoming.rglob("*")
        if path.is_file()
        and path.relative_to(incoming).as_posix()
        != "collection_manifest.json"
    }
    if actual_paths != expected_paths:
        raise ALTrainingError(
            "incomplete collection contains an unexpected file set"
        )
    files = _collection_file_inventory(incoming)
    collection = _sealed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at": _now(),
            "bundle_id": plan["bundle_id"],
            "deployment_contract_sha256": deployment["contract_sha256"],
            "task_payload_sha256": task["payload_sha256"],
            "task_id": int(submission["task_id"]),
            "scheduler_url": submission["scheduler_url"],
            "scheduler_status": scheduler_task["status"],
            "scheduler_exit_code": int(scheduler_task["exit_code"]),
            "quality_passed": authenticated.quality["passed"],
            "search_only_proposal": not authenticated.quality["passed"],
            "generation_relative": result["generation_relative"],
            "documentary_generation_path": result[
                "documentary_generation_path"
            ],
            "dataset_manifest_payload_sha256": task["dataset"][
                "manifest_payload_sha256"
            ],
            "dataset_sha256": task["dataset"]["sha256"],
            "code_revision": task["code"]["revision"],
            "targets": list(TARGETS),
            "targets_sha256": TARGETS_SHA256,
            "file_count": len(files),
            "files": files,
            "transfer": {
                "transport": "existing_verified_sftp",
                "download_count": len(transfers),
                "connection_count": connection.connection_count,
                "all_remote_pre_post_and_local_sha_verified": True,
            },
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "canonical_dataset_mutated": False,
            "quality_threshold_relaxation_performed": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    manifest_path = incoming / "collection_manifest.json"
    _atomic_json(manifest_path, collection)
    validate_collection(manifest_path)
    resume_path.unlink()
    os.replace(incoming, output)
    return output / "collection_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, stage, submit, and collect one immutable eight-CPU "
            "25-target MFT goal active-learning training task."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--dataset-manifest", type=Path, required=True)
    plan.add_argument("--code-root", type=Path, required=True)
    plan.add_argument("--expected-code-revision", required=True)
    plan.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    plan.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)

    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--plan", type=Path, required=True)
    stage_parser.add_argument(
        "--accounts",
        type=Path,
        default=DEFAULT_ACCOUNTS,
    )
    stage_parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=DEFAULT_SCHEDULER_SOURCE,
    )
    stage_parser.add_argument(
        "--staging-account",
        default=DEFAULT_STAGING_ACCOUNT,
    )
    stage_parser.add_argument("--resume-incoming")
    stage_parser.add_argument("--apply", action="store_true")

    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--scheduler-url",
        default=DEFAULT_SCHEDULER_URL,
    )
    submit_parser.add_argument(
        "--accounts",
        type=Path,
        default=DEFAULT_ACCOUNTS,
    )
    submit_parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=DEFAULT_SCHEDULER_SOURCE,
    )
    submit_parser.add_argument(
        "--staging-account",
        default=DEFAULT_STAGING_ACCOUNT,
    )
    submit_parser.add_argument("--priority", type=int, default=100)
    submit_parser.add_argument("--apply", action="store_true")

    execute_parser = commands.add_parser(
        "execute",
        help="Scheduler-only worker entry point.",
    )
    execute_parser.add_argument("--payload", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--submission", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument(
        "--scheduler-url",
        default=DEFAULT_SCHEDULER_URL,
    )
    collect_parser.add_argument(
        "--accounts",
        type=Path,
        default=DEFAULT_ACCOUNTS,
    )
    collect_parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=DEFAULT_SCHEDULER_SOURCE,
    )
    collect_parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    validate = commands.add_parser("validate-collection")
    validate.add_argument("--collection", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan, deployment, task = build_plan(
            dataset_manifest_path=args.dataset_manifest,
            code_root=args.code_root,
            expected_code_revision=args.expected_code_revision,
            local_root=args.local_root,
            remote_root=args.remote_root,
        )
        output: Any = {
            "plan": plan,
            "deployment": deployment,
            "task": task,
        }
    elif args.command == "stage":
        output = stage(
            plan_path=args.plan,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            resume_incoming=args.resume_incoming,
            apply=args.apply,
        )
    elif args.command == "submit":
        output = submit(
            plan_path=args.plan,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            priority=args.priority,
            apply=args.apply,
        )
    elif args.command == "execute":
        output = {
            "result": str(
                execute(
                    payload_path=args.payload,
                    output=args.output,
                )
            )
        }
    elif args.command == "collect":
        output = {
            "collection": str(
                collect(
                    plan_path=args.plan,
                    submission_path=args.submission,
                    output=args.output,
                    scheduler_url=args.scheduler_url,
                    accounts_path=args.accounts,
                    scheduler_source=args.scheduler_source,
                    retries=args.retries,
                )
            )
        }
    else:
        collection = validate_collection(args.collection)
        output = {
            "collection": str(args.collection.resolve(strict=True)),
            "payload_sha256": collection["payload_sha256"],
            "quality_passed": collection["quality_passed"],
            "search_only_proposal": collection["search_only_proposal"],
        }
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ALTrainingError as exc:
        print(f"[mft-goal-al-train] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
