"""Plan immutable Slurm bundles for corrected current7 Tier-1 seed lanes.

This is an additive packaging path.  It does not import, modify, or invoke the
historical all11 seed runner.  Planning is local and side-effect free outside
the requested plan directory; this module deliberately has no stage or submit
command.  Scheduler envelopes can be rendered for review and handed to a
separate, explicitly authorized publisher after the remote bundle has an
atomic READY seal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import (
        ADAPTER_SCHEMA,
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest_view,
        canonical_sha256,
        expected_generation_artifacts,
        validate_adapter_receipt,
    )
    from tier1_deep_crossover_contract import (
        DEEP_CROSSOVER_ISLANDS,
        FIXED_GENERATIONS,
        INFERENCE_THREADS,
        POPULATION,
        island_profile,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import (
        ADAPTER_SCHEMA,
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest_view,
        canonical_sha256,
        expected_generation_artifacts,
        validate_adapter_receipt,
    )
    from tools.tier1_deep_crossover_contract import (
        DEEP_CROSSOVER_ISLANDS,
        FIXED_GENERATIONS,
        INFERENCE_THREADS,
        POPULATION,
        island_profile,
    )


PLAN_SCHEMA = "mft-tier1-current7-slurm-plan-v1"
BUNDLE_SCHEMA = "mft-tier1-current7-slurm-bundle-v1"
RELOCATION_SCHEMA = "mft-tier1-current7-relocation-v1"
TASK_SCHEMA = "mft-tier1-current7-slurm-seed-task-v1"
STATUS_SCHEMA = "mft-tier1-current7-slurm-seed-status-v1"
FAST_RAMP_SCHEMA = "mft-tier1-current7-fast-ramp-v1"
REFILL_SCHEMA = "mft-tier1-current7-open-ended-refill-v1"
SEED_LEDGER_SCHEMA = "mft-tier1-current7-seed-ledger-v1"
SEARCH_INTERFACE_SCHEMA = "mft-tier1-current7-search-seed-cli-v1"
REMOTE_PREFLIGHT_SCHEMA = "mft-tier1-current7-remote-model-load-v1"
RESULT_SCHEMA = "mft-tier1-current7-search-seed-v1"
READY_SCHEMA = "mft-tier1-current7-slurm-ready-v1"

DEFAULT_REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_tier1_current7_bundles"
DEFAULT_CPUS = 8
DEFAULT_MEMORY_MB = 65_536
DEFAULT_MAX_WORKERS_PER_NODE = 4
DEFAULT_TIMEOUT_SECONDS = 86_400
DEFAULT_PRIORITY = 0
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 1_200
DEFAULT_MAX_RSS_BYTES = 42 * 1024**3

CRITICAL_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "joblib",
    "pymoo",
    "lightgbm",
    "xgboost",
    "catboost",
)

PROHIBITED_LEGACY_CODE = frozenset(
    {
        "tools/tier1_slurm_seed_runner.py",
        "tools/slurm_nsga_lane_runner.py",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
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


def _hex(value: Any, length: int, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(
            f"{label} must be a {length}-character hexadecimal digest"
        )
    return normalized


def _safe_relative(value: str, label: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"bundle source is not a file: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"bundle source SHA-256 mismatch: {path}")
    return {"sha256": digest, "size": path.stat().st_size}


def _synthetic_record(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _documentary_paths(receipt: Mapping[str, Any]) -> dict[str, str]:
    adapter = adapter_manifest_view(receipt)
    paths = {
        "generation": str(adapter.get("generation") or ""),
        "registry": str(adapter.get("registry") or ""),
        "train_report": str((adapter.get("train_report") or {}).get("path") or ""),
        "candidate": str((adapter.get("candidate") or {}).get("path") or ""),
        "quality_status": str(
            (adapter.get("quality_status") or {}).get("path") or ""
        ),
        "dataset": str((adapter.get("dataset") or {}).get("path") or ""),
        "profile": str((adapter.get("profile") or {}).get("path") or ""),
        "adapter_code": str((adapter.get("code") or {}).get("path") or ""),
    }
    if any(not value for value in paths.values()):
        raise RuntimeError("adapter receipt documentary path inventory is incomplete")
    return paths


def runtime_package_versions() -> dict[str, str]:
    versions = {}
    for package in CRITICAL_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"critical runtime package is unavailable: {package}") from exc
    return versions


def authenticate_clean_code_root(
    code_root: Path, expected_revision: str,
) -> dict[str, Any]:
    root = code_root.resolve(strict=True)
    expected = _hex(expected_revision, 40, "bundle code revision")
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = root.as_posix()
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    actual = str(revision.stdout or "").strip().lower()
    if revision.returncode != 0 or actual != expected:
        raise RuntimeError("bundle code revision mismatch")
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("bundle code root must be clean before planning")
    return {"path": str(root), "revision": actual, "clean": True}


def collect_tracked_code_sources(
    code_root: Path,
    *,
    optimizer_entrypoint: str,
    extra_code_files: Iterable[str] = (),
) -> dict[str, Path]:
    """Collect the current7 runtime without adding the legacy all11 runner."""

    root = code_root.resolve(strict=True)
    entrypoint = _safe_relative(optimizer_entrypoint, "optimizer entrypoint")
    required_tools = {
        entrypoint,
        "tools/tier1_corrected_generation_adapter.py",
        "tools/tier1_corrected_current7_receipt.py",
        "tools/tier1_corrected_current7_slurm_bundle.py",
        "tools/tier1_corrected_current7_slurm_seed_runner.py",
        "tools/tier1_deep_crossover_contract.py",
        "tools/tier1_n1_6_anchor_island_contract.py",
        *(_safe_relative(item, "extra code file") for item in extra_code_files),
    }
    if required_tools.intersection(PROHIBITED_LEGACY_CODE):
        raise RuntimeError("legacy all11 runner cannot enter a current7 bundle")

    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = root.as_posix()
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
        env=environment,
    )
    if listed.returncode != 0:
        raise RuntimeError("cannot enumerate committed bundle code")
    tracked = {
        item.decode("utf-8").replace("\\", "/")
        for item in listed.stdout.split(b"\0")
        if item
    }
    missing = sorted(required_tools.difference(tracked))
    if missing:
        raise RuntimeError(f"required current7 code is not committed: {missing}")

    selected = set(required_tools)
    selected.update(
        item
        for item in tracked
        if item.startswith("regression_260707/")
        and PurePosixPath(item).suffix in {".py", ".json"}
    )
    selected.difference_update(PROHIBITED_LEGACY_CODE)
    return {
        f"artifacts/code/{relative}": (root / relative).resolve(strict=True)
        for relative in sorted(selected)
    }


def _generation_sources(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    dataset_path: Path,
    profile_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    identity = validate_adapter_receipt(receipt)
    generation = generation.resolve(strict=True)
    if not generation.is_dir() or generation.name != identity.training_run_id:
        raise RuntimeError("generation directory differs from adapter receipt")
    report_path = generation / "train_report.json"
    report = read_json(report_path)
    _record(report_path, expected_sha256=identity.train_report_sha256)
    if report.get("schema_version") != 2:
        raise RuntimeError("corrected train report schema mismatch")
    if (
        report.get("training_run_id") != identity.training_run_id
        or report.get("targets") != list(CORRECTED_GENERATION_TARGETS)
        or report.get("dataset_sha256") != identity.dataset_sha256
        or report.get("profile_sha256") != identity.profile_canonical_sha256
        or report.get("strict_full_rows") != identity.strict_full_rows
    ):
        raise RuntimeError("corrected train report identity mismatch")
    artifacts = report.get("artifacts")
    expected_artifacts = set(expected_generation_artifacts())
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise RuntimeError("corrected train report artifact inventory mismatch")

    receipt_path = receipt_path.resolve(strict=True)
    candidate_path = candidate_path.resolve(strict=True)
    quality_path = quality_path.resolve(strict=True)
    dataset_path = dataset_path.resolve(strict=True)
    profile_path = profile_path.resolve(strict=True)
    sources: dict[str, Path] = {
        "artifacts/evidence/adapter_receipt.json": receipt_path,
        "artifacts/evidence/candidate.json": candidate_path,
        "artifacts/evidence/quality_status.json": quality_path,
        "artifacts/dataset/strict_full.parquet": dataset_path,
        "artifacts/profile/standard.json": profile_path,
        (
            "artifacts/registry/generations/"
            f"{identity.training_run_id}/train_report.json"
        ): report_path,
    }
    expected_file_shas = {
        "artifacts/evidence/candidate.json": identity.candidate_sha256,
        "artifacts/evidence/quality_status.json": identity.quality_status_sha256,
        "artifacts/dataset/strict_full.parquet": identity.dataset_sha256,
        (
            "artifacts/registry/generations/"
            f"{identity.training_run_id}/train_report.json"
        ): identity.train_report_sha256,
    }
    for relative, expected in expected_file_shas.items():
        _record(sources[relative], expected_sha256=expected)
    profile = read_json(profile_path)
    if canonical_sha256(profile) != identity.profile_canonical_sha256:
        raise RuntimeError("relocated profile canonical SHA-256 mismatch")

    generation_inventory: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_artifacts):
        source = (generation / relative).resolve(strict=True)
        try:
            source.relative_to(generation)
        except ValueError as exc:
            raise RuntimeError("generation artifact escaped its root") from exc
        expected_sha = _hex(
            artifacts.get(relative), 64, f"generation artifact SHA {relative}"
        )
        record = _record(source, expected_sha256=expected_sha)
        if record["size"] != identity.artifact_sizes_bytes[relative]:
            raise RuntimeError(f"generation artifact size mismatch: {relative}")
        bundle_relative = (
            "artifacts/registry/generations/"
            f"{identity.training_run_id}/{relative}"
        )
        sources[bundle_relative] = source
        generation_inventory[relative] = record

    relocation = {
        "schema_version": RELOCATION_SCHEMA,
        "source_absolute_paths_are_documentary_only": True,
        "local_adapter_authentication_replayed_remotely": False,
        "generation_report_bytes_mutated": False,
        "remote_git_checkout_required": False,
        "original_paths": _documentary_paths(receipt),
        "bundle_paths": {
            "adapter_receipt": "artifacts/evidence/adapter_receipt.json",
            "candidate": "artifacts/evidence/candidate.json",
            "quality_status": "artifacts/evidence/quality_status.json",
            "dataset": "artifacts/dataset/strict_full.parquet",
            "profile": "artifacts/profile/standard.json",
            "registry": "artifacts/registry",
            "generation": (
                "artifacts/registry/generations/"
                f"{identity.training_run_id}"
            ),
            "train_report": (
                "artifacts/registry/generations/"
                f"{identity.training_run_id}/train_report.json"
            ),
        },
        "relocated_identity": {
            **identity.to_dict(),
            "generation_artifacts": generation_inventory,
            "generation_artifact_inventory_sha256": canonical_sha256(
                generation_inventory
            ),
        },
    }
    return identity.to_dict(), sources, relocation


def _island_contracts(
    warm_starts: Mapping[str, Mapping[str, Path]],
) -> tuple[dict[str, Any], dict[str, Path]]:
    if set(warm_starts) != {"n1-5", "n1-6"}:
        raise RuntimeError("warm-start inventory must contain n1-5 and n1-6")
    warm_contracts = {}
    sources = {}
    for warm_id in ("n1-5", "n1-6"):
        value = warm_starts[warm_id]
        if set(value) != {"artifact", "contract"}:
            raise RuntimeError(f"{warm_id} warm start requires artifact and contract")
        artifact = Path(value["artifact"]).resolve(strict=True)
        contract = Path(value["contract"]).resolve(strict=True)
        artifact_relative = f"artifacts/warm/{warm_id}/coordinates.npy"
        contract_relative = f"artifacts/warm/{warm_id}/contract.json"
        artifact_record = _record(artifact)
        contract_record = _record(contract)
        sources[artifact_relative] = artifact
        sources[contract_relative] = contract
        warm_contracts[warm_id] = {
            "artifact": {"path": artifact_relative, **artifact_record},
            "contract": {"path": contract_relative, **contract_record},
        }

    islands = {}
    for island in DEEP_CROSSOVER_ISLANDS:
        source_profile = island_profile(island)
        current7_profile = {
            "schema_version": "mft-tier1-current7-deep-crossover-island-v1",
            "island_id": island.island_id,
            "variant": f"{island.variant}-current7",
            "source_deep_crossover_profile_sha256": source_profile["sha256"],
            "active_quota": island.active_quota,
            "seed_start": island.seed_start,
            "seed_window_end_exclusive": island.seed_window_end_exclusive,
            "fixed_primary_turns": island.fixed_primary_turns,
            "fixed_primary_turns_repair_required": True,
            "warm_family": island.warm_family,
            "optimizer_llt_scale_uH": island.optimizer_llt_scale_uh,
            "optimizer_all_current7_thermal_scale_C": (
                island.optimizer_all_thermal_scale_c
            ),
            "optimizer_resonance_allowance_Hz": (
                island.optimizer_resonance_allowance_hz
            ),
            "optimizer_llt_allowance_uH": island.optimizer_llt_allowance_uh,
            "optimizer_resonance_scale_Hz": source_profile[
                "optimizer_resonance_scale_Hz"
            ],
            "optimizer_termination_strategy": source_profile[
                "optimizer_termination_strategy"
            ],
            "temperature_targets": list(CURRENT_TEMPERATURE_TARGETS),
            "population": POPULATION,
            "fixed_generations": FIXED_GENERATIONS,
            "inference_threads": INFERENCE_THREADS,
            "topology_evolution_contract": source_profile[
                "topology_evolution_contract"
            ],
            "offspring_physics_repair_required": True,
            "terminal_physical_replay_required": True,
            "physical_hard_spec_mutation": False,
            "objective_mutation": False,
            "authoritative_terminal_G": "physical_unscaled_unchanged",
            "scheduler_priority_inherited_from_source_profile": False,
            "automatic_promotion_allowed": False,
        }
        current7_profile["sha256"] = canonical_sha256(current7_profile)
        warm_id = "n1-5" if island.fixed_primary_turns == 5 else "n1-6"
        islands[island.island_id] = {
            "source_profile": source_profile,
            "source_profile_sha256": source_profile["sha256"],
            "current7_profile": current7_profile,
            "current7_profile_sha256": current7_profile["sha256"],
            "warm_id": warm_id,
            "warm": warm_contracts[warm_id],
            "scheduler_priority_inherited_from_source_profile": False,
        }
    return islands, sources


def _resource_contract(priority: int) -> dict[str, Any]:
    if isinstance(priority, bool) or int(priority) != priority or int(priority) < 0:
        raise ValueError("current7 scheduler priority must be a non-negative integer")
    return {
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "aedt_used": False,
        "gpus": 0,
        "cpus_per_seed_task": DEFAULT_CPUS,
        "memory_mb_per_seed_task": DEFAULT_MEMORY_MB,
        "optimizer_processes_per_task": 1,
        "inference_threads_per_task": INFERENCE_THREADS,
        "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "priority": int(priority),
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "peak_rss_gate_bytes": DEFAULT_MAX_RSS_BYTES,
        "negative_priority_hardcoded": False,
    }


def seed_lanes() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canaries = []
    ramp = []
    occupied = set()
    for island in DEEP_CROSSOVER_ISLANDS:
        for offset in range(island.active_quota):
            seed = island.seed_start + offset
            if seed in occupied or seed >= island.seed_window_end_exclusive:
                raise RuntimeError("current7 seed allocation overlaps or escapes its window")
            occupied.add(seed)
            lane = {
                "island_id": island.island_id,
                "variant": f"{island.variant}-current7",
                "seed": seed,
                "fixed_primary_turns": island.fixed_primary_turns,
                "wave": "canary" if offset == 0 else "ramp",
            }
            (canaries if offset == 0 else ramp).append(lane)
    if len(canaries) != 4 or len(ramp) != 32 or len(occupied) != 36:
        raise RuntimeError("current7 fast-ramp seed shape must be exactly 4 + 32")
    return canaries, ramp


def build_plan(
    *,
    local_root: Path,
    remote_root: str,
    receipt_path: Path,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    dataset_path: Path,
    profile_path: Path,
    code_identity: Mapping[str, Any],
    code_sources: Mapping[str, Path],
    optimizer_entrypoint: str,
    warm_starts: Mapping[str, Mapping[str, Path]],
    runtime_packages: Mapping[str, str],
    priority: int = DEFAULT_PRIORITY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = receipt_path.resolve(strict=True)
    receipt = read_json(receipt_path)
    identity, sources, relocation = _generation_sources(
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
        dataset_path=dataset_path,
        profile_path=profile_path,
    )
    if code_identity.get("clean") is not True:
        raise RuntimeError("bundle code identity is not clean")
    code_revision = _hex(
        code_identity.get("revision"), 40, "bundle code revision"
    )
    entrypoint = _safe_relative(optimizer_entrypoint, "optimizer entrypoint")
    entrypoint_bundle = f"artifacts/code/{entrypoint}"
    runner_bundle = (
        "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py"
    )
    normalized_code_sources = {}
    for relative, source in code_sources.items():
        relative = _safe_relative(relative, "code bundle path")
        if relative in PROHIBITED_LEGACY_CODE or relative.removeprefix(
            "artifacts/code/"
        ) in PROHIBITED_LEGACY_CODE:
            raise RuntimeError("legacy all11 runner cannot enter a current7 bundle")
        if not relative.startswith("artifacts/code/"):
            raise RuntimeError("code source must be below artifacts/code")
        normalized_code_sources[relative] = Path(source).resolve(strict=True)
    if entrypoint_bundle not in normalized_code_sources:
        raise RuntimeError("current7 optimizer entrypoint is absent from code inventory")
    if runner_bundle not in normalized_code_sources:
        raise RuntimeError("current7 Slurm runner is absent from code inventory")
    if any(
        relative.endswith("/tier1_slurm_seed_runner.py")
        for relative in normalized_code_sources
    ):
        raise RuntimeError("legacy all11 seed runner entered current7 code inventory")
    sources.update(normalized_code_sources)

    islands, warm_sources = _island_contracts(warm_starts)
    sources.update(warm_sources)
    source_revision_bytes = (code_revision + "\n").encode("ascii")
    relocation_bytes = (
        json.dumps(relocation, indent=2, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    if set(runtime_packages) != set(CRITICAL_PACKAGES):
        raise RuntimeError("critical runtime package inventory mismatch")
    normalized_packages = {
        name: str(runtime_packages[name]).strip() for name in CRITICAL_PACKAGES
    }
    if any(not value for value in normalized_packages.values()):
        raise RuntimeError("critical runtime package version is empty")
    requirements_bytes = "".join(
        f"{name}=={normalized_packages[name]}\n" for name in sorted(normalized_packages)
    ).encode("ascii")

    synthetic = {
        "artifacts/code/.source-revision": source_revision_bytes,
        "artifacts/evidence/relocation.json": relocation_bytes,
        "artifacts/runtime/requirements.lock": requirements_bytes,
    }
    files = {relative: _record(source) for relative, source in sorted(sources.items())}
    files.update(
        {relative: _synthetic_record(payload) for relative, payload in synthetic.items()}
    )
    code_inventory = {
        relative: files[relative]
        for relative in sorted(files)
        if relative.startswith("artifacts/code/")
    }
    resource_contract = _resource_contract(priority)
    canaries, ramp = seed_lanes()
    fast_ramp = {
        "schema_version": FAST_RAMP_SCHEMA,
        "canary_count": 4,
        "ramp_count": 32,
        "canaries": canaries,
        "ramp": ramp,
        "release_trigger": "all_four_remote_model_load_and_rss_gates_pass",
        "wait_for_canary_optimizer_completion": False,
        "release_mode": "submit_all_remaining_32_in_one_immediate_wave",
        "preflight_timeout_seconds": DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        "maximum_peak_rss_bytes": DEFAULT_MAX_RSS_BYTES,
    }
    refill = {
        "schema_version": REFILL_SCHEMA,
        "policy": "maintain_per_island_active_quota_until_stop_requested",
        "initial_bootstrap_count": 36,
        "refill_waits_for_full_36_terminal_wave": False,
        "refill_on_each_observed_terminal_gap": True,
        "seed_reuse_allowed": False,
        "ledger_schema_version": SEED_LEDGER_SCHEMA,
        "island_active_quotas": {
            island.island_id: island.active_quota
            for island in DEEP_CROSSOVER_ISLANDS
        },
        "seed_windows": {
            island.island_id: {
                "start": island.seed_start,
                "end_exclusive": island.seed_window_end_exclusive,
            }
            for island in DEEP_CROSSOVER_ISLANDS
        },
        "stop_requires_explicit_ledger_flag": True,
        "scheduler_submission_performed": False,
    }
    stable_contract = {
        "schema_version": BUNDLE_SCHEMA,
        "adapter_schema_version": ADAPTER_SCHEMA,
        "task_schema_version": TASK_SCHEMA,
        "status_schema_version": STATUS_SCHEMA,
        "ready_schema_version": READY_SCHEMA,
        "bundle_code_revision": code_revision,
        "bundle_code_clean_at_plan": True,
        "remote_git_checkout_required": False,
        "files": files,
        "code_inventory": code_inventory,
        "code_inventory_sha256": canonical_sha256(code_inventory),
        "adapter_receipt": {
            "path": "artifacts/evidence/adapter_receipt.json",
            "file_sha256": files[
                "artifacts/evidence/adapter_receipt.json"
            ]["sha256"],
            "identity": identity,
            "launch_eligible": identity["launch_eligible"],
        },
        "relocation": {
            "path": "artifacts/evidence/relocation.json",
            "sha256": files["artifacts/evidence/relocation.json"]["sha256"],
            "contract_sha256": canonical_sha256(relocation),
        },
        "generation_artifacts": relocation["relocated_identity"][
            "generation_artifacts"
        ],
        "generation_artifact_inventory_sha256": relocation[
            "relocated_identity"
        ]["generation_artifact_inventory_sha256"],
        "islands": islands,
        "search_execution": {
            "schema_version": SEARCH_INTERFACE_SCHEMA,
            "entrypoint": entrypoint_bundle,
            "runner": runner_bundle,
            "remote_preflight_filename": "remote_preflight.json",
            "result_filename": "result.json",
            "remote_preflight_schema_version": REMOTE_PREFLIGHT_SCHEMA,
            "result_schema_version": RESULT_SCHEMA,
            "optimizer_processes_per_task": 1,
            "model_mapping_instances_per_process": 1,
        },
        "runtime": {
            "critical_packages": normalized_packages,
            "requirements_lock": "artifacts/runtime/requirements.lock",
        },
        "resources": resource_contract,
        "scheduler_api_contract": {
            "method": "POST",
            "endpoint_path": "/api/tasks",
            "required_fields": [
                "name",
                "remote_cwd",
                "command",
                "payload_json",
                "required_capability",
                "env_profile",
                "cpus",
                "memory_mb",
                "scheduling_profile",
                "aedt_backend",
                "gpus",
                "priority",
                "timeout_seconds",
                "dedupe_key",
                "max_workers_per_node",
            ],
            "requested_allocation_id_allowed": False,
            "active_dedupe_returns_existing_task": True,
            "scheduler_payload_environment_variable": (
                "SLURM_SCHEDULER_PAYLOAD_PATH"
            ),
            "scheduler_submission_performed": False,
        },
        "fast_ramp": fast_ramp,
        "open_ended_refill": refill,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    contract_sha = canonical_sha256(stable_contract)
    bundle_id = f"current7-{contract_sha[:20]}"
    manifest = {
        **stable_contract,
        "bundle_id": bundle_id,
        "contract_sha256": contract_sha,
    }
    local_root = local_root.resolve()
    plan_dir = local_root / bundle_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    synthetic_paths = {}
    for relative, payload in synthetic.items():
        destination = plan_dir / "synthetic" / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        synthetic_paths[relative] = destination
    source_map = {
        relative: str(source)
        for relative, source in sorted({**sources, **synthetic_paths}.items())
    }
    manifest_path = plan_dir / "bundle_manifest.json"
    source_map_path = plan_dir / "local_sources.json"
    relocation_path = synthetic_paths["artifacts/evidence/relocation.json"]
    atomic_json(manifest_path, manifest)
    atomic_json(source_map_path, source_map)
    manifest_sha = sha256_file(manifest_path)
    remote_bundle = remote_root.rstrip("/") + "/" + bundle_id
    plan = {
        "schema_version": PLAN_SCHEMA,
        "created_at": _now(),
        "bundle_id": bundle_id,
        "contract_sha256": contract_sha,
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": manifest_sha,
        "local_sources": str(source_map_path),
        "relocation_contract": str(relocation_path),
        "remote_root": remote_root.rstrip("/"),
        "remote_bundle": remote_bundle,
        "publication_contract": {
            "stage_below_unique_incoming_directory": True,
            "verify_every_file_sha256_before_ready": True,
            "verify_exact_runtime_packages_before_ready": True,
            "write_ready_after_all_verification": True,
            "atomic_rename_incoming_to_content_addressed_bundle": True,
            "make_artifacts_read_only_before_publication": True,
        },
        "resources": resource_contract,
        "fast_ramp": fast_ramp,
        "open_ended_refill": refill,
        "stage_performed": False,
        "submission_performed": False,
    }
    atomic_json(plan_dir / "offload_plan.json", plan)
    return plan, manifest


def _lane_by_seed(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    lanes = [
        *manifest["fast_ramp"]["canaries"],
        *manifest["fast_ramp"]["ramp"],
    ]
    matches = [lane for lane in lanes if int(lane["seed"]) == int(seed)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError("seed appears more than once in current7 lane inventory")
    refill = manifest.get("open_ended_refill") or {}
    matching_islands = []
    for island_id, window in (refill.get("seed_windows") or {}).items():
        if int(window["start"]) <= int(seed) < int(window["end_exclusive"]):
            matching_islands.append(island_id)
    if len(matching_islands) != 1:
        raise RuntimeError("seed is outside the sealed current7 refill windows")
    island_id = matching_islands[0]
    profile = manifest["islands"][island_id]["current7_profile"]
    return {
        "island_id": island_id,
        "variant": profile["variant"],
        "seed": int(seed),
        "fixed_primary_turns": profile["fixed_primary_turns"],
        "wave": "refill",
    }


def build_task_payload(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    seed: int,
    priority: int | None = None,
) -> dict[str, Any]:
    if manifest.get("bundle_id") != plan.get("bundle_id"):
        raise RuntimeError("plan/manifest bundle identity mismatch")
    receipt_identity = manifest["adapter_receipt"]["identity"]
    if (
        receipt_identity.get("launch_eligible") is not True
        or receipt_identity.get("offspring_physics_repair") is not True
        or receipt_identity.get("fixed_primary_turns_supported") != [5, 6]
        or receipt_identity.get("initial_repair_attested") is not True
        or receipt_identity.get("warm_repair_attested") is not True
        or receipt_identity.get(
            "every_offspring_decode_repair_attested"
        )
        is not True
        or receipt_identity.get("terminal_physical_replay_attested") is not True
    ):
        raise RuntimeError(
            "current7 receipt is smoke-only or lacks the mandatory "
            "fixed-turn offspring physics-repair/replay gate"
        )
    lane = _lane_by_seed(manifest, seed)
    island = manifest["islands"][lane["island_id"]]
    resources = dict(manifest["resources"])
    if priority is not None:
        resources = _resource_contract(priority)
    task_contract = {
        "schema_version": TASK_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "cohort_id": f"{plan['bundle_id']}-deep36",
        "lane": lane,
        "island_profile_sha256": island["current7_profile_sha256"],
        "warm_id": island["warm_id"],
        "warm_artifact_sha256": island["warm"]["artifact"]["sha256"],
        "warm_contract_sha256": island["warm"]["contract"]["sha256"],
        "seed": int(seed),
        "population": POPULATION,
        "max_generations": FIXED_GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "optimizer_processes": 1,
        "adapter_receipt_file_sha256": manifest["adapter_receipt"][
            "file_sha256"
        ],
        "adapter_manifest_sha256": manifest["adapter_receipt"]["identity"][
            "adapter_manifest_sha256"
        ],
        "train_report_sha256": manifest["adapter_receipt"]["identity"][
            "train_report_sha256"
        ],
        "dataset_sha256": manifest["adapter_receipt"]["identity"][
            "dataset_sha256"
        ],
        "profile_canonical_sha256": manifest["adapter_receipt"]["identity"][
            "profile_canonical_sha256"
        ],
        "required_model_targets_sha256": CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        "temperature_contract_sha256": manifest["adapter_receipt"]["identity"][
            "temperature_contract_sha256"
        ],
        "hard_constraint_contract_sha256": manifest["adapter_receipt"][
            "identity"
        ]["hard_constraint_contract_sha256"],
        "optimizer_repair_contract_sha256": receipt_identity[
            "optimizer_repair_contract_sha256"
        ],
        "offspring_physics_repair": True,
        "fixed_primary_turns_supported": [5, 6],
        "initial_repair_attested": True,
        "warm_repair_attested": True,
        "every_offspring_decode_repair_attested": True,
        "terminal_physical_replay_attested": True,
        "generation_artifact_inventory_sha256": manifest[
            "generation_artifact_inventory_sha256"
        ],
        "relocation_contract_sha256": manifest["relocation"][
            "contract_sha256"
        ],
        "search_interface_schema_version": SEARCH_INTERFACE_SCHEMA,
        "preflight_timeout_seconds": manifest["fast_ramp"][
            "preflight_timeout_seconds"
        ],
        "maximum_peak_rss_bytes": manifest["fast_ramp"][
            "maximum_peak_rss_bytes"
        ],
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    payload_sha = canonical_sha256(task_contract)
    command = "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/code${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            "exec python -u artifacts/code/tools/"
            "tier1_corrected_current7_slurm_seed_runner.py "
            '--bundle-root "$PWD" --payload "$payload_path" '
            '--payload-root "$payload_root" '
            f"--payload-sha256 {payload_sha}",
        ]
    )
    dedupe = canonical_sha256(
        {
            "bundle_id": plan["bundle_id"],
            "task": task_contract,
            "resources": resources,
        }
    )
    return {
        "name": (
            f"mft-t1c7-{lane['wave']}-{plan['bundle_id'][-8:]}-"
            f"{lane['island_id'][:12]}-{seed}"
        ),
        "remote_cwd": plan["remote_bundle"],
        "command": command,
        "payload_json": task_contract,
        "required_capability": resources["required_capability"],
        "env_profile": resources["env_profile"],
        "cpus": resources["cpus_per_seed_task"],
        "memory_mb": resources["memory_mb_per_seed_task"],
        "scheduling_profile": resources["scheduling_profile"],
        "aedt_backend": resources["aedt_backend"],
        "gpus": resources["gpus"],
        "priority": resources["priority"],
        "timeout_seconds": resources["timeout_seconds"],
        "dedupe_key": f"mft-tier1-current7:{dedupe}",
        "max_workers_per_node": resources["max_workers_per_node"],
    }


def build_task_waves(
    plan: Mapping[str, Any], manifest: Mapping[str, Any],
    *, priority: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for wave in ("canaries", "ramp"):
        result[wave] = [
            build_task_payload(
                plan, manifest, seed=int(lane["seed"]), priority=priority
            )
            for lane in manifest["fast_ramp"][wave]
        ]
    return result


ACTIVE_LEDGER_STATES = frozenset(
    {"planned", "submitted", "queued", "running"}
)
TERMINAL_LEDGER_STATES = frozenset(
    {"completed", "failed", "cancelled", "timed_out"}
)


def seal_seed_ledger(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    revision: int,
    parent_ledger_sha256: str | None = None,
    stop_requested: bool = False,
) -> dict[str, Any]:
    if isinstance(revision, bool) or int(revision) != revision or revision < 0:
        raise ValueError("seed-ledger revision must be a non-negative integer")
    normalized = [dict(entry) for entry in entries]
    seeds = [int(entry.get("seed", -1)) for entry in normalized]
    dedupe = [str(entry.get("dedupe_key") or "") for entry in normalized]
    if len(seeds) != len(set(seeds)) or len(dedupe) != len(set(dedupe)):
        raise RuntimeError("seed ledger contains a seed/dedupe identity duplicate")
    if any(not value for value in dedupe):
        raise RuntimeError("seed ledger contains an empty dedupe identity")
    refill = manifest.get("open_ended_refill") or {}
    if refill.get("schema_version") != REFILL_SCHEMA:
        raise RuntimeError("bundle has no open-ended refill contract")
    for entry in normalized:
        island_id = str(entry.get("island_id") or "")
        window = (refill.get("seed_windows") or {}).get(island_id)
        state = str(entry.get("state") or "")
        if (
            not window
            or not int(window["start"]) <= int(entry["seed"]) < int(
                window["end_exclusive"]
            )
            or state not in ACTIVE_LEDGER_STATES | TERMINAL_LEDGER_STATES
            or not isinstance(entry.get("task_payload_sha256"), str)
            or len(entry["task_payload_sha256"]) != 64
        ):
            raise RuntimeError("seed ledger entry violates its island contract")
    used = set(seeds)
    next_seed_by_island = {}
    for island_id, window in (refill.get("seed_windows") or {}).items():
        candidate = int(window["start"])
        while candidate in used and candidate < int(window["end_exclusive"]):
            candidate += 1
        next_seed_by_island[island_id] = candidate
    unsigned = {
        "schema_version": SEED_LEDGER_SCHEMA,
        "bundle_id": manifest.get("bundle_id"),
        "revision": int(revision),
        "parent_ledger_sha256": parent_ledger_sha256,
        "stop_requested": bool(stop_requested),
        "next_seed_by_island": next_seed_by_island,
        "entries": normalized,
        "scheduler_submission_performed_by_ledger": False,
    }
    return {**unsigned, "ledger_sha256": canonical_sha256(unsigned)}


def validate_seed_ledger(
    manifest: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in ledger.items() if key != "ledger_sha256"
    }
    if (
        ledger.get("schema_version") != SEED_LEDGER_SCHEMA
        or ledger.get("bundle_id") != manifest.get("bundle_id")
        or ledger.get("scheduler_submission_performed_by_ledger") is not False
        or ledger.get("ledger_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("seed ledger identity/SHA mismatch")
    resealed = seal_seed_ledger(
        manifest,
        ledger.get("entries") or [],
        revision=int(ledger.get("revision", -1)),
        parent_ledger_sha256=ledger.get("parent_ledger_sha256"),
        stop_requested=bool(ledger.get("stop_requested")),
    )
    if resealed != dict(ledger):
        raise RuntimeError("seed ledger canonical contract mismatch")
    return dict(ledger)


def initial_seed_ledger(
    plan: Mapping[str, Any], manifest: Mapping[str, Any],
    *, priority: int | None = None,
) -> dict[str, Any]:
    waves = build_task_waves(plan, manifest, priority=priority)
    entries = []
    for task in [*waves["canaries"], *waves["ramp"]]:
        payload = task["payload_json"]
        entries.append(
            {
                "island_id": payload["lane"]["island_id"],
                "seed": payload["seed"],
                "state": "planned",
                "wave": payload["lane"]["wave"],
                "task_payload_sha256": canonical_sha256(payload),
                "dedupe_key": task["dedupe_key"],
            }
        )
    return seal_seed_ledger(manifest, entries, revision=0)


def plan_refill_wave(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    priority: int | None = None,
) -> dict[str, Any]:
    ledger = validate_seed_ledger(manifest, ledger)
    if ledger["stop_requested"] is True:
        return {
            "schema_version": REFILL_SCHEMA,
            "bundle_id": manifest["bundle_id"],
            "eligible": False,
            "reason": "explicit_stop_requested",
            "tasks": [],
            "next_ledger": None,
            "scheduler_submission_performed": False,
        }
    refill = manifest["open_ended_refill"]
    entries = [dict(entry) for entry in ledger["entries"]]
    used = {int(entry["seed"]) for entry in entries}
    tasks = []
    active_before = {}
    gaps = {}
    for island_id, quota in refill["island_active_quotas"].items():
        active = sum(
            entry["island_id"] == island_id
            and entry["state"] in ACTIVE_LEDGER_STATES
            for entry in entries
        )
        if active > int(quota):
            raise RuntimeError("seed ledger exceeds a sealed island active quota")
        active_before[island_id] = active
        gaps[island_id] = int(quota) - active
        window = refill["seed_windows"][island_id]
        candidate = int(ledger["next_seed_by_island"][island_id])
        for _index in range(gaps[island_id]):
            while candidate in used and candidate < int(window["end_exclusive"]):
                candidate += 1
            if candidate >= int(window["end_exclusive"]):
                raise RuntimeError(f"seed window exhausted for island {island_id}")
            task = build_task_payload(
                plan, manifest, seed=candidate, priority=priority
            )
            tasks.append(task)
            payload = task["payload_json"]
            entries.append(
                {
                    "island_id": island_id,
                    "seed": candidate,
                    "state": "planned",
                    "wave": "refill",
                    "task_payload_sha256": canonical_sha256(payload),
                    "dedupe_key": task["dedupe_key"],
                }
            )
            used.add(candidate)
            candidate += 1
    next_ledger = seal_seed_ledger(
        manifest,
        entries,
        revision=int(ledger["revision"]) + 1,
        parent_ledger_sha256=ledger["ledger_sha256"],
    )
    terminal_count = sum(
        entry["state"] in TERMINAL_LEDGER_STATES for entry in ledger["entries"]
    )
    return {
        "schema_version": REFILL_SCHEMA,
        "bundle_id": manifest["bundle_id"],
        "eligible": bool(tasks),
        "reason": "terminal_quota_gaps" if tasks else "all_quotas_full",
        "active_before": active_before,
        "quota_gaps": gaps,
        "cumulative_terminal_count": terminal_count,
        "cumulative_seed_count_before": len(ledger["entries"]),
        "refill_count": len(tasks),
        "tasks": tasks,
        "next_ledger": next_ledger,
        "scheduler_submission_performed": False,
    }


def assess_fast_ramp(
    manifest: Mapping[str, Any], statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        int(lane["seed"]): lane for lane in manifest["fast_ramp"]["canaries"]
    }
    observed = {int(status.get("seed", -1)): status for status in statuses}
    reasons = []
    if set(observed) != set(expected):
        reasons.append("canary_seed_inventory_mismatch")
    for seed, lane in expected.items():
        status = observed.get(seed) or {}
        if status.get("schema_version") != STATUS_SCHEMA:
            reasons.append(f"seed_{seed}:status_schema")
            continue
        checks = {
            "bundle": status.get("bundle_id") == manifest.get("bundle_id"),
            "island": status.get("island_id") == lane["island_id"],
            "gate": status.get("ramp_gate_passed") is True,
            "process": status.get("optimizer_processes") == 1,
            "threads": status.get("inference_threads") == INFERENCE_THREADS,
            "models": status.get("loaded_model_count")
            == len(CURRENT_REQUIRED_MODEL_TARGETS),
            "auth_pass": status.get("full_generation_authentication_passes") == 1,
            "artifact_count": status.get("authenticated_artifact_count")
            == len(expected_generation_artifacts()),
            "rss": isinstance(status.get("observed_peak_rss_bytes"), int)
            and 0 < status["observed_peak_rss_bytes"]
            <= manifest["fast_ramp"]["maximum_peak_rss_bytes"],
            "running": status.get("optimizer_started") is True
            and status.get("terminal") is not True,
        }
        reasons.extend(
            f"seed_{seed}:{name}" for name, passed in checks.items() if not passed
        )
    eligible = not reasons
    return {
        "schema_version": FAST_RAMP_SCHEMA,
        "bundle_id": manifest.get("bundle_id"),
        "eligible": eligible,
        "reasons": reasons,
        "verified_canary_count": len(expected) if eligible else 0,
        "release_count": len(manifest["fast_ramp"]["ramp"]) if eligible else 0,
        "release_mode": (
            "submit_all_remaining_32_in_one_immediate_wave" if eligible else "hold"
        ),
        "wait_for_canary_optimizer_completion": False,
        "scheduler_submission_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--local-root", type=Path, required=True)
    plan.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    plan.add_argument("--receipt", type=Path, required=True)
    plan.add_argument("--generation", type=Path, required=True)
    plan.add_argument("--candidate", type=Path, required=True)
    plan.add_argument("--quality", type=Path, required=True)
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--code-root", type=Path, required=True)
    plan.add_argument("--code-revision", required=True)
    plan.add_argument("--optimizer-entrypoint", required=True)
    plan.add_argument("--extra-code-file", action="append", default=[])
    plan.add_argument("--n1-5-warm", type=Path, required=True)
    plan.add_argument("--n1-5-warm-contract", type=Path, required=True)
    plan.add_argument("--n1-6-warm", type=Path, required=True)
    plan.add_argument("--n1-6-warm-contract", type=Path, required=True)
    plan.add_argument("--priority", type=int, default=DEFAULT_PRIORITY)
    waves = commands.add_parser("render-task-waves")
    waves.add_argument("--plan", type=Path, required=True)
    waves.add_argument("--priority", type=int)
    ledger = commands.add_parser("render-initial-ledger")
    ledger.add_argument("--plan", type=Path, required=True)
    ledger.add_argument("--priority", type=int)
    refill = commands.add_parser("render-refill-wave")
    refill.add_argument("--plan", type=Path, required=True)
    refill.add_argument("--ledger", type=Path, required=True)
    refill.add_argument("--priority", type=int)
    assess = commands.add_parser("assess-fast-ramp")
    assess.add_argument("--manifest", type=Path, required=True)
    assess.add_argument("--statuses", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        code_identity = authenticate_clean_code_root(
            args.code_root, args.code_revision
        )
        code_sources = collect_tracked_code_sources(
            args.code_root,
            optimizer_entrypoint=args.optimizer_entrypoint,
            extra_code_files=args.extra_code_file,
        )
        value, _manifest = build_plan(
            local_root=args.local_root,
            remote_root=args.remote_root,
            receipt_path=args.receipt,
            generation=args.generation,
            candidate_path=args.candidate,
            quality_path=args.quality,
            dataset_path=args.dataset,
            profile_path=args.profile,
            code_identity=code_identity,
            code_sources=code_sources,
            optimizer_entrypoint=args.optimizer_entrypoint,
            warm_starts={
                "n1-5": {
                    "artifact": args.n1_5_warm,
                    "contract": args.n1_5_warm_contract,
                },
                "n1-6": {
                    "artifact": args.n1_6_warm,
                    "contract": args.n1_6_warm_contract,
                },
            },
            runtime_packages=runtime_package_versions(),
            priority=args.priority,
        )
    elif args.command in {
        "render-task-waves", "render-initial-ledger", "render-refill-wave",
    }:
        plan = read_json(args.plan)
        manifest = read_json(Path(plan["bundle_manifest"]))
        if args.command == "render-task-waves":
            value = build_task_waves(plan, manifest, priority=args.priority)
        elif args.command == "render-initial-ledger":
            value = initial_seed_ledger(
                plan, manifest, priority=args.priority
            )
        else:
            value = plan_refill_wave(
                plan,
                manifest,
                read_json(args.ledger),
                priority=args.priority,
            )
    else:
        manifest = read_json(args.manifest)
        statuses_value = json.loads(args.statuses.read_text(encoding="utf-8"))
        if not isinstance(statuses_value, list):
            raise RuntimeError("fast-ramp statuses must be a JSON array")
        value = assess_fast_ramp(manifest, statuses_value)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
