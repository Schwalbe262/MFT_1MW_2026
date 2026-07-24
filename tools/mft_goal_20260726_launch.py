"""Prepare and execute the isolated 2026-07-26 MFT goal campaign.

This entry point deliberately does not import the legacy current7 Slurm
bundle, receipt, or seed-runner contracts.  It produces a source-bound,
authenticated payload that the separate Scheduler project can submit without
mixing either project's implementation or release identity.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_G0_MODEL_TARGETS,
    GOAL_PRIMARY_TURN_STRATA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    GOAL_TEMPERATURE_TARGETS,
    canonical_sha256,
    dynamic_core_group_violation,
    validate_cw1_mm,
    validate_goal_stage_spec,
)
from tools import tier1_corrected_generation_adapter as adapter  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


LOCAL_PREFLIGHT_SCHEMA = "mft-goal-20260726-g0-local-preflight-v1"
BUNDLE_SCHEMA = "mft-goal-20260726-source-bound-bundle-v1"
TASK_PAYLOAD_SCHEMA = "mft-goal-20260726-nsga-task-v1"
SCHEDULER_MANIFEST_SCHEMA = "mft-goal-20260726-scheduler-manifest-v1"
RELOCATION_SCHEMA = "mft-goal-20260726-worker-relocation-v1"
SEARCH_RESULT_SCHEMA = "mft-goal-20260726-search-seed-v1"
GLOBAL_PARETO_SCHEMA = "mft-goal-20260726-global-pareto-v1"
CODE_MANIFEST_SCHEMA = preflight.GOAL_CODE_MANIFEST_SCHEMA
RUNTIME_SOURCE_ROLES = (
    "generation",
    "candidate",
    "quality_status",
    "code_root",
    "dataset",
    "profile",
)
GOAL_RUNTIME_TOOL_FILES = (
    "tools/mft_goal_20260726_launch.py",
    "tools/tier1_corrected_generation_adapter.py",
    "tools/tier1_corrected_generation_preflight.py",
    "tools/tier1_semlock_safe_inference_smoke.py",
)
POPULATION = 320
GENERATIONS = 300
# Pymoo increments ``algorithm.n_gen`` after evaluating the final requested
# generation.  Keep the scientific evolution count and the observed framework
# counter distinct: 300 evaluated generations produce an n_gen counter of 301.
EXPECTED_ALGORITHM_N_GEN_COUNTER = GENERATIONS + 1
INFERENCE_THREADS = 8
ROLLING_SEED_COUNT = 32
CANARY_SEED_COUNT = 4
DEADLINE_KST = "2026-07-26T18:00:00+09:00"
DEADLINE_UTC = "2026-07-26T09:00:00+00:00"
BODY_QUALITY_THRESHOLDS = {
    "min_r2": 0.85,
    "max_rmse": 5.0,
    "max_p90_ape_pct": 10.0,
    "max_interval_p90_width": 10.0,
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("payload already contains a seal")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Any, *, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("sealed payload must be an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} payload seal mismatch")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input must be an object: {path}")
    return value


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [str(path.resolve()), str(root.resolve())]
        ) == str(root.resolve())
    except (OSError, ValueError):
        return False


def _git_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = root.as_posix()
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    return environment


def _collect_goal_code_sources(code_root: Path) -> dict[str, Path]:
    """Collect the committed goal runtime without importing Scheduler code."""

    root = code_root.resolve(strict=True)
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
        env=_git_environment(root),
    )
    if listed.returncode != 0:
        raise RuntimeError("cannot enumerate committed goal runtime code")
    tracked = {
        item.decode("utf-8").replace("\\", "/")
        for item in listed.stdout.split(b"\0")
        if item
    }
    required = set(GOAL_RUNTIME_TOOL_FILES)
    missing = sorted(required - tracked)
    if missing:
        raise RuntimeError(f"required goal runtime code is not committed: {missing}")
    selected = set(required)
    selected.update(
        relative
        for relative in tracked
        if relative.startswith(("module/", "regression_260707/"))
        and PurePosixPath(relative).suffix in {".py", ".json"}
    )
    sources: dict[str, Path] = {}
    for relative in sorted(selected):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"goal runtime code path is unsafe: {relative}")
        source = root.joinpath(*pure.parts).resolve(strict=True)
        if not source.is_file() or not _path_is_below(source, root):
            raise RuntimeError(f"goal runtime code escaped checkout: {relative}")
        sources[f"artifacts/code/{relative}"] = source
    return sources


def _build_goal_code_manifest(
    sources: Mapping[str, Path], *, code_revision: str
) -> dict[str, Any]:
    revision = str(code_revision).lower()
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("goal code revision is invalid")
    records = {
        relative: {
            "sha256": adapter.sha256_file(Path(source)),
            "size": Path(source).stat().st_size,
        }
        for relative, source in sorted(sources.items())
    }
    marker_bytes = f"{revision}\n".encode("ascii")
    records["artifacts/code/.source-revision"] = {
        "sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "size": len(marker_bytes),
    }
    return _seal(
        {
            "schema_version": CODE_MANIFEST_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "code_revision": revision,
            "code_root_relative": "artifacts/code",
            "revision_marker": "artifacts/code/.source-revision",
            "files": records,
            "code_inventory": records,
            "code_inventory_sha256": canonical_sha256(records),
            "staged_path_rule": "bundle_root/<code_inventory_key>",
            "source_checkout_mutated": False,
            "remote_git_checkout_required": False,
            "scheduler_project_code_included": False,
        }
    )


def _stage_goal_code(
    output: Path,
    *,
    sources: Mapping[str, Path],
    manifest: Mapping[str, Any],
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    for relative, source in sorted(sources.items()):
        pure = PurePosixPath(relative)
        destination = output.joinpath(*pure.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source), destination)
    marker = output / "artifacts" / "code" / ".source-revision"
    marker.write_bytes(f"{manifest['code_revision']}\n".encode("ascii"))
    manifest_path = output / "code_manifest.json"
    _atomic_json(manifest_path, manifest)
    preflight.authenticate_goal_code_inventory(
        code_root=output / "artifacts" / "code",
        code_manifest_path=manifest_path,
        expected_manifest_payload_sha256=str(manifest["payload_sha256"]),
        expected_code_inventory_sha256=str(
            manifest["code_inventory_sha256"]
        ),
        expected_code_revision=str(manifest["code_revision"]),
    )
    return manifest_path


def seed_assignments(
    *,
    mode: str,
    seed_start: int,
    fixed_primary_turns: int = 5,
    seed_count: int | None = None,
    wave_size: int = 32,
) -> list[dict[str, Any]]:
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")
    if mode == "single":
        if fixed_primary_turns not in GOAL_PRIMARY_TURN_STRATA:
            raise ValueError("single-seed N1 must be from 5 through 8")
        raw = [(seed_start, int(fixed_primary_turns))]
    elif mode in {"rolling32", "rolling"}:
        count = ROLLING_SEED_COUNT if mode == "rolling32" else seed_count
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < ROLLING_SEED_COUNT
        ):
            raise ValueError("rolling seed_count must be an integer >= 32")
        if (
            isinstance(wave_size, bool)
            or not isinstance(wave_size, int)
            or wave_size < 1
        ):
            raise ValueError("wave_size must be a positive integer")
        raw = [
            (
                seed_start + index,
                GOAL_PRIMARY_TURN_STRATA[index % len(GOAL_PRIMARY_TURN_STRATA)],
            )
            for index in range(count)
        ]
    else:
        raise ValueError("mode must be single, rolling32, or rolling")
    result = []
    for index, (seed, turns) in enumerate(raw):
        canary = mode in {"rolling32", "rolling"} and index < CANARY_SEED_COUNT
        result.append(
            {
                "ordinal": index,
                "seed": seed,
                "fixed_primary_turns": turns,
                "phase": "canary" if canary else (
                    "single" if mode == "single" else "rolling"
                ),
                "wave": (
                    0
                    if canary or mode == "single"
                    else 1 + (index - CANARY_SEED_COUNT) // (
                        4 if mode == "rolling32" else wave_size
                    )
                ),
            }
        )
    return result


def _quality_contract(
    *, quality: Mapping[str, Any], code_root: Path
) -> dict[str, Any]:
    thresholds_path = (
        code_root
        / "regression_260707"
        / "training"
        / "model_quality_thresholds.json"
    ).resolve(strict=True)
    thresholds = _read_json(thresholds_path)
    targets = thresholds.get("targets") or {}
    quality_passed = quality.get("passed")
    if not isinstance(quality_passed, bool):
        raise RuntimeError("goal G0 quality status is not terminal")
    temperature_statuses: dict[str, Any] = {}
    for target in GOAL_TEMPERATURE_TARGETS:
        if targets.get(target) != BODY_QUALITY_THRESHOLDS:
            raise RuntimeError(
                f"goal temperature threshold drifted: {target}"
            )
        status = (quality.get("targets") or {}).get(target) or {}
        metrics = status.get("metrics")
        if (
            not isinstance(status.get("passed"), bool)
            or status.get("blocking") is not True
            or not isinstance(status.get("reasons"), list)
            or not isinstance(metrics, Mapping)
        ):
            raise RuntimeError(
                f"goal temperature quality evidence is incomplete: {target}"
            )
        metric_values = {}
        computed_reasons = []
        for threshold_name, limit in BODY_QUALITY_THRESHOLDS.items():
            metric = threshold_name.removeprefix(
                "min_"
            ).removeprefix("max_")
            try:
                observed = float(metrics[metric])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"goal temperature metric is unavailable: {target}:{metric}"
                ) from exc
            if not math.isfinite(observed):
                raise RuntimeError(
                    f"goal temperature metric is nonfinite: {target}:{metric}"
                )
            metric_values[metric] = observed
            if threshold_name.startswith("min_") and observed < float(limit):
                computed_reasons.append(f"metric_below_minimum:{metric}")
            if threshold_name.startswith("max_") and observed > float(limit):
                computed_reasons.append(f"metric_above_maximum:{metric}")
        if status.get("passed") is not (not status["reasons"]):
            raise RuntimeError(
                f"goal temperature quality status contradicts reasons: {target}"
            )
        if status.get("passed") is True and computed_reasons:
            raise RuntimeError(
                f"goal temperature quality status contradicts metrics: {target}"
            )
        if any(
            reason not in status["reasons"]
            for reason in computed_reasons
        ):
            raise RuntimeError(
                f"goal temperature quality reasons omit threshold failure: {target}"
            )
        temperature_statuses[target] = {
            "passed": status["passed"],
            "reasons": list(status["reasons"]),
            "metrics": metric_values,
            "computed_threshold_reasons": computed_reasons,
        }
    file_sha = adapter.sha256_file(thresholds_path)
    canonical_sha = canonical_sha256(thresholds)
    if (
        quality.get("quality_thresholds_sha256") != file_sha
        and quality.get("thresholds_sha256") != canonical_sha
    ):
        raise RuntimeError("goal G0 quality threshold identity mismatch")
    return {
        "path": str(thresholds_path),
        "file_sha256": file_sha,
        "canonical_sha256": canonical_sha,
        "temperature_target_count": len(GOAL_TEMPERATURE_TARGETS),
        "temperature_thresholds": copy.deepcopy(
            BODY_QUALITY_THRESHOLDS
        ),
        "quality_passed": quality_passed,
        "search_only_proposal": not quality_passed,
        "quality_blockers": list(quality.get("reasons") or []),
        "temperature_status": temperature_statuses,
        "thresholds_lowered_or_bypassed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }


def run_local_preflight(
    *,
    generation: Path,
    candidate: Path,
    quality_status: Path,
    code_root: Path,
    expected_code_revision: str,
) -> tuple[dict[str, Any], preflight.Current7Tier1Runner]:
    import numpy as np

    first = preflight.build_authenticated_runner(
        generation=generation,
        candidate_path=candidate,
        quality_path=quality_status,
        code_root=code_root,
        expected_code_revision=expected_code_revision,
        fixed_primary_turns=GOAL_PRIMARY_TURN_STRATA[0],
        stage_spec=GOAL_STAGE_SPEC,
        inference_threads=INFERENCE_THREADS,
    )
    quality_contract = _quality_contract(
        quality=first.authenticated.quality,
        code_root=code_root,
    )
    runners = {
        str(turns): (
            first
            if turns == GOAL_PRIMARY_TURN_STRATA[0]
            else preflight.runner_for_fixed_primary_turns(first, turns)
        )
        for turns in GOAL_PRIMARY_TURN_STRATA
    }
    strata: dict[str, Any] = {}
    for turns in GOAL_PRIMARY_TURN_STRATA:
        runner = runners[str(turns)]
        raw = np.full((1, runner.problem.n_var), 0.5, dtype=float)
        repaired, repair = runner.repair_coordinates(
            raw, stage="goal_local_preflight"
        )
        evaluated = runner.evaluate_coordinates(repaired)
        if (
            not bool(evaluated["decoder_valid"][0])
            or not np.isfinite(evaluated["F"]).all()
            or not np.isfinite(evaluated["G"]).all()
            or tuple(runner.problem.temperature_targets)
            != GOAL_TEMPERATURE_TARGETS
            or tuple(runner.problem.constraint_names)
            != preflight.GOAL_CONSTRAINT_NAMES
        ):
            raise RuntimeError(f"goal local preflight failed for N1={turns}")
        row = evaluated["frame"].iloc[0]
        observed_turns = int(row["N1_main"]) + int(row["N1_side"])
        if (
            observed_turns != turns
            or dynamic_core_group_violation(row) > 0.0
        ):
            raise RuntimeError(f"goal decoded contract failed for N1={turns}")
        cw1 = validate_cw1_mm(row["cw1"])
        model_smoke = preflight._smoke_every_model(
            runner.models, evaluated["frame"].iloc[[0]]
        )
        strata[str(turns)] = {
            "fixed_primary_turns": turns,
            "cw1_mm": cw1,
            "n_core_group": int(row["n_core_group"]),
            "repair_sha256": repair["sha256"],
            "coordinate_sha256": canonical_sha256(repaired.tolist()),
            "decoded_physical_params_sha256": canonical_sha256(
                preflight._jsonable_decoded_parameters(row)
            ),
            "objective_sha256": canonical_sha256(
                np.asarray(evaluated["F"], dtype=float).tolist()
            ),
            "physical_G_sha256": canonical_sha256(
                np.asarray(evaluated["G"], dtype=float).tolist()
            ),
            "model_smoke_sha256": canonical_sha256(model_smoke),
            "all_24_required_models_exercised": True,
            "decoded_and_finite": True,
        }
    artifacts = first.authenticated.report["artifacts"]
    value = {
        "schema_version": LOCAL_PREFLIGHT_SCHEMA,
        "status": "passed",
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "stage_spec": copy.deepcopy(GOAL_STAGE_SPEC),
        "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": (
            GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
        "hard_constraint_contract_sha256": (
            first.problem.hard_constraint_contract_sha256
        ),
        "operating_point_sha256": FIXED_OPERATING_IDENTITY_SHA256,
        "cooling_contract_sha256": FIXED_COOLING_IDENTITY_SHA256,
        "generation": first.authenticated.evidence["generation_relative"],
        "train_report_sha256": first.authenticated.evidence[
            "train_report"
        ]["sha256"],
        "candidate_sha256": first.authenticated.evidence["candidate"][
            "sha256"
        ],
        "quality_status_sha256": first.authenticated.evidence[
            "quality_status"
        ]["sha256"],
        "dataset_sha256": first.authenticated.evidence["dataset"]["sha256"],
        "profile_sha256": first.authenticated.evidence["profile"][
            "canonical_sha256"
        ],
        "evaluation_model_sha256": canonical_sha256(artifacts),
        "generation_targets": list(GOAL_G0_MODEL_TARGETS),
        "required_model_targets": list(adapter.GOAL_REQUIRED_MODEL_TARGETS),
        "required_model_targets_sha256": (
            adapter.GOAL_REQUIRED_MODEL_TARGETS_SHA256
        ),
        "quality": quality_contract,
        "search_only_proposal": quality_contract[
            "search_only_proposal"
        ],
        "code": copy.deepcopy(first.code_identity),
        "strata": strata,
        "population": POPULATION,
        "generations": GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "legacy_current7_stage_or_release_identity_reused": False,
        "scheduler_write_performed": False,
        "scheduler_submission_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    return _seal(value), first


def build_bundle_values(
    *,
    local_preflight: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    output_root: Path,
    source: Mapping[str, str],
    code_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _validate_seal(dict(local_preflight), schema=LOCAL_PREFLIGHT_SCHEMA)
    validated_code_manifest = _validate_seal(
        dict(code_manifest), schema=CODE_MANIFEST_SCHEMA
    )
    if (
        set(source)
        != {*RUNTIME_SOURCE_ROLES, "expected_code_revision"}
        or source.get("expected_code_revision")
        != validated_code_manifest.get("code_revision")
        or (local_preflight.get("code") or {}).get("revision")
        != validated_code_manifest.get("code_revision")
        or validated_code_manifest.get("code_inventory_sha256")
        != canonical_sha256(validated_code_manifest.get("code_inventory"))
        or validated_code_manifest.get("staged_path_rule")
        != "bundle_root/<code_inventory_key>"
        or validated_code_manifest.get("source_checkout_mutated") is not False
        or validated_code_manifest.get("remote_git_checkout_required")
        is not False
        or validated_code_manifest.get("scheduler_project_code_included")
        is not False
    ):
        raise RuntimeError("goal bundle source/code manifest mismatch")
    tasks: list[dict[str, Any]] = []
    for assignment in assignments:
        seed = int(assignment["seed"])
        turns = int(assignment["fixed_primary_turns"])
        task = _seal(
            {
                "schema_version": TASK_PAYLOAD_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "task_name": f"mft-goal-s{seed}-n1-{turns}",
                **copy.deepcopy(assignment),
                "population": POPULATION,
                "generations": GENERATIONS,
                "inference_threads": INFERENCE_THREADS,
                "stage_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "hard_constraint_contract_sha256": local_preflight[
                    "hard_constraint_contract_sha256"
                ],
                "search_only_proposal": local_preflight[
                    "search_only_proposal"
                ],
                "dataset_sha256": local_preflight["dataset_sha256"],
                "evaluation_model_sha256": local_preflight[
                    "evaluation_model_sha256"
                ],
                "operating_point_sha256": FIXED_OPERATING_IDENTITY_SHA256,
                "cooling_contract_sha256": FIXED_COOLING_IDENTITY_SHA256,
                "source": dict(source),
                "source_identity": {
                    "train_report_sha256": local_preflight[
                        "train_report_sha256"
                    ],
                    "candidate_sha256": local_preflight[
                        "candidate_sha256"
                    ],
                    "quality_status_sha256": local_preflight[
                        "quality_status_sha256"
                    ],
                    "dataset_sha256": local_preflight["dataset_sha256"],
                    "profile_sha256": local_preflight["profile_sha256"],
                    "evaluation_model_sha256": local_preflight[
                        "evaluation_model_sha256"
                    ],
                    "code_revision": local_preflight["code"]["revision"],
                    "code_manifest_payload_sha256": (
                        validated_code_manifest["payload_sha256"]
                    ),
                    "code_inventory_sha256": validated_code_manifest[
                        "code_inventory_sha256"
                    ],
                },
                "result_schema_version": SEARCH_RESULT_SCHEMA,
                "terminal_table_schema_version": (
                    preflight.GOAL_TERMINAL_TABLE_SCHEMA
                ),
                "resources": {
                    "cpus": INFERENCE_THREADS,
                    "memory_GiB": 64,
                    "gpus": 0,
                    "optimizer_processes": 1,
                },
                "legacy_current7_stage_or_release_identity_reused": False,
                "production_eligible": False,
                "automatic_promotion_allowed": False,
            }
        )
        tasks.append(task)
    bundle = _seal(
        {
            "schema_version": BUNDLE_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "deadline_kst": DEADLINE_KST,
            "deadline_utc": DEADLINE_UTC,
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": local_preflight[
                "hard_constraint_contract_sha256"
            ],
            "local_preflight_sha256": local_preflight["payload_sha256"],
            "search_only_proposal": local_preflight[
                "search_only_proposal"
            ],
            "task_count": len(tasks),
            "task_payload_sha256": [
                task["payload_sha256"] for task in tasks
            ],
            "task_relative_paths": [
                f"tasks/seed-{task['seed']}-n1-{task['fixed_primary_turns']}.json"
                for task in tasks
            ],
            "source_bound_paths": dict(source),
            "code_manifest": {
                "path": "code_manifest.json",
                "schema_version": CODE_MANIFEST_SCHEMA,
                "payload_sha256": validated_code_manifest["payload_sha256"],
                "code_inventory_sha256": validated_code_manifest[
                    "code_inventory_sha256"
                ],
                "code_revision": validated_code_manifest["code_revision"],
            },
            "relocation_contract": {
                "schema_version": RELOCATION_SCHEMA,
                "mode": (
                    "source_paths_or_explicit_worker_role_path_relocation"
                ),
                "roles": list(RUNTIME_SOURCE_ROLES),
                "code_manifest_path_required_for_relocation": True,
                "relocation_files_emitted": False,
                "stager_must_generate_task_bound_relocation": True,
                "source_absolute_paths_are_not_worker_authority": True,
                "relocated_paths_must_reauthenticate_all_bound_SHA256": True,
                "task_payload_sha256_must_match": True,
                "worker_override_file_supported": True,
            },
            "scheduler_project_modified": False,
            "legacy_current7_bundle_or_release_identity_reused": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    canary = [task["seed"] for task in tasks if task["phase"] == "canary"]
    rolling_waves = {}
    for task in tasks:
        if task["phase"] == "rolling":
            rolling_waves.setdefault(str(task["wave"]), []).append(task["seed"])
    scheduler = _seal(
        {
            "schema_version": SCHEDULER_MANIFEST_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "bundle_sha256": bundle["payload_sha256"],
            "maximum_parallel_tasks": len(tasks),
            "canary_seed_count": len(canary),
            "canary_seeds": canary,
            "canary_policy": (
                "submit_one_seed_per_N1_stratum_then_release_rolling_waves"
                if canary
                else "single_seed_local_preflight_then_submit"
            ),
            "rolling_waves": rolling_waves,
            "resources_per_task": {
                "cpus": INFERENCE_THREADS,
                "memory_GiB": 64,
                "optimizer_processes": 1,
            },
            "command_template": [
                "python3",
                "tools/mft_goal_20260726_launch.py",
                "execute-seed",
                "--payload",
                "<task-payload-path>",
                "--output",
                "<task-output-directory>",
                "--relocation",
                "<optional-worker-relocation-json>",
            ],
            "deadline_kst": DEADLINE_KST,
            "search_only_proposal": local_preflight[
                "search_only_proposal"
            ],
            "do_not_modify_scheduler_project_from_this_cli": True,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    return bundle, tasks, scheduler


def validate_task_payload(value: Any) -> dict[str, Any]:
    task = _validate_seal(value, schema=TASK_PAYLOAD_SCHEMA)
    validate_goal_stage_spec(task.get("stage_spec"))
    source_identity = task.get("source_identity") or {}
    digests = [
        source_identity.get(name)
        for name in (
            "train_report_sha256",
            "candidate_sha256",
            "quality_status_sha256",
            "dataset_sha256",
            "profile_sha256",
            "evaluation_model_sha256",
            "code_manifest_payload_sha256",
            "code_inventory_sha256",
        )
    ]
    if (
        task.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or task.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or task.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or task.get("population") != POPULATION
        or task.get("generations") != GENERATIONS
        or task.get("inference_threads") != INFERENCE_THREADS
        or task.get("fixed_primary_turns")
        not in GOAL_PRIMARY_TURN_STRATA
        or task.get("legacy_current7_stage_or_release_identity_reused")
        is not False
        or not isinstance(task.get("search_only_proposal"), bool)
        or task.get("production_eligible") is not False
        or task.get("automatic_promotion_allowed") is not False
        or set(task.get("source") or {})
        != {*RUNTIME_SOURCE_ROLES, "expected_code_revision"}
        or any(
            not isinstance((task.get("source") or {}).get(name), str)
            or not (task.get("source") or {})[name]
            for name in (*RUNTIME_SOURCE_ROLES, "expected_code_revision")
        )
        or set(task.get("source_identity") or {})
        != {
            "train_report_sha256",
            "candidate_sha256",
            "quality_status_sha256",
            "dataset_sha256",
            "profile_sha256",
            "evaluation_model_sha256",
            "code_revision",
            "code_manifest_payload_sha256",
            "code_inventory_sha256",
        }
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or not isinstance(source_identity.get("code_revision"), str)
        or len(source_identity["code_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_identity["code_revision"]
        )
        or source_identity["code_revision"]
        != (task.get("source") or {}).get("expected_code_revision")
        or task.get("dataset_sha256")
        != source_identity.get("dataset_sha256")
        or task.get("evaluation_model_sha256")
        != source_identity.get("evaluation_model_sha256")
    ):
        raise RuntimeError("goal task payload contract mismatch")
    return task


def resolve_worker_source(
    task: Mapping[str, Any], relocation_path: Path | None
) -> tuple[dict[str, str], Path | None]:
    source = dict(task["source"])
    if relocation_path is None:
        return source, None
    relocation = _validate_seal(
        _read_json(relocation_path.resolve(strict=True)),
        schema=RELOCATION_SCHEMA,
    )
    paths = relocation.get("paths")
    code_manifest_path = relocation.get("code_manifest_path")
    if (
        set(relocation)
        != {
            "schema_version",
            "task_payload_sha256",
            "paths",
            "code_manifest_path",
            "source_absolute_paths_are_documentary_only",
            "remote_git_checkout_required",
            "payload_sha256",
        }
        or relocation.get("task_payload_sha256") != task["payload_sha256"]
        or not isinstance(paths, dict)
        or set(paths) != set(RUNTIME_SOURCE_ROLES)
        or any(
            not isinstance(paths.get(name), str) or not paths[name]
            or not Path(paths[name]).is_absolute()
            for name in RUNTIME_SOURCE_ROLES
        )
        or not isinstance(code_manifest_path, str)
        or not code_manifest_path
        or not Path(code_manifest_path).is_absolute()
        or relocation.get("source_absolute_paths_are_documentary_only")
        is not True
        or relocation.get("remote_git_checkout_required") is not False
    ):
        raise RuntimeError("worker relocation contract mismatch")
    source.update({name: str(paths[name]) for name in paths})
    return source, Path(code_manifest_path)


def prepare_bundle(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    code_root = args.code_root.resolve(strict=True)
    if output.exists():
        raise RuntimeError("goal bundle output already exists")
    if _path_is_below(output, code_root):
        raise RuntimeError("goal bundle output must be outside the clean code root")
    local_preflight, runner = run_local_preflight(
        generation=args.generation,
        candidate=args.candidate,
        quality_status=args.quality_status,
        code_root=code_root,
        expected_code_revision=args.expected_code_revision,
    )
    code_sources = _collect_goal_code_sources(code_root)
    code_manifest = _build_goal_code_manifest(
        code_sources,
        code_revision=local_preflight["code"]["revision"],
    )
    _stage_goal_code(
        output,
        sources=code_sources,
        manifest=code_manifest,
    )
    documentary_generation = str(
        runner.authenticated.candidate.get("generation_path") or ""
    )
    if not documentary_generation:
        raise RuntimeError("goal candidate documentary generation path is missing")
    source = {
        "generation": documentary_generation,
        "candidate": str(args.candidate.resolve(strict=True)),
        "quality_status": str(args.quality_status.resolve(strict=True)),
        "code_root": str(code_root),
        "dataset": str(runner.authenticated.dataset_path),
        "profile": str(runner.authenticated.profile_path),
        "expected_code_revision": local_preflight["code"]["revision"],
    }
    assignments = seed_assignments(
        mode=args.mode,
        seed_start=args.seed_start,
        fixed_primary_turns=args.fixed_primary_turns,
        seed_count=args.seed_count,
        wave_size=args.wave_size,
    )
    bundle, tasks, scheduler = build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignments,
        output_root=output,
        source=source,
        code_manifest=code_manifest,
    )
    _atomic_json(output / "local_preflight.json", local_preflight)
    for task in tasks:
        path = (
            output
            / "tasks"
            / f"seed-{task['seed']}-n1-{task['fixed_primary_turns']}.json"
        )
        _atomic_json(path, task)
    _atomic_json(output / "scheduler_manifest.json", scheduler)
    _atomic_json(output / "bundle_manifest.json", bundle)
    return output / "bundle_manifest.json"


def execute_seed(args: argparse.Namespace) -> Path:
    task = validate_task_payload(_read_json(args.payload.resolve(strict=True)))
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("goal seed output already exists")
    source, relocated_code_manifest = resolve_worker_source(
        task, args.relocation
    )
    relocated = relocated_code_manifest is not None
    runner = preflight.build_authenticated_runner(
        generation=Path(source["generation"]),
        candidate_path=Path(source["candidate"]),
        quality_path=Path(source["quality_status"]),
        code_root=Path(source["code_root"]),
        expected_code_revision=source["expected_code_revision"],
        fixed_primary_turns=int(task["fixed_primary_turns"]),
        stage_spec=task["stage_spec"],
        inference_threads=INFERENCE_THREADS,
        dataset_path_override=(
            Path(source["dataset"]) if relocated else None
        ),
        profile_path_override=(
            Path(source["profile"]) if relocated else None
        ),
        expected_documentary_generation_path=(
            str(task["source"]["generation"]) if relocated else None
        ),
        code_manifest_path=relocated_code_manifest,
        expected_code_manifest_payload_sha256=(
            task["source_identity"]["code_manifest_payload_sha256"]
            if relocated
            else None
        ),
        expected_code_inventory_sha256=(
            task["source_identity"]["code_inventory_sha256"]
            if relocated
            else None
        ),
    )
    source_identity = task["source_identity"]
    if (
        runner.authenticated.evidence["train_report"]["sha256"]
        != source_identity["train_report_sha256"]
        or runner.authenticated.evidence["candidate"]["sha256"]
        != source_identity["candidate_sha256"]
        or runner.authenticated.evidence["quality_status"]["sha256"]
        != source_identity["quality_status_sha256"]
        or runner.authenticated.evidence["dataset"]["sha256"]
        != source_identity["dataset_sha256"]
        or runner.authenticated.evidence["profile"]["canonical_sha256"]
        != source_identity["profile_sha256"]
        or canonical_sha256(runner.authenticated.report["artifacts"])
        != source_identity["evaluation_model_sha256"]
        or runner.code_identity["revision"] != source_identity["code_revision"]
        or (
            relocated
            and runner.code_identity.get("code_inventory_sha256")
            != source_identity["code_inventory_sha256"]
        )
        or (
            relocated
            and (
                runner.code_identity.get("code_manifest") or {}
            ).get("payload_sha256")
            != source_identity["code_manifest_payload_sha256"]
        )
        or runner.problem.hard_constraint_contract_sha256
        != task["hard_constraint_contract_sha256"]
    ):
        raise RuntimeError("goal task input identity changed after bundling")
    output.mkdir(parents=True)
    physical_evaluate, scaling = preflight.install_optimizer_scaling(
        runner.problem,
        resonance_scale_hz=150.0,
        llt_scale_uh=0.55,
        all_thermal_scale_c=10.0,
    )
    runner.physical_evaluate = physical_evaluate
    runner.optimizer_scaling = scaling
    repair = runner.install_offspring_repair_operator(
        topology_contract=preflight.goal_topology_contract(
            int(task["fixed_primary_turns"])
        )
    )

    def seal_pre_optimization(value: Mapping[str, Any]) -> None:
        if (
            value.get("offspring_repair_operator_installed") is not True
            or value.get("optimizer_execution_started") is not False
        ):
            raise RuntimeError("goal pre-optimization gate failed")
        _atomic_json(
            output / "pre_optimization.json",
            _seal(
                {
                    "schema_version": (
                        "mft-goal-20260726-pre-optimization-v1"
                    ),
                    "task_payload_sha256": task["payload_sha256"],
                    "evidence": copy.deepcopy(dict(value)),
                    "repair_installation": repair,
                    "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                }
            ),
        )

    result = runner.run_one(
        seed=int(task["seed"]),
        population=POPULATION,
        max_generations=GENERATIONS,
        optimizer_termination_strategy=(
            preflight.FIXED_GENERATION_TERMINATION_STRATEGY
        ),
        pre_optimization_callback=seal_pre_optimization,
    )
    artifacts = preflight.persist_search_outputs(
        runner,
        result,
        output,
        source_identity={
            "seed": int(task["seed"]),
            "task_id": str(
                os.environ.get("SLURM_SCHED_TASK_ID")
                or task["task_name"]
            ),
            "bundle_id": task["payload_sha256"],
            "island_id": f"n1-{task['fixed_primary_turns']}",
            "dataset_sha256": task["dataset_sha256"],
            "model_artifacts_sha256": task["evaluation_model_sha256"],
            "model_generation_sha256": runner.authenticated.evidence[
                "train_report"
            ]["sha256"],
            "evaluation_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
        },
    )
    result_value = _seal(
        {
            "schema_version": SEARCH_RESULT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "task_payload_sha256": task["payload_sha256"],
            "seed": int(task["seed"]),
            "fixed_primary_turns": int(task["fixed_primary_turns"]),
            "population": POPULATION,
            "generations": GENERATIONS,
            "evaluated_generations": int(
                result.tier1_evaluated_generations
            ),
            "completed_generations": int(
                result.tier1_completed_generations
            ),
            "stage_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "constraint_version": (
                runner.problem.hard_constraint_contract["stage"]
            ),
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            "dataset_sha256": task["dataset_sha256"],
            "evaluation_model_sha256": task["evaluation_model_sha256"],
            "operating_point_sha256": FIXED_OPERATING_IDENTITY_SHA256,
            "cooling_contract_sha256": FIXED_COOLING_IDENTITY_SHA256,
            "constraint_names": list(runner.problem.constraint_names),
            "temperature_targets": list(GOAL_TEMPERATURE_TARGETS),
            "optimizer_scaling_contract": scaling,
            "optimizer_repair_audit": result.tier1_repair_audit,
            "optimizer_topology_evolution_audit": (
                result.tier1_topology_evolution_audit
            ),
            "terminal_population_count": artifacts[
                "terminal_population_count"
            ],
            "physical_feasible_count": artifacts[
                "physical_feasible_count"
            ],
            "feasible_pareto_count": artifacts[
                "feasible_pareto_count"
            ],
            "artifact_inventory": artifacts["artifact_inventory"],
            "artifact_inventory_sha256": artifacts[
                "artifact_inventory_sha256"
            ],
            "terminal_physical_candidates_manifest": artifacts[
                "terminal_physical_candidates_manifest"
            ],
            "legacy_current7_stage_or_release_identity_reused": False,
            "search_only_proposal": task["search_only_proposal"],
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        }
    )
    path = output / "result.json"
    _atomic_json(path, result_value)
    return path


def _contained_relative_file(root: Path, value: Any, label: str) -> Path:
    relative = PurePosixPath(str(value or ""))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise RuntimeError(f"{label} path is unsafe")
    base = root.resolve(strict=True)
    path = base.joinpath(*relative.parts).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped its authenticated root") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable")
    return path


def load_bundle_task_ledger(
    bundle_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Authenticate one original bundle and its complete sealed task ledger."""

    manifest_path = bundle_manifest_path.resolve(strict=True)
    bundle = _validate_seal(
        _read_json(manifest_path),
        schema=BUNDLE_SCHEMA,
    )
    paths = bundle.get("task_relative_paths")
    payloads = bundle.get("task_payload_sha256")
    count = bundle.get("task_count")
    code_record = bundle.get("code_manifest") or {}
    if (
        bundle.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or bundle.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or bundle.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(paths, list)
        or not isinstance(payloads, list)
        or len(paths) != count
        or len(payloads) != count
        or len(set(paths)) != count
        or len(set(payloads)) != count
        or code_record.get("schema_version") != CODE_MANIFEST_SCHEMA
    ):
        raise RuntimeError("goal bundle task ledger contract mismatch")
    code_manifest_path = _contained_relative_file(
        manifest_path.parent,
        code_record.get("path"),
        "goal code manifest",
    )
    code_manifest = _validate_seal(
        _read_json(code_manifest_path),
        schema=CODE_MANIFEST_SCHEMA,
    )
    if (
        code_manifest.get("payload_sha256")
        != code_record.get("payload_sha256")
        or code_manifest.get("code_inventory_sha256")
        != code_record.get("code_inventory_sha256")
        or code_manifest.get("code_revision")
        != code_record.get("code_revision")
    ):
        raise RuntimeError("goal bundle code manifest identity mismatch")

    ledger: dict[str, dict[str, Any]] = {}
    seeds: set[int] = set()
    turns: list[int] = []
    for relative, expected_payload in zip(paths, payloads):
        task_path = _contained_relative_file(
            manifest_path.parent,
            relative,
            "goal task payload",
        )
        task = validate_task_payload(_read_json(task_path))
        if (
            task["payload_sha256"] != expected_payload
            or task.get("source") != bundle.get("source_bound_paths")
            or task["source_identity"]["code_manifest_payload_sha256"]
            != code_manifest["payload_sha256"]
            or task["source_identity"]["code_inventory_sha256"]
            != code_manifest["code_inventory_sha256"]
            or task["source_identity"]["code_revision"]
            != code_manifest["code_revision"]
            or task.get("search_only_proposal")
            != bundle.get("search_only_proposal")
            or PurePosixPath(str(relative)).as_posix()
            != (
                f"tasks/seed-{task['seed']}-"
                f"n1-{task['fixed_primary_turns']}.json"
            )
        ):
            raise RuntimeError("goal bundle task ledger identity mismatch")
        seed = int(task["seed"])
        if seed in seeds:
            raise RuntimeError("goal bundle task ledger has duplicate seeds")
        seeds.add(seed)
        turns.append(int(task["fixed_primary_turns"]))
        ledger[task["payload_sha256"]] = task
    if len(ledger) != count:
        raise RuntimeError("goal bundle task ledger is incomplete")
    if count > 1 and set(turns) != set(GOAL_PRIMARY_TURN_STRATA):
        raise RuntimeError("goal bundle task ledger lacks all N1 strata")
    return bundle, ledger


def _validated_seed_table(
    result_path: Path,
    *,
    expected_task: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    import numpy as np
    import pandas as pd

    path = result_path.resolve(strict=True)
    result = _validate_seal(_read_json(path), schema=SEARCH_RESULT_SCHEMA)
    if (
        result.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or result.get("stage_spec") != GOAL_STAGE_SPEC
        or result.get("hard_spec") != GOAL_STAGE_SPEC
        or result.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or result.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or not isinstance(result.get("constraint_version"), str)
        or not result.get("constraint_version")
        or result.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or result.get("constraint_names")
        != list(preflight.GOAL_CONSTRAINT_NAMES)
        or result.get("temperature_targets")
        != list(GOAL_TEMPERATURE_TARGETS)
        or result.get("task_payload_sha256")
        != expected_task.get("payload_sha256")
        or result.get("seed") != expected_task.get("seed")
        or result.get("fixed_primary_turns")
        != expected_task.get("fixed_primary_turns")
        or result.get("population") != POPULATION
        or result.get("population") != expected_task.get("population")
        or result.get("generations") != GENERATIONS
        or result.get("generations") != expected_task.get("generations")
        or isinstance(result.get("evaluated_generations"), bool)
        or not isinstance(result.get("evaluated_generations"), int)
        or result.get("evaluated_generations") != GENERATIONS
        or isinstance(result.get("completed_generations"), bool)
        or not isinstance(result.get("completed_generations"), int)
        or result.get("completed_generations")
        != EXPECTED_ALGORITHM_N_GEN_COUNTER
        or result.get("terminal_population_count") != POPULATION
        or not isinstance(result.get("search_only_proposal"), bool)
        or result.get("search_only_proposal")
        != expected_task.get("search_only_proposal")
        or result.get("dataset_sha256")
        != expected_task.get("dataset_sha256")
        or result.get("evaluation_model_sha256")
        != expected_task.get("evaluation_model_sha256")
        or result.get("hard_constraint_contract_sha256")
        != expected_task.get("hard_constraint_contract_sha256")
        or result.get("cooling_contract_sha256")
        != expected_task.get("cooling_contract_sha256")
        or result.get("operating_point_sha256")
        != expected_task.get("operating_point_sha256")
        or result.get("production_eligible") is not False
        or result.get("automatic_promotion_allowed") is not False
        or result.get("legacy_current7_stage_or_release_identity_reused")
        is not False
    ):
        raise RuntimeError(f"goal seed result contract mismatch: {path}")
    inventory = result.get("artifact_inventory") or {}
    if result.get("artifact_inventory_sha256") != canonical_sha256(inventory):
        raise RuntimeError(f"goal seed artifact inventory mismatch: {path}")
    table_record = inventory.get("terminal_physical_candidates") or {}
    manifest_record = inventory.get(
        "terminal_physical_candidates_manifest"
    ) or {}
    table_path = _contained_relative_file(
        path.parent,
        table_record.get("path"),
        "goal terminal table",
    )
    manifest_path = _contained_relative_file(
        path.parent,
        manifest_record.get("path"),
        "goal terminal table manifest",
    )
    if (
        adapter.sha256_file(table_path) != table_record.get("sha256")
        or adapter.sha256_file(manifest_path) != manifest_record.get("sha256")
    ):
        raise RuntimeError(f"goal terminal table artifact mismatch: {path}")
    table_manifest = _validate_seal(
        _read_json(manifest_path),
        schema=preflight.GOAL_TERMINAL_TABLE_SCHEMA,
    )
    if (
        table_manifest != result.get("terminal_physical_candidates_manifest")
        or table_manifest.get("row_count") != POPULATION
        or (table_manifest.get("csv") or {}).get("sha256")
        != table_record.get("sha256")
        or table_manifest.get("physical_deduplication_key")
        != "physical_geometry_sha256"
    ):
        raise RuntimeError(f"goal terminal table manifest mismatch: {path}")
    table = pd.read_csv(table_path)
    required = set(table_manifest.get("required_identity_columns") or [])
    canonical_required = {
        "terminal_population_index",
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
        "physical_constraint_feasible",
        "physical_feasible",
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
        "objective_volume_L",
        "objective_total_loss_W",
    }
    physical_columns = [
        f"physical_G:{name}" for name in result.get("constraint_names") or []
    ]
    normalized_columns = [
        f"normalized_G:{name}" for name in result.get("constraint_names") or []
    ]
    common = {
        "dataset_sha256": result["dataset_sha256"],
        "evaluation_model_sha256": result["evaluation_model_sha256"],
        "evaluation_model_artifacts_sha256": result[
            "evaluation_model_sha256"
        ],
        "constraint_spec_sha256": result["stage_spec_sha256"],
        "evaluation_spec_sha256": result["stage_spec_sha256"],
        "evaluation_temperature_contract_sha256": result[
            "temperature_contract_sha256"
        ],
        "evaluation_hard_constraint_contract_sha256": result[
            "hard_constraint_contract_sha256"
        ],
        "cooling_contract_sha256": result["cooling_contract_sha256"],
        "operating_point_sha256": result["operating_point_sha256"],
        "source_bundle_id": result["task_payload_sha256"],
        "source_island_id": (
            f"n1-{int(expected_task['fixed_primary_turns'])}"
        ),
        "evaluation_model_generation_sha256": expected_task[
            "source_identity"
        ]["train_report_sha256"],
    }
    if (
        len(table) != POPULATION
        or table["terminal_population_index"].tolist()
        != list(range(POPULATION))
        or not canonical_required.issubset(required)
        or not required.issubset(table.columns)
        or table_manifest.get("columns") != list(table.columns)
        or table_manifest.get("goal_contract_required") is not True
        or table_manifest.get("stage_spec_sha256")
        != GOAL_STAGE_SPEC_SHA256
        or table_manifest.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or table_manifest.get("hard_constraint_contract_sha256")
        != result["hard_constraint_contract_sha256"]
        or not set(physical_columns + normalized_columns).issubset(
            table.columns
        )
        or table["source_seed"].nunique() != 1
        or int(table["source_seed"].iloc[0]) != int(result["seed"])
        or table["source_island_id"].isna().any()
        or table["source_task_id"].isna().any()
        or table["evaluation_model_generation_sha256"].isna().any()
        or not (
            table["candidate_physics_sha"]
            == table["physical_geometry_sha256"]
        ).all()
        or any(
            table[name].isna().any()
            or set(table[name].astype(str)) != {str(expected)}
            for name, expected in common.items()
        )
        or not np.isfinite(
            table[
                [
                    "objective_volume_L",
                    "objective_total_loss_W",
                    *physical_columns,
                    *normalized_columns,
                ]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise RuntimeError(f"goal terminal table content mismatch: {path}")
    try:
        decoded_turns = []
        for value in table["decoded_physical_params_json"]:
            decoded = json.loads(value)
            main = float(decoded["N1_main"])
            side = float(decoded["N1_side"])
            if (
                not math.isfinite(main)
                or not math.isfinite(side)
                or not main.is_integer()
                or not side.is_integer()
            ):
                raise ValueError("decoded primary turns are not integers")
            decoded_turns.append(int(main) + int(side))
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OverflowError,
    ) as exc:
        raise RuntimeError(
            f"goal terminal decoded N1 evidence mismatch: {path}"
        ) from exc
    if set(decoded_turns) != {int(expected_task["fixed_primary_turns"])}:
        raise RuntimeError(f"goal terminal decoded N1 evidence mismatch: {path}")
    flag_columns = (
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
        "physical_constraint_feasible",
        "physical_feasible",
    )
    if any(
        not pd.api.types.is_bool_dtype(table[name])
        for name in flag_columns
    ):
        raise RuntimeError(f"goal terminal boolean evidence mismatch: {path}")
    physical_values = table[physical_columns].to_numpy(dtype=float)
    expected_constraint_feasible = (
        table["decoder_valid"].to_numpy(dtype=bool)
        & np.all(physical_values <= 0.0, axis=1)
    )
    surrogate_valid = table["surrogate_physical_valid"].to_numpy(
        dtype=bool
    )
    expected_physical_feasible = (
        expected_constraint_feasible & surrogate_valid
    )
    if (
        not table["decoder_valid"].all()
        or not np.array_equal(
            table["surrogate_physicality_passed"].to_numpy(dtype=bool),
            surrogate_valid,
        )
        or not np.array_equal(
            table["physical_constraint_feasible"].to_numpy(dtype=bool),
            expected_constraint_feasible,
        )
        or not np.array_equal(
            table["physical_feasible"].to_numpy(dtype=bool),
            expected_physical_feasible,
        )
        or int(result.get("physical_feasible_count") or 0)
        != int(expected_physical_feasible.sum())
    ):
        raise RuntimeError(f"goal terminal physicality evidence mismatch: {path}")
    table = table.copy()
    table["source_result_path"] = str(path)
    table["source_result_sha256"] = adapter.sha256_file(path)
    return result, table


def _atomic_csv(path: Path, frame: Any) -> None:
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def aggregate_results(
    *,
    result_paths: list[Path],
    bundle_manifest_path: Path,
    output: Path,
    minimum_seeds: int,
) -> Path:
    import numpy as np
    import pandas as pd
    from tools.mft_goal_global_pareto import (
        rank_candidates,
        select_standard_candidates,
    )

    if (
        isinstance(minimum_seeds, bool)
        or not isinstance(minimum_seeds, int)
        or minimum_seeds < 2
    ):
        raise ValueError("minimum_seeds must be an integer >= 2")
    resolved = sorted(path.resolve(strict=True) for path in result_paths)
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("global Pareto inputs contain duplicate result paths")
    bundle, task_ledger = load_bundle_task_ledger(bundle_manifest_path)
    if len(resolved) < minimum_seeds:
        raise RuntimeError("authenticated seed result count is below minimum")
    if len(resolved) != len(task_ledger):
        raise RuntimeError(
            "global Pareto requires one result for every original bundle task"
        )
    output = output.resolve()
    if output.exists():
        raise RuntimeError("global Pareto output already exists")
    results = []
    frames = []
    observed_tasks: set[str] = set()
    for path in resolved:
        envelope = _validate_seal(
            _read_json(path),
            schema=SEARCH_RESULT_SCHEMA,
        )
        task_sha = str(envelope.get("task_payload_sha256") or "")
        expected_task = task_ledger.get(task_sha)
        if expected_task is None:
            raise RuntimeError(
                "global Pareto result is not in the original task ledger"
            )
        if task_sha in observed_tasks:
            raise RuntimeError(
                "global Pareto inputs contain duplicate task results"
            )
        observed_tasks.add(task_sha)
        result, frame = _validated_seed_table(
            path,
            expected_task=expected_task,
        )
        results.append(result)
        frames.append(frame)
    if observed_tasks != set(task_ledger):
        raise RuntimeError("global Pareto original task results are incomplete")
    seeds = [int(result["seed"]) for result in results]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("global Pareto inputs contain duplicate seeds")
    common_fields = (
        "dataset_sha256",
        "evaluation_model_sha256",
        "stage_spec_sha256",
        "temperature_contract_sha256",
        "hard_constraint_contract_sha256",
        "cooling_contract_sha256",
        "operating_point_sha256",
    )
    common_identity = {
        name: results[0][name] for name in common_fields
    }
    if any(
        any(result.get(name) != expected for name, expected in common_identity.items())
        for result in results
    ):
        raise RuntimeError("global Pareto seed identities are mixed")
    merged = pd.concat(frames, ignore_index=True)
    physical_columns = [
        column for column in merged.columns if column.startswith("physical_G:")
    ]
    normalized_columns = [
        column for column in merged.columns if column.startswith("normalized_G:")
    ]
    if (
        len(merged) != POPULATION * len(results)
        or not physical_columns
        or not normalized_columns
    ):
        raise RuntimeError("global terminal table inventory is incomplete")
    compare_columns = [
        "objective_volume_L",
        "objective_total_loss_W",
        "surrogate_physical_valid",
        "physical_constraint_feasible",
        "physical_feasible",
        *physical_columns,
        *normalized_columns,
    ]
    for _geometry, group in merged.groupby(
        "physical_geometry_sha256", sort=False
    ):
        values = group[compare_columns].to_numpy(dtype=float)
        if len(values) > 1 and not np.allclose(
            values,
            values[[0]],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            raise RuntimeError(
                "duplicate physical geometry has inconsistent evaluation"
            )
    deduplicated = merged.drop_duplicates(
        "physical_geometry_sha256", keep="first"
    ).copy()
    ranked = rank_candidates(
        deduplicated,
        objective_columns=(
            "objective_volume_L",
            "objective_total_loss_W",
        ),
        physical_constraint_columns=physical_columns,
        normalized_constraint_columns=normalized_columns,
    )
    ranked["global_physical_feasible"] = ranked["hard_feasible"]
    ranked["global_non_dominated_rank"] = ranked["feasible_rank"]
    feasible = ranked["hard_feasible"].to_numpy(dtype=bool)
    feasible_ranks = ranked.loc[
        ranked["feasible_rank"].ge(0), "feasible_rank"
    ]
    front_count = (
        0 if feasible_ranks.empty else int(feasible_ranks.max()) + 1
    )
    pareto = ranked.loc[
        ranked["global_non_dominated_rank"].eq(0)
    ].copy()
    pareto.sort_values(
        ["objective_volume_L", "objective_total_loss_W"],
        inplace=True,
    )
    standard_candidates = select_standard_candidates(
        ranked,
        objective_columns=(
            "objective_volume_L",
            "objective_total_loss_W",
        ),
        normalized_constraint_columns=normalized_columns,
    )
    output.mkdir(parents=True)
    all_path = output / "global_terminal_candidates.csv"
    pareto_path = output / "global_pareto_front.csv"
    standard_path = output / "standard_candidates.csv"
    _atomic_csv(all_path, ranked)
    _atomic_csv(pareto_path, pareto)
    _atomic_csv(standard_path, standard_candidates)
    manifest = _seal(
        {
            "schema_version": GLOBAL_PARETO_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "seed_count": len(results),
            "minimum_seed_count": minimum_seeds,
            "seeds": sorted(seeds),
            "input_terminal_row_count": int(len(merged)),
            "deduplicated_physical_geometry_count": int(len(ranked)),
            "physical_feasible_count": int(feasible.sum()),
            "non_dominated_front_count": front_count,
            "global_pareto_count": int(len(pareto)),
            "standard_candidate_count": int(len(standard_candidates)),
            "sorting_authority": (
                "all_authenticated_terminal_rows_then_physical_dedupe_"
                "then_decoder_and_physical_G_and_surrogate_physicality_"
                "feasible_then_exact_2d_nlogn_non_dominated_sort"
            ),
            "seed_local_pareto_merge_used": False,
            "common_identity": common_identity,
            "inputs": [
                {
                    "path": str(path),
                    "sha256": adapter.sha256_file(path),
                }
                for path in resolved
            ],
            "authenticated_bundle": {
                "path": str(bundle_manifest_path.resolve(strict=True)),
                "sha256": adapter.sha256_file(
                    bundle_manifest_path.resolve(strict=True)
                ),
                "payload_sha256": bundle["payload_sha256"],
                "task_count": bundle["task_count"],
                "task_ledger_sha256": canonical_sha256(
                    sorted(task_ledger)
                ),
                "all_four_N1_strata_covered": (
                    {
                        int(task["fixed_primary_turns"])
                        for task in task_ledger.values()
                    }
                    == set(GOAL_PRIMARY_TURN_STRATA)
                ),
            },
            "artifacts": {
                "global_terminal_candidates": {
                    "path": all_path.name,
                    "sha256": adapter.sha256_file(all_path),
                    "row_count": int(len(deduplicated)),
                },
                "global_pareto_front": {
                    "path": pareto_path.name,
                    "sha256": adapter.sha256_file(pareto_path),
                    "row_count": int(len(pareto)),
                },
                "standard_candidates": {
                    "path": standard_path.name,
                    "sha256": adapter.sha256_file(standard_path),
                    "row_count": int(len(standard_candidates)),
                    "selection": (
                        "deterministic_anchors_knee_margin_and_front_spread"
                    ),
                },
            },
            "search_only_proposal": any(
                result["search_only_proposal"] for result in results
            ),
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    path = output / "aggregate_manifest.json"
    _atomic_json(path, manifest)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the isolated MFT goal campaign."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--generation", type=Path, required=True)
    prepare.add_argument("--candidate", type=Path, required=True)
    prepare.add_argument("--quality-status", type=Path, required=True)
    prepare.add_argument("--code-root", type=Path, required=True)
    prepare.add_argument("--expected-code-revision", required=True)
    prepare.add_argument(
        "--mode",
        choices=("single", "rolling32", "rolling"),
        required=True,
    )
    prepare.add_argument("--seed-start", type=int, required=True)
    prepare.add_argument("--fixed-primary-turns", type=int, default=5)
    prepare.add_argument("--seed-count", type=int)
    prepare.add_argument("--wave-size", type=int, default=32)
    prepare.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("execute-seed")
    execute.add_argument("--payload", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--relocation", type=Path)
    validate = commands.add_parser("validate-task")
    validate.add_argument("payload", type=Path)
    aggregate = commands.add_parser("aggregate")
    sources = aggregate.add_mutually_exclusive_group(required=True)
    sources.add_argument("--result", type=Path, action="append")
    sources.add_argument("--results-root", type=Path)
    aggregate.add_argument("--bundle-manifest", type=Path, required=True)
    aggregate.add_argument("--minimum-seeds", type=int, default=32)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_bundle(args)
    elif args.command == "execute-seed":
        result = execute_seed(args)
    elif args.command == "validate-task":
        validate_task_payload(_read_json(args.payload.resolve(strict=True)))
        result = args.payload.resolve(strict=True)
    else:
        result_paths = (
            args.result
            if args.result
            else list(args.results_root.resolve(strict=True).rglob("result.json"))
        )
        result = aggregate_results(
            result_paths=result_paths,
            bundle_manifest_path=args.bundle_manifest,
            output=args.output,
            minimum_seeds=args.minimum_seeds,
        )
    print(
        json.dumps(
            {"status": "ok", "path": str(result)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
