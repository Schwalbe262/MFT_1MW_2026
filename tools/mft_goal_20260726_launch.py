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
import json
import math
import os
from pathlib import Path
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
POPULATION = 320
GENERATIONS = 300
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
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _validate_seal(dict(local_preflight), schema=LOCAL_PREFLIGHT_SCHEMA)
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
                    "evaluation_model_sha256": local_preflight[
                        "evaluation_model_sha256"
                    ],
                    "code_revision": local_preflight["code"]["revision"],
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
            "relocation_contract": {
                "schema_version": RELOCATION_SCHEMA,
                "mode": (
                    "source_paths_or_explicit_worker_role_path_relocation"
                ),
                "roles": [
                    "generation",
                    "candidate",
                    "quality_status",
                    "code_root",
                ],
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
            "evaluation_model_sha256",
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
        != {
            "generation",
            "candidate",
            "quality_status",
            "code_root",
            "expected_code_revision",
        }
        or set(task.get("source_identity") or {})
        != {
            "train_report_sha256",
            "candidate_sha256",
            "quality_status_sha256",
            "dataset_sha256",
            "evaluation_model_sha256",
            "code_revision",
        }
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
        or not isinstance(source_identity.get("code_revision"), str)
        or len(source_identity["code_revision"]) != 40
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
) -> dict[str, str]:
    source = dict(task["source"])
    if relocation_path is None:
        return source
    relocation = _validate_seal(
        _read_json(relocation_path.resolve(strict=True)),
        schema=RELOCATION_SCHEMA,
    )
    paths = relocation.get("paths")
    if (
        relocation.get("task_payload_sha256") != task["payload_sha256"]
        or not isinstance(paths, dict)
        or set(paths)
        != {"generation", "candidate", "quality_status", "code_root"}
    ):
        raise RuntimeError("worker relocation contract mismatch")
    source.update({name: str(paths[name]) for name in paths})
    return source


def prepare_bundle(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    code_root = args.code_root.resolve(strict=True)
    if output.exists():
        raise RuntimeError("goal bundle output already exists")
    if _path_is_below(output, code_root):
        raise RuntimeError("goal bundle output must be outside the clean code root")
    local_preflight, _runner = run_local_preflight(
        generation=args.generation,
        candidate=args.candidate,
        quality_status=args.quality_status,
        code_root=code_root,
        expected_code_revision=args.expected_code_revision,
    )
    source = {
        "generation": str(args.generation.resolve(strict=True)),
        "candidate": str(args.candidate.resolve(strict=True)),
        "quality_status": str(args.quality_status.resolve(strict=True)),
        "code_root": str(code_root),
        "expected_code_revision": args.expected_code_revision,
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
    )
    output.mkdir(parents=True)
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
    source = resolve_worker_source(task, args.relocation)
    runner = preflight.build_authenticated_runner(
        generation=Path(source["generation"]),
        candidate_path=Path(source["candidate"]),
        quality_path=Path(source["quality_status"]),
        code_root=Path(source["code_root"]),
        expected_code_revision=source["expected_code_revision"],
        fixed_primary_turns=int(task["fixed_primary_turns"]),
        stage_spec=task["stage_spec"],
        inference_threads=INFERENCE_THREADS,
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
        or canonical_sha256(runner.authenticated.report["artifacts"])
        != source_identity["evaluation_model_sha256"]
        or runner.code_identity["revision"] != source_identity["code_revision"]
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


def _validated_seed_table(result_path: Path) -> tuple[dict[str, Any], Any]:
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
        or result.get("population") != POPULATION
        or result.get("generations") != GENERATIONS
        or result.get("terminal_population_count") != POPULATION
        or not isinstance(result.get("search_only_proposal"), bool)
        or result.get("production_eligible") is not False
        or result.get("automatic_promotion_allowed") is not False
        or result.get("legacy_current7_stage_or_release_identity_reused")
        is not False
    ):
        raise RuntimeError(f"goal seed result contract mismatch: {path}")
    inventory = result.get("artifact_inventory") or {}
    table_record = inventory.get("terminal_physical_candidates") or {}
    manifest_record = inventory.get(
        "terminal_physical_candidates_manifest"
    ) or {}
    table_path = path.parent / str(table_record.get("path") or "")
    manifest_path = path.parent / str(manifest_record.get("path") or "")
    if (
        not table_path.is_file()
        or not manifest_path.is_file()
        or adapter.sha256_file(table_path) != table_record.get("sha256")
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
    *, result_paths: list[Path], output: Path, minimum_seeds: int
) -> Path:
    import numpy as np
    import pandas as pd
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    if (
        isinstance(minimum_seeds, bool)
        or not isinstance(minimum_seeds, int)
        or minimum_seeds < 2
    ):
        raise ValueError("minimum_seeds must be an integer >= 2")
    resolved = sorted({path.resolve(strict=True) for path in result_paths})
    if len(resolved) < minimum_seeds:
        raise RuntimeError("authenticated seed result count is below minimum")
    output = output.resolve()
    if output.exists():
        raise RuntimeError("global Pareto output already exists")
    results = []
    frames = []
    for path in resolved:
        result, frame = _validated_seed_table(path)
        results.append(result)
        frames.append(frame)
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
    feasible = deduplicated["physical_feasible"].to_numpy(dtype=bool)
    deduplicated["global_physical_feasible"] = feasible
    deduplicated["global_non_dominated_rank"] = -1
    feasible_indices = np.flatnonzero(feasible)
    fronts: list[Any] = []
    if len(feasible_indices):
        objectives = deduplicated.iloc[feasible_indices][
            ["objective_volume_L", "objective_total_loss_W"]
        ].to_numpy(dtype=float)
        fronts = NonDominatedSorting().do(objectives)
        rank_column = deduplicated.columns.get_loc(
            "global_non_dominated_rank"
        )
        for rank, local_indices in enumerate(fronts):
            global_indices = feasible_indices[
                np.asarray(local_indices, dtype=int)
            ]
            deduplicated.iloc[global_indices, rank_column] = rank
    pareto = deduplicated.loc[
        deduplicated["global_non_dominated_rank"].eq(0)
    ].copy()
    pareto.sort_values(
        ["objective_volume_L", "objective_total_loss_W"],
        inplace=True,
    )
    output.mkdir(parents=True)
    all_path = output / "global_terminal_candidates.csv"
    pareto_path = output / "global_pareto_front.csv"
    _atomic_csv(all_path, deduplicated)
    _atomic_csv(pareto_path, pareto)
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
            "deduplicated_physical_geometry_count": int(len(deduplicated)),
            "physical_feasible_count": int(feasible.sum()),
            "non_dominated_front_count": len(fronts),
            "global_pareto_count": int(len(pareto)),
            "sorting_authority": (
                "all_authenticated_terminal_rows_then_physical_dedupe_"
                "then_decoder_and_physical_G_and_surrogate_physicality_"
                "feasible_then_non_dominated_sort"
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
