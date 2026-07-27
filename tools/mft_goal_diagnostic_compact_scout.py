"""Prepare and run an isolated N1=6 compact surrogate-screening scout.

This diagnostic path is deliberately separate from the reserved fresh512
campaign.  It may use an older authenticated surrogate whose quality gate did
not pass, but every output remains screening-only and cannot claim production,
final design, FEA validation, or fresh512 activation.  This CLI writes local
artifacts only; it contains no Scheduler API mutation or submission client.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    GOAL_TEMPERATURE_TARGETS,
    canonical_sha256,
)
from tools import mft_goal_20260726_launch as goal_launch  # noqa: E402
from tools import tier1_corrected_generation_adapter as adapter  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


CAMPAIGN_ID = "mft-goal-diagnostic-n1-6-compact-scout"
FIXED_PRIMARY_TURNS = 6
ACTIVATION_SCHEMA = "mft-goal-diagnostic-n1-6-compact-activation-v1"
BUNDLE_SCHEMA = "mft-goal-diagnostic-n1-6-compact-bundle-v1"
TASK_SCHEMA = "mft-goal-diagnostic-n1-6-compact-task-v1"
SCHEDULER_SCHEMA = "mft-goal-diagnostic-n1-6-compact-scheduler-v1"
RESULT_SCHEMA = "mft-goal-diagnostic-n1-6-compact-result-v1"
SOURCE_AUDIT_SCHEMA = "mft-goal-diagnostic-n1-6-source-audit-v1"
POPULATION = goal_launch.POPULATION
GENERATIONS = goal_launch.GENERATIONS
INFERENCE_THREADS = goal_launch.INFERENCE_THREADS
DEFAULT_SEED_START = 2_607_264_000
DEFAULT_SEED_COUNT = 32
MAXIMUM_SEED_COUNT = 128


def _seed_interval(seed_start: int, seed_count: int) -> list[int]:
    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or isinstance(seed_count, bool)
        or not isinstance(seed_count, int)
        or not 1 <= seed_count <= MAXIMUM_SEED_COUNT
    ):
        raise ValueError("diagnostic seed interval is invalid")
    seeds = list(range(seed_start, seed_start + seed_count))
    reserved = set(
        range(
            goal_launch.FRESH_AL_SEED_START,
            goal_launch.FRESH_AL_SEED_END + 1,
        )
    )
    if reserved.intersection(seeds):
        raise RuntimeError(
            "diagnostic scout may not consume reserved fresh512 seeds"
        )
    return seeds


def _validate_activation(value: Mapping[str, Any]) -> dict[str, Any]:
    activation = goal_launch._validate_seal(
        dict(value), schema=ACTIVATION_SCHEMA
    )
    compact_contract = preflight.validate_goal_compact_search_contract(
        activation.get("compact_search_contract") or {},
        fixed_primary_turns=FIXED_PRIMARY_TURNS,
    )
    bank = activation.get("compact_coordinate_bank") or {}
    unsigned_bank = {
        key: item for key, item in bank.items() if key != "sha256"
    }
    source = activation.get("source_identity") or {}
    if (
        activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or activation.get("screening_only") is not True
        or activation.get("production_eligible") is not False
        or activation.get("final_design_claim_allowed") is not False
        or activation.get("fresh512_activation_evidence") is not False
        or activation.get("reserved_fresh512_seed_interval_used") is not False
        or activation.get("symmetric_FEA_validation_still_required") is not True
        or activation.get("quality_thresholds_lowered_or_bypassed") is not False
        or activation.get("invalid_or_near_band_fallback_allowed") is not False
        or activation.get(
            "core_center_gap_mm_symmetric_FEA_synthesis_required"
        )
        is not True
        or activation.get("physical_Lm_2mH_verified") is not False
        or activation.get("scheduler_write_performed") is not False
        or activation.get("scheduler_submission_performed") is not False
        or activation.get("compact_search_contract_sha256")
        != compact_contract["sha256"]
        or bank.get("schema_version") != preflight.GOAL_COMPACT_BANK_SCHEMA
        or bank.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or bank.get("compact_search_contract_sha256")
        != compact_contract["sha256"]
        or bank.get("sha256") != canonical_sha256(unsigned_bank)
        or activation.get("compact_coordinate_bank_sha256")
        != bank.get("sha256")
        or bank.get("near_band_fallback_used") is not False
        or activation.get("fixed_lm2mh_resonance_contract")
        != preflight.goal_fixed_lm2mh_resonance_contract()
        or activation.get("fixed_lm2mh_resonance_contract_sha256")
        != preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
        or not isinstance(activation.get("source_quality_passed"), bool)
        or set(source)
        != {
            "train_report_sha256",
            "candidate_sha256",
            "quality_status_sha256",
            "dataset_sha256",
            "profile_sha256",
            "evaluation_model_sha256",
            "code_revision",
        }
    ):
        raise RuntimeError("diagnostic compact activation contract mismatch")
    return activation


def _build_activation(
    runner: preflight.Current7Tier1Runner,
    *,
    quality_contract: Mapping[str, Any],
    compact_contract: Mapping[str, Any],
    compact_bank: Mapping[str, Any],
    fixed_lm_installation: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = runner.authenticated.report["artifacts"]
    source_identity = {
        "train_report_sha256": runner.authenticated.evidence[
            "train_report"
        ]["sha256"],
        "candidate_sha256": runner.authenticated.evidence["candidate"][
            "sha256"
        ],
        "quality_status_sha256": runner.authenticated.evidence[
            "quality_status"
        ]["sha256"],
        "dataset_sha256": runner.authenticated.evidence["dataset"]["sha256"],
        "profile_sha256": runner.authenticated.evidence["profile"][
            "canonical_sha256"
        ],
        "evaluation_model_sha256": canonical_sha256(artifacts),
        "code_revision": runner.code_identity["revision"],
    }
    return goal_launch._seal(
        {
            "schema_version": ACTIVATION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "source_identity": source_identity,
            "source_quality_passed": bool(
                quality_contract["quality_passed"]
            ),
            "source_quality_blockers": list(
                quality_contract["quality_blockers"]
            ),
            "quality_thresholds_lowered_or_bypassed": False,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "reserved_fresh512_seed_interval_used": False,
            "symmetric_FEA_validation_still_required": True,
            "fixed_lm2mh_resonance_contract": (
                preflight.goal_fixed_lm2mh_resonance_contract()
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "fixed_lm2mh_resonance_installation": copy.deepcopy(
                dict(fixed_lm_installation)
            ),
            "effective_hard_constraint_contract_sha256": (
                runner.problem.hard_constraint_contract_sha256
            ),
            "compact_search_contract": copy.deepcopy(dict(compact_contract)),
            "compact_search_contract_sha256": compact_contract["sha256"],
            "compact_coordinate_bank": copy.deepcopy(dict(compact_bank)),
            "compact_coordinate_bank_sha256": compact_bank["sha256"],
            "invalid_or_near_band_fallback_allowed": False,
            "core_center_gap_mm_symmetric_FEA_synthesis_required": True,
            "physical_Lm_2mH_verified": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )


def audit_source(args: argparse.Namespace) -> Path:
    """Authenticate immutable source bytes without loading model pickle files."""

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("diagnostic source audit output already exists")
    code_root = args.code_root.resolve(strict=True)
    code = adapter.authenticate_code_root(
        code_root, args.expected_code_revision
    )
    authenticated = adapter.authenticate_corrected_generation(
        generation=args.generation.resolve(strict=True),
        candidate_path=args.candidate.resolve(strict=True),
        quality_path=args.quality_status.resolve(strict=True),
        goal_campaign=True,
        dataset_path_override=(
            None
            if args.runtime_dataset is None
            else args.runtime_dataset.resolve(strict=True)
        ),
        profile_path_override=(
            None
            if args.runtime_profile is None
            else args.runtime_profile.resolve(strict=True)
        ),
        expected_documentary_generation_path=(
            args.expected_documentary_generation_path
        ),
    )
    quality = goal_launch._quality_contract(
        quality=authenticated.quality,
        code_root=code_root,
    )
    value = goal_launch._seal(
        {
            "schema_version": SOURCE_AUDIT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "code": code,
            "generation": authenticated.evidence["generation_relative"],
            "train_report_sha256": authenticated.evidence[
                "train_report"
            ]["sha256"],
            "candidate_sha256": authenticated.evidence["candidate"]["sha256"],
            "quality_status_sha256": authenticated.evidence[
                "quality_status"
            ]["sha256"],
            "dataset": copy.deepcopy(authenticated.evidence["dataset"]),
            "profile": copy.deepcopy(authenticated.evidence["profile"]),
            "source_quality_passed": quality["quality_passed"],
            "source_quality_blockers": quality["quality_blockers"],
            "diagnostic_screening_path_allowed": True,
            "fresh512_or_production_activation_allowed": False,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "full_model_pickle_authentication_performed": False,
            "full_model_pickle_authentication_required_before_scheduler_POST": True,
            "exact_real_prepare_and_local_dry_run_required_before_scheduler_POST": True,
            "source_files_mutated": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    goal_launch._atomic_json(output, value)
    return output


def prepare(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    code_root = args.code_root.resolve(strict=True)
    if output.exists():
        raise RuntimeError("diagnostic scout output already exists")
    if goal_launch._path_is_below(output, code_root):
        raise RuntimeError("diagnostic output must be outside the code root")
    seeds = _seed_interval(int(args.seed_start), int(args.seed_count))
    runner = preflight.build_authenticated_runner(
        generation=args.generation.resolve(strict=True),
        candidate_path=args.candidate.resolve(strict=True),
        quality_path=args.quality_status.resolve(strict=True),
        code_root=code_root,
        expected_code_revision=args.expected_code_revision,
        fixed_primary_turns=FIXED_PRIMARY_TURNS,
        stage_spec=GOAL_STAGE_SPEC,
        inference_threads=INFERENCE_THREADS,
        dataset_path_override=(
            None
            if args.runtime_dataset is None
            else args.runtime_dataset.resolve(strict=True)
        ),
        profile_path_override=(
            None
            if args.runtime_profile is None
            else args.runtime_profile.resolve(strict=True)
        ),
        expected_documentary_generation_path=(
            args.expected_documentary_generation_path
        ),
    )
    quality = goal_launch._quality_contract(
        quality=runner.authenticated.quality,
        code_root=code_root,
    )
    _base, fixed_lm_installation = (
        preflight.install_goal_fixed_lm2mh_resonance(runner.problem)
    )
    compact_contract = preflight.goal_compact_search_contract(
        FIXED_PRIMARY_TURNS
    )
    compact_bank = preflight.build_goal_compact_coordinate_bank(
        runner.problem,
        seed=seeds[0],
        compact_contract=compact_contract,
    )
    preflight.validate_goal_compact_coordinate_bank(
        runner.problem,
        compact_bank,
        compact_contract=compact_contract,
    )
    activation = _build_activation(
        runner,
        quality_contract=quality,
        compact_contract=compact_contract,
        compact_bank=compact_bank,
        fixed_lm_installation=fixed_lm_installation,
    )
    _validate_activation(activation)

    code_sources = goal_launch._collect_goal_code_sources(code_root)
    code_manifest = goal_launch._build_goal_code_manifest(
        code_sources,
        code_revision=runner.code_identity["revision"],
    )
    goal_launch._stage_goal_code(
        output,
        sources=code_sources,
        manifest=code_manifest,
    )
    documentary_generation = str(
        runner.authenticated.candidate.get("generation_path") or ""
    )
    if not documentary_generation:
        raise RuntimeError("diagnostic candidate lacks generation_path")
    source = {
        "generation": documentary_generation,
        "candidate": str(args.candidate.resolve(strict=True)),
        "quality_status": str(args.quality_status.resolve(strict=True)),
        "code_root": str(code_root),
        "dataset": str(runner.authenticated.dataset_path),
        "profile": str(runner.authenticated.profile_path),
        "expected_code_revision": runner.code_identity["revision"],
    }
    tasks = []
    for ordinal, seed in enumerate(seeds):
        compact_run_authorization = (
            preflight.seal_goal_compact_run_authorization(
                authorization_mode=preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC,
                source_activation_schema=ACTIVATION_SCHEMA,
                source_activation_payload_sha256=activation[
                    "payload_sha256"
                ],
                seed=seed,
                fixed_primary_turns=FIXED_PRIMARY_TURNS,
                dataset_sha256=activation["source_identity"][
                    "dataset_sha256"
                ],
                evaluation_model_sha256=activation["source_identity"][
                    "evaluation_model_sha256"
                ],
                quality_status_sha256=activation["source_identity"][
                    "quality_status_sha256"
                ],
                source_quality_passed=activation[
                    "source_quality_passed"
                ],
                effective_hard_constraint_contract_sha256=activation[
                    "effective_hard_constraint_contract_sha256"
                ],
                compact_search_contract_sha256=activation[
                    "compact_search_contract_sha256"
                ],
                compact_coordinate_bank_sha256=activation[
                    "compact_coordinate_bank_sha256"
                ],
            )
        )
        tasks.append(
            goal_launch._seal(
                {
                    "schema_version": TASK_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "task_name": f"mft-goal-diag-compact-n1-6-s{seed}",
                    "ordinal": ordinal,
                    "seed": seed,
                    "fixed_primary_turns": FIXED_PRIMARY_TURNS,
                    "population": POPULATION,
                    "generations": GENERATIONS,
                    "inference_threads": INFERENCE_THREADS,
                    "stage_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                    "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                    "temperature_contract_sha256": (
                        GOAL_TEMPERATURE_CONTRACT_SHA256
                    ),
                    "hard_constraint_contract_sha256": activation[
                        "effective_hard_constraint_contract_sha256"
                    ],
                    "source": source,
                    "source_identity": activation["source_identity"],
                    "code_manifest_payload_sha256": code_manifest[
                        "payload_sha256"
                    ],
                    "code_inventory_sha256": code_manifest[
                        "code_inventory_sha256"
                    ],
                    "activation": activation,
                    "compact_run_authorization": (
                        compact_run_authorization
                    ),
                    "screening_only": True,
                    "production_eligible": False,
                    "final_design_claim_allowed": False,
                    "fresh512_activation_evidence": False,
                    "symmetric_FEA_validation_still_required": True,
                }
            )
        )
    task_paths = [
        f"tasks/seed-{task['seed']}-n1-6.json" for task in tasks
    ]
    bundle = goal_launch._seal(
        {
            "schema_version": BUNDLE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "task_count": len(tasks),
            "task_relative_paths": task_paths,
            "task_payload_sha256": [
                task["payload_sha256"] for task in tasks
            ],
            "activation_payload_sha256": activation["payload_sha256"],
            "code_manifest_payload_sha256": code_manifest["payload_sha256"],
            "source_bound_paths": source,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "scheduler_project_modified": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    scheduler = goal_launch._seal(
        {
            "schema_version": SCHEDULER_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "bundle_payload_sha256": bundle["payload_sha256"],
            "task_count": len(tasks),
            "maximum_parallel_tasks": len(tasks),
            "resources_per_task": {
                "cpus": INFERENCE_THREADS,
                "memory_GiB": 64,
                "gpus": 0,
                "optimizer_processes": 1,
            },
            "command_template": [
                "python3",
                "tools/mft_goal_diagnostic_compact_scout.py",
                "execute",
                "--payload",
                "<task-payload-path>",
                "--output",
                "<unique-task-output-directory>",
                "--relocation",
                "<optional-worker-relocation-json>",
            ],
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "exact_real_dry_run_and_authentication_required_before_POST": True,
            "scheduler_project_modified": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    goal_launch._atomic_json(output / "activation.json", activation)
    for task, relative in zip(tasks, task_paths):
        goal_launch._atomic_json(output / relative, task)
    goal_launch._atomic_json(output / "bundle_manifest.json", bundle)
    goal_launch._atomic_json(output / "scheduler_manifest.json", scheduler)
    return output / "bundle_manifest.json"


def _validate_task(value: Mapping[str, Any]) -> dict[str, Any]:
    task = goal_launch._validate_seal(dict(value), schema=TASK_SCHEMA)
    activation = _validate_activation(task.get("activation") or {})
    seed = task.get("seed")
    _seed_interval(int(seed), 1)
    source = task.get("source") or {}
    source_identity = task.get("source_identity") or {}
    compact_authorization = (
        preflight.validate_goal_compact_run_authorization_seal(
            task.get("compact_run_authorization") or {}
        )
    )
    digest_fields = (
        "train_report_sha256",
        "candidate_sha256",
        "quality_status_sha256",
        "dataset_sha256",
        "profile_sha256",
        "evaluation_model_sha256",
    )
    if (
        task.get("campaign_id") != CAMPAIGN_ID
        or task.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or task.get("population") != POPULATION
        or task.get("generations") != GENERATIONS
        or task.get("inference_threads") != INFERENCE_THREADS
        or task.get("stage_spec") != GOAL_STAGE_SPEC
        or task.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or task.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or task.get("hard_constraint_contract_sha256")
        != activation["effective_hard_constraint_contract_sha256"]
        or task.get("source_identity") != activation["source_identity"]
        or compact_authorization.get("authorization_mode")
        != preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
        or compact_authorization.get("source_activation_schema")
        != ACTIVATION_SCHEMA
        or compact_authorization.get("seed") != seed
        or compact_authorization.get("fixed_primary_turns")
        != FIXED_PRIMARY_TURNS
        or compact_authorization.get("source_activation_payload_sha256")
        != activation["payload_sha256"]
        or compact_authorization.get("dataset_sha256")
        != source_identity.get("dataset_sha256")
        or compact_authorization.get("evaluation_model_sha256")
        != source_identity.get("evaluation_model_sha256")
        or compact_authorization.get("quality_status_sha256")
        != source_identity.get("quality_status_sha256")
        or compact_authorization.get(
            "effective_hard_constraint_contract_sha256"
        )
        != task.get("hard_constraint_contract_sha256")
        or compact_authorization.get("compact_search_contract_sha256")
        != activation.get("compact_search_contract_sha256")
        or compact_authorization.get("compact_coordinate_bank_sha256")
        != activation.get("compact_coordinate_bank_sha256")
        or compact_authorization.get("screening_only") is not True
        or compact_authorization.get("production_eligible") is not False
        or compact_authorization.get("final_design_claim_allowed") is not False
        or compact_authorization.get("dataset_authentication_sha256")
        is not None
        or set(source)
        != {*goal_launch.RUNTIME_SOURCE_ROLES, "expected_code_revision"}
        or any(
            not isinstance(source.get(name), str) or not source[name]
            for name in (*goal_launch.RUNTIME_SOURCE_ROLES, "expected_code_revision")
        )
        or any(
            not isinstance(source_identity.get(name), str)
            or len(source_identity[name]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_identity[name]
            )
            for name in digest_fields
        )
        or source_identity.get("code_revision")
        != source.get("expected_code_revision")
        or not isinstance(task.get("code_manifest_payload_sha256"), str)
        or len(task["code_manifest_payload_sha256"]) != 64
        or not isinstance(task.get("code_inventory_sha256"), str)
        or len(task["code_inventory_sha256"]) != 64
        or task.get("screening_only") is not True
        or task.get("production_eligible") is not False
        or task.get("final_design_claim_allowed") is not False
        or task.get("fresh512_activation_evidence") is not False
        or task.get("symmetric_FEA_validation_still_required") is not True
    ):
        raise RuntimeError("diagnostic compact task contract mismatch")
    return task


def execute(args: argparse.Namespace) -> Path:
    task = _validate_task(
        goal_launch._read_json(args.payload.resolve(strict=True))
    )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("diagnostic task output already exists")
    source, relocated_code_manifest = goal_launch.resolve_worker_source(
        task, args.relocation
    )
    relocated = relocated_code_manifest is not None
    runner = preflight.build_authenticated_runner(
        generation=Path(source["generation"]),
        candidate_path=Path(source["candidate"]),
        quality_path=Path(source["quality_status"]),
        code_root=Path(source["code_root"]),
        expected_code_revision=source["expected_code_revision"],
        fixed_primary_turns=FIXED_PRIMARY_TURNS,
        stage_spec=GOAL_STAGE_SPEC,
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
            task["code_manifest_payload_sha256"] if relocated else None
        ),
        expected_code_inventory_sha256=(
            task["code_inventory_sha256"] if relocated else None
        ),
    )
    activation = task["activation"]
    source_identity = activation["source_identity"]
    _base, fixed_installation = preflight.install_goal_fixed_lm2mh_resonance(
        runner.problem
    )
    observed_identity = {
        "train_report_sha256": runner.authenticated.evidence[
            "train_report"
        ]["sha256"],
        "candidate_sha256": runner.authenticated.evidence["candidate"][
            "sha256"
        ],
        "quality_status_sha256": runner.authenticated.evidence[
            "quality_status"
        ]["sha256"],
        "dataset_sha256": runner.authenticated.evidence["dataset"]["sha256"],
        "profile_sha256": runner.authenticated.evidence["profile"][
            "canonical_sha256"
        ],
        "evaluation_model_sha256": canonical_sha256(
            runner.authenticated.report["artifacts"]
        ),
        "code_revision": runner.code_identity["revision"],
    }
    if (
        observed_identity != source_identity
        or runner.problem.hard_constraint_contract_sha256
        != task["hard_constraint_contract_sha256"]
        or fixed_installation["resonance_contract_sha256"]
        != activation["fixed_lm2mh_resonance_contract_sha256"]
    ):
        raise RuntimeError("diagnostic task source identity drifted")
    compact_contract = activation["compact_search_contract"]
    compact_bank = activation["compact_coordinate_bank"]
    preflight.validate_goal_compact_coordinate_bank(
        runner.problem,
        compact_bank,
        compact_contract=compact_contract,
    )
    output.mkdir(parents=True)
    physical_evaluate, scaling = preflight.install_optimizer_scaling(
        runner.problem,
        resonance_scale_hz=150.0,
        llt_scale_uh=0.55,
        all_thermal_scale_c=10.0,
    )
    runner.physical_evaluate = physical_evaluate
    runner.optimizer_scaling = scaling
    repair_installation = runner.install_offspring_repair_operator(
        topology_contract=preflight.goal_topology_contract(
            FIXED_PRIMARY_TURNS
        )
    )

    def pre_optimization(evidence: Mapping[str, Any]) -> None:
        goal_launch._atomic_json(
            output / "pre_optimization.json",
            goal_launch._seal(
                {
                    "schema_version": (
                        "mft-goal-diagnostic-compact-pre-optimization-v1"
                    ),
                    "task_payload_sha256": task["payload_sha256"],
                    "evidence": copy.deepcopy(dict(evidence)),
                    "repair_installation": repair_installation,
                    "screening_only": True,
                    "production_eligible": False,
                    "final_design_claim_allowed": False,
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
        pre_optimization_callback=pre_optimization,
        compact_search_contract=compact_contract,
        compact_coordinate_bank=compact_bank,
        compact_run_authorization=task["compact_run_authorization"],
        compact_source_activation=activation,
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
            "island_id": "diagnostic-n1-6-compact",
            "dataset_sha256": source_identity["dataset_sha256"],
            "model_artifacts_sha256": source_identity[
                "evaluation_model_sha256"
            ],
            "model_generation_sha256": source_identity[
                "train_report_sha256"
            ],
            "evaluation_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
        },
    )
    value = goal_launch._seal(
        {
            "schema_version": RESULT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "task_payload_sha256": task["payload_sha256"],
            "seed": int(task["seed"]),
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "population": POPULATION,
            "generations": GENERATIONS,
            "temperature_targets": list(GOAL_TEMPERATURE_TARGETS),
            "optimizer_scaling_contract": scaling,
            "optimizer_repair_audit": result.tier1_repair_audit,
            "optimizer_topology_evolution_audit": (
                result.tier1_topology_evolution_audit
            ),
            "compact_run_authorization_sha256": task[
                "compact_run_authorization"
            ]["sha256"],
            "resonance_contract_schema": (
                preflight.GOAL_FIXED_LM_RESONANCE_SCHEMA
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "effective_self_resonance_authority": (
                preflight.goal_fixed_lm2mh_resonance_contract()[
                    "effective_self_resonance_authority"
                ]
            ),
            "terminal_population_count": artifacts[
                "terminal_population_count"
            ],
            "physical_feasible_count_is_surrogate_screening_only": artifacts[
                "physical_feasible_count"
            ],
            "feasible_pareto_count_is_surrogate_screening_only": artifacts[
                "feasible_pareto_count"
            ],
            "artifact_inventory": artifacts["artifact_inventory"],
            "artifact_inventory_sha256": artifacts[
                "artifact_inventory_sha256"
            ],
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    path = output / "result.json"
    goal_launch._atomic_json(path, value)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit-source")
    audit_parser.add_argument("--generation", type=Path, required=True)
    audit_parser.add_argument("--candidate", type=Path, required=True)
    audit_parser.add_argument("--quality-status", type=Path, required=True)
    audit_parser.add_argument("--code-root", type=Path, required=True)
    audit_parser.add_argument("--expected-code-revision", required=True)
    audit_parser.add_argument("--runtime-dataset", type=Path)
    audit_parser.add_argument("--runtime-profile", type=Path)
    audit_parser.add_argument(
        "--expected-documentary-generation-path"
    )
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(handler=audit_source)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--generation", type=Path, required=True)
    prepare_parser.add_argument("--candidate", type=Path, required=True)
    prepare_parser.add_argument("--quality-status", type=Path, required=True)
    prepare_parser.add_argument("--code-root", type=Path, required=True)
    prepare_parser.add_argument("--expected-code-revision", required=True)
    prepare_parser.add_argument("--runtime-dataset", type=Path)
    prepare_parser.add_argument("--runtime-profile", type=Path)
    prepare_parser.add_argument(
        "--expected-documentary-generation-path"
    )
    prepare_parser.add_argument(
        "--seed-start", type=int, default=DEFAULT_SEED_START
    )
    prepare_parser.add_argument(
        "--seed-count", type=int, default=DEFAULT_SEED_COUNT
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(handler=prepare)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--payload", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)
    execute_parser.add_argument("--relocation", type=Path)
    execute_parser.set_defaults(handler=execute)
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
