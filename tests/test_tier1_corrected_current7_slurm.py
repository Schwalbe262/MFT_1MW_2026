from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools import tier1_corrected_current7_receipt as receipt_contract
from tools import tier1_corrected_current7_slurm_bundle as bundle_tool
from tools import tier1_corrected_current7_slurm_controller as controller
from tools import tier1_corrected_current7_slurm_publish as publisher
from tools import tier1_corrected_current7_slurm_seed_runner as runner
from tools import tier1_corrected_generation_adapter as generation_adapter
from tools import tier1_corrected_generation_preflight as generation_preflight
from tools import tier1_final1000_multiseed_phase_b_contract as phase_b_contract


REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_search_source() -> str:
    return r"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

TEMPERATURES = [
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
]

def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("command")
for name in (
    "bundle-root", "relocation", "adapter-receipt", "registry",
    "generation", "dataset", "profile", "warm-start", "warm-contract",
    "output", "remote-preflight", "bundle-id", "island-id",
    "island-profile-sha256", "seed", "population", "max-generations",
    "inference-threads", "scheduler-cpus", "fixed-primary-turns",
    "optimizer-termination-strategy", "optimizer-resonance-scale-hz",
    "optimizer-llt-scale-uh", "optimizer-all-thermal-scale-c",
    "optimizer-repair-contract-sha256",
    "stage-spec-json", "stage-spec-sha256",
):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--optimizer-resonance-allowance-hz")
parser.add_argument("--optimizer-llt-allowance-uh")
args = parser.parse_args()
relocation = json.loads(Path(args.relocation).read_text())
identity = relocation["relocated_identity"]
bundle_manifest = json.loads(
    Path(args.bundle_root, "bundle_manifest.json").read_text()
)
topology_contract = bundle_manifest["islands"][args.island_id][
    "current7_profile"
]["topology_evolution_contract"]
topology_niche = topology_contract.get("final1000_topology_niche_contract")
stage_spec = json.loads(args.stage_spec_json)
assert stage_spec == identity["hard_spec"]
assert canonical(stage_spec) == args.stage_spec_sha256
semlock_stress = {
    "rounds": 8,
    "prediction_call_count": 8 * 20,
    "sklearn_extratrees_n_jobs": 1,
    "semaphore_entry_growth_count": 0,
    "enospc_observed": False,
    "stress_passed": True,
}
semlock_stress["sha256"] = canonical(semlock_stress)
preflight = {
    "schema_version": "mft-tier1-current7-remote-model-load-v1",
    "status": "passed",
    "bundle_id": args.bundle_id,
    "seed": int(args.seed),
    "island_id": args.island_id,
    "fixed_primary_turns": int(args.fixed_primary_turns),
    "optimizer_pid": os.getpid(),
    "optimizer_processes": 1,
    "model_mapping_instances": 1,
    "full_generation_authentication_passes": 1,
    "authenticated_artifact_count": 42,
    "generation_artifact_inventory_sha256": identity[
        "generation_artifact_inventory_sha256"
    ],
    "adapter_manifest_sha256": identity["adapter_manifest_sha256"],
    "train_report_sha256": identity["train_report_sha256"],
    "dataset_sha256": identity["dataset_sha256"],
    "profile_canonical_sha256": identity["profile_canonical_sha256"],
    "temperature_contract_sha256": identity["temperature_contract_sha256"],
    "hard_constraint_contract_sha256": identity[
        "hard_constraint_contract_sha256"
    ],
    "stage_spec_sha256": args.stage_spec_sha256,
    "island_profile_sha256": args.island_profile_sha256,
    "warm_artifact_sha256": hashlib.sha256(
        Path(args.warm_start).read_bytes()
    ).hexdigest(),
    "warm_contract_sha256": hashlib.sha256(
        Path(args.warm_contract).read_bytes()
    ).hexdigest(),
    "loaded_model_count": 20,
    "loaded_model_targets_sha256": (
        "3aa7a549f0c81cf68a38dcfded20d6ac3c081bda8c81d629201992cc02573ff9"
    ),
    "temperature_targets": TEMPERATURES,
    "inference_threads": int(args.inference_threads),
    "scheduler_cpus": int(args.scheduler_cpus),
    "optimizer_repair_contract_sha256": (
        args.optimizer_repair_contract_sha256
    ),
    "topology_evolution_contract_sha256": topology_contract["sha256"],
    "topology_niche_contract_sha256": (
        None if topology_niche is None else topology_niche["sha256"]
    ),
    "semlock_safe_prediction_stress": semlock_stress,
    "offspring_physics_repair": True,
    "initial_repair_attested": True,
    "warm_repair_attested": True,
    "offspring_repair_operator_installed": True,
    "offspring_repair_operator_contract_sha256": (
        args.optimizer_repair_contract_sha256
    ),
    "offspring_repair_execution_status": (
        "deferred_until_optimizer_execution"
    ),
    "every_offspring_decode_repair_attested": False,
    "terminal_physical_replay_required": True,
    "legacy_feedback_wrapper_used": False,
    "local_adapter_authentication_replayed": False,
    "observed_peak_rss_bytes": 1024 * 1024,
    "production_eligible": False,
    "fea_submission_approved": False,
    "fea_submission_performed": False,
    "aedt_used": False,
    "automatic_promotion_allowed": False,
}
preflight["payload_sha256"] = canonical(preflight)
Path(args.remote_preflight).write_text(json.dumps(preflight))
time.sleep(0.1)
hard_spec = identity["hard_spec"]
dummy_artifact = Path(args.output, "dummy.bin")
dummy_artifact.write_bytes(b"sealed-current7-dummy")
artifact_inventory = {
    "dummy": {
        "path": dummy_artifact.name,
        "sha256": hashlib.sha256(dummy_artifact.read_bytes()).hexdigest(),
        "size_bytes": dummy_artifact.stat().st_size,
    }
}
topology_audit = {
    "migration_events": 1,
    "paired_parent_pairs_emitted": 10,
    "survival_calls": int(args.max_generations),
    "terminal_epsilon_zero": True,
    "all_required_topologies_preserved": True,
}
topology_audit["sha256"] = canonical(topology_audit)
result = {
    "schema_version": "mft-tier1-current7-search-seed-v1",
    "bundle_id": args.bundle_id,
    "seed": int(args.seed),
    "island_id": args.island_id,
    "population": int(args.population),
    "max_generations": int(args.max_generations),
    "evaluated_generations": int(args.max_generations),
    "completed_generations": int(args.max_generations) + 1,
    "optimizer_processes": 1,
    "inference_threads": int(args.inference_threads),
    "scheduler_cpus": int(args.scheduler_cpus),
    "generation_artifact_inventory_sha256": identity[
        "generation_artifact_inventory_sha256"
    ],
    "adapter_manifest_sha256": identity["adapter_manifest_sha256"],
    "train_report_sha256": identity["train_report_sha256"],
    "dataset_sha256": identity["dataset_sha256"],
    "profile_canonical_sha256": identity["profile_canonical_sha256"],
    "temperature_contract_sha256": identity["temperature_contract_sha256"],
    "hard_constraint_contract_sha256": identity[
        "hard_constraint_contract_sha256"
    ],
    "hard_spec": hard_spec,
    "stage_spec_sha256": identity["hard_spec_sha256"],
    "constraint_version": identity["constraint_version"],
    "island_profile_sha256": args.island_profile_sha256,
    "warm_artifact_sha256": hashlib.sha256(
        Path(args.warm_start).read_bytes()
    ).hexdigest(),
    "warm_contract_sha256": hashlib.sha256(
        Path(args.warm_contract).read_bytes()
    ).hexdigest(),
    "temperature_targets": TEMPERATURES,
    "constraint_names": identity["constraint_names"],
    "fixed_primary_turns": int(args.fixed_primary_turns),
    "terminal_population_primary_turn_values": [int(args.fixed_primary_turns)],
    "terminal_population_fixed_primary_turns_verified": True,
    "offspring_physics_repair": True,
    "initial_population_repair_attested": True,
    "warm_start_repair_attested": True,
    "every_offspring_decode_repair_attested": True,
    "offspring_repair_operator_call_count": int(args.max_generations),
    "offspring_repair_operator_row_count": (
        int(args.population) * int(args.max_generations)
    ),
    "terminal_physical_replay_attested": True,
    "optimizer_repair_contract_sha256": args.optimizer_repair_contract_sha256,
    "optimizer_topology_evolution_contract": topology_contract,
    "topology_evolution_contract_sha256": topology_contract["sha256"],
    "topology_niche_contract_sha256": (
        None if topology_niche is None else topology_niche["sha256"]
    ),
    "optimizer_topology_evolution_audit": topology_audit,
    "semlock_safe_prediction_stress": semlock_stress,
    "semlock_safe_prediction_stress_sha256": semlock_stress["sha256"],
    "semlock_safe_prediction_stress_attested": True,
    "artifact_inventory": artifact_inventory,
    "artifact_inventory_sha256": canonical(artifact_inventory),
    "feasible_pareto_count": 0,
    "production_eligible": False,
    "fea_submission_approved": False,
    "fea_submission_performed": False,
    "aedt_used": False,
    "automatic_promotion_allowed": False,
}
result["payload_sha256"] = canonical(result)
Path(args.output, "result.json").write_text(json.dumps(result))
""".lstrip()


def _sealed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["sha256"] = receipt_contract.canonical_sha256(result)
    return result


def _v2_receipt(adapter: dict, *, repair_ready: bool) -> dict:
    expected_turns = list(generation_preflight.SUPPORTED_FIXED_PRIMARY_TURNS)
    expected_stages = {
        "initial_population": True,
        "warm_start": True,
        "every_offspring": True,
        "terminal_physical_replay": True,
    }
    strata = {}
    for turns in expected_turns:
        key = str(turns)
        contract = {
            "schema_version": generation_preflight.OPTIMIZER_REPAIR_SCHEMA,
            "projection_source_revision": (
                generation_preflight.PINNED_PROJECTION_SOURCE_REVISION
            ),
            "fixed_cw1_mm": 5.0,
            "cw1_enforcement": "inside_decoder_winding_budget",
            "fixed_primary_turns": turns,
            "required_stages": [
                "initial_population",
                "authenticated_warm_start",
                "every_pymoo_offspring",
                "terminal_unscaled_physical_replay",
            ],
        }
        repair_contract_sha = receipt_contract.canonical_sha256(contract)
        stage_evidence = _sealed(
            {
                "schema_version": "mft-tier1-current7-repair-smoke-v1",
                "fixed_primary_turns": turns,
                "stages": expected_stages,
                "same_problem_repair_used_for_all_stages": True,
            }
        )
        optimizer_repair = _sealed(
            {
                "schema_version": generation_preflight.OPTIMIZER_REPAIR_SCHEMA,
                "contract": contract,
                "contract_sha256": repair_contract_sha,
                "stages": expected_stages,
                "stage_evidence": stage_evidence,
                "fixed_primary_turns_repair": True,
                "fixed_primary_turns": turns,
                "launch_eligible": True,
            }
        )
        problem = _sealed(
            {
                "fixed_primary_turns": turns,
                "primary_winding_budget_identity_attested": True,
                "primary_winding_budget_identity": {"passed": True},
                "optimizer_repair_contract_sha256": repair_contract_sha,
                "launch_eligible": True,
            }
        )
        model_smoke = _sealed(
            {
                "target_count": len(
                    receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS
                ),
                "all_required_targets_exercised": True,
                "targets": {
                    target: {"finite": True}
                    for target in receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS
                },
                "additional_half_width_multiplier": 1.0,
                "semlock_safe_prediction_stress": _sealed(
                    {
                        "prediction_call_count": 8
                        * len(receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS),
                        "sklearn_extratrees_n_jobs": 1,
                        "semaphore_entry_growth_count": 0,
                        "enospc_observed": False,
                        "stress_passed": True,
                    }
                ),
            }
        )
        physical_smoke = _sealed(
            {
                "decoder_valid": True,
                "decoded_controls": {
                    "cw1": 5.0,
                    "N1_main": turns,
                    "N1_side": 0,
                },
                "objectives": {
                    "bounding_box_volume_L": 800.0,
                    "predicted_total_loss_W": 6000.0,
                },
                "physical_constraint_G": {
                    name: -1.0
                    for name in generation_preflight.CURRENT7_CONSTRAINT_NAMES
                },
                "finite_objectives": True,
                "finite_constraints": True,
                "design_feasibility_required_for_smoke": False,
            }
        )
        strata[key] = _sealed(
            {
                "fixed_primary_turns": turns,
                "problem": problem,
                "optimizer_repair": optimizer_repair,
                "model_smoke": model_smoke,
                "smoke": physical_smoke,
                "launch_eligible": True,
            }
        )

    model_load = {
        "required_targets": list(receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS),
        "required_targets_sha256": (
            receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS_SHA256
        ),
        "loaded_target_count": len(
            receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS
        ),
        "cache_load_calls": 1,
        "full_generation_authentication_passes": 1,
        "models_loaded_once_per_process": True,
        "generation_copy_performed": False,
        "inference_binding": {"target_count": 20, "threads_per_model": 1},
        "model_smoke_completed": True,
        "supported_fixed_primary_turns": expected_turns,
        "strata": {key: strata[key]["model_smoke"] for key in strata},
        "all_supported_strata_exercised": True,
        "semlock_safe_prediction_stress_required": True,
        "sklearn_extratrees_n_jobs": 1,
        "repeated_prediction_stress_rounds_per_stratum": 8,
        "dev_shm_semaphore_growth_allowed": False,
        "enospc_allowed": False,
    }
    problem_contract = {
        "stage_spec": generation_preflight.CURRENT_STAGE_SPEC,
        "stage_spec_sha256": generation_preflight.CURRENT_STAGE_SPEC_SHA256,
        "temperature_contract": generation_adapter.CURRENT_TEMPERATURE_CONTRACT,
        "temperature_contract_sha256": (
            generation_adapter.CURRENT_TEMPERATURE_CONTRACT_SHA256
        ),
        "hard_constraint_contract": (
            generation_adapter.CURRENT_STAGE_HARD_CONTRACT
        ),
        "hard_constraint_contract_sha256": (
            generation_adapter.CURRENT_STAGE_HARD_CONTRACT_SHA256
        ),
        "constraint_names": list(generation_preflight.CURRENT7_CONSTRAINT_NAMES),
        "constraint_count": len(generation_preflight.CURRENT7_CONSTRAINT_NAMES),
        "base_constraint_count": len(generation_preflight.BASE_CONSTRAINT_NAMES),
        "additive_hard_constraint_count": len(
            generation_preflight.ADDITIVE_HARD_CONSTRAINT_NAMES
        ),
        "base_secondary_vertical_insulation_retained": True,
        "minimum_physical_insulation_is_authoritative_superset": True,
        "variable_cooling_dimensions": list(
            generation_preflight.VARIABLE_COOLING_DIMENSIONS
        ),
        "fixed_cooling_pads_mm": generation_preflight.FIXED_COOLING_PADS_MM,
        "simple_base_20mm_plate_clamp_superseded": True,
        "primary_conductor_enforcement": (
            "fixed_cw1_consumed_inside_decoder_winding_budget"
        ),
        "side_temperature_condition_applied": True,
        "q90_additional_multiplier": 1.0,
        "supported_fixed_primary_turns": expected_turns,
        "strata": {key: strata[key]["problem"] for key in strata},
    }
    optimizer_repair = {
        "schema_version": generation_preflight.OPTIMIZER_REPAIR_SCHEMA,
        "supported_fixed_primary_turns": expected_turns,
        "strata": {key: strata[key]["optimizer_repair"] for key in strata},
        "all_supported_strata_passed": True,
        "launch_eligible": True,
    }
    smoke = {
        "coordinate": {
            "path": r"C:\original\smoke.npy",
            "sha256": "c" * 64,
            "shape": [1, 1],
            "coordinate_unit_sha256": "d" * 64,
        },
        "supported_fixed_primary_turns": expected_turns,
        "strata": {key: strata[key]["smoke"] for key in strata},
        "all_supported_strata_smoked": True,
    }
    receipt = {
        "schema_version": receipt_contract.SMOKE_RECEIPT_SCHEMA,
        "status": (
            "authenticated_dual_stratum_model_and_repair_smoke_passed_"
            "launch_eligible"
        ),
        "created_at": "2026-07-20T00:00:00+09:00",
        "supported_fixed_primary_turns": expected_turns,
        "strata": strata,
        "runner": {
            "schema_version": generation_preflight.RUNNER_SCHEMA,
            "problem_schema": generation_preflight.PROBLEM_SCHEMA,
            "run_interface": "Current7Tier1Runner.run_one",
            "current_run_nsga2_semantics_source": "optimization.run_nsga2.run_one",
            "current_initialization_semantics_source": (
                "optimization.run_nsga2.run_one"
            ),
            "supported_fixed_primary_turns": expected_turns,
            "full_nsga_executed": False,
            "launch_eligible": True,
        },
        "adapter_manifest": adapter,
        "adapter_manifest_sha256": receipt_contract.canonical_sha256(adapter),
        "code_modules": {"synthetic": True},
        "model_load": model_load,
        "model_load_sha256": receipt_contract.canonical_sha256(model_load),
        "problem_contract": problem_contract,
        "problem_contract_sha256": receipt_contract.canonical_sha256(
            problem_contract
        ),
        "optimizer_repair": optimizer_repair,
        "optimizer_repair_sha256": receipt_contract.canonical_sha256(
            optimizer_repair
        ),
        "smoke": smoke,
        "smoke_sha256": receipt_contract.canonical_sha256(smoke),
        "portability": generation_adapter.CURRENT_STAGE_HARD_CONTRACT[
            "portability"
        ],
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "canonical_pointer_write_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    if not repair_ready:
        receipt["strata"].pop("6")
    receipt["payload_sha256"] = receipt_contract.canonical_sha256(receipt)
    return receipt


def _fixture(
    tmp_path: Path,
    *,
    repair_ready: bool = True,
    profile_payload: dict | None = None,
    adapter_code_revision: str = "9" * 40,
) -> dict:
    run_id = "20260720T000000-current7"
    registry = tmp_path / "registry"
    generation = registry / "generations" / run_id
    generation.mkdir(parents=True)
    artifact_hashes = {}
    artifact_sizes = {}
    for index, relative in enumerate(receipt_contract.expected_generation_artifacts()):
        path = generation / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}-{relative}".encode())
        artifact_hashes[relative] = _sha(path)
        artifact_sizes[relative] = path.stat().st_size

    dataset = tmp_path / "strict.parquet"
    dataset.write_bytes(b"strict-current7-dataset")
    profile = tmp_path / "profile.json"
    if profile_payload is None:
        profile_payload = {"profile": "standard", "version": 7}
    _write_json(profile, profile_payload)
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, {"schema_version": 2, "candidate": "corrected"})
    quality = tmp_path / "quality_status.json"
    _write_json(
        quality,
        {"passed": False, "reasons": [f"blocker-{index}" for index in range(11)]},
    )
    report = {
        "schema_version": 2,
        "training_run_id": run_id,
        "dataset_path": r"C:\original\strict.parquet",
        "dataset_sha256": _sha(dataset),
        "profile_path": r"C:\original\profile.json",
        "profile_sha256": receipt_contract.training_profile_sha256(
            json.loads(profile.read_text())
        ),
        "strict_full_rows": 6151,
        "targets": list(receipt_contract.CORRECTED_GENERATION_TARGETS),
        "artifacts": artifact_hashes,
        "report": {
            target: {"r2": 0.9}
            for target in receipt_contract.CORRECTED_GENERATION_TARGETS
        },
    }
    report_path = generation / "train_report.json"
    _write_json(report_path, report)
    adapter = {
        "schema_version": receipt_contract.ADAPTER_SCHEMA,
        "generation": "C:\\original\\registry\\generations\\" + run_id,
        "registry": r"C:\original\registry",
        "generation_relative": f"generations/{run_id}",
        "training_run_id": run_id,
        "train_report": {
            "path": r"C:\original\generation\train_report.json",
            "sha256": _sha(report_path),
        },
        "candidate": {
            "path": r"C:\original\candidate.json",
            "sha256": _sha(candidate),
        },
        "quality_status": {
            "path": r"C:\original\quality_status.json",
            "sha256": _sha(quality),
            "passed": False,
            "blocking_reason_count": 11,
        },
        "dataset": {
            "path": r"C:\original\strict.parquet",
            "sha256": _sha(dataset),
            "strict_full_rows": 6151,
        },
        "profile": {
            "path": r"C:\original\profile.json",
            "canonical_sha256": report["profile_sha256"],
        },
        "capacitance_recovery": {
            "contract": generation_adapter.RECOVERY_CONTRACT,
            "max_allowed_abs_delta_F": (
                generation_adapter.RECOVERY_MAX_ABS_DELTA_F
            ),
            "row_count": 6151,
            "cap_enabled_row_count": 6151,
            "eligible_row_count": 6151,
            "recovered_row_count": 6151,
            "max_observed_abs_delta_F": 5.0e-11,
            "missing_columns": [],
            "status": generation_adapter.RECOVERY_STATUS,
            "guard_passed": True,
            "guard_target_count": 21,
            "guard_passed_target_count": 21,
        },
        "generation_targets": list(receipt_contract.CORRECTED_GENERATION_TARGETS),
        "generation_target_count": 21,
        "required_model_targets": list(receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS),
        "required_model_targets_sha256": (
            receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS_SHA256
        ),
        "artifact_count": 42,
        "artifact_sizes_bytes": artifact_sizes,
        "full_artifact_hash_pass_deferred_to_single_model_load": True,
        "temperature_contract": generation_adapter.CURRENT_TEMPERATURE_CONTRACT,
        "temperature_contract_sha256": (
            generation_adapter.CURRENT_TEMPERATURE_CONTRACT_SHA256
        ),
        "hard_constraint_contract": (
            generation_adapter.CURRENT_STAGE_HARD_CONTRACT
        ),
        "hard_constraint_contract_sha256": (
            generation_adapter.CURRENT_STAGE_HARD_CONTRACT_SHA256
        ),
        "code": {
            "path": r"C:\original\code",
            "revision": adapter_code_revision,
            "clean": True,
        },
        "model_loading": {
            "process_scope": "single_local_process",
            "cache_policy": "one_generation_authentication_and_unpickle_pass",
            "generation_copy_performed": False,
            "legacy_feedback_wrapper_used": False,
        },
        "portability": generation_adapter.CURRENT_STAGE_HARD_CONTRACT[
            "portability"
        ],
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "canonical_pointer_write_performed": False,
    }
    receipt = _v2_receipt(adapter, repair_ready=repair_ready)
    receipt_path = tmp_path / "authentication_receipt.json"
    _write_json(receipt_path, receipt)

    fake_search = tmp_path / "fake_current7_search.py"
    fake_search.write_text(_fake_search_source(), encoding="utf-8")
    code_sources = {
        "artifacts/code/tools/fake_current7_search.py": fake_search,
        "artifacts/code/tools/tier1_corrected_current7_receipt.py": (
            REPO / "tools" / "tier1_corrected_current7_receipt.py"
        ),
        "artifacts/code/tools/tier1_corrected_current7_slurm_bundle.py": (
            REPO / "tools" / "tier1_corrected_current7_slurm_bundle.py"
        ),
        "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py": (
            REPO / "tools" / "tier1_corrected_current7_slurm_seed_runner.py"
        ),
    }
    warm = {}
    for warm_id in ("n1-5", "n1-6"):
        artifact = tmp_path / warm_id / "coordinates.npy"
        artifact.parent.mkdir()
        artifact.write_bytes(f"warm-{warm_id}".encode())
        contract = tmp_path / warm_id / "contract.json"
        _write_json(contract, {"warm_id": warm_id, "authenticated": True})
        warm[warm_id] = {"artifact": artifact, "contract": contract}
    runtime_packages = {name: "1.0-test" for name in bundle_tool.CRITICAL_PACKAGES}
    return {
        "receipt": receipt_path,
        "generation": generation,
        "candidate": candidate,
        "quality": quality,
        "dataset": dataset,
        "profile": profile,
        "code_sources": code_sources,
        "warm": warm,
        "runtime_packages": runtime_packages,
    }


def _plan(
    tmp_path: Path,
    *,
    repair_ready: bool = True,
    profile_payload: dict | None = None,
):
    fixture = _fixture(
        tmp_path,
        repair_ready=repair_ready,
        profile_payload=profile_payload,
    )
    plan, manifest = bundle_tool.build_plan(
        local_root=tmp_path / "plans",
        remote_root="/remote/current7",
        receipt_path=fixture["receipt"],
        generation=fixture["generation"],
        candidate_path=fixture["candidate"],
        quality_path=fixture["quality"],
        dataset_path=fixture["dataset"],
        profile_path=fixture["profile"],
        code_identity={"revision": "9" * 40, "clean": True},
        code_sources=fixture["code_sources"],
        optimizer_entrypoint="tools/fake_current7_search.py",
        warm_starts=fixture["warm"],
        runtime_packages=fixture["runtime_packages"],
    )
    return fixture, plan, manifest


def _local_publish(tmp_path: Path, fixture: dict, plan: dict, manifest: dict) -> Path:
    bundle = tmp_path / "published" / manifest["bundle_id"]
    bundle.mkdir(parents=True)
    source_map = json.loads(Path(plan["local_sources"]).read_text())
    for relative, source in source_map.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(plan["bundle_manifest"], bundle / "bundle_manifest.json")
    (bundle / "runs").mkdir()
    _write_json(
        bundle / "READY.json",
        {
            "schema_version": bundle_tool.READY_SCHEMA,
            "bundle_id": manifest["bundle_id"],
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "every_file_sha256_verified": True,
            "runtime_verified": True,
            "code_inventory_sha256": manifest["code_inventory_sha256"],
            "relocation_contract_sha256": manifest["relocation"]["contract_sha256"],
            "remote_git_checkout_performed": False,
        },
    )
    return bundle


def test_receipt_mapping_seals_current7_and_repair_gate(tmp_path):
    fixture = _fixture(tmp_path)
    identity = receipt_contract.validate_adapter_receipt(
        json.loads(fixture["receipt"].read_text())
    )
    assert identity.launch_eligible is True
    assert identity.offspring_physics_repair is True
    assert identity.fixed_primary_turns_supported == (5, 6)
    assert set(
        identity.optimizer_repair_contract_sha256_by_fixed_primary_turns
    ) == {"5", "6"}
    assert (
        identity.optimizer_repair_contract_sha256_by_fixed_primary_turns["5"]
        != identity.optimizer_repair_contract_sha256_by_fixed_primary_turns["6"]
    )
    assert identity.hard_spec == generation_preflight.CURRENT_STAGE_SPEC
    assert identity.hard_spec_sha256 == generation_preflight.CURRENT_STAGE_SPEC_SHA256
    assert identity.constraint_names == generation_preflight.CURRENT7_CONSTRAINT_NAMES
    assert identity.artifact_count == 42
    assert identity.required_model_targets_sha256 == (
        receipt_contract.CURRENT_REQUIRED_MODEL_TARGETS_SHA256
    )


def test_legacy_or_incomplete_smoke_receipt_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    value = json.loads(fixture["receipt"].read_text())
    value["schema_version"] = "mft-tier1-corrected-generation-smoke-receipt-v1"
    value["status"] = "authenticated_model_smoke_passed_launch_blocked"
    value.pop("payload_sha256")
    value["payload_sha256"] = receipt_contract.canonical_sha256(value)
    with pytest.raises(RuntimeError, match="unsupported"):
        receipt_contract.validate_adapter_receipt(value)


@pytest.mark.parametrize(
    "tamper",
    ("missing_stratum", "repair_contract_sha", "repair_stage_sha"),
)
def test_receipt_v2_stratum_and_repair_seals_are_fail_closed(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    value = json.loads(fixture["receipt"].read_text())
    if tamper == "missing_stratum":
        value["strata"].pop("6")
    elif tamper == "repair_contract_sha":
        value["strata"]["5"]["optimizer_repair"]["contract_sha256"] = "0" * 64
    else:
        value["strata"]["6"]["optimizer_repair"]["stage_evidence"][
            "sha256"
        ] = "0" * 64
    value.pop("payload_sha256")
    value["payload_sha256"] = receipt_contract.canonical_sha256(value)
    with pytest.raises(RuntimeError):
        receipt_contract.validate_adapter_receipt(value)


def test_bundle_is_content_addressed_and_relocates_absolute_paths(tmp_path):
    fixture, plan, manifest = _plan(tmp_path)
    relocation = json.loads(Path(plan["relocation_contract"]).read_text())
    assert manifest["bundle_id"].startswith("current7-")
    assert len(manifest["generation_artifacts"]) == 42
    assert relocation["source_absolute_paths_are_documentary_only"] is True
    assert relocation["local_adapter_authentication_replayed_remotely"] is False
    assert relocation["generation_report_bytes_mutated"] is False
    assert relocation["bundle_paths"]["dataset"] == (
        "artifacts/dataset/strict_full.parquet"
    )
    assert relocation["original_paths"]["dataset"].startswith("C:\\")
    assert manifest["remote_git_checkout_required"] is False
    assert manifest["bundle_code_revision"] == "9" * 40
    assert manifest["adapter_receipt"]["identity"]["adapter_code_revision"] == (
        manifest["bundle_code_revision"]
    )
    assert manifest["scheduler_api_contract"]["endpoint_path"] == "/api/tasks"
    assert (
        manifest["scheduler_api_contract"]["requested_allocation_id_allowed"] is False
    )
    assert manifest["scheduler_api_contract"]["scheduler_submission_performed"] is False
    assert all(
        not relative.endswith("tier1_slurm_seed_runner.py")
        for relative in manifest["code_inventory"]
    )
    assert plan["stage_performed"] is False
    assert plan["submission_performed"] is False

    second_plan, second_manifest = bundle_tool.build_plan(
        local_root=tmp_path / "second-plan",
        remote_root="/remote/current7",
        receipt_path=fixture["receipt"],
        generation=fixture["generation"],
        candidate_path=fixture["candidate"],
        quality_path=fixture["quality"],
        dataset_path=fixture["dataset"],
        profile_path=fixture["profile"],
        code_identity={"revision": "9" * 40, "clean": True},
        code_sources=fixture["code_sources"],
        optimizer_entrypoint="tools/fake_current7_search.py",
        warm_starts=fixture["warm"],
        runtime_packages=fixture["runtime_packages"],
    )
    assert second_plan["bundle_id"] == plan["bundle_id"]
    assert second_manifest == manifest


def test_phase_b_bundle_build_seals_every_required_remote_code_file(tmp_path):
    fixture = _fixture(tmp_path)
    required = bundle_tool.phase_b_remote_code_files()
    required_hashes = bundle_tool.phase_b_required_code_sha256()
    assert required == tuple(sorted(phase_b_contract.REMOTE_CODE_FILES))
    optimizer_entrypoint = "regression_260707/optimization/nsga2_problem.py"
    collected_sources = bundle_tool.collect_phase_b_tracked_code_sources(
        REPO,
        optimizer_entrypoint=optimizer_entrypoint,
    )
    tracked_module_files = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "module").rglob("*.py")
    }
    assert tracked_module_files
    assert {
        relative.removeprefix("artifacts/code/")
        for relative in collected_sources
        if relative.startswith("artifacts/code/module/")
    } == tracked_module_files
    assert "artifacts/code/module/input_parameter_260706.py" in collected_sources
    # Keep the publish fixture short enough for Windows while exercising the
    # exact optimizer import closure that failed on the topology canary.
    optimizer_import_files = {
        "regression_260707/optimization/nsga2_problem.py",
        "regression_260707/optimization/geometry_metrics.py",
        "regression_260707/optimization/design_summary.py",
        "regression_260707/model_targets.py",
    }
    code_sources = dict(fixture["code_sources"])
    code_sources.update(
        {
            f"artifacts/code/{relative}": REPO / relative
            for relative in {*required, *optimizer_import_files}
        }
    )
    plan, manifest = bundle_tool.build_plan(
        local_root=tmp_path / "phase-b-plan",
        remote_root="/remote/current7",
        receipt_path=fixture["receipt"],
        generation=fixture["generation"],
        candidate_path=fixture["candidate"],
        quality_path=fixture["quality"],
        dataset_path=fixture["dataset"],
        profile_path=fixture["profile"],
        code_identity={"revision": "9" * 40, "clean": True},
        code_sources=code_sources,
        optimizer_entrypoint=optimizer_entrypoint,
        warm_starts=fixture["warm"],
        runtime_packages=fixture["runtime_packages"],
        required_code_files=required,
        required_code_sha256=required_hashes,
    )

    attestation = bundle_tool.validate_required_runtime_code(
        manifest, required, required_code_sha256=required_hashes
    )
    assert attestation == manifest["required_runtime_code"]
    for relative in required:
        bundle_relative = f"artifacts/code/{relative}"
        assert manifest["code_inventory"][bundle_relative]["sha256"] == _sha(
            REPO / relative
        )

    published = _local_publish(tmp_path, fixture, plan, manifest)
    bundle_tool.validate_required_runtime_code(
        manifest,
        required,
        required_code_sha256=required_hashes,
        bundle_root=published,
    )
    import_smoke = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import pathlib, sys, types; "
                f"sys.path.insert(0, {str(published / 'artifacts' / 'code')!r}); "
                f"sys.path.insert(0, {str(published / 'artifacts' / 'code' / 'regression_260707')!r}); "
                "pymoo=types.ModuleType('pymoo'); "
                "core=types.ModuleType('pymoo.core'); "
                "problem=types.ModuleType('pymoo.core.problem'); "
                "problem.Problem=type('Problem', (), {}); "
                "sys.modules.update({'pymoo':pymoo,'pymoo.core':core,'pymoo.core.problem':problem}); "
                "import tools.tier1_final1000_multiseed_phase_b_contract; "
                "import tools.tier1_final1000_multiseed_phase_b_runner; "
                "import optimization.nsga2_problem as nsga; "
                "import module.input_parameter_260706 as parameters; "
                f"root=pathlib.Path({str(published / 'artifacts' / 'code')!r}).resolve(); "
                "assert pathlib.Path(nsga.__file__).resolve().is_relative_to(root); "
                "assert pathlib.Path(parameters.__file__).resolve().is_relative_to(root)"
            ),
        ],
        cwd=published,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_smoke.returncode == 0, import_smoke.stderr
    first = published / "artifacts" / "code" / Path(required[0])
    first.write_bytes(first.read_bytes() + b"\n# drift\n")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        bundle_tool.validate_required_runtime_code(
            manifest,
            required,
            required_code_sha256=required_hashes,
            bundle_root=published,
        )


def test_phase_b_bundle_build_refuses_missing_required_remote_code(tmp_path):
    fixture = _fixture(tmp_path)
    required = bundle_tool.phase_b_remote_code_files()
    with pytest.raises(RuntimeError, match="required runtime code is absent"):
        bundle_tool.build_plan(
            local_root=tmp_path / "missing-phase-b-plan",
            remote_root="/remote/current7",
            receipt_path=fixture["receipt"],
            generation=fixture["generation"],
            candidate_path=fixture["candidate"],
            quality_path=fixture["quality"],
            dataset_path=fixture["dataset"],
            profile_path=fixture["profile"],
            code_identity={"revision": "9" * 40, "clean": True},
            code_sources=fixture["code_sources"],
            optimizer_entrypoint="tools/fake_current7_search.py",
            warm_starts=fixture["warm"],
            runtime_packages=fixture["runtime_packages"],
            required_code_files=required,
            required_code_sha256=bundle_tool.phase_b_required_code_sha256(),
        )


def test_phase_b_bundle_build_refuses_semlock_helper_byte_drift(tmp_path):
    fixture = _fixture(tmp_path)
    required = bundle_tool.phase_b_remote_code_files()
    code_sources = dict(fixture["code_sources"])
    code_sources.update(
        {
            f"artifacts/code/{relative}": REPO / relative
            for relative in required
        }
    )
    helper_relative = next(iter(bundle_tool.phase_b_required_code_sha256()))
    drifted_helper = tmp_path / "drifted-semlock-helper.py"
    original = (REPO / helper_relative).read_bytes()
    drifted_helper.write_bytes(original.replace(b"\r\n", b"\n") + b"\n# drift\n")
    code_sources[f"artifacts/code/{helper_relative}"] = drifted_helper

    with pytest.raises(RuntimeError, match="hard hash mismatch"):
        bundle_tool.build_plan(
            local_root=tmp_path / "drifted-phase-b-plan",
            remote_root="/remote/current7",
            receipt_path=fixture["receipt"],
            generation=fixture["generation"],
            candidate_path=fixture["candidate"],
            quality_path=fixture["quality"],
            dataset_path=fixture["dataset"],
            profile_path=fixture["profile"],
            code_identity={"revision": "9" * 40, "clean": True},
            code_sources=code_sources,
            optimizer_entrypoint="tools/fake_current7_search.py",
            warm_starts=fixture["warm"],
            runtime_packages=fixture["runtime_packages"],
            required_code_files=required,
            required_code_sha256=bundle_tool.phase_b_required_code_sha256(),
        )


def test_bundle_refuses_adapter_code_revision_mismatch_before_plan_write(tmp_path):
    fixture = _fixture(tmp_path, adapter_code_revision="4" * 40)
    plan_root = tmp_path / "mismatched-plan-must-not-exist"

    with pytest.raises(RuntimeError, match="bundle/adapter code identity mismatch"):
        bundle_tool.build_plan(
            local_root=plan_root,
            remote_root="/remote/current7",
            receipt_path=fixture["receipt"],
            generation=fixture["generation"],
            candidate_path=fixture["candidate"],
            quality_path=fixture["quality"],
            dataset_path=fixture["dataset"],
            profile_path=fixture["profile"],
            code_identity={"revision": "9" * 40, "clean": True},
            code_sources=fixture["code_sources"],
            optimizer_entrypoint="tools/fake_current7_search.py",
            warm_starts=fixture["warm"],
            runtime_packages=fixture["runtime_packages"],
        )

    assert not plan_root.exists()


def test_training_profile_hash_accepts_non_ascii_locally_and_remotely(
    tmp_path, monkeypatch
):
    profile_payload = {"profile": "표준", "version": 7}
    training_sha = receipt_contract.training_profile_sha256(profile_payload)
    assert training_sha != receipt_contract.canonical_sha256(profile_payload)

    fixture, plan, manifest = _plan(
        tmp_path,
        profile_payload=profile_payload,
    )
    identity = manifest["adapter_receipt"]["identity"]
    assert identity["profile_canonical_sha256"] == training_sha

    bundle = _local_publish(tmp_path, fixture, plan, manifest)
    task = bundle_tool.build_task_waves(plan, manifest)["canaries"][0]
    payload = task["payload_json"]
    payload_root = tmp_path / "scheduler-runs"
    payload_path = payload_root / "task-1" / "payload.json"
    _write_json(payload_path, payload)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: manifest["runtime"]["critical_packages"][name],
    )

    verified, _observed_manifest, _relocation = runner.verify_payload(
        bundle,
        payload_path,
        payload_root,
        receipt_contract.canonical_sha256(payload),
    )
    assert verified == payload


def test_task_waves_are_4_plus_32_and_use_requested_resources(tmp_path):
    _fixture_value, plan, manifest = _plan(tmp_path)
    waves = bundle_tool.build_task_waves(plan, manifest)
    assert len(waves["canaries"]) == 4
    assert len(waves["ramp"]) == 32
    tasks = waves["canaries"] + waves["ramp"]
    assert len({task["payload_json"]["seed"] for task in tasks}) == 36
    assert len({task["dedupe_key"] for task in tasks}) == 36
    assert [task["payload_json"]["seed"] for task in waves["canaries"]] == [
        2_107_210_000,
        2_207_210_000,
        2_307_210_000,
        2_407_210_000,
    ]
    for task in tasks:
        assert task["payload_json"]["max_generations"] == 300
        assert (
            task["payload_json"]["population"]
            * task["payload_json"]["max_generations"]
            == 96_000
        )
        assert task["cpus"] == 8
        assert task["memory_mb"] == 65_536
        assert task["max_workers_per_node"] == 4
        assert task["scheduling_profile"] == "standard"
        assert task["aedt_backend"] == "standalone"
        assert task["gpus"] == 0
        assert task["priority"] == 0
        assert task["payload_json"]["optimizer_processes"] == 1
        assert task["payload_json"]["offspring_physics_repair"] is True
        fixed_turns = str(task["payload_json"]["lane"]["fixed_primary_turns"])
        assert task["payload_json"]["optimizer_repair_contract_sha256"] == (
            manifest["adapter_receipt"]["identity"][
                "optimizer_repair_contract_sha256_by_fixed_primary_turns"
            ][fixed_turns]
        )
        assert "$PWD/artifacts/python-site" in task["command"]
        assert "tier1_corrected_current7_slurm_seed_runner.py" in task["command"]
        assert "tier1_slurm_seed_runner.py" not in task["command"]


def test_smoke_only_or_repair_false_receipt_cannot_render_tasks(tmp_path):
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _plan(tmp_path, repair_ready=False)


def test_fast_ramp_releases_all_32_without_waiting_for_completion(tmp_path):
    _fixture_value, _plan_value, manifest = _plan(tmp_path)
    statuses = []
    for lane in manifest["fast_ramp"]["canaries"]:
        statuses.append(
            {
                "schema_version": bundle_tool.STATUS_SCHEMA,
                "bundle_id": manifest["bundle_id"],
                "seed": lane["seed"],
                "island_id": lane["island_id"],
                "ramp_gate_passed": True,
                "optimizer_processes": 1,
                "inference_threads": 8,
                "loaded_model_count": 20,
                "full_generation_authentication_passes": 1,
                "authenticated_artifact_count": 42,
                "observed_peak_rss_bytes": 20 * 1024**3,
                "optimizer_started": True,
                "terminal": False,
            }
        )
    decision = bundle_tool.assess_fast_ramp(manifest, statuses)
    assert decision["eligible"] is True
    assert decision["release_count"] == 32
    assert decision["wait_for_canary_optimizer_completion"] is False
    assert decision["scheduler_submission_performed"] is False

    over_rss = copy.deepcopy(statuses)
    over_rss[0]["observed_peak_rss_bytes"] = 43 * 1024**3
    assert bundle_tool.assess_fast_ramp(manifest, over_rss)["eligible"] is False
    terminal = copy.deepcopy(statuses)
    terminal[0]["terminal"] = True
    assert bundle_tool.assess_fast_ramp(manifest, terminal)["eligible"] is False


def test_hash_chained_ledger_refills_open_ended_24_4_4_4_quotas(tmp_path):
    _fixture_value, plan, manifest = _plan(tmp_path)
    ledger = bundle_tool.initial_seed_ledger(plan, manifest)
    assert len(ledger["entries"]) == 36
    assert list(ledger["next_seed_by_island"].values()) == [
        2_107_210_024,
        2_207_210_004,
        2_307_210_004,
        2_407_210_004,
    ]
    assert bundle_tool.plan_refill_wave(plan, manifest, ledger)["refill_count"] == 0

    for _cycle in range(3):
        terminal_entries = [
            {
                **entry,
                "state": (
                    "completed"
                    if entry["state"] in bundle_tool.ACTIVE_LEDGER_STATES
                    else entry["state"]
                ),
            }
            for entry in ledger["entries"]
        ]
        terminal = bundle_tool.seal_seed_ledger(
            manifest,
            terminal_entries,
            revision=ledger["revision"] + 1,
            parent_ledger_sha256=ledger["ledger_sha256"],
        )
        refill = bundle_tool.plan_refill_wave(plan, manifest, terminal)
        assert refill["eligible"] is True
        assert refill["refill_count"] == 36
        assert refill["scheduler_submission_performed"] is False
        assert list(refill["quota_gaps"].values()) == [24, 4, 4, 4]
        ledger = refill["next_ledger"]
        assert ledger["parent_ledger_sha256"] == terminal["ledger_sha256"]

    assert len(ledger["entries"]) == 144
    assert len({entry["seed"] for entry in ledger["entries"]}) == 144
    active_counts = {
        island_id: sum(
            entry["island_id"] == island_id
            and entry["state"] in bundle_tool.ACTIVE_LEDGER_STATES
            for entry in ledger["entries"]
        )
        for island_id in manifest["open_ended_refill"]["island_active_quotas"]
    }
    assert active_counts == manifest["open_ended_refill"]["island_active_quotas"]

    stopped = bundle_tool.seal_seed_ledger(
        manifest,
        ledger["entries"],
        revision=ledger["revision"] + 1,
        parent_ledger_sha256=ledger["ledger_sha256"],
        stop_requested=True,
    )
    decision = bundle_tool.plan_refill_wave(plan, manifest, stopped)
    assert decision["eligible"] is False
    assert decision["reason"] == "explicit_stop_requested"
    assert decision["tasks"] == []


def test_remote_runner_verifies_relocation_and_completes_fake_seed(
    tmp_path, monkeypatch
):
    fixture, plan, manifest = _plan(tmp_path)
    bundle = _local_publish(tmp_path, fixture, plan, manifest)
    task = bundle_tool.build_task_waves(plan, manifest)["canaries"][0]
    payload = task["payload_json"]
    payload_root = tmp_path / "scheduler-runs"
    payload_path = payload_root / "task-1" / "payload.json"
    _write_json(payload_path, payload)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: manifest["runtime"]["critical_packages"][name],
    )
    payload_sha = receipt_contract.canonical_sha256(payload)
    verified, observed_manifest, relocation = runner.verify_payload(
        bundle, payload_path, payload_root, payload_sha
    )
    assert verified == payload
    assert observed_manifest["bundle_id"] == manifest["bundle_id"]
    assert relocation["source_absolute_paths_are_documentary_only"] is True

    monkeypatch.setenv("SLURM_SCHED_TASK_ID", "local-smoke-1")
    assert (
        runner.run(
            bundle, payload_path, payload_root, payload_sha, heartbeat_seconds=0.02
        )
        == 0
    )
    status = json.loads(
        (bundle / "runs" / "task-local-smoke-1" / "seed_status.json").read_text()
    )
    assert status["state"] == "completed"
    assert status["terminal"] is True
    assert status["full_generation_authentication_passes"] == 1
    assert status["authenticated_artifact_count"] == 42
    assert status["observed_peak_rss_bytes"] <= 42 * 1024**3
    result_path = next(
        path
        for path in (bundle / "runs" / "task-local-smoke-1").rglob("result.json")
    )
    valid_result = json.loads(result_path.read_text())
    assert valid_result["optimizer_processes"] == 1
    runner.validate_result(valid_result, payload=payload, manifest=manifest)

    missing_process_count = copy.deepcopy(valid_result)
    missing_process_count.pop("optimizer_processes")
    missing_process_count.pop("payload_sha256")
    missing_process_count["payload_sha256"] = receipt_contract.canonical_sha256(
        missing_process_count
    )
    with pytest.raises(RuntimeError, match="terminal current7 result"):
        runner.validate_result(
            missing_process_count, payload=payload, manifest=manifest
        )

    wrong_process_count = copy.deepcopy(valid_result)
    wrong_process_count["optimizer_processes"] = 2
    wrong_process_count.pop("payload_sha256")
    wrong_process_count["payload_sha256"] = receipt_contract.canonical_sha256(
        wrong_process_count
    )
    with pytest.raises(RuntimeError, match="terminal current7 result"):
        runner.validate_result(wrong_process_count, payload=payload, manifest=manifest)


def test_remote_runner_rejects_relocated_artifact_tamper(tmp_path, monkeypatch):
    fixture, plan, manifest = _plan(tmp_path)
    bundle = _local_publish(tmp_path, fixture, plan, manifest)
    task = bundle_tool.build_task_waves(plan, manifest)["canaries"][0]
    payload = task["payload_json"]
    payload_root = tmp_path / "scheduler-runs"
    payload_path = payload_root / "task-1" / "payload.json"
    _write_json(payload_path, payload)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: manifest["runtime"]["critical_packages"][name],
    )
    artifact = next(iter(manifest["generation_artifacts"]))
    generation = bundle / (
        "artifacts/registry/generations/"
        + manifest["adapter_receipt"]["identity"]["training_run_id"]
    )
    (generation / artifact).write_bytes(b"tampered-size")
    with pytest.raises(RuntimeError, match="missing/wrong size"):
        runner.verify_payload(
            bundle,
            payload_path,
            payload_root,
            receipt_contract.canonical_sha256(payload),
        )


def test_remote_runner_rejects_cross_stratum_repair_contract(tmp_path, monkeypatch):
    fixture, plan, manifest = _plan(tmp_path)
    bundle = _local_publish(tmp_path, fixture, plan, manifest)
    task = bundle_tool.build_task_waves(plan, manifest)["canaries"][0]
    payload = copy.deepcopy(task["payload_json"])
    assert payload["lane"]["fixed_primary_turns"] == 5
    payload["optimizer_repair_contract_sha256"] = manifest[
        "adapter_receipt"
    ]["identity"]["optimizer_repair_contract_sha256_by_fixed_primary_turns"]["6"]
    payload_root = tmp_path / "scheduler-runs"
    payload_path = payload_root / "task-1" / "payload.json"
    _write_json(payload_path, payload)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: manifest["runtime"]["critical_packages"][name],
    )
    with pytest.raises(RuntimeError, match="N1 stratum"):
        runner.verify_payload(
            bundle,
            payload_path,
            payload_root,
            receipt_contract.canonical_sha256(payload),
        )


class _FakePublicationTransport:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/"}
        self.modes: dict[str, int] = {"/": 0o755}
        self.write_count = 0
        self.operations: list[tuple[str, str]] = []
        self.file_writes: list[str] = []
        self.fail_after: int | None = None

    def _write(self, operation: str, path: str) -> None:
        if self.fail_after is not None and self.write_count >= self.fail_after:
            raise RuntimeError("injected publication interruption")
        self.write_count += 1
        self.operations.append((operation, path))

    def exists(self, path: str) -> bool:
        return path in self.files or path in self.dirs

    def is_dir(self, path: str) -> bool:
        return path in self.dirs

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        if path in self.dirs:
            return
        self._write(f"mkdir:{mode:o}", path)
        pieces = path.rstrip("/").split("/")
        for index in range(1, len(pieces) + 1):
            value = "/".join(pieces[:index]) or "/"
            if path.startswith("/"):
                value = "/" + value.lstrip("/")
            self.dirs.add(value)
            self.modes.setdefault(value, mode if value == path else 0o755)

    def upload_file(self, local: Path, remote: str) -> None:
        self._write("upload", remote)
        self.files[remote] = local.read_bytes()
        self.file_writes.append(remote)

    def write_bytes(self, path: str, value: bytes) -> None:
        self._write("write", path)
        self.files[path] = bytes(value)
        self.file_writes.append(path)

    def write_ready(self, path: str, value: bytes) -> None:
        self.write_bytes(path, value)
        self.modes[path] = 0o444
        self.modes[path.rsplit("/", 1)[0]] = 0o555

    def replace(self, source: str, destination: str) -> None:
        self._write("replace", destination)
        self.files[destination] = self.files.pop(source)

    def file_record(self, path: str):
        value = self.files.get(path)
        if value is None:
            return None
        return {
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }

    def verify_runtime(
        self,
        root: str,
        *,
        requirements_relative: str,
        expected_packages: dict[str, str],
    ):
        assert requirements_relative == "artifacts/runtime/requirements.lock"
        self._write("runtime", root)
        return dict(expected_packages)

    def seal_permissions(self, root: str) -> None:
        self._write("seal", root)
        self.modes[root] = 0o755
        for path in self.dirs:
            if path == root + "/artifacts" or path.startswith(root + "/artifacts/"):
                self.modes[path] = 0o555
        for path in self.files:
            if path.startswith(root + "/artifacts/"):
                self.modes[path] = 0o444
        self.modes[root + "/bundle_manifest.json"] = 0o444
        self.modes[root + "/.publication-journal.json"] = 0o444
        self.modes[root + "/runs"] = 0o1777

    def promote_directory(self, incoming: str, destination: str) -> None:
        self._write("promote", destination)
        moved_files = {
            destination + path.removeprefix(incoming): value
            for path, value in self.files.items()
            if path == incoming or path.startswith(incoming + "/")
        }
        for path in list(self.files):
            if path == incoming or path.startswith(incoming + "/"):
                del self.files[path]
        moved_dirs = {
            destination + path.removeprefix(incoming)
            for path in self.dirs
            if path == incoming or path.startswith(incoming + "/")
        }
        self.dirs = {
            path
            for path in self.dirs
            if path != incoming and not path.startswith(incoming + "/")
        }
        self.files.update(moved_files)
        self.dirs.update(moved_dirs)
        moved_modes = {
            destination + path.removeprefix(incoming): mode
            for path, mode in self.modes.items()
            if path == incoming or path.startswith(incoming + "/")
        }
        self.modes = {
            path: mode
            for path, mode in self.modes.items()
            if path != incoming and not path.startswith(incoming + "/")
        }
        self.modes.update(moved_modes)


class _FakeScheduler:
    def __init__(self):
        self.post_count = 0
        self.next_id = 1000
        self.tasks: dict[int, dict] = {}
        self.by_dedupe: dict[str, int] = {}
        self.seed_statuses: dict[int, dict] = {}
        self.crash_after_create_once = False

    def find_task_by_dedupe(self, dedupe_key: str):
        task_id = self.by_dedupe.get(dedupe_key)
        return None if task_id is None else dict(self.tasks[task_id])

    def submit_task(self, payload):
        dedupe = payload["dedupe_key"]
        assert "requested_allocation_id" not in payload
        if dedupe in self.by_dedupe:
            return dict(self.tasks[self.by_dedupe[dedupe]])
        task_id = self.next_id
        self.next_id += 1
        task = {
            **copy.deepcopy(dict(payload)),
            "id": task_id,
            "task_id": task_id,
            "status": "queued",
        }
        self.tasks[task_id] = task
        self.by_dedupe[dedupe] = task_id
        self.post_count += 1
        if self.crash_after_create_once:
            self.crash_after_create_once = False
            raise RuntimeError("injected post-response interruption")
        return dict(task)

    def get_task(self, task_id: int):
        value = self.tasks.get(task_id)
        return None if value is None else dict(value)

    def read_seed_status(self, task_id: int):
        value = self.seed_statuses.get(task_id)
        return None if value is None else dict(value)


def _published_fixture(tmp_path: Path):
    fixture, plan, manifest = _plan(tmp_path)
    remote = _FakePublicationTransport()
    plan_path = Path(plan["bundle_manifest"]).parent / "offload_plan.json"
    publication = publisher.publish_bundle(plan_path, apply=True, transport=remote)
    return fixture, plan, manifest, plan_path, remote, publication


def _pass_canary_gates(scheduler: _FakeScheduler, manifest: dict) -> None:
    for lane in manifest["fast_ramp"]["canaries"]:
        matching = [
            task
            for task in scheduler.tasks.values()
            if task["payload_json"]["seed"] == lane["seed"]
        ]
        assert len(matching) == 1
        task = matching[0]
        scheduler.seed_statuses[task["id"]] = {
            "schema_version": bundle_tool.STATUS_SCHEMA,
            "bundle_id": manifest["bundle_id"],
            "seed": lane["seed"],
            "island_id": lane["island_id"],
            "ramp_gate_passed": True,
            "optimizer_processes": 1,
            "inference_threads": 8,
            "loaded_model_count": 20,
            "full_generation_authentication_passes": 1,
            "authenticated_artifact_count": 42,
            "observed_peak_rss_bytes": 20 * 1024**3,
            "optimizer_started": True,
            "terminal": False,
        }


def test_publisher_dry_run_write_zero_and_interrupted_apply_resumes_ready_last(
    tmp_path,
):
    _fixture_value, plan, manifest = _plan(tmp_path)
    plan_path = Path(plan["bundle_manifest"]).parent / "offload_plan.json"
    remote = _FakePublicationTransport()
    dry = publisher.publish_bundle(plan_path, apply=False, transport=remote)
    assert dry["publication_complete"] is False
    assert dry["remote_write_performed"] is False
    assert remote.write_count == 0

    remote.fail_after = 12
    with pytest.raises(RuntimeError, match="interruption"):
        publisher.publish_bundle(plan_path, apply=True, transport=remote)
    assert not remote.exists(plan["remote_bundle"] + "/READY.json")
    partial_count = sum(
        remote.file_record(dry["incoming_bundle"] + "/" + relative) == expected
        for relative, expected in manifest["files"].items()
    )
    assert partial_count > 0

    remote.fail_after = None
    receipt = publisher.publish_bundle(plan_path, apply=True, transport=remote)
    assert receipt["publication_complete"] is True
    assert receipt["resumed_files"] >= partial_count
    assert remote.exists(plan["remote_bundle"] + "/READY.json")
    assert remote.file_writes[-1].endswith("/READY.json")
    assert remote.operations[-1] == ("promote", plan["remote_bundle"])
    ready = receipt["ready"]
    assert ready["every_file_sha256_verified"] is True
    assert ready["code_inventory_sha256"] == manifest["code_inventory_sha256"]
    assert (
        ready["relocation_contract_sha256"] == manifest["relocation"]["contract_sha256"]
    )
    assert ready["remote_git_checkout_performed"] is False
    final = plan["remote_bundle"]
    assert remote.modes[final] & 0o005 == 0o005
    assert remote.modes[final] & 0o222 == 0
    assert remote.modes[final + "/bundle_manifest.json"] & 0o004
    assert remote.modes[final + "/READY.json"] == 0o444
    assert remote.modes[final + "/runs"] == 0o1777
    for relative in manifest["files"]:
        path = final + "/" + relative
        assert remote.modes[path] & 0o004, path
        assert remote.modes[path] & 0o222 == 0, path

    before = remote.write_count
    repeated = publisher.publish_bundle(plan_path, apply=True, transport=remote)
    assert repeated["already_ready"] is True
    assert remote.write_count == before


def test_publisher_ready_and_pinned_site_satisfy_remote_runner_contract(
    tmp_path, monkeypatch
):
    _fixture_value, plan, manifest, _plan_path, remote, _publication = (
        _published_fixture(tmp_path)
    )
    remote_prefix = plan["remote_bundle"] + "/"
    local_bundle = tmp_path / "publisher-materialized"
    for path, value in remote.files.items():
        if not path.startswith(remote_prefix):
            continue
        destination = local_bundle / path.removeprefix(remote_prefix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
    task = bundle_tool.build_task_waves(plan, manifest)["canaries"][0]
    payload = task["payload_json"]
    payload_root = tmp_path / "scheduler-runs"
    payload_path = payload_root / "task-51" / "payload.json"
    _write_json(payload_path, payload)
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: manifest["runtime"]["critical_packages"][name],
    )
    verified, observed_manifest, _relocation = runner.verify_payload(
        local_bundle,
        payload_path,
        payload_root,
        receipt_contract.canonical_sha256(payload),
    )
    assert verified == payload
    assert observed_manifest["bundle_id"] == manifest["bundle_id"]
    assert "$PWD/artifacts/python-site" in task["command"]


def test_controller_dry_run_has_zero_local_and_scheduler_writes(tmp_path):
    _fixture_value, plan, _manifest, plan_path, remote, publication = (
        _published_fixture(tmp_path)
    )
    scheduler = _FakeScheduler()
    state_path = tmp_path / "controller.json"
    result = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=False,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert result["scheduler_post_count"] == 0
    assert result["state_writes"] == 0
    assert not state_path.exists()
    assert scheduler.post_count == 0
    assert sum(action["action"] == "would_submit" for action in result["actions"]) == 4
    assert plan["submission_performed"] is False


def test_controller_rejects_repair_false_before_any_write(tmp_path):
    fixture, _plan_value, _manifest, plan_path, _remote, publication = (
        _published_fixture(tmp_path)
    )
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["strata"].pop("6")
    receipt.pop("payload_sha256")
    receipt["payload_sha256"] = receipt_contract.canonical_sha256(receipt)
    _write_json(fixture["receipt"], receipt)
    scheduler = _FakeScheduler()
    with pytest.raises(RuntimeError):
        controller.control_once(
            plan_path,
            publication,
            state_path=tmp_path / "blocked.json",
            apply=True,
            scheduler=scheduler,
        )
    assert scheduler.post_count == 0
    assert not (tmp_path / "blocked.json").exists()


def test_controller_recovers_post_crash_then_releases_exact_32_once(tmp_path):
    _fixture_value, _plan_value, manifest, plan_path, remote, publication = (
        _published_fixture(tmp_path)
    )
    scheduler = _FakeScheduler()
    scheduler.crash_after_create_once = True
    state_path = tmp_path / "controller.json"
    with pytest.raises(RuntimeError, match="post-response"):
        controller.control_once(
            plan_path,
            publication,
            state_path=state_path,
            apply=True,
            scheduler=scheduler,
            ready_probe=controller.TransportReadyProbe(remote),
        )
    assert scheduler.post_count == 1
    assert len(scheduler.tasks) == 1

    held = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert held["submission_count"] == 4
    assert scheduler.post_count == 4
    assert held["ramp_released"] is False

    _pass_canary_gates(scheduler, manifest)
    released = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert released["ramp_released"] is True
    assert released["submission_count"] == 36
    assert scheduler.post_count == 36
    assert (
        sum(
            action["action"] == "submitted" and action["wave"] == "ramp"
            for action in released["actions"]
        )
        == 32
    )

    repeated = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert repeated["submission_count"] == 36
    assert scheduler.post_count == 36


def test_controller_refills_each_terminal_island_gap_without_duplicates(tmp_path):
    _fixture_value, _plan_value, manifest, plan_path, remote, publication = (
        _published_fixture(tmp_path)
    )
    scheduler = _FakeScheduler()
    state_path = tmp_path / "controller.json"
    controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    _pass_canary_gates(scheduler, manifest)
    controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert scheduler.post_count == 36
    for island_id in manifest["open_ended_refill"]["island_active_quotas"]:
        matching = [
            task
            for task in scheduler.tasks.values()
            if task["payload_json"]["lane"]["island_id"] == island_id
        ]
        scheduler.tasks[matching[0]["id"]]["status"] = "completed"
    result = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert scheduler.post_count == 40
    assert result["submission_count"] == 40
    assert result["ledger_active_count"] == 36
    assert result["ledger_terminal_count"] == 4
    assert len(scheduler.by_dedupe) == 40

    repeated = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
    )
    assert scheduler.post_count == 40
    assert repeated["ledger_active_count"] == 36

    # Even with fresh terminal gaps available, the explicit stop is sealed
    # before the controller can reserve or POST another refill.
    refill_tasks = [
        task
        for task in scheduler.tasks.values()
        if task["payload_json"]["lane"]["wave"] == "refill"
    ]
    assert len(refill_tasks) == 4
    for task in refill_tasks:
        scheduler.tasks[task["id"]]["status"] = "completed"
    stopped = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
        request_stop=True,
    )
    assert stopped["stop_requested"] is True
    assert scheduler.post_count == 40


def test_controller_stop_flag_precedes_canary_or_refill_submission(tmp_path):
    _fixture_value, _plan_value, _manifest, plan_path, remote, publication = (
        _published_fixture(tmp_path)
    )
    scheduler = _FakeScheduler()
    state_path = tmp_path / "controller.json"
    result = controller.control_once(
        plan_path,
        publication,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=controller.TransportReadyProbe(remote),
        request_stop=True,
    )
    assert result["stop_requested"] is True
    assert result["scheduler_post_count"] == 0
    assert scheduler.post_count == 0
    state = json.loads(state_path.read_text())
    assert state["ledger"]["stop_requested"] is True


def test_controller_rejects_live_ready_mismatch_before_scheduler_post(tmp_path):
    _fixture_value, plan, _manifest, plan_path, remote, publication = (
        _published_fixture(tmp_path)
    )
    ready_path = plan["remote_bundle"] + "/READY.json"
    value = json.loads(remote.files[ready_path])
    value["bundle_manifest_sha256"] = "0" * 64
    remote.files[ready_path] = json.dumps(value).encode()
    scheduler = _FakeScheduler()
    with pytest.raises(RuntimeError, match="live remote READY"):
        controller.control_once(
            plan_path,
            publication,
            state_path=tmp_path / "controller.json",
            apply=True,
            scheduler=scheduler,
            ready_probe=controller.TransportReadyProbe(remote),
        )
    assert scheduler.post_count == 0
