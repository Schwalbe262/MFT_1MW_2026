from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def _variant_probe(environment=None):
    code = r'''
import json
from tools import tier1_thermal_balanced_crossover_slurm_rolling as entry
r = entry.rolling
hard = {
  "Llt_target_uH": 27.5,
  "Llt_tol_uH": 0.55,
  "T_limit_C": 110.0,
  "B_limit_T": 1.2,
  "insulation_min_mm": 40.0,
  "n_core_group_max": 4,
  "primary_conductor_thickness_mm": 5.0,
  "resonance_min_Hz": 15000.0,
  "magnetizing_inductance_factor": 0.5,
  "size_W_max_mm": 1200.0,
  "size_L_max_mm": 1200.0,
  "size_H_max_mm": 750.0,
  "q_sigma": 1.0,
  "uncertainty_contract": "q90_conformal_half_width_physical_v1",
}
thermal = r.temperature_constraint_contract(hard)
acquisition = r.thermal_crossover_acquisition_contract(hard)
pointer = {"current": {
  "cohort_id": "res15k-balanced4c-fixture-0123456789",
  "bundle_id": "tier1-res15k-balanced4c-fixture-0123456789",
  "bundle_manifest_sha256": "a"*64,
  "constraint_version": (
    "mft-tier1-envelope-1200x1200x750-res15k-"
    "t110all11-core4-5t-lmhalf-v4"
  ),
  "hard_spec": hard,
  "hard_spec_sha256": r.canonical_sha(hard),
  "attestation_schema_version": r.ATTESTATION_SCHEMA,
  "task_schema_version": r.TASK_SCHEMA,
  "controller_source_sha256": "b"*64,
  "temperature_constraint_contract": thermal,
  "temperature_constraint_contract_sha256": r.canonical_sha(thermal),
  "source_model_manifest_sha256": "c"*64,
  "deployment_model_manifest_sha256": "d"*64,
  "nsga_code_revision": "e"*40,
  "warm_start_sha256": "f"*64,
  "base_python_site": "/gpfs/base/python-site",
  "remote_bundle": "/gpfs/tier1/bundle",
  "rolling_target": r.ROLLING_TARGET,
  "refill_allowed": True,
  "search_profile": r.search_profile(),
  "optimizer_resonance_scale_Hz": r.OPTIMIZER_RESONANCE_SCALE_HZ,
  "optimizer_core_thermal_scale_C": r.OPTIMIZER_CORE_THERMAL_SCALE_C,
  "acquisition_ranking_contract": acquisition,
  "acquisition_ranking_contract_sha256": acquisition["sha256"],
  "production_eligible": False,
  "fea_submission_approved": False,
  "fea_submission_performed": False,
  "aedt_used": False,
  "automatic_promotion_allowed": False,
}}
task = r.task_payload(pointer, r.SEED_START)
print(json.dumps({
  "runtime": str(r.RUNTIME),
  "warm": str(r.DEFAULT_WARM),
  "profile": r.search_profile(),
  "prefix": task["name"],
  "priority": task["priority"],
  "dedupe": task["dedupe_key"],
  "payload": task["payload_json"],
  "hard": hard,
  "thermal": thermal,
  "acquisition": acquisition,
  "thermal_overrides": r.optimizer_core_thermal_overrides(),
}, sort_keys=True))
'''
    env = os.environ.copy()
    env.pop("MFT_TIER1_SEARCH_VARIANT", None)
    if environment:
        env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_balanced_crossover_has_distinct_fail_closed_identity():
    process = _variant_probe()
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    assert value["runtime"].endswith(
        r"mft_tier1_nsga_thermal_balanced4c_t110_res15k_260719"
    )
    assert value["warm"].endswith(
        r"t110-res15k-thermal-crossover-balanced-v1\next_warm_start.npy"
    )
    acquisition_sha = profile.pop("acquisition_ranking_contract_sha256")
    assert len(acquisition_sha) == 64
    assert set(acquisition_sha) <= set("0123456789abcdef")
    assert profile == {
        "schema_version": "mft-tier1-nsga-search-profile-v1",
        "namespace": (
            "thermal-crossover-balanced4c-res250-p320-g600-warm64-v1"
        ),
        "variant": "thermal-crossover-balanced4c-v1",
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "rolling_target": 32,
        "seed_start": 1_907_194_000,
        "task_priority": -5,
        "optimizer_resonance_scale_Hz": 250.0,
        "optimizer_scale_scope": (
            "search_pressure_only_physical_G_unchanged"
        ),
        "optimizer_core_thermal_scale_C": 4.0,
        "optimizer_core_thermal_constraints": [
            "temperature_robust_limit:T_max_core",
            "temperature_robust_limit:Tprobe_core_center_max",
            "temperature_robust_limit:Tprobe_core_top_yoke_max",
        ],
        "soft_axis_target_W_mm": 1_000.0,
        "soft_axis_target_L_mm": 1_000.0,
        "soft_axis_pressure_scope": (
            "ranking_only_positive_excess_no_hard_G_mutation"
        ),
        "legacy_migration_from_priority": -6,
        "priority_migration_seed_cutoff_exclusive": 1_907_194_000,
        "refill_policy": "continuous-unbounded-seeds-v1",
    }
    assert value["prefix"].startswith("mft-t1therm4-")
    assert value["priority"] == -5
    assert value["dedupe"].startswith(
        "mft-tier1-thermal-crossover-balanced4c-nsga:"
    )
    payload = value["payload"]
    assert payload["hard_spec"] == value["hard"]
    assert payload["hard_spec"]["T_limit_C"] == 110.0
    assert payload["hard_spec"]["resonance_min_Hz"] == 15_000.0
    assert payload["hard_spec"]["size_H_max_mm"] == 750.0
    assert value["thermal"]["robust_upper_bound_C"] == 110.0
    assert payload["optimizer_resonance_scale_Hz"] == 250.0
    assert payload["optimizer_core_thermal_scale_C"] == 4.0
    assert payload["seed"] == 1_907_194_000
    assert payload["population"] == 320
    assert payload["max_generations"] == 600
    assert payload["inference_threads"] == 8
    assert value["thermal_overrides"] == {
        name: 4.0
        for name in profile["optimizer_core_thermal_constraints"]
    }
    assert value["acquisition"]["core_thermal_positive_G_scale_C"] == 4.0
    assert value["acquisition"]["authoritative_terminal_G"] == (
        "physical_unscaled_unchanged"
    )
    assert value["acquisition"]["hard_constraint_mutation"] is False
    for field in (
        "production_eligible",
        "fea_submission_approved",
        "fea_submission_performed",
        "aedt_used",
        "automatic_promotion_allowed",
    ):
        assert payload[field] is False


def test_balanced_crossover_entrypoint_rejects_conflicting_variant():
    process = _variant_probe({
        "MFT_TIER1_SEARCH_VARIANT": "thermal-crossover-core1c-v1"
    })
    assert process.returncode != 0
    assert "conflicts with balanced thermal-crossover entrypoint" in (
        process.stderr
    )


def test_balanced_crossover_entrypoint_rejects_cached_default_module():
    code = r'''
import os
from tools import tier1_slurm_rolling
os.environ.pop("MFT_TIER1_SEARCH_VARIANT", None)
from tools import tier1_thermal_balanced_crossover_slurm_rolling
'''
    env = os.environ.copy()
    env.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "cached tier1_slurm_rolling module" in process.stderr
