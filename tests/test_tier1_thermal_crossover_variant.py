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
from tools import tier1_thermal_crossover_slurm_rolling as entry
r = entry.rolling
hard = {
  "T_limit_C": 110.0,
  "size_W_max_mm": 1200.0,
  "size_L_max_mm": 1200.0,
}
thermal = r.temperature_constraint_contract(hard)
acquisition = r.thermal_crossover_acquisition_contract(hard)
pointer = {"current": {
  "cohort_id": "res15k-fixture-0123456789",
  "bundle_id": "tier1-res15k-fixture-0123456789",
  "bundle_manifest_sha256": "a"*64,
  "constraint_version": "fixture",
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
  "runtime": str(r.RUNTIME), "warm": str(r.DEFAULT_WARM),
  "profile": r.search_profile(), "prefix": task["name"],
  "priority": task["priority"], "dedupe": task["dedupe_key"],
  "payload": task["payload_json"],
}, sort_keys=True))
'''
    env = os.environ.copy()
    env.pop("MFT_TIER1_SEARCH_VARIANT", None)
    if environment:
        env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env,
        capture_output=True, text=True, check=False,
    )


def test_thermal_crossover_variant_has_distinct_attested_identity():
    process = _variant_probe()
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    assert value["runtime"].endswith(
        r"mft_tier1_nsga_thermal_crossover_t110_res15k_260719"
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
            "thermal-crossover-core1c-res250-p320-g600-warm64-v1"
        ),
        "variant": "thermal-crossover-core1c-v1",
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "rolling_target": 32,
        "seed_start": 1_907_193_000,
        "task_priority": -6,
        "optimizer_resonance_scale_Hz": 250.0,
        "optimizer_scale_scope": "search_pressure_only_physical_G_unchanged",
        "optimizer_core_thermal_scale_C": 1.0,
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
        "legacy_migration_from_priority": -7,
        "priority_migration_seed_cutoff_exclusive": 1_907_193_000,
        "refill_policy": "continuous-unbounded-seeds-v1",
    }
    payload = value["payload"]
    assert value["prefix"].startswith("mft-t1therm1-")
    assert value["priority"] == -6
    assert value["dedupe"].startswith(
        "mft-tier1-thermal-crossover-core1c-nsga:"
    )
    assert payload["optimizer_resonance_scale_Hz"] == 250.0
    assert payload["optimizer_core_thermal_scale_C"] == 1.0
    assert payload["seed"] == 1_907_193_000
    assert payload["population"] == 320
    assert payload["max_generations"] == 600
    assert payload["inference_threads"] == 8
    assert payload["production_eligible"] is False
    assert payload["fea_submission_approved"] is False
    assert payload["fea_submission_performed"] is False
    assert payload["aedt_used"] is False
    assert payload["automatic_promotion_allowed"] is False


def test_thermal_crossover_entrypoint_rejects_conflicting_variant():
    process = _variant_probe({
        "MFT_TIER1_SEARCH_VARIANT": "resonance-focus-250hz-v1"
    })
    assert process.returncode != 0
    assert "conflicts with thermal-crossover entrypoint" in process.stderr


def test_thermal_crossover_entrypoint_rejects_cached_default_module():
    code = r'''
import os
from tools import tier1_slurm_rolling
os.environ.pop("MFT_TIER1_SEARCH_VARIANT", None)
from tools import tier1_thermal_crossover_slurm_rolling
'''
    env = os.environ.copy()
    env.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "cached tier1_slurm_rolling module" in process.stderr
