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
from tools import tier1_resonance_focus_slurm_rolling as entry
r = entry.rolling
hard = {"T_limit_C": 110.0}
thermal = r.temperature_constraint_contract(hard)
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
}}
task = r.task_payload(pointer, r.SEED_START)
print(json.dumps({
  "runtime": str(r.RUNTIME), "warm": str(r.DEFAULT_WARM),
  "profile": r.search_profile(), "prefix": task["name"],
  "priority": task["priority"], "dedupe": task["dedupe_key"],
  "payload_scale": task["payload_json"]["optimizer_resonance_scale_Hz"],
  "migration_cutoff": r.PRIORITY_MIGRATION_SEED_CUTOFF,
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


def test_resonance_focus_variant_has_distinct_immutable_defaults():
    process = _variant_probe()
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    assert value["runtime"].endswith(r"t1r250_260719")
    assert value["warm"].endswith(
        r"t110-res15k-resonance-focus-50x50-v1\next_warm_start.npy"
    )
    assert profile == {
        "schema_version": "mft-tier1-nsga-search-profile-v1",
        "namespace": "resonance-focus250-p320-g600-warm64-v1",
        "variant": "resonance-focus-250hz-v1",
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "rolling_target": 32,
        "seed_start": 1_907_192_000,
        "task_priority": -7,
        "optimizer_resonance_scale_Hz": 250.0,
        "optimizer_scale_scope": "search_pressure_only_physical_G_unchanged",
        "legacy_migration_from_priority": -8,
        "priority_migration_seed_cutoff_exclusive": 1_907_192_000,
        "refill_policy": "continuous-unbounded-seeds-v1",
    }
    assert value["prefix"].startswith("mft-t1res250-")
    assert value["priority"] == -7
    assert value["payload_scale"] == 250.0
    assert value["dedupe"].startswith("mft-tier1-resonance-focus250-nsga:")
    assert value["migration_cutoff"] == 1_907_192_000


def test_resonance_focus_entrypoint_rejects_conflicting_variant():
    process = _variant_probe({"MFT_TIER1_SEARCH_VARIANT": "normalized-warm-v1"})
    assert process.returncode != 0
    assert "conflicts with resonance-focus entrypoint" in process.stderr


def test_resonance_focus_entrypoint_rejects_cached_normalized_module():
    code = r'''
import os
from tools import tier1_slurm_rolling
os.environ.pop("MFT_TIER1_SEARCH_VARIANT", None)
from tools import tier1_resonance_focus_slurm_rolling
'''
    env = os.environ.copy()
    env.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "cached tier1_slurm_rolling module" in process.stderr
