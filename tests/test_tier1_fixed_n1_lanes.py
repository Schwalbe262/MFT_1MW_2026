from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.tier1_resonance_feedback import (
    install_fixed_primary_turns_repair,
    prepare_fixed_primary_turns_warm_start,
)


REPO = Path(__file__).resolve().parents[1]


class _FakeProblem:
    n_var = 3

    def __init__(self):
        self.evaluated = None

    def repair_unit_coordinates(self, coordinates):
        return np.clip(np.asarray(coordinates, dtype=float), 0.0, 1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        self.evaluated = np.asarray(X, dtype=float).copy()
        out["F"] = np.zeros((len(self.evaluated), 2))

    def filter_warm_start_coordinates(self, coordinates):
        values = np.asarray(coordinates, dtype=float)
        kept = []
        seen = set()
        for row in values:
            identity = tuple(row.tolist())
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(row)
        filtered = np.asarray(kept, dtype=float).reshape(-1, self.n_var)
        return filtered, {
            "input_count": len(values),
            "hard_feasible_count": len(values),
            "decoded_unique_count": len(filtered),
            "rejected_count": len(values) - len(filtered),
        }


def test_fixed_primary_turn_repair_covers_1d_2d_and_fails_closed():
    problem = _FakeProblem()
    contract = install_fixed_primary_turns_repair(
        problem, 5, coordinate_index=1, minimum_turns=5, maximum_turns=8,
    )
    expected_unit = (5 - 5 + 0.5) / (8 - 5 + 0.9999)
    assert contract["fixed_primary_turns"] == 5
    assert contract["hard_constraint_mutation"] is False
    assert contract["objective_mutation"] is False
    assert len(contract["sha256"]) == 64

    repaired_1d = problem.repair_unit_coordinates([0.2, 0.9, 0.8])
    repaired_2d = problem.repair_unit_coordinates([
        [0.1, 0.0, 0.3], [0.7, 1.0, 0.9],
    ])
    assert repaired_1d.shape == (3,)
    assert repaired_2d.shape == (2, 3)
    assert repaired_1d[1] == pytest.approx(expected_unit, abs=1e-15)
    assert np.equal(repaired_2d[:, 1], expected_unit).all()

    output = {}
    problem._evaluate(repaired_2d, output)
    assert output["F"].shape == (2, 2)
    with pytest.raises(RuntimeError, match="bypassed fixed primary-turn"):
        problem._evaluate(np.asarray([[0.1, 0.9, 0.3]]), {})
    with pytest.raises(RuntimeError, match="conflicting fixed primary-turn"):
        install_fixed_primary_turns_repair(
            problem, 6, coordinate_index=1, minimum_turns=5, maximum_turns=8,
        )


def test_fixed_turn_warm_pool_is_repaired_then_deduped_with_diversity_gate():
    problem = _FakeProblem()
    contract = install_fixed_primary_turns_repair(
        problem, 6, coordinate_index=1, minimum_turns=5, maximum_turns=8,
    )
    warm = np.column_stack([
        np.linspace(0.0, 0.975, 40),
        np.linspace(0.0, 1.0, 40),
        np.linspace(0.975, 0.0, 40),
    ])
    filtered, audit = prepare_fixed_primary_turns_warm_start(
        problem, warm, contract, source_sha_verified=True,
    )
    assert filtered.shape == (40, 3)
    assert audit["source_coordinate_count"] == 40
    assert audit["post_repair_decoded_unique_count"] == 40
    assert audit["minimum_unique_count"] == 32
    assert audit["fixed_primary_turn_coordinate_verified"] is True
    assert audit["source_sha_verified_before_repair"] is True
    assert len(audit["post_repair_coordinates_sha256"]) == 64

    with pytest.raises(RuntimeError, match="source SHA must be verified"):
        prepare_fixed_primary_turns_warm_start(
            problem, warm, contract, source_sha_verified=False,
        )

    collapsed = np.repeat(warm[:1], 64, axis=0)
    with pytest.raises(RuntimeError, match="lost required post-repair diversity"):
        prepare_fixed_primary_turns_warm_start(
            problem, collapsed, contract, source_sha_verified=True,
        )


@pytest.mark.parametrize(
    (
        "module", "variant", "turns", "resonance_scale", "core_scale",
        "seed_start", "focus", "runtime_suffix",
        "priority", "legacy_priority", "task_stem", "dedupe_namespace",
    ),
    [
        (
            "tools.tier1_fixed_n1_5_thermal_llt_slurm_rolling",
            "fixed-n1-5-thermal-llt-v1", 5, 250.0, 1.0,
            1_907_195_000,
            "thermal_then_llt_with_resonance_preserved",
            "mft_tier1_nsga_n1_5_thermal_llt_t110_res15k_260719",
            -4, -5, "mft-t1n5therm",
            "mft-tier1-fixed-n1-5-thermal-llt-nsga",
        ),
        (
            "tools.tier1_fixed_n1_6_resonance_llt_slurm_rolling",
            "fixed-n1-6-resonance-llt-v1", 6, 100.0, 4.0,
            1_907_196_000,
            "resonance_then_llt_with_thermal_preserved",
            "mft_tier1_nsga_n1_6_resonance_llt_t110_res15k_260719",
            -3, -4, "mft-t1n6res",
            "mft-tier1-fixed-n1-6-resonance-llt-nsga",
        ),
        (
            "tools.tier1_fixed_n1_6_thermal_bridge_slurm_rolling",
            "fixed-n1-6-thermal-bridge-v1", 6, 150.0, 2.0,
            1_907_197_000,
            "low_thermal_resonance_pass_llt_bridge",
            "mft_tier1_nsga_n1_6_thermal_bridge_t110_res15k_260719",
            -2, -3, "mft-t1n6bridge",
            "mft-tier1-fixed-n1-6-thermal-bridge-nsga",
        ),
    ],
)
def test_fixed_turn_lane_identity_is_distinct_and_fail_closed(
    module, variant, turns, resonance_scale, core_scale, seed_start, focus,
    runtime_suffix, priority, legacy_priority, task_stem, dedupe_namespace,
):
    code = f'''\
import importlib, json
entry = importlib.import_module({module!r})
r = entry.rolling
hard = {{"size_W_max_mm": 1200.0, "size_L_max_mm": 1200.0}}
print(json.dumps({{
    "variant": r.SEARCH_VARIANT,
    "turns": r.FIXED_PRIMARY_TURNS,
    "resonance_scale": r.OPTIMIZER_RESONANCE_SCALE_HZ,
    "core_scale": r.OPTIMIZER_CORE_THERMAL_SCALE_C,
    "seed_start": r.SEED_START,
    "priority": r.TASK_PRIORITY,
    "legacy_priority": r.LEGACY_TASK_PRIORITY,
    "task_stem": r.TASK_NAME_STEM,
    "dedupe_namespace": r.DEDUPE_NAMESPACE,
    "runtime": str(r.RUNTIME),
    "profile": r.search_profile(),
    "thermal_overrides": r.optimizer_core_thermal_overrides(),
    "acquisition": r.thermal_crossover_acquisition_contract(hard),
}}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    assert value["variant"] == variant
    assert value["turns"] == turns
    assert value["resonance_scale"] == resonance_scale
    assert value["core_scale"] == core_scale
    assert value["seed_start"] == seed_start
    assert value["priority"] == priority
    assert value["legacy_priority"] == legacy_priority
    assert value["task_stem"] == task_stem
    assert value["dedupe_namespace"] == dedupe_namespace
    assert value["runtime"].endswith(runtime_suffix)
    assert profile["fixed_primary_turns"] == turns
    assert profile["search_focus"] == focus
    assert profile["optimizer_Llt_scale_uH"] == 0.55
    assert profile["population"] == 320
    assert profile["max_generations"] == 600
    assert profile["rolling_target"] == 32
    assert profile["task_priority"] == priority
    assert profile["legacy_migration_from_priority"] == legacy_priority
    assert value["thermal_overrides"] == {
        name: core_scale
        for name in profile["optimizer_core_thermal_constraints"]
    }
    assert value["acquisition"][
        "core_thermal_positive_G_scale_C"
    ] == core_scale
    assert value["acquisition"]["hard_constraint_mutation"] is False


def test_fixed_turn_entrypoint_rejects_conflicting_cached_variant():
    environment = os.environ.copy()
    environment["MFT_TIER1_SEARCH_VARIANT"] = (
        "fixed-n1-6-resonance-llt-v1"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tools.tier1_fixed_n1_5_thermal_llt_slurm_rolling",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "conflicts with fixed-N1=5 entrypoint" in process.stderr


def test_orthogonal_fixed_n1_lane_has_distinct_fail_closed_identity():
    code = '''\
import importlib, json
entry = importlib.import_module(
    "tools.tier1_fixed_n1_6_orthogonal_bridge_slurm_rolling"
)
r = entry.rolling
print(json.dumps({
    "variant": r.SEARCH_VARIANT,
    "namespace": r.SEARCH_NAMESPACE,
    "turns": r.FIXED_PRIMARY_TURNS,
    "resonance": r.OPTIMIZER_RESONANCE_SCALE_HZ,
    "core": r.OPTIMIZER_CORE_THERMAL_SCALE_C,
    "llt": r.OPTIMIZER_LLT_SCALE_UH,
    "all_thermal": r.OPTIMIZER_ALL_THERMAL_SCALE_C,
    "seed_start": r.SEED_START,
    "priority": r.TASK_PRIORITY,
    "legacy_priority": r.LEGACY_TASK_PRIORITY,
    "task_stem": r.TASK_NAME_STEM,
    "dedupe": r.DEDUPE_NAMESPACE,
    "runtime": str(r.RUNTIME),
    "profile": r.search_profile(),
    "core_overrides": r.optimizer_core_thermal_overrides(),
    "all_overrides": r.optimizer_all_thermal_overrides(),
    "explicit_overrides": r.optimizer_explicit_overrides(),
    "acquisition": r.thermal_crossover_acquisition_contract({
        "size_W_max_mm": 1200.0, "size_L_max_mm": 1200.0,
    }),
}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    assert value["variant"] == "fixed-n1-6-orthogonal-bridge-v1"
    assert value["namespace"] == (
        "fixed-n1-6-orthogonal-allthermal2c-llt0p25-res150-"
        "p320-g600-warm64-v1"
    )
    assert value["turns"] == 6
    assert value["resonance"] == 150.0
    assert value["core"] is None
    assert value["llt"] == 0.25
    assert value["all_thermal"] == 2.0
    assert value["seed_start"] == 1_907_198_000
    assert value["priority"] == -1
    assert value["legacy_priority"] == -2
    assert value["task_stem"] == "mft-t1n6ortho"
    assert value["dedupe"] == (
        "mft-tier1-fixed-n1-6-orthogonal-bridge-nsga"
    )
    assert value["runtime"].endswith(
        "mft_tier1_nsga_n1_6_orthogonal_bridge_t110_res15k_260719"
    )
    assert profile["search_focus"] == (
        "orthogonal_thermal_llt_resonance_bridge"
    )
    assert profile["optimizer_Llt_scale_uH"] == 0.25
    assert profile["optimizer_all_active_thermal_scale_C"] == 2.0
    assert len(profile["optimizer_all_active_thermal_constraints"]) == 11
    assert profile["soft_axis_pressure_scope"] == (
        "disabled_until_stage_feasible"
    )
    assert "soft_axis_target_W_mm" not in profile
    assert "soft_axis_target_L_mm" not in profile
    assert value["core_overrides"] is None
    assert value["acquisition"] is None
    assert value["all_overrides"] == {
        name: 2.0
        for name in profile["optimizer_all_active_thermal_constraints"]
    }
    assert value["explicit_overrides"] == {
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        **value["all_overrides"],
    }
