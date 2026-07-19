from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import tier1_n1_6_anchor_island_contract as contract
from tools import tier1_n1_6_anchor_island_preflight as preflight
from tools import tier1_resonance_focus_warm_handoff as sealed


REPO = Path(__file__).resolve().parents[1]


CONSTRAINT_NAMES = (
    contract.RESONANCE_CONSTRAINT,
    contract.LLT_CONSTRAINT,
    "temperature_robust_limit:T_max_core",
)


class FakeProblem:
    constraint_names = CONSTRAINT_NAMES

    def __init__(self):
        def normalized_evaluate(X, out, *args, **kwargs):
            del args, kwargs
            rows = len(np.atleast_2d(X))
            physical = np.tile([150.0, 0.05, 4.0], (rows, 1))
            out["G"] = physical / np.asarray([150.0, 0.05, 2.0])
            out["physical_marker"] = physical

        self._evaluate = normalized_evaluate


def _normalization() -> dict:
    return {
        "constraint_order": list(CONSTRAINT_NAMES),
        "authoritative_terminal_G": "physical_unscaled",
        "scales": {
            contract.RESONANCE_CONSTRAINT: 150.0,
            contract.LLT_CONSTRAINT: 0.05,
            "temperature_robust_limit:T_max_core": 2.0,
        },
    }


def test_six_islands_have_sealed_parallel_quota_and_staircase_identity():
    assert len(contract.ANCHOR_ISLANDS) == 6
    assert contract.TOTAL_ACTIVE_QUOTA == 36
    assert [item.active_quota for item in contract.ANCHOR_ISLANDS] == [
        16, 4, 4, 4, 4, 4,
    ]
    assert [
        item.resonance_allowance_hz
        for item in contract.ANCHOR_ISLANDS[1:]
    ] == [2_000.0, 1_500.0, 1_000.0, 500.0, 0.0]
    assert len(contract.BY_VARIANT) == len(contract.ANCHOR_ISLANDS)
    anchor_a = contract.ANCHOR_ISLANDS[0]
    assert anchor_a.seed_start == 1_907_205_000
    assert "seed205" in anchor_a.runtime_suffix
    assert "seed205" in anchor_a.namespace
    assert "seed205" in anchor_a.dedupe_namespace
    seed_ranges = [
        set(range(item.seed_start, item.seed_start + item.active_quota))
        for item in contract.ANCHOR_ISLANDS
    ]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(seed_ranges)
        for right in seed_ranges[index + 1:]
    )
    for island in contract.ANCHOR_ISLANDS:
        profile = contract.island_profile(island)
        assert profile["fixed_primary_turns"] == 6
        assert profile["soft_axis_pressure_enabled"] is False
        assert profile["physical_hard_spec_mutation"] is False
        unsigned = {key: value for key, value in profile.items() if key != "sha256"}
        assert profile["sha256"] == contract.canonical_sha(unsigned)


def test_optimizer_allowance_changes_only_optimizer_g_and_preserves_metadata():
    problem = FakeProblem()
    physical_evaluate = problem._evaluate
    normalization = _normalization()
    sealed = contract.install_optimizer_allowance(
        problem,
        normalization,
        resonance_allowance_hz=150.0,
        llt_allowance_uh=0.05,
    )
    optimizer = {}
    problem._evaluate(np.zeros((2, 3)), optimizer)
    physical = {}
    physical_evaluate(np.zeros((2, 3)), physical)

    assert np.array_equal(physical["G"], np.tile([1.0, 1.0, 2.0], (2, 1)))
    assert np.array_equal(optimizer["G"], np.tile([0.0, 0.0, 2.0], (2, 1)))
    assert np.array_equal(optimizer["physical_marker"], physical["physical_marker"])
    assert sealed["hard_constraint_mutation"] is False
    assert sealed["result_replay_uses_physical_evaluator"] is True
    assert sealed["authoritative_terminal_G"] == "physical_unscaled_unchanged"
    unsigned = {key: value for key, value in sealed.items() if key != "sha256"}
    assert sealed["sha256"] == contract.canonical_sha(unsigned)


@pytest.mark.parametrize(
    ("resonance", "llt"),
    [
        (None, 0.0),
        (-1.0, 0.0),
        (2_000.0001, 0.0),
        (0.0, -0.001),
        (0.0, 0.0501),
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (True, 0.0),
    ],
)
def test_allowance_contract_rejects_unsealed_values(resonance, llt):
    with pytest.raises(RuntimeError):
        contract.optimizer_allowance_contract(
            CONSTRAINT_NAMES,
            resonance_allowance_hz=resonance,
            llt_allowance_uh=llt,
        )


def test_allowance_install_fails_closed_on_order_or_scale_drift():
    for mutation in ("order", "scale"):
        normalization = copy.deepcopy(_normalization())
        if mutation == "order":
            normalization["constraint_order"].reverse()
        else:
            normalization["scales"].pop(contract.LLT_CONSTRAINT)
        with pytest.raises(RuntimeError):
            contract.install_optimizer_allowance(
                FakeProblem(),
                normalization,
                resonance_allowance_hz=0.0,
                llt_allowance_uh=0.0,
            )


@pytest.mark.parametrize("island", contract.ANCHOR_ISLANDS)
def test_anchor_entrypoint_seals_each_independent_quota(island):
    code = """\
import json
from tools import tier1_fixed_n1_6_anchor_island_slurm_rolling as entry
r = entry.rolling
print(json.dumps({
    "variant": r.SEARCH_VARIANT,
    "runtime": str(r.RUNTIME),
    "target": r.ROLLING_TARGET,
    "res_allow": r.OPTIMIZER_RESONANCE_ALLOWANCE_HZ,
    "llt_allow": r.OPTIMIZER_LLT_ALLOWANCE_UH,
    "profile": r.search_profile(),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    environment["MFT_TIER1_ANCHOR_ISLAND_VARIANT"] = island.variant
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    assert value["variant"] == island.variant
    assert value["runtime"].endswith(island.runtime_suffix)
    assert value["target"] == island.active_quota
    assert value["res_allow"] == island.resonance_allowance_hz
    assert value["llt_allow"] == island.llt_allowance_uh
    assert value["profile"]["rolling_target"] == island.active_quota
    assert value["profile"]["decoded_params_sha256_duplicate_cap"] == 1
    assert value["profile"]["soft_axis_pressure_scope"] == (
        "disabled_until_stage_feasible"
    )
    assert value["profile"]["physical_hard_spec_mutation"] is False


def test_anchor_entrypoint_has_no_default_and_rejects_variant_conflict():
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    environment.pop("MFT_TIER1_ANCHOR_ISLAND_VARIANT", None)
    process = subprocess.run(
        [
            sys.executable, "-c",
            "import tools.tier1_fixed_n1_6_anchor_island_slurm_rolling",
        ],
        cwd=REPO, env=environment, capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "must name one sealed anchor island" in process.stderr

    environment["MFT_TIER1_ANCHOR_ISLAND_VARIANT"] = (
        contract.ANCHOR_ISLANDS[0].variant
    )
    environment["MFT_TIER1_SEARCH_VARIANT"] = (
        contract.ANCHOR_ISLANDS[1].variant
    )
    process = subprocess.run(
        [
            sys.executable, "-c",
            "import tools.tier1_fixed_n1_6_anchor_island_slurm_rolling",
        ],
        cwd=REPO, env=environment, capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "variants disagree" in process.stderr


def _preflight_result(island) -> dict:
    names = list(sealed.EXPECTED_CONSTRAINT_NAMES)
    thermal = [
        name for name in names
        if name.startswith("temperature_robust_limit:")
    ]
    scales = {name: 1.0 for name in names}
    scales.update({
        contract.RESONANCE_CONSTRAINT: contract.RESONANCE_SCALE_HZ,
        contract.LLT_CONSTRAINT: contract.LLT_SCALE_UH,
        "Llt_ensemble_disagreement": 2.0 * contract.LLT_SCALE_UH,
        **{name: contract.ALL_THERMAL_SCALE_C for name in thermal},
    })
    physical = {name: -1.0 for name in names}
    return {
        "schema_version": preflight.RESULT_SCHEMA,
        "seed": island.seed_start - 1,
        "population": 64,
        "max_generations": 1,
        "completed_generations": 2,
        "constraint_version": preflight.CONSTRAINT_VERSION,
        "hard_spec": preflight.HARD_SPEC,
        "hard_spec_sha256": contract.canonical_sha(preflight.HARD_SPEC),
        "nsga_code_revision": preflight.EXPECTED_NSGA_REVISION,
        "fixed_primary_turns": 6,
        "terminal_population_fixed_primary_turns_verified": True,
        "terminal_population_primary_turn_values": [6],
        "constraint_names": names,
        "optimizer_constraint_normalization": {
            "constraint_order": names,
            "authoritative_terminal_G": "physical_unscaled",
            "scales": scales,
        },
        "optimizer_resonance_allowance_Hz": (
            island.resonance_allowance_hz
        ),
        "optimizer_Llt_allowance_uH": island.llt_allowance_uh,
        "optimizer_constraint_allowance_contract": (
            contract.optimizer_allowance_contract(
                names,
                resonance_allowance_hz=island.resonance_allowance_hz,
                llt_allowance_uh=island.llt_allowance_uh,
            )
        ),
        "acquisition_ranking_contract": None,
        "constraint_minimum_G": physical,
        "terminal_population_best_constraint_G": physical,
        "optimizer_terminal_best_physical_constraint_G": physical,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }


def test_preflight_summary_replays_all_six_physical_result_seals():
    for island in contract.ANCHOR_ISLANDS:
        result = _preflight_result(island)
        summary = preflight._result_summary(result, island)
        assert summary["variant"] == island.variant
        assert summary["authoritative_terminal_G"] == (
            "physical_unscaled_unchanged"
        )
        assert summary["scheduler_write_performed"] is False
        mutated = copy.deepcopy(result)
        mutated["optimizer_resonance_allowance_Hz"] += 1.0
        with pytest.raises(RuntimeError, match="result seal mismatch"):
            preflight._result_summary(mutated, island)
