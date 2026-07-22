from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from tools import tier1_deep_crossover_contract as contract
from tools import tier1_corrected_generation_preflight as preflight
from tools import tier1_fixed_n1_5_deep_crossover_warm_handoff as warm
from tools import tier1_resonance_feedback as feedback
from tools import tier1_slurm_seed_runner as runner


REPO = Path(__file__).resolve().parents[1]


def test_deep_inventory_is_globally_disjoint_and_exact_36_slots():
    contract.validate_inventory()
    islands = contract.DEEP_CROSSOVER_ISLANDS
    assert sum(item.active_quota for item in islands) == 36
    assert [item.active_quota for item in islands] == [24, 4, 4, 4]
    assert len({item.variant for item in islands}) == len(islands)
    assert len({item.namespace for item in islands}) == len(islands)
    assert len({item.task_name_stem for item in islands}) == len(islands)
    assert len({item.dedupe_namespace for item in islands}) == len(islands)
    windows = sorted(
        (item.seed_start, item.seed_window_end_exclusive) for item in islands
    )
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))


def test_fixed_generation_contract_is_exact_counter_plus_one():
    observed = feedback.optimizer_termination_contract(
        contract.TERMINATION_STRATEGY, contract.FIXED_GENERATIONS,
    )
    assert observed["rule"] == {
        "kind": "pymoo_minimize_tuple_n_gen",
        "n_max_gen": 300,
        "early_stop_allowed": False,
        "completed_evolution_generations": 300,
        "expected_algorithm_n_gen_counter": 301,
    }
    assert observed == runner.expected_optimizer_termination_contract(
        contract.TERMINATION_STRATEGY, contract.FIXED_GENERATIONS,
    )


def test_fixed_optimizer_reuses_pinned_initialization_and_repair(monkeypatch):
    calls = {}
    pinned = types.ModuleType("optimization.run_nsga2")

    def initial(problem, *, seed, pop, warm_X):
        calls["initial"] = (problem, seed, pop, warm_X)
        return np.zeros((pop, 2)), {"sealed": True}

    repair = object()
    pinned._initial_population = initial
    pinned._physics_repair_operator = lambda: repair
    optimization = types.ModuleType("optimization")
    optimization.run_nsga2 = pinned
    monkeypatch.setitem(sys.modules, "optimization", optimization)
    monkeypatch.setitem(sys.modules, "optimization.run_nsga2", pinned)

    nsga2_module = types.ModuleType("pymoo.algorithms.moo.nsga2")

    class NSGA2:
        def __init__(self, **kwargs):
            calls["algorithm"] = kwargs

    nsga2_module.NSGA2 = NSGA2
    optimize_module = types.ModuleType("pymoo.optimize")

    def minimize(problem, algorithm, termination, **kwargs):
        calls["minimize"] = (problem, algorithm, termination, kwargs)
        return types.SimpleNamespace()

    optimize_module.minimize = minimize
    monkeypatch.setitem(sys.modules, "pymoo.algorithms.moo.nsga2", nsga2_module)
    monkeypatch.setitem(sys.modules, "pymoo.optimize", optimize_module)

    problem = object()
    warm_values = np.ones((3, 2))
    result = feedback._run_optimizer(
        problem, seed=17, population=8, warm_start=warm_values,
        max_generations=300,
        termination_strategy=contract.TERMINATION_STRATEGY,
    )
    assert calls["initial"] == (problem, 17, 8, warm_values)
    assert calls["algorithm"]["sampling"].shape == (8, 2)
    assert calls["algorithm"]["repair"] is repair
    assert calls["algorithm"]["eliminate_duplicates"] is True
    assert calls["minimize"][2] == ("n_gen", 300)
    assert result.initialization_audit == {"sealed": True}


def test_turn_split_operators_execute_pairing_migration_and_epsilon_survival(
    monkeypatch,
):
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair

    class ToyProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=25, n_obj=2, n_ieq_constr=2,
                xl=np.zeros(25), xu=np.ones(25),
            )

        def repair_unit_coordinates(self, values):
            return np.clip(values, 0.0, 1.0)

        def _evaluate(self, values, out, *args, **kwargs):
            out["F"] = np.c_[values[:, 3], values[:, 4]]
            out["G"] = np.c_[values[:, 5] - 0.5, values[:, 6] - 0.5]

    class ToyRepair(Repair):
        def _do(self, problem, values, **kwargs):
            return problem.repair_unit_coordinates(values)

    pinned = types.ModuleType("optimization.run_nsga2")
    pinned._initial_population = lambda problem, seed, pop, warm_X: (
        np.random.default_rng(seed).random((pop, problem.n_var)),
        {"warm_filter": {"decoded_unique_count": 64}},
    )
    pinned._physics_repair_operator = lambda: ToyRepair()
    optimization = types.ModuleType("optimization")
    optimization.run_nsga2 = pinned
    monkeypatch.setitem(sys.modules, "optimization", optimization)
    monkeypatch.setitem(sys.modules, "optimization.run_nsga2", pinned)

    result = feedback._run_optimizer(
        ToyProblem(), seed=17, population=64, warm_start=None,
        max_generations=2,
        termination_strategy=contract.TERMINATION_STRATEGY,
        topology_evolution_contract=contract.topology_evolution_contract(5),
    )
    audit = result.topology_operator_audit
    assert result.algorithm.n_gen == 3
    assert audit["paired_parent_pairs_emitted"] > 0
    assert audit["migration_events"] == 1
    assert audit["migrants_created"] == 6
    assert audit["survival_calls"] == 2
    assert min(audit["last_topology_counts"].values()) >= 4
    assert result.initialization_audit["turn_split_sub_islands"][
        "source_prediction_or_pass_classification_inherited"
    ] is False


@pytest.mark.parametrize(
    "factory",
    [
        preflight.create_deep_topology_components,
        feedback._deep_topology_components,
    ],
    ids=["preflight", "resonance-runtime"],
)
def test_epsilon_admission_remains_aggregate_before_minimax_order(factory):
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair

    class ToyProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=25,
                n_obj=2,
                n_ieq_constr=2,
                xl=np.zeros(25),
                xu=np.ones(25),
            )

        def repair_unit_coordinates(self, values):
            return np.asarray(values, dtype=float)

    class ToyRepair(Repair):
        def _do(self, problem, values, **kwargs):
            return np.asarray(values, dtype=float)

    problem = ToyProblem()
    topology = contract.topology_evolution_contract(6)
    topologies = topology["turn_split_sub_islands_N2_main"]
    copies = topology["initial_repaired_copies_per_sub_island"]
    count = len(topologies) * copies
    x = np.random.default_rng(1901).random((count, problem.n_var))
    for topology_index, n2_main in enumerate(topologies):
        start = topology_index * copies
        x[start : start + copies, 2] = (60 - n2_main) / 48.0
    f = np.c_[np.arange(count, dtype=float), np.arange(count, dtype=float)]
    g = np.full((count, 2), -1.0, dtype=float)
    g[0] = [15.0, 15.0]  # aggregate 30 > epsilon 20
    g[1] = [19.0, -1.0]  # aggregate 19 <= epsilon 20
    population = Population.new("X", x, "F", f, "G", g)
    _selection, _mating, survival = factory(
        problem, topology, ToyRepair()
    )

    survival._do(
        problem,
        population,
        n_survive=count,
        random_state=np.random.default_rng(1902),
        algorithm=types.SimpleNamespace(n_gen=1),
    )

    assert survival.last_epsilon == 20.0
    assert population[0].get("rank") == count - 1
    assert population[0].get("crowding") == -15.0
    assert population[1].get("rank") < population[0].get("rank")


def test_warm_subset_selection_is_deterministic_and_unique():
    values = np.arange(64 * 25, dtype=float).reshape(64, 25)
    values /= values.max()
    first = warm._diverse_indices(values)
    second = warm._diverse_indices(values.copy())
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert all(0 <= index < 64 for index in first)


def test_anchor_replay_returns_dict_with_source_identity_and_current_gate():
    source_decoded = {"N1": 5, "N2_main": 28, "space_mm": 40.0}
    current_decoded = {"N1": 5, "N2_main": 28, "space_mm": 39.6}
    source_g = {"thermal": 2.0, "resonance": -2.0e-9}
    replayed_g = {"thermal": 2.2, "resonance": -1.99e-9}
    source_candidate = {
        "decoded_params": dict(source_decoded),
        "constraint_G": dict(source_g),
    }
    evidence = contract.validate_anchor_current_replay(
        current_decoded=current_decoded, replayed_physical_g=replayed_g,
        source_candidate=source_candidate,
        expected_decoded_sha256=contract.canonical_sha(source_decoded),
        expected_source_physical_g_sha256=contract.canonical_sha(source_g),
    )
    assert isinstance(evidence, dict)
    assert evidence["source_decoded_params"] == source_candidate["decoded_params"]
    assert evidence["current_replayed_decoded_params"] == current_decoded
    assert evidence["physical_constraint_pass_classification_identical"] is True
    assert evidence[
        "source_vs_current_physical_G_numerical_equality_claimed"
    ] is False
    assert evidence["optimizer_deterministic_replay_claimed"] is False
    assert evidence["original_terminal_chromosome_authority"] is False


def test_anchor_replay_rejects_source_vs_current_classification_change():
    decoded = {"N1": 5}
    source_g = {"thermal": 0.0}
    with pytest.raises(RuntimeError, match="differs from source class"):
        contract.validate_anchor_current_replay(
            current_decoded=decoded,
            replayed_physical_g={"thermal": 2.0e-9},
            source_candidate={
                "decoded_params": dict(decoded), "constraint_G": source_g,
            },
            expected_decoded_sha256=contract.canonical_sha(decoded),
            expected_source_physical_g_sha256=contract.canonical_sha(source_g),
        )


@pytest.mark.parametrize("fixed_primary_turns", [5, 6])
def test_turn_split_contract_has_real_pairing_migration_and_epsilon_survival(
    fixed_primary_turns,
):
    value = contract.topology_evolution_contract(fixed_primary_turns)
    if fixed_primary_turns == 5:
        topologies = [28, 29, 30, 31, 32, 33]
        pairs = {
            (28, 31), (29, 32), (30, 33),
            (28, 33), (29, 31), (30, 32),
        }
        minimum_each = 4
    else:
        topologies = [34, 35, 36, 37, 38, 39, 60]
        pairs = set(contract.BASIN_TURN_SPLIT_PARENT_PAIRS_N1_6)
        minimum_each = 8
        assert value["basin_donor_lanes"] == (
            contract.BASIN_DONOR_LANES_N1_6
        )
        assert {tuple(pair) for pair in value["cross_lane_parent_pairs"]} >= {
            (34, 38), (37, 60), (39, 60)
        }
    assert value["requested_global_turn_split_N2_main"] == topologies
    assert value["turn_split_sub_islands_N2_main"] == topologies
    assert value[
        "unavailable_requested_topologies_due_pinned_physics_repair"
    ] == []
    assert {tuple(pair) for pair in value["turn_split_parent_pair_schedule"]} == pairs
    assert value["paired_mating"].startswith("cross_distinct")
    assert value["migration"]["period_generations"] == 5
    assert value["migration"]["migrants_per_event"] == len(topologies)
    assert value["survival"] == {
        "kind": (
            "normalized_positive_G_sum_epsilon_then_"
            "positive_count_max_sum_then_rank_crowding"
        ),
        "initial_epsilon": 20.0,
        "decay_to_zero_generation": 160,
        "terminal_epsilon": 0.0,
        "epsilon_feasibility": (
            "sum_normalized_positive_G_less_than_or_equal_to_epsilon"
        ),
        "infeasible_order": (
            "positive_constraint_count_then_max_normalized_positive_G_"
            "then_sum_normalized_positive_G"
        ),
        "physical_G_mutation": False,
        "objective_mutation": False,
        "minimum_survivors_per_turn_split_sub_island": minimum_each,
    }
    budget = value["bounded_diversity_budget"]
    assert budget["additional_model_evaluations"] == 0
    assert budget["population_change"] == budget["generation_change"] == 0
    if fixed_primary_turns == 6:
        assert budget["protected_slots"] == 56
        assert budget["protected_population_fraction"] == 0.175
        assert budget["maximum_single_protected_topology_count"] == 272
        assert budget["maximum_single_protected_topology_fraction"] == 0.85
    assert value["minimum_evolution_generations"] == 300
    assert value["ftol_early_stop_allowed"] is False
    assert value["warm_donor_prediction_inheritance_allowed"] is False
    assert value["model_or_training_data_provenance_mutation"] is False
    assert value["capacitance_label_precision_remediation_included"] is False
    assert contract.canonical_sha({
        key: item for key, item in value.items() if key != "sha256"
    }) == value["sha256"]


@pytest.mark.parametrize("island", contract.DEEP_CROSSOVER_ISLANDS)
def test_rolling_profile_matches_each_sealed_deep_variant(island):
    script = """
import json
from tools import tier1_slurm_rolling as r
print(json.dumps({
  'variant': r.SEARCH_VARIANT,
  'profile': r.search_profile(),
  'target': r.ROLLING_TARGET,
  'generations': r.MAX_GENERATIONS,
  'termination': r.optimizer_termination_contract(),
}))
"""
    environment = os.environ.copy()
    environment["MFT_TIER1_SEARCH_VARIANT"] = island.variant
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=environment,
        text=True, capture_output=True, check=True,
    )
    value = json.loads(completed.stdout)
    profile = value["profile"]
    assert value["variant"] == island.variant
    assert value["target"] == island.active_quota
    assert value["generations"] == contract.FIXED_GENERATIONS
    assert profile["deep_crossover_island"] == contract.island_profile(island)
    assert profile["optimizer_termination_strategy"] == (
        contract.TERMINATION_STRATEGY
    )
    assert profile["seed_window_end_exclusive"] == (
        island.seed_window_end_exclusive
    )
    assert value["termination"]["rule"][
        "expected_algorithm_n_gen_counter"
    ] == 301


def test_unsupported_termination_fails_closed():
    with pytest.raises(ValueError, match="unsupported"):
        feedback.optimizer_termination_contract(
            "ftol-maybe", contract.FIXED_GENERATIONS,
        )
    with pytest.raises(RuntimeError, match="unsupported"):
        runner.expected_optimizer_termination_contract(
            "ftol-maybe", contract.FIXED_GENERATIONS,
        )
