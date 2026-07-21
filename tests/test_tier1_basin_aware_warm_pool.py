from __future__ import annotations

import types

import numpy as np

from tools import tier1_deep_crossover_contract as contract
from tools.tier1_corrected_generation_preflight import (
    seed_turn_split_sub_islands,
    select_basin_warm_start_indices,
)


def _coordinate(n2_main: int) -> float:
    return (60 - int(n2_main)) / 48.0


def _topologies(values: np.ndarray) -> np.ndarray:
    return 60 - np.rint(48.0 * np.clip(values[:, 2], 0.0, 1.0)).astype(int)


class _RepairOnlyProblem:
    n_var = 25

    def __init__(self) -> None:
        self.repair_calls = 0
        self.model_evaluation_calls = 0

    def repair_unit_coordinates(self, values):
        self.repair_calls += 1
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def test_rare_basin_warm_topologies_are_never_lost_by_half_pool_sampling():
    topology = contract.topology_evolution_contract(6)
    values = np.random.default_rng(11).random((320, 25))
    values[:, 2] = _coordinate(34)
    for index, n2_main in enumerate(
        topology["turn_split_sub_islands_N2_main"]
    ):
        values[index, 2] = _coordinate(n2_main)

    first, first_audit = select_basin_warm_start_indices(
        values,
        count=160,
        contract=topology,
        random_state=np.random.default_rng(901),
    )
    second, second_audit = select_basin_warm_start_indices(
        values.copy(),
        count=160,
        contract=topology,
        random_state=np.random.default_rng(901),
    )

    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["rare_available_topology_dropped"] is False
    selected_topologies = set(_topologies(values[first]))
    assert selected_topologies >= set(
        topology["turn_split_sub_islands_N2_main"]
    )
    assert first_audit["additional_model_evaluations"] == 0


def test_basin_seeding_reuses_same_lane_genomes_and_is_deterministic():
    topology = contract.topology_evolution_contract(6)
    rng = np.random.default_rng(17)
    initial = rng.random((320, 25))
    initial[:, 2] = _coordinate(34)
    topologies = topology["turn_split_sub_islands_N2_main"]
    for copy_index in range(8):
        for topology_index, n2_main in enumerate(topologies):
            initial[copy_index * len(topologies) + topology_index, 2] = (
                _coordinate(n2_main)
            )
    problem = _RepairOnlyProblem()

    first, first_audit = seed_turn_split_sub_islands(
        problem,
        initial,
        topology,
        warm_donor_count=160,
    )
    second, second_audit = seed_turn_split_sub_islands(
        problem,
        initial.copy(),
        topology,
        warm_donor_count=160,
    )

    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert problem.model_evaluation_calls == 0
    assert first_audit["additional_model_evaluations"] == 0
    assert first_audit["same_or_same_lane_donor_count"] == 56
    assert all(
        count >= 8
        for count in first_audit[
            "topology_counts_after_current_repair"
        ].values()
    )
    exact_size_sources = [
        item
        for item in first_audit["donor_sources"]
        if item["target_N2_main"] == 60
    ]
    assert len(exact_size_sources) == 8
    assert {item["source_N2_main"] for item in exact_size_sources} == {60}
    assert first_audit["basin_lane_topologies"] == {
        "llt_target_mean_q90": [34, 37],
        "thermal_split": [35, 36, 37, 38, 39],
        "exact_size_no_side": [60],
    }


def test_n1_6_operator_keeps_all_basin_topologies_in_terminal_population(
    monkeypatch,
):
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from tools import tier1_resonance_feedback as feedback

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
    monkeypatch.setitem(__import__("sys").modules, "optimization", optimization)
    monkeypatch.setitem(
        __import__("sys").modules, "optimization.run_nsga2", pinned
    )

    result = feedback._run_optimizer(
        ToyProblem(),
        seed=23,
        population=64,
        warm_start=None,
        max_generations=2,
        termination_strategy=contract.TERMINATION_STRATEGY,
        topology_evolution_contract=contract.topology_evolution_contract(6),
    )

    counts = result.topology_operator_audit["last_topology_counts"]
    assert set(map(int, counts)) == {34, 35, 36, 37, 38, 39, 60}
    assert min(counts.values()) >= 8
    assert max(counts.values()) <= 16
    assert result.topology_operator_audit["migrants_created"] == 7
    seeded = result.initialization_audit["turn_split_sub_islands"][
        "basin_seeded_counts"
    ]
    assert seeded["llt_target_mean_q90"] >= 16
    assert seeded["thermal_split"] >= 40
    assert seeded["exact_size_no_side"] >= 8
