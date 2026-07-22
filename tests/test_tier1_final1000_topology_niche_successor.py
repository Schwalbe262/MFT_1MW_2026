from __future__ import annotations

import copy
import json
import types

import numpy as np
import pytest

from tools import tier1_final1000_topology_niche_contract as niche
from tools.tier1_corrected_generation_preflight import (
    Current7Tier1Runner,
    build_topology_niche_initial_population,
    create_deep_topology_components,
    install_optimizer_scaling,
    select_remote_warm_preflight_filter,
    validate_topology_niche_warm_partition,
)
from tools.tier1_deep_crossover_contract import topology_evolution_contract
from tools.tier1_final1000_topology_successor_plan import (
    EXPECTED_EMPTY_POOL_CAPACITY_HISTOGRAM,
    OPTIMIZER_ISLAND_BY_STAGE,
    build_plan,
    decompose_capacity_histogram,
    mixed_parent_partition,
    pack_logical_children,
    validate_current_pool_capacity_snapshot,
    validate_plan,
)
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as stage_profiles


def _coordinate(topology: int) -> float:
    return (60 - int(topology)) / 48.0


def _warm_fixture() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(731)
    groups = {}
    rows = []
    cursor = 0
    for name, expected in niche.WARM_SOURCE_GROUPS.items():
        group_rows = []
        for topology, count in expected["topology_counts"].items():
            values = rng.random((int(count), 25))
            values[:, 2] = _coordinate(int(topology))
            group_rows.extend(values)
        group = np.asarray(group_rows, dtype=float)
        rows.extend(group)
        groups[name] = {
            "start": cursor,
            "count": int(expected["count"]),
            "topology_counts": {
                str(key): int(value)
                for key, value in expected["topology_counts"].items()
            },
            "coordinate_sha256": niche.canonical_sha256(group.tolist()),
            "current7_physics_repair_required": True,
        }
        cursor += len(group)
    values = np.asarray(rows, dtype=float)
    partition = {
        "schema_version": niche.WARM_PARTITION_SCHEMA,
        "fixed_primary_turns": 6,
        "total_count": len(values),
        "groups": groups,
        "warm_topology_counts": {
            str(key): value for key, value in niche.WARM_TOPOLOGY_COUNTS.items()
        },
        "N2_main_60_warm_count": 0,
        "same_current7_physics_repair_required": True,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
    }
    partition["sha256"] = niche.canonical_sha256(partition)
    return values, partition


def test_contract_fills_complete_population_and_terminal_allowances_are_zero():
    contract = niche.validate_contract(niche.contract())

    assert sum(niche.TOPOLOGY_QUOTA_BY_N2_MAIN.values()) == 320
    assert sum(niche.WARM_TOPOLOGY_COUNTS.values()) == 160
    assert sum(niche.FRESH_SOBOL_TOPOLOGY_COUNTS.values()) == 160
    assert niche.WARM_TOPOLOGY_COUNTS[60] == 0
    assert niche.FRESH_SOBOL_TOPOLOGY_COUNTS[60] == 8
    assert niche.optimizer_allowances(37, 0)["resonance_Hz"] == 3000.0
    assert niche.optimizer_allowances(37, 40)["resonance_Hz"] == 2500.0
    assert niche.optimizer_allowances(34, 0)["temperature_C"] == 40.0
    assert niche.optimizer_allowances(34, 0)["Llt_uH"] == 1.75
    for topology in niche.TOPOLOGY_QUOTA_BY_N2_MAIN:
        assert set(niche.optimizer_allowances(topology, 240).values()) == {0.0}
        assert set(niche.optimizer_allowances(topology, 300).values()) == {0.0}
    assert contract["physical_constraint_G_mutation"] is False
    assert contract["physical_objective_mutation"] is False


def test_warm_partition_authenticates_semantic_groups_and_rejects_tamper():
    values, partition = _warm_fixture()

    assert validate_topology_niche_warm_partition(partition, values) == partition
    forged = copy.deepcopy(partition)
    forged["groups"]["llt_thermal_37"]["count"] -= 1
    unsigned = dict(forged)
    unsigned.pop("sha256")
    forged["sha256"] = niche.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="group range"):
        validate_topology_niche_warm_partition(forged, values)


class _RepairProblem:
    n_var = 25

    def __init__(self):
        self.repair_calls = 0
        self.model_evaluation_calls = 0

    def repair_unit_coordinates(self, values):
        self.repair_calls += 1
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def test_initial_population_is_deterministic_exact_160_warm_160_sobol():
    pytest.importorskip("scipy")
    warm, _partition = _warm_fixture()
    contract = topology_evolution_contract(
        6, enable_final1000_topology_niche=True
    )
    problem = _RepairProblem()

    first, first_audit = build_topology_niche_initial_population(
        problem, warm, contract, seed=26072201
    )
    second, second_audit = build_topology_niche_initial_population(
        problem, warm.copy(), contract, seed=26072201
    )

    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["authenticated_warm_count"] == 160
    assert first_audit["fresh_sobol_count"] == 160
    assert first_audit["fresh_N2_main_60_sentinel_count"] == 8
    assert first_audit["warm_N2_main_60_count"] == 0
    assert first_audit["complete_topology_counts"] == {
        str(key): value
        for key, value in niche.TOPOLOGY_QUOTA_BY_N2_MAIN.items()
    }
    assert problem.model_evaluation_calls == 0


def test_topology_warm_preflight_preserves_exact_partition_before_dedupe():
    warm, partition = _warm_fixture()

    class StructuralProblem:
        n_var = 25
        fixed_primary_turns = 6
        fixed_primary_turn_coordinate_index = 0
        fixed_primary_turn_unit_coordinate = 0.5

        def repair_unit_coordinates(self, values):
            repaired = np.asarray(values, dtype=float).copy()
            repaired[:, self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            return repaired

        def filter_structural_donor_coordinates(self, values):
            # Production structural filtering reports every semantic row as
            # accepted, but its ordinary return value is decoded-geometry
            # deduplicated.  Niche quotas must retain the authenticated rows.
            unique = np.asarray(values, dtype=float)[:52]
            return unique, {
                "structurally_accepted_count": 160,
                "structurally_rejected_count": 0,
                "decoded_unique_count": 52,
            }

    runner = Current7Tier1Runner(
        authenticated=None,
        code_identity={},
        modules=None,
        adapter_evidence={},
        model_cache=None,
        models={},
        inference_binding={},
        density_gate=None,
        problem=StructuralProblem(),
    )
    retained, structural, audit = runner.prepare_authenticated_warm_start(
        warm,
        role_partition=None,
        topology_niche_partition=partition,
        stage="topology_niche_smoke",
    )

    assert retained.shape == (160, 25)
    assert structural.shape == (0, 25)
    assert audit["retained_count"] == 160
    assert audit["decoded_unique_count"] == 52
    assert audit["duplicate_geometry_count"] == 108
    assert audit["exact_semantic_partition_rows_preserved_before_dedupe"] is True
    assert audit["topology_counts"] == {
        str(key): value for key, value in niche.WARM_TOPOLOGY_COUNTS.items()
    }


def test_remote_topology_preflight_selects_structural_filter_evidence():
    evidence = {
        "schema_version": "mft-tier1-authenticated-topology-niche-warm-audit-v1",
        "structural_geometry_filter": {
            "structurally_accepted_count": 160,
            "decoded_unique_count": 52,
        },
    }

    assert select_remote_warm_preflight_filter(
        evidence, topology_niche_enabled=True
    ) == evidence["structural_geometry_filter"]
    with pytest.raises(RuntimeError, match="role/schema mismatch"):
        select_remote_warm_preflight_filter(
            evidence, topology_niche_enabled=False
        )


def test_remote_standard_preflight_still_requires_hard_filter_evidence():
    evidence = {
        "schema_version": "mft-tier1-authenticated-warm-role-audit-v1",
        "standard_hard_geometry_filter": {"hard_feasible_count": 1},
    }

    assert select_remote_warm_preflight_filter(
        evidence, topology_niche_enabled=False
    ) == evidence["standard_hard_geometry_filter"]
    forged = dict(evidence)
    forged.pop("standard_hard_geometry_filter")
    with pytest.raises(RuntimeError, match="filter evidence is missing"):
        select_remote_warm_preflight_filter(
            forged, topology_niche_enabled=False
        )


def test_survival_preserves_exact_counts_every_generation_and_never_mutates_G():
    pytest.importorskip("pymoo")
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair

    class ToyProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=25,
                n_obj=2,
                n_ieq_constr=3,
                xl=np.zeros(25),
                xu=np.ones(25),
            )

        def repair_unit_coordinates(self, values):
            return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)

    class ToyRepair(Repair):
        def _do(self, problem, values, **kwargs):
            return problem.repair_unit_coordinates(values)

    rng = np.random.default_rng(922)
    x = rng.random((320, 25))
    cursor = 0
    for topology, count in niche.TOPOLOGY_QUOTA_BY_N2_MAIN.items():
        x[cursor : cursor + count, 2] = _coordinate(topology)
        cursor += count
    f = rng.random((320, 2))
    g = rng.normal(size=(320, 3))
    doubled_x = np.vstack([x, x.copy()])
    doubled_f = np.vstack([f, f + 0.01])
    doubled_g = np.vstack([g, g + 0.01])
    original_g = doubled_g.copy()
    pop = Population.new(X=doubled_x, F=doubled_f, G=doubled_g)
    problem = ToyProblem()
    selection, mating, survival = create_deep_topology_components(
        problem,
        topology_evolution_contract(6, enable_final1000_topology_niche=True),
        ToyRepair(),
    )
    algorithm = types.SimpleNamespace(n_gen=1)

    for generation in range(1, 7):
        algorithm.n_gen = generation
        survivors = survival._do(
            problem,
            pop,
            n_survive=320,
            random_state=np.random.default_rng(1000 + generation),
            algorithm=algorithm,
        )
        observed = 60 - np.rint(48 * survivors.get("X")[:, 2]).astype(int)
        assert {
            topology: int(np.count_nonzero(observed == topology))
            for topology in niche.TOPOLOGY_QUOTA_BY_N2_MAIN
        } == niche.TOPOLOGY_QUOTA_BY_N2_MAIN
    np.testing.assert_array_equal(pop.get("G"), original_g)
    assert len(survival.generation_topology_counts) == 6
    assert all(
        item["exact_quota_verified"] is True
        and item["counts"] == {
            str(key): value
            for key, value in niche.TOPOLOGY_QUOTA_BY_N2_MAIN.items()
        }
        for item in survival.generation_topology_counts
    )

    parent_pop = Population.new(X=x, F=f, G=g)
    pairs = selection._do(
        problem,
        parent_pop,
        n_select=160,
        n_parents=2,
        random_state=np.random.default_rng(77),
    )
    parent_topology = 60 - np.rint(48 * x[:, 2]).astype(int)
    left = parent_topology[pairs[:, 0]]
    right = parent_topology[pairs[:, 1]]
    cross = np.logical_or(
        np.logical_and(left == 36, right == 37),
        np.logical_and(left == 37, right == 36),
    )
    assert int(np.count_nonzero(cross)) == 64
    assert int(np.count_nonzero(left == right)) == 96
    assert selection.cross_36x37_pairs_emitted == 64
    assert selection.generation_parent_pair_counts[-1][
        "cross_36x37_pair_count"
    ] == 64
    record = selection.generation_parent_pair_counts[-1]
    assert record["sha256"] == niche.canonical_sha256({
        key: value for key, value in record.items() if key != "sha256"
    })

    offspring = mating._do(
        problem,
        parent_pop,
        n_offsprings=320,
        random_state=np.random.default_rng(991),
        algorithm=types.SimpleNamespace(n_gen=2),
    )
    offspring_topology = 60 - np.rint(48 * offspring.get("X")[:, 2]).astype(int)
    assert mating.cross_36x37_offspring_attributed > 0
    assert np.count_nonzero(offspring_topology == 36) > 0
    assert np.count_nonzero(offspring_topology == 37) > 0
    assert all(
        item["sha256"] == niche.canonical_sha256({
            key: value for key, value in item.items() if key != "sha256"
        })
        for item in mating.generation_cross_offspring_counts
    )


def test_dynamic_allowance_is_optimizer_only_and_zero_for_terminal_replay():
    class ScalingProblem:
        constraint_names = (
            "Llt_robust_band",
            "Llt_ensemble_disagreement",
            "temperature_robust_limit:Tprobe_core_center_max",
            "half_magnetizing_resonance_minimum",
            "half_magnetizing_resonance_maximum",
            "minimum_physical_insulation",
        )

        def __init__(self):
            self._evaluate = self._physical

        def _physical(self, values, out, *args, **kwargs):
            out["F"] = np.zeros((len(values), 2), dtype=float)
            out["G"] = np.tile(
                np.asarray([2.0, 2.0, 50.0, 3500.0, 3500.0, 3.0]),
                (len(values), 1),
            )

    problem = ScalingProblem()
    physical, contract = install_optimizer_scaling(
        problem,
        resonance_scale_hz=150.0,
        llt_scale_uh=0.05,
        all_thermal_scale_c=2.5,
        resonance_allowance_hz=0.0,
        llt_allowance_uh=0.0,
        topology_allowance_contract=niche.contract(),
    )
    x = np.zeros((1, 25), dtype=float)
    x[:, 2] = _coordinate(37)
    physical_out = {}
    physical(x, physical_out)
    before = np.asarray(physical_out["G"], dtype=float).copy()

    problem._tier1_optimizer_generation = 0
    early = {}
    problem._evaluate(x, early)
    problem._tier1_optimizer_generation = 240
    terminal = {}
    problem._evaluate(x, terminal)

    np.testing.assert_array_equal(physical_out["G"], before)
    np.testing.assert_allclose(
        terminal["G"],
        before / np.asarray(contract["scale_vector"], dtype=float),
    )
    assert early["G"][0, 3] < terminal["G"][0, 3]
    assert contract["physical_hard_constraint_mutation"] is False
    assert contract["physical_objective_mutation"] is False
    assert contract["dynamic_allowances_zero_from_generation"] == 240


def test_plan_is_deterministic_lane_independent_and_44cpu_packs_8_plus_3():
    first = build_plan()
    second = build_plan()
    assert first == second
    assert first["canary"]["logical_seed_count"] == 16
    assert first["canary"]["stage_counts"] == {
        "entry-1200-t125": 6,
        "bridge-1150-t115": 6,
        "close-1075-t107p5": 2,
        "final-1000-t100": 2,
    }
    assert first["production"]["initial_stage_quotas"] == {
        "entry-1200-t125": 300,
        "bridge-1150-t115": 150,
        "close-1075-t107p5": 40,
        "final-1000-t100": 10,
    }
    children = first["production"]["logical_children"][:11]
    identities = [
        (item["seed"], item["payload_sha256"], item["dedupe_key"])
        for item in children
    ]
    packed = pack_logical_children(
        children, available_cpus=44, maximum_lane_count=8
    )
    assert [group["lane_count"] for group in packed["groups"]] == [8, 3]
    assert [
        identity
        for group in packed["groups"]
        for identity in zip(
            group["logical_child_seeds"],
            group["logical_child_payload_sha256"],
            group["logical_child_dedupe_keys"],
        )
    ] == identities
    assert first["bundle_bindings_complete"] is False
    assert first["launch_eligible"] is False
    assert first["scheduler_submission_performed"] is False

    forged = copy.deepcopy(first)
    forged["production"]["logical_children"][0]["seed"] += 1
    unsigned = {key: value for key, value in forged.items() if key != "sha256"}
    forged["sha256"] = niche.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="logical child science seal"):
        validate_plan(forged)


def test_topology_plan_binds_exact500_to_existing_single_seed_controller(
    tmp_path, monkeypatch
):
    science = build_plan()
    science_path = tmp_path / "science.json"
    science_path.write_text(json.dumps(science), encoding="utf-8")
    bindings = {}
    for stage in stage_profiles.STAGES:
        ready = {
            "schema_version": "mft-tier1-current7-slurm-ready-v1",
            "bundle_id": f"topology-{stage.stage_id}",
            "stage_spec_sha256": stage_profiles.stage_profile(stage)[
                "stage_spec_sha256"
            ],
        }
        bindings[stage.stage_id] = {
            "plan": {
                "bundle_id": f"topology-{stage.stage_id}",
                "bundle_manifest_sha256": "a" * 64,
                "remote_bundle": f"/gpfs/topology/{stage.stage_id}",
            },
            "manifest": {},
            "publication": {
                "receipt_sha256": "b" * 64,
                "ready": ready,
                "ready_sha256": launch.canonical_sha256(ready),
            },
            "base_island_id": OPTIMIZER_ISLAND_BY_STAGE[stage.stage_id],
        }

    child_by_seed = {
        int(child["seed"]): child
        for child in science["production"]["logical_children"]
    }

    def fake_task(plan, _manifest, *, seed, priority):
        child = child_by_seed[int(seed)]
        stage = stage_profiles.BY_ID[child["stage_id"]]
        payload = {
            "schema_version": "mft-tier1-current7-slurm-seed-task-v1",
            "bundle_id": plan["bundle_id"],
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "lane": {
                "island_id": child["optimizer_island_id"],
                "variant": child["optimizer_island_id"],
                "seed": int(seed),
                "fixed_primary_turns": 6,
                "wave": "refill",
            },
            "seed": int(seed),
            "population": 320,
            "max_generations": 300,
            "inference_threads": stage_profiles.INFERENCE_THREADS,
            "optimizer_processes": 1,
            "topology_niche_contract_sha256": science[
                "topology_niche_contract_sha256"
            ],
            "maximum_peak_rss_bytes": 42 * 1024**3,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        }
        payload_sha = launch.canonical_sha256(payload)
        return {
            "name": "base",
            "remote_cwd": plan["remote_bundle"],
            "command": (
                "set -euo pipefail\nexec python -u artifacts/code/tools/"
                "tier1_corrected_current7_slurm_seed_runner.py "
                f"--payload-sha256 {payload_sha}"
            ),
            "payload_json": payload,
            "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1",
            "cpus": 8,
            "memory_mb": 65_536,
            "scheduling_profile": "standard",
            "aedt_backend": "standalone",
            "gpus": 0,
            "priority": priority,
            "timeout_seconds": 86_400,
            "dedupe_key": "base:" + "c" * 64,
            "max_workers_per_node": 4,
        }

    monkeypatch.setattr(launch, "load_stage_bindings", lambda _path: bindings)
    monkeypatch.setattr(launch, "build_current7_task_payload", fake_task)
    rendered = launch.build_topology_successor_launch_plan(
        tmp_path / "unused-bindings.json", science_path
    )

    assert len(rendered["task_waves"]["canaries"]) == 4
    assert len(rendered["task_waves"]["ramp"]) == 496
    assert rendered["open_ended_refill"]["stage_active_quotas"] == {
        "entry-1200-t125": 300,
        "bridge-1150-t115": 150,
        "close-1075-t107p5": 40,
        "final-1000-t100": 10,
    }
    tasks = [
        *rendered["task_waves"]["canaries"],
        *rendered["task_waves"]["ramp"],
    ]
    assert all(
        task["payload_json"]["topology_niche_contract_sha256"]
        == science["topology_niche_contract_sha256"]
        for task in tasks
    )
    assert len(
        {
            task["payload_json"]["topology_successor_logical_dedupe_key"]
            for task in tasks
        }
    ) == 500
    assert rendered["topology_successor_science_identity"][
        "pre_cutover_canary_logical_seed_count"
    ] == 16

    # The 16-seed terminal gate uses the same task renderer but remains a
    # separate, non-submitting artifact from the controller's 4+496 shape.
    canary_children = {
        int(child["seed"]): child
        for child in science["canary"]["logical_children"]
    }
    child_by_seed.update(canary_children)
    canary = launch.build_topology_pre_cutover_canary_plan(
        tmp_path / "unused-bindings.json", science_path
    )
    assert canary["logical_seed_count"] == 16
    assert canary["stage_counts"] == {
        "entry-1200-t125": 6,
        "bridge-1150-t115": 6,
        "close-1075-t107p5": 2,
        "final-1000-t100": 2,
    }
    assert canary["scheduler_write_performed"] is False
    assert canary["production_launch_allowed_before_all_16_terminal"] is False


def test_mixed_exact500_partition_preserves_stage_and_science_identity():
    plan = build_plan()
    production = plan["production"]["logical_children"]
    packed = mixed_parent_partition(production)

    assert packed["logical_child_count"] == 500
    assert packed["parent_count"] == 71
    assert packed["parent_count_by_lane"] == {
        "8": 50,
        "7": 7,
        "6": 2,
        "5": 4,
        "4": 1,
        "3": 3,
        "2": 2,
        "1": 2,
    }
    assert sum(
        group["lane_count"] for group in packed["groups"]
    ) == 500
    assert packed["science_identity_mutation"] is False


def test_current_pool_capacity_snapshot_proves_431_vs_fixed8_336():
    decomposed = decompose_capacity_histogram(
        EXPECTED_EMPTY_POOL_CAPACITY_HISTOGRAM
    )
    assert decomposed == {8: 42, 7: 7, 6: 2, 5: 3, 4: 1, 3: 3, 2: 2, 1: 2}
    snapshot = {
        "schema_version": "mft-tier1-empty-pool-capacity-snapshot-v1",
        "observed_at": "2026-07-22T21:00:00+09:00",
        "source_snapshot_sha256": "a" * 64,
        "logical_capacity_histogram": {
            str(key): value
            for key, value in EXPECTED_EMPTY_POOL_CAPACITY_HISTOGRAM.items()
        },
        "logical_capacity": 431,
        "fixed_eight_lane_capacity": 336,
        "decomposed_parent_counts": {
            str(key): value for key, value in decomposed.items()
        },
        "scheduler_write_performed": False,
    }
    snapshot["sha256"] = niche.canonical_sha256(snapshot)
    assert validate_current_pool_capacity_snapshot(snapshot) == snapshot
