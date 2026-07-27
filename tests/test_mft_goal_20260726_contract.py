from __future__ import annotations

import copy
import json
from pathlib import Path
import types
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from module import mft_goal_20260726_contract as goal
from regression_260707.verify import finalize
from tools import tier1_corrected_generation_adapter as adapter
from tools import tier1_corrected_generation_preflight as preflight
from tools import tier1_final1000_multiseed_consumer as consumer
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_diagnostic_compact_scout as diagnostic_scout


REPO = Path(__file__).resolve().parents[1]


def test_local_preflight_relocation_requires_complete_triplet(
    monkeypatch,
):
    called = False

    def unexpected_runner(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authentication must not start")

    monkeypatch.setattr(
        preflight,
        "build_authenticated_runner",
        unexpected_runner,
    )
    with pytest.raises(
        RuntimeError,
        match="requires runtime dataset, runtime profile",
    ):
        launch.run_local_preflight(
            generation=Path("generation"),
            candidate=Path("candidate.json"),
            quality_status=Path("quality.json"),
            code_root=Path("code"),
            expected_code_revision="a" * 40,
            dataset_path_override=Path("dataset.parquet"),
        )
    assert called is False


def test_local_preflight_forwards_exact_remote_documentary_identity(
    monkeypatch,
):
    observed = {}

    class AuthenticationReached(RuntimeError):
        pass

    def capture_runner(**kwargs):
        observed.update(kwargs)
        raise AuthenticationReached

    monkeypatch.setattr(
        preflight,
        "build_authenticated_runner",
        capture_runner,
    )
    dataset = Path("harvested/strict_al.parquet")
    profile = Path("harvested/goal_standard.json")
    documentary = (
        "/gpfs/tmp_cpu2/mft_goal_20260726/al/"
        "runs/task-123/registry/generations/example"
    )
    with pytest.raises(AuthenticationReached):
        launch.run_local_preflight(
            generation=Path("harvested/registry/generations/example"),
            candidate=Path("harvested/candidate.json"),
            quality_status=Path("harvested/quality_status.json"),
            code_root=Path("code"),
            expected_code_revision="b" * 40,
            dataset_path_override=dataset,
            profile_path_override=profile,
            expected_documentary_generation_path=documentary,
        )
    assert observed["dataset_path_override"] is dataset
    assert observed["profile_path_override"] is profile
    assert (
        observed["expected_documentary_generation_path"]
        == documentary
    )


def test_prepare_bundle_seals_all_three_relocation_cli_inputs(
    monkeypatch,
    tmp_path,
):
    revision = "c" * 40
    code_root = tmp_path / "code"
    code_root.mkdir()
    candidate = tmp_path / "candidate.json"
    quality = tmp_path / "quality.json"
    runtime_dataset = tmp_path / "strict_al.parquet"
    runtime_profile = tmp_path / "goal_standard.json"
    for path in (
        candidate,
        quality,
        runtime_dataset,
        runtime_profile,
    ):
        path.write_text("{}", encoding="utf-8")
    documentary = (
        "/gpfs/tmp_cpu2/mft_goal_20260726/al_training/"
        "bundle/runs/task-123/registry/generations/g1"
    )
    relocation = {
        "enabled": True,
        "source_absolute_paths_are_documentary_only": True,
        "documentary_generation_path": documentary,
        "runtime_generation_path": str(tmp_path / "generation"),
        "documentary_dataset_path": (
            "/gpfs/example/artifacts/input/strict_al.parquet"
        ),
        "runtime_dataset_path": str(runtime_dataset),
        "documentary_profile_path": (
            "/gpfs/example/artifacts/code/goal_standard.json"
        ),
        "runtime_profile_path": str(runtime_profile),
        "content_identities_reauthenticated": True,
    }
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "profile_sha256": "2" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "source_relocation": relocation,
            "code": {"revision": revision},
            "search_only_proposal": False,
        }
    )
    authenticated = types.SimpleNamespace(
        candidate={"generation_path": documentary},
        dataset_path=runtime_dataset,
        profile_path=runtime_profile,
    )
    observed = {}

    def fake_preflight(**kwargs):
        observed.update(kwargs)
        return local_preflight, types.SimpleNamespace(
            authenticated=authenticated
        )

    monkeypatch.setattr(launch, "run_local_preflight", fake_preflight)
    monkeypatch.setattr(
        launch,
        "_collect_goal_code_sources",
        lambda _root: {},
    )
    monkeypatch.setattr(
        launch,
        "_build_goal_code_manifest",
        lambda _sources, *, code_revision: (
            _synthetic_goal_code_manifest(code_revision)
        ),
    )

    def fake_stage(output, *, sources, manifest):
        assert sources == {}
        output.mkdir(parents=True)
        launch._atomic_json(output / "code_manifest.json", manifest)
        return output / "code_manifest.json"

    monkeypatch.setattr(launch, "_stage_goal_code", fake_stage)
    output = tmp_path / "campaign"
    args = launch._parser().parse_args(
        [
            "prepare",
            "--generation",
            str(tmp_path / "generation"),
            "--candidate",
            str(candidate),
            "--quality-status",
            str(quality),
            "--code-root",
            str(code_root),
            "--expected-code-revision",
            revision,
            "--runtime-dataset",
            str(runtime_dataset),
            "--runtime-profile",
            str(runtime_profile),
            "--expected-documentary-generation-path",
            documentary,
            "--mode",
            "single",
            "--seed-start",
            "2607263000",
            "--output",
            str(output),
        ]
    )
    launch.prepare_bundle(args)
    assert observed["dataset_path_override"] == runtime_dataset
    assert observed["profile_path_override"] == runtime_profile
    assert observed["expected_documentary_generation_path"] == documentary
    sealed = json.loads(
        (output / "local_preflight.json").read_text(encoding="utf-8")
    )
    launch._validate_seal(
        sealed,
        schema=launch.LOCAL_PREFLIGHT_SCHEMA,
    )
    assert sealed["source_relocation"] == relocation
    task = json.loads(
        (
            output
            / "tasks"
            / f"seed-{launch.FRESH_AL_SEED_START}-n1-5.json"
        ).read_text(encoding="utf-8")
    )
    assert task["fresh512_compact_active"] is False
    assert task["fresh512_search_activation"] is None
    assert task["compact_run_authorization"] is None
    assert task["hard_constraint_contract_sha256"] == "d" * 64


def _synthetic_goal_code_manifest(revision: str = "c" * 40):
    records = {
        "artifacts/code/.source-revision": {
            "sha256": "9" * 64,
            "size": 41,
        }
    }
    return launch._seal(
        {
            "schema_version": launch.CODE_MANIFEST_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "code_revision": revision,
            "code_root_relative": "artifacts/code",
            "revision_marker": "artifacts/code/.source-revision",
            "files": records,
            "code_inventory": records,
            "code_inventory_sha256": goal.canonical_sha256(records),
            "staged_path_rule": "bundle_root/<code_inventory_key>",
            "source_checkout_mutated": False,
            "remote_git_checkout_required": False,
            "scheduler_project_code_included": False,
        }
    )


class _Predictor:
    def __init__(self, value: float, half_width: float = 0.25):
        self.value = float(value)
        self.half_width = float(half_width)
        self.features = ["l1"]

    def predict_mu_sigma(self, frame, conformal=True):
        assert conformal is True
        return (
            np.full(len(frame), self.value),
            np.full(len(frame), self.half_width),
        )

    def disagreement(self, frame):
        return np.zeros(len(frame))


def _models():
    values = {
        target: 1.0 for target in adapter.GOAL_REQUIRED_MODEL_TARGETS
    }
    values.update(
        {
            "Llt_phys": 27.5,
            "k": 0.9,
            "C_tx_tx_F": 1.0e-12,
            "C_rx_rx_F": 1.0e-14,
            "C_tx_rx_F": 5.0e-10,
            **{
                target: 100.0
                for target in goal.WINDING_TEMPERATURE_TARGETS
            },
            **{
                target: 119.0
                for target in goal.CORE_TEMPERATURE_TARGETS
            },
        }
    )
    return {target: _Predictor(value) for target, value in values.items()}


def _goal_problem(primary_turns=5, *, geometry_constraint_profile=None):
    modules = preflight.load_current7_modules(REPO)
    problem_class = preflight.create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(
            modules.nsga2_problem.design_analytical_b_field_t
        ),
    )
    return problem_class(
        _models(),
        spec=goal.GOAL_STAGE_SPEC,
        density_gate=lambda frame: np.full(len(frame), -1.0),
        fixed_primary_turns=primary_turns,
        geometry_constraint_profile=geometry_constraint_profile,
    )


def _coordinate(problem, *, cw1_unit=0.5, side_unit=0.4):
    result = np.full((1, problem.n_var), 0.5)
    result[0, problem.cw1_coordinate_index] = cw1_unit
    result[0, 2] = side_unit
    return problem.repair_unit_coordinates(result)


def test_goal_contract_is_exact_and_forbids_legacy_scalar_fields():
    assert goal.validate_goal_stage_spec(goal.GOAL_STAGE_SPEC) == (
        goal.GOAL_STAGE_SPEC
    )
    assert goal.GOAL_SIZE_LIMITS_MM == {
        "W": 1200.0,
        "L": 1000.0,
        "H": 750.0,
    }
    assert goal.TEMPERATURE_FAMILY_LIMITS_C == {
        "primary_winding": 100.0,
        "secondary_winding": 120.0,
        "core": 120.0,
    }
    assert {
        target: goal.temperature_limit_for_target(target)
        for target in goal.PRIMARY_WINDING_TEMPERATURE_TARGETS
    } == {
        target: 100.0
        for target in goal.PRIMARY_WINDING_TEMPERATURE_TARGETS
    }
    assert {
        target: goal.temperature_limit_for_target(target)
        for target in goal.SECONDARY_WINDING_TEMPERATURE_TARGETS
    } == {
        target: 120.0
        for target in goal.SECONDARY_WINDING_TEMPERATURE_TARGETS
    }
    assert goal.GOAL_RESONANCE_MIN_HZ == 15_000.0
    assert "resonance_max_Hz" not in goal.GOAL_STAGE_SPEC
    for legacy in (
        "T_limit_C",
        "n_core_group_max",
        "primary_conductor_thickness_mm",
        "f1_split",
    ):
        mutated = copy.deepcopy(goal.GOAL_STAGE_SPEC)
        mutated[legacy] = 100.0
        with pytest.raises(goal.GoalContractError, match="legacy fields"):
            goal.validate_goal_stage_spec(mutated)


def test_goal_geometry_profile_requires_explicit_diagnostic_clearance_override():
    official = preflight.GOAL_OFFICIAL_GEOMETRY_CONSTRAINT_PROFILE
    assert official["profile_role"] == "official"
    assert official["primary_axial_clearance"] == {
        "column": "h_gap1",
        "minimum_mm": 40.0,
        "official_minimum_mm": 40.0,
        "diagnostic_override": False,
    }
    assert official["winding_height_alignment"]["mode"] == "off"
    assert official["production_or_final_design_claim_allowed"] is True
    assert official["temperature_contract_mutated"] is False
    assert official["cooling_contract_mutated"] is False
    assert preflight.validate_goal_geometry_constraint_profile(
        official
    ) == official

    for clearance in (20.0, 30.0):
        with pytest.raises(RuntimeError, match="diagnostic-only"):
            preflight.goal_geometry_constraint_profile(
                primary_axial_clearance_min_mm=clearance,
            )
    with pytest.raises(RuntimeError, match="explicit override"):
        preflight.goal_geometry_constraint_profile(
            winding_height_alignment_mode="diagnostic",
        )
    tampered = copy.deepcopy(official)
    tampered["primary_axial_clearance"]["minimum_mm"] = 20.0
    with pytest.raises(RuntimeError):
        preflight.validate_goal_geometry_constraint_profile(tampered)


@pytest.mark.parametrize("primary_axial_minimum_mm", [20.0, 30.0, 40.0])
def test_aligned_goal_lane_repairs_to_decoded_clearance_overlap_and_size(
    primary_axial_minimum_mm,
):
    profile = preflight.goal_geometry_constraint_profile(
        primary_axial_clearance_min_mm=primary_axial_minimum_mm,
        winding_height_alignment_mode="hard",
        diagnostic_override=(primary_axial_minimum_mm < 40.0),
    )
    problem = _goal_problem(
        6,
        geometry_constraint_profile=profile,
    )
    sobol_dims = preflight.load_current7_modules(
        REPO
    ).input_parameter._SOBOL_DIMS
    dimensions = {
        name: index
        for index, (name, _lower, _upper) in enumerate(
            sobol_dims
        )
    }
    coordinate = np.full((1, problem.n_var), 0.5)
    coordinate[0, dimensions["wh1"]] = 1.0
    coordinate[0, dimensions["wh2"]] = 0.0
    repaired = problem.repair_unit_coordinates(coordinate)
    assert np.array_equal(
        problem.repair_unit_coordinates(repaired),
        repaired,
    )
    output = {}
    problem._evaluate(repaired, output)
    assert output["decoder_valid"].tolist() == [True]
    row = output["frame"].iloc[0]
    overlap = min(row["nwh1"], row["nwh2"]) / max(
        row["nwh1"], row["nwh2"]
    )
    _volume, (width, length, height) = problem._goal_bounding_box_lit(row)
    assert row["h_gap1"] >= primary_axial_minimum_mm
    assert row["h_gap2"] >= 40.0
    assert overlap >= 0.9
    assert width <= goal.GOAL_SIZE_LIMITS_MM["W"]
    assert length <= goal.GOAL_SIZE_LIMITS_MM["L"]
    assert height <= goal.GOAL_SIZE_LIMITS_MM["H"]
    constraints = dict(zip(problem.constraint_names, output["G"][0]))
    assert constraints["minimum_physical_insulation"] <= 0.0
    assert (
        constraints[preflight.WINDING_HEIGHT_ALIGNMENT_CONSTRAINT]
        <= 0.0
    )
    assert (
        problem.geometry_constraint_profile_sha256
        == profile["sha256"]
    )
    assert (
        problem.hard_constraint_contract["geometry_constraint_profile_sha256"]
        == profile["sha256"]
    )


def test_diagnostic_alignment_reports_without_mutating_hard_constraint_schema():
    profile = preflight.goal_geometry_constraint_profile(
        primary_axial_clearance_min_mm=30.0,
        winding_height_alignment_mode="diagnostic",
        diagnostic_override=True,
    )
    problem = _goal_problem(6, geometry_constraint_profile=profile)
    assert preflight.WINDING_HEIGHT_ALIGNMENT_CONSTRAINT not in (
        problem.constraint_names
    )
    audit = problem.hard_geometry_audit(
        np.full((1, problem.n_var), 0.5)
    )
    assert audit["winding_height_alignment_is_diagnostic"] is True
    assert audit["winding_height_alignment_is_hard"] is False
    assert np.isfinite(audit["winding_height_alignment_G"]).all()


def test_fixed_lm2mh_resonance_uses_leakage_and_reflected_turn_ratio():
    screen = preflight.derive_fixed_lm2mh_self_resonance(
        {
            "Llt_phys": 27.5,
            "C_tx_tx_F": 1.2e-9,
            "C_rx_rx_F": 12.0e-12,
        },
        {
            "N1_main": 6,
            "N1_side": 0,
            "N2_main": 60,
            "N2_side": 0,
        },
    )
    l_tx = 0.002 + 27.5e-6
    l_rx = l_tx * 100.0
    assert screen["primary_resonant_inductance_H"] == pytest.approx(l_tx)
    assert screen["secondary_resonant_inductance_H"] == pytest.approx(l_rx)
    assert screen["f_res_tx_fixed_lm2mh_Hz"] == pytest.approx(
        1.0 / (2.0 * np.pi * np.sqrt(l_tx * 1.2e-9))
    )
    assert screen["f_res_rx_fixed_lm2mh_Hz"] == pytest.approx(
        1.0 / (2.0 * np.pi * np.sqrt(l_rx * 12.0e-12))
    )
    assert screen["resonance_contract_schema"] == (
        preflight.GOAL_FIXED_LM_RESONANCE_SCHEMA
    )


def test_fixed_lm2mh_wrapper_changes_only_resonance_G_before_scaling():
    problem = _goal_problem(6)
    coordinate = _coordinate(problem, cw1_unit=0.5, side_unit=0.4)
    before = {}
    problem._evaluate(coordinate, before)
    base_hash = problem.hard_constraint_contract_sha256

    _base_evaluate, installation = (
        preflight.install_goal_fixed_lm2mh_resonance(problem)
    )
    after = {}
    problem._evaluate(coordinate, after)
    resonance_index = problem.constraint_index[
        preflight.RESONANCE_MINIMUM_CONSTRAINT
    ]
    other = [
        index
        for index in range(problem.n_ieq_constr)
        if index != resonance_index
    ]
    assert np.array_equal(before["G"][:, other], after["G"][:, other])
    row = after["frame"].iloc[0]
    expected = preflight.derive_fixed_lm2mh_self_resonance(
        {
            "Llt_phys": 27.5,
            "C_tx_tx_F": 1.0e-12,
            "C_rx_rx_F": 1.0e-14,
        },
        row,
    )
    assert after["G"][0, resonance_index] == pytest.approx(
        goal.GOAL_RESONANCE_MIN_HZ
        - expected["f_res_min_tx_rx_only_Hz"]
    )
    assert problem.hard_constraint_contract_sha256 != base_hash
    assert installation["base_hard_constraint_contract_sha256"] == base_hash
    assert installation["other_physical_constraints_mutated"] is False
    assert (
        problem._goal_fixed_lm2mh_resonance_contract["surrogate_k_used"]
        is False
    )
    assert installation["resonance_contract_schema"] == (
        preflight.GOAL_FIXED_LM_RESONANCE_SCHEMA
    )
    assert installation[
        "base_stage_magnetizing_inductance_factor_ignored"
    ] is True
    assert problem.hard_constraint_contract[
        "effective_self_resonance_authority"
    ] == "fixed_primary_Lm_2mH_plus_physical_leakage"


def _compact_authenticated(*, quality_passed: bool):
    artifacts = {"target/models.pkl": "a" * 64}
    return types.SimpleNamespace(
        quality={"passed": quality_passed},
        report={"artifacts": artifacts},
        evidence={
            "dataset": {"sha256": "b" * 64},
            "quality_status": {"sha256": "c" * 64},
        },
    )


def _compact_authorization(
    problem,
    *,
    mode,
    seed,
    contract,
    bank,
    authenticated,
):
    diagnostic = mode == preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
    dataset_authentication = (
        {}
        if diagnostic
        else {
            "thermal_mesh_policy": preflight.GOAL_B7_THERMAL_MESH_POLICY,
            "thermal_mesh_plan_contract_version": (
                preflight.GOAL_B7_THERMAL_MESH_PLAN
            ),
            "every_authenticated_row_exact_B7_v8": True,
            "dataset_sha256": authenticated.evidence[
                "dataset"
            ]["sha256"],
            "authenticated_row_count": 8,
            "unique_source_tasks": 4,
        }
    )
    activation_value = (
        {
            "schema_version": diagnostic_scout.ACTIVATION_SCHEMA,
            "campaign_id": diagnostic_scout.CAMPAIGN_ID,
            "fixed_primary_turns": 6,
            "source_identity": {
                "dataset_sha256": authenticated.evidence[
                    "dataset"
                ]["sha256"],
                "evaluation_model_sha256": goal.canonical_sha256(
                    authenticated.report["artifacts"]
                ),
                "quality_status_sha256": authenticated.evidence[
                    "quality_status"
                ]["sha256"],
            },
            "source_quality_passed": authenticated.quality["passed"],
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "reserved_fresh512_seed_interval_used": False,
            "effective_hard_constraint_contract_sha256": (
                problem.hard_constraint_contract_sha256
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "compact_search_contract_sha256": contract["sha256"],
            "compact_coordinate_bank_sha256": bank["sha256"],
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
        if diagnostic
        else {
            "schema_version": launch.FRESH512_ACTIVATION_SCHEMA,
            "dataset_sha256": authenticated.evidence[
                "dataset"
            ]["sha256"],
            "evaluation_model_sha256": goal.canonical_sha256(
                authenticated.report["artifacts"]
            ),
            "quality_status_sha256": authenticated.evidence[
                "quality_status"
            ]["sha256"],
            "quality_passed": True,
            "search_only_proposal": False,
            "seed_start": launch.FRESH_AL_SEED_START,
            "seed_end_inclusive": launch.FRESH_AL_SEED_END,
            "seed_count": launch.FRESH_AL_SEED_COUNT,
            "seeds_per_N1_stratum": 128,
            "effective_hard_constraint_contract_sha256": (
                problem.hard_constraint_contract_sha256
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "dataset_authentication": dataset_authentication,
            "compact_by_N1": {
                str(turns): {
                    "contract_sha256": contract["sha256"],
                    "coordinate_bank_sha256": bank["sha256"],
                }
                for turns in goal.GOAL_PRIMARY_TURN_STRATA
            },
            "global_exact_A_B_C_and_independent_H_coverage": True,
            "invalid_or_near_band_fallback_allowed": False,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    activation = launch._seal(activation_value)
    authorization = preflight.seal_goal_compact_run_authorization(
        authorization_mode=mode,
        source_activation_schema=(
            diagnostic_scout.ACTIVATION_SCHEMA
            if diagnostic
            else launch.FRESH512_ACTIVATION_SCHEMA
        ),
        source_activation_payload_sha256=activation["payload_sha256"],
        seed=seed,
        fixed_primary_turns=problem.fixed_primary_turns,
        dataset_sha256=authenticated.evidence["dataset"]["sha256"],
        evaluation_model_sha256=goal.canonical_sha256(
            authenticated.report["artifacts"]
        ),
        quality_status_sha256=authenticated.evidence[
            "quality_status"
        ]["sha256"],
        source_quality_passed=authenticated.quality["passed"],
        effective_hard_constraint_contract_sha256=(
            problem.hard_constraint_contract_sha256
        ),
        compact_search_contract_sha256=contract["sha256"],
        compact_coordinate_bank_sha256=bank["sha256"],
        dataset_authentication_sha256=(
            None
            if diagnostic
            else goal.canonical_sha256(dataset_authentication)
        ),
    )
    return authorization, activation


def test_direct_runner_compact_call_without_per_seed_authority_fails_closed():
    problem = _goal_problem(6)
    preflight.install_goal_fixed_lm2mh_resonance(problem)
    runner = types.SimpleNamespace(problem=problem)
    with pytest.raises(
        RuntimeError,
        match="authenticated per-seed authority",
    ):
        preflight.Current7Tier1Runner.run_one(
            runner,
            seed=launch.FRESH_AL_SEED_START,
            population=4,
            max_generations=1,
            compact_search_contract=(
                preflight.goal_compact_search_contract(6)
            ),
            compact_coordinate_bank={},
        )


def test_compact_authority_binds_fresh512_seed_dataset_model_and_quality():
    problem = _goal_problem(6)
    preflight.install_goal_fixed_lm2mh_resonance(problem)
    contract = preflight.goal_compact_search_contract(6)
    bank = {"sha256": "f" * 64}
    authenticated = _compact_authenticated(quality_passed=True)
    with pytest.raises(RuntimeError, match="quality state must be boolean"):
        preflight.seal_goal_compact_run_authorization(
            authorization_mode=preflight.GOAL_COMPACT_AUTH_FRESH512,
            source_activation_schema=launch.FRESH512_ACTIVATION_SCHEMA,
            source_activation_payload_sha256="0" * 64,
            seed=launch.FRESH_AL_SEED_START,
            fixed_primary_turns=6,
            dataset_sha256="1" * 64,
            evaluation_model_sha256="2" * 64,
            quality_status_sha256="3" * 64,
            source_quality_passed="true",
            effective_hard_constraint_contract_sha256="4" * 64,
            compact_search_contract_sha256="5" * 64,
            compact_coordinate_bank_sha256="6" * 64,
            dataset_authentication_sha256="7" * 64,
        )
    seed = launch.FRESH_AL_SEED_START
    authorization, activation = _compact_authorization(
        problem,
        mode=preflight.GOAL_COMPACT_AUTH_FRESH512,
        seed=seed,
        contract=contract,
        bank=bank,
        authenticated=authenticated,
    )
    observed = preflight.validate_goal_compact_run_authorization(
        authorization,
        source_activation=activation,
        problem=problem,
        seed=seed,
        authenticated=authenticated,
        compact_contract=contract,
        compact_bank=bank,
    )
    assert observed["every_authenticated_row_exact_B7_v8"] is True
    assert observed["screening_only"] is False
    assert observed["production_eligible"] is False

    wrong_dataset = _compact_authenticated(quality_passed=True)
    wrong_dataset.evidence["dataset"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runtime binding"):
        preflight.validate_goal_compact_run_authorization(
            authorization,
            source_activation=activation,
            problem=problem,
            seed=seed,
            authenticated=wrong_dataset,
            compact_contract=contract,
            compact_bank=bank,
        )
    tampered = copy.deepcopy(authorization)
    tampered["seed"] += 1
    with pytest.raises(RuntimeError, match="seal mismatch"):
        preflight.validate_goal_compact_run_authorization_seal(tampered)
    tampered_activation = copy.deepcopy(activation)
    tampered_activation["dataset_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source activation seal"):
        preflight.validate_goal_compact_run_authorization(
            authorization,
            source_activation=tampered_activation,
            problem=problem,
            seed=seed,
            authenticated=authenticated,
            compact_contract=contract,
            compact_bank=bank,
        )


def test_signed_diagnostic_compact_authority_is_nonreserved_screening_only():
    problem = _goal_problem(6)
    preflight.install_goal_fixed_lm2mh_resonance(problem)
    contract = preflight.goal_compact_search_contract(6)
    assert "fresh512_only" not in contract
    assert contract["authorized_execution_roles"] == [
        preflight.GOAL_COMPACT_AUTH_FRESH512,
        preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC,
    ]
    assert contract["coverage_requirement_by_authorization_role"][
        preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
    ] == "exact_active_strata_for_N1_6_screening_only"
    bank = {"sha256": "f" * 64}
    authenticated = _compact_authenticated(quality_passed=False)
    seed = diagnostic_scout.DEFAULT_SEED_START
    authorization, activation = _compact_authorization(
        problem,
        mode=preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC,
        seed=seed,
        contract=contract,
        bank=bank,
        authenticated=authenticated,
    )
    observed = preflight.validate_goal_compact_run_authorization(
        authorization,
        source_activation=activation,
        problem=problem,
        seed=seed,
        authenticated=authenticated,
        compact_contract=contract,
        compact_bank=bank,
    )
    assert observed["screening_only"] is True
    assert observed["production_eligible"] is False
    assert observed["final_design_claim_allowed"] is False
    assert observed["fresh512_activation_evidence"] is False
    assert observed["reserved_fresh512_seed"] is False


def test_n1_5_compact_initialization_does_not_claim_missing_C_bridge():
    contract = preflight.goal_compact_search_contract(5)
    claims = preflight.goal_compact_initialization_claims(
        contract,
        {
            "membership_counts": {
                "compact_A": 2,
                "compact_B": 2,
                "compact_C": 0,
                "height_boundary": 4,
            }
        },
    )
    assert claims["active_strata_for_this_N1"] == [
        "compact_A",
        "compact_B",
        "height_boundary",
    ]
    assert claims["exact_A_B_C_bridge_inserted"] is False
    assert claims["global_fresh512_compact_C_coverage_deferred"] is True


def test_local_preflight_preserves_base_hard_contract_without_fixed_lm(
    monkeypatch,
    tmp_path,
):
    created = {}
    artifacts = {"target/models.pkl": "a" * 64}

    def make_runner(turns):
        problem = _goal_problem(turns)
        created[turns] = problem
        authenticated = types.SimpleNamespace(
            quality={"passed": True},
            report={"artifacts": artifacts},
            evidence={
                "generation_relative": "generations/g1",
                "train_report": {"sha256": "1" * 64},
                "candidate": {"sha256": "2" * 64},
                "quality_status": {"sha256": "3" * 64},
                "dataset": {"sha256": "4" * 64},
                "profile": {"canonical_sha256": "5" * 64},
                "relocation": {"enabled": False},
            },
        )
        runner = types.SimpleNamespace(
            problem=problem,
            authenticated=authenticated,
            code_identity={"revision": "6" * 40},
            models=_models(),
        )

        def repair_coordinates(values, *, stage):
            repaired = problem.repair_unit_coordinates(values)
            return repaired, {
                "stage": stage,
                "sha256": goal.canonical_sha256(repaired.tolist()),
            }

        def evaluate_coordinates(values):
            out = {}
            problem._evaluate(values, out)
            return out

        runner.repair_coordinates = repair_coordinates
        runner.evaluate_coordinates = evaluate_coordinates
        return runner

    first = make_runner(5)
    monkeypatch.setattr(
        preflight,
        "build_authenticated_runner",
        lambda **_kwargs: first,
    )
    monkeypatch.setattr(
        preflight,
        "runner_for_fixed_primary_turns",
        lambda _runner, turns: make_runner(turns),
    )
    monkeypatch.setattr(
        launch,
        "_quality_contract",
        lambda **_kwargs: {
            "quality_passed": True,
            "search_only_proposal": False,
        },
    )

    def forbidden_install(_problem):
        raise AssertionError("local preflight must preserve base resonance")

    monkeypatch.setattr(
        preflight,
        "install_goal_fixed_lm2mh_resonance",
        forbidden_install,
    )
    local, observed_first = launch.run_local_preflight(
        generation=tmp_path / "generation",
        candidate=tmp_path / "candidate.json",
        quality_status=tmp_path / "quality.json",
        code_root=tmp_path,
        expected_code_revision="6" * 40,
    )
    assert observed_first is first
    assert local["fixed_lm2mh_resonance_installed"] is False
    assert local["hard_constraint_contract_sha256"] == (
        first.problem.hard_constraint_contract_sha256
    )
    assert local["base_hard_constraint_contract_sha256"] == (
        local["hard_constraint_contract_sha256"]
    )
    assert all(
        item["fixed_lm2mh_resonance_installed"] is False
        for item in local["strata"].values()
    )
    assert all(
        not getattr(
            problem,
            "_goal_fixed_lm2mh_resonance_installed",
            False,
        )
        for problem in created.values()
    )


def test_nonreserved_execute_seed_does_not_install_fixed_lm(
    monkeypatch,
    tmp_path,
):
    artifacts = {"target/models.pkl": "a" * 64}
    model_sha = goal.canonical_sha256(artifacts)
    local = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "4" * 64,
            "profile_sha256": "5" * 64,
            "evaluation_model_sha256": model_sha,
            "train_report_sha256": "1" * 64,
            "candidate_sha256": "2" * 64,
            "quality_status_sha256": "3" * 64,
            "code": {"revision": "6" * 40},
            "search_only_proposal": False,
        }
    )
    source = {
        "generation": "generation",
        "candidate": "candidate",
        "quality_status": "quality",
        "code_root": "code",
        "dataset": "dataset",
        "profile": "profile",
        "expected_code_revision": "6" * 40,
    }
    _bundle, tasks, _scheduler = launch.build_bundle_values(
        local_preflight=local,
        assignments=[
            {
                "seed": diagnostic_scout.DEFAULT_SEED_START,
                "ordinal": 0,
                "phase": "single",
                "wave": 0,
                "fixed_primary_turns": 6,
            }
        ],
        output_root=tmp_path,
        source=source,
        code_manifest=_synthetic_goal_code_manifest("6" * 40),
    )
    task = tasks[0]
    payload = tmp_path / "task.json"
    launch._atomic_json(payload, task)
    problem = types.SimpleNamespace(
        hard_constraint_contract_sha256="d" * 64,
        hard_constraint_contract={"stage": "base-legacy"},
        constraint_names=preflight.GOAL_CONSTRAINT_NAMES,
        temperature_targets=goal.GOAL_TEMPERATURE_TARGETS,
    )
    authenticated = types.SimpleNamespace(
        quality={"passed": True},
        report={"artifacts": artifacts},
        evidence={
            "train_report": {"sha256": "1" * 64},
            "candidate": {"sha256": "2" * 64},
            "quality_status": {"sha256": "3" * 64},
            "dataset": {"sha256": "4" * 64},
            "profile": {"canonical_sha256": "5" * 64},
        },
    )
    observed = {}
    runner = types.SimpleNamespace(
        problem=problem,
        authenticated=authenticated,
        code_identity={"revision": "6" * 40},
    )
    runner.install_offspring_repair_operator = (
        lambda **_kwargs: {"installed": True}
    )

    def run_one(**kwargs):
        observed.update(kwargs)
        kwargs["pre_optimization_callback"](
            {
                "offspring_repair_operator_installed": True,
                "optimizer_execution_started": False,
            }
        )
        return types.SimpleNamespace(
            tier1_evaluated_generations=launch.GENERATIONS,
            tier1_completed_generations=(
                launch.EXPECTED_ALGORITHM_N_GEN_COUNTER
            ),
            tier1_repair_audit={},
            tier1_topology_evolution_audit={},
        )

    runner.run_one = run_one
    monkeypatch.setattr(
        preflight,
        "build_authenticated_runner",
        lambda **_kwargs: runner,
    )

    def forbidden_install(_problem):
        raise AssertionError("nonreserved execute must preserve base resonance")

    monkeypatch.setattr(
        preflight,
        "install_goal_fixed_lm2mh_resonance",
        forbidden_install,
    )
    monkeypatch.setattr(
        preflight,
        "install_optimizer_scaling",
        lambda _problem, **_kwargs: (
            lambda *_args, **_inner_kwargs: None,
            {"schema_version": "test-scaling"},
        ),
    )
    monkeypatch.setattr(
        preflight,
        "persist_search_outputs",
        lambda *_args, **_kwargs: {
            "terminal_population_count": launch.POPULATION,
            "physical_feasible_count": 0,
            "feasible_pareto_count": 0,
            "artifact_inventory": {},
            "artifact_inventory_sha256": goal.canonical_sha256({}),
            "terminal_physical_candidates_manifest": {},
        },
    )
    output = tmp_path / "result"
    path = launch.execute_seed(
        types.SimpleNamespace(
            payload=payload,
            output=output,
            relocation=None,
        )
    )
    result = launch._read_json(path)
    assert observed["compact_search_contract"] is None
    assert observed["compact_coordinate_bank"] is None
    assert observed["compact_run_authorization"] is None
    assert observed["compact_source_activation"] is None
    assert result.get("fixed_lm2mh_resonance_installation") is None
    assert result.get("resonance_contract_schema") is None
    assert problem.hard_constraint_contract_sha256 == "d" * 64


def test_n1_6_compact_bank_bridges_exact_A_B_C_and_independent_H():
    problem = _goal_problem(6)
    contract = preflight.goal_compact_search_contract(6)
    bank = preflight.build_goal_compact_coordinate_bank(
        problem,
        seed=2_607_263_006,
        compact_contract=contract,
        maximum_attempts_per_stratum=128,
    )
    repaired, audit = preflight.validate_goal_compact_coordinate_bank(
        problem,
        bank,
        compact_contract=contract,
    )
    assert repaired.shape[1] == problem.n_var
    assert bank["membership_counts"] == {
        "compact_A": 2,
        "compact_B": 2,
        "compact_C": 2,
        "height_boundary": 4,
    }
    assert audit["exact_stratum_membership_verified"] is True
    assert sum(bool(value) for value in bank["topology_counts"].values()) >= 4
    assert all(
        record["W_mm"] <= 1200.0
        and record["L_mm"] <= 1000.0
        and record["H_mm"] <= 750.0
        for record in bank["rows"]
    )
    assert any(
        "compact_C" in record["memberships"]
        and 1160.0 <= record["W_mm"] <= 1170.0
        and 960.0 <= record["L_mm"] <= 975.0
        for record in bank["rows"]
    )
    rng = np.random.default_rng(2_607_263_106)
    coordinates = np.asarray(bank["coordinates"], dtype=float)
    for stratum in contract["active_strata_for_this_N1"]:
        donors = coordinates[
            [
                index
                for index, record in enumerate(bank["rows"])
                if stratum in record["memberships"]
            ]
        ]
        mutant, evidence = preflight._goal_compact_joint_mutant(
            problem,
            donors,
            stratum=stratum,
            compact_contract=contract,
            random_state=rng,
        )
        _replayed, records = preflight._goal_compact_coordinate_replay(
            problem,
            mutant.reshape(1, -1),
            strata=contract["strata"],
        )
        assert stratum in records[0]["memberships"]
        assert evidence["joint_coordinates_changed_after_repair"] >= 3
        assert evidence["near_band_fallback_used"] is False


def test_cw1_grid_has_exact_endpoints_and_no_independent_f1_split():
    assert goal.cw1_from_unit_coordinate(0.0) == 1.0
    assert goal.cw1_from_unit_coordinate(0.5) == 5.5
    assert goal.cw1_from_unit_coordinate(1.0) == 10.0
    for value in (1.0, 1.01, 5.5, 9.99, 10.0):
        assert goal.validate_cw1_mm(value) == value
        assert goal.cw1_from_unit_coordinate(
            goal.cw1_unit_coordinate(value)
        ) == value
    for value in (0.99, 1.005, 10.01):
        with pytest.raises(goal.GoalContractError):
            goal.validate_cw1_mm(value)
    assert (
        goal.GOAL_STAGE_SPEC["cw1_search_mm"][
            "f1_split_independent_search_allowed"
        ]
        is False
    )


@pytest.mark.parametrize("primary_turns", [5, 6, 7, 8])
@pytest.mark.parametrize(
    ("unit", "expected_cw1"),
    [(0.0, 1.0), (0.5, 5.5), (1.0, 10.0)],
)
def test_goal_problem_uses_split_temperature_and_generalized_budget(
    primary_turns, unit, expected_cw1
):
    problem = _goal_problem(primary_turns)
    repaired = _coordinate(problem, cw1_unit=unit)
    assert np.array_equal(
        problem.repair_unit_coordinates(repaired), repaired
    )
    output = {}
    problem._evaluate(repaired, output)
    assert output["decoder_valid"].tolist() == [True]
    row = output["frame"].iloc[0]
    assert row["cw1"] == expected_cw1
    assert (
        int(row["N1_main"]) + int(row["N1_side"])
        == primary_turns
    )
    assert 2 <= int(row["n_core_group"]) <= 10
    assert goal.dynamic_core_group_violation(row) <= 0.0
    budget = preflight.winding_budget_identity(
        row, expected_cw1_mm=expected_cw1
    )
    assert budget["passed"] is True
    assert budget["post_decode_field_override_performed"] is False
    winding_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_Tx_leeward_max"
    )
    body_winding_index = problem.constraint_names.index(
        "temperature_robust_limit:T_max_Tx"
    )
    secondary_winding_index = problem.constraint_names.index(
        "temperature_robust_limit:T_max_Rx_main"
    )
    core_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_core_center_max"
    )
    assert output["G"][0, winding_index] == pytest.approx(0.25)
    assert output["G"][0, body_winding_index] == pytest.approx(0.25)
    assert output["G"][0, secondary_winding_index] == pytest.approx(-19.75)
    assert output["G"][0, core_index] == pytest.approx(-0.75)
    assert "core_group_dynamic_validity" in problem.constraint_names
    assert "core_group_manufacturability_limit" not in (
        problem.constraint_names
    )
    assert problem.temperature_contract == goal.GOAL_TEMPERATURE_CONTRACT
    assert problem.spec.get("T_limit_C") is None


def test_goal_problem_rejects_old_turn_strata_and_fixed_identity_override():
    with pytest.raises(ValueError, match="campaign strata"):
        _goal_problem(4)
    with pytest.raises(ValueError, match="campaign strata"):
        _goal_problem(9)

    modules = preflight.load_current7_modules(REPO)
    problem_class = preflight.create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(
            modules.nsga2_problem.design_analytical_b_field_t
        ),
    )
    with pytest.raises(ValueError, match="fan_velocity"):
        problem_class(
            _models(),
            spec=goal.GOAL_STAGE_SPEC,
            density_gate=lambda frame: np.full(len(frame), -1.0),
            fixed_overrides={"fan_velocity": 2.0},
            fixed_primary_turns=5,
        )


def test_goal_side_winding_masks_both_body_and_probe_constraints():
    problem = _goal_problem(5)
    coordinate = _coordinate(problem, side_unit=0.0)
    output = {}
    problem._evaluate(coordinate, output)
    assert int(output["frame"].iloc[0]["N2_side"]) == 0
    for target in ("T_max_Rx_side", "Tprobe_Rx_side_leeward_max"):
        index = problem.constraint_names.index(
            f"temperature_robust_limit:{target}"
        )
        assert output["G"][0, index] == -preflight.BIG


def test_goal_g0_has_25_targets_and_exact_body_quality_thresholds():
    assert tuple(adapter.GOAL_REQUIRED_MODEL_TARGETS[-11:]) == (
        goal.GOAL_TEMPERATURE_TARGETS
    )
    assert len(goal.GOAL_G0_MODEL_TARGETS) == 25
    assert goal.GOAL_G0_MODEL_TARGETS[-11:] == (
        *goal.PROBE_TEMPERATURE_TARGETS,
        *goal.BODY_WINDING_TEMPERATURE_TARGETS,
        *goal.BODY_CORE_TEMPERATURE_TARGETS,
    )
    assert set(goal.GOAL_G0_MODEL_TARGETS) == (
        set(adapter.GOAL_REQUIRED_MODEL_TARGETS) | {"B_max_core"}
    )
    thresholds = json.loads(
        (
            REPO
            / "regression_260707"
            / "training"
            / "model_quality_thresholds.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "min_r2": 0.85,
        "max_rmse": 5.0,
        "max_p90_ape_pct": 10.0,
        "max_interval_p90_width": 10.0,
    }
    for target in ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core"):
        assert thresholds["targets"][target] == expected

    from regression_260707.training import checkpoint_train

    for target in goal.BODY_WINDING_TEMPERATURE_TARGETS + (
        *goal.BODY_CORE_TEMPERATURE_TARGETS,
    ):
        assert target not in checkpoint_train.TARGETS
        assert checkpoint_train.OPTIONAL_TARGETS[target] == {
            "transform": "t50",
            "metric_focus": "rmse",
        }
        outlier = pd.DataFrame(
            {
                "_strict_valid_full": [True],
                target: [5000.0],
                "physics_data_revision": ["goal-g0"],
            }
        )
        assert checkpoint_train.filter_valid_training_rows(
            outlier, target
        ).empty
    assert len(checkpoint_train.TARGETS) == 21
    assert set(checkpoint_train.TRAINABLE_TARGETS) == set(
        goal.GOAL_G0_MODEL_TARGETS
    )


def test_goal_24_model_inference_binding_is_accepted():
    binding = {
        "threads_per_model": 8,
        "target_count": len(adapter.GOAL_REQUIRED_MODEL_TARGETS),
        "model_count": len(adapter.GOAL_REQUIRED_MODEL_TARGETS),
        "families": ["extratrees"],
        "family_threads": {"extratrees": 1},
        "semaphore_free_families": ["extratrees"],
        "semaphore_free_sklearn_forest": True,
        "policy": preflight.FAMILY_SPECIFIC_INFERENCE_POLICY,
    }
    assert preflight._terminal_inference_binding_contract(binding) == (True, 8)


def test_goal_launcher_builds_isolated_32_seed_canary_rolling_payloads(
    tmp_path,
):
    assignments = launch.seed_assignments(
        mode="rolling32", seed_start=2607261000
    )
    assert len(assignments) == 32
    assert [item["fixed_primary_turns"] for item in assignments[:4]] == [
        5,
        6,
        7,
        8,
    ]
    assert [item["phase"] for item in assignments[:4]] == ["canary"] * 4
    assert {
        turns: sum(
            item["fixed_primary_turns"] == turns for item in assignments
        )
        for turns in goal.GOAL_PRIMARY_TURN_STRATA
    } == {5: 8, 6: 8, 7: 8, 8: 8}
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "profile_sha256": "2" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "code": {"revision": "c" * 40},
            "search_only_proposal": False,
        }
    )
    source = {
        "generation": "G0",
        "candidate": "candidate.json",
        "quality_status": "quality.json",
        "code_root": "repo",
        "dataset": "strict.parquet",
        "profile": "profile.json",
        "expected_code_revision": "c" * 40,
    }
    code_manifest = _synthetic_goal_code_manifest()
    bundle, tasks, scheduler = launch.build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignments,
        output_root=tmp_path,
        source=source,
        code_manifest=code_manifest,
    )
    assert bundle["schema_version"] == launch.BUNDLE_SCHEMA
    assert bundle["task_count"] == 32
    assert bundle["legacy_current7_bundle_or_release_identity_reused"] is False
    assert bundle["relocation_contract"]["schema_version"] == (
        launch.RELOCATION_SCHEMA
    )
    assert bundle["relocation_contract"][
        "source_absolute_paths_are_not_worker_authority"
    ] is True
    assert bundle["relocation_contract"]["roles"] == list(
        launch.RUNTIME_SOURCE_ROLES
    )
    assert bundle["code_manifest"]["payload_sha256"] == (
        code_manifest["payload_sha256"]
    )
    assert code_manifest["staged_path_rule"] == (
        "bundle_root/<code_inventory_key>"
    )
    assert code_manifest["source_checkout_mutated"] is False
    assert bundle["relocation_contract"]["relocation_files_emitted"] is False
    assert bundle["relocation_contract"][
        "stager_must_generate_task_bound_relocation"
    ] is True
    assert scheduler["maximum_parallel_tasks"] == 32
    assert scheduler["canary_seed_count"] == 4
    assert scheduler["scheduler_submission_performed"] is False
    assert all(launch.validate_task_payload(task) is task for task in tasks)
    assert all(task["population"] == 320 for task in tasks)
    assert all(task["generations"] == 300 for task in tasks)
    assert all(
        task["temperature_contract_sha256"]
        == goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        for task in tasks
    )


def test_goal_launcher_task_rejects_legacy_scalar_temperature():
    assignment = launch.seed_assignments(
        mode="single", seed_start=2607260001, fixed_primary_turns=8
    )
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "profile_sha256": "2" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "code": {"revision": "c" * 40},
            "search_only_proposal": True,
        }
    )
    _bundle, tasks, _scheduler = launch.build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignment,
        output_root=Path("."),
        source={
            "generation": "G0",
            "candidate": "candidate.json",
            "quality_status": "quality.json",
            "code_root": "repo",
            "dataset": "strict.parquet",
            "profile": "profile.json",
            "expected_code_revision": "c" * 40,
        },
        code_manifest=_synthetic_goal_code_manifest(),
    )
    forged = copy.deepcopy(tasks[0])
    forged.pop("payload_sha256")
    forged["stage_spec"]["T_limit_C"] = 100.0
    forged = launch._seal(forged)
    with pytest.raises(goal.GoalContractError, match="legacy fields"):
        launch.validate_task_payload(forged)


def test_goal_launcher_scales_to_authenticated_512_seed_rollout():
    assignments = launch.seed_assignments(
        mode="rolling",
        seed_start=2607262000,
        seed_count=512,
        wave_size=32,
    )
    assert len(assignments) == 512
    assert sum(item["phase"] == "canary" for item in assignments) == 4
    assert len({item["seed"] for item in assignments}) == 512
    assert {
        turns: sum(
            item["fixed_primary_turns"] == turns for item in assignments
        )
        for turns in goal.GOAL_PRIMARY_TURN_STRATA
    } == {5: 128, 6: 128, 7: 128, 8: 128}
    assert max(item["wave"] for item in assignments) == 16


def _synthetic_compact_activation_bank(turns, contract):
    required = {
        name: (
            contract["initialization"][
                "minimum_exact_rows_height_boundary"
            ]
            if name == "height_boundary"
            else contract["initialization"][
                "minimum_exact_rows_per_WL_stratum"
            ]
        )
        for name in contract["active_strata_for_this_N1"]
    }
    rows = [
        {"memberships": [name]}
        for name, count in required.items()
        for _index in range(count)
    ]
    coordinates = [[float(index)] for index in range(len(rows))]
    memberships = {
        name: sum(name in row["memberships"] for row in rows)
        for name in contract["strata"]
    }
    topology_counts = {
        str(topology): int(index < 4)
        for index, topology in enumerate(
            contract["turn_split_topologies_N2_main"]
        )
    }
    unsigned = {
        "schema_version": preflight.GOAL_COMPACT_BANK_SCHEMA,
        "fixed_primary_turns": turns,
        "seed": launch.FRESH_AL_SEED_START + turns,
        "compact_search_contract_sha256": contract["sha256"],
        "coordinates": coordinates,
        "coordinate_sha256": goal.canonical_sha256(coordinates),
        "rows": rows,
        "membership_counts": memberships,
        "topology_counts": topology_counts,
        "generation_attempts": {name: 1 for name in required},
        "decoder_repair_and_exact_dimension_replay_performed": True,
        "archive_coordinate_donor_used": False,
        "near_band_fallback_used": False,
    }
    return {**unsigned, "sha256": goal.canonical_sha256(unsigned)}


def test_reserved_fresh512_dry_run_manifest_seals_compact_activation_without_post(
    tmp_path,
):
    local = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "base_hard_constraint_contract_sha256": "0" * 64,
            "fixed_lm2mh_resonance_contract": (
                preflight.goal_fixed_lm2mh_resonance_contract()
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "fixed_lm2mh_resonance_installed": False,
            "dataset_sha256": "a" * 64,
            "profile_sha256": "2" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "quality": {"quality_passed": True},
            "code": {"revision": "c" * 40},
            "search_only_proposal": False,
        }
    )
    compact = {}
    for turns in goal.GOAL_PRIMARY_TURN_STRATA:
        contract = preflight.goal_compact_search_contract(turns)
        bank = _synthetic_compact_activation_bank(turns, contract)
        compact[str(turns)] = {
            "contract": contract,
            "contract_sha256": contract["sha256"],
            "coordinate_bank": bank,
            "coordinate_bank_sha256": bank["sha256"],
        }
    global_counts = {
        name: sum(
            compact[str(turns)]["coordinate_bank"]["membership_counts"][
                name
            ]
            for turns in goal.GOAL_PRIMARY_TURN_STRATA
        )
        for name in preflight.GOAL_COMPACT_STRATA
    }
    activation = launch._seal(
        {
            "schema_version": launch.FRESH512_ACTIVATION_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "local_preflight_payload_sha256": local["payload_sha256"],
            "dataset_authentication": {
                "dataset_sha256": local["dataset_sha256"],
                "authenticated_row_count": 8,
                "unique_source_tasks": 4,
                "thermal_mesh_policy": (
                    "b7-rxmain-l5-shared-region-wcp-pad-"
                    "symmetry-contact-clipped-v1"
                ),
                "thermal_mesh_plan_contract_version": "thermal-mesh-plan-v8",
                "every_authenticated_row_exact_B7_v8": True,
            },
            "dataset_sha256": local["dataset_sha256"],
            "evaluation_model_sha256": local[
                "evaluation_model_sha256"
            ],
            "quality_status_sha256": local["quality_status_sha256"],
            "quality_passed": True,
            "search_only_proposal": False,
            "seed_start": launch.FRESH_AL_SEED_START,
            "seed_end_inclusive": launch.FRESH_AL_SEED_END,
            "seed_count": launch.FRESH_AL_SEED_COUNT,
            "seeds_per_N1_stratum": 128,
            "fixed_lm2mh_resonance_contract": (
                preflight.goal_fixed_lm2mh_resonance_contract()
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
            ),
            "base_hard_constraint_contract_sha256": local[
                "hard_constraint_contract_sha256"
            ],
            "effective_hard_constraint_contract_sha256": local[
                "hard_constraint_contract_sha256"
            ],
            "fixed_lm2mh_installation_by_N1": {
                str(turns): {
                    "base_hard_constraint_contract_sha256": local[
                        "hard_constraint_contract_sha256"
                    ],
                    "effective_hard_constraint_contract_sha256": local[
                        "hard_constraint_contract_sha256"
                    ],
                    "resonance_contract_schema": (
                        preflight.GOAL_FIXED_LM_RESONANCE_SCHEMA
                    ),
                    "resonance_contract_sha256": (
                        preflight.goal_fixed_lm2mh_resonance_contract()[
                            "sha256"
                        ]
                    ),
                }
                for turns in goal.GOAL_PRIMARY_TURN_STRATA
            },
            "compact_by_N1": compact,
            "global_compact_membership_counts": global_counts,
            "global_exact_A_B_C_and_independent_H_coverage": True,
            "core_center_gap_mm_search_coordinate": False,
            "core_center_gap_mm_symmetric_FEA_synthesis_required": True,
            "physical_Lm_2mH_verified": False,
            "compact_scout_role": "surrogate_screening_only",
            "compact_scout_production_or_final_design_claim_allowed": False,
            "symmetric_FEA_validation_still_required": True,
            "invalid_or_near_band_fallback_allowed": False,
            "legacy_campaign_behavior_changed": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    assignments = launch.seed_assignments(
        mode="rolling",
        seed_start=launch.FRESH_AL_SEED_START,
        seed_count=launch.FRESH_AL_SEED_COUNT,
        wave_size=32,
    )
    bundle, tasks, scheduler = launch.build_bundle_values(
        local_preflight=local,
        assignments=assignments,
        output_root=tmp_path,
        source={
            "generation": "G0",
            "candidate": "candidate.json",
            "quality_status": "quality.json",
            "code_root": "repo",
            "dataset": "strict.parquet",
            "profile": "profile.json",
            "expected_code_revision": "c" * 40,
        },
        code_manifest=_synthetic_goal_code_manifest(),
        fresh512_search_activation=activation,
    )
    assert bundle["task_count"] == 512
    assert bundle["fresh512_compact_active"] is True
    assert scheduler["maximum_parallel_tasks"] == 512
    assert scheduler["scheduler_submission_performed"] is False
    assert scheduler["scheduler_write_performed"] is False
    assert tasks[0]["fresh512_compact_active"] is True
    assert tasks[-1]["fresh512_search_activation"]["payload_sha256"] == (
        activation["payload_sha256"]
    )
    launch.validate_task_payload(tasks[0])
    launch.validate_task_payload(tasks[-1])


def test_diagnostic_n1_6_compact_contract_is_separate_screening_only():
    with pytest.raises(RuntimeError, match="reserved fresh512"):
        diagnostic_scout._seed_interval(
            launch.FRESH_AL_SEED_START,
            1,
        )
    assert diagnostic_scout._seed_interval(
        diagnostic_scout.DEFAULT_SEED_START,
        4,
    ) == list(
        range(
            diagnostic_scout.DEFAULT_SEED_START,
            diagnostic_scout.DEFAULT_SEED_START + 4,
        )
    )
    contract = preflight.goal_compact_search_contract(6)
    bank = _synthetic_compact_activation_bank(6, contract)
    source_identity = {
        "train_report_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "quality_status_sha256": "3" * 64,
        "dataset_sha256": "4" * 64,
        "profile_sha256": "5" * 64,
        "evaluation_model_sha256": "6" * 64,
        "code_revision": "7" * 40,
    }
    activation = launch._seal(
        {
            "schema_version": diagnostic_scout.ACTIVATION_SCHEMA,
            "campaign_id": diagnostic_scout.CAMPAIGN_ID,
            "fixed_primary_turns": 6,
            "source_identity": source_identity,
            "source_quality_passed": False,
            "source_quality_blockers": ["quality_not_passed"],
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
            "fixed_lm2mh_resonance_installation": {},
            "effective_hard_constraint_contract_sha256": "8" * 64,
            "compact_search_contract": contract,
            "compact_search_contract_sha256": contract["sha256"],
            "compact_coordinate_bank": bank,
            "compact_coordinate_bank_sha256": bank["sha256"],
            "invalid_or_near_band_fallback_allowed": False,
            "core_center_gap_mm_symmetric_FEA_synthesis_required": True,
            "physical_Lm_2mH_verified": False,
            "scheduler_write_performed": False,
            "scheduler_submission_performed": False,
        }
    )
    assert (
        diagnostic_scout._validate_activation(activation)["screening_only"]
        is True
    )
    task = launch._seal(
        {
            "schema_version": diagnostic_scout.TASK_SCHEMA,
            "campaign_id": diagnostic_scout.CAMPAIGN_ID,
            "task_name": "diagnostic-fixture",
            "ordinal": 0,
            "seed": diagnostic_scout.DEFAULT_SEED_START,
            "fixed_primary_turns": 6,
            "population": diagnostic_scout.POPULATION,
            "generations": diagnostic_scout.GENERATIONS,
            "inference_threads": diagnostic_scout.INFERENCE_THREADS,
            "stage_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": "8" * 64,
            "source": {
                "generation": "generation",
                "candidate": "candidate",
                "quality_status": "quality",
                "code_root": "code",
                "dataset": "dataset",
                "profile": "profile",
                "expected_code_revision": "7" * 40,
            },
            "source_identity": source_identity,
            "code_manifest_payload_sha256": "9" * 64,
            "code_inventory_sha256": "a" * 64,
            "activation": activation,
            "compact_run_authorization": (
                preflight.seal_goal_compact_run_authorization(
                    authorization_mode=(
                        preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
                    ),
                    source_activation_schema=(
                        diagnostic_scout.ACTIVATION_SCHEMA
                    ),
                    source_activation_payload_sha256=activation[
                        "payload_sha256"
                    ],
                    seed=diagnostic_scout.DEFAULT_SEED_START,
                    fixed_primary_turns=6,
                    dataset_sha256=source_identity["dataset_sha256"],
                    evaluation_model_sha256=source_identity[
                        "evaluation_model_sha256"
                    ],
                    quality_status_sha256=source_identity[
                        "quality_status_sha256"
                    ],
                    source_quality_passed=False,
                    effective_hard_constraint_contract_sha256="8" * 64,
                    compact_search_contract_sha256=contract["sha256"],
                    compact_coordinate_bank_sha256=bank["sha256"],
                )
            ),
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "fresh512_activation_evidence": False,
            "symmetric_FEA_validation_still_required": True,
        }
    )
    assert diagnostic_scout._validate_task(task) == task
    forged = copy.deepcopy(task)
    forged.pop("payload_sha256")
    forged["production_eligible"] = True
    forged = launch._seal(forged)
    with pytest.raises(RuntimeError, match="task contract"):
        diagnostic_scout._validate_task(forged)


def test_failed_g0_quality_is_sealed_as_search_only_without_lowering_thresholds():
    thresholds_path = (
        REPO
        / "regression_260707"
        / "training"
        / "model_quality_thresholds.json"
    )
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    metrics = {
        "r2": 0.80,
        "rmse": 5.5,
        "p90_ape_pct": 11.0,
        "interval_p90_width": 11.0,
    }
    quality = {
        "passed": False,
        "reasons": ["T_max_Tx:metric_below_minimum:r2"],
        "thresholds_sha256": goal.canonical_sha256(thresholds),
        "targets": {
                target: {
                    "passed": False,
                    "blocking": True,
                    "reasons": [
                        "metric_below_minimum:r2",
                        "metric_above_maximum:rmse",
                        "metric_above_maximum:p90_ape_pct",
                        "metric_above_maximum:interval_p90_width",
                    ],
                "metrics": metrics,
            }
            for target in goal.GOAL_TEMPERATURE_TARGETS
        },
    }
    contract = launch._quality_contract(quality=quality, code_root=REPO)
    assert contract["quality_passed"] is False
    assert contract["search_only_proposal"] is True
    assert contract["thresholds_lowered_or_bypassed"] is False
    assert contract["temperature_target_count"] == 11
    assert set(contract["temperature_status"]) == set(
        goal.GOAL_TEMPERATURE_TARGETS
    )
    assert contract["production_eligible"] is False
    assert contract["automatic_promotion_allowed"] is False


def _valid_goal_fea_result():
    problem = _goal_problem(5)
    coordinate = _coordinate(problem, cw1_unit=0.5, side_unit=0.0)
    frame, _shrink, valid = problem.decode_batch(coordinate)
    assert valid.tolist() == [True]
    result = frame.iloc[0].to_dict()
    result.update(
        {
            "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "Llt": 27.5,
            "full_model": 1,
            "B_max_core": 1.0,
            "T_max_Tx": 100.0,
            "T_max_Rx_main": 120.0,
            "T_max_core": 120.0,
            "P_winding_total": 3.0,
            "P_Tx_main_group": 1.0,
            "P_Rx_main_group": 2.0,
            "P_Rx_side_total": 0.0,
            "P_core_total": 1.0,
            "P_core_plate_total": 1.0,
            "P_wcp_total": 1.0,
            "f_res_min_tx_rx_only_Hz": 15_000.0,
            "thermal_pad_conductivity_W_mK": 0.2,
            **{
                target: 99.0
                for target in goal.PROBE_PRIMARY_WINDING_TEMPERATURE_TARGETS
            },
            **{
                target: 119.0
                for target in goal.PROBE_SECONDARY_WINDING_TEMPERATURE_TARGETS
            },
            **{
                target: 119.0
                for target in goal.PROBE_CORE_TEMPERATURE_TARGETS
            },
        }
    )
    return result


def test_final_gate_uses_split_temperature_box_resonance_and_identity():
    valid = _valid_goal_fea_result()
    assert finalize.physical_spec_reasons(valid) == []

    winding_hot = dict(valid, T_max_Tx=100.01)
    assert "temperature_out_of_spec:T_max_Tx" in (
        finalize.physical_spec_reasons(winding_hot)
    )
    core_hot = dict(valid, T_max_core=120.01)
    assert "temperature_out_of_spec:T_max_core" in (
        finalize.physical_spec_reasons(core_hot)
    )
    hot_probe = dict(valid, Tprobe_Tx_leeward_max=100.01)
    assert "temperature_out_of_spec:Tprobe_Tx_leeward_max" in (
        finalize.physical_spec_reasons(hot_probe)
    )
    secondary_at_limit = dict(valid, T_max_Rx_main=120.0)
    assert finalize.physical_spec_reasons(secondary_at_limit) == []
    secondary_hot = dict(valid, T_max_Rx_main=120.01)
    assert "temperature_out_of_spec:T_max_Rx_main" in (
        finalize.physical_spec_reasons(secondary_hot)
    )
    scalar = dict(valid, T_limit_C=100.0)
    assert "goal_legacy_scalar_temperature_forbidden" in (
        finalize.physical_spec_reasons(scalar)
    )
    wrong_fan = dict(valid, fan_velocity=1.6)
    assert "fixed_identity_mismatch:fan_velocity" in (
        finalize.physical_spec_reasons(wrong_fan)
    )
    low_resonance = dict(valid, f_res_min_tx_rx_only_Hz=14_999.9)
    assert "self_resonance_below_minimum" in (
        finalize.physical_spec_reasons(low_resonance)
    )


def test_consumer_rejects_scalar_or_wrong_split_contract_identity():
    value = {
        "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
        "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
        "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": (
            goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
    }
    assert consumer.validate_goal_result_contract(value) is True
    with pytest.raises(RuntimeError, match="scalar T_limit_C"):
        consumer.validate_goal_result_contract({**value, "T_limit_C": 100.0})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        consumer.validate_goal_result_contract(
            {**value, "temperature_contract_sha256": "0" * 64}
        )


def test_terminal_320_table_contains_physical_dedupe_and_provenance():
    problem = _goal_problem(5)
    one = _coordinate(problem, cw1_unit=0.5, side_unit=0.0)
    coordinates = np.repeat(one, preflight.PRODUCTION_POPULATION, axis=0)
    frame, _shrink, valid = problem.decode_batch(coordinates)
    count = preflight.PRODUCTION_POPULATION
    objectives = np.column_stack(
        (np.full(count, 100.0), np.full(count, 200.0))
    )
    physical_g = np.full((count, problem.n_ieq_constr), -1.0)
    normalized_g = np.full((count, problem.n_ieq_constr), -0.5)
    surrogate_valid = np.ones(count, dtype=bool)
    surrogate_valid[7] = False
    source = {
        "seed": 42,
        "task_id": "87052",
        "bundle_id": "goal-bundle",
        "island_id": "n1-5",
        "dataset_sha256": "c" * 64,
        "model_artifacts_sha256": "a" * 64,
        "model_generation_sha256": "b" * 64,
        "evaluation_spec_sha256": problem.stage_spec_sha256,
        "temperature_contract_sha256": (
            problem.temperature_contract_sha256
        ),
        "hard_constraint_contract_sha256": (
            problem.hard_constraint_contract_sha256
        ),
    }
    table = preflight._terminal_physical_candidate_frame(
        types.SimpleNamespace(problem=problem),
        coordinates=coordinates,
        objectives=objectives,
        optimizer_constraints=normalized_g,
        physical_constraints=physical_g,
        frame=frame,
        decoder_valid=valid,
        surrogate_physical_valid=surrogate_valid,
        source_identity=source,
    )
    assert len(table) == 320
    assert table["terminal_population_index"].tolist() == list(range(320))
    assert table["candidate_physics_sha"].nunique() == 1
    assert (
        table["candidate_physics_sha"]
        == table["physical_geometry_sha256"]
    ).all()
    assert table["source_seed"].unique().tolist() == [42]
    assert table["source_task_id"].unique().tolist() == ["87052"]
    assert table["source_bundle_id"].unique().tolist() == ["goal-bundle"]
    assert table["surrogate_physical_valid"].sum() == 319
    assert table["surrogate_physicality_passed"].sum() == 319
    assert table["physical_constraint_feasible"].all()
    assert table["physical_feasible"].sum() == 319
    assert not bool(table.loc[7, "physical_feasible"])
    assert table["evaluation_model_artifacts_sha256"].unique().tolist() == [
        "a" * 64
    ]
    assert table["dataset_sha256"].unique().tolist() == ["c" * 64]
    assert table["evaluation_model_sha256"].unique().tolist() == ["a" * 64]
    assert table["constraint_spec_sha256"].unique().tolist() == [
        goal.GOAL_STAGE_SPEC_SHA256
    ]
    assert table["cooling_contract_sha256"].unique().tolist() == [
        goal.FIXED_COOLING_IDENTITY_SHA256
    ]
    assert table["operating_point_sha256"].unique().tolist() == [
        goal.FIXED_OPERATING_IDENTITY_SHA256
    ]
    assert all(
        f"physical_G:{name}" in table.columns
        and f"normalized_G:{name}" in table.columns
        for name in problem.constraint_names
    )
    with pytest.raises(RuntimeError, match="requires 320 rows"):
        preflight._terminal_physical_candidate_frame(
            types.SimpleNamespace(problem=problem),
            coordinates=coordinates[:-1],
            objectives=objectives[:-1],
            optimizer_constraints=normalized_g[:-1],
            physical_constraints=physical_g[:-1],
            frame=frame.iloc[:-1],
            decoder_valid=valid[:-1],
            surrogate_physical_valid=surrogate_valid[:-1],
            source_identity=source,
        )


def _write_synthetic_goal_seed_result(
    root: Path,
    *,
    task: Mapping[str, object],
    force_all_infeasible: bool = False,
) -> Path:
    root.mkdir()
    seed = int(task["seed"])
    fixed_primary_turns = int(task["fixed_primary_turns"])
    constraints = list(preflight.GOAL_CONSTRAINT_NAMES)
    task_sha = str(task["payload_sha256"])
    rows = []
    for index in range(launch.POPULATION):
        if index == 0:
            if seed == 101:
                volume, loss = 100.0, 200.0
            elif seed == 102:
                volume, loss = 110.0, 190.0
            else:
                volume, loss = 500.0 + seed, 500.0 + seed
        elif seed == 101 and index == 1:
            # Physical G alone passes, but a negative objective must remain
            # quarantined by the canonical surrogate-physicality evidence.
            volume, loss = -100.0, -100.0
        else:
            volume, loss = 1000.0 + index, 1000.0 + index
        surrogate_valid = not (seed == 101 and index == 1)
        geometry_sha = goal.canonical_sha256(
            {"seed": seed, "terminal_population_index": index}
        )
        physical_constraint_feasible = not force_all_infeasible
        row = {
            "terminal_population_index": index,
            "decoder_valid": True,
            "surrogate_physical_valid": surrogate_valid,
            "surrogate_physicality_passed": surrogate_valid,
            "physical_constraint_feasible": physical_constraint_feasible,
            "physical_feasible": (
                surrogate_valid and physical_constraint_feasible
            ),
            "physical_geometry_sha256": geometry_sha,
            "canonical_physical_params_sha256": geometry_sha,
            "candidate_physics_sha": geometry_sha,
            "objective_volume_L": volume,
            "objective_total_loss_W": loss,
            "physical_G_json": "{}",
            "normalized_G_json": "{}",
            "coordinate_unit_json": "[]",
            "decoded_physical_params_json": json.dumps(
                {
                    "N1_main": fixed_primary_turns,
                    "N1_side": 0,
                },
                sort_keys=True,
            ),
            "source_seed": seed,
            "source_task_id": f"task-{seed}",
            "source_bundle_id": task_sha,
            "source_island_id": f"n1-{fixed_primary_turns}",
            "dataset_sha256": task["dataset_sha256"],
            "evaluation_model_sha256": task["evaluation_model_sha256"],
            "constraint_spec_sha256": task["stage_spec_sha256"],
            "cooling_contract_sha256": (
                goal.FIXED_COOLING_IDENTITY_SHA256
            ),
            "operating_point_sha256": (
                goal.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "evaluation_model_artifacts_sha256": task[
                "evaluation_model_sha256"
            ],
            "evaluation_model_generation_sha256": task[
                "source_identity"
            ]["train_report_sha256"],
            "evaluation_spec_sha256": task["stage_spec_sha256"],
            "evaluation_temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "evaluation_hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
        }
        row.update({f"physical_G:{name}": -1.0 for name in constraints})
        row.update({f"normalized_G:{name}": -0.5 for name in constraints})
        if index == 0 and seed == 101:
            row["physical_G:exterior_width_limit"] = -40.0
            row["normalized_G:exterior_width_limit"] = -40.0 / 1200.0
        if index == 0 and seed == 102:
            row["physical_G:exterior_length_limit"] = -30.0
            row["normalized_G:exterior_length_limit"] = -30.0 / 1000.0
        if force_all_infeasible:
            row["physical_G:Llt_robust_band"] = 0.25
            row["normalized_G:Llt_robust_band"] = 0.5
        rows.append(row)
    table = pd.DataFrame(rows)
    table_path = root / "terminal_physical_candidates.csv"
    table.to_csv(table_path, index=False)
    manifest = launch._seal(
        {
            "schema_version": preflight.GOAL_TERMINAL_TABLE_SCHEMA,
            "goal_contract_required": True,
            "row_count": launch.POPULATION,
            "terminal_population_index_min": 0,
            "terminal_population_index_max": launch.POPULATION - 1,
            "columns": list(table.columns),
            "required_identity_columns": list(table.columns),
            "csv": {
                "path": table_path.name,
                "sha256": adapter.sha256_file(table_path),
                "size_bytes": table_path.stat().st_size,
            },
            "source_identity": {"seed": seed},
            "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            "one_row_per_terminal_individual": True,
            "physical_deduplication_key": "physical_geometry_sha256",
            "global_pareto_provenance_ready": True,
        }
    )
    manifest_path = root / "terminal_physical_candidates.manifest.json"
    launch._atomic_json(manifest_path, manifest)
    inventory = {
        "terminal_physical_candidates": {
            "path": table_path.name,
            "sha256": adapter.sha256_file(table_path),
            "size_bytes": table_path.stat().st_size,
        },
        "terminal_physical_candidates_manifest": {
            "path": manifest_path.name,
            "sha256": adapter.sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    result = launch._seal(
        {
            "schema_version": launch.SEARCH_RESULT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
            "task_payload_sha256": task_sha,
            "seed": seed,
            "fixed_primary_turns": fixed_primary_turns,
            "population": launch.POPULATION,
            "generations": launch.GENERATIONS,
            "evaluated_generations": launch.GENERATIONS,
            "completed_generations": launch.EXPECTED_ALGORITHM_N_GEN_COUNTER,
            "stage_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "hard_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "constraint_version": "mft-goal-20260726",
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            "dataset_sha256": task["dataset_sha256"],
            "evaluation_model_sha256": task["evaluation_model_sha256"],
            "operating_point_sha256": (
                goal.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "cooling_contract_sha256": (
                goal.FIXED_COOLING_IDENTITY_SHA256
            ),
            "constraint_names": constraints,
            "temperature_targets": list(goal.GOAL_TEMPERATURE_TARGETS),
            "terminal_population_count": launch.POPULATION,
            "physical_feasible_count": (
                0
                if force_all_infeasible
                else launch.POPULATION - (1 if seed == 101 else 0)
            ),
            "feasible_pareto_count": 0 if force_all_infeasible else 1,
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": goal.canonical_sha256(inventory),
            "terminal_physical_candidates_manifest": manifest,
            "legacy_current7_stage_or_release_identity_reused": False,
            "search_only_proposal": task["search_only_proposal"],
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        }
    )
    result_path = root / "result.json"
    launch._atomic_json(result_path, result)
    return result_path


def _write_synthetic_goal_bundle(
    root: Path,
    *,
    seeds: tuple[int, int, int, int],
    turns: tuple[int, int, int, int] = (5, 6, 7, 8),
) -> tuple[Path, list[dict[str, object]]]:
    root.mkdir()
    assignments = launch.seed_assignments(
        mode="rolling32",
        seed_start=seeds[0],
    )[:4]
    for assignment, seed, fixed_turns in zip(assignments, seeds, turns):
        assignment["seed"] = seed
        assignment["fixed_primary_turns"] = fixed_turns
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "profile_sha256": "2" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "code": {"revision": "c" * 40},
            "search_only_proposal": False,
        }
    )
    source = {
        "generation": r"Z:\documentary\registry\generations\G0",
        "candidate": r"Z:\documentary\candidate.json",
        "quality_status": r"Z:\documentary\quality.json",
        "code_root": r"Z:\documentary\code",
        "dataset": r"Z:\documentary\strict.parquet",
        "profile": r"Z:\documentary\profile.json",
        "expected_code_revision": "c" * 40,
    }
    code_manifest = _synthetic_goal_code_manifest()
    bundle, tasks, _scheduler = launch.build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignments,
        output_root=root,
        source=source,
        code_manifest=code_manifest,
    )
    launch._atomic_json(root / "code_manifest.json", code_manifest)
    for task in tasks:
        launch._atomic_json(
            root
            / "tasks"
            / f"seed-{task['seed']}-n1-{task['fixed_primary_turns']}.json",
            task,
        )
    bundle_path = root / "bundle_manifest.json"
    launch._atomic_json(bundle_path, bundle)
    return bundle_path, tasks


def test_global_pareto_recomputes_from_all_terminal_rows_and_physicality(
    tmp_path,
):
    bundle_path, tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle",
        seeds=(101, 102, 103, 104),
    )
    results = [
        _write_synthetic_goal_seed_result(
            tmp_path / f"seed-{task['seed']}",
            task=task,
        )
        for task in tasks
    ]
    manifest_path = launch.aggregate_results(
        result_paths=results,
        bundle_manifest_path=bundle_path,
        output=tmp_path / "global",
        minimum_seeds=4,
    )
    manifest = launch._validate_seal(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        schema=launch.GLOBAL_PARETO_SCHEMA,
    )
    assert manifest["input_terminal_row_count"] == 1280
    assert manifest["physical_feasible_count"] == 1279
    assert manifest["global_pareto_count"] == 2
    assert manifest["global_objective_front_count"] == 1
    assert manifest["compactness_domain_count"] == 2
    assert manifest["compactness_hard_feasible_count"] == 2
    assert manifest["compactness_acquisition_count"] == 2
    assert (
        manifest["compactness_acquisition_contract"]["trigger"]["allowed"]
        is False
    )
    assert (
        "fresh_2607263000_3511_campaign_incomplete"
        in manifest["compactness_acquisition_contract"]["trigger"][
            "reasons"
        ]
    )
    assert manifest["seed_local_pareto_merge_used"] is False
    assert manifest["authenticated_bundle"]["task_count"] == 4
    assert manifest["authenticated_bundle"]["all_four_N1_strata_covered"] is True
    pareto = pd.read_csv(tmp_path / "global" / "global_pareto_front.csv")
    assert sorted(
        zip(
            pareto["objective_volume_L"],
            pareto["objective_total_loss_W"],
        )
    ) == [(100.0, 200.0), (110.0, 190.0)]
    objective_front = pd.read_csv(
        tmp_path / "global" / "global_objective_front.csv"
    )
    assert list(
        zip(
            objective_front["objective_volume_L"],
            objective_front["objective_total_loss_W"],
        )
    ) == [(-100.0, -100.0)]
    assert (
        manifest["artifacts"]["global_objective_front"][
            "constraint_authority"
        ]
        == "unconstrained_audit_only_not_fea_eligible"
    )
    merged = pd.read_csv(
        tmp_path / "global" / "global_terminal_candidates.csv"
    )
    quarantined = merged.loc[
        (merged["source_seed"] == 101)
        & (merged["terminal_population_index"] == 1)
    ].iloc[0]
    assert bool(quarantined["physical_feasible"]) is False
    assert quarantined["global_non_dominated_rank"] == -1
    compact = pd.read_csv(
        tmp_path / "global" / "compactness_acquisition_candidates.csv"
    )
    assert set(compact["source_seed"]) == {101, 102}
    assert compact["hard_feasible"].all()
    assert (
        manifest["artifacts"]["compactness_acquisition_candidates"][
            "invalid_or_near_feasible_fallback_used"
        ]
        is False
    )


def test_global_pareto_retains_nonempty_audit_front_when_feasible_front_empty(
    tmp_path,
):
    bundle_path, tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle",
        seeds=(101, 102, 103, 104),
    )
    results = [
        _write_synthetic_goal_seed_result(
            tmp_path / f"seed-{task['seed']}",
            task=task,
            force_all_infeasible=True,
        )
        for task in tasks
    ]
    manifest_path = launch.aggregate_results(
        result_paths=results,
        bundle_manifest_path=bundle_path,
        output=tmp_path / "global",
        minimum_seeds=4,
    )
    manifest = launch._validate_seal(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        schema=launch.GLOBAL_PARETO_SCHEMA,
    )
    assert manifest["physical_feasible_count"] == 0
    assert manifest["global_pareto_count"] == 0
    assert manifest["global_objective_front_count"] == 1
    assert manifest["compactness_acquisition_count"] == 0
    assert (
        "no_hard_feasible_compact_candidate"
        in manifest["compactness_acquisition_contract"]["trigger"][
            "reasons"
        ]
    )
    assert pd.read_csv(
        tmp_path / "global" / "global_pareto_front.csv"
    ).empty
    objective_front = pd.read_csv(
        tmp_path / "global" / "global_objective_front.csv"
    )
    assert len(objective_front) == 1
    assert bool(objective_front.iloc[0]["hard_feasible"]) is False
    standard = pd.read_csv(
        tmp_path / "global" / "standard_candidates.csv"
    )
    assert set(standard["standard_selection_basis"]) == {
        "near_feasible_fallback"
    }


def _write_goal_code_inventory_fixture(root: Path):
    code_root = root / "artifacts" / "code"
    runtime_file = code_root / "module" / "runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("VALUE = 1\n", encoding="utf-8")
    revision = "a" * 40
    marker = code_root / ".source-revision"
    marker.write_bytes(f"{revision}\n".encode("ascii"))
    records = {
        "artifacts/code/.source-revision": {
            "sha256": adapter.sha256_file(marker),
            "size": marker.stat().st_size,
        },
        "artifacts/code/module/runtime.py": {
            "sha256": adapter.sha256_file(runtime_file),
            "size": runtime_file.stat().st_size,
        },
    }
    manifest = launch._seal(
        {
            "schema_version": launch.CODE_MANIFEST_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "code_revision": revision,
            "code_root_relative": "artifacts/code",
            "revision_marker": "artifacts/code/.source-revision",
            "files": records,
            "code_inventory": records,
            "code_inventory_sha256": goal.canonical_sha256(records),
            "staged_path_rule": "bundle_root/<code_inventory_key>",
            "source_checkout_mutated": False,
            "remote_git_checkout_required": False,
            "scheduler_project_code_included": False,
        }
    )
    manifest_path = root / "code_manifest.json"
    launch._atomic_json(manifest_path, manifest)
    return code_root, runtime_file, marker, manifest_path, manifest


def test_checkout_free_goal_code_inventory_rehashes_and_fails_closed(tmp_path):
    code_root, runtime_file, _marker, manifest_path, manifest = (
        _write_goal_code_inventory_fixture(tmp_path / "valid")
    )
    evidence = preflight.authenticate_goal_code_inventory(
        code_root=code_root,
        code_manifest_path=manifest_path,
        expected_manifest_payload_sha256=manifest["payload_sha256"],
        expected_code_inventory_sha256=manifest["code_inventory_sha256"],
        expected_code_revision=manifest["code_revision"],
    )
    assert evidence["authentication_mode"] == "sealed_goal_code_inventory"
    assert evidence["git_checkout_required"] is False
    assert evidence["verified_file_count"] == 2
    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        preflight.authenticate_goal_code_inventory(
            code_root=code_root,
            code_manifest_path=manifest_path,
            expected_manifest_payload_sha256="0" * 64,
            expected_code_inventory_sha256=manifest["code_inventory_sha256"],
            expected_code_revision=manifest["code_revision"],
        )

    runtime_file.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="code file authentication failed"):
        preflight.authenticate_goal_code_inventory(
            code_root=code_root,
            code_manifest_path=manifest_path,
            expected_manifest_payload_sha256=manifest["payload_sha256"],
            expected_code_inventory_sha256=manifest["code_inventory_sha256"],
            expected_code_revision=manifest["code_revision"],
        )

    code_root, _runtime_file, _marker, manifest_path, manifest = (
        _write_goal_code_inventory_fixture(tmp_path / "extra")
    )
    (code_root / "module" / "shadow.py").write_text(
        "RAISE = True\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="inventory is not exact"):
        preflight.authenticate_goal_code_inventory(
            code_root=code_root,
            code_manifest_path=manifest_path,
            expected_manifest_payload_sha256=manifest["payload_sha256"],
            expected_code_inventory_sha256=manifest["code_inventory_sha256"],
            expected_code_revision=manifest["code_revision"],
        )

    code_root, _runtime_file, marker, manifest_path, manifest = (
        _write_goal_code_inventory_fixture(tmp_path / "marker")
    )
    marker.write_bytes(f"{'b' * 40}\n".encode("ascii"))
    with pytest.raises(RuntimeError, match="code file authentication failed"):
        preflight.authenticate_goal_code_inventory(
            code_root=code_root,
            code_manifest_path=manifest_path,
            expected_manifest_payload_sha256=manifest["payload_sha256"],
            expected_code_inventory_sha256=manifest["code_inventory_sha256"],
            expected_code_revision=manifest["code_revision"],
        )


def test_worker_relocation_requires_six_roles_and_bound_code_manifest(tmp_path):
    bundle_path, tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle",
        seeds=(201, 202, 203, 204),
    )
    task = tasks[0]
    runtime_paths = {
        name: str((tmp_path / "worker" / name).resolve())
        for name in launch.RUNTIME_SOURCE_ROLES
    }
    relocation = launch._seal(
        {
            "schema_version": launch.RELOCATION_SCHEMA,
            "task_payload_sha256": task["payload_sha256"],
            "paths": runtime_paths,
            "code_manifest_path": str(
                bundle_path.parent / "code_manifest.json"
            ),
            "source_absolute_paths_are_documentary_only": True,
            "remote_git_checkout_required": False,
        }
    )
    relocation_path = tmp_path / "relocation.json"
    launch._atomic_json(relocation_path, relocation)
    resolved, code_manifest_path = launch.resolve_worker_source(
        task, relocation_path
    )
    assert {
        name: resolved[name] for name in launch.RUNTIME_SOURCE_ROLES
    } == runtime_paths
    assert resolved["expected_code_revision"] == "c" * 40
    assert code_manifest_path == bundle_path.parent / "code_manifest.json"
    assert task["source"]["generation"].startswith("Z:")

    forged = copy.deepcopy(relocation)
    forged.pop("payload_sha256")
    forged["paths"].pop("dataset")
    launch._atomic_json(relocation_path, launch._seal(forged))
    with pytest.raises(RuntimeError, match="relocation contract mismatch"):
        launch.resolve_worker_source(task, relocation_path)


def test_global_pareto_rejects_results_outside_original_task_ledger(tmp_path):
    bundle_path, tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle-a",
        seeds=(301, 302, 303, 304),
    )
    _other_bundle, other_tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle-b",
        seeds=(401, 402, 403, 404),
    )
    results = [
        _write_synthetic_goal_seed_result(
            tmp_path / f"seed-a-{task['seed']}",
            task=task,
        )
        for task in tasks
    ]
    results[0] = _write_synthetic_goal_seed_result(
        tmp_path / "self-sealed-foreign-result",
        task=other_tasks[0],
    )
    with pytest.raises(RuntimeError, match="not in the original task ledger"):
        launch.aggregate_results(
            result_paths=results,
            bundle_manifest_path=bundle_path,
            output=tmp_path / "forged-global",
            minimum_seeds=4,
        )

    valid_results = [
        _write_synthetic_goal_seed_result(
            tmp_path / f"seed-valid-{task['seed']}",
            task=task,
        )
        for task in tasks
    ]
    with pytest.raises(RuntimeError, match="one result for every"):
        launch.aggregate_results(
            result_paths=valid_results[:-1],
            bundle_manifest_path=bundle_path,
            output=tmp_path / "partial-global",
            minimum_seeds=3,
        )


def test_global_pareto_rejects_tampered_generation_and_escaped_artifact(
    tmp_path,
):
    bundle_path, tasks = _write_synthetic_goal_bundle(
        tmp_path / "bundle",
        seeds=(501, 502, 503, 504),
    )
    results = [
        _write_synthetic_goal_seed_result(
            tmp_path / f"seed-{task['seed']}",
            task=task,
        )
        for task in tasks
    ]
    first = json.loads(results[0].read_text(encoding="utf-8"))
    first.pop("payload_sha256")
    first["completed_generations"] = launch.GENERATIONS
    launch._atomic_json(results[0], launch._seal(first))
    with pytest.raises(RuntimeError, match="seed result contract mismatch"):
        launch.aggregate_results(
            result_paths=results,
            bundle_manifest_path=bundle_path,
            output=tmp_path / "short-global",
            minimum_seeds=4,
        )

    first.pop("completed_generations")
    first["completed_generations"] = (
        launch.EXPECTED_ALGORITHM_N_GEN_COUNTER + 1
    )
    launch._atomic_json(results[0], launch._seal(first))
    with pytest.raises(RuntimeError, match="seed result contract mismatch"):
        launch.aggregate_results(
            result_paths=results,
            bundle_manifest_path=bundle_path,
            output=tmp_path / "overrun-global",
            minimum_seeds=4,
        )

    first.pop("completed_generations")
    first["completed_generations"] = (
        launch.EXPECTED_ALGORITHM_N_GEN_COUNTER
    )
    launch._atomic_json(results[0], launch._seal(first))
    second = json.loads(results[1].read_text(encoding="utf-8"))
    second.pop("payload_sha256")
    second["artifact_inventory"]["terminal_physical_candidates"][
        "path"
    ] = "../terminal_physical_candidates.csv"
    second["artifact_inventory_sha256"] = goal.canonical_sha256(
        second["artifact_inventory"]
    )
    launch._atomic_json(results[1], launch._seal(second))
    with pytest.raises(RuntimeError, match="path is unsafe"):
        launch.aggregate_results(
            result_paths=results,
            bundle_manifest_path=bundle_path,
            output=tmp_path / "escaped-global",
            minimum_seeds=4,
        )


def test_bundle_task_ledger_requires_all_four_primary_turn_strata(tmp_path):
    bundle_path, _tasks = _write_synthetic_goal_bundle(
        tmp_path / "single-stratum-bundle",
        seeds=(601, 602, 603, 604),
        turns=(5, 5, 5, 5),
    )
    with pytest.raises(RuntimeError, match="lacks all N1 strata"):
        launch.load_bundle_task_ledger(bundle_path)
