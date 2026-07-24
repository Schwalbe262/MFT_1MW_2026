import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from module.input_parameter_260706 import (
    create_input_parameter,
    get_drawing_default_params,
    validation_check,
)
from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY,
    FIXED_OPERATING_IDENTITY,
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    GOAL_TEMPERATURE_TARGETS,
)
from regression_260707.verify import scheduler_client
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_fea_handoff as handoff
from tools import mft_runtime_license_snapshot as runtime_license
from tools import tier1_corrected_generation_adapter as adapter
from tools import tier1_corrected_generation_preflight as preflight


def _decoded_params(primary_turns=6):
    params = get_drawing_default_params()
    params.update(FIXED_OPERATING_IDENTITY)
    params.update(
        {
            key: value
            for key, value in FIXED_COOLING_IDENTITY.items()
            if key != "thermal_pad_conductivity_W_mK"
        }
    )
    params["n_core_group"] = 5
    valid, frame, errors = validation_check(
        create_input_parameter(params),
        strict=False,
        return_errors=True,
    )
    assert valid, errors
    decoded = preflight._jsonable_decoded_parameters(frame.iloc[0])
    decoded["N1_main"] = int(primary_turns)
    decoded["N1_side"] = 0
    # Synthetic terminal evidence uses the exact goal constraint vector as
    # authority. Keep the mocked final FEA row independently gate-feasible.
    for key in (
        "cc_w2c_space_x",
        "cc_w2c_space_y",
        "w2c_w1c_space_x",
        "w2c_w1c_space_y",
        "w1c_w2s_gap_x_actual",
        "w1s_cs_space_x",
        "cs_w1s_space_y",
        "h_gap2",
        "w2s_w1s_space_x",
        "w1s_w2s_space_y",
    ):
        decoded[key] = 45.0
    return decoded


def _bundle(tmp_path: Path):
    local = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "c" * 64,
            "candidate_sha256": "e" * 64,
            "quality_status_sha256": "f" * 64,
            "code": {"revision": "1" * 40},
            "search_only_proposal": False,
        }
    )
    assignments = launch.seed_assignments(
        mode="rolling32",
        seed_start=2607269001,
    )
    bundle, tasks, _scheduler = launch.build_bundle_values(
        local_preflight=local,
        assignments=assignments,
        output_root=tmp_path,
        source={
            "generation": "G0",
            "candidate": "candidate.json",
            "quality_status": "quality.json",
            "code_root": "repo",
            "expected_code_revision": "1" * 40,
        },
    )
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle_path = bundle_root / "bundle_manifest.json"
    launch._atomic_json(bundle_path, bundle)
    task_paths = {}
    for task, relative in zip(tasks, bundle["task_relative_paths"]):
        task_path = bundle_root / relative
        task_path.parent.mkdir(parents=True, exist_ok=True)
        launch._atomic_json(task_path, task)
        task_paths[task["payload_sha256"]] = task_path
    return bundle, bundle_path, tasks, task_paths


def _write_seed_result(tmp_path: Path, task, decoded):
    geometry = {
        name: decoded[name]
        for name in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
    }
    geometry_sha = handoff.canonical_sha256(geometry)
    decoded_sha = handoff.canonical_sha256(decoded)
    constraints = list(preflight.GOAL_CONSTRAINT_NAMES)
    physical = {name: -1.0 for name in constraints}
    normalized = {name: -0.5 for name in constraints}
    rows = []
    for index in range(launch.POPULATION):
        row = {
            "terminal_population_index": index,
            "decoder_valid": True,
            "surrogate_physical_valid": True,
            "surrogate_physicality_passed": True,
            "physical_constraint_feasible": True,
            "physical_feasible": True,
            "physical_geometry_sha256": geometry_sha,
            "canonical_physical_params_sha256": decoded_sha,
            "candidate_physics_sha": geometry_sha,
            "objective_volume_L": 500.0 + task["fixed_primary_turns"],
            "objective_total_loss_W": (
                1000.0 - 10.0 * task["fixed_primary_turns"]
            ),
            "physical_G_json": json.dumps(
                physical, sort_keys=True, separators=(",", ":")
            ),
            "normalized_G_json": json.dumps(
                normalized, sort_keys=True, separators=(",", ":")
            ),
            "coordinate_unit_json": "[]",
            "decoded_physical_params_json": json.dumps(
                decoded, sort_keys=True, separators=(",", ":")
            ),
            "source_seed": task["seed"],
            "source_task_id": f"scheduler-task-{task['seed']}",
            "source_bundle_id": task["payload_sha256"],
            "source_island_id": f"n1-{task['fixed_primary_turns']}",
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "constraint_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "cooling_contract_sha256": (
                launch.FIXED_COOLING_IDENTITY_SHA256
            ),
            "operating_point_sha256": (
                launch.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "evaluation_model_artifacts_sha256": "b" * 64,
            "evaluation_model_generation_sha256": "c" * 64,
            "evaluation_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "evaluation_temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "evaluation_hard_constraint_contract_sha256": "d" * 64,
        }
        row.update({f"physical_G:{name}": -1.0 for name in constraints})
        row.update({f"normalized_G:{name}": -0.5 for name in constraints})
        rows.append(row)
    table = pd.DataFrame(rows)
    terminal_path = tmp_path / "terminal_physical_candidates.csv"
    tmp_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(terminal_path, index=False)
    terminal_manifest = launch._seal(
        {
            "schema_version": preflight.GOAL_TERMINAL_TABLE_SCHEMA,
            "goal_contract_required": True,
            "row_count": launch.POPULATION,
            "terminal_population_index_min": 0,
            "terminal_population_index_max": launch.POPULATION - 1,
            "columns": list(table.columns),
            "required_identity_columns": list(table.columns),
            "csv": {
                "path": terminal_path.name,
                "sha256": adapter.sha256_file(terminal_path),
                "size_bytes": terminal_path.stat().st_size,
            },
            "source_identity": {"seed": task["seed"]},
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": "d" * 64,
            "one_row_per_terminal_individual": True,
            "physical_deduplication_key": "physical_geometry_sha256",
            "global_pareto_provenance_ready": True,
        }
    )
    terminal_manifest_path = (
        tmp_path / "terminal_physical_candidates.manifest.json"
    )
    launch._atomic_json(terminal_manifest_path, terminal_manifest)
    inventory = {
        "terminal_physical_candidates": {
            "path": terminal_path.name,
            "sha256": adapter.sha256_file(terminal_path),
            "size_bytes": terminal_path.stat().st_size,
        },
        "terminal_physical_candidates_manifest": {
            "path": terminal_manifest_path.name,
            "sha256": adapter.sha256_file(terminal_manifest_path),
            "size_bytes": terminal_manifest_path.stat().st_size,
        },
    }
    result = launch._seal(
        {
            "schema_version": launch.SEARCH_RESULT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "task_payload_sha256": task["payload_sha256"],
            "seed": task["seed"],
            "fixed_primary_turns": task["fixed_primary_turns"],
            "population": launch.POPULATION,
            "generations": launch.GENERATIONS,
            "evaluated_generations": launch.GENERATIONS,
            "completed_generations": launch.GENERATIONS,
            "stage_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "constraint_version": "mft-goal-20260726",
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "operating_point_sha256": (
                launch.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "cooling_contract_sha256": (
                launch.FIXED_COOLING_IDENTITY_SHA256
            ),
            "constraint_names": constraints,
            "temperature_targets": list(GOAL_TEMPERATURE_TARGETS),
            "terminal_population_count": launch.POPULATION,
            "physical_feasible_count": launch.POPULATION,
            "feasible_pareto_count": 1,
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": handoff.canonical_sha256(inventory),
            "terminal_physical_candidates_manifest": terminal_manifest,
            "legacy_current7_stage_or_release_identity_reused": False,
            "search_only_proposal": False,
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        }
    )
    result_path = tmp_path / "result.json"
    launch._atomic_json(result_path, result)
    return result_path, geometry_sha


def _fixture(tmp_path: Path):
    bundle, bundle_path, tasks, task_paths = _bundle(tmp_path)
    decoded_by_turns = {
        turns: _decoded_params(turns) for turns in range(5, 9)
    }
    result_paths = []
    geometry_by_turns = {}
    result_by_payload = {}
    for task in tasks:
        result_path, geometry_sha = _write_seed_result(
            tmp_path
            / "seed-results"
            / f"seed-{task['seed']}-n1-{task['fixed_primary_turns']}",
            task,
            decoded_by_turns[task["fixed_primary_turns"]],
        )
        result_paths.append(result_path)
        geometry_by_turns[task["fixed_primary_turns"]] = geometry_sha
        result_by_payload[task["payload_sha256"]] = result_path
    aggregate_path = launch.aggregate_results(
        result_paths=result_paths,
        output=tmp_path / "aggregate",
        minimum_seeds=launch.ROLLING_SEED_COUNT,
    )
    selected_task = next(
        task for task in tasks if task["fixed_primary_turns"] == 6
    )
    global_path = aggregate_path.parent / "global_pareto_front.csv"
    return {
        "task": selected_task,
        "task_path": task_paths[selected_task["payload_sha256"]],
        "bundle_path": bundle_path,
        "result_path": result_by_payload[selected_task["payload_sha256"]],
        "aggregate_path": aggregate_path,
        "global_path": global_path,
        "geometry_sha": geometry_by_turns[6],
        "decoded": decoded_by_turns[6],
    }


def _make_plan(tmp_path: Path, fixture):
    return handoff.create_plan(
        aggregate_manifest_path=fixture["aggregate_path"],
        bundle_manifest_path=fixture["bundle_path"],
        standard_candidates_path=None,
        standard_candidates_sha256=None,
        candidate_physics_sha256=fixture["geometry_sha"],
        source_result_path=fixture["result_path"],
        task_payload_path=fixture["task_path"],
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "plan",
    )


def _result(plan_path: Path, stage: str, submission, *, hot=False):
    plan, params, selected = handoff._load_plan(plan_path)
    profile = handoff._read_json(
        plan_path.parent / plan["profiles"][stage]["path"]
    )
    runtime_license_sha = "8" * 64 if stage == "full" else ""
    runtime_auth = (
        handoff._core_auth(
            plan["solver_revision"],
            16,
            license_contract=handoff.FULL_LICENSE_CONTRACT,
            license_snapshot_sha256=runtime_license_sha,
        )
        if stage == "full"
        else submission["core_policy"]["auth_sha256"]
    )
    result = {
        **selected["row_contract"]["decoded_params"],
        **handoff._effective_params(params, profile),
        "git_hash": plan["solver_revision"],
        "git_dirty": 0,
        "pyaedt_library_git_hash": plan["library_revision"],
        "pyaedt_library_git_dirty": 0,
        "project_name": f"mock-{stage}-project",
        "Llt": 13.75 if stage == "standard" else 27.5,
        "B_max_core": 1.0,
        "f_res_min_tx_rx_only_Hz": 15000.0,
        "thermal_pad_conductivity_W_mK": 0.2,
        "P_winding_total": 3.0,
        "P_Tx_main_group": 1.0,
        "P_Rx_main_group": 1.0,
        "P_Rx_side_total": 1.0,
        "P_core_total": 1.0,
        "P_core_plate_total": 1.0,
        "P_wcp_total": 1.0,
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": submission["core_policy"]["contract"],
        "solver_core_opt_in": 1,
        "solver_core_backend": "standalone",
        "solver_num_cores_requested": submission["resources"]["cpus"],
        "solver_num_cores_effective": submission["resources"]["cpus"],
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": submission["resources"]["cpus"],
        "solver_core_slurm_cpus_per_task_readback": str(
            submission["resources"]["cpus"]
        ),
        "solver_core_scheduler_task_id_readback": str(submission["task_id"]),
        "solver_core_slurm_job_id_readback": "81234",
        "solver_core_auth_sha256": runtime_auth,
        "solver_matrix_hpc_num_cores_readback": submission["resources"]["cpus"],
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "9" * 64,
        "solver_core_license_contract": (
            handoff.FULL_LICENSE_CONTRACT if stage == "full" else ""
        ),
        "solver_core_license_snapshot_sha256": runtime_license_sha,
        "solver_core_license_checked_at_readback": (
            "2026-07-24T12:00:00+00:00" if stage == "full" else ""
        ),
        "solver_core_license_snapshot_age_seconds_readback": (
            1.0 if stage == "full" else ""
        ),
        "solver_core_license_headroom_readback_json": (
            json.dumps(
                {
                    "anshpc": 48,
                    "elec_solve_maxwell": 3,
                    "electronics_desktop": 3,
                    "electronics3d_gui": 3,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if stage == "full"
            else "{}"
        ),
        **{
            target: (
                101.0
                if hot and target == "T_max_Tx"
                else 99.0
                if "core" not in target
                else 119.0
            )
            for target in GOAL_TEMPERATURE_TARGETS
        },
    }
    return result


class _FakeScheduler:
    RESULT_VALID = scheduler_client.RESULT_VALID

    def __init__(self):
        self.calls = []
        self.result = None
        self.next_id = 70001

    def submit_verification(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        task_id = self.next_id
        self.next_id += 1
        return task_id

    def get_status(self, _task_id):
        return "completed"

    def fetch_result(self, *_args, **_kwargs):
        return SimpleNamespace(state=self.RESULT_VALID, result=self.result)

    @staticmethod
    def result_matches_params(result, params, required_keys=None):
        return scheduler_client.result_matches_params(
            result, params, required_keys=required_keys
        )


def _remote_reader(submission, result, artifact_bytes):
    retained = submission["retained_aedt"]
    marker_payload = {
        **retained["marker_contract"],
        "created_at": "2026-07-24T12:00:00Z",
    }
    marker = handoff._json_bytes(marker_payload)
    receipt = {
        "schema_version": handoff.REMOTE_RECEIPT_SCHEMA,
        "stage": submission["stage"],
        "dedupe_key": retained["dedupe_key"],
        "parameter_digest": retained["parameter_digest"],
        "solver_revision": submission["solver_revision"],
        "library_revision": submission["library_revision"],
        "profile_sha256": submission["profile_sha256"],
        "artifact_path": retained["artifact_path"],
        "marker_path": retained["marker_path"],
        "retention_required": True,
        "prune_protection_required": True,
        "scheduler_cleanup_exclusion_required": True,
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_size_bytes": len(artifact_bytes),
        "marker_sha256": hashlib.sha256(marker).hexdigest(),
        "marker_contract_sha256": retained["marker_contract_sha256"],
        "transport_schema_version": retained["transport"]["schema_version"],
        "transport_encoding": retained["transport"]["encoding"],
        "transport_chunk_directory": retained["transport"]["chunk_directory"],
        "transport_raw_chunk_bytes": retained["transport"]["raw_chunk_bytes"],
        "transport_max_encoded_chunk_bytes": retained["transport"][
            "max_encoded_chunk_bytes"
        ],
        "transport_chunk_count": int(
            np.ceil(
                len(artifact_bytes)
                / retained["transport"]["raw_chunk_bytes"]
            )
        ),
        "source_project_filename": f"{result['project_name']}.aedt",
        "source_project_name": result["project_name"],
    }
    receipt_bytes = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()

    def read(**kwargs):
        if kwargs["relative_path"] == retained["receipt_path"]:
            return receipt_bytes
        if kwargs["relative_path"] == retained["marker_path"]:
            return marker
        raise AssertionError(kwargs)

    return read


def _license_snapshot(path: Path):
    value = {
        "schema": "mft-aedt-license-headroom-snapshot-v1",
        "server_up": True,
        "server": "1055@172.16.10.81",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "features": {
            "anshpc": {"total": 64, "used": 0},
            "elec_solve_maxwell": {"total": 4, "used": 0},
            "electronics_desktop": {"total": 4, "used": 0},
            "electronics3d_gui": {"total": 4, "used": 0},
        },
    }
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    return path


def test_plan_authenticates_global_row_source_result_task_and_profiles(tmp_path):
    fixture = _fixture(tmp_path)
    plan_path = _make_plan(tmp_path, fixture)
    plan, params, selected = handoff._load_plan(plan_path)

    assert plan["hard_spec"] == GOAL_STAGE_SPEC
    assert plan["candidate_physics_sha256"] == fixture["geometry_sha"]
    assert set(params) == set(handoff.ALL_INPUT_KEYS)
    assert selected["source_result_identity"]["task_payload_sha256"] == (
        fixture["task"]["payload_sha256"]
    )
    authority = selected["selection_source"]["aggregate_authority"]
    assert authority["seed_count"] == 32
    assert authority["minimum_seed_count"] == 32
    assert authority["all_bundle_seed_results_reauthenticated"] is True
    assert authority["global_nds_recomputed"] is True
    assert plan["stages"]["standard"]["resources"] == {
        "cpus": 8,
        "timeout_seconds": 14400,
    }
    assert plan["stages"]["full"]["resources"] == {
        "cpus": 16,
        "timeout_seconds": 43200,
    }
    for stage in ("standard", "full"):
        profile = handoff._read_json(
            plan_path.parent / plan["profiles"][stage]["path"]
        )
        assert profile["param_overrides"]["keep_project"] == 1
        assert profile["fixed_boundary_contract"][
            "thermal_pad_conductivity_W_mK"
        ] == 0.2
        assert plan["stages"][stage]["retained_aedt"][
            "prune_protection_required"
        ] is True
        assert plan["stages"][stage]["retained_aedt"]["marker_path"].endswith(
            "/.slurm-scheduler-preserve.json"
        )
        assert plan["stages"][stage]["retained_aedt"]["marker_contract"][
            "schema"
        ] == "slurm-scheduler-prune-protection-v1"


def test_plan_accepts_authenticated_standard_candidates_bytes(tmp_path):
    fixture = _fixture(tmp_path)
    plan_path = handoff.create_plan(
        aggregate_manifest_path=None,
        bundle_manifest_path=None,
        standard_candidates_path=fixture["global_path"],
        standard_candidates_sha256=adapter.sha256_file(
            fixture["global_path"]
        ),
        candidate_physics_sha256=fixture["geometry_sha"],
        source_result_path=fixture["result_path"],
        task_payload_path=fixture["task_path"],
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "plan-standard-bytes",
    )
    plan, _params, selected = handoff._load_plan(plan_path)
    assert selected["selection_source"]["kind"] == (
        "authenticated_standard_candidates_bytes"
    )
    assert plan["scheduler_submission_performed"] is False
    assert plan["submission_eligible"] is False
    with pytest.raises(
        handoff.HandoffContractError, match="inspect-only"
    ):
        handoff.submit_stage(
            plan_path=plan_path,
            stage="standard",
            output=tmp_path / "forbidden_submission.json",
            scheduler=_FakeScheduler(),
        )


def test_submit_reauthenticates_sources_after_joint_plan_reseal(tmp_path):
    fixture = _fixture(tmp_path)
    plan_path = _make_plan(tmp_path, fixture)
    plan = handoff._read_json(plan_path)
    selected_path = plan_path.parent / plan["selected_candidate"]["path"]
    selected = handoff._read_json(selected_path)

    selected_unsigned = dict(selected)
    selected_unsigned.pop("payload_sha256")
    selected_unsigned["selected_row"] = dict(selected_unsigned["selected_row"])
    selected_unsigned["selected_row"]["source_seed"] = 999999
    resealed_selected = handoff._seal(selected_unsigned)
    selected_path.chmod(0o666)
    selected_path.write_bytes(handoff._json_bytes(resealed_selected))
    selected_path.chmod(0o444)

    plan_unsigned = dict(plan)
    plan_unsigned.pop("payload_sha256")
    plan_unsigned["selected_candidate"] = handoff._file_record(selected_path)
    plan_unsigned["search_authority_sha256"] = handoff.canonical_sha256(
        handoff._selected_authentication_payload(resealed_selected)
    )
    plan_path.chmod(0o666)
    plan_path.write_bytes(handoff._json_bytes(handoff._seal(plan_unsigned)))
    plan_path.chmod(0o444)

    fake = _FakeScheduler()
    with pytest.raises(
        handoff.HandoffContractError,
        match="freshly authenticated search authority",
    ):
        handoff.submit_stage(
            plan_path=plan_path,
            stage="standard",
            output=tmp_path / "forbidden_resealed_submission.json",
            scheduler=fake,
        )
    assert fake.calls == []


def test_plan_fails_if_aggregate_pareto_bytes_drift(tmp_path):
    fixture = _fixture(tmp_path)
    frame = pd.read_csv(fixture["global_path"])
    frame.loc[0, "objective_total_loss_W"] += 1
    frame.to_csv(fixture["global_path"], index=False)

    with pytest.raises(
        handoff.HandoffContractError,
        match="aggregate artifact bytes drifted: global_pareto_front",
    ):
        _make_plan(tmp_path, fixture)


def test_plan_fails_if_aggregate_omits_a_bundle_seed(tmp_path):
    fixture = _fixture(tmp_path)
    aggregate = handoff._read_json(fixture["aggregate_path"])
    unsigned = dict(aggregate)
    unsigned.pop("payload_sha256")
    unsigned["inputs"] = unsigned["inputs"][:-1]
    unsigned["seed_count"] -= 1
    unsigned["seeds"] = unsigned["seeds"][:-1]
    launch._atomic_json(fixture["aggregate_path"], launch._seal(unsigned))
    with pytest.raises(
        handoff.HandoffContractError,
        match="aggregate seed inventory|omits one or more bundle",
    ):
        _make_plan(tmp_path, fixture)


def test_mocked_standard_gate_full_and_package_flow(tmp_path):
    fixture = _fixture(tmp_path)
    plan_path = _make_plan(tmp_path, fixture)
    fake = _FakeScheduler()

    standard_submission_path = handoff.submit_stage(
        plan_path=plan_path,
        stage="standard",
        output=tmp_path / "standard_submission.json",
        scheduler=fake,
    )
    standard_call = fake.calls[-1]
    assert standard_call[1]["cpus"] == 8
    assert standard_call[1]["aedt_backend"] == "standalone"
    assert standard_call[1]["submission_env"][
        "MFT_STANDALONE_CORE_COUNT"
    ] == "8"
    assert standard_call[0][3]["timeout_seconds"] == 14400
    assert standard_call[0][3]["param_overrides"]["keep_project"] == 1
    with pytest.raises(
        handoff.HandoffContractError, match="Scheduler origin differs"
    ):
        handoff.collect_stage(
            plan_path=plan_path,
            submission_path=standard_submission_path,
            output=tmp_path / "wrong_origin_collection.json",
            scheduler_url="http://other-scheduler.invalid",
            scheduler=fake,
        )

    standard_submission = handoff._read_json(standard_submission_path)
    standard_result = _result(
        plan_path, "standard", standard_submission
    )
    fake.result = standard_result
    symmetric_bytes = b"mock symmetric AEDT bytes"
    standard_collection_path = handoff.collect_stage(
        plan_path=plan_path,
        submission_path=standard_submission_path,
        output=tmp_path / "standard_collection.json",
        scheduler=fake,
        remote_reader=_remote_reader(
            standard_submission, standard_result, symmetric_bytes
        ),
    )
    gate_path, passed = handoff.create_standard_gate(
        plan_path=plan_path,
        standard_collection_path=standard_collection_path,
        output=tmp_path / "standard_gate.json",
    )
    assert passed is True

    full_submission_path = handoff.submit_stage(
        plan_path=plan_path,
        stage="full",
        output=tmp_path / "full_submission.json",
        standard_gate_path=gate_path,
        standard_collection_path=standard_collection_path,
        license_snapshot_path=_license_snapshot(
            tmp_path / "license_snapshot.json"
        ),
        scheduler=fake,
    )
    full_call = fake.calls[-1]
    assert full_call[1]["cpus"] == 16
    assert full_call[0][3]["timeout_seconds"] == 43200
    assert full_call[1]["submission_env"][
        "MFT_STANDALONE_CORE_COUNT"
    ] == "16"
    assert full_call[1]["submission_env"][
        scheduler_client.RUNTIME_LICENSE_REFRESH_ENV
    ] == "1"
    assert "MFT_STANDALONE_CORE_LICENSE_SNAPSHOT_JSON" not in (
        full_call[1]["submission_env"]
    )
    assert full_call[0][3]["param_overrides"]["full_model"] == 1
    assert full_call[0][3]["param_overrides"]["keep_project"] == 1

    full_submission = handoff._read_json(full_submission_path)
    assert full_submission["preceding_standard_authority"][
        "gate_payload_sha256"
    ] == handoff._read_json(gate_path)["payload_sha256"]
    full_result = _result(plan_path, "full", full_submission)
    fake.result = full_result
    full_bytes = b"mock full AEDT bytes"
    full_collection_path = handoff.collect_stage(
        plan_path=plan_path,
        submission_path=full_submission_path,
        output=tmp_path / "full_collection.json",
        scheduler=fake,
        remote_reader=_remote_reader(
            full_submission, full_result, full_bytes
        ),
    )
    payloads = {
        hashlib.sha256(symmetric_bytes).hexdigest(): symmetric_bytes,
        hashlib.sha256(full_bytes).hexdigest(): full_bytes,
    }

    def fetcher(**kwargs):
        payload = payloads[kwargs["expected_sha256"]]
        assert len(payload) == kwargs["expected_size"]
        kwargs["destination"].write_bytes(payload)

    package_path = handoff.package_results(
        plan_path=plan_path,
        standard_collection_path=standard_collection_path,
        standard_gate_path=gate_path,
        full_collection_path=full_collection_path,
        output=tmp_path / "package",
        remote_fetcher=fetcher,
    )
    package = handoff._validate_seal(
        handoff._read_json(package_path), handoff.PACKAGE_SCHEMA
    )
    assert (package_path.parent / "symmetric.aedt").read_bytes() == (
        symmetric_bytes
    )
    assert (package_path.parent / "full.aedt").read_bytes() == full_bytes
    assert package["prune_protection_marker_verified_for_both"] is True
    assert set(package["prune_protection_markers"]) == {"standard", "full"}
    for marker in package["prune_protection_markers"].values():
        assert marker["schema"] == "slurm-scheduler-prune-protection-v1"
        assert marker["source_remote_path"].endswith(
            "/.slurm-scheduler-preserve.json"
        )
        assert len(marker["sha256"]) == 64
        assert len(marker["contract_sha256"]) == 64
    assert package["source_remote_artifacts_must_not_be_pruned_before_package"]
    assert package["result_identities"]["standard"]["full_model"] == 0
    assert package["result_identities"]["full"]["full_model"] == 1
    assert package["standard_to_full_transition"][
        "authenticated_standard_pass_preceded_full_submission"
    ] is True
    assert package["scheduler_repository_modified"] is False
    assert package["package_fea_execution_self_contained"] is True
    assert {
        "fea_params",
        "standard_profile",
        "full_profile",
        "standard_submission",
        "full_submission",
    }.issubset(package["evidence"])


def test_hot_standard_gate_blocks_full_submission(tmp_path):
    fixture = _fixture(tmp_path)
    plan_path = _make_plan(tmp_path, fixture)
    fake = _FakeScheduler()
    submission_path = handoff.submit_stage(
        plan_path=plan_path,
        stage="standard",
        output=tmp_path / "standard_submission.json",
        scheduler=fake,
    )
    submission = handoff._read_json(submission_path)
    hot_result = _result(plan_path, "standard", submission, hot=True)
    fake.result = hot_result
    collection_path = handoff.collect_stage(
        plan_path=plan_path,
        submission_path=submission_path,
        output=tmp_path / "hot_collection.json",
        scheduler=fake,
        remote_reader=_remote_reader(submission, hot_result, b"hot project"),
    )
    gate_path, passed = handoff.create_standard_gate(
        plan_path=plan_path,
        standard_collection_path=collection_path,
        output=tmp_path / "hot_gate.json",
    )
    assert passed is False
    before = len(fake.calls)
    with pytest.raises(
        handoff.HandoffContractError,
        match="authenticated Standard PASS gate is absent",
    ):
        handoff.submit_stage(
            plan_path=plan_path,
            stage="full",
            output=tmp_path / "forbidden_full_submission.json",
            standard_gate_path=gate_path,
            standard_collection_path=collection_path,
            license_snapshot_path=_license_snapshot(
                tmp_path / "license_snapshot.json"
            ),
            scheduler=fake,
        )
    assert len(fake.calls) == before


def test_scheduler_retention_contract_is_opt_in_and_dedupe_scoped():
    profile, _record = handoff._profile_content("standard")
    retained = scheduler_client.retained_aedt_identity(
        "mft-goal-standard-test",
        {"x": 1},
        profile,
        "2" * 40,
        "3" * 40,
    )
    assert retained["artifact_path"].endswith("/symmetric.aedt")
    assert retained["marker_path"].endswith("/.slurm-scheduler-preserve.json")
    assert retained["marker_contract"]["schema"] == (
        "slurm-scheduler-prune-protection-v1"
    )
    assert retained["marker_contract"]["preserve"] is True
    command = scheduler_client._retained_aedt_export_command(retained)
    assert "find \"$MFT_WORKDIR\" -type f -name '*.aedt'" in command
    assert "MFT_TASK_ROOT" in command
    assert "source_project_name" in command
    assert ".aedt.chunks" in command
    assert "base64.b64encode" in command
    assert scheduler_client.retained_aedt_identity(
        "legacy",
        {"x": 1},
        {"param_overrides": {}},
        "2" * 40,
        "3" * 40,
    ) is None


def test_profiles_are_exactly_pinned_and_reject_cli_or_physics_drift():
    profile, _record = handoff._profile_content("standard")
    cli_drift = copy.deepcopy(profile)
    cli_drift["cli_flags"] = "--model-only; touch /tmp/forbidden"
    with pytest.raises(handoff.HandoffContractError, match="profile contract"):
        handoff._validate_profile("standard", cli_drift)
    physics_drift = copy.deepcopy(profile)
    physics_drift["param_overrides"]["thermal_on"] = 0
    with pytest.raises(handoff.HandoffContractError, match="profile contract"):
        handoff._validate_profile("standard", physics_drift)
    with pytest.raises(ValueError, match="reviewed immutable profile"):
        scheduler_client.retained_aedt_identity(
            "drift",
            {"x": 1},
            cli_drift,
            "2" * 40,
            "3" * 40,
        )


def test_binary_aedt_is_reconstructed_from_text_safe_base64_chunks(
    tmp_path, monkeypatch
):
    payload = bytes(range(256)) * 3001
    raw_chunk_bytes = scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
    chunks = [
        payload[index : index + raw_chunk_bytes]
        for index in range(0, len(payload), raw_chunk_bytes)
    ]
    encoded = {
        f"goal/chunks/{index:08d}.b64": __import__("base64").b64encode(chunk)
        for index, chunk in enumerate(chunks)
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(request, timeout):
        assert timeout == 120.0
        query = handoff.urllib.parse.parse_qs(
            handoff.urllib.parse.urlparse(request.full_url).query
        )
        assert int(query["max_bytes"][0]) == 1024 * 1024
        return Response(encoded[query["path"][0]])

    monkeypatch.setattr(handoff.urllib.request, "urlopen", urlopen)
    destination = tmp_path / "reconstructed.aedt"
    handoff._fetch_remote_to_path(
        scheduler_url="http://scheduler.invalid",
        task_id=70001,
        relative_path="goal/full.aedt",
        transport_chunk_directory="goal/chunks",
        transport_raw_chunk_bytes=raw_chunk_bytes,
        transport_max_encoded_chunk_bytes=(
            scheduler_client.RETAINED_AEDT_MAX_ENCODED_CHUNK_BYTES
        ),
        transport_chunk_count=len(chunks),
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        destination=destination,
    )
    assert destination.read_bytes() == payload


def test_retained_export_script_writes_scheduler_marker_receipt_and_chunks(
    tmp_path,
):
    profile, _record = handoff._profile_content("standard")
    retained = scheduler_client.retained_aedt_identity(
        "export-test",
        {"x": 1},
        profile,
        "2" * 40,
        "3" * 40,
    )
    command = scheduler_client._retained_aedt_export_command(retained)
    tokens = shlex.split(command)
    index = tokens.index("-c")
    script = tokens[index + 1]
    marker_contract_json = tokens[index + 7]
    receipt_context_json = tokens[index + 8]
    source = tmp_path / "mock-project.aedt"
    source.write_bytes(bytes(range(256)) * 4000)
    destination = tmp_path / retained["artifact_path"]
    receipt = tmp_path / retained["receipt_path"]
    marker = tmp_path / retained["marker_path"]
    chunk_dir = tmp_path / retained["transport"]["chunk_directory"]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            str(destination),
            str(receipt),
            str(marker),
            str(chunk_dir),
            marker_contract_json,
            receipt_context_json,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    assert marker.name == ".slurm-scheduler-preserve.json"
    assert set(marker_value) == {
        "schema",
        "preserve",
        "created_at",
        "reason",
        "owner",
    }
    assert marker_value["schema"] == "slurm-scheduler-prune-protection-v1"
    assert receipt_value["artifact_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert receipt_value["marker_sha256"] == hashlib.sha256(
        marker.read_bytes()
    ).hexdigest()
    assert len(list(chunk_dir.glob("*.b64"))) == receipt_value[
        "transport_chunk_count"
    ]


def test_runtime_license_snapshot_is_compute_start_bound(
    tmp_path, monkeypatch
):
    lmutil = tmp_path / "lmutil"
    lmutil.write_text("mock", encoding="utf-8")
    lmutil.chmod(0o755)
    lmstat = "\n".join(
        [
            "Users of anshpc:  (Total of 64 licenses issued; "
            "Total of 8 licenses in use)",
            "Users of elec_solve_maxwell:  (Total of 4 licenses issued; "
            "Total of 1 license in use)",
            "Users of electronics_desktop:  (Total of 4 licenses issued; "
            "Total of 1 license in use)",
            "Users of electronics3d_gui:  (Total of 4 licenses issued; "
            "Total of 1 license in use)",
        ]
    )
    monkeypatch.setattr(
        runtime_license.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=lmstat
        ),
    )
    output = tmp_path / "runtime-license"
    captured = runtime_license.capture(
        output_dir=output,
        solver_revision="2" * 40,
        lmutil=str(lmutil),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    snapshot_json = (output / "snapshot.json").read_text(encoding="utf-8")
    assert hashlib.sha256(snapshot_json.encode()).hexdigest() == (
        captured["snapshot_sha256"]
    )
    assert captured["auth_sha256"] == handoff._core_auth(
        "2" * 40,
        16,
        license_contract=handoff.FULL_LICENSE_CONTRACT,
        license_snapshot_sha256=captured["snapshot_sha256"],
    )
    full_profile, _record = handoff._profile_content("full")
    retained = scheduler_client.retained_aedt_identity(
        "full-runtime",
        {"x": 1},
        full_profile,
        "2" * 40,
        "3" * 40,
    )
    refresh = scheduler_client._runtime_license_refresh_command(
        retained,
        {scheduler_client.RUNTIME_LICENSE_REFRESH_ENV: "1"},
        "2" * 40,
    )
    assert "mft_runtime_license_snapshot.py" in refresh
    assert "snapshot.json" in refresh
    assert "MFT_STANDALONE_CORE_AUTH_SHA256" in refresh


def test_admission_license_snapshot_rejects_unknown_shell_bearing_fields(
    tmp_path,
):
    path = _license_snapshot(tmp_path / "license.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["untrusted"] = "$(touch /tmp/forbidden)"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        handoff.HandoffContractError, match="identity drifted"
    ):
        handoff._validate_license_snapshot(path)
