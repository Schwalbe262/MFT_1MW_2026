import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.recover_full_postsolve_project import (
    PRESERVER_IDENTITY_SCHEMA,
    PRESERVER_MANIFEST_SCHEMA,
    TERMINAL_WATCH_SCHEMA,
    _canonical_sha256,
    _exact_solved_design,
    _overlay_single_row,
    _sha256,
    _validate_thermal_mesh_result,
    _validate_thermal_pad_result,
    attest_recovery_runtime,
    attest_recovery_solver_core_policy,
    attest_staged_thermal_wrapper,
    attest_preserved_seal,
    attest_terminal_watch,
    deterministic_full_loss_object_groups,
    discover_preserved_project,
    hydrate_sealed_loss_evidence,
    hydrate_full_loss_object_groups,
    load_sealed_source_result,
    prepare_staged_thermal_rebuild,
    recover,
    seal_fresh_thermal_project,
    stage_preserved_project,
    validate_full_loss_object_groups,
)


def _fake_thermal_mesh_fields():
    object_operation_readbacks = [
        {
            "name": f"mesh_{index}",
            "operation_type": "object_level",
            "assignment_count": 1,
            "assignment_sha256": "c" * 64,
            "level": 5,
            "separate_objects": True,
        }
        for index in range(29)
    ]
    region_operation_readbacks = [
        {
            "name": f"wcp_pad_mesh_region_{index}",
            "operation_type": "mesh_region",
            "assignment_count": 1,
            "assignment_sha256": "f" * 64,
            "level": 5,
            "separate_objects": None,
            "region_object_count": 1,
        }
        for index in range(8)
    ]
    operation_readbacks = (
        object_operation_readbacks + region_operation_readbacks
    )
    artifact = {
        "directory": "icepak_thermal.results/DV1_Meshes0_V0.sd",
        "name": "DV1_Meshes0_V0.sd",
        "grid_output_size": 100,
        "grid_output_sha256_sample": "d" * 64,
        "grid_mapping_size": 50,
        "grid_mapping_sha256_sample": "e" * 64,
        "mtime_ns": 1,
    }
    mapping_readbacks = [{
        "schema": "thermal-grid-mapping-readback-v1",
        "region_name": "GlobalRegion",
        "parent_region": 0,
        "has_mesh": True,
        "object_count": 56,
        "objects_with_domains": [
            f"required_object_{index}" for index in range(56)
        ],
        "object_domain_value_counts": {
            f"required_object_{index}": 1 for index in range(56)
        },
        "overlapping_mr_face_count": 0,
        "artifact_name": "DV1_Meshes0_V0.sd",
        "grid_mapping_size": 50,
        "grid_mapping_sha256_sample": "e" * 64,
    }] + [
        {
            "schema": "thermal-grid-mapping-readback-v1",
            "region_name": f"wcp_pad_mesh_region_{index}",
            "parent_region": 6,
            "has_mesh": True,
            "object_count": 1,
            "objects_with_domains": [f"wcp_pad_{index}"],
            "object_domain_value_counts": {f"wcp_pad_{index}": 1},
            "overlapping_mr_face_count": 1,
            "artifact_name": f"DV1_Meshes{index}_V0.sd",
            "grid_mapping_size": 50,
            "grid_mapping_sha256_sample": "e" * 64,
        }
        for index in range(1, 9)
    ]
    preflight = {
        "schema": "thermal-mesh-preflight-v2",
        "passed": True,
        "static_contract_passed": True,
        "generate_mesh_returned": True,
        "message_scan_complete": True,
        "analysis_dispatched_after_premesh": True,
        "native_operation_readback_passed": True,
        "mesh_artifact_readback_passed": True,
        "mesh_mapping_coverage_passed": True,
        "mesh_mapping_coverage": {
            "schema": "thermal-grid-mapping-coverage-v1",
            "passed": True,
            "fresh_grid_mapping_count": 9,
            "parse_errors": [],
            "global_region_count": 1,
            "required_object_count": 56,
            "mapped_required_object_count": 56,
            "required_objects_missing": [],
            "expected_local_region_count": 8,
            "missing_local_regions": [],
            "local_regions_without_mesh": [],
            "uncoupled_local_regions": [],
            "local_region_objects_missing": [],
            "readbacks": mapping_readbacks,
        },
        "postflight_identity_passed": True,
        "standalone_idle_barrier_passed": True,
        "required_objects_missing": [],
        "unmeshed_objects": [],
        "native_errors": [],
        "mesh_policy": "b3-rxmain-l5-wcp-pad-padded-regions-v1",
        "mesh_plan_sha256": "b" * 64,
        "fresh_mesh_artifact_count": 9,
        "fresh_mesh_artifacts": [artifact],
        "mesh_operation_count": 37,
        "mesh_assigned_object_count": 93,
        "object_level_operation_count": 29,
        "mesh_region_operation_count": 8,
        "wcp_pad_mesh_region_count": 8,
        "core_plate_assembly_count": 15,
        "wcp_assembly_count": 4,
        "rx_retained_pack_count": 3,
        "native_operation_readback": {
            "missing_operation_names": [],
            "required_thin_objects_missing": [],
            "expected_operation_count": 37,
            "assigned_object_count": 93,
            "required_thin_object_count": 56,
            "object_level_operation_count": 29,
            "mesh_region_operation_count": 8,
            "mesh_region_part_readback_passed": True,
            "operation_readbacks": operation_readbacks,
        },
        "standalone_idle_barrier": {
            "passed": True,
            "last_running": False,
            "idle_observations": 2,
            "stable_artifact_observations": 2,
            "fresh_mesh_artifacts": [artifact],
        },
    }
    return {
        "thermal_mesh_policy": "b3-rxmain-l5-wcp-pad-padded-regions-v1",
        "thermal_mesh_plan_contract_version": "thermal-mesh-plan-v4",
        "thermal_mesh_plan_sha256": "b" * 64,
        "thermal_mesh_operation_count": 37,
        "thermal_mesh_assigned_object_count": 93,
        "thermal_mesh_required_thin_object_count": 56,
        "thermal_mesh_shared_operation_count": 0,
        "thermal_mesh_separate_object_operation_count": 29,
        "thermal_mesh_object_level_operation_count": 29,
        "thermal_mesh_region_operation_count": 8,
        "thermal_mesh_wcp_pad_region_count": 8,
        "thermal_mesh_core_plate_assembly_count": 15,
        "thermal_mesh_wcp_assembly_count": 4,
        "thermal_mesh_rx_retained_pack_count": 3,
        "thermal_mesh_preflight_contract_version": (
            "thermal-mesh-preflight-v2"
        ),
        "thermal_mesh_preflight_status": (
            "passed_standalone_native_premesh"
        ),
        "thermal_mesh_native_generation_passed": 1,
        "thermal_mesh_unmeshed_object_count": 0,
        "thermal_mesh_unmeshed_objects_json": "[]",
        "thermal_mesh_preflight_json": json.dumps(
            preflight, sort_keys=True, separators=(",", ":")
        ),
        "thermal_mesh_postsolve_probe_object_count": 9,
        "thermal_mesh_postsolve_probe_missing_count": 0,
        "thermal_mesh_postsolve_probe_missing_objects_json": "[]",
        "thermal_mesh_postsolve_probe_complete": 1,
    }


def _fake_validated_parameter_row():
    return {
        "full_model": 1,
        "matrix_on": 1,
        "cap_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "matrix_percent_error": 0.5,
        "matrix_min_converged": 1,
        "percent_error": 0.5,
        "min_converged": 1,
        "cap_percent_error": 0.5,
        "n_core_group": 1,
        "n_explicit_turns": 2,
        "N1_main": 1,
        "N1_side": 0,
        "N2_main": 1,
        "N2_side": 0,
        "core_plate_on": 0,
        "core_plate_pad_t": 2.0,
        "wcp_on": 0,
        "wcp_pad_t": 2.0,
        "fan_velocity": 1.5,
        "k_ins": 0.2,
    }


def test_validates_candidate_exact_thermal_mesh_release_evidence():
    frame = pd.DataFrame([_fake_thermal_mesh_fields()])

    evidence = _validate_thermal_mesh_result(frame)

    assert evidence["counts"][
        "thermal_mesh_operation_count"
    ] == 37
    assert evidence["counts"][
        "thermal_mesh_assigned_object_count"
    ] == 93
    assert evidence["counts"][
        "thermal_mesh_required_thin_object_count"
    ] == 56
    assert evidence["counts"][
        "thermal_mesh_region_operation_count"
    ] == 8
    assert evidence["counts"][
        "thermal_mesh_postsolve_probe_object_count"
    ] == 9
    assert evidence["postsolve_prior_nine_probe_complete"] is True


def test_rejects_missing_prior_nine_postsolve_mesh_probe():
    fields = _fake_thermal_mesh_fields()
    fields["thermal_mesh_postsolve_probe_missing_count"] = 1
    fields["thermal_mesh_postsolve_probe_complete"] = 0
    fields["thermal_mesh_postsolve_probe_missing_objects_json"] = json.dumps(
        ["Tx_main_wcp_pad_1_in_p"]
    )

    with pytest.raises(
        RuntimeError,
        match="rebuilt thermal result integer contract mismatch",
    ):
        _validate_thermal_mesh_result(pd.DataFrame([fields]))


def test_rejects_native_thermal_mesh_error_evidence():
    fields = _fake_thermal_mesh_fields()
    preflight = json.loads(fields["thermal_mesh_preflight_json"])
    preflight["native_errors"] = ["Failed to run solver"]
    fields["thermal_mesh_preflight_json"] = json.dumps(preflight)

    with pytest.raises(RuntimeError, match="preflight contract mismatch"):
        _validate_thermal_mesh_result(pd.DataFrame([fields]))


def _fake_source_result_row(project_name="simulation_123_456"):
    return {
        **_fake_validated_parameter_row(),
        "physics_data_revision": "physics-revision",
        "project_name": project_name,
        "saved_at": "2026-07-24 11:09:21",
        "git_hash": "146142be579e2f3e45f12961214018a4e28445c5",
        "git_dirty": 0,
        "pyaedt_library_git_hash": (
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
        ),
        "pyaedt_library_git_dirty": 0,
        "Ltx": 100.0,
        "Lrx": 25.0,
        "M": 45.0,
        "k": 0.9,
        "Lmt": 81.0,
        "Lmr": 20.25,
        "Llt": 19.0,
        "Llr": 4.75,
        "conv_passes_matrix": 3,
        "conv_consecutive_matrix": 2,
        "conv_error_pct_matrix": 0.1,
        "conv_delta_pct_matrix": 0.1,
        "C_tx_tx_F": 1e-9,
        "C_rx_rx_F": 2e-9,
        "C_tx_rx_F": 0.5e-9,
        "f_res_tx_self_Hz": 100_000.0,
        "f_res_rx_self_Hz": 200_000.0,
        "f_res_interwinding_Hz": 50_000.0,
        "conv_passes_cap": 3,
        "conv_consecutive_cap": 2,
        "conv_error_pct_cap": 0.1,
        "conv_delta_pct_cap": 0.1,
        "P_core_total": 100.0,
        "P_core_plate_total": 0.0,
        "P_wcp_total": 0.0,
        "P_winding_total": 250.0,
        "B_mean_core": 0.5,
        "B_max_core": 0.8,
        "P_core_1": 100.0,
        "P_Tx_main_group": 200.0,
        "P_Rx_main_group": 50.0,
        "P_turn_Tx_main_0_0": 200.0,
        "P_turn_Rx_main_0_0": 50.0,
        "conv_passes_loss": 3,
        "conv_consecutive_loss": 2,
        "conv_error_pct_loss": 0.1,
        "conv_delta_pct_loss": 0.1,
        "result_valid_em": 1,
        "em_validity_reason": "valid",
        "result_valid_thermal": 0,
        "matrix_solve_attempts": 1,
        "cap_solve_attempts": 1,
        "loss_solve_attempts": 1,
        "matrix_extraction_backend": "export_rl_matrix",
        "cap_extraction_backend": "export_c_matrix",
        "loss_extraction_backend": "get_solution_data_per_variation",
        "thermal_solve_attempts": 1,
        "thermal_analyze_call_ok": 1,
        "thermal_dispatch_status": "success",
        "thermal_convergence_available": 1,
        "thermal_converged": 1,
        "thermal_iterations": 65,
        "thermal_solution_data_available": 1,
        "thermal_solved": 0,
        "thermal_extraction_complete": 0,
        "thermal_extraction_failure_reason": (
            "required_volume_temperature_missing"
        ),
        "thermal_pad_conductivity_W_mK": 3.0,
        "thermal_pad_material_policy": (
            "deadline_tim_k3_native_attested_3WmK_"
            "electrically_insulating_v2"
        ),
        "thermal_pad_native_readback_contract_version": (
            "thermal-pad-native-material-readback-v1"
        ),
        "thermal_pad_native_readback_attested": 1,
        "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
        "thermal_pad_native_electrical_conductivity_S_m": 0.0,
    }


def test_discovers_one_adjacent_preserved_project_pair(tmp_path):
    project = tmp_path / "current" / "repo" / "simulation" / "sim"
    project.mkdir(parents=True)
    aedt = project / "sim.aedt"
    aedt.write_bytes(b"project")
    results = project / "sim.aedtresults"
    results.mkdir()
    (results / "fields.dat").write_bytes(b"fields")

    assert discover_preserved_project(tmp_path) == (
        aedt.resolve(),
        results.resolve(),
    )


def test_discovery_rejects_project_without_results_sidecar(tmp_path):
    (tmp_path / "orphan.aedt").write_bytes(b"project")

    with pytest.raises(RuntimeError, match="found 0"):
        discover_preserved_project(tmp_path)


def test_discovery_filters_exact_project_name(tmp_path):
    for name in ("sim_a", "sim_b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f"{name}.aedt").write_bytes(b"project")
        (folder / f"{name}.aedtresults").mkdir()

    aedt, results = discover_preserved_project(tmp_path, "sim_b")

    assert aedt.name == "sim_b.aedt"
    assert results.name == "sim_b.aedtresults"


def _write_sealed_current(
    tmp_path,
    termination_reason,
    *,
    replica_of_task_id=0,
    source_rows=None,
):
    candidate_digest = "a" * 64
    profile_sha = "b" * 64
    source_payload = {
        "fidelity": "actual_full_geometry",
        "complete_pipeline": True,
        "matrix_enabled": True,
        "capacitance_enabled": True,
        "loss_enabled": True,
        "thermal_enabled": True,
        "n_explicit_turns": 2,
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutation_allowed": False,
        "physical_candidate_digest": candidate_digest,
        "followup_profile_sha256": profile_sha,
    }
    if replica_of_task_id:
        source_payload.update({
            "replica_of_task_id": int(replica_of_task_id),
            "replica_execution_sha256": "9" * 64,
        })
    identity = {
        "schema_version": PRESERVER_IDENTITY_SCHEMA,
        "source_task_id": 123,
        "source_account_name": "account",
        "source_node_name": "n100",
        "source_allocation_id": 456,
        "source_slurm_job_id": "789",
        "source_workdir": "/enroot/mft_campaign-b428-hit-full-test",
        "source_command_sha256": "c" * 64,
        "source_payload_sha256": _canonical_sha256(source_payload),
        "source_payload": source_payload,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    stable_binding = {
        "schema_version": "mft-deadline-full-preserver-binding-v1",
        "source_task_id": 123,
        "source_account_name": identity["source_account_name"],
        "source_node_name": identity["source_node_name"],
        "source_allocation_id": identity["source_allocation_id"],
        "source_slurm_job_id": identity["source_slurm_job_id"],
        "source_workdir": identity["source_workdir"],
        "source_command_sha256": identity["source_command_sha256"],
        "source_payload_sha256": identity["source_payload_sha256"],
    }
    token = _canonical_sha256(stable_binding)[:16]
    gpfs_root = tmp_path / f"mft_full_recovery_task_123_{token}"
    current = gpfs_root / "current"
    project_dir = current / "repo" / "simulation" / "simulation_123_456"
    project_dir.mkdir(parents=True)
    aedt = project_dir / "simulation_123_456.aedt"
    aedt.write_bytes(b"project")
    results = project_dir / "simulation_123_456.aedtresults"
    results.mkdir()
    (results / "fields.bin").write_bytes(b"fields")
    params = current / "repo" / "cand.json"
    params.write_text('{"full_model":1}', encoding="utf-8")
    source_result = current / "repo" / "simulation_results_260706.csv"
    pd.DataFrame(
        source_rows
        if source_rows is not None
        else [_fake_source_result_row()]
    ).to_csv(
        source_result, index=False
    )

    identity_path = gpfs_root / "source_identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    files = []
    for path in sorted(current.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(current).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    project_relative = aedt.relative_to(current).as_posix()
    results_relative = results.relative_to(current).as_posix()
    manifest = {
        "schema_version": PRESERVER_MANIFEST_SCHEMA,
        "source_task_id": 123,
        "source_identity_sha256": identity["identity_sha256"],
        "termination_reason": termination_reason,
        "current_directory": str(current.resolve()),
        "aedt_and_aedtresults_adjacent_atomic_publish": True,
        "last_sync": {
            "project_paths": [project_relative],
            "aedtresults_paths": [results_relative],
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "source_task_mutated": False,
        "source_task_cancelled": False,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = gpfs_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(
        gpfs_root=gpfs_root,
        current=current,
        manifest=manifest,
        manifest_path=manifest_path,
        identity_path=identity_path,
        aedt=aedt,
        results=results,
        params=params,
        source_result=source_result,
        params_sha=_sha256(params),
        params_canonical_sha=_canonical_sha256({"full_model": 1}),
        candidate_digest=candidate_digest,
        profile_sha=profile_sha,
    )


def _attest(seal):
    return attest_preserved_seal(
        mirror_root=seal.current,
        manifest_path=seal.manifest_path,
        source_identity_path=seal.identity_path,
        params_path=seal.params,
        project_name="simulation_123_456",
        expected_source_task_id=123,
        expected_manifest_sha256=seal.manifest["manifest_sha256"],
        expected_source_identity_sha256=(
            json.loads(seal.identity_path.read_text())["identity_sha256"]
        ),
        expected_params_sha256=seal.params_sha,
        expected_params_canonical_sha256=seal.params_canonical_sha,
        expected_candidate_digest=seal.candidate_digest,
        expected_profile_sha256=seal.profile_sha,
    )


def _write_terminal_watch(
    tmp_path, seal, *, source_status="failed", sidecar_exit_code=0
):
    sealed = _attest(seal)
    identity = json.loads(seal.identity_path.read_text(encoding="utf-8"))
    project_relative = sealed["project_relative_path"]
    results_relative = str(
        Path(project_relative).with_name(
            "simulation_123_456.aedtresults"
        ).as_posix()
    )
    validation = {
        name: True
        for name in (
            "source_terminal",
            "sidecar_completed_zero",
            "one_preserver_result",
            "zero_preserver_fatal",
            "execution_id_bound",
            "manifest_id_bound",
            "source_identity_id_bound",
            "current_is_atomic_leaf",
            "manifest_valid",
            "source_identity_valid",
            "project_and_results_adjacent",
            "params_in_manifest",
            "params_canonical_bound",
            "all_pass",
        )
    }
    source_finished_at = "2026-07-24T09:59:00+09:00"
    sidecar_finished_at = "2026-07-24T09:59:30+09:00"

    def write_status(
        filename, task_id, name, status, exit_code, finished_at
    ):
        record = {
            "id": task_id,
            "task_id": task_id,
            "name": name,
            "status": status,
            "state": (
                "succeeded" if status == "completed" else status
            ),
            "exit_code": exit_code,
            "finished_at": finished_at,
            "started_at": "2026-07-24T09:00:00+09:00",
            "account_name": "account",
            "requested_account_name": "account",
            "slurm_job_id": "789",
            "allocation_id": 456,
            "assigned_allocation": 456,
            "node_name": "n100",
            "allocation_node_name": "n100",
            "actual_node_name": "n100",
            "dedupe_key": f"dedupe-{task_id}",
        }
        path = tmp_path / filename
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "path": filename,
            "file_sha256": _sha256(path),
            "canonical_sha256": _canonical_sha256(record),
        }

    source_exit_code = 1 if source_status == "failed" else 0
    source_status_evidence = write_status(
        "source-status.json",
        123,
        "source-task",
        source_status,
        source_exit_code,
        source_finished_at,
    )
    sidecar_status_evidence = write_status(
        "sidecar-status.json",
        124,
        "preserver-sidecar",
        "completed",
        sidecar_exit_code,
        sidecar_finished_at,
    )
    payload = {
        "schema_version": TERMINAL_WATCH_SCHEMA,
        "captured_at": "2026-07-24T10:00:00+09:00",
        "execution": {
            "source_task_id": 123,
            "source_status": source_status,
            "source_exit_code": source_exit_code,
            "source_finished_at": source_finished_at,
            "source_status_evidence": source_status_evidence,
            "sidecar_task_id": 124,
            "sidecar_status": "completed",
            "sidecar_exit_code": sidecar_exit_code,
            "sidecar_finished_at": sidecar_finished_at,
            "sidecar_status_evidence": sidecar_status_evidence,
        },
        "seal": {
            "gpfs_root": str(seal.gpfs_root.resolve()),
            "current_directory": str(seal.current.resolve()),
            "manifest": {
                "path": str(seal.manifest_path.resolve()),
                "file_sha256": _sha256(seal.manifest_path),
                "canonical_sha256": sealed["manifest_sha256"],
                "recorded_manifest_sha256": sealed["manifest_sha256"],
            },
            "source_identity": {
                "path": str(seal.identity_path.resolve()),
                "file_sha256": _sha256(seal.identity_path),
                "canonical_sha256": sealed["source_identity_sha256"],
                "recorded_identity_sha256": sealed[
                    "source_identity_sha256"
                ],
            },
            "file_count": sealed["file_count"],
            "project": {
                "stem": "simulation_123_456",
                "relative_path": project_relative,
                "absolute_path": str(
                    (seal.current / project_relative).resolve()
                ),
                "aedtresults_relative_path": results_relative,
                "aedtresults_absolute_path": str(
                    (seal.current / results_relative).resolve()
                ),
            },
            "params": {
                "relative_path": "repo/cand.json",
                "absolute_path": str(seal.params.resolve()),
                "raw_sha256": sealed["params_sha256"],
                "canonical_sha256": sealed["params_canonical_sha256"],
            },
        },
        "identity": {
            "physical_candidate_digest": sealed["candidate_digest"],
            "followup_profile_sha256": sealed["profile_sha256"],
            "source_payload_sha256": identity["source_payload_sha256"],
            "replica_of_task_id": sealed["replica_of_task_id"],
            "replica_execution_sha256": sealed[
                "replica_execution_sha256"
            ],
        },
        "validation": validation,
        "source_task_mutated": False,
        "sidecar_task_mutated": False,
    }
    payload["terminal_watch_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "terminal-watch-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


@pytest.mark.parametrize(
    "termination_reason",
    ["source_exit_code_published", "source_workdir_disappeared"],
)
def test_attests_only_complete_terminal_preserver_publication(
    tmp_path, termination_reason
):
    seal = _write_sealed_current(tmp_path, termination_reason)

    evidence = _attest(seal)

    assert evidence["termination_reason"] == termination_reason
    assert evidence["project_relative_path"].endswith(
        "simulation_123_456.aedt"
    )
    assert evidence["params_sha256"] == seal.params_sha
    assert evidence["source_result_relative_path"] == (
        "repo/simulation_results_260706.csv"
    )
    assert evidence["source_result_sha256"] == _sha256(
        seal.source_result
    )


def _fake_bound_df_plus():
    frame = pd.DataFrame([_fake_validated_parameter_row()])
    frame["physics_data_revision"] = "physics-revision"
    return frame


def test_loads_one_manifest_bound_source_result_and_validates_all_stages(
    tmp_path,
):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    row, evidence = load_sealed_source_result(
        mirror_root=seal.current,
        seal=_attest(seal),
        project_name="simulation_123_456",
        df_plus=_fake_bound_df_plus(),
    )

    assert float(row["Llt"].iloc[0]) == 19.0
    assert evidence["project_row_count"] == 1
    assert evidence["em_validity_reason"] == "valid"
    assert evidence["cap_validity_reason"] == "valid"
    assert evidence["source_thermal_failure_reason"] == (
        "required_volume_temperature_missing"
    )
    assert evidence["source_historical_thermal_postsolve"] == {
        "available": False,
        "availability_reason": "source_columns_absent",
        "expected_column_names": [
            "thermal_mesh_postsolve_probe_object_count",
            "thermal_mesh_postsolve_probe_missing_count",
            "thermal_mesh_postsolve_probe_complete",
        ],
        "present_column_names": [],
        "counts": {},
    }

    rebuilt = pd.DataFrame([_fake_thermal_mesh_fields()])
    _validate_thermal_mesh_result(rebuilt)
    result = _overlay_single_row(row, rebuilt)
    assert int(
        result["thermal_mesh_postsolve_probe_missing_count"].iloc[0]
    ) == 0
    assert (
        "thermal_mesh_postsolve_probe_missing_count"
        not in row.columns
    )


def test_attests_optional_complete_legacy_source_postsolve_columns(
    tmp_path,
):
    source = _fake_source_result_row()
    source.update({
        "thermal_mesh_postsolve_probe_object_count": 9,
        "thermal_mesh_postsolve_probe_missing_count": 9,
        "thermal_mesh_postsolve_probe_complete": 0,
    })
    seal = _write_sealed_current(
        tmp_path,
        "source_exit_code_published",
        source_rows=[source],
    )

    _row, evidence = load_sealed_source_result(
        mirror_root=seal.current,
        seal=_attest(seal),
        project_name="simulation_123_456",
        df_plus=_fake_bound_df_plus(),
    )

    assert evidence["source_historical_thermal_postsolve"] == {
        "available": True,
        "availability_reason": "legacy_source_columns_attested",
        "expected_column_names": [
            "thermal_mesh_postsolve_probe_object_count",
            "thermal_mesh_postsolve_probe_missing_count",
            "thermal_mesh_postsolve_probe_complete",
        ],
        "present_column_names": [
            "thermal_mesh_postsolve_probe_object_count",
            "thermal_mesh_postsolve_probe_missing_count",
            "thermal_mesh_postsolve_probe_complete",
        ],
        "counts": {
            "thermal_mesh_postsolve_probe_object_count": 9,
            "thermal_mesh_postsolve_probe_missing_count": 9,
            "thermal_mesh_postsolve_probe_complete": 0,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        (
            lambda rows: rows[0].update(
                project_name="different_project"
            ),
            "exactly one project row",
        ),
        (
            lambda rows: rows.append(dict(rows[0])),
            "exactly one project row",
        ),
        (
            lambda rows: rows[0].update(result_valid_em=0),
            "integer contract mismatch",
        ),
        (
            lambda rows: rows[0].update(
                thermal_mesh_postsolve_probe_object_count=9,
                thermal_mesh_postsolve_probe_missing_count=8,
                thermal_mesh_postsolve_probe_complete=0,
            ),
            "sealed source result integer contract mismatch",
        ),
        (
            lambda rows: rows[0].update(
                thermal_mesh_postsolve_probe_missing_count=9
            ),
            "historical thermal postsolve columns are partially present",
        ),
        (
            lambda rows: rows[0].update(
                thermal_pad_native_thermal_conductivity_W_mK=0.2
            ),
            "thermal/TIM evidence contract failed",
        ),
        (
            lambda rows: rows[0].update(Llt=float("nan")),
            "solved EM evidence is invalid",
        ),
    ],
)
def test_rejects_unbound_or_invalid_source_result_rows(
    tmp_path, mutation, error_pattern
):
    rows = [_fake_source_result_row()]
    mutation(rows)
    seal = _write_sealed_current(
        tmp_path,
        "source_exit_code_published",
        source_rows=rows,
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        load_sealed_source_result(
            mirror_root=seal.current,
            seal=_attest(seal),
            project_name="simulation_123_456",
            df_plus=_fake_bound_df_plus(),
        )


def test_rejects_source_result_changed_after_manifest_attestation(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    evidence = _attest(seal)
    seal.source_result.write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after manifest"):
        load_sealed_source_result(
            mirror_root=seal.current,
            seal=evidence,
            project_name="simulation_123_456",
            df_plus=_fake_bound_df_plus(),
        )


def test_sealed_loss_hydration_checks_required_keys_and_power_balances():
    frame = pd.DataFrame([_fake_source_result_row()])
    groups = {
        "Tx_windings_main": ["Tx_main_0_0"],
        "Tx_windings_side": [],
        "Tx_windings_side2": [],
        "Rx_windings_main": ["Rx_main_0_0"],
        "Rx_windings_side": [],
        "Rx_windings_side2": [],
        "core_objs": [
            f"core_1_{piece}"
            for piece in (
                "leg_left",
                "leg_center",
                "leg_right",
                "yoke_bottom",
                "yoke_top",
            )
        ],
        "core_flux_sheets": ["core_flux_section_1"],
        "core_plates": [],
        "core_pads": [],
        "wcp_plates": [],
        "wcp_pads": [],
    }

    loss_map, evidence = hydrate_sealed_loss_evidence(
        frame, groups, _fake_bound_df_plus()
    )

    assert loss_map["P_core_1"] == 100.0
    assert loss_map["P_turn_Tx_main_0_0"] == 200.0
    assert evidence["required_key_count"] == 5
    assert evidence["core_group_sum_w"] == 100.0


def test_task_95074_actual_row_loss_preflight_uses_enumerated_wcp_names():
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "task_95074_loss_preflight.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["source_task_id"] == 95074
    assert fixture["terminal_snapshot_sha256"] == (
        "e67490a85eaf84f6b8ee84f42d51eddbf377d1c02c15d54ba92cbeb1929c2af9"
    )
    frame = pd.DataFrame([fixture["result"]])

    groups = deterministic_full_loss_object_groups(frame)
    loss_map, evidence = hydrate_sealed_loss_evidence(
        frame, groups, frame
    )

    assert groups["wcp_plates"] == fixture["expected"]["wcp_plates"]
    assert groups["wcp_pads"] == fixture["expected"]["wcp_pads"]
    assert sorted(
        key for key in loss_map if key.startswith("P_Tx_main_wcp_")
    ) == [
        "P_Tx_main_wcp_1_n",
        "P_Tx_main_wcp_1_p",
        "P_Tx_main_wcp_2_n",
        "P_Tx_main_wcp_2_p",
    ]
    for key, expected in fixture["expected"]["loss_evidence"].items():
        assert evidence[key] == expected


def _write_clean_runtime_repo(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Recovery Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "config",
            "user.email",
            "recovery@example.invalid",
        ],
        check=True,
    )
    (source / "base.txt").write_text("base\n", encoding="ascii")
    subprocess.run(["git", "-C", str(source), "add", "base.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "base"],
        check=True,
    )
    base = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    (source / "recovery.txt").write_text("recovery\n", encoding="ascii")
    subprocess.run(
        ["git", "-C", str(source), "add", "recovery.txt"], check=True
    )
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "recovery"],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return source, base, revision


def _write_clean_library_repo(tmp_path):
    library = tmp_path / "pyaedt_library"
    library.mkdir()
    subprocess.run(["git", "init", "-q", str(library)], check=True)
    subprocess.run(
        ["git", "-C", str(library), "config", "user.name", "Library Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(library),
            "config",
            "user.email",
            "library@example.invalid",
        ],
        check=True,
    )
    (library / "src").mkdir()
    (library / "src" / "module.py").write_text(
        "VALUE = 1\n", encoding="ascii"
    )
    subprocess.run(["git", "-C", str(library), "add", "src"], check=True)
    subprocess.run(
        ["git", "-C", str(library), "commit", "-q", "-m", "library"],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(library), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return library, revision


def test_recovery_runtime_attests_clean_bundle_checkout_and_library(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    source, base, revision = _write_clean_runtime_repo(tmp_path)
    bundle_sha = "b" * 64
    library, library_revision = _write_clean_library_repo(tmp_path)
    monkeypatch.setattr(module, "_SOURCE_RESULT_GIT_HASH", base)
    monkeypatch.setattr(
        module,
        "_SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(
        module,
        "PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(module, "PYAEDT_LIBRARY_GIT_DIRTY", 0)
    monkeypatch.setattr(module, "GIT_HASH", revision)
    monkeypatch.setattr(module, "GIT_DIRTY", 0)

    evidence = attest_recovery_runtime(
        recovery_revision=revision,
        bundle_sha256=bundle_sha,
        repo_root=source,
        environ={
            "MFT_FULL_RECOVERY_SOURCE_REVISION": revision,
            "MFT_FULL_RECOVERY_BUNDLE_SHA256": bundle_sha,
            "MFT_PYAEDT_LIBRARY_ROOT": str(library),
        },
    )

    assert evidence["recovery_revision"] == revision
    assert evidence["source_bundle_sha256"] == bundle_sha
    assert evidence["pyaedt_library_git_dirty"] == 0
    assert evidence["provenance_mode"] == "clean_git_bundle_checkout_and_sha256"


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        (
            "MFT_FULL_RECOVERY_SOURCE_REVISION",
            "c" * 40,
            "revision environment binding",
        ),
        (
            "MFT_FULL_RECOVERY_BUNDLE_SHA256",
            "d" * 64,
            "bundle SHA-256 environment binding",
        ),
        ("MFT_PYAEDT_LIBRARY_ROOT", "", "explicit library root"),
    ],
)
def test_recovery_runtime_rejects_unbound_inputs(
    tmp_path, monkeypatch, field, value, error_pattern
):
    import tools.recover_full_postsolve_project as module

    source, base, revision = _write_clean_runtime_repo(tmp_path)
    bundle_sha = "b" * 64
    library, library_revision = _write_clean_library_repo(tmp_path)
    monkeypatch.setattr(module, "_SOURCE_RESULT_GIT_HASH", base)
    monkeypatch.setattr(
        module,
        "_SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(
        module,
        "PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(module, "PYAEDT_LIBRARY_GIT_DIRTY", 0)
    monkeypatch.setattr(module, "GIT_HASH", revision)
    monkeypatch.setattr(module, "GIT_DIRTY", 0)
    environ = {
        "MFT_FULL_RECOVERY_SOURCE_REVISION": revision,
        "MFT_FULL_RECOVERY_BUNDLE_SHA256": bundle_sha,
        "MFT_PYAEDT_LIBRARY_ROOT": str(library),
    }
    environ[field] = value

    with pytest.raises(RuntimeError, match=error_pattern):
        attest_recovery_runtime(
            recovery_revision=revision,
            bundle_sha256=bundle_sha,
            repo_root=source,
            environ=environ,
        )


def test_recovery_runtime_rejects_dirty_solver_checkout(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    source, base, revision = _write_clean_runtime_repo(tmp_path)
    library, library_revision = _write_clean_library_repo(tmp_path)
    (source / "untracked.txt").write_text("dirty\n", encoding="ascii")
    monkeypatch.setattr(module, "_SOURCE_RESULT_GIT_HASH", base)
    monkeypatch.setattr(
        module,
        "_SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(module, "GIT_HASH", revision)
    monkeypatch.setattr(module, "GIT_DIRTY", 0)
    monkeypatch.setattr(
        module, "PYAEDT_LIBRARY_GIT_HASH", library_revision
    )
    monkeypatch.setattr(module, "PYAEDT_LIBRARY_GIT_DIRTY", 0)

    with pytest.raises(RuntimeError, match="exact clean recovery revision"):
        attest_recovery_runtime(
            recovery_revision=revision,
            bundle_sha256="b" * 64,
            repo_root=source,
            environ={
                "MFT_FULL_RECOVERY_SOURCE_REVISION": revision,
                "MFT_FULL_RECOVERY_BUNDLE_SHA256": "b" * 64,
                "MFT_PYAEDT_LIBRARY_ROOT": str(library),
            },
        )


@pytest.mark.parametrize("failure_mode", ["wrong_hash", "dirty"])
def test_recovery_runtime_rejects_wrong_or_dirty_library(
    tmp_path, monkeypatch, failure_mode
):
    import tools.recover_full_postsolve_project as module

    source, base, revision = _write_clean_runtime_repo(tmp_path)
    library, library_revision = _write_clean_library_repo(tmp_path)
    monkeypatch.setattr(module, "_SOURCE_RESULT_GIT_HASH", base)
    monkeypatch.setattr(
        module,
        "_SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH",
        library_revision,
    )
    monkeypatch.setattr(module, "GIT_HASH", revision)
    monkeypatch.setattr(module, "GIT_DIRTY", 0)
    monkeypatch.setattr(
        module,
        "PYAEDT_LIBRARY_GIT_HASH",
        "0" * 40 if failure_mode == "wrong_hash" else library_revision,
    )
    monkeypatch.setattr(
        module,
        "PYAEDT_LIBRARY_GIT_DIRTY",
        1 if failure_mode == "dirty" else 0,
    )
    if failure_mode == "dirty":
        (library / "untracked.txt").write_text("dirty\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="library provenance"):
        attest_recovery_runtime(
            recovery_revision=revision,
            bundle_sha256="b" * 64,
            repo_root=source,
            environ={
                "MFT_FULL_RECOVERY_SOURCE_REVISION": revision,
                "MFT_FULL_RECOVERY_BUNDLE_SHA256": "b" * 64,
                "MFT_PYAEDT_LIBRARY_ROOT": str(library),
            },
        )


def _exact_recovery_core_policy():
    import tools.recover_full_postsolve_project as module

    return {
        "schema": "mft-solver-core-policy-v1",
        "contract_version": "mft-standalone-core-16-optin-v1",
        "opt_in": True,
        "backend": "standalone",
        "requested_num_cores": 16,
        "effective_num_cores": 16,
        "num_tasks": 1,
        "affinity_count_readback": 16,
        "slurm_cpus_per_task_readback": 16,
        "scheduler_task_id_readback": 123,
        "slurm_job_id_readback": 456,
        "solver_revision": module.GIT_HASH,
        "solver_dirty": 0,
        "auth_sha256": "a" * 64,
        "license_contract": "mft-aedt-hpc-license-snapshot-v1",
        "license_snapshot_sha256": "b" * 64,
        "license_snapshot_age_seconds_readback": 1.0,
        "license_headroom_readback": {
            "anshpc": 16,
            "elec_solve_maxwell": 1,
            "electronics_desktop": 1,
            "electronics3d_gui": 1,
        },
    }


def test_recovery_solver_core_policy_requires_exact_sixteen_cores():
    policy = _exact_recovery_core_policy()
    sim = SimpleNamespace(
        solver_core_policy=policy,
        NUM_CORE=16,
        NUM_TASK=1,
    )

    assert attest_recovery_solver_core_policy(sim) == policy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "pooled"),
        ("effective_num_cores", 8),
        ("affinity_count_readback", 15),
        ("slurm_cpus_per_task_readback", 8),
        ("scheduler_task_id_readback", 0),
        ("scheduler_task_id_readback", False),
        ("slurm_job_id_readback", 0),
        ("slurm_job_id_readback", False),
        ("solver_revision", "0" * 40),
        ("solver_dirty", 1),
        ("license_snapshot_age_seconds_readback", 601.0),
    ],
)
def test_recovery_solver_core_policy_rejects_wrong_contract(field, value):
    policy = _exact_recovery_core_policy()
    policy[field] = value
    sim = SimpleNamespace(
        solver_core_policy=policy,
        NUM_CORE=16,
        NUM_TASK=1,
    )

    with pytest.raises(RuntimeError, match="core policy mismatch"):
        attest_recovery_solver_core_policy(sim)


def test_recovery_solver_core_policy_rejects_license_headroom_shortage():
    policy = _exact_recovery_core_policy()
    policy["license_headroom_readback"]["anshpc"] = 15
    sim = SimpleNamespace(
        solver_core_policy=policy,
        NUM_CORE=16,
        NUM_TASK=1,
    )

    with pytest.raises(RuntimeError, match="core policy mismatch"):
        attest_recovery_solver_core_policy(sim)


def test_recovery_solver_core_policy_rejects_native_core_mismatch():
    policy = _exact_recovery_core_policy()
    policy = {
        **policy,
    }
    sim = SimpleNamespace(
        solver_core_policy=policy,
        NUM_CORE=4,
        NUM_TASK=1,
    )

    with pytest.raises(RuntimeError, match="core policy mismatch"):
        attest_recovery_solver_core_policy(sim)


def test_fresh_thermal_project_is_atomically_sealed_with_inventory(tmp_path):
    source = tmp_path / "working" / "simulation_123_456"
    results = source / "simulation_123_456.aedtresults"
    results.mkdir(parents=True)
    (source / "simulation_123_456.aedt").write_bytes(b"aedt-project")
    (results / "fields.bin").write_bytes(b"fields")
    output = tmp_path / "durable-output"
    output.mkdir()

    seal = seal_fresh_thermal_project(
        source_dir=source,
        output_dir=output,
        project_name="simulation_123_456",
        require_gpfs=False,
    )

    target = output / "fresh-thermal-project"
    assert seal["status"] == "atomically_published_read_only"
    assert seal["atomic_publish"] is True
    assert seal["storage_root_contract"] == "test_non_gpfs"
    assert Path(seal["aedt"]) == target / "simulation_123_456.aedt"
    assert Path(seal["aedtresults"]) == (
        target / "simulation_123_456.aedtresults"
    )
    assert seal["file_count"] == 2
    assert seal["total_bytes"] == len(b"aedt-project") + len(b"fields")
    assert len(seal["tree_sha256"]) == 64
    assert len(seal["aedt_sha256"]) == 64
    assert not (output / ".incoming-fresh-thermal-project").exists()

    with pytest.raises(RuntimeError, match="already exists"):
        seal_fresh_thermal_project(
            source_dir=source,
            output_dir=output,
            project_name="simulation_123_456",
            require_gpfs=False,
        )


def test_fresh_thermal_project_rejects_non_gpfs_durable_output(tmp_path):
    source = tmp_path / "working" / "simulation_123_456"
    (source / "simulation_123_456.aedtresults").mkdir(parents=True)
    (source / "simulation_123_456.aedt").write_bytes(b"aedt-project")
    output = tmp_path / "not-gpfs"
    output.mkdir()

    with pytest.raises(RuntimeError, match="below /gpfs"):
        seal_fresh_thermal_project(
            source_dir=source,
            output_dir=output,
            project_name="simulation_123_456",
            require_gpfs=True,
        )


def test_sealed_loss_hydration_rejects_aggregate_mismatch():
    row = _fake_source_result_row()
    row["P_winding_total"] = 251.0

    with pytest.raises(RuntimeError, match="winding groups"):
        hydrate_sealed_loss_evidence(
            pd.DataFrame([row]),
            {
                "Tx_windings_main": ["Tx_main_0_0"],
                "Rx_windings_main": ["Rx_main_0_0"],
                "Rx_windings_side": [],
                "core_objs": [
                    f"core_1_{piece}"
                    for piece in (
                        "leg_left",
                        "leg_center",
                        "leg_right",
                        "yoke_bottom",
                        "yoke_top",
                    )
                ],
                "core_plates": [],
                "wcp_plates": [],
            },
            _fake_bound_df_plus(),
        )


def test_result_overlay_replaces_stale_thermal_fields_without_duplicates():
    result = _overlay_single_row(
        pd.DataFrame([{
            "Llt": 27.1,
            "thermal_solved": 0,
            "T_max_core": float("nan"),
        }]),
        pd.DataFrame([{
            "thermal_solved": 1,
            "T_max_core": 94.0,
        }]),
    )

    assert list(result.columns).count("thermal_solved") == 1
    assert int(result["thermal_solved"].iloc[0]) == 1
    assert float(result["T_max_core"].iloc[0]) == 94.0
    assert float(result["Llt"].iloc[0]) == 27.1


def test_attests_independent_source_and_sidecar_terminal_watch(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    path, payload = _write_terminal_watch(tmp_path, seal)
    sealed = _attest(seal)
    assert sealed["params_canonical_sha256"] != sealed["candidate_digest"]

    evidence = attest_terminal_watch(
        terminal_watch_path=path,
        expected_terminal_watch_sha256=payload[
            "terminal_watch_sha256"
        ],
        sealed_root=seal.gpfs_root,
        seal=sealed,
        project_name="simulation_123_456",
    )

    assert evidence["source_status"] == "failed"
    assert evidence["sidecar_status"] == "completed"
    assert evidence["sidecar_exit_code"] == 0


def test_rejects_terminal_watch_status_bundle_hash_drift(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    path, payload = _write_terminal_watch(tmp_path, seal)
    (tmp_path / "source-status.json").write_text(
        '{"changed":true}\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="status hash mismatch"):
        attest_terminal_watch(
            terminal_watch_path=path,
            expected_terminal_watch_sha256=payload[
                "terminal_watch_sha256"
            ],
            sealed_root=seal.gpfs_root,
            seal=_attest(seal),
            project_name="simulation_123_456",
        )


def test_attests_positive_replica_provenance_without_id_substitution(
    tmp_path,
):
    seal = _write_sealed_current(
        tmp_path,
        "source_exit_code_published",
        replica_of_task_id=99,
    )
    path, payload = _write_terminal_watch(tmp_path, seal)

    evidence = attest_terminal_watch(
        terminal_watch_path=path,
        expected_terminal_watch_sha256=payload[
            "terminal_watch_sha256"
        ],
        sealed_root=seal.gpfs_root,
        seal=_attest(seal),
        project_name="simulation_123_456",
    )

    assert evidence["replica_of_task_id"] == 99
    assert evidence["replica_execution_sha256"] == "9" * 64
    assert payload["execution"]["source_task_id"] == 123


def test_rejects_terminal_watch_before_sidecar_zero_exit(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    path, payload = _write_terminal_watch(
        tmp_path, seal, sidecar_exit_code=1
    )

    with pytest.raises(RuntimeError, match="terminal contract"):
        attest_terminal_watch(
            terminal_watch_path=path,
            expected_terminal_watch_sha256=payload[
                "terminal_watch_sha256"
            ],
            sealed_root=seal.gpfs_root,
            seal=_attest(seal),
            project_name="simulation_123_456",
        )


def test_rejects_watch_deadline_publication(tmp_path):
    seal = _write_sealed_current(tmp_path, "watch_deadline_reached")

    with pytest.raises(RuntimeError, match="not terminal"):
        _attest(seal)


def test_rejects_sealed_file_hash_drift(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    (seal.results / "fields.bin").write_bytes(b"FIELDS")

    with pytest.raises(RuntimeError, match="file hash/size mismatch"):
        _attest(seal)


def test_rejects_parameter_digest_not_bound_to_manifest(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_workdir_disappeared"
    )
    seal.params_sha = "c" * 64

    with pytest.raises(RuntimeError, match="parameter-file SHA-256"):
        _attest(seal)


def test_staging_is_fresh_and_does_not_modify_sealed_pair(tmp_path):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    work_dir = tmp_path / "writable-copy"

    staged_aedt, staged_results = stage_preserved_project(
        seal.aedt, seal.results, work_dir
    )

    assert staged_aedt.read_bytes() == seal.aedt.read_bytes()
    assert (staged_results / "fields.bin").read_bytes() == b"fields"
    assert seal.aedt.read_bytes() == b"project"
    with pytest.raises(RuntimeError, match="must not already exist"):
        stage_preserved_project(seal.aedt, seal.results, work_dir)


class _Modeler:
    def __init__(self, names):
        self.object_names = list(names)
        self._objects = {
            name: SimpleNamespace(name=name) for name in self.object_names
        }

    def get_object_from_name(self, name):
        return self._objects.get(name)


def test_hydrates_only_deterministic_full_loss_object_groups():
    design = SimpleNamespace(modeler=_Modeler([
        "Rx_main_10_0",
        "core_0_yoke_top",
        "Tx_main_1_0",
        "Rx_side2_0_0",
        "Tx_main_0_0",
        "core_flux_section_1",
        "core_plate_1_center",
        "core_plate_pad_1_a_center",
        "Tx_main_wcp_1_p",
        "Tx_main_wcp_pad_1_in_p",
        "Rx_main_2_0",
        "Rx_side_0_0",
        "Region",
        "Tx_main_0_0_Section1",
    ]))

    groups = hydrate_full_loss_object_groups(design)

    assert groups["Tx_windings_main"] == ["Tx_main_0_0", "Tx_main_1_0"]
    assert groups["Rx_windings_main"] == ["Rx_main_2_0", "Rx_main_10_0"]
    assert groups["Rx_windings_side"] == ["Rx_side_0_0"]
    assert groups["Rx_windings_side2"] == ["Rx_side2_0_0"]
    assert groups["core_objs"] == ["core_0_yoke_top"]
    assert groups["core_plates"] == ["core_plate_1_center"]
    assert groups["core_pads"] == ["core_plate_pad_1_a_center"]
    assert groups["wcp_plates"] == ["Tx_main_wcp_1_p"]
    assert groups["wcp_pads"] == ["Tx_main_wcp_pad_1_in_p"]
    assert [item.name for item in design.Rx_windings] == [
        "Rx_main_2_0",
        "Rx_main_10_0",
        "Rx_side_0_0",
        "Rx_side2_0_0",
    ]


def _real_full_names():
    names = [
        "Tx_main_0_0",
        "Tx_main_1_0",
        "Rx_main_0_0",
        "Rx_main_1_0",
        "Tx_side_0_0",
        "Tx_side2_0_0",
        "Rx_side_0_0",
        "Rx_side2_0_0",
        "core_flux_section_1",
        "Tx_main_wcp_1_p",
        "Tx_main_wcp_1_n",
    ]
    names.extend(
        f"core_1_{piece}"
        for piece in (
            "leg_left",
            "leg_center",
            "leg_right",
            "yoke_bottom",
            "yoke_top",
        )
    )
    names.extend(
        f"core_plate_{interface}_{side}"
        for interface in (1, 2)
        for side in ("side_left", "center", "side_right")
    )
    names.extend(
        f"core_plate_pad_{interface}_{layer}_{side}"
        for interface in (1, 2)
        for layer in ("a", "b")
        for side in ("side_left", "center", "side_right")
    )
    names.extend(
        f"Tx_main_wcp_pad_1_{layer}_{side}"
        for layer in ("in", "out")
        for side in ("p", "n")
    )
    return names


def _real_full_params():
    return pd.DataFrame([{
        "N1_main": 2,
        "N1_side": 1,
        "N2_main": 2,
        "N2_side": 1,
        "n_core_group": 1,
        "core_plate_on": 1,
        "core_plate_pad_t": 2.0,
        "wcp_on": 1,
        "wcp_t": 20.0,
        "wcp_pad_t": 2.0,
        "gap1": 10.0,
    }])


def test_validates_exact_real_full_model_object_counts():
    design = SimpleNamespace(modeler=_Modeler(_real_full_names()))

    groups = hydrate_full_loss_object_groups(design)
    expected = validate_full_loss_object_groups(groups, _real_full_params())

    assert expected["core_objs"] == 5
    assert expected["core_flux_sheets"] == 1
    assert expected["core_plates"] == 6
    assert expected["core_pads"] == 12
    assert expected["wcp_plates"] == 2
    assert expected["wcp_pads"] == 4


def test_rejects_missing_enabled_winding_cooling_plate():
    names = _real_full_names()
    names.remove("Tx_main_wcp_1_n")
    design = SimpleNamespace(modeler=_Modeler(names))
    groups = hydrate_full_loss_object_groups(design)

    with pytest.raises(
        RuntimeError, match=r'"wcp_plates".*"actual": 1.*"expected": 2'
    ):
        validate_full_loss_object_groups(groups, _real_full_params())


class _NativeDesign:
    def __init__(
        self,
        name,
        solution,
        *,
        design_type="Maxwell 3D",
        setups=("Setup1",),
    ):
        self.name = name
        self.solution = solution
        self.design_type = design_type
        self.setups = tuple(setups)

    def GetName(self):
        return f"1;{self.name}"

    def GetDesignType(self):
        return self.design_type

    def GetSolutionType(self):
        return self.solution

    def GetModule(self, name):
        assert name == "AnalysisSetup"
        return SimpleNamespace(GetSetups=lambda: list(self.setups))

    def Analyze(self, *_args, **_kwargs):
        raise AssertionError("post-solve recovery must not analyze Maxwell")


class _NativeProject:
    def __init__(self, calls=None, project_path=None, designs=None):
        self.calls = calls if calls is not None else []
        self.project_path = (
            Path(project_path).parent
            if project_path is not None
            else Path.cwd()
        )
        self.designs = list(designs) if designs is not None else [
            _NativeDesign("maxwell_matrix", "AC Magnetic"),
            _NativeDesign("maxwell_cap", "Electrostatic"),
            _NativeDesign("maxwell_loss", "AC Magnetic"),
        ]

    def GetName(self):
        return "simulation_123_456"

    def GetDesigns(self):
        return list(self.designs)

    def GetPath(self):
        return str(self.project_path)

    def SetActiveDesign(self, name):
        return next(item for item in self.designs if item.name == name)

    def DeleteDesign(self, name):
        matches = [item for item in self.designs if item.name == name]
        if len(matches) != 1:
            raise RuntimeError(name)
        self.calls.append(("delete", name))
        self.designs.remove(matches[0])


class _Project:
    def __init__(self, calls, project_path=None, designs=None):
        self.project = _NativeProject(calls, project_path, designs)
        self.calls = calls

    def create_design(self, name, solver, solution):
        self.calls.append(("bind", name, solver, solution))
        native = self.project.SetActiveDesign(name)
        app = SimpleNamespace(
            oproject=self.project,
            odesign=native,
        )
        return SimpleNamespace(
            design_name=name,
            solver_instance=app,
            modeler=_Modeler([]),
        )


class _Desktop:
    def __init__(
        self, calls, *_args, preserved_thermal=False, **_kwargs
    ):
        self.calls = calls
        self.preserved_thermal = preserved_thermal
        self.odesktop = SimpleNamespace(SetActiveProject=lambda _name: None)

    def create_project(self, path, name):
        self.calls.append(("create-project", Path(path).name, name))
        project_dir = Path(path)
        project_dir.mkdir(parents=True)
        (project_dir / f"{name}.aedt").write_bytes(b"fresh-project")
        (project_dir / f"{name}.aedtresults").mkdir()
        return _Project(
            self.calls,
            project_path=project_dir / f"{name}.aedt",
            designs=[],
        )

    def load_project(self, path):
        self.calls.append(("load", Path(path).name))
        designs = None
        if self.preserved_thermal:
            designs = [
                _NativeDesign("maxwell_matrix", "AC Magnetic"),
                _NativeDesign("maxwell_cap", "Electrostatic"),
                _NativeDesign("maxwell_loss", "AC Magnetic"),
                _NativeDesign(
                    "icepak_thermal",
                    "SteadyState",
                    design_type="Icepak",
                    setups=("ThermalSetup",),
                ),
            ]
        return _Project(self.calls, project_path=path, designs=designs)

    def release_desktop(self, **kwargs):
        self.calls.append(("release", kwargs))
        return None


def _thermal_native_design(
    *,
    design_type="Icepak",
    solution="SteadyState",
    setups=("ThermalSetup",),
):
    return _NativeDesign(
        "icepak_thermal",
        solution,
        design_type=design_type,
        setups=setups,
    )


def _thermal_rebuild_fixture(tmp_path, designs):
    sealed_root = tmp_path / "sealed" / "current"
    source_dir = sealed_root / "simulation"
    staged_dir = tmp_path / "staged"
    source_dir.mkdir(parents=True)
    staged_dir.mkdir()
    source = source_dir / "simulation_123_456.aedt"
    staged = staged_dir / "simulation_123_456.aedt"
    source.write_bytes(b"sealed-source-project")
    staged.write_bytes(source.read_bytes())
    calls = []
    project = _Project(calls, project_path=staged, designs=designs)
    kwargs = {
        "staged_aedt": staged,
        "sealed_source_aedt": source,
        "sealed_root": sealed_root,
        "expected_source_aedt_sha256": _sha256(source),
    }
    return project, calls, source, kwargs


def test_prepares_exact_saved_thermal_by_deleting_only_staged_copy(
    tmp_path,
):
    designs = [
        _NativeDesign("maxwell_matrix", "AC Magnetic"),
        _NativeDesign("maxwell_cap", "Electrostatic"),
        _NativeDesign("maxwell_loss", "AC Magnetic"),
        _thermal_native_design(),
    ]
    project, calls, source, kwargs = _thermal_rebuild_fixture(
        tmp_path, designs
    )
    source_before = source.read_bytes()

    lifecycle = prepare_staged_thermal_rebuild(project, **kwargs)

    assert lifecycle["preserved_thermal_present"] is True
    assert lifecycle["preserved_thermal_validated"] is True
    assert lifecycle["preserved_thermal_deleted_from_staged_copy"] is True
    assert lifecycle["sealed_source_mutated"] is False
    assert lifecycle["preserved_thermal_identity"] == {
        "design_name": "icepak_thermal",
        "design_type": "Icepak",
        "solution_type": "SteadyState",
        "setups": ["ThermalSetup"],
    }
    assert calls == [("delete", "icepak_thermal")]
    assert source.read_bytes() == source_before
    assert [item.name for item in project.project.designs] == [
        "maxwell_matrix",
        "maxwell_cap",
        "maxwell_loss",
    ]


def test_prepares_exact_three_maxwell_designs_without_deletion(tmp_path):
    designs = [
        _NativeDesign("maxwell_matrix", "AC Magnetic"),
        _NativeDesign("maxwell_cap", "Electrostatic"),
        _NativeDesign("maxwell_loss", "AC Magnetic"),
    ]
    project, calls, _source, kwargs = _thermal_rebuild_fixture(
        tmp_path, designs
    )

    lifecycle = prepare_staged_thermal_rebuild(project, **kwargs)

    assert lifecycle["preserved_thermal_present"] is False
    assert lifecycle["preserved_thermal_action"] == "absent_build_new"
    assert calls == []


@pytest.mark.parametrize("native_location", ["sealed_source", "wrong_stage"])
def test_thermal_prepare_rejects_wrong_native_project_before_delete(
    tmp_path, native_location
):
    designs = [
        _NativeDesign("maxwell_matrix", "AC Magnetic"),
        _NativeDesign("maxwell_cap", "Electrostatic"),
        _NativeDesign("maxwell_loss", "AC Magnetic"),
        _thermal_native_design(),
    ]
    project, calls, source, kwargs = _thermal_rebuild_fixture(
        tmp_path, designs
    )
    if native_location == "sealed_source":
        native_file = source
    else:
        wrong_dir = tmp_path / "wrong-stage"
        wrong_dir.mkdir()
        native_file = wrong_dir / "simulation_123_456.aedt"
        native_file.write_bytes(b"wrong-staged-project")
    project.project.project_path = native_file.parent
    source_before = source.read_bytes()

    with pytest.raises(
        RuntimeError, match="not bound to the exact staged copy"
    ):
        prepare_staged_thermal_rebuild(project, **kwargs)

    assert calls == []
    assert source.read_bytes() == source_before
    assert project.project.designs[-1].name == "icepak_thermal"


@pytest.mark.parametrize(
    "escaped_component",
    ["project", "results"],
)
def test_thermal_wrapper_rejects_sealed_path_escape(
    tmp_path, escaped_component
):
    sealed_root = tmp_path / "sealed" / "current"
    sealed_dir = sealed_root / "simulation"
    staged_dir = tmp_path / "staged"
    sealed_dir.mkdir(parents=True)
    staged_dir.mkdir()
    sealed_project = sealed_dir / "simulation_123_456.aedt"
    staged_project = staged_dir / "simulation_123_456.aedt"
    sealed_results = sealed_dir / "simulation_123_456.aedtresults"
    staged_results = staged_dir / "simulation_123_456.aedtresults"
    sealed_project.write_bytes(b"sealed")
    staged_project.write_bytes(b"staged")
    sealed_results.mkdir()
    staged_results.mkdir()
    thermal = SimpleNamespace(
        project_file=str(
            sealed_project
            if escaped_component == "project"
            else staged_project
        ),
        results_directory=str(
            sealed_results
            if escaped_component == "results"
            else staged_results
        ),
    )

    with pytest.raises(
        RuntimeError, match="escaped the staged project/results pair"
    ):
        attest_staged_thermal_wrapper(
            thermal,
            staged_aedt=staged_project,
            staged_results=staged_results,
            sealed_root=sealed_root,
        )


@pytest.mark.parametrize(
    ("extra_designs", "error_pattern"),
    [
        (
            [_NativeDesign("rogue", "AC Magnetic")],
            "not an exact Full recovery design set",
        ),
        (
            [_thermal_native_design(), _thermal_native_design()],
            "duplicate design names",
        ),
    ],
)
def test_rejects_wrong_or_duplicate_saved_design_extras(
    tmp_path, extra_designs, error_pattern
):
    designs = [
        _NativeDesign("maxwell_matrix", "AC Magnetic"),
        _NativeDesign("maxwell_cap", "Electrostatic"),
        _NativeDesign("maxwell_loss", "AC Magnetic"),
        *extra_designs,
    ]
    project, calls, _source, kwargs = _thermal_rebuild_fixture(
        tmp_path, designs
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        prepare_staged_thermal_rebuild(project, **kwargs)

    assert calls == []


@pytest.mark.parametrize(
    "thermal",
    [
        _thermal_native_design(design_type="Maxwell 3D"),
        _thermal_native_design(
            solution="SteadyState TemperatureAndFlow"
        ),
        _thermal_native_design(solution="Transient"),
        _thermal_native_design(setups=("ThermalSetup", "Setup2")),
        _thermal_native_design(setups=("WrongSetup",)),
    ],
)
def test_rejects_saved_thermal_with_wrong_native_identity(
    tmp_path, thermal
):
    designs = [
        _NativeDesign("maxwell_matrix", "AC Magnetic"),
        _NativeDesign("maxwell_cap", "Electrostatic"),
        _NativeDesign("maxwell_loss", "AC Magnetic"),
        thermal,
    ]
    project, calls, _source, kwargs = _thermal_rebuild_fixture(
        tmp_path, designs
    )

    with pytest.raises(RuntimeError, match="thermal design identity mismatch"):
        prepare_staged_thermal_rebuild(project, **kwargs)

    assert calls == []


def test_exact_solved_design_rejects_magnetostatic_replacement():
    project = _Project([])
    project.project.designs[0].solution = "Magnetostatic"

    with pytest.raises(RuntimeError, match="physics mismatch"):
        _exact_solved_design(
            project, "maxwell_matrix", "ac_magnetic"
        )


class _Simulation:
    def __init__(self, calls, desktop=None):
        import tools.recover_full_postsolve_project as module

        self.calls = calls
        self.solve_attempts = {"matrix": 0, "cap": 0, "loss": 0}
        self.NUM_CORE = 16
        self.NUM_TASK = 1
        self.solver_core_policy = {
            "schema": "mft-solver-core-policy-v1",
            "contract_version": "mft-standalone-core-16-optin-v1",
            "opt_in": True,
            "backend": "standalone",
            "requested_num_cores": 16,
            "effective_num_cores": 16,
            "num_tasks": 1,
            "affinity_count_readback": 16,
            "slurm_cpus_per_task_readback": 16,
            "scheduler_task_id_readback": 123,
            "slurm_job_id_readback": 456,
            "solver_revision": module.GIT_HASH,
            "solver_dirty": 0,
            "auth_sha256": "c" * 64,
            "license_contract": "mft-aedt-hpc-license-snapshot-v1",
            "license_snapshot_sha256": "d" * 64,
            "license_snapshot_age_seconds_readback": 1.0,
            "license_headroom_readback": {
                "anshpc": 16,
                "elec_solve_maxwell": 1,
                "electronics_desktop": 1,
                "electronics3d_gui": 1,
            },
        }

    def _remember_native_desktop_handle(self, _desktop):
        self.calls.append(("remember",))

    def get_magnetic_parameter(self):
        raise AssertionError("live Maxwell matrix extraction is forbidden")

    def get_capacitance_parameter(self):
        raise AssertionError("live Maxwell capacitance extraction is forbidden")

    def save_calculation(self):
        raise AssertionError("live Maxwell loss calculation is forbidden")

    def save_loss_reports(self):
        raise AssertionError("live Maxwell loss extraction is forbidden")

    def get_convergence_info(self, label):
        raise AssertionError(
            f"live Maxwell convergence extraction is forbidden: {label}"
        )


def _fake_recovery_args(tmp_path, seal, watch_path, watch_payload):
    return SimpleNamespace(
        sealed_root=str(seal.gpfs_root),
        source_task_id=123,
        manifest_sha256=seal.manifest["manifest_sha256"],
        source_identity_sha256=json.loads(
            seal.identity_path.read_text(encoding="utf-8")
        )["identity_sha256"],
        terminal_watch_result=str(watch_path),
        terminal_watch_sha256=watch_payload["terminal_watch_sha256"],
        params_sha256=seal.params_sha,
        params_canonical_sha256=seal.params_canonical_sha,
        candidate_digest=seal.candidate_digest,
        profile_sha256=seal.profile_sha,
        project_name="simulation_123_456",
        recovery_revision="a" * 40,
        bundle_sha256="b" * 64,
        work_dir=str(tmp_path / "recovery-work"),
        output_dir=str(tmp_path / "recovery-output"),
        aedt_version="2025.2",
    )


def _install_fake_recovery(
    monkeypatch, calls, thermal_valid, *, preserved_thermal=False
):
    import tools.recover_full_postsolve_project as module

    df_plus = pd.DataFrame([_fake_validated_parameter_row()])
    monkeypatch.setattr(
        module,
        "attest_recovery_runtime",
        lambda **_kwargs: {
            "schema": "mft-full-recovery-runtime-evidence-v1",
            "provenance_mode": "clean_git_bundle_checkout_and_sha256",
            "recovery_revision": "a" * 40,
            "package_sha256": "b" * 64,
            "source_bundle_sha256": "b" * 64,
            "runtime_git_root": "/source",
            "source_revision_is_descendant_of": (
                "146142be579e2f3e45f12961214018a4e28445c5"
            ),
            "solver_runtime_git_hash": "a" * 40,
            "solver_runtime_git_dirty": 0,
            "pyaedt_library_root": "/enroot/pyaedt_library",
            "pyaedt_library_git_hash": (
                "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
            ),
            "pyaedt_library_git_dirty": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "pyDesktop",
        lambda *args, **kwargs: _Desktop(
            calls,
            *args,
            preserved_thermal=preserved_thermal,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        module,
        "Simulation",
        lambda desktop=None: _Simulation(calls, desktop),
    )
    monkeypatch.setattr(
        module,
        "_load_fixed_input_parameter",
        lambda _params: (df_plus.copy(), "physics-revision"),
    )
    monkeypatch.setattr(
        module,
        "validation_check",
        lambda _input, strict: (True, df_plus.copy()),
    )
    monkeypatch.setattr(
        module,
        "hydrate_full_loss_object_groups",
        lambda _design: {
            "Tx_windings_main": ["Tx_main_0_0"],
            "Tx_windings_side": [],
            "Tx_windings_side2": [],
            "Rx_windings_main": ["Rx_main_0_0"],
            "Rx_windings_side": [],
            "Rx_windings_side2": [],
            "core_objs": ["core_1"],
            "core_flux_sheets": ["core_flux_section_1"],
            "core_plates": [],
            "core_pads": [],
            "wcp_plates": [],
            "wcp_pads": [],
        },
    )
    monkeypatch.setattr(
        module,
        "validate_full_loss_object_groups",
        lambda _groups, _df: {"core_objs": 1},
    )
    def fake_run_thermal(sim):
        calls.append(("thermal",))
        sim.project.project.designs.append(_NativeDesign(
            "icepak_thermal",
            "SteadyState",
            design_type="Icepak",
            setups=("ThermalSetup",),
        ))
        project_file = (
            Path(sim.project_path) / f"{sim.PROJECT_NAME}.aedt"
        )
        sim.design_thermal = SimpleNamespace(
            project_file=str(project_file),
            results_directory=str(
                project_file.with_suffix(".aedtresults")
            ),
        )
        return pd.DataFrame([{
            "thermal_solved": int(thermal_valid),
            **_fake_thermal_mesh_fields(),
            "thermal_pad_conductivity_W_mK": 0.2,
            "thermal_pad_material_policy": (
                "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
                "electrically_insulating_v1"
            ),
            "thermal_pad_native_readback_contract_version": (
                "thermal-pad-native-material-readback-v1"
            ),
            "thermal_pad_native_readback_attested": 1,
            "thermal_pad_native_thermal_conductivity_W_mK": 0.2,
            "thermal_pad_native_electrical_conductivity_S_m": 0.0,
            "thermal_rx_explicit_insulation_model": (
                "solid_interturn_candidate_k_ins_v1"
            ),
            "thermal_rx_explicit_insulation_count": 6,
            "thermal_rx_explicit_insulation_policy": (
                "rx-explicit-interturn-solid-candidate-k-ins-"
                "native-attested-v1"
            ),
            "thermal_rx_explicit_insulation_material": (
                "winding_insulation"
            ),
            "thermal_rx_explicit_insulation_counts_json": json.dumps({
                "Rx_main_insulation": 2,
                "Rx_side_insulation": 2,
                "Rx_side2_insulation": 2,
            }, sort_keys=True),
            "thermal_rx_explicit_insulation_"
            "native_readback_contract_version": (
                "rx-explicit-insulation-native-material-readback-v1"
            ),
            "thermal_rx_explicit_insulation_"
            "native_readback_attested": 1,
            "thermal_rx_explicit_insulation_"
            "native_thermal_conductivity_W_mK": 0.2,
            "thermal_rx_explicit_insulation_"
            "native_electrical_conductivity_S_m": 0.0,
        }])

    monkeypatch.setattr(module, "run_thermal_analysis", fake_run_thermal)
    monkeypatch.setattr(
        module,
        "_thermal_result_is_valid",
        lambda _frame, physics_data_revision: thermal_valid,
    )


def test_complete_recovery_reuses_sealed_em_without_any_live_maxwell_call(
    tmp_path, monkeypatch
):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=True)

    manifest = recover(
        _fake_recovery_args(tmp_path, seal, watch_path, watch_payload)
    )

    assert manifest["status"] == "complete_full_recovery"
    assert manifest["maxwell_analyze_calls"] == 0
    assert manifest["result_valid_em"] is True
    assert manifest["result_valid_cap"] is True
    assert manifest["result_valid_thermal"] is True
    assert manifest["live_maxwell_extraction_calls"] == 0
    assert not [item for item in calls if item[0] == "extract"]
    assert not [item for item in calls if item[0] == "bind"]
    assert not [item for item in calls if item[0] == "load"]
    assert manifest["source_result_evidence"]["project_row_count"] == 1
    assert manifest["loss_evidence"]["required_key_count"] == 5
    assert manifest["thermal_pad_evidence"]["native_readback_attested"] is True
    assert manifest["fixed_boundary_evidence"][
        "authoritative_fixed_boundary_attested"
    ] is True
    assert manifest["fixed_boundary_evidence"]["mismatches"] == []
    assert manifest["thermal_mesh_evidence"][
        "postsolve_prior_nine_probe_complete"
    ] is True
    assert manifest["recovery_runtime"]["thermal_solver_cores"] == 16
    assert manifest["recovery_runtime"]["solver_core_policy"][
        "contract_version"
    ] == "mft-standalone-core-16-optin-v1"
    durable = manifest["thermal_project"]["durable_seal"]
    assert durable["status"] == "atomically_published_read_only"
    assert durable["file_count"] == 1
    assert Path(durable["aedt"]).is_file()
    assert Path(durable["aedtresults"]).is_dir()
    assert manifest["thermal_rebuild"]["durable_project_seal"][
        "tree_sha256"
    ] == durable["tree_sha256"]
    result = pd.read_csv(
        tmp_path / "recovery-output" / "postsolve-recovery-result.csv"
    )
    assert float(result["Llt"].iloc[0]) == 19.0
    assert float(result["C_tx_rx_F"].iloc[0]) == 0.5e-9
    assert int(
        result["postsolve_recovery_maxwell_analyze_calls"].iloc[0]
    ) == 0
    assert int(
        result[
            "postsolve_recovery_live_maxwell_extraction_calls"
        ].iloc[0]
    ) == 0
    assert int(
        result["fixed_boundary_authoritative_attested"].iloc[0]
    ) == 1
    assert int(result["fixed_boundary_diagnostic_only"].iloc[0]) == 0
    assert float(
        result["fixed_boundary_fan_velocity_m_s"].iloc[0]
    ) == 1.5
    assert float(
        result["fixed_boundary_core_plate_pad_t_mm"].iloc[0]
    ) == 2.0
    assert float(
        result[
            "fixed_boundary_thermal_pad_conductivity_W_mK"
        ].iloc[0]
    ) == 0.2
    assert (
        result["postsolve_recovery_em_evidence_source"].iloc[0]
        == "manifest_bound_simulation_results_260706_csv"
    )
    assert result["postsolve_recovery_solver_revision"].iloc[0] == "a" * 40
    assert result["postsolve_recovery_package_sha256"].iloc[0] == "b" * 64
    assert int(
        result["postsolve_recovery_thermal_solver_cores"].iloc[0]
    ) == 16
    assert (
        result["postsolve_recovery_pyaedt_library_git_hash"].iloc[0]
        == "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
    )
    assert Path(
        result["postsolve_recovery_durable_thermal_aedt"].iloc[0]
    ).is_file()
    assert len(
        result[
            "postsolve_recovery_durable_thermal_tree_sha256"
        ].iloc[0]
    ) == 64
    assert int(
        result["postsolve_recovery_preserved_thermal_present"].iloc[0]
    ) == 0
    assert int(
        result[
            "postsolve_recovery_sealed_source_thermal_mutated"
        ].iloc[0]
    ) == 0
    assert calls[-1][0] == "release"


def test_recovery_rejects_deadline_boundary_override_before_aedt_launch(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=True)
    deadline = pd.DataFrame([{
        **_fake_validated_parameter_row(),
        "fan_velocity": 6.0,
        "core_plate_pad_t": 1.0,
        "wcp_pad_t": 1.0,
    }])
    monkeypatch.setattr(
        module,
        "_load_fixed_input_parameter",
        lambda _params: (deadline.copy(), "physics-revision"),
    )
    monkeypatch.setattr(
        module,
        "validation_check",
        lambda _input, strict: (True, deadline.copy()),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "non-authoritative fixed-boundary candidate rejected: "
            "fan_velocity.*core_plate_pad_t.*wcp_pad_t"
        ),
    ):
        recover(
            _fake_recovery_args(
                tmp_path, seal, watch_path, watch_payload
            )
        )

    assert calls == []
    assert not (tmp_path / "recovery-work").exists()
    assert not (tmp_path / "recovery-output").exists()


def test_complete_recovery_never_opens_or_copies_saved_source_project(
    tmp_path, monkeypatch
):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(
        monkeypatch,
        calls,
        thermal_valid=True,
        preserved_thermal=True,
    )
    source_before = seal.aedt.read_bytes()

    manifest = recover(
        _fake_recovery_args(tmp_path, seal, watch_path, watch_payload)
    )

    assert manifest["maxwell_analyze_calls"] == 0
    assert manifest["live_maxwell_extraction_calls"] == 0
    assert manifest["design_names"] == ["icepak_thermal"]
    assert manifest["thermal_project"]["source_project_opened"] is False
    assert manifest["thermal_project"]["source_results_copied"] is False
    rebuild = manifest["thermal_rebuild"]
    assert rebuild["preserved_thermal_validated"] is False
    assert rebuild["preserved_thermal_deleted_from_staged_copy"] is False
    assert rebuild["status"] == "fresh_thermal_complete"
    assert rebuild["sealed_source_mutated"] is False
    assert (
        rebuild["rebuilt_thermal_wrapper"][
            "paths_bound_to_staged_copy"
        ]
        is True
    )
    assert "sealed_source_results_metadata_sha256" in rebuild
    assert seal.aedt.read_bytes() == source_before
    assert not [item for item in calls if item[0] == "load"]
    assert not [item for item in calls if item[0] == "delete"]
    assert [item for item in calls if item[0] == "create-project"]
    result = pd.read_csv(
        tmp_path / "recovery-output" / "postsolve-recovery-result.csv"
    )
    assert int(
        result["postsolve_recovery_preserved_thermal_present"].iloc[0]
    ) == 0
    assert int(
        result[
            "postsolve_recovery_preserved_thermal_deleted_from_staged_copy"
        ].iloc[0]
    ) == 0
    assert int(
        result["postsolve_recovery_maxwell_analyze_calls"].iloc[0]
    ) == 0
    assert int(
        result["postsolve_recovery_source_project_opened"].iloc[0]
    ) == 0


def test_thermal_invalid_recovery_never_publishes_complete_manifest(
    tmp_path, monkeypatch
):
    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=False)

    with pytest.raises(RuntimeError, match="thermal result failed"):
        recover(
            _fake_recovery_args(tmp_path, seal, watch_path, watch_payload)
        )

    output = tmp_path / "recovery-output"
    assert not (output / "postsolve-recovery-manifest.json").exists()
    assert (output / "postsolve-recovery-failure.json").exists()
    assert (output / "postsolve-recovery-rejected-result.csv").exists()
    assert calls[-1][0] == "release"


def test_uncertain_native_thermal_failure_never_calls_desktop_release(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=False)

    def fail_with_native_engine_running(sim):
        sim.solver_may_be_running = True
        raise RuntimeError("completion barrier timed out")

    monkeypatch.setattr(
        module, "run_thermal_analysis", fail_with_native_engine_running
    )

    with pytest.raises(RuntimeError, match="completion barrier timed out"):
        recover(
            _fake_recovery_args(tmp_path, seal, watch_path, watch_payload)
        )

    assert not [item for item in calls if item[0] == "release"]
    output = tmp_path / "recovery-output"
    assert not (output / "postsolve-recovery-manifest.json").exists()
    assert (output / "postsolve-recovery-failure.json").exists()
    failure = json.loads(
        (output / "postsolve-recovery-failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["standalone_solver_may_be_running"] is True


def test_desktop_release_failure_never_publishes_success_csv(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=True)

    class ReleaseFailDesktop(_Desktop):
        def release_desktop(self, **kwargs):
            self.calls.append(("release", kwargs))
            return False

    monkeypatch.setattr(
        module,
        "pyDesktop",
        lambda *args, **kwargs: ReleaseFailDesktop(
            calls, *args, **kwargs
        ),
    )

    with pytest.raises(RuntimeError, match="release returned False"):
        recover(
            _fake_recovery_args(tmp_path, seal, watch_path, watch_payload)
        )

    output = tmp_path / "recovery-output"
    assert not (output / "postsolve-recovery-result.csv").exists()
    assert not (output / "postsolve-recovery-manifest.json").exists()
    assert (output / "postsolve-recovery-failure.json").exists()


def test_cli_uncertain_solver_uses_hard_exit_instead_of_system_exit(
    monkeypatch,
):
    import tools.recover_full_postsolve_project as module

    class HardExit(BaseException):
        pass

    exit_codes = []
    monkeypatch.setattr(module, "parse_args", lambda _argv: object())
    monkeypatch.setattr(
        module,
        "recover",
        lambda _args: (_ for _ in ()).throw(
            module.UncertainStandaloneSolverExit("engine still running")
        ),
    )

    def hard_exit(code):
        exit_codes.append(code)
        raise HardExit()

    monkeypatch.setattr(module.os, "_exit", hard_exit)

    with pytest.raises(HardExit):
        module.main([])

    assert exit_codes == [
        module.UncertainStandaloneSolverExit.exit_code
    ]


def test_cli_stream_failures_cannot_prevent_uncertain_hard_exit(
    monkeypatch,
):
    import tools.recover_full_postsolve_project as module

    class HardExit(BaseException):
        pass

    class BrokenStream:
        def write(self, _value):
            raise OSError("write failed")

        def flush(self):
            raise OSError("flush failed")

    exit_codes = []
    monkeypatch.setattr(module, "parse_args", lambda _argv: object())
    monkeypatch.setattr(
        module,
        "recover",
        lambda _args: (_ for _ in ()).throw(
            module.UncertainStandaloneSolverExit("engine still running")
        ),
    )
    monkeypatch.setattr(module.sys, "stdout", BrokenStream())
    monkeypatch.setattr(module.sys, "stderr", BrokenStream())

    def hard_exit(code):
        exit_codes.append(code)
        raise HardExit()

    monkeypatch.setattr(module.os, "_exit", hard_exit)

    with pytest.raises(HardExit):
        module.main([])

    assert exit_codes == [
        module.UncertainStandaloneSolverExit.exit_code
    ]


def test_uncertain_failure_artifact_error_still_reaches_cli_hard_exit(
    tmp_path, monkeypatch
):
    import tools.recover_full_postsolve_project as module

    class HardExit(BaseException):
        pass

    seal = _write_sealed_current(
        tmp_path, "source_exit_code_published"
    )
    watch_path, watch_payload = _write_terminal_watch(tmp_path, seal)
    args = _fake_recovery_args(
        tmp_path, seal, watch_path, watch_payload
    )
    calls = []
    _install_fake_recovery(monkeypatch, calls, thermal_valid=False)

    def fail_with_native_engine_running(sim):
        sim.solver_may_be_running = True
        raise RuntimeError("completion barrier timed out")

    monkeypatch.setattr(
        module, "run_thermal_analysis", fail_with_native_engine_running
    )
    monkeypatch.setattr(
        module,
        "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("GPFS fsync failed")
        ),
    )
    monkeypatch.setattr(module, "parse_args", lambda _argv: args)
    exit_codes = []

    def hard_exit(code):
        exit_codes.append(code)
        raise HardExit()

    monkeypatch.setattr(module.os, "_exit", hard_exit)

    with pytest.raises(HardExit):
        module.main([])

    assert exit_codes == [
        module.UncertainStandaloneSolverExit.exit_code
    ]
    assert not [item for item in calls if item[0] == "release"]
