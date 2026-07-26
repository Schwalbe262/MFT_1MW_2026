from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from module.mft_goal_20260726_contract import (
    GOAL_TEMPERATURE_TARGETS,
    fixed_identity_expectations,
)
from tools import mft_goal_official5_direct_terminal_collector as collector


def _task96338() -> dict:
    return {
        "id": collector.TASK_ID,
        "task_id": collector.TASK_ID,
        "name": collector.TASK_NAME,
        "dedupe_key": collector.DEDUPE_KEY,
        "project": collector.SCHEDULER_PROJECT,
        "account_name": collector.ACCOUNT_NAME,
        "requested_account_name": collector.ACCOUNT_NAME,
        "requested_node_name": collector.NODE_NAME,
        "requested_node_name_policy": "strict",
        "node_name": collector.NODE_NAME,
        "node_name_policy": "strict",
        "actual_node_name": collector.NODE_NAME,
        "allocation_node_name": collector.NODE_NAME,
        "same_node_as_node_name": collector.NODE_NAME,
        "allocation_id": collector.ALLOCATION_ID,
        "assigned_allocation": collector.ALLOCATION_ID,
        "same_node_as_allocation_id": collector.ALLOCATION_ID,
        "same_node_as_task_id": collector.SAME_NODE_TASK_ID,
        "slurm_job_id": collector.SLURM_JOB_ID,
        "placement_contract_satisfied": True,
        "strict_node_placement": True,
        "preferred_node_relaxed": False,
        "cpus": 8,
        "memory_mb": 98_304,
        "gpus": 0,
        "max_workers_per_node": 2,
        "timeout_seconds": 45_300,
        "priority": 100,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": (
            "slurm_scheduler/runs/2026-07-26/task-96338-1785073843"
        ),
        "status": "running",
        "state": "running",
        "exit_code": None,
    }


def _task96337() -> dict:
    return {
        "id": collector.PRIOR_FAILURE_TASK_ID,
        "task_id": collector.PRIOR_FAILURE_TASK_ID,
        "name": collector.PRIOR_FAILURE_TASK_NAME,
        "dedupe_key": collector.PRIOR_FAILURE_DEDUPE_KEY,
        "project": collector.SCHEDULER_PROJECT,
        "account_name": collector.ACCOUNT_NAME,
        "actual_node_name": collector.NODE_NAME,
        "allocation_node_name": collector.NODE_NAME,
        "allocation_id": collector.ALLOCATION_ID,
        "same_node_as_allocation_id": collector.ALLOCATION_ID,
        "same_node_as_task_id": collector.SAME_NODE_TASK_ID,
        "slurm_job_id": collector.SLURM_JOB_ID,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "state": "failed",
        "status": "failed",
        "exit_code": 1,
        "placement_contract_satisfied": True,
        "failure_message": (
            "RuntimeError: standalone core opt-in authentication digest mismatch"
        ),
    }


def _result() -> dict:
    result = {
        **fixed_identity_expectations(),
        "git_hash": collector.SOLVER_REVISION,
        "git_dirty": 0,
        "pyaedt_library_git_hash": collector.LIBRARY_REVISION,
        "pyaedt_library_git_dirty": 0,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "N2_side": 23,
        "l1": 68.0,
        "l2": 357.5,
        "h1": 571.0,
        "w1": 476.0,
        "sl1_main_x": 700.0,
        "nwl1_main": 20.0,
        "sl1_main_y": 400.0,
        "nwb1_main_y": 20.0,
        "sl2_side_x": 220.0,
        "sl2_side_y": 220.0,
        "nwl2_side": 20.0,
        "f_res_min_tx_rx_only_Hz": 16_000.0,
        "P_winding_total": 1_000.0,
        "P_core_total": 2_000.0,
        "P_core_plate_total": 100.0,
        "P_wcp_total": 100.0,
        "thermal_mesh_preflight_status": collector.DIRECT_PREFLIGHT_STATUS,
        "thermal_mesh_native_generation_passed": 0,
        "thermal_mesh_direct_analyze_opt_in": 1,
        "thermal_mesh_direct_analyze_contract_version": (
            collector.DIRECT_CONTRACT
        ),
        "thermal_mesh_direct_analyze_contract_sha256": "a" * 64,
        "thermal_mesh_unmeshed_object_count": 0,
        "thermal_mesh_unmeshed_objects_json": "[]",
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": collector.CORE_CONTRACT,
        "solver_core_opt_in": 1,
        "solver_core_backend": "standalone",
        "solver_num_cores_requested": 8,
        "solver_num_cores_effective": 8,
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": 72,
        "solver_core_slurm_cpus_per_task_readback": "8",
        "solver_core_scheduler_task_id_readback": str(collector.TASK_ID),
        "solver_core_slurm_job_id_readback": collector.SLURM_JOB_ID,
        "solver_core_auth_sha256": collector.CORE_AUTH_SHA256,
        "solver_core_license_contract": "",
        "solver_core_license_snapshot_sha256": "",
        "solver_matrix_hpc_num_cores_readback": 8,
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "b" * 64,
        "solver_core_dispatch_evidence_json": json.dumps(
            {"matrix": {"requested_cores": 8}},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "solver_core_readback_evidence_json": json.dumps(
            {
                "matrix_hpc_acf": {
                    "num_cores_readback": 8,
                    "num_engines_readback": 1,
                    "acf_sha256": "b" * 64,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_solved": 1,
        "thermal_convergence_available": 1,
        "thermal_converged": 1,
        "thermal_iterations": 30,
        "thermal_extraction_complete": 1,
        "thermal_required_missing_count": 0,
        "thermal_required_group_mask": 15,
        "thermal_rx_power_balance_ok": 1,
        "thermal_rx_power_balance_group_count": 3,
        "thermal_rx_expected_power_w": 10_000.0,
        "thermal_rx_assigned_power_w": 10_000.0,
        "thermal_rx_power_balance_max_abs_w": 0.0,
        "thermal_rx_model": "homogenized_blocks",
        "thermal_residual_flow_limit": 1e-3,
        "thermal_residual_energy_limit": 1e-7,
        "thermal_residual_continuity": 1e-4,
        "thermal_residual_x_velocity": 1e-4,
        "thermal_residual_y_velocity": 1e-4,
        "thermal_residual_z_velocity": 1e-4,
        "thermal_residual_energy": 1e-8,
        "physics_data_revision": "mft1mw-1k101-native-lamination-kf0p85-v3",
        "Tprobe_Rx_side_leeward_mean": 87.0,
        "Tprobe_Rx_side_outer_max": 87.0,
        "Tprobe_Rx_side_outer_mean": 86.0,
        "Tprobe_Rx_side_inner_max": 88.0,
        "Tprobe_Rx_side_inner_mean": 87.0,
        "thermal_rx_side_probe_contract_version": (
            "rx-side-transformer-inner-outer-v1"
        ),
        "thermal_rx_side_probe_max_rule": (
            "max_across_all_transformer_inner_and_outer_faces"
        ),
        "thermal_rx_side_probe_mean_rule": (
            "mean_of_face_selected_by_max_no_cross_face_average"
        ),
        "thermal_rx_side_probe_selected_face": "Tprobe_Rx_side1_inner",
        "thermal_rx_side_probe_face_count": 2,
    }
    for target in GOAL_TEMPERATURE_TARGETS:
        result[target] = (
            90.0
            if target in {
                "T_max_Tx",
                "T_max_Rx_main",
                "T_max_Rx_side",
                "Tprobe_Tx_leeward_max",
                "Tprobe_Rx_main_leeward_max",
                "Tprobe_Rx_side_leeward_max",
            }
            else 110.0
        )
    return result


def test_exact_official5_source_contract_authenticates() -> None:
    if not collector.DEFAULT_SOURCE_ROOT.exists():
        pytest.skip("exact task96338 source package is not present")
    contract = collector.load_contract()
    assert contract["task_id"] == collector.TASK_ID
    assert contract["same_node_as_task_id"] == collector.SAME_NODE_TASK_ID
    assert contract["same_node_as_allocation_id"] == collector.ALLOCATION_ID
    assert contract["slurm_job_id"] == collector.SLURM_JOB_ID
    assert contract["parameter_digest"] == collector.PARAMETER_DIGEST
    assert contract["selected"]["selected_row"]["candidate_physics_sha"] == (
        collector.CANDIDATE_PHYSICS_SHA256
    )


def test_get_task_requires_exact_same_node_placement() -> None:
    contract = {"scheduler_url": collector.SCHEDULER_URL}
    task = _task96338()
    methods = []

    def getter(url: str, *, max_bytes: int, timeout: float) -> bytes:
        methods.append((url, max_bytes, timeout))
        return collector.canonical_bytes(task)

    actual = collector.get_task(contract, getter=getter)
    assert actual["same_node_as_task_id"] == collector.SAME_NODE_TASK_ID
    assert methods and all("/api/tasks/96338" in item[0] for item in methods)

    drifted = copy.deepcopy(task)
    drifted["same_node_as_task_id"] = 0

    def drifted_getter(
        _url: str, *, max_bytes: int, timeout: float
    ) -> bytes:
        del max_bytes, timeout
        return collector.canonical_bytes(drifted)

    with pytest.raises(collector.CollectionError, match="same_node_as_task_id"):
        collector.get_task(contract, getter=drifted_getter)


def test_scientific_attestation_preserves_all_truth_gates() -> None:
    evidence = collector.attest_result(_result())
    assert evidence["direct_analyze_authentication_passed"] is True
    assert evidence["solver_core_authentication_passed"] is True
    assert evidence["thermal_truth_authentication_passed"] is True
    assert evidence["goal_physical_spec_passed"] is True
    assert evidence["goal_physical_spec"]["gates"]["resonance_Hz"][
        "passed"
    ] is True
    assert evidence["fixed_identity_attestation"]["attested"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "thermal_mesh_direct_analyze_opt_in",
            0,
            "direct Analyze evidence drifted",
        ),
        (
            "solver_core_auth_sha256",
            "0" * 64,
            "solver core evidence drifted",
        ),
        ("thermal_converged", 0, "scientific truth gate failed"),
        ("fan_velocity", 2.0, "fixed operating/cooling identity failed"),
    ),
)
def test_scientific_attestation_fails_closed(
    field: str, value: object, message: str
) -> None:
    result = _result()
    result[field] = value
    with pytest.raises(collector.CollectionError, match=message):
        collector.attest_result(result)


def test_goal_failure_is_recorded_without_discarding_valid_fea() -> None:
    result = _result()
    result["T_max_Tx"] = 101.0
    evidence = collector.attest_result(result)
    assert evidence["thermal_truth_authentication_passed"] is True
    assert evidence["goal_physical_spec_passed"] is False
    assert any(
        reason.startswith("winding_max_C failed")
        or reason.startswith("T_max_Tx failed")
        for reason in evidence["goal_physical_spec_reasons"]
    )


def test_prior_failure_ledger_is_get_only_and_idempotent(
    tmp_path: Path,
) -> None:
    calls = []
    task = _task96337()

    def getter(url: str, *, max_bytes: int, timeout: float) -> bytes:
        calls.append((url, max_bytes, timeout))
        if url.endswith(f"/api/tasks/{collector.PRIOR_FAILURE_TASK_ID}"):
            return collector.canonical_bytes(task)
        assert "/stdout?" in url or "/stderr?" in url
        return b"standalone core opt-in authentication digest mismatch\n"

    path = tmp_path / "terminal_failure_task96337_v1.failure_ledger.json"
    first = collector.ensure_prior_failure_ledger(path, getter=getter)
    call_count = len(calls)
    second = collector.ensure_prior_failure_ledger(path, getter=getter)
    assert first["event"] == "prior_failure_ledger"
    assert second["event"] == "prior_failure_ledger_exists"
    assert len(calls) == call_count
    ledger = json.loads(path.read_text("utf-8"))
    assert ledger["task_id"] == collector.PRIOR_FAILURE_TASK_ID
    assert ledger["excluded_from_selection_and_nds"] is True
    assert ledger["scheduler_get_only_collection"] is True
    assert ledger["scheduler_mutation_performed"] is False
    assert all("/api/tasks/" in url for url, _, _ in calls)


def test_collector_source_exposes_no_scheduler_mutation_call() -> None:
    source = Path(collector.__file__).read_text("utf-8")
    assert 'method="POST"' not in source
    assert "method='POST'" not in source
    assert "/cancel" not in source
    assert "/api/tasks/" in source
