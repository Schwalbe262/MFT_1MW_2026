from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_terminal_collector as collector


REPO_ROOT = Path(__file__).resolve().parents[1]


def _offline_contract(target: str) -> tuple[collector.Contract, bytes]:
    sealed = collector.load_contract(target, REPO_ROOT)
    task_sh = sealed.command + b"\n"
    return (
        replace(
            sealed,
            task_sh_size=len(task_sh),
            task_sh_sha256=hashlib.sha256(task_sh).hexdigest(),
        ),
        task_sh,
    )


def _task(contract: collector.Contract, *, status: str, exit_code: int | None):
    return {
        "id": contract.task_id,
        "task_id": contract.task_id,
        "name": contract.name,
        "dedupe_key": contract.dedupe_key,
        "account_name": contract.account,
        "requested_account_name": contract.account,
        "requested_node_name": contract.node,
        "node_name": contract.node,
        "actual_node_name": contract.node,
        "allocation_node_name": contract.node,
        "node_name_policy": "strict",
        "strict_node_placement": True,
        "cpus": contract.cpus,
        "memory_mb": contract.memory_mb,
        "gpus": 0,
        "timeout_seconds": contract.timeout_seconds,
        "project": "MFT_1MW_2026v1",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "status": status,
        "state": status,
        "exit_code": exit_code,
    }


class FakeGetClient:
    def __init__(self, task: dict[str, Any], task_sh: bytes):
        self.task = task
        self.task_sh = task_sh
        self.remote_files_calls = 0
        self.output_calls = 0

    def get_json(self, endpoint: str) -> dict[str, Any]:
        assert endpoint == f"/api/tasks/{self.task['task_id']}"
        return self.task

    def remote_file(
        self,
        task_id: int,
        path: str,
        *,
        base: str = "remote_cwd",
        max_bytes: int = collector.MAX_REMOTE_FILE_BYTES,
    ) -> bytes:
        assert task_id == self.task["task_id"]
        assert path == "task.sh"
        assert base == "remote_dir"
        assert max_bytes == collector.MAX_REMOTE_FILE_BYTES
        return self.task_sh

    def remote_files(
        self,
        task_id: int,
        glob: str,
        *,
        base: str = "remote_cwd",
    ) -> list[str]:
        assert task_id == self.task["task_id"]
        assert base == "remote_cwd"
        assert glob
        self.remote_files_calls += 1
        return []

    def task_output(self, task_id: int, stream: str) -> bytes:
        self.output_calls += 1
        raise AssertionError((task_id, stream))


def test_exact_sealed_contracts_load_with_nonproduction_classification():
    full = collector.load_contract("full96326", REPO_ROOT)
    thermal = collector.load_contract("thermal96324", REPO_ROOT)

    assert (full.task_id, full.node, full.command_sha256) == (
        96326,
        "n116",
        "979da42b7df65a1804892a783ed9756383863ea258cdda2cd1db20dfae792652",
    )
    assert (thermal.task_id, thermal.node, thermal.command_sha256) == (
        96324,
        "n111",
        "41dfe4f93c47a7dedba9e86ed5fb4cbb7481845d19e2a5e881d504e29ef63a7b",
    )
    assert thermal.plan["contract"]["fixed_physics"] == {
        "fan_velocity_m_per_s": 1.5,
        "thermal_pad_thickness_mm": 2.0,
        "tim_conductivity_w_per_mk": 0.2,
    }
    assert collector.CLASSIFICATION["scientific_pass_claimed"] is False
    assert collector.CLASSIFICATION["production_claimed"] is False


@pytest.mark.parametrize("target", ["full96326", "thermal96324"])
def test_running_poll_revalidates_exact_task_and_command(
    target: str,
    tmp_path: Path,
):
    contract, task_sh = _offline_contract(target)
    fake = FakeGetClient(_task(contract, status="running", exit_code=None), task_sh)

    event = collector.poll_once(
        contract,
        fake,  # type: ignore[arg-type]
        tmp_path,
        poll_number=1,
    )

    assert event["state"] == "watching"
    assert event["contract"]["task_identity_verified"] is True
    assert event["contract"]["strict_node_verified"] is True
    assert event["contract"]["fixed_physics_verified"] is True
    assert event["contract"]["command_sha256"] == contract.command_sha256
    assert fake.remote_files_calls == 0
    assert fake.output_calls == 0


def test_identity_drift_is_fail_closed():
    contract, task_sh = _offline_contract("full96326")
    task = _task(contract, status="running", exit_code=None)
    task["dedupe_key"] += ":drift"

    with pytest.raises(collector.CollectorError, match="dedupe_key"):
        collector.validate_live_task(task, task_sh, contract)


def test_failure_is_ledger_only_without_artifact_or_log_get(tmp_path: Path):
    contract, task_sh = _offline_contract("thermal96324")
    fake = FakeGetClient(_task(contract, status="failed", exit_code=1), task_sh)

    event = collector.poll_once(
        contract,
        fake,  # type: ignore[arg-type]
        tmp_path,
        poll_number=4,
    )

    assert event["state"] == "failure_ledger_only"
    assert event["artifact_collection_attempted"] is False
    assert event["failure_or_timeout"] is True
    assert fake.remote_files_calls == 0
    assert fake.output_calls == 0


def test_full_success_without_remote_retention_is_pending(tmp_path: Path):
    contract, task_sh = _offline_contract("full96326")
    fake = FakeGetClient(_task(contract, status="completed", exit_code=0), task_sh)

    event = collector.poll_once(
        contract,
        fake,  # type: ignore[arg-type]
        tmp_path,
        poll_number=8,
    )

    assert event["state"] == "success_pending_collection"
    assert event["artifact_collection_attempted"] is True
    assert event["scientific_pass_claimed"] is False
    assert event["production_claimed"] is False
    assert event["pending_reasons"]
    assert fake.remote_files_calls == 1
    assert fake.output_calls == 0


def test_scheduler_client_exposes_no_post_or_mutation_surface():
    public = {name for name in dir(collector.GetOnlyClient) if not name.startswith("_")}

    assert public == {
        "get_bytes",
        "get_json",
        "remote_file",
        "remote_files",
        "task_output",
    }


def test_full_terminal_result_is_bound_to_candidate_solver_and_job():
    contract = collector.load_contract("full96326", REPO_ROOT)
    candidate = collector._validate_full_candidate(contract.command)
    result = {
        **candidate,
        "git_hash": "a1e4f70cefa1af04673c73a6131bf490c0cc14b5",
        "pyaedt_library_git_hash": (
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
        ),
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": "mft-standalone-core-16-optin-v1",
        "solver_core_backend": "standalone",
        "solver_core_license_contract": "mft-aedt-hpc-license-snapshot-v1",
        "solver_core_opt_in": 1,
        "solver_num_cores_requested": 16,
        "solver_num_cores_effective": 16,
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": 16,
        "solver_core_slurm_cpus_per_task_readback": "16",
        "solver_core_scheduler_task_id_readback": "96326",
        "solver_core_slurm_job_id_readback": "839534",
        "solver_matrix_hpc_num_cores_readback": 16,
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "a" * 64,
        "solver_core_license_snapshot_sha256": "b" * 64,
        "solver_core_auth_sha256": "c" * 64,
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_solved": 1,
        "thermal_extraction_complete": 1,
        "thermal_convergence_available": 1,
        "thermal_converged": 1,
        "thermal_required_missing_count": 0,
        "f_res_min_tx_rx_only_Hz": 15_200.0,
    }
    stdout = (
        "MFT_LIBRARY_GIT_HASH "
        "e6b9b9d20a832ff5c3f7ca97218737a0b8650781\n"
        f"RESULT_JSON {json.dumps(result, sort_keys=True)}\n"
    ).encode()

    observed, identity = collector._full_result_from_stdout(
        stdout,
        contract=contract,
        expected_slurm_job_id="839534",
    )

    assert observed == result
    assert identity["candidate_physics_sha256"] == collector.FULL_CANDIDATE_SHA256
    assert identity["slurm_job_id"] == "839534"
