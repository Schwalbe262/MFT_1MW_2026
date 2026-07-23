import json
from types import SimpleNamespace

import pytest

from run_simulation_260706 import (
    STANDALONE_CORE_AUTH_ENV,
    STANDALONE_CORE_CONTRACT_ENV,
    STANDALONE_CORE_CONTRACT_VERSION,
    STANDALONE_CORE_COUNT_ENV,
    Simulation,
    resolve_solver_core_policy,
    standalone_core_contract_auth_sha256,
)


REVISION = "a" * 40


def _authenticated_environment(**overrides):
    environment = {
        STANDALONE_CORE_CONTRACT_ENV: STANDALONE_CORE_CONTRACT_VERSION,
        STANDALONE_CORE_COUNT_ENV: "8",
        STANDALONE_CORE_AUTH_ENV: standalone_core_contract_auth_sha256(
            REVISION, 8
        ),
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_SCHED_TASK_ID": "95010",
        "SLURM_JOB_ID": "815500",
    }
    environment.update(overrides)
    return environment


def test_default_policy_remains_capped_at_four_without_opt_in():
    policy = resolve_solver_core_policy(
        "standalone",
        environ={
            "SLURM_CPUS_PER_TASK": "64",
            "SLURM_SCHED_TASK_ID": "42",
            "SLURM_JOB_ID": "84",
        },
        affinity_count=64,
        solver_revision=REVISION,
        solver_dirty=0,
    )

    assert policy["opt_in"] is False
    assert policy["requested_num_cores"] == 4
    assert policy["effective_num_cores"] == 4
    assert policy["slurm_cpus_per_task_readback"] == "64"


def test_authenticated_standalone_policy_enables_exactly_eight_cores():
    policy = resolve_solver_core_policy(
        "standalone",
        environ=_authenticated_environment(),
        affinity_count=16,
        solver_revision=REVISION,
        solver_dirty=0,
    )

    assert policy == {
        "schema": "mft-solver-core-policy-v1",
        "contract_version": STANDALONE_CORE_CONTRACT_VERSION,
        "opt_in": True,
        "backend": "standalone",
        "requested_num_cores": 8,
        "effective_num_cores": 8,
        "num_tasks": 1,
        "affinity_count_readback": 16,
        "slurm_cpus_per_task_readback": 8,
        "scheduler_task_id_readback": 95010,
        "slurm_job_id_readback": 815500,
        "auth_sha256": standalone_core_contract_auth_sha256(REVISION, 8),
        "solver_revision": REVISION,
        "solver_dirty": 0,
    }


@pytest.mark.parametrize(
    ("backend", "environment", "affinity", "revision", "dirty", "match"),
    [
        (
            "standalone",
            {
                STANDALONE_CORE_CONTRACT_ENV:
                    STANDALONE_CORE_CONTRACT_VERSION
            },
            8,
            REVISION,
            0,
            "partial standalone core opt-in",
        ),
        (
            "pooled",
            _authenticated_environment(),
            8,
            REVISION,
            0,
            "forbidden for backend",
        ),
        (
            "standalone",
            _authenticated_environment(),
            8,
            REVISION,
            1,
            "clean committed solver revision",
        ),
        (
            "standalone",
            _authenticated_environment(
                **{STANDALONE_CORE_AUTH_ENV: "0" * 64}
            ),
            8,
            REVISION,
            0,
            "authentication digest mismatch",
        ),
        (
            "standalone",
            _authenticated_environment(SLURM_CPUS_PER_TASK="7"),
            8,
            REVISION,
            0,
            "Slurm allocation mismatch",
        ),
        (
            "standalone",
            _authenticated_environment(),
            7,
            REVISION,
            0,
            "affinity is smaller",
        ),
        (
            "standalone",
            _authenticated_environment(SLURM_SCHED_TASK_ID=""),
            8,
            REVISION,
            0,
            "positive decimal SLURM_SCHED_TASK_ID",
        ),
    ],
)
def test_opt_in_rejects_every_unattested_state(
        backend, environment, affinity, revision, dirty, match):
    with pytest.raises(RuntimeError, match=match):
        resolve_solver_core_policy(
            backend,
            environ=environment,
            affinity_count=affinity,
            solver_revision=revision,
            solver_dirty=dirty,
        )


def test_standalone_dispatch_passes_exact_eight_core_argument():
    calls = []

    class _Setup:
        @staticmethod
        def analyze(**kwargs):
            calls.append(kwargs)
            return None

    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "standalone"
    simulation.NUM_CORE = 8
    simulation.NUM_TASK = 1
    simulation.solve_attempts = {"matrix": 0}
    simulation.stage_timings = {}
    simulation.solver_core_dispatch_evidence = {}
    simulation.design1 = SimpleNamespace(setup=_Setup())
    simulation.save_project = lambda *args, **kwargs: True
    simulation._log_recent_aedt_messages = lambda _label: None

    simulation.analyze_and_extract("matrix", lambda: None)

    assert calls == [{"cores": 8}]
    assert simulation.solver_core_dispatch_evidence["matrix"] == {
        "schema": "mft-solver-core-dispatch-v1",
        "stage": "matrix",
        "dispatch": "pyaedt_setup_analyze",
        "backend": "standalone",
        "cores_argument": 8,
        "tasks_argument": None,
        "gpus_argument": None,
    }


def test_acf_and_result_telemetry_preserve_exact_core_readback(tmp_path):
    acf = tmp_path / "pyaedt_config.acf"
    acf.write_text(
        "\n".join([
            "$begin 'DSOConfig'",
            "ConfigName='pyaedt_config'",
            "DesignType='Maxwell 3D'",
            "MachineName='localhost'",
            "NumEngines=1",
            "NumCores=8",
            "NumGPUs=0",
            "UseAutoSettings=True",
            "$end 'DSOConfig'",
        ]) + "\n",
        encoding="utf-8",
    )
    policy = resolve_solver_core_policy(
        "standalone",
        environ=_authenticated_environment(),
        affinity_count=8,
        solver_revision=REVISION,
        solver_dirty=0,
    )
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "standalone"
    simulation.NUM_CORE = 8
    simulation.NUM_TASK = 1
    simulation.solver_core_policy = policy
    simulation.solver_core_dispatch_evidence = {
        "matrix": {
            "schema": "mft-solver-core-dispatch-v1",
            "stage": "matrix",
            "cores_argument": 8,
        }
    }
    simulation.solver_core_readback_evidence = {}
    simulation.solve_attempts = {}
    simulation.extraction_attempts = {}
    simulation.extraction_backends = {}

    assert simulation._validated_matrix_hpc_acf(acf) == str(acf)
    telemetry = simulation.get_execution_telemetry().iloc[0]

    assert telemetry["solver_core_opt_in"] == 1
    assert telemetry["solver_num_cores_requested"] == 8
    assert telemetry["solver_num_cores_effective"] == 8
    assert telemetry["solver_core_slurm_cpus_per_task_readback"] == "8"
    assert telemetry["solver_matrix_hpc_num_cores_readback"] == 8
    assert telemetry["solver_matrix_hpc_num_engines_readback"] == 1
    assert len(telemetry["solver_matrix_hpc_acf_sha256"]) == 64
    dispatch = json.loads(telemetry["solver_core_dispatch_evidence_json"])
    readback = json.loads(telemetry["solver_core_readback_evidence_json"])
    assert dispatch["matrix"]["cores_argument"] == 8
    assert readback["matrix_hpc_acf"]["num_cores_readback"] == 8
