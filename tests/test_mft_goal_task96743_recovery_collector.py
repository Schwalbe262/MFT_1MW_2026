from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from module.fixed_boundary_contract import FIXED_BOUNDARY_CONTRACT_SHA256
from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
)
from tools import mft_goal_final_artifact_pipeline as pipeline
from tools import mft_goal_task96743_recovery_collector as collector
from tools import mft_goal_terminal_collector as terminal


def _task(*, status: str = "running", state: str = "running") -> dict[str, Any]:
    return {
        "task_id": collector.TASK_ID,
        "id": collector.TASK_ID,
        "name": collector.TASK_NAME,
        "dedupe_key": collector.DEDUPE_KEY,
        "project": collector.SCHEDULER_PROJECT,
        "account_name": collector.ACCOUNT_NAME,
        "requested_node_name": collector.NODE_NAME,
        "requested_node_name_policy": "strict",
        "node_name": collector.NODE_NAME,
        "node_name_policy": "strict",
        "actual_node_name": collector.NODE_NAME,
        "allocation_node_name": collector.NODE_NAME,
        "assigned_allocation": collector.ALLOCATION_ID,
        "allocation_id": collector.ALLOCATION_ID,
        "slurm_job_id": collector.SLURM_JOB_ID,
        "remote_cwd": collector.REMOTE_CWD,
        "remote_dir": collector.REMOTE_DIR,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "priority": collector.PRIORITY,
        "timeout_seconds": collector.TIMEOUT_SECONDS,
        "max_workers_per_node": collector.MAX_WORKERS_PER_NODE,
        "same_node_as_task_id": 0,
        "cpus": collector.CPUS,
        "memory_mb": collector.MEMORY_MB,
        "gpus": 0,
        "placement_contract_satisfied": True,
        "strict_node_placement": True,
        "preferred_node_relaxed": False,
        "status": status,
        "state": state,
    }


def test_exact_retry_task_identity_is_fail_closed() -> None:
    assert collector.validate_task(_task())["task_id"] == 96743

    old_task = _task()
    old_task["task_id"] = 96483
    with pytest.raises(collector.RecoveryError, match="task_id"):
        collector.validate_task(old_task)

    drifted = _task()
    drifted["dedupe_key"] += ":drift"
    with pytest.raises(collector.RecoveryError, match="dedupe_key"):
        collector.validate_task(drifted)


@pytest.mark.parametrize(
    ("status", "state"),
    [
        ("queued", "queued"),
        ("running", "running"),
        ("completed", "succeeded"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ],
)
def test_supported_task_lifecycle_is_explicit(status: str, state: str) -> None:
    task = collector.validate_task(_task(status=status, state=state))
    assert collector._task_state(task) == (status, state)  # noqa: SLF001


def test_stdout_result_is_bound_to_retry_library_revision() -> None:
    stdout = (
        b"MFT_LIBRARY_GIT_HASH "
        + collector.LIBRARY_REVISION.encode()
        + b"\nRESULT_JSON {\"sequence\":1}\n"
        + b"RESULT_JSON {\"sequence\":2}\n"
    )
    assert collector._extract_result(stdout) == {"sequence": 2}  # noqa: SLF001

    wrong = stdout.replace(
        collector.LIBRARY_REVISION.encode(),
        b"0" * 40,
    )
    with pytest.raises(collector.RecoveryError, match="library marker"):
        collector._extract_result(wrong)  # noqa: SLF001


def test_task96743_legacy_capacitance_is_never_final_resonance() -> None:
    truth = collector._capacitance_truth_boundary(  # noqa: SLF001
        {"f_res_min_tx_rx_only_Hz": 99_999.0},
        source={"effective_params": {}},
    )
    assert truth["legacy_two_equipotential_capacitance_present"] is True
    assert truth["legacy_two_equipotential_resonance_is_final_truth"] is False
    assert truth["actual_turn_graded_Tx_authenticated"] is False
    assert truth["actual_turn_graded_Rx_authenticated"] is False
    assert truth["separate_sweep_automatically_merged"] is False
    assert truth["winner_authority_allowed"] is False


def test_exact_remote_inventory_rejects_missing_or_extra_files() -> None:
    receipt = {"transport_chunk_count": 2}
    expected = [
        collector.ARTIFACT_PATH,
        collector.RECEIPT_PATH,
        collector.MARKER_PATH,
        f"{collector.CHUNK_DIRECTORY}/00000000.b64",
        f"{collector.CHUNK_DIRECTORY}/00000001.b64",
    ]
    assert collector._inventory_contract(expected, receipt) == sorted(  # noqa: SLF001
        expected
    )
    with pytest.raises(terminal.TransientGetError, match="incomplete"):
        collector._inventory_contract(expected[:-1], receipt)  # noqa: SLF001
    with pytest.raises(terminal.TransientGetError, match="incomplete"):
        collector._inventory_contract(  # noqa: SLF001
            [*expected, f"{collector.RETAINED_ROOT}/unexpected.txt"],
            receipt,
        )


def test_chunk_reconstruction_binds_exact_sha_and_size(tmp_path: Path) -> None:
    raw = b"task96743-authenticated-aedt"
    encoded = base64.b64encode(raw)

    class FakeClient:
        def remote_file(
            self,
            task_id: int,
            path: str,
            *,
            max_bytes: int,
        ) -> bytes:
            assert task_id == collector.TASK_ID
            assert path == f"{collector.CHUNK_DIRECTORY}/00000000.b64"
            assert max_bytes == collector.MAX_ENCODED_CHUNK_BYTES
            return encoded

    receipt = {
        "transport_chunk_count": 1,
        "artifact_size_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    artifact, inventory = collector._reconstruct_chunks(  # noqa: SLF001
        client=FakeClient(),
        receipt=receipt,
        staging=tmp_path,
    )
    assert artifact.read_bytes() == raw
    assert inventory == [
        pipeline.file_record(
            tmp_path / "symmetric.aedt.chunks" / "00000000.b64",
            relative_to=tmp_path,
        )
    ]


def test_task96743_cannot_bridge_legacy_cap_to_final_artifact_authority(
    tmp_path: Path,
) -> None:
    collection = {
        "result_attestation": {
            "measured_hard_constraints_passed": True,
            "graded_capacitance_provenance_authenticated": False,
        }
    }
    assert collector._winner_authority(  # noqa: SLF001
        root=tmp_path,
        collection=collection,
        source={"effective_params": {}},
    ) is None


def test_nonpassing_collection_cannot_emit_winner_authority(
    tmp_path: Path,
) -> None:
    assert (
        collector._winner_authority(  # noqa: SLF001
            root=tmp_path,
            collection={
                "result_attestation": {
                    "measured_hard_constraints_passed": False
                }
            },
            source={"effective_params": {}},
        )
        is None
    )


def test_collector_exposes_get_only_scheduler_surface() -> None:
    public = {
        name
        for name in dir(terminal.GetOnlyClient)
        if not name.startswith("_")
    }
    assert public == {
        "get_bytes",
        "get_json",
        "remote_file",
        "remote_files",
        "task_output",
    }
    source = Path(collector.__file__).read_text(encoding="utf-8")
    assert "requests.post" not in source
    assert "urlopen(Request" not in source
    assert "scheduler_mutation_performed\": False" in source


def test_exact_contract_constants_are_task96743_specific() -> None:
    assert collector.TASK_ID == 96743
    assert collector.TASK_NAME.endswith("-r1")
    assert collector.SOLVER_REVISION in collector.DEDUPE_KEY
    assert collector.LIBRARY_REVISION in collector.DEDUPE_KEY
    assert collector.PARAMETER_DIGEST in collector.DEDUPE_KEY
    assert collector.TASK_SH_SHA256 == (
        "05b4c71d9a085b8be582bd2c1b8f3386d92c7d23b41809f2b85af75f272a6cd1"
    )
    assert collector.MARKER_CONTRACT_SHA256 == (
        "0aa32db17cefe2493d2f7f91d346129cf3039f7cb27d1bb3fea55cb9b6df060e"
    )
    assert collector.FIXED_BOUNDARY_CONTRACT_SHA256 == (
        FIXED_BOUNDARY_CONTRACT_SHA256
    )
    assert collector.GOAL_CONTRACT_SCHEMA == GOAL_CONTRACT_SCHEMA
    assert collector.GOAL_STAGE_SPEC_SHA256 == GOAL_STAGE_SPEC_SHA256
    assert collector.GOAL_TEMPERATURE_CONTRACT_SHA256 == (
        GOAL_TEMPERATURE_CONTRACT_SHA256
    )
    assert collector.FIXED_OPERATING_IDENTITY_SHA256 == (
        FIXED_OPERATING_IDENTITY_SHA256
    )
    assert collector.FIXED_COOLING_IDENTITY_SHA256 == (
        FIXED_COOLING_IDENTITY_SHA256
    )


def test_live_sealed_sources_and_task_script_reauthenticate_when_present() -> None:
    if not all(
        path.is_file()
        for path in (
            collector.CAMPAIGN_PATH,
            collector.TUNED_MANIFEST_PATH,
            collector.PARAMS_PATH,
            collector.PROFILE_PATH,
        )
    ):
        pytest.skip("task96743 immutable source evidence is not mounted")
    source = collector.load_source_contract()
    assert source["effective_params"]["cw1"] == 5.0
    assert source["effective_params"]["gap1"] == 1.6
    assert source["effective_params"]["core_center_gap_mm"] == 0.860423
    assert source["effective_params"]["matrix_max_passes"] == 20
    assert source["effective_params"]["matrix_percent_error"] == 1.5
    assert source["decoded_params"]["N1_main"] == 8
    assert (
        int(source["decoded_params"]["N2_main"])
        + int(source["decoded_params"]["N2_side"])
        == 80
    )
    task_sh = (
        collector.DEFAULT_OUTPUT / "recovery_source" / "task.sh"
    )
    if not task_sh.is_file():
        pytest.skip("live task96743 task.sh has not been GET-sealed locally")
    validation = collector.validate_task_script(
        task_sh.read_bytes(),
        effective_params=source["effective_params"],
    )
    assert validation["cancelled_task96483_authority_reused"] is False
