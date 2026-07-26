from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import (
    mft_goal_official5_rounded_mesh_recovery_execute as executor,
)
from tools import (
    mft_goal_official5_rounded_mesh_recovery_prepare as recovery,
)


OBSERVED_AT = datetime(2026, 7, 26, 18, 0, tzinfo=timezone.utc)
REVISION = "d" * 40


def _contract() -> dict[str, Any]:
    return {
        "task_id": recovery.SOURCE_TASK_ID,
        "task_name": recovery.rounded.TASK_NAME,
        "dedupe_key": "rounded-task96340-dedupe",
        "candidate_physics_sha256": recovery.SOURCE_CANDIDATE_SHA256,
        "solver_revision": (
            "623a5345ae1b85b974cb248bcb0c8dfaadf1a867"
        ),
        "library_revision": recovery.LIBRARY_REVISION,
        "node_name": recovery.NODE_NAME,
    }


def _task(status: str = "failed") -> dict[str, Any]:
    return {
        "task_id": recovery.SOURCE_TASK_ID,
        "name": recovery.rounded.TASK_NAME,
        "dedupe_key": "rounded-task96340-dedupe",
        "project": recovery.PROJECT,
        "requested_account_name": recovery.ACCOUNT_NAME,
        "requested_node_name": recovery.NODE_NAME,
        "requested_node_name_policy": "strict",
        "cpus": recovery.CPUS,
        "memory_mb": recovery.MEMORY_MB,
        "timeout_seconds": recovery.SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
        "status": status,
        "failure_message": (
            "RESULT_JSON: thermal_extraction_failure_reason="
            "solve_not_converged:native_terminal_error"
        ),
    }


def _trigger_stream() -> bytes:
    return (
        b"Solver failed because of poor mesh quality. "
        b"Try adjusting the mesh settings.\n"
        b"Block_ABC123 is intersecting with MeshRegion "
        b"wcp_pad_mesh_region_1_in_p\n"
    )


def test_gate_accepts_only_exact_terminal_poor_mesh_wcp_intersection() -> None:
    gate = recovery.evaluate_gate(
        contract=_contract(),
        task=_task(),
        stdout=_trigger_stream(),
        stderr=b"",
        observed_at=OBSERVED_AT,
    )
    recovery.validate_seal(gate, recovery.GATE_SCHEMA)
    assert gate["prepare_allowed"] is True
    assert gate["submission_allowed"] is False
    assert gate["wcp_intersection_region_names"] == [
        "wcp_pad_mesh_region_1_in_p"
    ]
    assert gate["scheduler_get_calls"] == 3
    assert gate["scheduler_mutation_performed"] is False


@pytest.mark.parametrize(
    ("status", "stdout", "reason"),
    (
        (
            "succeeded",
            _trigger_stream(),
            "source_succeeded_recovery_forbidden",
        ),
        (
            "running",
            _trigger_stream(),
            "source_not_terminal",
        ),
        (
            "failed",
            (
                b"Block_ABC123 is intersecting with MeshRegion "
                b"wcp_pad_mesh_region_1_in_p\n"
            ),
            "failure_is_not_native_poor_mesh",
        ),
        (
            "failed",
            b"Solver failed because of poor mesh quality.\n",
            "poor_mesh_has_no_wcp_intersection_evidence",
        ),
    ),
)
def test_gate_forbids_success_active_and_other_failure_evidence(
    status: str, stdout: bytes, reason: str
) -> None:
    gate = recovery.evaluate_gate(
        contract=_contract(),
        task=_task(status),
        stdout=stdout,
        stderr=b"",
        observed_at=OBSERVED_AT,
    )
    assert gate["prepare_allowed"] is False
    assert gate["submission_allowed"] is False
    assert gate["reason"] == reason


def test_gate_fails_closed_on_exact_task_identity_drift() -> None:
    task = _task()
    task["dedupe_key"] = "wrong"
    gate = recovery.evaluate_gate(
        contract=_contract(),
        task=task,
        stdout=_trigger_stream(),
        stderr=b"",
        observed_at=OBSERVED_AT,
    )
    assert gate["prepare_allowed"] is False
    assert gate["reason"] == "source_identity_drift"
    assert set(gate["identity_drift"]) == {"dedupe_key"}


def test_profile_changes_no_candidate_or_fixed_thermal_inputs() -> None:
    profile = recovery.load_recovery_profile()
    source = recovery.read_json(
        recovery.rounded.OUTPUT_ROOT / recovery.rounded.PROFILE_NAME
    )
    assert profile["param_overrides"] == source["param_overrides"]
    assert (
        profile["fixed_boundary_contract"]
        == source["fixed_boundary_contract"]
    )
    assert profile["artifact_retention"] == source["artifact_retention"]
    mesh = profile["mesh_recovery_contract"]
    assert mesh["source_eighth_padding_mm"] == [
        0.0,
        2.0,
        0.0,
        0.0,
        2.0,
        0.0,
    ]
    assert mesh["target_eighth_padding_mm"] == [
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert mesh["expected_wcp_mesh_region_count"] == 4
    assert mesh["missing_thin_solids_required"] == 0
    assert mesh["direct_analyze_forbidden"] is True


def test_executor_authenticates_exact_params_and_rejects_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "N1_main": 6,
        "N2_main": 37,
        "N2_side": 23,
    }
    monkeypatch.setattr(
        executor, "EXPECTED_PARAMS_SHA256", executor._canonical_sha256(params)
    )
    path = tmp_path / "cand.json"
    path.write_text(json.dumps(params), encoding="utf-8")
    environ = {
        executor.OPT_IN_ENV: executor.OPT_IN_TOKEN,
        executor.PADDING_ENV: "1.0",
    }
    marker = executor.authenticate_invocation(
        ["--fixed", "--thermal", "--headless", "--params", str(path)],
        environ,
    )
    assert marker["full_model"] == 0
    assert marker["target_eighth_padding_mm"] == [
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    with pytest.raises(executor.RecoveryExecutionError):
        executor.authenticate_invocation(
            ["--full", "--params", str(path)], environ
        )


def test_executor_enforces_four_native_regions_and_mapping_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_padding = list(executor.EXPECTED_EIGHTH_PADDING_MM)
    directions = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    operations = [
        {
            "category": "wcp_pad_region",
            "level": 5,
            "objects": [f"Tx_main_wcp_pad_{index}_in_p"],
            "padding_values_mm": expected_padding,
            "padding_by_direction_mm": dict(
                zip(
                    directions,
                    expected_padding,
                )
            ),
        }
        for index in range(executor.EXPECTED_WCP_REGION_COUNT)
    ]
    plan = {
        "policy": executor.TARGET_POLICY,
        "symmetry_mode": "eighth",
        "required_objects_missing": [],
        "wcp_pad_mesh_region_count": 4,
        "operations": operations,
    }
    evidence = {
        "passed": True,
        "generate_mesh_returned": True,
        "mesh_mapping_coverage_passed": True,
        "required_objects_missing": [],
        "unmeshed_objects": [],
        "wcp_pad_mesh_region_count": 4,
        "mesh_mapping_coverage": {
            "passed": True,
            "expected_local_region_count": 4,
            "missing_local_regions": [],
            "local_regions_without_mesh": [],
            "uncoupled_local_regions": [],
            "local_region_objects_missing": [],
            "required_objects_missing": [],
        },
        "native_operation_readback": {
            "required_thin_objects_missing": [],
            "mesh_region_operation_count": 4,
            "mesh_region_part_readback_passed": True,
        },
    }
    fake_thermal = SimpleNamespace(
        THERMAL_MESH_POLICY=executor.SOURCE_POLICY,
        WCP_PAD_MESH_REGION_CONTRACT_VERSION=(
            executor.SOURCE_REGION_CONTRACT
        ),
        WCP_PAD_MESH_REGION_PADDING_MM=2.0,
        _assign_thermal_mesh=(
            lambda *_args, **_kwargs: copy.deepcopy(plan)
        ),
        _generate_and_attest_thermal_mesh=(
            lambda *_args, **_kwargs: copy.deepcopy(evidence)
        ),
    )
    fake_thermal._wcp_pad_mesh_region_padding_mm = lambda _mode: dict(
        zip(
            directions,
            (0.0, fake_thermal.WCP_PAD_MESH_REGION_PADDING_MM,
             0.0, 0.0, fake_thermal.WCP_PAD_MESH_REGION_PADDING_MM, 0.0),
        )
    )
    monkeypatch.setattr(executor, "thermal", fake_thermal)
    executor.install_recovery_policy()
    assert fake_thermal.WCP_PAD_MESH_REGION_PADDING_MM == 1.0
    assert fake_thermal._assign_thermal_mesh()["required_objects_missing"] == []
    assert fake_thermal._generate_and_attest_thermal_mesh()["passed"] is True
    broken = copy.deepcopy(evidence)
    broken["mesh_mapping_coverage"]["uncoupled_local_regions"] = ["wcp"]
    with pytest.raises(
        executor.RecoveryExecutionError, match="domain/overlap"
    ):
        executor._validate_native_preflight(broken)


def test_payload_uses_preflight_executor_and_has_no_direct_analyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = recovery.load_recovery_profile()

    def fake_derive(
        _params: dict[str, Any],
        _profile: dict[str, Any],
        revision: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assert recovery.rounded.TASK_NAME == recovery.TASK_NAME
        command = (
            f"git checkout {revision}; "
            f'export {recovery.rounded.direct.DIRECT_ENV_NAME}='
            f'"{recovery.rounded.direct.DIRECT_ENV_TOKEN}"; '
            "timeout 10800s python run_simulation_260706.py "
            "--fixed --thermal --headless "
            "--symmetry-thermal-direct-analyze --params cand.json"
        )
        return (
            {
                "name": recovery.TASK_NAME,
                "project": recovery.PROJECT,
                "account_name": recovery.ACCOUNT_NAME,
                "node_name": recovery.NODE_NAME,
                "node_name_policy": "strict",
                "cpus": recovery.CPUS,
                "memory_mb": recovery.MEMORY_MB,
                "timeout_seconds": recovery.SCHEDULER_SECONDS,
                "max_workers_per_node": recovery.MAX_WORKERS_PER_NODE,
                "aedt_backend": "standalone",
                "command": command,
            },
            {
                recovery.rounded.direct.DIRECT_ENV_NAME: (
                    recovery.rounded.direct.DIRECT_ENV_TOKEN
                )
            },
            {
                "stage": "standard",
                "artifact_path": "root/symmetric.aedt",
                "retention_required": True,
                "prune_protection_required": True,
            },
        )

    monkeypatch.setattr(
        recovery.rounded, "derive_scheduler_payload", fake_derive
    )
    payload, environment, _retained = recovery.derive_scheduler_payload(
        {}, profile, REVISION
    )
    command = payload["command"]
    assert (
        f"python {recovery.EXECUTOR_RELATIVE}" in command
    )
    assert "python run_simulation_260706.py" not in command
    assert "--symmetry-thermal-direct-analyze" not in command
    assert recovery.rounded.direct.DIRECT_ENV_NAME not in environment
    assert environment[executor.OPT_IN_ENV] == executor.OPT_IN_TOKEN
    assert environment[executor.PADDING_ENV] == "1.0"


def test_cli_has_no_submit_command() -> None:
    parser = recovery._parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {"gate", "prepare"}
