from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from module import mft_goal_20260726_contract as goal
from tools import mft_goal_final_package_gate as gate
from tools import mft_goal_terminal_collector as terminal


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_RETRY_PLAN = (
    REPO_ROOT / "artifacts/mft_goal_postdeadline_full_retry_plan_v1.json"
)


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _record(path: Path) -> dict[str, Any]:
    return gate._file_record(path)


def _event(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["event_sha256"] = terminal._sha256(
        terminal._canonical_bytes(result)
    )
    return result


def _candidate_and_retry() -> tuple[dict[str, Any], dict[str, Any]]:
    retry = json.loads(FULL_RETRY_PLAN.read_text(encoding="utf-8"))
    candidate = terminal._validate_full_candidate(
        retry["payload"]["command"].encode("utf-8")
    )
    return candidate, retry


def _authority_plan(path: Path) -> Path:
    value = gate._seal(
        {
            "schema_version": "mft-goal-provisional-full-precompute-plan-v1",
            "campaign_id": gate.CAMPAIGN_ID,
            "candidate_physics_sha256": gate.LOGICAL_CANDIDATE_SHA256,
            "effective_full_params_sha256": gate.FULL_EFFECTIVE_PARAMS_SHA256,
            "solver_revision": gate.SOURCE_SOLVER_REVISION,
            "library_revision": gate.LIBRARY_REVISION,
            "goal_hard_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "fixed_boundary": {
                "fan_velocity_m_s": 1.5,
                "fan_config": "dual",
                "thermal_pad_conductivity_W_mK": 0.2,
                "core_plate_pad_t_mm": 2.0,
                "wcp_pad_t_mm": 2.0,
                "mutable": False,
            },
            "diagnostic_only": True,
            "production_eligible": False,
            "automatic_promotion": False,
        }
    )
    _json(path, value)
    return path


def _full_result(candidate: dict[str, Any], *, resonance_hz: float) -> dict[str, Any]:
    result = copy.deepcopy(candidate)
    result.update(
        {
            "git_hash": gate.SOURCE_SOLVER_REVISION,
            "pyaedt_library_git_hash": gate.LIBRARY_REVISION,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
            "thermal_solved": 1,
            "thermal_extraction_complete": 1,
            "thermal_convergence_available": 1,
            "thermal_converged": 1,
            "thermal_required_missing_count": 0,
            "f_res_min_tx_rx_only_Hz": resonance_hz,
            "T_max_Tx": 92.0,
            "T_max_Rx_main": 93.0,
            "T_max_Rx_side": 94.0,
            "T_max_core": 111.0,
            "solver_core_scheduler_task_id_readback": str(gate.FULL_TASK_ID),
            "solver_core_slurm_job_id_readback": "839534",
        }
    )
    return result


def _native_fixed() -> dict[str, Any]:
    return {
        "schema": "mft-corrected-thermal-native-fixed-readback-v2",
        "passed": True,
        "fan_boundary": {
            "name": "fan_inlet",
            "velocity_vector_m_per_s": {"X": 0.0, "Y": -1.5, "Z": 0.0},
            "passed": True,
        },
        "tim_material": {"thermal_conductivity_W_mK": 0.2},
        "geometry": {
            "passed": True,
            "dimensions_mm": {
                "width_x": 1186.354,
                "length_y": 970.822,
                "height_z": 698.0,
            },
            "physical_pad_families": {
                "core_plate": ["core_plate_pad_1"],
                "wcp": ["Tx_main_wcp_pad_1"],
            },
            "physical_pad_solids": {
                "core_plate_pad_1": {
                    "family": "core_plate",
                    "material": "thermal_pad",
                    "y_thickness_mm": 2.0,
                },
                "Tx_main_wcp_pad_1": {
                    "family": "wcp",
                    "material": "thermal_pad",
                    "y_thickness_mm": 2.0,
                },
            },
        },
    }


def _thermal_result(*, candidate_sha256: str | None = None) -> dict[str, Any]:
    temperatures = {
        "T_max_Tx": 94.0,
        "T_max_Rx_main": 95.0,
        "T_max_Rx_side": 96.0,
        "T_max_core": 112.0,
    }
    return {
        "schema": "mft-corrected-thermal-diagnostic-result-v1",
        "diagnostic_only": True,
        "canonical": False,
        "source_provenance": {
            "candidate_sha256": (
                candidate_sha256 or gate.LOGICAL_CANDIDATE_SHA256
            ),
            "solver_revision": gate.SOURCE_SOLVER_REVISION,
            "library_revision": gate.LIBRARY_REVISION,
        },
        "executor_provenance": {
            "imported": {
                "executor_solver_revision": gate.THERMAL_EXECUTOR_REVISION,
                "executor_solver_dirty": 0,
                "pyaedt_library_revision": gate.LIBRARY_REVISION,
                "pyaedt_library_dirty": 0,
            }
        },
        "constraint_observation": {
            "winding_max_c": 96.0,
            "winding_limit_c": 100.0,
            "winding_pass": True,
            "core_max_c": 112.0,
            "core_limit_c": 120.0,
            "core_pass": True,
            "all_temperature_constraints_pass": True,
            "diagnostic_only": True,
        },
        "temperatures": temperatures,
        "temperature_extraction": {"passed": True},
        "convergence": {"thermal_converged": 1},
        "parallel_attestation": {"passed": True},
        "native_fixed_readback": _native_fixed(),
    }


def _fixture(
    tmp_path: Path,
    *,
    full_state: str = "success_collected_diagnostic",
    resonance_hz: float = 15_200.0,
    thermal_candidate_sha256: str | None = None,
    include_full_result: bool = True,
) -> dict[str, Path]:
    candidate, retry = _candidate_and_retry()
    full_root = tmp_path / "full96326"
    full_collection = full_root / "collection"
    full_collection.mkdir(parents=True)
    full_aedt = full_collection / "full.aedt"
    full_aedt.write_bytes(b"actual solver full AEDT")
    full_result_path = full_collection / "full_result.json"
    full_result = _full_result(candidate, resonance_hz=resonance_hz)
    _json(full_result_path, full_result)
    full_result_record = {
        **_record(full_result_path),
        "identity": {
            "candidate_physics_sha256": gate.FULL_EFFECTIVE_PARAMS_SHA256,
            "solver_revision": gate.SOURCE_SOLVER_REVISION,
            "library_revision": gate.LIBRARY_REVISION,
            "scheduler_task_id": gate.FULL_TASK_ID,
            "slurm_job_id": "839534",
            "full_model": 1,
            "thermal_symmetry": "full",
            "result_payload_sha256": terminal._sha256(
                terminal._canonical_bytes(full_result)
            ),
        },
    }
    _retry_name = retry["payload"]["name"]
    _retry_dedupe = retry["payload"]["dedupe_key"]
    full_event = _event(
        {
            "schema": "mft-terminal-collector-event-v1",
            "observed_at_utc": "2026-07-26T10:00:00Z",
            "poll_number": 1,
            "target": "full96326",
            "task": {
                "task_id": gate.FULL_TASK_ID,
                "name": _retry_name,
                "dedupe_key": _retry_dedupe,
                "slurm_job_id": "839534",
            },
            "contract": {
                "task_identity_verified": True,
                "fixed_physics_verified": True,
                "strict_node_verified": True,
                "local_evidence": {"plan": _record(FULL_RETRY_PLAN)},
            },
            "scheduler_state": "success",
            "state": full_state,
            "artifact_collection_attempted": (
                full_state == "success_collected_diagnostic"
            ),
            "failure_or_timeout": False,
            "reassembled_full_aedt": _record(full_aedt),
            "full_result_truth": (
                full_result_record if include_full_result else None
            ),
            "aedtresults_remote_paths": [
                "goal-fea-retained/x/full.aedtresults/file"
            ],
            "receipt": {
                "stage": "full",
                "solver_revision": gate.SOURCE_SOLVER_REVISION,
                "library_revision": gate.LIBRARY_REVISION,
                "parameter_digest": gate.FULL_EFFECTIVE_PARAMS_SHA256[:16],
                "artifact_sha256": _record(full_aedt)["sha256"],
                "artifact_size_bytes": _record(full_aedt)["size_bytes"],
            },
            **terminal.CLASSIFICATION,
        }
    )
    full_event_path = full_root / "latest.json"
    _json(full_event_path, full_event)

    thermal_root = tmp_path / "thermal96324"
    thermal_collection = thermal_root / "collection"
    artifacts = thermal_collection / "artifacts"
    artifacts.mkdir(parents=True)
    symmetric = artifacts / "symmetric.aedt"
    symmetric.write_bytes(b"actual solver corrected symmetric AEDT")
    corrected_path = artifacts / "corrected_result.json"
    corrected = _thermal_result(
        candidate_sha256=thermal_candidate_sha256
    )
    _json(corrected_path, corrected)
    thermal_plan = {
        "task_identity": {
            "candidate_sha256": gate.LOGICAL_CANDIDATE_SHA256
        },
        "contract": {
            "fixed_physics": copy.deepcopy(gate.FIXED_THERMAL_BOUNDARY),
            "library_revision": gate.LIBRARY_REVISION,
        },
        "executor": {"revision": gate.THERMAL_EXECUTOR_REVISION},
        "plan_payload_sha256": "a" * 64,
    }
    thermal_plan_path = tmp_path / "thermal_plan.json"
    _json(thermal_plan_path, thermal_plan)
    rows = [
        {
            "path": "symmetric.aedt",
            "size_bytes": symmetric.stat().st_size,
            "sha256": gate.diagnostic_handoff.sha256_file(symmetric),
        },
        {
            "path": "corrected_result.json",
            "size_bytes": corrected_path.stat().st_size,
            "sha256": gate.diagnostic_handoff.sha256_file(corrected_path),
        },
    ]
    manifest_unsigned = {
        "schema": "mft-corrected-thermal-minimum-retained-package-v1",
        "diagnostic_only": True,
        "canonical": False,
        "candidate_sha256": gate.LOGICAL_CANDIDATE_SHA256,
        "source_solver_revision": gate.SOURCE_SOLVER_REVISION,
        "executor_solver_revision": gate.THERMAL_EXECUTOR_REVISION,
        "files": rows,
    }
    manifest = {
        **manifest_unsigned,
        "payload_sha256": goal.canonical_sha256(manifest_unsigned),
    }
    thermal_event = _event(
        {
            "schema": "mft-terminal-collector-event-v1",
            "observed_at_utc": "2026-07-26T10:00:00Z",
            "poll_number": 1,
            "target": "thermal96324",
            "task": {
                "task_id": gate.THERMAL_TASK_ID,
                "name": "corrected-thermal",
                "dedupe_key": "corrected-thermal-dedupe",
                "slurm_job_id": "839461",
            },
            "contract": {
                "task_identity_verified": True,
                "fixed_physics_verified": True,
                "strict_node_verified": True,
                "local_evidence": {"plan": _record(thermal_plan_path)},
            },
            "scheduler_state": "success",
            "state": "success_collected_diagnostic",
            "artifact_collection_attempted": True,
            "failure_or_timeout": False,
            "manifest": manifest,
            "collected_files": rows,
            **terminal.CLASSIFICATION,
        }
    )
    thermal_event_path = thermal_root / "latest.json"
    _json(thermal_event_path, thermal_event)
    return {
        "full_event": full_event_path,
        "thermal_event": thermal_event_path,
        "authority": _authority_plan(tmp_path / "authority.json"),
        "output": tmp_path / "output",
    }


def _run(paths: dict[str, Path]) -> dict[str, Any]:
    return gate.evaluate_and_publish(
        full_event_path=paths["full_event"],
        thermal_event_path=paths["thermal_event"],
        authority_plan_path=paths["authority"],
        full_retry_plan_path=FULL_RETRY_PLAN,
        output_root=paths["output"],
    )


def test_running_collectors_create_only_one_pending_manifest(tmp_path: Path):
    paths = _fixture(tmp_path, full_state="watching")

    result = _run(paths)

    assert result["status"] == "pending"
    assert {path.name for path in paths["output"].iterdir()} == {
        "pending_manifest.json"
    }
    assert result["final_package_created"] is False


def test_open_only_aedt_without_full_result_truth_is_never_promoted(
    tmp_path: Path,
):
    paths = _fixture(tmp_path, include_full_result=False)

    result = _run(paths)

    assert result["status"] == "authority_invalid"
    assert "Full result/AEDT retention authority is incomplete" in " ".join(
        result["pending_reasons"]
    )
    assert not (paths["output"] / gate.PACKAGE_NAME).exists()
    assert list(paths["output"].glob("**/*.aedt")) == []


def test_candidate_mismatch_is_fail_closed_with_no_model_copy(tmp_path: Path):
    paths = _fixture(tmp_path, thermal_candidate_sha256="f" * 64)

    result = _run(paths)

    assert result["status"] == "authority_invalid"
    assert not (paths["output"] / gate.PACKAGE_NAME).exists()
    assert list(paths["output"].glob("**/*.aedt")) == []


def test_actual_constraint_failure_keeps_only_pending_manifest(tmp_path: Path):
    paths = _fixture(tmp_path, resonance_hz=14_999.0)

    result = _run(paths)

    assert result["status"] == "actual_constraints_failed"
    assert {path.name for path in paths["output"].iterdir()} == {
        "pending_manifest.json"
    }
    assert "resonance" in " ".join(result["pending_reasons"])


def test_actual_pass_publishes_authenticated_models_and_truth_idempotently(
    tmp_path: Path,
):
    paths = _fixture(tmp_path)

    first = _run(paths)
    second = _run(paths)

    package = paths["output"] / gate.PACKAGE_NAME
    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert not (paths["output"] / "pending_manifest.json").exists()
    assert {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    } == gate.EXPECTED_PACKAGE_FILES
    truth = json.loads((package / "result_truth.json").read_text(encoding="utf-8"))
    assert truth["all_explicit_goal_constraints_actual_pass"] is True
    assert truth["constraints"]["resonance"]["actual_hz"] == 15_200.0
    assert truth["constraints"]["winding_temperature"]["actual_maximum_c"] == 96.0
    assert truth["open_only_diagnostic_snapshot_promoted"] is False


def test_watch_rechecks_mutating_latest_events_without_busy_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = {
        name: tmp_path / f"{name}.json"
        for name in ("full", "thermal", "authority", "retry")
    }
    for path in inputs.values():
        path.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "pending"}

    monkeypatch.setattr(gate, "evaluate_and_publish", evaluate)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)

    status = gate.main(
        [
            "--full-event",
            str(inputs["full"]),
            "--thermal-event",
            str(inputs["thermal"]),
            "--candidate-authority-plan",
            str(inputs["authority"]),
            "--full-retry-plan",
            str(inputs["retry"]),
            "--output-root",
            str(tmp_path / "output"),
            "--watch",
            "--interval-seconds",
            "7",
            "--max-cycles",
            "2",
        ]
    )

    assert status == 2
    assert len(calls) == 2
    assert sleeps == [7]
