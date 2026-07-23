from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import deadline_core_sibling_adapter as adapter


def _json(path: Path, value: dict) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _authority_false() -> dict:
    return {
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutated": False,
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
        "final_design_approved": False,
    }


def _core_contract(task_id: int) -> dict:
    spec = adapter.CANDIDATES[task_id]
    value = {
        "schema": "mft-solver-core-policy-v1",
        "contract_version": spec["core_contract_version"],
        "backend": "standalone",
        "opt_in": True,
        "requested_num_cores": spec["cores"],
        "effective_num_cores": spec["cores"],
        "affinity_count_readback": spec["cores"],
        "slurm_cpus_per_task_readback": spec["cores"],
        "scheduler_task_id_readback": task_id,
        "solver_revision": spec["runtime_revision"],
        "solver_dirty": 0,
        "auth_sha256": spec["core_auth"],
    }
    if spec["cores"] == 16:
        value.update({
            "license_contract": "mft-aedt-hpc-license-snapshot-v1",
            "license_snapshot_sha256":
                spec["license_snapshot_sha256"],
        })
    return value


def _dispatch(task_id: int) -> list[dict]:
    return [{
        "schema": "mft-solver-core-dispatch-v1",
        "backend": "standalone",
        "cores_argument": adapter.CANDIDATES[task_id]["cores"],
        "stage": "matrix",
        "dispatch": "pyaedt_setup_analyze",
    }]


@pytest.mark.parametrize("task_id", [95016, 95030])
def test_core_contract_accepts_exact_8_and_16_core_readback(task_id):
    spec = adapter.CANDIDATES[task_id]
    adapter._validate_core_contract(_core_contract(task_id), task_id, spec)
    adapter._validate_dispatches(_dispatch(task_id), spec)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("effective_num_cores", 1),
        ("slurm_cpus_per_task_readback", 1),
        ("scheduler_task_id_readback", 95009),
        ("solver_revision", adapter.PHYSICS_SOLVER_REVISION),
        ("solver_dirty", 1),
    ],
)
def test_core_contract_fails_closed_on_runtime_drift(field, replacement):
    task_id = 95030
    value = _core_contract(task_id)
    value[field] = replacement
    with pytest.raises(RuntimeError, match="runtime core contract drifted"):
        adapter._validate_core_contract(
            value, task_id, adapter.CANDIDATES[task_id]
        )


def test_standard_pair_allowlist_is_exact():
    assert adapter.CANDIDATES[95016]["standard_task_ids"] == {
        95014, 95015
    }
    assert 95009 not in adapter.CANDIDATES[95016]["standard_task_ids"]
    assert adapter.CANDIDATES[95019]["standard_task_ids"] == {95035}
    assert adapter.CANDIDATES[95030]["standard_task_ids"] == {95022}
    assert adapter.CANDIDATES[95032]["standard_task_ids"] == {95031, 95043}
    assert adapter.CANDIDATES[95033]["standard_task_ids"] == {95037}
    assert adapter.CANDIDATES[95039]["standard_task_ids"] == {95034}
    assert adapter.CANDIDATES[95040]["standard_task_ids"] == {95038}
    assert adapter.CANDIDATES[95042]["standard_task_ids"] == {95041}


def test_revision_attestation_matches_exact_reviewed_chain():
    repo = Path(__file__).resolve().parents[1]
    evidence = adapter.build_revision_attestation(repo)
    assert evidence["revision_chain_valid"] is True
    assert evidence[
        "physical_model_or_result_extraction_change"
    ] is False
    assert [
        set(edge["changed_paths"]) for edge in evidence["edges"]
    ] == [
        {
            "run_simulation_260706.py",
            "tests/test_standalone_core_optin.py",
        },
        {
            "run_simulation_260706.py",
            "tests/test_standalone_core_optin.py",
        },
    ]
    assert evidence["payload_sha256"] == adapter._payload_sha256(evidence)


def test_revision_attestation_rejects_rehashed_edge_tamper(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    evidence = adapter.build_revision_attestation(repo)
    evidence["edges"][0]["changed_paths"]["physical_model.py"] = "M"
    evidence["payload_sha256"] = adapter._payload_sha256(evidence)
    path = _json(tmp_path / "tampered-revision.json", evidence)
    with pytest.raises(
        RuntimeError, match="revision ancestry/diff attestation drifted"
    ):
        adapter._validate_revision_attestation(
            path, adapter._sha256(path)
        )


def _terminal_fixture(tmp_path: Path, task_id: int):
    spec = adapter.CANDIDATES[task_id]
    plan = _json(tmp_path / "plan.json", {"fixture": "plan"})
    receipt = _json(tmp_path / "receipt.json", {"fixture": "receipt"})
    start = _json(tmp_path / "start.json", {"fixture": "start"})
    source = _json(tmp_path / "source.json", {"fixture": "source"})
    native_result = {
        "git_hash": spec["runtime_revision"],
        "git_dirty": 0,
        "pyaedt_library_git_hash": adapter.LIBRARY_REVISION,
        "pyaedt_library_git_dirty": 0,
        "solver_core_contract_version": spec["core_contract_version"],
        "solver_num_cores_effective": spec["cores"],
        "solver_core_scheduler_task_id_readback": task_id,
        "untouched_physical_value": 123.456,
    }
    stdout = "native stdout\n"
    stderr = ""
    capture = {
        "schema_version": adapter.TERMINAL_CAPTURE_SCHEMA,
        "task_id": task_id,
        "task_name": spec["task_name"],
        "task_status": "completed",
        "exit_code": 0,
        "failure_message": "",
        "plan": adapter._file_reference(plan),
        "source_submission_receipt": adapter._file_reference(source),
        "canonical_adapter_receipt": adapter._file_reference(receipt),
        "start_evidence": adapter._file_reference(start),
        "candidate_identity_sha256": spec["identity"],
        "candidate_digest": spec["digest"],
        "physics_solver_revision": adapter.PHYSICS_SOLVER_REVISION,
        "runtime_solver_revision": spec["runtime_revision"],
        "library_revision": adapter.LIBRARY_REVISION,
        "core_contract_valid": True,
        "core_contract": _core_contract(task_id),
        "dispatch_records": _dispatch(task_id),
        "result_count": 1,
        "result_runtime_identity_valid": True,
        "result": native_result,
        "collection_errors": [],
        "collection_valid": True,
        "actual_hard_spec_pass": False,
        "actual_hard_spec_evaluation_pending": True,
        "stdout": {
            "sha256": adapter._sha256_bytes(stdout.encode()),
            "size_bytes": len(stdout.encode()),
            "value": stdout,
        },
        "stderr": {
            "sha256": adapter._sha256_bytes(stderr.encode()),
            "size_bytes": len(stderr.encode()),
            "value": stderr,
        },
        **_authority_false(),
    }
    capture_path = _json(tmp_path / "capture.json", capture)
    return capture, capture_path, plan, receipt, start, native_result


@pytest.mark.parametrize("task_id", [95016, 95030])
def test_terminal_capture_preserves_native_result(task_id, tmp_path):
    capture, path, plan, receipt, start, native = _terminal_fixture(
        tmp_path, task_id
    )
    observed = adapter._validate_terminal_capture(
        capture,
        path,
        plan,
        receipt,
        start,
        task_id,
        adapter.CANDIDATES[task_id],
    )
    assert observed == native
    assert observed["untouched_physical_value"] == 123.456


def test_terminal_capture_rejects_missing_result(tmp_path):
    task_id = 95016
    capture, path, plan, receipt, start, _ = _terminal_fixture(
        tmp_path, task_id
    )
    capture["result_count"] = 0
    capture["result"] = None
    capture["collection_valid"] = False
    with pytest.raises(RuntimeError, match="not clean and complete"):
        adapter._validate_terminal_capture(
            capture,
            path,
            plan,
            receipt,
            start,
            task_id,
            adapter.CANDIDATES[task_id],
        )


def _canonical_bundle(tmp_path: Path, task_id=95016):
    spec = adapter.CANDIDATES[task_id]
    native = {
        "git_hash": spec["runtime_revision"],
        "native_physics": {"Llt": 27.5, "loss": 4321.0},
    }
    plan_path = _json(tmp_path / "plan.json", {"plan": 1})
    receipt_path = _json(tmp_path / "receipt.json", {"receipt": 1})
    start_path = _json(tmp_path / "start.json", {"start": 1})
    revision_path = _json(tmp_path / "revision.json", {"revision": 1})
    capture_path = _json(tmp_path / "capture.json", {"capture": 1})
    scheduler_client = SimpleNamespace(
        effective_verification_params=lambda params, profile: params,
        result_matches_params=lambda result, effective, required_keys: True,
    )
    gate = SimpleNamespace(
        _load_submission_contract=lambda root: (
            ("cw1",),
            lambda result, **kwargs: SimpleNamespace(full_valid=True),
            lambda result: (1.0, (1.0, 1.0, 1.0)),
            scheduler_client,
        ),
        _read_json=lambda path: {"param_overrides": {}},
        _profile_contract=lambda profile, fidelity: None,
        _solver_result_variant_contract=lambda plan, result: True,
        _actual_hard_gate=lambda *args: {"pass": True, "reasons": []},
    )
    plan = {
        "plan_kind": "diagnostic-near-truth",
        "candidate_surrogate_hard_feasible": False,
        "candidate": {
            "decoded_params": {"cw1": 5.0},
        },
        "solver_contract": {
            "fine_profile": {"path": str(tmp_path / "fine.json")},
        },
        "target": {},
        "source": {"temperature_targets": []},
        "_resonance_contract_kind": "final-minimum-15khz",
        "_tim_solver_contract_kind": "corrected-native-readback",
        "_fan_velocity_profile_identity": {"matches": True},
    }
    capture = {
        "core_contract_valid": True,
        "result_runtime_identity_valid": True,
        "collection_valid": True,
    }
    return {
        "gate": gate,
        "plan": plan,
        "plan_path": plan_path,
        "receipt_path": receipt_path,
        "start_path": start_path,
        "revision_path": revision_path,
        "capture_path": capture_path,
        "capture": capture,
        "result": native,
        "task_id": task_id,
        "spec": spec,
    }


def test_canonicalize_copies_native_result_without_edit(
    monkeypatch, tmp_path
):
    bundle = _canonical_bundle(tmp_path)
    native_before = copy.deepcopy(bundle["result"])
    monkeypatch.setattr(
        adapter, "_validate_start_bundle", lambda args, require_terminal: bundle
    )
    execution = tmp_path / "runtime-execution-attestation.json"
    output = tmp_path / "full-canonical-result.json"
    args = SimpleNamespace(
        submission_code_root=str(tmp_path),
        execution_attestation_output=str(execution),
        output=str(output),
    )
    assert adapter.command_canonicalize_full(args) == 0
    canonical = json.loads(output.read_text(encoding="utf-8"))
    assert canonical["result"] == native_before
    assert bundle["result"] == native_before
    assert canonical["actual_hard_spec_pass"] is True
    assert canonical["runtime_execution_evidence"] == (
        adapter._file_reference(execution)
    )
    assert canonical["automatic_promotion_allowed"] is False
    assert canonical["final_design_approved"] is False


def test_canonicalize_refuses_nonpass_actual_gate(monkeypatch, tmp_path):
    bundle = _canonical_bundle(tmp_path)
    bundle["gate"]._actual_hard_gate = lambda *args: {
        "pass": False,
        "reasons": ["temperature_out_of_spec"],
    }
    monkeypatch.setattr(
        adapter, "_validate_start_bundle", lambda args, require_terminal: bundle
    )
    output = tmp_path / "full-canonical-result.json"
    args = SimpleNamespace(
        submission_code_root=str(tmp_path),
        execution_attestation_output=str(
            tmp_path / "runtime-execution-attestation.json"
        ),
        output=str(output),
    )
    with pytest.raises(RuntimeError, match="failed actual hard gate"):
        adapter.command_canonicalize_full(args)
    assert not output.exists()


def test_output_is_write_once(tmp_path):
    output = _json(tmp_path / "sealed.json", {"already": "present"})
    with pytest.raises(RuntimeError, match="refusing to replace"):
        adapter._atomic_write_once(output, {"replacement": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "already": "present"
    }


def _join_args(monkeypatch, tmp_path, standard_task_id):
    task_id = 95016
    spec = adapter.CANDIDATES[task_id]
    plan_path = _json(tmp_path / "join-plan.json", {"fixture": "plan"})
    standard_path = _json(
        tmp_path / "standard-result.json", {"fixture": "standard"}
    )
    full_path = _json(tmp_path / "full-result.json", {"fixture": "full"})
    plan = {
        "candidate": {
            "candidate_identity_sha256": spec["identity"],
            "candidate_digest": spec["digest"],
        },
        "solver_contract": {
            "solver_variant": "deadline-tim-k3",
            "thermal_pad_conductivity_W_mK": 3.0,
            "thermal_pad_material_policy":
                "deadline_tim_k3_native_attested_3WmK_"
                "electrically_insulating_v2",
            "thermal_pad_native_readback_contract_version":
                "thermal-pad-native-material-readback-v1",
        },
    }
    standard = {
        "task_id": standard_task_id,
        "task_status": "completed",
        "result_state": "valid",
        "actual_hard_spec_gate": {"pass": True},
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
    }
    full = {
        "task_id": task_id,
        "runtime_execution_evidence": {
            "path": "immutable",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    }
    execution = {"collection_valid": True}
    gate = SimpleNamespace(
        _validate_plan=lambda path, sha: plan,
        _validate_result_evidence=lambda path, sha, value: standard,
    )
    monkeypatch.setattr(adapter, "_load_deadline_gate", lambda root: gate)
    monkeypatch.setattr(
        adapter,
        "_validate_canonical_full",
        lambda path, sha, value: (full, execution),
    )
    return SimpleNamespace(
        plan=str(plan_path),
        plan_sha256=adapter._sha256(plan_path),
        standard_result=str(standard_path),
        standard_result_sha256=adapter._sha256(standard_path),
        full_result=str(full_path),
        full_result_sha256=adapter._sha256(full_path),
        deadline_gate_code_root=str(tmp_path),
        output=str(tmp_path / "speculative-full-join.json"),
    )


def test_join_accepts_only_exact_standard_pass_sibling(monkeypatch, tmp_path):
    args = _join_args(monkeypatch, tmp_path, 95014)
    assert adapter.command_join(args) == 0
    joined = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert joined["standard_task_id"] == 95014
    assert joined["speculative_full_task_id"] == 95016
    assert joined["exact_standard_task_pair_allowlist_pass"] is True
    assert joined["sendable_design_evidence_ready"] is True
    assert joined["final_design_approved"] is False


def test_join_rejects_nonallowlisted_standard_even_if_marked_pass(
    monkeypatch, tmp_path
):
    args = _join_args(monkeypatch, tmp_path, 95009)
    with pytest.raises(RuntimeError, match="exact allowed PASS sibling"):
        adapter.command_join(args)
    assert not Path(args.output).exists()
