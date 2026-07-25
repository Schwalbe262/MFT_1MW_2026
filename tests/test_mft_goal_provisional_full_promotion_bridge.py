from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from module.input_parameter_260706 import ALL_INPUT_KEYS
from module.mft_goal_20260726_contract import canonical_sha256
from regression_260707.verify import scheduler_client
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_provisional_full_promotion_bridge as bridge
from tests import test_mft_goal_provisional_full_collector as collector_test


SOLVER = "a" * 40
LIBRARY = "b" * 40
CANDIDATE = "c" * 64
STANDARD_RESULT = "d" * 64
COLLECTION_PAYLOAD = "e" * 64


def test_module_contains_no_scheduler_or_claim_mutator() -> None:
    source = Path(bridge.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "submit_verification",
        "cancel_task",
        "cancel",
        "post",
        "patch",
        "delete",
        "unlink",
        "acquire_claim",
        "finalize_claim",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)


def _touch(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _strict_standard_execution() -> dict[str, object]:
    return {
        "task_id": 96304,
        "name": "standard",
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "failure_message": "",
        "slurm_job_id": "824575",
        "allocation_id": 14492,
        "account_name": "r1jae262",
        "actual_node_name": "n114",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": "standard-dedupe",
        "remote_cwd": "runs",
        "remote_dir": "task-96304",
        "finished_at": "2026-07-26T05:00:00+09:00",
        "scheduling_profile": "fea_bursty",
        "node_name": "n114",
        "requested_node_name": "n114",
        "node_name_policy": "strict",
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
        "assigned_allocation": 14492,
        "allocation_node_name": "n114",
        "requested_account_name": "r1jae262",
        "same_node_as_task_id": 0,
        "started_at": "2026-07-25T18:00:00+09:00",
    }


def _profile() -> dict[str, object]:
    return {
        "schema_version": "profile",
        "stage": "full",
        "comment": "test",
        "reviewed_solver_path": "run --full",
        "cli_flags": "--thermal --headless --full",
        "param_overrides": {
            "full_model": 1,
            "thermal_symmetry": "full",
            "fan_velocity": 1.5,
        },
        "fixed_boundary_contract": {
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "thermal_pad_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
        },
        "artifact_retention": {"stage": "full"},
        "mem_mb": 98304,
        "cpus": 16,
        "timeout_seconds": 43200,
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    standard_plan_path = _touch(tmp_path / "standard_plan.json", "standard")
    standard_collection_path = _touch(
        tmp_path / "standard_collection.json", "collection"
    )
    standard_collection_record = production._file_record(standard_collection_path)
    nds_authority = [
        {
            "logical_authority_task_id": 96230,
            "collection_payload_sha256": standard_collection_record["sha256"],
        }
    ]
    nds_key = canonical_sha256(nds_authority)[:16]
    truth_directory = tmp_path / "truth_snapshots" / f"n01-{nds_key}"
    truth_directory.mkdir(parents=True)
    truth_manifest_path = _touch(
        truth_directory / "truth_pareto_manifest.json", "truth-manifest"
    )
    truth_manifest_record = production._file_record(truth_manifest_path)
    nds_receipt_path = truth_directory.parent / f"{truth_directory.name}.receipt.json"
    production._write_immutable_json(
        nds_receipt_path,
        production._seal(
            {
                "schema_version": bridge.terminal.NDS_RECEIPT_SCHEMA,
                "campaign_id": bridge.CAMPAIGN_ID,
                "authority_key": nds_key,
                "logical_authority_task_ids": [96230],
                "input_collections": [standard_collection_record],
                "truth_manifest": truth_manifest_record,
                "global_nondominated_sort_performed": True,
                "scheduler_mutation_performed": False,
                "created_at_utc": "2026-07-26T05:00:00+00:00",
            }
        ),
    )
    canonical_full_plan_path = _touch(
        tmp_path / "canonical_full_plan.json", "canonical-full"
    )
    provisional_success_path = _touch(
        tmp_path / "provisional_success.json", "provisional-success"
    )
    provisional_plan_path = _touch(
        tmp_path / "provisional_plan.json", "provisional-plan"
    )
    provisional_submission_path = _touch(
        tmp_path / "provisional_submission.json", "provisional-submission"
    )
    local_full_path = _touch(tmp_path / "full.aedt", "full-aedt")

    retry = {
        "schema_version": bridge.SOURCE_RETRY_SCHEMA,
        "logical_authority_task_id": 96230,
        "fixed_physics_unchanged": True,
        "timeout_change_only": True,
    }
    params = {name: 1.0 for name in ALL_INPUT_KEYS}
    params.update(
        {
            name: copy.deepcopy(value)
            for name, value in production.FIXED_PROFILE_FIELDS.items()
            if name in params
        }
    )
    params["N1_main"] = 12.0
    params["N2_main"] = 10.0
    profile = _profile()
    effective = production._effective_params(params, profile)
    standard_plan = {
        "payload_sha256": "1" * 64,
        "retry_of_timeout12h": retry,
    }
    standard_submission = {
        "task_id": 96304,
        "retry_of_timeout12h": copy.deepcopy(retry),
    }
    standard_collection = {
        "plan": production._file_record(standard_plan_path),
        "task_id": 96304,
        "scheduler_url": "http://127.0.0.1:8002",
        "candidate_physics_sha256": CANDIDATE,
        "scheduler_task_execution": _strict_standard_execution(),
        "goal_physical_spec_passed": True,
        "actual_body_probe_temperature_gate_passed": True,
        "remote_aedt_bundle_receipt": {
            "artifact_path": "symmetric.aedt",
            "artifact_sha256": "2" * 64,
            "artifact_size_bytes": 100,
        },
    }
    fixed_attestation = {
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "thermal_pad_conductivity_W_mK": 0.2,
    }
    standard_truth = {
        "collection": standard_collection_record,
        "collection_payload_sha256": COLLECTION_PAYLOAD,
        "standard_result_sha256": STANDARD_RESULT,
        "standard_task_id": 96304,
        "candidate_physics_sha256": CANDIDATE,
        "actual_dimensions_mm": {"W": 1100.0, "L": 900.0, "H": 700.0},
        "actual_volume_L": 693.0,
        "actual_constraint_margins": {"temperature_C": {"T_max_Tx": 4.0}},
        "fixed_identity_attestation": fixed_attestation,
    }
    selected = {
        "row_contract": {"fixed_identity_attestation": copy.deepcopy(fixed_attestation)}
    }
    standard_view = {
        "collection": standard_collection,
        "plan": standard_plan,
        "params": params,
        "selected": selected,
        "submission": standard_submission,
    }
    full_plan = {
        "payload_sha256": "3" * 64,
        "truth_promotion_manifest": truth_manifest_record,
        "truth_promotion_payload_sha256": "5" * 64,
        "truth_row": {
            "truth_row_sha256": "6" * 64,
            "truth_non_dominated_rank": 0,
        },
        "candidate_physics_sha256": CANDIDATE,
        "standard_collection": standard_collection_record,
        "standard_task_id": 96304,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "fea_params_sha256": canonical_sha256(params),
        "stage": {"effective_params_sha256": canonical_sha256(effective)},
    }
    source_authority = {
        "account_name": "r1jae262",
        "node_name": "n114",
        "task_id": 96304,
        "node_name_policy": "strict",
    }
    provisional_plan_record = production._file_record(provisional_plan_path)
    provisional_submission_record = production._file_record(provisional_submission_path)
    provisional_plan = {
        "payload_sha256": "7" * 64,
        "target_source_plan": production._file_record(standard_plan_path),
        "target_source_plan_payload_sha256": standard_plan["payload_sha256"],
        "source_fea_params_sha256": canonical_sha256(params),
        "candidate_physics_sha256": CANDIDATE,
        "logical_authority_task_id": 96230,
        "source_actual_standard_task_id": 96304,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "effective_full_params_sha256": canonical_sha256(effective),
        "fixed_boundary": {**bridge.FIXED_BOUNDARY, "mutable": False},
        "source_standard_execution_authority": source_authority,
        "scheduler_url": "http://127.0.0.1:8002",
        "scheduler_project": scheduler_client.MFT_PROJECT,
        "selected_account_name": "dhj02",
        "requested_node_name": "n116",
    }
    provisional_submission = {
        "payload_sha256": "8" * 64,
        "task_id": 96307,
    }
    full_task = {
        "task_id": 96307,
        "name": "provisional-full",
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "account_name": "dhj02",
        "actual_node_name": "n116",
    }
    local_full = production._file_record(local_full_path)
    remote_full = {
        "artifact_path": "full.aedt",
        "artifact_sha256": local_full["sha256"],
        "artifact_size_bytes": local_full["size_bytes"],
    }
    success = {
        "payload_sha256": "9" * 64,
        "candidate_physics_sha256": CANDIDATE,
        "task_id": 96307,
        "result_identity": {
            "solver_revision": SOLVER,
            "library_revision": LIBRARY,
        },
        "provisional_plan": provisional_plan_record,
        "submission": provisional_submission_record,
        "local_full_aedt": local_full,
    }
    provisional = {
        "success_path": provisional_success_path,
        "success": success,
        "plan": provisional_plan,
        "params": params,
        "profile": copy.deepcopy(profile),
        "submission": provisional_submission,
        "task": full_task,
        "result": {"project_name": "full"},
        "receipt": remote_full,
    }
    full_truth = {
        "candidate_physics_sha256": CANDIDATE,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "scheduler_task_id": 96307,
        "actual_dimensions_mm": {"W": 1100.0, "L": 900.0, "H": 700.0},
        "actual_volume_L": 693.0,
        "goal_physical_spec_passed": True,
        "fixed_identity_attestation": copy.deepcopy(fixed_attestation),
    }
    return {
        "canonical_full_plan_path": canonical_full_plan_path,
        "provisional_success_path": provisional_success_path,
        "full_plan": full_plan,
        "params": params,
        "profile": profile,
        "standard_view": standard_view,
        "standard_truth": standard_truth,
        "provisional": provisional,
        "full_truth": full_truth,
    }


def _patch_authorities(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, object],
) -> None:
    monkeypatch.setattr(
        bridge.promotion,
        "_load_full_plan",
        lambda _path, predictor=None: (
            data["full_plan"],
            data["params"],
            data["profile"],
            data["standard_view"],
            data["standard_truth"],
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_authenticate_provisional_success",
        lambda _path: data["provisional"],
    )
    monkeypatch.setattr(
        bridge,
        "_full_actual_truth",
        lambda **_kwargs: copy.deepcopy(data["full_truth"]),
    )


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    data = _fixture(tmp_path)
    _patch_authorities(monkeypatch, data)
    value = bridge._build_bridge(
        canonical_full_plan_path=data["canonical_full_plan_path"],
        provisional_success_path=data["provisional_success_path"],
    )
    return data, value


def test_build_bridge_is_explicit_and_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data, value = _build(tmp_path, monkeypatch)
    assert set(value) == bridge.BRIDGE_FIELDS
    assert value["schema_version"] == bridge.BRIDGE_SCHEMA
    assert value["production_eligible"] is True
    assert value["bridge_passed"] is True
    assert value["authenticated_standard_constraint_pass"] is True
    assert value["authenticated_provisional_full_constraint_pass"] is True
    assert value["rank0_standard_truth_authority"] is True
    assert value["exact_geometry_identity"] is True
    assert value["automatic_promotion"] is False
    assert value["scheduler_mutation_performed"] is False
    assert value["canonical_claim_modified"] is False
    assert value["canonical_v1_promotion_allowed"] is False
    assert value["drop_in_truth_full_collection_v1_compatible"] is False
    assert value["drop_in_truth_full_package_v1_compatible"] is False
    assert value["full_aedtresults_available"] is False
    assert value["full_model"]["aedtresults"] is None
    unsigned = dict(value)
    assert unsigned.pop("payload_sha256") == canonical_sha256(unsigned)


@pytest.mark.parametrize(
    ("standard_pass", "full_pass", "geometry_width", "eligible"),
    [
        (False, True, 1100.0, False),
        (True, False, 1100.0, False),
        (True, True, 1099.0, False),
        (True, True, 1100.0, True),
    ],
)
def test_production_eligible_requires_both_passes_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_pass: bool,
    full_pass: bool,
    geometry_width: float,
    eligible: bool,
) -> None:
    data = _fixture(tmp_path)
    data["standard_view"]["collection"]["goal_physical_spec_passed"] = standard_pass
    data["full_truth"]["goal_physical_spec_passed"] = full_pass
    data["full_truth"]["actual_dimensions_mm"]["W"] = geometry_width
    _patch_authorities(monkeypatch, data)
    value = bridge._build_bridge(
        canonical_full_plan_path=data["canonical_full_plan_path"],
        provisional_success_path=data["provisional_success_path"],
    )
    assert value["production_eligible"] is eligible
    assert value["project_only_model_package_eligible"] is eligible


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate",
        "params",
        "revision",
        "source_plan",
        "logical",
        "standard_task",
        "standard_node",
        "profile",
        "fixed",
    ],
)
def test_identity_drift_fails_closed_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    data = _fixture(tmp_path)
    if mutation == "candidate":
        data["provisional"]["plan"]["candidate_physics_sha256"] = "f" * 64
    elif mutation == "params":
        data["provisional"]["params"] = copy.deepcopy(data["params"])
        data["provisional"]["params"]["N1_main"] = 99.0
    elif mutation == "revision":
        data["provisional"]["plan"]["solver_revision"] = "f" * 40
    elif mutation == "source_plan":
        other = _touch(tmp_path / "other-plan.json", "other")
        data["standard_view"]["collection"]["plan"] = production._file_record(other)
    elif mutation == "logical":
        data["standard_view"]["plan"]["retry_of_timeout12h"][
            "logical_authority_task_id"
        ] = 96231
        data["standard_view"]["submission"]["retry_of_timeout12h"][
            "logical_authority_task_id"
        ] = 96231
    elif mutation == "standard_task":
        data["standard_view"]["collection"]["task_id"] = 96305
    elif mutation == "standard_node":
        data["standard_view"]["collection"]["scheduler_task_execution"][
            "actual_node_name"
        ] = "n115"
    elif mutation == "fixed":
        data["full_truth"]["fixed_identity_attestation"] = {"drift": True}
    else:
        data["provisional"]["profile"]["reviewed_solver_path"] = "different solver path"
    _patch_authorities(monkeypatch, data)
    with pytest.raises(bridge.HandoffContractError):
        bridge._build_bridge(
            canonical_full_plan_path=data["canonical_full_plan_path"],
            provisional_success_path=data["provisional_success_path"],
        )


def test_non_rank0_authority_is_never_production_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fixture(tmp_path)
    data["full_plan"]["truth_row"]["truth_non_dominated_rank"] = 1
    _patch_authorities(monkeypatch, data)
    value = bridge._build_bridge(
        canonical_full_plan_path=data["canonical_full_plan_path"],
        provisional_success_path=data["provisional_success_path"],
    )
    assert value["rank0_standard_truth_authority"] is False
    assert value["bridge_passed"] is False
    assert value["production_eligible"] is False


def test_existing_bridge_is_fully_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, value = _build(tmp_path, monkeypatch)
    path = tmp_path / "bridge.json"
    production._write_immutable_json(path, value)
    authenticated = bridge.authenticate_bridge(path)
    assert authenticated["production_eligible"] is True
    assert authenticated["bridge"] == value
    assert authenticated["canonical_full_plan_path"] == str(
        data["canonical_full_plan_path"].resolve()
    )
    assert authenticated["provisional_full_success_path"] == str(
        data["provisional_success_path"].resolve()
    )
    # The public ``validate`` command prints this exact view as JSON.
    production._json_bytes(authenticated)

    drifted = copy.deepcopy(value)
    drifted["production_eligible"] = False
    drifted = production._seal(
        {key: item for key, item in drifted.items() if key != "payload_sha256"}
    )
    other = tmp_path / "bridge-drifted.json"
    production._write_immutable_json(other, drifted)
    with pytest.raises(
        bridge.HandoffContractError,
        match="authority recomputation drifted",
    ):
        bridge.authenticate_bridge(other)


def test_terminal_global_nds_receipt_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fixture(tmp_path)
    _patch_authorities(monkeypatch, data)
    manifest = Path(data["full_plan"]["truth_promotion_manifest"]["path"])
    receipt_path = manifest.parent.parent / f"{manifest.parent.name}.receipt.json"
    receipt = production._read_json(receipt_path)
    unsigned = {key: item for key, item in receipt.items() if key != "payload_sha256"}
    unsigned["authority_key"] = "0" * 16
    receipt_path.write_bytes(production._json_bytes(production._seal(unsigned)))
    with pytest.raises(
        bridge.HandoffContractError,
        match="global-NDS authority mapping drifted",
    ):
        bridge._build_bridge(
            canonical_full_plan_path=data["canonical_full_plan_path"],
            provisional_success_path=data["provisional_success_path"],
        )


def test_create_does_not_write_when_authentication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _touch(tmp_path / "plan.json", "plan")
    success = _touch(tmp_path / "success.json", "success")
    output = tmp_path / "bridge.json"

    def fail(**_kwargs: object) -> dict[str, object]:
        raise bridge.HandoffContractError("blocked")

    monkeypatch.setattr(bridge, "_build_bridge", fail)
    with pytest.raises(bridge.HandoffContractError, match="blocked"):
        bridge.create_bridge(
            canonical_full_plan_path=plan,
            provisional_success_path=success,
            output=output,
        )
    assert not output.exists()


def test_source_logical_authority_requires_exact_timeout12h_lineage() -> None:
    retry = {
        "schema_version": bridge.SOURCE_RETRY_SCHEMA,
        "logical_authority_task_id": 96230,
        "fixed_physics_unchanged": True,
        "timeout_change_only": True,
    }
    assert (
        bridge._source_logical_authority(
            {"retry_of_timeout12h": retry},
            {"retry_of_timeout12h": copy.deepcopy(retry)},
        )
        == 96230
    )
    drifted = copy.deepcopy(retry)
    drifted["fixed_physics_unchanged"] = False
    with pytest.raises(bridge.HandoffContractError):
        bridge._source_logical_authority(
            {"retry_of_timeout12h": drifted},
            {"retry_of_timeout12h": copy.deepcopy(drifted)},
        )


def test_provisional_success_is_reauthenticated_from_collector_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = collector_test._fixture(tmp_path, monkeypatch)
    state, _scheduler = collector_test._success_cycle(fixture)
    assert state["status"] == "completed_provisional_full_collected"
    monkeypatch.setattr(
        bridge.scheduler_client,
        "result_matches_params",
        lambda *_args, **_kwargs: True,
    )
    success_path = (
        fixture["watch_plan_path"].parent / "provisional_full_collection_receipt.json"
    )
    authenticated = bridge._authenticate_provisional_success(success_path)
    assert authenticated["success"]["task_id"] == collector_test.TASK_ID
    assert authenticated["task"]["actual_node_name"] == collector_test.NODE
    assert (
        authenticated["receipt"]["artifact_sha256"]
        == authenticated["success"]["local_full_aedt"]["sha256"]
    )


def test_strict_standard_execution_rejects_non_strict_placement() -> None:
    source = {
        "account_name": "r1jae262",
        "node_name": "n114",
        "task_id": 96304,
        "node_name_policy": "strict",
    }
    execution = _strict_standard_execution()
    assert (
        bridge._strict_standard_execution(
            execution,
            source_authority=source,
            standard_submission={"task_id": 96304},
        )
        == execution
    )
    execution["requested_node_name_policy"] = "prefer"
    with pytest.raises(bridge.HandoffContractError):
        bridge._strict_standard_execution(
            execution,
            source_authority=source,
            standard_submission={"task_id": 96304},
        )
