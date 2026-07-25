from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

from module.mft_goal_20260726_contract import (
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    canonical_sha256,
)
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_provisional_full_package_v2 as package
from tools import mft_goal_terminal_success_watcher as terminal
from tools import mft_goal_truth_promotion as promotion


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    return production._write_immutable_json(path, value)


def _authority(
    root: Path,
    *,
    nds_binds_standard: bool = True,
) -> dict[str, Any]:
    candidate = "a" * 64
    truth_row = {
        "candidate_physics_sha256": candidate,
        "truth_non_dominated_rank": 0,
    }
    truth_row["truth_row_sha256"] = canonical_sha256(truth_row)

    standard_path = _write_json(
        root / "standard_collection.source.json",
        {
            "scheduler_url": "http://127.0.0.1:8002",
            "task_id": 96304,
        },
    )
    standard_record = production._file_record(standard_path)
    full_plan_path = _write_json(root / "full_plan.source.json", {"plan": 1})
    success_path = _write_json(root / "provisional_success.source.json", {"success": 1})
    full_bytes = b"provisional-full-project"
    full_path = root / "source_full.aedt"
    full_path.write_bytes(full_bytes)
    full_record = production._file_record(full_path)

    logical_id = 96230
    collection_record = standard_record
    if not nds_binds_standard:
        other_path = _write_json(root / "other_collection.json", {"other": 1})
        collection_record = production._file_record(other_path)
    authority_rows = [
        {
            "logical_authority_task_id": logical_id,
            "collection_payload_sha256": collection_record["sha256"],
        }
    ]
    authority_key = canonical_sha256(authority_rows)[:16]
    snapshot_root = root / "truth_snapshots"
    snapshot = snapshot_root / f"n01-{authority_key}"
    snapshot.mkdir(parents=True)
    csv_bytes = (
        "candidate_physics_sha256,truth_row_sha256,"
        "truth_non_dominated_rank\n"
        f"{candidate},{truth_row['truth_row_sha256']},0\n"
    ).encode()
    csv_path = snapshot / "truth_validated_pareto_front.csv"
    csv_path.write_bytes(csv_bytes)
    truth_manifest = production._seal(
        {
            "schema_version": promotion.TRUTH_MANIFEST_SCHEMA,
            "truth_validated_pareto_front": {
                "path": csv_path.name,
                "sha256": _sha(csv_bytes),
                "size_bytes": len(csv_bytes),
                "row_count": 1,
                "columns": [
                    "candidate_physics_sha256",
                    "truth_row_sha256",
                    "truth_non_dominated_rank",
                ],
            },
            "ranked_rows": [truth_row],
        }
    )
    truth_path = _write_json(snapshot / "truth_pareto_manifest.json", truth_manifest)
    truth_record = production._file_record(truth_path)
    nds_receipt = production._seal(
        {
            "schema_version": terminal.NDS_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "authority_key": authority_key,
            "logical_authority_task_ids": [logical_id],
            "input_collections": [collection_record],
            "truth_manifest": truth_record,
            "global_nondominated_sort_performed": True,
            "scheduler_mutation_performed": False,
            "created_at_utc": "2026-07-26T00:00:00Z",
        }
    )
    nds_path = _write_json(
        snapshot_root / f"n01-{authority_key}.receipt.json", nds_receipt
    )

    bridge_path = _write_json(root / "promotion_bridge.source.json", {"v": 1})
    symmetric_bytes = b"authenticated-standard-symmetric-project"
    symmetric_receipt = {
        "artifact_path": "retained/symmetric.aedt",
        "artifact_sha256": _sha(symmetric_bytes),
        "artifact_size_bytes": len(symmetric_bytes),
        "source_project_name": "MFT_goal_standard",
        "transport_chunk_directory": "retained/chunks",
        "transport_raw_chunk_bytes": 1024,
        "transport_max_encoded_chunk_bytes": 2048,
        "transport_chunk_count": 1,
        "results_path": "retained/symmetric.aedtresults",
        "results_manifest_path": "retained/results.manifest.json",
        "results_manifest_sha256": "b" * 64,
        "results_tree_sha256": "c" * 64,
        "results_file_count": 1,
        "results_size_bytes": 10,
    }
    bridge = {
        "schema_version": package.bridge_adapter.BRIDGE_SCHEMA,
        "payload_sha256": "d" * 64,
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "candidate_physics_sha256": candidate,
        "logical_authority_task_id": logical_id,
        "standard_task_id": 96304,
        "provisional_full_task_id": 96307,
        "solver_revision": "e" * 40,
        "library_revision": "f" * 40,
        "fea_params_sha256": "1" * 64,
        "effective_full_params_sha256": "2" * 64,
        "fixed_boundary": {
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "thermal_pad_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
        },
        "fixed_identity_attestation": {"sha256": "3" * 64},
        "truth_promotion_manifest": truth_record,
        "truth_promotion_payload_sha256": truth_manifest["payload_sha256"],
        "truth_nds_receipt": production._file_record(nds_path),
        "truth_nds_receipt_payload_sha256": nds_receipt["payload_sha256"],
        "truth_row_sha256": truth_row["truth_row_sha256"],
        "truth_non_dominated_rank": 0,
        "standard_collection": standard_record,
        "provisional_full_success": production._file_record(success_path),
        "canonical_full_plan": production._file_record(full_plan_path),
        "symmetric_model": {
            "retained_remote_receipt": symmetric_receipt,
            "full_model": 0,
            "thermal_symmetry": "eighth",
        },
        "full_model": {
            "local_artifact": full_record,
            "retained_remote_receipt": {"task_id": 96307},
            "full_model": 1,
            "thermal_symmetry": "full",
            "aedtresults": None,
        },
        "standard_actual_truth": {
            "goal_physical_spec_passed": True,
            "candidate_physics_sha256": candidate,
        },
        "provisional_full_actual_truth": {
            "goal_physical_spec_passed": True,
            "candidate_physics_sha256": candidate,
        },
        "authenticated_standard_constraint_pass": True,
        "authenticated_provisional_full_constraint_pass": True,
        "exact_same_candidate_parameters_revisions": True,
        "exact_fixed_cooling_identity": True,
        "exact_geometry_identity": True,
        "rank0_standard_truth_authority": True,
        "project_only_model_package_eligible": True,
        "bridge_passed": True,
        "production_eligible": True,
        "canonical_v1_promotion_allowed": False,
        "consumer_contract": {
            "schema_version": ("mft-goal-project-only-full-package-consumer-v1"),
            "bridge_reauthentication_required": True,
            "production_eligible_true_required": True,
            "allowed_primary_outputs": [
                "symmetric_model.aedt",
                "full_model.aedt",
            ],
            "full_aedtresults_must_not_be_synthesized": True,
            "truth_full_collection_v1_must_not_be_emitted": True,
            "truth_full_package_v1_must_not_be_emitted": True,
        },
        **package.bridge_adapter._static_flags(),
    }

    def authenticate_bridge(
        _path: Path, *, predictor: Any | None = None
    ) -> dict[str, Any]:
        del predictor
        return {
            "bridge": bridge,
            "production_eligible": bridge["production_eligible"],
        }

    def load_truth(
        path: Path, *, predictor: Any | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        del predictor
        value = production._read_json(path)
        return value, [copy.deepcopy(truth_row)], {}

    fetches: list[dict[str, Any]] = []

    def fetch_symmetric(**kwargs: Any) -> None:
        fetches.append(dict(kwargs))
        assert kwargs["scheduler_url"] == "http://127.0.0.1:8002"
        assert kwargs["task_id"] == 96304
        assert kwargs["expected_sha256"] == _sha(symmetric_bytes)
        kwargs["destination"].write_bytes(symmetric_bytes)

    return {
        "bridge": bridge,
        "bridge_path": bridge_path,
        "authenticate_bridge": authenticate_bridge,
        "load_truth": load_truth,
        "fetch_symmetric": fetch_symmetric,
        "fetches": fetches,
        "symmetric_bytes": symmetric_bytes,
        "full_bytes": full_bytes,
    }


def _build(authority: dict[str, Any], output: Path) -> Path:
    return package.build_package(
        bridge_receipt_path=authority["bridge_path"],
        output=output,
        bridge_authenticator=authority["authenticate_bridge"],
        truth_loader=authority["load_truth"],
        remote_fetcher=authority["fetch_symmetric"],
    )


def _validate(authority: dict[str, Any], output: Path) -> dict[str, Any]:
    return package.authenticate_package(
        output / package.MANIFEST_NAME,
        bridge_authenticator=authority["authenticate_bridge"],
        truth_loader=authority["load_truth"],
    )


def test_builds_and_authenticates_explicit_project_only_package(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    output = _build(authority, tmp_path / "package")

    assert {path.name for path in output.iterdir()} == (
        package.PAYLOAD_NAMES | {package.MANIFEST_NAME}
    )
    assert (output / "symmetric_model.aedt").read_bytes() == authority[
        "symmetric_bytes"
    ]
    assert (output / "full_model.aedt").read_bytes() == authority["full_bytes"]
    authenticated = _validate(authority, output)
    manifest = authenticated["manifest"]
    assert manifest["schema_version"] == package.PACKAGE_SCHEMA
    assert manifest["schema_version"] != promotion.PACKAGE_SCHEMA
    assert manifest["full_aedtresults_available"] is False
    assert manifest["full_aedtresults_synthesized"] is False
    assert manifest["self_contained_deep_validation_supported"] is False
    assert manifest["external_authority_required_for_deep_reauthentication"] is True
    assert manifest["package_directory_relocatable_with_external_authority"] is True
    assert manifest["standalone_cross_machine_relocatable"] is False
    assert manifest["models"]["symmetric_model"]["relative_path"] == (
        "symmetric_model.aedt"
    )
    assert manifest["models"]["full_model"]["relative_path"] == ("full_model.aedt")
    assert manifest["result_trees"]["full_model"]["aedtresults"] is None
    assert (
        manifest["actual_truth_global_nds"]["global_nondominated_sort_performed"]
        is True
    )
    assert len(authority["fetches"]) == 1


@pytest.mark.parametrize("field", ["bridge_passed", "production_eligible"])
def test_fails_closed_before_fetch_when_bridge_is_ineligible(
    tmp_path: Path, field: str
) -> None:
    authority = _authority(tmp_path / "authority")
    authority["bridge"][field] = False

    with pytest.raises(
        production.HandoffContractError,
        match="not production-eligible|not package-eligible",
    ):
        _build(authority, tmp_path / "package")
    assert not (tmp_path / "package").exists()
    assert authority["fetches"] == []


def test_fails_closed_when_nds_does_not_include_standard_authority(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority", nds_binds_standard=False)

    with pytest.raises(
        production.HandoffContractError,
        match="does not bind the Standard authority",
    ):
        _build(authority, tmp_path / "package")
    assert not (tmp_path / "package").exists()
    assert authority["fetches"] == []


def test_fails_closed_when_symmetric_retained_receipt_is_absent(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    del authority["bridge"]["symmetric_model"]["retained_remote_receipt"]

    with pytest.raises(production.HandoffContractError):
        _build(authority, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_authenticator_rejects_tampered_packaged_model(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "authority")
    output = _build(authority, tmp_path / "package")
    model = output / "full_model.aedt"
    os.chmod(model, 0o666)
    model.write_bytes(b"tampered")

    with pytest.raises(
        production.HandoffContractError,
        match="inventory drifted",
    ):
        _validate(authority, output)


def test_authenticator_cross_binds_resealed_model_inventory(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    output = _build(authority, tmp_path / "package")
    model = output / "full_model.aedt"
    os.chmod(model, 0o666)
    model.write_bytes(b"tampered-and-resealed")

    manifest_path = output / package.MANIFEST_NAME
    manifest = production._read_json(manifest_path)
    for item in manifest["inventory"]:
        if item["relative_path"] == model.name:
            item["sha256"] = production._sha256_file(model)
            item["size_bytes"] = model.stat().st_size
            break
    manifest["package_tree_sha256"] = canonical_sha256(manifest["inventory"])
    resealed = production._seal(
        {key: value for key, value in manifest.items() if key != "payload_sha256"}
    )
    os.chmod(manifest_path, 0o666)
    manifest_path.write_bytes(production._json_bytes(resealed))

    with pytest.raises(
        production.HandoffContractError,
        match="AEDT bytes drifted from authenticated source",
    ):
        _validate(authority, output)


def test_package_directory_is_relocatable_with_authority_sources(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "authority")
    output = _build(authority, tmp_path / "package")
    relocated = tmp_path / "relocated-package"
    shutil.copytree(output, relocated)

    authenticated = _validate(authority, relocated)
    assert authenticated["package_root"] == relocated.resolve()
    assert (
        authenticated["manifest"][
            "package_directory_relocatable_with_external_authority"
        ]
        is True
    )
