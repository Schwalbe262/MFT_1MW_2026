from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import tier1_fixed_n1_6_bridge_warm_handoff as bridge
from tools import tier1_slurm_rolling as rolling


def _constraints(*, thermal_pass: bool, resonance_pass: bool) -> dict:
    values = {name: -1.0 for name in bridge.sealed.EXPECTED_CONSTRAINT_NAMES}
    if not thermal_pass:
        for name in bridge.CORE_THERMAL_CONSTRAINTS:
            values[name] = 10.0
    values[bridge.RESONANCE_CONSTRAINT] = -100.0 if resonance_pass else 2_000.0
    values["Llt_robust_band"] = 0.25
    return values


def _row(role: str, task_id: int, index: int, *, thermal: bool, resonance: bool):
    params = {"N1": 5 if role == "resfocus" and task_id == 57505 else 6,
              "coordinate_index": index}
    digest = bridge._json_sha(params)
    status_digest = f"{10_000 + index:064x}"
    result_digest = f"{20_000 + index:064x}"
    return {
        "role": role,
        "decoded_params": params,
        "decoded_params_sha256": digest,
        "constraint_G": _constraints(
            thermal_pass=thermal, resonance_pass=resonance,
        ),
        "reference": {
            "cohort_id": "cohort",
            "task_id": task_id,
            "seed": 1_000 + index,
            "seed_status_sha256": status_digest,
            "result_sha256": result_digest,
        },
    }


def test_bridge_handoff_interleaves_quota_and_required_anchors(
    tmp_path: Path, monkeypatch,
):
    def write_json(path: Path, value: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8",
        )
        return bridge.sealed._sha_file(path)

    rows = {}
    next_index = 0
    specifications = {
        "normalized": (20, 57059, True, False),
        "thermal": (20, 57919, True, False),
        "fixed6": (30, 58486, False, True),
        "resfocus": (10, 57505, False, True),
    }
    for role, (count, anchor, thermal, resonance) in specifications.items():
        role_rows = []
        for offset in range(count):
            task_id = anchor if offset == 0 else 60_000 + next_index
            role_rows.append(_row(
                role, task_id, next_index,
                thermal=thermal, resonance=resonance,
            ))
            next_index += 1
        rows[role] = role_rows

    for role, role_rows in rows.items():
        for row in role_rows:
            reference = row["reference"]
            result_path = (
                tmp_path / role / "results"
                / f"task-{reference['task_id']}" / "result.json"
            )
            result_sha = write_json(result_path, {
                "schema_version": bridge.sealed.SEARCH_SCHEMA,
                "seed": reference["seed"],
                "nsga_code_revision": bridge.EXPECTED_NSGA_REVISION,
                "constraint_version": bridge.CONSTRAINT_VERSION,
                "hard_spec": bridge.HARD_SPEC,
                "hard_spec_sha256": bridge._json_sha(bridge.HARD_SPEC),
                "model_manifest_sha256": (
                    "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
                ),
                "temperature_constraint_contract_sha256": (
                    "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
                ),
                "constraint_names": list(
                    bridge.sealed.EXPECTED_CONSTRAINT_NAMES
                ),
                "production_eligible": False,
                "fea_submission_approved": False,
                "automatic_promotion_allowed": False,
                "next_target_fea_batch_plan": {"candidates": [{
                    "decoded_params": row["decoded_params"],
                    "decoded_params_sha256": row["decoded_params_sha256"],
                    "constraint_G": row["constraint_G"],
                }]},
            })
            seed_status_path = result_path.with_name("seed_status.json")
            seed_status_sha = write_json(seed_status_path, {
                "schema_version": bridge.sealed.SEED_STATUS_SCHEMA,
                "seed": reference["seed"],
                "task_id": str(reference["task_id"]),
                "cohort_id": reference["cohort_id"],
                "state": "completed",
                "exit_code": 0,
                "result_sha256": result_sha,
                "bundle_manifest_sha256": "a" * 64,
                "hard_spec_sha256": bridge._json_sha(bridge.HARD_SPEC),
                "temperature_constraint_contract_sha256": (
                    "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
                ),
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
            })
            reference.update({
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "seed_status_path": str(seed_status_path),
                "seed_status_sha256": seed_status_sha,
            })

    generation_identity = {
        "constraint_version": bridge.CONSTRAINT_VERSION,
        "hard_spec_sha256": bridge._json_sha(bridge.HARD_SPEC),
        "source_model_manifest_sha256": (
            "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
        ),
        "deployment_model_manifest_sha256": (
            "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
        ),
        "temperature_constraint_contract_sha256": (
            "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
        ),
        "hard_spec": bridge.HARD_SPEC,
        "nsga_code_revision": bridge.EXPECTED_NSGA_REVISION,
    }
    controller_shas = {
        "normalized": (
            "0b48bcf01e691be0ee217700f173d6cf42f80902f5457251684bf19ac0de7e86"
        ),
        "thermal": (
            "ffc451375af92ac13910f59c9dd6888a0e1d595e6767ba41f6927cee3bccc71d"
        ),
        "fixed6": (
            "fbc1c784d06e2a7b3fc4b60fab0db665e8707fafb658451fd7a4e22a6762f2a2"
        ),
        "resfocus": (
            "5097c476f983f6460861ff7a53aaef4472fc67eda0924006db3d4429fd27052a"
        ),
    }
    profiles = {
        "normalized": None,
        "thermal": {
            "namespace": bridge.ROLE_NAMESPACES["thermal"],
            "variant": "thermal-crossover-core1c-v1",
        },
        "fixed6": {
            "namespace": bridge.ROLE_NAMESPACES["fixed6"],
            "variant": "fixed-n1-6-resonance-llt-v1",
        },
        "resfocus": {
            "namespace": bridge.ROLE_NAMESPACES["resfocus"],
            "variant": "resonance-focus-250hz-v1",
        },
    }
    all_result_shas = sorted(
        row["reference"]["result_sha256"]
        for role_rows in rows.values() for row in role_rows
    )
    all_status_shas = sorted(
        row["reference"]["seed_status_sha256"]
        for role_rows in rows.values() for row in role_rows
    )
    all_terminal_rows = [
        {
            "authenticated": True,
            "terminal_state": "completed",
            "scheduler_state": "completed",
            "scheduler_exit_code": 0,
            "task_id": row["reference"]["task_id"],
            "seed": row["reference"]["seed"],
            "cohort_id": row["reference"]["cohort_id"],
            "result": {"sha256": row["reference"]["result_sha256"]},
            "remote_status": {
                "sha256": row["reference"]["seed_status_sha256"]
            },
        }
        for role_rows in rows.values() for row in role_rows
    ]

    def fake_authenticate(root, role):
        root = root.resolve()
        cohort = "cohort"
        profile = profiles[role]
        current = {
            **{
                key: generation_identity[key]
                for key in (
                    "constraint_version", "hard_spec_sha256",
                    "source_model_manifest_sha256",
                    "deployment_model_manifest_sha256",
                    "temperature_constraint_contract_sha256",
                )
            },
            "hard_spec": bridge.HARD_SPEC,
            "nsga_code_revision": bridge.EXPECTED_NSGA_REVISION,
            "cohort_id": cohort,
            "bundle_manifest_sha256": "a" * 64,
            "controller_source_sha256": controller_shas[role],
            "search_profile": profile,
        }
        pointer_path = root / "sealed" / "pointer.json"
        pointer_sha = write_json(pointer_path, {
            "schema_version": bridge.sealed.POINTER_SCHEMA,
            "current": current,
        })
        status_path = root / "sealed" / "status.json"
        status_sha = write_json(status_path, {
            "schema_version": bridge.sealed.STATUS_SCHEMA,
            **{
                key: generation_identity[key]
                for key in (
                    "constraint_version", "hard_spec_sha256",
                    "source_model_manifest_sha256",
                    "deployment_model_manifest_sha256",
                    "temperature_constraint_contract_sha256",
                )
            },
            "hard_spec": bridge.HARD_SPEC,
            "nsga_code_revision": bridge.EXPECTED_NSGA_REVISION,
            "cohort_id": cohort,
            "updated_at": "2026-07-19T00:00:00Z",
            "search_profile": profile,
            "healthy": True,
            "error": None,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "terminal_results": all_terminal_rows,
        })
        index_path = root / "canonical" / "index.json"
        index_sha = write_json(index_path, {
            "schema_version": bridge.sealed.INDEX_SCHEMA,
            **{
                key: generation_identity[key]
                for key in (
                    "constraint_version", "hard_spec_sha256",
                    "source_model_manifest_sha256",
                    "deployment_model_manifest_sha256",
                    "temperature_constraint_contract_sha256",
                )
            },
            "hard_spec": bridge.HARD_SPEC,
            "active_cohort_id": cohort,
            "updated_at": "2026-07-19T00:00:00Z",
            "model_pointer": {
                "schema_version": bridge.sealed.POINTER_SCHEMA,
                "path": str(pointer_path),
                "sha256": pointer_sha,
            },
            "status": {
                "schema_version": bridge.sealed.STATUS_SCHEMA,
                "path": str(status_path),
                "sha256": status_sha,
            },
        })
        return {
            "evidence": {
                "role": role,
                "index": {"path": str(index_path), "sha256": index_sha},
                "pointer": {"path": str(pointer_path), "sha256": pointer_sha},
                "status": {"path": str(status_path), "sha256": status_sha},
                "generation_identity": generation_identity,
                "cohort_id": cohort,
                "bundle_manifest_sha256": "a" * 64,
                "controller_source_sha256": controller_shas[role],
                "search_profile": profile,
                "terminal_result_sha256": all_result_shas,
                "terminal_seed_status_sha256": all_status_shas,
                "terminal_result_set_sha256": bridge._json_sha(all_result_shas),
                "terminal_seed_status_set_sha256": bridge._json_sha(
                    all_status_shas
                ),
            },
            "rows": rows[role],
        }

    monkeypatch.setattr(bridge, "_authenticate", fake_authenticate)
    monkeypatch.setattr(
        bridge, "_git_revision", lambda _root: bridge.EXPECTED_NSGA_REVISION,
    )
    monkeypatch.setattr(
        bridge, "_verify_nsga_code_root",
        lambda _root: {
            "revision": bridge.EXPECTED_NSGA_REVISION,
            "clean": True,
            "decoder_relative_path": bridge.NSGA_DECODER_RELATIVE_PATH,
            "decoder_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        bridge, "_prepare_rows",
        lambda source: [dict(row) for row in source["rows"]],
    )
    monkeypatch.setattr(
        bridge,
        "_load_sobol_schema",
        lambda _root: ([object()] * 25, 5, 8, {}),
    )

    def fake_decoded(params, *_args):
        index = int(params["coordinate_index"])
        coordinate = np.zeros(25, dtype=float)
        coordinate[0] = (index + 1) / 100.0
        coordinate[1] = ((index * 7) % 89 + 1) / 100.0
        return coordinate

    monkeypatch.setattr(bridge, "decoded_to_unit", fake_decoded)
    monkeypatch.setattr(
        rolling, "_fixed_bridge_decode_coordinates",
        lambda params, _root: np.vstack([
            fake_decoded(item) for item in params
        ]),
    )
    for role_rows in rows.values():
        for row in role_rows:
            constraints = row["constraint_G"]
            row.update({
                "original_N1": int(row["decoded_params"]["N1"]),
                "all_thermal_pass": all(
                    constraints[name] <= 0.0
                    for name in bridge.THERMAL_CONSTRAINTS
                ),
                "resonance_pass": constraints[bridge.RESONANCE_CONSTRAINT] <= 0.0,
                "core_positive_sum_C": sum(
                    max(constraints[name], 0.0)
                    for name in bridge.CORE_THERMAL_CONSTRAINTS
                ),
                "core_positive_max_C": max(
                    max(constraints[name], 0.0)
                    for name in bridge.CORE_THERMAL_CONSTRAINTS
                ),
            })

    (tmp_path / "code").mkdir()
    result = bridge.build_handoff(
        normalized_root=tmp_path / "normalized",
        thermal_root=tmp_path / "thermal",
        fixed6_root=tmp_path / "fixed6",
        resonance_focus_root=tmp_path / "resfocus",
        nsga_code_root=tmp_path / "code",
        output=tmp_path / "output",
    )

    def seal_synthetic_preflight(warm_sha: str) -> None:
        repair = {
            "schema_version": "mft-tier1-fixed-primary-turns-repair-v1",
            "fixed_primary_turns": 6,
            "coordinate_index": 0,
            "hard_constraint_mutation": False,
            "objective_mutation": False,
            "warm_start_post_repair_minimum_unique_count": 32,
        }
        repair["sha256"] = bridge._json_sha(repair)
        scales = {
            name: 1.0 for name in bridge.sealed.EXPECTED_CONSTRAINT_NAMES
        }
        scales.update({
            "Llt_robust_band": 0.55,
            bridge.RESONANCE_CONSTRAINT: 150.0,
            **{name: 2.0 for name in bridge.CORE_THERMAL_CONSTRAINTS},
        })
        normalization = {
            "schema_version": (
                "mft-tier1-optimizer-constraint-normalization-v1"
            ),
            "constraint_order": list(
                bridge.sealed.EXPECTED_CONSTRAINT_NAMES
            ),
            "authoritative_terminal_G": "physical_unscaled",
            "scales": scales,
            "explicit_optimizer_only_scale_overrides": {
                name: 2.0 for name in bridge.CORE_THERMAL_CONSTRAINTS
            },
        }
        normalization["sha256"] = bridge._json_sha(normalization)
        preflight_path = tmp_path / "preflight" / "result.json"
        write_json(preflight_path, {
            "schema_version": bridge.sealed.SEARCH_SCHEMA,
            "seed": 1_907_197_999,
            "population": 64,
            "max_generations": 1,
            "completed_generations": 2,
            "nsga_code_revision": bridge.EXPECTED_NSGA_REVISION,
            "constraint_version": bridge.CONSTRAINT_VERSION,
            "hard_spec": bridge.HARD_SPEC,
            "hard_spec_sha256": bridge._json_sha(bridge.HARD_SPEC),
            "model_manifest_sha256": (
                "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
            ),
            "temperature_constraint_contract_sha256": (
                "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
            ),
            "constraint_names": list(
                bridge.sealed.EXPECTED_CONSTRAINT_NAMES
            ),
            "fixed_primary_turns": 6,
            "fixed_primary_turns_contract": repair,
            "terminal_population_fixed_primary_turns_verified": True,
            "terminal_population_primary_turn_values": [6],
            "warm_start": {
                "sha256": warm_sha,
                "coordinate_count": 64,
                "dimension_count": 25,
                "post_repair_coordinate_count": 64,
                "repaired_by_current_simultaneous_hard_constraint_problem": True,
                "fixed_primary_turns_repair": repair,
                "fixed_primary_turns_warm_audit": {
                    "schema_version": (
                        "mft-tier1-fixed-primary-turns-warm-audit-v1"
                    ),
                    "source_coordinate_count": 64,
                    "post_repair_coordinate_shape": [64, 25],
                    "post_repair_coordinate_dtype": "float64",
                    "post_repair_coordinates_sha256": "b" * 64,
                    "post_repair_decoded_unique_count": 64,
                    "post_repair_hard_feasible_count": 64,
                    "post_repair_rejected_count": 0,
                    "minimum_unique_count": 32,
                    "fixed_primary_turns": 6,
                    "fixed_primary_turn_coordinate_verified": True,
                    "physical_geometry_dedupe_performed": True,
                    "source_sha_verified_before_repair": True,
                },
            },
            "initialization_audit": {"warm_filter": {
                "input_count": 64,
                "hard_feasible_count": 64,
                "decoded_unique_count": 64,
                "rejected_count": 0,
            }},
            "optimizer_constraint_normalization": normalization,
            "acquisition_ranking_contract": {
                "active": True,
                "hard_constraint_mutation": False,
                "core_thermal_positive_G_scale_C": 2.0,
            },
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        })
        bridge.seal_post_repair_preflight(
            tmp_path / "output" / bridge.CONTRACT_FILENAME,
            preflight_path,
            tmp_path / "code",
        )

    seal_synthetic_preflight(result["warm_start"]["sha256"])

    assert result["warm_start"]["shape"] == [64, 25]
    assert result["category_counts"] == {
        "low_temperature_branch": 32,
        "resonance_pass_branch": 32,
    }
    assert result["source_role_counts"] == {
        "normalized": 16,
        "thermal": 16,
        "fixed6": 24,
        "resfocus": 8,
    }
    assert result["branch_source_role_counts"] == {
        "low_temperature_branch": {"normalized": 16, "thermal": 16},
        "resonance_pass_branch": {"fixed6": 24, "resfocus": 8},
    }
    assert {
        role: anchor["task_id"] for role, anchor in result["anchors"].items()
    } == bridge.REQUIRED_ANCHOR_TASKS
    selected = result["selected_provenance"]
    assert [item["warm_index"] for item in selected] == list(range(64))
    assert all(
        item["category"] == (
            "low_temperature_branch" if index % 2 == 0
            else "resonance_pass_branch"
        )
        for index, item in enumerate(selected)
    )
    assert result["anchors"]["resfocus"]["original_N1"] == 5
    assert result["anchors"]["resfocus"]["fixed_N1_repair_required"] is True
    assert result["optimizer_resonance_scale_Hz"] == 150.0
    assert result["optimizer_core_thermal_scale_C"] == 2.0
    assert result["optimizer_Llt_scale_uH"] == 0.55
    assert np.load(
        tmp_path / "output" / "next_warm_start.npy", allow_pickle=False,
    ).shape == (64, 25)
    for role in bridge.SOURCE_ROLES:
        for kind in bridge.SOURCE_SNAPSHOT_KINDS:
            snapshot = result["source_snapshots"][role][kind]
            snapshot_path = tmp_path / "output" / snapshot["path"]
            assert snapshot_path.is_file()
            assert bridge.sealed._sha_file(snapshot_path) == snapshot["sha256"]

    warm_path = tmp_path / "output" / "next_warm_start.npy"
    assert rolling.validate_fixed_n1_6_bridge_warm_contract(
        warm_path, tmp_path / "code",
    )[0] == (
        tmp_path / "output" / bridge.CONTRACT_FILENAME
    )

    contract_path = tmp_path / "output" / bridge.CONTRACT_FILENAME
    pristine_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pristine_coordinates = np.load(warm_path, allow_pickle=False)
    captured_warm = rolling._capture_validated_warm_artifacts(
        warm_path, (contract_path, rolling.sha256(contract_path)),
    )
    pristine_warm_bytes = warm_path.read_bytes()
    warm_path.write_bytes(pristine_warm_bytes + b"moving-writer-tick")
    assert captured_warm["warm_start_sha256"] == result["warm_start"]["sha256"]
    assert captured_warm["files"][warm_path.name] == pristine_warm_bytes
    warm_path.write_bytes(pristine_warm_bytes)

    bundle = tmp_path / "bundle"
    copied = rolling._copy_warm_handoff_evidence(
        contract_path, bundle / "artifacts" / "warm",
    )
    assert len(copied) == 141  # 12 heads + 64 pairs + repair preflight.
    bundled_contract = bundle / "artifacts" / "warm" / contract_path.name
    bundled_contract.parent.mkdir(parents=True, exist_ok=True)
    bundled_contract.write_bytes(contract_path.read_bytes())
    bundled_files = rolling._deployment_files(bundle)
    assert all(
        destination.relative_to(bundle).as_posix() in bundled_files
        for destination in copied
    )
    evidence_relatives = sorted(
        destination.relative_to(bundle).as_posix() for destination in copied
    )
    evidence_manifest = {
        "warm_handoff_contract": bundled_contract.relative_to(bundle).as_posix(),
        "warm_handoff_evidence": evidence_relatives,
        "files": bundled_files,
        "high_value_sha256": {
            relative: bundled_files[relative]["sha256"]
            for relative in evidence_relatives
        },
    }
    evidence_plan = {
        "warm_handoff_contract_sha256": rolling.sha256(bundled_contract)
    }
    assert rolling._bridge_bundle_evidence_attested(
        bundle.resolve(), evidence_plan, evidence_manifest,
    )
    incomplete_manifest = copy.deepcopy(evidence_manifest)
    incomplete_manifest["warm_handoff_evidence"].pop()
    assert not rolling._bridge_bundle_evidence_attested(
        bundle.resolve(), evidence_plan, incomplete_manifest,
    )
    forged_manifest = copy.deepcopy(evidence_manifest)
    forged_relative = evidence_relatives[0]
    forged_manifest["files"][forged_relative]["sha256"] = "e" * 64
    forged_manifest["high_value_sha256"][forged_relative] = "e" * 64
    assert not rolling._bridge_bundle_evidence_attested(
        bundle.resolve(), evidence_plan, forged_manifest,
    )

    selected_result = (
        tmp_path / "output"
        / pristine_contract["selected_artifact_snapshots"][0]["result"]["path"]
    )
    selected_result_bytes = selected_result.read_bytes()
    selected_result.write_bytes(selected_result_bytes + b" ")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        rolling.validate_fixed_n1_6_bridge_warm_contract(
            warm_path, tmp_path / "code",
        )
    selected_result.write_bytes(selected_result_bytes)

    repair_preflight = (
        tmp_path / "output"
        / pristine_contract["post_repair_preflight"]["result"]["path"]
    )
    repair_preflight_bytes = repair_preflight.read_bytes()
    repair_preflight.write_bytes(repair_preflight_bytes + b" ")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        rolling.validate_fixed_n1_6_bridge_warm_contract(
            warm_path, tmp_path / "code",
        )
    repair_preflight.write_bytes(repair_preflight_bytes)

    terminal_tampered = copy.deepcopy(pristine_contract)
    selected_result_shas = {
        item["result_sha256"]
        for item in terminal_tampered["selected_provenance"]
    }
    normalized = terminal_tampered["source_evidence"]["normalized"]
    unused_index = next(
        index for index, digest in enumerate(normalized["terminal_result_sha256"])
        if digest not in selected_result_shas
    )
    normalized["terminal_result_sha256"][unused_index] = "f" * 64
    normalized["terminal_result_sha256"].sort()
    normalized["terminal_result_set_sha256"] = bridge._json_sha(
        normalized["terminal_result_sha256"]
    )
    write_json(contract_path, terminal_tampered)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        rolling.validate_fixed_n1_6_bridge_warm_contract(
            warm_path, tmp_path / "code",
        )
    write_json(contract_path, pristine_contract)

    invalid_arrays = []
    changed_coordinate = pristine_coordinates.copy()
    changed_coordinate[1, 2] = 0.25
    invalid_arrays.append(changed_coordinate)
    duplicate_coordinate = pristine_coordinates.copy()
    duplicate_coordinate[1] = duplicate_coordinate[0]
    invalid_arrays.append(duplicate_coordinate)
    nonfinite_coordinate = pristine_coordinates.copy()
    nonfinite_coordinate[1, 2] = np.nan
    invalid_arrays.append(nonfinite_coordinate)
    below_range_coordinate = pristine_coordinates.copy()
    below_range_coordinate[1, 2] = -0.1
    invalid_arrays.append(below_range_coordinate)
    above_range_coordinate = pristine_coordinates.copy()
    above_range_coordinate[1, 2] = 1.1
    invalid_arrays.append(above_range_coordinate)
    invalid_arrays.append(pristine_coordinates.astype(np.float32))
    for invalid in invalid_arrays:
        with warm_path.open("wb") as stream:
            np.save(stream, invalid, allow_pickle=False)
        invalid_contract = copy.deepcopy(pristine_contract)
        invalid_contract["warm_start"]["sha256"] = bridge.sealed._sha_file(
            warm_path
        )
        write_json(contract_path, invalid_contract)
        with pytest.raises(RuntimeError, match="contract mismatch"):
            rolling.validate_fixed_n1_6_bridge_warm_contract(
                warm_path, tmp_path / "code",
            )

    with warm_path.open("wb") as stream:
        np.save(stream, pristine_coordinates, allow_pickle=False)
    write_json(contract_path, pristine_contract)

    # A quota drift that remains branch-valid must still fail actual-count checks.
    quota_tampered = copy.deepcopy(pristine_contract)
    changed = next(
        item for item in quota_tampered["selected_provenance"]
        if item["source_role"] == "normalized"
        and item["task_id"] != bridge.REQUIRED_ANCHOR_TASKS["normalized"]
    )
    changed["source_role"] = "thermal"
    quota_tampered["selected_provenance_sha256"] = bridge._json_sha(
        quota_tampered["selected_provenance"]
    )
    write_json(contract_path, quota_tampered)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        rolling.validate_fixed_n1_6_bridge_warm_contract(
            warm_path, tmp_path / "code",
        )

    # Restore exact original bytes with a fresh deterministic rebuild, then prove
    # that a mutable-head snapshot is mandatory.
    result = bridge.build_handoff(
        normalized_root=tmp_path / "normalized",
        thermal_root=tmp_path / "thermal",
        fixed6_root=tmp_path / "fixed6",
        resonance_focus_root=tmp_path / "resfocus",
        nsga_code_root=tmp_path / "code",
        output=tmp_path / "output",
    )
    seal_synthetic_preflight(result["warm_start"]["sha256"])
    snapshot_path = (
        tmp_path / "output"
        / result["source_snapshots"]["normalized"]["index"]["path"]
    )
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        rolling.validate_fixed_n1_6_bridge_warm_contract(
            warm_path, tmp_path / "code",
        )


def test_bridge_handoff_rejects_wrong_nsga_revision(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr(bridge, "_git_revision", lambda _root: "0" * 40)
    with pytest.raises(RuntimeError, match="EXPECTED_NSGA_REVISION"):
        bridge.build_handoff(
            normalized_root=tmp_path,
            thermal_root=tmp_path,
            fixed6_root=tmp_path,
            resonance_focus_root=tmp_path,
            nsga_code_root=tmp_path,
            output=tmp_path / "output",
        )


def test_authenticated_capture_retries_a_canonical_tick_race(
    tmp_path: Path, monkeypatch,
):
    terminal_evidence = {
        "terminal_snapshot_count": 1,
        "terminal_result_order_sha256": "1" * 64,
        "terminal_seed_status_order_sha256": "2" * 64,
        "terminal_reference_order_sha256": "3" * 64,
        "terminal_reference_set_sha256": "4" * 64,
        "terminal_result_sha256": ["5" * 64],
        "terminal_seed_status_sha256": ["6" * 64],
    }
    calls = 0

    def fake_authenticate(root: Path, role: str) -> dict:
        nonlocal calls
        calls += 1
        evidence = {
            "role": role,
            "terminal_result_sha256": ["5" * 64],
            "terminal_seed_status_sha256": ["6" * 64],
            "terminal_result_set_sha256": bridge._json_sha(["5" * 64]),
            "terminal_seed_status_set_sha256": bridge._json_sha(["6" * 64]),
        }
        for kind in bridge.SOURCE_SNAPSHOT_KINDS:
            path = root / f"{kind}.json"
            digest = write_json_for_race(path, {"attempt": calls, "kind": kind})
            evidence[kind] = {"path": str(path), "sha256": digest}
        if calls == 1:
            # Simulate the 10-second writer tick after authentication returned its
            # SHA but before the consumer could capture the index bytes.
            write_json_for_race(
                Path(evidence["index"]["path"]),
                {"attempt": calls + 100, "kind": "index"},
            )
        return {"evidence": evidence, "results": []}

    def write_json_for_race(path: Path, value: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return bridge.sealed._sha_file(path)

    monkeypatch.setattr(bridge, "_authenticate", fake_authenticate)
    monkeypatch.setattr(
        bridge, "_terminal_snapshot_evidence",
        lambda _status, _role: terminal_evidence,
    )
    captured = bridge._capture_authenticated_source(tmp_path, "normalized")

    assert calls == 2
    assert json.loads(captured["_snapshot_payloads"]["index"])["attempt"] == 2
