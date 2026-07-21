from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.tier1_corrected_generation_preflight import (
    CURRENT_STAGE_SPEC,
    authenticate_warm_handoff,
    canonical_sha256,
    stage_constraint_names,
    validate_stage_spec,
)
from tools.tier1_final1000_warm_pool import (
    CONTRACT_SCHEMA,
    build_warm_pool,
    transform_physical_g,
    upgrade_existing_handoff_contract,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage() -> dict:
    return validate_stage_spec(
        {
            **CURRENT_STAGE_SPEC,
            "T_limit_C": 100.0,
            "size_W_max_mm": 1_000.0,
            "size_L_max_mm": 1_000.0,
            "resonance_min_Hz": 15_000.0,
            "resonance_max_Hz": 20_000.0,
        }
    )


def test_transform_reconstructs_strict_resonance_and_new_limits():
    names = tuple(stage_constraint_names(CURRENT_STAGE_SPEC))
    position = {name: index for index, name in enumerate(names)}
    source = np.zeros((2, len(names)), dtype=float)
    source[:, position["half_magnetizing_resonance_minimum"]] = [0.0, -5_000.0]
    source[:, position["exterior_width_limit"]] = [-200.0, -199.0]
    source[:, position["temperature_robust_limit:Tprobe_core_center_max"]] = [
        -10.0,
        -9.0,
    ]

    transformed, target_names = transform_physical_g(
        source,
        source_names=names,
        source_spec=CURRENT_STAGE_SPEC,
        stage_spec=_stage(),
    )
    target = {name: index for index, name in enumerate(target_names)}

    assert transformed[0, target["half_magnetizing_resonance_minimum"]] == 0.0
    assert transformed[1, target["half_magnetizing_resonance_minimum"]] == -5_000.0
    assert transformed[0, target["half_magnetizing_resonance_maximum"]] < 0.0
    assert transformed[1, target["half_magnetizing_resonance_maximum"]] > 0.0
    assert transformed[:, target["exterior_width_limit"]].tolist() == [0.0, 1.0]
    assert transformed[:, target[
        "temperature_robust_limit:Tprobe_core_center_max"
    ]].tolist() == [0.0, 1.0]


def _artifact(path: Path) -> dict:
    return {
        "local_cache_path": str(path),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    cache = root / "cache"
    canonical = root / "canonical"
    cache.mkdir(parents=True)
    canonical.mkdir(parents=True)
    names = tuple(stage_constraint_names(CURRENT_STAGE_SPEC))
    rng = np.random.default_rng(7)
    terminals = []
    for seed, island in ((11, "n1-6-test"), (12, "n1-5-ignored")):
        x = rng.random((8, 25))
        g = np.zeros((8, len(names)), dtype=float)
        g[:, names.index("Llt_robust_band")] = np.linspace(0.0, 2.0, 8)
        g[:, names.index("exterior_width_limit")] = np.linspace(-200.0, -150.0, 8)
        g[:, names.index("half_magnetizing_resonance_minimum")] = -1_000.0
        f = np.column_stack((np.linspace(500.0, 600.0, 8), np.linspace(5_000, 6_000, 8)))
        paths = {}
        for key, values in {
            "terminal_X": x,
            "terminal_G_physical": g,
            "terminal_F": f,
        }.items():
            path = cache / f"{seed}-{key}.npy"
            np.save(path, values, allow_pickle=False)
            paths[key] = _artifact(path)
        terminals.append(
            {
                "seed": seed,
                "island_id": island,
                "authenticated": True,
                "terminal_state": "completed",
                "artifact_objects": paths,
            }
        )
    status = {
        "schema_version": "mft-tier1-current7-slurm-rolling-status-v1",
        "bundle_id": "fixture-bundle",
        "constraint_names": list(names),
        "hard_spec": CURRENT_STAGE_SPEC,
        "hard_spec_sha256": canonical_sha256(CURRENT_STAGE_SPEC),
        "terminal_results": terminals,
    }
    status_path = canonical / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    index = {
        "schema_version": "mft-tier1-current7-slurm-rolling-index-v1",
        "bundle_id": "fixture-bundle",
        "path_containment_root": str(root),
        "constraint_names": list(names),
        "hard_spec": CURRENT_STAGE_SPEC,
        "hard_spec_sha256": canonical_sha256(CURRENT_STAGE_SPEC),
        "snapshot_sha256": "1" * 64,
        "status": {
            "path": str(status_path),
            "sha256": _sha(status_path),
        },
    }
    index_path = canonical / "current7-index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def test_build_pool_authenticates_n1_6_and_writes_sealed_artifacts(tmp_path: Path):
    index = _fixture(tmp_path)
    coordinates, contract_path = build_warm_pool(
        index_path=index,
        stage_spec=_stage(),
        output_dir=tmp_path / "out",
        pool_size=4,
        candidate_limit=8,
        minimum_distance=0.0,
    )

    values = np.load(coordinates, allow_pickle=False)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    assert values.shape == (4, 25)
    assert contract["schema_version"] == CONTRACT_SCHEMA
    assert contract["source_status"]["authenticated_terminal_records"] == 1
    assert contract["coordinate_artifact"]["sha256"] == _sha(coordinates)
    assert contract["contract_sha256"] == canonical_sha256(unsigned)
    assert contract["fixed_primary_turns"] == 6
    assert contract["warm_start"]["sha256"] == _sha(coordinates)
    assert contract["warm_start"]["shape"] == [4, 25]
    authenticated, evidence = authenticate_warm_handoff(
        coordinates,
        contract_path,
        fixed_primary_turns=6,
        n_var=25,
        expected_contract_file_sha256=_sha(contract_path),
    )
    assert authenticated.shape == (4, 25)
    assert evidence["coordinates_only"] is True
    assert all(item["island_id"] == "n1-6-test" for item in contract["selection"])


def test_build_pool_seals_n1_5_identity_for_remote_authenticator(tmp_path: Path):
    coordinates, contract_path = build_warm_pool(
        index_path=_fixture(tmp_path),
        stage_spec=_stage(),
        output_dir=tmp_path / "out-n1-5",
        pool_size=4,
        candidate_limit=8,
        minimum_distance=0.0,
        island_prefix="n1-5-",
    )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["fixed_primary_turns"] == 5
    values, _evidence = authenticate_warm_handoff(
        coordinates,
        contract_path,
        fixed_primary_turns=5,
        n_var=25,
        expected_contract_file_sha256=_sha(contract_path),
    )
    assert values.shape == (4, 25)


def test_existing_pool_contract_can_be_upgraded_without_rewriting_coordinates(
    tmp_path: Path,
):
    coordinates, contract_path = build_warm_pool(
        index_path=_fixture(tmp_path),
        stage_spec=_stage(),
        output_dir=tmp_path / "out",
        pool_size=4,
        candidate_limit=8,
        minimum_distance=0.0,
    )
    original_coordinate_sha = _sha(coordinates)
    legacy = json.loads(contract_path.read_text(encoding="utf-8"))
    for field in (
        "fixed_primary_turns",
        "fixed_primary_turns_scope",
        "warm_start",
        "warm_rows_are_coordinate_donors_only",
        "physical_hard_spec_mutation",
        "objective_mutation",
    ):
        legacy.pop(field)
    legacy.pop("contract_sha256")
    legacy["contract_sha256"] = canonical_sha256(legacy)
    legacy_path = tmp_path / "legacy-contract.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    upgraded_path = tmp_path / "current7-contract.json"

    actual = upgrade_existing_handoff_contract(
        warm_start=coordinates,
        source_contract=legacy_path,
        output_contract=upgraded_path,
        fixed_primary_turns=6,
    )

    assert actual == upgraded_path
    assert _sha(coordinates) == original_coordinate_sha
    upgraded = json.loads(upgraded_path.read_text(encoding="utf-8"))
    assert upgraded["compatibility_upgrade"]["coordinate_bytes_rewritten"] is False
    values, _evidence = authenticate_warm_handoff(
        coordinates,
        upgraded_path,
        fixed_primary_turns=6,
        n_var=25,
        expected_contract_file_sha256=_sha(upgraded_path),
    )
    assert values.shape == (4, 25)


def test_build_pool_rejects_status_tampering(tmp_path: Path):
    index_path = _fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    status_path = Path(index["status"]["path"])
    status_path.write_text(status_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match="status reference authentication failed"):
        build_warm_pool(
            index_path=index_path,
            stage_spec=_stage(),
            output_dir=tmp_path / "out",
            pool_size=4,
            candidate_limit=8,
        )
