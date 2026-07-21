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
    legacy_unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"contract_sha256", "sha256"}
    }
    handoff_unsigned = {
        key: value for key, value in contract.items() if key != "sha256"
    }
    assert values.shape == (4, 25)
    assert contract["schema_version"] == CONTRACT_SCHEMA
    assert contract["source_status"]["authenticated_terminal_records"] == 1
    assert contract["coordinate_artifact"]["sha256"] == _sha(coordinates)
    assert contract["contract_sha256"] == canonical_sha256(legacy_unsigned)
    assert contract["sha256"] == canonical_sha256(handoff_unsigned)
    assert contract["fixed_primary_turns"] == 6
    assert contract["warm_rows_are_coordinate_donors_only"] is True
    loaded, evidence = authenticate_warm_handoff(
        coordinates,
        contract_path,
        fixed_primary_turns=6,
        n_var=values.shape[1],
        expected_contract_file_sha256=_sha(contract_path),
    )
    np.testing.assert_array_equal(loaded, values)
    assert evidence["contract_canonical_sha256"] == contract["sha256"]
    assert evidence["coordinates_only"] is True
    assert all(item["island_id"] == "n1-6-test" for item in contract["selection"])


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


def _basin_fixture(
    tmp_path: Path,
    *,
    name: str,
    seed: int,
    topologies: list[int],
    result_seed: int | None = None,
) -> Path:
    root = tmp_path / name
    cache = root / "cache"
    canonical = root / "canonical"
    cache.mkdir(parents=True)
    canonical.mkdir(parents=True)
    names = tuple(stage_constraint_names(CURRENT_STAGE_SPEC))
    rng = np.random.default_rng(seed)
    x = rng.random((len(topologies), 25))
    x[:, 2] = np.asarray([(60 - value) / 48.0 for value in topologies])
    g = np.zeros((len(x), len(names)), dtype=float)
    g[:, names.index("half_magnetizing_resonance_minimum")] = -1_000.0
    g[:, names.index("exterior_width_limit")] = -250.0
    g[:, names.index("exterior_length_limit")] = -250.0
    g[:, names.index("exterior_height_limit")] = -10.0
    g[:, names.index("Llt_robust_band")] = np.linspace(-0.2, 0.2, len(x))
    f = np.column_stack((
        np.linspace(500.0, 600.0, len(x)),
        np.linspace(5_000.0, 6_000.0, len(x)),
    ))
    artifacts = {}
    for key, values in {
        "terminal_X": x,
        "terminal_G_physical": g,
        "terminal_F": f,
    }.items():
        path = cache / f"{seed}-{key}.npy"
        np.save(path, values, allow_pickle=False)
        artifacts[key] = _artifact(path)
    result_path = cache / f"{seed}-result.json"
    result_path.write_text(
        json.dumps({
            "seed": seed if result_seed is None else result_seed,
            "island_id": "n1-6-test",
        }),
        encoding="utf-8",
    )
    status = {
        "schema_version": "mft-tier1-current7-slurm-rolling-status-v1",
        "bundle_id": f"fixture-{name}",
        "constraint_names": list(names),
        "hard_spec": CURRENT_STAGE_SPEC,
        "hard_spec_sha256": canonical_sha256(CURRENT_STAGE_SPEC),
        "terminal_results": [{
            "seed": seed,
            "island_id": "n1-6-test",
            "authenticated": True,
            "terminal_state": "completed",
            "artifact_objects": artifacts,
            "result_object": _artifact(result_path),
        }],
    }
    status_path = canonical / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    index = {
        "schema_version": "mft-tier1-current7-slurm-rolling-index-v1",
        "bundle_id": f"fixture-{name}",
        "path_containment_root": str(root),
        "constraint_names": list(names),
        "hard_spec": CURRENT_STAGE_SPEC,
        "hard_spec_sha256": canonical_sha256(CURRENT_STAGE_SPEC),
        "snapshot_sha256": f"{seed % 16:x}" * 64,
        "status": {"path": str(status_path), "sha256": _sha(status_path)},
    }
    index_path = canonical / "current7-index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def test_basin_pool_combines_indexes_and_seals_actual_source_result_provenance(
    tmp_path: Path,
):
    primary = _basin_fixture(
        tmp_path,
        name="primary",
        seed=31,
        topologies=[34] * 40 + [35] * 8 + [60] * 8,
    )
    donor = _basin_fixture(
        tmp_path,
        name="donor",
        seed=32,
        topologies=[36] * 8 + [37] * 8 + [38] * 8 + [39] * 2,
    )

    coordinates, contract_path = build_warm_pool(
        index_path=primary,
        donor_index_paths=(donor,),
        basin_aware=True,
        stage_spec=_stage(),
        output_dir=tmp_path / "basin-out",
        pool_size=56,
        candidate_limit=82,
        minimum_distance=0.0,
    )

    values = np.load(coordinates, allow_pickle=False)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    basin = payload["basin_aware_selection"]
    observed = 60 - np.rint(48.0 * values[:, 2]).astype(int)
    assert set(observed) >= {34, 35, 36, 37, 38, 39, 60}
    assert len(payload["source_indexes"]) == 2
    assert basin["source_available_counts"] == {
        "34": 40,
        "35": 8,
        "36": 8,
        "37": 8,
        "38": 8,
        "39": 2,
        "60": 8,
    }
    assert basin["protected_selection_count"] == 50
    assert basin["protected_selection_cap"] == 56
    assert basin["downstream_fresh_random_count"] == 160
    assert all(
        len(item["source_result_sha256"]) == 64
        and len(item["source_index_sha256"]) == 64
        for item in basin["protected_selection"]
    )
    unsigned_basin = {
        key: value for key, value in basin.items() if key != "sha256"
    }
    assert basin["sha256"] == canonical_sha256(unsigned_basin)


def test_basin_pool_fails_closed_when_a_required_topology_is_absent(
    tmp_path: Path,
):
    primary = _basin_fixture(
        tmp_path,
        name="primary-missing",
        seed=41,
        topologies=[34] * 40 + [35] * 8 + [60] * 8,
    )
    donor = _basin_fixture(
        tmp_path,
        name="donor-missing",
        seed=42,
        topologies=[36] * 8 + [37] * 8 + [38] * 8,
    )

    with pytest.raises(RuntimeError, match="omitted required 39/21 donor"):
        build_warm_pool(
            index_path=primary,
            donor_index_paths=(donor,),
            basin_aware=True,
            stage_spec=_stage(),
            output_dir=tmp_path / "missing-out",
            pool_size=56,
            candidate_limit=80,
            minimum_distance=0.0,
        )


def test_basin_pool_fails_closed_on_source_result_identity_mismatch(
    tmp_path: Path,
):
    primary = _basin_fixture(
        tmp_path,
        name="primary-mismatch",
        seed=51,
        result_seed=999,
        topologies=[34] * 40 + [35] * 8 + [60] * 8,
    )

    with pytest.raises(RuntimeError, match="source result identity mismatch"):
        build_warm_pool(
            index_path=primary,
            basin_aware=True,
            stage_spec=_stage(),
            output_dir=tmp_path / "mismatch-out",
            pool_size=56,
            candidate_limit=56,
            minimum_distance=0.0,
        )
