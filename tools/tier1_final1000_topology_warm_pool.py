"""Build the 160-row authenticated Final1000 topology-niche warm artifact.

Only locally cached, authenticated terminal X/F/physical-G arrays are read.
The source status is frozen by its canonical index SHA before selection.  No
surrogate is loaded and no scheduler, AEDT, FEA, controller, or UI state is
modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

try:
    from tier1_final1000_topology_niche_contract import (
        WARM_PARTITION_SCHEMA,
        WARM_SOURCE_GROUPS,
        WARM_TOPOLOGY_COUNTS,
        canonical_sha256,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_final1000_topology_niche_contract import (
        WARM_PARTITION_SCHEMA,
        WARM_SOURCE_GROUPS,
        WARM_TOPOLOGY_COUNTS,
        canonical_sha256,
    )


SCHEMA = "mft-tier1-final1000-topology-warm-pool-v1"
INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
REQUIRED_TOPOLOGIES = (34, 35, 36, 37, 38, 39)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    base = root.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped the canonical containment root") from exc
    return resolved


def _artifact(
    result: Mapping[str, Any], name: str, root: Path, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    record = (result.get("artifact_objects") or {}).get(name) or {}
    path = _contained(
        Path(str(record.get("local_cache_path") or "")), root, name
    )
    expected_sha = str(record.get("sha256") or "")
    expected_size = record.get("size_bytes")
    if (
        len(expected_sha) != 64
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or path.stat().st_size != expected_size
        or _sha256_file(path) != expected_sha
        or record.get("shape") != list(expected_shape)
    ):
        raise RuntimeError(f"authenticated terminal artifact mismatch: {name}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise RuntimeError(f"terminal artifact has invalid values: {name}")
    return values, {
        "path": str(path),
        "sha256": expected_sha,
        "size_bytes": expected_size,
        "shape": list(expected_shape),
    }


def _topologies(coordinates: np.ndarray) -> np.ndarray:
    return 60 - np.rint(
        60.0 * np.clip(coordinates[:, 2], 0.0, 1.0) * 0.8
    ).astype(int)


def _coordinate(topology: int) -> float:
    return (60 - int(topology)) / 48.0


def _constraint_scales(names: tuple[str, ...], spec: Mapping[str, Any]) -> np.ndarray:
    result = []
    for name in names:
        if name.startswith("temperature_robust_limit:"):
            scale = max(5.0, 0.05 * float(spec["T_limit_C"]))
        elif name in {
            "half_magnetizing_resonance_minimum",
            "half_magnetizing_resonance_maximum",
        }:
            scale = 1000.0
        elif name in {"Llt_robust_band", "Llt_ensemble_disagreement"}:
            scale = float(spec["Llt_tol_uH"])
        elif name in {"exterior_width_limit", "exterior_length_limit"}:
            scale = max(25.0, 0.05 * float(spec["size_W_max_mm"]))
        elif name == "exterior_height_limit":
            scale = max(25.0, 0.05 * float(spec["size_H_max_mm"]))
        elif name == "analytical_flux_density_limit":
            scale = 0.1
        elif name == "strict_full_density_support":
            scale = 0.05
        else:
            scale = 1.0
        result.append(scale)
    scales = np.asarray(result, dtype=float)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise RuntimeError("warm selection constraint scales are invalid")
    return scales


def _category_names(group: str, names: tuple[str, ...]) -> tuple[str, ...]:
    thermal = tuple(name for name in names if name.startswith("temperature_"))
    llt = tuple(name for name in names if name.startswith("Llt_"))
    resonance = tuple(name for name in names if "resonance" in name)
    size = tuple(name for name in names if name.startswith("exterior_"))
    if group == "llt_thermal_37":
        return llt + thermal
    if group == "resonance_size_34_35":
        return resonance + size
    if group == "transition_36":
        return llt + thermal + resonance + size
    if group == "boundary_38_39":
        return llt + thermal + resonance + size + (
            "analytical_flux_density_limit",
        )
    raise RuntimeError(f"unsupported terminal warm category: {group}")


def _row_sha(row: np.ndarray) -> str:
    return canonical_sha256(np.asarray(row, dtype=float).tolist())


def _select(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    group: str,
    names: tuple[str, ...],
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    positions = {name: index for index, name in enumerate(names)}
    subset = np.asarray(
        [positions[name] for name in _category_names(group, names)], dtype=int
    )
    if not len(subset):
        raise RuntimeError("warm selection category has no constraints")
    ranked = []
    for item in candidates:
        positive = np.maximum(item["G"], 0.0) / scales
        pressure = positive[subset]
        key = (
            int(np.count_nonzero(pressure > 0.0)),
            float(np.max(pressure, initial=0.0)),
            float(np.sum(pressure)),
            int(np.count_nonzero(positive > 0.0)),
            float(np.max(positive, initial=0.0)),
            float(np.sum(positive)),
            float(item["F"][0]),
            float(item["F"][1]),
            int(item["seed"]),
            int(item["row"]),
            item["coordinate_sha256"],
        )
        ranked.append((key, item))
    selected = []
    seen = set()
    for _key, item in sorted(ranked, key=lambda value: value[0]):
        if item["coordinate_sha256"] in seen:
            continue
        selected.append(item)
        seen.add(item["coordinate_sha256"])
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(
            f"authenticated warm category underfilled: {group} {len(selected)}/{count}"
        )
    return selected


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    staged = Path(temporary)
    try:
        with staged.open("wb") as stream:
            np.save(stream, np.asarray(values, dtype="<f8"), allow_pickle=False)
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def build(index_path: Path, output: Path) -> Path:
    index_path = index_path.resolve(strict=True)
    index = _read_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise RuntimeError("unsupported canonical Current7 index")
    root = Path(str(index.get("path_containment_root") or "")).resolve(strict=True)
    status_ref = index.get("status") or {}
    status_path = _contained(
        Path(str(status_ref.get("path") or "")), root, "canonical status"
    )
    if _sha256_file(status_path) != status_ref.get("sha256"):
        raise RuntimeError("canonical status no longer matches frozen index")
    status = _read_json(status_path)
    names = tuple(status.get("constraint_names") or [])
    spec = status.get("hard_spec") or {}
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("hard_spec_sha256") != index.get("hard_spec_sha256")
        or status.get("constraint_names") != index.get("constraint_names")
        or len(names) < 1
    ):
        raise RuntimeError("canonical index/status science identity mismatch")
    scales = _constraint_scales(names, spec)
    by_topology: dict[int, list[dict[str, Any]]] = {
        topology: [] for topology in REQUIRED_TOPOLOGIES
    }
    authenticated_results = 0
    artifact_sets = []
    for result in status.get("terminal_results") or []:
        if result.get("authenticated") is not True:
            continue
        count = 320
        x, x_record = _artifact(result, "terminal_X", root, (count, 25))
        g, g_record = _artifact(
            result, "terminal_G_physical", root, (count, len(names))
        )
        f, f_record = _artifact(result, "terminal_F", root, (count, 2))
        observed = _topologies(x)
        used = False
        for topology in REQUIRED_TOPOLOGIES:
            for row in np.flatnonzero(observed == topology):
                row = int(row)
                coordinate = np.asarray(x[row], dtype=float)
                by_topology[topology].append({
                    "X": coordinate,
                    "G": np.asarray(g[row], dtype=float),
                    "F": np.asarray(f[row], dtype=float),
                    "topology": topology,
                    "seed": int(result["seed"]),
                    "row": row,
                    "bundle_id": str(result["bundle_id"]),
                    "result_sha256": str(
                        (result.get("result_object") or {}).get("sha256") or ""
                    ),
                    "coordinate_sha256": _row_sha(coordinate),
                })
                used = True
        if used:
            authenticated_results += 1
            artifact_sets.append({
                "seed": int(result["seed"]),
                "bundle_id": str(result["bundle_id"]),
                "terminal_X": x_record,
                "terminal_G_physical": g_record,
                "terminal_F": f_record,
            })
    availability = {key: len(value) for key, value in by_topology.items()}
    if any(availability[topology] < 48 for topology in REQUIRED_TOPOLOGIES):
        raise RuntimeError("canonical terminal population lacks topology diversity")

    selected_groups: dict[str, list[dict[str, Any]]] = {}
    selected_groups["llt_thermal_37"] = _select(
        by_topology[37], count=48, group="llt_thermal_37", names=names, scales=scales
    )
    selected_groups["resonance_size_34_35"] = (
        _select(by_topology[34], count=24, group="resonance_size_34_35", names=names, scales=scales)
        + _select(by_topology[35], count=24, group="resonance_size_34_35", names=names, scales=scales)
    )
    selected_groups["transition_36"] = _select(
        by_topology[36], count=32, group="transition_36", names=names, scales=scales
    )
    selected_groups["boundary_38_39"] = (
        _select(by_topology[38], count=8, group="boundary_38_39", names=names, scales=scales)
        + _select(by_topology[39], count=8, group="boundary_38_39", names=names, scales=scales)
    )

    crossover_targets = [
        topology
        for topology, count in WARM_SOURCE_GROUPS["repaired_crossover"][
            "topology_counts"
        ].items()
        for _ in range(int(count))
    ]
    crossover = []
    for crossover_index, topology in enumerate(crossover_targets):
        # The 11,555-completion audit still had zero physically feasible
        # rows.  Its strongest transition signal was the entry-stage N36/N37
        # boundary, so every crossover donor carries one authenticated N36
        # parent and one authenticated N37 parent before being projected into
        # its requested protected topology niche.
        left = selected_groups["transition_36"][crossover_index % 32]
        right = selected_groups["llt_thermal_37"][crossover_index % 48]
        alpha = 0.35 if crossover_index % 2 == 0 else 0.65
        coordinate = np.clip(
            alpha * left["X"] + (1.0 - alpha) * right["X"], 0.0, 1.0
        )
        coordinate[2] = _coordinate(topology)
        crossover.append({
            "X": coordinate,
            "topology": int(topology),
            "coordinate_sha256": _row_sha(coordinate),
            "parents": [
                {"seed": left["seed"], "row": left["row"], "sha256": left["coordinate_sha256"]},
                {"seed": right["seed"], "row": right["row"], "sha256": right["coordinate_sha256"]},
            ],
            "source_pair_topologies": [36, 37],
            "entry_N36_N37_crossover_priority": True,
            "alpha": alpha,
            "downstream_current7_physics_repair_required": True,
        })
    selected_groups["repaired_crossover"] = crossover

    ordered = [
        item
        for name in WARM_SOURCE_GROUPS
        for item in selected_groups[name]
    ]
    coordinates = np.asarray([item["X"] for item in ordered], dtype="<f8")
    if coordinates.shape != (160, 25) or not np.isfinite(coordinates).all():
        raise RuntimeError("topology warm output shape is invalid")
    observed = _topologies(coordinates)
    aggregate = {
        topology: int(np.count_nonzero(observed == topology))
        for topology in WARM_TOPOLOGY_COUNTS
    }
    if aggregate != WARM_TOPOLOGY_COUNTS or aggregate[60] != 0:
        raise RuntimeError("topology warm output mix is invalid")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    coordinate_path = output / "coordinates.npy"
    _atomic_npy(coordinate_path, coordinates)
    groups = {}
    cursor = 0
    selection = {}
    for name, expected in WARM_SOURCE_GROUPS.items():
        count = int(expected["count"])
        rows = coordinates[cursor : cursor + count]
        groups[name] = {
            "start": cursor,
            "count": count,
            "topology_counts": {
                str(key): int(item)
                for key, item in expected["topology_counts"].items()
            },
            "coordinate_sha256": canonical_sha256(rows.tolist()),
            "current7_physics_repair_required": True,
        }
        selection[name] = [
            {
                key: value
                for key, value in item.items()
                if key not in {"X", "G", "F"}
            }
            for item in selected_groups[name]
        ]
        cursor += count
    partition = {
        "schema_version": WARM_PARTITION_SCHEMA,
        "fixed_primary_turns": 6,
        "total_count": 160,
        "groups": groups,
        "warm_topology_counts": {
            str(key): item for key, item in aggregate.items()
        },
        "N2_main_60_warm_count": 0,
        "same_current7_physics_repair_required": True,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
    }
    partition["sha256"] = canonical_sha256(partition)
    contract = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "fixed_primary_turns": 6,
        "stage_id": index.get("final_goal_stage_id"),
        "stage_spec": spec,
        "stage_spec_sha256": index.get("hard_spec_sha256"),
        "constraint_names": list(names),
        "canonical_source": {
            "index_path": str(index_path),
            "index_snapshot_sha256": index.get("snapshot_sha256"),
            "status_path": str(status_path),
            "status_file_sha256": status_ref.get("sha256"),
            "authenticated_terminal_result_count_used": authenticated_results,
            "available_topology_rows": {
                str(key): item for key, item in availability.items()
            },
            "artifact_set_inventory_sha256": canonical_sha256(artifact_sets),
        },
        "selection_policy": {
            "groups": list(WARM_SOURCE_GROUPS),
            "rank": (
                "group_positive_count_then_max_then_sum_then_all_constraint_"
                "count_max_sum_then_objectives_seed_row_coordinate_sha"
            ),
            "source_prediction_or_pass_classification_inherited": False,
            "source_physical_G_used_for_offline_rank_only": True,
            "source_physical_G_not_written_to_warm_artifact": True,
            "selection": selection,
        },
        "topology_niche_partition": partition,
        "warm_start": {
            "path": "coordinates.npy",
            "shape": [160, 25],
            "dtype": "float64-little-endian",
            "sha256": _sha256_file(coordinate_path),
            "coordinate_contract": (
                "authenticated_terminal_and_crossover_coordinates_then_"
                "same_current7_physics_repair_v1"
            ),
        },
        "warm_rows_are_coordinate_donors_only": True,
        "downstream_repair_required": True,
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "prior_constraints_reused_for_optimizer": False,
        "prior_objectives_reused_for_optimizer": False,
        "surrogate_only": True,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    contract["sha256"] = canonical_sha256(contract)
    contract_path = output / "contract.json"
    _atomic_json(contract_path, contract)
    return contract_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only-derived Final1000 topology warm pool."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(build(args.index, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
