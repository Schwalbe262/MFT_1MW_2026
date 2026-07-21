"""Build an authenticated staged warm pool from Current7 terminal populations.

The source campaign is read-only.  Its physical, unscaled terminal ``G``
arrays are transformed algebraically to one explicitly supplied staged hard
spec.  Coordinates are ranked by staged constraint count and normalized
positive violation, then greedily diversified before an atomic ``.npy`` and
JSON contract are written.

No surrogate is loaded and no scheduler, AEDT, FEA, or production pointer is
mutated by this tool.
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
    from tier1_corrected_generation_preflight import (
        RESONANCE_MAXIMUM_CONSTRAINT,
        RESONANCE_MINIMUM_CONSTRAINT,
        canonical_sha256,
        stage_constraint_names,
        stage_spec_from_json_identity,
        validate_stage_spec,
    )
except ImportError:  # pragma: no cover - repository module import path
    from tools.tier1_corrected_generation_preflight import (
        RESONANCE_MAXIMUM_CONSTRAINT,
        RESONANCE_MINIMUM_CONSTRAINT,
        canonical_sha256,
        stage_constraint_names,
        stage_spec_from_json_identity,
        validate_stage_spec,
    )


CONTRACT_SCHEMA = "mft-tier1-final1000-warm-pool-v1"
INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
TEMPERATURE_PREFIX = "temperature_robust_limit:"
WIDTH_CONSTRAINT = "exterior_width_limit"
LENGTH_CONSTRAINT = "exterior_length_limit"
HEIGHT_CONSTRAINT = "exterior_height_limit"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    base = root.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the Current7 containment root") from exc
    return resolved


def _artifact(record: Mapping[str, Any], root: Path, label: str) -> Path:
    raw = str(record.get("local_cache_path") or "")
    expected = str(record.get("sha256") or "")
    if not raw or len(expected) != 64:
        raise RuntimeError(f"{label} artifact identity is incomplete")
    path = _contained(Path(raw), root, label)
    expected_size = record.get("size_bytes", record.get("size"))
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or path.stat().st_size != expected_size
        or _sha256_file(path) != expected
    ):
        raise RuntimeError(f"{label} artifact authentication failed")
    return path


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _load_source(index_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    index_path = index_path.resolve(strict=True)
    index = _read_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise RuntimeError("unsupported Current7 canonical index schema")
    root = Path(str(index.get("path_containment_root") or "")).resolve(strict=True)
    status_ref = index.get("status") or {}
    status_path = _contained(
        Path(str(status_ref.get("path") or "")), root, "Current7 status"
    )
    if (
        len(str(status_ref.get("sha256") or "")) != 64
        or _sha256_file(status_path) != status_ref.get("sha256")
    ):
        raise RuntimeError("Current7 status reference authentication failed")
    status = _read_json(status_path)
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("bundle_id") != index.get("bundle_id")
        or status.get("constraint_names") != index.get("constraint_names")
        or status.get("hard_spec") != index.get("hard_spec")
        or status.get("hard_spec_sha256") != index.get("hard_spec_sha256")
    ):
        raise RuntimeError("Current7 index/status identity mismatch")
    return index, status, root


def _resonance_frequency(
    source_g: np.ndarray,
    source_names: tuple[str, ...],
    source_spec: Mapping[str, Any],
) -> np.ndarray:
    positions = {name: index for index, name in enumerate(source_names)}
    lower = source_spec.get("resonance_min_Hz")
    if lower is not None and RESONANCE_MINIMUM_CONSTRAINT in positions:
        return float(lower) - source_g[:, positions[RESONANCE_MINIMUM_CONSTRAINT]]
    upper = source_spec.get("resonance_max_Hz")
    if upper is not None and RESONANCE_MAXIMUM_CONSTRAINT in positions:
        strict = np.nextafter(float(upper), -np.inf)
        return source_g[:, positions[RESONANCE_MAXIMUM_CONSTRAINT]] + strict
    raise RuntimeError("source Current7 status cannot reconstruct resonance")


def transform_physical_g(
    source_g: np.ndarray,
    *,
    source_names: tuple[str, ...],
    source_spec: Mapping[str, Any],
    stage_spec: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Transform unscaled physical G without reusing optimizer-normalized G."""

    source_g = np.asarray(source_g, dtype=float)
    if (
        source_g.ndim != 2
        or source_g.shape[1] != len(source_names)
        or not np.isfinite(source_g).all()
    ):
        raise RuntimeError("source terminal physical G has an invalid schema")
    normalized = validate_stage_spec(stage_spec)
    target_names = tuple(stage_constraint_names(normalized))
    source_position = {name: index for index, name in enumerate(source_names)}
    result = np.empty((len(source_g), len(target_names)), dtype=float)
    resonance = _resonance_frequency(source_g, source_names, source_spec)
    old_temperature = _finite(source_spec.get("T_limit_C"), "source T limit")
    old_limits = {
        WIDTH_CONSTRAINT: _finite(source_spec.get("size_W_max_mm"), "source W"),
        LENGTH_CONSTRAINT: _finite(source_spec.get("size_L_max_mm"), "source L"),
        HEIGHT_CONSTRAINT: _finite(source_spec.get("size_H_max_mm"), "source H"),
    }
    new_limits = {
        WIDTH_CONSTRAINT: float(normalized["size_W_max_mm"]),
        LENGTH_CONSTRAINT: float(normalized["size_L_max_mm"]),
        HEIGHT_CONSTRAINT: float(normalized["size_H_max_mm"]),
    }
    for target_index, name in enumerate(target_names):
        if name.startswith(TEMPERATURE_PREFIX):
            if name not in source_position:
                raise RuntimeError(f"source terminal G omitted {name}")
            result[:, target_index] = (
                source_g[:, source_position[name]]
                + old_temperature
                - float(normalized["T_limit_C"])
            )
        elif name in new_limits:
            if name not in source_position:
                raise RuntimeError(f"source terminal G omitted {name}")
            result[:, target_index] = (
                source_g[:, source_position[name]]
                + old_limits[name]
                - new_limits[name]
            )
        elif name == RESONANCE_MINIMUM_CONSTRAINT:
            result[:, target_index] = float(normalized["resonance_min_Hz"]) - resonance
        elif name == RESONANCE_MAXIMUM_CONSTRAINT:
            strict = np.nextafter(float(normalized["resonance_max_Hz"]), -np.inf)
            result[:, target_index] = resonance - strict
        elif name in source_position:
            result[:, target_index] = source_g[:, source_position[name]]
        else:
            raise RuntimeError(f"source terminal G cannot derive {name}")
    if not np.isfinite(result).all():
        raise RuntimeError("transformed staged physical G is non-finite")
    return result, target_names


def _constraint_scales(
    names: tuple[str, ...], stage_spec: Mapping[str, Any]
) -> np.ndarray:
    scales = []
    for name in names:
        if name.startswith(TEMPERATURE_PREFIX):
            value = max(5.0, 0.05 * float(stage_spec["T_limit_C"]))
        elif name in {RESONANCE_MINIMUM_CONSTRAINT, RESONANCE_MAXIMUM_CONSTRAINT}:
            value = 1_000.0
        elif name == "Llt_robust_band":
            value = float(stage_spec["Llt_tol_uH"])
        elif name == "Llt_ensemble_disagreement":
            value = float(stage_spec["Llt_tol_uH"])
        elif name == "analytical_flux_density_limit":
            value = 0.1
        elif name == "strict_full_density_support":
            value = 0.05
        elif name in {WIDTH_CONSTRAINT, LENGTH_CONSTRAINT}:
            value = max(25.0, 0.05 * float(stage_spec["size_W_max_mm"]))
        elif name == HEIGHT_CONSTRAINT:
            value = max(25.0, 0.05 * float(stage_spec["size_H_max_mm"]))
        else:
            value = 1.0
        scales.append(value)
    result = np.asarray(scales, dtype=float)
    if np.any(result <= 0.0) or not np.isfinite(result).all():
        raise RuntimeError("warm-pool constraint scales are invalid")
    return result


def _diverse_indices(
    coordinates: np.ndarray,
    order: np.ndarray,
    *,
    pool_size: int,
    candidate_limit: int,
    minimum_distance: float,
) -> list[int]:
    candidates = [int(value) for value in order[:candidate_limit]]
    selected: list[int] = []
    for index in candidates:
        row = coordinates[index]
        if not selected or min(
            float(np.linalg.norm(row - coordinates[prior])) for prior in selected
        ) >= minimum_distance:
            selected.append(index)
        if len(selected) == pool_size:
            return selected
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) == pool_size:
            break
    return selected


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_warm_pool(
    *,
    index_path: Path,
    stage_spec: Mapping[str, Any],
    output_dir: Path,
    pool_size: int = 320,
    candidate_limit: int = 8_192,
    minimum_distance: float = 0.01,
    island_prefix: str = "n1-6-",
) -> tuple[Path, Path]:
    if isinstance(pool_size, bool) or not 4 <= int(pool_size) <= 4096:
        raise ValueError("pool_size must be from 4 through 4096")
    if isinstance(candidate_limit, bool) or int(candidate_limit) < int(pool_size):
        raise ValueError("candidate_limit must be at least pool_size")
    if not math.isfinite(float(minimum_distance)) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    normalized_stage = validate_stage_spec(stage_spec)
    canonical_index, status, containment_root = _load_source(index_path)
    source_names = tuple(str(name) for name in status["constraint_names"])
    source_spec = status["hard_spec"]
    coordinates = []
    physical_g = []
    objectives = []
    metadata: list[dict[str, Any]] = []
    authenticated_artifacts: list[dict[str, Any]] = []
    for terminal in status.get("terminal_results") or []:
        if (
            terminal.get("authenticated") is not True
            or terminal.get("terminal_state") != "completed"
            or not str(terminal.get("island_id") or "").startswith(island_prefix)
        ):
            continue
        artifacts = terminal.get("artifact_objects") or {}
        paths = {
            key: _artifact(artifacts.get(key) or {}, containment_root, key)
            for key in ("terminal_X", "terminal_G_physical", "terminal_F")
        }
        try:
            x = np.load(paths["terminal_X"], allow_pickle=False)
            g = np.load(paths["terminal_G_physical"], allow_pickle=False)
            f = np.load(paths["terminal_F"], allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError("authenticated terminal NumPy artifact is invalid") from exc
        if (
            x.ndim != 2
            or g.shape != (len(x), len(source_names))
            or f.shape != (len(x), 2)
            or not np.isfinite(x).all()
            or not np.isfinite(g).all()
            or not np.isfinite(f).all()
        ):
            raise RuntimeError("authenticated terminal array schema mismatch")
        coordinates.append(np.asarray(x, dtype=float))
        physical_g.append(np.asarray(g, dtype=float))
        objectives.append(np.asarray(f, dtype=float))
        metadata.extend(
            {
                "seed": int(terminal["seed"]),
                "island_id": str(terminal["island_id"]),
                "terminal_population_index": row,
            }
            for row in range(len(x))
        )
        authenticated_artifacts.append(
            {
                "seed": int(terminal["seed"]),
                "island_id": str(terminal["island_id"]),
                "objects": {
                    key: {
                        "sha256": str(artifacts[key]["sha256"]),
                        "size_bytes": int(
                            artifacts[key].get("size_bytes", artifacts[key].get("size"))
                        ),
                    }
                    for key in paths
                },
            }
        )
    if not coordinates:
        raise RuntimeError("no authenticated terminal coordinates matched the island prefix")
    x_all = np.vstack(coordinates)
    g_all = np.vstack(physical_g)
    f_all = np.vstack(objectives)
    unique_keys = np.ascontiguousarray(x_all).view(
        np.dtype((np.void, x_all.dtype.itemsize * x_all.shape[1]))
    ).ravel()
    _, unique_indices = np.unique(unique_keys, return_index=True)
    unique_indices = np.sort(unique_indices)
    x_all = x_all[unique_indices]
    g_all = g_all[unique_indices]
    f_all = f_all[unique_indices]
    metadata = [metadata[int(index)] for index in unique_indices]
    staged_g, target_names = transform_physical_g(
        g_all,
        source_names=source_names,
        source_spec=source_spec,
        stage_spec=normalized_stage,
    )
    scales = _constraint_scales(target_names, normalized_stage)
    positive = np.maximum(staged_g, 0.0)
    violation_count = np.count_nonzero(staged_g > 0.0, axis=1)
    normalized_sum = np.sum(positive / scales, axis=1)
    order = np.lexsort((f_all[:, 1], f_all[:, 0], normalized_sum, violation_count))
    selected = _diverse_indices(
        x_all,
        order,
        pool_size=min(int(pool_size), len(x_all)),
        candidate_limit=min(int(candidate_limit), len(x_all)),
        minimum_distance=float(minimum_distance),
    )
    if len(selected) < 4:
        raise RuntimeError("staged warm selection produced fewer than four coordinates")
    selected_x = np.asarray(x_all[selected], dtype="<f8")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("warm-pool output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_path = output_dir / "coordinates.npy"
    contract_path = output_dir / "contract.json"
    _atomic_npy(coordinate_path, selected_x)
    selected_records = []
    for rank, index in enumerate(selected):
        row = np.asarray(x_all[index], dtype="<f8")
        selected_records.append(
            {
                "rank": rank,
                **metadata[index],
                "positive_constraint_count": int(violation_count[index]),
                "normalized_positive_G_sum": float(normalized_sum[index]),
                "objective_volume_L": float(f_all[index, 0]),
                "objective_total_loss_W": float(f_all[index, 1]),
                "coordinate_sha256": hashlib.sha256(row.tobytes()).hexdigest(),
            }
        )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "created_at": _now(),
        "source_index": {
            "path": str(index_path.resolve(strict=True)),
            "sha256": _sha256_file(index_path.resolve(strict=True)),
            "snapshot_sha256": canonical_index.get("snapshot_sha256"),
            "bundle_id": canonical_index.get("bundle_id"),
        },
        "source_status": {
            "path": str(
                Path(canonical_index["status"]["path"]).resolve(strict=True)
            ),
            "sha256": str(canonical_index["status"]["sha256"]),
            "authenticated_terminal_records": len(authenticated_artifacts),
            "artifact_inventory_sha256": canonical_sha256(authenticated_artifacts),
        },
        "source_constraint_names": list(source_names),
        "source_hard_spec": source_spec,
        "stage_spec": normalized_stage,
        "stage_spec_sha256": canonical_sha256(normalized_stage),
        "target_constraint_names": list(target_names),
        "constraint_scales": {
            name: float(scale) for name, scale in zip(target_names, scales)
        },
        "ranking": (
            "positive_constraint_count_then_sum_positive_physical_G_over_"
            "sealed_scale_then_volume_then_loss"
        ),
        "source_row_count": int(sum(len(value) for value in coordinates)),
        "unique_coordinate_count": int(len(x_all)),
        "source_exact_stage_feasible_count": int(
            np.count_nonzero(np.all(staged_g <= 0.0, axis=1))
        ),
        "selected_count": int(len(selected_x)),
        "candidate_limit": min(int(candidate_limit), len(x_all)),
        "minimum_coordinate_l2_distance": float(minimum_distance),
        "selection": selected_records,
        "coordinate_artifact": {
            "path": coordinate_path.name,
            "sha256": _sha256_file(coordinate_path),
            "coordinate_unit_sha256": canonical_sha256(selected_x.tolist()),
            "shape": list(selected_x.shape),
            "dtype": "float64-little-endian",
            "allow_pickle": False,
        },
        "downstream_repair_required": True,
        "prior_objectives_reused_for_optimizer": False,
        "prior_constraints_reused_for_optimizer": False,
        "terminal_physical_replay_required": True,
        "surrogate_only": True,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    _atomic_json(contract_path, contract)
    return coordinate_path, contract_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--stage-spec-json", required=True)
    parser.add_argument("--stage-spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=320)
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--minimum-distance", type=float, default=0.01)
    parser.add_argument("--island-prefix", default="n1-6-")
    args = parser.parse_args(argv)
    stage_spec = stage_spec_from_json_identity(
        args.stage_spec_json, args.stage_spec_sha256
    )
    coordinates, contract = build_warm_pool(
        index_path=args.index,
        stage_spec=stage_spec,
        output_dir=args.output,
        pool_size=args.pool_size,
        candidate_limit=args.candidate_limit,
        minimum_distance=args.minimum_distance,
        island_prefix=args.island_prefix,
    )
    print(
        json.dumps(
            {"coordinates": str(coordinates), "contract": str(contract)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
