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
        STRUCTURAL_DONOR_OPTIMIZER_GATES,
        STRUCTURAL_DONOR_REQUIRED_GATES,
        WARM_ROLE_PARTITION_SCHEMA,
        authenticate_warm_handoff,
        canonical_sha256,
        stage_constraint_names,
        stage_spec_from_json_identity,
        validate_stage_spec,
    )
    from tier1_deep_crossover_contract import topology_evolution_contract
except ImportError:  # pragma: no cover - repository module import path
    from tools.tier1_corrected_generation_preflight import (
        RESONANCE_MAXIMUM_CONSTRAINT,
        RESONANCE_MINIMUM_CONSTRAINT,
        STRUCTURAL_DONOR_OPTIMIZER_GATES,
        STRUCTURAL_DONOR_REQUIRED_GATES,
        WARM_ROLE_PARTITION_SCHEMA,
        authenticate_warm_handoff,
        canonical_sha256,
        stage_constraint_names,
        stage_spec_from_json_identity,
        validate_stage_spec,
    )
    from tools.tier1_deep_crossover_contract import topology_evolution_contract


CONTRACT_SCHEMA = "mft-tier1-final1000-warm-pool-v1"
INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
TEMPERATURE_PREFIX = "temperature_robust_limit:"
WIDTH_CONSTRAINT = "exterior_width_limit"
LENGTH_CONSTRAINT = "exterior_length_limit"
HEIGHT_CONSTRAINT = "exterior_height_limit"
BASIN_POOL_SCHEMA = "mft-tier1-final1000-basin-warm-selection-v1"
BASIN_PROTECTED_COPIES_EACH = 8
BASIN_PROTECTED_SELECTION_CAP = 56
JOINT_NICHE_POOL_SCHEMA = (
    "mft-tier1-final1000-joint-niche-warm-selection-v1"
)
JOINT_NICHE_NAMES = (
    "llt_robust_joint",
    "core_center_top_joint",
    "rx_main_joint",
    "width_joint",
)
JOINT_NICHE_COPIES_EACH = 2
JOINT_NICHE_SOURCE_MINIMUMS = {
    "final-1000-t100": 8,
    "close-1075-t107p5": 16,
}


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


def _fixed_primary_turns(
    island_prefix: str, explicit: int | None = None
) -> int:
    """Resolve the fixed-turn identity that remote warm repair authenticates."""

    if explicit is not None:
        if isinstance(explicit, bool) or int(explicit) not in (5, 6):
            raise ValueError("fixed_primary_turns must be 5 or 6")
        turns = int(explicit)
    elif str(island_prefix).startswith("n1-5-"):
        turns = 5
    elif str(island_prefix).startswith("n1-6-"):
        turns = 6
    else:
        raise ValueError(
            "fixed_primary_turns is required for a nonstandard island prefix"
        )
    expected_prefix = f"n1-{turns}-"
    if not str(island_prefix).startswith(expected_prefix):
        raise ValueError("fixed_primary_turns and island_prefix disagree")
    return turns


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


def _n1_6_secondary_main_turns(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] <= 2:
        raise RuntimeError("basin warm coordinates have an invalid schema")
    return 60 - np.rint(48.0 * np.clip(values[:, 2], 0.0, 1.0)).astype(int)


def _basin_aware_indices(
    coordinates: np.ndarray,
    staged_g: np.ndarray,
    objectives: np.ndarray,
    *,
    target_names: tuple[str, ...],
    global_order: np.ndarray,
    pool_size: int,
    candidate_limit: int,
    minimum_distance: float,
) -> tuple[list[int], dict[str, Any]]:
    """Reserve authenticated donors for each audited N1=6 basin."""

    topology = topology_evolution_contract(6)
    required_topologies = tuple(
        int(value) for value in topology["turn_split_sub_islands_N2_main"]
    )
    if (
        int(pool_size) < BASIN_PROTECTED_SELECTION_CAP
        or required_topologies != (34, 35, 36, 37, 38, 39, 60)
        or topology["bounded_diversity_budget"]["protected_slots"]
        != BASIN_PROTECTED_SELECTION_CAP
    ):
        raise RuntimeError("basin warm protected budget contract mismatch")
    positions = {name: index for index, name in enumerate(target_names)}
    required_names = {
        "Llt_robust_band",
        "Llt_ensemble_disagreement",
        WIDTH_CONSTRAINT,
        LENGTH_CONSTRAINT,
        HEIGHT_CONSTRAINT,
    }
    temperature_indices = [
        index
        for index, name in enumerate(target_names)
        if name.startswith(TEMPERATURE_PREFIX)
    ]
    if not required_names <= set(positions) or not temperature_indices:
        raise RuntimeError("basin warm source omitted a ranking constraint")
    n2_main = _n1_6_secondary_main_turns(coordinates)
    global_rank = np.empty(len(global_order), dtype=int)
    global_rank[np.asarray(global_order, dtype=int)] = np.arange(len(global_order))
    llt_order = np.lexsort((
        objectives[:, 1],
        global_rank,
        staged_g[:, positions["Llt_ensemble_disagreement"]],
        staged_g[:, positions["Llt_robust_band"]],
    ))
    temperature = staged_g[:, temperature_indices]
    thermal_order = np.lexsort((
        objectives[:, 1],
        global_rank,
        np.maximum(temperature, 0.0).sum(axis=1),
        temperature.max(axis=1),
    ))
    size_indices = [
        positions[name]
        for name in (WIDTH_CONSTRAINT, LENGTH_CONSTRAINT, HEIGHT_CONSTRAINT)
    ]
    size_g = staged_g[:, size_indices]
    size_order = np.lexsort((
        objectives[:, 1],
        global_rank,
        np.maximum(size_g, 0.0).sum(axis=1),
        np.count_nonzero(size_g > 0.0, axis=1),
    ))
    lane_for_topology = {
        34: "llt_target_mean_q90",
        37: "llt_target_mean_q90",
        35: "thermal_split",
        36: "thermal_split",
        38: "thermal_split",
        39: "thermal_split",
        60: "exact_size_no_side",
    }
    order_for_lane = {
        "llt_target_mean_q90": llt_order,
        "thermal_split": thermal_order,
        "exact_size_no_side": size_order,
    }
    selected: list[int] = []
    protected: list[dict[str, Any]] = []
    availability: dict[str, int] = {}
    for target in required_topologies:
        lane = lane_for_topology[target]
        candidates = [
            int(index)
            for index in order_for_lane[lane]
            if n2_main[int(index)] == target
            and (
                target != 60
                or bool(np.all(size_g[int(index)] <= 0.0))
            )
        ]
        availability[str(target)] = len(candidates)
        if not candidates:
            side = 60 - target
            raise RuntimeError(
                f"basin warm source omitted required {target}/{side} donor"
            )
        topology_selected: list[int] = []
        for index in candidates:
            row = coordinates[index]
            if not topology_selected or min(
                float(np.linalg.norm(row - coordinates[prior]))
                for prior in topology_selected
            ) >= minimum_distance:
                topology_selected.append(index)
            if len(topology_selected) == BASIN_PROTECTED_COPIES_EACH:
                break
        for index in candidates:
            if index not in topology_selected:
                topology_selected.append(index)
            if len(topology_selected) == BASIN_PROTECTED_COPIES_EACH:
                break
        for index in topology_selected:
            if index in selected:
                raise RuntimeError("basin warm donor topology identity overlapped")
            selected.append(index)
            protected.append({
                "source_row_index": index,
                "lane": lane,
                "N2_main": target,
                "N2_side": 60 - target,
            })
    if not 0 < len(protected) <= BASIN_PROTECTED_SELECTION_CAP:
        raise RuntimeError("basin warm protected selection escaped its cap")
    candidates = [
        int(value) for value in global_order[:candidate_limit]
        if int(value) not in selected
    ]
    for index in candidates:
        row = coordinates[index]
        if not selected or min(
            float(np.linalg.norm(row - coordinates[prior]))
            for prior in selected
        ) >= minimum_distance:
            selected.append(index)
        if len(selected) == pool_size:
            break
    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) == pool_size:
            break
    if len(selected) != pool_size:
        raise RuntimeError("basin warm selection could not fill the requested pool")
    audit = {
        "schema_version": BASIN_POOL_SCHEMA,
        "required_topologies_N2_main": list(required_topologies),
        "source_available_counts": availability,
        "requested_copies_each": BASIN_PROTECTED_COPIES_EACH,
        "protected_selection_count": len(protected),
        "protected_selection_cap": BASIN_PROTECTED_SELECTION_CAP,
        "protected_selection": protected,
        "missing_required_topologies": [],
        "coordinate_donors_only": True,
        "prior_prediction_or_pass_classification_inherited": False,
        "downstream_population": int(
            topology["bounded_diversity_budget"]["population"]
        ),
        "downstream_protected_seed_count": BASIN_PROTECTED_SELECTION_CAP,
        "downstream_maximum_combined_authenticated_count": 160,
        "downstream_maximum_standard_hard_feasible_count": 104,
        "downstream_minimum_fresh_random_count": 160,
        "additional_model_evaluations_during_selection": 0,
    }
    audit["sha256"] = canonical_sha256(audit)
    return selected, audit


def _joint_niche_indices(
    coordinates: np.ndarray,
    staged_g: np.ndarray,
    objectives: np.ndarray,
    *,
    target_names: tuple[str, ...],
    constraint_scales: np.ndarray,
    global_order: np.ndarray,
    source_stage_ids: tuple[str, ...],
    pool_size: int,
    minimum_distance: float,
) -> tuple[list[int], dict[str, Any]]:
    """Preserve four exact-final joint niches in every audited N1=6 basin.

    The 56-row reservoir is deliberately the exact downstream protected
    donor budget: two independent donors for each of four physical niches in
    each of seven turn-split topologies.  Consequently the downstream
    half-warm policy consumes every structural donor instead of sampling away
    a rare niche.  Source-stage minimums are enforced without changing any
    objective, physical ``G``, hard tolerance, or pass classification.
    """

    topology = topology_evolution_contract(6)
    required_topologies = tuple(
        int(value) for value in topology["turn_split_sub_islands_N2_main"]
    )
    expected_count = (
        len(required_topologies)
        * len(JOINT_NICHE_NAMES)
        * JOINT_NICHE_COPIES_EACH
    )
    if (
        required_topologies != (34, 35, 36, 37, 38, 39, 60)
        or expected_count != BASIN_PROTECTED_SELECTION_CAP
        or int(pool_size) != expected_count
        or topology["bounded_diversity_budget"]["protected_slots"]
        != expected_count
    ):
        raise RuntimeError("joint-niche warm protected budget mismatch")
    coordinates = np.asarray(coordinates, dtype=float)
    staged_g = np.asarray(staged_g, dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    scales = np.asarray(constraint_scales, dtype=float)
    if (
        coordinates.ndim != 2
        or staged_g.shape != (len(coordinates), len(target_names))
        or objectives.shape != (len(coordinates), 2)
        or scales.shape != (len(target_names),)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(staged_g).all()
        or not np.isfinite(objectives).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
    ):
        raise RuntimeError("joint-niche ranking inventory mismatch")
    if len(source_stage_ids) != len(coordinates):
        raise RuntimeError("joint-niche source-stage inventory mismatch")

    positions = {name: index for index, name in enumerate(target_names)}
    core_center = f"{TEMPERATURE_PREFIX}Tprobe_core_center_max"
    core_top = f"{TEMPERATURE_PREFIX}Tprobe_core_top_yoke_max"
    rx_main = f"{TEMPERATURE_PREFIX}Tprobe_Rx_main_leeward_max"
    required_names = {
        "Llt_robust_band",
        "Llt_ensemble_disagreement",
        core_center,
        core_top,
        rx_main,
        WIDTH_CONSTRAINT,
        LENGTH_CONSTRAINT,
        HEIGHT_CONSTRAINT,
    }
    if not required_names <= set(positions):
        raise RuntimeError("joint-niche warm source omitted a ranking constraint")

    n2_main = _n1_6_secondary_main_turns(coordinates)
    global_order = np.asarray(global_order, dtype=int)
    if (
        global_order.shape != (len(coordinates),)
        or len(np.unique(global_order)) != len(coordinates)
        or np.any(global_order < 0)
        or np.any(global_order >= len(coordinates))
    ):
        raise RuntimeError("joint-niche global ranking is not a permutation")
    global_rank = np.empty(len(global_order), dtype=int)
    global_rank[global_order] = np.arange(len(global_order))
    normalized_positive = np.maximum(staged_g, 0.0) / scales

    def niche_order(indices: tuple[int, ...]) -> np.ndarray:
        positive = normalized_positive[:, indices]
        return np.lexsort((
            objectives[:, 1],
            objectives[:, 0],
            global_rank,
            positive.sum(axis=1),
            positive.max(axis=1),
            np.count_nonzero(positive > 0.0, axis=1),
        ))

    size_indices = tuple(
        positions[name]
        for name in (WIDTH_CONSTRAINT, LENGTH_CONSTRAINT, HEIGHT_CONSTRAINT)
    )
    width_positive = normalized_positive[:, positions[WIDTH_CONSTRAINT]]
    size_positive = normalized_positive[:, size_indices]
    orders = {
        "llt_robust_joint": niche_order((
            positions["Llt_robust_band"],
            positions["Llt_ensemble_disagreement"],
        )),
        "core_center_top_joint": niche_order((
            positions[core_center], positions[core_top]
        )),
        "rx_main_joint": niche_order((positions[rx_main],)),
        "width_joint": np.lexsort((
            objectives[:, 1],
            objectives[:, 0],
            global_rank,
            size_positive.sum(axis=1),
            size_positive.max(axis=1),
            np.count_nonzero(size_positive > 0.0, axis=1),
            width_positive,
            width_positive > 0.0,
        )),
    }
    required_source_stage = {
        34: "final-1000-t100",
        35: "close-1075-t107p5",
        36: "close-1075-t107p5",
        37: "close-1075-t107p5",
        38: "close-1075-t107p5",
        39: None,
        60: "final-1000-t100",
    }
    source_stage_array = np.asarray(source_stage_ids, dtype=object)
    selected: list[int] = []
    protected: list[dict[str, Any]] = []
    availability: dict[str, dict[str, int]] = {}

    def choose(
        candidates: list[int],
        topology_selected: list[int],
        *,
        stage_id: str | None,
    ) -> int:
        stage_eligible = [
            index for index in candidates
            if stage_id is None or source_stage_array[index] == stage_id
        ]
        if not stage_eligible:
            label = stage_id or "any"
            raise RuntimeError(
                f"joint-niche warm source omitted required {label} donor"
            )
        eligible = [index for index in stage_eligible if index not in selected]
        for index in eligible:
            row = coordinates[index]
            if not topology_selected or min(
                float(np.linalg.norm(row - coordinates[prior]))
                for prior in topology_selected
            ) >= minimum_distance:
                return index
        if eligible:
            return eligible[0]
        # One audited topology currently has only seven distinct source rows
        # for eight protected slots.  Reusing a fully authenticated coordinate
        # is explicit and preferable to silently dropping its joint niche;
        # downstream current repair still replays every row.
        return stage_eligible[0]

    for target in required_topologies:
        topology_selected: list[int] = []
        availability[str(target)] = {}
        stage_id = required_source_stage[target]
        for niche in JOINT_NICHE_NAMES:
            candidates = [
                int(index)
                for index in orders[niche]
                if n2_main[int(index)] == target
                and (
                    target != 60
                    or bool(np.all(staged_g[int(index), size_indices] <= 0.0))
                )
            ]
            availability[str(target)][niche] = len(candidates)
            if len(candidates) < JOINT_NICHE_COPIES_EACH:
                side = 60 - target
                raise RuntimeError(
                    f"joint-niche warm source omitted required "
                    f"{target}/{side}/{niche} donors"
                )
            for copy_index in range(JOINT_NICHE_COPIES_EACH):
                source_requirement = stage_id if copy_index == 0 else None
                index = choose(
                    candidates,
                    topology_selected,
                    stage_id=source_requirement,
                )
                selected.append(index)
                topology_selected.append(index)
                protected.append({
                    "source_row_index": index,
                    "lane": niche,
                    "N2_main": target,
                    "N2_side": 60 - target,
                    "required_source_stage_id": source_requirement,
                })

    if len(selected) != expected_count:
        raise RuntimeError("joint-niche warm selection escaped its exact budget")
    reused_source_row_count = len(selected) - len(set(selected))
    selected_stage_counts = {
        stage_id: sum(source_stage_ids[index] == stage_id for index in selected)
        for stage_id in sorted(set(source_stage_ids))
    }
    required_stage_slot_counts = {
        stage_id: sum(
            item["required_source_stage_id"] == stage_id
            for item in protected
        )
        for stage_id in JOINT_NICHE_SOURCE_MINIMUMS
    }
    if required_stage_slot_counts != JOINT_NICHE_SOURCE_MINIMUMS:
        raise RuntimeError("joint-niche source quota slot contract drifted")
    for stage_id, minimum in JOINT_NICHE_SOURCE_MINIMUMS.items():
        if selected_stage_counts.get(stage_id, 0) < minimum:
            raise RuntimeError(
                f"joint-niche source quota underfilled for {stage_id}"
            )
    selected_niche_counts = {
        niche: sum(item["lane"] == niche for item in protected)
        for niche in JOINT_NICHE_NAMES
    }
    selected_topology_counts = {
        str(target): sum(item["N2_main"] == target for item in protected)
        for target in required_topologies
    }
    if (
        set(selected_niche_counts.values()) != {14}
        or set(selected_topology_counts.values()) != {8}
    ):
        raise RuntimeError("joint-niche warm balance contract mismatch")
    audit = {
        "schema_version": JOINT_NICHE_POOL_SCHEMA,
        "required_topologies_N2_main": list(required_topologies),
        "joint_niches": list(JOINT_NICHE_NAMES),
        "copies_per_topology_niche": JOINT_NICHE_COPIES_EACH,
        "source_available_counts": availability,
        "protected_selection_count": len(protected),
        "protected_selection_cap": expected_count,
        "protected_selection": protected,
        "unique_source_row_count": len(set(selected)),
        "reused_source_row_count": reused_source_row_count,
        "authenticated_source_row_reuse_allowed_only_after_unique_exhaustion": True,
        "selected_niche_counts": selected_niche_counts,
        "selected_topology_counts": selected_topology_counts,
        "selected_source_stage_counts": selected_stage_counts,
        "source_stage_minimum_counts": dict(JOINT_NICHE_SOURCE_MINIMUMS),
        "required_source_stage_slot_counts": required_stage_slot_counts,
        "joint_feasibility_ranking": (
            "positive_constraint_count_then_max_normalized_positive_G_"
            "then_sum_normalized_positive_G"
        ),
        "niche_feasibility_rankings": {
            "llt_robust_joint": (
                "positive_constraint_count_then_max_normalized_positive_G_"
                "then_sum_normalized_positive_G"
            ),
            "core_center_top_joint": (
                "positive_constraint_count_then_max_normalized_positive_G_"
                "then_sum_normalized_positive_G"
            ),
            "rx_main_joint": (
                "positive_constraint_count_then_max_normalized_positive_G_"
                "then_sum_normalized_positive_G"
            ),
            "width_joint": (
                "width_positive_state_then_width_normalized_positive_G_then_"
                "size_positive_constraint_count_then_max_normalized_"
                "positive_G_then_sum_normalized_positive_G"
            ),
        },
        "ranking_constraint_scales": {
            name: float(scale) for name, scale in zip(target_names, scales)
        },
        "max_normalized_positive_G_pressure": True,
        "whole_authenticated_population_ranked": True,
        "missing_required_topologies": [],
        "coordinate_donors_only": True,
        "prior_prediction_or_pass_classification_inherited": False,
        "downstream_population": int(
            topology["bounded_diversity_budget"]["population"]
        ),
        "downstream_protected_seed_count": expected_count,
        "downstream_maximum_combined_authenticated_count": 160,
        "downstream_maximum_standard_hard_feasible_count": 104,
        "downstream_minimum_fresh_random_count": 160,
        "additional_model_evaluations_during_selection": 0,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
        "terminal_physical_replay_required": True,
    }
    audit["sha256"] = canonical_sha256(audit)
    return selected, audit


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
    fixed_primary_turns: int | None = None,
    donor_index_paths: tuple[Path, ...] = (),
    basin_aware: bool = False,
    joint_niche_aware: bool = False,
    standard_warm_start: Path | None = None,
    standard_warm_contract: Path | None = None,
) -> tuple[Path, Path]:
    if isinstance(pool_size, bool) or not 4 <= int(pool_size) <= 4096:
        raise ValueError("pool_size must be from 4 through 4096")
    if isinstance(candidate_limit, bool) or int(candidate_limit) < int(pool_size):
        raise ValueError("candidate_limit must be at least pool_size")
    if not math.isfinite(float(minimum_distance)) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be finite and non-negative")
    fixed_turns = _fixed_primary_turns(island_prefix, fixed_primary_turns)
    if not isinstance(basin_aware, bool):
        raise ValueError("basin_aware must be boolean")
    if not isinstance(joint_niche_aware, bool):
        raise ValueError("joint_niche_aware must be boolean")
    if joint_niche_aware and not basin_aware:
        raise ValueError("joint-niche selection requires basin-aware mode")
    if joint_niche_aware and int(pool_size) != BASIN_PROTECTED_SELECTION_CAP:
        raise ValueError(
            "joint-niche selection requires the exact 56-row donor budget"
        )
    normalized_stage = validate_stage_spec(stage_spec)
    if bool(standard_warm_start) != bool(standard_warm_contract):
        raise ValueError(
            "standard warm artifact and contract must be supplied together"
        )
    if basin_aware and standard_warm_start is None:
        raise ValueError(
            "basin-aware output requires an existing standard warm handoff"
        )
    if not basin_aware and standard_warm_start is not None:
        raise ValueError("standard warm composition requires basin-aware mode")
    standard_values = None
    standard_authentication = None
    standard_contract_value = None
    if standard_warm_start is not None and standard_warm_contract is not None:
        standard_contract_path = Path(standard_warm_contract).resolve(strict=True)
        standard_contract_value = _read_json(standard_contract_path)
        if (
            standard_contract_value.get("stage_spec") != normalized_stage
            or standard_contract_value.get("stage_spec_sha256")
            != canonical_sha256(normalized_stage)
            or int(
                (standard_contract_value.get("hard_geometry_audit") or {}).get(
                    "joint_count", 0
                )
            )
            < 1
        ):
            raise RuntimeError(
                "standard warm handoff lacks exact-stage hard-feasible evidence"
            )
        standard_values, standard_authentication = authenticate_warm_handoff(
            Path(standard_warm_start),
            standard_contract_path,
            fixed_primary_turns=fixed_turns,
            n_var=25,
            expected_contract_file_sha256=_sha256_file(
                standard_contract_path
            ),
        )
    index_paths = tuple(
        path.resolve(strict=True)
        for path in (Path(index_path), *map(Path, donor_index_paths))
    )
    if len(index_paths) != len(set(index_paths)):
        raise RuntimeError("basin warm source indexes must be unique")
    coordinates = []
    staged_g_blocks = []
    objectives = []
    metadata: list[dict[str, Any]] = []
    authenticated_artifacts: list[dict[str, Any]] = []
    source_indexes = []
    canonical_index = None
    source_names = None
    source_spec = None
    target_names = None
    for source_index_rank, current_index_path in enumerate(index_paths):
        current_index, status, containment_root = _load_source(
            current_index_path
        )
        source_stage_id = str(
            current_index.get("final_goal_stage_id") or ""
        )
        current_names = tuple(
            str(name) for name in status["constraint_names"]
        )
        current_spec = status["hard_spec"]
        if canonical_index is None:
            canonical_index = current_index
            source_names = current_names
            source_spec = current_spec
        source_terminal_count = 0
        for terminal in status.get("terminal_results") or []:
            if (
                terminal.get("authenticated") is not True
                or terminal.get("terminal_state") != "completed"
                or not str(terminal.get("island_id") or "").startswith(
                    island_prefix
                )
            ):
                continue
            artifacts = terminal.get("artifact_objects") or {}
            paths = {
                key: _artifact(
                    artifacts.get(key) or {}, containment_root, key
                )
                for key in ("terminal_X", "terminal_G_physical", "terminal_F")
            }
            result_sha256 = None
            if basin_aware:
                result_record = terminal.get("result_object") or {}
                result_path = _artifact(
                    result_record, containment_root, "source result"
                )
                result_value = _read_json(result_path)
                result_sha256 = str(result_record.get("sha256") or "")
                if (
                    int(result_value.get("seed", -1))
                    != int(terminal["seed"])
                    or result_value.get("island_id")
                    != terminal.get("island_id")
                ):
                    raise RuntimeError("basin warm source result identity mismatch")
            try:
                x = np.load(paths["terminal_X"], allow_pickle=False)
                g = np.load(paths["terminal_G_physical"], allow_pickle=False)
                f = np.load(paths["terminal_F"], allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "authenticated terminal NumPy artifact is invalid"
                ) from exc
            if (
                x.ndim != 2
                or g.shape != (len(x), len(current_names))
                or f.shape != (len(x), 2)
                or not np.isfinite(x).all()
                or not np.isfinite(g).all()
                or not np.isfinite(f).all()
            ):
                raise RuntimeError("authenticated terminal array schema mismatch")
            current_staged_g, current_target_names = transform_physical_g(
                g,
                source_names=current_names,
                source_spec=current_spec,
                stage_spec=normalized_stage,
            )
            if target_names is None:
                target_names = current_target_names
            elif current_target_names != target_names:
                raise RuntimeError("basin warm target constraint schema drifted")
            coordinates.append(np.asarray(x, dtype=float))
            staged_g_blocks.append(current_staged_g)
            objectives.append(np.asarray(f, dtype=float))
            metadata.extend(
                {
                    "seed": int(terminal["seed"]),
                    "island_id": str(terminal["island_id"]),
                    "terminal_population_index": row,
                    "source_index_rank": source_index_rank,
                    "source_stage_id": source_stage_id,
                    "source_index_sha256": _sha256_file(current_index_path),
                    "source_result_sha256": result_sha256,
                }
                for row in range(len(x))
            )
            object_records = {
                key: {
                    "sha256": str(artifacts[key]["sha256"]),
                    "size_bytes": int(
                        artifacts[key].get(
                            "size_bytes", artifacts[key].get("size")
                        )
                    ),
                }
                for key in paths
            }
            if result_sha256 is not None:
                object_records["source_result"] = {
                    "sha256": result_sha256,
                    "size_bytes": int(
                        (terminal["result_object"]).get(
                            "size_bytes", terminal["result_object"].get("size")
                        )
                    ),
                }
            authenticated_artifacts.append({
                "seed": int(terminal["seed"]),
                "island_id": str(terminal["island_id"]),
                "source_index_rank": source_index_rank,
                "objects": object_records,
            })
            source_terminal_count += 1
        source_indexes.append({
            "rank": source_index_rank,
            "path": str(current_index_path),
            "sha256": _sha256_file(current_index_path),
            "snapshot_sha256": current_index.get("snapshot_sha256"),
            "bundle_id": current_index.get("bundle_id"),
            "final_goal_stage_id": source_stage_id,
            "status": {
                "path": str(Path(current_index["status"]["path"]).resolve()),
                "sha256": str(current_index["status"]["sha256"]),
            },
            "authenticated_terminal_records": source_terminal_count,
        })
    if not coordinates:
        raise RuntimeError("no authenticated terminal coordinates matched the island prefix")
    x_all = np.vstack(coordinates)
    staged_g = np.vstack(staged_g_blocks)
    f_all = np.vstack(objectives)
    unique_keys = np.ascontiguousarray(x_all).view(
        np.dtype((np.void, x_all.dtype.itemsize * x_all.shape[1]))
    ).ravel()
    _, unique_indices = np.unique(unique_keys, return_index=True)
    unique_indices = np.sort(unique_indices)
    x_all = x_all[unique_indices]
    staged_g = staged_g[unique_indices]
    f_all = f_all[unique_indices]
    metadata = [metadata[int(index)] for index in unique_indices]
    if canonical_index is None or source_names is None or source_spec is None:
        raise RuntimeError("warm source identity was not initialized")
    if target_names is None:
        raise RuntimeError("warm target constraint schema was not initialized")
    scales = _constraint_scales(target_names, normalized_stage)
    positive = np.maximum(staged_g, 0.0)
    violation_count = np.count_nonzero(staged_g > 0.0, axis=1)
    normalized_positive = positive / scales
    normalized_sum = np.sum(normalized_positive, axis=1)
    normalized_max = np.max(normalized_positive, axis=1)
    if joint_niche_aware:
        # Chebyshev/minimax pressure prevents one very large hard-constraint
        # miss from hiding behind a small aggregate CV.  Raw physical G and
        # objectives remain untouched and are still replayed downstream.
        order = np.lexsort((
            f_all[:, 1],
            f_all[:, 0],
            normalized_sum,
            normalized_max,
            violation_count,
        ))
    else:
        order = np.lexsort((
            f_all[:, 1], f_all[:, 0], normalized_sum, violation_count
        ))
    basin_selection = None
    if basin_aware:
        if joint_niche_aware:
            selected, basin_selection = _joint_niche_indices(
                x_all,
                staged_g,
                f_all,
                target_names=target_names,
                constraint_scales=scales,
                global_order=order,
                source_stage_ids=tuple(
                    str(item["source_stage_id"]) for item in metadata
                ),
                pool_size=int(pool_size),
                minimum_distance=float(minimum_distance),
            )
        else:
            selected, basin_selection = _basin_aware_indices(
                x_all,
                staged_g,
                f_all,
                target_names=target_names,
                global_order=order,
                pool_size=int(pool_size),
                candidate_limit=min(int(candidate_limit), len(x_all)),
                minimum_distance=float(minimum_distance),
            )
        for item in basin_selection["protected_selection"]:
            source = metadata[int(item["source_row_index"])]
            result_sha = str(source.get("source_result_sha256") or "")
            if len(result_sha) != 64:
                raise RuntimeError(
                    "basin warm protected donor lacks source-result SHA"
                )
            item.update({
                "source_seed": int(source["seed"]),
                "source_island_id": str(source["island_id"]),
                "source_terminal_population_index": int(
                    source["terminal_population_index"]
                ),
                "source_index_rank": int(source["source_index_rank"]),
                "source_stage_id": str(source["source_stage_id"]),
                "source_index_sha256": str(source["source_index_sha256"]),
                "source_result_sha256": result_sha,
            })
        basin_selection.pop("sha256", None)
        basin_selection["sha256"] = canonical_sha256(basin_selection)
    else:
        selected = _diverse_indices(
            x_all,
            order,
            pool_size=min(int(pool_size), len(x_all)),
            candidate_limit=min(int(candidate_limit), len(x_all)),
            minimum_distance=float(minimum_distance),
        )
    if len(selected) < 4:
        raise RuntimeError("staged warm selection produced fewer than four coordinates")
    basin_selected_x = np.asarray(x_all[selected], dtype="<f8")
    standard_count = 0
    if standard_values is not None:
        standard_x = np.asarray(standard_values, dtype="<f8")
        standard_count = int(len(standard_x))
        selected_x = np.vstack((standard_x, basin_selected_x))
    else:
        selected_x = basin_selected_x
    if basin_selection is not None:
        if joint_niche_aware:
            protected = basin_selection["protected_selection"]
            if len(protected) != len(selected):
                raise RuntimeError(
                    "joint-niche protected artifact mapping mismatch"
                )
            for basin_rank, (item, source_index) in enumerate(
                zip(protected, selected)
            ):
                if int(item["source_row_index"]) != int(source_index):
                    raise RuntimeError(
                        "joint-niche protected selection order drifted"
                    )
                item["artifact_row_index"] = standard_count + basin_rank
        else:
            basin_positions = {
                int(source_index): standard_count + basin_rank
                for basin_rank, source_index in enumerate(selected)
            }
            for item in basin_selection["protected_selection"]:
                item["artifact_row_index"] = basin_positions[
                    int(item["source_row_index"])
                ]
        basin_selection.pop("sha256", None)
        basin_selection["sha256"] = canonical_sha256(basin_selection)
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
                "rank": standard_count + rank,
                "basin_rank": rank,
                "artifact_row_index": standard_count + rank,
                **metadata[index],
                "positive_constraint_count": int(violation_count[index]),
                "max_normalized_positive_G": float(normalized_max[index]),
                "normalized_positive_G_sum": float(normalized_sum[index]),
                "objective_volume_L": float(f_all[index, 0]),
                "objective_total_loss_W": float(f_all[index, 1]),
                "coordinate_sha256": hashlib.sha256(row.tobytes()).hexdigest(),
            }
        )
    warm_role_partition = None
    standard_warm_source = None
    if basin_aware:
        if (
            standard_values is None
            or standard_authentication is None
            or standard_contract_value is None
        ):
            raise RuntimeError("basin warm role composition was not authenticated")
        standard_slice = selected_x[:standard_count]
        basin_slice = selected_x[standard_count:]
        standard_warm_source = {
            "artifact": {
                "path": str(Path(standard_warm_start).resolve(strict=True)),
                "sha256": standard_authentication["artifact"]["sha256"],
                "coordinate_unit_sha256": standard_authentication[
                    "artifact"
                ]["coordinate_unit_sha256"],
                "shape": list(standard_slice.shape),
            },
            "contract": {
                "path": str(Path(standard_warm_contract).resolve(strict=True)),
                "file_sha256": standard_authentication[
                    "contract_file_sha256"
                ],
                "canonical_sha256": standard_authentication[
                    "contract_canonical_sha256"
                ],
                "schema_version": standard_authentication[
                    "contract_schema_version"
                ],
            },
            "hard_geometry_joint_count": int(
                standard_contract_value["hard_geometry_audit"]["joint_count"]
            ),
            "remote_current_problem_hard_filter_required": True,
        }
        standard_warm_source["sha256"] = canonical_sha256(
            standard_warm_source
        )
        warm_role_partition = {
            "schema_version": WARM_ROLE_PARTITION_SCHEMA,
            "fixed_primary_turns": fixed_turns,
            "total_count": int(len(selected_x)),
            "standard_hard_feasible_candidates": {
                "start": 0,
                "count": standard_count,
                "coordinate_unit_sha256": canonical_sha256(
                    standard_slice.tolist()
                ),
                "full_hard_geometry_filter_required": True,
                "ordinary_warm_sampling_allowed": True,
                "source_artifact_sha256": standard_authentication[
                    "artifact"
                ]["sha256"],
                "source_contract_file_sha256": standard_authentication[
                    "contract_file_sha256"
                ],
                "source_hard_geometry_joint_count": int(
                    standard_contract_value["hard_geometry_audit"][
                        "joint_count"
                    ]
                ),
            },
            "basin_structural_coordinate_donors": {
                "start": standard_count,
                "count": int(len(basin_slice)),
                "coordinate_unit_sha256": canonical_sha256(
                    basin_slice.tolist()
                ),
                "required_gate_names": list(
                    STRUCTURAL_DONOR_REQUIRED_GATES
                ),
                "optimizer_gate_names_not_used_for_donor_admission": list(
                    STRUCTURAL_DONOR_OPTIMIZER_GATES
                ),
                "required_topologies_N2_main": list(
                    basin_selection["required_topologies_N2_main"]
                ),
                "source_selection_contract_sha256": basin_selection[
                    "sha256"
                ],
                "coordinate_donors_only": True,
                "protected_topology_slots_only": True,
                "ordinary_warm_sampling_allowed": False,
                "prior_prediction_or_pass_classification_inherited": False,
            },
            "maximum_combined_authenticated_fraction": 0.5,
            "minimum_fresh_random_fraction": 0.5,
            "physical_constraint_G_mutation": False,
            "physical_objective_mutation": False,
            "terminal_physical_replay_required": True,
        }
        warm_role_partition["sha256"] = canonical_sha256(
            warm_role_partition
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
        "source_indexes": source_indexes,
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
            "positive_constraint_count_then_max_positive_physical_G_over_"
            "sealed_scale_then_sum_positive_physical_G_over_sealed_scale_"
            "then_volume_then_loss"
            if joint_niche_aware
            else
            "positive_constraint_count_then_sum_positive_physical_G_over_"
            "sealed_scale_then_volume_then_loss"
        ),
        "source_row_count": int(sum(len(value) for value in coordinates)),
        "unique_coordinate_count": int(len(x_all)),
        "source_exact_stage_feasible_count": int(
            np.count_nonzero(np.all(staged_g <= 0.0, axis=1))
        ),
        "selected_count": int(len(selected_x)),
        "standard_hard_feasible_candidate_count": standard_count,
        "basin_structural_candidate_count": int(len(basin_selected_x)),
        "selection_record_scope": "basin_structural_candidate_rows_only",
        "candidate_limit": min(int(candidate_limit), len(x_all)),
        "minimum_coordinate_l2_distance": float(minimum_distance),
        "basin_aware_selection": basin_selection,
        "joint_niche_aware_selection": joint_niche_aware,
        "standard_warm_source": standard_warm_source,
        "selection": selected_records,
        "coordinate_artifact": {
            "path": coordinate_path.name,
            "sha256": _sha256_file(coordinate_path),
            "coordinate_unit_sha256": canonical_sha256(selected_x.tolist()),
            "shape": list(selected_x.shape),
            "dtype": "float64-little-endian",
            "allow_pickle": False,
        },
        # These aliases are the exact fail-closed interface consumed by
        # tier1_corrected_generation_preflight.authenticate_warm_handoff.
        # The optimizer repairs every donor after authentication; no prior
        # objective or constraint classification crosses the handoff.
        "fixed_primary_turns": fixed_turns,
        "fixed_primary_turns_scope": (
            "runner_repair_after_authenticated_inverse_coordinate_handoff"
        ),
        "warm_rows_are_coordinate_donors_only": True,
        "warm_start": {
            "path": coordinate_path.name,
            "sha256": _sha256_file(coordinate_path),
            "shape": list(selected_x.shape),
            "coordinate_contract": (
                "authenticated_coordinate_donors_then_current_repair_v1"
            ),
        },
        "warm_role_partition": warm_role_partition,
        "downstream_repair_required": True,
        "prior_objectives_reused_for_optimizer": False,
        "prior_constraints_reused_for_optimizer": False,
        "source_prediction_or_pass_classification_inherited": False,
        "physical_hard_spec_mutation": False,
        "hard_constraint_mutation": False,
        "objective_mutation": False,
        "terminal_physical_replay_required": True,
        "surrogate_only": True,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    contract["sha256"] = canonical_sha256(contract)
    _atomic_json(contract_path, contract)
    authenticate_warm_handoff(
        coordinate_path,
        contract_path,
        fixed_primary_turns=fixed_turns,
        n_var=int(selected_x.shape[1]),
        expected_contract_file_sha256=_sha256_file(contract_path),
    )
    return coordinate_path, contract_path


def upgrade_existing_handoff_contract(
    *,
    warm_start: Path,
    source_contract: Path,
    output_contract: Path,
    fixed_primary_turns: int,
) -> Path:
    """Reseal an existing staged pool for the Current7 remote authenticator.

    This is intentionally a metadata-only upgrade.  The coordinate bytes are
    authenticated against the original final1000 contract and never copied or
    rewritten, so already generated pools can be made launch-ready quickly.
    """

    turns = _fixed_primary_turns(f"n1-{int(fixed_primary_turns)}-", fixed_primary_turns)
    warm_path = warm_start.resolve(strict=True)
    source_path = source_contract.resolve(strict=True)
    output_path = output_contract.resolve()
    if output_path.exists():
        raise RuntimeError("upgraded warm contract output must not already exist")
    source = _read_json(source_path)
    if source.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("unsupported final1000 source warm contract schema")
    unsigned_source = dict(source)
    handoff_source_seal = unsigned_source.pop("sha256", None)
    if (
        handoff_source_seal is not None
        and handoff_source_seal != canonical_sha256(unsigned_source)
    ):
        raise RuntimeError("source final1000 warm handoff seal mismatch")
    source_seal = unsigned_source.pop("contract_sha256", None)
    if source_seal != canonical_sha256(unsigned_source):
        raise RuntimeError("source final1000 warm contract seal mismatch")
    artifact = source.get("coordinate_artifact") or {}
    expected_sha = str(artifact.get("sha256") or "").lower()
    if _sha256_file(warm_path) != expected_sha:
        raise RuntimeError("source final1000 warm coordinate SHA-256 mismatch")
    try:
        values = np.asarray(np.load(warm_path, allow_pickle=False), dtype=float)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError("source final1000 warm coordinate array is invalid") from exc
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 25
        or list(values.shape) != artifact.get("shape")
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError("source final1000 warm coordinate shape/range mismatch")
    stage_spec = validate_stage_spec(source.get("stage_spec") or {})
    if source.get("stage_spec_sha256") != canonical_sha256(stage_spec):
        raise RuntimeError("source final1000 warm stage identity mismatch")

    upgraded = dict(source)
    upgraded.pop("sha256", None)
    upgraded.pop("contract_sha256", None)
    upgraded.update(
        {
            "fixed_primary_turns": turns,
            "fixed_primary_turns_scope": (
                "runner_repair_after_authenticated_inverse_coordinate_handoff"
            ),
            "warm_start": {
                "path": warm_path.name,
                "sha256": expected_sha,
                "shape": list(values.shape),
                "coordinate_contract": (
                    "authenticated_coordinate_donors_then_current_repair_v1"
                ),
            },
            "warm_rows_are_coordinate_donors_only": True,
            "physical_hard_spec_mutation": False,
            "objective_mutation": False,
            "compatibility_upgrade": {
                "schema_version": (
                    "mft-tier1-final1000-current7-warm-compatibility-v1"
                ),
                "source_contract_file_sha256": _sha256_file(source_path),
                "source_contract_canonical_sha256": source_seal,
                "coordinate_bytes_rewritten": False,
            },
        }
    )
    upgraded["contract_sha256"] = canonical_sha256(upgraded)
    upgraded["sha256"] = canonical_sha256(upgraded)
    _atomic_json(output_path, upgraded)
    authenticate_warm_handoff(
        warm_path,
        output_path,
        fixed_primary_turns=turns,
        n_var=int(values.shape[1]),
        expected_contract_file_sha256=_sha256_file(output_path),
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--stage-spec-json")
    parser.add_argument("--stage-spec-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pool-size", type=int, default=320)
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--minimum-distance", type=float, default=0.01)
    parser.add_argument("--island-prefix", default="n1-6-")
    parser.add_argument("--fixed-primary-turns", type=int, choices=(5, 6))
    parser.add_argument("--upgrade-existing-contract", type=Path)
    parser.add_argument("--upgrade-existing-warm", type=Path)
    parser.add_argument("--upgrade-output-contract", type=Path)
    parser.add_argument(
        "--donor-index",
        type=Path,
        action="append",
        default=[],
        help="additional authenticated stage index (repeatable)",
    )
    parser.add_argument("--basin-aware", action="store_true")
    parser.add_argument("--joint-niche-aware", action="store_true")
    parser.add_argument("--standard-warm-start", type=Path)
    parser.add_argument("--standard-warm-contract", type=Path)
    args = parser.parse_args(argv)
    upgrade_values = (
        args.upgrade_existing_contract,
        args.upgrade_existing_warm,
        args.upgrade_output_contract,
    )
    if any(value is not None for value in upgrade_values):
        if (
            any(value is None for value in upgrade_values)
            or args.fixed_primary_turns is None
            or any(
                value is not None
                for value in (
                    args.index,
                    args.stage_spec_json,
                    args.stage_spec_sha256,
                    args.output,
                    args.standard_warm_start,
                    args.standard_warm_contract,
                )
            )
            or bool(args.donor_index)
            or args.basin_aware
            or args.joint_niche_aware
        ):
            parser.error(
                "contract upgrade requires the three --upgrade-* paths and "
                "--fixed-primary-turns, without pool-build inputs"
            )
        upgraded = upgrade_existing_handoff_contract(
            warm_start=args.upgrade_existing_warm,
            source_contract=args.upgrade_existing_contract,
            output_contract=args.upgrade_output_contract,
            fixed_primary_turns=args.fixed_primary_turns,
        )
        print(json.dumps({"contract": str(upgraded)}, sort_keys=True))
        return 0
    if any(
        value is None
        for value in (
            args.index,
            args.stage_spec_json,
            args.stage_spec_sha256,
            args.output,
        )
    ):
        parser.error(
            "pool build requires --index, --stage-spec-json, "
            "--stage-spec-sha256, and --output"
        )
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
        fixed_primary_turns=args.fixed_primary_turns,
        donor_index_paths=tuple(args.donor_index),
        basin_aware=args.basin_aware,
        joint_niche_aware=args.joint_niche_aware,
        standard_warm_start=args.standard_warm_start,
        standard_warm_contract=args.standard_warm_contract,
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
