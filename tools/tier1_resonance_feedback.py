"""Authenticated target-only feedback from Tier-1 resonance FEA.

This module deliberately does *not* append Tier-1 rows to the canonical
``train.parquet``.  Tier-1 solves contain matrix and electrostatic outputs but
not the loss and thermal families required by the strict-full contract.  The
only admissible use is an isolated target-specific sidecar and experimental
models for ``Llt_phys``, ``k`` and the three capacitance coefficients.

The command has six independent stages:

``audit``
    Authenticate the immutable plan and every verified result, then write a
    SHA-addressed target-only JSONL sidecar.
``compose``
    Combine independently authenticated sidecars without trusting either
    runtime as a superset.  Every source audit, row and scheduler identity is
    re-authenticated and cross-cohort duplicates fail closed.
``train``
    Recompute the exact solver/library strict-full cohort, evaluate every
    authenticated Tier-1 point out-of-fold, tune its cohort weight, and build an isolated
    five-target model.  Loss and thermal models are referenced only from an
    authenticated strict-full generation.
``search-seed``
    Run one corrected hard-constraint NSGA seed using an explicitly pinned
    NSGA code revision.  It never publishes a model or submits FEA.
``launch``
    Start several independent ``search-seed`` processes and record their PIDs.
``monitor``
    Continuously attest process identity, resource use, result state and the
    last completed generation for every launched seed.

All mutable output lives below an operator-supplied runtime directory.
Production eligibility and automatic promotion are always false.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tier1_n1_6_anchor_island_contract import (
        install_optimizer_allowance,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_n1_6_anchor_island_contract import (
        install_optimizer_allowance,
    )


SIDECAR_SCHEMA = "mft-tier1-target-measurement-sidecar-v1"
AUDIT_SCHEMA = "mft-tier1-target-measurement-audit-v1"
COMPOSITION_SCHEMA = "mft-tier1-target-measurement-composition-v1"
MODEL_SCHEMA = "mft-tier1-target-model-v1"
SEARCH_SCHEMA = "mft-tier1-corrected-search-seed-v1"
MANIFEST_SCHEMA = "mft-tier1-feedback-manifest-v1"
TARGETS = ("Llt_phys", "k", "C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F")
CAPACITANCE_TARGETS = ("C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F")
TRANSFORMS = {
    "Llt_phys": "log",
    "k": None,
    "C_tx_tx_F": "log",
    "C_rx_rx_F": "log",
    "C_tx_rx_F": "log",
}
WEIGHT_CANDIDATES = (1.0, 3.0, 8.0, 16.0, 32.0)
CONFORMAL_HALF_WIDTH_CONTRACT = "q90_conformal_half_width_physical_v1"
TARGET_CONFORMAL_CALIBRATION_POLICY = (
    "authenticated_tier1_partial_oof_q90_target_specific_v2"
)
EXPECTED_NSGA_REVISION = "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
CONSTRAINT_VERSION = (
    "mft-tier1-envelope-1200x1200x750-res15k-t110all11-core4-5t-lmhalf-v4"
)
HARD_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    # Revision 175 evaluates every SURROGATE_TEMPERATURE_TARGETS member:
    # nine unconditional maxima/probes and two side-winding targets when
    # N2_side > 0.  One robust upper bound applies to the complete set.
    "T_limit_C": 110.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "n_core_group_max": 4,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": 15_000.0,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_200.0,
    "size_L_max_mm": 1_200.0,
    "size_H_max_mm": 750.0,
    "q_sigma": 1.0,
    "uncertainty_contract": CONFORMAL_HALF_WIDTH_CONTRACT,
}
TEMPERATURE_TARGETS = (
    "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
    "Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max", "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
SIDE_TEMPERATURE_TARGETS = (
    "T_max_Rx_side", "Tprobe_Rx_side_leeward_max",
)
TEMPERATURE_CONSTRAINT_CONTRACT = {
    "schema_version": "mft-tier1-temperature-constraint-contract-v3",
    "semantic_version": "all11-robust-q90-half-width-t110-v4",
    "robust_upper_bound_C": 110.0,
    "hard_spec_temperature_key": "T_limit_C",
    "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
    "uncertainty_contract": CONFORMAL_HALF_WIDTH_CONTRACT,
    "targets": list(TEMPERATURE_TARGETS),
    "target_count": len(TEMPERATURE_TARGETS),
    "unconditional_targets": [
        target for target in TEMPERATURE_TARGETS
        if target not in SIDE_TEMPERATURE_TARGETS
    ],
    "unconditional_target_count": 9,
    "side_winding_conditional_targets": list(SIDE_TEMPERATURE_TARGETS),
    "side_winding_conditional_target_count": 2,
    "side_winding_activation": "finite_N2_side_gt_0",
    "side_winding_absent_behavior": "finite_N2_side_eq_0_disables_with_negative_BIG",
    "side_winding_missing_behavior": "missing_or_nonfinite_N2_side_fails_with_positive_BIG",
    "constraint_name_format": "temperature_robust_limit:{target}",
    "source": "model_targets.SURROGATE_TEMPERATURE_TARGETS",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


TEMPERATURE_CONSTRAINT_CONTRACT_SHA256 = _json_sha(
    TEMPERATURE_CONSTRAINT_CONTRACT
)
OPTIMIZER_CONSTRAINT_NORMALIZATION_SCHEMA = (
    "mft-tier1-optimizer-constraint-normalization-v1"
)
CORE_THERMAL_PRESSURE_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
ALL_THERMAL_PRESSURE_CONSTRAINTS = tuple(
    f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
)
THERMAL_CROSSOVER_ACQUISITION_SCHEMA = (
    "mft-tier1-thermal-crossover-acquisition-ranking-v1"
)
FIXED_PRIMARY_TURNS_SCHEMA = "mft-tier1-fixed-primary-turns-repair-v1"
OPTIMIZER_TERMINATION_SCHEMA = "mft-tier1-optimizer-termination-v1"
DEFAULT_TERMINATION_STRATEGY = "default-ftol-period30-v1"
FIXED_GENERATION_TERMINATION_STRATEGY = "fixed-n-gen-no-ftol-v1"
THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM = 1_000.0
THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM = 100.0
if HARD_SPEC.get("T_limit_C") != TEMPERATURE_CONSTRAINT_CONTRACT[
    "robust_upper_bound_C"
]:
    raise RuntimeError("hard spec and temperature contract upper bounds differ")


def optimizer_termination_contract(
    strategy: str, max_generations: int,
) -> dict[str, Any]:
    """Seal the optimizer stop rule without changing physical evaluation."""

    if (
        isinstance(max_generations, bool)
        or int(max_generations) != max_generations
        or int(max_generations) < 1
    ):
        raise ValueError("max generations must be a positive integer")
    max_generations = int(max_generations)
    if strategy == DEFAULT_TERMINATION_STRATEGY:
        rule = {
            "kind": "pymoo_DefaultMultiObjectiveTermination",
            "ftol": 0.0025,
            "period": 30,
            "n_max_gen": max_generations,
            "early_stop_allowed": True,
        }
    elif strategy == FIXED_GENERATION_TERMINATION_STRATEGY:
        rule = {
            "kind": "pymoo_minimize_tuple_n_gen",
            "n_max_gen": max_generations,
            "early_stop_allowed": False,
            "completed_evolution_generations": max_generations,
            "expected_algorithm_n_gen_counter": max_generations + 1,
        }
    else:
        raise ValueError(f"unsupported optimizer termination strategy: {strategy}")
    contract = {
        "schema_version": OPTIMIZER_TERMINATION_SCHEMA,
        "strategy": strategy,
        "rule": rule,
        "pinned_initial_population": (
            "optimization.run_nsga2._initial_population"
        ),
        "pinned_physics_repair": (
            "optimization.run_nsga2._physics_repair_operator"
        ),
        "physical_evaluator_mutation": False,
        "physical_constraint_mutation": False,
        "objective_mutation": False,
        "authoritative_terminal_G": "physical_unscaled_replay",
    }
    contract["sha256"] = _json_sha(contract)
    return contract


def optimizer_constraint_normalization(
    constraint_names: Iterable[str], spec: dict[str, Any],
    *, resonance_scale_hz: float | None = None,
    core_thermal_scale_c: float | None = None,
    llt_scale_uh: float | None = None,
    all_thermal_scale_c: float | None = None,
) -> dict[str, Any]:
    """Return positive engineering scales used only for optimizer pressure.

    The authoritative physical ``G`` values remain unchanged and are replayed
    for every terminal result.  Dividing by these scales prevents a resonance
    miss expressed in Hz, or a size miss expressed in mm, from numerically
    drowning Llt, density, and thermal violations merely because their units
    are smaller.
    """

    names = tuple(str(name) for name in constraint_names)
    if not names or len(set(names)) != len(names):
        raise RuntimeError("optimizer constraint names must be unique")
    if resonance_scale_hz is not None:
        if isinstance(resonance_scale_hz, bool):
            raise RuntimeError("optimizer resonance scale must be finite and positive")
        resonance_scale_hz = float(resonance_scale_hz)
        if not math.isfinite(resonance_scale_hz) or resonance_scale_hz <= 0.0:
            raise RuntimeError("optimizer resonance scale must be finite and positive")
    if core_thermal_scale_c is not None:
        if isinstance(core_thermal_scale_c, bool):
            raise RuntimeError(
                "optimizer core thermal scale must be finite and positive"
            )
        core_thermal_scale_c = float(core_thermal_scale_c)
        if (
            not math.isfinite(core_thermal_scale_c)
            or core_thermal_scale_c <= 0.0
        ):
            raise RuntimeError(
                "optimizer core thermal scale must be finite and positive"
            )
        missing = set(CORE_THERMAL_PRESSURE_CONSTRAINTS).difference(names)
        if missing:
            raise RuntimeError(
                "optimizer core thermal pressure requires exactly the sealed "
                f"three constraints; missing {sorted(missing)}"
            )
    if all_thermal_scale_c is not None:
        if core_thermal_scale_c is not None:
            raise RuntimeError(
                "optimizer core and all-thermal scale overrides are mutually "
                "exclusive"
            )
        if isinstance(all_thermal_scale_c, bool):
            raise RuntimeError(
                "optimizer all-thermal scale must be finite and positive"
            )
        all_thermal_scale_c = float(all_thermal_scale_c)
        if (
            not math.isfinite(all_thermal_scale_c)
            or all_thermal_scale_c <= 0.0
        ):
            raise RuntimeError(
                "optimizer all-thermal scale must be finite and positive"
            )
        missing = set(ALL_THERMAL_PRESSURE_CONSTRAINTS).difference(names)
        if missing:
            raise RuntimeError(
                "optimizer all-active thermal pressure requires the sealed "
                f"all-11 constraint set; missing {sorted(missing)}"
            )
    if llt_scale_uh is not None:
        if isinstance(llt_scale_uh, bool):
            raise RuntimeError(
                "optimizer Llt scale must be finite and positive"
            )
        llt_scale_uh = float(llt_scale_uh)
        if not math.isfinite(llt_scale_uh) or llt_scale_uh <= 0.0:
            raise RuntimeError(
                "optimizer Llt scale must be finite and positive"
            )
        missing = {
            "Llt_robust_band", "Llt_ensemble_disagreement",
        }.difference(names)
        if missing:
            raise RuntimeError(
                "optimizer Llt pressure requires both sealed Llt constraints; "
                f"missing {sorted(missing)}"
            )
    scales: dict[str, float] = {}
    for name in names:
        if name == "Llt_robust_band":
            scale = (
                llt_scale_uh
                if llt_scale_uh is not None
                else float(spec["Llt_tol_uH"])
            )
        elif name.startswith("temperature_robust_limit:"):
            scale = max(5.0, 0.05 * float(spec["T_limit_C"]))
        elif name == "analytical_flux_density_limit":
            scale = max(0.05, 0.1 * float(spec["B_limit_T"]))
        elif name == "decoded_space_shrink":
            scale = 1.0
        elif name == "minimum_physical_insulation":
            scale = max(2.0, 0.1 * float(spec["insulation_min_mm"]))
        elif name == "strict_full_density_support":
            scale = 0.1
        elif name == "Llt_ensemble_disagreement":
            scale = 2.0 * (
                llt_scale_uh
                if llt_scale_uh is not None
                else float(spec["Llt_tol_uH"])
            )
        elif name == "core_group_manufacturability_limit":
            scale = 1.0
        elif name == "half_magnetizing_resonance_minimum":
            scale = (
                resonance_scale_hz
                if resonance_scale_hz is not None
                else max(1_000.0, 0.05 * float(spec["resonance_min_Hz"]))
            )
        elif name == "exterior_width_limit":
            scale = max(25.0, 0.05 * float(spec["size_W_max_mm"]))
        elif name == "exterior_length_limit":
            scale = max(25.0, 0.05 * float(spec["size_L_max_mm"]))
        elif name == "exterior_height_limit":
            scale = max(25.0, 0.05 * float(spec["size_H_max_mm"]))
        else:
            raise RuntimeError(
                f"optimizer constraint normalization is undefined: {name}"
            )
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                f"optimizer constraint normalization is invalid: {name}"
            )
        if (
            core_thermal_scale_c is not None
            and name in CORE_THERMAL_PRESSURE_CONSTRAINTS
        ):
            scale = core_thermal_scale_c
        if (
            all_thermal_scale_c is not None
            and name in ALL_THERMAL_PRESSURE_CONSTRAINTS
        ):
            scale = all_thermal_scale_c
        scales[name] = float(scale)
    contract = {
        "schema_version": OPTIMIZER_CONSTRAINT_NORMALIZATION_SCHEMA,
        "purpose": "optimizer_search_pressure_only",
        "authoritative_terminal_G": "physical_unscaled",
        "formula": "G_optimizer=G_physical/engineering_scale",
        "constraint_order": list(names),
        "scales": scales,
    }
    if core_thermal_scale_c is not None:
        contract.update({
            "optimizer_core_thermal_scale_C": core_thermal_scale_c,
            "explicit_optimizer_only_scale_overrides": {
                name: core_thermal_scale_c
                for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
            },
            "override_scope": (
                "exactly_three_core_thermal_constraints_"
                "physical_G_unchanged"
            ),
        })
    if llt_scale_uh is not None or all_thermal_scale_c is not None:
        explicit_overrides = dict(
            contract.get("explicit_optimizer_only_scale_overrides") or {}
        )
        if llt_scale_uh is not None:
            explicit_overrides.update({
                "Llt_robust_band": llt_scale_uh,
                "Llt_ensemble_disagreement": 2.0 * llt_scale_uh,
            })
            contract["optimizer_Llt_scale_uH"] = llt_scale_uh
        if all_thermal_scale_c is not None:
            explicit_overrides.update({
                name: all_thermal_scale_c
                for name in ALL_THERMAL_PRESSURE_CONSTRAINTS
            })
            contract.update({
                "optimizer_all_active_thermal_scale_C": (
                    all_thermal_scale_c
                ),
                "optimizer_all_active_thermal_constraints": list(
                    ALL_THERMAL_PRESSURE_CONSTRAINTS
                ),
                "all_active_thermal_side_activation": (
                    "finite_N2_side_gt_0_else_physical_negative_BIG"
                ),
            })
        contract["explicit_optimizer_only_scale_overrides"] = (
            explicit_overrides
        )
        contract["override_scope"] = (
            "sealed_Llt_and_all_active_thermal_constraints_"
            "physical_G_unchanged"
        )
    contract["sha256"] = _json_sha(contract)
    return contract


def install_optimizer_constraint_normalization(
    problem: Any, spec: dict[str, Any],
    *, resonance_scale_hz: float | None = None,
    core_thermal_scale_c: float | None = None,
    llt_scale_uh: float | None = None,
    all_thermal_scale_c: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Normalize a problem in-place while retaining its physical evaluator."""

    names = tuple(str(name) for name in problem.constraint_names)
    contract = optimizer_constraint_normalization(
        names, spec,
        resonance_scale_hz=resonance_scale_hz,
        core_thermal_scale_c=core_thermal_scale_c,
        llt_scale_uh=llt_scale_uh,
        all_thermal_scale_c=all_thermal_scale_c,
    )
    scale_vector = np.asarray(
        [contract["scales"][name] for name in names], dtype=float,
    )
    physical_evaluate = problem._evaluate

    def normalized_evaluate(X, out, *args, **kwargs):
        physical: dict[str, Any] = {}
        physical_evaluate(X, physical, *args, **kwargs)
        if "G" not in physical:
            raise RuntimeError("physical optimizer evaluation omitted G")
        physical_g = np.asarray(physical["G"], dtype=float)
        if physical_g.ndim != 2 or physical_g.shape[1] != len(scale_vector):
            raise RuntimeError("physical optimizer constraint width drifted")
        out.update(physical)
        out["G"] = physical_g / scale_vector

    problem._evaluate = normalized_evaluate
    return physical_evaluate, contract


def install_fixed_primary_turns_repair(
    problem: Any,
    fixed_primary_turns: int | None,
    *,
    coordinate_index: int,
    minimum_turns: int,
    maximum_turns: int,
) -> dict[str, Any] | None:
    """Force one trained primary-turn stratum through every repair call.

    The NSGA code remains the authenticated revision pinned by this feedback
    lane.  This adapter narrows only ``u_N1`` before the code revision's own
    fixed-point physics repair, and rejects any evaluation that bypasses the
    adapter.  All physical constraints and both physical objectives are left
    unchanged.
    """

    if fixed_primary_turns is None:
        return None
    if (
        isinstance(fixed_primary_turns, bool)
        or int(fixed_primary_turns) != fixed_primary_turns
    ):
        raise ValueError("fixed primary turns must be an integer")
    fixed_primary_turns = int(fixed_primary_turns)
    minimum_turns = int(minimum_turns)
    maximum_turns = int(maximum_turns)
    coordinate_index = int(coordinate_index)
    if not minimum_turns <= fixed_primary_turns <= maximum_turns:
        raise ValueError(
            "fixed primary turns are outside the authenticated decoder range"
        )
    if not 0 <= coordinate_index < int(problem.n_var):
        raise ValueError("fixed primary-turn coordinate index is invalid")
    scale = maximum_turns - minimum_turns + 0.9999
    fixed_unit = float(np.clip(
        (fixed_primary_turns - minimum_turns + 0.5) / scale, 0.0, 1.0,
    ))
    contract = {
        "schema_version": FIXED_PRIMARY_TURNS_SCHEMA,
        "fixed_primary_turns": fixed_primary_turns,
        "coordinate_name": "u_N1",
        "coordinate_index": coordinate_index,
        "fixed_unit_coordinate": fixed_unit,
        "decoder_minimum_turns": minimum_turns,
        "decoder_maximum_turns": maximum_turns,
        "repair_order": "force_u_N1_then_authenticated_physics_fixed_point",
        "scope": "initialization_warm_start_and_every_offspring_repair",
        "evaluation_bypass_policy": "fail_closed",
        "warm_start_source_authentication": (
            "verify_source_sha256_before_deterministic_fixed_turn_repair"
        ),
        "warm_start_post_repair_dedupe": "decoded_physical_geometry_identity",
        "warm_start_post_repair_minimum_unique_count": 32,
        "hard_constraint_mutation": False,
        "objective_mutation": False,
    }
    contract["sha256"] = _json_sha(contract)

    existing = getattr(problem, "_fixed_primary_turns_contract", None)
    if existing is not None:
        if existing != contract:
            raise RuntimeError("conflicting fixed primary-turn repair installed")
        return existing
    original_repair = problem.repair_unit_coordinates
    original_evaluate = problem._evaluate

    def fixed_repair(coordinates):
        values = np.asarray(coordinates, dtype=float)
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != int(problem.n_var):
            raise ValueError("NSGA coordinates have the wrong shape")
        forced = values.copy()
        forced[:, coordinate_index] = fixed_unit
        repaired = np.asarray(original_repair(forced), dtype=float)
        if repaired.ndim == 1:
            repaired = repaired.reshape(1, -1)
        if (
            repaired.shape != forced.shape
            or not np.isclose(
                repaired[:, coordinate_index], fixed_unit,
                rtol=0.0, atol=1e-15,
            ).all()
        ):
            raise RuntimeError(
                "authenticated physics repair escaped fixed primary turns"
            )
        return repaired[0] if one_dimensional else repaired

    def fixed_evaluate(X, out, *args, **kwargs):
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if (
            values.ndim != 2
            or values.shape[1] != int(problem.n_var)
            or not np.isclose(
                values[:, coordinate_index], fixed_unit,
                rtol=0.0, atol=1e-15,
            ).all()
        ):
            raise RuntimeError(
                "optimizer evaluation bypassed fixed primary-turn repair"
            )
        return original_evaluate(X, out, *args, **kwargs)

    problem.repair_unit_coordinates = fixed_repair
    problem._evaluate = fixed_evaluate
    problem._fixed_primary_turns_contract = contract
    return contract


def prepare_fixed_primary_turns_warm_start(
    problem: Any,
    warm: np.ndarray,
    contract: dict[str, Any] | None,
    *,
    source_sha_verified: bool,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Repair, hard-filter, and geometry-dedupe an authenticated warm pool."""

    values = np.asarray(warm, dtype=float)
    if source_sha_verified is not True:
        raise RuntimeError("warm source SHA must be verified before repair")
    if contract is None:
        return values, None
    if values.ndim != 2 or values.shape[1] != int(problem.n_var):
        raise RuntimeError("fixed-turn warm coordinate schema mismatch")
    repaired = np.asarray(problem.repair_unit_coordinates(values), dtype=float)
    coordinate_index = int(contract["coordinate_index"])
    fixed_unit = float(contract["fixed_unit_coordinate"])
    fixed_verified = bool(
        repaired.shape == values.shape
        and np.isclose(
            repaired[:, coordinate_index], fixed_unit,
            rtol=0.0, atol=1e-15,
        ).all()
    )
    if not fixed_verified:
        raise RuntimeError("warm pool escaped fixed primary-turn repair")
    filtered, filter_audit = problem.filter_warm_start_coordinates(repaired)
    filtered = np.asarray(filtered, dtype=float)
    minimum_unique = int(
        contract["warm_start_post_repair_minimum_unique_count"]
    )
    unique_count = int(filter_audit.get("decoded_unique_count", -1))
    if (
        filtered.ndim != 2
        or filtered.shape[1] != int(problem.n_var)
        or unique_count != len(filtered)
        or unique_count < minimum_unique
    ):
        raise RuntimeError(
            "fixed-turn warm pool lost required post-repair diversity"
        )
    byte_view = np.ascontiguousarray(filtered, dtype=np.float64)
    return filtered, {
        "schema_version": "mft-tier1-fixed-primary-turns-warm-audit-v1",
        "source_coordinate_count": int(len(values)),
        "post_repair_hard_feasible_count": int(
            filter_audit.get("hard_feasible_count", -1)
        ),
        "post_repair_decoded_unique_count": unique_count,
        "post_repair_rejected_count": int(
            filter_audit.get("rejected_count", -1)
        ),
        "minimum_unique_count": minimum_unique,
        "fixed_primary_turns": int(contract["fixed_primary_turns"]),
        "fixed_primary_turn_coordinate_verified": fixed_verified,
        "post_repair_coordinates_sha256": hashlib.sha256(
            byte_view.tobytes(order="C")
        ).hexdigest(),
        "post_repair_coordinate_shape": list(byte_view.shape),
        "post_repair_coordinate_dtype": "float64",
        "source_sha_verified_before_repair": source_sha_verified,
        "physical_geometry_dedupe_performed": True,
    }


def thermal_crossover_acquisition_contract(
    spec: dict[str, Any], core_thermal_scale_c: float | None,
) -> dict[str, Any]:
    """Seal ranking-only thermal and per-axis size pressure semantics."""

    active = core_thermal_scale_c is not None
    if active:
        if isinstance(core_thermal_scale_c, bool):
            raise RuntimeError(
                "acquisition core thermal scale must be finite and positive"
            )
        core_thermal_scale_c = float(core_thermal_scale_c)
        if (
            not math.isfinite(core_thermal_scale_c)
            or core_thermal_scale_c <= 0.0
        ):
            raise RuntimeError(
                "acquisition core thermal scale must be finite and positive"
            )
    contract = {
        "schema_version": THERMAL_CROSSOVER_ACQUISITION_SCHEMA,
        "active": active,
        "purpose": "candidate_acquisition_ranking_only",
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "hard_constraint_mutation": False,
        "core_thermal_constraints": list(CORE_THERMAL_PRESSURE_CONSTRAINTS),
        "core_thermal_positive_G_scale_C": core_thermal_scale_c,
        "soft_axis_targets_mm": {
            "exterior_width": THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM,
            "exterior_length": THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM,
        },
        "soft_axis_positive_excess_scales_mm": {
            "exterior_width": THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM,
            "exterior_length": THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM,
        },
        "soft_axis_mode": (
            "independent_positive_excess_above_1000mm_no_new_hard_G"
        ),
        "physical_axis_recovery": {
            "exterior_width_mm": (
                "G_exterior_width_limit+hard_spec.size_W_max_mm"
            ),
            "exterior_length_mm": (
                "G_exterior_length_limit+hard_spec.size_L_max_mm"
            ),
        },
        "hard_axis_limits_mm": {
            "exterior_width": float(spec["size_W_max_mm"]),
            "exterior_length": float(spec["size_L_max_mm"]),
        },
        "ranking_formula": (
            "base_target_score+sum(max(core_thermal_G_C,0)/core_scale_C)"
            "+max(exterior_width_mm-1000,0)/100"
            "+max(exterior_length_mm-1000,0)/100"
        ),
    }
    contract["sha256"] = _json_sha(contract)
    return contract


def thermal_crossover_acquisition_pressure(
    physical_g: np.ndarray,
    constraint_names: Iterable[str],
    spec: dict[str, Any],
    core_thermal_scale_c: float | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Compute ranking pressure without mutating physical constraint values."""

    names = tuple(str(name) for name in constraint_names)
    matrix = np.asarray(physical_g, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise RuntimeError("acquisition physical G shape mismatch")
    if not np.isfinite(matrix).all():
        raise RuntimeError("acquisition physical G is non-finite")
    contract = thermal_crossover_acquisition_contract(
        spec, core_thermal_scale_c,
    )
    zero = np.zeros(matrix.shape[0], dtype=float)
    if not contract["active"]:
        components = {
            "core_thermal_positive_normalized": zero.copy(),
            "soft_width_positive_excess_normalized": zero.copy(),
            "soft_length_positive_excess_normalized": zero.copy(),
        }
        return zero, components, contract
    required = (
        *CORE_THERMAL_PRESSURE_CONSTRAINTS,
        "exterior_width_limit",
        "exterior_length_limit",
    )
    missing = set(required).difference(names)
    if missing:
        raise RuntimeError(
            f"thermal crossover acquisition constraints missing: {sorted(missing)}"
        )
    positions = {name: names.index(name) for name in required}
    core = sum(
        np.maximum(matrix[:, positions[name]], 0.0)
        / float(core_thermal_scale_c)
        for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
    )
    width_mm = (
        matrix[:, positions["exterior_width_limit"]]
        + float(spec["size_W_max_mm"])
    )
    length_mm = (
        matrix[:, positions["exterior_length_limit"]]
        + float(spec["size_L_max_mm"])
    )
    width = np.maximum(
        width_mm - THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM, 0.0,
    ) / THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM
    length = np.maximum(
        length_mm - THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM, 0.0,
    ) / THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM
    components = {
        "core_thermal_positive_normalized": np.asarray(core, dtype=float),
        "soft_width_positive_excess_normalized": np.asarray(
            width, dtype=float
        ),
        "soft_length_positive_excess_normalized": np.asarray(
            length, dtype=float
        ),
    }
    total = sum(components.values(), start=zero.copy())
    return np.asarray(total, dtype=float), components, contract


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: os.PathLike[str] | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(
        value, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n")


def _atomic_pickle(path: Path, value: Any) -> None:
    _atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{name} must be a positive finite number")
    return result


def _lc_hz(inductance_h: float, capacitance_f: float) -> float:
    return 1.0 / (2.0 * math.pi * math.sqrt(inductance_h * capacitance_f))


def derive_resonance_screen(measurements: dict, params: dict) -> dict:
    """Derive the unchanged 0.5-magnetizing NSGA resonance screen."""

    leakage_h = _finite_positive(measurements.get("Llt_phys"), "Llt_phys") * 1e-6
    coupling = float(measurements.get("k"))
    if not math.isfinite(coupling) or not 0.0 < coupling < 1.0:
        raise RuntimeError("k must satisfy 0 < k < 1")
    c_tx = _finite_positive(measurements.get("C_tx_tx_F"), "C_tx_tx_F")
    c_rx = _finite_positive(measurements.get("C_rx_rx_F"), "C_rx_rx_F")
    c_cross = _finite_positive(measurements.get("C_tx_rx_F"), "C_tx_rx_F")
    n1 = float(params.get("N1_main", 0)) + float(params.get("N1_side", 0))
    n2 = float(params.get("N2_main", 0)) + float(params.get("N2_side", 0))
    if n1 <= 0.0 or n2 <= 0.0:
        raise RuntimeError("positive primary and secondary turns are required")
    tx_self_h = leakage_h / (1.0 - coupling * coupling)
    tx_magnetizing_h = tx_self_h * coupling * coupling
    rx_magnetizing_h = tx_magnetizing_h * (n2 / n1) ** 2
    factor = HARD_SPEC["magnetizing_inductance_factor"]
    values = {
        "f_res_tx_screen_Hz": _lc_hz(factor * tx_magnetizing_h, c_tx),
        "f_res_rx_screen_Hz": _lc_hz(factor * rx_magnetizing_h, c_rx),
        "f_res_interwinding_screen_Hz": _lc_hz(leakage_h, c_cross),
    }
    values["f_res_min_screen_Hz"] = min(values.values())
    return values


def _duplicates(values: Iterable[Any]) -> list[Any]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def authenticate_v6(
    runtime: Path,
    expected_verified: int = 63,
    excluded_candidate_ids: Iterable[str] = (),
) -> tuple[list[dict], dict]:
    """Authenticate verified results and return non-strict sidecar rows."""

    runtime = runtime.resolve()
    plan_path = runtime / "plan.json"
    state_path = runtime / "state.json"
    plan = _read_json(plan_path)
    state = _read_json(state_path)
    plan_sha = _sha256(plan_path)
    if state.get("plan_sha256") != plan_sha:
        raise RuntimeError("state/plan SHA mismatch")
    if state.get("production_eligible") is not False:
        raise RuntimeError("Tier-1 state unexpectedly production eligible")
    if plan.get("production_eligible") is not False:
        raise RuntimeError("Tier-1 plan unexpectedly production eligible")
    if plan.get("automatic_model_or_candidate_promotion") is not False:
        raise RuntimeError("Tier-1 plan unexpectedly permits automatic promotion")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("plan candidate inventory is missing")
    by_digest = {item.get("candidate_digest"): item for item in candidates}
    if None in by_digest or len(by_digest) != len(candidates):
        raise RuntimeError("plan candidate digests are missing or duplicated")
    solver_contract = plan.get("solver_contract") or {}
    expected_solver = solver_contract.get("solver_revision")
    expected_library = solver_contract.get("library_revision")
    if not all(
        isinstance(value, str) and len(value) == 40
        for value in (expected_solver, expected_library)
    ):
        raise RuntimeError("plan solver/library revision pins are invalid")

    excluded_candidate_ids = frozenset(str(value) for value in excluded_candidate_ids)
    unknown_exclusions = excluded_candidate_ids.difference(
        str(item.get("candidate_id")) for item in candidates
    )
    if unknown_exclusions:
        raise RuntimeError(f"unknown excluded candidate ids: {sorted(unknown_exclusions)}")
    rows: list[dict] = []
    errors: list[str] = []
    state_candidates = state.get("candidates") or {}
    for digest, status in sorted(state_candidates.items()):
        if status.get("state") != "measurement_verified":
            continue
        if str(status.get("candidate_id")) in excluded_candidate_ids:
            continue
        candidate = by_digest.get(digest)
        if candidate is None:
            errors.append(f"{digest}:not_in_plan")
            continue
        reference = status.get("result_reference") or {}
        result_path = Path(str(reference.get("path") or "")).resolve()
        try:
            if os.path.commonpath([result_path, runtime]) != str(runtime):
                raise RuntimeError("result escaped runtime")
            result_sha = _sha256(result_path)
            result = _read_json(result_path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{digest}:result_read:{exc}")
            continue
        checks = {
            "result_sha": result_sha == reference.get("sha256"),
            "result_size": result_path.stat().st_size == reference.get("size"),
            "candidate_digest": result.get("candidate_digest") == digest,
            "candidate_id_result": result.get("candidate_id") == status.get("candidate_id"),
            "candidate_id_plan": candidate.get("candidate_id") == status.get("candidate_id"),
            "task_id": result.get("task_id") == status.get("task_id"),
            "task_name": candidate.get("task_name") == status.get("task_name"),
            "dedupe": candidate.get("scheduler_dedupe_key") == status.get("scheduler_dedupe_key"),
            "plan_sha": result.get("source_plan_sha256") == plan_sha,
            "measurement": result.get("measurement_verified") is True,
            "production": result.get("production_eligible") is False,
            "approval": result.get("fea_or_production_approval_granted") is False,
            "solver": result.get("solver_revision") == expected_solver,
            "library": result.get("library_revision") == expected_library,
        }
        if not all(checks.values()):
            errors.append(
                f"{digest}:" + ",".join(key for key, passed in checks.items() if not passed)
            )
            continue
        measurements = {target: _finite_positive(result.get(target), target) for target in TARGETS if target != "k"}
        measurements["k"] = float(result.get("k"))
        if not 0.0 < measurements["k"] < 1.0:
            errors.append(f"{digest}:invalid_k")
            continue
        params = candidate.get("effective_params")
        if not isinstance(params, dict) or not params:
            errors.append(f"{digest}:missing_params")
            continue
        screen = derive_resonance_screen(measurements, params)
        reported_resonance = {
            name: _finite_positive(result.get(name), name)
            for name in (
                "f_res_tx_self_Hz", "f_res_rx_self_Hz",
                "f_res_interwinding_Hz",
            )
        }
        row = {
            "schema_version": SIDECAR_SCHEMA,
            "row_kind": "target_specific_partial_measurement",
            "source_plan": {
                "path": str(plan_path), "sha256": plan_sha,
                "plan_id": plan.get("plan_id"),
            },
            "source_result": {
                "path": str(result_path), "sha256": result_sha,
                "size": result_path.stat().st_size,
            },
            "source_task": {
                "task_id": int(status["task_id"]),
                "task_name": status["task_name"],
                "scheduler_dedupe_key": status["scheduler_dedupe_key"],
                "terminal_status": status.get("task_status"),
            },
            "solver_revision": expected_solver,
            "library_revision": expected_library,
            "candidate_id": status["candidate_id"],
            "candidate_digest": digest,
            "candidate_params_sha256": _json_sha(params),
            "effective_params": params,
            "measurements": measurements,
            "reported_solver_resonance_Hz": reported_resonance,
            "derived_half_magnetizing_resonance_Hz": screen,
            "measurement_verified": True,
            "strict_full_eligible": False,
            "loss_thermal_eligible": False,
            "eligible_targets": list(TARGETS),
            "ineligible_target_families": ["loss", "flux_density", "thermal"],
            "resonance_is_derived_not_independent_target": True,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
        row["row_sha256"] = _json_sha(row)
        rows.append(row)

    if errors:
        raise RuntimeError("Tier-1 authentication failed: " + "; ".join(errors))
    if len(rows) != int(expected_verified):
        raise RuntimeError(
            f"expected exactly {expected_verified} verified results, got {len(rows)}"
        )
    identities = {
        "candidate_digest": [row["candidate_digest"] for row in rows],
        "candidate_id": [row["candidate_id"] for row in rows],
        "task_id": [row["source_task"]["task_id"] for row in rows],
        "scheduler_dedupe_key": [
            row["source_task"]["scheduler_dedupe_key"] for row in rows
        ],
        "result_sha256": [row["source_result"]["sha256"] for row in rows],
        "row_sha256": [row["row_sha256"] for row in rows],
    }
    duplicate_report = {name: _duplicates(values) for name, values in identities.items()}
    if any(duplicate_report.values()):
        raise RuntimeError(f"verified Tier-1 identity duplicates: {duplicate_report}")
    non_verified = [
        {
            "candidate_id": value.get("candidate_id"),
            "state": value.get("state"),
            "task_id": value.get("task_id"),
            "task_status": value.get("task_status"),
        }
        for value in state_candidates.values()
        if value.get("state") != "measurement_verified"
    ]
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "created_at": _now(),
        "runtime": str(runtime),
        "plan_sha256": plan_sha,
        "plan_id": plan.get("plan_id"),
        "state_sha256_observed": _sha256(state_path),
        "planned_count": len(candidates),
        "verified_count": len(rows),
        "excluded_non_verified": non_verified,
        "excluded_verified_after_frozen_cohort_cutoff": sorted(
            excluded_candidate_ids
        ),
        "duplicate_counts": {
            name: len(values) for name, values in duplicate_report.items()
        },
        "solver_revision": expected_solver,
        "library_revision": expected_library,
        "strict_full_append_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "measurement_ranges": {
            target: {
                "minimum": min(row["measurements"][target] for row in rows),
                "maximum": max(row["measurements"][target] for row in rows),
            }
            for target in TARGETS
        },
        "minimum_screen_resonance_Hz": {
            "minimum": min(
                row["derived_half_magnetizing_resonance_Hz"]["f_res_min_screen_Hz"]
                for row in rows
            ),
            "maximum": max(
                row["derived_half_magnetizing_resonance_Hz"]["f_res_min_screen_Hz"]
                for row in rows
            ),
        },
    }
    return rows, audit


def write_audit(
    runtime: Path,
    output: Path,
    expected_verified: int = 63,
    excluded_candidate_ids: Iterable[str] = (),
    delta_from_audit: Path | None = None,
) -> dict:
    rows, audit = authenticate_v6(
        runtime,
        expected_verified=expected_verified,
        excluded_candidate_ids=excluded_candidate_ids,
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = output / "tier1_target_measurements.jsonl"
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _atomic_bytes(sidecar_path, payload)
    audit["sidecar"] = {
        "path": str(sidecar_path),
        "sha256": _sha256(sidecar_path),
        "size": sidecar_path.stat().st_size,
        "row_count": len(rows),
        "row_sha256": [row["row_sha256"] for row in rows],
    }
    if delta_from_audit is not None:
        parent_audit_path = delta_from_audit.resolve()
        parent_audit = _read_json(parent_audit_path)
        parent_ref = parent_audit.get("sidecar") or {}
        parent_rows = _load_sidecar(
            Path(str(parent_ref.get("path") or "")).resolve(),
            expected_sha=parent_ref.get("sha256"),
            expected_count=int(parent_ref.get("row_count") or 0),
        )
        parent_digests = {row["candidate_digest"] for row in parent_rows}
        current_digests = {row["candidate_digest"] for row in rows}
        if not parent_digests.issubset(current_digests):
            raise RuntimeError("delta parent sidecar is not a subset of combined cohort")
        delta_rows = [row for row in rows if row["candidate_digest"] not in parent_digests]
        if not delta_rows:
            raise RuntimeError("combined cohort has no delta rows")
        delta_path = output / "tier1_target_measurements_delta.jsonl"
        _atomic_bytes(
            delta_path,
            b"".join(_canonical_bytes(row) + b"\n" for row in delta_rows),
        )
        audit["frozen_parent_cohort"] = {
            "audit_path": str(parent_audit_path),
            "audit_sha256": _sha256(parent_audit_path),
            "sidecar_sha256": parent_ref.get("sha256"),
            "row_count": len(parent_rows),
        }
        audit["delta_sidecar"] = {
            "path": str(delta_path),
            "sha256": _sha256(delta_path),
            "size": delta_path.stat().st_size,
            "row_count": len(delta_rows),
            "candidate_ids": [row["candidate_id"] for row in delta_rows],
            "candidate_digests": [row["candidate_digest"] for row in delta_rows],
            "source_result_sha256": [
                row["source_result"]["sha256"] for row in delta_rows
            ],
        }
        audit["combined_cohort"] = {
            "sidecar_sha256": audit["sidecar"]["sha256"],
            "row_count": len(rows),
            "composition": {
                "frozen_parent_rows": len(parent_rows),
                "delta_rows": len(delta_rows),
            },
        }
    audit_path = output / "audit.json"
    _atomic_json(audit_path, audit)
    return audit


def _load_sidecar(
    path: Path,
    expected_sha: str | None = None,
    expected_count: int | None = None,
) -> list[dict]:
    if expected_sha and _sha256(path) != expected_sha:
        raise RuntimeError("sidecar SHA mismatch")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            reported = row.pop("row_sha256", None)
            actual = _json_sha(row)
            row["row_sha256"] = reported
            if reported != actual:
                raise RuntimeError(f"sidecar row SHA mismatch at line {line_number}")
            if (
                row.get("measurement_verified") is not True
                or row.get("strict_full_eligible") is not False
                or row.get("loss_thermal_eligible") is not False
                or row.get("production_eligible") is not False
                or tuple(row.get("eligible_targets") or ()) != TARGETS
            ):
                raise RuntimeError(f"sidecar eligibility contract failed at line {line_number}")
            rows.append(row)
    if expected_count is not None and len(rows) != int(expected_count):
        raise RuntimeError(
            f"expected {expected_count} authenticated rows, got {len(rows)}"
        )
    return rows


def compose_audits(*, audit_paths: Iterable[Path], output: Path) -> dict:
    """Compose separately authenticated Tier-1 cohorts.

    A newly measured canary normally has its own immutable plan/runtime, so it
    cannot truthfully be audited as a superset of a frozen historical cohort.
    This composition step authenticates each input audit and sidecar, then
    enforces identity uniqueness across the complete union before emitting a
    new SHA-addressed sidecar.  It never changes either source cohort.
    """

    resolved_paths = [Path(path).resolve() for path in audit_paths]
    if len(resolved_paths) < 2:
        raise RuntimeError("composition requires at least two source audits")
    if len(resolved_paths) != len(set(resolved_paths)):
        raise RuntimeError("composition source audit paths must be unique")

    sources: list[dict] = []
    rows: list[dict] = []
    expected_solver: str | None = None
    expected_library: str | None = None
    for audit_path in resolved_paths:
        audit_sha = _sha256(audit_path)
        audit = _read_json(audit_path)
        if audit.get("schema_version") not in {AUDIT_SCHEMA, COMPOSITION_SCHEMA}:
            raise RuntimeError(f"unsupported source audit schema: {audit_path}")
        if (
            audit.get("strict_full_append_performed") is not False
            or audit.get("production_eligible") is not False
            or audit.get("automatic_promotion_allowed") is not False
        ):
            raise RuntimeError(f"source audit eligibility contract failed: {audit_path}")
        solver = audit.get("solver_revision")
        library = audit.get("library_revision")
        if expected_solver is None:
            expected_solver, expected_library = solver, library
        elif (solver, library) != (expected_solver, expected_library):
            raise RuntimeError("source audit solver/library revisions differ")

        sidecar_ref = audit.get("sidecar") or {}
        sidecar_path = Path(str(sidecar_ref.get("path") or "")).resolve()
        source_rows = _load_sidecar(
            sidecar_path,
            expected_sha=sidecar_ref.get("sha256"),
            expected_count=int(sidecar_ref.get("row_count") or 0),
        )
        if not source_rows:
            raise RuntimeError(f"source audit has no authenticated rows: {audit_path}")
        for row in source_rows:
            if (
                row.get("solver_revision") != expected_solver
                or row.get("library_revision") != expected_library
            ):
                raise RuntimeError("sidecar row solver/library revision mismatch")
        sources.append({
            "audit_path": str(audit_path),
            "audit_sha256": audit_sha,
            "audit_schema_version": audit.get("schema_version"),
            "source_plan_sha256": audit.get("plan_sha256"),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sidecar_ref.get("sha256"),
            "row_count": len(source_rows),
        })
        rows.extend(source_rows)

    identities = {
        "candidate_digest": [row["candidate_digest"] for row in rows],
        "candidate_id": [row["candidate_id"] for row in rows],
        "task_id": [row["source_task"]["task_id"] for row in rows],
        "scheduler_dedupe_key": [
            row["source_task"]["scheduler_dedupe_key"] for row in rows
        ],
        "result_sha256": [row["source_result"]["sha256"] for row in rows],
        "row_sha256": [row["row_sha256"] for row in rows],
    }
    duplicate_report = {name: _duplicates(values) for name, values in identities.items()}
    if any(duplicate_report.values()):
        raise RuntimeError(f"composed Tier-1 identity duplicates: {duplicate_report}")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = output / "tier1_target_measurements.jsonl"
    _atomic_bytes(
        sidecar_path,
        b"".join(_canonical_bytes(row) + b"\n" for row in rows),
    )
    composition_identity = {
        "schema_version": COMPOSITION_SCHEMA,
        "ordered_source_audit_sha256": [source["audit_sha256"] for source in sources],
        "ordered_source_sidecar_sha256": [
            source["sidecar_sha256"] for source in sources
        ],
        "combined_sidecar_sha256": _sha256(sidecar_path),
        "row_count": len(rows),
    }
    audit = {
        "schema_version": COMPOSITION_SCHEMA,
        "created_at": _now(),
        "plan_sha256": _json_sha(composition_identity),
        "composition_identity": composition_identity,
        "source_audits": sources,
        "solver_revision": expected_solver,
        "library_revision": expected_library,
        "verified_count": len(rows),
        "duplicate_counts": {
            name: len(values) for name, values in duplicate_report.items()
        },
        "sidecar": {
            "path": str(sidecar_path),
            "sha256": composition_identity["combined_sidecar_sha256"],
            "size": sidecar_path.stat().st_size,
            "row_count": len(rows),
            "row_sha256": [row["row_sha256"] for row in rows],
        },
        "composition": {
            "source_count": len(sources),
            "source_row_counts": [source["row_count"] for source in sources],
            "combined_row_count": len(rows),
        },
        "strict_full_append_performed": False,
        "canonical_dataset_write_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    _atomic_json(output / "audit.json", audit)
    return audit


def _repo_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    regression = repo / "regression_260707"
    for path in (str(repo), str(regression), str(regression / "training")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo, regression


def _load_strict_cohort(
    dataset: Path, solver_revision: str, library_revision: str,
) -> tuple[pd.DataFrame, str, int]:
    """Read one stable snapshot and recompute the exact revision cohort."""

    _repo_paths()
    from regression_260707.quality_contract import annotate_validity
    from regression_260707.training.checkpoint_train import to_physical

    before = _sha256(dataset)
    frame = pd.read_parquet(dataset)
    after = _sha256(dataset)
    if before != after:
        raise RuntimeError("canonical dataset changed while it was read")
    audited = annotate_validity(
        frame,
        expected_solver_revision=solver_revision,
        expected_library_revision=library_revision,
    )
    strict = to_physical(audited.loc[
        audited["_strict_valid_full"].fillna(False).astype(bool)
    ].copy())
    if strict.empty:
        raise RuntimeError("exact revision strict-full cohort is empty")
    revisions = sorted(set(
        strict.get("physics_data_revision", pd.Series(dtype=str))
        .fillna("").astype(str).str.strip()
    ))
    if len(revisions) != 1 or not revisions[0]:
        raise RuntimeError(f"strict-full physics revision cohort is not singular: {revisions}")
    return strict, before, len(frame)


def _derive_partial_features(rows: list[dict]) -> pd.DataFrame:
    _repo_paths()
    from module.input_parameter_260706 import KEYS, create_input_parameter, validation_check

    derived = []
    for row in rows:
        params = row["effective_params"]
        base = create_input_parameter({key: params[key] for key in KEYS if key in params})
        valid, completed = validation_check(base, strict=False)
        if not valid or len(completed) != 1:
            raise RuntimeError(f"candidate geometry no longer validates: {row['candidate_id']}")
        record = completed.iloc[0].to_dict()
        record.update({key: value for key, value in params.items() if key not in record})
        record.update(row["measurements"])
        record["candidate_id"] = row["candidate_id"]
        record["candidate_digest"] = row["candidate_digest"]
        record["row_sha256"] = row["row_sha256"]
        derived.append(record)
    return pd.DataFrame(derived)


def _transform(values: np.ndarray, kind: str | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if kind == "log":
        return np.log(np.clip(values, np.finfo(float).tiny, None))
    return values


def _inverse(values: np.ndarray, kind: str | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if kind == "log":
        return np.exp(values)
    return values


def _fit_lgbm(X: pd.DataFrame, y: np.ndarray, weights: np.ndarray, seed: int):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y, sample_weight=weights)
    return model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    denominator = np.maximum(np.abs(actual), np.finfo(float).tiny)
    ape = np.abs(error) / denominator
    scale = max(
        float(np.quantile(actual, 0.9) - np.quantile(actual, 0.1)),
        float(np.median(np.abs(actual))), np.finfo(float).tiny,
    )
    variance = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "count": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "normalized_rmse_pct": float(np.sqrt(np.mean(error ** 2)) / scale * 100.0),
        "mae": float(np.mean(np.abs(error))),
        "mape_pct": float(np.mean(ape) * 100.0),
        "p90_ape_pct": float(np.quantile(ape, 0.9) * 100.0),
        "r2": float(1.0 - np.sum(error ** 2) / (variance or 1e-30)),
    }


def _primary_metric(target: str, metrics: dict) -> float:
    return float(metrics["rmse"] if target == "k" else metrics["mape_pct"])


def _q90_half_width(errors: np.ndarray) -> tuple[float, float]:
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or len(errors) < 30 or not np.isfinite(errors).all():
        raise RuntimeError("conformal calibration needs at least 30 finite errors")
    if (errors < 0.0).any():
        raise RuntimeError("conformal calibration errors must be non-negative")
    half_width = float(np.quantile(errors, 0.9, method="higher"))
    coverage = float(np.mean(errors <= half_width))
    if coverage < 0.9:
        raise RuntimeError("conformal q90 empirical coverage is below 90%")
    return half_width, coverage


def _folds(partial: pd.DataFrame, count: int = 5) -> np.ndarray:
    order = np.argsort(partial["candidate_digest"].astype(str).to_numpy())
    fold = np.empty(len(partial), dtype=int)
    fold[order] = np.arange(len(partial)) % int(count)
    return fold


class TargetPredictor:
    """Minimal conformal predictor consumed by the corrected NSGA problem."""

    predict_mu_sigma_contract = CONFORMAL_HALF_WIDTH_CONTRACT

    def __init__(self, bundle: dict):
        if bundle.get("schema_version") != MODEL_SCHEMA:
            raise RuntimeError("unsupported target model bundle")
        self.bundle = bundle
        self.features = list(bundle["features"])
        self.kind = bundle["transform"]
        self.models = list(bundle["models"])
        self.absolute_half_width = float(bundle["absolute_q90_half_width"])

    def configure_inference_threads(self, threads: int = 1) -> dict:
        threads = int(threads)
        if threads < 1 or threads > 8:
            raise ValueError("inference threads must be between 1 and 8")
        for _family, model in self.models:
            model.n_jobs = threads
        return {
            "threads": threads,
            "model_count": len(self.models),
            "families": sorted({family for family, _ in self.models}),
        }

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.features).fillna(0.0)
        return np.stack([model.predict(X) for _family, model in self.models])

    def predict_mu_sigma(self, frame: pd.DataFrame, conformal: bool = True):
        matrix = self._matrix(frame)
        mu_t = np.median(matrix, axis=0)
        mu = _inverse(mu_t, self.kind)
        raw = _inverse(matrix, self.kind)
        spread = np.std(raw, axis=0)
        if conformal:
            spread = np.maximum(spread, self.absolute_half_width)
        return mu, np.maximum(spread, np.abs(mu) * np.finfo(float).eps)

    def disagreement(self, frame: pd.DataFrame):
        raw = _inverse(self._matrix(frame), self.kind)
        return raw.max(axis=0) - raw.min(axis=0)


def _authenticate_base_generation(path: Path) -> dict:
    path = path.resolve()
    report_path = path / "train_report.json"
    report = _read_json(report_path)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("base generation artifact manifest is missing")
    for relative, expected in artifacts.items():
        artifact = (path / relative).resolve()
        if os.path.commonpath([artifact, path]) != str(path):
            raise RuntimeError("base generation artifact escaped generation")
        if _sha256(artifact) != expected:
            raise RuntimeError(f"base generation artifact SHA mismatch: {relative}")
    required_strict_targets = {
        "P_winding_total", "P_Tx_main_group", "P_Rx_main_group",
        "P_Rx_side_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
        "B_mean_core", "B_max_core", "T_max_Tx", "T_max_Rx_main",
        "T_max_Rx_side", "T_max_core", "Tprobe_Tx_leeward_max",
        "Tprobe_Rx_main_leeward_max", "Tprobe_Rx_side_leeward_max",
        "Tprobe_core_center_max", "Tprobe_core_center_leg_max",
        "Tprobe_core_side_leg_max", "Tprobe_core_top_yoke_max",
    }
    missing = sorted(required_strict_targets.difference(report.get("targets") or []))
    if missing:
        raise RuntimeError(f"base strict-full generation is incomplete: {missing}")
    if int(report.get("strict_full_rows") or 0) < 1:
        raise RuntimeError("base generation has no strict-full rows")
    return {
        "path": str(path),
        "train_report_sha256": _sha256(report_path),
        "training_run_id": report.get("training_run_id"),
        "dataset_sha256": report.get("dataset_sha256"),
        "strict_full_rows": int(report["strict_full_rows"]),
        "artifact_count": len(artifacts),
        "targets": report.get("targets"),
        "loss_thermal_source_policy": "authenticated_strict_full_generation_only",
    }


def _resonance_predictions(
    target_predictions: dict[str, np.ndarray], partial_rows: list[dict],
) -> np.ndarray:
    output = []
    for index, row in enumerate(partial_rows):
        values = {target: target_predictions[target][index] for target in TARGETS}
        output.append(derive_resonance_screen(values, row["effective_params"])[
            "f_res_min_screen_Hz"
        ])
    return np.asarray(output, dtype=float)


def train_feedback(
    *, audit_dir: Path, dataset: Path, base_generation: Path, output: Path,
) -> dict:
    audit = _read_json(audit_dir / "audit.json")
    sidecar_ref = audit.get("sidecar") or {}
    sidecar = Path(str(sidecar_ref.get("path") or "")).resolve()
    rows = _load_sidecar(
        sidecar,
        expected_sha=sidecar_ref.get("sha256"),
        expected_count=int(sidecar_ref.get("row_count") or 0),
    )
    solver = audit["solver_revision"]
    library = audit["library_revision"]
    strict, dataset_sha, raw_rows = _load_strict_cohort(dataset, solver, library)
    partial = _derive_partial_features(rows)

    _repo_paths()
    from regression_260707.training.checkpoint_train import feature_columns

    candidate_features = feature_columns(strict)
    features = [
        name for name in candidate_features
        if name in partial.columns
        and pd.to_numeric(partial[name], errors="coerce").map(np.isfinite).all()
        and pd.to_numeric(strict[name], errors="coerce").map(np.isfinite).all()
    ]
    if len(features) < 20:
        raise RuntimeError(f"too few common design-time features: {len(features)}")
    strict_X = strict[features].astype(float).reset_index(drop=True)
    partial_X = partial[features].astype(float).reset_index(drop=True)
    fold_ids = _folds(partial)
    base_generation_ref = _authenticate_base_generation(base_generation)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    strict_snapshot = strict.loc[:, [*features, *TARGETS]].copy()
    strict_snapshot_path = output / "strict_target_snapshot.parquet"
    strict_snapshot.to_parquet(strict_snapshot_path, index=False)

    report_targets: dict[str, dict] = {}
    before_predictions: dict[str, np.ndarray] = {}
    after_predictions: dict[str, np.ndarray] = {}
    artifact_inventory: dict[str, str] = {}
    for target_index, target in enumerate(TARGETS):
        y_strict = pd.to_numeric(strict[target], errors="coerce").to_numpy(dtype=float)
        y_partial = pd.to_numeric(partial[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y_strict).all() or not np.isfinite(y_partial).all():
            raise RuntimeError(f"non-finite target values: {target}")
        if target != "k" and (y_strict <= 0).any():
            raise RuntimeError(f"non-positive strict target values: {target}")
        kind = TRANSFORMS[target]
        transformed_strict = _transform(y_strict, kind)
        transformed_partial = _transform(y_partial, kind)

        baseline_model = _fit_lgbm(
            strict_X, transformed_strict, np.ones(len(strict_X)),
            seed=7100 + target_index,
        )
        baseline_pred = _inverse(baseline_model.predict(partial_X), kind)
        baseline_metrics = _metrics(y_partial, baseline_pred)
        before_predictions[target] = baseline_pred

        trials = []
        trial_predictions: dict[float, np.ndarray] = {}
        for weight in WEIGHT_CANDIDATES:
            oof = np.full(len(partial), np.nan)
            for fold in range(5):
                fit_partial = fold_ids != fold
                held = fold_ids == fold
                X_fit = pd.concat(
                    [strict_X, partial_X.loc[fit_partial]], ignore_index=True
                )
                y_fit = np.concatenate([
                    transformed_strict, transformed_partial[fit_partial]
                ])
                weights = np.concatenate([
                    np.ones(len(strict_X)),
                    np.full(int(fit_partial.sum()), float(weight)),
                ])
                model = _fit_lgbm(
                    X_fit, y_fit, weights,
                    seed=7200 + 100 * target_index + fold,
                )
                oof[held] = _inverse(model.predict(partial_X.loc[held]), kind)
            metrics = _metrics(y_partial, oof)
            trials.append({
                "partial_sample_weight": weight,
                "held_out_metrics": metrics,
                "primary_metric": _primary_metric(target, metrics),
            })
            trial_predictions[weight] = oof
        selected = min(trials, key=lambda item: (item["primary_metric"], item["partial_sample_weight"]))
        selected_weight = float(selected["partial_sample_weight"])
        selected_pred = trial_predictions[selected_weight]
        after_predictions[target] = selected_pred
        improvement = (
            _primary_metric(target, baseline_metrics)
            - _primary_metric(target, selected["held_out_metrics"])
        )
        target_gate = math.isfinite(improvement) and improvement >= 0.0

        # Keep 10% of strict-full rows only for conformal calibration.  Every
        # Tier-1 measurement participates in the final experimental fit after
        # having been evaluated out-of-fold above.
        calibration = np.arange(len(strict_X)) % 10 == (target_index % 10)
        fit_strict = ~calibration
        X_fit = pd.concat([strict_X.loc[fit_strict], partial_X], ignore_index=True)
        y_fit = np.concatenate([
            transformed_strict[fit_strict], transformed_partial,
        ])
        weights = np.concatenate([
            np.ones(int(fit_strict.sum())),
            np.full(len(partial_X), selected_weight),
        ])
        models = []
        for repeat in range(4):
            model = _fit_lgbm(
                X_fit, y_fit, weights,
                seed=7600 + 100 * target_index + repeat,
            )
            models.append(("lightgbm", model))
        calibration_matrix = np.stack([
            _inverse(model.predict(strict_X.loc[calibration]), kind)
            for _family, model in models
        ])
        calibration_mu = np.median(calibration_matrix, axis=0)
        calibration_errors = np.abs(y_strict[calibration] - calibration_mu)
        partial_oof_errors = np.abs(y_partial - selected_pred)
        # The search model is explicitly target-specific: its admissible
        # measurement cohort is the authenticated Tier-1 matrix/capacitance
        # lane.  Calibrate the operational half-width on predictions that held
        # every Tier-1 point out exactly once.  Broad strict-full calibration
        # remains an immutable diagnostic and is never silently discarded or
        # presented as target-lane uncertainty.
        absolute_half_width, partial_coverage = _q90_half_width(
            partial_oof_errors
        )
        strict_half_width, strict_coverage = _q90_half_width(
            calibration_errors
        )
        mixed_half_width, mixed_coverage = _q90_half_width(np.concatenate([
            calibration_errors, partial_oof_errors,
        ]))
        target_gate = target_gate and partial_coverage >= 0.9
        bundle = {
            "schema_version": MODEL_SCHEMA,
            "target": target,
            "features": features,
            "transform": kind,
            "models": models,
            "absolute_q90_half_width": absolute_half_width,
            "uncertainty_contract": CONFORMAL_HALF_WIDTH_CONTRACT,
            "conformal_calibration_policy": TARGET_CONFORMAL_CALIBRATION_POLICY,
            "target_partial_oof_calibration_count": len(partial_oof_errors),
            "target_partial_oof_empirical_coverage": partial_coverage,
            "strict_full_diagnostic_q90_half_width": strict_half_width,
            "strict_full_diagnostic_empirical_coverage": strict_coverage,
            "mixed_legacy_diagnostic_q90_half_width": mixed_half_width,
            "mixed_legacy_diagnostic_empirical_coverage": mixed_coverage,
            "source_dataset_sha256": dataset_sha,
            "source_plan_sha256": audit["plan_sha256"],
            "source_sidecar_sha256": sidecar_ref["sha256"],
            "solver_revision": solver,
            "library_revision": library,
            "strict_full_rows": len(strict),
            "partial_measurement_rows": len(partial),
            "partial_sample_weight": selected_weight,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
        target_dir = output / "models" / target
        model_path = target_dir / "models.pkl"
        meta_path = target_dir / "meta.json"
        _atomic_pickle(model_path, bundle)
        metadata = {
            key: value for key, value in bundle.items() if key not in {"models"}
        }
        metadata.update({
            "baseline_target_holdout_metrics": baseline_metrics,
            "augmented_target_oof_metrics": selected["held_out_metrics"],
            "weight_trials": trials,
            "target_gate_passed": target_gate,
            "primary_metric_improvement": improvement,
            "model_sha256": _sha256(model_path),
        })
        _atomic_json(meta_path, metadata)
        artifact_inventory[str(model_path.relative_to(output)).replace("\\", "/")] = _sha256(model_path)
        artifact_inventory[str(meta_path.relative_to(output)).replace("\\", "/")] = _sha256(meta_path)
        report_targets[target] = metadata

    actual_targets = {
        target: partial[target].to_numpy(dtype=float) for target in TARGETS
    }
    actual_resonance = _resonance_predictions(actual_targets, rows)
    before_resonance = _resonance_predictions(before_predictions, rows)
    after_resonance = _resonance_predictions(after_predictions, rows)
    resonance_metrics = {
        "baseline_target_holdout_metrics": _metrics(actual_resonance, before_resonance),
        "augmented_target_oof_metrics": _metrics(actual_resonance, after_resonance),
    }
    resonance_gate = (
        resonance_metrics["augmented_target_oof_metrics"]["mape_pct"]
        <= resonance_metrics["baseline_target_holdout_metrics"]["mape_pct"]
    )
    target_gates = {
        target: bool(report_targets[target]["target_gate_passed"])
        for target in TARGETS
    }
    experimental_gate = all(target_gates.values()) and resonance_gate
    report = {
        "schema_version": MODEL_SCHEMA,
        "created_at": _now(),
        "source_dataset": {
            "path": str(dataset.resolve()), "sha256": dataset_sha,
            "raw_rows": raw_rows, "exact_revision_strict_full_rows": len(strict),
        },
        "source_plan_sha256": audit["plan_sha256"],
        "source_sidecar_sha256": sidecar_ref["sha256"],
        "solver_revision": solver,
        "library_revision": library,
        "features": features,
        "target_measurement_rows": len(partial),
        "target_reports": report_targets,
        "derived_resonance_held_out": resonance_metrics,
        "experimental_target_gate": {
            "passed": experimental_gate,
            "target_gates": target_gates,
            "derived_resonance_gate": resonance_gate,
            "all_target_points_held_out_once": True,
            "fold_count": 5,
            "conformal_calibration_policy": (
                TARGET_CONFORMAL_CALIBRATION_POLICY
            ),
            "target_partial_oof_empirical_coverage_minimum": min(
                report_targets[target][
                    "target_partial_oof_empirical_coverage"
                ]
                for target in TARGETS
            ),
        },
        "base_strict_full_generation": base_generation_ref,
        "loss_thermal_model_policy": (
            "unchanged authenticated base generation; Tier-1 rows never eligible"
        ),
        "canonical_dataset_write_performed": False,
        "production_registry_write_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "artifacts": artifact_inventory,
    }
    report_path = output / "training_report.json"
    _atomic_json(report_path, report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": _now(),
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "source_plan_sha256": audit["plan_sha256"],
        "source_sidecar_sha256": sidecar_ref["sha256"],
        "files": {
            "training_report.json": _sha256(report_path),
            "strict_target_snapshot.parquet": _sha256(strict_snapshot_path),
            **artifact_inventory,
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return report


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    # Immutable Slurm bundles intentionally omit ``.git``.  Their builder
    # writes this marker only after hashing every staged code byte, and the
    # remote runner authenticates that inventory before importing this module.
    marker = path / ".source-revision"
    if marker.is_file():
        revision = marker.read_text(encoding="ascii").strip()
        if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
            return revision
    raise RuntimeError(f"cannot authenticate NSGA source revision below {path}")


def _load_search_models(model_dir: Path, nsga_code_root: Path):
    regression = nsga_code_root / "regression_260707"
    for path in (
        str(nsga_code_root), str(regression), str(regression / "training"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    from optimization.run_nsga2 import REQUIRED_MODEL_TARGETS
    from predictor import EnsemblePredictor

    report = _read_json(model_dir / "training_report.json")
    manifest = _read_json(model_dir / "manifest.json")
    for relative, expected in (manifest.get("files") or {}).items():
        if _sha256(model_dir / relative) != expected:
            raise RuntimeError(f"feedback artifact SHA mismatch: {relative}")
    if report.get("production_eligible") is not False:
        raise RuntimeError("feedback model unexpectedly production eligible")
    if not report.get("experimental_target_gate", {}).get("passed"):
        raise RuntimeError("experimental target gate did not pass")
    base = Path(report["base_strict_full_generation"]["path"])
    base_report = _read_json(base / "train_report.json")
    models = {}
    target_set = set(TARGETS)
    for target in REQUIRED_MODEL_TARGETS:
        if target in target_set:
            with open(model_dir / "models" / target / "models.pkl", "rb") as handle:
                models[target] = TargetPredictor(pickle.load(handle))
        else:
            relative = f"{target}/models.pkl"
            expected = (base_report.get("artifacts") or {}).get(relative)
            if not expected or _sha256(base / relative) != expected:
                raise RuntimeError(f"strict-full base model failed authentication: {target}")
            with open(base / relative, "rb") as handle:
                models[target] = EnsemblePredictor(pickle.load(handle))
    return models, report, manifest


def _deep_topology_contract(fixed_primary_turns: int) -> dict[str, Any]:
    try:
        from tier1_deep_crossover_contract import (  # noqa: PLC0415
            topology_evolution_contract,
        )
    except ImportError:  # pragma: no cover - repository import path
        from tools.tier1_deep_crossover_contract import (  # noqa: PLC0415
            topology_evolution_contract,
        )
    return topology_evolution_contract(fixed_primary_turns)


def _turn_split_main_values(
    coordinates: np.ndarray, *, fixed_primary_turns: int,
    coordinate_index: int,
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    secondary_turns = 10 * int(fixed_primary_turns)
    side_turns = np.rint(
        secondary_turns * np.clip(values[:, coordinate_index], 0.0, 1.0) * 0.8
    ).astype(int)
    return secondary_turns - side_turns


def _turn_split_unit_coordinate(
    n2_main: int, *, fixed_primary_turns: int,
) -> float:
    secondary_turns = 10 * int(fixed_primary_turns)
    n2_side = secondary_turns - int(n2_main)
    value = n2_side / (0.8 * secondary_turns)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError("turn-split migration coordinate escaped unit interval")
    return float(value)


def _seed_turn_split_sub_islands(
    problem: Any, initial: np.ndarray, contract: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(initial, dtype=float).copy()
    topologies = tuple(contract["turn_split_sub_islands_N2_main"])
    copies = int(contract["initial_repaired_copies_per_sub_island"])
    coordinate_index = int(contract["coordinate_index"])
    required = len(topologies) * copies
    if values.ndim != 2 or len(values) < required:
        raise RuntimeError("population is too small for turn-split initialization")
    lanes = contract.get("basin_donor_lanes") or {}
    lane_topologies = {
        str(name): tuple(int(pair[0]) for pair in lane[
            "topologies_N2_main_N2_side"
        ])
        for name, lane in lanes.items()
    }
    if not lane_topologies:
        raise RuntimeError("turn-split contract has no basin donor lanes")
    source_values = values.copy()
    source_observed = _turn_split_main_values(
        source_values,
        fixed_primary_turns=contract["fixed_primary_turns"],
        coordinate_index=coordinate_index,
    )
    donor_sources = []
    for copy_index in range(copies):
        for topology_index, n2_main in enumerate(topologies):
            row = copy_index * len(topologies) + topology_index
            same_lane = {
                topology
                for values_in_lane in lane_topologies.values()
                if n2_main in values_in_lane
                for topology in values_in_lane
            }
            candidate_tiers = (
                ("same_topology", source_observed == n2_main),
                ("same_basin_lane", np.isin(source_observed, sorted(same_lane))),
                ("protected_basin_topology", np.isin(source_observed, topologies)),
                ("any_initial_coordinate", np.ones(len(values), dtype=bool)),
            )
            ordered_candidates = []
            candidate_source_tier = {}
            for tier, mask in candidate_tiers:
                candidates = np.flatnonzero(mask)
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate not in candidate_source_tier:
                        ordered_candidates.append(candidate)
                        candidate_source_tier[candidate] = tier
            if not ordered_candidates:  # pragma: no cover
                raise RuntimeError("turn-split basin has no coordinate donor")
            source_index = ordered_candidates[copy_index % len(ordered_candidates)]
            source_tier = candidate_source_tier[source_index]
            values[row] = source_values[source_index]
            values[row, coordinate_index] = _turn_split_unit_coordinate(
                n2_main,
                fixed_primary_turns=contract["fixed_primary_turns"],
            )
            donor_sources.append({
                "target_N2_main": n2_main,
                "copy_index": copy_index,
                "source_row": source_index,
                "source_N2_main": int(source_observed[source_index]),
                "source_tier": source_tier,
            })
    values = np.asarray(problem.repair_unit_coordinates(values), dtype=float)
    observed = _turn_split_main_values(
        values, fixed_primary_turns=contract["fixed_primary_turns"],
        coordinate_index=coordinate_index,
    )
    counts = {
        str(topology): int(np.count_nonzero(observed == topology))
        for topology in topologies
    }
    if any(counts[str(topology)] < copies for topology in topologies):
        raise RuntimeError("turn-split initialization repair lost a sub-island")
    audit = {
        "schema_version": "mft-tier1-basin-aware-initialization-v2",
        "topology_counts_after_current_repair": counts,
        "minimum_copies_each_verified": True,
        "basin_lane_topologies": {
            name: list(values_in_lane)
            for name, values_in_lane in lane_topologies.items()
        },
        "basin_seeded_counts": {
            name: sum(counts[str(topology)] for topology in values_in_lane)
            for name, values_in_lane in lane_topologies.items()
        },
        "donor_sources": donor_sources,
        "same_or_same_lane_donor_count": sum(
            item["source_tier"] in {"same_topology", "same_basin_lane"}
            for item in donor_sources
        ),
        "warm_coordinates_are_donors_only": True,
        "source_prediction_or_pass_classification_inherited": False,
        "additional_model_evaluations": 0,
    }
    audit["sha256"] = _json_sha(audit)
    return values, audit


def _deep_topology_components(
    problem: Any, contract: dict[str, Any], repair: Any,
) -> tuple[Any, Any, Any]:
    """Create paired mating, migration and epsilon survival operators."""

    from pymoo.core.duplicate import DefaultDuplicateElimination  # noqa: PLC0415
    from pymoo.core.mating import Mating  # noqa: PLC0415
    from pymoo.core.selection import Selection  # noqa: PLC0415
    from pymoo.core.survival import Survival  # noqa: PLC0415
    from pymoo.operators.crossover.sbx import SBX  # noqa: PLC0415
    from pymoo.operators.mutation.pm import PM  # noqa: PLC0415
    from pymoo.operators.survival.rank_and_crowding import (  # noqa: PLC0415
        RankAndCrowding,
    )

    topologies = tuple(int(value) for value in (
        contract["turn_split_sub_islands_N2_main"]
    ))
    parent_pairs = tuple(
        tuple(int(value) for value in pair)
        for pair in contract["turn_split_parent_pair_schedule"]
    )
    fixed_turns = int(contract["fixed_primary_turns"])
    coordinate_index = int(contract["coordinate_index"])
    minimum_each = int(
        contract["survival"][
            "minimum_survivors_per_turn_split_sub_island"
        ]
    )
    initial_epsilon = float(contract["survival"]["initial_epsilon"])
    epsilon_decay_generation = int(
        contract["survival"]["decay_to_zero_generation"]
    )
    migration_period = int(contract["migration"]["period_generations"])
    migrants_per_event = int(contract["migration"]["migrants_per_event"])

    def topology_values(population) -> np.ndarray:
        return _turn_split_main_values(
            population.get("X"), fixed_primary_turns=fixed_turns,
            coordinate_index=coordinate_index,
        )

    def individual_score(individual) -> tuple[float, float, float]:
        rank = individual.get("rank")
        crowding = individual.get("crowding")
        constraints = np.asarray(individual.get("G"), dtype=float)
        positive_g = float(np.maximum(constraints, 0.0).sum())
        rank_value = float(rank) if rank is not None else math.inf
        crowding_value = (
            float(crowding) if crowding is not None else -math.inf
        )
        return (rank_value, positive_g, -crowding_value)

    class TurnSplitPairedSelection(Selection):
        def __init__(self):
            super().__init__()
            self.selection_calls = 0
            self.parent_pairs_emitted = 0

        def _pick(self, pop, candidates, random_state):
            candidates = np.asarray(candidates, dtype=int)
            if len(candidates) == 0:
                candidates = np.arange(len(pop), dtype=int)
            draw = random_state.choice(
                candidates, size=min(2, len(candidates)), replace=False,
            )
            return int(min(draw, key=lambda index: individual_score(pop[index])))

        def _do(
            self, problem, pop, n_select, n_parents, *args,
            random_state=None, **kwargs,
        ):
            if int(n_parents) != 2:
                raise RuntimeError("turn-split paired selection requires two parents")
            observed = topology_values(pop)
            selected = np.empty((int(n_select), 2), dtype=int)
            offset = self.parent_pairs_emitted
            for index in range(int(n_select)):
                left, right = parent_pairs[(offset + index) % len(parent_pairs)]
                selected[index, 0] = self._pick(
                    pop, np.flatnonzero(observed == left), random_state,
                )
                selected[index, 1] = self._pick(
                    pop, np.flatnonzero(observed == right), random_state,
                )
            self.selection_calls += 1
            self.parent_pairs_emitted += int(n_select)
            return selected

    class TurnSplitMigrationMating(Mating):
        def __init__(self, selection):
            super().__init__(
                selection, SBX(eta=15, prob=0.9), PM(eta=20),
                repair=repair,
                eliminate_duplicates=DefaultDuplicateElimination(),
                n_max_iterations=100,
            )
            self.migration_events = 0
            self.migrants_created = 0
            self.last_migration_generation = None

        def _do(
            self, problem, pop, n_offsprings, parents=None,
            random_state=None, **kwargs,
        ):
            offspring = super()._do(
                problem, pop, n_offsprings, parents=parents,
                random_state=random_state, **kwargs,
            )
            algorithm = kwargs.get("algorithm")
            generation = int(getattr(algorithm, "n_gen", 0) or 0)
            migrate = (
                self.last_migration_generation is None
                or generation - self.last_migration_generation >= migration_period
            )
            if not migrate or len(offspring) == 0:
                return offspring
            observed = topology_values(pop)
            migrants = []
            for target_index, target in enumerate(topologies):
                source = topologies[target_index - 1]
                candidates = np.flatnonzero(observed == source)
                if len(candidates) == 0:
                    continue
                elite_index = min(
                    candidates, key=lambda index: individual_score(pop[index]),
                )
                coordinate = np.asarray(pop[int(elite_index)].X, dtype=float).copy()
                coordinate[coordinate_index] = _turn_split_unit_coordinate(
                    target, fixed_primary_turns=fixed_turns,
                )
                migrants.append(coordinate)
            if migrants:
                while len(migrants) < migrants_per_event:
                    migrants.extend(list(migrants))
                migrants = migrants[: min(migrants_per_event, len(offspring))]
                for index, coordinate in enumerate(migrants):
                    offspring[index].set("X", coordinate)
                self.migration_events += 1
                self.migrants_created += len(migrants)
                self.last_migration_generation = generation
            return offspring

    class TurnSplitEpsilonSurvival(Survival):
        def __init__(self):
            super().__init__(filter_infeasible=False)
            self.ranking = RankAndCrowding()
            self.survival_calls = 0
            self.last_epsilon = None
            self.last_topology_counts = {}
            self.minimum_topology_count_observed = math.inf
            self.maximum_single_topology_count_observed = 0

        def _do(
            self, problem, pop, *args, n_survive=None,
            random_state=None, **kwargs,
        ):
            n_survive = min(int(n_survive), len(pop))
            algorithm = kwargs.get("algorithm")
            generation = int(getattr(algorithm, "n_gen", 1) or 1)
            evolution_generation = max(0, generation - 1)
            fraction = max(
                0.0, 1.0 - evolution_generation / epsilon_decay_generation,
            )
            epsilon = initial_epsilon * fraction * fraction
            constraints = np.asarray(pop.get("G"), dtype=float)
            positive_sum = np.maximum(constraints, 0.0).sum(axis=1)
            epsilon_feasible = np.flatnonzero(positive_sum <= epsilon)
            epsilon_infeasible = np.flatnonzero(positive_sum > epsilon)
            global_order = []
            if len(epsilon_feasible):
                ranked = self.ranking._do(
                    problem, pop[epsilon_feasible],
                    n_survive=len(epsilon_feasible),
                    random_state=random_state,
                )
                identity = {id(individual): index for index, individual in enumerate(pop)}
                global_order.extend(identity[id(individual)] for individual in ranked)
            for rank_offset, index in enumerate(
                epsilon_infeasible[np.argsort(
                    positive_sum[epsilon_infeasible], kind="stable",
                )]
            ):
                pop[int(index)].set("rank", len(global_order) + rank_offset)
                pop[int(index)].set("crowding", -float(positive_sum[int(index)]))
                global_order.append(int(index))
            observed = topology_values(pop)
            selected = []
            for topology in topologies:
                selected.extend([
                    index for index in global_order
                    if observed[index] == topology
                    and index not in selected
                ][:minimum_each])
            selected.extend(
                index for index in global_order if index not in selected
            )
            selected = selected[:n_survive]
            survivors = pop[np.asarray(selected, dtype=int)]
            survivor_topologies = topology_values(survivors)
            counts = {
                str(topology): int(np.count_nonzero(
                    survivor_topologies == topology
                ))
                for topology in topologies
            }
            if any(counts[str(topology)] < minimum_each for topology in topologies):
                raise RuntimeError(
                    "epsilon survival lost a required turn-split sub-island"
                )
            self.survival_calls += 1
            self.last_epsilon = float(epsilon)
            self.last_topology_counts = counts
            self.minimum_topology_count_observed = min(
                self.minimum_topology_count_observed, min(counts.values()),
            )
            self.maximum_single_topology_count_observed = max(
                self.maximum_single_topology_count_observed,
                max(counts.values()),
            )
            return survivors

    selection = TurnSplitPairedSelection()
    mating = TurnSplitMigrationMating(selection)
    survival = TurnSplitEpsilonSurvival()
    return selection, mating, survival


def _run_optimizer(
    problem: Any, *, seed: int, population: int,
    warm_start: np.ndarray | None, max_generations: int,
    termination_strategy: str,
    topology_evolution_contract: dict[str, Any] | None = None,
):
    """Run the pinned NSGA implementation with one explicitly sealed stop rule."""

    from optimization import run_nsga2 as pinned

    if termination_strategy == DEFAULT_TERMINATION_STRATEGY:
        return pinned.run_one(
            problem, seed=int(seed), pop=int(population), warm_X=warm_start,
            max_gen=int(max_generations),
        )
    if termination_strategy != FIXED_GENERATION_TERMINATION_STRATEGY:
        raise ValueError(
            f"unsupported optimizer termination strategy: {termination_strategy}"
        )
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    initial, initialization_audit = pinned._initial_population(
        problem, seed=int(seed), pop=int(population), warm_X=warm_start,
    )
    repair = pinned._physics_repair_operator()
    algorithm_kwargs = {}
    if topology_evolution_contract is not None:
        initial, topology_initialization_audit = _seed_turn_split_sub_islands(
            problem, initial, topology_evolution_contract,
        )
        initialization_audit = dict(initialization_audit or {})
        initialization_audit["turn_split_sub_islands"] = (
            topology_initialization_audit
        )
        selection, mating, survival = _deep_topology_components(
            problem, topology_evolution_contract, repair,
        )
        algorithm_kwargs.update({
            "selection": selection, "mating": mating,
            "survival": survival,
        })
    algorithm = NSGA2(
        pop_size=int(population), sampling=initial, repair=repair,
        eliminate_duplicates=True, **algorithm_kwargs,
    )
    result = minimize(
        problem, algorithm, ("n_gen", int(max_generations)),
        seed=int(seed), verbose=False, save_history=False,
    )
    result.initialization_audit = initialization_audit
    if topology_evolution_contract is not None:
        executed = result.algorithm
        result.topology_operator_audit = {
            "paired_selection_calls": int(
                executed.mating.selection.selection_calls
            ),
            "paired_parent_pairs_emitted": int(
                executed.mating.selection.parent_pairs_emitted
            ),
            "migration_events": int(executed.mating.migration_events),
            "migrants_created": int(executed.mating.migrants_created),
            "survival_calls": int(executed.survival.survival_calls),
            "last_optimizer_epsilon": float(
                executed.survival.last_epsilon
            ),
            "minimum_topology_count_observed": int(
                executed.survival.minimum_topology_count_observed
            ),
            "maximum_single_topology_count_observed": int(
                executed.survival.maximum_single_topology_count_observed
            ),
            "last_topology_counts": dict(
                executed.survival.last_topology_counts
            ),
        }
    return result


def run_search_seed(
    *, model_dir: Path, nsga_code_root: Path, output: Path, seed: int,
    population: int, max_generations: int,
    warm_start: Path | None = None,
    warm_start_sha256: str | None = None,
    inference_threads: int = 1,
    optimizer_resonance_scale_hz: float | None = None,
    optimizer_core_thermal_scale_c: float | None = None,
    optimizer_llt_scale_uh: float | None = None,
    optimizer_all_thermal_scale_c: float | None = None,
    optimizer_resonance_allowance_hz: float | None = None,
    optimizer_llt_allowance_uh: float | None = None,
    fixed_primary_turns: int | None = None,
    optimizer_termination_strategy: str = DEFAULT_TERMINATION_STRATEGY,
) -> dict:
    revision = _git_revision(nsga_code_root)
    if revision != EXPECTED_NSGA_REVISION:
        raise RuntimeError(
            f"NSGA code revision must be {EXPECTED_NSGA_REVISION}, got {revision}"
        )
    models, report, manifest = _load_search_models(model_dir, nsga_code_root)
    from optimization.nsga2_problem import (
        MFTProblem, SIDE_TEMPERATURE_TARGETS as CODE_SIDE_TEMPERATURE_TARGETS,
        T_TARGETS as CODE_TEMPERATURE_TARGETS,
    )
    from predictor import DensityGate
    from module.input_parameter_260706 import (
        N1_MAX_TURNS, N1_MIN_TURNS, _SOBOL_DIMS,
    )

    if tuple(CODE_TEMPERATURE_TARGETS) != TEMPERATURE_TARGETS:
        raise RuntimeError("NSGA temperature target order differs from v3 contract")
    if tuple(CODE_SIDE_TEMPERATURE_TARGETS) != SIDE_TEMPERATURE_TARGETS:
        raise RuntimeError("NSGA side-temperature semantics differ from v3 contract")

    if isinstance(inference_threads, bool) or not 1 <= int(inference_threads) <= 8:
        raise ValueError("inference_threads must be an integer from 1 through 8")
    inference_threads = int(inference_threads)
    thread_bindings = {}
    for target, model in models.items():
        configure = getattr(model, "configure_inference_threads", None)
        if not callable(configure):
            raise RuntimeError("model inference thread binding is unavailable")
        try:
            binding = configure(inference_threads)
        except ValueError:
            # Revision 175 predates the eight-CPU Slurm lane and caps the
            # public helper at four.  Keep its fitted bytes and optimizer code
            # exact while applying the same supported-family controls here.
            configured = []
            for family, fitted in model.bundle["models"]:
                family_name = str(family).lower()
                if family_name not in {
                    "lightgbm", "xgboost", "catboost", "extratrees",
                }:
                    raise RuntimeError(
                        f"cannot bind unsupported family: {family_name}"
                    )
                if family_name != "catboost":
                    if not hasattr(fitted, "n_jobs"):
                        raise RuntimeError(
                            f"{family_name} model has no n_jobs control"
                        )
                    fitted.n_jobs = inference_threads
                configured.append(family_name)
            model.inference_threads = inference_threads
            binding = {
                "threads": inference_threads,
                "model_count": len(configured),
                "families": configured,
                "adapter": "tier1_eight_cpu_supported_family_binding_v1",
            }
        if int(binding.get("threads", -1)) != inference_threads:
            raise RuntimeError("model inference thread binding failed")
        thread_bindings[target] = binding
    strict = pd.read_parquet(model_dir / "strict_target_snapshot.parquet")
    features = report["features"]
    density_gate = DensityGate(strict, features)
    problem = MFTProblem(models, spec=HARD_SPEC, density_gate=density_gate)
    sobol_names = [str(item[0]) for item in _SOBOL_DIMS]
    if sobol_names.count("u_N1") != 1:
        raise RuntimeError("authenticated decoder has no unique u_N1 coordinate")
    fixed_turns_contract = install_fixed_primary_turns_repair(
        problem,
        fixed_primary_turns,
        coordinate_index=sobol_names.index("u_N1"),
        minimum_turns=N1_MIN_TURNS,
        maximum_turns=N1_MAX_TURNS,
    )
    physical_evaluate, optimizer_normalization = (
        install_optimizer_constraint_normalization(
            problem, HARD_SPEC,
            resonance_scale_hz=optimizer_resonance_scale_hz,
            core_thermal_scale_c=optimizer_core_thermal_scale_c,
            llt_scale_uh=optimizer_llt_scale_uh,
            all_thermal_scale_c=optimizer_all_thermal_scale_c,
        )
    )
    if ((optimizer_resonance_allowance_hz is None)
            != (optimizer_llt_allowance_uh is None)):
        raise RuntimeError(
            "optimizer resonance and Llt allowances must be supplied together"
        )
    optimizer_allowance = None
    if optimizer_resonance_allowance_hz is not None:
        optimizer_allowance = install_optimizer_allowance(
            problem,
            optimizer_normalization,
            resonance_allowance_hz=optimizer_resonance_allowance_hz,
            llt_allowance_uh=optimizer_llt_allowance_uh,
        )
    if bool(warm_start) != bool(warm_start_sha256):
        raise RuntimeError("warm start path and SHA must be supplied together")
    warm = None
    warm_audit = None
    if warm_start is not None:
        warm_start = warm_start.resolve()
        actual_warm_sha = _sha256(warm_start)
        if actual_warm_sha != warm_start_sha256:
            raise RuntimeError("warm start SHA mismatch")
        warm = np.asarray(np.load(warm_start, allow_pickle=False), dtype=float)
        if warm.ndim != 2 or warm.shape[1] != problem.n_var:
            raise RuntimeError("warm start coordinate schema mismatch")
        if not np.isfinite(warm).all():
            raise RuntimeError("warm start coordinates are not finite")
        source_coordinate_count = int(len(warm))
        warm, fixed_warm_audit = prepare_fixed_primary_turns_warm_start(
            problem, warm, fixed_turns_contract, source_sha_verified=True,
        )
        warm_audit = {
            "path": str(warm_start), "sha256": actual_warm_sha,
            "coordinate_count": source_coordinate_count,
            "post_repair_coordinate_count": len(warm),
            "dimension_count": warm.shape[1],
            "repaired_by_current_simultaneous_hard_constraint_problem": True,
            "fixed_primary_turns_repair": fixed_turns_contract,
            "fixed_primary_turns_warm_audit": fixed_warm_audit,
        }
    termination_contract = optimizer_termination_contract(
        optimizer_termination_strategy, max_generations,
    )
    topology_contract = None
    if (
        optimizer_termination_strategy
        == FIXED_GENERATION_TERMINATION_STRATEGY
        and fixed_primary_turns in (5, 6)
    ):
        topology_contract = _deep_topology_contract(fixed_primary_turns)
        if sobol_names[topology_contract["coordinate_index"]] != (
            topology_contract["coordinate_name"]
        ):
            raise RuntimeError("turn-split topology coordinate schema drifted")
    result = _run_optimizer(
        problem, seed=int(seed), population=int(population),
        warm_start=warm, max_generations=int(max_generations),
        termination_strategy=optimizer_termination_strategy,
        topology_evolution_contract=topology_contract,
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feasible_count = 0
    candidates = []

    def jsonable_params(series: pd.Series) -> dict:
        output_params = {}
        for key, value in series.to_dict().items():
            if not isinstance(value, (str, int, float, np.integer, np.floating)):
                continue
            normalized = value.item() if hasattr(value, "item") else value
            if isinstance(normalized, float) and not math.isfinite(normalized):
                continue
            output_params[key] = normalized
        return output_params

    if result.X is not None:
        coordinates = np.atleast_2d(result.X)
        objectives = np.atleast_2d(result.F)
        frame, _shrink, valid = problem.decode_batch(coordinates)
        for index in range(len(coordinates)):
            predictions = {}
            uncertainty = {}
            for target, model in models.items():
                mu, half = model.predict_mu_sigma(frame.iloc[[index]])
                predictions[target] = float(mu[0])
                uncertainty[target] = float(half[0])
            screen = derive_resonance_screen(predictions, frame.iloc[index].to_dict())
            candidates.append({
                "index": index,
                "volume_L": float(objectives[index, 0]),
                "total_loss_W": float(objectives[index, 1]),
                "decoded_params": jsonable_params(frame.iloc[index]),
                "predictions": predictions,
                "conformal_half_width": uncertainty,
                "derived_resonance": screen,
                "decoder_valid": bool(valid[index]),
                "production_eligible": False,
            })
        feasible_count = len(candidates)
    population_X = np.asarray(result.algorithm.pop.get("X"), dtype=float)
    evaluated = {}
    # Re-run the terminal population through the retained physical evaluator.
    # The optimizer saw dimensionless G values with identical signs, while all
    # persisted evidence and every downstream gate continue to use engineering
    # units exactly as before.
    physical_evaluate(population_X, evaluated)
    constraint_matrix = np.asarray(evaluated["G"], dtype=float)
    normalization_vector = np.asarray([
        optimizer_normalization["scales"][name]
        for name in problem.constraint_names
    ], dtype=float)
    allowance_vector = np.zeros(len(problem.constraint_names), dtype=float)
    if optimizer_allowance is not None:
        allowance_vector = np.asarray([
            optimizer_allowance["allowances"][name]
            for name in problem.constraint_names
        ], dtype=float)
    normalized_constraint_matrix = (
        constraint_matrix - allowance_vector
    ) / normalization_vector
    terminal_frame = evaluated["frame"].reset_index(drop=True)
    primary_turn_values = pd.to_numeric(
        terminal_frame["N1_main"], errors="coerce"
    ) + pd.to_numeric(terminal_frame["N1_side"], errors="coerce")
    unique_primary_turn_values = sorted({
        int(value) for value in primary_turn_values.to_numpy(dtype=float)
        if math.isfinite(float(value)) and float(value).is_integer()
    })
    fixed_turns_verified = None
    if fixed_primary_turns is not None:
        fixed_turns_verified = bool(
            np.isfinite(primary_turn_values.to_numpy(dtype=float)).all()
            and np.equal(
                primary_turn_values.to_numpy(dtype=float),
                float(fixed_primary_turns),
            ).all()
        )
        if not fixed_turns_verified:
            raise RuntimeError(
                "terminal population escaped fixed primary-turn stratum"
            )
    topology_evolution_audit = None
    if topology_contract is not None:
        topologies = tuple(
            topology_contract["turn_split_sub_islands_N2_main"]
        )
        terminal_n2_main = pd.to_numeric(
            terminal_frame["N2_main"], errors="coerce",
        ).to_numpy(dtype=float)
        terminal_topology_counts = {
            str(topology): int(np.count_nonzero(
                terminal_n2_main == float(topology)
            ))
            for topology in topologies
        }
        minimum_each = int(
            topology_contract["survival"][
                "minimum_survivors_per_turn_split_sub_island"
            ]
        )
        diversity_budget = topology_contract["bounded_diversity_budget"]
        maximum_single_topology = int(
            diversity_budget["maximum_single_protected_topology_count"]
        )
        lane_topologies = {
            str(name): tuple(
                int(pair[0])
                for pair in lane["topologies_N2_main_N2_side"]
            )
            for name, lane in topology_contract["basin_donor_lanes"].items()
        }
        terminal_lane_counts = {
            name: sum(
                terminal_topology_counts[str(topology)]
                for topology in values
            )
            for name, values in lane_topologies.items()
        }
        operator_audit = getattr(result, "topology_operator_audit", None)
        if (
            not isinstance(operator_audit, dict)
            or operator_audit.get("paired_selection_calls", 0) < 1
            or operator_audit.get("paired_parent_pairs_emitted", 0) < 1
            or operator_audit.get("migration_events", 0) < 1
            or operator_audit.get("migrants_created", 0) < len(topologies)
            or operator_audit.get("survival_calls", 0) < 2
            or operator_audit.get("minimum_topology_count_observed", 0)
            < minimum_each
            or operator_audit.get(
                "maximum_single_topology_count_observed", math.inf
            ) > maximum_single_topology
            or operator_audit.get("last_topology_counts")
            != terminal_topology_counts
            or any(
                terminal_topology_counts[str(topology)] < minimum_each
                for topology in topologies
            )
        ):
            raise RuntimeError("turn-split topology evolution audit failed")
        expected_last_epsilon = float(
            topology_contract["survival"]["initial_epsilon"]
        ) * max(
            0.0,
            1.0 - max(0, int(max_generations) - 1) / int(
                topology_contract["survival"]["decay_to_zero_generation"]
            ),
        ) ** 2
        if not math.isclose(
            float(operator_audit["last_optimizer_epsilon"]),
            expected_last_epsilon, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError("turn-split epsilon schedule drifted")
        topology_evolution_audit = {
            "schema_version": "mft-tier1-turn-split-evolution-audit-v1",
            **operator_audit,
            "terminal_topology_counts": terminal_topology_counts,
            "terminal_basin_lane_counts": terminal_lane_counts,
            "bounded_diversity_budget": diversity_budget,
            "all_required_topologies_preserved": True,
            "single_topology_collapse_prevented": True,
            "terminal_epsilon_zero": bool(expected_last_epsilon == 0.0),
            "physical_constraint_G_mutation": False,
            "physical_objective_mutation": False,
            "warm_donor_prediction_inheritance": False,
        }
        topology_evolution_audit["sha256"] = _json_sha(
            topology_evolution_audit
        )
    total_violation = np.maximum(constraint_matrix, 0.0).sum(axis=1)
    best_index = int(np.argmin(total_violation))
    normalized_total_violation = np.maximum(
        normalized_constraint_matrix, 0.0,
    ).sum(axis=1)
    normalized_best_index = int(np.argmin(normalized_total_violation))
    observed_generation_counter = int(result.algorithm.n_gen)
    if (
        optimizer_termination_strategy
        == FIXED_GENERATION_TERMINATION_STRATEGY
        and observed_generation_counter != int(max_generations) + 1
    ):
        raise RuntimeError(
            "fixed-generation optimizer counter differs from its sealed gate"
        )

    def terminal_design_snapshot(index: int) -> dict[str, Any]:
        coordinate = [float(value) for value in population_X[index]]
        params = jsonable_params(terminal_frame.iloc[index])
        return {
            "terminal_population_index": int(index),
            "coordinate_unit": coordinate,
            "coordinate_unit_sha256": _json_sha(coordinate),
            "decoded_params": params,
            "decoded_params_sha256": _json_sha(params),
            "physical_constraint_G": {
                name: float(constraint_matrix[index, position])
                for position, name in enumerate(problem.constraint_names)
            },
            "physical_constraint_G_sha256": _json_sha({
                name: float(constraint_matrix[index, position])
                for position, name in enumerate(problem.constraint_names)
            }),
        }

    physical_best_design = terminal_design_snapshot(best_index)
    optimizer_best_design = terminal_design_snapshot(normalized_best_index)
    constraint_minima = {
        name: float(np.nanmin(constraint_matrix[:, index]))
        for index, name in enumerate(problem.constraint_names)
    }
    constraint_index = {
        name: index for index, name in enumerate(problem.constraint_names)
    }
    geometry_names = (
        "analytical_flux_density_limit", "decoded_space_shrink",
        "minimum_physical_insulation", "core_group_manufacturability_limit",
        "exterior_width_limit", "exterior_length_limit", "exterior_height_limit",
    )
    geometry_ok = np.ones(len(population_X), dtype=bool)
    for name in geometry_names:
        geometry_ok &= constraint_matrix[:, constraint_index[name]] <= 1e-9
    base_target_score = (
        np.maximum(constraint_matrix[:, constraint_index["Llt_robust_band"]], 0.0)
        + np.maximum(
            constraint_matrix[:, constraint_index["half_magnetizing_resonance_minimum"]],
            0.0,
        ) / 1_000.0
        + np.maximum(
            constraint_matrix[:, constraint_index["strict_full_density_support"]],
            0.0,
        )
        + np.maximum(
            constraint_matrix[:, constraint_index["Llt_ensemble_disagreement"]],
            0.0,
        )
    )
    (
        crossover_acquisition_pressure,
        crossover_acquisition_components,
        crossover_acquisition_contract,
    ) = thermal_crossover_acquisition_pressure(
        constraint_matrix,
        problem.constraint_names,
        HARD_SPEC,
        optimizer_core_thermal_scale_c,
    )
    if optimizer_all_thermal_scale_c is not None:
        if (
            crossover_acquisition_contract.get("active") is not False
            or np.any(crossover_acquisition_pressure != 0.0)
            or any(
                np.any(values != 0.0)
                for values in crossover_acquisition_components.values()
            )
        ):
            raise RuntimeError(
                "orthogonal all-thermal lane unexpectedly activated "
                "soft-axis acquisition pressure"
            )
        crossover_acquisition_contract = None
    target_score = base_target_score + crossover_acquisition_pressure
    ranked_pool = np.where(geometry_ok & np.isfinite(target_score))[0]
    ranked_pool = ranked_pool[np.argsort(target_score[ranked_pool], kind="stable")]
    # Greedy max-min diversity over a good-score pool.  This is a plan only;
    # the scheduler cap must be checked afresh before any later submission.
    ranked_pool = ranked_pool[: min(len(ranked_pool), 256)]
    selected_indices: list[int] = []
    if len(ranked_pool):
        selected_indices.append(int(ranked_pool[0]))
    while len(selected_indices) < min(32, len(ranked_pool)):
        remaining = [index for index in ranked_pool if int(index) not in selected_indices]
        if not remaining:
            break
        distances = np.asarray([
            min(
                float(np.linalg.norm(population_X[index] - population_X[chosen]))
                for chosen in selected_indices
            )
            for index in remaining
        ])
        score_scale = np.asarray([target_score[index] for index in remaining])
        score_scale = score_scale / max(float(np.max(score_scale)), 1e-12)
        chosen_position = int(np.argmax(distances - 0.1 * score_scale))
        selected_indices.append(int(remaining[chosen_position]))
    next_fea_candidates = []
    seen_parameter_sha = set()
    for index in selected_indices:
        params = jsonable_params(terminal_frame.iloc[index])
        params_sha = _json_sha(params)
        if params_sha in seen_parameter_sha:
            continue
        seen_parameter_sha.add(params_sha)
        target_predictions = {}
        target_uncertainty = {}
        for target in TARGETS:
            mu, half = models[target].predict_mu_sigma(terminal_frame.iloc[[index]])
            target_predictions[target] = float(mu[0])
            target_uncertainty[target] = float(half[0])
        next_fea_candidates.append({
            "rank": len(next_fea_candidates) + 1,
            "terminal_population_index": index,
            "decoded_params": params,
            "decoded_params_sha256": params_sha,
            "target_predictions": target_predictions,
            "target_conformal_half_width": target_uncertainty,
            "derived_resonance": derive_resonance_screen(target_predictions, params),
            "target_acquisition_score": float(target_score[index]),
            "target_acquisition_score_components": {
                "base_target_normalized_positive_violation": float(
                    base_target_score[index]
                ),
                "core_thermal_positive_normalized": float(
                    crossover_acquisition_components[
                        "core_thermal_positive_normalized"
                    ][index]
                ),
                "soft_width_positive_excess_normalized": float(
                    crossover_acquisition_components[
                        "soft_width_positive_excess_normalized"
                    ][index]
                ),
                "soft_length_positive_excess_normalized": float(
                    crossover_acquisition_components[
                        "soft_length_positive_excess_normalized"
                    ][index]
                ),
                "total": float(target_score[index]),
            },
            "constraint_G": {
                name: float(constraint_matrix[index, position])
                for position, name in enumerate(problem.constraint_names)
            },
            "production_eligible": False,
            "fea_submission_approved": False,
            "eligible_for_submission": False,
        })
    thermal_constraint_names = [
        name for name in problem.constraint_names
        if name.startswith("temperature_robust_limit:")
    ]
    expected_thermal_constraint_names = [
        f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
    ]
    if thermal_constraint_names != expected_thermal_constraint_names:
        raise RuntimeError("result thermal constraint order differs from v3 contract")
    search = {
        "schema_version": SEARCH_SCHEMA,
        "created_at": _now(),
        "seed": int(seed),
        "population": int(population),
        "max_generations": int(max_generations),
        "inference_threads": inference_threads,
        "thread_bindings": thread_bindings,
        "completed_generations": observed_generation_counter,
        "evaluated_generations": (
            int(max_generations)
            if optimizer_termination_strategy
            == FIXED_GENERATION_TERMINATION_STRATEGY
            else None
        ),
        "nsga_code_revision": revision,
        "model_manifest_sha256": _sha256(model_dir / "manifest.json"),
        "source_plan_sha256": report["source_plan_sha256"],
        "hard_spec": HARD_SPEC,
        "constraint_version": CONSTRAINT_VERSION,
        "hard_spec_sha256": _json_sha(HARD_SPEC),
        "temperature_constraint_contract": TEMPERATURE_CONSTRAINT_CONTRACT,
        "temperature_constraint_contract_sha256": (
            TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        ),
        "warm_start": warm_audit,
        "initialization_audit": getattr(result, "initialization_audit", None),
        "optimizer_termination_contract": termination_contract,
        "optimizer_termination_strategy": optimizer_termination_strategy,
        "optimizer_topology_evolution_contract": topology_contract,
        "optimizer_topology_evolution_audit": topology_evolution_audit,
        "fixed_generation_gate_satisfied": (
            observed_generation_counter == int(max_generations) + 1
            if optimizer_termination_strategy
            == FIXED_GENERATION_TERMINATION_STRATEGY
            else None
        ),
        "optimizer_constraint_normalization": optimizer_normalization,
        "optimizer_resonance_scale_Hz": optimizer_resonance_scale_hz,
        "optimizer_core_thermal_scale_C": optimizer_core_thermal_scale_c,
        "optimizer_Llt_scale_uH": optimizer_llt_scale_uh,
        "optimizer_all_active_thermal_scale_C": (
            optimizer_all_thermal_scale_c
        ),
        "optimizer_resonance_allowance_Hz": (
            optimizer_resonance_allowance_hz
        ),
        "optimizer_Llt_allowance_uH": optimizer_llt_allowance_uh,
        "optimizer_constraint_allowance_contract": optimizer_allowance,
        "fixed_primary_turns": fixed_primary_turns,
        "fixed_primary_turns_contract": fixed_turns_contract,
        "terminal_population_primary_turn_values": unique_primary_turn_values,
        "terminal_population_fixed_primary_turns_verified": (
            fixed_turns_verified
        ),
        "acquisition_ranking_contract": crossover_acquisition_contract,
        "constraint_names": list(problem.constraint_names),
        "constraint_minimum_G": constraint_minima,
        "terminal_population_minimum_total_positive_violation": float(total_violation[best_index]),
        "terminal_population_best_constraint_G": {
            name: float(constraint_matrix[best_index, index])
            for index, name in enumerate(problem.constraint_names)
        },
        "terminal_population_best_design": physical_best_design,
        "optimizer_terminal_minimum_normalized_total_positive_violation": (
            float(normalized_total_violation[normalized_best_index])
        ),
        "optimizer_terminal_best_physical_constraint_G": {
            name: float(constraint_matrix[normalized_best_index, index])
            for index, name in enumerate(problem.constraint_names)
        },
        "optimizer_terminal_best_design": optimizer_best_design,
        "feasible_pareto_count": feasible_count,
        "candidates": candidates,
        "next_target_fea_batch_plan": {
            "schema_version": "mft-tier1-next-target-fea-batch-plan-v1",
            "selection": "hard-geometry then target-violation plus coordinate diversity",
            "candidate_count": len(next_fea_candidates),
            "candidates": next_fea_candidates,
            "submission_performed": False,
            "scheduler_cap": 500,
            "scheduler_cap_verified_before_submission": False,
            "submission_requires_fresh_cap_check": True,
            "submission_requires_all_hard_constraints_pass": True,
            "current_candidates_eligible_for_submission": False,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    _atomic_json(output / "result.json", search)
    return search


def launch_searches(
    *, python: Path, script: Path, model_dir: Path, nsga_code_root: Path,
    output: Path, seeds: list[int], population: int, max_generations: int,
    warm_start: Path | None = None,
    warm_start_sha256: str | None = None,
    optimizer_resonance_scale_hz: float | None = None,
    optimizer_core_thermal_scale_c: float | None = None,
    optimizer_llt_scale_uh: float | None = None,
    optimizer_all_thermal_scale_c: float | None = None,
    optimizer_resonance_allowance_hz: float | None = None,
    optimizer_llt_allowance_uh: float | None = None,
    fixed_primary_turns: int | None = None,
    optimizer_termination_strategy: str = DEFAULT_TERMINATION_STRATEGY,
) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if ((optimizer_resonance_allowance_hz is None)
            != (optimizer_llt_allowance_uh is None)):
        raise RuntimeError(
            "optimizer resonance and Llt allowances must be supplied together"
        )
    if bool(warm_start) != bool(warm_start_sha256):
        raise RuntimeError("warm start path and SHA must be supplied together")
    if warm_start is not None:
        warm_start = warm_start.resolve()
        if _sha256(warm_start) != warm_start_sha256:
            raise RuntimeError("warm start SHA mismatch before launch")
    jobs = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for seed in seeds:
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), str(script), "search-seed",
            "--model-dir", str(model_dir),
            "--nsga-code-root", str(nsga_code_root),
            "--output", str(seed_dir),
            "--seed", str(seed),
            "--population", str(population),
            "--max-generations", str(max_generations),
            "--optimizer-termination-strategy",
            optimizer_termination_strategy,
        ]
        if warm_start is not None:
            command.extend([
                "--warm-start", str(warm_start),
                "--warm-start-sha256", str(warm_start_sha256),
            ])
        if optimizer_resonance_scale_hz is not None:
            command.extend([
                "--optimizer-resonance-scale-hz",
                str(float(optimizer_resonance_scale_hz)),
            ])
        if optimizer_core_thermal_scale_c is not None:
            command.extend([
                "--optimizer-core-thermal-scale-c",
                str(float(optimizer_core_thermal_scale_c)),
            ])
        if optimizer_llt_scale_uh is not None:
            command.extend([
                "--optimizer-llt-scale-uh",
                str(float(optimizer_llt_scale_uh)),
            ])
        if optimizer_all_thermal_scale_c is not None:
            command.extend([
                "--optimizer-all-thermal-scale-c",
                str(float(optimizer_all_thermal_scale_c)),
            ])
        if optimizer_resonance_allowance_hz is not None:
            command.extend([
                "--optimizer-resonance-allowance-hz",
                str(float(optimizer_resonance_allowance_hz)),
            ])
        if optimizer_llt_allowance_uh is not None:
            command.extend([
                "--optimizer-llt-allowance-uh",
                str(float(optimizer_llt_allowance_uh)),
            ])
        if fixed_primary_turns is not None:
            command.extend([
                "--fixed-primary-turns", str(int(fixed_primary_turns)),
            ])
        stdout = open(seed_dir / "stdout.log", "ab", buffering=0)
        stderr = open(seed_dir / "stderr.log", "ab", buffering=0)
        process = subprocess.Popen(
            command, cwd=str(nsga_code_root), stdout=stdout, stderr=stderr,
            stdin=subprocess.DEVNULL, creationflags=creationflags,
        )
        stdout.close()
        stderr.close()
        jobs.append({
            "seed": seed, "pid": process.pid, "command": command,
            "output": str(seed_dir),
        })
    launch = {
        "schema_version": "mft-tier1-corrected-search-launch-v1",
        "created_at": _now(),
        "model_manifest_sha256": _sha256(model_dir / "manifest.json"),
        "nsga_code_revision": _git_revision(nsga_code_root),
        "population": population,
        "max_generations": max_generations,
        "optimizer_termination_strategy": optimizer_termination_strategy,
        "optimizer_termination_contract": optimizer_termination_contract(
            optimizer_termination_strategy, max_generations,
        ),
        "optimizer_resonance_scale_hz": optimizer_resonance_scale_hz,
        "optimizer_core_thermal_scale_c": optimizer_core_thermal_scale_c,
        "optimizer_llt_scale_uh": optimizer_llt_scale_uh,
        "optimizer_all_thermal_scale_c": optimizer_all_thermal_scale_c,
        "optimizer_resonance_allowance_hz": (
            optimizer_resonance_allowance_hz
        ),
        "optimizer_llt_allowance_uh": optimizer_llt_allowance_uh,
        "fixed_primary_turns": fixed_primary_turns,
        "warm_start": (
            {
                "path": str(warm_start.resolve()),
                "sha256": warm_start_sha256,
            }
            if warm_start is not None else None
        ),
        "jobs": jobs,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    _atomic_json(output / "launch.json", launch)
    return launch


def monitor_searches(
    *, launch_path: Path, output: Path, poll_seconds: float = 15.0,
    once: bool = False, identity_status_path: Path | None = None,
) -> dict:
    """Attest launched seed processes without changing search or FEA state."""
    launch_path = launch_path.resolve()
    output = output.resolve()
    if poll_seconds <= 0:
        raise RuntimeError("poll_seconds must be positive")
    launch = _read_json(launch_path)
    if launch.get("schema_version") != "mft-tier1-corrected-search-launch-v1":
        raise RuntimeError("unrecognized launch manifest schema")

    durable_launch_identity: dict[int, bool] = {}
    if identity_status_path is not None:
        identity_status_path = identity_status_path.resolve()
        identity_status = _read_json(identity_status_path)
        if identity_status.get("launch_sha256") != _sha256(launch_path):
            raise RuntimeError("launch identity status is bound to another launch")
        durable_launch_identity.update({
            int(job["seed"]): True
            for job in identity_status.get("jobs", [])
            if job.get("command_identity_verified") is True
            or job.get("launch_command_identity_verified") is True
        })

    try:
        import psutil
    except ImportError as error:  # pragma: no cover - deployed environment has it
        raise RuntimeError("psutil is required for PID identity attestation") from error

    while True:
        if output.is_file():
            previous = _read_json(output)
            if previous.get("launch_sha256") == _sha256(launch_path):
                durable_launch_identity.update({
                    int(job["seed"]): True
                    for job in previous.get("jobs", [])
                    if job.get("launch_command_identity_verified") is True
                    or (
                        job.get("process_alive") is True
                        and job.get("command_identity_verified") is True
                    )
                })
        jobs = []
        for launched in launch["jobs"]:
            seed = int(launched["seed"])
            pid = int(launched["pid"])
            seed_output = Path(launched["output"]).resolve()
            result_path = seed_output / "result.json"
            stdout_path = seed_output / "stdout.log"
            stderr_path = seed_output / "stderr.log"
            process_alive = False
            command_identity_verified = False
            rss_bytes = None
            cpu_seconds = None
            process_created_at = None
            try:
                process = psutil.Process(pid)
                process_alive = process.is_running() and (
                    process.status() != psutil.STATUS_ZOMBIE
                )
                command_line = process.cmdline()
                expected_tokens = [str(token) for token in launched["command"]]
                command_identity_verified = command_line == expected_tokens
                memory = process.memory_info()
                times = process.cpu_times()
                rss_bytes = int(memory.rss)
                cpu_seconds = float(times.user + times.system)
                process_created_at = datetime.fromtimestamp(
                    process.create_time(), tz=timezone.utc
                ).astimezone().isoformat(timespec="seconds")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                process_alive = False
            if process_alive and command_identity_verified:
                durable_launch_identity[seed] = True
            launch_command_identity_verified = durable_launch_identity.get(seed, False)
            runtime_command_identity_verified = (
                command_identity_verified if process_alive else None
            )

            completed_generations = None
            feasible_pareto_count = None
            result_sha256 = None
            if result_path.is_file():
                result = _read_json(result_path)
                if int(result.get("seed", -1)) != seed:
                    raise RuntimeError(f"seed result identity mismatch: {seed}")
                if result.get("fea_submission_approved") is not False:
                    raise RuntimeError(f"FEA approval fail-closed violation: {seed}")
                completed_generations = int(result["completed_generations"])
                feasible_pareto_count = int(result["feasible_pareto_count"])
                result_sha256 = _sha256(result_path)
                terminal_result_verified = True
                state = "completed"
            elif process_alive and command_identity_verified:
                # The pinned run_one API exposes n_gen only when minimize returns.
                # Null therefore truthfully means "inside an uncheckpointed
                # generation", never an invented progress estimate.
                state = "running"
                terminal_result_verified = None
            elif process_alive:
                state = "pid_identity_mismatch"
                terminal_result_verified = None
            else:
                state = "failed_before_result"
                terminal_result_verified = False
            jobs.append({
                "seed": seed,
                "pid": pid,
                "state": state,
                "process_alive": process_alive,
                # Backward-compatible aggregate: after launch it represents
                # durable launch attestation, not a query against a dead PID.
                "command_identity_verified": launch_command_identity_verified,
                "launch_command_identity_verified": (
                    launch_command_identity_verified
                ),
                "runtime_command_identity_verified": (
                    runtime_command_identity_verified
                ),
                "terminal_result_verified": terminal_result_verified,
                "process_created_at": process_created_at,
                "rss_bytes": rss_bytes,
                "cpu_seconds": cpu_seconds,
                "completed_generations": completed_generations,
                "generation_progress_contract": (
                    "terminal_exact_intermediate_uncheckpointed_v1"
                ),
                "feasible_pareto_count": feasible_pareto_count,
                "result_sha256": result_sha256,
                "output": str(seed_output),
                "stdout_size": stdout_path.stat().st_size
                if stdout_path.is_file() else 0,
                "stderr_size": stderr_path.stat().st_size
                if stderr_path.is_file() else 0,
            })
        counts = Counter(job["state"] for job in jobs)
        status = {
            "schema_version": "mft-tier1-corrected-search-monitor-v1",
            "updated_at": _now(),
            "monitor_pid": os.getpid(),
            "launch_path": str(launch_path),
            "launch_sha256": _sha256(launch_path),
            "model_manifest_sha256": launch["model_manifest_sha256"],
            "nsga_code_revision": launch["nsga_code_revision"],
            "population": int(launch["population"]),
            "max_generations": int(launch["max_generations"]),
            "state_counts": dict(sorted(counts.items())),
            "jobs": jobs,
            "production_eligible": False,
            "fea_submission_approved": False,
            "submission_performed": False,
        }
        _atomic_json(output, status)
        if once or not any(job["state"] == "running" for job in jobs):
            return status
        time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--runtime", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-verified", type=int, default=63)
    audit.add_argument(
        "--exclude-candidate-id",
        action="append",
        default=[],
        help="verified result excluded from an explicitly frozen cohort",
    )
    audit.add_argument(
        "--delta-from-audit",
        type=Path,
        default=None,
        help="authenticate this cohort as a superset and emit only new rows",
    )

    compose = commands.add_parser("compose")
    compose.add_argument(
        "--audit",
        dest="audits",
        type=Path,
        action="append",
        required=True,
        help="source audit.json; pass once per independently authenticated cohort",
    )
    compose.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--audit-dir", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--base-generation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)

    search = commands.add_parser("search-seed")
    search.add_argument("--model-dir", type=Path, required=True)
    search.add_argument("--nsga-code-root", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--seed", type=int, required=True)
    search.add_argument("--population", type=int, default=160)
    search.add_argument("--max-generations", type=int, default=240)
    search.add_argument(
        "--optimizer-termination-strategy",
        choices=(
            DEFAULT_TERMINATION_STRATEGY,
            FIXED_GENERATION_TERMINATION_STRATEGY,
        ),
        default=DEFAULT_TERMINATION_STRATEGY,
    )
    search.add_argument("--warm-start", type=Path, default=None)
    search.add_argument("--warm-start-sha256", default=None)
    search.add_argument("--inference-threads", type=int, default=1)
    search.add_argument(
        "--optimizer-resonance-scale-hz", type=float, default=None,
        help="optimizer-only scale; physical terminal G remains unscaled",
    )
    search.add_argument(
        "--optimizer-core-thermal-scale-c", type=float, default=None,
        help=(
            "optimizer-only scale applied to exactly the three sealed core "
            "thermal constraints; physical terminal G remains unscaled"
        ),
    )
    search.add_argument(
        "--optimizer-llt-scale-uh", type=float, default=None,
        help=(
            "optimizer-only robust Llt scale; disagreement uses exactly 2x; "
            "physical terminal G remains unscaled"
        ),
    )
    search.add_argument(
        "--optimizer-all-thermal-scale-c", type=float, default=None,
        help=(
            "optimizer-only scale for all active sealed thermal constraints; "
            "physical terminal G and side-winding activation remain unchanged"
        ),
    )
    search.add_argument(
        "--optimizer-resonance-allowance-hz", type=float, default=None,
        help=(
            "optimizer-only resonance preservation allowance; physical "
            "terminal G remains unchanged"
        ),
    )
    search.add_argument(
        "--optimizer-llt-allowance-uh", type=float, default=None,
        help=(
            "optimizer-only Llt preservation allowance; physical terminal "
            "G remains unchanged"
        ),
    )
    search.add_argument(
        "--fixed-primary-turns", type=int, default=None,
        help=(
            "force the authenticated u_N1 decoder stratum before every "
            "physics repair; physical constraints/objectives remain unchanged"
        ),
    )

    launch = commands.add_parser("launch")
    launch.add_argument("--python", type=Path, required=True)
    launch.add_argument("--model-dir", type=Path, required=True)
    launch.add_argument("--nsga-code-root", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--seeds", default="26071801,26071802,26071803,26071804")
    launch.add_argument("--population", type=int, default=160)
    launch.add_argument("--max-generations", type=int, default=240)
    launch.add_argument(
        "--optimizer-termination-strategy",
        choices=(
            DEFAULT_TERMINATION_STRATEGY,
            FIXED_GENERATION_TERMINATION_STRATEGY,
        ),
        default=DEFAULT_TERMINATION_STRATEGY,
    )
    launch.add_argument("--warm-start", type=Path, default=None)
    launch.add_argument("--warm-start-sha256", default=None)
    launch.add_argument(
        "--optimizer-resonance-scale-hz", type=float, default=None,
    )
    launch.add_argument(
        "--optimizer-core-thermal-scale-c", type=float, default=None,
    )
    launch.add_argument(
        "--optimizer-llt-scale-uh", type=float, default=None,
    )
    launch.add_argument(
        "--optimizer-all-thermal-scale-c", type=float, default=None,
    )
    launch.add_argument(
        "--optimizer-resonance-allowance-hz", type=float, default=None,
    )
    launch.add_argument(
        "--optimizer-llt-allowance-uh", type=float, default=None,
    )
    launch.add_argument("--fixed-primary-turns", type=int, default=None)

    monitor = commands.add_parser("monitor")
    monitor.add_argument("--launch", type=Path, required=True)
    monitor.add_argument("--output", type=Path, required=True)
    monitor.add_argument("--poll-seconds", type=float, default=15.0)
    monitor.add_argument("--once", action="store_true")
    monitor.add_argument("--identity-status", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit":
        result = write_audit(
            args.runtime,
            args.output,
            args.expected_verified,
            args.exclude_candidate_id,
            args.delta_from_audit,
        )
    elif args.command == "compose":
        result = compose_audits(audit_paths=args.audits, output=args.output)
    elif args.command == "train":
        result = train_feedback(
            audit_dir=args.audit_dir, dataset=args.dataset,
            base_generation=args.base_generation, output=args.output,
        )
    elif args.command == "search-seed":
        result = run_search_seed(
            model_dir=args.model_dir, nsga_code_root=args.nsga_code_root,
            output=args.output, seed=args.seed, population=args.population,
            max_generations=args.max_generations,
            warm_start=args.warm_start,
            warm_start_sha256=args.warm_start_sha256,
            inference_threads=args.inference_threads,
            optimizer_resonance_scale_hz=(
                args.optimizer_resonance_scale_hz
            ),
            optimizer_core_thermal_scale_c=(
                args.optimizer_core_thermal_scale_c
            ),
            optimizer_llt_scale_uh=args.optimizer_llt_scale_uh,
            optimizer_all_thermal_scale_c=(
                args.optimizer_all_thermal_scale_c
            ),
            optimizer_resonance_allowance_hz=(
                args.optimizer_resonance_allowance_hz
            ),
            optimizer_llt_allowance_uh=args.optimizer_llt_allowance_uh,
            fixed_primary_turns=args.fixed_primary_turns,
            optimizer_termination_strategy=(
                args.optimizer_termination_strategy
            ),
        )
    elif args.command == "launch":
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
        if len(seeds) != len(set(seeds)) or not seeds:
            raise SystemExit("--seeds must contain distinct integers")
        result = launch_searches(
            python=args.python, script=Path(__file__).resolve(),
            model_dir=args.model_dir, nsga_code_root=args.nsga_code_root,
            output=args.output, seeds=seeds, population=args.population,
            max_generations=args.max_generations,
            warm_start=args.warm_start,
            warm_start_sha256=args.warm_start_sha256,
            optimizer_resonance_scale_hz=(
                args.optimizer_resonance_scale_hz
            ),
            optimizer_core_thermal_scale_c=(
                args.optimizer_core_thermal_scale_c
            ),
            optimizer_llt_scale_uh=args.optimizer_llt_scale_uh,
            optimizer_all_thermal_scale_c=(
                args.optimizer_all_thermal_scale_c
            ),
            optimizer_resonance_allowance_hz=(
                args.optimizer_resonance_allowance_hz
            ),
            optimizer_llt_allowance_uh=args.optimizer_llt_allowance_uh,
            fixed_primary_turns=args.fixed_primary_turns,
            optimizer_termination_strategy=(
                args.optimizer_termination_strategy
            ),
        )
    else:
        result = monitor_searches(
            launch_path=args.launch, output=args.output,
            poll_seconds=args.poll_seconds, once=args.once,
            identity_status_path=args.identity_status,
        )
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
