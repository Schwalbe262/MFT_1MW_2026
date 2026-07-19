"""Fail-closed optimizer-only contracts for fixed-N1=6 anchor islands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


SCHEMA = "mft-tier1-fixed-n1-6-anchor-island-v1"
ALLOWANCE_SCHEMA = "mft-tier1-optimizer-constraint-allowance-v1"
RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
LLT_CONSTRAINT = "Llt_robust_band"
RESONANCE_SCALE_HZ = 150.0
LLT_SCALE_UH = 0.05
ALL_THERMAL_SCALE_C = 2.0
MAX_RESONANCE_ALLOWANCE_HZ = 2_000.0
MAX_LLT_ALLOWANCE_UH = 0.05


@dataclass(frozen=True)
class AnchorIsland:
    island_id: str
    variant: str
    runtime_suffix: str
    namespace: str
    task_name_stem: str
    dedupe_namespace: str
    search_focus: str
    active_quota: int
    seed_start: int
    resonance_allowance_hz: float
    llt_allowance_uh: float
    staircase_order: int | None


def _staircase(cap_hz: int, order: int, seed_start: int) -> AnchorIsland:
    token = str(cap_hz)
    return AnchorIsland(
        island_id=f"thermal-llt-rescap-{token}hz",
        variant=f"fixed-n1-6-anchor-b-rescap-{token}hz-v1",
        runtime_suffix=(
            f"mft_tier1_nsga_n1_6_anchor_b_rescap_{token}_t110_res15k_260719"
        ),
        namespace=(
            f"fixed-n1-6-anchor-b-allthermal2c-llt0p05-res150-"
            f"cap{token}-p320-g600-warm64-v1"
        ),
        task_name_stem=f"mft-t1n6b{token}",
        dedupe_namespace=f"mft-tier1-fixed-n1-6-anchor-b-{token}-nsga",
        search_focus=(
            "preserve_all11_thermal_and_Llt_then_tighten_resonance_"
            f"cap_{token}Hz"
        ),
        active_quota=4,
        seed_start=seed_start,
        resonance_allowance_hz=float(cap_hz),
        llt_allowance_uh=0.0,
        staircase_order=order,
    )


ANCHOR_ISLANDS = (
    AnchorIsland(
        island_id="resonance-llt-anchor",
        variant="fixed-n1-6-anchor-a-resllt-v1",
        runtime_suffix=(
            "mft_tier1_nsga_n1_6_anchor_a_resllt_seed205_"
            "t110_res15k_260719"
        ),
        namespace=(
            "fixed-n1-6-anchor-a-allthermal2c-llt0p05-res150-"
            "eps150-p320-g600-warm64-seed205-v1"
        ),
        task_name_stem="mft-t1n6a205",
        dedupe_namespace="mft-tier1-fixed-n1-6-anchor-a-seed205-nsga",
        search_focus=(
            "preserve_resonance_G_le_150Hz_and_Llt_G_le_0p05uH_"
            "then_minimize_all11_robust_temperature"
        ),
        active_quota=16,
        seed_start=1_907_205_000,
        resonance_allowance_hz=150.0,
        llt_allowance_uh=0.05,
        staircase_order=None,
    ),
    _staircase(2_000, 0, 1_907_200_000),
    _staircase(1_500, 1, 1_907_201_000),
    _staircase(1_000, 2, 1_907_202_000),
    _staircase(500, 3, 1_907_203_000),
    _staircase(0, 4, 1_907_204_000),
)
BY_VARIANT = {island.variant: island for island in ANCHOR_ISLANDS}
BY_ID = {island.island_id: island for island in ANCHOR_ISLANDS}
TOTAL_ACTIVE_QUOTA = sum(island.active_quota for island in ANCHOR_ISLANDS)


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def island_profile(island: AnchorIsland) -> dict[str, Any]:
    profile = {
        "schema_version": SCHEMA,
        **asdict(island),
        "fixed_primary_turns": 6,
        "optimizer_resonance_scale_Hz": RESONANCE_SCALE_HZ,
        "optimizer_Llt_scale_uH": LLT_SCALE_UH,
        "optimizer_all_active_thermal_scale_C": ALL_THERMAL_SCALE_C,
        "soft_axis_pressure_enabled": False,
        "physical_hard_spec_mutation": False,
        "authoritative_terminal_G": "physical_unscaled_unchanged",
    }
    profile["sha256"] = canonical_sha(profile)
    return profile


def _allowance(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite and non-negative")
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(output) or not 0.0 <= output <= maximum:
        raise RuntimeError(
            f"{label} must be within the sealed [0, {maximum}] range"
        )
    return output


def optimizer_allowance_contract(
    constraint_names: Iterable[str], *, resonance_allowance_hz: Any,
    llt_allowance_uh: Any,
) -> dict[str, Any]:
    names = tuple(str(name) for name in constraint_names)
    if not names or len(set(names)) != len(names):
        raise RuntimeError("optimizer allowance constraint names must be unique")
    missing = {RESONANCE_CONSTRAINT, LLT_CONSTRAINT}.difference(names)
    if missing:
        raise RuntimeError(
            f"optimizer allowance constraints are missing: {sorted(missing)}"
        )
    resonance = _allowance(
        resonance_allowance_hz,
        "optimizer resonance allowance",
        MAX_RESONANCE_ALLOWANCE_HZ,
    )
    llt = _allowance(
        llt_allowance_uh,
        "optimizer Llt allowance",
        MAX_LLT_ALLOWANCE_UH,
    )
    allowances = {
        name: 0.0 for name in names
    }
    allowances[RESONANCE_CONSTRAINT] = resonance
    allowances[LLT_CONSTRAINT] = llt
    contract = {
        "schema_version": ALLOWANCE_SCHEMA,
        "purpose": "optimizer_anchor_preservation_only",
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "formula": (
            "G_optimizer=(G_physical-optimizer_only_allowance)/"
            "engineering_scale"
        ),
        "constraint_order": list(names),
        "allowances": allowances,
        "nonzero_allowances": {
            name: value for name, value in allowances.items() if value != 0.0
        },
        "optimizer_resonance_allowance_Hz": resonance,
        "optimizer_Llt_allowance_uH": llt,
        "hard_constraint_mutation": False,
        "result_replay_uses_physical_evaluator": True,
    }
    contract["sha256"] = canonical_sha(contract)
    return contract


def install_optimizer_allowance(
    problem: Any, normalization_contract: dict[str, Any], *,
    resonance_allowance_hz: Any, llt_allowance_uh: Any,
) -> dict[str, Any]:
    """Apply allowances after normalization without touching physical replay."""

    names = tuple(str(name) for name in problem.constraint_names)
    if (
        normalization_contract.get("constraint_order") != list(names)
        or normalization_contract.get("authoritative_terminal_G")
        != "physical_unscaled"
    ):
        raise RuntimeError("optimizer normalization/allowance order mismatch")
    scales = normalization_contract.get("scales")
    if not isinstance(scales, dict) or set(scales) != set(names):
        raise RuntimeError("optimizer allowance has no sealed scale vector")
    scale_vector = np.asarray([float(scales[name]) for name in names], dtype=float)
    if not np.isfinite(scale_vector).all() or (scale_vector <= 0.0).any():
        raise RuntimeError("optimizer allowance scale vector is invalid")
    contract = optimizer_allowance_contract(
        names,
        resonance_allowance_hz=resonance_allowance_hz,
        llt_allowance_uh=llt_allowance_uh,
    )
    allowance_vector = np.asarray([
        float(contract["allowances"][name]) for name in names
    ], dtype=float)
    normalized_evaluate = problem._evaluate

    def allowance_evaluate(X, out, *args, **kwargs):
        normalized: dict[str, Any] = {}
        normalized_evaluate(X, normalized, *args, **kwargs)
        if "G" not in normalized:
            raise RuntimeError("normalized optimizer evaluation omitted G")
        normalized_g = np.asarray(normalized["G"], dtype=float)
        if normalized_g.ndim != 2 or normalized_g.shape[1] != len(names):
            raise RuntimeError("optimizer allowance constraint width drifted")
        out.update(normalized)
        out["G"] = normalized_g - allowance_vector / scale_vector

    problem._evaluate = allowance_evaluate
    problem._optimizer_constraint_allowance_contract = contract
    return contract
