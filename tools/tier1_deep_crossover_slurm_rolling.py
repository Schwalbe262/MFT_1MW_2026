"""Authenticated entrypoint for one sealed deep-crossover Tier-1 island."""

from __future__ import annotations

import os


variant = os.environ.get("MFT_TIER1_DEEP_CROSSOVER_VARIANT")
if not variant:
    raise RuntimeError("MFT_TIER1_DEEP_CROSSOVER_VARIANT is required")
configured = os.environ.get("MFT_TIER1_SEARCH_VARIANT")
if configured is not None and configured != variant:
    raise RuntimeError("Tier-1 deep crossover variant environment mismatch")
os.environ["MFT_TIER1_SEARCH_VARIANT"] = variant

try:
    from tier1_deep_crossover_contract import BY_VARIANT
    import tier1_slurm_rolling as rolling
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_deep_crossover_contract import BY_VARIANT
    from tools import tier1_slurm_rolling as rolling


if variant not in BY_VARIANT:
    raise RuntimeError(f"unsupported deep crossover variant: {variant}")
island = BY_VARIANT[variant]
if (
    rolling.SEARCH_VARIANT != variant
    or rolling.DEEP_CROSSOVER_ISLAND != island
    or rolling.SEED_START != island.seed_start
    or rolling.ROLLING_TARGET != island.active_quota
    or rolling.FIXED_PRIMARY_TURNS != island.fixed_primary_turns
    or rolling.OPTIMIZER_LLT_SCALE_UH != island.optimizer_llt_scale_uh
    or rolling.OPTIMIZER_ALL_THERMAL_SCALE_C
    != island.optimizer_all_thermal_scale_c
    or rolling.OPTIMIZER_RESONANCE_ALLOWANCE_HZ
    != island.optimizer_resonance_allowance_hz
    or rolling.OPTIMIZER_LLT_ALLOWANCE_UH
    != island.optimizer_llt_allowance_uh
    or rolling.OPTIMIZER_TERMINATION_STRATEGY
    != "fixed-n-gen-no-ftol-v1"
    or rolling.MAX_GENERATIONS != 200
):
    raise RuntimeError("deep crossover rolling contract drift")


main = rolling.main


if __name__ == "__main__":
    main()
