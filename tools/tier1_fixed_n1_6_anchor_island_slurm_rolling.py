"""Fail-closed entrypoint for one fixed-N1=6 anchor-island controller.

The exact island is mandatory in ``MFT_TIER1_ANCHOR_ISLAND_VARIANT``.  One
immutable source file can therefore serve all six independently quota-limited
controllers without silently falling back to a production/default variant.
"""

from __future__ import annotations

import os

try:
    from tier1_n1_6_anchor_island_contract import BY_VARIANT
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_n1_6_anchor_island_contract import BY_VARIANT


VARIANT = os.environ.get("MFT_TIER1_ANCHOR_ISLAND_VARIANT")
if VARIANT not in BY_VARIANT:
    raise RuntimeError(
        "MFT_TIER1_ANCHOR_ISLAND_VARIANT must name one sealed anchor island"
    )
prior = os.environ.get("MFT_TIER1_SEARCH_VARIANT")
if prior is not None and prior != VARIANT:
    raise RuntimeError("anchor-island and rolling search variants disagree")
os.environ["MFT_TIER1_SEARCH_VARIANT"] = VARIANT

try:
    import tier1_slurm_rolling as rolling
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_slurm_rolling as rolling


SEALED = BY_VARIANT[VARIANT]
if (
    rolling.SEARCH_VARIANT != SEALED.variant
    or rolling.ANCHOR_ISLAND != SEALED
    or rolling.ROLLING_TARGET != SEALED.active_quota
    or rolling.OPTIMIZER_RESONANCE_ALLOWANCE_HZ
    != SEALED.resonance_allowance_hz
    or rolling.OPTIMIZER_LLT_ALLOWANCE_UH != SEALED.llt_allowance_uh
    or rolling.FIXED_PRIMARY_TURNS != 6
    or rolling.search_profile().get("physical_hard_spec_mutation") is not False
):
    raise RuntimeError("anchor-island rolling controller identity drifted")


if __name__ == "__main__":
    rolling.main()
