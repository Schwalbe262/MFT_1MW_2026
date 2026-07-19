"""Entrypoint for the isolated Tier-1 balanced thermal crossover sidecar."""

from __future__ import annotations

import os
from pathlib import Path
import sys


VARIANT = "thermal-crossover-balanced4c-v1"
_existing = os.environ.get("MFT_TIER1_SEARCH_VARIANT")
if _existing not in {None, VARIANT}:
    raise RuntimeError(
        "MFT_TIER1_SEARCH_VARIANT conflicts with balanced thermal-crossover "
        f"entrypoint: {_existing}"
    )
os.environ["MFT_TIER1_SEARCH_VARIANT"] = VARIANT

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_slurm_rolling as rolling  # noqa: E402


if (
    rolling.SEARCH_VARIANT != VARIANT
    or rolling.SEARCH_NAMESPACE
    != "thermal-crossover-balanced4c-res250-p320-g600-warm64-v1"
    or rolling.OPTIMIZER_RESONANCE_SCALE_HZ != 250.0
    or rolling.OPTIMIZER_CORE_THERMAL_SCALE_C != 4.0
    or rolling.SEED_START != 1_907_194_000
    or rolling.ROLLING_TARGET != 32
    or rolling.POPULATION != 320
    or rolling.MAX_GENERATIONS != 600
    or rolling.INFERENCE_THREADS != 8
    or rolling.TASK_PRIORITY != -5
    or rolling.LEGACY_TASK_PRIORITY != -6
    or rolling.PRIORITY_MIGRATION_SEED_CUTOFF != 1_907_194_000
    or rolling.TASK_NAME_STEM != "mft-t1therm4"
    or rolling.DEDUPE_NAMESPACE
    != "mft-tier1-thermal-crossover-balanced4c-nsga"
    or not str(rolling.RUNTIME).endswith(
        r"mft_tier1_nsga_thermal_balanced4c_t110_res15k_260719"
    )
):
    raise RuntimeError(
        "cached tier1_slurm_rolling module has a conflicting search identity"
    )


def main() -> int:
    return rolling.main()


if __name__ == "__main__":
    raise SystemExit(main())
