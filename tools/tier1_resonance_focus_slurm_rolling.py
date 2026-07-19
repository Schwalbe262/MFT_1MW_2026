"""Entrypoint for the isolated 250 Hz Tier-1 resonance-focus sidecar.

The shared rolling implementation is selected through a fail-closed variant
environment contract.  The supervisor inherits that contract when it starts
workers, so a worker launched under the normalized sidecar identity cannot
silently operate this runtime (or vice versa).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


VARIANT = "resonance-focus-250hz-v1"
_existing = os.environ.get("MFT_TIER1_SEARCH_VARIANT")
if _existing not in {None, VARIANT}:
    raise RuntimeError(
        "MFT_TIER1_SEARCH_VARIANT conflicts with resonance-focus entrypoint: "
        f"{_existing}"
    )
os.environ["MFT_TIER1_SEARCH_VARIANT"] = VARIANT

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_slurm_rolling as rolling  # noqa: E402

if (
    rolling.SEARCH_VARIANT != VARIANT
    or rolling.SEARCH_NAMESPACE
    != "resonance-focus250-p320-g600-warm64-v1"
    or rolling.OPTIMIZER_RESONANCE_SCALE_HZ != 250.0
    or rolling.SEED_START != 1_907_192_000
    or rolling.TASK_PRIORITY != -7
    or not str(rolling.RUNTIME).endswith(
        r"t1r250_260719"
    )
):
    raise RuntimeError(
        "cached tier1_slurm_rolling module has a conflicting search identity"
    )


def main() -> int:
    return rolling.main()


if __name__ == "__main__":
    raise SystemExit(main())
