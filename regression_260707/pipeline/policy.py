"""Versioned scheduling policy for continuous model/optimization generations."""

from __future__ import annotations

from dataclasses import dataclass
import math


MIN_MODEL_ACTIVATION_ROWS = 3000
FIRST_TUNING_ROWS = 4000
TUNING_MIN_GROWTH_ROWS = 2000
TUNING_MIN_GROWTH_FRACTION = 0.20
STANDARD_VERIFICATION_COUNT = 33
FINE_VERIFICATION_COUNT = 3
NSGA_RESTARTS = 16
NSGA_POPULATION = 200
NSGA_MAX_WORKERS = 4


def checkpoint_sequence(strict_full_rows: int) -> list[int]:
    count = max(0, int(strict_full_rows))
    fixed = [500, 1000, 2000, 3000]
    if count >= 4000:
        fixed.extend(range(4000, count + 1, 1000))
    return [value for value in fixed if value <= count]


def next_training_checkpoint(
    strict_full_rows: int, active_strict_full_rows: int = 0
) -> int | None:
    active = max(0, int(active_strict_full_rows))
    due = [
        value for value in checkpoint_sequence(strict_full_rows)
        if value > active
    ]
    return due[0] if due else None


@dataclass(frozen=True)
class TuningDecision:
    due: bool
    reason: str
    threshold_rows: int


def tuning_decision(
    strict_full_rows: int,
    *,
    last_tuned_rows: int | None = None,
    drift_detected: bool = False,
    quality_regression: bool = False,
) -> TuningDecision:
    """Apply the production Optuna cadence.

    Tuning starts only at 4k strict-full rows.  Thereafter it runs when the
    cohort grew by at least ``max(2k, 20%)`` or an explicit drift/quality
    signal is present.  A signal never bypasses the 4k evidence floor.
    """
    rows = max(0, int(strict_full_rows))
    if rows < FIRST_TUNING_ROWS:
        return TuningDecision(False, "below_first_tuning_gate", FIRST_TUNING_ROWS)
    if last_tuned_rows is None:
        return TuningDecision(True, "first_tuning_generation", FIRST_TUNING_ROWS)
    last = max(0, int(last_tuned_rows))
    growth = max(
        TUNING_MIN_GROWTH_ROWS,
        int(math.ceil(last * TUNING_MIN_GROWTH_FRACTION)),
    )
    threshold = last + growth
    if drift_detected:
        return TuningDecision(True, "dataset_drift", threshold)
    if quality_regression:
        return TuningDecision(True, "quality_regression", threshold)
    if rows >= threshold:
        return TuningDecision(True, "cohort_growth", threshold)
    return TuningDecision(False, "insufficient_cohort_growth", threshold)


def bounded_nsga_workers(requested: int, restarts: int = NSGA_RESTARTS) -> int:
    value = int(requested)
    if value < 1:
        raise ValueError("NSGA worker count must be positive")
    return min(value, NSGA_MAX_WORKERS, max(1, int(restarts)))
