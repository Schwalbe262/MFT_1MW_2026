"""Physics-consistent first-order resonance estimates for surrogate outputs.

The Maxwell capacitance stage uses the full-physical Tx self inductance, Rx
self inductance, and primary-referred leakage inductance for its three LC
estimates.  The surrogate predicts only ``Llt_phys`` and ``k``.  Maxwell's
matrix contract defines ``Llt = Ltx * (1 - k**2)``, so Tx self inductance can
be recovered exactly from those predictions.  Rx self inductance is referred
through the physical turns ratio because it is not otherwise identifiable
from ``Llt_phys`` and ``k`` alone.

``Llt_phys`` is expressed in microhenries; capacitance inputs are expressed in
farads; the returned frequencies are expressed in hertz.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


RESONANCE_OUTPUTS = (
    "f_res_tx_self_Hz",
    "f_res_rx_self_Hz",
    "f_res_interwinding_Hz",
)
DEFAULT_PERCENTILES = (90.0, 95.0, 99.0)
DEFAULT_MEDIAN_RELATIVE_ERROR_THRESHOLD = 0.05


def _finite_number(value, name: str) -> float:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_number(value, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _turns(params: Mapping, total: str, main: str, side: str) -> float:
    if main in params and side in params:
        main_turns = _finite_number(params[main], main)
        side_turns = _finite_number(params[side], side)
        if main_turns < 0.0 or side_turns < 0.0:
            raise ValueError(f"{main} and {side} must be non-negative")
        turns = main_turns + side_turns
        if turns <= 0.0:
            raise ValueError(f"{main} + {side} must be positive")
        return turns
    if total in params:
        return _positive_number(params[total], total)
    raise KeyError(f"params must provide {main}/{side} or {total}")


def _lc_resonance_hz(inductance_h: float, capacitance_f: float) -> float:
    inductance_h = _positive_number(inductance_h, "inductance_h")
    capacitance_f = _positive_number(capacitance_f, "capacitance_f")
    frequency = 1.0 / (2.0 * math.pi * math.sqrt(inductance_h * capacitance_f))
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("LC resonance frequency is not positive and finite")
    return frequency


def derive_resonances(predictions: dict, params: dict) -> dict:
    """Derive the three solver-compatible LC resonance estimates.

    ``predictions`` must contain ``Llt_phys`` (microhenries), ``k``, and the
    three full-physical capacitances in farads.  ``params`` supplies either the
    four split turn counts (preferred) or the derived totals ``N1`` and ``N2``.

    The Rx self-inductance relation uses the turns-ratio approximation
    ``Lrx = Ltx * (N2 / N1)**2``.  The exact solver ``Lrx`` cannot be recovered
    from only primary leakage and coupling coefficient.
    """
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    leakage_uh = _positive_number(predictions.get("Llt_phys"), "Llt_phys")
    coupling = _finite_number(predictions.get("k"), "k")
    if coupling < 0.0 or coupling >= 1.0:
        raise ValueError("k must satisfy 0 <= k < 1")

    c_tx_tx_f = _positive_number(predictions.get("C_tx_tx_F"), "C_tx_tx_F")
    c_rx_rx_f = _positive_number(predictions.get("C_rx_rx_F"), "C_rx_rx_F")
    c_tx_rx_f = _positive_number(predictions.get("C_tx_rx_F"), "C_tx_rx_F")

    primary_turns = _turns(params, "N1", "N1_main", "N1_side")
    secondary_turns = _turns(params, "N2", "N2_main", "N2_side")
    turns_ratio = secondary_turns / primary_turns

    leakage_h = leakage_uh * 1e-6
    tx_self_h = leakage_h / (1.0 - coupling * coupling)
    rx_self_h = tx_self_h * turns_ratio * turns_ratio

    return {
        "f_res_tx_self_Hz": _lc_resonance_hz(tx_self_h, c_tx_tx_f),
        "f_res_rx_self_Hz": _lc_resonance_hz(rx_self_h, c_rx_rx_f),
        "f_res_interwinding_Hz": _lc_resonance_hz(leakage_h, c_tx_rx_f),
    }


def _physical_leakage_uh(row: Mapping) -> float:
    value = row.get("Llt_phys")
    try:
        return _positive_number(value, "Llt_phys")
    except ValueError:
        pass

    leakage_raw_uh = _positive_number(row.get("Llt"), "Llt")
    full_model = _finite_number(row.get("full_model"), "full_model")
    if full_model not in (0.0, 1.0):
        raise ValueError("full_model must be 0 or 1")
    return leakage_raw_uh * (2.0 if full_model == 0.0 else 1.0)


def _percentile_key(percentile: float) -> str:
    token = f"{percentile:g}".replace(".", "p")
    return f"p{token}_relative_error"


def _error_summary(errors: Iterable[float], percentiles: tuple[float, ...]) -> dict:
    values = np.asarray(tuple(errors), dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarize an empty relative-error sample")
    median = float(np.median(values))
    summary = {
        "count": int(values.size),
        "median_relative_error": median,
        "median_relative_error_pct": median * 100.0,
    }
    for percentile in percentiles:
        value = float(np.percentile(values, percentile))
        key = _percentile_key(percentile)
        summary[key] = value
        summary[f"{key}_pct"] = value * 100.0
    maximum = float(np.max(values))
    summary["max_relative_error"] = maximum
    summary["max_relative_error_pct"] = maximum * 100.0
    return summary


def validate_resonances(
    rows,
    *,
    median_relative_error_threshold: float = (
        DEFAULT_MEDIAN_RELATIVE_ERROR_THRESHOLD
    ),
    percentiles: Iterable[float] = DEFAULT_PERCENTILES,
    assert_threshold: bool = False,
) -> dict:
    """Compare derived and solver resonance fields for dataset rows.

    Rows with ``cap_on != 1`` (when that column exists), missing fields, or
    non-physical values are skipped.  The threshold applies independently to
    every output; ``passed`` is false if any output median is not strictly
    below it.  Set ``assert_threshold`` to raise ``AssertionError`` on failure.
    """
    threshold = _positive_number(
        median_relative_error_threshold,
        "median_relative_error_threshold",
    )
    normalized_percentiles = tuple(
        _finite_number(value, "percentile") for value in percentiles
    )
    if any(value < 0.0 or value > 100.0 for value in normalized_percentiles):
        raise ValueError("percentiles must be between 0 and 100")

    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif isinstance(rows, Mapping):
        frame = pd.DataFrame([dict(rows)])
    else:
        frame = pd.DataFrame(rows)

    errors = {name: [] for name in RESONANCE_OUTPUTS}
    eligible_rows = 0
    valid_rows = 0
    has_cap_on = "cap_on" in frame.columns
    for row in frame.to_dict(orient="records"):
        if has_cap_on:
            try:
                if _finite_number(row.get("cap_on"), "cap_on") != 1.0:
                    continue
            except ValueError:
                continue
        eligible_rows += 1

        try:
            predictions = {
                "Llt_phys": _physical_leakage_uh(row),
                "k": row.get("k"),
                "C_tx_tx_F": row.get("C_tx_tx_F"),
                "C_rx_rx_F": row.get("C_rx_rx_F"),
                "C_tx_rx_F": row.get("C_tx_rx_F"),
            }
            derived = derive_resonances(predictions, row)
            solver = {
                name: _positive_number(row.get(name), name)
                for name in RESONANCE_OUTPUTS
            }
        except (KeyError, TypeError, ValueError):
            continue

        valid_rows += 1
        for name in RESONANCE_OUTPUTS:
            errors[name].append(abs(derived[name] - solver[name]) / solver[name])

    if valid_rows == 0:
        raise ValueError("no rows contain complete positive capacitance resonance data")

    per_output = {
        name: _error_summary(values, normalized_percentiles)
        for name, values in errors.items()
    }
    combined = _error_summary(
        (value for values in errors.values() for value in values),
        normalized_percentiles,
    )
    failed_outputs = [
        name
        for name, summary in per_output.items()
        if summary["median_relative_error"] >= threshold
    ]
    report = {
        "rows_total": int(len(frame)),
        "rows_eligible": int(eligible_rows),
        "rows_valid": int(valid_rows),
        "rows_skipped": int(len(frame) - valid_rows),
        "median_relative_error_threshold": threshold,
        "median_relative_error_threshold_pct": threshold * 100.0,
        "passed": not failed_outputs,
        "failed_outputs": failed_outputs,
        "outputs": per_output,
        "combined": combined,
    }
    if assert_threshold and failed_outputs:
        details = ", ".join(
            f"{name}={per_output[name]['median_relative_error_pct']:.6g}%"
            for name in failed_outputs
        )
        raise AssertionError(
            "resonance median relative-error threshold failed: "
            f"{details}; required < {threshold * 100.0:.6g}%"
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate derived surrogate resonances against solver fields."
    )
    parser.add_argument("dataset", type=Path, help="campaign parquet to read")
    parser.add_argument(
        "--median-threshold-pct",
        type=float,
        default=DEFAULT_MEDIAN_RELATIVE_ERROR_THRESHOLD * 100.0,
        help="strict per-output median relative-error threshold (default: 5)",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="report a failed threshold without returning an error",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    frame = pd.read_parquet(args.dataset)
    report = validate_resonances(
        frame,
        median_relative_error_threshold=args.median_threshold_pct / 100.0,
        assert_threshold=not args.no_assert,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
