"""Compare accuracy and wall-clock results from an MFT efficiency A/B run.

The comparator intentionally has no pandas dependency so it can run in the
lightweight controller environment as well as on an AEDT worker node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_THRESHOLDS = {
    "electromagnetic_relative_pct": 0.5,
    "loss_relative_pct": 2.0,
    "b_relative_pct": 2.0,
    "temperature_absolute_c": 2.0,
}

_EXACT_ELECTROMAGNETIC_TARGETS = {"Llt", "k"}
_TIME_FIELDS = (
    "ab_process_wall_s",
    "stage_time_pre_result_s",
    "time",
)


@dataclass(frozen=True)
class TargetDelta:
    target: str
    family: str
    baseline: float | None
    variant: float | None
    delta: float | None
    relative_pct: float | None
    threshold: float
    threshold_kind: str
    passed: bool
    reason: str


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_result(path: str | Path, row: int = -1) -> dict[str, Any]:
    """Load one result record from JSON or select a row from CSV.

    JSON may contain one object or a one-element/list-of-records payload. CSV
    defaults to its final row because campaign result CSVs are append-only.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"result file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload:
            try:
                selected = payload[row]
            except IndexError as exc:
                raise ValueError(
                    f"JSON row {row} is outside {len(payload)} records in {source}"
                ) from exc
            if isinstance(selected, dict):
                return selected
        raise ValueError(f"JSON result must contain an object or records: {source}")
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError(f"CSV result has no data rows: {source}")
        try:
            return dict(records[row])
        except IndexError as exc:
            raise ValueError(
                f"CSV row {row} is outside {len(records)} records in {source}"
            ) from exc
    raise ValueError(f"result must be .json or .csv: {source}")


def target_family(name: str) -> str | None:
    if name in _EXACT_ELECTROMAGNETIC_TARGETS:
        return "Llt/k"
    if name.startswith("Tprobe_"):
        return "temperature"
    if name.startswith("P_") and name != "P_target":
        return "loss"
    if (name.startswith("B_") or name.startswith("Bavg_")) and not name.endswith(
        ("_rel_error", "_tolerance_rel", "_attested")
    ):
        return "B"
    return None


def discover_targets(
    baseline: Mapping[str, Any], variant: Mapping[str, Any]
) -> list[tuple[str, str]]:
    targets = []
    for name in sorted(set(baseline) | set(variant) | _EXACT_ELECTROMAGNETIC_TARGETS):
        family = target_family(name)
        if family is not None:
            targets.append((name, family))
    return targets


def _relative_pct(baseline: float, variant: float) -> float:
    # The baseline is the reference solution; using the larger arm as the
    # denominator would make an increase slightly easier to pass than a drop.
    scale = max(abs(baseline), 1e-12)
    return 100.0 * abs(variant - baseline) / scale


def compare_records(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"unknown thresholds: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in thresholds.items()})
    if any(not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("thresholds must be finite and non-negative")

    deltas = []
    for target, family in discover_targets(baseline, variant):
        base_value = _finite_float(baseline.get(target))
        variant_value = _finite_float(variant.get(target))
        if family == "temperature":
            threshold = limits["temperature_absolute_c"]
            threshold_kind = "absolute_C"
        elif family == "Llt/k":
            threshold = limits["electromagnetic_relative_pct"]
            threshold_kind = "relative_pct"
        elif family == "loss":
            threshold = limits["loss_relative_pct"]
            threshold_kind = "relative_pct"
        else:
            threshold = limits["b_relative_pct"]
            threshold_kind = "relative_pct"

        if (base_value is None and variant_value is None
                and family == "Llt/k"):
            delta = TargetDelta(
                target, family, None, None, None, None, threshold,
                threshold_kind, False, "missing_or_non_finite",
            )
        elif base_value is None and variant_value is None:
            # Optional targets such as an absent side winding can be NaN in
            # both arms. They carry no A/B evidence and are reported as skip.
            delta = TargetDelta(
                target, family, None, None, None, None, threshold,
                threshold_kind, True, "both_non_finite_skipped",
            )
        elif base_value is None or variant_value is None:
            delta = TargetDelta(
                target, family, base_value, variant_value, None, None,
                threshold, threshold_kind, False, "missing_or_non_finite",
            )
        else:
            signed_delta = variant_value - base_value
            relative_pct = _relative_pct(base_value, variant_value)
            observed = (
                abs(signed_delta)
                if threshold_kind == "absolute_C"
                else relative_pct
            )
            delta = TargetDelta(
                target, family, base_value, variant_value, signed_delta,
                relative_pct, threshold, threshold_kind,
                observed <= threshold, "within_threshold" if observed <= threshold
                else "threshold_exceeded",
            )
        deltas.append(delta)

    time_summary = compare_time(baseline, variant)
    evaluated = [item for item in deltas if item.reason != "both_non_finite_skipped"]
    loss_expected = (
        (_finite_float(baseline.get("loss_on")) or 0) != 0
        or (_finite_float(variant.get("loss_on")) or 0) != 0
        or any(item.family == "loss" for item in evaluated)
    )
    thermal_expected = (
        (_finite_float(baseline.get("thermal_on")) or 0) != 0
        or (_finite_float(variant.get("thermal_on")) or 0) != 0
        or any(item.family == "temperature" for item in evaluated)
    )
    expected_families = {"Llt/k"}
    if loss_expected:
        expected_families.update({"loss", "B"})
    if thermal_expected:
        expected_families.add("temperature")

    family_counts = {}
    for family in ("Llt/k", "loss", "B", "temperature"):
        selected = [item for item in evaluated if item.family == family]
        family_counts[family] = {
            "evaluated": len(selected),
            "failed": sum(not item.passed for item in selected),
            "required": family in expected_families,
            "passed": (
                bool(selected) and all(item.passed for item in selected)
                if family in expected_families else all(item.passed for item in selected)
            ),
        }

    family_gate_passed = all(
        details["passed"] for details in family_counts.values()
    )
    em_expected = (
        (_finite_float(baseline.get("matrix_on")) or 0) != 0
        or (_finite_float(variant.get("matrix_on")) or 0) != 0
        or loss_expected
    )
    required_quality_fields = []
    if em_expected:
        required_quality_fields.append("result_valid_em")
    if thermal_expected:
        required_quality_fields.extend([
            "result_valid_thermal",
            "thermal_converged",
            "thermal_extraction_complete",
            "thermal_rx_power_balance_ok",
        ])

    quality_checks = []
    for arm, record in (("baseline", baseline), ("variant", variant)):
        for field in required_quality_fields:
            value = _finite_float(record.get(field))
            quality_checks.append({
                "arm": arm,
                "field": field,
                "value": value,
                "expected": 1.0,
                "passed": value == 1.0,
            })
        if "ab_return_code" in record:
            value = _finite_float(record.get("ab_return_code"))
            quality_checks.append({
                "arm": arm,
                "field": "ab_return_code",
                "value": value,
                "expected": 0.0,
                "passed": value == 0.0,
            })
    baseline_identity = str(
        baseline.get("ab_experiment_sha256") or ""
    ).strip()
    variant_identity = str(
        variant.get("ab_experiment_sha256") or ""
    ).strip()
    if baseline_identity or variant_identity:
        quality_checks.append({
            "arm": "pair",
            "field": "ab_experiment_sha256",
            "value": variant_identity or None,
            "expected": baseline_identity or None,
            "passed": bool(
                baseline_identity
                and variant_identity
                and baseline_identity == variant_identity
            ),
        })
    if "ab_arm" in baseline or "ab_arm" in variant:
        for expected_arm, record in (
            ("baseline", baseline), ("variant", variant)
        ):
            observed_arm = str(record.get("ab_arm") or "").strip()
            quality_checks.append({
                "arm": expected_arm,
                "field": "ab_arm",
                "value": observed_arm or None,
                "expected": expected_arm,
                "passed": observed_arm == expected_arm,
            })
    quality_gate_passed = bool(quality_checks) and all(
        item["passed"] for item in quality_checks
    )
    return {
        "schema": "mft-efficiency-ab-comparison-v1",
        "passed": (
            bool(evaluated)
            and all(item.passed for item in evaluated)
            and family_gate_passed
            and quality_gate_passed
        ),
        "thresholds": limits,
        "time": time_summary,
        "family_counts": family_counts,
        "quality_checks": quality_checks,
        "targets": [asdict(item) for item in deltas],
    }


def _record_time(record: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for field in _TIME_FIELDS:
        value = _finite_float(record.get(field))
        if value is not None and value >= 0:
            return value, field
    stage_fields = (
        ("matrix_on", "time_matrix"),
        ("loss_on", "time_loss"),
        ("thermal_on", "time_thermal"),
    )
    enabled_fields = [
        time_field for flag, time_field in stage_fields
        if (_finite_float(record.get(flag)) or 0) != 0
    ]
    selected_fields = enabled_fields or [
        time_field for _, time_field in stage_fields
        if _finite_float(record.get(time_field)) is not None
    ]
    components = [
        _finite_float(record.get(field)) for field in selected_fields
    ]
    if selected_fields and all(
        value is not None and value >= 0 for value in components
    ):
        return sum(components), f"sum({','.join(selected_fields)})"
    return None, None


def compare_time(
    baseline: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    base_seconds, base_source = _record_time(baseline)
    variant_seconds, variant_source = _record_time(variant)
    if base_seconds is None or variant_seconds is None:
        return {
            "baseline_seconds": base_seconds,
            "variant_seconds": variant_seconds,
            "saved_seconds": None,
            "saved_pct": None,
            "baseline_source": base_source,
            "variant_source": variant_source,
            "comparable": False,
        }
    if base_source != variant_source:
        return {
            "baseline_seconds": base_seconds,
            "variant_seconds": variant_seconds,
            "saved_seconds": None,
            "saved_pct": None,
            "baseline_source": base_source,
            "variant_source": variant_source,
            "comparable": False,
        }
    saved = base_seconds - variant_seconds
    return {
        "baseline_seconds": base_seconds,
        "variant_seconds": variant_seconds,
        "saved_seconds": saved,
        "saved_pct": 100.0 * saved / base_seconds if base_seconds > 0 else None,
        "baseline_source": base_source,
        "variant_source": variant_source,
        "comparable": True,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.{digits}g}"


def _display(value: Any) -> str:
    number = _finite_float(value)
    if number is not None:
        return f"{number:.6g}"
    return "n/a" if value is None else str(value)


def render_markdown(comparison: Mapping[str, Any]) -> str:
    status = "PASS" if comparison["passed"] else "FAIL"
    timing = comparison["time"]
    missing_families = [
        family for family, details in comparison["family_counts"].items()
        if details["required"] and not details["evaluated"]
    ]
    lines = [
        f"Efficiency A/B accuracy gate: **{status}**",
        "",
    ]
    if missing_families:
        lines.extend([
            "Missing required finite target families: "
            + ", ".join(missing_families),
            "",
        ])
    lines.extend([
        "| baseline s | variant s | saved s | saved % | timing source |",
        "|---:|---:|---:|---:|---|",
        "| "
        + " | ".join(
            [
                _fmt(timing["baseline_seconds"], 7),
                _fmt(timing["variant_seconds"], 7),
                _fmt(timing["saved_seconds"], 7),
                _fmt(timing["saved_pct"], 5),
                f"{timing['baseline_source']} / {timing['variant_source']}",
            ]
        )
        + " |",
        "",
        "| arm | validity field | observed | expected | status |",
        "|---|---|---:|---:|---|",
    ])
    for item in comparison["quality_checks"]:
        lines.append(
            "| "
            + " | ".join([
                str(item["arm"]),
                str(item["field"]),
                _display(item["value"]),
                _display(item["expected"]),
                "PASS" if item["passed"] else "FAIL",
            ])
            + " |"
        )
    lines.extend([
        "",
        "| target | family | baseline | variant | delta | relative % | limit | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for item in comparison["targets"]:
        limit_suffix = " C" if item["threshold_kind"] == "absolute_C" else "%"
        item_status = (
            "SKIP" if item["reason"] == "both_non_finite_skipped"
            else "PASS" if item["passed"] else "FAIL"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["target"]),
                    str(item["family"]),
                    _fmt(item["baseline"]),
                    _fmt(item["variant"]),
                    _fmt(item["delta"]),
                    _fmt(item["relative_pct"], 5),
                    f"{_fmt(item['threshold'], 5)}{limit_suffix}",
                    item_status,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="baseline result JSON or CSV")
    parser.add_argument("variant", help="variant result JSON or CSV")
    parser.add_argument("--baseline-row", type=int, default=-1)
    parser.add_argument("--variant-row", type=int, default=-1)
    parser.add_argument("--em-relative-pct", type=float, default=0.5)
    parser.add_argument("--loss-relative-pct", type=float, default=2.0)
    parser.add_argument("--b-relative-pct", type=float, default=2.0)
    parser.add_argument("--temperature-absolute-c", type=float, default=2.0)
    parser.add_argument("--json-output", help="also write the comparison as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline = load_result(args.baseline, args.baseline_row)
    variant = load_result(args.variant, args.variant_row)
    comparison = compare_records(
        baseline,
        variant,
        {
            "electromagnetic_relative_pct": args.em_relative_pct,
            "loss_relative_pct": args.loss_relative_pct,
            "b_relative_pct": args.b_relative_pct,
            "temperature_absolute_c": args.temperature_absolute_c,
        },
    )
    print(render_markdown(comparison))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
