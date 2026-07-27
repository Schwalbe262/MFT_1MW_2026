"""Seal the B/core-loss alignment audit used by the compact core rescue.

The audit deliberately separates:

* deterministic design equations from same-row FEA labels;
* disjoint seed-42/43 evaluation predictions from in-sample comparisons; and
* the broad 6/60 cohort from the unsupported compact fixed-20T/5T slice.

It is an evidence generator, not an acceptance gate.  In particular, the
pointwise ``B_max_core`` field is retained only as a diagnostic because its
source contract identifies edge and segment-interface spikes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCHEMA = "mft-goal-core-physics-alignment-audit-v1"
ANCHOR_SHA256 = (
    "86778c97c5d6e48fb9e76f53a75cf1e4e59d163a3600ce8a59d4c151d0695be3"
)
ANCHOR_B_T = 1.4617010926508007
ANCHOR_AE_EFFECTIVE_M2 = 0.028505600000000002
ANCHOR_PRIMARY_TURNS = 6
TARGET_B_NOMINAL_T = 0.8
TARGET_B_BAND_T = (0.75, 0.85)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["payload_sha256"] = _sha256_bytes(_canonical_bytes(value))
    return sealed


def _metrics(prediction: Iterable[float], reference: Iterable[float]) -> dict:
    predicted = np.asarray(prediction, dtype=float)
    truth = np.asarray(reference, dtype=float)
    mask = np.isfinite(predicted) & np.isfinite(truth)
    predicted = predicted[mask]
    truth = truth[mask]
    if not len(predicted):
        return {"n": 0}
    error = predicted - truth
    nonzero = np.abs(truth) > 1e-12
    ape = np.abs(error[nonzero] / truth[nonzero])
    return {
        "n": int(len(predicted)),
        "bias": float(np.mean(error)),
        "median_bias": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.9)),
        "bias_percent_of_reference_mean": float(
            np.mean(error) / np.mean(truth) * 100.0
        ),
        "mape_percent": float(np.mean(ape) * 100.0),
        "p90_ape_percent": float(np.quantile(ape, 0.9) * 100.0),
        "pearson_correlation": float(np.corrcoef(predicted, truth)[0, 1]),
    }


def _summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not len(numeric):
        return {"n": 0}
    return {
        "n": int(len(numeric)),
        "minimum": float(numeric.min()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.9)),
        "maximum": float(numeric.max()),
    }


def _add_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n2_side = pd.to_numeric(
        result["N2_side"], errors="coerce"
    ).fillna(-1)
    core_x = 4.0 * result["l1"] + 2.0 * result["l2"]
    center_x = result["sl1_main_x"] + 2.0 * result["nwl1_main"]
    center_y = result["sl1_main_y"] + 2.0 * result["nwb1_main_y"]
    side_x = 2.0 * (
        result["l1"]
        + result["l2"]
        + result["l1"] / 2.0
        + result["sl2_side_x"] / 2.0
        + result["nwl2_side"]
    )
    side_y = 2.0 * (
        result["sl2_side_y"] / 2.0 + result["nwl2_side"]
    )
    result["audit_W_mm"] = np.maximum.reduce(
        [
            core_x.to_numpy(),
            center_x.to_numpy(),
            np.where(n2_side > 0.0, side_x, -np.inf),
        ]
    )
    result["audit_L_mm"] = np.maximum.reduce(
        [
            result["w1"].to_numpy(),
            center_y.to_numpy(),
            np.where(n2_side > 0.0, side_y, -np.inf),
        ]
    )
    result["audit_H_mm"] = result["h1"] + 2.0 * result["l1"]
    return result


def _evaluation_indices(count: int) -> np.ndarray:
    indices = np.arange(count)
    _, holdout = train_test_split(
        indices, test_size=0.20, random_state=42
    )
    _, evaluation = train_test_split(
        holdout, test_size=0.50, random_state=43
    )
    return np.asarray(evaluation, dtype=int)


def _load_bundle_prediction(
    generation: Path,
    target: str,
    frame: pd.DataFrame,
    evaluation: np.ndarray,
    regression_root: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    training_root = regression_root / "training"
    for path in (str(training_root), str(regression_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # The deployed module intentionally supports flat imports.
    from train_models import _ensemble_prediction  # noqa: PLC0415

    model_path = generation / target / "models.pkl"
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)  # noqa: S301 - authenticated local model
    features = list(bundle["features"])
    x = frame[features].fillna(0.0).reset_index(drop=True)
    mean, sigma = _ensemble_prediction(
        bundle["models"],
        x.iloc[evaluation],
        bundle["transform"],
    )
    return bundle, mean, sigma, _sha256_file(model_path)


def _temperature_group(frame: pd.DataFrame, mask: pd.Series) -> dict:
    subset = frame.loc[mask]
    fields = (
        "B_design_square_material_analytic",
        "T_max_core",
        "Tprobe_core_center_max",
        "Tprobe_core_center_leg_max",
        "Tprobe_core_side_leg_max",
        "Tprobe_core_top_yoke_max",
        "P_core_total",
        "Ae_effective_m2",
        "l1",
        "w1",
        "n_core_group",
        "core_depth_each",
    )
    return {
        "n": int(len(subset)),
        "statistics": {
            field: _summary(subset[field]) for field in fields
        },
    }


def build_audit(
    dataset: Path,
    generation: Path,
    regression_root: Path,
) -> dict[str, Any]:
    model_meta = json.loads(
        (generation / "P_core_total" / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    features = list(model_meta["features"])
    audit_columns = [
        "N1_main",
        "N1_side",
        "N2_main",
        "N2_side",
        "l1",
        "l2",
        "h1",
        "w1",
        "n_core_group",
        "core_depth_each",
        "core_plate_t",
        "wcp_t",
        "cw1",
        "gap1",
        "cw2",
        "gap2",
        "nwh2",
        "Ae_effective_m2",
        "sl1_main_x",
        "sl1_main_y",
        "nwl1_main",
        "nwb1_main_y",
        "sl2_side_x",
        "sl2_side_y",
        "nwl2_side",
        "fan_velocity",
        "air_temp",
        "plate_temp",
        "k_ins",
        "core_k_thermal",
        "physics_data_revision",
        "core_material_contract_version",
        "B_design_square_material_analytic",
        "B_ac_sine_material_analytic",
        "B_mean_core",
        "B_mean_core_material",
        "B_max_core",
        "B_max_core_material",
        "B_max_core_usage",
        "P_core_total",
        "P_core_total_expected_from_faraday_mass",
        "P_core_total_native_raw_W",
        "P_core_total_expected_native_raw_W",
        "core_loss_native_attested",
        "T_max_core",
        "Tprobe_core_center_max",
        "Tprobe_core_center_leg_max",
        "Tprobe_core_side_leg_max",
        "Tprobe_core_top_yoke_max",
    ]
    columns = list(dict.fromkeys([*features, *audit_columns]))
    frame = pd.read_parquet(dataset, columns=columns)
    frame = _add_dimensions(frame)
    evaluation = _evaluation_indices(len(frame))

    b_bundle, b_mean, b_sigma, b_model_sha = _load_bundle_prediction(
        generation,
        "B_mean_core",
        frame,
        evaluation,
        regression_root,
    )
    p_bundle, p_mean, p_sigma, p_model_sha = _load_bundle_prediction(
        generation,
        "P_core_total",
        frame,
        evaluation,
        regression_root,
    )
    b_truth = frame["B_mean_core_material"].to_numpy(float)[evaluation]
    b_design = frame[
        "B_design_square_material_analytic"
    ].to_numpy(float)[evaluation]
    b_sine = frame["B_ac_sine_material_analytic"].to_numpy(float)[evaluation]
    p_truth = frame["P_core_total"].to_numpy(float)[evaluation]
    p_formula = frame[
        "P_core_total_expected_from_faraday_mass"
    ].to_numpy(float)[evaluation]

    n1 = frame["N1_main"] + frame["N1_side"]
    n2 = frame["N2_main"] + frame["N2_side"]
    is_6_60 = (n1 == 6) & (n2 == 60)
    is_fixed20 = (
        is_6_60
        & np.isclose(frame["core_plate_t"], 20.0)
        & np.isclose(frame["wcp_t"], 20.0)
    )
    is_primary_5t = (
        is_fixed20
        & np.isclose(frame["cw1"], 5.0)
        & np.isclose(frame["gap1"], 1.6)
    )
    is_compact = (
        is_primary_5t
        & (frame["audit_W_mm"] <= 1200.0)
        & (frame["audit_L_mm"] <= 900.0)
        & (frame["audit_H_mm"] <= 750.0)
    )
    b_all = frame["B_design_square_material_analytic"]
    band_70_80 = (b_all >= 0.70) & (b_all <= 0.80)
    band_75_85 = (b_all >= 0.75) & (b_all <= 0.85)

    nominal_multiplier = ANCHOR_B_T / TARGET_B_NOMINAL_T
    band_multipliers = {
        str(target): ANCHOR_B_T / target
        for target in TARGET_B_BAND_T
    }
    target_ae = {
        str(target): ANCHOR_AE_EFFECTIVE_M2 * ANCHOR_B_T / target
        for target in (
            TARGET_B_BAND_T[0],
            TARGET_B_NOMINAL_T,
            TARGET_B_BAND_T[1],
        )
    }

    return _seal(
        {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "dataset": str(dataset),
                "dataset_sha256": _sha256_file(dataset),
                "dataset_rows": int(len(frame)),
                "generation": str(generation),
                "training_run_id": p_bundle["training_run_id"],
                "B_mean_core_model_sha256": b_model_sha,
                "P_core_total_model_sha256": p_model_sha,
                "evaluation_partition": {
                    "train_holdout_seed": 42,
                    "calibration_evaluation_seed": 43,
                    "row_count": int(len(evaluation)),
                    "absolute_indices_int64_sha256": _sha256_bytes(
                        evaluation.astype(np.int64).tobytes()
                    ),
                    "leakage_status": "disjoint_from_fit_and_calibration",
                },
            },
            "identity": {
                "physics_data_revision": sorted(
                    frame["physics_data_revision"].dropna().unique().tolist()
                ),
                "core_material_contract_version": sorted(
                    frame[
                        "core_material_contract_version"
                    ].dropna().unique().tolist()
                ),
                "cooling_unique_values": {
                    field: sorted(frame[field].dropna().unique().tolist())
                    for field in (
                        "fan_velocity",
                        "air_temp",
                        "plate_temp",
                        "k_ins",
                        "core_k_thermal",
                    )
                },
                "native_core_loss_attested_rows": int(
                    np.isclose(
                        frame["core_loss_native_attested"], 1.0
                    ).sum()
                ),
                "B_max_core_usage": sorted(
                    frame["B_max_core_usage"].dropna().unique().tolist()
                ),
            },
            "same_row_full_cohort": {
                "B_design_square_vs_FEA_mean_material": _metrics(
                    frame["B_design_square_material_analytic"],
                    frame["B_mean_core_material"],
                ),
                "B_sine_analytic_vs_FEA_mean_material": _metrics(
                    frame["B_ac_sine_material_analytic"],
                    frame["B_mean_core_material"],
                ),
                "P_core_formula_vs_FEA_margin_adjusted": _metrics(
                    frame["P_core_total_expected_from_faraday_mass"],
                    frame["P_core_total"],
                ),
            },
            "disjoint_evaluation": {
                "B_mean_surrogate_vs_FEA": _metrics(b_mean, b_truth),
                "B_design_square_vs_FEA_mean_material": _metrics(
                    b_design, b_truth
                ),
                "B_sine_analytic_vs_FEA_mean_material": _metrics(
                    b_sine, b_truth
                ),
                "B_mean_surrogate_interval": {
                    "bundle_metrics": b_bundle["metrics"],
                    "recomputed_coverage": float(
                        np.mean(
                            np.abs(b_mean - b_truth)
                            <= b_bundle["q90"] * b_sigma
                        )
                    ),
                },
                "P_core_surrogate_vs_FEA": _metrics(p_mean, p_truth),
                "P_core_formula_vs_FEA": _metrics(p_formula, p_truth),
                "P_core_formula_vs_surrogate_mean": _metrics(
                    p_formula, p_mean
                ),
                "P_core_surrogate_interval": {
                    "bundle_metrics": p_bundle["metrics"],
                    "recomputed_coverage": float(
                        np.mean(
                            np.abs(p_mean - p_truth)
                            <= p_bundle["q90"] * p_sigma
                        )
                    ),
                },
            },
            "temperature_evidence": {
                "B_0p70_to_0p80_all": _temperature_group(
                    frame, band_70_80
                ),
                "B_0p70_to_0p80_6_60": _temperature_group(
                    frame, band_70_80 & is_6_60
                ),
                "B_0p75_to_0p85_all": _temperature_group(
                    frame, band_75_85
                ),
                "B_0p75_to_0p85_6_60": _temperature_group(
                    frame, band_75_85 & is_6_60
                ),
                "B_0p75_to_0p85_6_60_fixed20": _temperature_group(
                    frame, band_75_85 & is_fixed20
                ),
            },
            "compact_support_audit": {
                "N1_6_N2_60_rows": int(is_6_60.sum()),
                "plus_core_and_wcp_fixed20_rows": int(is_fixed20.sum()),
                "plus_primary_cw1_5_gap1_1p6_rows": int(
                    is_primary_5t.sum()
                ),
                "plus_W1200_L900_H750_rows": int(is_compact.sum()),
                "assessment": (
                    "unsupported_combination_extrapolation_FEA_required"
                    if not int(is_compact.sum())
                    else "exact_support_present"
                ),
            },
            "anchor_core_rescue": {
                "anchor_physical_geometry_sha256": ANCHOR_SHA256,
                "anchor_primary_turns": ANCHOR_PRIMARY_TURNS,
                "anchor_B_design_square_material_analytic_T": ANCHOR_B_T,
                "anchor_Ae_effective_m2": ANCHOR_AE_EFFECTIVE_M2,
                "target_B_nominal_T": TARGET_B_NOMINAL_T,
                "target_B_band_T": list(TARGET_B_BAND_T),
                "required_N_times_Ae_multiplier_nominal": nominal_multiplier,
                "required_N_times_Ae_multiplier_by_band_edge": (
                    band_multipliers
                ),
                "required_Ae_effective_m2_for_N1_6": target_ae,
                "required_Ae_effective_m2_for_N1_7": {
                    target: value * 6.0 / 7.0
                    for target, value in target_ae.items()
                },
                "N1_7_N2_70_role": "comparison_only",
            },
            "authority": {
                "analytical_B_role": (
                    "conservative_monotonic_design_constraint"
                ),
                "B_max_core_role": "diagnostic_only_not_acceptance_truth",
                "core_loss_formula_role": (
                    "conservative_monotonic_reference_with_observed_bias"
                ),
                "surrogate_role": (
                    "acquisition_only_in_unsupported_compact_fixed20_slice"
                ),
                "temperature_acceptance": "symmetric_FEA_required",
            },
        }
    )


def _markdown(audit: dict[str, Any]) -> str:
    evaluation = audit["disjoint_evaluation"]
    temperature = audit["temperature_evidence"][
        "B_0p70_to_0p80_6_60"
    ]["statistics"]["T_max_core"]
    support = audit["compact_support_audit"]
    anchor = audit["anchor_core_rescue"]
    b_metric = evaluation["B_design_square_vs_FEA_mean_material"]
    p_formula = evaluation["P_core_formula_vs_FEA"]
    p_model = evaluation["P_core_surrogate_vs_FEA"]
    return "\n".join(
        [
            "# Compact core-rescue physics alignment audit",
            "",
            f"- Schema: `{audit['schema_version']}`",
            f"- Payload SHA-256: `{audit['payload_sha256']}`",
            (
                "- Evaluation: 616 rows, seed-42/43 partition, "
                "disjoint from fit and calibration."
            ),
            "",
            "## Alignment",
            "",
            (
                "- Design-square B vs FEA mean material B: "
                f"bias {b_metric['bias']:+.5f} T "
                f"({b_metric['bias_percent_of_reference_mean']:+.2f}%), "
                f"RMSE {b_metric['rmse']:.5f} T, "
                f"p90 |error| {b_metric['p90_absolute_error']:.5f} T, "
                f"r={b_metric['pearson_correlation']:.5f}."
            ),
            (
                "- Core-loss formula vs FEA: "
                f"bias {p_formula['bias']:+.2f} W "
                f"({p_formula['bias_percent_of_reference_mean']:+.2f}%), "
                f"RMSE {p_formula['rmse']:.2f} W, "
                f"p90 |error| {p_formula['p90_absolute_error']:.2f} W, "
                f"r={p_formula['pearson_correlation']:.5f}."
            ),
            (
                "- Pcore surrogate vs FEA: "
                f"bias {p_model['bias']:+.2f} W "
                f"({p_model['bias_percent_of_reference_mean']:+.2f}%), "
                f"RMSE {p_model['rmse']:.2f} W, "
                f"p90 |error| {p_model['p90_absolute_error']:.2f} W, "
                f"r={p_model['pearson_correlation']:.5f}."
            ),
            "",
            "## B=0.70–0.80 T thermal evidence",
            "",
            (
                f"- 6/60 rows: n={temperature['n']}; "
                f"Tcore mean {temperature['mean']:.2f} °C, "
                f"median {temperature['median']:.2f} °C, "
                f"p90 {temperature['p90']:.2f} °C, "
                f"max {temperature['maximum']:.2f} °C."
            ),
            "",
            "## Compact support and rescue target",
            "",
            (
                "- Exact support counts: "
                f"6/60={support['N1_6_N2_60_rows']}, "
                f"+fixed20={support['plus_core_and_wcp_fixed20_rows']}, "
                f"+5T/1.6={support['plus_primary_cw1_5_gap1_1p6_rows']}, "
                f"+compact={support['plus_W1200_L900_H750_rows']}."
            ),
            (
                "- Anchor B="
                f"{anchor['anchor_B_design_square_material_analytic_T']:.6f} T "
                "requires N·Ae multiplier "
                f"{anchor['required_N_times_Ae_multiplier_nominal']:.6f} "
                "for nominal 0.8 T."
            ),
            "",
            "All compact fixed20 candidates are extrapolative until symmetric "
            "EM+thermal FEA; surrogate temperatures are acquisition values only.",
            "",
        ]
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = build_audit(
        args.dataset.resolve(),
        args.generation.resolve(),
        args.regression_root.resolve(),
    )
    json_path = output / "core_physics_alignment_audit.json"
    md_path = output / "core_physics_alignment_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(audit), encoding="utf-8")
    manifest = _seal(
        {
            "schema_version": "mft-goal-core-physics-alignment-manifest-v1",
            "audit_payload_sha256": audit["payload_sha256"],
            "files": {
                json_path.name: {
                    "sha256": _sha256_file(json_path),
                    "bytes": json_path.stat().st_size,
                },
                md_path.name: {
                    "sha256": _sha256_file(md_path),
                    "bytes": md_path.stat().st_size,
                },
            },
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(
            manifest, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "audit_payload_sha256": audit["payload_sha256"],
                "manifest_payload_sha256": manifest["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
