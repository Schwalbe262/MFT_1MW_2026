"""Enumerate and seal compact 6/60 fixed-20T B≈0.8 core geometries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.optimization.design_summary import (  # noqa: E402
    B_AREA_BASIS_GROSS_WITH_LAMINATION,
    design_analytical_b_field_t,
)
from regression_260707.optimization.geometry_metrics import (  # noqa: E402
    bounding_box_lit,
)


SCHEMA = "mft-goal-core-rescue-geometry-sweep-v1"
ANCHOR_SHA = (
    "86778c97c5d6e48fb9e76f53a75cf1e4e59d163a3600ce8a59d4c151d0695be3"
)
ANCHOR_B_T = 1.4617010926508007
ANCHOR_AE_M2 = 0.028505600000000002
GRID = {
    "B_target_T": [0.75, 0.775, 0.8, 0.825, 0.85],
    "l1_mm": list(range(70, 101, 5)),
    "l2_mm": [356.0, 357.0, 359.0],
    "cw2_mm": [0.45, 0.55, 0.65, 0.75, 0.85],
    "gap2_mm": [0.3, 0.5, 0.7, 0.9, 1.1],
    "N2_main": list(range(24, 47)),
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return result


def _load_anchor(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["physical_geometry_sha256"] == ANCHOR_SHA:
                return json.loads(row["decoded_physical_params_json"])
    raise RuntimeError(f"anchor {ANCHOR_SHA} was not found")


def _algebraic_rows() -> list[tuple]:
    rows = []
    for target_b, l1, l2, cw2, gap2, n2_main in itertools.product(
        GRID["B_target_T"],
        GRID["l1_mm"],
        GRID["l2_mm"],
        GRID["cw2_mm"],
        GRID["gap2_mm"],
        GRID["N2_main"],
    ):
        ae_required = 1000.0 / (4.0 * 1000.0 * 6.0 * target_b)
        iron_depth = ae_required / (0.85 * 2.0 * l1 * 1e-6)
        core_groups = math.ceil(iron_depth / 120.0)
        if not 2 <= core_groups <= 10:
            continue
        if iron_depth / core_groups < 60.0:
            continue
        w1 = round(iron_depth + (core_groups + 1) * 24.0)
        core_depth_each = (
            w1 - (core_groups + 1) * 24.0
        ) / core_groups
        if not 60.0 <= core_depth_each <= 120.0:
            continue
        n2_side = 60 - n2_main
        nwl2_main = n2_main * cw2 + (n2_main - 1) * gap2
        nwl2_side = n2_side * cw2 + (n2_side - 1) * gap2
        width = 4.0 * l1 + 2.0 * l2 + 80.0 + 2.0 * nwl2_side
        length = w1 + 325.6 + 2.0 * nwl2_main
        height = 547.0 + 2.0 * l1
        if width <= 1200.0 and length <= 900.0 and height <= 750.0:
            rows.append(
                (
                    abs(target_b - 0.8),
                    abs(width - length),
                    width * length * height,
                    target_b,
                    l1,
                    l2,
                    w1,
                    core_groups,
                    core_depth_each,
                    cw2,
                    gap2,
                    n2_main,
                    n2_side,
                )
            )
    return rows


def _decode(anchor: dict[str, Any], algebraic: tuple) -> dict | None:
    (
        _,
        _,
        _,
        target_b,
        l1,
        l2,
        w1,
        core_groups,
        _,
        cw2,
        gap2,
        n2_main,
        n2_side,
    ) = algebraic
    values = dict(anchor)
    values.update(
        {
            "N1_main": 6,
            "N1_side": 0,
            "N2_main": n2_main,
            "N2_side": n2_side,
            "l1": l1,
            "l2": l2,
            "h1": 547,
            "w1": w1,
            "n_core_group": core_groups,
            "core_plate_t": 20.0,
            "wcp_t": 20.0,
            "cw1": 5.0,
            "gap1": 1.6,
            "cw2": cw2,
            "gap2": gap2,
            "nwh1": 453.1,
            "nwh2": 453.1,
            "cc_w2c_space_x": 40.0,
            "cc_w2c_space_y": 40.0,
            "w2c_w1c_space_x": 40.0,
            "w2c_w1c_space_y": 40.0,
            "w1c_w2s_space_x": 32.4,
            "w1s_cs_space_x": 40.0,
            "cs_w1s_space_y": 40.0,
            "round_corner": 0,
            "full_model": 0,
        }
    )
    try:
        frame = create_input_parameter(
            {key: values[key] for key in KEYS if key in values}
        )
        valid, derived = validation_check(frame, strict=False)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not valid:
        return None
    row = derived.iloc[0]
    volume_l, dimensions = bounding_box_lit(row)
    b_design = design_analytical_b_field_t(
        row,
        core_lamination_factor=0.85,
        area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
    )
    width, length, height = map(float, dimensions)
    if not (
        0.75 <= b_design <= 0.85
        and width <= 1200.0
        and length <= 900.0
        and height <= 750.0
    ):
        return None
    decoded = row.to_dict()
    identity = {
        key: decoded[key]
        for key in (
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
            "nwh1",
            "nwh2",
            "Ae_effective_m2",
        )
    }
    geometry_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    return {
        "geometry_sweep_sha256": geometry_sha,
        "target_B_grid_T": float(target_b),
        "B_design_square_material_analytic_T": float(b_design),
        "W_mm": width,
        "L_mm": length,
        "H_mm": height,
        "volume_L": float(volume_l),
        "N_times_Ae_multiplier_vs_anchor": (
            float(decoded["N1"]) * float(decoded["Ae_effective_m2"])
        )
        / (6.0 * ANCHOR_AE_M2),
        "exact_training_support_count": 0,
        "extrapolation_status": (
            "fixed20_compact_5T_1p6_exact_support_absent_FEA_required"
        ),
        "full_decoded_params_json": json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ranked-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-limit", type=int, default=5000)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    anchor = _load_anchor(args.source_ranked_csv.resolve())
    algebraic = sorted(_algebraic_rows())
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    for candidate in algebraic[: args.validation_limit]:
        decoded = _decode(anchor, candidate)
        if decoded is not None:
            rows.append(decoded)
    rows.sort(
        key=lambda item: (
            abs(item["B_design_square_material_analytic_T"] - 0.8),
            item["volume_L"],
            item["geometry_sweep_sha256"],
        )
    )
    if not rows:
        raise RuntimeError("core-rescue geometry sweep produced no valid rows")

    csv_path = output / "validated_core_rescue_geometries.csv"
    _write_csv(csv_path, rows)
    summary = _seal(
        {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_ranked_csv": str(args.source_ranked_csv.resolve()),
            "source_ranked_csv_sha256": _file_sha(
                args.source_ranked_csv.resolve()
            ),
            "anchor_physical_geometry_sha256": ANCHOR_SHA,
            "anchor_B_design_square_material_analytic_T": ANCHOR_B_T,
            "anchor_Ae_effective_m2": ANCHOR_AE_M2,
            "target_B_band_T": [0.75, 0.85],
            "target_B_nominal_T": 0.8,
            "required_N_times_Ae_multiplier_nominal": ANCHOR_B_T / 0.8,
            "grid": GRID,
            "algebraic_feasible_count": len(algebraic),
            "validation_limit": args.validation_limit,
            "decoded_valid_count": len(rows),
            "fixed_identity": {
                "turns": "6/60",
                "cw1_mm": 5.0,
                "gap1_mm": 1.6,
                "core_plate_t_mm": 20.0,
                "wcp_t_mm": 20.0,
                "nwh1_equals_nwh2": True,
                "fan_velocity_m_s": 1.5,
                "TIM_modified": False,
                "round_corner": 0,
                "full_model": 0,
            },
            "limits": {"W_mm": 1200.0, "L_mm": 900.0, "H_mm": 750.0},
            "support": {
                "strict6151_exact_compact_fixed20_5T_1p6_rows": 0,
                "authority": "geometry_only_FEA_required",
            },
            "best_nominal_rows": [
                {
                    key: row[key]
                    for key in (
                        "geometry_sweep_sha256",
                        "B_design_square_material_analytic_T",
                        "W_mm",
                        "L_mm",
                        "H_mm",
                        "volume_L",
                        "N_times_Ae_multiplier_vs_anchor",
                        "full_decoded_params_json",
                    )
                }
                for row in rows[:12]
            ],
        }
    )
    summary_path = output / "geometry_sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    manifest = _seal(
        {
            "schema_version": "mft-goal-core-rescue-geometry-manifest-v1",
            "summary_payload_sha256": summary["payload_sha256"],
            "files": {
                csv_path.name: {
                    "sha256": _file_sha(csv_path),
                    "bytes": csv_path.stat().st_size,
                },
                summary_path.name: {
                    "sha256": _file_sha(summary_path),
                    "bytes": summary_path.stat().st_size,
                },
            },
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decoded_valid_count": len(rows),
                "summary_payload_sha256": summary["payload_sha256"],
                "manifest_payload_sha256": manifest["payload_sha256"],
                "csv_sha256": manifest["files"][csv_path.name]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
