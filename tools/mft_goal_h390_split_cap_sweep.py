"""Prepare direct symmetric Matrix + turn-graded-cap checks for h390 splits."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.optimization.geometry_metrics import (  # noqa: E402
    bounding_box_lit,
)
from tools import mft_goal_compact50f_gap2_cap_sweep as compact  # noqa: E402
from tools import mft_goal_core_rescue_fea_feeder as feeder  # noqa: E402


SOURCE_ROOT = (
    compact.RUNTIME / "compact50f_height_cap_sweep_v1"
)
SOURCE_PLAN = SOURCE_ROOT / "plan.json"
SOURCE_PARAMS = Path(
    r"\\raidrive-peets\ANSYS\git\MFT_1MW_2026"
    r"\artifacts\h390_llt_correction_v1"
    r"\h390_36_24_equal_gap_0p4046737_params.json"
)
SOURCE_PROFILE = SOURCE_ROOT / "profiles" / "cap-screen.json"
FULL_PROFILE = (
    compact.RUNTIME
    / "compact50f_height390_full_v1"
    / "profiles"
    / "full-physics.json"
)
SOURCE_TASK_ID = 97881
SOURCE_PARAMS_SHA256 = (
    "e122cc95e9b97612fb3d9bb0059f98601b155f2c61d4201668f0695f19c5ad07"
)
SPLITS = (35, 36, 37)


def _builtin(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _validated(
    raw: dict[str, Any],
    split_main: int,
    expected_gap_mm: float,
    expected_h1_mm: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    ok, frame, errors = validation_check(
        create_input_parameter(raw),
        strict=True,
        return_errors=True,
    )
    if not ok or errors or len(frame) != 1:
        raise compact.CompactSweepError(
            "strict validation failed: " + " / ".join(errors)
        )
    row = frame.iloc[0]
    _volume, dimensions = bounding_box_lit(row)
    width, length, height = map(float, dimensions)
    checks = {
        "turns_and_requested_split": (
            int(row["N1"]) == 6
            and int(row["N2"]) == 60
            and int(row["N2_main"]) == split_main
            and int(row["N2_side"]) == 60 - split_main
        ),
        "fixed_h390_geometry": (
            int(row["n_core_group"]) == 5
            and float(row["l2"]) == 280.0
            and float(row["l1"]) == 90.0
            and float(row["h1"]) == expected_h1_mm
            and float(row["w1"]) == 484.0
            and float(row["nwh1"]) == 390.0
            and float(row["nwh2"]) == 390.0
            and float(row["gap2"]) == 0.85
            and float(row["core_center_gap_mm"]) == expected_gap_mm
        ),
        "fixed_foils_and_cooling": (
            float(row["cw1"]) == 5.0
            and float(row["gap1"]) == 1.6
            and float(row["cw2"]) == 0.3
            and float(row["core_plate_t"]) == 20.0
            and float(row["wcp_t"]) == 20.0
            and float(row["core_plate_pad_t"]) == 2.0
            and float(row["wcp_pad_t"]) == 2.0
            and float(row["fan_velocity"]) == 1.5
            and float(row["k_ins"]) == 0.2
        ),
        "nonrounded_eighth": (
            int(row["round_corner"]) == 0
            and int(row["full_model"]) == 0
            and str(row["thermal_symmetry"]) == "eighth"
        ),
        "equal_three_leg_gap": (
            int(row["core_equal_three_leg_air_gap"]) == 1
            and int(row["core_air_gap_gapped_leg_count"]) == 3
            and float(row["core_center_gap_mm"]) > 0.0
        ),
        "envelope": width <= 1200.0 and length <= 900.0 and height <= 750.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise compact.CompactSweepError(
            "split contract failed: " + ", ".join(failed)
        )
    effective = {
        key: _builtin(row[key]) for key in sorted(ALL_INPUT_KEYS)
    }
    return effective, {
        "W_mm": width,
        "L_mm": length,
        "H_mm": height,
    }


def prepare(
    output: Path,
    splits: tuple[int, ...],
    *,
    full: bool = False,
    matrix_authority: bool = False,
    cap_authority: bool = False,
    cap_active_winding: str = "Rx",
    cap_authority_max_passes: int = 12,
    cap_authority_percent_error: float = 0.25,
    equal_gaps_mm: tuple[float, ...] | None = None,
    h1_values_mm: tuple[float, ...] | None = None,
) -> Path:
    if sum((bool(full), bool(matrix_authority), bool(cap_authority))) > 1:
        raise compact.CompactSweepError(
            "--full, --matrix-authority, and --cap-authority are "
            "mutually exclusive"
        )
    if (
        cap_authority_max_passes < 12
        or cap_authority_percent_error <= 0.0
    ):
        raise compact.CompactSweepError(
            "Cap authority requires at least 12 passes and positive error"
        )
    if cap_active_winding not in {"Tx", "Rx"}:
        raise compact.CompactSweepError(
            "cap_active_winding must be exactly Tx or Rx"
        )
    destination = output.resolve()
    if destination.exists():
        raise compact.CompactSweepError(f"output exists: {destination}")
    destination.mkdir(parents=True)
    source_plan = feeder._validate_seal(
        feeder._read(SOURCE_PLAN), feeder.SCHEMA
    )
    if feeder._file_sha(SOURCE_PARAMS) != SOURCE_PARAMS_SHA256:
        raise compact.CompactSweepError("sealed h390 correction params drifted")
    source_params = feeder._read(SOURCE_PARAMS)
    selected_profile_path = FULL_PROFILE if full else SOURCE_PROFILE
    profile = feeder._read(selected_profile_path)
    if matrix_authority:
        profile = copy.deepcopy(profile)
        profile.update(
            {
                "comment": (
                    "Selected h390 35/25 authority Matrix solve: "
                    "10 passes, one converged pass, 1.5% target"
                ),
                "stage": "matrix_authority_10pass",
                "timeout_seconds": 14400,
            }
        )
        profile["param_overrides"].update(
            {
                "matrix_on": 1,
                "matrix_max_passes": 10,
                "matrix_min_converged": 1,
                "matrix_percent_error": 1.5,
                "cap_on": 0,
                "loss_on": 0,
                "thermal_on": 0,
                "keep_project": 1,
            }
        )
    elif cap_authority:
        profile = copy.deepcopy(profile)
        profile.update(
            {
                "comment": (
                    "Selected compact h390 high-accuracy turn-graded Rx "
                    f"Cap authority: {cap_authority_max_passes} passes and "
                    f"{cap_authority_percent_error}% target"
                ),
                "stage": "turn_graded_rx_cap_authority",
                "timeout_seconds": 14400,
            }
        )
        profile["param_overrides"].update(
            {
                "matrix_on": 1,
                "matrix_max_passes": 20,
                "matrix_min_converged": 1,
                "matrix_percent_error": 1.5,
                "cap_on": 1,
                "cap_max_passes": int(cap_authority_max_passes),
                "cap_percent_error": float(
                    cap_authority_percent_error
                ),
                "loss_on": 0,
                "thermal_on": 0,
                "keep_project": 1,
            }
        )
    profile_record = feeder._profile_record(
        profile,
        destination,
        (
            "profiles/full-physics.json"
            if full
            else (
                "profiles/matrix-authority.json"
                if matrix_authority
                else (
                    "profiles/cap-authority.json"
                    if cap_authority
                    else "profiles/cap-screen.json"
                )
            )
        ),
    )
    gaps = equal_gaps_mm or (0.4046737,)
    h1_values = h1_values_mm or (float(source_params["h1"]),)
    specs = tuple(
        (split, gap, h1)
        for split in splits
        for gap in gaps
        for h1 in h1_values
    )
    lanes: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for lane_index, (split_main, equal_gap_mm, h1_mm) in enumerate(
        specs, start=1
    ):
        if split_main < 12 or split_main > 59:
            raise compact.CompactSweepError("split is outside 12..59")
        raw = copy.deepcopy(source_params)
        raw.update(
            {
                "N2_main": int(split_main),
                "N2_side": int(60 - split_main),
                "h1": float(h1_mm),
                "core_center_gap_mm": float(equal_gap_mm),
                "cap_turn_graded_active_winding": cap_active_winding,
                "cap_turn_graded_section_order": (
                    "auto" if cap_active_winding == "Tx" else "main,side"
                ),
                "cap_turn_graded_reverse_sections": "none",
                "cap_turn_graded_reverse_terminal_polarity": 0,
                "cap_turn_graded_side_polarity": 1,
                "cap_turn_graded_side2_polarity": 1,
                "matrix_on": 1,
                "cap_on": int(not matrix_authority),
                "loss_on": int(full),
                "thermal_on": int(full),
                "round_corner": 0,
                "full_model": 0,
                "keep_project": int(
                    full or matrix_authority or cap_authority
                ),
            }
        )
        effective, dimensions = _validated(
            raw, split_main, float(equal_gap_mm), float(h1_mm)
        )
        candidate_sha = feeder._sha(effective)
        gap_token = f"{int(round(equal_gap_mm * 10_000_000)):08d}"
        candidate_id = (
            f"h390-split-{cap_active_winding.lower()}-"
            f"{split_main:02d}-{60 - split_main:02d}-"
            f"h1-{int(round(h1_mm)):03d}-g{gap_token}"
        )
        params_path = feeder._write(
            destination / "params" / f"lane-{lane_index:02d}.json",
            effective,
        )
        name = (
            f"mft-h390-split-"
            f"{'full' if full else ('capauth' if cap_authority else 'cap')}-"
            f"{cap_active_winding.lower()}-"
            f"{lane_index:02d}-"
            f"{split_main:02d}x{60 - split_main:02d}-"
            f"h{int(round(h1_mm)):03d}-g{gap_token}-"
            f"{candidate_sha[:10]}"
        )
        identity = feeder.scheduler_client.verification_submission_identity(
            name,
            effective,
            profile,
            compact.SOLVER_REVISION,
            compact.LIBRARY_REVISION,
        )
        audits.append(
            {
                "lane_index": lane_index,
                "N2_main": split_main,
                "N2_side": 60 - split_main,
                "h1_mm": float(h1_mm),
                "vertical_clearance_each_mm": (
                    float(h1_mm) - float(effective["nwh1"])
                )
                / 2.0,
                "equal_three_leg_gap_mm": float(equal_gap_mm),
                "candidate_sha256": candidate_sha,
                **dimensions,
            }
        )
        lanes.append(
            {
                "lane_index": lane_index,
                "candidate_index": lane_index,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "source_task_id": SOURCE_TASK_ID,
                "source_plan_payload_sha256": source_plan["payload_sha256"],
                "core_center_gap_mm": float(effective["core_center_gap_mm"]),
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "mode": (
                    "matrix_turngraded_cap_loss_thermal"
                    if full
                    else (
                        "matrix_only_10pass_authority"
                        if matrix_authority
                        else (
                            "matrix_turngraded_cap_high_accuracy_authority"
                            if cap_authority
                            else compact.CAP_MODE
                        )
                    )
                ),
                "params": feeder._record(params_path, destination),
                "params_sha256": feeder._sha(effective),
                "profile": profile_record,
                "scheduler": {
                    "project": feeder.scheduler_client.MFT_PROJECT,
                    "name": name,
                    "workdir": name.replace("-", "_"),
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": feeder._sha(identity["merged"]),
                    "cpus": feeder.CPUS,
                    "memory_mb": feeder.MEMORY_MB,
                    "timeout_seconds": int(profile["timeout_seconds"]),
                    "priority": feeder.PRIORITY,
                    "max_workers_per_node": 1,
                    "environment": feeder._core_environment(
                        compact.SOLVER_REVISION
                    ),
                },
            }
        )
    plan = feeder._seal(
        {
            "schema_version": feeder.SCHEMA,
            "created_at_utc": compact._now(),
            "source": {
                "plan": compact._external_record(SOURCE_PLAN),
                "params": compact._external_record(SOURCE_PARAMS),
                "profile": compact._external_record(selected_profile_path),
                "task_id": SOURCE_TASK_ID,
            },
            "solver_revision": compact.SOLVER_REVISION,
            "library_revision": compact.LIBRARY_REVISION,
            "selection": {
                "strategy": (
                    "h390_selected_35_25_full_symmetric_validation"
                    if full
                    else (
                        "h390_selected_35_25_matrix_10pass_authority"
                        if matrix_authority
                        else (
                            "h390_compact_selected_high_accuracy_cap_authority"
                            if cap_authority
                            else (
                                "h390_direct_split_bracket_after_32_28_FEA_bias"
                            )
                        )
                    )
                ),
                "audits": audits,
            },
            "regression_contract": {
                "turns": "6/60",
                "secondary_splits": [
                    f"{value}/{60 - value}" for value in splits
                ],
                "cap_turn_graded_active_winding": cap_active_winding,
                "equal_winding_height_mm": 390.0,
                "gap2_mm": 0.85,
                "cw2_mm": 0.3,
                "equal_three_leg_gap_values_mm": sorted(
                    {float(gap) for _split, gap, _h1 in specs}
                ),
                "h1_values_mm": sorted(
                    {float(h1) for _split, _gap, h1 in specs}
                ),
                "fixed_20T_cooling_plates": True,
                "equal_identical_physical_gap_all_three_legs": True,
                "nonrounded_eighth": True,
            },
            "lanes": lanes,
            "parallel_execution_requested": len(lanes) > 1,
            "scheduler_project_source_included": False,
            "scheduler_project_modified": False,
            "scheduler_submission_performed": False,
            "final_promotion_allowed": False,
        }
    )
    return feeder._write(destination / "plan.json", plan)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-main", action="append", type=int)
    parser.add_argument("--equal-gap", action="append", type=float)
    parser.add_argument("--h1", action="append", type=float)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--matrix-authority", action="store_true")
    parser.add_argument("--cap-authority", action="store_true")
    parser.add_argument(
        "--cap-active-winding",
        choices=("Tx", "Rx"),
        default="Rx",
    )
    parser.add_argument(
        "--cap-authority-max-passes", type=int, default=12
    )
    parser.add_argument(
        "--cap-authority-percent-error", type=float, default=0.25
    )
    args = parser.parse_args(argv)
    splits = tuple(args.split_main or SPLITS)
    print(
        prepare(
            args.output,
            splits,
            full=args.full,
            matrix_authority=args.matrix_authority,
            cap_authority=args.cap_authority,
            cap_active_winding=args.cap_active_winding,
            cap_authority_max_passes=args.cap_authority_max_passes,
            cap_authority_percent_error=args.cap_authority_percent_error,
            equal_gaps_mm=(
                tuple(args.equal_gap) if args.equal_gap else None
            ),
            h1_values_mm=(tuple(args.h1) if args.h1 else None),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
