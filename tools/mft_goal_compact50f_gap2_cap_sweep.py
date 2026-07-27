"""Prepare strict compact50f gap2 or equal-height direct-cap sweeps.

The exact task-97585 geometry is kept fixed except for either secondary
inter-turn spacing or the common primary/secondary winding height.  Each point
is re-derived through the production strict validator and must remain inside
1200 x 900 x 750 mm (with the tighter 1150-mm width audit) before a
feeder-compatible, sealed cap-screen plan is emitted.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


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
from tools import mft_goal_core_rescue_fea_feeder as feeder  # noqa: E402


RUNTIME = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
)
BASE_ROOT = RUNTIME / "core_rescue_exact_gap_n5_full_v2"
BASE_PLAN = BASE_ROOT / "plan.json"
BASE_PARAMS = BASE_ROOT / "params" / "lane-01.json"
CAP_PROFILE = (
    RUNTIME
    / "crx_nearpass_97581_held_v1"
    / "profiles"
    / "cap-screen.json"
)
SOLVER_REVISION = "06726ed20c9a9a0eeb472f2457fde0d332bc4405"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
BASE_TASK_ID = 97585
DEFAULT_GAPS = (0.85, 0.95, 1.05, 1.10)
DEFAULT_HEIGHT_MM = 490.0
CAP_MODE = "matrix_turngraded_rx_cap"


class CompactSweepError(RuntimeError):
    """The exact source, strict geometry, or fixed contract drifted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _builtin(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _external_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": feeder._file_sha(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validated(
    raw: Mapping[str, Any],
    *,
    expected_height_mm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        ok, frame, errors = validation_check(
            create_input_parameter(dict(raw)),
            strict=True,
            return_errors=True,
        )
    except ValueError as exc:
        raise CompactSweepError(f"strict validation failed: {exc}") from exc
    if not ok or errors or len(frame) != 1:
        raise CompactSweepError(
            "strict validation failed: " + " / ".join(errors)
        )
    row = frame.iloc[0]
    _volume, dimensions = bounding_box_lit(row)
    width, length, height = map(float, dimensions)
    checks = {
        "turns_6_60_split_32_28": (
            int(row["N1"]) == 6
            and int(row["N2"]) == 60
            and int(row["N2_main"]) == 32
            and int(row["N2_side"]) == 28
        ),
        "fixed_core": (
            int(row["n_core_group"]) == 5
            and float(row["l2"]) == 280.0
            and float(row["w1"]) == 484.0
        ),
        "equal_winding_height_requested": (
            float(row["nwh1"]) == expected_height_mm
            and float(row["nwh2"]) == expected_height_mm
        ),
        "fixed_foils": (
            float(row["cw1"]) == 5.0
            and float(row["gap1"]) == 1.6
            and float(row["cw2"]) == 0.3
        ),
        "fixed_cooling": (
            float(row["core_plate_t"]) == 20.0
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
        "equal_positive_three_leg_gap": (
            int(row["core_equal_three_leg_air_gap"]) == 1
            and int(row["core_air_gap_gapped_leg_count"]) == 3
            and float(row["core_center_gap_mm"]) > 0.0
        ),
        "within_primary_envelope": (
            width <= 1200.0 and length <= 900.0 and height <= 750.0
        ),
        "within_compact_width_1150": width <= 1150.0,
        "wcp_length_preserved": float(row["wcp_len_x"]) == 324.6,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise CompactSweepError("contract failed: " + ", ".join(failed))
    effective = {
        key: _builtin(row[key]) for key in sorted(ALL_INPUT_KEYS)
    }
    return effective, {
        "checks": checks,
        "W_mm": width,
        "L_mm": length,
        "H_mm": height,
        "W_headroom_to_1150_mm": 1150.0 - width,
        "L_headroom_to_900_mm": 900.0 - length,
    }


def prepare(
    output: Path,
    *,
    specs: tuple[tuple[float, float], ...],
) -> Path:
    destination = output.resolve()
    if destination.exists():
        raise CompactSweepError(f"output exists: {destination}")
    destination.mkdir(parents=True)

    base_plan = feeder._validate_seal(
        feeder._read(BASE_PLAN), feeder.SCHEMA
    )
    base_params = feeder._read(BASE_PARAMS)
    if (
        base_plan["library_revision"] != LIBRARY_REVISION
        or float(base_params["core_center_gap_mm"])
        != 0.402982779173143
    ):
        raise CompactSweepError("exact compact50f source drifted")
    cap_profile = feeder._read(CAP_PROFILE)
    overrides = cap_profile["param_overrides"]
    if (
        int(overrides.get("matrix_on", 0)) != 1
        or int(overrides.get("cap_on", 0)) != 1
        or int(overrides.get("loss_on", 1)) != 0
        or int(overrides.get("thermal_on", 1)) != 0
        or int(overrides.get("round_corner", 1)) != 0
        or int(overrides.get("full_model", 1)) != 0
    ):
        raise CompactSweepError("cap profile drifted")
    profile_record = feeder._profile_record(
        cap_profile, destination, "profiles/cap-screen.json"
    )

    rows: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    for lane_index, (gap2, winding_height) in enumerate(specs, start=1):
        raw = copy.deepcopy(base_params)
        raw.update(
            {
                "gap2": float(gap2),
                "nwh1": float(winding_height),
                "nwh2": float(winding_height),
                "matrix_on": 1,
                "cap_on": 1,
                "cap_turn_graded_active_winding": "Rx",
                "cap_turn_graded_voltage_policy": "turn_midpoint",
                "cap_turn_graded_section_order": "main,side",
                "loss_on": 0,
                "thermal_on": 0,
                "keep_project": 0,
                "loss_from_copy": 0,
                "round_corner": 0,
                "full_model": 0,
            }
        )
        effective, audit = _validated(
            raw,
            expected_height_mm=float(winding_height),
        )
        candidate_sha = feeder._sha(effective)
        candidate_id = (
            f"gap2_{gap2:.3f}_h{winding_height:.1f}".replace(".", "p")
        )
        params_path = feeder._write(
            destination / "params" / f"lane-{lane_index:02d}.json",
            effective,
        )
        name = (
            f"mft-compact50f-cap-{lane_index:02d}-{candidate_id}-"
            f"{candidate_sha[:10]}"
        )
        identity = feeder.scheduler_client.verification_submission_identity(
            name,
            effective,
            cap_profile,
            SOLVER_REVISION,
            LIBRARY_REVISION,
        )
        rows.append(
            {
                "lane_index": lane_index,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "gap2_mm": float(gap2),
                "equal_winding_height_mm": float(winding_height),
                **audit,
            }
        )
        lanes.append(
            {
                "lane_index": lane_index,
                "candidate_index": lane_index,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "source_task_id": BASE_TASK_ID,
                "source_plan_payload_sha256": base_plan["payload_sha256"],
                "core_center_gap_mm": float(effective["core_center_gap_mm"]),
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "mode": CAP_MODE,
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
                    "timeout_seconds": int(cap_profile["timeout_seconds"]),
                    "priority": feeder.PRIORITY,
                    "max_workers_per_node": 1,
                    "environment": feeder._core_environment(SOLVER_REVISION),
                },
            }
        )

    plan = feeder._seal(
        {
            "schema_version": feeder.SCHEMA,
            "created_at_utc": _now(),
            "source": {
                "base_task_id": BASE_TASK_ID,
                "base_plan": _external_record(BASE_PLAN),
                "base_plan_payload_sha256": base_plan["payload_sha256"],
                "base_params": _external_record(BASE_PARAMS),
                "profile_source": _external_record(CAP_PROFILE),
            },
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "regression_contract": {
                "turns": "6/60",
                "secondary_split": "32/28",
                "n_core_group": 5,
                "l2_mm": 280.0,
                "w1_mm": 484.0,
                "equal_winding_height_values_mm": sorted(
                    {height for _gap, height in specs}
                ),
                "cw1_mm": 5.0,
                "gap1_mm": 1.6,
                "cw2_mm": 0.3,
                "core_plate_t_mm": 20.0,
                "winding_plate_t_mm": 20.0,
                "pad_t_mm": 2.0,
                "fan_velocity_m_s": 1.5,
                "TIM_k_W_mK": 0.2,
                "equal_identical_physical_gap_all_three_legs": True,
                "physical_gap_mm": 0.402982779173143,
                "nonrounded_eighth": True,
                "W_max_mm": 1150.0,
                "L_max_mm": 900.0,
                "H_max_mm": 750.0,
            },
            "selection": {
                "candidate_count": len(rows),
                "gap2_values_mm": sorted({gap for gap, _height in specs}),
                "equal_winding_height_values_mm": sorted(
                    {height for _gap, height in specs}
                ),
                "strict_geometry_audit": rows,
                "direct_cap_FEA_is_final_cap_authority": True,
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
    parser.add_argument(
        "--gap2",
        type=float,
        action="append",
        help="repeat for each requested gap; defaults to .85/.95/1.05/1.10",
    )
    parser.add_argument(
        "--height",
        type=float,
        action="append",
        help=(
            "repeat for an equal-winding-height sweep; when present, gap2 is "
            "held at --fixed-gap2"
        ),
    )
    parser.add_argument("--fixed-gap2", type=float, default=0.85)
    args = parser.parse_args(argv)
    if args.height and args.gap2:
        raise CompactSweepError("--height and --gap2 are mutually exclusive")
    if args.height:
        heights = tuple(args.height)
        specs = tuple((float(args.fixed_gap2), height) for height in heights)
    else:
        gaps = tuple(args.gap2) if args.gap2 else DEFAULT_GAPS
        specs = tuple((gap, DEFAULT_HEIGHT_MM) for gap in gaps)
    if (
        len(specs) != len(set(specs))
        or any(gap <= 0.0 or height <= 0.0 for gap, height in specs)
    ):
        raise CompactSweepError("gap2/height specs must be unique and positive")
    path = prepare(args.output, specs=specs)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
