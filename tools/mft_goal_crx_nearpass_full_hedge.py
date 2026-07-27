"""Prepare full-physics hedge lanes from authenticated Crx cap screens.

The source near-pass plan remains immutable.  This tool promotes only selected
candidate indices to Matrix + turn-graded Rx Cap + Loss + Thermal while
preserving the fixed 6/60, cooling, plate, symmetry, and three-leg-gap
contracts.  It prepares a sealed feeder-compatible plan; submission remains a
separate explicit ``mft_goal_core_rescue_fea_feeder.py submit --apply`` step.
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

from tools import mft_goal_core_rescue_fea_feeder as feeder


FULL_MODE = "matrix_turngraded_cap_loss_thermal"
SOURCE_PLAN = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "crx_nearpass_97581_held_v1"
    / "plan.json"
)
SOURCE_RECEIPT = SOURCE_PLAN.parent / "submission_receipt.json"
DIRECT_CAP_TASKS = {1: 97766, 2: 97767, 3: 97768, 4: 97769}


class FullHedgeError(RuntimeError):
    """The immutable source or fixed design contract drifted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_close(
    value: Mapping[str, Any], key: str, expected: float, tol: float = 1e-12
) -> None:
    if abs(float(value.get(key, float("nan"))) - expected) > tol:
        raise FullHedgeError(f"{key} drifted from {expected}")


def _assert_contract(params: Mapping[str, Any]) -> None:
    if (
        int(params.get("N1_main", -1)) + int(params.get("N1_side", -1))
        != 6
        or int(params.get("N2_main", -1))
        + int(params.get("N2_side", -1))
        != 60
        or int(params.get("core_equal_three_leg_air_gap", 0)) != 1
        or float(params.get("core_center_gap_mm", 0.0)) <= 0.0
        or int(params.get("round_corner", 1)) != 0
        or int(params.get("full_model", 1)) != 0
        or float(params.get("nwh1", -1.0))
        != float(params.get("nwh2", -2.0))
        or float(params.get("cw2", 99.0)) > 1.0
    ):
        raise FullHedgeError("fixed turns/gap/symmetry/height/cw2 contract drifted")
    for key, expected in (
        ("cw1", 5.0),
        ("gap1", 1.6),
        ("core_plate_t", 20.0),
        ("core_plate_pad_t", 2.0),
        ("wcp_t", 20.0),
        ("wcp_pad_t", 2.0),
        ("fan_velocity", 1.5),
    ):
        _require_close(params, key, expected)


def _source_lane_by_index(
    source_plan: Mapping[str, Any], candidate_index: int
) -> Mapping[str, Any]:
    matches = [
        lane
        for lane in source_plan["lanes"]
        if int(lane.get("candidate_index", -1)) == candidate_index
        and str(lane.get("mode")) == "matrix_turngraded_rx_cap"
    ]
    if len(matches) != 1:
        raise FullHedgeError(
            f"expected one source cap lane for candidate {candidate_index}"
        )
    return matches[0]


def prepare(
    output: Path,
    *,
    source_plan_path: Path,
    source_receipt_path: Path,
    candidate_indices: tuple[int, ...],
) -> Path:
    destination = output.resolve()
    if destination.exists():
        raise FullHedgeError(f"output exists: {destination}")
    destination.mkdir(parents=True)

    source_plan = feeder._validate_seal(
        feeder._read(source_plan_path), feeder.SCHEMA
    )
    source_receipt = feeder._validate_seal(
        feeder._read(source_receipt_path), feeder.SUBMISSION_SCHEMA
    )
    if (
        source_receipt.get("scheduler_POST_performed") is not True
        or source_receipt.get("plan_payload_sha256")
        != source_plan["payload_sha256"]
    ):
        raise FullHedgeError("source submission receipt is incomplete or unbound")
    submitted = {
        int(row["candidate_index"]): int(row["task_id"])
        for row in source_receipt["submissions"]
        if str(row.get("mode")) == "matrix_turngraded_rx_cap"
    }
    if any(
        submitted.get(index) != DIRECT_CAP_TASKS.get(index)
        for index in candidate_indices
    ):
        raise FullHedgeError("direct cap task binding drifted")

    source_root = source_plan_path.resolve(strict=True).parent
    source_full_lanes = [
        lane for lane in source_plan["lanes"] if lane["mode"] == FULL_MODE
    ]
    if len(source_full_lanes) != 1:
        raise FullHedgeError("source full profile is ambiguous")
    full_profile = feeder._read(
        source_root / source_full_lanes[0]["profile"]["path"]
    )
    if (
        int(full_profile["param_overrides"].get("matrix_on", 0)) != 1
        or int(full_profile["param_overrides"].get("cap_on", 0)) != 1
        or int(full_profile["param_overrides"].get("loss_on", 0)) != 1
        or int(full_profile["param_overrides"].get("thermal_on", 0)) != 1
        or int(full_profile.get("cpus", feeder.CPUS)) != feeder.CPUS
    ):
        raise FullHedgeError("source full profile drifted")
    full_profile_record = feeder._profile_record(
        full_profile, destination, "profiles/full-physics.json"
    )

    lanes: list[dict[str, Any]] = []
    for lane_index, candidate_index in enumerate(candidate_indices, start=1):
        source_lane = _source_lane_by_index(source_plan, candidate_index)
        params = copy.deepcopy(
            feeder._read(source_root / source_lane["params"]["path"])
        )
        params.update(
            {
                "matrix_on": 1,
                "cap_on": 1,
                "cap_turn_graded_active_winding": "Rx",
                "cap_turn_graded_voltage_policy": "turn_midpoint",
                "cap_turn_graded_section_order": "main,side",
                "loss_on": 1,
                "thermal_on": 1,
                "keep_project": 1,
                "loss_from_copy": 0,
            }
        )
        _assert_contract(params)
        params_path = feeder._write(
            destination / "params" / f"lane-{lane_index:02d}.json", params
        )
        candidate_id = str(source_lane["candidate_id"])
        candidate_sha = str(source_lane["candidate_sha256"])
        name = (
            f"mft-crx-nearpass-full-hedge-{candidate_index:02d}-"
            f"{candidate_id}-{candidate_sha[:10]}"
        )
        identity = feeder.scheduler_client.verification_submission_identity(
            name,
            params,
            full_profile,
            source_plan["solver_revision"],
            source_plan["library_revision"],
        )
        lanes.append(
            {
                "lane_index": lane_index,
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "source_task_id": int(DIRECT_CAP_TASKS[candidate_index]),
                "source_plan_payload_sha256": source_plan["payload_sha256"],
                "source_receipt_payload_sha256": source_receipt["payload_sha256"],
                "core_center_gap_mm": float(params["core_center_gap_mm"]),
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "mode": FULL_MODE,
                "params": feeder._record(params_path, destination),
                "params_sha256": feeder._sha(params),
                "profile": full_profile_record,
                "scheduler": {
                    "project": feeder.scheduler_client.MFT_PROJECT,
                    "name": name,
                    "workdir": name.replace("-", "_"),
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": feeder._sha(identity["merged"]),
                    "cpus": feeder.CPUS,
                    "memory_mb": feeder.MEMORY_MB,
                    "timeout_seconds": int(full_profile["timeout_seconds"]),
                    "priority": feeder.PRIORITY,
                    "max_workers_per_node": 1,
                    "environment": feeder._core_environment(
                        source_plan["solver_revision"]
                    ),
                },
            }
        )

    plan = feeder._seal(
        {
            "schema_version": feeder.SCHEMA,
            "created_at_utc": _now(),
            "source": {
                "plan": {
                    "path": str(source_plan_path.resolve(strict=True)),
                    "sha256": feeder._file_sha(
                        source_plan_path.resolve(strict=True)
                    ),
                    "size_bytes": source_plan_path.resolve(strict=True).stat().st_size,
                },
                "plan_payload_sha256": source_plan["payload_sha256"],
                "receipt": {
                    "path": str(source_receipt_path.resolve(strict=True)),
                    "sha256": feeder._file_sha(
                        source_receipt_path.resolve(strict=True)
                    ),
                    "size_bytes": (
                        source_receipt_path.resolve(strict=True).stat().st_size
                    ),
                },
                "receipt_payload_sha256": source_receipt["payload_sha256"],
                "direct_cap_tasks": {
                    str(index): DIRECT_CAP_TASKS[index]
                    for index in candidate_indices
                },
            },
            "solver_revision": source_plan["solver_revision"],
            "library_revision": source_plan["library_revision"],
            "regression_contract": copy.deepcopy(
                source_plan["regression_contract"]
            ),
            "selection": {
                "candidate_indices": list(candidate_indices),
                "full_physics_lane_count": len(lanes),
                "direct_cap_screen_required_before_promotion": True,
                "final_authority": "direct full FEA",
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
    parser.add_argument("--source-plan", type=Path, default=SOURCE_PLAN)
    parser.add_argument("--source-receipt", type=Path, default=SOURCE_RECEIPT)
    parser.add_argument(
        "--candidate-index",
        type=int,
        action="append",
        required=True,
        choices=tuple(DIRECT_CAP_TASKS),
    )
    args = parser.parse_args(argv)
    indices = tuple(dict.fromkeys(args.candidate_index))
    path = prepare(
        args.output,
        source_plan_path=args.source_plan,
        source_receipt_path=args.source_receipt,
        candidate_indices=indices,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
