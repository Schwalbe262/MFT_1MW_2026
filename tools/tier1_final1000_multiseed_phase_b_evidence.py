"""Build immutable, capacity-aware Phase-B dry evidence.

The command performs no Scheduler/API call and no FEA/AEDT work.  It uses the
reviewed current active39 empty-pool histogram, produces exact 431-running /
69-queued placement for 500 logical seeds, and writes JSON only to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:
    from tier1_final1000_multiseed_contract import canonical_sha256, json_bytes
    from tier1_final1000_multiseed_phase_b_successor import (
        build_placement_evidence,
        build_placement_plan,
        current_empty_pool_inventory,
    )
    from tier1_final1000_slurm_launch import validate_launch_plan
    from tier1_final1000_stage_profiles import STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_final1000_multiseed_contract import canonical_sha256, json_bytes
    from tools.tier1_final1000_multiseed_phase_b_successor import (
        build_placement_evidence,
        build_placement_plan,
        current_empty_pool_inventory,
    )
    from tools.tier1_final1000_slurm_launch import validate_launch_plan
    from tools.tier1_final1000_stage_profiles import STAGES


EVIDENCE_SCHEMA = "mft-tier1-final1000-multiseed-phase-b-dry-run-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("launch plan must be a JSON object")
    return value


def _next_seed_inventory(plan: Mapping[str, Any]) -> dict[str, int]:
    tasks = [
        *plan["task_waves"]["canaries"],
        *plan["task_waves"]["ramp"],
    ]
    result: dict[str, int] = {}
    for stage in STAGES:
        seeds = [
            int(task["payload_json"]["seed"])
            for task in tasks
            if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
        ]
        if not seeds:
            raise RuntimeError("launch plan has no stage seed inventory")
        result[stage.stage_id] = max(seeds) + 1
    return result


def build_dry_run_evidence(
    plan: Mapping[str, Any], *, source_file_sha256: str
) -> dict[str, Any]:
    source = validate_launch_plan(plan)
    inventory = current_empty_pool_inventory()
    placement_with_tasks = build_placement_plan(
        source,
        next_seed_by_stage=_next_seed_inventory(source),
        allocation_inventory=inventory,
    )
    placement = {
        key: item for key, item in placement_with_tasks.items() if key != "parent_tasks"
    }
    capacity_evidence = build_placement_evidence(placement, inventory)
    unsigned = {
        "schema_version": EVIDENCE_SCHEMA,
        "source_launch_plan_sha256": source["launch_plan_sha256"],
        "source_file_sha256": str(source_file_sha256),
        "allocation_inventory_sha256": capacity_evidence["allocation_inventory_sha256"],
        "placement_plan_sha256": capacity_evidence["placement_plan_sha256"],
        "allocation_histogram": capacity_evidence["allocation_histogram"],
        "stage_logical_quotas": capacity_evidence["stage_logical_quotas"],
        "logical_seed_count": capacity_evidence["logical_seed_count"],
        "running_logical_count": capacity_evidence["running_logical_count"],
        "queued_logical_count": capacity_evidence["queued_logical_count"],
        "physical_parent_count": capacity_evidence["physical_parent_count"],
        "running_parent_count": capacity_evidence["running_parent_count"],
        "queued_parent_count": capacity_evidence["queued_parent_count"],
        "maxjobs_reduction": capacity_evidence["logical_seed_count"]
        - capacity_evidence["physical_parent_count"],
        "shape_parent_counts": capacity_evidence["shape_parent_counts"],
        "running_stage_shapes": capacity_evidence["running_stage_shapes"],
        "queued_stage_shapes": capacity_evidence["queued_stage_shapes"],
        "parent_task_inventory_sha256": capacity_evidence[
            "parent_task_inventory_sha256"
        ],
        "logical_child_inventory_sha256": capacity_evidence[
            "logical_child_inventory_sha256"
        ],
        "one_lane_repack_logical_child_inventory_sha256": capacity_evidence[
            "one_lane_repack_logical_child_inventory_sha256"
        ],
        "lane_count_independent_science_identity": True,
        "capacity_snapshot_recheck_required": True,
        "capacity_snapshot_drift_action": "rerender-or-fail-closed",
        "dispatch_admission_policy": capacity_evidence["dispatch_admission_policy"],
        "static_eight_first_pack_production_eligible": False,
        "documentary_only": True,
        "production_eligible": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "virtual_scheduler_task_ids_created": False,
    }
    return {**unsigned, "evidence_sha256": canonical_sha256(unsigned)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = args.launch_plan.read_bytes()
    plan = _read_json(args.launch_plan)
    evidence = build_dry_run_evidence(
        plan, source_file_sha256=hashlib.sha256(payload).hexdigest()
    )
    sys.stdout.buffer.write(json_bytes(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
