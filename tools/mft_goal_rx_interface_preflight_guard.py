"""Read-only guard for the corrected Rx shared-interface canary preflight.

This tool performs Scheduler GET requests only.  A passing predispatch marker
does not authorize cancellation; corrected replacement receipts must also be
sealed and authenticated before the old acquisition lanes can be released.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Any, Mapping
import urllib.request


MARKER_PREFIX = "THERMAL_RX_INTERFACE_PREFLIGHT_JSON="
EXPECTED_SCHEMA = "thermal-rx-interface-predispatch-v1"
EXPECTED_INTERFACE_CONTRACT = "thermal-rx-block-interface-coverage-v1"
EXPECTED_MESH_POLICY = (
    "b6-rx-block-shared-region-wcp-pad-symmetry-contact-clipped-v1"
)
EXPECTED_MESH_PLAN_CONTRACT = "thermal-mesh-plan-v7"
EXPECTED_RX_MAIN_OBJECTS = ["Rx_main_block_xn", "Rx_main_block_yp"]
EXPECTED_FIXED_COOLING_SCHEMA = "mft-fixed-thermal-boundary-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GuardError(RuntimeError):
    """The Scheduler task or canary preflight evidence is unavailable."""


def _scheduler_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _stdout_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for name in ("stdout", "output", "text"):
            if isinstance(value.get(name), str):
                return value[name]
    raise GuardError("Scheduler stdout response has no text")


def _extract_marker(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        position = line.find(MARKER_PREFIX)
        if position < 0:
            continue
        encoded = line[position + len(MARKER_PREFIX) :].strip()
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise GuardError("Rx-interface preflight marker is malformed") from exc
        if not isinstance(value, dict):
            raise GuardError("Rx-interface preflight marker is not an object")
        return value
    raise GuardError("Rx-interface preflight marker is absent")


def _validate_marker(marker: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []

    def exact(name: str, expected: Any) -> None:
        if marker.get(name) != expected:
            reasons.append(f"{name}_mismatch")

    exact("schema", EXPECTED_SCHEMA)
    exact("passed", True)
    exact(
        "thermal_rx_block_interface_contract_version",
        EXPECTED_INTERFACE_CONTRACT,
    )
    exact("mesh_policy", EXPECTED_MESH_POLICY)
    exact("mesh_plan_contract_version", EXPECTED_MESH_PLAN_CONTRACT)
    exact("rx_main_objects", EXPECTED_RX_MAIN_OBJECTS)

    pack_count = marker.get("rx_block_shared_pack_count")
    if (
        not isinstance(pack_count, int)
        or isinstance(pack_count, bool)
        or pack_count < 1
    ):
        reasons.append("rx_block_shared_pack_count_invalid")

    operations = marker.get("rx_main_shared_operations")
    if not isinstance(operations, list) or len(operations) != 1:
        reasons.append("rx_main_shared_operations_invalid")
    else:
        operation = operations[0]
        if not isinstance(operation, dict):
            reasons.append("rx_main_shared_operation_not_object")
        else:
            expected_operation = {
                "plan_name": "rx_main_block_mesh_level",
                "objects": EXPECTED_RX_MAIN_OBJECTS,
                "shared_region": True,
                "separate_objects": False,
                "native_separate_objects": False,
            }
            for name, expected in expected_operation.items():
                if operation.get(name) != expected:
                    reasons.append(f"rx_main_shared_operation_{name}_mismatch")

    exact("shared_intent_and_native_readback_passed", True)
    exact("native_operation_readback_passed", True)

    premesh_status = marker.get("premesh_status")
    if not isinstance(premesh_status, str) or not (
        "direct" in premesh_status.lower()
        or "pass" in premesh_status.lower()
    ):
        reasons.append("premesh_status_not_direct_or_passed_native")
    if (
        marker.get("generate_mesh_returned") is not True
        and marker.get("direct_analyze_gate_passed") is not True
    ):
        reasons.append("premesh_dispatch_evidence_missing")

    cooling = marker.get("fixed_cooling_identity")
    if not isinstance(cooling, dict):
        reasons.append("fixed_cooling_identity_missing")
    else:
        if cooling.get("schema") != EXPECTED_FIXED_COOLING_SCHEMA:
            reasons.append("fixed_cooling_identity_schema_mismatch")
        cooling_sha = cooling.get("contract_sha256")
        if not isinstance(cooling_sha, str) or not HEX64.fullmatch(cooling_sha):
            reasons.append("fixed_cooling_identity_hash_invalid")
        try:
            fan_velocity = float(cooling.get("fan_velocity_m_s"))
            pad_conductivity = float(
                cooling.get("thermal_pad_conductivity_W_mK")
            )
        except (TypeError, ValueError, OverflowError):
            reasons.append("fixed_cooling_identity_numeric_invalid")
        else:
            if not math.isclose(
                fan_velocity, 1.5, rel_tol=0.0, abs_tol=1e-12
            ):
                reasons.append("fixed_cooling_identity_fan_velocity_mismatch")
            if not math.isclose(
                pad_conductivity, 0.2, rel_tol=0.0, abs_tol=1e-12
            ):
                reasons.append(
                    "fixed_cooling_identity_pad_conductivity_mismatch"
                )
        if cooling.get("cooling_boundary_modified") is not False:
            reasons.append("fixed_cooling_identity_boundary_modified")

    exact("terminal_interface_coverage_pending", True)
    exact("thermal_rx_main_interface_coverage_passed", False)
    exact("thermal_result_scientific_valid", False)
    return reasons


def inspect_task(
    *,
    scheduler_url: str,
    task_id: int,
    expected_name: str | None = None,
) -> dict[str, Any]:
    origin = scheduler_url.rstrip("/")
    task = _scheduler_json(f"{origin}/api/tasks/{task_id}")
    if isinstance(task, dict) and isinstance(task.get("task"), dict):
        task = task["task"]
    if not isinstance(task, dict):
        raise GuardError("Scheduler task response is not an object")
    if int(task.get("id", task.get("task_id", -1))) != task_id:
        raise GuardError("Scheduler task identity drifted")
    if expected_name is not None and task.get("name") != expected_name:
        raise GuardError("Scheduler task name drifted")
    if task.get("status") not in {"attaching", "running", "completed"}:
        raise GuardError("corrected canary is not attached/running/completed")

    stdout = _stdout_text(
        _scheduler_json(f"{origin}/api/tasks/{task_id}/stdout")
    )
    marker = _extract_marker(stdout)
    reasons = _validate_marker(marker)
    return {
        "schema": "mft-goal-rx-interface-preflight-guard-v1",
        "task_id": task_id,
        "task_name": task.get("name"),
        "task_status": task.get("status"),
        "actual_node_name": task.get("actual_node_name"),
        "allocation_id": task.get("allocation_id"),
        "slurm_job_id": task.get("slurm_job_id"),
        "marker": marker,
        "gate_passed": not reasons,
        "reasons": reasons,
        "cancellation_authorized": False,
        "next_required_gate": (
            "sealed corrected replacement receipts and accepted task GETs"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--scheduler-url", default="http://127.0.0.1:8002"
    )
    parser.add_argument("--expected-name")
    args = parser.parse_args()
    try:
        report = inspect_task(
            scheduler_url=args.scheduler_url,
            task_id=args.task_id,
            expected_name=args.expected_name,
        )
    except GuardError as exc:
        print(json.dumps({"gate_passed": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
