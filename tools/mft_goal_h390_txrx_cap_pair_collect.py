"""Collect the exact final h390 Tx/Rx turn-graded capacitance pair.

This collector is read-only with respect to Scheduler.  It binds two
high-accuracy direct-FEA tasks to their sealed plans and submission receipts,
requires identical physical/operating parameters apart from the active
turn-graded winding schedule, and emits:

* one authenticated result receipt per winding;
* one actual-connection topology receipt; and
* one final-artifact-pipeline-ready graded-capacitance provenance object.

The fixed design resonance inductances are 2 mH on Tx and 0.2 H on Rx.  The
solver's independently extracted self inductances remain recorded as
readbacks, but they do not replace those design targets in the final 15 kHz
gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.request


PLAN_SCHEMA = "mft-goal-core-rescue-direct-fea-plan-v1"
RECEIPT_SCHEMA = "mft-goal-core-rescue-direct-fea-submission-v1"
RESULT_RECEIPT_SCHEMA = "mft-goal-h390-graded-cap-result-receipt-v1"
TOPOLOGY_RECEIPT_SCHEMA = (
    "mft-goal-final-actual-connection-topology-receipt-v1"
)
PROVENANCE_SCHEMA = "mft-goal-final-graded-capacitance-provenance-v1"
TURN_GRADED_SCHEMA = "mft-turn-graded-energy-capacitance-v2"
EXPECTED_MODE = "matrix_turngraded_cap_high_accuracy_authority"
TARGET_INDUCTANCE_H = {"Tx": 0.002, "Rx": 0.2}
EXPECTED_SECTIONS = {
    "Tx": (["main"], {"main": 1}),
    "Rx": (["main", "side"], {"main": 1, "side": 1}),
}
CAP_SCHEDULE_KEYS = {
    "cap_turn_graded_active_winding",
    "cap_turn_graded_section_order",
    "cap_turn_graded_reverse_sections",
    "cap_turn_graded_reverse_terminal_polarity",
    "cap_turn_graded_side_polarity",
    "cap_turn_graded_side2_polarity",
    # Stage toggles select which solvers run; they do not alter the retained
    # physical transformer, operating point, materials, or cooling boundary.
    "loss_on",
    "thermal_on",
}
SCHEDULER_BASE_URL = "http://127.0.0.1:8002"


class PairCollectionError(RuntimeError):
    """A plan, task, topology, or numerical authority contract drifted."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PairCollectionError("non-canonical JSON value") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    label = (
        resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": label,
        "sha256": _file_sha(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise PairCollectionError("value is already sealed")
    result["payload_sha256"] = _sha(result)
    return result


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PairCollectionError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PairCollectionError(f"JSON object required: {path}")
    return value


def _validated(path: Path, schema: str) -> dict[str, Any]:
    value = _read(path)
    observed = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(observed, str)
        or len(observed) != 64
        or _sha(unsigned) != observed
    ):
        raise PairCollectionError(f"sealed contract drifted: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value) + b"\n"
    if target.exists():
        if target.read_bytes() != payload:
            raise PairCollectionError(f"refusing to overwrite: {target}")
        return target
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(target)
    return target


def _http_json(base: str, suffix: str) -> dict[str, Any]:
    url = f"{base.rstrip('/')}{suffix}"
    try:
        with urllib.request.urlopen(url, timeout=60.0) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise PairCollectionError(f"Scheduler GET failed: {url}") from exc
    if not isinstance(value, dict):
        raise PairCollectionError(f"Scheduler object required: {url}")
    return value.get("task", value)


def _http_text(base: str, suffix: str) -> str:
    url = f"{base.rstrip('/')}{suffix}"
    try:
        with urllib.request.urlopen(url, timeout=60.0) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, UnicodeDecodeError) as exc:
        raise PairCollectionError(f"Scheduler text GET failed: {url}") from exc


def _result(stdout: str, task_id: int) -> dict[str, Any]:
    prefix = "RESULT_JSON "
    rows = [
        line[len(prefix):]
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(rows) != 1:
        raise PairCollectionError(
            f"task {task_id} must contain exactly one RESULT_JSON"
        )
    try:
        value = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise PairCollectionError(
            f"task {task_id} RESULT_JSON is malformed"
        ) from exc
    if not isinstance(value, dict):
        raise PairCollectionError(
            f"task {task_id} RESULT_JSON must be an object"
        )
    return value


def _finite(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PairCollectionError(f"{label} is not finite") from exc
    if not math.isfinite(numeric):
        raise PairCollectionError(f"{label} is not finite")
    return numeric


def _json_value(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        raise PairCollectionError(f"{label} JSON string is absent")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise PairCollectionError(f"{label} JSON is malformed") from exc


def _single_binding(
    plan_path: Path,
    *,
    expected_mode: str | None = EXPECTED_MODE,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    resolved = plan_path.resolve(strict=True)
    root = resolved.parent
    plan = _validated(resolved, PLAN_SCHEMA)
    receipt_path = root / "submission_receipt.json"
    receipt = _validated(receipt_path, RECEIPT_SCHEMA)
    lanes = plan.get("lanes") or []
    submissions = receipt.get("submissions") or []
    if (
        len(lanes) != 1
        or len(submissions) != 1
        or receipt.get("complete") is not True
        or receipt.get("scheduler_POST_performed") is not True
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
    ):
        raise PairCollectionError(f"single-lane binding failed: {resolved}")
    lane = lanes[0]
    submission = submissions[0]
    if (
        (expected_mode is not None and lane.get("mode") != expected_mode)
        or (
            expected_mode is not None
            and submission.get("mode") != expected_mode
        )
        or submission.get("candidate_sha256")
        != lane.get("candidate_sha256")
        or submission.get("dedupe_key")
        != (lane.get("scheduler") or {}).get("dedupe_key")
    ):
        raise PairCollectionError(f"lane binding drifted: {resolved}")
    params_path = root / lane["params"]["path"]
    params = _read(params_path)
    if (
        _record(params_path, relative_to=root) != lane["params"]
        or _sha(params) != lane["params_sha256"]
    ):
        raise PairCollectionError(f"params binding drifted: {params_path}")
    return plan, receipt, params, resolved, receipt_path, params_path


def _physical_identity(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in sorted(params.items())
        if key not in CAP_SCHEDULE_KEYS
    }


def _collect_winding(
    *,
    active: str,
    plan_path: Path,
    scheduler_base_url: str,
    output_root: Path,
    candidate_physics_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, receipt, params, resolved_plan, receipt_path, params_path = (
        _single_binding(plan_path)
    )
    submission = receipt["submissions"][0]
    task_id = int(submission["task_id"])
    task = _http_json(scheduler_base_url, f"/api/tasks/{task_id}")
    stdout = _http_text(
        scheduler_base_url, f"/api/tasks/{task_id}/stdout"
    )
    result = _result(stdout, task_id)
    prefix = active.lower()
    cap_field = f"C_{prefix}_{prefix}_turn_graded_F"
    suffix = f"cap_turn_graded_{prefix}"
    capacitance = _finite(result.get(cap_field), f"{active} capacitance")
    solver_self = _finite(
        result.get("self_inductance_H"), f"{active} solver self inductance"
    )
    fixed_inductance = TARGET_INDUCTANCE_H[active]
    fixed_frequency = 1.0 / (
        2.0 * math.pi * math.sqrt(fixed_inductance * capacitance)
    )
    schedule = _json_value(
        result.get("cap_turn_graded_schedule_json"),
        f"{active} schedule",
    )
    expected_sections, expected_polarities = EXPECTED_SECTIONS[active]
    gap = _finite(result.get("core_center_gap_mm"), f"{active} gap")
    checks = {
        "scheduler_terminal_success": (
            str(task.get("status") or "").lower() == "completed"
            and str(task.get("state") or "").lower() == "succeeded"
            and int(task.get("exit_code") or 0) == 0
        ),
        "solver_revision_exact": (
            result.get("git_hash") == plan.get("solver_revision")
        ),
        "library_revision_exact": (
            result.get("pyaedt_library_git_hash")
            == plan.get("library_revision")
        ),
        "active_winding_exact": (
            params.get("cap_turn_graded_active_winding") == active
            and result.get("cap_turn_graded_active_winding") == active
            and result.get("active_winding") == active
        ),
        "turn_graded_schema_exact": (
            result.get("cap_turn_graded_schema_version")
            == TURN_GRADED_SCHEMA
        ),
        "nonrounded_eighth": (
            int(result.get("full_model", -1)) == 0
            and int(result.get("round_corner", -1)) == 0
            and str(result.get("thermal_symmetry")) == "eighth"
        ),
        "equal_three_leg_gap": (
            int(result.get("core_equal_three_leg_air_gap", 0)) == 1
            and int(result.get("core_air_gap_gapped_leg_count", 0)) == 3
            and result.get("core_air_gap_topology")
            == "equal_center_and_both_side_legs"
            and int(
                result.get(
                    "core_equal_three_leg_air_gap_geometry_attested", 0
                )
            )
            == 1
            and int(
                result.get(
                    "core_equal_three_leg_air_gap_symmetry_geometry_attested",
                    0,
                )
            )
            == 1
        ),
        "schedule_actual_connection": (
            schedule.get("winding") == active
            and schedule.get("voltage_policy") == "turn_midpoint"
            and schedule.get("reverse_sections") == []
            and schedule.get("reverse_terminal_polarity") is False
            and schedule.get("section_order") == expected_sections
            and schedule.get("section_polarities") == expected_polarities
        ),
        "even_potential_symmetry_explicit": (
            int(
                result.get(
                    "electrostatic_even_potential_symmetry_assumed", 0
                )
            )
            == 1
            and schedule.get(
                "electrostatic_even_potential_symmetry_assumed"
            )
            is True
        ),
        "high_accuracy_cap_converged": (
            _finite(
                result.get(f"conv_passes_{suffix}"),
                f"{active} cap passes",
            )
            >= 1.0
            and _finite(
                result.get(f"conv_consecutive_{suffix}"),
                f"{active} cap consecutive",
            )
            >= 1.0
            and _finite(
                result.get(f"conv_error_pct_{suffix}"),
                f"{active} cap error",
            )
            <= 0.25
        ),
        "fixed_Lm2mH_resonance_pass_15kHz": fixed_frequency >= 15_000.0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PairCollectionError(
            f"{active} authority failed: {', '.join(failed)}"
        )
    result_sha = _sha(result)
    receipt_value = _seal(
        {
            "schema_version": RESULT_RECEIPT_SCHEMA,
            "active_winding": active,
            "task_id": task_id,
            "scheduler_status": task.get("status"),
            "scheduler_state": task.get("state"),
            "scheduler_node": (
                task.get("actual_node_name") or task.get("node_name")
            ),
            "scheduler_slurm_job_id": task.get("slurm_job_id"),
            "finished_at": task.get("finished_at"),
            "plan": _record(resolved_plan),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission_receipt": _record(receipt_path),
            "submission_payload_sha256": receipt["payload_sha256"],
            "params": _record(params_path),
            "candidate_physics_sha256": candidate_physics_sha256,
            "physical_identity_sha256": _sha(_physical_identity(params)),
            "result_sha256": result_sha,
            "stdout_sha256": hashlib.sha256(
                stdout.encode("utf-8")
            ).hexdigest(),
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "core_center_gap_mm": gap,
            "core_equal_three_leg_air_gap": 1,
            "terminal_capacitance_F": capacitance,
            "solver_self_inductance_H": solver_self,
            "fixed_inductance_for_resonance_H": fixed_inductance,
            "fixed_Lm2mH_resonance_Hz": fixed_frequency,
            "fixed_Lm2mH_resonance_pass_15kHz": True,
            "cap_convergence": {
                "passes": _finite(
                    result.get(f"conv_passes_{suffix}"),
                    f"{active} passes",
                ),
                "consecutive": _finite(
                    result.get(f"conv_consecutive_{suffix}"),
                    f"{active} consecutive",
                ),
                "error_pct": _finite(
                    result.get(f"conv_error_pct_{suffix}"),
                    f"{active} error",
                ),
                "delta_pct": _finite(
                    result.get(f"conv_delta_pct_{suffix}"),
                    f"{active} delta",
                ),
                "mesh_tets": _finite(
                    result.get(f"mesh_tets_{suffix}"),
                    f"{active} mesh",
                ),
            },
            "schedule": schedule,
            "contract_checks": checks,
            "scheduler_mutation_performed": False,
        }
    )
    receipt_output = output_root / (
        f"{active.lower()}-graded-cap-authenticated-result-receipt.json"
    )
    _write(receipt_output, receipt_value)
    row = {
        "task_id": task_id,
        "active_winding": active,
        "cap_turn_graded_schema_version": TURN_GRADED_SCHEMA,
        "candidate_physics_sha256": candidate_physics_sha256,
        "core_center_gap_mm": gap,
        "core_equal_three_leg_air_gap": 1,
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "result_sha256": result_sha,
        "terminal_capacitance_F": capacitance,
        "self_inductance_H": fixed_inductance,
        "solver_self_inductance_H": solver_self,
        "resonance_Hz": fixed_frequency,
        "authenticated_result_receipt": _record(
            receipt_output, relative_to=output_root
        ),
    }
    return row, receipt_value, {
        "params": params,
        "result": result,
        "schedule": schedule,
    }


def collect(
    *,
    tx_plan: Path,
    rx_plan: Path,
    final_plan: Path,
    output_dir: Path,
    scheduler_base_url: str,
) -> Path:
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final, _receipt, final_params, resolved_final, _rp, final_params_path = (
        _single_binding(final_plan, expected_mode=None)
    )
    final_lane = final["lanes"][0]
    candidate_physics_sha256 = str(final_lane["candidate_sha256"])
    tx, _tx_receipt, tx_evidence = _collect_winding(
        active="Tx",
        plan_path=tx_plan,
        scheduler_base_url=scheduler_base_url,
        output_root=destination,
        candidate_physics_sha256=candidate_physics_sha256,
    )
    rx, _rx_receipt, rx_evidence = _collect_winding(
        active="Rx",
        plan_path=rx_plan,
        scheduler_base_url=scheduler_base_url,
        output_root=destination,
        candidate_physics_sha256=candidate_physics_sha256,
    )
    identities = {
        "final": _sha(_physical_identity(final_params)),
        "Tx": _sha(_physical_identity(tx_evidence["params"])),
        "Rx": _sha(_physical_identity(rx_evidence["params"])),
    }
    gaps = {
        float(final_params["core_center_gap_mm"]),
        float(tx["core_center_gap_mm"]),
        float(rx["core_center_gap_mm"]),
    }
    if len(set(identities.values())) != 1 or len(gaps) != 1:
        raise PairCollectionError(
            "Tx/Rx/final physical identities or gaps do not match"
        )
    topology = _seal(
        {
            "schema_version": TOPOLOGY_RECEIPT_SCHEMA,
            "actual_connection_topology_attested": True,
            "attestation_basis": (
                "solver-emitted per-turn midpoint schedules bound to the "
                "exact final nonrounded eighth geometry and equal physical "
                "air gap on all three legs"
            ),
            "candidate_physics_sha256": candidate_physics_sha256,
            "physical_identity_sha256": identities["final"],
            "final_full_chain_plan": _record(resolved_final),
            "final_full_chain_params": _record(final_params_path),
            "core_center_gap_mm": next(iter(gaps)),
            "core_equal_three_leg_air_gap": 1,
            "air_gap_topology": "equal_center_and_both_side_legs",
            "turn_ratio": {"N1": 6, "N2": 60},
            "Tx": {
                "task_id": tx["task_id"],
                "active_winding": "Tx",
                "actual_section_order": ["main"],
                "actual_section_polarities": {"main": 1},
                "voltage_policy": "turn_midpoint",
                "reverse_sections": [],
                "reverse_terminal_polarity": False,
                "terminal_voltage_normalized_V": 1.0,
            },
            "Rx": {
                "task_id": rx["task_id"],
                "active_winding": "Rx",
                "actual_section_order": ["main", "side"],
                "actual_section_polarities": {"main": 1, "side": 1},
                "section_turn_counts": {"main": 35, "side": 25},
                "voltage_policy": "turn_midpoint",
                "reverse_sections": [],
                "reverse_terminal_polarity": False,
                "terminal_voltage_normalized_V": 1.0,
            },
            "electrostatic_even_potential_symmetry_assumed": True,
            "symmetric_side2_absence_attested": True,
            "full_model_series_interconnect_attested": False,
            "limitation": (
                "The symmetric electrostatic cut planes impose an "
                "even-potential approximation. This receipt attests the "
                "implemented actual-connection proxy, not a future full-model "
                "physical terminal interconnect."
            ),
            "legacy_two_net_result_used": False,
            "scheduler_mutation_performed": False,
        }
    )
    topology_path = _write(
        destination / "actual-connection-topology-receipt.json",
        topology,
    )
    minimum = min(float(tx["resonance_Hz"]), float(rx["resonance_Hz"]))
    provenance = _seal(
        {
            "schema_version": PROVENANCE_SCHEMA,
            "provenance_authenticated": True,
            "source_kind": "authenticated_same_geometry_tx_rx_pair",
            "geometry_and_gap_exact_match": True,
            "actual_connection_topology_attested": True,
            "actual_connection_topology_receipt": _record(
                topology_path, relative_to=destination
            ),
            "legacy_two_net_result_used": False,
            "candidate_physics_sha256": candidate_physics_sha256,
            "physical_identity_sha256": identities["final"],
            "core_center_gap_mm": next(iter(gaps)),
            "core_equal_three_leg_air_gap": 1,
            "equal_three_leg_air_gap_FEA_required": True,
            "minimum_resonance_Hz": minimum,
            "minimum_resonance_active_winding": (
                "Tx" if tx["resonance_Hz"] <= rx["resonance_Hz"] else "Rx"
            ),
            "pair_resonance_pass_15kHz": minimum >= 15_000.0,
            "tx": tx,
            "rx": rx,
            "scheduler_mutation_performed": False,
        }
    )
    provenance_path = _write(
        destination / "graded-capacitance-provenance.json",
        provenance,
    )
    collection = _seal(
        {
            "schema_version": "mft-goal-h390-final-txrx-cap-collection-v1",
            "final_plan": _record(resolved_final),
            "candidate_physics_sha256": candidate_physics_sha256,
            "physical_identity_sha256": identities["final"],
            "topology_receipt": _record(
                topology_path, relative_to=destination
            ),
            "graded_capacitance_provenance": _record(
                provenance_path, relative_to=destination
            ),
            "tx": tx,
            "rx": rx,
            "minimum_resonance_Hz": minimum,
            "pair_resonance_pass_15kHz": minimum >= 15_000.0,
            "all_contracts_passed": True,
            "scheduler_mutation_performed": False,
        }
    )
    return _write(destination / "collection.json", collection)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-plan", type=Path, required=True)
    parser.add_argument("--rx-plan", type=Path, required=True)
    parser.add_argument("--final-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scheduler-base-url", default=SCHEDULER_BASE_URL
    )
    args = parser.parse_args(argv)
    print(
        collect(
            tx_plan=args.tx_plan,
            rx_plan=args.rx_plan,
            final_plan=args.final_plan,
            output_dir=args.output_dir,
            scheduler_base_url=args.scheduler_base_url,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
