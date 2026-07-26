#!/usr/bin/env python3
"""Fail-closed final Full/Symmetric solver package gate.

The gate consumes the local outputs of the GET-only task96326 Full collector
and task96324 corrected-thermal collector.  It publishes solver-produced AEDT
files only when both collectors are terminal-success authorities, both model
results are bound to the same candidate and fixed cooling contract, and every
explicit goal constraint has finite actual solver evidence and passes.

Until then the output root contains only ``pending_manifest.json``.  An
AEDT-open receipt or the older diagnostic fallback snapshots can never satisfy
this gate because neither is a terminal collector result authority.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import mft_goal_20260726_contract as goal  # noqa: E402
from tools import mft_goal_collect_live_aedt_transport as aedt_transport  # noqa: E402
from tools import mft_goal_diagnostic_final_handoff_v2 as diagnostic_handoff  # noqa: E402
from tools import mft_goal_terminal_collector as terminal_collector  # noqa: E402


CAMPAIGN_ID = "mft-goal-20260726"
PENDING_SCHEMA = "mft-goal-final-solver-package-pending-v1"
TRUTH_SCHEMA = "mft-goal-final-solver-result-truth-v1"
PACKAGE_SCHEMA = "mft-goal-final-solver-package-v1"
SEAL_SCHEMA = "mft-goal-final-solver-package-seal-v1"
PACKAGE_NAME = "final_solver_package_v1"
FULL_TASK_ID = 96326
THERMAL_TASK_ID = 96324
LOGICAL_CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
FULL_EFFECTIVE_PARAMS_SHA256 = (
    "62f2b846d05bbc5880165c6086fe9897e04d0588b0b8767fceb35102e4be9a44"
)
SOURCE_SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
THERMAL_EXECUTOR_REVISION = "a0208331949f70c21cde948e853a331d7f7f9824"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
FIXED_THERMAL_BOUNDARY = {
    "fan_velocity_m_per_s": 1.5,
    "thermal_pad_thickness_mm": 2.0,
    "tim_conductivity_w_per_mk": 0.2,
}
EXPECTED_PACKAGE_FILES = {
    "evidence/full_collector_event.json",
    "evidence/thermal_collector_event.json",
    "models/full.aedt",
    "models/symmetric.aedt",
    "results/corrected_thermal_result.json",
    "results/full_result.json",
    "result_truth.json",
    "package_manifest.json",
    "package_seal.json",
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class FinalPackageGateError(RuntimeError):
    """An authority or final-package invariant failed closed."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _payload_sha256(value: Any) -> str:
    return diagnostic_handoff.canonical_value_sha256(value)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(value))
    if "payload_sha256" in sealed:
        raise FinalPackageGateError("payload is already sealed")
    sealed["payload_sha256"] = _payload_sha256(sealed)
    return sealed


def _validate_payload_seal(
    value: Mapping[str, Any],
    *,
    schema: str,
    schema_field: str = "schema",
) -> None:
    if value.get(schema_field) != schema:
        raise FinalPackageGateError(f"{schema} schema drifted")
    claimed = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if not isinstance(claimed, str) or _payload_sha256(unsigned) != claimed:
        raise FinalPackageGateError(f"{schema} payload seal drifted")


def _read_json(path: Path, label: str, *, maximum_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    try:
        diagnostic_handoff._regular_file(path, maximum_bytes=maximum_bytes)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        diagnostic_handoff.HandoffError,
    ) as exc:
        raise FinalPackageGateError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise FinalPackageGateError(f"{label} root is not an object")
    return value


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    try:
        return diagnostic_handoff._record(path, relative_to=relative_to)
    except (OSError, ValueError, diagnostic_handoff.HandoffError) as exc:
        raise FinalPackageGateError(f"unsafe package authority file: {path}") from exc


def _is_contained_file(path: Path, root: Path, label: str) -> Path:
    try:
        resolved_root = diagnostic_handoff._real_directory(root, label=label)
        resolved = path.resolve(strict=True)
        diagnostic_handoff._regular_file(resolved)
    except (OSError, diagnostic_handoff.HandoffError) as exc:
        raise FinalPackageGateError(f"{label} is not a safe file") from exc
    if resolved_root not in resolved.parents:
        raise FinalPackageGateError(f"{label} escaped its collector root")
    return resolved


def _record_matches(path: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    actual = _file_record(path)
    if (
        actual["sha256"] != record.get("sha256")
        or actual["size_bytes"] != record.get("size_bytes")
    ):
        raise FinalPackageGateError(f"{label} file record drifted")
    return actual


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FinalPackageGateError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalPackageGateError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise FinalPackageGateError(f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer():
        raise FinalPackageGateError(f"{label} must be an integer")
    return int(number)


def _exact_float(value: Any, expected: float, label: str) -> float:
    observed = _finite(value, label)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise FinalPackageGateError(
            f"{label} drifted: {observed!r} != {expected!r}"
        )
    return observed


def _validate_terminal_event(
    path: Path,
    *,
    target: str,
    task_id: int,
) -> tuple[dict[str, Any], list[str]]:
    event = _read_json(path, f"{target} collector event")
    claimed = event.get("event_sha256")
    unsigned = dict(event)
    unsigned.pop("event_sha256", None)
    if (
        event.get("schema") != "mft-terminal-collector-event-v1"
        or event.get("target") != target
        or not isinstance(claimed, str)
        or terminal_collector._sha256(
            terminal_collector._canonical_bytes(unsigned)
        )
        != claimed
        or event.get("diagnostic_only") is not True
        or event.get("canonical") is not False
        or event.get("production_truth_eligible") is not False
        or event.get("scientific_pass_claimed") is not False
        or event.get("production_claimed") is not False
    ):
        raise FinalPackageGateError(f"{target} collector event seal/classification drifted")
    task = event.get("task")
    contract = event.get("contract")
    if (
        not isinstance(task, dict)
        or task.get("task_id") != task_id
        or not isinstance(contract, dict)
        or contract.get("task_identity_verified") is not True
        or contract.get("fixed_physics_verified") is not True
        or contract.get("strict_node_verified") is not True
    ):
        raise FinalPackageGateError(f"{target} collector task contract drifted")
    pending: list[str] = []
    state = str(event.get("state") or "")
    if state != "success_collected_diagnostic":
        pending.append(f"{target} collector state is {state or 'missing'}")
    if event.get("artifact_collection_attempted") is not True:
        pending.append(f"{target} collector has not collected terminal artifacts")
    if event.get("failure_or_timeout") is True:
        pending.append(f"{target} task failed or timed out")
    return event, pending


def _validate_authority_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(path, "Full candidate authority plan")
    _validate_payload_seal(
        plan,
        schema="mft-goal-provisional-full-precompute-plan-v1",
        schema_field="schema_version",
    )
    fixed = plan.get("fixed_boundary")
    expected_fixed = {
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "mutable": False,
    }
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("candidate_physics_sha256") != LOGICAL_CANDIDATE_SHA256
        or plan.get("effective_full_params_sha256")
        != FULL_EFFECTIVE_PARAMS_SHA256
        or plan.get("solver_revision") != SOURCE_SOLVER_REVISION
        or plan.get("library_revision") != LIBRARY_REVISION
        or plan.get("goal_hard_spec_sha256") != goal.GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        or fixed != expected_fixed
        or plan.get("diagnostic_only") is not True
        or plan.get("production_eligible") is not False
        or plan.get("automatic_promotion") is not False
    ):
        raise FinalPackageGateError("Full candidate authority plan drifted")
    return plan, _file_record(path)


def _validate_full_retry_plan(
    path: Path,
    *,
    authority_plan: Mapping[str, Any],
    full_event: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _read_json(path, "post-deadline Full retry plan")
    unsigned = dict(plan)
    seal = unsigned.pop("seal", None)
    if (
        not isinstance(seal, dict)
        or not isinstance(seal.get("sha256"), str)
        or _payload_sha256(unsigned) != seal["sha256"]
    ):
        raise FinalPackageGateError("post-deadline Full retry plan seal drifted")
    classification = plan.get("classification")
    payload = plan.get("payload")
    command_reuse = plan.get("command_reuse")
    if (
        not isinstance(classification, dict)
        or classification.get("diagnostic_only") is not True
        or classification.get("production") is not False
        or classification.get("canonical") is not False
        or not isinstance(payload, dict)
        or not isinstance(command_reuse, dict)
        or command_reuse.get("physics_candidate", {}).get("sha256")
        != authority_plan["effective_full_params_sha256"]
        or command_reuse.get("physics_candidate", {}).get("byte_exact_unchanged")
        is not True
    ):
        raise FinalPackageGateError("post-deadline Full retry classification drifted")
    command = str(payload.get("command") or "").encode("utf-8")
    candidate = terminal_collector._validate_full_candidate(command)
    if _payload_sha256(candidate) != authority_plan["effective_full_params_sha256"]:
        raise FinalPackageGateError("Full effective candidate cross-binding drifted")
    task = full_event["task"]
    if (
        payload.get("name") != task.get("name")
        or payload.get("dedupe_key") != task.get("dedupe_key")
    ):
        raise FinalPackageGateError("Full retry task identity drifted")
    event_plan = full_event["contract"].get("local_evidence", {}).get("plan")
    plan_record = _file_record(path)
    if (
        not isinstance(event_plan, dict)
        or event_plan.get("sha256") != plan_record["sha256"]
        or event_plan.get("size_bytes") != plan_record["size_bytes"]
    ):
        raise FinalPackageGateError("Full collector did not bind the retry plan")
    return plan, candidate, plan_record


def _parameters_match(result: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    for key, expected in candidate.items():
        if key not in result:
            return False
        observed = result[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                if not math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
        elif str(observed) != str(expected):
            return False
    return True


def _load_full_sources(
    event_path: Path,
    event: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    collector_root = event_path.parent.resolve(strict=True)
    result_record = event.get("full_result_truth")
    aedt_record = event.get("reassembled_full_aedt")
    receipt = event.get("receipt")
    task = event["task"]
    if (
        not isinstance(result_record, dict)
        or not isinstance(aedt_record, dict)
        or not isinstance(receipt, dict)
        or not event.get("aedtresults_remote_paths")
    ):
        raise FinalPackageGateError("Full result/AEDT retention authority is incomplete")
    result_path = _is_contained_file(
        Path(str(result_record.get("path") or "")),
        collector_root / "collection",
        "Full result JSON",
    )
    aedt_path = _is_contained_file(
        Path(str(aedt_record.get("path") or "")),
        collector_root / "collection",
        "Full AEDT",
    )
    _record_matches(result_path, result_record, "Full result JSON")
    _record_matches(aedt_path, aedt_record, "Full AEDT")
    result = _read_json(result_path, "Full result JSON")
    identity = result_record.get("identity")
    expected_identity = {
        "candidate_physics_sha256": FULL_EFFECTIVE_PARAMS_SHA256,
        "solver_revision": SOURCE_SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "scheduler_task_id": FULL_TASK_ID,
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "full_model": 1,
        "thermal_symmetry": "full",
        "result_payload_sha256": terminal_collector._sha256(
            terminal_collector._canonical_bytes(result)
        ),
    }
    if identity != expected_identity:
        raise FinalPackageGateError("Full result identity record drifted")
    if (
        receipt.get("stage") != "full"
        or receipt.get("solver_revision") != SOURCE_SOLVER_REVISION
        or receipt.get("library_revision") != LIBRARY_REVISION
        or receipt.get("parameter_digest")
        != FULL_EFFECTIVE_PARAMS_SHA256[:16]
        or receipt.get("artifact_sha256") != aedt_record.get("sha256")
        or receipt.get("artifact_size_bytes") != aedt_record.get("size_bytes")
        or result.get("git_hash") != SOURCE_SOLVER_REVISION
        or result.get("pyaedt_library_git_hash") != LIBRARY_REVISION
        or _integer(result.get("full_model"), "Full result model mode") != 1
        or str(result.get("thermal_symmetry") or "").lower() != "full"
        or str(result.get("solver_core_scheduler_task_id_readback") or "")
        != str(FULL_TASK_ID)
        or str(result.get("solver_core_slurm_job_id_readback") or "")
        != str(task.get("slurm_job_id") or "")
        or not _parameters_match(result, candidate)
    ):
        raise FinalPackageGateError("Full result candidate/solver/AEDT binding drifted")
    required_flags = (
        "result_valid_em",
        "result_valid_thermal",
        "thermal_solved",
        "thermal_extraction_complete",
        "thermal_convergence_available",
        "thermal_converged",
    )
    if any(_integer(result.get(key), key) != 1 for key in required_flags):
        raise FinalPackageGateError("Full result scientific validity is incomplete")
    if _integer(result.get("thermal_required_missing_count"), "thermal missing count") != 0:
        raise FinalPackageGateError("Full result has missing thermal targets")
    return aedt_path, result_path, result


def _thermal_manifest_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    claimed = manifest.get("payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256", None)
    if (
        manifest.get("schema")
        != "mft-corrected-thermal-minimum-retained-package-v1"
        or not isinstance(claimed, str)
        or goal.canonical_sha256(unsigned) != claimed
    ):
        raise FinalPackageGateError("corrected thermal manifest seal drifted")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise FinalPackageGateError("corrected thermal manifest has no inventory")
    rows: dict[str, dict[str, Any]] = {}
    for raw in files:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"path", "size_bytes", "sha256"}
            or not isinstance(raw.get("path"), str)
            or raw["path"] in rows
            or HEX64.fullmatch(str(raw.get("sha256") or "")) is None
        ):
            raise FinalPackageGateError("corrected thermal manifest row drifted")
        rows[raw["path"]] = raw
    return rows


def _load_thermal_sources(
    event_path: Path,
    event: Mapping[str, Any],
    *,
    authority_plan: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    collector_root = event_path.parent.resolve(strict=True)
    collection = collector_root / "collection"
    manifest = event.get("manifest")
    if not isinstance(manifest, dict):
        raise FinalPackageGateError("corrected thermal retained manifest is absent")
    rows = _thermal_manifest_rows(manifest)
    collected = event.get("collected_files")
    if not isinstance(collected, list):
        raise FinalPackageGateError("corrected thermal collected inventory is absent")
    collected_rows = {
        str(row.get("path")): row for row in collected if isinstance(row, dict)
    }
    required = {"symmetric.aedt", "corrected_result.json"}
    if not required.issubset(rows) or not required.issubset(collected_rows):
        raise FinalPackageGateError("corrected thermal result/AEDT was not byte-collected")
    for relative in required:
        if rows[relative] != collected_rows[relative]:
            raise FinalPackageGateError("corrected thermal inventory cross-binding drifted")
    symmetric = _is_contained_file(
        collection / "artifacts" / "symmetric.aedt",
        collection,
        "corrected symmetric AEDT",
    )
    corrected_path = _is_contained_file(
        collection / "artifacts" / "corrected_result.json",
        collection,
        "corrected thermal result",
    )
    _record_matches(symmetric, rows["symmetric.aedt"], "corrected symmetric AEDT")
    _record_matches(
        corrected_path,
        rows["corrected_result.json"],
        "corrected thermal result",
    )
    result = _read_json(corrected_path, "corrected thermal result")
    source = result.get("source_provenance")
    executor = result.get("executor_provenance")
    imported = executor.get("imported") if isinstance(executor, dict) else None
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("candidate_sha256") != LOGICAL_CANDIDATE_SHA256
        or manifest.get("source_solver_revision") != SOURCE_SOLVER_REVISION
        or manifest.get("executor_solver_revision") != THERMAL_EXECUTOR_REVISION
        or result.get("schema") != "mft-corrected-thermal-diagnostic-result-v1"
        or result.get("diagnostic_only") is not True
        or result.get("canonical") is not False
        or not isinstance(source, dict)
        or source.get("candidate_sha256") != LOGICAL_CANDIDATE_SHA256
        or source.get("solver_revision") != SOURCE_SOLVER_REVISION
        or source.get("library_revision") != LIBRARY_REVISION
        or not isinstance(imported, dict)
        or imported.get("executor_solver_revision") != THERMAL_EXECUTOR_REVISION
        or imported.get("executor_solver_dirty") != 0
        or imported.get("pyaedt_library_revision") != LIBRARY_REVISION
        or imported.get("pyaedt_library_dirty") != 0
        or authority_plan.get("candidate_physics_sha256")
        != source.get("candidate_sha256")
    ):
        raise FinalPackageGateError("corrected thermal candidate/solver binding drifted")
    plan_record = event["contract"].get("local_evidence", {}).get("plan")
    if not isinstance(plan_record, dict):
        raise FinalPackageGateError("corrected thermal plan authority is absent")
    plan_path = Path(str(plan_record.get("path") or ""))
    _record_matches(plan_path, plan_record, "corrected thermal execution plan")
    thermal_plan = _read_json(plan_path, "corrected thermal execution plan")
    if (
        thermal_plan.get("task_identity", {}).get("candidate_sha256")
        != LOGICAL_CANDIDATE_SHA256
        or thermal_plan.get("contract", {}).get("fixed_physics")
        != FIXED_THERMAL_BOUNDARY
        or thermal_plan.get("executor", {}).get("revision")
        != THERMAL_EXECUTOR_REVISION
        or thermal_plan.get("contract", {}).get("library_revision")
        != LIBRARY_REVISION
    ):
        raise FinalPackageGateError("corrected thermal execution plan drifted")
    return symmetric, corrected_path, result, thermal_plan


def _temperature_targets(result: Mapping[str, Any], label: str) -> dict[str, float]:
    temperatures = result.get("temperatures")
    source: Mapping[str, Any] = (
        temperatures if isinstance(temperatures, dict) else result
    )
    return {
        name: _finite(source.get(name), f"{label} {name}")
        for name in ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core")
    }


def _constraint_truth(
    full_result: Mapping[str, Any],
    thermal_result: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = thermal_result.get("native_fixed_readback")
    if (
        not isinstance(fixed, dict)
        or fixed.get("schema")
        != "mft-corrected-thermal-native-fixed-readback-v2"
        or fixed.get("passed") is not True
    ):
        raise FinalPackageGateError("native fixed thermal readback is absent")
    fan = fixed.get("fan_boundary")
    tim = fixed.get("tim_material")
    geometry = fixed.get("geometry")
    if (
        not isinstance(fan, dict)
        or fan.get("passed") is not True
        or fan.get("name") != "fan_inlet"
        or fan.get("velocity_vector_m_per_s")
        != {"X": 0.0, "Y": -1.5, "Z": 0.0}
        or not isinstance(tim, dict)
        or not isinstance(geometry, dict)
        or geometry.get("passed") is not True
    ):
        raise FinalPackageGateError("native cooling/geometry readback drifted")
    _exact_float(
        tim.get("thermal_conductivity_W_mK"),
        0.2,
        "native TIM conductivity",
    )
    pad_solids = geometry.get("physical_pad_solids")
    pad_families = geometry.get("physical_pad_families")
    if (
        not isinstance(pad_solids, dict)
        or not pad_solids
        or not isinstance(pad_families, dict)
        or set(pad_families) != {"core_plate", "wcp"}
        or any(not pad_families[name] for name in pad_families)
    ):
        raise FinalPackageGateError("native thermal pad inventory is incomplete")
    for name, row in pad_solids.items():
        if (
            not isinstance(row, dict)
            or row.get("material") != "thermal_pad"
        ):
            raise FinalPackageGateError(f"native thermal pad drifted: {name}")
        _exact_float(
            row.get("y_thickness_mm"),
            2.0,
            f"native thermal pad thickness {name}",
        )
    dimensions = geometry.get("dimensions_mm")
    if not isinstance(dimensions, dict):
        raise FinalPackageGateError("native dimensions are absent")
    dimension_values = {
        "W": _finite(dimensions.get("width_x"), "native width"),
        "L": _finite(dimensions.get("length_y"), "native length"),
        "H": _finite(dimensions.get("height_z"), "native height"),
    }
    dimension_pass = {
        axis: 0.0 < dimension_values[axis] <= goal.GOAL_SIZE_LIMITS_MM[axis]
        for axis in ("W", "L", "H")
    }
    resonance = _finite(
        full_result.get("f_res_min_tx_rx_only_Hz"),
        "Full actual resonance",
    )
    lane_temperatures = {
        "full": _temperature_targets(full_result, "Full result"),
        "symmetric_corrected": _temperature_targets(
            thermal_result,
            "corrected symmetric result",
        ),
    }
    winding_targets = {
        lane: {
            name: values[name]
            for name in ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side")
        }
        for lane, values in lane_temperatures.items()
    }
    winding_max = max(
        value
        for values in winding_targets.values()
        for value in values.values()
    )
    core_targets = {
        lane: values["T_max_core"] for lane, values in lane_temperatures.items()
    }
    core_max = max(core_targets.values())
    observation = thermal_result.get("constraint_observation")
    corrected_winding_max = max(winding_targets["symmetric_corrected"].values())
    if (
        not isinstance(observation, dict)
        or not math.isclose(
            _finite(observation.get("winding_max_c"), "reported winding max"),
            corrected_winding_max,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _finite(observation.get("core_max_c"), "reported core max"),
            core_targets["symmetric_corrected"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or _finite(observation.get("winding_limit_c"), "reported winding limit")
        != 100.0
        or _finite(observation.get("core_limit_c"), "reported core limit")
        != 120.0
    ):
        raise FinalPackageGateError("corrected thermal constraint observation drifted")
    convergence = thermal_result.get("convergence")
    parallel = thermal_result.get("parallel_attestation")
    if (
        not isinstance(convergence, dict)
        or _integer(convergence.get("thermal_converged"), "corrected convergence")
        != 1
        or not isinstance(parallel, dict)
        or parallel.get("passed") is not True
    ):
        raise FinalPackageGateError("corrected thermal solve attestation is incomplete")
    checks = {
        "dimensions": {
            "actual_mm": dimension_values,
            "limits_mm": dict(goal.GOAL_SIZE_LIMITS_MM),
            "axis_pass": dimension_pass,
            "pass": all(dimension_pass.values()),
            "actual_native_geometry": True,
        },
        "resonance": {
            "actual_hz": resonance,
            "minimum_hz": goal.GOAL_RESONANCE_MIN_HZ,
            "pass": resonance >= goal.GOAL_RESONANCE_MIN_HZ,
            "source_model": "full",
            "predicted_or_surrogate": False,
        },
        "winding_temperature": {
            "actual_targets_c": winding_targets,
            "actual_maximum_c": winding_max,
            "maximum_c": 100.0,
            "pass": winding_max <= 100.0,
            "both_model_lanes_required": True,
        },
        "core_temperature": {
            "actual_targets_c": core_targets,
            "actual_maximum_c": core_max,
            "maximum_c": 120.0,
            "pass": core_max <= 120.0,
            "both_model_lanes_required": True,
        },
        "fixed_cooling": {
            "expected": copy.deepcopy(FIXED_THERMAL_BOUNDARY),
            "fan_native_readback": copy.deepcopy(fan),
            "tim_native_readback": copy.deepcopy(tim),
            "pad_native_readback_count": len(pad_solids),
            "pass": True,
        },
    }
    checks["all_explicit_goal_constraints_actual_pass"] = all(
        checks[name]["pass"]
        for name in (
            "dimensions",
            "resonance",
            "winding_temperature",
            "core_temperature",
            "fixed_cooling",
        )
    )
    return checks


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    try:
        aedt_transport._atomic_bytes(path, _canonical_bytes(value, newline=True))
    except (OSError, aedt_transport.CollectionError) as exc:
        raise FinalPackageGateError(f"cannot write {path.name} atomically") from exc


def _ensure_output_root(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
    try:
        root = diagnostic_handoff._real_directory(path, label="final output root")
    except (OSError, diagnostic_handoff.HandoffError) as exc:
        raise FinalPackageGateError("final output root is unsafe") from exc
    allowed = {"pending_manifest.json", PACKAGE_NAME}
    unexpected = sorted(item.name for item in root.iterdir() if item.name not in allowed)
    if unexpected:
        raise FinalPackageGateError(
            "final output root has unexpected members: " + ", ".join(unexpected)
        )
    return root


def _write_pending(
    output_root: Path,
    *,
    reasons: Sequence[str],
    input_records: Mapping[str, Any],
    status: str = "pending",
) -> dict[str, Any]:
    if (output_root / PACKAGE_NAME).exists():
        raise FinalPackageGateError("cannot downgrade an existing final package")
    manifest = _seal(
        {
            "schema": PENDING_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "status": status,
            "created_at_utc": _now(),
            "final_package_created": False,
            "full_aedt_promoted": False,
            "symmetric_aedt_promoted": False,
            "solver_result_truth_included": False,
            "open_only_diagnostic_snapshot_promotion_allowed": False,
            "production_truth_eligible": False,
            "canonical": False,
            "pending_reasons": sorted(set(str(item) for item in reasons)),
            "inputs": copy.deepcopy(dict(input_records)),
            "scheduler_methods_used": [],
            "scheduler_mutation_performed": False,
        }
    )
    path = output_root / "pending_manifest.json"
    _write_json_atomic(path, manifest)
    return {
        "status": status,
        "pending_manifest_path": str(path),
        "pending_manifest_sha256": diagnostic_handoff.sha256_file(path),
        "pending_reasons": manifest["pending_reasons"],
        "final_package_created": False,
    }


def _copy_authenticated(source: Path, destination: Path) -> dict[str, Any]:
    before = _file_record(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    after = _file_record(source)
    copied = _file_record(destination)
    if (
        before != after
        or copied["sha256"] != before["sha256"]
        or copied["size_bytes"] != before["size_bytes"]
    ):
        raise FinalPackageGateError(f"source changed during copy: {source}")
    return copied


def _inventory(root: Path) -> list[dict[str, Any]]:
    try:
        paths = diagnostic_handoff._inventory_paths(root)
    except (OSError, diagnostic_handoff.HandoffError) as exc:
        raise FinalPackageGateError("package inventory is unsafe") from exc
    return [
        _file_record(root / relative, relative_to=root)
        for relative in sorted(paths)
    ]


def _validate_existing_package(destination: Path) -> dict[str, Any]:
    try:
        root = diagnostic_handoff._real_directory(
            destination,
            label="existing final package",
        )
    except (OSError, diagnostic_handoff.HandoffError) as exc:
        raise FinalPackageGateError("existing final package is unsafe") from exc
    actual_paths = set(diagnostic_handoff._inventory_paths(root))
    if actual_paths != EXPECTED_PACKAGE_FILES:
        raise FinalPackageGateError("existing final package file set drifted")
    manifest_path = root / "package_manifest.json"
    seal_path = root / "package_seal.json"
    manifest = _read_json(manifest_path, "final package manifest")
    seal = _read_json(seal_path, "final package seal")
    truth = _read_json(root / "result_truth.json", "final result truth")
    _validate_payload_seal(manifest, schema=PACKAGE_SCHEMA)
    _validate_payload_seal(seal, schema=SEAL_SCHEMA)
    _validate_payload_seal(truth, schema=TRUTH_SCHEMA)
    if (
        manifest.get("all_explicit_goal_constraints_actual_pass") is not True
        or manifest.get("solver_result_truth_included") is not True
        or manifest.get("open_only_diagnostic_snapshot_promoted") is not False
        or truth.get("all_explicit_goal_constraints_actual_pass") is not True
        or seal.get("package_manifest_sha256")
        != diagnostic_handoff.sha256_file(manifest_path)
        or seal.get("package_manifest_payload_sha256")
        != manifest["payload_sha256"]
    ):
        raise FinalPackageGateError("existing final package truth gate drifted")
    sealed_files = seal.get("sealed_files")
    if not isinstance(sealed_files, list):
        raise FinalPackageGateError("existing final package inventory seal is absent")
    expected_without_seal = sorted(EXPECTED_PACKAGE_FILES - {"package_seal.json"})
    actual_without_seal = [
        _file_record(root / relative, relative_to=root)
        for relative in expected_without_seal
    ]
    if (
        sealed_files != actual_without_seal
        or seal.get("sealed_file_inventory_sha256")
        != _payload_sha256({"files": actual_without_seal})
    ):
        raise FinalPackageGateError("existing final package inventory seal drifted")
    return {
        "status": "already_published",
        "package_path": str(root),
        "manifest_sha256": diagnostic_handoff.sha256_file(manifest_path),
        "seal_sha256": diagnostic_handoff.sha256_file(seal_path),
        "all_explicit_goal_constraints_actual_pass": True,
    }


def _publish(
    output_root: Path,
    *,
    full_event_path: Path,
    thermal_event_path: Path,
    full_aedt: Path,
    symmetric_aedt: Path,
    full_result_path: Path,
    corrected_result_path: Path,
    authority_plan_record: Mapping[str, Any],
    full_retry_plan_record: Mapping[str, Any],
    full_event: Mapping[str, Any],
    thermal_event: Mapping[str, Any],
    thermal_plan: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    destination = output_root / PACKAGE_NAME
    if destination.exists():
        return _validate_existing_package(destination)
    staging = output_root / f".{PACKAGE_NAME}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        copied: dict[str, dict[str, Any]] = {}
        sources = {
            "evidence/full_collector_event.json": full_event_path,
            "evidence/thermal_collector_event.json": thermal_event_path,
            "models/full.aedt": full_aedt,
            "models/symmetric.aedt": symmetric_aedt,
            "results/full_result.json": full_result_path,
            "results/corrected_thermal_result.json": corrected_result_path,
        }
        for relative, source in sources.items():
            copied[relative] = _copy_authenticated(source, staging / relative)
            copied[relative]["path"] = relative
        truth = _seal(
            {
                "schema": TRUTH_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": _now(),
                "candidate_sha256": LOGICAL_CANDIDATE_SHA256,
                "effective_full_params_sha256": FULL_EFFECTIVE_PARAMS_SHA256,
                "source_solver_revision": SOURCE_SOLVER_REVISION,
                "thermal_executor_revision": THERMAL_EXECUTOR_REVISION,
                "library_revision": LIBRARY_REVISION,
                "goal_stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
                "goal_temperature_contract_sha256": (
                    goal.GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "fixed_thermal_boundary": copy.deepcopy(FIXED_THERMAL_BOUNDARY),
                "fixed_thermal_boundary_sha256": _payload_sha256(
                    FIXED_THERMAL_BOUNDARY
                ),
                "tasks": {
                    "full": {
                        "task_id": FULL_TASK_ID,
                        "slurm_job_id": full_event["task"]["slurm_job_id"],
                        "collector_event_sha256": full_event["event_sha256"],
                    },
                    "symmetric_corrected": {
                        "task_id": THERMAL_TASK_ID,
                        "slurm_job_id": thermal_event["task"]["slurm_job_id"],
                        "collector_event_sha256": thermal_event["event_sha256"],
                    },
                },
                "constraints": copy.deepcopy(dict(constraints)),
                "all_explicit_goal_constraints_actual_pass": True,
                "actual_solver_results_used": ["full", "symmetric_corrected"],
                "solver_result_truth_included": True,
                "surrogate_result_used_for_actual_pass": False,
                "open_only_diagnostic_snapshot_used": False,
                "open_only_diagnostic_snapshot_promoted": False,
                "source_tasks_diagnostic_only": True,
                "canonical": False,
                "production_truth_eligible": False,
                "automatic_promotion_allowed": False,
            }
        )
        truth_path = staging / "result_truth.json"
        _write_json_atomic(truth_path, truth)
        copied["result_truth.json"] = _file_record(
            truth_path,
            relative_to=staging,
        )
        manifest = _seal(
            {
                "schema": PACKAGE_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "package_name": PACKAGE_NAME,
                "created_at_utc": _now(),
                "package_kind": "actual_solver_result_artifacts",
                "candidate_sha256": LOGICAL_CANDIDATE_SHA256,
                "effective_full_params_sha256": FULL_EFFECTIVE_PARAMS_SHA256,
                "source_solver_revision": SOURCE_SOLVER_REVISION,
                "thermal_executor_revision": THERMAL_EXECUTOR_REVISION,
                "library_revision": LIBRARY_REVISION,
                "authority_plan": copy.deepcopy(dict(authority_plan_record)),
                "full_retry_plan": copy.deepcopy(dict(full_retry_plan_record)),
                "thermal_execution_plan_payload_sha256": thermal_plan.get(
                    "plan_payload_sha256"
                ),
                "goal_contract": {
                    "maximum_dimensions_mm": dict(goal.GOAL_SIZE_LIMITS_MM),
                    "minimum_resonance_hz": goal.GOAL_RESONANCE_MIN_HZ,
                    "maximum_winding_temperature_c": 100.0,
                    "maximum_core_temperature_c": 120.0,
                    "fixed_thermal_boundary": copy.deepcopy(
                        FIXED_THERMAL_BOUNDARY
                    ),
                },
                "result_truth": copied["result_truth.json"],
                "artifact_inventory": [
                    copied[name] for name in sorted(copied)
                ],
                "artifact_inventory_sha256": _payload_sha256(
                    {"files": [copied[name] for name in sorted(copied)]}
                ),
                "full_aedt_solver_produced": True,
                "symmetric_aedt_solver_produced": True,
                "solver_result_truth_included": True,
                "all_explicit_goal_constraints_actual_pass": True,
                "open_only_diagnostic_snapshot_promoted": False,
                "source_tasks_diagnostic_only": True,
                "canonical": False,
                "production_truth_eligible": False,
                "scheduler_methods_used": [],
                "scheduler_mutation_performed": False,
            }
        )
        manifest_path = staging / "package_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        sealed_files = [
            _file_record(staging / relative, relative_to=staging)
            for relative in sorted(
                EXPECTED_PACKAGE_FILES - {"package_seal.json"}
            )
        ]
        seal = _seal(
            {
                "schema": SEAL_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "package_name": PACKAGE_NAME,
                "created_at_utc": _now(),
                "atomic_publication": True,
                "publication_mode": "no_replace_directory_rename",
                "package_manifest_sha256": diagnostic_handoff.sha256_file(
                    manifest_path
                ),
                "package_manifest_payload_sha256": manifest["payload_sha256"],
                "sealed_files": sealed_files,
                "sealed_file_inventory_sha256": _payload_sha256(
                    {"files": sealed_files}
                ),
                "solver_result_truth_included": True,
                "all_explicit_goal_constraints_actual_pass": True,
                "open_only_diagnostic_snapshot_promoted": False,
                "source_tasks_diagnostic_only": True,
                "canonical": False,
                "production_truth_eligible": False,
            }
        )
        _write_json_atomic(staging / "package_seal.json", seal)
        if set(diagnostic_handoff._inventory_paths(staging)) != EXPECTED_PACKAGE_FILES:
            raise FinalPackageGateError("staged final package file set drifted")
        os.rename(staging, destination)
        published = _validate_existing_package(destination)
        pending = output_root / "pending_manifest.json"
        if pending.exists():
            pending.unlink()
        published["status"] = "published"
        return published
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def evaluate_and_publish(
    *,
    full_event_path: Path,
    thermal_event_path: Path,
    authority_plan_path: Path,
    full_retry_plan_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = _ensure_output_root(output_root)
    destination = root / PACKAGE_NAME
    if destination.exists():
        return _validate_existing_package(destination)
    input_records: dict[str, Any] = {}
    pending: list[str] = []
    try:
        full_event, full_pending = _validate_terminal_event(
            full_event_path,
            target="full96326",
            task_id=FULL_TASK_ID,
        )
        input_records["full_collector_event"] = _file_record(full_event_path)
        pending.extend(full_pending)
    except FinalPackageGateError as exc:
        return _write_pending(
            root,
            reasons=[f"Full collector authority invalid: {exc}"],
            input_records=input_records,
            status="authority_invalid",
        )
    try:
        thermal_event, thermal_pending = _validate_terminal_event(
            thermal_event_path,
            target="thermal96324",
            task_id=THERMAL_TASK_ID,
        )
        input_records["thermal_collector_event"] = _file_record(thermal_event_path)
        pending.extend(thermal_pending)
    except FinalPackageGateError as exc:
        return _write_pending(
            root,
            reasons=[f"thermal collector authority invalid: {exc}"],
            input_records=input_records,
            status="authority_invalid",
        )
    try:
        authority_plan, authority_record = _validate_authority_plan(
            authority_plan_path
        )
        input_records["candidate_authority_plan"] = authority_record
        _full_retry_plan, candidate, retry_record = _validate_full_retry_plan(
            full_retry_plan_path,
            authority_plan=authority_plan,
            full_event=full_event,
        )
        input_records["full_retry_plan"] = retry_record
    except FinalPackageGateError as exc:
        return _write_pending(
            root,
            reasons=[f"candidate/Full plan authority invalid: {exc}"],
            input_records=input_records,
            status="authority_invalid",
        )
    if pending:
        return _write_pending(
            root,
            reasons=pending,
            input_records=input_records,
        )
    try:
        full_aedt, full_result_path, full_result = _load_full_sources(
            full_event_path,
            full_event,
            candidate=candidate,
        )
        (
            symmetric_aedt,
            corrected_result_path,
            thermal_result,
            thermal_plan,
        ) = _load_thermal_sources(
            thermal_event_path,
            thermal_event,
            authority_plan=authority_plan,
        )
        constraints = _constraint_truth(full_result, thermal_result)
    except FinalPackageGateError as exc:
        return _write_pending(
            root,
            reasons=[f"actual result authority incomplete: {exc}"],
            input_records=input_records,
            status="authority_invalid",
        )
    if constraints["all_explicit_goal_constraints_actual_pass"] is not True:
        failed = [
            name
            for name in (
                "dimensions",
                "resonance",
                "winding_temperature",
                "core_temperature",
                "fixed_cooling",
            )
            if constraints[name]["pass"] is not True
        ]
        input_records["actual_constraint_observation_sha256"] = _payload_sha256(
            constraints
        )
        return _write_pending(
            root,
            reasons=[
                "actual goal constraints failed: " + ", ".join(failed)
            ],
            input_records=input_records,
            status="actual_constraints_failed",
        )
    return _publish(
        root,
        full_event_path=full_event_path,
        thermal_event_path=thermal_event_path,
        full_aedt=full_aedt,
        symmetric_aedt=symmetric_aedt,
        full_result_path=full_result_path,
        corrected_result_path=corrected_result_path,
        authority_plan_record=authority_record,
        full_retry_plan_record=retry_record,
        full_event=full_event,
        thermal_event=thermal_event,
        thermal_plan=thermal_plan,
        constraints=constraints,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-event", type=Path, required=True)
    parser.add_argument("--thermal-event", type=Path, required=True)
    parser.add_argument("--candidate-authority-plan", type=Path, required=True)
    parser.add_argument("--full-retry-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_and_publish(
        full_event_path=args.full_event.resolve(strict=True),
        thermal_event_path=args.thermal_event.resolve(strict=True),
        authority_plan_path=args.candidate_authority_plan.resolve(strict=True),
        full_retry_plan_path=args.full_retry_plan.resolve(strict=True),
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {"published", "already_published"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
