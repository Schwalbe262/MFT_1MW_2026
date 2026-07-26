"""GET-only collector for the single rounded candidate-5 Full task.

The collector authenticates the prepare plan and exact-once submission
receipt, verifies Full runtime/core/license provenance, validates the retained
AEDT bundle and native results manifest, reconstructs ``full_model.aedt`` from
bounded authenticated chunks, and writes an atomic local collection.  It has
no Scheduler mutation path.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib import parse


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_official5_rounded_full_prepare as prepare_only  # noqa: E402
from tools import mft_goal_official5_rounded_full_submit as submit_only  # noqa: E402
from tools import mft_goal_postdeadline_standard_collector as base  # noqa: E402
from tools import mft_goal_truth_promotion as truth  # noqa: E402


ContractError = prepare_only.ContractError
JsonReader = prepare_only.continuation.JsonReader
Getter = Callable[..., bytes]

DEFAULT_OUTPUT = prepare_only.OUTPUT_ROOT / "authenticated_get_collection"
COLLECTION_SCHEMA = "mft-goal-official5-rounded-full-collection-v1"
COLLECTION_SEAL_SCHEMA = (
    "mft-goal-official5-rounded-full-collection-seal-v1"
)
FAILURE_SCHEMA = "mft-goal-official5-rounded-full-failure-v1"
CHECKPOINT_COLLECTION_SCHEMA = (
    "mft-goal-rounded-full-diagnostic-checkpoint-collection-v1"
)
CHECKPOINT_COLLECTION_SEAL_SCHEMA = (
    "mft-goal-rounded-full-diagnostic-checkpoint-collection-seal-v1"
)
ACTIVE_STATES = {"queued", "attaching", "running"}
FAILURE_STATES = {"failed", "cancelled", "timed_out", "timeout"}
MAX_STDOUT_BYTES = 64 * 1024 * 1024


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_bytes(prepare_only.canonical_bytes(value) + b"\n")
    return path


def _write_atomic_seal(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = prepare_only.sealed(value)
    raw = prepare_only.canonical_bytes(payload) + b"\n"
    if target.exists():
        existing = prepare_only.validate_seal(
            prepare_only.read_json(target),
            str(value.get("schema_version") or ""),
        )
        if existing != payload:
            raise ContractError(f"existing {target.name} identity drifted")
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_contract(
    *,
    plan_path: Path,
    submission_path: Path,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    submission_path = submission_path.resolve(strict=True)
    plan = prepare_only.load_plan(plan_path)
    submission = prepare_only.validate_seal(
        prepare_only.read_json(submission_path),
        submit_only.RECEIPT_SCHEMA,
    )
    payload = plan["scheduler_payload"]
    retained = plan["retained_full_aedt_bundle"]
    if (
        submission.get("plan") != prepare_only._file_record(plan_path)  # noqa: SLF001
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
        or submission.get("source_standard_task_id")
        != prepare_only.SOURCE_STANDARD_TASK_ID
        or submission.get("source_candidate_physics_sha256")
        != prepare_only.SOURCE_CANDIDATE_SHA256
        or submission.get("task_name") != prepare_only.TASK_NAME
        or submission.get("dedupe_key") != payload["dedupe_key"]
        or submission.get("maximum_scheduler_posts") != 1
        or submission.get("scheduler_post_attempts_consumed") != 1
        or submission.get("scheduler_repository_modified") is not False
        or submission.get("mft_and_scheduler_functionality_mixed")
        is not False
        or retained.get("stage") != "full"
        or not str(retained.get("artifact_path") or "").endswith(
            "/full_model.aedt"
        )
    ):
        raise ContractError("rounded Full plan/submission identity drifted")
    task_id = submission.get("full_task_id")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
    ):
        raise ContractError("rounded Full task ID is invalid")
    profile_path = prepare_only.rounded._contained(  # noqa: SLF001
        plan_path.parent, plan["full_profile"], "Full profile"
    )
    params_path = prepare_only.rounded._contained(  # noqa: SLF001
        plan_path.parent, plan["full_params"], "Full params"
    )
    profile = prepare_only.read_json(profile_path)
    params = prepare_only.read_json(params_path)
    return {
        "plan": plan,
        "submission": submission,
        "plan_path": plan_path,
        "submission_path": submission_path,
        "task_id": task_id,
        "task_name": prepare_only.TASK_NAME,
        "dedupe_key": payload["dedupe_key"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "profile": profile,
        "params": params,
        "retained": retained,
        "checkpoint": plan["diagnostic_geometry_setup_checkpoint"],
        "target_lane": plan["target_lane"],
        "scheduler_url": prepare_only.SCHEDULER_URL.rstrip("/"),
    }


def _checkpoint_metadata(
    contract: Mapping[str, Any],
    *,
    getter: Getter = base.http_get,
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    expected = contract["checkpoint"]
    receipt_raw = base.remote_get(
        contract,
        expected["receipt_path"],
        1024 * 1024,
        getter=getter,
    )
    marker_raw = base.remote_get(
        contract,
        expected["marker_path"],
        1024 * 1024,
        getter=getter,
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
        marker = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(
            "rounded Full checkpoint metadata is invalid JSON"
        ) from exc
    required = {
        "schema_version",
        "stage",
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "transport_schema_version",
        "transport_encoding",
        "transport_chunk_directory",
        "transport_raw_chunk_bytes",
        "transport_max_encoded_chunk_bytes",
        "transport_chunk_count",
        "marker_path",
        "marker_sha256",
        "marker_contract_sha256",
        "source_project_filename",
        "source_standard_task_id",
        "source_candidate_physics_sha256",
        "solver_revision",
        "library_revision",
        "profile_sha256",
        "parameter_digest",
        "rounding_policy",
        "fixed_boundary",
        "checkpoint_created_before_full_solve",
        "terminal_success_required_for_collection",
        "scientific_pass",
        "thermal_pass",
        "production_promotion_eligible",
        "diagnostic_only",
    }
    integer_fields_valid = (
        type(receipt.get("artifact_size_bytes")) is int
        and 0
        < receipt["artifact_size_bytes"]
        <= scheduler_client.RETAINED_AEDT_MAX_BYTES
        and type(receipt.get("transport_chunk_count")) is int
        and receipt["transport_chunk_count"]
        == math.ceil(
            receipt["artifact_size_bytes"]
            / expected["transport_raw_chunk_bytes"]
        )
    )
    identity = {
        "schema_version": expected["receipt_schema_version"],
        "stage": expected["stage"],
        "artifact_path": expected["artifact_path"],
        "transport_schema_version": expected[
            "transport_schema_version"
        ],
        "transport_encoding": expected["transport_encoding"],
        "transport_chunk_directory": expected["chunk_directory"],
        "transport_raw_chunk_bytes": expected[
            "transport_raw_chunk_bytes"
        ],
        "transport_max_encoded_chunk_bytes": expected[
            "transport_max_encoded_chunk_bytes"
        ],
        "marker_path": expected["marker_path"],
        "marker_contract_sha256": expected[
            "marker_contract_sha256"
        ],
        "source_standard_task_id": expected[
            "source_standard_task_id"
        ],
        "source_candidate_physics_sha256": expected[
            "source_candidate_physics_sha256"
        ],
        "solver_revision": expected["solver_revision"],
        "library_revision": expected["library_revision"],
        "profile_sha256": expected["profile_sha256"],
        "parameter_digest": expected["parameter_digest"],
        "rounding_policy": expected["rounding_policy"],
        "fixed_boundary": expected["fixed_boundary"],
        "checkpoint_created_before_full_solve": True,
        "terminal_success_required_for_collection": False,
        "scientific_pass": False,
        "thermal_pass": False,
        "production_promotion_eligible": False,
        "diagnostic_only": True,
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required
        or not integer_fields_valid
        or any(receipt.get(key) != value for key, value in identity.items())
        or not isinstance(receipt.get("artifact_sha256"), str)
        or len(receipt["artifact_sha256"]) != 64
        or prepare_only.payload_sha256(
            expected["marker_contract"]
        )
        != expected["marker_contract_sha256"]
    ):
        raise ContractError(
            "rounded Full checkpoint receipt identity drifted"
        )
    marker_contract = dict(marker) if isinstance(marker, dict) else {}
    marker_contract.pop("created_at", None)
    if (
        marker_contract != expected["marker_contract"]
        or prepare_only.payload_sha256(marker_contract)
        != expected["marker_contract_sha256"]
        or base.sha256_bytes(marker_raw) != receipt["marker_sha256"]
    ):
        raise ContractError("rounded Full checkpoint marker drifted")
    return receipt, marker, receipt_raw, marker_raw


def collect_diagnostic_checkpoint(
    *,
    contract: Mapping[str, Any],
    output: Path,
    getter: Getter = base.http_get,
    remote_fetcher: Any = production._fetch_remote_to_path,  # noqa: SLF001
) -> dict[str, Any]:
    """Collect pre-solve geometry/setup even when the Full task failed."""

    receipt, marker, receipt_raw, marker_raw = _checkpoint_metadata(
        contract, getter=getter
    )
    destination = output.resolve()
    if destination.exists():
        seal = prepare_only.validate_seal(
            prepare_only.read_json(destination / "collection_seal.json"),
            CHECKPOINT_COLLECTION_SEAL_SCHEMA,
        )
        if seal.get("task_id") != contract["task_id"]:
            raise ContractError(
                "existing diagnostic checkpoint identity drifted"
            )
        return {
            "event": "rounded_full_checkpoint_already_collected",
            "task_id": contract["task_id"],
            "output": str(destination),
            "scientific_pass": False,
            "production_promotion_eligible": False,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        artifact = (
            staging / "full_model_geometry_setup_checkpoint.aedt"
        )
        remote_fetcher(
            scheduler_url=contract["scheduler_url"],
            task_id=contract["task_id"],
            relative_path=receipt["artifact_path"],
            transport_chunk_directory=receipt[
                "transport_chunk_directory"
            ],
            transport_raw_chunk_bytes=receipt[
                "transport_raw_chunk_bytes"
            ],
            transport_max_encoded_chunk_bytes=receipt[
                "transport_max_encoded_chunk_bytes"
            ],
            transport_chunk_count=receipt["transport_chunk_count"],
            expected_size=receipt["artifact_size_bytes"],
            expected_sha256=receipt["artifact_sha256"],
            destination=artifact,
        )
        if (
            artifact.stat().st_size != receipt["artifact_size_bytes"]
            or prepare_only.sha256_file(artifact)
            != receipt["artifact_sha256"]
        ):
            raise ContractError(
                "diagnostic checkpoint reconstruction drifted"
            )
        (staging / "remote_checkpoint_receipt.json").write_bytes(
            receipt_raw
        )
        (staging / "remote_checkpoint_marker.json").write_bytes(marker_raw)
        collection = prepare_only.sealed(
            {
                "schema_version": CHECKPOINT_COLLECTION_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": contract["task_id"],
                "task_name": contract["task_name"],
                "source_standard_task_id": (
                    prepare_only.SOURCE_STANDARD_TASK_ID
                ),
                "source_candidate_physics_sha256": (
                    prepare_only.SOURCE_CANDIDATE_SHA256
                ),
                "solver_revision": contract["solver_revision"],
                "library_revision": contract["library_revision"],
                "rounded_identity": copy.deepcopy(
                    contract["checkpoint"]["rounding_policy"]
                ),
                "fixed_boundary": copy.deepcopy(
                    contract["checkpoint"]["fixed_boundary"]
                ),
                "checkpoint_stage": (
                    "full_model_geometry_setup_pre_solve"
                ),
                "retained_checkpoint_aedt": {
                    "path": artifact.name,
                    "sha256": prepare_only.sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                },
                "remote_receipt_payload_sha256": (
                    prepare_only.payload_sha256(receipt)
                ),
                "remote_marker": marker,
                "scheduler_terminal_success_required": False,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
                "scientific_result_available": False,
                "scientific_pass": False,
                "thermal_pass": False,
                "production_promotion_eligible": False,
                "production_package_eligible": False,
                "diagnostic_only": True,
            }
        )
        collection_path = _write_json(
            staging / "collection_receipt.json", collection
        )
        seal = prepare_only.sealed(
            {
                "schema_version": CHECKPOINT_COLLECTION_SEAL_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": contract["task_id"],
                "collection_receipt": {
                    "path": collection_path.name,
                    "sha256": prepare_only.sha256_file(collection_path),
                    "size_bytes": collection_path.stat().st_size,
                },
                "artifact_sha256": receipt["artifact_sha256"],
                "scientific_pass": False,
                "production_promotion_eligible": False,
                "atomic_directory_collection": True,
                "scheduler_get_only": True,
            }
        )
        _write_json(staging / "collection_seal.json", seal)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "event": "rounded_full_checkpoint_collected",
        "task_id": contract["task_id"],
        "output": str(destination),
        "artifact_sha256": receipt["artifact_sha256"],
        "scientific_pass": False,
        "thermal_pass": False,
        "production_promotion_eligible": False,
        "scheduler_post_calls": 0,
    }


def get_task(
    contract: Mapping[str, Any],
    *,
    reader: JsonReader = prepare_only.continuation.get_json,
) -> dict[str, Any]:
    task = reader(f"/api/tasks/{contract['task_id']}", None)
    lane = contract["target_lane"]
    expected = {
        "task_id": contract["task_id"],
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": prepare_only.PROJECT,
        "requested_account_name": lane["account_name"],
        "requested_node_name": lane["node_name"],
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": 0,
        "requested_allocation_id": 0,
        "cpus": prepare_only.CPUS,
        "memory_mb": prepare_only.MEMORY_MB,
        "timeout_seconds": prepare_only.SCHEDULER_TIMEOUT_SECONDS,
        "max_workers_per_node": prepare_only.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": value, "actual": task.get(key)}
        for key, value in expected.items()
        if not isinstance(task, Mapping) or task.get(key) != value
    }
    status = str(task.get("status") or "")
    if drift or status not in (
        ACTIVE_STATES | FAILURE_STATES | {"completed", "succeeded"}
    ):
        raise ContractError(
            f"rounded Full Scheduler task identity drifted: {drift}"
        )
    if status not in {"queued"} and (
        task.get("account_name") != lane["account_name"]
        or task.get("actual_node_name") != lane["node_name"]
        or task.get("strict_node_placement") is not True
        or task.get("placement_contract_satisfied") is not True
    ):
        raise ContractError("rounded Full strict placement drifted")
    return copy.deepcopy(dict(task))


def _stdout_result(
    contract: Mapping[str, Any],
    *,
    getter: Getter = base.http_get,
) -> tuple[dict[str, Any], bytes]:
    query = parse.urlencode({"max_bytes": MAX_STDOUT_BYTES})
    stdout = getter(
        f"{contract['scheduler_url']}/api/tasks/"
        f"{contract['task_id']}/stdout?{query}",
        max_bytes=MAX_STDOUT_BYTES,
        timeout=120.0,
    )
    try:
        text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise ContractError("rounded Full stdout is not UTF-8") from exc
    result = None
    library = None
    for line in reversed(text.splitlines()):
        if result is None and line.startswith("RESULT_JSON "):
            try:
                candidate = json.loads(line.removeprefix("RESULT_JSON "))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
        if library is None and line.startswith("MFT_LIBRARY_GIT_HASH "):
            library = line.removeprefix("MFT_LIBRARY_GIT_HASH ").strip()
        if result is not None and library is not None:
            break
    if result is None or library != contract["library_revision"]:
        raise ContractError("rounded Full RESULT_JSON/library marker absent")
    effective = scheduler_client.effective_verification_params(
        contract["params"], contract["profile"]
    )
    if (
        not scheduler_client.result_matches_params(
            result, effective, required_keys=set(ALL_INPUT_KEYS)
        )
        or result.get("git_hash") != contract["solver_revision"]
        or result.get("pyaedt_library_git_hash")
        != contract["library_revision"]
        or float(result.get("full_model")) != 1.0
        or str(result.get("thermal_symmetry") or "") != "full"
        or float(result.get("round_corner")) != 1.0
        or float(result.get("corner_radius")) != 10.0
        or float(result.get("corner_segments")) != 4.0
    ):
        raise ContractError("rounded Full RESULT_JSON identity drifted")
    fixed = {
        "fan_config": result.get("fan_config"),
        "fan_velocity_m_s": float(result.get("fan_velocity")),
        "thermal_pad_conductivity_W_mK": float(result.get("k_ins")),
        "core_plate_pad_t_mm": float(result.get("core_plate_pad_t")),
        "wcp_pad_t_mm": float(result.get("wcp_pad_t")),
    }
    if fixed != prepare_only.FIXED_BOUNDARY:
        raise ContractError("rounded Full fixed thermal identity drifted")
    return result, stdout


def _submission_shadow(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "full",
        "task_id": contract["task_id"],
        "task_name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "solver_revision": contract["solver_revision"],
        "library_revision": contract["library_revision"],
        "profile_sha256": contract["plan"]["full_profile_sha256"],
        "retained_aedt_bundle": contract["retained"],
        "retained_aedt": contract["retained"],
        "core_policy": contract["plan"]["core_policy"],
    }


def _actual_evidence(
    result: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (float(value) for value in dimensions)
    resonance = float(result["f_res_min_tx_rx_only_Hz"])
    active, temperatures, temperature_passed = (
        production._temperature_gate_evidence(result)  # noqa: SLF001
    )
    passed = (
        width <= float(GOAL_SIZE_LIMITS_MM["W"])
        and length <= float(GOAL_SIZE_LIMITS_MM["L"])
        and height <= float(GOAL_SIZE_LIMITS_MM["H"])
        and resonance >= float(GOAL_STAGE_SPEC["resonance_min_Hz"])
        and temperature_passed
    )
    source_dimensions = source["actual_dimensions_mm"]
    source_resonance = float(source["actual_resonance_Hz"])
    source_temperatures = source["hard_constraint_evidence"][
        "body_temperatures"
    ]["evidence"]
    deltas = {
        "volume_L": float(volume_l) - float(source["actual_volume_L"]),
        "width_mm": width - float(source_dimensions["W"]),
        "length_mm": length - float(source_dimensions["L"]),
        "height_mm": height - float(source_dimensions["H"]),
        "resonance_Hz": resonance - source_resonance,
        "temperatures_C": {
            name: float(item["actual_C"])
            - float(source_temperatures[name]["actual_C"])
            for name, item in temperatures.items()
            if name in source_temperatures
        },
    }
    return {
        "actual_volume_L": float(volume_l),
        "actual_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_resonance_Hz": resonance,
        "active_temperature_targets": active,
        "actual_temperatures": temperatures,
        "goal_constraints_passed": passed,
        "symmetric_to_full_deltas": deltas,
    }


def collect(
    *,
    contract: Mapping[str, Any],
    output: Path,
    reader: JsonReader = prepare_only.continuation.get_json,
    getter: Getter = base.http_get,
    remote_reader: Any = production._remote_bytes,  # noqa: SLF001
    manifest_reader: Any = diagnostic._remote_manifest_bytes,  # noqa: SLF001
    remote_fetcher: Any = production._fetch_remote_to_path,  # noqa: SLF001
) -> dict[str, Any]:
    task = get_task(contract, reader=reader)
    status = str(task.get("status") or "")
    state = str(task.get("state") or "")
    if status in ACTIVE_STATES:
        return {
            "event": "rounded_full_task_active",
            "task_id": contract["task_id"],
            "status": status,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    if status in FAILURE_STATES:
        checkpoint_result = None
        checkpoint_error = None
        try:
            checkpoint_result = collect_diagnostic_checkpoint(
                contract=contract,
                output=(
                    output.resolve().parent
                    / "diagnostic_checkpoint_collection"
                ),
                getter=getter,
                remote_fetcher=remote_fetcher,
            )
        except Exception as exc:
            checkpoint_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        failure_path = output.resolve().parent / "terminal_failure.json"
        _write_atomic_seal(
            failure_path,
            {
                "schema_version": FAILURE_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "task_id": contract["task_id"],
                "diagnostic_checkpoint": checkpoint_result,
                "diagnostic_checkpoint_error": checkpoint_error,
                "scientific_pass": False,
                "thermal_pass": False,
                "production_promotion_eligible": False,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            },
        )
        return {
            "event": "rounded_full_task_failed",
            "task_id": contract["task_id"],
            "failure_ledger": str(failure_path),
            "diagnostic_checkpoint": checkpoint_result,
            "diagnostic_checkpoint_error": checkpoint_error,
            "scientific_pass": False,
            "production_promotion_eligible": False,
            "scheduler_post_calls": 0,
        }
    if (status, state) != ("completed", "succeeded") or task.get(
        "exit_code"
    ) != 0:
        raise ContractError(
            f"rounded Full terminal state invalid: {status}/{state}"
        )
    result, stdout = _stdout_result(contract, getter=getter)
    shadow = _submission_shadow(contract)
    try:
        production._validate_result_core_policy(  # noqa: SLF001
            result, shadow
        )
        (
            remote_receipt,
            marker,
            results_manifest,
            results_manifest_sha,
        ) = truth._validated_full_remote_bundle(  # noqa: SLF001
            submission=shadow,
            result=result,
            scheduler_url=contract["scheduler_url"],
            remote_reader=remote_reader,
            manifest_reader=manifest_reader,
        )
    except Exception as exc:
        raise ContractError(
            "rounded Full retained bundle/provenance invalid"
        ) from exc
    plan = contract["plan"]
    authority_path = prepare_only.rounded._contained(  # noqa: SLF001
        contract["plan_path"].parent,
        plan["source_standard_success_authority"],
        "rounded Standard success authority",
    )
    source = prepare_only.validate_seal(
        prepare_only.read_json(authority_path),
        prepare_only.SOURCE_AUTHORITY_SCHEMA,
    )
    actual = _actual_evidence(result, source=source)
    destination = output.resolve()
    if destination.exists():
        seal = prepare_only.validate_seal(
            prepare_only.read_json(destination / "collection_seal.json"),
            COLLECTION_SEAL_SCHEMA,
        )
        if seal.get("task_id") != contract["task_id"]:
            raise ContractError("existing Full collection identity drifted")
        return {
            "event": "rounded_full_already_collected",
            "task_id": contract["task_id"],
            "output": str(destination),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        artifact = staging / "full_model.aedt"
        remote_fetcher(
            scheduler_url=contract["scheduler_url"],
            task_id=contract["task_id"],
            relative_path=remote_receipt["artifact_path"],
            transport_chunk_directory=remote_receipt[
                "transport_chunk_directory"
            ],
            transport_raw_chunk_bytes=remote_receipt[
                "transport_raw_chunk_bytes"
            ],
            transport_max_encoded_chunk_bytes=remote_receipt[
                "transport_max_encoded_chunk_bytes"
            ],
            transport_chunk_count=remote_receipt[
                "transport_chunk_count"
            ],
            expected_size=remote_receipt["artifact_size_bytes"],
            expected_sha256=remote_receipt["artifact_sha256"],
            destination=artifact,
        )
        if (
            artifact.stat().st_size
            != remote_receipt["artifact_size_bytes"]
            or prepare_only.sha256_file(artifact)
            != remote_receipt["artifact_sha256"]
        ):
            raise ContractError("reconstructed rounded Full AEDT drifted")
        _write_json(staging / "result.json", result)
        (staging / "scheduler_stdout.log").write_bytes(stdout)
        _write_json(staging / "scheduler_terminal_task.json", task)
        _write_json(
            staging / "remote_bundle_receipt.json", remote_receipt
        )
        _write_json(staging / "prune_protection_marker.json", marker)
        _write_json(
            staging / "full_model.aedtresults.manifest.json",
            results_manifest,
        )
        shutil.copy2(contract["plan_path"], staging / "source_plan.json")
        shutil.copy2(
            contract["submission_path"],
            staging / "source_submission_receipt.json",
        )
        files = {
            name: prepare_only._file_record(staging / name)  # noqa: SLF001
            for name in (
                "full_model.aedt",
                "result.json",
                "scheduler_stdout.log",
                "scheduler_terminal_task.json",
                "remote_bundle_receipt.json",
                "prune_protection_marker.json",
                "full_model.aedtresults.manifest.json",
                "source_plan.json",
                "source_submission_receipt.json",
            )
        }
        for record in files.values():
            record["path"] = Path(record["path"]).name
        receipt = prepare_only.sealed(
            {
                "schema_version": COLLECTION_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": contract["task_id"],
                "task_name": contract["task_name"],
                "dedupe_key": contract["dedupe_key"],
                "source_standard_task_id": (
                    prepare_only.SOURCE_STANDARD_TASK_ID
                ),
                "source_candidate_physics_sha256": (
                    prepare_only.SOURCE_CANDIDATE_SHA256
                ),
                "solver_revision": contract["solver_revision"],
                "library_revision": contract["library_revision"],
                "full_model": 1,
                "thermal_symmetry": "full",
                "rounding_policy": copy.deepcopy(
                    prepare_only.ROUNDING_POLICY
                ),
                "fixed_boundary": copy.deepcopy(
                    prepare_only.FIXED_BOUNDARY
                ),
                "runtime_core_policy_verified": True,
                "runtime_license_refresh_verified": True,
                "actual_evidence": actual,
                "retained_full_aedt": files["full_model.aedt"],
                "remote_artifact_sha256": remote_receipt[
                    "artifact_sha256"
                ],
                "remote_results_manifest_sha256": (
                    results_manifest_sha
                ),
                "remote_results_tree_sha256": remote_receipt[
                    "results_tree_sha256"
                ],
                "remote_results_file_count": remote_receipt[
                    "results_file_count"
                ],
                "remote_results_size_bytes": remote_receipt[
                    "results_size_bytes"
                ],
                "files": files,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
                "scheduler_mutation_performed": False,
                "synthetic_aedt_or_results_used": False,
            }
        )
        receipt_path = _write_json(
            staging / "collection_receipt.json", receipt
        )
        seal = prepare_only.sealed(
            {
                "schema_version": COLLECTION_SEAL_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": contract["task_id"],
                "source_candidate_physics_sha256": (
                    prepare_only.SOURCE_CANDIDATE_SHA256
                ),
                "collection_receipt": {
                    "path": "collection_receipt.json",
                    "sha256": prepare_only.sha256_file(receipt_path),
                    "size_bytes": receipt_path.stat().st_size,
                },
                "collection_receipt_payload_sha256": receipt[
                    "payload_sha256"
                ],
                "artifact_sha256": remote_receipt["artifact_sha256"],
                "results_tree_sha256": remote_receipt[
                    "results_tree_sha256"
                ],
                "atomic_directory_collection": True,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            }
        )
        _write_json(staging / "collection_seal.json", seal)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "event": "rounded_full_collected",
        "task_id": contract["task_id"],
        "output": str(destination),
        "artifact_sha256": remote_receipt["artifact_sha256"],
        "results_tree_sha256": remote_receipt["results_tree_sha256"],
        "goal_constraints_passed": actual["goal_constraints_passed"],
        "scheduler_post_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=prepare_only.OUTPUT_ROOT / prepare_only.PLAN_NAME,
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=(
            prepare_only.OUTPUT_ROOT
            / submit_only.SUBMISSION_DIRECTORY
            / submit_only.RECEIPT_NAME
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_contract(
        plan_path=args.plan, submission_path=args.submission
    )
    result = collect(contract=contract, output=args.output)
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_full_collector_error",
                    "error": str(exc),
                    "scheduler_get_only": True,
                    "scheduler_post_calls": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
