"""Prepare the bounded B6 recovery for rounded task96340; never submit it.

The gate opens only when the exact authenticated rounded Standard task96340
has terminal state ``failed`` and its immutable output contains both:

* the native Icepak poor-mesh-quality marker, and
* a WCP-specific ``is intersecting with MeshRegion
  wcp_pad_mesh_region_*`` marker.

A success, an active task, or any other failure cause is ineligible.  Even an
eligible result creates only a sealed local plan: this module has no Scheduler
mutation or submission command.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_collector as source_collector,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as rounded,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_submit as source_submit,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_mesh_recovery_execute as executor,
)


ContractError = rounded.PostdeadlineContractError
SOURCE_TASK_ID = 96340
SOURCE_CANDIDATE_SHA256 = rounded.SOURCE_CANDIDATE_SHA256
SOURCE_PLAN_PATH = rounded.OUTPUT_ROOT / rounded.PLAN_NAME
SOURCE_FINAL_PATH = (
    rounded.OUTPUT_ROOT
    / source_submit.SUBMISSION_DIRECTORY_NAME
    / source_submit.FINAL_NAME
)
PROJECT = rounded.PROJECT
ACCOUNT_NAME = rounded.ACCOUNT_NAME
NODE_NAME = rounded.NODE_NAME
CPUS = rounded.CPUS
MEMORY_MB = rounded.MEMORY_MB
SOLVER_SECONDS = rounded.SOLVER_SECONDS
SCHEDULER_SECONDS = rounded.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = rounded.MAX_WORKERS_PER_NODE
LIBRARY_REVISION = rounded.LIBRARY_REVISION

TASK_NAME = (
    "mft-goal-final-standard-official5-rounded-r10-s4-"
    "mesh-recovery-b6-v1-909d249ebe45"
)
WORKDIR = (
    "mft_goal_final_standard_official5_rounded_r10_s4_"
    "mesh_recovery_b6_v1_909d249ebe45"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\final_standard_official5_rounded_r10_s4_mesh_recovery_b6_prepare_v1"
)
PROFILE_PATH = (
    REPOSITORY
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_rounded_mesh_recovery_b6.json"
)
EXECUTOR_PATH = (
    REPOSITORY
    / "tools"
    / "mft_goal_official5_rounded_mesh_recovery_execute.py"
)
EXECUTOR_RELATIVE = EXECUTOR_PATH.relative_to(REPOSITORY).as_posix()

PROFILE_SCHEMA = (
    "mft-goal-diagnostic-standard-rounded-mesh-recovery-b6-profile-v1"
)
MESH_CONTRACT_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-b6-v1"
GATE_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-b6-gate-v1"
REVISION_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-b6-revision-v1"
PLAN_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-b6-prepare-plan-v1"
RECEIPT_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-b6-receipt-v1"

PLAN_NAME = "rounded_mesh_recovery_b6_prepare_plan.json"
RECEIPT_NAME = "prepare_receipt.json"
GATE_NAME = "source_task96340_gate.json"
PARAMS_NAME = "rounded_fea_params_unchanged.json"
PROFILE_NAME = "mesh_recovery_b6_execution_profile.json"

MAX_TASK_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 64 * 1024 * 1024
POOR_MESH_MARKER = "Solver failed because of poor mesh quality."
WCP_INTERSECTION = re.compile(
    r"\bis intersecting with MeshRegion "
    r"(?P<region>wcp_pad_mesh_region_\d+_(?:in|out)_[pn])\b"
)
ACTIVE_STATES = frozenset({"queued", "attaching", "attached", "running"})
SUCCESS_STATES = frozenset({"completed", "succeeded", "success"})
FAILURE_STATES = frozenset({"failed"})

sealed = rounded.sealed
validate_seal = rounded.validate_seal
payload_sha256 = rounded.payload_sha256
read_json = rounded.read_json
write_immutable_json = rounded.write_immutable_json
sha256_file = rounded.sha256_file


def _status(task: Mapping[str, Any]) -> str:
    return str(task.get("status", task.get("state", ""))).strip().casefold()


def _task_id(task: Mapping[str, Any]) -> int:
    try:
        return int(task.get("task_id", task.get("id", 0)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("task96340 identity is invalid") from exc


def _default_task_reader(task_id: int) -> dict[str, Any]:
    raw = source_collector.base.http_get(
        f"{rounded.SCHEDULER_URL.rstrip('/')}/api/tasks/{task_id}",
        max_bytes=MAX_TASK_BYTES,
        timeout=30.0,
    )
    try:
        task = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("task96340 GET is invalid JSON") from exc
    if not isinstance(task, dict):
        raise ContractError("task96340 GET is not an object")
    return task


def _default_stream_reader(task_id: int, stream: str) -> bytes:
    if stream not in {"stdout", "stderr"}:
        raise ContractError("unsupported Scheduler stream")
    return source_collector.base.http_get(
        (
            f"{rounded.SCHEDULER_URL.rstrip('/')}/api/tasks/{task_id}/"
            f"{stream}?max_bytes={MAX_STREAM_BYTES}"
        ),
        max_bytes=MAX_STREAM_BYTES,
        timeout=120.0,
    )


def _source_contract(
    loader: Callable[..., dict[str, Any]] = source_collector.load_contract,
) -> dict[str, Any]:
    contract = loader(
        plan_path=SOURCE_PLAN_PATH,
        final_path=SOURCE_FINAL_PATH,
    )
    if (
        contract.get("task_id") != SOURCE_TASK_ID
        or contract.get("task_name") != rounded.TASK_NAME
        or contract.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or contract.get("solver_revision")
        != "623a5345ae1b85b974cb248bcb0c8dfaadf1a867"
        or contract.get("library_revision") != LIBRARY_REVISION
        or contract.get("node_name") != NODE_NAME
    ):
        raise ContractError("task96340 local source contract drifted")
    return contract


def evaluate_gate(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    stdout: bytes,
    stderr: bytes,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Classify the exact source without making any external mutation."""

    if len(stdout) > MAX_STREAM_BYTES or len(stderr) > MAX_STREAM_BYTES:
        raise ContractError("task96340 stream exceeds bounded gate size")
    try:
        text = (stdout + b"\n" + stderr).decode("utf-8")
    except UnicodeError as exc:
        raise ContractError("task96340 streams are not UTF-8") from exc
    status = _status(task)
    identity_expected = {
        "task_id": SOURCE_TASK_ID,
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
    }
    identity_actual = {
        "task_id": _task_id(task),
        **{
            key: task.get(key)
            for key in identity_expected
            if key != "task_id"
        },
    }
    identity_drift = {
        key: {"expected": expected, "actual": identity_actual.get(key)}
        for key, expected in identity_expected.items()
        if identity_actual.get(key) != expected
    }
    poor_mesh = POOR_MESH_MARKER.casefold() in text.casefold()
    region_names = sorted(set(
        match.group("region")
        for match in WCP_INTERSECTION.finditer(text)
    ))
    wcp_intersection = bool(region_names)
    terminal_failed = status in FAILURE_STATES
    eligible = (
        not identity_drift
        and terminal_failed
        and poor_mesh
        and wcp_intersection
    )
    if identity_drift:
        reason = "source_identity_drift"
    elif status in ACTIVE_STATES:
        reason = "source_not_terminal"
    elif status in SUCCESS_STATES:
        reason = "source_succeeded_recovery_forbidden"
    elif not terminal_failed:
        reason = "source_not_exact_failed_state"
    elif not poor_mesh:
        reason = "failure_is_not_native_poor_mesh"
    elif not wcp_intersection:
        reason = "poor_mesh_has_no_wcp_intersection_evidence"
    else:
        reason = "exact_poor_mesh_wcp_intersection_trigger"
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ContractError("B6 gate observation time is naive")
    return sealed(
        {
            "schema_version": GATE_SCHEMA,
            "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "scheduler_get_only": True,
            "scheduler_get_calls": 3,
            "scheduler_mutation_performed": False,
            "source_task_id": SOURCE_TASK_ID,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "source_status": status,
            "source_failure_message": str(
                task.get("failure_message") or ""
            )[:4000],
            "identity_drift": identity_drift,
            "terminal_failed": terminal_failed,
            "native_poor_mesh_marker_present": poor_mesh,
            "wcp_intersection_marker_present": wcp_intersection,
            "wcp_intersection_region_names": region_names,
            "stdout": {
                "size_bytes": len(stdout),
                "sha256": hashlib.sha256(stdout).hexdigest(),
            },
            "stderr": {
                "size_bytes": len(stderr),
                "sha256": hashlib.sha256(stderr).hexdigest(),
            },
            "prepare_allowed": eligible,
            "submission_allowed": False,
            "reason": reason,
            "success_forbids_recovery": True,
            "different_failure_forbids_recovery": True,
            "recovery_attempt_budget": 1,
        }
    )


def evaluate_live_gate(
    *,
    task_reader: Callable[[int], dict[str, Any]] = _default_task_reader,
    stream_reader: Callable[[int, str], bytes] = _default_stream_reader,
    source_loader: Callable[..., dict[str, Any]] = (
        source_collector.load_contract
    ),
    observed_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _source_contract(source_loader)
    task = task_reader(SOURCE_TASK_ID)
    stdout = stream_reader(SOURCE_TASK_ID, "stdout")
    stderr = stream_reader(SOURCE_TASK_ID, "stderr")
    return contract, evaluate_gate(
        contract=contract,
        task=task,
        stdout=stdout,
        stderr=stderr,
        observed_at=observed_at,
    )


def load_recovery_profile() -> dict[str, Any]:
    profile = read_json(PROFILE_PATH.resolve(strict=True))
    source = read_json(
        (
            rounded.OUTPUT_ROOT / rounded.PROFILE_NAME
        ).resolve(strict=True)
    )
    recovery = profile.get("mesh_recovery_contract")
    if (
        profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("stage") != "standard"
        or profile.get("cpus") != CPUS
        or profile.get("mem_mb") != MEMORY_MB
        or profile.get("timeout_seconds") != SOLVER_SECONDS
        or profile.get("param_overrides")
        != source.get("param_overrides")
        or profile.get("fixed_boundary_contract")
        != source.get("fixed_boundary_contract")
        or profile.get("artifact_retention")
        != source.get("artifact_retention")
        or not isinstance(recovery, dict)
        or recovery.get("schema_version") != MESH_CONTRACT_SCHEMA
        or recovery.get("source_task_id") != SOURCE_TASK_ID
        or recovery.get("source_mesh_policy") != executor.SOURCE_POLICY
        or recovery.get("target_mesh_policy") != executor.TARGET_POLICY
        or recovery.get("source_eighth_padding_mm")
        != [0.0, 2.0, 0.0, 0.0, 2.0, 0.0]
        or recovery.get("target_eighth_padding_mm")
        != list(executor.EXPECTED_EIGHTH_PADDING_MM)
        or recovery.get("expected_wcp_mesh_region_count")
        != executor.EXPECTED_WCP_REGION_COUNT
        or recovery.get("missing_thin_solids_required") != 0
        or recovery.get("direct_analyze_forbidden") is not True
    ):
        raise ContractError("B6 recovery profile drifted")
    return profile


def attest_solver_revision(
    revision: str,
    *,
    source_reader: Callable[[str, str], bytes] = rounded.reviewed._git_show,
) -> dict[str, Any]:
    base = rounded.attest_rounded_solver_revision(
        revision, source_reader=source_reader
    )
    try:
        wrapper = source_reader(EXECUTOR_RELATIVE, revision)
        profile = source_reader(
            PROFILE_PATH.relative_to(REPOSITORY).as_posix(), revision
        )
    except Exception as exc:
        raise ContractError("B6 committed source cannot be read") from exc
    markers = (
        b"TARGET_TANGENTIAL_PADDING_MM = 1.0",
        b"EXPECTED_EIGHTH_PADDING_MM = (0.0, 1.0, 0.0, 0.0, 1.0, 0.0)",
        b"_validate_native_preflight",
        b"direct_analyze\": False",
    )
    if (
        any(marker not in wrapper for marker in markers)
        or PROFILE_SCHEMA.encode("utf-8") not in profile
        or MESH_CONTRACT_SCHEMA.encode("utf-8") not in profile
    ):
        raise ContractError("B6 revision lacks bounded recovery source")
    return sealed(
        {
            "schema_version": REVISION_SCHEMA,
            "solver_revision": revision,
            "rounded_base_attestation": base,
            "executor": {
                "path": EXECUTOR_RELATIVE,
                "sha256": hashlib.sha256(wrapper).hexdigest(),
                "size_bytes": len(wrapper),
            },
            "profile": {
                "path": PROFILE_PATH.relative_to(REPOSITORY).as_posix(),
                "sha256": hashlib.sha256(profile).hexdigest(),
                "size_bytes": len(profile),
            },
            "process_local_mesh_override_only": True,
            "scheduler_submission_capability": False,
        }
    )


@contextmanager
def _rounded_identity_patch() -> Any:
    replacements = {"TASK_NAME": TASK_NAME, "WORKDIR": WORKDIR}
    previous = {key: getattr(rounded, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(rounded, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(rounded, key, value)


def _without_direct_analyze(command: str) -> str:
    direct_flag = " --symmetry-thermal-direct-analyze"
    direct_export = (
        f'export {rounded.direct.DIRECT_ENV_NAME}='
        f'"{rounded.direct.DIRECT_ENV_TOKEN}"; '
    )
    source_timeout = "timeout --signal=TERM --kill-after=300s 43200s "
    target_timeout = (
        "timeout --signal=TERM --kill-after=300s "
        f"{SOLVER_SECONDS}s "
    )
    if (
        command.count(direct_flag) != 1
        or command.count(direct_export) != 1
        or command.count(source_timeout) != 1
    ):
        raise ContractError("rounded direct-Analyze command shape drifted")
    command = command.replace(direct_flag, "", 1)
    command = command.replace(direct_export, "", 1)
    command = command.replace(source_timeout, target_timeout, 1)
    return command


def derive_scheduler_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
    solver_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive in-process, then install the no-direct B6 launcher."""

    with _rounded_identity_patch():
        payload, environment, retained = rounded.derive_scheduler_payload(
            params, profile, solver_revision
        )
    payload = copy.deepcopy(payload)
    environment = copy.deepcopy(environment)
    retained = copy.deepcopy(retained)
    command = str(payload.get("command") or "")
    source_runner = "python run_simulation_260706.py"
    target_runner = f"python {EXECUTOR_RELATIVE}"
    if command.count(source_runner) != 1:
        raise ContractError("B6 source runner command drifted")
    command = command.replace(source_runner, target_runner, 1)
    command = _without_direct_analyze(command)
    payload["command"] = command
    if environment.pop(rounded.direct.DIRECT_ENV_NAME, None) != (
        rounded.direct.DIRECT_ENV_TOKEN
    ):
        raise ContractError("B6 direct-Analyze environment drifted")
    environment[executor.OPT_IN_ENV] = executor.OPT_IN_TOKEN
    environment[executor.PADDING_ENV] = str(
        executor.TARGET_TANGENTIAL_PADDING_MM
    )
    validate_recovery_payload(
        payload,
        environment,
        retained,
        params=params,
        profile=profile,
        solver_revision=solver_revision,
    )
    return payload, environment, retained


def validate_recovery_payload(
    payload: Mapping[str, Any],
    environment: Mapping[str, Any],
    retained: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    solver_revision: str,
) -> None:
    effective = {**params, **profile.get("param_overrides", {})}
    rounded._validate_effective_contract(effective)
    command = str(payload.get("command") or "")
    expected = {
        "name": TASK_NAME,
        "project": PROJECT,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if (
        drift
        or "same_node_as_task_id" in payload
        or "requested_allocation_id" in payload
        or command.count(f"python {EXECUTOR_RELATIVE}") != 1
        or "python run_simulation_260706.py" in command
        or "--symmetry-thermal-direct-analyze" in command
        or "--full" in command
        or command.count(
            "timeout --signal=TERM --kill-after=300s "
            f"{SOLVER_SECONDS}s "
        )
        != 1
        or "timeout --signal=TERM --kill-after=300s 43200s " in command
        or solver_revision not in command
        or rounded.direct.DIRECT_ENV_NAME in environment
        or environment.get(executor.OPT_IN_ENV) != executor.OPT_IN_TOKEN
        or environment.get(executor.PADDING_ENV) != "1.0"
        or retained.get("stage") != "standard"
        or not str(retained.get("artifact_path", "")).endswith(
            "/symmetric.aedt"
        )
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
    ):
        raise ContractError(f"B6 payload/retention drifted: {drift}")


def _record(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ContractError("B6 artifact escapes output root") from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def prepare(
    *,
    solver_revision: str,
    output: Path = OUTPUT_ROOT,
    gate_loader: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        evaluate_live_gate
    ),
    revision_attester: Callable[[str], dict[str, Any]] = (
        attest_solver_revision
    ),
    payload_builder: Callable[
        [dict[str, Any], dict[str, Any], str],
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = derive_scheduler_payload,
    observed_at: datetime | None = None,
) -> Path:
    """Seal one prepare-only package after the exact terminal gate."""

    contract, gate = gate_loader(observed_at=observed_at)
    validate_seal(gate, GATE_SCHEMA)
    if gate.get("prepare_allowed") is not True:
        raise ContractError(
            f"B6 recovery prepare forbidden: {gate.get('reason')}"
        )
    revision = revision_attester(solver_revision)
    validate_seal(revision, REVISION_SCHEMA)
    source_plan = contract.get("plan")
    if not isinstance(source_plan, dict):
        raise ContractError("task96340 source plan is absent")
    source_root = Path(contract["plan_path"]).resolve(strict=True).parent
    params_path = rounded._contained(
        source_root,
        source_plan.get("rounded_fea_params"),
        "task96340 rounded params",
    )
    params = read_json(params_path)
    if payload_sha256(params) != executor.EXPECTED_PARAMS_SHA256:
        raise ContractError("task96340 rounded params drifted")
    profile = load_recovery_profile()
    payload, environment, retained = payload_builder(
        params, profile, solver_revision
    )
    validate_recovery_payload(
        payload,
        environment,
        retained,
        params=params,
        profile=profile,
        solver_revision=solver_revision,
    )
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ContractError("B6 prepare time is naive")
    target = output.resolve()
    if target.exists():
        raise ContractError(f"immutable B6 output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        gate_path = write_immutable_json(staging / GATE_NAME, gate)
        params_out = write_immutable_json(staging / PARAMS_NAME, params)
        profile_out = write_immutable_json(staging / PROFILE_NAME, profile)
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                "created_at_utc": now.astimezone(timezone.utc).isoformat(),
                "prepare_only": True,
                "scheduler_get_calls": 3,
                "scheduler_post_calls": 0,
                "scheduler_mutation_performed": False,
                "submission_capability_present": False,
                "submission_authorized": False,
                "source_task_id": SOURCE_TASK_ID,
                "source_candidate_physics_sha256": (
                    SOURCE_CANDIDATE_SHA256
                ),
                "source_plan": {
                    "path": str(Path(contract["plan_path"]).resolve()),
                    "sha256": contract["plan_file_sha256"],
                },
                "source_submission": {
                    "path": str(
                        Path(contract["submission_path"]).resolve()
                    ),
                    "sha256": contract["submission_file_sha256"],
                },
                "trigger_gate": _record(staging, gate_path),
                "rounded_fea_params_unchanged": _record(
                    staging, params_out
                ),
                "rounded_fea_params_sha256": payload_sha256(params),
                "mesh_recovery_profile": _record(
                    staging, profile_out
                ),
                "solver_revision_attestation": revision,
                "solver_revision": solver_revision,
                "library_revision": LIBRARY_REVISION,
                "mesh_recovery_contract": copy.deepcopy(
                    profile["mesh_recovery_contract"]
                ),
                "fixed_boundary_contract": copy.deepcopy(
                    profile["fixed_boundary_contract"]
                ),
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(
                    environment
                ),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "bounded_recovery_policy": {
                    "attempt_budget": 1,
                    "same_candidate": True,
                    "same_geometry": True,
                    "same_turns": True,
                    "round_corner": 1,
                    "corner_radius_mm": 10.0,
                    "corner_segments": 4,
                    "full_model": 0,
                    "thermal_symmetry": "eighth",
                    "fan_velocity_m_s": 1.5,
                    "thermal_pad_conductivity_W_mK": 0.2,
                    "core_plate_pad_t_mm": 2.0,
                    "wcp_pad_t_mm": 2.0,
                    "only_numerical_delta": (
                        "nonzero WCP tangential MeshRegion padding "
                        "2.0mm -> 1.0mm"
                    ),
                    "automatic_full_trigger": False,
                },
            }
        )
        plan_path = write_immutable_json(staging / PLAN_NAME, plan)
        receipt = sealed(
            {
                "schema_version": RECEIPT_SCHEMA,
                "created_at_utc": now.astimezone(timezone.utc).isoformat(),
                "prepare_only": True,
                "plan": _record(staging, plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "source_task_id": SOURCE_TASK_ID,
                "source_candidate_physics_sha256": (
                    SOURCE_CANDIDATE_SHA256
                ),
                "scheduler_get_calls": 3,
                "scheduler_post_calls": 0,
                "scheduler_mutation_performed": False,
                "submission_capability_present": False,
                "recovery_attempt_budget": 1,
            }
        )
        write_immutable_json(staging / RECEIPT_NAME, receipt)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser(
        "gate", help="GET-only classification of task96340"
    )
    gate.add_argument("--observed-at", default="")
    prepare_parser = commands.add_parser(
        "prepare", help="seal a local package only when the gate is exact"
    )
    prepare_parser.add_argument("--solver-revision", required=True)
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser


def _parse_observed_at(value: str) -> datetime | None:
    if not value:
        return None
    observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if observed.tzinfo is None:
        raise ContractError("--observed-at must be timezone-aware")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    observed_at = _parse_observed_at(getattr(args, "observed_at", ""))
    if args.command == "gate":
        _contract, result = evaluate_live_gate(observed_at=observed_at)
    else:
        path = prepare(
            solver_revision=args.solver_revision,
            output=args.output,
            observed_at=observed_at,
        )
        plan = validate_seal(read_json(path), PLAN_SCHEMA)
        result = {
            "schema_version": RECEIPT_SCHEMA,
            "event": "rounded_mesh_recovery_b6_prepared",
            "plan": str(path),
            "plan_payload_sha256": plan["payload_sha256"],
            "source_task_id": SOURCE_TASK_ID,
            "scheduler_get_calls": 3,
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
            "submission_capability_present": False,
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_mesh_recovery_b6_prepare_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scheduler_post_calls": 0,
                    "scheduler_mutation_performed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
