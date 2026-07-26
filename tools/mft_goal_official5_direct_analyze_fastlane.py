"""Fail-closed one-shot Standard 1/8 direct-Analyze same-node fast lane.

The immutable source authority is official candidate #5 and its submitted
task96332 package.  A new plan may only co-locate with that still-running
source task on its exact ``jji0930/n115`` allocation.  The plan is diagnostic,
post-deadline, noncanonical, Standard only, and retains a unique symmetric
AEDT/result bundle.

``prepare`` requires an explicit committed solver revision.  The revision is
read with ``git show`` and must contain the versioned direct-Analyze source
path in both the thermal module and CLI before a payload can be sealed.

``submit`` is a separate exact-authorization action.  It repeats storage,
account, node, license, health, collision, and capacity GET gates before and
inside the campaign lock.  An immutable intent and exclusive attempt ledger
are written before the sole possible POST.  Any ambiguous outcome consumes
the lifetime budget and can only be reconciled by exact GET; it is never
reposted.
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
    mft_goal_official_standard_hedge_prepare as source_prepare,
)
from tools import (  # noqa: E402
    mft_goal_official_standard_hedge_submit as source_submit,
)


reviewed = source_prepare.reviewed
PostdeadlineContractError = source_prepare.PostdeadlineContractError
JsonReader = source_prepare.JsonReader
Poster = Callable[
    [str, Mapping[str, Any]],
    tuple[int | None, dict[str, Any] | None, str | None],
]

CAMPAIGN_ID = source_prepare.CAMPAIGN_ID
SCHEDULER_URL = source_prepare.SCHEDULER_URL
PROJECT = source_prepare.PROJECT
CPUS = source_prepare.CPUS
MEMORY_MB = source_prepare.MEMORY_MB
SOLVER_SECONDS = source_prepare.SOLVER_SECONDS
KILL_GRACE_SECONDS = source_prepare.KILL_GRACE_SECONDS
RETENTION_SECONDS = source_prepare.RETENTION_SECONDS
SCHEDULER_SECONDS = source_prepare.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = 2
PRIORITY = source_prepare.PRIORITY
LIBRARY_REVISION = source_prepare.LIBRARY_REVISION
ORIGINAL_DEADLINE_UTC = source_prepare.ORIGINAL_DEADLINE_UTC
OPENING_QUEUE_REASON = source_prepare.OPENING_QUEUE_REASON
FIXED_BOUNDARY = copy.deepcopy(source_prepare.FIXED_BOUNDARY)
SAFETY_FLAGS = copy.deepcopy(source_prepare.SAFETY_FLAGS)

SOURCE_SPEC = next(
    spec for spec in source_prepare.CANDIDATES if spec.selection_order == 5
)
SOURCE_TASK_ID = 96332
SOURCE_TASK_NAME = SOURCE_SPEC.task_name
SOURCE_TASK_DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-postdeadline-official5-"
    "s95913-909d249ebe45-n115:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:d71e655337a09462"
)
SOURCE_PLAN_SHA256 = (
    "46a3f43c4b31e6dd0c31d777b7a93aa105badfe18f65923c0afb43d93a5741b4"
)
SOURCE_PLAN_PAYLOAD_SHA256 = (
    "f4df1255b7b61abde4ca4c5e8ece4763e8685cd72fee1542dc27adcf7ddcfff8"
)
SOURCE_SELECTED_SHA256 = (
    "3c702f8f7c4e10205344c3e0f97b18a246c6e7f439cda8ba62b755754a22c65e"
)
SOURCE_SELECTED_PAYLOAD_SHA256 = (
    "a45dd34861bdcd5ace45b661994839c9e24456b58d3dd4845a61c166af1f7aca"
)
SOURCE_PARAMS_SHA256 = (
    "45c0790fe91d0135dcc2d8598bc85d8e849da5f3a9ba30a7ce6d881939011bb1"
)
SOURCE_PROFILE_SHA256 = (
    "6ec34d9555434b10050c07edc589dcdc1b6aabcacafd738528bbb53d733261f8"
)
SOURCE_SUBMISSION_RECEIPT = (
    source_submit.OUTPUT_ROOT
    / SOURCE_SPEC.lane_name
    / source_submit.RECEIPT_NAME
)
SOURCE_SUBMISSION_RECEIPT_SHA256 = (
    "4275d9e8e004f3f9f57d9483ca9e7b8a48d410778f39a80848c95c396ed2171c"
)
SOURCE_SUBMISSION_RECEIPT_PAYLOAD_SHA256 = (
    "84c30fe0067855f505d1b2a187050353d7497530b507625d4a55377494fec7a2"
)
SOURCE_FINAL_SEAL = (
    source_submit.OUTPUT_ROOT
    / SOURCE_SPEC.lane_name
    / source_submit.LANE_FINAL_NAME
)
SOURCE_FINAL_SEAL_SHA256 = (
    "ed28dd4d0c4ec904d2910323bc63bee2e13afddb59cd8e4d7f1a9d3d11487477"
)
SOURCE_FINAL_SEAL_PAYLOAD_SHA256 = (
    "00fc32efe74ae20d2d80bb69a974c93a7da1e977e24b0b9786147f155fa5dec9"
)

CANDIDATE_SHA256 = SOURCE_SPEC.candidate_sha256
ACCOUNT_NAME = "jji0930"
NODE_NAME = "n115"
SAME_NODE_AS_TASK_ID = SOURCE_TASK_ID
SOURCE_ALLOCATION_ID = 14650
SOURCE_SLURM_JOB_ID = "840582"
STORAGE_AUDIT_TASK_ID = 96334
STORAGE_AUDIT_TASK_NAME = (
    "mft-goal-readonly-live-standard5-inventory-t96332-v1"
)
STORAGE_AUDIT_DEDUPE_KEY = (
    "mft-goal-readonly-live-standard5-inventory:"
    "t96332:v1:909d249ebe45:n115"
)
MAX_STORAGE_AUDIT_AGE_SECONDS = 4 * 60 * 60
TASK_NAME = (
    "mft-goal-diag-standard-official5-direct-analyze-samenode-r1-v2-"
    f"{SOURCE_SPEC.short_sha}-{NODE_NAME}"
)
WORKDIR = (
    "mft_goal_diag_standard_official5_direct_analyze_samenode_r1_v2_"
    f"{SOURCE_SPEC.short_sha}_{NODE_NAME}"
)
DIRECT_ENV_NAME = "MFT_SYMMETRY_THERMAL_DIRECT_ANALYZE"
DIRECT_ENV_TOKEN = "standard-eighth-direct-analyze-v1"
DIRECT_ENV_TOKEN_SHA256 = hashlib.sha256(
    DIRECT_ENV_TOKEN.encode("utf-8")
).hexdigest()
STANDALONE_CORE_CONTRACT = "mft-standalone-core-optin-v1"


def standalone_core_auth_sha256(solver_revision: str) -> str:
    """Bind the 8-core opt-in to the exact committed solver revision."""

    payload = {
        "backend": "standalone",
        "contract_version": STANDALONE_CORE_CONTRACT,
        "requested_num_cores": CPUS,
        "required_slurm_cpus_per_task": CPUS,
        "solver_revision": solver_revision,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


POST_AUTHORIZATION = (
    "authorize-official5-direct-analyze-samenode-r1-96332-one-post-v2"
)

OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official5_direct_analyze_samenode_r1_v2"
)
PLAN_NAME = "direct_analyze_plan.json"
PREPARE_RECEIPT_NAME = "prepare_receipt.json"
SELECTED_NAME = "selected_candidate.json"
PARAMS_NAME = "fea_params.json"
PROFILE_NAME = "execution_profile.json"
SUBMISSION_DIRECTORY_NAME = "submission"
INTENT_NAME = "submission_intent.json"
ATTEMPT_NAME = "scheduler_post_attempt.json"
RECEIPT_NAME = "submission_receipt.json"
FINAL_NAME = "final_seal.json"
AMBIGUOUS_NAME = "post_ambiguous.json"

PLAN_SCHEMA = "mft-goal-official5-direct-analyze-plan-v1"
PREPARE_RECEIPT_SCHEMA = (
    "mft-goal-official5-direct-analyze-prepare-receipt-v1"
)
REVISION_SCHEMA = "mft-goal-direct-analyze-solver-revision-attestation-v1"
SOURCE_AUTHORITY_SCHEMA = "mft-goal-official5-task96332-authority-v1"
LIVE_GATE_SCHEMA = "mft-goal-official5-direct-analyze-live-gate-v1"
INTENT_SCHEMA = "mft-goal-official5-direct-analyze-submit-intent-v1"
ATTEMPT_SCHEMA = "mft-goal-official5-direct-analyze-post-attempt-v1"
RECEIPT_SCHEMA = "mft-goal-official5-direct-analyze-submission-v1"
FINAL_SCHEMA = "mft-goal-official5-direct-analyze-final-v1"
AMBIGUOUS_SCHEMA = "mft-goal-official5-direct-analyze-ambiguous-v1"
SINGLE_ATTEMPT_SCHEMA = (
    "mft-goal-official5-direct-analyze-single-attempt-v1"
)

MIN_STORAGE_FREE_GB = 10.0
LICENSE_FEATURE = source_submit.LICENSE_FEATURE
LICENSE_RESERVE_FLOOR = source_submit.LICENSE_RESERVE_FLOOR
MAX_NODE_METRICS_AGE_SECONDS = (
    source_submit.MAX_NODE_METRICS_AGE_SECONDS
)
NODE_STATE_ALLOWLIST = source_submit.NODE_STATE_ALLOWLIST
ACTIVE_TASK_STATES = frozenset(
    ("queued", "attaching", "running")
)
SOURCE_TASK_STATES = frozenset(
    (
        "queued",
        "attaching",
        "attached",
        "running",
        "completed",
        "succeeded",
        "failed",
    )
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")

canonical_bytes = source_prepare.canonical_bytes
payload_sha256 = source_prepare.payload_sha256
sealed = source_prepare.sealed
validate_seal = source_prepare.validate_seal
sha256_file = source_prepare.sha256_file
file_record = source_prepare.file_record
read_json = source_prepare.read_json
write_immutable_json = source_prepare.write_immutable_json
write_exclusive_json = source_submit.write_exclusive_json
get_json = source_prepare.get_json
scheduler_campaign_lock = source_submit.scheduler_campaign_lock


def _require_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PostdeadlineContractError(f"{label} is unavailable") from exc
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or sha256_file(resolved) != expected_sha256
    ):
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return resolved


def _contained_artifact(
    root: Path,
    record: Any,
    label: str,
) -> Path:
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise PostdeadlineContractError(f"{label} record is malformed")
    resolved_root = root.resolve(strict=True)
    raw = Path(str(record["path"]))
    path = (
        raw.resolve(strict=True)
        if raw.is_absolute()
        else (resolved_root / raw).resolve(strict=True)
    )
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise PostdeadlineContractError(
            f"{label} escapes its sealed root"
        ) from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["size_bytes"]
    ):
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return path


def attest_solver_revision(
    revision: str,
    *,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
) -> dict[str, Any]:
    """Verify that one committed revision contains the direct path."""

    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(
        revision
    ):
        raise PostdeadlineContractError(
            "solver revision must be an exact lowercase 40-hex commit"
        )
    try:
        thermal = source_reader("module/thermal_260706.py", revision)
        runner = source_reader("run_simulation_260706.py", revision)
    except Exception as exc:
        raise PostdeadlineContractError(
            "solver revision cannot be read as committed source"
        ) from exc
    thermal_markers = (
        DIRECT_ENV_NAME,
        DIRECT_ENV_TOKEN,
        "_symmetry_thermal_direct_analyze_request",
        "_symmetry_thermal_direct_analyze_preflight",
        "generate_mesh_call_policy",
        "forbidden_before_analyze",
    )
    runner_markers = (
        "--symmetry-thermal-direct-analyze",
        "SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV",
        "SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN",
    )
    if (
        any(marker.encode("utf-8") not in thermal for marker in thermal_markers)
        or any(
            marker.encode("utf-8") not in runner for marker in runner_markers
        )
    ):
        raise PostdeadlineContractError(
            "solver revision predates or lacks the committed direct path"
        )
    return sealed(
        {
            "schema_version": REVISION_SCHEMA,
            "solver_revision": revision,
            "committed_source_readback": True,
            "direct_path_present": True,
            "thermal_source": {
                "path": "module/thermal_260706.py",
                "sha256": hashlib.sha256(thermal).hexdigest(),
                "size_bytes": len(thermal),
            },
            "runner_source": {
                "path": "run_simulation_260706.py",
                "sha256": hashlib.sha256(runner).hexdigest(),
                "size_bytes": len(runner),
            },
            "required_env_name": DIRECT_ENV_NAME,
            "required_env_token_sha256": DIRECT_ENV_TOKEN_SHA256,
            "full_model_allowed": False,
        }
    )


def authenticate_source_authority() -> dict[str, Any]:
    """Reauthenticate the exact official #5 plan and task96332 seals."""

    manifest, lanes = source_submit.authenticate_prepare_package()
    matches = [
        lane for lane in lanes if lane["spec"].selection_order == 5
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            "official #5 source lane is absent or ambiguous"
        )
    lane = matches[0]
    plan_path = _require_file(
        lane["plan_path"], SOURCE_PLAN_SHA256, "source official #5 plan"
    )
    plan = lane["plan"]
    if (
        plan.get("payload_sha256") != SOURCE_PLAN_PAYLOAD_SHA256
        or plan.get("candidate_physics_sha256") != CANDIDATE_SHA256
        or plan.get("official_standard_selection_order") != 5
        or plan.get("standard_only") is not True
        or plan.get("symmetric_model") is not True
        or plan.get("full_model") is not False
        or plan.get("thermal_symmetry") != "eighth"
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("fixed_physics_unchanged") is not True
        or plan.get("scheduler_post_calls") != 0
    ):
        raise PostdeadlineContractError(
            "source official #5 plan authority drifted"
        )
    root = plan_path.parent
    selected_path = _contained_artifact(
        root, plan.get("selected_candidate"), "source selected candidate"
    )
    params_path = _contained_artifact(
        root, plan.get("fea_params"), "source FEA params"
    )
    profile_path = _contained_artifact(
        root, plan.get("execution_profile"), "source execution profile"
    )
    if (
        sha256_file(selected_path) != SOURCE_SELECTED_SHA256
        or sha256_file(params_path) != SOURCE_PARAMS_SHA256
        or sha256_file(profile_path) != SOURCE_PROFILE_SHA256
    ):
        raise PostdeadlineContractError(
            "source official #5 candidate artifact drifted"
        )
    selected = validate_seal(
        read_json(selected_path), source_prepare.CANDIDATE_SCHEMA
    )
    params = read_json(params_path)
    profile = read_json(profile_path)
    fresh_selected, fresh_params = source_prepare.authenticate_candidate(
        SOURCE_SPEC
    )
    fresh_profile = source_prepare.candidate8.reviewed_profile()
    if (
        selected.get("payload_sha256")
        != SOURCE_SELECTED_PAYLOAD_SHA256
        or selected != fresh_selected
        or params != fresh_params
        or profile != fresh_profile
        or payload_sha256(params) != SOURCE_SPEC.raw_fea_params_sha256
        or payload_sha256(source_prepare._profiled_params(params, profile))
        != SOURCE_SPEC.profiled_fea_params_sha256
    ):
        raise PostdeadlineContractError(
            "source official #5 candidate reauthentication drifted"
        )
    receipt_path = _require_file(
        SOURCE_SUBMISSION_RECEIPT,
        SOURCE_SUBMISSION_RECEIPT_SHA256,
        "source task96332 receipt",
    )
    receipt = validate_seal(
        read_json(receipt_path), source_submit.RECEIPT_SCHEMA
    )
    final_path = _require_file(
        SOURCE_FINAL_SEAL,
        SOURCE_FINAL_SEAL_SHA256,
        "source task96332 final seal",
    )
    final = validate_seal(
        read_json(final_path), source_submit.LANE_FINAL_SCHEMA
    )
    if (
        receipt.get("payload_sha256")
        != SOURCE_SUBMISSION_RECEIPT_PAYLOAD_SHA256
        or receipt.get("selection_order") != 5
        or receipt.get("candidate_physics_sha256") != CANDIDATE_SHA256
        or receipt.get("task_id") != SOURCE_TASK_ID
        or receipt.get("task_name") != SOURCE_TASK_NAME
        or receipt.get("dedupe_key") != SOURCE_TASK_DEDUPE_KEY
        or receipt.get("account_name") != SOURCE_SPEC.account_name
        or receipt.get("node_name") != SOURCE_SPEC.node_name
        or receipt.get("node_name_policy") != "strict"
        or receipt.get("preferred_node_relaxed") is not False
        or receipt.get("scheduler_post_calls") != 1
        or receipt.get("scheduler_project_mutation_performed") is not False
        or receipt.get("scheduler_repository_modified") is not False
        or final.get("payload_sha256") != SOURCE_FINAL_SEAL_PAYLOAD_SHA256
        or final.get("task_id") != SOURCE_TASK_ID
        or final.get("scheduler_post_calls") != 1
    ):
        raise PostdeadlineContractError(
            "source task96332 submission authority drifted"
        )
    authority = sealed(
        {
            "schema_version": SOURCE_AUTHORITY_SCHEMA,
            **SAFETY_FLAGS,
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "official_standard_selection_order": 5,
            "source_task_id": SOURCE_TASK_ID,
            "source_task_name": SOURCE_TASK_NAME,
            "source_task_dedupe_key": SOURCE_TASK_DEDUPE_KEY,
            "prepare_manifest": file_record(
                source_submit.PREPARE_MANIFEST
            ),
            "prepare_manifest_payload_sha256": manifest["payload_sha256"],
            "source_plan": file_record(plan_path),
            "source_plan_payload_sha256": plan["payload_sha256"],
            "source_selected_candidate": file_record(selected_path),
            "source_selected_candidate_payload_sha256": selected[
                "payload_sha256"
            ],
            "source_fea_params": file_record(params_path),
            "source_execution_profile": file_record(profile_path),
            "source_submission_receipt": file_record(receipt_path),
            "source_submission_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "source_final_seal": file_record(final_path),
            "source_final_seal_payload_sha256": final["payload_sha256"],
            "source_reauthenticated": True,
            "source_task_scheduler_post_calls": 1,
            "new_lane_scheduler_post_calls": 0,
        }
    )
    return {
        "authority": authority,
        "selected": selected,
        "params": params,
        "profile": profile,
        "source_plan": plan,
    }


@contextmanager
def _payload_contract(solver_revision: str) -> Any:
    replacements = {
        "SOLVER_REVISION": solver_revision,
        "LIBRARY_REVISION": LIBRARY_REVISION,
        "ACCOUNT_NAME": ACCOUNT_NAME,
        "NODE_NAME": NODE_NAME,
        "TASK_NAME": TASK_NAME,
        "WORKDIR": WORKDIR,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
        "CORE_AUTH_SHA256": standalone_core_auth_sha256(
            solver_revision
        ),
    }
    previous = {name: getattr(reviewed, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(reviewed, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(reviewed, name, value)


def _direct_command(command: str) -> str:
    solver = (
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json; simulation_rc=$?;"
    )
    direct = (
        f'export {DIRECT_ENV_NAME}="{DIRECT_ENV_TOKEN}"; '
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--symmetry-thermal-direct-analyze --params cand.json; "
        "simulation_rc=$?;"
    )
    if command.count(solver) != 1 or DIRECT_ENV_NAME in command:
        raise PostdeadlineContractError(
            "reviewed command cannot be converted exactly once"
        )
    return command.replace(solver, direct, 1)


def derive_direct_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
    solver_revision: str,
    *,
    base_builder: Callable[
        [dict[str, Any], dict[str, Any]],
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive a unique retained Standard payload and add the exact opt-in."""

    with _payload_contract(solver_revision):
        payload, environment, retained = (
            base_builder(params, profile)
            if base_builder is not None
            else reviewed._capture_scheduler_payload(params, profile)
        )
        reviewed.validate_scheduler_payload(payload, retained)
    payload = copy.deepcopy(payload)
    environment = copy.deepcopy(environment)
    retained = copy.deepcopy(retained)
    payload["same_node_as_task_id"] = SAME_NODE_AS_TASK_ID
    payload["command"] = _direct_command(str(payload.get("command") or ""))
    environment[DIRECT_ENV_NAME] = DIRECT_ENV_TOKEN
    validate_direct_payload(
        payload,
        environment,
        retained,
        params=params,
        solver_revision=solver_revision,
    )
    return payload, environment, retained


def validate_direct_payload(
    payload: Mapping[str, Any],
    environment: Mapping[str, Any],
    retained: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    solver_revision: str,
) -> None:
    expected = {
        "name": TASK_NAME,
        "project": PROJECT,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "timeout_seconds": SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
    }
    drift = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    command = str(payload.get("command") or "")
    dedupe = str(payload.get("dedupe_key") or "")
    artifact = str(retained.get("artifact_path") or "")
    results = str(retained.get("results_path") or "")
    chunks = str(
        retained.get("transport", {}).get("chunk_directory") or ""
    )
    boundary = {
        "fan_velocity": params.get("fan_velocity"),
        "k_ins": params.get("k_ins"),
        "core_plate_pad_t": params.get("core_plate_pad_t"),
        "wcp_pad_t": params.get("wcp_pad_t"),
        "thermal_symmetry": params.get("thermal_symmetry"),
        "full_model": params.get("full_model"),
    }
    core_auth = standalone_core_auth_sha256(solver_revision)
    if (
        drift
        or environment.get(DIRECT_ENV_NAME) != DIRECT_ENV_TOKEN
        or environment.get("MFT_STANDALONE_CORE_CONTRACT")
        != STANDALONE_CORE_CONTRACT
        or environment.get("MFT_STANDALONE_CORE_COUNT") != str(CPUS)
        or environment.get("MFT_STANDALONE_CORE_AUTH_SHA256")
        != core_auth
        or command.count(
            f'export MFT_STANDALONE_CORE_AUTH_SHA256="{core_auth}"'
        )
        != 1
        or command.count(
            f'export {DIRECT_ENV_NAME}="{DIRECT_ENV_TOKEN}";'
        )
        != 1
        or command.count("--symmetry-thermal-direct-analyze") != 1
        or command.count(
            "python run_simulation_260706.py --fixed --thermal --headless"
        )
        != 1
        or "--full" in command
        or solver_revision not in command
        or not dedupe
        or dedupe == SOURCE_TASK_DEDUPE_KEY
        or retained.get("dedupe_key") != dedupe
        or retained.get("solver_revision") != solver_revision
        or retained.get("library_revision") != LIBRARY_REVISION
        or retained.get("stage") != "standard"
        or not artifact.endswith("/symmetric.aedt")
        or not results.endswith("/symmetric.aedtresults")
        or not chunks.endswith("/symmetric.aedt.chunks")
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
        or boundary
        != {
            "fan_velocity": 1.5,
            "k_ins": 0.2,
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
            "thermal_symmetry": "eighth",
            "full_model": 0,
        }
    ):
        raise PostdeadlineContractError(
            f"direct Analyze payload/retention contract drifted: {drift}"
        )


def capacity_query() -> list[tuple[str, Any]]:
    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", ACCOUNT_NAME),
        ("node_name", NODE_NAME),
        ("same_node_as_task_id", SAME_NODE_AS_TASK_ID),
    ]


def _task_rows(value: Any) -> list[dict[str, Any]]:
    return reviewed._task_rows(value)


def _account_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PostdeadlineContractError(
            "Scheduler account inventory is malformed"
        )
    return [row for row in value if isinstance(row, dict)]


def _capability_accounts(value: Any) -> set[str]:
    if not isinstance(value, list):
        raise PostdeadlineContractError(
            "Scheduler capability inventory is malformed"
        )
    for row in value:
        if (
            isinstance(row, dict)
            and row.get("capability") == "conda:pyaedt2026v1"
            and isinstance(row.get("accounts"), list)
        ):
            return {str(account) for account in row["accounts"]}
    return set()


def _explicit_account_storage(
    account: Mapping[str, Any],
    *,
    storage_audit: Mapping[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    """Attest account caps and storage, using the exact n115 df audit."""

    cap = source_submit._account_storage_gate(account)
    storage = cap.get("storage")
    if (
        isinstance(storage, Mapping)
        and storage.get("mode") == "declared_quota"
        and float(storage.get("free_gb") or 0.0) >= MIN_STORAGE_FREE_GB
    ):
        return cap
    expected = {
        "task_id": STORAGE_AUDIT_TASK_ID,
        "name": STORAGE_AUDIT_TASK_NAME,
        "dedupe_key": STORAGE_AUDIT_DEDUPE_KEY,
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "project": PROJECT,
        "account_name": ACCOUNT_NAME,
        "requested_account_name": ACCOUNT_NAME,
        "allocation_id": SOURCE_ALLOCATION_ID,
        "assigned_allocation": SOURCE_ALLOCATION_ID,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "node_name": NODE_NAME,
        "requested_node_name": NODE_NAME,
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "same_node_as_node_name": NODE_NAME,
        "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
    }
    drift = {
        key: {"expected": expected_value, "actual": storage_audit.get(key)}
        for key, expected_value in expected.items()
        if storage_audit.get(key) != expected_value
    }
    stdout = str(storage_audit.get("stdout") or "")
    match = re.search(
        r"(?m)^/dev/\S+\s+\d+\s+\d+\s+(\d+)\s+\d+%\s+/enroot\s*$",
        stdout,
    )
    finished_at = source_submit._parse_scheduler_time(
        storage_audit.get("finished_at")
    )
    age = (observed_at - finished_at).total_seconds()
    available_kib = int(match.group(1)) if match else 0
    free_gb = available_kib / (1024**2)
    if (
        drift
        or "MFT_LIVE_INVENTORY_V1" not in stdout
        or "source=/enroot/mft_campaign-" not in stdout
        or "node=n115 job=840582" not in stdout
        or not match
        or free_gb < MIN_STORAGE_FREE_GB
        or age < -60
        or age > MAX_STORAGE_AUDIT_AGE_SECONDS
    ):
        raise PostdeadlineContractError(
            "same-node storage audit is not admissible: "
            f"identity={drift}, free_gb={free_gb:.3f}, age_s={age:.1f}"
        )
    result = copy.deepcopy(cap)
    result["storage"] = {
        "mode": "same_node_enroot_df_audit",
        "path": "/enroot",
        "free_gb": free_gb,
        "available_kib": available_kib,
        "audit_task_id": STORAGE_AUDIT_TASK_ID,
        "audit_task_name": STORAGE_AUDIT_TASK_NAME,
        "audit_finished_at": finished_at.isoformat(),
        "audit_age_seconds": age,
        "minimum_free_gb": MIN_STORAGE_FREE_GB,
        "source_task96332_allocation_attested": True,
    }
    return result


def _same_node_allocation_gate(
    value: Any,
    *,
    source_task: Mapping[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    """Attest the exact live allocation inherited from task96332."""

    if isinstance(value, dict) and isinstance(value.get("allocations"), list):
        rows = value["allocations"]
    elif isinstance(value, list):
        rows = value
    else:
        raise PostdeadlineContractError(
            "Scheduler allocation inventory is malformed"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and int(row.get("id") or 0) == SOURCE_ALLOCATION_ID
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            "source same-node allocation is absent or ambiguous"
        )
    allocation = matches[0]
    metrics_at = source_submit._parse_scheduler_time(
        allocation.get("node_metrics_observed_at")
    )
    age = (observed_at - metrics_at).total_seconds()
    expected = {
        "id": SOURCE_ALLOCATION_ID,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "state": "active",
        "resource_pool": "cpu",
    }
    drift = {
        key: {"expected": expected_value, "actual": allocation.get(key)}
        for key, expected_value in expected.items()
        if allocation.get(key) != expected_value
    }
    if (
        drift
        or int(source_task.get("allocation_id") or 0)
        != SOURCE_ALLOCATION_ID
        or str(source_task.get("slurm_job_id") or "")
        != SOURCE_SLURM_JOB_ID
        or allocation.get("node_pestat_state") not in NODE_STATE_ALLOWLIST
        or int(allocation.get("node_cpu_total") or 0) < CPUS
        or int(allocation.get("node_memory_total_mb") or 0) < MEMORY_MB
        or int(allocation.get("node_memory_free_mb") or 0) < MEMORY_MB
        or int(allocation.get("free_cpus") or 0) < CPUS
        or int(allocation.get("free_memory_mb") or 0) < MEMORY_MB
        or age < -60
        or age > MAX_NODE_METRICS_AGE_SECONDS
    ):
        raise PostdeadlineContractError(
            f"source same-node allocation gate failed: {drift}"
        )
    return {
        "allocation_id": SOURCE_ALLOCATION_ID,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "state": allocation["state"],
        "node_state": allocation["node_pestat_state"],
        "free_cpus": int(allocation["free_cpus"]),
        "free_memory_mb": int(allocation["free_memory_mb"]),
        "node_cpu_total": int(allocation["node_cpu_total"]),
        "node_cpu_used": int(allocation.get("node_cpu_used") or 0),
        "node_memory_total_mb": int(allocation["node_memory_total_mb"]),
        "node_memory_free_mb": int(allocation["node_memory_free_mb"]),
        "metrics_observed_at": metrics_at.isoformat(),
        "metrics_age_seconds": age,
        "exact_source_allocation_attested": True,
    }


def live_gate(
    *,
    expected_dedupe_key: str,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """GET-only exact task96332 same-node admission gate."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "live gate time must be timezone-aware"
        )
    now = now.astimezone(timezone.utc)
    if now <= ORIGINAL_DEADLINE_UTC:
        raise PostdeadlineContractError(
            "post-deadline gate cannot precede the original deadline"
        )
    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{SOURCE_TASK_ID}", None)
    storage_audit = reader(
        f"/api/tasks/{STORAGE_AUDIT_TASK_ID}",
        [("include_output", "true"), ("output_limit", 65536)],
    )
    capacity = reader("/api/task-capacity", capacity_query())
    allocations = reader("/api/allocations", [("limit", 10000)])
    licenses = reader("/api/licenses", None)
    accounts = reader("/api/accounts/status/live", None)
    capabilities = reader("/api/capabilities", None)
    active = reader(
        "/api/tasks",
        [
            ("status", "queued"),
            ("status", "attaching"),
            ("status", "running"),
            ("limit", 10000),
        ],
    )
    inventory = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    if (
        not isinstance(health, Mapping)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")
    expected_source = {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
        "project": PROJECT,
        "requested_account_name": SOURCE_SPEC.account_name,
        "requested_node_name": SOURCE_SPEC.node_name,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "status": "running",
        "allocation_id": SOURCE_ALLOCATION_ID,
        "assigned_allocation": SOURCE_ALLOCATION_ID,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "same_node_as_task_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
    }
    source_drift = {
        key: {
            "expected": expected,
            "actual": (
                source_task.get(key)
                if isinstance(source_task, Mapping)
                else None
            ),
        }
        for key, expected in expected_source.items()
        if not isinstance(source_task, Mapping)
        or source_task.get(key) != expected
    }
    if source_drift:
        raise PostdeadlineContractError(
            f"source task96332 GET identity drifted: {source_drift}"
        )
    matches = [
        account
        for account in _account_rows(accounts)
        if account.get("account_name") == ACCOUNT_NAME
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            "target account status is absent or ambiguous"
        )
    account_gate = _explicit_account_storage(
        matches[0],
        storage_audit=storage_audit,
        observed_at=now,
    )
    if ACCOUNT_NAME not in _capability_accounts(capabilities):
        raise PostdeadlineContractError(
            "target account lacks pyaedt2026v1 capability"
        )
    license_gate = source_submit._license_gate(
        licenses, required_headroom=1
    )
    node_gate = _same_node_allocation_gate(
        allocations,
        source_task=source_task,
        observed_at=now,
    )
    capacity_allocations = (
        capacity.get("allocations")
        if isinstance(capacity, Mapping)
        else None
    )
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("queue_state") != "ready"
        or capacity.get("queue_reason")
        != f"ready to attach to allocation {SOURCE_ALLOCATION_ID}"
        or int(capacity.get("ready_fit_slots") or 0) < 1
        or int(capacity.get("pending_fit_slots") or 0) != 0
        or int(capacity.get("inflight_fit_slots") or 0) < 1
        or capacity.get("memory_pressure_state") != "ok"
        or capacity.get("preferred_node_relaxed") is not False
        or int(capacity.get("standalone_aedt_available") or 0) < 1
        or not isinstance(capacity_allocations, list)
        or len(capacity_allocations) != 1
        or int(capacity_allocations[0].get("allocation_id") or 0)
        != SOURCE_ALLOCATION_ID
        or capacity_allocations[0].get("account_name") != ACCOUNT_NAME
        or capacity_allocations[0].get("node_name") != NODE_NAME
        or capacity_allocations[0].get("memory_pressure_state") != "ok"
        or int(capacity_allocations[0].get("fit_slots") or 0) < 1
        or int(capacity_allocations[0].get("free_cpus") or 0) < CPUS
        or int(capacity_allocations[0].get("free_memory_mb") or 0)
        < MEMORY_MB
    ):
        raise PostdeadlineContractError(
            "exact task96332 same-node capacity gate is not ready"
        )
    active_fea = [
        row
        for row in _task_rows(active)
        if row.get("status") in ACTIVE_TASK_STATES
        and (
            row.get("node_name") == NODE_NAME
            or row.get("requested_node_name") == NODE_NAME
            or row.get("actual_node_name") == NODE_NAME
        )
        and (
            row.get("aedt_backend") == "standalone"
            or row.get("scheduling_profile") == "fea_bursty"
        )
    ]
    active_target_ids = {
        int(row.get("task_id") or row.get("id") or 0)
        for row in active_fea
    }
    if active_target_ids != {SOURCE_TASK_ID}:
        raise PostdeadlineContractError(
            "n115 active FEA set is not the exact source task96332"
        )
    collisions = [
        row
        for row in _task_rows(inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collisions:
        raise PostdeadlineContractError(
            "direct fast-lane task name or dedupe already exists"
        )
    return sealed(
        {
            "schema_version": LIVE_GATE_SCHEMA,
            **SAFETY_FLAGS,
            "observed_at_utc": now.isoformat(),
            "task_name": TASK_NAME,
            "dedupe_key": expected_dedupe_key,
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "source_allocation_id": SOURCE_ALLOCATION_ID,
            "source_slurm_job_id": SOURCE_SLURM_JOB_ID,
            "scheduler_health": copy.deepcopy(dict(health)),
            "source_task96332": copy.deepcopy(dict(source_task)),
            "storage_audit_task96334": copy.deepcopy(
                dict(storage_audit)
            ),
            "capacity_query": capacity_query(),
            "capacity": copy.deepcopy(dict(capacity)),
            "account_cap_and_storage_gate": account_gate,
            "license_gate": license_gate,
            "node_gate": node_gate,
            "active_target_node_fea_count": 1,
            "active_target_node_fea_task_ids": [SOURCE_TASK_ID],
            "collision_count": 0,
            "same_node_ready_only": True,
            "preferred_node_relaxed": False,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    )


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise PostdeadlineContractError(
            "prepared artifact escapes output root"
        ) from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def _attempt_nonce(
    *,
    solver_revision: str,
    payload: Mapping[str, Any],
) -> str:
    return payload_sha256(
        {
            "schema_version": SINGLE_ATTEMPT_SCHEMA,
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "solver_revision": solver_revision,
            "task_name": TASK_NAME,
            "dedupe_key": payload["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "source_allocation_id": SOURCE_ALLOCATION_ID,
            "scheduler_payload_sha256": payload_sha256(payload),
            "direct_env_token_sha256": DIRECT_ENV_TOKEN_SHA256,
        }
    )


SourceAuthenticator = Callable[[], dict[str, Any]]
RevisionAttester = Callable[[str], dict[str, Any]]
PayloadBuilder = Callable[
    [dict[str, Any], dict[str, Any], str],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]
Gate = Callable[[str, datetime | None], dict[str, Any]]


def prepare(
    *,
    solver_revision: str,
    output: Path | None = None,
    source_authenticator: SourceAuthenticator = authenticate_source_authority,
    revision_attester: RevisionAttester | None = None,
    payload_builder: PayloadBuilder = derive_direct_payload,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> Path:
    """Authenticate, GET-gate, and atomically seal a no-POST plan."""

    target = (output or OUTPUT_ROOT).resolve()
    if target != OUTPUT_ROOT.resolve():
        raise PostdeadlineContractError(
            "prepare output root is not the fixed fast-lane root"
        )
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable fast-lane output already exists: {target}"
        )
    attester = revision_attester or attest_solver_revision
    revision = attester(solver_revision)
    if (
        validate_seal(revision, REVISION_SCHEMA) is not revision
        or revision.get("solver_revision") != solver_revision
        or revision.get("direct_path_present") is not True
    ):
        raise PostdeadlineContractError(
            "direct-path solver revision attestation drifted"
        )
    source = source_authenticator()
    authority = source.get("authority")
    selected = source.get("selected")
    params = source.get("params")
    profile = source.get("profile")
    if (
        not isinstance(authority, dict)
        or validate_seal(authority, SOURCE_AUTHORITY_SCHEMA) is not authority
        or authority.get("source_task_id") != SOURCE_TASK_ID
        or not isinstance(selected, dict)
        or not isinstance(params, dict)
        or not isinstance(profile, dict)
    ):
        raise PostdeadlineContractError(
            "source official #5 authority is incomplete"
        )
    payload, environment, retained = payload_builder(
        params, profile, solver_revision
    )
    validate_direct_payload(
        payload,
        environment,
        retained,
        params=params,
        solver_revision=solver_revision,
    )
    gate = live_gate(
        expected_dedupe_key=str(payload["dedupe_key"]),
        reader=reader,
        observed_at=observed_at,
    )
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "prepare time must be timezone-aware"
        )
    now = now.astimezone(timezone.utc)
    nonce = _attempt_nonce(
        solver_revision=solver_revision, payload=payload
    )
    attempt_path = target / ATTEMPT_NAME
    submission_path = target / SUBMISSION_DIRECTORY_NAME
    staging = target.with_name(
        f".{target.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    if staging.exists():
        raise PostdeadlineContractError("prepare staging path exists")
    staging.mkdir(parents=True)
    try:
        selected_path = write_immutable_json(
            staging / SELECTED_NAME, selected
        )
        params_path = write_immutable_json(
            staging / PARAMS_NAME, params
        )
        profile_path = write_immutable_json(
            staging / PROFILE_NAME, profile
        )
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "campaign_id": CAMPAIGN_ID,
                "prepare_only": True,
                "standard_only": True,
                "symmetric_model": True,
                "full_model": False,
                "thermal_symmetry": "eighth",
                "direct_analyze_fast_lane": True,
                "same_node_fast_lane": True,
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "official_standard_selection_order": 5,
                "source_authority": authority,
                "selected_candidate": _relative_record(
                    staging, selected_path
                ),
                "selected_candidate_payload_sha256": selected[
                    "payload_sha256"
                ],
                "fea_params": _relative_record(staging, params_path),
                "fea_params_sha256": payload_sha256(params),
                "execution_profile": _relative_record(
                    staging, profile_path
                ),
                "execution_profile_sha256": payload_sha256(profile),
                "solver_revision": solver_revision,
                "solver_revision_attestation": revision,
                "library_revision": LIBRARY_REVISION,
                "task_name": TASK_NAME,
                "workdir": WORKDIR,
                "dedupe_key": payload["dedupe_key"],
                "placement": {
                    "account_name": ACCOUNT_NAME,
                    "node_name": NODE_NAME,
                    "node_name_policy": "strict",
                    "preferred_node_relaxed_allowed": False,
                    "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
                    "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
                    "same_node_as_slurm_job_id": SOURCE_SLURM_JOB_ID,
                    "dependency_task_id": 0,
                },
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "solver_seconds": SOLVER_SECONDS,
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                    "retention_seconds": RETENTION_SECONDS,
                    "scheduler_timeout_seconds": SCHEDULER_SECONDS,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                },
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "fixed_physics_unchanged": True,
                "direct_analyze_opt_in": {
                    "env_name": DIRECT_ENV_NAME,
                    "env_value": DIRECT_ENV_TOKEN,
                    "env_value_sha256": DIRECT_ENV_TOKEN_SHA256,
                    "cli_flag": "--symmetry-thermal-direct-analyze",
                    "exact_opt_in_required": True,
                    "standalone_core_contract": (
                        STANDALONE_CORE_CONTRACT
                    ),
                    "standalone_core_count": CPUS,
                    "standalone_core_auth_sha256": (
                        standalone_core_auth_sha256(solver_revision)
                    ),
                },
                "scheduler_url": SCHEDULER_URL,
                "scheduler_project": PROJECT,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                "scheduler_post_calls": 0,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(
                    environment
                ),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "unique_dedupe_and_retention_identity": True,
                "live_gate_at_prepare": gate,
                "single_attempt_contract": {
                    "schema_version": SINGLE_ATTEMPT_SCHEMA,
                    "attempt_ledger_path": str(attempt_path),
                    "submission_output_path": str(submission_path),
                    "attempt_nonce": nonce,
                    "post_call_budget": 1,
                    "immutable_intent_required_before_post": True,
                    "attempt_consumed_before_network": True,
                    "ambiguous_retry_allowed": False,
                    "output_override_allowed": False,
                },
                "submit_authorization": POST_AUTHORIZATION,
                "automatic_full_trigger": False,
                "full_continuation_allowed": False,
            }
        )
        plan_path = write_immutable_json(staging / PLAN_NAME, plan)
        receipt = sealed(
            {
                "schema_version": PREPARE_RECEIPT_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "prepare_only": True,
                "plan": _relative_record(staging, plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "solver_revision": solver_revision,
                "task_name": TASK_NAME,
                "dedupe_key": payload["dedupe_key"],
                "account_name": ACCOUNT_NAME,
                "node_name": NODE_NAME,
                "node_name_policy": "strict",
                "scheduler_get_gate_performed": True,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "submit_command_argv": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "submit",
                    "--same-node-task-id",
                    str(SAME_NODE_AS_TASK_ID),
                    "--authorize-post",
                    POST_AUTHORIZATION,
                ],
                "ready_for_explicit_submit": True,
                "automatic_full_trigger": False,
            }
        )
        write_immutable_json(
            staging / PREPARE_RECEIPT_NAME, receipt
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def load_plan(
    plan_path: Path | None = None,
    *,
    source_authenticator: SourceAuthenticator = authenticate_source_authority,
    revision_attester: RevisionAttester | None = None,
    payload_builder: PayloadBuilder = derive_direct_payload,
) -> dict[str, Any]:
    path = (plan_path or (OUTPUT_ROOT / PLAN_NAME)).resolve(strict=True)
    if path != (OUTPUT_ROOT.resolve() / PLAN_NAME):
        raise PostdeadlineContractError("plan path escaped fixed output root")
    plan = validate_seal(read_json(path), PLAN_SCHEMA)
    single = plan.get("single_attempt_contract")
    placement = plan.get("placement")
    direct = plan.get("direct_analyze_opt_in")
    if (
        any(plan.get(key) is not value for key, value in SAFETY_FLAGS.items())
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("prepare_only") is not True
        or plan.get("standard_only") is not True
        or plan.get("symmetric_model") is not True
        or plan.get("full_model") is not False
        or plan.get("thermal_symmetry") != "eighth"
        or plan.get("direct_analyze_fast_lane") is not True
        or plan.get("same_node_fast_lane") is not True
        or plan.get("candidate_physics_sha256") != CANDIDATE_SHA256
        or plan.get("official_standard_selection_order") != 5
        or plan.get("task_name") != TASK_NAME
        or plan.get("workdir") != WORKDIR
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("fixed_physics_unchanged") is not True
        or plan.get("scheduler_url") != SCHEDULER_URL
        or plan.get("scheduler_project") != PROJECT
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("scheduler_post_calls") != 0
        or plan.get("automatic_full_trigger") is not False
        or plan.get("full_continuation_allowed") is not False
        or not isinstance(placement, Mapping)
        or placement.get("account_name") != ACCOUNT_NAME
        or placement.get("node_name") != NODE_NAME
        or placement.get("node_name_policy") != "strict"
        or placement.get("preferred_node_relaxed_allowed") is not False
        or placement.get("same_node_as_task_id") != SAME_NODE_AS_TASK_ID
        or placement.get("same_node_as_allocation_id")
        != SOURCE_ALLOCATION_ID
        or placement.get("same_node_as_slurm_job_id")
        != SOURCE_SLURM_JOB_ID
        or not isinstance(direct, Mapping)
        or direct.get("env_name") != DIRECT_ENV_NAME
        or direct.get("env_value") != DIRECT_ENV_TOKEN
        or direct.get("env_value_sha256") != DIRECT_ENV_TOKEN_SHA256
        or direct.get("exact_opt_in_required") is not True
        or direct.get("standalone_core_contract")
        != STANDALONE_CORE_CONTRACT
        or direct.get("standalone_core_count") != CPUS
        or direct.get("standalone_core_auth_sha256")
        != standalone_core_auth_sha256(
            str(plan.get("solver_revision") or "")
        )
        or not isinstance(single, Mapping)
        or single.get("schema_version") != SINGLE_ATTEMPT_SCHEMA
        or Path(str(single.get("attempt_ledger_path") or "")).resolve()
        != OUTPUT_ROOT.resolve() / ATTEMPT_NAME
        or Path(str(single.get("submission_output_path") or "")).resolve()
        != OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
        or single.get("post_call_budget") != 1
        or single.get("immutable_intent_required_before_post") is not True
        or single.get("attempt_consumed_before_network") is not True
        or single.get("ambiguous_retry_allowed") is not False
        or single.get("output_override_allowed") is not False
    ):
        raise PostdeadlineContractError("sealed direct plan drifted")
    solver_revision = str(plan.get("solver_revision") or "")
    attester = revision_attester or attest_solver_revision
    revision = attester(solver_revision)
    if (
        plan.get("solver_revision_attestation") != revision
        or revision.get("direct_path_present") is not True
    ):
        raise PostdeadlineContractError(
            "sealed solver revision attestation drifted"
        )
    root = path.parent
    selected_path = _contained_artifact(
        root, plan.get("selected_candidate"), "selected candidate"
    )
    params_path = _contained_artifact(
        root, plan.get("fea_params"), "FEA params"
    )
    profile_path = _contained_artifact(
        root, plan.get("execution_profile"), "execution profile"
    )
    selected = read_json(selected_path)
    params = read_json(params_path)
    profile = read_json(profile_path)
    source = source_authenticator()
    if (
        plan.get("source_authority") != source.get("authority")
        or selected != source.get("selected")
        or params != source.get("params")
        or profile != source.get("profile")
    ):
        raise PostdeadlineContractError(
            "sealed plan differs from task96332 source authority"
        )
    payload, environment, retained = payload_builder(
        params, profile, solver_revision
    )
    validate_direct_payload(
        payload,
        environment,
        retained,
        params=params,
        solver_revision=solver_revision,
    )
    if (
        plan.get("scheduler_payload") != payload
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(payload)
        or plan.get("dedupe_key") != payload["dedupe_key"]
        or plan.get("submission_environment") != environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(environment)
        or plan.get("retained_aedt_bundle") != retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(retained)
        or single.get("attempt_nonce")
        != _attempt_nonce(
            solver_revision=solver_revision, payload=payload
        )
    ):
        raise PostdeadlineContractError(
            "sealed plan no longer matches rederived direct payload"
        )
    return plan


def _inventory_exact(
    reader: JsonReader,
    *,
    dedupe_key: str,
) -> list[dict[str, Any]]:
    rows = _task_rows(
        reader(
            "/api/tasks",
            [
                ("limit", 10000),
                ("project", PROJECT),
                ("name_prefix", TASK_NAME),
            ],
        )
    )
    return [
        row
        for row in rows
        if row.get("name") == TASK_NAME
        and row.get("dedupe_key") == dedupe_key
    ]


def _authenticated_readback(
    reader: JsonReader,
    *,
    task_id: int,
    dedupe_key: str,
) -> dict[str, Any]:
    value = reader(f"/api/tasks/{task_id}", None)
    expected = {
        "task_id": task_id,
        "name": TASK_NAME,
        "dedupe_key": dedupe_key,
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {
            "expected": expected_value,
            "actual": (
                value.get(key) if isinstance(value, Mapping) else None
            ),
        }
        for key, expected_value in expected.items()
        if not isinstance(value, Mapping)
        or value.get(key) != expected_value
    }
    live_placement_drift: dict[str, dict[str, Any]] = {}
    if (
        isinstance(value, Mapping)
        and value.get("status") in {"attaching", "attached", "running"}
    ):
        live_expected = {
            "allocation_id": SOURCE_ALLOCATION_ID,
            "assigned_allocation": SOURCE_ALLOCATION_ID,
            "slurm_job_id": SOURCE_SLURM_JOB_ID,
            "account_name": ACCOUNT_NAME,
            "actual_node_name": NODE_NAME,
            "allocation_node_name": NODE_NAME,
            "same_node_as_node_name": NODE_NAME,
            "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
        }
        live_placement_drift = {
            key: {
                "expected": expected_value,
                "actual": value.get(key),
            }
            for key, expected_value in live_expected.items()
            if value.get(key) != expected_value
        }
    if (
        drift
        or live_placement_drift
        or value.get("status")
        not in {"queued", "attaching", "attached", "running"}
        or (
            value.get("account_name")
            and value.get("account_name") != ACCOUNT_NAME
        )
        or (
            value.get("node_name")
            and value.get("node_name") != NODE_NAME
        )
    ):
        raise PostdeadlineContractError(
            "exact Scheduler task readback drifted: "
            f"identity={drift}, live_placement={live_placement_drift}"
        )
    return copy.deepcopy(dict(value))


def _write_ambiguous(
    *,
    root: Path,
    plan: Mapping[str, Any],
    attempt_path: Path,
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    exact_count: int,
) -> Path:
    path = root / AMBIGUOUS_NAME
    if path.exists():
        validate_seal(read_json(path), AMBIGUOUS_SCHEMA)
        return path
    value = sealed(
        {
            "schema_version": AMBIGUOUS_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_payload_sha256": plan["payload_sha256"],
            "attempt_ledger": file_record(attempt_path),
            "http_status": http_status,
            "http_response": copy.deepcopy(response),
            "http_error": error,
            "exact_get_reconciliation_count": exact_count,
            "scheduler_post_calls": 1,
            "attempt_consumed": True,
            "retry_allowed": False,
            "automatic_full_trigger": False,
        }
    )
    return write_immutable_json(path, value)


def _finish(
    *,
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    intent_path: Path,
    attempt_path: Path,
    task_id: int,
    readback: Mapping[str, Any],
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    reconciled: bool,
) -> Path:
    receipt = sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "intent": file_record(intent_path),
            "attempt_ledger": file_record(attempt_path),
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "solver_revision": plan["solver_revision"],
            "scheduler_payload_sha256": plan[
                "scheduler_payload_sha256"
            ],
            "scheduler_post_calls": 1,
            "scheduler_post_http_status": http_status,
            "scheduler_post_response": copy.deepcopy(response),
            "scheduler_post_error": error,
            "exact_get_reconciled": reconciled,
            "task_id": task_id,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
            "task_readback": copy.deepcopy(dict(readback)),
            "task_readback_sha256": payload_sha256(readback),
            "preferred_node_relaxed": False,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
            "automatic_full_trigger": False,
        }
    )
    receipt_path = root / RECEIPT_NAME
    if receipt_path.exists():
        current = validate_seal(read_json(receipt_path), RECEIPT_SCHEMA)
        if current.get("task_id") != task_id:
            raise PostdeadlineContractError(
                "existing direct receipt task identity drifted"
            )
        receipt = current
    else:
        receipt_path = write_immutable_json(receipt_path, receipt)
    final = sealed(
        {
            "schema_version": FINAL_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "intent": file_record(intent_path),
            "attempt_ledger": file_record(attempt_path),
            "submission_receipt": file_record(receipt_path),
            "submission_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "task_id": task_id,
            "scheduler_post_call_budget": 1,
            "scheduler_post_calls_evidenced": 1,
            "retry_allowed": False,
            "full_model_started": False,
            "immutable_evidence_complete": True,
        }
    )
    final_path = root / FINAL_NAME
    if final_path.exists():
        current = validate_seal(read_json(final_path), FINAL_SCHEMA)
        if current.get("task_id") != task_id:
            raise PostdeadlineContractError(
                "existing final seal task identity drifted"
            )
        return final_path
    return write_immutable_json(final_path, final)


def _reconcile_consumed(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    reader: JsonReader,
) -> Path:
    root = OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    intent_path = root / INTENT_NAME
    attempt_path = OUTPUT_ROOT.resolve() / ATTEMPT_NAME
    if not intent_path.is_file() or not attempt_path.is_file():
        raise PostdeadlineContractError(
            "consumed attempt lacks immutable intent/ledger"
        )
    validate_seal(read_json(intent_path), INTENT_SCHEMA)
    validate_seal(read_json(attempt_path), ATTEMPT_SCHEMA)
    exact = _inventory_exact(reader, dedupe_key=plan["dedupe_key"])
    if len(exact) != 1:
        _write_ambiguous(
            root=root,
            plan=plan,
            attempt_path=attempt_path,
            http_status=None,
            response=None,
            error="consumed attempt exact GET reconciliation is not unique",
            exact_count=len(exact),
        )
        raise PostdeadlineContractError(
            "POST attempt already consumed; exact GET is not unique"
        )
    task_id = int(exact[0].get("task_id") or exact[0].get("id") or 0)
    readback = _authenticated_readback(
        reader, task_id=task_id, dedupe_key=plan["dedupe_key"]
    )
    return _finish(
        root=root,
        plan_path=plan_path,
        plan=plan,
        intent_path=intent_path,
        attempt_path=attempt_path,
        task_id=task_id,
        readback=readback,
        http_status=None,
        response=None,
        error="reconciled consumed attempt by exact GET",
        reconciled=True,
    )


def submit(
    *,
    authorize_post: str,
    plan_path: Path | None = None,
    reader: JsonReader = get_json,
    poster: Poster = reviewed._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    plan_loader: Callable[[Path | None], dict[str, Any]] = load_plan,
) -> Path:
    """Consume no more than one lifetime POST after immutable evidence."""

    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit direct fast-lane one-shot authorization is absent"
        )
    resolved_plan = (
        plan_path or (OUTPUT_ROOT / PLAN_NAME)
    ).resolve(strict=True)
    plan = plan_loader(resolved_plan)
    root = OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    attempt_path = OUTPUT_ROOT.resolve() / ATTEMPT_NAME
    final_path = root / FINAL_NAME
    if final_path.is_file():
        final = validate_seal(read_json(final_path), FINAL_SCHEMA)
        _authenticated_readback(
            reader,
            task_id=int(final["task_id"]),
            dedupe_key=plan["dedupe_key"],
        )
        return final_path
    if attempt_path.exists():
        return _reconcile_consumed(
            plan_path=resolved_plan, plan=plan, reader=reader
        )
    initial_gate = live_gate(
        expected_dedupe_key=plan["dedupe_key"],
        reader=reader,
        observed_at=observed_at,
    )
    root.mkdir(parents=True, exist_ok=False)
    intent = sealed(
        {
            "schema_version": INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(resolved_plan),
            "plan_payload_sha256": plan["payload_sha256"],
            "initial_live_gate": initial_gate,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
            "authorization": POST_AUTHORIZATION,
            "post_call_budget": 1,
            "attempt_ledger_must_precede_network": True,
            "ambiguous_retry_allowed": False,
            "automatic_full_trigger": False,
        }
    )
    intent_path = write_immutable_json(root / INTENT_NAME, intent)
    with lock_factory():
        if attempt_path.exists():
            raise PostdeadlineContractError(
                "direct fast-lane POST attempt was consumed concurrently"
            )
        locked_gate = live_gate(
            expected_dedupe_key=plan["dedupe_key"],
            reader=reader,
            observed_at=observed_at,
        )
        attempt = sealed(
            {
                "schema_version": ATTEMPT_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan": file_record(resolved_plan),
                "intent": file_record(intent_path),
                "initial_live_gate": initial_gate,
                "locked_pre_submit_live_gate": locked_gate,
                "attempt_nonce": plan["single_attempt_contract"][
                    "attempt_nonce"
                ],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "campaign_mutation_lock_acquired": True,
                "post_call_budget": 1,
                "post_call_consumed_before_network": True,
                "scheduler_post_calls_before": 0,
                "scheduler_post_calls_authorized": 1,
                "ambiguous_retry_allowed": False,
            }
        )
        attempt_path = write_exclusive_json(attempt_path, attempt)
        try:
            status, response, error = poster(
                f"{SCHEDULER_URL}/api/tasks",
                plan["scheduler_payload"],
            )
        except Exception as exc:
            status, response, error = None, None, str(exc)
    exact = _inventory_exact(reader, dedupe_key=plan["dedupe_key"])
    response_task_id = (
        response.get("task_id") or response.get("id")
        if isinstance(response, Mapping)
        else None
    )
    exact_task_id = (
        exact[0].get("task_id") or exact[0].get("id")
        if len(exact) == 1
        else None
    )
    if (
        len(exact) != 1
        or exact_task_id is None
        or (
            response_task_id is not None
            and int(response_task_id) != int(exact_task_id)
        )
    ):
        _write_ambiguous(
            root=root,
            plan=plan,
            attempt_path=attempt_path,
            http_status=status,
            response=response,
            error=error,
            exact_count=len(exact),
        )
        raise PostdeadlineContractError(
            "POST consumed but exact GET reconciliation failed; no retry"
        )
    task_id = int(exact_task_id)
    readback = _authenticated_readback(
        reader, task_id=task_id, dedupe_key=plan["dedupe_key"]
    )
    return _finish(
        root=root,
        plan_path=resolved_plan,
        plan=plan,
        intent_path=intent_path,
        attempt_path=attempt_path,
        task_id=task_id,
        readback=readback,
        http_status=status,
        response=response,
        error=error,
        reconciled=(status != 201 or response_task_id is None),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="GET-gate and seal a no-POST direct plan"
    )
    prepare_parser.add_argument("--solver-revision", required=True)
    prepare_parser.add_argument(
        "--same-node-task-id", required=True, type=int
    )
    inspect_parser = commands.add_parser(
        "inspect", help="reauthenticate the plan and repeat GET gates"
    )
    inspect_parser.add_argument(
        "--same-node-task-id", required=True, type=int
    )
    submit_parser = commands.add_parser(
        "submit", help="consume the one-shot POST budget"
    )
    submit_parser.add_argument(
        "--same-node-task-id", required=True, type=int
    )
    submit_parser.add_argument("--authorize-post", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.same_node_task_id != SAME_NODE_AS_TASK_ID:
        raise PostdeadlineContractError(
            "exact same-node source task96332 authorization is absent"
        )
    if args.command == "prepare":
        plan_path = prepare(solver_revision=args.solver_revision)
        plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
        result = {
            "event": "prepared",
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "solver_revision": plan["solver_revision"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "scheduler_post_calls": 0,
        }
    elif args.command == "inspect":
        plan = load_plan()
        gate = live_gate(expected_dedupe_key=plan["dedupe_key"])
        result = {
            "event": "admissible",
            "plan": str(OUTPUT_ROOT / PLAN_NAME),
            "solver_revision": plan["solver_revision"],
            "gate": gate,
            "scheduler_post_calls": 0,
        }
    else:
        final_path = submit(authorize_post=args.authorize_post)
        final = validate_seal(read_json(final_path), FINAL_SCHEMA)
        result = {
            "event": "submitted",
            "final_seal": str(final_path),
            "task_id": final["task_id"],
            "scheduler_post_calls": final[
                "scheduler_post_calls_evidenced"
            ],
        }
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PostdeadlineContractError as exc:
        print(
            json.dumps(
                {
                    "event": "direct_analyze_fastlane_error",
                    "error": str(exc),
                    "scheduler_post_calls_unproven": 0,
                    "automatic_full_trigger": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
