"""GET-only terminal collector for the exact official5 direct task 96338.

This module is intentionally task-specific.  It authenticates the immutable
official5 submission package, the same-node Scheduler placement, the direct
thermal Analyze opt-in, the revision-bound eight-core policy, the unchanged
scientific gates, and the retained symmetric AEDT bundle.  It never mutates
Scheduler state.
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
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import parse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_RESONANCE_MIN_HZ,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    TEMPERATURE_FAMILY_LIMITS_C,
    TEMPERATURE_TARGET_FAMILIES,
    TEMPERATURE_TARGET_LIMITS_C,
    attest_fixed_identity,
)
from tools import mft_goal_postdeadline_standard_collector as base  # noqa: E402


TASK_ID = 96_338
TASK_NAME = (
    "mft-goal-diag-standard-official5-direct-analyze-samenode-r1-v2-"
    "909d249ebe45-n115"
)
DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-official5-direct-analyze-samenode-r1-v2-"
    "909d249ebe45-n115:d4e78cfa757888f7c47964aa392e4ff96f1e453f:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:12aa95d67bc957bc"
)
CANDIDATE_PHYSICS_SHA256 = (
    "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
)
SOLVER_REVISION = "d4e78cfa757888f7c47964aa392e4ff96f1e453f"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PROFILE_SHA256 = (
    "eecac71177e1af05fdc06deb4bea9525dc837cab80628f9d681004aeff0aa426"
)
PARAMETER_DIGEST = "12aa95d67bc957bc"
CORE_AUTH_SHA256 = (
    "0305537ff06f209faa990c87240262bd5b6da4b8166a6205b16c870334ce8fd5"
)
CORE_CONTRACT = "mft-standalone-core-optin-v1"
DIRECT_ENV_TOKEN = "standard-eighth-direct-analyze-v1"
DIRECT_CONTRACT = "mft-symmetry-thermal-direct-analyze-v1"
DIRECT_PREFLIGHT_STATUS = "explicit_opt_in_symmetry_direct_analyze"
SAME_NODE_TASK_ID = 96_332
ALLOCATION_ID = 14_650
SLURM_JOB_ID = "840582"
NODE_NAME = "n115"
ACCOUNT_NAME = "jji0930"
SCHEDULER_PROJECT = "MFT_1MW_2026v1"
SCHEDULER_URL = "http://127.0.0.1:8002"
RETAINED_ROOT = "goal-fea-retained/aa68dae03113f244"
MARKER_CONTRACT_SHA256 = (
    "59f833e07ea1aaa7c50afb31ce696567279bb5724394826774489d41cba88bf6"
)

PRIOR_FAILURE_TASK_ID = 96_337
PRIOR_FAILURE_TASK_NAME = (
    "mft-goal-diag-standard-official5-direct-analyze-samenode-v1-"
    "909d249ebe45-n115"
)
PRIOR_FAILURE_DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-official5-direct-analyze-samenode-v1-"
    "909d249ebe45-n115:d4e78cfa757888f7c47964aa392e4ff96f1e453f:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:12aa95d67bc957bc"
)

DEFAULT_SOURCE_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official5_direct_analyze_samenode_r1_v2"
)
DEFAULT_CAMPAIGN_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official5_direct_analyze_retry_task96338_v1"
)
DEFAULT_OUTPUT = (
    DEFAULT_CAMPAIGN_ROOT / "authenticated_get_collection_task96338_v1"
)
DEFAULT_PRIOR_FAILURE_LEDGER = (
    DEFAULT_CAMPAIGN_ROOT / "terminal_failure_task96337_v1.failure_ledger.json"
)

PLAN_SCHEMA = "mft-goal-official5-direct-analyze-plan-v1"
PREPARE_SCHEMA = "mft-goal-official5-direct-analyze-prepare-receipt-v1"
POST_ATTEMPT_SCHEMA = "mft-goal-official5-direct-analyze-post-attempt-v1"
INTENT_SCHEMA = "mft-goal-official5-direct-analyze-submit-intent-v1"
SUBMISSION_SCHEMA = "mft-goal-official5-direct-analyze-submission-v1"
FINAL_SCHEMA = "mft-goal-official5-direct-analyze-final-v1"
COLLECTION_SCHEMA = "mft-goal-official5-direct-terminal-collection-v1"
COLLECTION_SEAL_SCHEMA = (
    "mft-goal-official5-direct-terminal-collection-seal-v1"
)
AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-official5-direct-authenticated-collection-v1"
)
SCIENTIFIC_GATE_SCHEMA = "mft-goal-official5-direct-scientific-gates-v1"
FAILURE_SCHEMA = "mft-goal-official5-direct-terminal-failure-ledger-v1"
PRIOR_FAILURE_SCHEMA = "mft-goal-official5-direct-pre-em-failure-ledger-v1"
POLL_SCHEMA = "mft-goal-official5-direct-terminal-poll-v1"

CLASSIFICATION = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "production_eligible": False,
    "original_deadline_missed": True,
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
MAX_LOG_BYTES = 16 * 1024 * 1024
ACTIVE_STATES = {"queued", "attaching", "running"}
TERMINAL_SUCCESS = {("completed", "succeeded")}
TERMINAL_FAILURE_STATES = {"failed", "cancelled", "timed_out", "timeout"}

SOURCE_FILE_SHA256 = {
    "direct_analyze_plan.json": (
        "03e14ff007c6fa9ed8c5b01af1013f31233ca673acb008f1659d6834018d561d"
    ),
    "execution_profile.json": (
        "6ec34d9555434b10050c07edc589dcdc1b6aabcacafd738528bbb53d733261f8"
    ),
    "fea_params.json": (
        "45c0790fe91d0135dcc2d8598bc85d8e849da5f3a9ba30a7ce6d881939011bb1"
    ),
    "prepare_receipt.json": (
        "fa58c39fdcdb90f2821c4eaac418a2aeb72a55047a1589e74c143b7d46282854"
    ),
    "scheduler_post_attempt.json": (
        "760ad062a760c99be94d025a6e585b732a11859858e18654b21a574e149be936"
    ),
    "selected_candidate.json": (
        "3c702f8f7c4e10205344c3e0f97b18a246c6e7f439cda8ba62b755754a22c65e"
    ),
    "submission/final_seal.json": (
        "4e2802cd321727114926da7af8631d8f267fab37fa801d966966fdb54b75946e"
    ),
    "submission/submission_intent.json": (
        "af5b0a3706f52673fac2963683a4c4cfd8839c7b912e9c15e5760f4fd8bb69ff"
    ),
    "submission/submission_receipt.json": (
        "0731cf22e3f78f93da943cc9102b7d96a2719bfbe1ca65e782789b2d98bc30dd"
    ),
}

FIXED_BOUNDARY = {
    "core_plate_on": 1,
    "core_plate_pad_t_mm": 2.0,
    "fan_config": "dual",
    "fan_velocity_m_s": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "wcp_on": 1,
    "wcp_pad_t_mm": 2.0,
}
SUBMISSION_ENVIRONMENT = {
    "MFT_STANDALONE_CORE_AUTH_SHA256": CORE_AUTH_SHA256,
    "MFT_STANDALONE_CORE_CONTRACT": CORE_CONTRACT,
    "MFT_STANDALONE_CORE_COUNT": "8",
    "MFT_SYMMETRY_THERMAL_DIRECT_ANALYZE": DIRECT_ENV_TOKEN,
}


CollectionError = base.CollectionError
canonical_bytes = base.canonical_bytes
payload_sha256 = base.payload_sha256
seal = base.seal
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
file_record = base.file_record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"{label} must be a JSON object")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = copy.deepcopy(dict(value))
    expected = body.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(expected, str)
        or not HEX64.fullmatch(expected)
        or payload_sha256(body) != expected
    ):
        raise CollectionError(f"{label} seal drifted")


def _write_atomic(path: Path, data: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() == data:
            return
        raise CollectionError(f"immutable output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json(
    path: Path, value: Mapping[str, Any], *, replace: bool = False
) -> None:
    _write_atomic(path, canonical_bytes(value) + b"\n", replace=replace)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CollectionError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CollectionError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise CollectionError(f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer():
        raise CollectionError(f"{label} must be integral")
    return int(number)


def _numeric_equal(actual: Any, expected: Any) -> bool:
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _verify_record(
    record: Any, *, root: Path, expected_name: str, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise CollectionError(f"{label} file record is absent")
    candidate = Path(str(record.get("path") or ""))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    if resolved.name != expected_name:
        raise CollectionError(f"{label} filename drifted")
    if (
        record.get("sha256") != sha256_file(resolved)
        or record.get("size_bytes") != resolved.stat().st_size
    ):
        raise CollectionError(f"{label} file record drifted")
    return resolved


def _selected_adapter(
    selected: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    official_row = selected.get("official_row")
    decoded = selected.get("decoded_physical_params")
    authentication = selected.get("authentication")
    if (
        not isinstance(official_row, Mapping)
        or not isinstance(decoded, Mapping)
        or not isinstance(authentication, Mapping)
        or official_row.get("candidate_physics_sha")
        != CANDIDATE_PHYSICS_SHA256
        or authentication.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
    ):
        raise CollectionError("official5 selected candidate identity drifted")
    return {
        "schema_version": "mft-goal-official5-selected-adapter-v1",
        "selected_row": copy.deepcopy(dict(official_row)),
        "row_contract": {
            "decoded_params": copy.deepcopy(dict(decoded)),
            "fea_params_sha256": payload_sha256(params),
        },
        "task_identity": {
            "seed": _integer(authentication.get("source_seed"), "source seed"),
            "fixed_primary_turns": _integer(
                params.get("N1_main"), "primary turns"
            ),
        },
        "official5_source": copy.deepcopy(dict(selected)),
    }


def load_contract(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    scheduler_url: str = SCHEDULER_URL,
) -> dict[str, Any]:
    """Authenticate the exact immutable official5 task96338 source package."""

    root = source_root.resolve(strict=True)
    paths: dict[str, Path] = {}
    for relative, expected_sha in SOURCE_FILE_SHA256.items():
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CollectionError("official5 source path escaped its root") from exc
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise CollectionError(f"official5 source bytes drifted: {relative}")
        paths[relative] = path

    plan = _read_json(paths["direct_analyze_plan.json"], "official5 plan")
    prepare = _read_json(paths["prepare_receipt.json"], "prepare receipt")
    attempt = _read_json(
        paths["scheduler_post_attempt.json"], "Scheduler attempt ledger"
    )
    intent = _read_json(
        paths["submission/submission_intent.json"], "submission intent"
    )
    submission = _read_json(
        paths["submission/submission_receipt.json"], "submission receipt"
    )
    final = _read_json(paths["submission/final_seal.json"], "final seal")
    selected = _read_json(
        paths["selected_candidate.json"], "selected candidate"
    )
    params = _read_json(paths["fea_params.json"], "FEA parameters")
    profile = _read_json(paths["execution_profile.json"], "execution profile")

    for value, schema, label in (
        (plan, PLAN_SCHEMA, "official5 plan"),
        (prepare, PREPARE_SCHEMA, "prepare receipt"),
        (attempt, POST_ATTEMPT_SCHEMA, "Scheduler attempt ledger"),
        (intent, INTENT_SCHEMA, "submission intent"),
        (submission, SUBMISSION_SCHEMA, "submission receipt"),
        (final, FINAL_SCHEMA, "final seal"),
        (
            selected,
            str(selected.get("schema_version") or ""),
            "selected candidate",
        ),
    ):
        _validate_seal(value, schema, label)

    expected_common = {
        "task_name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
    }
    for label, value in (
        ("plan", plan),
        ("prepare", prepare),
        ("intent", intent),
        ("submission", submission),
    ):
        for key, expected in expected_common.items():
            if value.get(key) != expected:
                raise CollectionError(f"{label} exact identity drifted: {key}")
    if (
        plan.get("candidate_physics_sha256") != CANDIDATE_PHYSICS_SHA256
        or prepare.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or submission.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or submission.get("task_id") != TASK_ID
        or final.get("task_id") != TASK_ID
        or plan.get("solver_revision") != SOLVER_REVISION
        or plan.get("library_revision") != LIBRARY_REVISION
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("submission_environment") != SUBMISSION_ENVIRONMENT
        or plan.get("full_model") is not False
        or plan.get("symmetric_model") is not True
        or str(plan.get("thermal_symmetry") or "").lower() != "eighth"
        or plan.get("direct_analyze_fast_lane") is not True
        or plan.get("direct_analyze_opt_in", {}).get(
            "standalone_core_auth_sha256"
        )
        != CORE_AUTH_SHA256
    ):
        raise CollectionError("official5 plan hard identity drifted")

    for key, record_name in (
        ("fea_params", "fea_params.json"),
        ("selected_candidate", "selected_candidate.json"),
        ("execution_profile", "execution_profile.json"),
    ):
        resolved = _verify_record(
            plan.get(key),
            root=root,
            expected_name=record_name,
            label=f"plan {key}",
        )
        if resolved != paths[record_name]:
            raise CollectionError(f"plan source binding drifted: {key}")

    scheduler_payload = plan.get("scheduler_payload")
    resources = plan.get("resources")
    placement = plan.get("placement")
    if (
        not isinstance(scheduler_payload, Mapping)
        or payload_sha256(scheduler_payload)
        != plan.get("scheduler_payload_sha256")
        or not isinstance(resources, Mapping)
        or not isinstance(placement, Mapping)
    ):
        raise CollectionError("official5 Scheduler envelope is malformed")
    scheduler_expected = {
        "name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "project": SCHEDULER_PROJECT,
        "account_name": ACCOUNT_NAME,
        "cpus": 8,
        "memory_mb": 98_304,
        "gpus": 0,
        "max_workers_per_node": 2,
        "timeout_seconds": 45_300,
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "same_node_as_task_id": SAME_NODE_TASK_ID,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "priority": 100,
    }
    for key, expected in scheduler_expected.items():
        if scheduler_payload.get(key) != expected:
            raise CollectionError(f"sealed Scheduler payload drifted: {key}")
    if (
        resources
        != {
            "cpus": 8,
            "kill_grace_seconds": 300,
            "max_workers_per_node": 2,
            "memory_mb": 98_304,
            "retention_seconds": 1_800,
            "scheduler_timeout_seconds": 45_300,
            "solver_seconds": 43_200,
        }
        or placement.get("same_node_as_task_id") != SAME_NODE_TASK_ID
        or placement.get("same_node_as_allocation_id") != ALLOCATION_ID
        or placement.get("same_node_as_slurm_job_id") != SLURM_JOB_ID
        or placement.get("node_name") != NODE_NAME
        or placement.get("node_name_policy") != "strict"
        or placement.get("account_name") != ACCOUNT_NAME
    ):
        raise CollectionError("official5 resource/placement contract drifted")

    command = str(scheduler_payload.get("command") or "")
    required_fragments = (
        f"git fetch -q origin {SOLVER_REVISION}",
        f"git checkout -q --detach {SOLVER_REVISION}",
        f"fetch -q origin {LIBRARY_REVISION}",
        f"checkout -q --detach {LIBRARY_REVISION}",
        f'MFT_STANDALONE_CORE_AUTH_SHA256="{CORE_AUTH_SHA256}"',
        f'MFT_STANDALONE_CORE_CONTRACT="{CORE_CONTRACT}"',
        'MFT_STANDALONE_CORE_COUNT="8"',
        f'MFT_SYMMETRY_THERMAL_DIRECT_ANALYZE="{DIRECT_ENV_TOKEN}"',
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--symmetry-thermal-direct-analyze --params cand.json",
    )
    if any(command.count(fragment) != 1 for fragment in required_fragments):
        raise CollectionError("official5 direct/core command authentication failed")

    effective = copy.deepcopy(params)
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, Mapping):
        raise CollectionError("official5 profile overrides are malformed")
    effective.update(overrides)
    parameter_json = json.dumps(
        effective, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    if (
        payload_sha256(profile) != PROFILE_SHA256
        or plan.get("execution_profile_sha256") != PROFILE_SHA256
        or sha256_bytes(parameter_json.encode("utf-8"))[:16]
        != PARAMETER_DIGEST
        or plan.get("fea_params_sha256") != payload_sha256(params)
        or effective.get("fan_velocity") != 1.5
        or effective.get("k_ins") != 0.2
        or effective.get("thermal_symmetry") != "eighth"
        or effective.get("full_model") != 0
        or effective.get("thermal_on") != 1
        or effective.get("loss_on") != 1
        or effective.get("thermal_max_iterations") != 250
        or effective.get("thermal_rx_side_block_mesh_level") != 5
    ):
        raise CollectionError("official5 effective FEA parameter identity drifted")

    retained = plan.get("retained_aedt_bundle")
    if not isinstance(retained, Mapping):
        raise CollectionError("official5 retained bundle contract is absent")
    expected_retained = {
        "relative_directory": RETAINED_ROOT,
        "artifact_path": f"{RETAINED_ROOT}/symmetric.aedt",
        "receipt_path": f"{RETAINED_ROOT}/symmetric.aedt.receipt.json",
        "marker_path": f"{RETAINED_ROOT}/.slurm-scheduler-preserve.json",
        "results_path": f"{RETAINED_ROOT}/symmetric.aedtresults",
        "results_manifest_path": (
            f"{RETAINED_ROOT}/symmetric.aedtresults.manifest.json"
        ),
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "profile_sha256": PROFILE_SHA256,
        "parameter_digest": PARAMETER_DIGEST,
        "dedupe_key": DEDUPE_KEY,
        "marker_contract_sha256": MARKER_CONTRACT_SHA256,
    }
    for key, expected in expected_retained.items():
        if retained.get(key) != expected:
            raise CollectionError(f"retained bundle contract drifted: {key}")
    transport = retained.get("transport")
    if (
        not isinstance(transport, Mapping)
        or transport.get("chunk_directory")
        != f"{RETAINED_ROOT}/symmetric.aedt.chunks"
        or transport.get("raw_chunk_bytes") != base.RAW_CHUNK_BYTES
        or transport.get("max_encoded_chunk_bytes")
        != base.MAX_ENCODED_CHUNK_BYTES
        or transport.get("schema_version") != base.TRANSPORT_SCHEMA
        or transport.get("encoding") != "base64"
    ):
        raise CollectionError("retained bundle transport contract drifted")

    readback = submission.get("task_readback")
    if (
        not isinstance(readback, Mapping)
        or readback.get("task_id", readback.get("id")) != TASK_ID
        or readback.get("same_node_as_task_id") != SAME_NODE_TASK_ID
        or readback.get("same_node_as_allocation_id") != ALLOCATION_ID
        or readback.get("allocation_id") != ALLOCATION_ID
        or str(readback.get("slurm_job_id") or "") != SLURM_JOB_ID
        or readback.get("actual_node_name") != NODE_NAME
        or readback.get("placement_contract_satisfied") is not True
        or submission.get("same_node_as_task_id") != SAME_NODE_TASK_ID
        or submission.get("same_node_as_allocation_id") != ALLOCATION_ID
        or submission.get("solver_revision") != SOLVER_REVISION
        or final.get("submission_receipt_payload_sha256")
        != submission.get("payload_sha256")
    ):
        raise CollectionError("official5 submission placement lineage drifted")

    marker_contract = retained.get("marker_contract")
    if (
        not isinstance(marker_contract, Mapping)
        or payload_sha256(marker_contract) != MARKER_CONTRACT_SHA256
    ):
        raise CollectionError("official5 prune marker contract drifted")

    selected_adapter = _selected_adapter(selected, params)
    return {
        "plan": plan,
        "prepare": prepare,
        "attempt": attempt,
        "intent": intent,
        "submission": submission,
        "final": final,
        "params": params,
        "effective_params": effective,
        "profile": profile,
        "selected_source": selected,
        "selected": selected_adapter,
        "source_root": root,
        "source_paths": paths,
        "source_inventory": {
            name: file_record(path, relative_to=root)
            for name, path in sorted(paths.items())
        },
        "plan_path": paths["direct_analyze_plan.json"],
        "submission_path": paths[
            "submission/submission_receipt.json"
        ],
        "plan_file_sha256": SOURCE_FILE_SHA256[
            "direct_analyze_plan.json"
        ],
        "submission_file_sha256": SOURCE_FILE_SHA256[
            "submission/submission_receipt.json"
        ],
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "profile_sha256": PROFILE_SHA256,
        "parameter_digest": PARAMETER_DIGEST,
        "scheduler_url": scheduler_url.rstrip("/"),
        "node_name": NODE_NAME,
        "same_node_as_task_id": SAME_NODE_TASK_ID,
        "same_node_as_allocation_id": ALLOCATION_ID,
        "slurm_job_id": SLURM_JOB_ID,
        "retained": {
            "root": RETAINED_ROOT,
            "artifact_path": f"{RETAINED_ROOT}/symmetric.aedt",
            "receipt_path": f"{RETAINED_ROOT}/symmetric.aedt.receipt.json",
            "marker_path": f"{RETAINED_ROOT}/.slurm-scheduler-preserve.json",
            "chunk_directory": (
                f"{RETAINED_ROOT}/symmetric.aedt.chunks"
            ),
            "results_path": f"{RETAINED_ROOT}/symmetric.aedtresults",
            "results_manifest_path": (
                f"{RETAINED_ROOT}/symmetric.aedtresults.manifest.json"
            ),
            "marker_contract": copy.deepcopy(dict(marker_contract)),
            "marker_contract_sha256": MARKER_CONTRACT_SHA256,
        },
    }


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"{label} must be an object")
    return value


def get_task(
    contract: Mapping[str, Any],
    *,
    getter: Callable[..., bytes] = base.http_get,
) -> dict[str, Any]:
    raw = getter(
        f"{contract['scheduler_url']}/api/tasks/{TASK_ID}",
        max_bytes=1024 * 1024,
        timeout=30.0,
    )
    task = _decode_json(raw, "Scheduler task GET")
    expected = {
        "name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "project": SCHEDULER_PROJECT,
        "account_name": ACCOUNT_NAME,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "same_node_as_node_name": NODE_NAME,
        "allocation_id": ALLOCATION_ID,
        "assigned_allocation": ALLOCATION_ID,
        "same_node_as_allocation_id": ALLOCATION_ID,
        "same_node_as_task_id": SAME_NODE_TASK_ID,
        "slurm_job_id": SLURM_JOB_ID,
        "placement_contract_satisfied": True,
        "strict_node_placement": True,
        "preferred_node_relaxed": False,
        "cpus": 8,
        "memory_mb": 98_304,
        "gpus": 0,
        "max_workers_per_node": 2,
        "timeout_seconds": 45_300,
        "priority": 100,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
    }
    if task.get("task_id", task.get("id")) != TASK_ID:
        raise CollectionError("Scheduler task identity drifted: task_id")
    if task.get("id", TASK_ID) != TASK_ID:
        raise CollectionError("Scheduler task identity drifted: id")
    for key, expected_value in expected.items():
        if task.get(key) != expected_value:
            raise CollectionError(f"Scheduler task identity drifted: {key}")
    remote_dir = str(task.get("remote_dir") or "")
    if not remote_dir.startswith(
        "slurm_scheduler/runs/2026-07-26/task-96338-"
    ):
        raise CollectionError("Scheduler task remote directory drifted")
    return task


def _get_log(
    contract: Mapping[str, Any],
    stream: str,
    *,
    getter: Callable[..., bytes] = base.http_get,
    max_bytes: int = MAX_LOG_BYTES,
) -> bytes:
    if stream not in {"stdout", "stderr"}:
        raise CollectionError("unsupported Scheduler log stream")
    query = parse.urlencode({"max_bytes": max_bytes})
    return getter(
        f"{contract['scheduler_url']}/api/tasks/{contract['task_id']}/"
        f"{stream}?{query}",
        max_bytes=max_bytes,
        timeout=120.0,
    )


def get_result(
    contract: Mapping[str, Any],
    *,
    getter: Callable[..., bytes] = base.http_get,
) -> tuple[dict[str, Any], bytes]:
    stdout = _get_log(contract, "stdout", getter=getter)
    try:
        text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise CollectionError("Scheduler stdout is not UTF-8") from exc
    result: dict[str, Any] | None = None
    library_marker: str | None = None
    for line in reversed(text.splitlines()):
        if library_marker is None and line.startswith("MFT_LIBRARY_GIT_HASH "):
            candidate = line.removeprefix("MFT_LIBRARY_GIT_HASH ").strip()
            if HEX40.fullmatch(candidate):
                library_marker = candidate
        if result is None and line.startswith("RESULT_JSON "):
            try:
                candidate_result = json.loads(
                    line.removeprefix("RESULT_JSON ")
                )
            except json.JSONDecodeError:
                continue
            if isinstance(candidate_result, dict):
                result = candidate_result
        if result is not None and library_marker is not None:
            break
    if result is None or library_marker != LIBRARY_REVISION:
        raise CollectionError("valid RESULT_JSON/library marker is absent")
    if (
        result.get("git_hash") != SOLVER_REVISION
        or result.get("pyaedt_library_git_hash") != LIBRARY_REVISION
        or _integer(result.get("git_dirty"), "solver dirty flag") != 0
        or _integer(
            result.get("pyaedt_library_git_dirty"), "library dirty flag"
        )
        != 0
    ):
        raise CollectionError("RESULT_JSON revision identity drifted")
    for key, expected in contract["effective_params"].items():
        if key not in result:
            raise CollectionError(f"RESULT_JSON omitted submitted input: {key}")
        actual = result[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matched = _numeric_equal(actual, expected)
        else:
            matched = str(actual) == str(expected)
        if not matched:
            raise CollectionError(f"RESULT_JSON submitted input drifted: {key}")
    return result, stdout


def _validate_direct_analyze(result: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "thermal_mesh_preflight_status": DIRECT_PREFLIGHT_STATUS,
        "thermal_mesh_native_generation_passed": 0,
        "thermal_mesh_direct_analyze_opt_in": 1,
        "thermal_mesh_direct_analyze_contract_version": DIRECT_CONTRACT,
        "thermal_mesh_unmeshed_object_count": 0,
        "thermal_mesh_unmeshed_objects_json": "[]",
    }
    for key, expected_value in expected.items():
        actual = result.get(key)
        if isinstance(expected_value, int):
            passed = _numeric_equal(actual, expected_value)
        else:
            passed = actual == expected_value
        if not passed:
            raise CollectionError(f"direct Analyze evidence drifted: {key}")
    contract_sha = str(
        result.get("thermal_mesh_direct_analyze_contract_sha256") or ""
    )
    if not HEX64.fullmatch(contract_sha):
        raise CollectionError("direct Analyze contract SHA is absent")
    return {
        "authenticated": True,
        "environment_name": "MFT_SYMMETRY_THERMAL_DIRECT_ANALYZE",
        "environment_token": DIRECT_ENV_TOKEN,
        "contract_version": DIRECT_CONTRACT,
        "contract_sha256": contract_sha,
        "preflight_status": DIRECT_PREFLIGHT_STATUS,
        "native_generate_mesh_bypassed_by_explicit_opt_in": True,
        "unmeshed_object_count": 0,
        "cooling_boundary_modified": False,
        "scientific_truth_gates_unchanged": True,
    }


def _parse_json_object(value: Any, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise CollectionError(f"{label} JSON is invalid") from exc
    if not isinstance(parsed, dict):
        raise CollectionError(f"{label} JSON must be an object")
    return parsed


def _validate_core_policy(result: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": CORE_CONTRACT,
        "solver_core_backend": "standalone",
        "solver_core_slurm_cpus_per_task_readback": "8",
        "solver_core_scheduler_task_id_readback": str(TASK_ID),
        "solver_core_slurm_job_id_readback": SLURM_JOB_ID,
        "solver_core_auth_sha256": CORE_AUTH_SHA256,
        "solver_core_license_contract": "",
        "solver_core_license_snapshot_sha256": "",
    }
    for key, expected in exact.items():
        if str(result.get(key) or "") != expected:
            raise CollectionError(f"solver core evidence drifted: {key}")
    integer_exact = {
        "solver_core_opt_in": 1,
        "solver_num_cores_requested": 8,
        "solver_num_cores_effective": 8,
        "solver_num_tasks_effective": 1,
        "solver_matrix_hpc_num_cores_readback": 8,
        "solver_matrix_hpc_num_engines_readback": 1,
    }
    for key, expected in integer_exact.items():
        if _integer(result.get(key), key) != expected:
            raise CollectionError(f"solver core evidence drifted: {key}")
    affinity = _integer(
        result.get("solver_core_affinity_count_readback"),
        "solver core affinity",
    )
    acf_sha = str(result.get("solver_matrix_hpc_acf_sha256") or "")
    if affinity < 8 or not HEX64.fullmatch(acf_sha):
        raise CollectionError("solver matrix HPC readback drifted")
    dispatch = _parse_json_object(
        result.get("solver_core_dispatch_evidence_json"),
        "solver core dispatch evidence",
    )
    readback = _parse_json_object(
        result.get("solver_core_readback_evidence_json"),
        "solver core readback evidence",
    )
    matrix = readback.get("matrix_hpc_acf")
    if (
        not dispatch
        or not isinstance(matrix, Mapping)
        or _integer(matrix.get("num_cores_readback"), "matrix ACF cores") != 8
        or _integer(matrix.get("num_engines_readback"), "matrix ACF engines")
        != 1
        or matrix.get("acf_sha256") != acf_sha
    ):
        raise CollectionError("solver core structured readback drifted")
    return {
        "authenticated": True,
        "contract_version": CORE_CONTRACT,
        "auth_sha256": CORE_AUTH_SHA256,
        "requested_cores": 8,
        "effective_cores": 8,
        "num_tasks": 1,
        "affinity_count_readback": affinity,
        "scheduler_task_id_readback": str(TASK_ID),
        "slurm_job_id_readback": SLURM_JOB_ID,
        "matrix_hpc_num_cores_readback": 8,
        "matrix_hpc_num_engines_readback": 1,
        "matrix_hpc_acf_sha256": acf_sha,
        "dispatch_evidence_sha256": payload_sha256(dispatch),
        "readback_evidence_sha256": payload_sha256(readback),
    }


def _validate_thermal_truth(result: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "result_valid_em",
        "result_valid_thermal",
        "thermal_solved",
        "thermal_convergence_available",
        "thermal_converged",
        "thermal_extraction_complete",
        "thermal_rx_power_balance_ok",
    ):
        if _integer(result.get(key), key) != 1:
            raise CollectionError(f"scientific truth gate failed: {key}")
    iterations = _integer(result.get("thermal_iterations"), "thermal iterations")
    if iterations <= 0:
        raise CollectionError("thermal iteration evidence is invalid")
    if _integer(
        result.get("thermal_required_missing_count"),
        "thermal required missing count",
    ) != 0:
        raise CollectionError("thermal result has missing required groups")
    required_mask = _integer(
        result.get("thermal_required_group_mask"), "thermal required mask"
    )
    if required_mask & 11 != 11 or required_mask & ~15:
        raise CollectionError("thermal required group mask drifted")
    if _finite(result.get("N2_side"), "N2_side") > 0 and not required_mask & 4:
        raise CollectionError("thermal side group is not required")
    rx_expected = _finite(
        result.get("thermal_rx_expected_power_w"), "thermal RX expected power"
    )
    rx_assigned = _finite(
        result.get("thermal_rx_assigned_power_w"), "thermal RX assigned power"
    )
    rx_error = _finite(
        result.get("thermal_rx_power_balance_max_abs_w"),
        "thermal RX balance error",
    )
    if (
        result.get("thermal_rx_model")
        not in {"homogenized_blocks", "hybrid_explicit"}
        or rx_expected < 0
        or rx_assigned < 0
        or not math.isclose(
            rx_assigned, rx_expected, rel_tol=1e-12, abs_tol=1e-9
        )
        or not 0 <= rx_error <= 1e-9
        or _finite(
            result.get("thermal_rx_power_balance_group_count"),
            "thermal RX balance groups",
        )
        < 1
    ):
        raise CollectionError("thermal RX power balance gate failed")
    flow_limit = _finite(
        result.get("thermal_residual_flow_limit"), "thermal flow limit"
    )
    energy_limit = _finite(
        result.get("thermal_residual_energy_limit"), "thermal energy limit"
    )
    flow_residuals = {
        key: _finite(result.get(key), key)
        for key in (
            "thermal_residual_continuity",
            "thermal_residual_x_velocity",
            "thermal_residual_y_velocity",
            "thermal_residual_z_velocity",
        )
    }
    energy_residual = _finite(
        result.get("thermal_residual_energy"), "thermal energy residual"
    )
    if (
        not 0 < flow_limit <= 1e-3
        or not 0 < energy_limit <= 1e-7
        or any(not 0 <= value <= flow_limit for value in flow_residuals.values())
        or not 0 <= energy_residual <= energy_limit
    ):
        raise CollectionError("thermal convergence residual gate failed")
    trusted = {}
    mandatory = (
        "T_max_Tx",
        "T_max_Rx_main",
        "T_max_core",
        "Tprobe_Tx_leeward_max",
        "Tprobe_Rx_main_leeward_max",
        "Tprobe_core_center_max",
    )
    side = ("T_max_Rx_side", "Tprobe_Rx_side_leeward_max")
    for key in (*mandatory, *side):
        value = _finite(result.get(key), key)
        if not -273.15 < value < 4_700.0:
            raise CollectionError(f"untrusted thermal temperature: {key}")
        trusted[key] = value
    if (
        str(result.get("physics_data_revision") or "")
        == "mft1mw-1k101-native-lamination-kf0p85-v3"
    ):
        for key in (
            "Tprobe_Rx_side_leeward_mean",
            "Tprobe_Rx_side_outer_max",
            "Tprobe_Rx_side_outer_mean",
            "Tprobe_Rx_side_inner_max",
            "Tprobe_Rx_side_inner_mean",
        ):
            value = _finite(result.get(key), key)
            if not -273.15 < value < 4_700.0:
                raise CollectionError(f"untrusted thermal temperature: {key}")
            trusted[key] = value
        if (
            result.get("thermal_rx_side_probe_contract_version")
            != "rx-side-transformer-inner-outer-v1"
            or result.get("thermal_rx_side_probe_max_rule")
            != "max_across_all_transformer_inner_and_outer_faces"
            or result.get("thermal_rx_side_probe_mean_rule")
            != "mean_of_face_selected_by_max_no_cross_face_average"
            or result.get("thermal_rx_side_probe_selected_face")
            not in {
                "Tprobe_Rx_side_side",
                "Tprobe_Rx_side1_inner",
                "Tprobe_Rx_side2_side",
                "Tprobe_Rx_side2_inner",
            }
            or _integer(
                result.get("thermal_rx_side_probe_face_count"),
                "thermal RX side probe face count",
            )
            != 2
        ):
            raise CollectionError("thermal RX side face probe contract drifted")
    return {
        "authenticated": True,
        "result_valid_em": True,
        "result_valid_thermal": True,
        "thermal_solved": True,
        "thermal_converged": True,
        "thermal_extraction_complete": True,
        "iterations": iterations,
        "required_group_mask": required_mask,
        "rx_power_balance_error_W": rx_error,
        "flow_residual_limit": flow_limit,
        "flow_residuals": flow_residuals,
        "energy_residual_limit": energy_limit,
        "energy_residual": energy_residual,
        "trusted_temperature_sha256": payload_sha256(trusted),
    }


def _bounding_box(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    l1 = _finite(result.get("l1"), "l1")
    l2 = _finite(result.get("l2"), "l2")
    h1 = _finite(result.get("h1"), "h1")
    w1 = _finite(result.get("w1"), "w1")
    n2_side = _integer(result.get("N2_side"), "N2_side")
    x_candidates = [
        4 * l1 + 2 * l2,
        _finite(result.get("sl1_main_x"), "sl1_main_x")
        + 2 * _finite(result.get("nwl1_main"), "nwl1_main"),
    ]
    y_candidates = [
        w1,
        _finite(result.get("sl1_main_y"), "sl1_main_y")
        + 2 * _finite(result.get("nwb1_main_y"), "nwb1_main_y"),
    ]
    if n2_side > 0:
        offset = l1 + l2 + l1 / 2
        side_x = (
            offset
            + _finite(result.get("sl2_side_x"), "sl2_side_x") / 2
            + _finite(result.get("nwl2_side"), "nwl2_side")
        )
        side_y = (
            _finite(result.get("sl2_side_y"), "sl2_side_y") / 2
            + _finite(result.get("nwl2_side"), "nwl2_side")
        )
        x_candidates.append(2 * side_x)
        y_candidates.append(2 * side_y)
    width = max(x_candidates)
    length = max(y_candidates)
    height = h1 + 2 * l1
    volume_l = width * length * height * 1e-6
    return volume_l, width, length, height


def _goal_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    volume, width, length, height = _bounding_box(result)
    resonance = _finite(
        result.get("f_res_min_tx_rx_only_Hz"),
        "minimum self resonance",
    )
    active_targets = []
    temperatures = {}
    family_values: dict[str, list[float]] = {"winding": [], "core": []}
    n2_side = _finite(result.get("N2_side"), "N2_side")
    for target, family in TEMPERATURE_TARGET_FAMILIES.items():
        if target in {
            "T_max_Rx_side",
            "Tprobe_Rx_side_leeward_max",
        } and n2_side <= 0:
            continue
        value = _finite(result.get(target), target)
        limit = float(TEMPERATURE_TARGET_LIMITS_C[target])
        active_targets.append(target)
        family_values[family].append(value)
        temperatures[target] = {
            "actual_C": value,
            "limit_C": limit,
            "passed": value <= limit,
        }
    if not family_values["winding"] or not family_values["core"]:
        raise CollectionError("temperature family evidence is incomplete")
    winding_max = max(family_values["winding"])
    core_max = max(family_values["core"])
    gates = {
        "width_mm": {
            "actual": width,
            "limit": float(GOAL_SIZE_LIMITS_MM["W"]),
            "relation": "<=",
            "passed": width <= float(GOAL_SIZE_LIMITS_MM["W"]),
        },
        "length_mm": {
            "actual": length,
            "limit": float(GOAL_SIZE_LIMITS_MM["L"]),
            "relation": "<=",
            "passed": length <= float(GOAL_SIZE_LIMITS_MM["L"]),
        },
        "height_mm": {
            "actual": height,
            "limit": float(GOAL_SIZE_LIMITS_MM["H"]),
            "relation": "<=",
            "passed": height <= float(GOAL_SIZE_LIMITS_MM["H"]),
        },
        "resonance_Hz": {
            "actual": resonance,
            "limit": float(GOAL_RESONANCE_MIN_HZ),
            "relation": ">=",
            "passed": resonance >= float(GOAL_RESONANCE_MIN_HZ),
        },
        "winding_max_C": {
            "actual": winding_max,
            "limit": float(TEMPERATURE_FAMILY_LIMITS_C["winding"]),
            "relation": "<=",
            "passed": winding_max
            <= float(TEMPERATURE_FAMILY_LIMITS_C["winding"]),
        },
        "core_max_C": {
            "actual": core_max,
            "limit": float(TEMPERATURE_FAMILY_LIMITS_C["core"]),
            "relation": "<=",
            "passed": core_max
            <= float(TEMPERATURE_FAMILY_LIMITS_C["core"]),
        },
    }
    for item in gates.values():
        if item["relation"] == "<=":
            item["margin"] = item["limit"] - item["actual"]
        else:
            item["margin"] = item["actual"] - item["limit"]
    losses = {
        name: _finite(result.get(name), name)
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    reasons = [
        f"{name} failed: actual={item['actual']}, "
        f"relation={item['relation']}, limit={item['limit']}"
        for name, item in gates.items()
        if not item["passed"]
    ]
    failed_targets = [
        target for target, item in temperatures.items() if not item["passed"]
    ]
    for target in failed_targets:
        item = temperatures[target]
        reasons.append(
            f"{target} failed: actual_C={item['actual_C']}, "
            f"limit_C={item['limit_C']}"
        )
    return {
        "goal_contract_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "volume_L": volume,
        "dimensions_mm": {
            "width": width,
            "length": length,
            "height": height,
        },
        "active_temperature_targets": active_targets,
        "temperature_targets": temperatures,
        "temperature_target_gate_passed": not failed_targets,
        "gates": gates,
        "losses_W": losses,
        "total_loss_W": sum(losses.values()),
        "reasons": reasons,
        "passed": not reasons,
    }


def attest_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate fixed physics and unchanged scientific truth gates."""

    try:
        fixed = attest_fixed_identity(
            result, require_thermal_pad_metadata=True
        )
    except Exception as exc:
        raise CollectionError("fixed operating/cooling identity failed") from exc
    direct = _validate_direct_analyze(result)
    core = _validate_core_policy(result)
    thermal = _validate_thermal_truth(result)
    goal = _goal_evidence(result)
    return seal(
        {
            "schema_version": SCIENTIFIC_GATE_SCHEMA,
            "created_at_utc": _now(),
            **CLASSIFICATION,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "fixed_identity_attestation": fixed,
            "direct_analyze_authentication": direct,
            "solver_core_authentication": core,
            "thermal_truth_authentication": thermal,
            "direct_analyze_authentication_passed": True,
            "solver_core_authentication_passed": True,
            "thermal_truth_authentication_passed": True,
            "goal_physical_spec": goal,
            "goal_physical_spec_passed": goal["passed"],
            "goal_physical_spec_reasons": goal["reasons"],
            "scientific_truth_gates_unchanged": True,
        }
    )


def _task_is_success(task: Mapping[str, Any]) -> bool:
    return (
        str(task.get("status") or "").lower(),
        str(task.get("state") or "").lower(),
    ) in TERMINAL_SUCCESS and task.get("exit_code") == 0


def _source_copy(
    contract: Mapping[str, Any], destination: Path
) -> dict[str, dict[str, Any]]:
    source_destination = destination / "official5_source"
    inventory = {}
    for relative, source in sorted(contract["source_paths"].items()):
        target = source_destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory[relative] = file_record(target, relative_to=destination)
    return inventory


def collect_success(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    output: Path,
    getter: Callable[..., bytes] = base.http_get,
) -> dict[str, Any]:
    """Atomically collect and authenticate task96338 terminal-success evidence."""

    destination = output.resolve()
    if destination.exists():
        view = authenticate_collection(destination)
        return {
            "event": "already_collected",
            "task_id": TASK_ID,
            "output": str(destination),
            "collection_payload_sha256": view["collection"]["payload_sha256"],
        }
    if not _task_is_success(task):
        raise CollectionError("task96338 is not exact terminal success")
    result, stdout = get_result(contract, getter=getter)
    stderr = _get_log(contract, "stderr", getter=getter)
    gates = attest_result(result)
    (
        remote_receipt,
        marker,
        manifest,
        remote_receipt_raw,
        marker_raw,
        manifest_raw,
    ) = base.fetch_remote_metadata(contract, result, getter=getter)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        artifact, chunk_inventory = base._fetch_aedt_and_chunks(  # noqa: SLF001
            contract=contract,
            receipt=remote_receipt,
            staging=staging,
            getter=getter,
        )
        (staging / "result.json").write_bytes(canonical_bytes(result) + b"\n")
        (staging / "scheduler_stdout.log").write_bytes(stdout)
        (staging / "scheduler_stderr.log").write_bytes(stderr)
        (staging / "remote_bundle_receipt.json").write_bytes(
            remote_receipt_raw
        )
        (staging / "prune_protection_marker.json").write_bytes(marker_raw)
        (staging / "symmetric.aedtresults.manifest.json").write_bytes(
            manifest_raw
        )
        (staging / "scheduler_terminal_task.json").write_bytes(
            canonical_bytes(task) + b"\n"
        )
        (staging / "scientific_gate_evidence.json").write_bytes(
            canonical_bytes(gates) + b"\n"
        )
        source_inventory = _source_copy(contract, staging)
        shutil.copy2(
            staging / "official5_source" / "direct_analyze_plan.json",
            staging / "source_plan.json",
        )
        shutil.copy2(
            staging
            / "official5_source"
            / "submission"
            / "submission_receipt.json",
            staging / "source_submission_receipt.json",
        )
        source_files = {
            name: file_record(staging / name, relative_to=staging)
            for name in (
                "source_plan.json",
                "source_submission_receipt.json",
                "scheduler_terminal_task.json",
                "scheduler_stdout.log",
                "scheduler_stderr.log",
                "remote_bundle_receipt.json",
                "prune_protection_marker.json",
            )
        }
        artifact_record = file_record(artifact, relative_to=staging)
        manifest_record = file_record(
            staging / "symmetric.aedtresults.manifest.json",
            relative_to=staging,
        )
        result_record = file_record(
            staging / "result.json", relative_to=staging
        )
        gate_record = file_record(
            staging / "scientific_gate_evidence.json",
            relative_to=staging,
        )
        receipt = seal(
            {
                "schema_version": COLLECTION_SCHEMA,
                "created_at_utc": _now(),
                **CLASSIFICATION,
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "dedupe_key": DEDUPE_KEY,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "same_node_as_task_id": SAME_NODE_TASK_ID,
                "same_node_as_allocation_id": ALLOCATION_ID,
                "slurm_job_id": SLURM_JOB_ID,
                "fixed_physics_unchanged": True,
                "source_plan_file_sha256": contract["plan_file_sha256"],
                "source_plan_payload_sha256": contract["plan"][
                    "payload_sha256"
                ],
                "source_submission_file_sha256": contract[
                    "submission_file_sha256"
                ],
                "source_submission_payload_sha256": contract["submission"][
                    "payload_sha256"
                ],
                "source_lineage": source_inventory,
                "source_lineage_sha256": payload_sha256(source_inventory),
                "scheduler_terminal_task_sha256": payload_sha256(task),
                "result_sha256": payload_sha256(result),
                "scientific_gate_evidence": gate_record,
                "scientific_gate_evidence_payload_sha256": gates[
                    "payload_sha256"
                ],
                "direct_analyze_authentication_passed": True,
                "solver_core_authentication_passed": True,
                "thermal_truth_authentication_passed": True,
                "goal_physical_spec_passed": gates[
                    "goal_physical_spec_passed"
                ],
                "goal_physical_spec_reasons": gates[
                    "goal_physical_spec_reasons"
                ],
                "retained_symmetric_aedt": artifact_record,
                "retained_symmetric_aedt_remote_sha256": remote_receipt[
                    "artifact_sha256"
                ],
                "retained_aedt_chunks": chunk_inventory,
                "retained_aedt_chunk_inventory_sha256": payload_sha256(
                    chunk_inventory
                ),
                "aedtresults_manifest": manifest_record,
                "aedtresults_manifest_payload_sha256": payload_sha256(
                    manifest
                ),
                "aedtresults_remote_tree_sha256": remote_receipt[
                    "results_tree_sha256"
                ],
                "aedtresults_remote_file_count": remote_receipt[
                    "results_file_count"
                ],
                "aedtresults_remote_size_bytes": remote_receipt[
                    "results_size_bytes"
                ],
                "aedtresults_remote_tree_authenticated": True,
                "aedtresults_local_materialized": False,
                "result_json": result_record,
                "remote_bundle_receipt_payload_sha256": payload_sha256(
                    remote_receipt
                ),
                "prune_protection_marker_payload_sha256": payload_sha256(
                    marker
                ),
                "source_files": source_files,
            }
        )
        receipt_path = staging / "collection_receipt.json"
        receipt_path.write_bytes(canonical_bytes(receipt) + b"\n")
        seal_value = seal(
            {
                "schema_version": COLLECTION_SEAL_SCHEMA,
                "created_at_utc": _now(),
                **CLASSIFICATION,
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "dedupe_key": DEDUPE_KEY,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "same_node_as_task_id": SAME_NODE_TASK_ID,
                "same_node_as_allocation_id": ALLOCATION_ID,
                "slurm_job_id": SLURM_JOB_ID,
                "source_submission_file_sha256": contract[
                    "submission_file_sha256"
                ],
                "collection_receipt": file_record(
                    receipt_path, relative_to=staging
                ),
                "collection_receipt_payload_sha256": receipt[
                    "payload_sha256"
                ],
                "artifact_sha256": remote_receipt["artifact_sha256"],
                "aedtresults_manifest_sha256": remote_receipt[
                    "results_manifest_sha256"
                ],
                "aedtresults_tree_sha256": remote_receipt[
                    "results_tree_sha256"
                ],
                "scientific_gate_evidence_payload_sha256": gates[
                    "payload_sha256"
                ],
                "direct_analyze_authentication_passed": True,
                "solver_core_authentication_passed": True,
                "thermal_truth_authentication_passed": True,
                "goal_physical_spec_passed": gates[
                    "goal_physical_spec_passed"
                ],
                "atomic_directory_collection": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
            }
        )
        (staging / "collection_seal.json").write_bytes(
            canonical_bytes(seal_value) + b"\n"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "event": "collected",
        "task_id": TASK_ID,
        "output": str(destination),
        "artifact_sha256": remote_receipt["artifact_sha256"],
        "aedtresults_tree_sha256": remote_receipt["results_tree_sha256"],
        "goal_physical_spec_passed": gates["goal_physical_spec_passed"],
        "collection_receipt_payload_sha256": receipt["payload_sha256"],
        "collection_seal_payload_sha256": seal_value["payload_sha256"],
    }


def _resolve_local_record(
    root: Path, record: Any, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise CollectionError(f"{label} record is absent")
    relative = base._safe_relative(record.get("path"), label)  # noqa: SLF001
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CollectionError(f"{label} escaped collection root") from exc
    if (
        not path.is_file()
        or record.get("sha256") != sha256_file(path)
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise CollectionError(f"{label} record bytes drifted")
    return path


def authenticate_collection(collection_path: Path) -> dict[str, Any]:
    """Return a downstream-compatible authenticated official5 collection view."""

    supplied = collection_path.resolve(strict=True)
    root = supplied if supplied.is_dir() else supplied.parent
    receipt_path = (
        root / "collection_receipt.json"
        if supplied.is_dir()
        else supplied
    )
    if receipt_path != root / "collection_receipt.json":
        raise CollectionError("collection path must be root or collection receipt")
    receipt = _read_json(receipt_path, "collection receipt")
    seal_value = _read_json(root / "collection_seal.json", "collection seal")
    _validate_seal(receipt, COLLECTION_SCHEMA, "collection receipt")
    _validate_seal(
        seal_value, COLLECTION_SEAL_SCHEMA, "collection seal"
    )
    for value, label in (
        (receipt, "collection receipt"),
        (seal_value, "collection seal"),
    ):
        if any(value.get(key) is not expected for key, expected in CLASSIFICATION.items()):
            raise CollectionError(f"{label} classification drifted")
        for key, expected in (
            ("task_id", TASK_ID),
            ("task_name", TASK_NAME),
            ("dedupe_key", DEDUPE_KEY),
            ("candidate_physics_sha256", CANDIDATE_PHYSICS_SHA256),
            ("same_node_as_task_id", SAME_NODE_TASK_ID),
            ("same_node_as_allocation_id", ALLOCATION_ID),
            ("slurm_job_id", SLURM_JOB_ID),
        ):
            if value.get(key) != expected:
                raise CollectionError(f"{label} identity drifted: {key}")
    receipt_record_path = _resolve_local_record(
        root, seal_value.get("collection_receipt"), "collection receipt"
    )
    if (
        receipt_record_path != receipt_path
        or seal_value.get("collection_receipt_payload_sha256")
        != receipt.get("payload_sha256")
    ):
        raise CollectionError("collection receipt/seal binding drifted")

    contract = load_contract(
        source_root=root / "official5_source",
        scheduler_url=SCHEDULER_URL,
    )
    if (
        receipt.get("source_plan_file_sha256")
        != contract["plan_file_sha256"]
        or receipt.get("source_plan_payload_sha256")
        != contract["plan"]["payload_sha256"]
        or receipt.get("source_submission_file_sha256")
        != contract["submission_file_sha256"]
        or receipt.get("source_submission_payload_sha256")
        != contract["submission"]["payload_sha256"]
        or receipt.get("source_lineage_sha256")
        != payload_sha256(receipt.get("source_lineage"))
    ):
        raise CollectionError("collection official5 lineage drifted")
    lineage = receipt.get("source_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(
        SOURCE_FILE_SHA256
    ):
        raise CollectionError("collection source lineage inventory drifted")
    for relative, record in lineage.items():
        path = _resolve_local_record(root, record, f"source lineage {relative}")
        if sha256_file(path) != SOURCE_FILE_SHA256[relative]:
            raise CollectionError(f"source lineage drifted: {relative}")

    terminal = _read_json(
        _resolve_local_record(
            root,
            receipt.get("source_files", {}).get(
                "scheduler_terminal_task.json"
            ),
            "terminal task",
        ),
        "terminal task",
    )
    get_task_contract = dict(contract)
    if (
        get_task_contract["task_id"] != TASK_ID
        or not _task_is_success(terminal)
        or receipt.get("scheduler_terminal_task_sha256")
        != payload_sha256(terminal)
    ):
        raise CollectionError("terminal Scheduler evidence drifted")
    # Validate the exact placement snapshot without issuing another network GET.
    def terminal_getter(
        _url: str, *, max_bytes: int, timeout: float
    ) -> bytes:
        del max_bytes, timeout
        return canonical_bytes(terminal)

    get_task(contract, getter=terminal_getter)

    result_path = _resolve_local_record(
        root, receipt.get("result_json"), "result JSON"
    )
    result = _read_json(result_path, "result JSON")
    if receipt.get("result_sha256") != payload_sha256(result):
        raise CollectionError("collected result payload drifted")
    gates = attest_result(result)
    stored_gate_path = _resolve_local_record(
        root,
        receipt.get("scientific_gate_evidence"),
        "scientific gate evidence",
    )
    stored_gates = _read_json(stored_gate_path, "scientific gate evidence")
    _validate_seal(
        stored_gates, SCIENTIFIC_GATE_SCHEMA, "scientific gate evidence"
    )
    stable_gate_fields = {
        key: value
        for key, value in gates.items()
        if key not in {"created_at_utc", "payload_sha256"}
    }
    stable_stored_fields = {
        key: value
        for key, value in stored_gates.items()
        if key not in {"created_at_utc", "payload_sha256"}
    }
    if (
        stable_gate_fields != stable_stored_fields
        or receipt.get("scientific_gate_evidence_payload_sha256")
        != stored_gates.get("payload_sha256")
        or seal_value.get("scientific_gate_evidence_payload_sha256")
        != stored_gates.get("payload_sha256")
    ):
        raise CollectionError("stored scientific gate evidence drifted")

    remote_receipt_path = _resolve_local_record(
        root,
        receipt.get("source_files", {}).get("remote_bundle_receipt.json"),
        "remote bundle receipt",
    )
    marker_path = _resolve_local_record(
        root,
        receipt.get("source_files", {}).get(
            "prune_protection_marker.json"
        ),
        "prune marker",
    )
    manifest_path = _resolve_local_record(
        root, receipt.get("aedtresults_manifest"), "results manifest"
    )
    remote_receipt = _read_json(
        remote_receipt_path, "remote bundle receipt"
    )
    marker = _read_json(marker_path, "prune marker")
    manifest = _read_json(manifest_path, "results manifest")
    authenticated_remote = base._validate_remote_receipt(  # noqa: SLF001
        remote_receipt, contract, result
    )
    base._validate_marker(  # noqa: SLF001
        marker, marker_path.read_bytes(), authenticated_remote, contract
    )
    base._validate_manifest(  # noqa: SLF001
        manifest, manifest_path.read_bytes(), authenticated_remote, result
    )
    artifact_path = _resolve_local_record(
        root, receipt.get("retained_symmetric_aedt"), "symmetric AEDT"
    )
    if sha256_file(artifact_path) != authenticated_remote["artifact_sha256"]:
        raise CollectionError("collected symmetric AEDT bytes drifted")
    chunks = receipt.get("retained_aedt_chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != authenticated_remote["transport_chunk_count"]
        or receipt.get("retained_aedt_chunk_inventory_sha256")
        != payload_sha256(chunks)
    ):
        raise CollectionError("collected AEDT chunk inventory drifted")
    for index, record in enumerate(chunks):
        _resolve_local_record(root, record, f"AEDT chunk {index}")

    goal = stored_gates["goal_physical_spec"]
    collection = seal(
        {
            "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
            **CLASSIFICATION,
            "stage": "standard",
            "scheduler_status": "completed",
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "dedupe_key": DEDUPE_KEY,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "same_node_as_task_id": SAME_NODE_TASK_ID,
            "same_node_as_allocation_id": ALLOCATION_ID,
            "slurm_job_id": SLURM_JOB_ID,
            "source_collection_receipt": file_record(receipt_path),
            "source_collection_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "source_collection_seal": file_record(
                root / "collection_seal.json"
            ),
            "source_collection_seal_payload_sha256": seal_value[
                "payload_sha256"
            ],
            "retained_symmetric_aedt": file_record(artifact_path),
            "result": result,
            "result_sha256": payload_sha256(result),
            "result_json": file_record(result_path),
            "goal_physical_spec_reasons": goal["reasons"],
            "goal_physical_spec_passed": goal["passed"],
            "fixed_identity_attestation": stored_gates[
                "fixed_identity_attestation"
            ],
            "scientific_gate_evidence": stored_gates,
            "direct_analyze_authentication_passed": True,
            "solver_core_authentication_passed": True,
            "thermal_truth_authentication_passed": True,
            "strict_al_adapter_authorized": True,
        }
    )
    return {
        "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
        "collection": collection,
        "plan": copy.deepcopy(contract["plan"]),
        "params": copy.deepcopy(contract["params"]),
        "selected": copy.deepcopy(contract["selected"]),
        "submission": copy.deepcopy(contract["submission"]),
    }


def _failure_task_get(
    *,
    getter: Callable[..., bytes] = base.http_get,
) -> dict[str, Any]:
    raw = getter(
        f"{SCHEDULER_URL}/api/tasks/{PRIOR_FAILURE_TASK_ID}",
        max_bytes=1024 * 1024,
        timeout=30.0,
    )
    task = _decode_json(raw, "task96337 Scheduler GET")
    expected = {
        "name": PRIOR_FAILURE_TASK_NAME,
        "dedupe_key": PRIOR_FAILURE_DEDUPE_KEY,
        "project": SCHEDULER_PROJECT,
        "account_name": ACCOUNT_NAME,
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "allocation_id": ALLOCATION_ID,
        "same_node_as_allocation_id": ALLOCATION_ID,
        "same_node_as_task_id": SAME_NODE_TASK_ID,
        "slurm_job_id": SLURM_JOB_ID,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "state": "failed",
        "status": "failed",
        "exit_code": 1,
        "placement_contract_satisfied": True,
    }
    if task.get("task_id", task.get("id")) != PRIOR_FAILURE_TASK_ID:
        raise CollectionError("task96337 identity drifted")
    for key, expected_value in expected.items():
        if task.get(key) != expected_value:
            raise CollectionError(f"task96337 identity drifted: {key}")
    if "authentication digest mismatch" not in str(
        task.get("failure_message") or ""
    ):
        raise CollectionError("task96337 failure reason drifted")
    return task


def ensure_prior_failure_ledger(
    path: Path = DEFAULT_PRIOR_FAILURE_LEDGER,
    *,
    getter: Callable[..., bytes] = base.http_get,
) -> dict[str, Any]:
    """Emit the exact task96337 pre-EM terminal failure ledger once."""

    destination = path.resolve()
    if destination.exists():
        value = _read_json(destination, "task96337 failure ledger")
        _validate_seal(
            value, PRIOR_FAILURE_SCHEMA, "task96337 failure ledger"
        )
        if (
            value.get("task_id") != PRIOR_FAILURE_TASK_ID
            or value.get("excluded_from_selection_and_nds") is not True
        ):
            raise CollectionError("existing task96337 ledger identity drifted")
        return {
            "event": "prior_failure_ledger_exists",
            "task_id": PRIOR_FAILURE_TASK_ID,
            "path": str(destination),
            "payload_sha256": value["payload_sha256"],
        }
    task = _failure_task_get(getter=getter)
    logs = {}
    for stream in ("stdout", "stderr"):
        query = parse.urlencode({"max_bytes": MAX_LOG_BYTES})
        raw = getter(
            f"{SCHEDULER_URL}/api/tasks/{PRIOR_FAILURE_TASK_ID}/"
            f"{stream}?{query}",
            max_bytes=MAX_LOG_BYTES,
            timeout=120.0,
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise CollectionError(
                f"task96337 {stream} is not UTF-8"
            ) from exc
        logs[stream] = {
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
            "utf8_text": text,
        }
    combined = logs["stdout"]["utf8_text"] + logs["stderr"]["utf8_text"]
    if "authentication digest mismatch" not in combined:
        raise CollectionError("task96337 log failure evidence is absent")
    ledger = seal(
        {
            "schema_version": PRIOR_FAILURE_SCHEMA,
            "created_at_utc": _now(),
            **CLASSIFICATION,
            "scientific_pass_claimed": False,
            "scientific_infeasible_claimed": False,
            "scheduler_failure_only": True,
            "failure_phase": "pre_em_aedt_startup",
            "collection_performed": False,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "excluded_from_selection_and_nds": True,
            "replacement_task_id": TASK_ID,
            "task_id": PRIOR_FAILURE_TASK_ID,
            "task_name": PRIOR_FAILURE_TASK_NAME,
            "dedupe_key": PRIOR_FAILURE_DEDUPE_KEY,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "same_node_as_task_id": SAME_NODE_TASK_ID,
            "same_node_as_allocation_id": ALLOCATION_ID,
            "slurm_job_id": SLURM_JOB_ID,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "scheduler_terminal_task": task,
            "scheduler_terminal_task_sha256": payload_sha256(task),
            "logs": logs,
            "failure_class": (
                "standalone_core_authentication_digest_mismatch_before_em"
            ),
        }
    )
    _write_json(destination, ledger)
    return {
        "event": "prior_failure_ledger",
        "task_id": PRIOR_FAILURE_TASK_ID,
        "path": str(destination),
        "payload_sha256": ledger["payload_sha256"],
    }


def write_failure_ledger(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    output: Path,
    getter: Callable[..., bytes] = base.http_get,
) -> dict[str, Any]:
    path = output.resolve().parent / f"{output.name}.failure_ledger.json"
    if path.exists():
        value = _read_json(path, "task96338 failure ledger")
        _validate_seal(value, FAILURE_SCHEMA, "task96338 failure ledger")
        return {
            "event": "failure_ledger_exists",
            "task_id": TASK_ID,
            "path": str(path),
            "payload_sha256": value["payload_sha256"],
        }
    logs = {}
    for stream in ("stdout", "stderr"):
        raw = _get_log(contract, stream, getter=getter)
        logs[stream] = {
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
            "utf8_text": raw.decode("utf-8", errors="replace"),
        }
    ledger = seal(
        {
            "schema_version": FAILURE_SCHEMA,
            "created_at_utc": _now(),
            **CLASSIFICATION,
            "scientific_pass_claimed": False,
            "scientific_infeasible_claimed": False,
            "scheduler_failure_only": True,
            "collection_performed": False,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "excluded_from_selection_and_nds": True,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "dedupe_key": DEDUPE_KEY,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "same_node_as_task_id": SAME_NODE_TASK_ID,
            "same_node_as_allocation_id": ALLOCATION_ID,
            "slurm_job_id": SLURM_JOB_ID,
            "scheduler_terminal_task": copy.deepcopy(dict(task)),
            "scheduler_terminal_task_sha256": payload_sha256(task),
            "logs": logs,
            "failure_class": "scheduler_terminal_failure_or_timeout",
        }
    )
    _write_json(path, ledger)
    return {
        "event": "failure_ledger",
        "task_id": TASK_ID,
        "path": str(path),
        "payload_sha256": ledger["payload_sha256"],
    }


def _record_poll(
    output: Path, contract: Mapping[str, Any], task: Mapping[str, Any], sequence: int
) -> Path:
    poll = seal(
        {
            "schema_version": POLL_SCHEMA,
            "observed_at_utc": _now(),
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "dedupe_key": DEDUPE_KEY,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "same_node_as_task_id": SAME_NODE_TASK_ID,
            "same_node_as_allocation_id": ALLOCATION_ID,
            "slurm_job_id": SLURM_JOB_ID,
            "state": task.get("state"),
            "status": task.get("status"),
            "exit_code": task.get("exit_code"),
            "actual_node_name": task.get("actual_node_name"),
            "source_submission_file_sha256": contract[
                "submission_file_sha256"
            ],
            "task_snapshot_sha256": payload_sha256(task),
        }
    )
    watch_root = output.resolve().parent / f".{output.name}.watch"
    polls = watch_root / "polls"
    polls.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    immutable = polls / f"{stamp}-{sequence:06d}.json"
    _write_json(immutable, poll)
    _write_json(watch_root / "latest_poll.json", poll, replace=True)
    return immutable


def poll_once(
    *,
    contract: Mapping[str, Any],
    output: Path,
    prior_failure_ledger: Path = DEFAULT_PRIOR_FAILURE_LEDGER,
    sequence: int = 0,
    getter: Callable[..., bytes] = base.http_get,
) -> tuple[bool, dict[str, Any]]:
    prior = ensure_prior_failure_ledger(
        prior_failure_ledger, getter=getter
    )
    task = get_task(contract, getter=getter)
    poll_path = _record_poll(output, contract, task, sequence)
    status = str(task.get("status") or "").lower()
    state = str(task.get("state") or "").lower()
    if _task_is_success(task):
        event = collect_success(
            contract=contract, task=task, output=output, getter=getter
        )
        event["poll_path"] = str(poll_path)
        event["prior_failure_ledger"] = prior["path"]
        return True, event
    if (
        state in TERMINAL_FAILURE_STATES
        or status in TERMINAL_FAILURE_STATES
        or task.get("exit_code") not in {None, 0}
    ):
        event = write_failure_ledger(
            contract=contract,
            task=task,
            output=output,
            getter=getter,
        )
        event["poll_path"] = str(poll_path)
        event["prior_failure_ledger"] = prior["path"]
        return True, event
    if state not in ACTIVE_STATES and status not in ACTIVE_STATES:
        raise CollectionError(
            f"unsupported Scheduler transition: status={status!r}, "
            f"state={state!r}"
        )
    return False, {
        "event": "active",
        "task_id": TASK_ID,
        "state": state,
        "status": status,
        "actual_node_name": task.get("actual_node_name"),
        "allocation_id": task.get("allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "same_node_as_task_id": task.get("same_node_as_task_id"),
        "poll_path": str(poll_path),
        "prior_failure_ledger": prior["path"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prior-failure-ledger",
        type=Path,
        default=DEFAULT_PRIOR_FAILURE_LEDGER,
    )
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 10 <= args.interval <= 300:
        raise CollectionError("watch interval must be between 10 and 300 seconds")
    if args.scheduler_url.rstrip("/") != SCHEDULER_URL:
        raise CollectionError("task96338 collector rejects Scheduler origin drift")
    contract = load_contract(
        source_root=args.source_root, scheduler_url=args.scheduler_url
    )
    sequence = 0
    while True:
        terminal, event = poll_once(
            contract=contract,
            output=args.output,
            prior_failure_ledger=args.prior_failure_ledger,
            sequence=sequence,
        )
        print(json.dumps(event, sort_keys=True), flush=True)
        if terminal:
            return 0 if event["event"] in {"collected", "already_collected"} else 2
        if args.once:
            return 0
        sequence += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
