#!/usr/bin/env python3
"""Plan, submit, retain, and collect one corrected thermal continuation.

The default submission mode is read-only.  A Scheduler task POST is reachable
only through ``submit --apply`` after a sealed-plan check, a complete paged
namespace scan, fresh project/license/node/storage gates, and a durable atomic
claim.  This module never mutates Scheduler project configuration.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import mft_campaign_atomic_claim as atomic_claim  # noqa: E402


PLAN_SCHEMA = "mft-corrected-thermal-submission-plan-v1"
EXECUTION_PLAN_SCHEMA = "mft-corrected-thermal-execution-plan-v1"
CHECKPOINT_SCHEMA = "mft-corrected-thermal-static-checkpoint-v3"
RETENTION_SCHEMA = "mft-corrected-thermal-retained-bundle-v1"
RETENTION_MARKER_SCHEMA = "mft-corrected-thermal-retention-marker-v1"
SOURCE_EVIDENCE_SCHEMA = "mft-corrected-source-em-evidence-v1"
COLLECTION_SCHEMA = "mft-corrected-thermal-mixed-provenance-collection-v1"
RUNTIME_QUOTA_AUTHORITY_SCHEMA = (
    "mft-corrected-thermal-runtime-quota-authority-v1"
)

CAMPAIGN_ID = "mft-goal-20260726"
PROJECT = "MFT_1MW_2026v1"
SCHEDULER_URL = "http://127.0.0.1:8002"
TASK_NAME = (
    "mft-goal-corrected-thermal-l96230-b7c30cb70b95-native-r3"
)
ACCOUNT = "r1jae262"
ACCOUNT_UID = 1455
TARGET_NODE = "n109"
FORBIDDEN_NODE = "n114"
SOURCE_LOGICAL_TASK_ID = 96230
SOURCE_EXECUTION_TASK_ID = 96304
SOURCE_TASK_NAME = (
    "mft-goal-diag-standard-timeout12h-r4-l96230-b7c30cb70b95"
)
SOURCE_TASK_DEDUPE = (
    "mft-al:mft-goal-diag-standard-timeout12h-r4-l96230-b7c30cb70b95:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:74cb13f5e316abc2"
)
SOURCE_SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
EXECUTOR_REQUIRED_ANCESTOR = "a927ef7ba7d4b5577e47a43377922dacd77b1993"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
SOURCE_PLAN_IDENTITY_SHA256 = (
    "08a6e426d261803660e933ad85985e17af37fea8e2b4a29a9ee2b5eaaf4483c3"
)
DEADLINE = datetime(2026, 7, 26, 18, 0, 0, tzinfo=timezone.utc).astimezone(
    timezone.utc
)
DEADLINE_TEXT = "2026-07-26T18:00:00+09:00"
DEADLINE_EPOCH = int(
    datetime.fromisoformat(DEADLINE_TEXT).astimezone(timezone.utc).timestamp()
)

CPUS = 8
MEMORY_MB = 294_912
GPUS = 0
MEMORY_MARGIN_MB = 65_536
NODE_TELEMETRY_MAX_AGE_SECONDS = 120.0
NODE_TELEMETRY_FUTURE_TOLERANCE_SECONDS = 5.0
RUNTIME_QUOTA_AUTHORITY_MAX_AGE_SECONDS = 1_800
RUNTIME_QUOTA_DRIFT_RESERVE_BYTES = 16 * 1024**3
RUNTIME_QUOTA_DRIFT_RESERVE_INODES = 4_096
CORE_POLICY_SCHEMA = "mft-corrected-thermal-core-policy-v1"
CORE_CONTRACT_VERSION = "mft-standalone-core-optin-v1"
CORE_CONTRACT_ENV = "MFT_STANDALONE_CORE_CONTRACT"
CORE_COUNT_ENV = "MFT_STANDALONE_CORE_COUNT"
CORE_AUTH_ENV = "MFT_STANDALONE_CORE_AUTH_SHA256"
MAX_TIMEOUT_SECONDS = 21_600
MIN_RUNTIME_SECONDS = 10_800
PACKAGE_RESERVE_SECONDS = 1_800
TIMEOUT_QUANTUM_SECONDS = 300
MINIMUM_SCRATCH_WORKING_SHADOW_BYTES = 256 * 1024**3
MAXIMUM_MINIMUM_BUNDLE_BYTES = 4 * 1024**3
MIN_STORAGE_HEADROOM_BYTES = 8 * 1024**3
MIN_STORAGE_HEADROOM_INODES = 4096
PHYSICAL_FREE_RESERVE_BYTES = 50 * 1024**3
MAX_RESPONSE_BYTES = 128 * 1024**2
INVENTORY_PAGE_SIZE = 10_000
MAX_INVENTORY_PAGES = 100
ACTIVE_STATUSES = {"attaching", "running"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
CHECKPOINT_ROOT = PurePosixPath(
    "/gpfs/home1/r1jae262/slurm_scheduler/mft_goal_20260726/"
    "immutable_sources_v1"
)
RETAINED_ROOT = (
    "/gpfs/home1/r1jae262/slurm_scheduler/mft_goal_20260726/"
    "corrected_thermal_minimum_native_r3"
)
RETRY_GENERATION = "corrected-thermal-native-r3"
INFRASTRUCTURE_RETRY_SOURCE = {
    "task_id": 96310,
    "task_name": (
        "mft-goal-corrected-thermal-l96230-b7c30cb70b95-core8-r2"
    ),
    "dedupe_key": (
        "mft-al:mft-goal-corrected-thermal-l96230-b7c30cb70b95-core8-r2:"
        "abf407d1bd6174674f4e8b2a231973bb40e05e98:"
        "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:b19bdee88bf422bf"
    ),
    "executor_revision": "abf407d1bd6174674f4e8b2a231973bb40e05e98",
    "plan_payload_sha256": (
        "e90d5bc1dfe25d5027401b4b1f4dda8987f21a6c6987fd506f9ca1ab797d8016"
    ),
    "plan_file_sha256": (
        "efe0202c763bed871d23c7909922e3ed0ad3e1c8b35f6ce88e460eccfce610d6"
    ),
    "submission_receipt_payload_sha256": (
        "ae3d05f1c9a79756df8660b3ce790c76e9460b91b83dd480f4f9f14834c3e6dd"
    ),
    "submission_receipt_file_sha256": (
        "d01c67b675ae81d1c80cfc2eba60d8f15c3e92d01d155aa70474d70b0a0e3f6b"
    ),
    "stdout_sha256": (
        "f8b8dc9b2f5fcac059bcca67a7104b233c3ec8837ba6d39deb2f9959f42ffd6b"
    ),
    "stderr_sha256": (
        "5980b82d7e2cbbefec05c8952d475d0f0c77a424f13b675672fd674c187f989a"
    ),
    "requested_node": "n109",
    "allocation_id": 14638,
    "slurm_job_id": "838099",
    "failure_class": "missing_native_fan_design_variable_before_solver",
    "retry_kind": "infrastructure",
}
REMOTE_CWD = "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs"
EXECUTOR_ENTRYPOINT = "tools/mft_goal_execute_corrected_thermal_checkpoint.py"
STAGE_ENTRYPOINT = "tools/mft_goal_corrected_thermal_continuation.py"
SUBMISSION_ENTRYPOINT = "tools/mft_goal_corrected_thermal_submission.py"
EXECUTOR_REPOSITORY = "https://github.com/Schwalbe262/MFT_1MW_2026.git"
EXECUTOR_REMOTE_REF = "refs/heads/integration/mft-goal-20260726"
LIBRARY_REPOSITORY = "https://github.com/Schwalbe262/pyaedt_library.git"
MUTATION_LOCK_NAME = "campaign-mutation.lock"

PROJECT_REPOS = [
    {
        "url": EXECUTOR_REPOSITORY,
        "ref": "main",
        "subdir": "MFT_1MW_2026",
    },
    {
        "url": LIBRARY_REPOSITORY,
        "ref": "main",
    },
]
PROJECT_SETUP = (
    "source /etc/profile.d/lmod.sh 2>/dev/null || true\n"
    "module load ansys-electronics/v252 2>/dev/null || "
    "export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64\n"
    "export FLEXLM_TIMEOUT=3000000"
)
PROJECT_ENTRYPOINTS = [
    {
        "path": "run_simulation_260706.py",
        "conda_env": "pyaedt2026v1",
        "workdir": "MFT_1MW_2026",
    },
    {
        "path": "run_campaign.py",
        "conda_env": "pyaedt2026v1",
        "workdir": "MFT_1MW_2026",
    },
]
PROJECT_CLEANUP_GLOBS = "*.aedtresults"
PROJECT_OUTPUT_GLOBS = (
    "simulation_results_*.csv,failed_samples_260706.jsonl,"
    "results_parts_260706/*.parquet"
)
OPTIONAL_FIELD_BUNDLE_REASON = (
    "deadline minimum-retention policy avoids duplicating checkpoint "
    "premesh and unbounded thermal field data"
)
RETENTION_CONTROL_EVIDENCE = {
    "retention_receipt.json": (
        "published in the same no-replace transaction and read-only/"
        "hash-verified, but intentionally outside the payload file inventory "
        "to avoid a self-hash cycle"
    ),
    ".slurm-scheduler-preserve.json": "read-only prune-protection marker",
}
RETENTION_EVIDENCE_SEMANTICS = (
    "read-only package-control evidence; transactionally published and "
    "post-publish hash verified"
)
PRUNE_MARKER_REASON = (
    "Retain corrected diagnostic thermal AEDT/evidence through deadline "
    "review; never promote automatically"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class CorrectedThermalError(RuntimeError):
    """A corrected-thermal fail-closed contract violation."""


class SubmissionUncertain(CorrectedThermalError):
    """A POST may have succeeded, but no unique durable row was observed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def core_contract_auth_sha256(solver_revision: Any) -> str:
    revision = _sha1(solver_revision, "core-policy solver revision")
    payload = {
        "backend": "standalone",
        "contract_version": CORE_CONTRACT_VERSION,
        "requested_num_cores": CPUS,
        "required_slurm_cpus_per_task": CPUS,
        "solver_revision": revision,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _core_policy(solver_revision: Any) -> dict[str, Any]:
    revision = _sha1(solver_revision, "core-policy solver revision")
    auth = core_contract_auth_sha256(revision)
    return {
        "schema": CORE_POLICY_SCHEMA,
        "backend": "standalone",
        "contract_version": CORE_CONTRACT_VERSION,
        "requested_num_cores": CPUS,
        "num_tasks": 1,
        "required_slurm_cpus_per_task": CPUS,
        "solver_revision": revision,
        "auth_sha256": auth,
        "environment": {
            CORE_CONTRACT_ENV: CORE_CONTRACT_VERSION,
            CORE_COUNT_ENV: str(CPUS),
            CORE_AUTH_ENV: auth,
        },
    }


def validate_core_policy(value: Mapping[str, Any], solver_revision: Any) -> dict:
    policy = _mapping(value, "solver core policy")
    expected = _core_policy(solver_revision)
    if policy != expected:
        raise CorrectedThermalError("solver core policy drifted")
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise CorrectedThermalError(f"{label} is not lowercase SHA-256")
    return text


def _sha1(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if _SHA1_RE.fullmatch(text) is None:
        raise CorrectedThermalError(f"{label} is not lowercase 40-hex")
    return text


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CorrectedThermalError(f"{label} is not a valid integer")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrectedThermalError(f"{label} is not an object")
    return copy.deepcopy(dict(value))


def sealed(value: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if digest_field in result:
        raise CorrectedThermalError(f"{digest_field} already exists")
    result[digest_field] = canonical_sha256(result)
    return result


def validate_seal(
    value: Mapping[str, Any],
    *,
    schema_field: str,
    schema: str,
    digest_field: str,
) -> dict[str, Any]:
    result = _mapping(value, schema)
    if result.get(schema_field) != schema:
        raise CorrectedThermalError(f"{schema} schema mismatch")
    digest = _sha256(result.get(digest_field), f"{schema} payload digest")
    unsigned = dict(result)
    unsigned.pop(digest_field, None)
    if canonical_sha256(unsigned) != digest:
        raise CorrectedThermalError(f"{schema} payload digest mismatch")
    return result


def _read_json(path: Path, label: str, *, max_bytes: int = 64 * 1024**2) -> dict:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CorrectedThermalError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise CorrectedThermalError(f"{label} is not a plain file: {path}")
    if metadata.st_size <= 0 or metadata.st_size > max_bytes:
        raise CorrectedThermalError(f"{label} size is outside its bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectedThermalError(f"{label} is not valid UTF-8 JSON") from exc
    return _mapping(value, label)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path = path.absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise CorrectedThermalError(f"existing immutable output differs: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _utc_now(value: datetime | None = None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise CorrectedThermalError("time must be timezone-aware")
    return result.astimezone(timezone.utc)


def _scheduler_timestamp(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise CorrectedThermalError(f"{label} is absent")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CorrectedThermalError(f"{label} is not ISO-8601") from exc
    # Scheduler SQLite timestamps are UTC even when the legacy API omits the
    # offset.  Attach UTC explicitly so all age arithmetic remains aware.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def calculate_timeout(now: datetime | None = None) -> int:
    remaining = math.floor(
        (
            datetime.fromisoformat(DEADLINE_TEXT).astimezone(timezone.utc)
            - _utc_now(now)
        ).total_seconds()
    )
    allowed = min(MAX_TIMEOUT_SECONDS, remaining - PACKAGE_RESERVE_SECONDS)
    rounded = (allowed // TIMEOUT_QUANTUM_SECONDS) * TIMEOUT_QUANTUM_SECONDS
    if rounded < MIN_RUNTIME_SECONDS:
        raise CorrectedThermalError(
            "deadline leaves less than the sealed minimum corrected-thermal runtime"
        )
    return rounded


def _git(
    repo: Path,
    *arguments: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def resolve_executor_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    head = _sha1(_git(repo, "rev-parse", "HEAD").stdout, "executor HEAD")
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        EXECUTOR_REQUIRED_ANCESTOR,
        head,
        check=False,
    )
    if ancestor.returncode != 0:
        raise CorrectedThermalError("executor HEAD does not contain the a927 fix")
    if _git(repo, "diff", "--quiet", check=False).returncode != 0:
        raise CorrectedThermalError("executor worktree has tracked unstaged changes")
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
        raise CorrectedThermalError("executor worktree has staged changes")

    entries: dict[str, dict[str, str]] = {}
    for relative in (
        STAGE_ENTRYPOINT,
        EXECUTOR_ENTRYPOINT,
        SUBMISSION_ENTRYPOINT,
    ):
        tracked = _git(
            repo, "ls-files", "--error-unmatch", "--", relative, check=False
        )
        if tracked.returncode != 0:
            raise CorrectedThermalError(
                f"executor entrypoint is not tracked at HEAD: {relative}"
            )
        raw = _git(repo, "show", f"{head}:{relative}", text=False).stdout
        blob_sha256 = hashlib.sha256(raw).hexdigest()
        path = repo / relative
        if not path.is_file() or sha256_file(path) != blob_sha256:
            raise CorrectedThermalError(
                f"executor entrypoint bytes differ from HEAD: {relative}"
            )
        entries[relative] = {
            "git_blob_sha1": _git(
                repo, "rev-parse", f"{head}:{relative}"
            ).stdout.strip(),
            "payload_sha256": blob_sha256,
        }
    remote = _git(
        repo,
        "ls-remote",
        "--heads",
        EXECUTOR_REPOSITORY,
        EXECUTOR_REMOTE_REF,
        check=False,
    )
    remote_rows = [
        line.split()
        for line in remote.stdout.splitlines()
        if line.strip()
    ]
    if (
        remote.returncode != 0
        or remote_rows != [[head, EXECUTOR_REMOTE_REF]]
    ):
        raise CorrectedThermalError(
            "executor HEAD is not the exact pushed integration ref"
        )
    return {
        "revision": head,
        "required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
        "tracked_worktree_clean": True,
        "remote_repository": EXECUTOR_REPOSITORY,
        "remote_ref": EXECUTOR_REMOTE_REF,
        "remote_revision_verified": True,
        "entrypoints": entries,
    }


def _safe_checkpoint_relative(value: Any) -> str:
    text = str(value or "")
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != text
    ):
        raise CorrectedThermalError(f"unsafe checkpoint path: {text!r}")
    return text


def authenticate_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path, "checkpoint manifest")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CorrectedThermalError("checkpoint schema mismatch")
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("composite_reauthentication_required") is not True
        or manifest.get("post_login_quota_reauthentication_required") is not True
    ):
        raise CorrectedThermalError("checkpoint diagnostic policy drifted")
    payload_digest = _sha256(
        manifest.get("manifest_payload_sha256"),
        "checkpoint manifest payload digest",
    )
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if canonical_sha256(unsigned) != payload_digest:
        raise CorrectedThermalError("checkpoint manifest payload digest mismatch")

    destination = PurePosixPath(str(manifest.get("destination") or ""))
    if (
        not destination.is_absolute()
        or destination == CHECKPOINT_ROOT
        or CHECKPOINT_ROOT not in destination.parents
        or ".." in destination.parts
    ):
        raise CorrectedThermalError("checkpoint destination escaped immutable root")

    source = _mapping(manifest.get("source_provenance"), "source provenance")
    exact_source = {
        "solver_revision": SOURCE_SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "candidate_sha256": CANDIDATE_SHA256,
        "logical_task_id": SOURCE_LOGICAL_TASK_ID,
        "execution_task_id": SOURCE_EXECUTION_TASK_ID,
        "source_plan_identity_sha256": SOURCE_PLAN_IDENTITY_SHA256,
    }
    for field, expected in exact_source.items():
        if source.get(field) != expected:
            raise CorrectedThermalError(
                f"checkpoint source provenance drifted: {field}"
            )
    for field in ("slurm_job_id", "allocation_id"):
        _positive_int(source.get(field), f"source {field}")
    if source.get("node") != FORBIDDEN_NODE:
        raise CorrectedThermalError("checkpoint source node identity drifted")
    source_project_sha = _sha256(
        source.get("source_project_sha256"), "source project digest"
    )
    source_metadata_sha = _sha256(
        source.get("source_static_metadata_sha256"),
        "source static metadata digest",
    )

    physics = _mapping(manifest.get("physics_boundary"), "fixed physics")
    exact_physics = {
        "fan_velocity_m_per_s": 1.5,
        "tim_conductivity_w_per_mk": 0.2,
        "thermal_pad_thickness_mm": 2.0,
    }
    for field, expected in exact_physics.items():
        try:
            observed = float(physics[field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CorrectedThermalError(f"invalid fixed physics {field}") from exc
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise CorrectedThermalError(f"fixed physics drifted: {field}")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 25:
        raise CorrectedThermalError("checkpoint inventory is not the 25-file allowlist")
    normalized_files: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(files):
        row = _mapping(raw, f"checkpoint files[{index}]")
        relative = _safe_checkpoint_relative(row.get("path"))
        if relative in names:
            raise CorrectedThermalError("checkpoint inventory has a duplicate path")
        names.add(relative)
        size = _positive_int(
            row.get("size"), f"checkpoint size {relative}", allow_zero=True
        )
        allocated = _positive_int(
            row.get("allocated"),
            f"checkpoint allocated bytes {relative}",
            allow_zero=True,
        )
        normalized_files.append(
            {
                "path": relative,
                "size": size,
                "allocated": allocated,
                "mtime_ns": _positive_int(
                    row.get("mtime_ns"), f"checkpoint mtime {relative}"
                ),
                "mode": str(row.get("mode") or ""),
                "sha256": _sha256(
                    row.get("sha256"), f"checkpoint content digest {relative}"
                ),
            }
        )
    aedt = [row for row in normalized_files if row["path"].endswith(".aedt")]
    grids = [
        row
        for row in normalized_files
        if row["path"].endswith("/grid_mapping")
        or row["path"].endswith("/grid_output")
    ]
    if (
        len(aedt) != 1
        or aedt[0]["sha256"] != source_project_sha
        or len(grids) != 20
    ):
        raise CorrectedThermalError("checkpoint AEDT/premesh inventory drifted")

    snapshots = []
    for name in ("source_snapshot_before", "source_snapshot_after"):
        snapshot = _mapping(manifest.get(name), name)
        normalized = {
            "metadata_sha256": _sha256(
                snapshot.get("metadata_sha256"), f"{name} metadata digest"
            ),
            "required_budget_bytes": _positive_int(
                snapshot.get("required_budget_bytes"), f"{name} byte budget"
            ),
            "required_inodes": _positive_int(
                snapshot.get("required_inodes"), f"{name} inode budget"
            ),
            "included_count": _positive_int(
                snapshot.get("included_count"), f"{name} file count"
            ),
            "directory_count": _positive_int(
                snapshot.get("directory_count"),
                f"{name} directory count",
                allow_zero=True,
            ),
            "logical_bytes": _positive_int(
                snapshot.get("logical_bytes"),
                f"{name} logical bytes",
                allow_zero=True,
            ),
            "allocated_bytes": _positive_int(
                snapshot.get("allocated_bytes"),
                f"{name} allocated bytes",
                allow_zero=True,
            ),
        }
        if (
            normalized["metadata_sha256"] != source_metadata_sha
            or normalized["included_count"] != len(normalized_files)
        ):
            raise CorrectedThermalError(f"{name} is not bound to source inventory")
        snapshots.append(normalized)
    if snapshots[0] != snapshots[1]:
        raise CorrectedThermalError("checkpoint source changed across static copy")

    quota_evidence = _mapping(
        manifest.get("quota_before_login_evidence"),
        "checkpoint login-node quota evidence",
    )
    quota_fields = {
        "filesystem",
        "quota_type",
        "uid",
        "usage_bytes",
        "soft_limit_bytes",
        "hard_limit_bytes",
        "in_doubt_bytes",
        "files_used",
        "files_soft_limit",
        "files_hard_limit",
        "files_in_doubt",
        "observed_at_epoch",
        "source",
        "canonical_sha256",
        "age_seconds_at_validation",
    }
    if set(quota_evidence) != quota_fields:
        raise CorrectedThermalError("checkpoint quota-evidence fields drifted")
    if (
        quota_evidence.get("filesystem") != "gpfs"
        or quota_evidence.get("quota_type") != "USR"
        or quota_evidence.get("source") != "gate2:mmlsquota-Y"
        or quota_evidence.get("uid") != ACCOUNT_UID
    ):
        raise CorrectedThermalError("checkpoint quota-evidence authority drifted")
    for field in (
        "uid",
        "usage_bytes",
        "soft_limit_bytes",
        "hard_limit_bytes",
        "in_doubt_bytes",
        "files_used",
        "files_soft_limit",
        "files_hard_limit",
        "files_in_doubt",
    ):
        _positive_int(
            quota_evidence.get(field),
            f"checkpoint quota evidence {field}",
            allow_zero=True,
        )
    try:
        observed_epoch = float(quota_evidence["observed_at_epoch"])
        quota_age = float(quota_evidence["age_seconds_at_validation"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedThermalError("checkpoint quota evidence time drifted") from exc
    if (
        not math.isfinite(observed_epoch)
        or observed_epoch < 0
        or not math.isfinite(quota_age)
        or quota_age < -5
        or quota_age > 120
    ):
        raise CorrectedThermalError("checkpoint quota evidence is stale")
    quota_claim = _sha256(
        quota_evidence.get("canonical_sha256"),
        "checkpoint quota evidence digest",
    )
    quota_unsigned = dict(quota_evidence)
    quota_unsigned.pop("canonical_sha256")
    quota_unsigned.pop("age_seconds_at_validation")
    if canonical_sha256(quota_unsigned) != quota_claim:
        raise CorrectedThermalError("checkpoint quota evidence digest drifted")

    filesystem: dict[str, dict[str, Any]] = {}
    filesystem_fields = {
        "anchor_device",
        "bavail",
        "block_size",
        "free_bytes",
        "fsid",
        "readonly",
        "required_free_bytes",
    }
    for field in ("filesystem_before", "filesystem_after_copy"):
        evidence = _mapping(manifest.get(field), field)
        if set(evidence) != filesystem_fields or evidence.get("readonly") is not False:
            raise CorrectedThermalError(f"{field} contract drifted")
        normalized = {"readonly": False}
        for name in filesystem_fields - {"readonly"}:
            normalized[name] = _positive_int(
                evidence.get(name), f"{field}.{name}", allow_zero=True
            )
        if (
            normalized["free_bytes"]
            != normalized["bavail"] * normalized["block_size"]
            or normalized["free_bytes"] < normalized["required_free_bytes"]
        ):
            raise CorrectedThermalError(f"{field} free-space accounting drifted")
        filesystem[field] = normalized
    before_filesystem = filesystem["filesystem_before"]
    after_filesystem = filesystem["filesystem_after_copy"]
    if (
        before_filesystem["anchor_device"]
        != after_filesystem["anchor_device"]
        or before_filesystem["fsid"] != after_filesystem["fsid"]
        or before_filesystem["required_free_bytes"]
        != snapshots[0]["required_budget_bytes"] + PHYSICAL_FREE_RESERVE_BYTES
        or after_filesystem["required_free_bytes"]
        != PHYSICAL_FREE_RESERVE_BYTES
    ):
        raise CorrectedThermalError("checkpoint filesystem identity/reserve drifted")

    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": sha256_file(path),
        "manifest_payload_sha256": payload_digest,
        "destination": destination.as_posix(),
        "source_provenance": source,
        "physics_boundary": exact_physics,
        "inventory_sha256": canonical_sha256(normalized_files),
        "file_count": len(normalized_files),
        "source_project_sha256": source_project_sha,
        "source_static_metadata_sha256": source_metadata_sha,
        "required_budget_bytes": snapshots[0]["required_budget_bytes"],
        "required_inodes": snapshots[0]["required_inodes"],
        "logical_bytes": snapshots[0]["logical_bytes"],
        "quota_before_login_evidence_sha256": quota_claim,
        "filesystem_evidence_sha256": canonical_sha256(filesystem),
    }


def _campaign_authority_sha256() -> str:
    return canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "candidate_sha256": CANDIDATE_SHA256,
            "deadline": DEADLINE_TEXT,
            "executor_required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
            "fixed_boundary": {
                "fan_velocity_m_per_s": 1.5,
                "tim_conductivity_w_per_mk": 0.2,
                "thermal_pad_thickness_mm": 2.0,
            },
            "library_revision": LIBRARY_REVISION,
            "logical_task_id": SOURCE_LOGICAL_TASK_ID,
            "source_execution_task_id": SOURCE_EXECUTION_TASK_ID,
            "source_solver_revision": SOURCE_SOLVER_REVISION,
        }
    )


def _retention_contract(checkpoint: Mapping[str, Any], identity: str) -> dict:
    return {
        "mode": "node_local_scratch_with_gpfs_minimum_retention",
        "scratch_root": f"/enroot/mft-corrected-{identity}/output",
        "minimum_scratch_working_shadow_bytes": (
            MINIMUM_SCRATCH_WORKING_SHADOW_BYTES
        ),
        "retained_filesystem": "gpfs",
        "retained_root": RETAINED_ROOT,
        "maximum_minimum_bundle_bytes": MAXIMUM_MINIMUM_BUNDLE_BYTES,
        "minimum_retained_headroom_after_bytes": MIN_STORAGE_HEADROOM_BYTES,
        "minimum_retained_inode_headroom_after": (
            MIN_STORAGE_HEADROOM_INODES
        ),
    }


def _execution_contract(
    *,
    checkpoint: Mapping[str, Any],
    executor: Mapping[str, Any],
    contract_digest: str,
    task_name: str,
    dedupe_key: str,
    retention: Mapping[str, Any],
    runtime_quota_authority: Mapping[str, Any],
    core_policy: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "checkpoint_manifest_sha256": checkpoint["file_sha256"],
        "checkpoint_manifest_payload_sha256": checkpoint[
            "manifest_payload_sha256"
        ],
        "executor_revision": executor["revision"],
        "required_executor_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
        "tool_payload_sha256": executor["entrypoints"][EXECUTOR_ENTRYPOINT][
            "payload_sha256"
        ],
        "submission_tool_payload_sha256": executor["entrypoints"][
            SUBMISSION_ENTRYPOINT
        ]["payload_sha256"],
        "dispatch": {
            "cores": CPUS,
            "tasks": 1,
            "use_auto_settings": False,
        },
        "core_policy": copy.deepcopy(dict(core_policy)),
        "runtime_quota_authority": copy.deepcopy(
            dict(runtime_quota_authority)
        ),
        "output_storage": copy.deepcopy(dict(retention)),
        "submission_contract_sha256": contract_digest,
        "task_name": task_name,
        "dedupe_key": dedupe_key,
    }
    return sealed(value, digest_field="plan_payload_sha256")


def validate_execution_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = validate_seal(
        value,
        schema_field="schema",
        schema=EXECUTION_PLAN_SCHEMA,
        digest_field="plan_payload_sha256",
    )
    if (
        plan.get("diagnostic_only") is not True
        or plan.get("canonical") is not False
        or plan.get("production_truth_eligible") is not False
        or plan.get("required_executor_ancestor")
        != EXECUTOR_REQUIRED_ANCESTOR
        or plan.get("dispatch")
        != {"cores": CPUS, "tasks": 1, "use_auto_settings": False}
    ):
        raise CorrectedThermalError("execution contract policy drifted")
    _sha1(plan.get("executor_revision"), "execution contract executor")
    _sha256(plan.get("tool_payload_sha256"), "execution tool payload")
    _sha256(
        plan.get("submission_tool_payload_sha256"), "submission tool payload"
    )
    _sha256(
        plan.get("checkpoint_manifest_sha256"), "execution checkpoint file"
    )
    _sha256(
        plan.get("submission_contract_sha256"), "execution submission contract"
    )
    validate_core_policy(
        _mapping(plan.get("core_policy"), "execution core policy"),
        plan["executor_revision"],
    )
    retention = _mapping(plan.get("output_storage"), "output storage")
    if (
        retention.get("mode")
        != "node_local_scratch_with_gpfs_minimum_retention"
        or not str(retention.get("scratch_root") or "").startswith(
            "/enroot/mft-corrected-"
        )
        or int(retention.get("minimum_scratch_working_shadow_bytes") or 0)
        < MINIMUM_SCRATCH_WORKING_SHADOW_BYTES
        or retention.get("retained_filesystem") != "gpfs"
        or retention.get("retained_root") != RETAINED_ROOT
        or int(retention.get("maximum_minimum_bundle_bytes") or 0)
        < MAXIMUM_MINIMUM_BUNDLE_BYTES
        or int(retention.get("minimum_retained_headroom_after_bytes") or 0)
        < MIN_STORAGE_HEADROOM_BYTES
        or int(retention.get("minimum_retained_inode_headroom_after") or 0)
        < MIN_STORAGE_HEADROOM_INODES
    ):
        raise CorrectedThermalError("execution output-storage policy drifted")
    return plan


def _build_command(
    *,
    checkpoint: Mapping[str, Any],
    executor: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    retention: Mapping[str, Any],
) -> str:
    encoded_plan = base64.b64encode(
        canonical_json_bytes(execution_contract)
    ).decode("ascii")
    revision = executor["revision"]
    executor_hash = executor["entrypoints"][EXECUTOR_ENTRYPOINT][
        "payload_sha256"
    ]
    submission_hash = executor["entrypoints"][SUBMISSION_ENTRYPOINT][
        "payload_sha256"
    ]
    core_policy = validate_core_policy(
        _mapping(execution_contract.get("core_policy"), "execution core policy"),
        revision,
    )
    output_root = str(retention["scratch_root"])
    scratch = str(PurePosixPath(output_root).parent)
    minimum_scratch_kib = math.ceil(
        (
            int(checkpoint["logical_bytes"])
            + int(retention["minimum_scratch_working_shadow_bytes"])
        )
        / 1024
    )
    lines = [
        "set -euo pipefail",
        f"scratch={_shell_quote(scratch)}",
        'case "$scratch" in /enroot/mft-corrected-*) ;; *) exit 90 ;; esac',
        'test ! -e "$scratch"',
        'mkdir -m 700 -- "$scratch"',
        'cleanup(){ rm -rf -- "$scratch"; }',
        "trap cleanup EXIT HUP INT TERM",
        'host="$(hostname -s)"',
        f'test "$host" = {_shell_quote(TARGET_NODE)}',
        f'test "$host" != {_shell_quote(FORBIDDEN_NODE)}',
        'free_kib="$(df -Pk /enroot | awk \'NR==2 {print $4}\')"',
        f'test "${{free_kib:-0}}" -ge {minimum_scratch_kib}',
        'mkdir -m 700 -- "$scratch/executor" "$scratch/library"',
        'git -C "$scratch/executor" init -q',
        (
            'git -C "$scratch/executor" remote add origin '
            f"{_shell_quote(EXECUTOR_REPOSITORY)}"
        ),
        (
            'git -C "$scratch/executor" fetch -q --depth=128 origin '
            f"{_shell_quote(EXECUTOR_REMOTE_REF)}"
        ),
        (
            'git -C "$scratch/executor" checkout -q --detach '
            f"{_shell_quote(revision)}"
        ),
        (
            'test "$(git -C "$scratch/executor" rev-parse HEAD)" = '
            f"{_shell_quote(revision)}"
        ),
        (
            'git -C "$scratch/executor" merge-base --is-ancestor '
            f"{_shell_quote(EXECUTOR_REQUIRED_ANCESTOR)} "
            f"{_shell_quote(revision)}"
        ),
        (
            'test -z "$(git -C "$scratch/executor" status '
            '--porcelain --untracked-files=no)"'
        ),
        (
            f"printf '%s  %s\\n' {_shell_quote(executor_hash)} "
            f"{_shell_quote(EXECUTOR_ENTRYPOINT)} | "
            '(cd "$scratch/executor" && sha256sum -c -)'
        ),
        (
            f"printf '%s  %s\\n' {_shell_quote(submission_hash)} "
            f"{_shell_quote(SUBMISSION_ENTRYPOINT)} | "
            '(cd "$scratch/executor" && sha256sum -c -)'
        ),
        'git -C "$scratch/library" init -q',
        (
            'git -C "$scratch/library" remote add origin '
            f"{_shell_quote(LIBRARY_REPOSITORY)}"
        ),
        (
            'git -C "$scratch/library" fetch -q --depth=1 origin '
            f"{_shell_quote(LIBRARY_REVISION)}"
        ),
        (
            'git -C "$scratch/library" checkout -q --detach '
            f"{_shell_quote(LIBRARY_REVISION)}"
        ),
        (
            'test "$(git -C "$scratch/library" rev-parse HEAD)" = '
            f"{_shell_quote(LIBRARY_REVISION)}"
        ),
        (
            f"printf '%s' {_shell_quote(encoded_plan)} | base64 -d "
            '> "$scratch/execution-plan.json"'
        ),
        (
            'test "$(sha256sum "$scratch/execution-plan.json" | '
            'awk \'{print $1}\')" = '
            f"{_shell_quote(hashlib.sha256(canonical_json_bytes(execution_contract)).hexdigest())}"
        ),
        f"deadline_epoch={DEADLINE_EPOCH}",
        f"reserve_seconds={PACKAGE_RESERVE_SECONDS}",
        f"minimum_runtime={MIN_RUNTIME_SECONDS}",
        'remaining="$((deadline_epoch - $(date +%s) - reserve_seconds))"',
        'test "$remaining" -ge "$minimum_runtime"',
        f'if [ "$remaining" -gt {MAX_TIMEOUT_SECONDS} ]; then '
        f"remaining={MAX_TIMEOUT_SECONDS}; fi",
        f'test "$scratch/output" = {_shell_quote(output_root)}',
        'mkdir -m 700 -- "$scratch/output"',
        (
            f"export {CORE_CONTRACT_ENV}="
            f"{_shell_quote(core_policy['environment'][CORE_CONTRACT_ENV])}"
        ),
        (
            f"export {CORE_COUNT_ENV}="
            f"{_shell_quote(core_policy['environment'][CORE_COUNT_ENV])}"
        ),
        (
            f"export {CORE_AUTH_ENV}="
            f"{_shell_quote(core_policy['environment'][CORE_AUTH_ENV])}"
        ),
        "set +e",
        (
            'timeout --signal=TERM --kill-after=300s "${remaining}s" '
            'python "$scratch/executor/'
            f'{EXECUTOR_ENTRYPOINT}" execute '
            f"--checkpoint {_shell_quote(checkpoint['destination'])} "
            '--output-root "$scratch/output" '
            '--library-root "$scratch/library" '
            '--execution-plan "$scratch/execution-plan.json"'
        ),
        "solve_rc=$?",
        "set -e",
        'if [ "$solve_rc" -ne 0 ]; then exit "$solve_rc"; fi',
        (
            'python "$scratch/executor/'
            f'{SUBMISSION_ENTRYPOINT}" retain '
            '--execution-plan "$scratch/execution-plan.json" '
            '--execution-root "$scratch/output" '
            '--execution-exit-code "$solve_rc"'
        ),
        "exit 0",
    ]
    return "\n".join(lines)


def _shell_quote(value: Any) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def build_plan(
    *,
    checkpoint_manifest: Path,
    claim_root: Path,
    runtime_quota_snapshot: Mapping[str, Any],
    executor_identity: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_now = _utc_now(now)
    checkpoint = authenticate_checkpoint_manifest(checkpoint_manifest)
    executor = (
        resolve_executor_identity(repo_root)
        if executor_identity is None
        else copy.deepcopy(dict(executor_identity))
    )
    revision = _sha1(executor.get("revision"), "executor revision")
    if executor.get("required_ancestor") != EXECUTOR_REQUIRED_ANCESTOR:
        raise CorrectedThermalError("executor required ancestor drifted")
    if (
        executor.get("tracked_worktree_clean") is not True
        or executor.get("remote_repository") != EXECUTOR_REPOSITORY
        or executor.get("remote_ref") != EXECUTOR_REMOTE_REF
        or executor.get("remote_revision_verified") is not True
    ):
        raise CorrectedThermalError("executor clean/pushed authority is absent")
    entries = _mapping(executor.get("entrypoints"), "executor entrypoints")
    for relative in (
        STAGE_ENTRYPOINT,
        EXECUTOR_ENTRYPOINT,
        SUBMISSION_ENTRYPOINT,
    ):
        row = _mapping(entries.get(relative), f"executor entrypoint {relative}")
        _sha256(row.get("payload_sha256"), f"{relative} payload")

    provisional_identity = canonical_sha256(
        {
            "checkpoint_manifest_sha256": checkpoint["file_sha256"],
            "candidate_sha256": CANDIDATE_SHA256,
            "executor_revision": revision,
            "library_revision": LIBRARY_REVISION,
        }
    )[:16]
    retention = _retention_contract(checkpoint, provisional_identity)
    runtime_quota_authority = build_runtime_quota_authority(
        runtime_quota_snapshot,
        retention=retention,
        now=observed_now,
    )
    core_policy = _core_policy(revision)
    contract = {
        "checkpoint": {
            key: checkpoint[key]
            for key in (
                "file_sha256",
                "manifest_payload_sha256",
                "destination",
                "inventory_sha256",
                "file_count",
                "source_project_sha256",
                "source_static_metadata_sha256",
                "required_budget_bytes",
                "required_inodes",
                "logical_bytes",
                "quota_before_login_evidence_sha256",
                "filesystem_evidence_sha256",
            )
        },
        "source_provenance": checkpoint["source_provenance"],
        "fixed_physics": checkpoint["physics_boundary"],
        "executor": {
            "revision": revision,
            "required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
            "entrypoints": entries,
        },
        "library_revision": LIBRARY_REVISION,
        "dispatch": {"cpus": CPUS, "tasks": 1, "use_auto_settings": False},
        "core_policy": core_policy,
        "retention": retention,
        "runtime_quota_authority": runtime_quota_authority,
        "retry_of_infrastructure": copy.deepcopy(
            INFRASTRUCTURE_RETRY_SOURCE
        ),
    }
    contract_digest = canonical_sha256(contract)
    dedupe = (
        f"mft-al:{TASK_NAME}:{revision}:{LIBRARY_REVISION}:"
        f"{contract_digest[:16]}"
    )
    execution = _execution_contract(
        checkpoint=checkpoint,
        executor=executor,
        contract_digest=contract_digest,
        task_name=TASK_NAME,
        dedupe_key=dedupe,
        retention=retention,
        runtime_quota_authority=runtime_quota_authority,
        core_policy=core_policy,
    )
    command = _build_command(
        checkpoint=checkpoint,
        executor=executor,
        execution_contract=execution,
        retention=retention,
    )
    timeout = calculate_timeout(observed_now)
    profile = {
        "project": PROJECT,
        "remote_cwd": REMOTE_CWD,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": GPUS,
        "account_name": ACCOUNT,
        "node_name": TARGET_NODE,
        "node_name_policy": "strict",
        "max_workers_per_node": 1,
        "same_node_as_task_id": 0,
        "priority": 100,
        "cleanup_globs": "mft-corrected-b7c30cb70b95-*",
        "timeout_seconds": timeout,
    }
    claim_root_absolute = claim_root.absolute()
    authority_sha = _campaign_authority_sha256()
    root_id = canonical_sha256(
        {
            "campaign": CAMPAIGN_ID,
            "authority": authority_sha,
            "root": str(claim_root_absolute),
        }
    )[:32]
    value = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": observed_now.isoformat(),
        "deadline": {
            "deadline_kst": DEADLINE_TEXT,
            "package_reserve_seconds": PACKAGE_RESERVE_SECONDS,
            "minimum_runtime_seconds": MIN_RUNTIME_SECONDS,
            "maximum_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "timeout_quantum_seconds": TIMEOUT_QUANTUM_SECONDS,
        },
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "live_submission_default": False,
        "scheduler": {
            "url": SCHEDULER_URL,
            "project": PROJECT,
            "get_only_without_apply": True,
            "post_endpoint": "/api/tasks",
            "maximum_post_calls": 1,
            "project_mutation_allowed": False,
        },
        "checkpoint_manifest": checkpoint,
        "contract": contract,
        "contract_digest_sha256": contract_digest,
        "executor": executor,
        "execution_contract": execution,
        "retention": retention,
        "core_policy": core_policy,
        "runtime_quota_authority": runtime_quota_authority,
        "retry_of_infrastructure": copy.deepcopy(
            INFRASTRUCTURE_RETRY_SOURCE
        ),
        "task_identity": {
            "name": TASK_NAME,
            "dedupe_key": dedupe,
            "candidate_sha256": CANDIDATE_SHA256,
            "logical_authority_task_id": SOURCE_LOGICAL_TASK_ID,
            "retry_generation": RETRY_GENERATION,
        },
        "submission_profile": profile,
        "submission_profile_sha256": canonical_sha256(profile),
        "canonical_command": command,
        "canonical_command_sha256": hashlib.sha256(
            command.encode("utf-8")
        ).hexdigest(),
        "claim": {
            "root": str(claim_root_absolute),
            "root_id": root_id,
            "campaign_authority_sha256": authority_sha,
        },
    }
    return sealed(value, digest_field="plan_payload_sha256")


def write_plan(
    *,
    checkpoint_manifest: Path,
    claim_root: Path,
    runtime_quota_snapshot: Mapping[str, Any],
    output: Path,
    repo_root: Path = REPO_ROOT,
    now: datetime | None = None,
) -> Path:
    plan = build_plan(
        checkpoint_manifest=checkpoint_manifest,
        claim_root=claim_root,
        runtime_quota_snapshot=runtime_quota_snapshot,
        repo_root=repo_root,
        now=now,
    )
    return _atomic_write_json(output, plan)


def load_plan(
    path: Path,
    *,
    verify_local_files: bool = True,
    executor_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = validate_seal(
        _read_json(path, "corrected thermal submission plan"),
        schema_field="schema_version",
        schema=PLAN_SCHEMA,
        digest_field="plan_payload_sha256",
    )
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("diagnostic_only") is not True
        or plan.get("canonical") is not False
        or plan.get("production_truth_eligible") is not False
        or plan.get("live_submission_default") is not False
    ):
        raise CorrectedThermalError("submission plan policy drifted")
    scheduler = _mapping(plan.get("scheduler"), "scheduler contract")
    if scheduler != {
        "url": SCHEDULER_URL,
        "project": PROJECT,
        "get_only_without_apply": True,
        "post_endpoint": "/api/tasks",
        "maximum_post_calls": 1,
        "project_mutation_allowed": False,
    }:
        raise CorrectedThermalError("scheduler authority drifted")
    contract = _mapping(plan.get("contract"), "submission contract")
    if plan.get("contract_digest_sha256") != canonical_sha256(contract):
        raise CorrectedThermalError("submission contract digest drifted")
    if (
        plan.get("retry_of_infrastructure")
        != INFRASTRUCTURE_RETRY_SOURCE
        or contract.get("retry_of_infrastructure")
        != INFRASTRUCTURE_RETRY_SOURCE
    ):
        raise CorrectedThermalError(
            "infrastructure retry ancestry drifted"
        )
    runtime_quota_authority = validate_runtime_quota_authority(
        _mapping(
            plan.get("runtime_quota_authority"),
            "runtime quota authority",
        ),
        retention=_mapping(plan.get("retention"), "retention contract"),
    )
    if contract.get("runtime_quota_authority") != runtime_quota_authority:
        raise CorrectedThermalError(
            "runtime quota authority contract binding drifted"
        )
    executor = _mapping(plan.get("executor"), "plan executor")
    revision = _sha1(executor.get("revision"), "plan executor revision")
    core_policy = validate_core_policy(
        _mapping(plan.get("core_policy"), "plan core policy"),
        revision,
    )
    if contract.get("core_policy") != core_policy:
        raise CorrectedThermalError("core policy contract binding drifted")
    if executor.get("required_ancestor") != EXECUTOR_REQUIRED_ANCESTOR:
        raise CorrectedThermalError("plan executor ancestor drifted")
    if (
        executor.get("tracked_worktree_clean") is not True
        or executor.get("remote_repository") != EXECUTOR_REPOSITORY
        or executor.get("remote_ref") != EXECUTOR_REMOTE_REF
        or executor.get("remote_revision_verified") is not True
    ):
        raise CorrectedThermalError("plan executor is not the pushed clean ref")
    task = _mapping(plan.get("task_identity"), "task identity")
    expected_dedupe = (
        f"mft-al:{TASK_NAME}:{revision}:{LIBRARY_REVISION}:"
        f"{plan['contract_digest_sha256'][:16]}"
    )
    if (
        task.get("name") != TASK_NAME
        or task.get("dedupe_key") != expected_dedupe
        or task.get("candidate_sha256") != CANDIDATE_SHA256
        or task.get("logical_authority_task_id")
        != SOURCE_LOGICAL_TASK_ID
        or task.get("retry_generation") != RETRY_GENERATION
    ):
        raise CorrectedThermalError("task identity drifted")
    profile = _mapping(plan.get("submission_profile"), "submission profile")
    if plan.get("submission_profile_sha256") != canonical_sha256(profile):
        raise CorrectedThermalError("submission profile digest drifted")
    expected_profile = {
        "project": PROJECT,
        "remote_cwd": REMOTE_CWD,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": GPUS,
        "account_name": ACCOUNT,
        "node_name": TARGET_NODE,
        "node_name_policy": "strict",
        "max_workers_per_node": 1,
        "same_node_as_task_id": 0,
        "priority": 100,
        "cleanup_globs": "mft-corrected-b7c30cb70b95-*",
    }
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise CorrectedThermalError("submission resource/placement profile drifted")
    timeout = _positive_int(profile.get("timeout_seconds"), "plan timeout")
    if (
        timeout < MIN_RUNTIME_SECONDS
        or timeout > MAX_TIMEOUT_SECONDS
        or timeout % TIMEOUT_QUANTUM_SECONDS
    ):
        raise CorrectedThermalError("plan timeout policy drifted")
    command = str(plan.get("canonical_command") or "")
    if (
        not command
        or hashlib.sha256(command.encode("utf-8")).hexdigest()
        != plan.get("canonical_command_sha256")
        or f'test "$host" = {_shell_quote(TARGET_NODE)}' not in command
        or f'test "$host" != {_shell_quote(FORBIDDEN_NODE)}' not in command
        or "--execution-plan" not in command
        or "timeout --signal=TERM" not in command
        or (
            f"export {CORE_CONTRACT_ENV}="
            f"{_shell_quote(CORE_CONTRACT_VERSION)}"
        )
        not in command
        or f"export {CORE_COUNT_ENV}={_shell_quote(str(CPUS))}" not in command
        or (
            f"export {CORE_AUTH_ENV}="
            f"{_shell_quote(core_policy['auth_sha256'])}"
        )
        not in command
    ):
        raise CorrectedThermalError("canonical command drifted")
    execution = validate_execution_contract(plan.get("execution_contract") or {})
    if (
        execution.get("executor_revision") != revision
        or execution.get("submission_contract_sha256")
        != plan.get("contract_digest_sha256")
        or execution.get("task_name") != TASK_NAME
        or execution.get("dedupe_key") != expected_dedupe
        or execution.get("output_storage") != plan.get("retention")
        or execution.get("core_policy") != core_policy
        or execution.get("runtime_quota_authority")
        != runtime_quota_authority
    ):
        raise CorrectedThermalError("execution/submission contract binding drifted")
    claim = _mapping(plan.get("claim"), "claim contract")
    if (
        claim.get("campaign_authority_sha256")
        != _campaign_authority_sha256()
        or not Path(str(claim.get("root") or "")).is_absolute()
        or not re.fullmatch(r"[0-9a-f]{32}", str(claim.get("root_id") or ""))
    ):
        raise CorrectedThermalError("atomic claim contract drifted")

    checkpoint = _mapping(
        plan.get("checkpoint_manifest"), "checkpoint manifest record"
    )
    if verify_local_files:
        refreshed_checkpoint = authenticate_checkpoint_manifest(
            Path(checkpoint["path"])
        )
        if refreshed_checkpoint != checkpoint:
            raise CorrectedThermalError("checkpoint manifest bytes drifted")
        actual_executor = (
            resolve_executor_identity(REPO_ROOT)
            if executor_identity is None
            else copy.deepcopy(dict(executor_identity))
        )
        if actual_executor != executor:
            raise CorrectedThermalError("fresh executor HEAD differs from plan")
    return plan


class SchedulerHTTP:
    """Small bounded Scheduler client; POST is intentionally a separate method."""

    def __init__(self, base_url: str = SCHEDULER_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.post_calls = 0

    def _url(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def get_json(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        request = urllib.request.Request(
            self._url(path, params),
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except Exception as exc:
            raise CorrectedThermalError(f"Scheduler GET failed: {path}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CorrectedThermalError("Scheduler JSON response exceeds bound")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CorrectedThermalError("Scheduler returned invalid JSON") from exc

    def get_text(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> str:
        request = urllib.request.Request(
            self._url(path, params),
            headers={"Accept": "text/plain"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except Exception as exc:
            raise CorrectedThermalError(f"Scheduler GET failed: {path}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CorrectedThermalError("Scheduler text response exceeds bound")
        try:
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise CorrectedThermalError("Scheduler stdout is not UTF-8") from exc

    def post_json(self, path: str, value: Mapping[str, Any]) -> Any:
        self.post_calls += 1
        if self.post_calls != 1:
            raise CorrectedThermalError("more than one Scheduler POST attempted")
        request = urllib.request.Request(
            self._url(path),
            data=canonical_json_bytes(value),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except Exception as exc:
            raise SubmissionUncertain("Scheduler POST outcome is uncertain") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise SubmissionUncertain("Scheduler POST response exceeds bound")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SubmissionUncertain("Scheduler POST returned invalid JSON") from exc


def _page_rows(payload: Mapping[str, Any], *, page: int, before_id: int) -> list:
    value = _mapping(payload, "Scheduler task page")
    rows = value.get("items")
    filters = value.get("filters")
    if (
        not isinstance(rows, list)
        or len(rows) > INVENTORY_PAGE_SIZE
        or value.get("page") != page
        or value.get("page_size") != INVENTORY_PAGE_SIZE
        or value.get("sort_by") != "id"
        or value.get("sort_order") != "desc"
        or not isinstance(filters, Mapping)
        or filters.get("before_id") != before_id
    ):
        raise CorrectedThermalError("Scheduler task page contract drifted")
    return rows


def _inventory_params(*, page: int, before_id: int) -> dict[str, Any]:
    return {
        "compact": "false",
        "paged": "true",
        "page": page,
        "page_size": INVENTORY_PAGE_SIZE,
        "before_id": before_id,
        "sort_by": "id",
        "sort_order": "desc",
    }


def complete_task_inventory(client: Any) -> dict[str, Any]:
    head_payload = client.get_json(
        "/api/tasks", _inventory_params(page=1, before_id=0)
    )
    head_rows = _page_rows(head_payload, page=1, before_id=0)
    head_ids = [_task_id(row) for row in head_rows]
    high_watermark = max(head_ids, default=0)
    before_id = high_watermark + 1 if high_watermark else 0

    first = client.get_json(
        "/api/tasks", _inventory_params(page=1, before_id=before_id)
    )
    first_rows = _page_rows(first, page=1, before_id=before_id)
    page_count = first.get("page_count")
    total = first.get("filtered_total")
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or not 1 <= page_count <= MAX_INVENTORY_PAGES
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total < 0
        or page_count != max(1, math.ceil(total / INVENTORY_PAGE_SIZE))
    ):
        raise CorrectedThermalError("Scheduler inventory metadata drifted")
    rows: list[dict[str, Any]] = []
    for raw in first_rows:
        rows.append(_mapping(raw, "Scheduler task"))
    for page in range(2, page_count + 1):
        payload = client.get_json(
            "/api/tasks", _inventory_params(page=page, before_id=before_id)
        )
        page_rows = _page_rows(payload, page=page, before_id=before_id)
        if (
            payload.get("page_count") != page_count
            or payload.get("filtered_total") != total
        ):
            raise CorrectedThermalError("Scheduler inventory changed between pages")
        rows.extend(_mapping(row, "Scheduler task") for row in page_rows)
    if len(rows) != total:
        raise CorrectedThermalError("Scheduler inventory is truncated")
    ids = [_task_id(row) for row in rows]
    if (
        len(ids) != len(set(ids))
        or any(task_id >= before_id for task_id in ids if before_id)
    ):
        raise CorrectedThermalError("Scheduler inventory ID snapshot drifted")

    tail_payload = client.get_json(
        "/api/tasks", _inventory_params(page=1, before_id=0)
    )
    tail_rows_raw = _page_rows(tail_payload, page=1, before_id=0)
    tail_rows = [_mapping(row, "Scheduler tail task") for row in tail_rows_raw]
    new_rows = [row for row in tail_rows if _task_id(row) > high_watermark]
    if (
        len(tail_rows) == INVENTORY_PAGE_SIZE
        and tail_rows
        and min(_task_id(row) for row in tail_rows) > high_watermark
    ):
        raise CorrectedThermalError("Scheduler inventory tail lost snapshot overlap")
    combined = [*new_rows, *rows]
    combined_ids = [_task_id(row) for row in combined]
    if len(combined_ids) != len(set(combined_ids)):
        raise CorrectedThermalError("Scheduler inventory has duplicate IDs")
    return {
        "rows": combined,
        "high_watermark_task_id": high_watermark,
        "snapshot_before_id": before_id,
        "page_count": page_count,
        "snapshot_total": total,
        "tail_new_count": len(new_rows),
    }


def _task_id(row: Any) -> int:
    value = _mapping(row, "Scheduler task")
    candidates = [
        value[field]
        for field in ("id", "task_id")
        if value.get(field) is not None
    ]
    if not candidates:
        raise CorrectedThermalError("Scheduler task has no ID")
    task_id = _positive_int(candidates[0], "Scheduler task ID")
    if any(candidate != task_id for candidate in candidates):
        raise CorrectedThermalError("Scheduler task IDs disagree")
    return task_id


def classify_collisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_name: str,
    dedupe_key: str,
) -> dict[str, Any]:
    name_rows = [dict(row) for row in rows if row.get("name") == task_name]
    dedupe_rows = [
        dict(row) for row in rows if row.get("dedupe_key") == dedupe_key
    ]
    exact = [
        dict(row)
        for row in rows
        if row.get("name") == task_name and row.get("dedupe_key") == dedupe_key
    ]
    partial_ids = {
        _task_id(row)
        for row in [*name_rows, *dedupe_rows]
        if not (
            row.get("name") == task_name and row.get("dedupe_key") == dedupe_key
        )
    }
    if partial_ids or len(exact) > 1:
        raise CorrectedThermalError(
            "Scheduler exact-name/dedupe namespace has a partial or duplicate "
            "collision"
        )
    if exact and str(exact[0].get("project") or "").strip() != PROJECT:
        raise CorrectedThermalError("exact task identity exists in another project")
    return {
        "name_match_count": len(name_rows),
        "dedupe_match_count": len(dedupe_rows),
        "exact_match_count": len(exact),
        "exact_matches": exact,
    }


def validate_project_gate(
    project: Mapping[str, Any], inventory_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    value = _mapping(project, "Scheduler project")
    exact = {
        "name": PROJECT,
        "repos": PROJECT_REPOS,
        "setup": PROJECT_SETUP,
        "entrypoints": PROJECT_ENTRYPOINTS,
        "cleanup_globs": PROJECT_CLEANUP_GLOBS,
        "output_globs": PROJECT_OUTPUT_GLOBS,
        "sim_subdir": "simulation",
        "auto_pull": False,
        "max_active_tasks": 500,
        "aedt_backend": "standalone",
    }
    failures = {
        field: value.get(field) == expected for field, expected in exact.items()
    }
    if not all(failures.values()):
        raise CorrectedThermalError(f"Scheduler project contract drifted: {failures}")
    deployments = value.get("deployments")
    if not isinstance(deployments, list) or not any(
        isinstance(row, Mapping)
        and row.get("account_name") == ACCOUNT
        and row.get("status") == "deployed"
        for row in deployments
    ):
        raise CorrectedThermalError("r1 project deployment is unavailable")
    active = [
        row
        for row in inventory_rows
        if str(row.get("project") or "").strip() == PROJECT
        and str(row.get("status") or row.get("state") or "").lower()
        in ACTIVE_STATUSES
    ]
    logical_active = _positive_int(
        value.get("logical_active_count"),
        "project logical active count",
        allow_zero=True,
    )
    if len(active) != logical_active or logical_active >= 500:
        raise CorrectedThermalError("project active-task capacity drifted")
    return {
        "project": PROJECT,
        "maximum": 500,
        "active": logical_active,
        "open_slots": 500 - logical_active,
        "r1_deployed": True,
    }


def validate_license_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    license_value = _mapping(value, "license snapshot")
    admission = _mapping(license_value.get("admission"), "license admission")
    feature = _mapping(
        _mapping(admission.get("features"), "license admission features").get(
            "electronics_desktop"
        ),
        "electronics_desktop admission",
    )
    try:
        age = float(admission.get("snapshot_age_seconds"))
        maximum_age = float(admission.get("snapshot_max_age_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedThermalError("license age is invalid") from exc
    total = _positive_int(feature.get("total"), "license total")
    used = _positive_int(feature.get("used"), "license used", allow_zero=True)
    reserve = _positive_int(
        feature.get("reserve"), "license reserve", allow_zero=True
    )
    headroom = _positive_int(
        feature.get("admit_headroom"),
        "license admit headroom",
        allow_zero=True,
    )
    if (
        license_value.get("server_up") is not True
        or str(license_value.get("error") or "") != ""
        or admission.get("enabled") is not True
        or admission.get("snapshot_valid") is not True
        or not math.isfinite(age)
        or not math.isfinite(maximum_age)
        or age < 0
        or age > maximum_age
        or str(admission.get("blocked_reason") or "") != ""
        or total - used - reserve < 1
        or headroom < 1
    ):
        raise CorrectedThermalError("electronics_desktop admission is unsafe")
    return {
        "server_up": True,
        "snapshot_age_seconds": age,
        "snapshot_max_age_seconds": maximum_age,
        "used": used,
        "total": total,
        "reserve": reserve,
        "admit_headroom": headroom,
    }


def validate_node_gate(
    allocations: Any,
    capacity: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(allocations, list):
        raise CorrectedThermalError("allocation inventory is malformed")
    candidates = [
        dict(row)
        for row in allocations
        if isinstance(row, Mapping)
        and row.get("node_name") == TARGET_NODE
        and row.get("node_metrics_observed_at")
    ]
    if not candidates:
        raise CorrectedThermalError(
            f"{TARGET_NODE} has no fresh node telemetry"
        )
    candidates.sort(
        key=lambda row: (
            _scheduler_timestamp(
                row.get("node_metrics_observed_at"),
                f"{TARGET_NODE} telemetry timestamp",
            ),
            int(row.get("id") or 0),
        )
    )
    node = candidates[-1]
    observed_at = _scheduler_timestamp(
        node.get("node_metrics_observed_at"),
        f"{TARGET_NODE} telemetry timestamp",
    )
    age_seconds = (_utc_now(now) - observed_at).total_seconds()
    state = str(node.get("node_pestat_state") or "").lower()
    cpu_total = _positive_int(
        node.get("node_cpu_total"), f"{TARGET_NODE} CPU total"
    )
    cpu_used = _positive_int(
        node.get("node_cpu_used"),
        f"{TARGET_NODE} CPU used",
        allow_zero=True,
    )
    memory_free = _positive_int(
        node.get("node_memory_free_mb"), f"{TARGET_NODE} free memory"
    )
    if (
        state not in {"idle", "mix"}
        or cpu_total - cpu_used < CPUS
        or memory_free < MEMORY_MB + MEMORY_MARGIN_MB
        or age_seconds > NODE_TELEMETRY_MAX_AGE_SECONDS
        or age_seconds < -NODE_TELEMETRY_FUTURE_TOLERANCE_SECONDS
        or FORBIDDEN_NODE
        in {
            str(node.get(field) or "")
            for field in (
                "node_name",
                "requested_node_name",
                "actual_node_name",
                "allocation_node_name",
            )
        }
    ):
        raise CorrectedThermalError(
            f"{TARGET_NODE} node telemetry cannot admit the task"
        )

    cap = _mapping(capacity, "task capacity")
    ready = _positive_int(
        cap.get("ready_fit_slots"), "ready fit slots", allow_zero=True
    )
    queue_state = str(cap.get("queue_state") or "").lower()
    capacity_allocations = cap.get("allocations")
    if (
        cap.get("memory_pressure_state") != "ok"
        or cap.get("preferred_node_relaxed") is not False
        or queue_state not in {"ready", "opening"}
        or (queue_state == "ready" and ready < 1)
        or not isinstance(capacity_allocations, list)
    ):
        raise CorrectedThermalError(
            f"strict {TARGET_NODE} task capacity is unsafe"
        )
    for row in capacity_allocations:
        item = _mapping(row, "capacity allocation")
        observed_nodes = {
            str(item.get(field) or "")
            for field in (
                "node_name",
                "requested_node_name",
                "actual_node_name",
                "allocation_node_name",
            )
        }
        if FORBIDDEN_NODE in observed_nodes or (
            observed_nodes - {""} and TARGET_NODE not in observed_nodes
        ):
            raise CorrectedThermalError(
                f"capacity response relaxed away from {TARGET_NODE}"
            )
    return {
        "node": TARGET_NODE,
        "node_state": state,
        "telemetry_carrier_allocation_state": str(
            node.get("state") or ""
        ).lower(),
        "node_free_cpus": cpu_total - cpu_used,
        "node_free_memory_mb": memory_free,
        "node_metrics_observed_at": observed_at.isoformat(),
        "node_metrics_age_seconds": age_seconds,
        "node_metrics_max_age_seconds": NODE_TELEMETRY_MAX_AGE_SECONDS,
        "node_metrics_future_tolerance_seconds": (
            NODE_TELEMETRY_FUTURE_TOLERANCE_SECONDS
        ),
        "queue_state": queue_state,
        "queue_reason": str(cap.get("queue_reason") or ""),
        "ready_fit_slots": ready,
        "preferred_node_relaxed": False,
    }


def parse_gpfs_quota(output: str) -> dict[str, Any]:
    rows = [
        line
        for line in output.splitlines()
        if line.startswith("mmlsquota:user:0:")
    ]
    if len(rows) != 1:
        raise CorrectedThermalError("expected exactly one GPFS user quota row")
    values = rows[0].split(":")
    if len(values) < 20:
        raise CorrectedThermalError("GPFS quota row is truncated")
    try:
        return {
            "filesystem": values[6],
            "quota_type": values[7],
            "uid": int(values[8]),
            "name": values[9],
            "usage_bytes": int(values[10]) * 1024,
            "soft_limit_bytes": int(values[11]) * 1024,
            "hard_limit_bytes": int(values[12]) * 1024,
            "in_doubt_bytes": int(values[13]) * 1024,
            "files_used": int(values[15]),
            "files_soft_limit": int(values[16]),
            "files_hard_limit": int(values[17]),
            "files_in_doubt": int(values[18]),
        }
    except (ValueError, IndexError) as exc:
        raise CorrectedThermalError("GPFS quota row has invalid numbers") from exc


def _gpfs_quota_command() -> str:
    return (
        "export LC_ALL=C; "
        "q=/usr/lpp/mmfs/bin/mmlsquota; "
        'test -x "$q"; '
        'u="$(id -un)"; n="$(id -u)"; '
        f'test "$u" = {_shell_quote(ACCOUNT)}; '
        f'test "$n" = {_shell_quote(ACCOUNT_UID)}; '
        '"$q" -u "$u" -Y gpfs'
    )


def probe_gpfs_quota(
    *,
    host: str,
    username: str,
    private_key: Path,
    known_hosts: Path,
    port: int = 22,
) -> dict[str, Any]:
    try:
        import paramiko
    except ImportError as exc:
        raise CorrectedThermalError("paramiko is required for fresh GPFS gate") from exc
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts.resolve(strict=True)))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            key_filename=str(private_key.resolve(strict=True)),
            allow_agent=False,
            look_for_keys=False,
            timeout=20,
            auth_timeout=20,
            banner_timeout=20,
        )
        command = _gpfs_quota_command()
        _stdin, stdout, stderr = client.exec_command(command, timeout=60)
        raw = stdout.read(2 * 1024**2 + 1)
        error = stderr.read(64 * 1024 + 1)
        return_code = stdout.channel.recv_exit_status()
    except Exception as exc:
        raise CorrectedThermalError("fresh read-only GPFS quota probe failed") from exc
    finally:
        client.close()
    if return_code != 0 or len(raw) > 2 * 1024**2 or len(error) > 64 * 1024:
        raise CorrectedThermalError("fresh read-only GPFS quota probe failed")
    try:
        return parse_gpfs_quota(raw.decode("utf-8"))
    except UnicodeError as exc:
        raise CorrectedThermalError("GPFS quota output is not UTF-8") from exc


def validate_storage_gate(
    quota: Mapping[str, Any], retention: Mapping[str, Any]
) -> dict[str, Any]:
    value = _mapping(quota, "GPFS quota")
    if (
        value.get("filesystem") != "gpfs"
        or value.get("quota_type") != "USR"
        or value.get("name") != ACCOUNT
        or value.get("uid") != ACCOUNT_UID
    ):
        raise CorrectedThermalError("GPFS quota identity drifted")
    byte_limit = min(
        _positive_int(value.get("soft_limit_bytes"), "GPFS soft byte limit"),
        _positive_int(value.get("hard_limit_bytes"), "GPFS hard byte limit"),
    )
    inode_limit = min(
        _positive_int(value.get("files_soft_limit"), "GPFS soft inode limit"),
        _positive_int(value.get("files_hard_limit"), "GPFS hard inode limit"),
    )
    available_bytes = (
        byte_limit
        - _positive_int(
            value.get("usage_bytes"), "GPFS byte usage", allow_zero=True
        )
        - _positive_int(
            value.get("in_doubt_bytes"),
            "GPFS in-doubt bytes",
            allow_zero=True,
        )
    )
    available_inodes = (
        inode_limit
        - _positive_int(
            value.get("files_used"), "GPFS used files", allow_zero=True
        )
        - _positive_int(
            value.get("files_in_doubt"),
            "GPFS in-doubt files",
            allow_zero=True,
        )
    )
    byte_budget = _positive_int(
        retention.get("maximum_minimum_bundle_bytes"),
        "minimum retention byte budget",
    )
    inode_budget = 32
    after_bytes = available_bytes - byte_budget
    after_inodes = available_inodes - inode_budget
    if (
        after_bytes
        < _positive_int(
            retention.get("minimum_retained_headroom_after_bytes"),
            "retention byte reserve",
        )
        or after_inodes
        < _positive_int(
            retention.get("minimum_retained_inode_headroom_after"),
            "retention inode reserve",
        )
    ):
        raise CorrectedThermalError("GPFS quota cannot admit bounded retention")
    return {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "available_before_bytes": available_bytes,
        "retention_budget_bytes": byte_budget,
        "headroom_after_bytes": after_bytes,
        "available_before_inodes": available_inodes,
        "retention_budget_inodes": inode_budget,
        "headroom_after_inodes": after_inodes,
    }


_RUNTIME_QUOTA_FIELDS = {
    "filesystem",
    "quota_type",
    "uid",
    "name",
    "usage_bytes",
    "soft_limit_bytes",
    "hard_limit_bytes",
    "in_doubt_bytes",
    "files_used",
    "files_soft_limit",
    "files_hard_limit",
    "files_in_doubt",
}


def build_runtime_quota_authority(
    quota: Mapping[str, Any],
    *,
    retention: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    raw = _mapping(quota, "runtime GPFS quota")
    if set(raw) != _RUNTIME_QUOTA_FIELDS:
        raise CorrectedThermalError("runtime GPFS quota fields drifted")
    normalized: dict[str, Any] = {
        "filesystem": str(raw["filesystem"]),
        "quota_type": str(raw["quota_type"]),
        "name": str(raw["name"]),
    }
    for field in _RUNTIME_QUOTA_FIELDS - {
        "filesystem",
        "quota_type",
        "name",
    }:
        normalized[field] = _positive_int(
            raw[field],
            f"runtime GPFS quota {field}",
            allow_zero=field
            in {
                "usage_bytes",
                "in_doubt_bytes",
                "files_used",
                "files_in_doubt",
            },
        )
    conservative = dict(normalized)
    conservative["in_doubt_bytes"] += RUNTIME_QUOTA_DRIFT_RESERVE_BYTES
    conservative["files_in_doubt"] += RUNTIME_QUOTA_DRIFT_RESERVE_INODES
    admission = validate_storage_gate(conservative, retention)
    observed_epoch = _utc_now(now).timestamp()
    return sealed(
        {
            "schema": RUNTIME_QUOTA_AUTHORITY_SCHEMA,
            "source": "submission-login:mmlsquota-Y",
            "account_name": ACCOUNT,
            "account_uid": ACCOUNT_UID,
            "observed_at_epoch": observed_epoch,
            "maximum_age_seconds": RUNTIME_QUOTA_AUTHORITY_MAX_AGE_SECONDS,
            "conservatism": {
                "maximum_usage_growth_bytes": (
                    RUNTIME_QUOTA_DRIFT_RESERVE_BYTES
                ),
                "maximum_file_growth": RUNTIME_QUOTA_DRIFT_RESERVE_INODES,
            },
            "quota": normalized,
            "admission": admission,
        },
        digest_field="payload_sha256",
    )


def validate_runtime_quota_authority(
    value: Mapping[str, Any],
    *,
    retention: Mapping[str, Any],
    now: datetime | None = None,
    enforce_fresh: bool = False,
) -> dict[str, Any]:
    authority = validate_seal(
        value,
        schema_field="schema",
        schema=RUNTIME_QUOTA_AUTHORITY_SCHEMA,
        digest_field="payload_sha256",
    )
    if set(authority) != {
        "schema",
        "source",
        "account_name",
        "account_uid",
        "observed_at_epoch",
        "maximum_age_seconds",
        "conservatism",
        "quota",
        "admission",
        "payload_sha256",
    }:
        raise CorrectedThermalError("runtime quota authority fields drifted")
    if (
        authority.get("source") != "submission-login:mmlsquota-Y"
        or authority.get("account_name") != ACCOUNT
        or authority.get("account_uid") != ACCOUNT_UID
        or authority.get("maximum_age_seconds")
        != RUNTIME_QUOTA_AUTHORITY_MAX_AGE_SECONDS
    ):
        raise CorrectedThermalError("runtime quota authority identity drifted")
    try:
        observed_epoch = float(authority["observed_at_epoch"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedThermalError(
            "runtime quota authority observation time drifted"
        ) from exc
    if not math.isfinite(observed_epoch) or observed_epoch < 0:
        raise CorrectedThermalError(
            "runtime quota authority observation time drifted"
        )
    quota = _mapping(authority.get("quota"), "runtime quota snapshot")
    if set(quota) != _RUNTIME_QUOTA_FIELDS:
        raise CorrectedThermalError("runtime quota snapshot fields drifted")
    conservatism = _mapping(
        authority.get("conservatism"),
        "runtime quota authority conservatism",
    )
    if conservatism != {
        "maximum_usage_growth_bytes": RUNTIME_QUOTA_DRIFT_RESERVE_BYTES,
        "maximum_file_growth": RUNTIME_QUOTA_DRIFT_RESERVE_INODES,
    }:
        raise CorrectedThermalError(
            "runtime quota authority conservatism drifted"
        )
    conservative = dict(quota)
    conservative["in_doubt_bytes"] += RUNTIME_QUOTA_DRIFT_RESERVE_BYTES
    conservative["files_in_doubt"] += RUNTIME_QUOTA_DRIFT_RESERVE_INODES
    expected_admission = validate_storage_gate(conservative, retention)
    if authority.get("admission") != expected_admission:
        raise CorrectedThermalError("runtime quota admission drifted")
    if enforce_fresh:
        age = _utc_now(now).timestamp() - observed_epoch
        if (
            age < -NODE_TELEMETRY_FUTURE_TOLERANCE_SECONDS
            or age > RUNTIME_QUOTA_AUTHORITY_MAX_AGE_SECONDS
        ):
            raise CorrectedThermalError(
                f"runtime quota authority is stale/future: {age:.3f}s"
            )
    return authority


def validate_infrastructure_retry_source(
    task: Mapping[str, Any],
    *,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    value = _mapping(task, "infrastructure retry source task")
    source = INFRASTRUCTURE_RETRY_SOURCE
    status = str(value.get("status") or value.get("state") or "").lower()
    failure = str(value.get("failure_message") or "")
    marker_prefix = "SOLVER_CORE_CONTRACT_JSON "
    marker_rows = [
        line[len(marker_prefix):]
        for line in stdout.splitlines()
        if line.startswith(marker_prefix)
    ]
    try:
        core_readback = (
            json.loads(marker_rows[0]) if len(marker_rows) == 1 else None
        )
    except json.JSONDecodeError:
        core_readback = None
    expected_core_readback = {
        "contract_version": CORE_CONTRACT_VERSION,
        "opt_in": True,
        "backend": "standalone",
        "requested_num_cores": 8,
        "effective_num_cores": 8,
        "num_tasks": 1,
        "slurm_cpus_per_task_readback": 8,
        "scheduler_task_id_readback": source["task_id"],
        "slurm_job_id_readback": int(source["slurm_job_id"]),
        "auth_sha256": core_contract_auth_sha256(
            source["executor_revision"]
        ),
        "solver_revision": source["executor_revision"],
        "solver_dirty": 0,
    }
    if (
        _task_id(value) != source["task_id"]
        or value.get("name") != source["task_name"]
        or value.get("dedupe_key") != source["dedupe_key"]
        or value.get("project") != PROJECT
        or value.get("account_name") != ACCOUNT
        or status != "failed"
        or isinstance(value.get("exit_code"), bool)
        or value.get("exit_code") != 1
        or value.get("requested_node_name") != source["requested_node"]
        or value.get("actual_node_name") != source["requested_node"]
        or value.get("allocation_node_name") != source["requested_node"]
        or value.get("placement_contract_satisfied") is not True
        or value.get("allocation_id") != source["allocation_id"]
        or str(value.get("slurm_job_id") or "") != source["slurm_job_id"]
        or failure
        != (
            "ansys.aedt.core.internal.errors.GrpcApiError: "
            "Failed to execute gRPC AEDT command: GetVariableValue"
        )
        or "attest_native_fixed_model" not in stderr
        or '"fan_velocity": _native_design_variable' not in stderr
        or "Failed to execute gRPC AEDT command: GetVariableValue" not in stderr
        or not isinstance(core_readback, Mapping)
        or any(
            core_readback.get(key) != expected
            for key, expected in expected_core_readback.items()
        )
        or "CORRECTED_THERMAL_JSON " in stdout
    ):
        raise CorrectedThermalError(
            "infrastructure retry source is not the exact pre-solver "
            "missing native fan design-variable failure"
        )
    return {
        "task_id": source["task_id"],
        "status": "failed",
        "exit_code": 1,
        "failure_class": source["failure_class"],
        "node": source["requested_node"],
        "allocation_id": source["allocation_id"],
        "slurm_job_id": source["slurm_job_id"],
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
    }


def _capacity_params() -> dict[str, Any]:
    return {
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "project": PROJECT,
        "account_name": ACCOUNT,
        "node_name": TARGET_NODE,
        "node_name_policy": "strict",
    }


def fresh_gates(
    plan: Mapping[str, Any],
    *,
    client: Any,
    storage_probe: Callable[[], Mapping[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    planned_timeout = int(plan["submission_profile"]["timeout_seconds"])
    allowed_timeout = calculate_timeout(now)
    if planned_timeout > allowed_timeout:
        raise CorrectedThermalError(
            "sealed plan timeout no longer fits the deadline; create a fresh plan"
        )
    runtime_quota = validate_runtime_quota_authority(
        _mapping(
            plan.get("runtime_quota_authority"),
            "runtime quota authority",
        ),
        retention=plan["retention"],
        now=now,
        enforce_fresh=True,
    )
    source_task_id = int(INFRASTRUCTURE_RETRY_SOURCE["task_id"])
    infrastructure_retry = validate_infrastructure_retry_source(
        client.get_json(f"/api/tasks/{source_task_id}"),
        stdout=client.get_text(
            f"/api/tasks/{source_task_id}/stdout",
            {"max_bytes": 16 * 1024**2},
        ),
        stderr=client.get_text(
            f"/api/tasks/{source_task_id}/stderr",
            {"max_bytes": 16 * 1024**2},
        ),
    )
    inventory = complete_task_inventory(client)
    collision = classify_collisions(
        inventory["rows"],
        task_name=plan["task_identity"]["name"],
        dedupe_key=plan["task_identity"]["dedupe_key"],
    )
    project = validate_project_gate(
        client.get_json(f"/api/projects/{PROJECT}"), inventory["rows"]
    )
    licenses = validate_license_gate(client.get_json("/api/licenses"))
    node = validate_node_gate(
        client.get_json("/api/allocations"),
        client.get_json("/api/task-capacity", _capacity_params()),
        now=now,
    )
    fresh_quota = _mapping(storage_probe(), "fresh GPFS quota")
    storage = validate_storage_gate(fresh_quota, plan["retention"])
    sealed_quota = _mapping(
        runtime_quota["quota"], "sealed runtime quota snapshot"
    )
    conservatism = _mapping(
        runtime_quota["conservatism"],
        "sealed runtime quota conservatism",
    )
    fresh_usage = int(fresh_quota["usage_bytes"]) + int(
        fresh_quota["in_doubt_bytes"]
    )
    sealed_usage_ceiling = int(sealed_quota["usage_bytes"]) + int(
        sealed_quota["in_doubt_bytes"]
    ) + int(conservatism["maximum_usage_growth_bytes"])
    fresh_files = int(fresh_quota["files_used"]) + int(
        fresh_quota["files_in_doubt"]
    )
    sealed_files_ceiling = int(sealed_quota["files_used"]) + int(
        sealed_quota["files_in_doubt"]
    ) + int(conservatism["maximum_file_growth"])
    if (
        fresh_usage > sealed_usage_ceiling
        or fresh_files > sealed_files_ceiling
    ):
        raise CorrectedThermalError(
            "fresh GPFS quota exceeded the sealed conservative authority"
        )
    return {
        "observed_at_utc": _utc_now(now).isoformat(),
        "deadline_timeout_seconds": planned_timeout,
        "runtime_quota_authority": {
            "payload_sha256": runtime_quota["payload_sha256"],
            "observed_at_epoch": runtime_quota["observed_at_epoch"],
            "maximum_age_seconds": runtime_quota["maximum_age_seconds"],
        },
        "retry_of_infrastructure": infrastructure_retry,
        "inventory": {
            key: value for key, value in inventory.items() if key != "rows"
        },
        "collisions": collision,
        "project": project,
        "licenses": licenses,
        "node": node,
        "storage": storage,
        "storage_vs_sealed_authority": {
            "fresh_usage_bytes": fresh_usage,
            "sealed_usage_ceiling_bytes": sealed_usage_ceiling,
            "fresh_files": fresh_files,
            "sealed_files_ceiling": sealed_files_ceiling,
            "passed": True,
        },
    }


def build_submission_body(plan: Mapping[str, Any]) -> dict[str, Any]:
    profile = plan["submission_profile"]
    return {
        "name": plan["task_identity"]["name"],
        "project": PROJECT,
        "remote_cwd": profile["remote_cwd"],
        "command": plan["canonical_command"],
        "required_capability": profile["required_capability"],
        "env_profile": profile["env_profile"],
        "scheduling_profile": profile["scheduling_profile"],
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": GPUS,
        "account_name": ACCOUNT,
        "node_name": TARGET_NODE,
        "node_name_policy": "strict",
        "max_workers_per_node": 1,
        "same_node_as_task_id": 0,
        "priority": profile["priority"],
        "timeout_seconds": profile["timeout_seconds"],
        "dedupe_key": plan["task_identity"]["dedupe_key"],
        "cleanup_globs": profile["cleanup_globs"],
        "aedt_backend": "standalone",
    }


def validate_task_readback(
    raw: Mapping[str, Any],
    pending: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    task = _mapping(raw, "submitted task readback")
    _task_id(task)
    body = build_submission_body(plan)
    direct_fields = {
        "name": body["name"],
        "project": PROJECT,
        "dedupe_key": body["dedupe_key"],
        "required_capability": body["required_capability"],
        "env_profile": body["env_profile"],
        "scheduling_profile": body["scheduling_profile"],
        "aedt_backend": "standalone",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": GPUS,
        "account_name": ACCOUNT,
        "same_node_as_task_id": 0,
        "timeout_seconds": body["timeout_seconds"],
        "priority": body["priority"],
        "node_name_policy": "strict",
    }
    failures = {
        field: task.get(field) == expected
        for field, expected in direct_fields.items()
    }
    if not all(failures.values()):
        raise CorrectedThermalError(f"submitted task readback drifted: {failures}")
    if task.get("max_workers_per_node") != 1:
        raise CorrectedThermalError("submitted task worker limit drifted")
    requested_node = str(
        task.get("requested_node_name") or task.get("node_name") or ""
    )
    requested_policy = str(
        task.get("requested_node_name_policy")
        or task.get("node_name_policy")
        or ""
    )
    node_fields = {
        str(task.get(field) or "")
        for field in (
            "node_name",
            "requested_node_name",
            "actual_node_name",
            "allocation_node_name",
        )
    }
    node_fields.discard("")
    status = str(task.get("status") or task.get("state") or "").lower()
    if (
        requested_node != TARGET_NODE
        or requested_policy != "strict"
        or task.get("preferred_node_relaxed") is not False
        or FORBIDDEN_NODE in node_fields
        or (status == "running" and TARGET_NODE not in node_fields)
    ):
        raise CorrectedThermalError("submitted task placement readback drifted")
    winner = pending["winner"]
    if (
        task.get("name") != winner["task_name"]
        or task.get("dedupe_key") != winner["dedupe_key"]
    ):
        raise CorrectedThermalError("submitted task differs from claim winner")
    return task


def _post_task_id(response: Any) -> int:
    value = _mapping(response, "Scheduler POST response")
    if isinstance(value.get("task"), Mapping):
        value = dict(value["task"])
    return _task_id(value)


def _sibling_snapshot(
    collision: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    exact = copy.deepcopy(list(collision["exact_matches"]))
    return {
        "matching_task_count": len(exact),
        "matching_tasks": exact,
        "inventory_high_watermark_task_id": inventory[
            "high_watermark_task_id"
        ],
        "inventory_snapshot_before_id": inventory["snapshot_before_id"],
        "inventory_page_count": inventory["page_count"],
        "inventory_snapshot_total": inventory["snapshot_total"],
        "inventory_tail_new_count": inventory["tail_new_count"],
    }


def _reconcile(
    *,
    client: Any,
    plan: Mapping[str, Any],
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    last_inventory: dict[str, Any] | None = None
    last_collision: dict[str, Any] | None = None
    for attempt in range(attempts):
        last_inventory = complete_task_inventory(client)
        last_collision = classify_collisions(
            last_inventory["rows"],
            task_name=plan["task_identity"]["name"],
            dedupe_key=plan["task_identity"]["dedupe_key"],
        )
        if last_collision["exact_match_count"] == 1:
            row = last_collision["exact_matches"][0]
            task = client.get_json(f"/api/tasks/{_task_id(row)}")
            return (
                _mapping(task, "reconciled task"),
                last_inventory,
                last_collision,
            )
        if attempt + 1 < attempts:
            time.sleep(0.25 * (attempt + 1))
    raise SubmissionUncertain(
        "no unique exact task appeared after the sole possible POST"
    )


def _default_lock_path() -> Path:
    local = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "MFT_1MW_2026" / MUTATION_LOCK_NAME


@contextmanager
def mutation_lock(path: Path | None = None, timeout_seconds: int = 900):
    lock_path = (path or _default_lock_path()).absolute()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    stream.seek(0)
    if stream.read(1) == b"":
        stream.seek(0)
        stream.write(b"\0")
        stream.flush()
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while not acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise CorrectedThermalError(
                        "timed out acquiring campaign mutation lock"
                    ) from None
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def submit_plan(
    plan_path: Path,
    *,
    apply: bool = False,
    client: Any | None = None,
    storage_probe: Callable[[], Mapping[str, Any]],
    receipt_path: Path | None = None,
    now: datetime | None = None,
    verify_local_files: bool = True,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    plan = load_plan(plan_path, verify_local_files=verify_local_files)
    scheduler = client or SchedulerHTTP(plan["scheduler"]["url"])
    if not apply:
        gates = fresh_gates(
            plan, client=scheduler, storage_probe=storage_probe, now=now
        )
        return {
            "schema": "mft-corrected-thermal-plan-only-readiness-v1",
            "plan_payload_sha256": plan["plan_payload_sha256"],
            "gates": gates,
            "scheduler_submission_performed": False,
            "scheduler_post_calls": int(getattr(scheduler, "post_calls", 0)),
            "apply_required_for_submission": True,
        }

    with mutation_lock(lock_path):
        gates = fresh_gates(
            plan, client=scheduler, storage_probe=storage_probe, now=now
        )
        claim_config = plan["claim"]
        claim_root = Path(claim_config["root"])
        authority = atomic_claim.initialize_claim_root(
            claim_root,
            campaign_id=CAMPAIGN_ID,
            campaign_authority_sha256=claim_config[
                "campaign_authority_sha256"
            ],
            root_id=claim_config["root_id"],
        )
        reference = atomic_claim.build_claim_reference(
            authority,
            candidate_physics_sha256=CANDIDATE_SHA256,
            logical_authority_task_id=SOURCE_LOGICAL_TASK_ID,
            retry_generation=RETRY_GENERATION,
        )
        body = build_submission_body(plan)
        winner = {
            "immediate_task_id": INFRASTRUCTURE_RETRY_SOURCE["task_id"],
            "immediate_retry_kind": "infrastructure",
            "plan_payload_sha256": plan["plan_payload_sha256"],
            "plan_file_sha256": sha256_file(plan_path),
            "profile_sha256": canonical_sha256(body),
            "resources": {
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "timeout_seconds": body["timeout_seconds"],
                "same_node_as_task_id": 0,
            },
            "task_name": body["name"],
            "dedupe_key": body["dedupe_key"],
        }
        acquired = atomic_claim.acquire_claim(
            claim_root, reference, winner
        )
        if acquired["status"] == "existing_finalized":
            return {
                "schema": "mft-corrected-thermal-submit-result-v1",
                "plan_payload_sha256": plan["plan_payload_sha256"],
                "claim_status": "existing_finalized",
                "finalized_claim": acquired["claim"],
                "scheduler_submission_performed": False,
                "scheduler_post_calls": int(
                    getattr(scheduler, "post_calls", 0)
                ),
            }
        pending = acquired["claim"]
        exact = gates["collisions"]["exact_matches"]
        performed = False
        post_error: Exception | None = None
        if acquired["status"] == "fresh_pending" and not exact:
            try:
                response = scheduler.post_json("/api/tasks", body)
                _post_task_id(response)
                performed = True
            except SubmissionUncertain as exc:
                post_error = exc
        elif acquired["status"] == "existing_pending" and not exact:
            raise SubmissionUncertain(
                "existing pending claim has no exact task; never re-POSTing"
            )

        try:
            task, inventory, collision = _reconcile(
                client=scheduler, plan=plan
            )
        except SubmissionUncertain:
            if post_error is not None:
                raise post_error
            raise
        if post_error is not None:
            # The namespace was empty before the sole POST and now contains
            # one exact row, so GET evidence resolves the response loss.
            performed = True
        normalized = validate_task_readback(
            task, pending, plan=plan
        )
        siblings = _sibling_snapshot(collision, inventory)
        finalized = atomic_claim.recover_pending_claim(
            claim_root,
            reference,
            pending,
            matching_tasks=[normalized],
            sibling_snapshot=siblings,
            evidence_validator=lambda row, claim: validate_task_readback(
                row, claim, plan=plan
            ),
        )
        result = sealed(
            {
                "schema": "mft-corrected-thermal-submit-result-v1",
                "plan_payload_sha256": plan["plan_payload_sha256"],
                "gates": gates,
                "claim_status": acquired["status"],
                "finalized_claim": finalized,
                "task_id": finalized["task_id"],
                "scheduler_submission_performed": performed,
                "scheduler_post_calls": int(
                    getattr(scheduler, "post_calls", int(performed))
                ),
            },
            digest_field="receipt_payload_sha256",
        )
        if receipt_path is not None:
            _atomic_write_json(receipt_path, result)
        return result


def _validate_retained_metadata(
    metadata: os.stat_result,
    *,
    directory: bool,
    label: str,
    posix: bool | None = None,
) -> None:
    enforce = os.name != "nt" if posix is None else posix
    if not enforce:
        return
    expected_mode = 0o500 if directory else 0o400
    if (
        metadata.st_uid != ACCOUNT_UID
        or stat.S_IMODE(metadata.st_mode) != expected_mode
        or (not directory and metadata.st_nlink != 1)
    ):
        raise CorrectedThermalError(
            f"{label} owner/mode/link contract drifted"
        )


def _validate_optional_field_bundle(value: Any) -> dict[str, Any]:
    optional = _mapping(value, "optional field bundle")
    if (
        set(optional)
        != {
            "retained",
            "reason",
            "observed_results_tree_logical_bytes",
            "observed_results_tree_file_count",
            "can_be_retained_only_by_separate_size_and_quota_preflight",
        }
        or optional.get("retained") is not False
        or optional.get("reason") != OPTIONAL_FIELD_BUNDLE_REASON
        or isinstance(
            optional.get("observed_results_tree_logical_bytes"), bool
        )
        or not isinstance(
            optional.get("observed_results_tree_logical_bytes"), int
        )
        or optional.get("observed_results_tree_logical_bytes") < 0
        or isinstance(optional.get("observed_results_tree_file_count"), bool)
        or not isinstance(
            optional.get("observed_results_tree_file_count"), int
        )
        or optional.get("observed_results_tree_file_count") < 0
        or optional.get(
            "can_be_retained_only_by_separate_size_and_quota_preflight"
        )
        is not True
    ):
        raise CorrectedThermalError("optional full field bundle policy drifted")
    return optional


def _validate_retention_control_records(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    destination: Path,
    manifest_path: Path,
    minimum_bundle_bytes: int,
) -> dict[str, Any]:
    if manifest.get("control_evidence") != RETENTION_CONTROL_EVIDENCE:
        raise CorrectedThermalError(
            "minimum retained manifest control evidence drifted"
        )
    optional = _validate_optional_field_bundle(
        manifest.get("optional_field_bundle")
    )
    expected_receipt_fields = {
        "schema",
        "destination",
        "manifest_path",
        "manifest_sha256",
        "prune_marker_path",
        "file_count",
        "minimum_bundle_bytes",
        "optional_field_bundle",
        "published_atomically_with_package",
        "manifest_inventory_membership",
        "evidence_semantics",
        "passed",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema")
        != "mft-corrected-thermal-minimum-retention-receipt-v1"
        or receipt.get("destination") != str(destination)
        or receipt.get("manifest_path") != str(manifest_path)
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("prune_marker_path")
        != str(destination / ".slurm-scheduler-preserve.json")
        or receipt.get("file_count") != 5
        or receipt.get("minimum_bundle_bytes") != minimum_bundle_bytes
        or receipt.get("optional_field_bundle") != optional
        or receipt.get("published_atomically_with_package") is not True
        or receipt.get("manifest_inventory_membership") is not False
        or receipt.get("evidence_semantics")
        != RETENTION_EVIDENCE_SEMANTICS
        or receipt.get("passed") is not True
    ):
        raise CorrectedThermalError("minimum retention receipt drifted")
    return optional


def retain_execution(
    *,
    execution_plan: Path,
    execution_root: Path,
    execution_exit_code: int,
    allowed_scratch_root: Path = Path("/enroot"),
) -> dict[str, Any]:
    """Authenticate the executor's compact GPFS package and emit a GET marker.

    The execution consumer performs the bounded atomic copy.  This adapter is
    deliberately read-only: it neither duplicates the checkpoint/premesh nor
    creates another GPFS bundle.
    """
    contract = validate_execution_contract(
        _read_json(execution_plan, "execution contract")
    )
    storage = contract["output_storage"]
    source = execution_root.resolve(strict=True)
    scratch = allowed_scratch_root.resolve(strict=True)
    if source == scratch or scratch not in source.parents:
        raise CorrectedThermalError("execution output escaped node-local scratch")
    if str(source) != str(Path(storage["scratch_root"]).resolve(strict=True)):
        raise CorrectedThermalError("execution output differs from sealed scratch root")
    if execution_exit_code != 0:
        raise CorrectedThermalError(
            "failed corrected-thermal solve has no success retention marker"
        )
    retained_root = Path(storage["retained_root"]).resolve(strict=True)
    label = (
        f"b7c-{contract['checkpoint_manifest_sha256'][:12]}-"
        f"{contract['executor_revision'][:12]}"
    )
    destination = retained_root / label
    if destination.parent != retained_root:
        raise CorrectedThermalError("minimum retained destination escaped root")
    metadata = destination.lstat()
    if destination.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CorrectedThermalError("minimum retained package is not a directory")
    _validate_retained_metadata(
        metadata, directory=True, label="minimum retained package root"
    )

    manifest_path = destination / "manifest.json"
    manifest = validate_seal(
        _read_json(manifest_path, "minimum retained package manifest"),
        schema_field="schema",
        schema="mft-corrected-thermal-minimum-retained-package-v1",
        digest_field="payload_sha256",
    )
    exact_manifest = {
        "diagnostic_only": True,
        "canonical": False,
        "candidate_sha256": CANDIDATE_SHA256,
        "checkpoint_manifest_sha256": contract["checkpoint_manifest_sha256"],
        "source_solver_revision": SOURCE_SOLVER_REVISION,
        "executor_solver_revision": contract["executor_revision"],
        "executor_required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
        "aedt_mixed_provenance": (
            "a1e4f source project saved after thermal-only solve by the exact "
            "planned clean a927-descendant executor"
        ),
    }
    expected_manifest_fields = {
        "schema",
        *exact_manifest,
        "files",
        "minimum_bundle_bytes",
        "optional_field_bundle",
        "control_evidence",
        "payload_sha256",
    }
    if set(manifest) != expected_manifest_fields or any(
        manifest.get(key) != expected for key, expected in exact_manifest.items()
    ):
        raise CorrectedThermalError("minimum retained manifest identity drifted")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 5:
        raise CorrectedThermalError("minimum retained file inventory drifted")
    required_names = {
        "symmetric.aedt",
        "corrected_result.json",
        "execution_receipt.json",
    }
    observed_names = {str(row.get("path") or "") for row in files if isinstance(row, Mapping)}
    if (
        not required_names.issubset(observed_names)
        or len([name for name in observed_names if name.startswith("convergence/")])
        != 1
        or len([name for name in observed_names if name.startswith("profile/")])
        != 1
    ):
        raise CorrectedThermalError("minimum retained package members drifted")
    verified_files: list[dict[str, Any]] = []
    total = 0
    for raw in files:
        row = _mapping(raw, "minimum retained file")
        if set(row) != {"path", "size_bytes", "sha256"}:
            raise CorrectedThermalError(
                "minimum retained file record fields drifted"
            )
        relative = _safe_checkpoint_relative(row.get("path"))
        path = destination.joinpath(*PurePosixPath(relative).parts)
        item = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_size
            != _positive_int(
                row.get("size_bytes"),
                f"minimum retained size {relative}",
                allow_zero=True,
            )
            or sha256_file(path)
            != _sha256(
                row.get("sha256"), f"minimum retained digest {relative}"
            )
        ):
            raise CorrectedThermalError(
                f"minimum retained file authentication failed: {relative}"
            )
        total += item.st_size
        verified_files.append(
            {
                "path": relative,
                "size_bytes": item.st_size,
                "sha256": row["sha256"],
            }
        )
    if (
        total != manifest.get("minimum_bundle_bytes")
        or total > int(storage["maximum_minimum_bundle_bytes"])
    ):
        raise CorrectedThermalError("minimum retained bundle size drifted")
    preserve = _read_json(
        destination / ".slurm-scheduler-preserve.json",
        "minimum retained prune marker",
    )
    if (
        set(preserve)
        != {"schema", "owner", "preserve", "reason", "diagnostic_only"}
        or preserve.get("schema") != "slurm-scheduler-prune-protection-v1"
        or preserve.get("owner") != PROJECT
        or preserve.get("preserve") is not True
        or preserve.get("diagnostic_only") is not True
        or preserve.get("reason") != PRUNE_MARKER_REASON
    ):
        raise CorrectedThermalError("minimum retained prune marker drifted")
    retention_receipt_path = destination / "retention_receipt.json"
    retention_receipt = _read_json(
        retention_receipt_path, "minimum retention receipt"
    )
    optional = _validate_retention_control_records(
        manifest=manifest,
        receipt=retention_receipt,
        destination=destination,
        manifest_path=manifest_path,
        minimum_bundle_bytes=total,
    )
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, dir_names, file_names in os.walk(
        destination, followlinks=False
    ):
        directory_path = Path(directory)
        for name in dir_names:
            path = directory_path / name
            item = path.lstat()
            if path.is_symlink() or not stat.S_ISDIR(item.st_mode):
                raise CorrectedThermalError(
                    "minimum retained tree has a non-directory"
                )
            _validate_retained_metadata(
                item,
                directory=True,
                label=f"minimum retained directory {path}",
            )
            actual_directories.add(path.relative_to(destination).as_posix())
        for name in file_names:
            path = directory_path / name
            item = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(item.st_mode):
                raise CorrectedThermalError(
                    "minimum retained tree has a non-file"
                )
            _validate_retained_metadata(
                item,
                directory=False,
                label=f"minimum retained file {path}",
            )
            actual_files.add(path.relative_to(destination).as_posix())
    expected_files = observed_names | {
        "manifest.json",
        ".slurm-scheduler-preserve.json",
        "retention_receipt.json",
    }
    expected_directories = {
        PurePosixPath(name).parent.as_posix()
        for name in observed_names
        if len(PurePosixPath(name).parts) > 1
    }
    if (
        actual_files != expected_files
        or actual_directories != expected_directories
        or len(actual_files) != 8
        or len(actual_directories) != 2
    ):
        raise CorrectedThermalError("minimum retained tree inventory drifted")
    receipt_path = destination / "execution_receipt.json"
    receipt = _read_json(receipt_path, "corrected thermal execution receipt")
    if (
        receipt.get("schema")
        != "mft-corrected-thermal-checkpoint-execution-v1"
        or receipt.get("diagnostic_only") is not True
        or receipt.get("canonical") is not False
        or receipt.get("production_truth_eligible") is not False
        or receipt.get("status") != "diagnostic_complete"
        or receipt.get("source_checkpoint_manifest_sha256")
        != contract["checkpoint_manifest_sha256"]
    ):
        raise CorrectedThermalError("corrected thermal receipt identity drifted")
    executor = _mapping(
        receipt.get("executor_provenance"), "receipt executor provenance"
    )
    if executor.get("executor_solver_revision") != contract["executor_revision"]:
        raise CorrectedThermalError("receipt executor revision drifted")
    corrected_result = _read_json(
        destination / "corrected_result.json",
        "corrected thermal compact result",
    )
    if (
        corrected_result.get("schema")
        != "mft-corrected-thermal-diagnostic-result-v1"
        or corrected_result.get("diagnostic_only") is not True
        or corrected_result.get("canonical") is not False
        or corrected_result.get("source_provenance")
        != receipt.get("source_provenance")
        or corrected_result.get("executor_provenance")
        != receipt.get("executor_provenance")
    ):
        raise CorrectedThermalError("corrected compact result drifted")
    compact_receipt = {
        key: copy.deepcopy(receipt.get(key))
        for key in (
            "schema",
            "status",
            "diagnostic_only",
            "canonical",
            "production_truth_eligible",
            "source_checkpoint_manifest_sha256",
            "source_provenance",
            "executor_provenance",
            "fixed_physics",
            "parallel_attestation",
            "solve",
        )
        if key in receipt
    }
    compact_receipt.update(
        {
            key: copy.deepcopy(corrected_result.get(key))
            for key in (
                "parallel_attestation",
                "temperatures",
                "constraint_observation",
            )
        }
    )
    marker = sealed(
        {
            "schema": RETENTION_MARKER_SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "task_name": contract["task_name"],
            "dedupe_key": contract["dedupe_key"],
            "submission_contract_sha256": contract[
                "submission_contract_sha256"
            ],
            "execution_plan_payload_sha256": contract[
                "plan_payload_sha256"
            ],
            "retained_destination": str(destination),
            "retention_manifest_path": str(manifest_path),
            "retention_manifest_payload_sha256": manifest["payload_sha256"],
            "retention_manifest_file_sha256": sha256_file(manifest_path),
            "corrected_receipt_path": str(receipt_path),
            "corrected_receipt_sha256": sha256_file(receipt_path),
            "execution_exit_code": execution_exit_code,
            "verified_files": verified_files,
            "minimum_bundle_bytes": total,
            "optional_field_bundle": optional,
            "corrected_receipt": compact_receipt,
        },
        digest_field="payload_sha256",
    )
    print(
        "CORRECTED_THERMAL_JSON "
        + json.dumps(
            marker,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return marker


def _thermal_field(name: str) -> bool:
    lowered = name.lower()
    return (
        name.startswith("T_")
        or name.startswith("Tprobe_")
        or lowered.startswith("thermal_")
        or lowered.startswith("temperature_")
        or lowered.startswith("corrected_thermal_")
        or lowered
        in {
            "result_valid",
            "result_valid_thermal",
            "constraints_satisfied",
            "feasible",
            "winding_max_c",
            "core_max_c",
        }
    )


def create_maxwell_source_evidence(
    *,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    read_only_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    source_row = _mapping(row, "Maxwell recovered result row")
    if not source_row or any(_thermal_field(str(key)) for key in source_row):
        raise CorrectedThermalError(
            "Maxwell recovery evidence must contain nonthermal fields only"
        )
    attestation = _mapping(
        read_only_attestation, "Maxwell read-only attestation"
    )
    required = {
        "saved_results_opened_read_only": True,
        "maxwell_analysis_calls": 0,
        "project_write_calls": 0,
        "source_checkpoint_unchanged": True,
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise CorrectedThermalError("Maxwell recovery was not proven read-only")
    _sha256(
        attestation.get("source_results_manifest_sha256"),
        "Maxwell saved-results manifest",
    )
    return sealed(
        {
            "schema": SOURCE_EVIDENCE_SCHEMA,
            "source_kind": "checkpoint_maxwell_saved_results_readonly",
            "diagnostic_only": True,
            "canonical": False,
            "candidate_sha256": CANDIDATE_SHA256,
            "logical_task_id": SOURCE_LOGICAL_TASK_ID,
            "source_execution_task_id": SOURCE_EXECUTION_TASK_ID,
            "source_solver_revision": SOURCE_SOLVER_REVISION,
            "source_library_revision": LIBRARY_REVISION,
            "checkpoint_manifest_sha256": plan["checkpoint_manifest"][
                "file_sha256"
            ],
            "read_only_attestation": attestation,
            "nonthermal_row": source_row,
        },
        digest_field="payload_sha256",
    )


def validate_maxwell_source_evidence(
    value: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = validate_seal(
        value,
        schema_field="schema",
        schema=SOURCE_EVIDENCE_SCHEMA,
        digest_field="payload_sha256",
    )
    exact = {
        "source_kind": "checkpoint_maxwell_saved_results_readonly",
        "diagnostic_only": True,
        "canonical": False,
        "candidate_sha256": CANDIDATE_SHA256,
        "logical_task_id": SOURCE_LOGICAL_TASK_ID,
        "source_execution_task_id": SOURCE_EXECUTION_TASK_ID,
        "source_solver_revision": SOURCE_SOLVER_REVISION,
        "source_library_revision": LIBRARY_REVISION,
        "checkpoint_manifest_sha256": plan["checkpoint_manifest"][
            "file_sha256"
        ],
    }
    if any(evidence.get(key) != expected for key, expected in exact.items()):
        raise CorrectedThermalError("Maxwell source evidence identity drifted")
    attestation = _mapping(
        evidence.get("read_only_attestation"),
        "Maxwell read-only attestation",
    )
    required = {
        "saved_results_opened_read_only": True,
        "maxwell_analysis_calls": 0,
        "project_write_calls": 0,
        "source_checkpoint_unchanged": True,
    }
    if any(attestation.get(key) != value for key, value in required.items()):
        raise CorrectedThermalError("Maxwell source evidence is not read-only")
    _sha256(
        attestation.get("source_results_manifest_sha256"),
        "Maxwell saved-results manifest",
    )
    row = _mapping(evidence.get("nonthermal_row"), "Maxwell nonthermal row")
    if not row or any(_thermal_field(str(key)) for key in row):
        raise CorrectedThermalError("Maxwell evidence contains thermal fields")
    return evidence


def _latest_json_line(text: str, prefix: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _terminal_source(client: Any) -> dict[str, Any] | None:
    task = _mapping(
        client.get_json(f"/api/tasks/{SOURCE_EXECUTION_TASK_ID}"),
        "source task readback",
    )
    if (
        _task_id(task) != SOURCE_EXECUTION_TASK_ID
        or task.get("name") != SOURCE_TASK_NAME
        or task.get("dedupe_key") != SOURCE_TASK_DEDUPE
        or task.get("project") != PROJECT
    ):
        raise CorrectedThermalError("source task 96304 identity drifted")
    status = str(task.get("status") or task.get("state") or "").lower()
    if status not in TERMINAL_STATUSES:
        return None
    if (
        status != "completed"
        or isinstance(task.get("exit_code"), bool)
        or task.get("exit_code") != 0
    ):
        raise CorrectedThermalError("source task 96304 terminated without success")
    stdout = client.get_text(
        f"/api/tasks/{SOURCE_EXECUTION_TASK_ID}/stdout",
        {"max_bytes": 16 * 1024**2},
    )
    row = _latest_json_line(stdout, "RESULT_JSON ")
    if row is None:
        raise CorrectedThermalError("terminal source task has no RESULT_JSON")
    return {
        "source_kind": "terminal_source_task_96304",
        "task": task,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "row": row,
    }


def _validate_retention_marker(
    value: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    marker = validate_seal(
        value,
        schema_field="schema",
        schema=RETENTION_MARKER_SCHEMA,
        digest_field="payload_sha256",
    )
    if (
        marker.get("diagnostic_only") is not True
        or marker.get("canonical") is not False
        or marker.get("production_truth_eligible") is not False
        or marker.get("task_name") != TASK_NAME
        or marker.get("dedupe_key") != plan["task_identity"]["dedupe_key"]
        or marker.get("submission_contract_sha256")
        != plan["contract_digest_sha256"]
        or marker.get("execution_plan_payload_sha256")
        != plan["execution_contract"]["plan_payload_sha256"]
        or marker.get("execution_exit_code") != 0
    ):
        raise CorrectedThermalError("corrected thermal retention marker drifted")
    receipt = _mapping(
        marker.get("corrected_receipt"), "corrected thermal compact receipt"
    )
    if (
        receipt.get("schema")
        != "mft-corrected-thermal-checkpoint-execution-v1"
        or receipt.get("status") != "diagnostic_complete"
        or receipt.get("diagnostic_only") is not True
        or receipt.get("canonical") is not False
        or receipt.get("production_truth_eligible") is not False
        or receipt.get("source_checkpoint_manifest_sha256")
        != plan["checkpoint_manifest"]["file_sha256"]
        or receipt.get("fixed_physics")
        != plan["checkpoint_manifest"]["physics_boundary"]
    ):
        raise CorrectedThermalError("corrected thermal compact receipt drifted")
    source = _mapping(receipt.get("source_provenance"), "receipt source")
    if (
        source.get("candidate_sha256") != CANDIDATE_SHA256
        or source.get("logical_task_id") != SOURCE_LOGICAL_TASK_ID
        or source.get("execution_task_id") != SOURCE_EXECUTION_TASK_ID
        or source.get("solver_revision") != SOURCE_SOLVER_REVISION
    ):
        raise CorrectedThermalError("corrected thermal source provenance drifted")
    executor = _mapping(receipt.get("executor_provenance"), "receipt executor")
    if (
        executor.get("executor_solver_revision")
        != plan["executor"]["revision"]
        or executor.get("pyaedt_library_revision") != LIBRARY_REVISION
    ):
        raise CorrectedThermalError("corrected thermal executor provenance drifted")
    return marker


def collect_mixed_provenance(
    *,
    plan: Mapping[str, Any],
    corrected_task_id: int,
    client: Any,
    maxwell_source_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = _mapping(
        client.get_json(f"/api/tasks/{corrected_task_id}"),
        "corrected thermal task",
    )
    if (
        _task_id(task) != corrected_task_id
        or task.get("name") != TASK_NAME
        or task.get("dedupe_key") != plan["task_identity"]["dedupe_key"]
        or task.get("project") != PROJECT
        or str(task.get("status") or task.get("state") or "").lower()
        != "completed"
        or isinstance(task.get("exit_code"), bool)
        or task.get("exit_code") != 0
    ):
        raise CorrectedThermalError("corrected thermal task is not terminal-success")
    stdout = client.get_text(
        f"/api/tasks/{corrected_task_id}/stdout",
        {"max_bytes": 16 * 1024**2},
    )
    marker_raw = _latest_json_line(stdout, "CORRECTED_THERMAL_JSON ")
    if marker_raw is None:
        raise CorrectedThermalError("corrected thermal stdout has no marker")
    marker = _validate_retention_marker(marker_raw, plan)

    terminal = _terminal_source(client)
    maxwell = (
        None
        if maxwell_source_evidence is None
        else validate_maxwell_source_evidence(maxwell_source_evidence, plan)
    )
    if (terminal is None) == (maxwell is None):
        raise CorrectedThermalError(
            "exactly one nonthermal source is required; missing or ambiguous"
        )
    if maxwell is not None:
        source_kind = maxwell["source_kind"]
        source_row = copy.deepcopy(maxwell["nonthermal_row"])
        source_record = {
            "source_kind": source_kind,
            "evidence_payload_sha256": maxwell["payload_sha256"],
            "checkpoint_manifest_sha256": maxwell[
                "checkpoint_manifest_sha256"
            ],
        }
    else:
        assert terminal is not None
        source_kind = terminal["source_kind"]
        raw_source_row = _mapping(terminal["row"], "terminal source RESULT_JSON")
        source_row = {
            str(key): copy.deepcopy(value)
            for key, value in raw_source_row.items()
            if not _thermal_field(str(key))
        }
        source_record = {
            "source_kind": source_kind,
            "task_id": SOURCE_EXECUTION_TASK_ID,
            "task_name": SOURCE_TASK_NAME,
            "dedupe_key": SOURCE_TASK_DEDUPE,
            "stdout_sha256": terminal["stdout_sha256"],
        }
    if not source_row:
        raise CorrectedThermalError("nonthermal source row is empty")
    if any(_thermal_field(str(key)) for key in source_row):
        raise CorrectedThermalError("nonthermal source retained a thermal field")

    receipt = marker["corrected_receipt"]
    temperatures = _mapping(receipt.get("temperatures"), "corrected temperatures")
    constraints = _mapping(
        receipt.get("constraint_observation"), "corrected constraints"
    )
    required_temperatures = {
        "T_max_Tx",
        "T_max_Rx_main",
        "T_max_Rx_side",
        "T_max_core",
    }
    if not required_temperatures.issubset(temperatures):
        raise CorrectedThermalError("corrected temperatures are incomplete")
    thermal_row: dict[str, Any] = {}
    for field, raw in temperatures.items():
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CorrectedThermalError(
                f"corrected thermal field is nonnumeric: {field}"
            ) from exc
        if not math.isfinite(value):
            raise CorrectedThermalError(
                f"corrected thermal field is nonfinite: {field}"
            )
        thermal_row[str(field)] = value
    if (
        constraints.get("winding_limit_c") != 100.0
        or constraints.get("core_limit_c") != 120.0
        or constraints.get("diagnostic_only") is not True
    ):
        raise CorrectedThermalError("corrected thermal constraint policy drifted")
    thermal_row.update(
        {
            "thermal_corrected_checkpoint_continuation": 1,
            "thermal_corrected_winding_pass": int(
                constraints.get("winding_pass") is True
            ),
            "thermal_corrected_core_pass": int(
                constraints.get("core_pass") is True
            ),
            "thermal_corrected_all_constraints_pass": int(
                constraints.get("all_temperature_constraints_pass") is True
            ),
            "thermal_corrected_winding_max_c": float(
                constraints["winding_max_c"]
            ),
            "thermal_corrected_core_max_c": float(constraints["core_max_c"]),
        }
    )
    overlap = set(source_row) & set(thermal_row)
    if overlap:
        raise CorrectedThermalError(
            f"nonthermal/thermal field provenance overlaps: {sorted(overlap)}"
        )
    merged = {**source_row, **thermal_row}
    field_sources = {
        **{
            field: source_kind
            for field in sorted(source_row)
        },
        **{
            field: "corrected_thermal_checkpoint_continuation"
            for field in sorted(thermal_row)
        },
    }
    return sealed(
        {
            "schema": COLLECTION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "candidate_sha256": CANDIDATE_SHA256,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "truth_dataset_ingestion_allowed": False,
            "corrected_task": {
                "task_id": corrected_task_id,
                "task_name": TASK_NAME,
                "dedupe_key": plan["task_identity"]["dedupe_key"],
                "stdout_sha256": hashlib.sha256(
                    stdout.encode("utf-8")
                ).hexdigest(),
                "retention_marker_payload_sha256": marker["payload_sha256"],
            },
            "nonthermal_source": source_record,
            "source_solver_revision": SOURCE_SOLVER_REVISION,
            "executor_solver_revision": plan["executor"]["revision"],
            "pyaedt_library_revision": LIBRARY_REVISION,
            "source_and_executor_revisions_are_separate": True,
            "field_sources": field_sources,
            "merged_row": merged,
        },
        digest_field="payload_sha256",
    )


def _storage_probe_from_args(args: argparse.Namespace) -> Callable[[], dict]:
    if args.storage_snapshot is not None:
        if bool(getattr(args, "apply", False)):
            raise CorrectedThermalError(
                "--apply requires a fresh SSH GPFS quota probe"
            )
        snapshot_path = args.storage_snapshot
        return lambda: _read_json(snapshot_path, "fresh GPFS quota snapshot")
    required = {
        "--ssh-host": args.ssh_host,
        "--ssh-private-key": args.ssh_private_key,
        "--known-hosts": args.known_hosts,
    }
    missing = [name for name, value in required.items() if value in {None, ""}]
    if missing:
        raise CorrectedThermalError(
            "fresh GPFS gate is missing: " + ", ".join(missing)
        )
    return lambda: probe_gpfs_quota(
        host=args.ssh_host,
        username=ACCOUNT,
        private_key=args.ssh_private_key,
        known_hosts=args.known_hosts,
        port=args.ssh_port,
    )


def _runtime_quota_from_plan_args(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "--ssh-host": args.ssh_host,
        "--ssh-private-key": args.ssh_private_key,
        "--known-hosts": args.known_hosts,
    }
    missing = [name for name, value in required.items() if value in {None, ""}]
    if missing:
        raise CorrectedThermalError(
            "plan requires a fresh login-node GPFS quota probe: "
            + ", ".join(missing)
        )
    return probe_gpfs_quota(
        host=args.ssh_host,
        username=ACCOUNT,
        private_key=args.ssh_private_key,
        known_hosts=args.known_hosts,
        port=args.ssh_port,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--checkpoint-manifest", required=True, type=Path)
    plan.add_argument("--claim-root", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    plan.add_argument("--ssh-host", required=True)
    plan.add_argument("--ssh-port", type=int, default=22)
    plan.add_argument("--ssh-private-key", required=True, type=Path)
    plan.add_argument("--known-hosts", required=True, type=Path)

    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True, type=Path)

    submit = commands.add_parser("submit")
    submit.add_argument("--plan", required=True, type=Path)
    submit.add_argument("--apply", action="store_true")
    submit.add_argument("--receipt", type=Path)
    submit.add_argument("--storage-snapshot", type=Path)
    submit.add_argument("--ssh-host")
    submit.add_argument("--ssh-port", type=int, default=22)
    submit.add_argument("--ssh-private-key", type=Path)
    submit.add_argument("--known-hosts", type=Path)

    retain = commands.add_parser("retain")
    retain.add_argument("--execution-plan", required=True, type=Path)
    retain.add_argument("--execution-root", required=True, type=Path)
    retain.add_argument("--execution-exit-code", required=True, type=int)

    source = commands.add_parser("seal-maxwell-source")
    source.add_argument("--plan", required=True, type=Path)
    source.add_argument("--row", required=True, type=Path)
    source.add_argument("--read-only-attestation", required=True, type=Path)
    source.add_argument("--output", required=True, type=Path)

    collect = commands.add_parser("collect")
    collect.add_argument("--plan", required=True, type=Path)
    collect.add_argument("--task-id", required=True, type=int)
    collect.add_argument("--maxwell-source-evidence", type=Path)
    collect.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            path = write_plan(
                checkpoint_manifest=args.checkpoint_manifest,
                claim_root=args.claim_root,
                runtime_quota_snapshot=_runtime_quota_from_plan_args(args),
                output=args.output,
                repo_root=args.repo_root,
            )
            result = {
                "status": "plan_created",
                "path": str(path),
                "scheduler_submission_performed": False,
                "plan": load_plan(path),
            }
        elif args.command == "validate-plan":
            result = {
                "status": "plan_authenticated",
                "scheduler_submission_performed": False,
                "plan": load_plan(args.plan),
            }
        elif args.command == "submit":
            result = submit_plan(
                args.plan,
                apply=args.apply,
                storage_probe=_storage_probe_from_args(args),
                receipt_path=args.receipt,
            )
        elif args.command == "retain":
            result = retain_execution(
                execution_plan=args.execution_plan,
                execution_root=args.execution_root,
                execution_exit_code=args.execution_exit_code,
            )
        elif args.command == "seal-maxwell-source":
            submission_plan = load_plan(args.plan)
            result = create_maxwell_source_evidence(
                plan=submission_plan,
                row=_read_json(args.row, "Maxwell recovered row"),
                read_only_attestation=_read_json(
                    args.read_only_attestation,
                    "Maxwell read-only attestation",
                ),
            )
            _atomic_write_json(args.output, result)
        else:
            submission_plan = load_plan(args.plan)
            evidence = (
                None
                if args.maxwell_source_evidence is None
                else _read_json(
                    args.maxwell_source_evidence,
                    "Maxwell source evidence",
                )
            )
            result = collect_mixed_provenance(
                plan=submission_plan,
                corrected_task_id=args.task_id,
                client=SchedulerHTTP(submission_plan["scheduler"]["url"]),
                maxwell_source_evidence=evidence,
            )
            _atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        CorrectedThermalError,
        atomic_claim.ClaimContractError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            f"CORRECTED_THERMAL_SUBMISSION_ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
