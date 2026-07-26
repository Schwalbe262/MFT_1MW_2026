#!/usr/bin/env python3
"""Execute one authenticated, diagnostic-only Icepak checkpoint continuation.

The input is the immutable static checkpoint produced by
``mft_goal_corrected_thermal_continuation.py``.  This consumer verifies every
manifest-bound byte, clones the checkpoint to a unique writable directory, and
attaches to the already-existing ``icepak_thermal/ThermalSetup``.  It does not
build or edit Maxwell, geometry, materials, boundaries, setup controls, or
mesh.  In particular, the saved premesh is consumed as-is.

The retained result is deliberately diagnostic-only.  Source solver
provenance and continuation-executor provenance remain separate in the
receipt; this tool cannot promote a result to canonical production truth.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_SCHEMA = "mft-corrected-thermal-static-checkpoint-v3"
RECEIPT_SCHEMA = "mft-corrected-thermal-checkpoint-execution-v1"
EXECUTION_PLAN_SCHEMA = "mft-corrected-thermal-execution-plan-v1"
SOURCE_SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
EXECUTOR_REQUIRED_ANCESTOR = "a927ef7ba7d4b5577e47a43377922dacd77b1993"
PYAEDT_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SOURCE_CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
SOURCE_LOGICAL_TASK_ID = 96230
THERMAL_DESIGN = "icepak_thermal"
THERMAL_SETUP = "ThermalSetup"
CORES = 8
TASKS = 1
FAN_VELOCITY_M_S = 1.5
TIM_CONDUCTIVITY_W_MK = 0.2
PAD_THICKNESS_MM = 2.0
SIZE_LIMITS_MM = (1200.0, 1000.0, 750.0)
COPY_CHUNK_BYTES = 8 * 1024 * 1024
GPFS_QUOTA_BINARY = Path("/usr/lpp/mmfs/bin/mmlsquota")
MINIMUM_SCRATCH_WORKING_SHADOW_BYTES = 256 * 1024**3
MAXIMUM_MINIMUM_BUNDLE_BYTES = 4 * 1024**3
MINIMUM_RETAINED_HEADROOM_BYTES = 8 * 1024**3
MINIMUM_RETAINED_INODE_HEADROOM = 4096
PHYSICAL_FREE_RESERVE_BYTES = 50 * 1024**3
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class ContinuationError(RuntimeError):
    """A checkpoint or thermal-only continuation contract failed closed."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish without replacing a concurrently created destination."""
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError:
            raise
        except OSError as exc:
            if _lexists(destination):
                raise FileExistsError(
                    errno.EEXIST, "destination exists", str(destination)
                ) from exc
            raise
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ContinuationError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_noreplace(partial: Path, base: Path) -> Path:
    stamp = int(time.time())
    for counter in range(100_000):
        quarantine = base.with_name(
            f"{base.name}.incomplete.{stamp}.{os.getpid()}.{counter:05d}"
        )
        try:
            _rename_noreplace(partial, quarantine)
            _fsync_directory(base.parent)
            return quarantine
        except FileExistsError:
            continue
    raise ContinuationError(
        f"cannot allocate no-replace quarantine path for {partial}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Match the staging producer's canonical receipt serialization."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(COPY_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ContinuationError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(value.st_mode):
        raise ContinuationError(f"{label} is not a plain regular file: {path}")
    return value


def _plain_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ContinuationError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(value.st_mode):
        raise ContinuationError(f"{label} is not a plain directory: {path}")
    return value


def _exact_hex(value: Any, length: int, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(rf"[0-9a-f]{{{int(length)}}}", text):
        raise ContinuationError(f"{label} is not exact {length}-hex")
    return text


def _relative_file(value: Any) -> str:
    text = str(value or "")
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or any(not part for part in pure.parts)
        or text == ".checkpoint_manifest.json"
    ):
        raise ContinuationError(f"unsafe checkpoint relative path: {text!r}")
    normalized = pure.as_posix()
    if normalized != text:
        raise ContinuationError(f"noncanonical checkpoint path: {text!r}")
    return normalized


def _required_mapping(
    value: Any, label: str, required: Iterable[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationError(f"{label} is not an object")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ContinuationError(f"{label} misses required fields: {missing}")
    return value


def _verify_source_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = _required_mapping(
        manifest.get("source_provenance"),
        "source_provenance",
        {
            "solver_revision",
            "library_revision",
            "candidate_sha256",
            "logical_task_id",
            "execution_task_id",
            "slurm_job_id",
            "allocation_id",
            "node",
            "source_project_sha256",
            "source_static_metadata_sha256",
            "source_plan_identity_sha256",
        },
    )
    observed = {
        "solver_revision": _exact_hex(
            source["solver_revision"], 40, "source solver revision"
        ),
        "library_revision": _exact_hex(
            source["library_revision"], 40, "source library revision"
        ),
        "candidate_sha256": _exact_hex(
            source["candidate_sha256"], 64, "source candidate SHA-256"
        ),
        "logical_task_id": int(source["logical_task_id"]),
        "execution_task_id": int(source["execution_task_id"]),
        "slurm_job_id": int(source["slurm_job_id"]),
        "allocation_id": int(source["allocation_id"]),
        "node": str(source["node"] or "").strip(),
        "source_project_sha256": _exact_hex(
            source["source_project_sha256"], 64, "source project SHA-256"
        ),
        "source_static_metadata_sha256": _exact_hex(
            source["source_static_metadata_sha256"],
            64,
            "source static metadata SHA-256",
        ),
        "source_plan_identity_sha256": _exact_hex(
            source["source_plan_identity_sha256"],
            64,
            "source plan identity SHA-256",
        ),
    }
    if observed["solver_revision"] != SOURCE_SOLVER_REVISION:
        raise ContinuationError("checkpoint source solver revision is not a1e4f")
    if observed["library_revision"] != PYAEDT_LIBRARY_REVISION:
        raise ContinuationError("checkpoint source library revision drifted")
    if observed["candidate_sha256"] != SOURCE_CANDIDATE_SHA256:
        raise ContinuationError("checkpoint source candidate is not b7c")
    if observed["logical_task_id"] != SOURCE_LOGICAL_TASK_ID:
        raise ContinuationError("checkpoint source logical task is not 96230")
    if any(
        observed[name] <= 0
        for name in ("execution_task_id", "slurm_job_id", "allocation_id")
    ):
        raise ContinuationError("checkpoint source scheduler identity is invalid")
    if not re.fullmatch(r"n[0-9A-Za-z_-]+", observed["node"]):
        raise ContinuationError("checkpoint source node identity is invalid")
    return observed


def _verify_quota_before_login_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _required_mapping(
        manifest.get("quota_before_login_evidence"),
        "quota_before_login_evidence",
        {
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
        },
    )
    exact_fields = {
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
    if set(evidence) != exact_fields:
        raise ContinuationError("quota-before-login evidence fields mismatch")
    if (
        evidence["filesystem"] != "gpfs"
        or evidence["quota_type"] != "USR"
        or evidence["source"] != "gate2:mmlsquota-Y"
    ):
        raise ContinuationError("quota-before-login evidence authority mismatch")
    integer_fields = {
        "uid",
        "usage_bytes",
        "soft_limit_bytes",
        "hard_limit_bytes",
        "in_doubt_bytes",
        "files_used",
        "files_soft_limit",
        "files_hard_limit",
        "files_in_doubt",
    }
    for name in integer_fields:
        value = evidence[name]
        if type(value) is not int or value < 0:
            raise ContinuationError(
                f"quota-before-login field is not a nonnegative integer: {name}"
            )
    getuid = getattr(os, "getuid", None)
    if getuid is not None and int(evidence["uid"]) != int(getuid()):
        raise ContinuationError("quota-before-login UID mismatch")
    for name in ("observed_at_epoch", "age_seconds_at_validation"):
        value = evidence[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContinuationError(
                f"quota-before-login field is not finite: {name}"
            )
    if float(evidence["observed_at_epoch"]) < 0:
        raise ContinuationError("quota-before-login observation time is negative")
    age = float(evidence["age_seconds_at_validation"])
    if age < -5.0 or age > 120.0:
        raise ContinuationError(
            "quota-before-login validation age is outside the staging contract"
        )
    claimed = _exact_hex(
        evidence["canonical_sha256"],
        64,
        "quota-before-login canonical SHA-256",
    )
    canonical_payload = dict(evidence)
    canonical_payload.pop("canonical_sha256")
    canonical_payload.pop("age_seconds_at_validation")
    actual = hashlib.sha256(canonical_json_bytes(canonical_payload)).hexdigest()
    if actual != claimed:
        raise ContinuationError(
            "quota-before-login canonical SHA-256 mismatch"
        )
    return dict(evidence)


def _verify_filesystem_evidence(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, int | bool]]:
    required_fields = {
        "anchor_device",
        "bavail",
        "block_size",
        "free_bytes",
        "fsid",
        "readonly",
        "required_free_bytes",
    }

    def validate(name: str) -> dict[str, int | bool]:
        value = _required_mapping(
            manifest.get(name), name, required_fields
        )
        if set(value) != required_fields:
            raise ContinuationError(f"{name} fields mismatch")
        if value["readonly"] is not False:
            raise ContinuationError(f"{name} reports a read-only filesystem")
        normalized: dict[str, int | bool] = {"readonly": False}
        for field in required_fields - {"readonly"}:
            raw = value[field]
            if type(raw) is not int or raw < 0:
                raise ContinuationError(
                    f"{name}.{field} is not a nonnegative exact integer"
                )
            normalized[field] = raw
        if (
            int(normalized["free_bytes"])
            != int(normalized["bavail"]) * int(normalized["block_size"])
        ):
            raise ContinuationError(f"{name} free-byte accounting mismatch")
        if int(normalized["free_bytes"]) < int(
            normalized["required_free_bytes"]
        ):
            raise ContinuationError(f"{name} violates its free-space contract")
        return normalized

    before = validate("filesystem_before")
    after = validate("filesystem_after_copy")
    if (
        before["anchor_device"] != after["anchor_device"]
        or before["fsid"] != after["fsid"]
    ):
        raise ContinuationError("checkpoint destination filesystem identity drifted")
    if int(after["required_free_bytes"]) != PHYSICAL_FREE_RESERVE_BYTES:
        raise ContinuationError(
            "post-copy physical free reserve is not the fixed 50 GiB contract"
        )
    source_before = _required_mapping(
        manifest.get("source_snapshot_before"),
        "source_snapshot_before",
        {"required_budget_bytes"},
    )
    budget = source_before["required_budget_bytes"]
    if type(budget) is not int or budget < 0:
        raise ContinuationError("source snapshot budget is invalid")
    if (
        int(before["required_free_bytes"])
        != budget + PHYSICAL_FREE_RESERVE_BYTES
    ):
        raise ContinuationError(
            "pre-copy physical free contract is not budget plus 50 GiB"
        )
    return {"before": before, "after_copy": after}


def authenticate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Authenticate the complete immutable checkpoint tree and provenance."""
    root = checkpoint.absolute()
    _plain_directory(root, "checkpoint root")
    manifest_path = root / ".checkpoint_manifest.json"
    _regular_file(manifest_path, "checkpoint manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError("checkpoint manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ContinuationError("checkpoint manifest root is not an object")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ContinuationError("checkpoint schema version mismatch")
    if manifest.get("diagnostic_only") is not True:
        raise ContinuationError("checkpoint is not diagnostic-only")
    if manifest.get("canonical") is not False:
        raise ContinuationError("checkpoint must explicitly reject canonical use")
    if manifest.get("composite_reauthentication_required") is not True:
        raise ContinuationError(
            "checkpoint does not require composite reauthentication"
        )
    if manifest.get("post_login_quota_reauthentication_required") is not True:
        raise ContinuationError(
            "checkpoint does not require post-login quota reauthentication"
        )

    payload_hash = _exact_hex(
        manifest.get("manifest_payload_sha256"),
        64,
        "checkpoint manifest payload SHA-256",
    )
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != payload_hash:
        raise ContinuationError("checkpoint receipt payload digest mismatch")

    physics = _required_mapping(
        manifest.get("physics_boundary"),
        "physics_boundary",
        {
            "fan_velocity_m_per_s",
            "tim_conductivity_w_per_mk",
            "thermal_pad_thickness_mm",
        },
    )
    exact_physics = {
        "fan_velocity_m_per_s": FAN_VELOCITY_M_S,
        "tim_conductivity_w_per_mk": TIM_CONDUCTIVITY_W_MK,
        "thermal_pad_thickness_mm": PAD_THICKNESS_MM,
    }
    for name, expected in exact_physics.items():
        try:
            actual = float(physics[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContinuationError(f"invalid physics boundary {name}") from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ContinuationError(
                f"fixed physics boundary drifted: {name}={actual} != {expected}"
            )

    source_provenance = _verify_source_provenance(manifest)
    quota_before_login = _verify_quota_before_login_evidence(manifest)
    filesystem_evidence = _verify_filesystem_evidence(manifest)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContinuationError("checkpoint file inventory is empty")
    inventory: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(files):
        item = _required_mapping(
            raw,
            f"files[{index}]",
            {"path", "size", "mtime_ns", "mode", "sha256"},
        )
        relative = _relative_file(item["path"])
        if relative in names:
            raise ContinuationError(f"duplicate checkpoint file: {relative}")
        names.add(relative)
        try:
            size = int(item["size"])
            mtime_ns = int(item["mtime_ns"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContinuationError(
                f"invalid checkpoint metadata: {relative}"
            ) from exc
        if size < 0 or mtime_ns <= 0:
            raise ContinuationError(
                f"invalid checkpoint size/mtime: {relative}"
            )
        expected_sha = _exact_hex(
            item["sha256"], 64, f"checkpoint SHA-256 {relative}"
        )
        path = root.joinpath(*PurePosixPath(relative).parts)
        metadata = _regular_file(path, f"checkpoint file {relative}")
        if metadata.st_size != size or metadata.st_mtime_ns != mtime_ns:
            raise ContinuationError(
                f"checkpoint file metadata mismatch: {relative}"
            )
        if sha256_file(path) != expected_sha:
            raise ContinuationError(f"checkpoint file hash mismatch: {relative}")
        inventory.append(
            {
                "path": relative,
                "size": size,
                "mtime_ns": mtime_ns,
                "mode": str(item["mode"]),
                "sha256": expected_sha,
            }
        )

    actual_names = set()
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dir_names:
            _plain_directory(directory_path / name, "checkpoint subdirectory")
        for name in file_names:
            path = directory_path / name
            _regular_file(path, "checkpoint tree entry")
            actual_names.add(path.relative_to(root).as_posix())
    expected_names = names | {".checkpoint_manifest.json"}
    if actual_names != expected_names:
        raise ContinuationError(
            "checkpoint tree does not match its complete inventory: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    aedt = [item for item in inventory if item["path"].endswith(".aedt")]
    if len(aedt) != 1:
        raise ContinuationError(
            f"checkpoint must contain exactly one AEDT file, found {len(aedt)}"
        )
    premesh_grid_files = [
        item
        for item in inventory
        if item["path"].endswith("/grid_mapping")
        or item["path"].endswith("/grid_output")
    ]
    if len(premesh_grid_files) != 20:
        raise ContinuationError(
            "checkpoint does not contain the complete 20-file saved premesh: "
            f"found={len(premesh_grid_files)}"
        )
    aedt_relative = PurePosixPath(aedt[0]["path"])
    if len(aedt_relative.parts) != 1:
        raise ContinuationError("checkpoint AEDT must be a direct child")
    project_stem = aedt_relative.name[: -len(".aedt")]
    results_root = f"{project_stem}.aedtresults"
    thermal_results = f"{results_root}/icepak_thermal.results"
    exact_source_files = {
        aedt_relative.as_posix(),
        f"{results_root}/ManagedFiles_Design7.asol",
        f"{results_root}/icepak_thermal.asol",
        f"{thermal_results}/DV274_S271_V0.profile",
        f"{thermal_results}/DV274_S271_V275.profile",
    }
    for family, suffix in (
        ("DV274_Meshes", "_V213.sd"),
        ("DV274_S271_Meshes", "_V0.sd"),
    ):
        for index in (0, 9, 10, 11, 12):
            directory = f"{thermal_results}/{family}{index}{suffix}"
            exact_source_files.update(
                {
                    f"{directory}/grid_mapping",
                    f"{directory}/grid_output",
                }
            )
    if names != exact_source_files:
        raise ContinuationError(
            "checkpoint file naming is not the exact saved-premesh allowlist: "
            f"missing={sorted(exact_source_files - names)}, "
            f"extra={sorted(names - exact_source_files)}"
        )
    if aedt[0]["sha256"] != source_provenance["source_project_sha256"]:
        raise ContinuationError(
            "source project SHA-256 is not bound to the AEDT inventory entry"
        )
    before = _required_mapping(
        manifest.get("source_snapshot_before"),
        "source_snapshot_before",
        {"metadata_sha256"},
    )
    if (
        _exact_hex(
            before["metadata_sha256"], 64, "source snapshot metadata SHA-256"
        )
        != source_provenance["source_static_metadata_sha256"]
    ):
        raise ContinuationError(
            "source static metadata provenance does not match snapshot"
        )
    after = _required_mapping(
        manifest.get("source_snapshot_after"),
        "source_snapshot_after",
        {"metadata_sha256"},
    )
    if (
        _exact_hex(
            after["metadata_sha256"], 64, "post-copy source metadata SHA-256"
        )
        != source_provenance["source_static_metadata_sha256"]
    ):
        raise ContinuationError(
            "post-copy source metadata provenance does not match admission"
        )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "files": inventory,
        "aedt_relative_path": aedt_relative.as_posix(),
        "source_provenance": source_provenance,
        "physics_boundary": exact_physics,
        "quota_before_login_evidence": quota_before_login,
        "filesystem_evidence": filesystem_evidence,
    }


def authenticate_execution_plan(
    plan_path: Path,
    checkpoint_authentication: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a source-only checkpoint to one exact clean executor submission."""
    resolved = plan_path.resolve(strict=True)
    _regular_file(resolved, "corrected thermal execution plan")
    try:
        plan = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(
            "corrected thermal execution plan is not valid JSON"
        ) from exc
    if not isinstance(plan, dict):
        raise ContinuationError("corrected thermal execution plan is not an object")
    if plan.get("schema") != EXECUTION_PLAN_SCHEMA:
        raise ContinuationError("corrected thermal execution plan schema mismatch")
    if plan.get("diagnostic_only") is not True or plan.get("canonical") is not False:
        raise ContinuationError(
            "corrected thermal execution plan is not diagnostic-only"
        )
    payload_hash = _exact_hex(
        plan.get("plan_payload_sha256"),
        64,
        "corrected thermal execution plan payload SHA-256",
    )
    unsigned = dict(plan)
    unsigned.pop("plan_payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != payload_hash:
        raise ContinuationError("corrected thermal execution plan digest mismatch")
    checkpoint_hash = _exact_hex(
        plan.get("checkpoint_manifest_sha256"),
        64,
        "execution plan checkpoint manifest SHA-256",
    )
    if checkpoint_hash != checkpoint_authentication["manifest_sha256"]:
        raise ContinuationError(
            "execution plan is bound to a different checkpoint manifest"
        )
    executor_revision = _exact_hex(
        plan.get("executor_revision"), 40, "execution plan executor revision"
    )
    required_ancestor = _exact_hex(
        plan.get("required_executor_ancestor"),
        40,
        "execution plan required executor ancestor",
    )
    if required_ancestor != EXECUTOR_REQUIRED_ANCESTOR:
        raise ContinuationError("execution plan does not require the a927 fix")
    tool_hash = _exact_hex(
        plan.get("tool_payload_sha256"),
        64,
        "execution plan tool payload SHA-256",
    )
    if tool_hash != sha256_file(Path(__file__).resolve(strict=True)):
        raise ContinuationError("execution plan tool payload SHA-256 mismatch")
    expected_dispatch = {
        "cores": CORES,
        "tasks": TASKS,
        "use_auto_settings": False,
    }
    dispatch = _required_mapping(
        plan.get("dispatch"), "execution plan dispatch", expected_dispatch
    )
    for name, expected in expected_dispatch.items():
        if dispatch[name] != expected:
            raise ContinuationError(
                f"execution plan dispatch drifted: {name}={dispatch[name]!r}"
            )
    storage = _required_mapping(
        plan.get("output_storage"),
        "execution plan output storage",
        {
            "mode",
            "scratch_root",
            "minimum_scratch_working_shadow_bytes",
            "retained_filesystem",
            "retained_root",
            "maximum_minimum_bundle_bytes",
            "minimum_retained_headroom_after_bytes",
            "minimum_retained_inode_headroom_after",
        },
    )
    storage_contract = {
        "mode": str(storage["mode"] or "").strip(),
        "scratch_root": str(storage["scratch_root"] or "").strip(),
        "minimum_scratch_working_shadow_bytes": int(
            storage["minimum_scratch_working_shadow_bytes"]
        ),
        "retained_filesystem": str(
            storage["retained_filesystem"] or ""
        ).strip(),
        "retained_root": str(storage["retained_root"] or "").strip(),
        "maximum_minimum_bundle_bytes": int(
            storage["maximum_minimum_bundle_bytes"]
        ),
        "minimum_retained_headroom_after_bytes": int(
            storage["minimum_retained_headroom_after_bytes"]
        ),
        "minimum_retained_inode_headroom_after": int(
            storage["minimum_retained_inode_headroom_after"]
        ),
    }
    if (
        storage_contract["mode"]
        != "node_local_scratch_with_gpfs_minimum_retention"
        or not os.path.isabs(storage_contract["scratch_root"])
        or storage_contract["minimum_scratch_working_shadow_bytes"]
        < MINIMUM_SCRATCH_WORKING_SHADOW_BYTES
        or storage_contract["retained_filesystem"] != "gpfs"
        or not os.path.isabs(storage_contract["retained_root"])
        or storage_contract["maximum_minimum_bundle_bytes"]
        < MAXIMUM_MINIMUM_BUNDLE_BYTES
        or storage_contract["minimum_retained_headroom_after_bytes"]
        < MINIMUM_RETAINED_HEADROOM_BYTES
        or storage_contract["minimum_retained_inode_headroom_after"]
        < MINIMUM_RETAINED_INODE_HEADROOM
    ):
        raise ContinuationError(
            f"execution plan retained-output budget is insufficient: "
            f"{storage_contract}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "checkpoint_manifest_sha256": checkpoint_hash,
        "executor_revision": executor_revision,
        "required_executor_ancestor": required_ancestor,
        "tool_payload_sha256": tool_hash,
        "dispatch": expected_dispatch,
        "output_storage": storage_contract,
        "plan_payload_sha256": payload_hash,
    }


def _copy_verified_file(source: Path, destination: Path, item: Mapping[str, Any]) -> None:
    before = _regular_file(source, f"checkpoint clone source {item['path']}")
    if (
        before.st_size != int(item["size"])
        or before.st_mtime_ns != int(item["mtime_ns"])
    ):
        raise ContinuationError(f"checkpoint changed before clone: {item['path']}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb", buffering=0) as input_stream:
        with destination.open("xb", buffering=0) as output_stream:
            shutil.copyfileobj(
                input_stream, output_stream, length=COPY_CHUNK_BYTES
            )
            output_stream.flush()
            os.fsync(output_stream.fileno())
    after = _regular_file(source, f"checkpoint clone source {item['path']}")
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ContinuationError(f"checkpoint changed during clone: {item['path']}")
    if (
        destination.stat().st_size != int(item["size"])
        or sha256_file(destination) != item["sha256"]
    ):
        raise ContinuationError(f"checkpoint clone hash mismatch: {item['path']}")
    try:
        os.utime(
            destination,
            ns=(int(item["mtime_ns"]), int(item["mtime_ns"])),
            follow_symlinks=False,
        )
    except NotImplementedError:
        # Windows contract tests have no symlink-following utime support.  The
        # destination was created exclusively above, so this fallback cannot
        # follow an attacker-controlled path.
        os.utime(
            destination,
            ns=(int(item["mtime_ns"]), int(item["mtime_ns"])),
        )
    try:
        os.chmod(destination, 0o600, follow_symlinks=False)
    except NotImplementedError:
        os.chmod(destination, 0o600)


def clone_checkpoint(
    authentication: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Create a unique writable clone; never mutate or delete the checkpoint."""
    checkpoint = Path(authentication["root"])
    root = output_root.absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _plain_directory(root, "continuation output root")
    label = (
        f"ct_b7c_{time.strftime('%m%dT%H%M%SZ', time.gmtime())}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    destination = root / label
    incoming = root / f".{label}.incoming"
    if _lexists(destination) or _lexists(incoming):
        raise ContinuationError("unique continuation clone path already exists")
    incoming.mkdir(mode=0o700)
    published = False
    try:
        for item in authentication["files"]:
            relative = PurePosixPath(item["path"])
            _copy_verified_file(
                checkpoint.joinpath(*relative.parts),
                incoming.joinpath(*relative.parts),
                item,
            )
        aedt = incoming.joinpath(
            *PurePosixPath(authentication["aedt_relative_path"]).parts
        )
        _fsync_directory(incoming)
        _rename_noreplace(incoming, destination)
        published = True
        _fsync_directory(root)
        aedt = destination / aedt.relative_to(incoming)
        return {
            "clone_root": destination,
            "aedt_path": aedt,
            "results_path": aedt.with_suffix(".aedtresults"),
            "checkpoint_unchanged": True,
            "file_count": len(authentication["files"]),
        }
    except BaseException:
        # Retain partial bytes for diagnosis.  A continuation tool never
        # recursively deletes evidence or an input checkpoint.
        partial = destination if published else incoming
        if _lexists(partial):
            _quarantine_noreplace(partial, destination)
        raise


def _quota_snapshot(filesystem: str) -> dict[str, Any]:
    if not GPFS_QUOTA_BINARY.is_file():
        raise ContinuationError(
            f"required GPFS quota binary is unavailable: {GPFS_QUOTA_BINARY}"
        )
    process = subprocess.run(
        [
            str(GPFS_QUOTA_BINARY),
            "-u",
            str(os.getuid()),
            "-Y",
            filesystem,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise ContinuationError(
            f"mmlsquota failed rc={process.returncode}: "
            f"{process.stderr[-1000:]}"
        )
    rows = [
        row
        for row in process.stdout.splitlines()
        if row.startswith("mmlsquota:user:0:")
    ]
    if len(rows) != 1:
        raise ContinuationError(
            f"mmlsquota returned {len(rows)} user rows"
        )
    values = rows[0].split(":")
    if len(values) < 20:
        raise ContinuationError(f"unexpected mmlsquota row: {rows[0]}")
    snapshot = {
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
    if snapshot["uid"] != os.getuid() or snapshot["quota_type"] != "USR":
        raise ContinuationError(
            f"GPFS quota identity mismatch: {snapshot}"
        )
    return snapshot


def admit_retained_output_storage(
    authentication: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Admit node-local solve scratch and the bounded GPFS minimum package."""
    storage = execution_plan["output_storage"]
    planned_scratch = Path(storage["scratch_root"]).absolute()
    actual_scratch = output_root.absolute()
    if planned_scratch != actual_scratch:
        raise ContinuationError(
            f"execution plan scratch root mismatch: "
            f"{planned_scratch} != {actual_scratch}"
        )
    scratch_parent = actual_scratch if actual_scratch.exists() else actual_scratch.parent
    _plain_directory(scratch_parent, "scratch root or parent")
    checkpoint_root = Path(authentication["root"]).resolve(strict=True)
    if (
        os.name != "nt"
        and scratch_parent.resolve(strict=True).stat().st_dev
        == checkpoint_root.stat().st_dev
    ):
        raise ContinuationError(
            "thermal solve scratch must not share the GPFS checkpoint device"
        )
    clone_bytes = sum(int(item["size"]) for item in authentication["files"])
    scratch_required = (
        clone_bytes + int(storage["minimum_scratch_working_shadow_bytes"])
    )
    scratch_free = int(shutil.disk_usage(scratch_parent).free)
    if scratch_free < scratch_required:
        raise ContinuationError(
            f"node-local scratch is insufficient: "
            f"{scratch_free} < {scratch_required}"
        )

    retained_root = Path(storage["retained_root"]).absolute()
    retained_parent = (
        retained_root if retained_root.exists() else retained_root.parent
    )
    _plain_directory(retained_parent, "retained root or parent")
    if (
        os.name != "nt"
        and retained_parent.resolve(strict=True).stat().st_dev
        != checkpoint_root.stat().st_dev
    ):
        raise ContinuationError(
            "minimum retention root is not on the checkpoint GPFS device"
        )
    quota = _quota_snapshot(storage["retained_filesystem"])
    budget_bytes = int(storage["maximum_minimum_bundle_bytes"])
    clone_inodes = 32
    usage = int(quota["usage_bytes"]) + int(quota["in_doubt_bytes"])
    files_used = int(quota["files_used"]) + int(quota["files_in_doubt"])
    shadow = {
        "soft_headroom_after_bytes": (
            int(quota["soft_limit_bytes"]) - usage - budget_bytes
        ),
        "hard_headroom_after_bytes": (
            int(quota["hard_limit_bytes"]) - usage - budget_bytes
        ),
        "soft_inode_headroom_after": (
            int(quota["files_soft_limit"]) - files_used - clone_inodes
        ),
        "hard_inode_headroom_after": (
            int(quota["files_hard_limit"]) - files_used - clone_inodes
        ),
    }
    minimum_bytes = int(storage["minimum_retained_headroom_after_bytes"])
    minimum_inodes = int(storage["minimum_retained_inode_headroom_after"])
    if (
        shadow["soft_headroom_after_bytes"] < minimum_bytes
        or shadow["hard_headroom_after_bytes"] < minimum_bytes
        or shadow["soft_inode_headroom_after"] < minimum_inodes
        or shadow["hard_inode_headroom_after"] < minimum_inodes
    ):
        raise ContinuationError(
            f"retained-output quota admission failed: shadow={shadow}, "
            f"minimum_bytes={minimum_bytes}, "
            f"minimum_inodes={minimum_inodes}"
        )
    return {
        "schema": "mft-corrected-thermal-retained-output-admission-v1",
        "scratch_root": str(actual_scratch),
        "scratch_free_bytes": scratch_free,
        "scratch_required_bytes": scratch_required,
        "quota_before": quota,
        "clone_logical_bytes": clone_bytes,
        "minimum_bundle_budget_bytes": budget_bytes,
        "retained_root": str(retained_root),
        "shadow": shadow,
        "passed": True,
    }


def attest_saved_premesh(
    authentication: Mapping[str, Any], clone_root: Path
) -> dict[str, Any]:
    grid = [
        item
        for item in authentication["files"]
        if item["path"].endswith("/grid_mapping")
        or item["path"].endswith("/grid_output")
    ]
    rows = []
    for item in grid:
        path = clone_root.joinpath(*PurePosixPath(item["path"]).parts)
        metadata = _regular_file(path, f"cloned premesh {item['path']}")
        actual_sha = sha256_file(path)
        if (
            metadata.st_size != int(item["size"])
            or actual_sha != item["sha256"]
        ):
            raise ContinuationError(
                f"saved premesh changed before dispatch: {item['path']}"
            )
        rows.append(
            {
                "path": item["path"],
                "size": int(item["size"]),
                "sha256": actual_sha,
            }
        )
    if len(rows) != 20:
        raise ContinuationError(
            f"cloned premesh attestation count mismatch: {len(rows)}"
        )
    return {
        "schema": "mft-corrected-thermal-saved-premesh-readback-v1",
        "file_count": len(rows),
        "files": rows,
        "regenerated": False,
        "passed": True,
    }


def snapshot_profiles(results_path: Path) -> dict[str, tuple[int, int, str]]:
    design_results = results_path / f"{THERMAL_DESIGN}.results"
    snapshot = {}
    for path in sorted(design_results.glob("*.profile")):
        value = _regular_file(path, "Icepak profile")
        snapshot[str(path.resolve(strict=True))] = (
            int(value.st_size),
            int(value.st_mtime_ns),
            sha256_file(path),
        )
    return snapshot


def fresh_corrected_profile(
    results_path: Path,
    before: Mapping[str, tuple[int, int, str]],
) -> Path:
    after = snapshot_profiles(results_path)
    changed = [
        Path(path)
        for path, signature in after.items()
        if before.get(path) != signature
    ]
    if not changed:
        raise ContinuationError(
            "corrected thermal solve produced no fresh profile artifact"
        )
    return max(
        changed,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _git_identity(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip().lower()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "run_simulation_260706.py",
            "module",
        ],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    return {"root": str(resolved), "revision": revision, "solver_dirty": bool(dirty)}


_NUMBER_WITH_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([A-Za-z_/()]*)\s*$"
)


def _quantity(value: Any, expected_unit: str, label: str) -> float:
    match = _NUMBER_WITH_UNIT.fullmatch(str(value))
    if match is None:
        raise ContinuationError(f"cannot parse native {label}: {value!r}")
    number = float(match.group(1))
    unit = match.group(2).casefold()
    aliases = {
        "mm": {"", "mm"},
        "m_per_sec": {"m_per_sec", "m/s", "mpersec"},
    }
    if unit not in aliases[expected_unit]:
        raise ContinuationError(
            f"native {label} unit mismatch: {unit!r} != {expected_unit!r}"
        )
    if not math.isfinite(number):
        raise ContinuationError(f"native {label} is nonfinite")
    return number


def _native_design_variable(native_design: Any, name: str, unit: str) -> float:
    getter = getattr(native_design, "GetVariableValue", None)
    if not callable(getter):
        raise ContinuationError("native design variable readback is unavailable")
    return _quantity(getter(name), unit, name)


def _native_boundary_inventory(ipk: Any) -> list[dict[str, Any]]:
    inventory = []
    for boundary in list(getattr(ipk, "boundaries", ()) or ()):
        child = getattr(boundary, "_child_object", None)
        if child is None:
            raise ContinuationError(
                f"native boundary child is unavailable: {getattr(boundary, 'name', '')}"
            )
        names = sorted(str(name) for name in (child.GetPropNames() or ()))
        props = {}
        for name in names:
            try:
                value = child.GetPropValue(name)
            except Exception:
                # Some AEDT child properties are action-like or version-only
                # and cannot be read.  Required fan/loss properties remain
                # fail-closed in the dedicated attestations below.
                continue
            if isinstance(value, (list, tuple)):
                props[name] = [str(item) for item in value]
            else:
                props[name] = str(value)
        inventory.append(
            {
                "name": str(getattr(boundary, "name", "") or ""),
                "properties": props,
            }
        )
    if not inventory:
        raise ContinuationError("native boundary inventory is empty")
    return inventory


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _flatten_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _flatten_strings(nested)
    else:
        yield str(value)


def _attest_fan(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    fan = [
        item
        for item in boundaries
        if str(item["name"]).startswith("fan_inlet")
    ]
    if not fan:
        raise ContinuationError("native fan inlet boundary is missing")
    velocity_values = []
    for item in fan:
        for name, value in item["properties"].items():
            if "velocity" not in name.casefold():
                continue
            for text in _flatten_strings(value):
                for number in re.findall(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?",
                    text,
                ):
                    velocity_values.append(abs(float(number)))
    if not velocity_values or not any(
        math.isclose(
            value, FAN_VELOCITY_M_S, rel_tol=0.0, abs_tol=1e-12
        )
        for value in velocity_values
    ):
        raise ContinuationError(
            f"native fan boundary has no {FAN_VELOCITY_M_S}m/s readback"
        )
    if any(
        value not in {0.0, FAN_VELOCITY_M_S} for value in velocity_values
    ):
        raise ContinuationError(
            f"native fan boundary velocity drifted: {velocity_values}"
        )
    return {
        "names": [item["name"] for item in fan],
        "velocity_magnitudes_m_per_s": velocity_values,
        "passed": True,
    }


def _native_loss_assignments(
    boundaries: list[dict[str, Any]],
    editor: Any = None,
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    assigned_objects: set[str] = set()
    for item in boundaries:
        props = item["properties"]
        total_name = next(
            (name for name in ("Total Power",) if name in props),
            None,
        )
        assignment_name = next(
            (name for name in ("Objects", "Assignment", "Parts") if name in props),
            None,
        )
        if total_name is None or assignment_name is None:
            continue
        power_text = str(props[total_name])
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*"
            r"(mW|W|kW)?\s*",
            power_text,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ContinuationError(
                f"native block power is invalid: {power_text!r}"
            )
        factor = {"": 1.0, "mw": 1e-3, "w": 1.0, "kw": 1e3}[
            str(match.group(2) or "").casefold()
        ]
        power_w = float(match.group(1)) * factor
        if not math.isfinite(power_w) or power_w < 0:
            raise ContinuationError("native block power is nonfinite or negative")
        raw_objects = props[assignment_name]
        values = raw_objects if isinstance(raw_objects, list) else [raw_objects]
        objects = []
        for raw in values:
            objects.extend(
                value.strip().strip("'\"")
                for value in re.split(r"\s*[,;]\s*", str(raw).strip("[]()"))
                if value.strip().strip("'\"")
            )
        get_object_name = getattr(editor, "GetObjectNameByID", None)
        if callable(get_object_name):
            resolved_objects = []
            for name in objects:
                try:
                    resolved_objects.append(str(get_object_name(int(name))))
                except (TypeError, ValueError, OverflowError):
                    resolved_objects.append(name)
            objects = resolved_objects
        if not objects:
            raise ContinuationError("native block has no assigned object")
        overlap = assigned_objects.intersection(objects)
        if overlap:
            raise ContinuationError(
                f"native loss assignments overlap: {sorted(overlap)}"
            )
        assigned_objects.update(objects)
        assignments.append(
            {
                "boundary": item["name"],
                "objects": objects,
                "power_w": power_w,
            }
        )
    if not assignments:
        raise ContinuationError("native solid-block loss assignments are missing")
    categories = {
        "winding": any(
            re.match(r"^(?:Tx|Rx)_", name)
            for name in assigned_objects
        ),
        "core": any(name.startswith("core_") for name in assigned_objects),
    }
    if not all(categories.values()):
        raise ContinuationError(
            f"native loss assignment coverage is incomplete: {categories}"
        )
    return {
        "assignment_count": len(assignments),
        "assigned_object_count": len(assigned_objects),
        "total_power_w": sum(item["power_w"] for item in assignments),
        "categories": categories,
        "assignments": assignments,
        "no_overlap": True,
        "passed": True,
    }


def _native_geometry(ipk: Any) -> dict[str, Any]:
    modeler = ipk.modeler
    editor = getattr(ipk, "oeditor", None) or getattr(modeler, "oeditor", None)
    unit = str(getattr(modeler, "model_units", "") or "").casefold()
    factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    if unit not in factors:
        raise ContinuationError(f"unsupported native model unit: {unit!r}")
    get_group = getattr(editor, "GetObjectsInGroup", None)
    get_object_box = getattr(editor, "GetObjectBoundingBox", None)
    if not callable(get_group) or not callable(get_object_box):
        raise ContinuationError(
            "native per-solid bounding-box readback is unavailable"
        )
    solid_names = sorted(
        str(name)
        for name in (get_group("Solids") or ())
        if str(name).casefold() != "region"
    )
    if not solid_names:
        raise ContinuationError("native physical-solid inventory is empty")
    solid_boxes = {}
    for name in solid_names:
        raw = list(get_object_box(name) or ())
        if len(raw) != 6:
            raise ContinuationError(
                f"native solid bounding box is invalid: {name}={raw!r}"
            )
        box = []
        for value in raw:
            try:
                box.append(float(value) * factors[unit])
            except (TypeError, ValueError):
                match = _NUMBER_WITH_UNIT.fullmatch(str(value))
                if match is None:
                    raise ContinuationError(
                        f"cannot parse native solid bounding box: {value!r}"
                    )
                value_unit = match.group(2).casefold()
                if value_unit not in factors:
                    raise ContinuationError(
                        "unsupported native solid bounding-box unit: "
                        f"{value_unit!r}"
                    )
                box.append(float(match.group(1)) * factors[value_unit])
        if not all(math.isfinite(value) for value in box):
            raise ContinuationError(
                f"native solid bounding box is nonfinite: {name}"
            )
        solid_boxes[name] = box
    values = [
        min(box[index] for box in solid_boxes.values())
        for index in range(3)
    ] + [
        max(box[index] for box in solid_boxes.values())
        for index in range(3, 6)
    ]
    dimensions = (
        abs(values[3] - values[0]),
        abs(values[4] - values[1]),
        abs(values[5] - values[2]),
    )
    if any(
        value <= 0 or value > limit + 1e-9
        for value, limit in zip(dimensions, SIZE_LIMITS_MM)
    ):
        raise ContinuationError(
            f"native geometry exceeds size limits: {dimensions}"
        )

    object_names = solid_names
    pad_names = [
        name for name in object_names if "pad" in name.casefold()
    ]
    if not pad_names:
        raise ContinuationError("native thermal pad object inventory is empty")
    get_property = getattr(editor, "GetPropertyValue", None)
    if not callable(get_property):
        raise ContinuationError("native object material readback is unavailable")
    pad_materials = {}
    for name in pad_names:
        material = str(
            get_property("Geometry3DAttributeTab", name, "Material") or ""
        ).strip().strip('"')
        pad_materials[name] = material
        if material.casefold() != "thermal_pad":
            raise ContinuationError(
                f"native pad material drifted: {name}={material!r}"
            )
    return {
        "bounding_box_mm": values,
        "bounding_box_scope": "native_solids_excluding_air_region",
        "dimensions_mm": {
            "width_x": dimensions[0],
            "length_y": dimensions[1],
            "height_z": dimensions[2],
        },
        "limits_mm": {
            "width_x": SIZE_LIMITS_MM[0],
            "length_y": SIZE_LIMITS_MM[1],
            "height_z": SIZE_LIMITS_MM[2],
        },
        "object_count": len(object_names),
        "pad_materials": pad_materials,
        "passed": True,
    }


def attest_native_fixed_model(
    native_design: Any, ipk: Any
) -> dict[str, Any]:
    """Read, but never edit, the fixed geometry/physics/loss contracts."""
    from module.thermal_260706 import _thermal_pad_native_readback

    variables = {
        "fan_velocity": _native_design_variable(
            native_design, "fan_velocity", "m_per_sec"
        ),
        "wcp_pad_t": _native_design_variable(
            native_design, "wcp_pad_t", "mm"
        ),
        "core_plate_pad_t": _native_design_variable(
            native_design, "core_plate_pad_t", "mm"
        ),
    }
    expected = {
        "fan_velocity": FAN_VELOCITY_M_S,
        "wcp_pad_t": PAD_THICKNESS_MM,
        "core_plate_pad_t": PAD_THICKNESS_MM,
    }
    for name, wanted in expected.items():
        if not math.isclose(
            variables[name], wanted, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContinuationError(
                f"native fixed design variable drifted: "
                f"{name}={variables[name]} != {wanted}"
            )
    tim = _thermal_pad_native_readback(ipk.materials)
    if not math.isclose(
        float(tim["thermal_conductivity_W_mK"]),
        TIM_CONDUCTIVITY_W_MK,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContinuationError("native TIM conductivity drifted")
    boundaries = _native_boundary_inventory(ipk)
    editor = getattr(ipk, "oeditor", None) or getattr(
        ipk.modeler, "oeditor", None
    )
    return {
        "schema": "mft-corrected-thermal-native-fixed-readback-v1",
        "design_variables": variables,
        "tim_material": tim,
        "fan_boundary": _attest_fan(boundaries),
        "loss_assignments": _native_loss_assignments(
            boundaries, editor=editor
        ),
        "geometry": _native_geometry(ipk),
        "passed": True,
    }


def _native_design_names(native_project: Any) -> tuple[str, ...]:
    names = []
    for design in tuple(native_project.GetDesigns() or ()):
        getter = getattr(design, "GetName", None)
        value = getter() if callable(getter) else design
        names.append(str(value or "").split(";")[-1].strip())
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ContinuationError("native project has empty or duplicate designs")
    if names.count(THERMAL_DESIGN) != 1:
        raise ContinuationError(
            f"native project has no exact {THERMAL_DESIGN!r} design"
        )
    return tuple(names)


def _attach_existing_thermal(project: Any) -> tuple[Any, Any, Any]:
    native_project = project.project
    before = _native_design_names(native_project)
    native_design = native_project.SetActiveDesign(THERMAL_DESIGN)
    if native_design is None or native_design is False:
        raise ContinuationError("native existing Icepak activation failed")
    if "icepak" not in str(native_design.GetDesignType() or "").casefold():
        raise ContinuationError("existing thermal design is not Icepak")
    analysis = native_design.GetModule("AnalysisSetup")
    setups = tuple(str(value) for value in (analysis.GetSetups() or ()))
    if setups != (THERMAL_SETUP,):
        raise ContinuationError(
            f"existing thermal setup inventory drifted: {setups!r}"
        )

    # pyProject.create_design attaches its wrapper to the active same-name
    # native design in this library.  Bracketing native inventories prove that
    # no InsertDesign or Maxwell design creation occurred.
    wrapper = project.create_design(
        name=THERMAL_DESIGN,
        solver="icepak",
        solution="SteadyState TemperatureAndFlow",
    )
    ipk = wrapper.solver_instance
    after = _native_design_names(native_project)
    if before != after or str(ipk.design_name) != THERMAL_DESIGN:
        raise ContinuationError(
            "PyAEDT attach changed the native design inventory"
        )
    setup = ipk.get_setup(name=THERMAL_SETUP)
    if setup is None or setup is False:
        raise ContinuationError("PyAEDT could not attach existing ThermalSetup")
    return wrapper, ipk, setup, native_design


def _attest_parallel_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if evidence.get("passed") is not True:
        raise ContinuationError("standalone thermal parallel evidence failed")
    policy = _required_mapping(
        evidence.get("policy"),
        "thermal parallel policy",
        {
            "expected_fluent_processes",
            "expected_num_engines",
            "pyaedt_cores_argument",
            "pyaedt_tasks_argument",
            "pyaedt_use_auto_settings_argument",
        },
    )
    expected_policy = {
        "expected_fluent_processes": CORES,
        "expected_num_engines": TASKS,
        "pyaedt_cores_argument": CORES,
        "pyaedt_tasks_argument": TASKS,
        "pyaedt_use_auto_settings_argument": False,
    }
    for name, wanted in expected_policy.items():
        if policy[name] != wanted:
            raise ContinuationError(
                f"thermal parallel policy drifted: {name}={policy[name]!r}"
            )
    attempts = evidence.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ContinuationError("thermal parallel attempt evidence is empty")
    rows = []
    for attempt in attempts:
        row = _required_mapping(
            attempt, "parallel attempt", {"passed", "acf", "process"}
        )
        if row["passed"] is not True:
            raise ContinuationError("thermal parallel attempt did not pass")
        acf = _required_mapping(
            row["acf"],
            "thermal ACF evidence",
            {"passed", "num_cores_readback", "num_engines_readback"},
        )
        process = _required_mapping(
            row["process"],
            "Fluent process evidence",
            {
                "passed",
                "thread_count_readbacks",
                "nprocs_count_readbacks",
                "mismatched_process_counts",
            },
        )
        thread_counts = [int(value) for value in process["thread_count_readbacks"]]
        mpi_counts = [int(value) for value in process["nprocs_count_readbacks"]]
        if (
            acf["passed"] is not True
            or int(acf["num_engines_readback"]) != TASKS
            or int(acf["num_cores_readback"]) != CORES
            or process["passed"] is not True
            or not thread_counts
            or not mpi_counts
            or set(thread_counts) != {CORES}
            or set(mpi_counts) != {CORES}
            or list(process["mismatched_process_counts"])
        ):
            raise ContinuationError(
                "actual Fluent -t8/MPI8 or ACF 1-engine/8-core attestation failed"
            )
        rows.append(
            {
                "acf_num_engines": TASKS,
                "acf_num_cores": CORES,
                "fluent_thread_counts": thread_counts,
                "fluent_mpi_counts": mpi_counts,
                "passed": True,
            }
        )
    return {
        "schema": "mft-corrected-thermal-8way-attestation-v1",
        "attempts": rows,
        "passed": True,
    }


def _runtime_identity(
    library_root: Path, execution_plan: Mapping[str, Any]
) -> dict[str, Any]:
    executor = _git_identity(REPO_ROOT)
    library = _git_identity(library_root)
    lineage = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            EXECUTOR_REQUIRED_ANCESTOR,
            executor["revision"],
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if (
        executor["revision"] != execution_plan["executor_revision"]
        or executor["solver_dirty"]
        or lineage.returncode != 0
    ):
        raise ContinuationError(
            "thermal executor is not the exact clean planned descendant of "
            f"a927: {executor}"
        )
    if (
        library["revision"] != PYAEDT_LIBRARY_REVISION
        or library["solver_dirty"]
    ):
        raise ContinuationError(
            f"PyAEDT library provenance drifted: {library}"
        )
    return {
        "source_solver_revision": SOURCE_SOLVER_REVISION,
        "executor_solver_revision": executor["revision"],
        "executor_required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
        "executor_is_required_descendant": True,
        "pyaedt_library_revision": library["revision"],
        "source_and_executor_revisions_are_separate": True,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    os.replace(temporary, path)


def _tree_logical_size(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dir_names:
            _plain_directory(directory_path / name, "result directory")
        for name in file_names:
            value = _regular_file(
                directory_path / name, "result tree file"
            )
            total += int(value.st_size)
            count += 1
    return total, count


def _fsync_and_protect_descriptor(
    descriptor: int, mode: int, *, protect: bool
) -> None:
    """Persist content first, then protection metadata on the same open FD."""
    os.fsync(descriptor)
    if protect:
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)


def _durably_protect_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _fsync_and_protect_descriptor(
            descriptor, 0o500, protect=True
        )
    finally:
        os.close(descriptor)


def _protect_retained_tree(root: Path) -> dict[str, Any]:
    """Fsync and make a completed retention tree immutable to its owner."""
    expected_files: dict[str, dict[str, Any]] = {}
    expected_directories: set[str] = set()
    directories: list[Path] = []
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories.append(directory_path)
        if directory_path != root:
            expected_directories.add(
                directory_path.relative_to(root).as_posix()
            )
        for name in dir_names:
            _plain_directory(
                directory_path / name, "retained package directory"
            )
        for name in file_names:
            path = directory_path / name
            value = _regular_file(path, "retained package file")
            if value.st_nlink != 1:
                raise ContinuationError(
                    f"retained package file has multiple links: {path}"
                )
            relative = path.relative_to(root).as_posix()
            expected_files[relative] = {
                "size_bytes": int(value.st_size),
                "sha256": sha256_file(path),
            }
            with path.open("r+b", buffering=0) as stream:
                _fsync_and_protect_descriptor(
                    stream.fileno(), 0o400, protect=os.name != "nt"
                )
    for directory_path in reversed(directories):
        _durably_protect_directory(directory_path)
    return {
        "files": expected_files,
        "directories": sorted(expected_directories),
    }


def _verify_retained_tree(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    _plain_directory(root, "published retained package")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path != root:
            actual_directories.add(
                directory_path.relative_to(root).as_posix()
            )
        directory_value = _plain_directory(
            directory_path, "published retained directory"
        )
        if os.name != "nt" and stat.S_IMODE(directory_value.st_mode) != 0o500:
            raise ContinuationError(
                f"published retained directory is not mode 0500: {directory_path}"
            )
        for name in dir_names:
            _plain_directory(
                directory_path / name, "published retained subdirectory"
            )
        for name in file_names:
            path = directory_path / name
            value = _regular_file(path, "published retained file")
            relative = path.relative_to(root).as_posix()
            actual_files.add(relative)
            if relative not in expected["files"]:
                raise ContinuationError(
                    f"unexpected published retained file: {relative}"
                )
            row = expected["files"][relative]
            if (
                value.st_nlink != 1
                or int(value.st_size) != int(row["size_bytes"])
                or sha256_file(path) != row["sha256"]
                or (
                    os.name != "nt"
                    and stat.S_IMODE(value.st_mode) != 0o400
                )
            ):
                raise ContinuationError(
                    f"published retained file verification failed: {relative}"
                )
    if actual_files != set(expected["files"]):
        raise ContinuationError("published retained file inventory drifted")
    if actual_directories != set(expected["directories"]):
        raise ContinuationError("published retained directory inventory drifted")
    return {
        "file_count": len(actual_files),
        "directory_count": len(actual_directories),
        "tree_verified": True,
        "read_only": os.name != "nt",
    }


def retain_minimum_artifacts(
    *,
    authentication: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    clone: Mapping[str, Any],
    receipt_path: Path,
    corrected_result_path: Path,
    convergence: Mapping[str, Any],
    corrected_profile_path: Path,
) -> dict[str, Any]:
    """Retain only the deadline-critical diagnostic evidence on GPFS."""
    storage = execution_plan["output_storage"]
    retained_root = Path(storage["retained_root"]).absolute()
    if not retained_root.exists():
        retained_root.mkdir(mode=0o700)
    _plain_directory(retained_root, "retained package root")
    label = (
        f"b7c-{authentication['manifest_sha256'][:12]}-"
        f"{execution_plan['executor_revision'][:12]}"
    )
    destination = retained_root / label
    incoming = retained_root / f".{label}.incoming.{os.getpid()}"
    if _lexists(destination) or _lexists(incoming):
        raise ContinuationError(
            "minimum retained package destination already exists"
        )

    clone_root = Path(clone["clone_root"]).resolve(strict=True)
    results_path = Path(clone["results_path"]).resolve(strict=True)
    aedt_path = Path(clone["aedt_path"]).resolve(strict=True)
    monitor_text = str(convergence.get("thermal_monitor_file", "") or "")
    monitor_path = Path(monitor_text).resolve(strict=True)
    if clone_root not in monitor_path.parents:
        raise ContinuationError(
            f"convergence monitor escaped clone: {monitor_path}"
        )
    profile_path = corrected_profile_path.resolve(strict=True)
    if clone_root not in profile_path.parents:
        raise ContinuationError(
            f"corrected profile escaped clone: {profile_path}"
        )
    sources = (
        (aedt_path, "symmetric.aedt"),
        (corrected_result_path.resolve(strict=True), "corrected_result.json"),
        (receipt_path.resolve(strict=True), "execution_receipt.json"),
        (monitor_path, f"convergence/{monitor_path.name}"),
        (profile_path, f"profile/{profile_path.name}"),
    )
    source_rows = []
    source_total = 0
    for source, relative in sources:
        value = _regular_file(source, f"minimum retention source {relative}")
        source_total += int(value.st_size)
        source_rows.append(
            {
                "source": source,
                "relative": relative,
                "size_bytes": int(value.st_size),
                "sha256": sha256_file(source),
            }
        )
    budget = int(storage["maximum_minimum_bundle_bytes"])
    if source_total > budget:
        raise ContinuationError(
            f"minimum retained package exceeds budget: "
            f"{source_total} > {budget}"
        )
    full_results_bytes, full_results_files = _tree_logical_size(results_path)

    incoming.mkdir(mode=0o700)
    published = False
    try:
        retained_rows = []
        for row in source_rows:
            destination_file = incoming.joinpath(
                *PurePosixPath(row["relative"]).parts
            )
            destination_file.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
            shutil.copy2(row["source"], destination_file)
            copied_sha = sha256_file(destination_file)
            if (
                destination_file.stat().st_size != row["size_bytes"]
                or copied_sha != row["sha256"]
                or sha256_file(row["source"]) != row["sha256"]
            ):
                raise ContinuationError(
                    f"minimum retained copy drifted: {row['relative']}"
                )
            retained_rows.append(
                {
                    "path": row["relative"],
                    "size_bytes": row["size_bytes"],
                    "sha256": copied_sha,
                }
            )
        package_manifest = {
            "schema": "mft-corrected-thermal-minimum-retained-package-v1",
            "diagnostic_only": True,
            "canonical": False,
            "candidate_sha256": SOURCE_CANDIDATE_SHA256,
            "checkpoint_manifest_sha256": authentication["manifest_sha256"],
            "source_solver_revision": SOURCE_SOLVER_REVISION,
            "executor_solver_revision": execution_plan["executor_revision"],
            "executor_required_ancestor": EXECUTOR_REQUIRED_ANCESTOR,
            "aedt_mixed_provenance": (
                "a1e4f source project saved after thermal-only solve by "
                "the exact planned clean a927-descendant executor"
            ),
            "files": retained_rows,
            "minimum_bundle_bytes": sum(
                int(row["size_bytes"]) for row in retained_rows
            ),
            "optional_field_bundle": {
                "retained": False,
                "reason": (
                    "deadline minimum-retention policy avoids duplicating "
                    "checkpoint premesh and unbounded thermal field data"
                ),
                "observed_results_tree_logical_bytes": full_results_bytes,
                "observed_results_tree_file_count": full_results_files,
                "can_be_retained_only_by_separate_size_and_quota_preflight": True,
            },
            "control_evidence": {
                "retention_receipt.json": (
                    "published in the same no-replace transaction and "
                    "read-only/hash-verified, but intentionally outside the "
                    "payload file inventory to avoid a self-hash cycle"
                ),
                ".slurm-scheduler-preserve.json": (
                    "read-only prune-protection marker"
                ),
            },
        }
        package_manifest["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(package_manifest)
        ).hexdigest()
        _atomic_json(incoming / "manifest.json", package_manifest)
        marker = {
            "schema": "slurm-scheduler-prune-protection-v1",
            "owner": "MFT_1MW_2026v1",
            "preserve": True,
            "reason": (
                "Retain corrected diagnostic thermal AEDT/evidence through "
                "deadline review; never promote automatically"
            ),
            "diagnostic_only": True,
        }
        _atomic_json(
            incoming / ".slurm-scheduler-preserve.json", marker
        )
        retention_receipt = {
            "schema": "mft-corrected-thermal-minimum-retention-receipt-v1",
            "destination": str(destination),
            "manifest_path": str(destination / "manifest.json"),
            "manifest_sha256": sha256_file(incoming / "manifest.json"),
            "prune_marker_path": str(
                destination / ".slurm-scheduler-preserve.json"
            ),
            "file_count": len(retained_rows),
            "minimum_bundle_bytes": package_manifest[
                "minimum_bundle_bytes"
            ],
            "optional_field_bundle": package_manifest[
                "optional_field_bundle"
            ],
            "published_atomically_with_package": True,
            "manifest_inventory_membership": False,
            "evidence_semantics": (
                "read-only package-control evidence; transactionally "
                "published and post-publish hash verified"
            ),
            "passed": True,
        }
        _atomic_json(
            incoming / "retention_receipt.json", retention_receipt
        )
        actual_bundle_bytes, _actual_bundle_files = _tree_logical_size(
            incoming
        )
        if actual_bundle_bytes > budget:
            raise ContinuationError(
                "minimum retained package including control evidence exceeds "
                f"budget: {actual_bundle_bytes} > {budget}"
            )
        protected = _protect_retained_tree(incoming)
        _fsync_directory(retained_root)
        _rename_noreplace(incoming, destination)
        published = True
        _fsync_directory(retained_root)
        post_publish = _verify_retained_tree(destination, protected)
        retention_path = destination / "retention_receipt.json"
        return {
            **retention_receipt,
            "retention_receipt_path": str(retention_path),
            "retention_receipt_sha256": sha256_file(retention_path),
            "published_tree": post_publish,
        }
    except BaseException:
        partial = destination if published else incoming
        if _lexists(partial):
            _quarantine_noreplace(partial, destination)
        raise


def execute_checkpoint(
    checkpoint: Path,
    output_root: Path,
    library_root: Path,
    execution_plan_path: Path,
    *,
    aedt_version: str | None = None,
) -> dict[str, Any]:
    authentication = authenticate_checkpoint(checkpoint)
    execution_plan = authenticate_execution_plan(
        execution_plan_path, authentication
    )
    runtime = _runtime_identity(library_root, execution_plan)
    storage_admission = admit_retained_output_storage(
        authentication, execution_plan, output_root
    )
    clone = clone_checkpoint(authentication, output_root)
    clone_root = Path(clone["clone_root"])
    aedt = Path(clone["aedt_path"])
    receipt_path = clone_root / "corrected_thermal_diagnostic_receipt.json"
    started = time.time()
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "source_checkpoint": str(Path(checkpoint).absolute()),
        "source_checkpoint_manifest_sha256": authentication["manifest_sha256"],
        "source_provenance": authentication["source_provenance"],
        "executor_provenance": {
            **runtime,
            "tool_payload_sha256": execution_plan["tool_payload_sha256"],
            "execution_plan_path": execution_plan["path"],
            "execution_plan_sha256": execution_plan["sha256"],
            "execution_plan_payload_sha256": execution_plan[
                "plan_payload_sha256"
            ],
        },
        "fixed_physics": authentication["physics_boundary"],
        "retained_output_admission": storage_admission,
        "clone": {
            **{key: str(value) if isinstance(value, Path) else value
               for key, value in clone.items()},
            "retained_on_success": True,
            "retained_on_failure": True,
        },
        "forbidden_regeneration_calls": {
            "maxwell_analyze": 0,
            "maxwell_create_or_copy": 0,
            "geometry_create_or_edit": 0,
            "material_create_or_edit": 0,
            "boundary_create_or_edit": 0,
            "setup_create_or_edit": 0,
            "solution_cleanup": 0,
            "mesh_generation": 0,
        },
        "started_epoch": started,
        "status": "starting",
    }
    desktop = None
    sim = None
    failure_stage = "runtime_attach"
    try:
        os.environ["MFT_PYAEDT_LIBRARY_ROOT"] = str(
            library_root.resolve(strict=True)
        )
        from run_simulation_260706 import (
            GIT_DIRTY,
            GIT_HASH,
            PYAEDT_LIBRARY_GIT_DIRTY,
            PYAEDT_LIBRARY_GIT_HASH,
            Simulation,
            pyDesktop,
        )
        from module.thermal_260706 import _solve_exact_thermal_setup
        from regression_260707.verify.replay_thermal_mesh import (
            _extract_thermal_targets,
            _thermal_target_summary,
        )

        imported = {
            "executor_solver_revision": str(GIT_HASH).lower(),
            "executor_solver_dirty": int(GIT_DIRTY),
            "pyaedt_library_revision": str(PYAEDT_LIBRARY_GIT_HASH).lower(),
            "pyaedt_library_dirty": int(PYAEDT_LIBRARY_GIT_DIRTY),
        }
        if imported != {
            "executor_solver_revision": execution_plan["executor_revision"],
            "executor_solver_dirty": 0,
            "pyaedt_library_revision": PYAEDT_LIBRARY_REVISION,
            "pyaedt_library_dirty": 0,
        }:
            raise ContinuationError(
                f"imported solver provenance drifted: {imported}"
            )
        receipt["executor_provenance"]["imported"] = imported

        desktop = pyDesktop(
            version=aedt_version,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        failure_stage = "project_load"
        project = desktop.load_project(path=str(aedt))
        native_project = project.project
        project_name = str(native_project.GetName() or "").strip()
        if not project_name:
            raise ContinuationError("loaded checkpoint clone has no project name")
        native_path = (
            Path(str(native_project.GetPath()))
            / f"{project_name}.aedt"
        ).resolve(strict=True)
        if native_path != aedt.resolve(strict=True):
            raise ContinuationError(
                f"AEDT loaded a different project: {native_path} != {aedt}"
            )
        before_designs = _native_design_names(native_project)

        failure_stage = "thermal_attach"
        thermal_wrapper, ipk, setup, native_design = (
            _attach_existing_thermal(project)
        )
        sim = Simulation(desktop=desktop)
        if sim.NUM_CORE != CORES or sim.NUM_TASK != TASKS:
            raise ContinuationError(
                f"authenticated Slurm core policy is not 8x1: "
                f"{sim.NUM_CORE}x{sim.NUM_TASK}"
            )
        sim.PROJECT_NAME = project_name
        sim.project_path = str(aedt.parent)
        sim.project = project
        sim.design1 = thermal_wrapper
        sim.design_thermal = ipk
        sim.thermal_mesh_preflight = {
            "schema": "mft-corrected-checkpoint-saved-premesh-v1",
            "passed": True,
            "mesh_plan_sha256": authentication["source_provenance"][
                "source_static_metadata_sha256"
            ],
            "mesh_artifact_readback_passed": True,
            "mesh_mapping_coverage_passed": True,
            "analysis_dispatched_after_premesh": False,
        }
        sim._remember_native_desktop_handle(desktop)
        receipt["saved_premesh_readback"] = attest_saved_premesh(
            authentication, clone_root
        )

        failure_stage = "native_fixed_readback"
        native_fixed = attest_native_fixed_model(native_design, ipk)
        receipt["native_fixed_readback"] = native_fixed

        failure_stage = "thermal_setup_dispatch"
        profile_snapshot = snapshot_profiles(Path(clone["results_path"]))
        solve_started = time.monotonic()
        solve = _solve_exact_thermal_setup(sim, ipk, setup)
        solve_elapsed = time.monotonic() - solve_started
        corrected_profile = fresh_corrected_profile(
            Path(clone["results_path"]), profile_snapshot
        )
        parallel = _attest_parallel_evidence(sim.thermal_parallel_evidence)
        receipt["parallel_attestation"] = parallel
        receipt["solve"] = {
            **solve,
            "elapsed_seconds": solve_elapsed,
            "corrected_profile_path": str(corrected_profile),
            "corrected_profile_sha256": sha256_file(corrected_profile),
        }
        convergence = solve["convergence"]

        failure_stage = "temperature_extraction"
        if int(convergence.get("thermal_converged", 0)) != 1:
            raise ContinuationError(
                "corrected thermal checkpoint solve did not converge: "
                f"{convergence.get('thermal_convergence_reason')}"
            )
        sweeps = tuple(str(value) for value in (ipk.existing_analysis_sweeps or ()))
        if len(sweeps) != 1:
            raise ContinuationError(
                f"corrected thermal solution sweep count mismatch: {sweeps!r}"
            )
        values, extraction = _extract_thermal_targets(ipk, sweeps[0])
        temperatures = _thermal_target_summary(values)
        winding_max = max(
            float(temperatures["T_max_Tx"]),
            float(temperatures["T_max_Rx_main"]),
            float(temperatures["T_max_Rx_side"]),
        )
        core_max = float(temperatures["T_max_core"])
        if not all(math.isfinite(value) for value in (winding_max, core_max)):
            raise ContinuationError("extracted temperatures are nonfinite")
        receipt["temperature_extraction"] = extraction
        receipt["temperatures"] = temperatures
        receipt["constraint_observation"] = {
            "winding_max_c": winding_max,
            "winding_limit_c": 100.0,
            "winding_pass": winding_max <= 100.0,
            "core_max_c": core_max,
            "core_limit_c": 120.0,
            "core_pass": core_max <= 120.0,
            "all_temperature_constraints_pass": (
                winding_max <= 100.0 and core_max <= 120.0
            ),
            "diagnostic_only": True,
        }

        failure_stage = "retained_artifact_attestation"
        after_designs = _native_design_names(native_project)
        if after_designs != before_designs:
            raise ContinuationError(
                "thermal-only continuation changed native design inventory"
            )
        if not aedt.is_file() or not Path(clone["results_path"]).is_dir():
            raise ContinuationError(
                "retained AEDT project/results are incomplete"
            )
        receipt["retained_artifacts"] = {
            "aedt_path": str(aedt),
            "aedt_sha256": sha256_file(aedt),
            "results_path": str(Path(clone["results_path"])),
            "native_design_names": list(after_designs),
            "thermal_design": THERMAL_DESIGN,
            "thermal_setup": THERMAL_SETUP,
            "aedt_mixed_provenance": (
                "a1e4f source project saved after thermal-only solve by "
                "the exact planned clean a927-descendant executor"
            ),
        }
        corrected_result_path = clone_root / "corrected_result.json"
        corrected_result = {
            "schema": "mft-corrected-thermal-diagnostic-result-v1",
            "diagnostic_only": True,
            "canonical": False,
            "source_provenance": receipt["source_provenance"],
            "executor_provenance": receipt["executor_provenance"],
            "constraint_observation": receipt["constraint_observation"],
            "temperatures": receipt["temperatures"],
            "temperature_extraction": receipt["temperature_extraction"],
            "convergence": convergence,
            "parallel_attestation": receipt["parallel_attestation"],
            "native_fixed_readback": receipt["native_fixed_readback"],
        }
        _atomic_json(corrected_result_path, corrected_result)
        retained_label = (
            f"b7c-{authentication['manifest_sha256'][:12]}-"
            f"{execution_plan['executor_revision'][:12]}"
        )
        receipt["minimum_retention_plan"] = {
            "destination": str(
                Path(
                    execution_plan["output_storage"]["retained_root"]
                ).absolute()
                / retained_label
            ),
            "required_files": [
                "symmetric.aedt",
                "corrected_result.json",
                "execution_receipt.json",
                "convergence monitor",
                "latest profile",
                "manifest.json",
                ".slurm-scheduler-preserve.json",
            ],
            "duplicate_premesh": False,
            "optional_full_field_bundle": True,
        }
        receipt["status"] = "diagnostic_complete"
        receipt["completed_epoch"] = time.time()
        receipt["elapsed_seconds"] = receipt["completed_epoch"] - started
        _atomic_json(receipt_path, receipt)
        try:
            desktop.release_desktop(
                close_projects=True, close_on_exit=True
            )
        except Exception as exc:
            raise ContinuationError(
                "AEDT release failed before minimum retention"
            ) from exc
        desktop = None
        retention = retain_minimum_artifacts(
            authentication=authentication,
            execution_plan=execution_plan,
            clone=clone,
            receipt_path=receipt_path,
            corrected_result_path=corrected_result_path,
            convergence=convergence,
            corrected_profile_path=corrected_profile,
        )
        receipt["minimum_retention"] = {
            **retention,
        }
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = sha256_file(receipt_path)
        return receipt
    except BaseException as exc:
        receipt.update(
            {
                "status": "diagnostic_failed",
                "failure_stage": failure_stage,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:4000],
                "traceback": traceback.format_exc()[-16000:],
                "completed_epoch": time.time(),
            }
        )
        receipt["elapsed_seconds"] = receipt["completed_epoch"] - started
        _atomic_json(receipt_path, receipt)
        raise
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(
                    close_projects=True, close_on_exit=True
                )
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True, type=Path)
    execute = commands.add_parser("execute")
    execute.add_argument("--checkpoint", required=True, type=Path)
    execute.add_argument("--output-root", required=True, type=Path)
    execute.add_argument("--library-root", required=True, type=Path)
    execute.add_argument("--execution-plan", required=True, type=Path)
    execute.add_argument("--aedt-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = authenticate_checkpoint(args.checkpoint)
            output = {
                "status": "authenticated",
                "diagnostic_only": True,
                "manifest_sha256": result["manifest_sha256"],
                "file_count": len(result["files"]),
                "source_provenance": result["source_provenance"],
            }
        else:
            output = execute_checkpoint(
                args.checkpoint,
                args.output_root,
                args.library_root,
                args.execution_plan,
                aedt_version=args.aedt_version,
            )
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ContinuationError, OSError, subprocess.SubprocessError) as exc:
        print(
            f"CORRECTED_THERMAL_CONTINUATION_ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
