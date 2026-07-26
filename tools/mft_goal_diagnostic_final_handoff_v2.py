#!/usr/bin/env python3
"""Publish a fail-closed, reference-only diagnostic handoff package.

This campaign-specific tool never copies or changes scientific artifacts.  It
authenticates the existing Pareto, model, AEDT-open, Full-failure, and terminal
evidence roots, then publishes a three-file immutable reference index.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping
import uuid


PACKAGE_SCHEMA = "mft-goal-diagnostic-final-handoff-reference-package-v2"
SEAL_SCHEMA = "mft-goal-diagnostic-final-handoff-reference-package-seal-v2"
TERMINAL_SCHEMA = "mft-goal-task96313-terminal-evidence-v1"
PACKAGE_NAME = "diagnostic_final_handoff_reference_package_v2"
CAMPAIGN_ID = "mft_goal_20260726"
CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
FIXED_PHYSICS_SHA256 = (
    "c7d1cc3d0da7b2ef5068e71ff7dd907fbc8f1ee00774473ae8a7f742d7d87acf"
)

PARETO_ROOT = "aggregate_rolling512_d4e4d60"
MODEL_ROOT = "diagnostic_fallback_model_package_v1"
OPEN_ROOT = "aedt_open_validation_collected_v1"
FULL_FAILURE_ROOT = "provisional_full_collector_527a6aa4_task96307_260726"

EXPECTED_FILES = {
    "pareto_manifest": (
        f"{PARETO_ROOT}/aggregate_manifest.json",
        "0ea4feb43f9eb37c8b057e7f06720f0c23590b463111443f99d8ffa6d08c4a63",
    ),
    "pareto_terminal": (
        f"{PARETO_ROOT}/global_terminal_candidates.csv",
        "43f3ebeeb3ae64f9c643d339197def0bcd3e98f1b49aad97dbb530c290f50d39",
    ),
    "pareto_production": (
        f"{PARETO_ROOT}/global_pareto_front.csv",
        "50d15ec628c321f0111738de97eb994e8a1ce5b2adcb39d858528d7699a8cc89",
    ),
    "pareto_audit": (
        f"{PARETO_ROOT}/global_objective_front.csv",
        "0e1f5c4d72bf97b27b3e4814aa3777997fdf1ab4f0ee110a1bdfbf6b5c076f52",
    ),
    "pareto_standard": (
        f"{PARETO_ROOT}/standard_candidates.csv",
        "f99ee4b7fba62a83e5b958d24c31260a5ba48c6fa77e6d25f2f189896df3ff6d",
    ),
    "alternate_standard": (
        "diagnostic_standard_final_512_4f3ec75_260725/"
        "selection/diagnostic_candidates.csv",
        "73d94686b010b464e2881c65698cd2c781a52a2078858c7d72453310e7da69f0",
    ),
    "model_manifest": (
        f"{MODEL_ROOT}/package_manifest.json",
        "42eda03449a488bcfc91fce0d8c5b2a69c2e6f0f3813611b245a3654cac65818",
    ),
    "model_seal": (
        f"{MODEL_ROOT}/package_seal.json",
        "8499e4a7a175c99783bbac5bcb4fd40e07cd454c51bacf046eac9b5d4594e3dd",
    ),
    "full_aedt": (
        f"{MODEL_ROOT}/models/full.aedt",
        "b82245fb9d17617a566621728baf21afac4657111e4ea64e88ad3f49c6e7e1e8",
    ),
    "symmetric_aedt": (
        f"{MODEL_ROOT}/models/symmetric.aedt",
        "186ca4eaecf232420fa3c95901af13bbbe6064e5dcbba612616858d93ecf5d1c",
    ),
    "open_collection": (
        f"{OPEN_ROOT}/local_collection_receipt.json",
        "85cdbee60df7d373d6396dc275adf4d9410dfbbf27041317aa035a5707af5c47",
    ),
    "open_full": (
        f"{OPEN_ROOT}/full/validation_receipt.json",
        "5979d4ca496b1e5d3ae0e2517ac7f8b08b592cdcfab8d5b866eecdf733599de6",
    ),
    "open_symmetric": (
        f"{OPEN_ROOT}/symmetric/validation_receipt.json",
        "c8799b0e9ea74f7a6a402e4f9a74efdc309a15e617768b1fea66630f492dc7d8",
    ),
    "full_failure": (
        f"{FULL_FAILURE_ROOT}/failure_ledger.json",
        "523e2ba9902e844e734e400b9d5976eacad22a4fc0fe8837068d93f8e217552a",
    ),
}


class HandoffError(RuntimeError):
    """Raised when an authority or publication gate is not satisfied."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _real_directory(path: Path, *, label: str) -> Path:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise HandoffError(f"{label} is not one real directory")
    return path.resolve(strict=True)


def _regular_file(path: Path, *, maximum_bytes: int | None = None) -> Path:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or _is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (maximum_bytes is not None and metadata.st_size > maximum_bytes)
    ):
        raise HandoffError(f"unsafe regular file: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path, maximum_bytes=8 * 1024 * 1024)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"JSON root is not an object: {path}")
    return value


def _validate_seal(
    path: Path,
    *,
    schema_field: str,
    schema: str,
) -> dict[str, Any]:
    value = _read_json(path)
    claimed = str(value.get("payload_sha256") or "")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if (
        value.get(schema_field) != schema
        or len(claimed) != 64
        or canonical_sha256(unsigned) != claimed
    ):
        raise HandoffError(f"canonical payload seal failed: {path}")
    return value


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    _regular_file(path)
    name = str(path if relative_to is None else path.relative_to(relative_to))
    return {
        "path": name.replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_file(base: Path, relative: str, expected_sha: str) -> dict[str, Any]:
    path = base.joinpath(*relative.split("/"))
    record = _record(path, relative_to=base)
    if record["sha256"] != expected_sha:
        raise HandoffError(f"authority SHA drifted: {relative}")
    return record


def _require_truth_flags(value: Mapping[str, Any], *, label: str) -> None:
    if (
        value.get("diagnostic_only") is not True
        or value.get("canonical") is not False
        or value.get("production_truth_eligible") is not False
        or value.get("solver_result_truth_included") is not False
    ):
        raise HandoffError(f"{label} truth boundary drifted")


def _validate_pareto(base: Path) -> dict[str, Any]:
    records = {
        key: _require_file(base, relative, digest)
        for key, (relative, digest) in EXPECTED_FILES.items()
        if key.startswith("pareto_")
    }
    alternate = _require_file(base, *EXPECTED_FILES["alternate_standard"])
    manifest = _validate_seal(
        base / EXPECTED_FILES["pareto_manifest"][0],
        schema_field="schema_version",
        schema="mft-goal-20260726-global-pareto-v1",
    )
    expected = {
        "seed_count": 512,
        "minimum_seed_count": 512,
        "input_terminal_row_count": 163840,
        "deduplicated_physical_geometry_count": 133563,
        "physical_feasible_count": 0,
        "global_pareto_count": 0,
        "global_objective_front_count": 22,
        "standard_candidate_count": 12,
        "seed_local_pareto_merge_used": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "search_only_proposal": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise HandoffError("global Pareto authority contract drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HandoffError("global Pareto artifact inventory is absent")
    artifact_contract = {
        "global_terminal_candidates": ("pareto_terminal", 133563),
        "global_pareto_front": ("pareto_production", 0),
        "global_objective_front": ("pareto_audit", 22),
        "standard_candidates": ("pareto_standard", 12),
    }
    for name, (record_key, rows) in artifact_contract.items():
        item = artifacts.get(name)
        if (
            not isinstance(item, dict)
            or item.get("row_count") != rows
            or item.get("sha256") != records[record_key]["sha256"]
        ):
            raise HandoffError(f"global Pareto artifact drifted: {name}")
    return {
        "root": str(base / PARETO_ROOT),
        "manifest": records["pareto_manifest"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "files": records,
        "seed_count": 512,
        "terminal_rows": 163840,
        "deduplicated_geometries": 133563,
        "production_front_rows": 0,
        "audit_only_front_rows": 22,
        "authoritative_search_only_standard_rows": 12,
        "non_authoritative_alternate_standard": alternate,
        "seed_local_pareto_merge_used": False,
    }


def _validate_model_package(base: Path) -> dict[str, Any]:
    records = {
        key: _require_file(base, relative, digest)
        for key, (relative, digest) in EXPECTED_FILES.items()
        if key
        in {
            "model_manifest",
            "model_seal",
            "full_aedt",
            "symmetric_aedt",
        }
    }
    manifest = _validate_seal(
        base / EXPECTED_FILES["model_manifest"][0],
        schema_field="schema",
        schema="diagnostic_fallback_model_package_v1",
    )
    seal = _validate_seal(
        base / EXPECTED_FILES["model_seal"][0],
        schema_field="schema",
        schema="diagnostic_fallback_model_package_seal_v1",
    )
    _require_truth_flags(manifest, label="model manifest")
    _require_truth_flags(seal, label="model seal")
    if (
        manifest.get("candidate_sha256") != CANDIDATE_SHA256
        or manifest.get("fixed_physics_sha256") != FIXED_PHYSICS_SHA256
        or seal.get("package_manifest_sha256")
        != records["model_manifest"]["sha256"]
        or seal.get("package_manifest_payload_sha256")
        != manifest["payload_sha256"]
    ):
        raise HandoffError("model package cross-binding drifted")
    model_root = base / MODEL_ROOT
    sealed_files = seal.get("sealed_files")
    if not isinstance(sealed_files, list):
        raise HandoffError("model sealed-file inventory is absent")
    expected_paths: set[str] = set()
    for item in sealed_files:
        if not isinstance(item, dict):
            raise HandoffError("model sealed-file row is invalid")
        relative = str(item.get("path") or "")
        if (
            not relative
            or relative.casefold() in {value.casefold() for value in expected_paths}
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise HandoffError("model sealed-file path is unsafe or duplicated")
        expected_paths.add(relative)
        actual = _record(model_root / relative, relative_to=model_root)
        if (
            actual["sha256"] != item.get("sha256")
            or actual["size_bytes"] != item.get("size_bytes")
        ):
            raise HandoffError(f"model sealed file drifted: {relative}")
    actual_paths = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths | {"package_seal.json"}:
        raise HandoffError("model package file set drifted")
    return {
        "root": str(model_root),
        "manifest": records["model_manifest"],
        "manifest_payload_sha256": manifest["payload_sha256"],
        "seal": records["model_seal"],
        "seal_payload_sha256": seal["payload_sha256"],
        "full_aedt": records["full_aedt"],
        "symmetric_aedt": records["symmetric_aedt"],
        "sealed_file_count": seal.get("sealed_file_count"),
        "sealed_file_inventory_sha256": seal.get(
            "sealed_file_inventory_sha256"
        ),
    }


def _validate_open_collection(base: Path) -> dict[str, Any]:
    records = {
        key: _require_file(base, relative, digest)
        for key, (relative, digest) in EXPECTED_FILES.items()
        if key in {"open_collection", "open_full", "open_symmetric"}
    }
    collection_path = base / EXPECTED_FILES["open_collection"][0]
    receipt = _validate_seal(
        collection_path,
        schema_field="schema",
        schema="mft-live-aedt-open-validation-local-collection-v1",
    )
    _require_truth_flags(receipt, label="open collection")
    if (
        receipt.get("candidate_sha256") != CANDIDATE_SHA256
        or receipt.get("fixed_physics_sha256") != FIXED_PHYSICS_SHA256
        or receipt.get("scheduler_access") != "GET-only"
    ):
        raise HandoffError("open collection authority drifted")
    root = base / OPEN_ROOT
    lanes = receipt.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != {"full", "symmetric"}:
        raise HandoffError("open collection lanes drifted")
    expected_paths = {"local_collection_receipt.json"}
    for lane in ("full", "symmetric"):
        item = lanes[lane]
        files = item.get("files") if isinstance(item, dict) else None
        if not isinstance(files, dict):
            raise HandoffError(f"open collection {lane} files are absent")
        for name, digest in files.items():
            relative = f"{lane}/{name}"
            expected_paths.add(relative)
            if _record(root / relative, relative_to=root)["sha256"] != digest:
                raise HandoffError(f"open collection file drifted: {relative}")
        if not all(bool(value) for value in item.get("checks", {}).values()):
            raise HandoffError(f"open collection {lane} check failed")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise HandoffError("open collection file set drifted")
    return {
        "root": str(root),
        "receipt": records["open_collection"],
        "receipt_payload_sha256": receipt["payload_sha256"],
        "full_validation_receipt": records["open_full"],
        "symmetric_validation_receipt": records["open_symmetric"],
        "model_snapshot_open_validated": True,
    }


def _validate_full_failure(base: Path) -> dict[str, Any]:
    relative, digest = EXPECTED_FILES["full_failure"]
    record = _require_file(base, relative, digest)
    ledger = _validate_seal(
        base / relative,
        schema_field="schema_version",
        schema="mft-goal-provisional-full-collection-failure-v1",
    )
    task = ledger.get("scheduler_task")
    if (
        ledger.get("diagnostic_only") is not True
        or ledger.get("production_eligible") is not False
        or ledger.get("success_receipt_created") is not False
        or ledger.get("collection_blocked_fail_closed") is not True
        or ledger.get("scheduler_mutation_performed") is not False
        or not isinstance(task, dict)
        or task.get("task_id") != 96307
        or task.get("state") != "failed"
        or task.get("exit_code") != 124
        or task.get("timeout_seconds") != 43200
    ):
        raise HandoffError("Full failure authority drifted")
    return {
        "root": str(base / FULL_FAILURE_ROOT),
        "ledger": record,
        "ledger_payload_sha256": ledger["payload_sha256"],
        "task_id": 96307,
        "terminal_state": "failed",
        "exit_code": 124,
        "scientific_result_available": False,
    }


def validate_terminal_evidence(path: Path, base: Path) -> dict[str, Any]:
    evidence = _validate_seal(
        path,
        schema_field="schema",
        schema=TERMINAL_SCHEMA,
    )
    task = evidence.get("task")
    access = evidence.get("scheduler_access")
    if (
        evidence.get("diagnostic_only") is not True
        or evidence.get("canonical") is not False
        or evidence.get("production_truth_eligible") is not False
        or evidence.get("candidate_sha256") != CANDIDATE_SHA256
        or evidence.get("fixed_physics_sha256") != FIXED_PHYSICS_SHA256
        or not isinstance(task, dict)
        or task.get("task_id") != 96313
        or task.get("name")
        != "mft-goal-corrected-thermal-l96230-b7c30cb70b95-native-r5-n111"
        or task.get("account_name") != "r1jae262"
        or task.get("actual_node_name") != "n111"
        or task.get("assigned_allocation") != 14641
        or str(task.get("slurm_job_id")) != "838708"
        or task.get("cpus") != 8
        or task.get("memory_mb") != 294912
        or task.get("timeout_seconds") != 6600
        or not isinstance(access, dict)
        or access.get("methods_used") != ["GET"]
        or access.get("mutation_performed") is not False
    ):
        raise HandoffError("task96313 terminal evidence identity drifted")
    state = str(task.get("state") or "")
    status = str(task.get("status") or "")
    exit_code = task.get("exit_code")
    if state == "succeeded":
        if status != "completed" or exit_code != 0:
            raise HandoffError("task96313 success union is invalid")
        terminal_branch = "success"
    elif state in {"failed", "cancelled"}:
        if status not in {"failed", "cancelled"} or exit_code == 0:
            raise HandoffError("task96313 failure union is invalid")
        terminal_branch = "failure"
    else:
        raise HandoffError(
            "task96313 is pending/active/unknown; final publication refused"
        )
    terminal_root = _real_directory(path.parent, label="terminal evidence root")
    root = base.resolve(strict=True)
    if root != terminal_root and root not in terminal_root.parents:
        raise HandoffError("terminal evidence escaped the campaign root")
    streams = evidence.get("streams")
    if not isinstance(streams, dict) or set(streams) != {"stdout", "stderr"}:
        raise HandoffError("terminal stream evidence is incomplete")
    for stream in streams.values():
        if not isinstance(stream, dict):
            raise HandoffError("terminal stream record is invalid")
        relative = str(stream.get("path") or "")
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not relative
        ):
            raise HandoffError("terminal stream path is unsafe")
        actual = _record(terminal_root / relative, relative_to=terminal_root)
        if (
            actual["sha256"] != stream.get("sha256")
            or actual["size_bytes"] != stream.get("size_bytes")
        ):
            raise HandoffError("terminal stream evidence drifted")
    return {
        "root": str(terminal_root),
        "evidence": _record(path),
        "evidence_payload_sha256": evidence["payload_sha256"],
        "terminal_branch": terminal_branch,
        "task": task,
        "diagnostic_temperature_observation_available": bool(
            evidence.get("diagnostic_temperature_observation_available")
        )
        if terminal_branch == "success"
        else False,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        move = ctypes.windll.kernel32.MoveFileW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move.restype = ctypes.c_int
        if not move(str(source), str(destination)):
            raise ctypes.WinError()
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(destination))
            return
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(source, destination)


def _validate_existing(
    destination: Path,
    *,
    terminal_sha256: str,
    pareto: Mapping[str, Any],
    models: Mapping[str, Any],
    open_validation: Mapping[str, Any],
    full_failure: Mapping[str, Any],
) -> dict[str, Any]:
    destination = _real_directory(
        destination, label="existing handoff package"
    )
    actual_names = {
        path.name for path in destination.iterdir() if path.is_file()
    }
    if actual_names != {
        "DIAGNOSTIC_ONLY_DO_NOT_PROMOTE.txt",
        "handoff_manifest.json",
        "handoff_seal.json",
    }:
        raise HandoffError("existing handoff file set drifted")
    manifest_path = destination / "handoff_manifest.json"
    seal_path = destination / "handoff_seal.json"
    manifest = _validate_seal(
        manifest_path, schema_field="schema", schema=PACKAGE_SCHEMA
    )
    seal = _validate_seal(
        seal_path, schema_field="schema", schema=SEAL_SCHEMA
    )
    truth = manifest.get("truth_gates")
    if (
        manifest.get("package_kind") != "diagnostic_reference_only"
        or manifest.get("reference_only") is not True
        or manifest.get("self_contained") is not False
        or not isinstance(truth, dict)
        or truth.get("diagnostic_only") is not True
        or truth.get("canonical") is not False
        or truth.get("production_truth_eligible") is not False
        or truth.get("goal_fully_satisfied") is not False
        or seal.get("diagnostic_only") is not True
        or seal.get("canonical") is not False
        or seal.get("production_truth_eligible") is not False
        or manifest.get("pareto_reference") != dict(pareto)
        or manifest.get("model_package_reference") != dict(models)
        or manifest.get("open_validation_reference")
        != dict(open_validation)
        or manifest.get("full_failure_reference") != dict(full_failure)
    ):
        raise HandoffError("existing handoff truth/reference drifted")
    sealed_files = seal.get("sealed_files")
    if not isinstance(sealed_files, list) or len(sealed_files) != 2:
        raise HandoffError("existing handoff sealed inventory drifted")
    actual_sealed = []
    for item in sealed_files:
        if not isinstance(item, dict):
            raise HandoffError("existing handoff sealed row is invalid")
        relative = str(item.get("path") or "")
        actual = _record(destination / relative, relative_to=destination)
        if actual != item:
            raise HandoffError(f"existing handoff file drifted: {relative}")
        actual_sealed.append(actual)
    if (
        seal.get("sealed_file_inventory_sha256")
        != canonical_sha256({"files": actual_sealed})
    ):
        raise HandoffError("existing handoff inventory seal drifted")
    if (
        manifest.get("task96313_terminal_evidence", {})
        .get("evidence", {})
        .get("sha256")
        != terminal_sha256
        or seal.get("handoff_manifest_sha256")
        != sha256_file(manifest_path)
        or seal.get("handoff_manifest_payload_sha256")
        != manifest["payload_sha256"]
    ):
        raise HandoffError("existing handoff identity drifted")
    return {
        "status": "already_published",
        "package_path": str(destination),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "seal_sha256": sha256_file(seal_path),
        "seal_payload_sha256": seal["payload_sha256"],
    }


def publish_handoff(
    *,
    base_root: Path,
    terminal_evidence_path: Path,
    output_name: str = PACKAGE_NAME,
) -> dict[str, Any]:
    base = _real_directory(base_root, label="campaign root")
    if (
        not output_name
        or Path(output_name).name != output_name
        or output_name.startswith(".")
    ):
        raise HandoffError("output name is unsafe")
    terminal = validate_terminal_evidence(terminal_evidence_path, base)
    pareto_before = _validate_pareto(base)
    model_before = _validate_model_package(base)
    open_before = _validate_open_collection(base)
    full_failure_before = _validate_full_failure(base)
    destination = base / output_name
    if destination.exists():
        return _validate_existing(
            destination,
            terminal_sha256=terminal["evidence"]["sha256"],
            pareto=pareto_before,
            models=model_before,
            open_validation=open_before,
            full_failure=full_failure_before,
        )

    staging = base / f".{output_name}.staging-{uuid.uuid4().hex}"
    claim = base / f".{output_name}.publish.claim"
    claim_descriptor = os.open(
        claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    os.close(claim_descriptor)
    staging.mkdir(mode=0o700)
    try:
        warning_path = staging / "DIAGNOSTIC_ONLY_DO_NOT_PROMOTE.txt"
        warning = (
            "DIAGNOSTIC REFERENCE PACKAGE ONLY.\n"
            "It is not canonical, production eligible, or proof that the "
            "size, resonance, winding-temperature, or core-temperature "
            "requirements passed. External authorities remain required.\n"
        )
        descriptor = os.open(
            warning_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(warning.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        warning_record = _record(warning_path, relative_to=staging)
        manifest: dict[str, Any] = {
            "schema": PACKAGE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "package_name": output_name,
            "package_kind": "diagnostic_reference_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference_only": True,
            "self_contained": False,
            "relocatable": False,
            "external_authority_required": True,
            "goal_contract": {
                "maximum_dimensions_mm": [1200, 1000, 750],
                "minimum_resonance_hz": 15000,
                "maximum_winding_temperature_c": 100,
                "maximum_core_temperature_c": 120,
                "fan_velocity_m_per_s": 1.5,
                "thermal_interface_contract_modified": False,
            },
            "candidate_sha256": CANDIDATE_SHA256,
            "fixed_physics_sha256": FIXED_PHYSICS_SHA256,
            "pareto_reference": pareto_before,
            "authoritative_standard_selection": {
                "path": EXPECTED_FILES["pareto_standard"][0],
                "sha256": EXPECTED_FILES["pareto_standard"][1],
                "row_count": 12,
                "search_only": True,
            },
            "alternate_standard_selection": {
                "path": pareto_before[
                    "non_authoritative_alternate_standard"
                ]["path"],
                "sha256": pareto_before[
                    "non_authoritative_alternate_standard"
                ]["sha256"],
                "authoritative": False,
                "production_eligible": False,
            },
            "model_package_reference": model_before,
            "open_validation_reference": open_before,
            "full_failure_reference": full_failure_before,
            "task96313_terminal_evidence": terminal,
            "cross_bindings": {
                "candidate_equal": True,
                "fixed_physics_equal": True,
                "model_artifact_sha_equal_open_receipts": True,
                "full_timeout_not_scientific_infeasibility": True,
                "production_pareto_is_empty": True,
            },
            "truth_gates": {
                "diagnostic_only": True,
                "canonical": False,
                "production_truth_eligible": False,
                "production_final_deliverable": False,
                "production_final_wrapper_compatible": False,
                "truth_dataset_ingestion_allowed": False,
                "automatic_promotion_allowed": False,
                "promotion_authority_claimed": False,
                "goal_fully_satisfied": False,
                "size_production_verified": False,
                "resonance_production_verified": False,
                "temperature_production_verified": False,
                "model_snapshot_open_validated": True,
                "terminal_evidence_authenticated": True,
                "snapshot_solver_result_truth_included": False,
                "diagnostic_temperature_observation_available": terminal[
                    "diagnostic_temperature_observation_available"
                ],
            },
            "mutation_boundary": {
                "repository_mutation_performed": False,
                "scheduler_submission_performed": False,
                "scheduler_cancel_performed": False,
                "external_authority_mutation_performed": False,
            },
            "internal_inventory": [warning_record],
            "internal_tree_sha256": canonical_sha256(
                {"files": [warning_record]}
            ),
        }
        manifest["payload_sha256"] = canonical_sha256(manifest)
        manifest_path = staging / "handoff_manifest.json"
        _write_json_exclusive(manifest_path, manifest)
        internal = [
            warning_record,
            _record(manifest_path, relative_to=staging),
        ]
        seal: dict[str, Any] = {
            "schema": SEAL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "package_name": output_name,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "atomic_publication": True,
            "publication_mode": "no_replace_directory_rename",
            "handoff_manifest_sha256": internal[1]["sha256"],
            "handoff_manifest_payload_sha256": manifest["payload_sha256"],
            "sealed_files": internal,
            "sealed_file_inventory_sha256": canonical_sha256(
                {"files": internal}
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        seal["payload_sha256"] = canonical_sha256(seal)
        seal_path = staging / "handoff_seal.json"
        _write_json_exclusive(seal_path, seal)
        _fsync_directory(staging)

        # Re-authenticate every external authority after staging.
        if (
            _validate_pareto(base) != pareto_before
            or _validate_model_package(base) != model_before
            or _validate_open_collection(base) != open_before
            or _validate_full_failure(base) != full_failure_before
            or validate_terminal_evidence(terminal_evidence_path, base)
            != terminal
        ):
            raise HandoffError("external authority changed during staging")
        for path in staging.iterdir():
            os.chmod(path, stat.S_IREAD)
        _rename_no_replace(staging, destination)
        _fsync_directory(base)
        published = _validate_existing(
            destination,
            terminal_sha256=terminal["evidence"]["sha256"],
            pareto=pareto_before,
            models=model_before,
            open_validation=open_before,
            full_failure=full_failure_before,
        )
        published["status"] = "published"
        return published
    finally:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass
            shutil.rmtree(staging)
        try:
            os.chmod(claim, stat.S_IWRITE | stat.S_IREAD)
            claim.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--terminal-evidence", required=True, type=Path)
    parser.add_argument("--output-name", default=PACKAGE_NAME)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate all authorities but do not publish",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base = _real_directory(args.base_root, label="campaign root")
        if args.preflight_only:
            result = {
                "status": "preflight_passed",
                "terminal": validate_terminal_evidence(
                    args.terminal_evidence, base
                ),
                "pareto": _validate_pareto(base),
                "models": _validate_model_package(base),
                "open_validation": _validate_open_collection(base),
                "full_failure": _validate_full_failure(base),
            }
        else:
            result = publish_handoff(
                base_root=base,
                terminal_evidence_path=args.terminal_evidence,
                output_name=args.output_name,
            )
    except (HandoffError, OSError, ValueError) as exc:
        print(
            f"DIAGNOSTIC_HANDOFF_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
