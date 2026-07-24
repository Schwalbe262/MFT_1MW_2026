#!/usr/bin/env python3
"""Recover complete solved EM evidence and thermal FEA from a sealed run.

This entrypoint is intentionally post-solve only.  It attests the preserved
project/results pair without opening or copying either one, loads the exact
manifest-bound source result row, reconstructs its physical Full-model loss
map, and runs only the downstream Icepak stage in a fresh project.  It never
creates a Maxwell wrapper, queries a reopened Maxwell result, or calls a
Maxwell analyze method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import pandas as pd


class UncertainStandaloneSolverExit(RuntimeError):
    """Require a hard process exit so PyAEDT atexit cannot release AEDT."""

    exit_code = 70


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_simulation_260706 import (  # noqa: E402
    GIT_DIRTY,
    GIT_HASH,
    GUI,
    PYAEDT_LIBRARY_GIT_DIRTY,
    PYAEDT_LIBRARY_GIT_HASH,
    STANDALONE_CORE_16_CONTRACT_VERSION,
    STANDALONE_CORE_16_LICENSE_CONTRACT_VERSION,
    Simulation,
    _aedt_design_name,
    _em_result_validation,
    _is_ac_magnetic_solution,
    _load_fixed_input_parameter,
    _thermal_result_is_valid,
    pyDesktop,
    validation_check,
)
from module.input_parameter_260706 import get_tx_y_gaps  # noqa: E402
from module.fixed_boundary_contract import (  # noqa: E402
    attest_fixed_boundary,
    fixed_boundary_result_metadata,
)
from module.thermal_260706 import (  # noqa: E402
    RX_EXPLICIT_INSULATION_MATERIAL,
    RX_EXPLICIT_INSULATION_NATIVE_READBACK_CONTRACT_VERSION,
    RX_EXPLICIT_INSULATION_POLICY,
    THERMAL_PAD_CONDUCTIVITY_W_MK,
    THERMAL_PAD_MATERIAL_POLICY,
    THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION,
    THERMAL_MESH_PLAN_CONTRACT_VERSION,
    THERMAL_MESH_POLICY,
    THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
    run_thermal_analysis,
)


RECOVERY_SCHEMA = "mft-full-postsolve-recovery-v1"
PRESERVER_MANIFEST_SCHEMA = "mft-deadline-full-preserved-workdir-v1"
PRESERVER_IDENTITY_SCHEMA = "mft-deadline-full-preserver-source-identity-v1"
TERMINAL_WATCH_SCHEMA = "mft-deadline-full-preserver-terminal-watch-v1"
ALLOWED_TERMINATION_REASONS = frozenset({
    "source_exit_code_published",
    "source_workdir_disappeared",
})
_CORE_RE = re.compile(
    r"^core_\d+(?:_(?:leg_(?:left|center|right)|yoke_(?:top|bottom)))?$"
)
_RECOVERY_MAXWELL_DESIGNS = (
    "maxwell_matrix",
    "maxwell_cap",
    "maxwell_loss",
)
_RECOVERY_THERMAL_DESIGN = "icepak_thermal"
_RECOVERY_THERMAL_DESIGN_TYPE = "Icepak"
# InsertDesign uses "SteadyState TemperatureAndFlow", while the exact native
# GetSolutionType readback of the saved production Icepak design is
# "SteadyState".
_RECOVERY_THERMAL_NATIVE_SOLUTION_TYPE = "SteadyState"
_RECOVERY_THERMAL_SETUP = "ThermalSetup"
_SEALED_SOURCE_RESULT_RELATIVE_PATH = (
    "repo/simulation_results_260706.csv"
)
_SOURCE_RESULT_GIT_HASH = (
    "146142be579e2f3e45f12961214018a4e28445c5"
)
_SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH = (
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
)
_RECOVERY_REVISION_ENV = "MFT_FULL_RECOVERY_SOURCE_REVISION"
_RECOVERY_BUNDLE_SHA256_ENV = "MFT_FULL_RECOVERY_BUNDLE_SHA256"
_PYAEDT_LIBRARY_ROOT_ENV = "MFT_PYAEDT_LIBRARY_ROOT"
_SOURCE_RECOVERABLE_THERMAL_FAILURE = (
    "required_volume_temperature_missing"
)
# Task 95074 is immutable historical evidence produced by the superseded
# deadline-only k=3 W/(m*K) TIM policy.  Keep that sealed-source contract
# separate from the current fresh-Icepak material constants imported above:
# a recovery may authenticate the old row, but every rebuilt thermal result
# must attest the authoritative k=0.2 W/(m*K) material.
SOURCE_THERMAL_PAD_CONDUCTIVITY_W_MK = 3.0
SOURCE_THERMAL_PAD_MATERIAL_POLICY = (
    "deadline_tim_k3_native_attested_3WmK_electrically_insulating_v2"
)
_SOURCE_EXTRACTION_BACKENDS = {
    "matrix_extraction_backend": "export_rl_matrix",
    "cap_extraction_backend": "export_c_matrix",
    "loss_extraction_backend": "get_solution_data_per_variation",
}
_RX_EXPLICIT_INSULATION_MODEL = (
    "solid_interturn_candidate_k_ins_v1"
)
_RX_EXPLICIT_INSULATION_COUNTS = {
    "Rx_main_insulation": 2,
    "Rx_side_insulation": 2,
    "Rx_side2_insulation": 2,
}
_THERMAL_MESH_EXPECTED_COUNTS = {
    "thermal_mesh_operation_count": 37,
    "thermal_mesh_assigned_object_count": 93,
    "thermal_mesh_required_thin_object_count": 56,
    "thermal_mesh_shared_operation_count": 0,
    "thermal_mesh_separate_object_operation_count": 29,
    "thermal_mesh_object_level_operation_count": 29,
    "thermal_mesh_region_operation_count": 8,
    "thermal_mesh_wcp_pad_region_count": 8,
    "thermal_mesh_core_plate_assembly_count": 15,
    "thermal_mesh_wcp_assembly_count": 4,
    "thermal_mesh_rx_retained_pack_count": 3,
    "thermal_mesh_postsolve_probe_object_count": 9,
    "thermal_mesh_postsolve_probe_missing_count": 0,
    "thermal_mesh_postsolve_probe_complete": 1,
}
_SOURCE_HISTORICAL_THERMAL_POSTSOLVE_COUNTS = {
    "thermal_mesh_postsolve_probe_object_count": 9,
    "thermal_mesh_postsolve_probe_missing_count": 9,
    "thermal_mesh_postsolve_probe_complete": 0,
}


def attest_recovery_runtime(
    *,
    recovery_revision: str,
    bundle_sha256: str,
    repo_root: Path = REPO_ROOT,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind fresh thermal execution to one archive and PyAEDT library clone.

    The submit command verifies and clones an immutable Git bundle.  This
    second fail-closed gate binds its SHA to a clean checkout of the exact
    recovery revision and to the exact clean pyaedt_library checkout used by
    fresh Icepak.
    """
    env = os.environ if environ is None else environ
    revision = str(recovery_revision or "").strip().lower()
    bundle_sha = str(bundle_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("recovery revision must be exact 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha):
        raise RuntimeError("recovery bundle SHA-256 must be exact 64-hex")
    if str(env.get(_RECOVERY_REVISION_ENV, "") or "").strip().lower() != revision:
        raise RuntimeError("recovery revision environment binding failed")
    if (
        str(env.get(_RECOVERY_BUNDLE_SHA256_ENV, "") or "").strip().lower()
        != bundle_sha
    ):
        raise RuntimeError("recovery bundle SHA-256 environment binding failed")

    root_input = Path(repo_root)
    if root_input.is_symlink():
        raise RuntimeError("recovery source root must not be a symlink")
    root = root_input.resolve(strict=True)
    try:
        runtime_revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
        runtime_status = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        lineage = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                _SOURCE_RESULT_GIT_HASH,
                revision,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception as error:
        raise RuntimeError("recovery clean Git checkout is unreadable") from error
    if (
        runtime_revision != revision
        or runtime_status
        or lineage.returncode != 0
        or str(GIT_HASH).strip().lower() != revision
        or int(GIT_DIRTY) != 0
    ):
        raise RuntimeError(
            "fresh thermal solver is not the exact clean recovery revision"
        )

    library_override = str(
        env.get(_PYAEDT_LIBRARY_ROOT_ENV, "") or ""
    ).strip()
    if not library_override:
        raise RuntimeError("fresh thermal requires an explicit library root")
    library_input = Path(library_override)
    if library_input.is_symlink():
        raise RuntimeError("pyaedt_library root must not be a symlink")
    library_path = library_input.resolve(strict=True)
    library_root = (
        library_path.parent
        if library_path.name.casefold() == "src"
        else library_path
    )
    library_src = library_root / "src"
    if (
        not library_root.is_dir()
        or library_src.is_symlink()
        or not library_src.resolve(strict=True).is_dir()
    ):
        raise RuntimeError("pyaedt_library override has no regular src/")
    try:
        library_runtime_revision = subprocess.check_output(
            ["git", "-C", str(library_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
        library_runtime_status = subprocess.check_output(
            [
                "git",
                "-C",
                str(library_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as error:
        raise RuntimeError(
            "fresh thermal pyaedt_library Git checkout is unreadable"
        ) from error
    if (
        library_runtime_revision
        != _SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH
        or library_runtime_status
        or str(PYAEDT_LIBRARY_GIT_HASH).strip().lower()
        != _SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH
        or int(PYAEDT_LIBRARY_GIT_DIRTY) != 0
    ):
        raise RuntimeError(
            "fresh thermal pyaedt_library provenance is not the exact clean "
            "production revision"
        )
    return {
        "schema": "mft-full-recovery-runtime-evidence-v1",
        "provenance_mode": "clean_git_bundle_checkout_and_sha256",
        "recovery_revision": revision,
        "package_sha256": bundle_sha,
        "source_bundle_sha256": bundle_sha,
        "runtime_git_root": str(root),
        "source_revision_is_descendant_of": _SOURCE_RESULT_GIT_HASH,
        "solver_runtime_git_hash": str(GIT_HASH),
        "solver_runtime_git_dirty": int(GIT_DIRTY),
        "pyaedt_library_root": str(library_root),
        "pyaedt_library_runtime_git_hash": library_runtime_revision,
        "pyaedt_library_git_hash": str(PYAEDT_LIBRARY_GIT_HASH),
        "pyaedt_library_git_dirty": int(PYAEDT_LIBRARY_GIT_DIRTY),
    }


def attest_recovery_solver_core_policy(sim: Any) -> dict[str, Any]:
    """Require the authenticated production 16-core standalone policy."""
    policy = dict(getattr(sim, "solver_core_policy", {}) or {})
    expected = {
        "schema": "mft-solver-core-policy-v1",
        "contract_version": STANDALONE_CORE_16_CONTRACT_VERSION,
        "opt_in": True,
        "backend": "standalone",
        "requested_num_cores": 16,
        "effective_num_cores": 16,
        "num_tasks": 1,
        "license_contract": STANDALONE_CORE_16_LICENSE_CONTRACT_VERSION,
        "slurm_cpus_per_task_readback": 16,
        "solver_revision": str(GIT_HASH).strip().lower(),
        "solver_dirty": 0,
    }
    mismatches = {
        name: {"actual": policy.get(name), "expected": value}
        for name, value in expected.items()
        if policy.get(name) != value
    }
    try:
        native_num_core = int(sim.NUM_CORE)
        native_num_task = int(sim.NUM_TASK)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "fresh thermal solver core readback is unavailable"
        ) from error
    if native_num_core != 16 or native_num_task != 1:
        mismatches["native_readback"] = {
            "actual": [native_num_core, native_num_task],
            "expected": [16, 1],
        }
    affinity = policy.get("affinity_count_readback")
    if (
        isinstance(affinity, bool)
        or not isinstance(affinity, int)
        or affinity < 16
    ):
        mismatches["affinity_count_readback"] = {
            "actual": affinity,
            "expected": "integer>=16",
        }
    for identity_name in (
        "scheduler_task_id_readback",
        "slurm_job_id_readback",
    ):
        identity = policy.get(identity_name)
        if (
            isinstance(identity, bool)
            or not isinstance(identity, int)
            or identity < 1
        ):
            mismatches[identity_name] = {
                "actual": identity,
                "expected": "positive integer",
            }
    license_sha = str(
        policy.get("license_snapshot_sha256") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", license_sha):
        mismatches["license_snapshot_sha256"] = {
            "actual": license_sha,
            "expected": "64-hex",
        }
    auth_sha = str(policy.get("auth_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", auth_sha):
        mismatches["auth_sha256"] = {
            "actual": auth_sha,
            "expected": "64-hex",
        }
    try:
        license_age = float(
            policy["license_snapshot_age_seconds_readback"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        license_age = float("nan")
    if (
        not math.isfinite(license_age)
        or license_age < -30.0
        or license_age > 600.0
    ):
        mismatches["license_snapshot_age_seconds_readback"] = {
            "actual": policy.get(
                "license_snapshot_age_seconds_readback"
            ),
            "expected": "finite[-30,600]",
        }
    headroom = policy.get("license_headroom_readback")
    required_headroom = {
        "anshpc": 16,
        "elec_solve_maxwell": 1,
        "electronics_desktop": 1,
        "electronics3d_gui": 1,
    }
    if not isinstance(headroom, dict) or any(
        isinstance(headroom.get(name), bool)
        or not isinstance(headroom.get(name), int)
        or int(headroom[name]) < minimum
        for name, minimum in required_headroom.items()
    ):
        mismatches["license_headroom_readback"] = {
            "actual": headroom,
            "expected_minimum": required_headroom,
        }
    if mismatches:
        raise RuntimeError(
            "fresh thermal solver core policy mismatch: "
            + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
        )
    return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every regular file below one symlink-free directory."""
    candidate = Path(root)
    if candidate.is_symlink():
        raise RuntimeError(f"inventory root is a symlink: {root}")
    directory = candidate.resolve(strict=True)
    if not directory.is_dir():
        raise RuntimeError(f"inventory root is not a regular directory: {root}")
    inventory = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"inventory contains a symlink: {path}")
        if path.is_file():
            inventory.append({
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return inventory


def _directory_metadata_sha256(root: Path) -> str:
    """Return a fast mutation token without rereading multi-GB result files."""
    candidate = Path(root)
    if candidate.is_symlink():
        raise RuntimeError(f"metadata root is a symlink: {root}")
    directory = candidate.resolve(strict=True)
    if not directory.is_dir():
        raise RuntimeError(f"metadata root is not a directory: {root}")
    inventory = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"metadata inventory contains a symlink: {path}")
        if path.is_file():
            stat = path.stat()
            inventory.append({
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    return _canonical_sha256(inventory)


def seal_fresh_thermal_project(
    *,
    source_dir: Path,
    output_dir: Path,
    project_name: str,
    require_gpfs: bool = True,
) -> dict[str, Any]:
    """Atomically preserve the closed fresh Icepak project on durable storage."""
    source_input = Path(source_dir)
    output_input = Path(output_dir)
    if source_input.is_symlink() or output_input.is_symlink():
        raise RuntimeError("thermal project seal roots must not be symlinks")
    source = source_input.resolve(strict=True)
    output = output_input.resolve(strict=True)
    if not source.is_dir() or not output.is_dir():
        raise RuntimeError("thermal project seal roots must be directories")
    if require_gpfs and output.parts[:2] != (os.sep, "gpfs"):
        raise RuntimeError(
            "durable thermal project output must resolve below /gpfs"
        )
    if _is_within(output, source) or _is_within(source, output):
        raise RuntimeError("thermal project source/output must not overlap")

    project = str(project_name or "").strip()
    if not project or Path(project).name != project:
        raise RuntimeError("thermal project seal identity is invalid")
    source_aedt = source / f"{project}.aedt"
    source_results = source / f"{project}.aedtresults"
    if (
        source_aedt.is_symlink()
        or source_results.is_symlink()
        or not source_aedt.resolve(strict=True).is_file()
        or not source_results.resolve(strict=True).is_dir()
        or source_aedt.parent != source_results.parent
    ):
        raise RuntimeError(
            "fresh thermal .aedt/.aedtresults pair is not adjacent and regular"
        )

    source_inventory = _directory_inventory(source)
    if not source_inventory:
        raise RuntimeError("fresh thermal project inventory is empty")
    source_tree_sha = _canonical_sha256(source_inventory)
    target = output / "fresh-thermal-project"
    incoming = output / ".incoming-fresh-thermal-project"
    if (
        target.exists()
        or target.is_symlink()
        or incoming.exists()
        or incoming.is_symlink()
    ):
        raise RuntimeError(
            "durable thermal project target/incoming path already exists"
        )
    shutil.copytree(source, incoming, symlinks=False, copy_function=shutil.copy2)
    incoming_inventory = _directory_inventory(incoming)
    if (
        incoming_inventory != source_inventory
        or _canonical_sha256(incoming_inventory) != source_tree_sha
    ):
        raise RuntimeError("durable thermal project copy changed inventory")

    if os.name != "nt":
        for path in sorted(incoming.rglob("*"), reverse=True):
            if path.is_symlink():
                raise RuntimeError(
                    "durable thermal project copy contains a symlink"
                )
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(incoming, 0o555)
    os.replace(incoming, target)
    durable_aedt = target / source_aedt.name
    durable_results = target / source_results.name
    if (
        durable_aedt.is_symlink()
        or durable_results.is_symlink()
        or not durable_aedt.resolve(strict=True).is_file()
        or not durable_results.resolve(strict=True).is_dir()
        or durable_aedt.parent != durable_results.parent
    ):
        raise RuntimeError("durable thermal project adjacency check failed")
    durable_inventory = _directory_inventory(target)
    durable_tree_sha = _canonical_sha256(durable_inventory)
    if (
        durable_inventory != source_inventory
        or durable_tree_sha != source_tree_sha
    ):
        raise RuntimeError("durable thermal project post-publish audit failed")
    if os.name != "nt":
        writable = [
            item["path"]
            for item in durable_inventory
            if (target / item["path"]).stat().st_mode & 0o222
        ]
        writable_dirs = [
            path.relative_to(target).as_posix() or "."
            for path in [target, *target.rglob("*")]
            if path.is_dir() and path.stat().st_mode & 0o222
        ]
        if writable or writable_dirs:
            raise RuntimeError(
                "durable thermal project seal is unexpectedly writable"
            )
    results_prefix = source_results.name + "/"
    results_inventory = [
        item for item in durable_inventory
        if item["path"].startswith(results_prefix)
    ]
    return {
        "schema": "mft-fresh-thermal-project-seal-v1",
        "status": "atomically_published_read_only",
        "atomic_publish": True,
        "storage_root_contract": "gpfs" if require_gpfs else "test_non_gpfs",
        "resolved_output_dir": str(output),
        "read_only_attested": os.name != "nt",
        "path": str(target),
        "aedt": str(durable_aedt),
        "aedt_sha256": _sha256(durable_aedt),
        "aedt_size_bytes": durable_aedt.stat().st_size,
        "aedtresults": str(durable_results),
        "aedtresults_file_count": len(results_inventory),
        "aedtresults_total_bytes": sum(
            int(item["size_bytes"]) for item in results_inventory
        ),
        "aedtresults_tree_sha256": _canonical_sha256(results_inventory),
        "file_count": len(durable_inventory),
        "total_bytes": sum(
            int(item["size_bytes"]) for item in durable_inventory
        ),
        "tree_sha256": durable_tree_sha,
        "files": durable_inventory,
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _load_bound_json_object(
    path: Path,
    label: str,
    *,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
) -> dict[str, Any]:
    """Read once and bind both exact bytes and canonical JSON identity."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    if (
        hashlib.sha256(raw).hexdigest()
        != str(expected_raw_sha256 or "").strip().lower()
        or _canonical_sha256(payload)
        != str(expected_canonical_sha256 or "").strip().lower()
    ):
        raise RuntimeError(f"{label} hash binding changed during recovery")
    return payload


def _relative_manifest_path(value: Any) -> Path:
    token = str(value or "")
    if "\\" in token:
        raise RuntimeError(f"unsafe manifest-relative path: {value!r}")
    relative = Path(token)
    if (
        not token
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise RuntimeError(f"unsafe manifest-relative path: {value!r}")
    return relative


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _attest_scheduler_status_evidence(
    *,
    evidence: dict[str, Any],
    bundle_root: Path,
    label: str,
    expected_task_id: int,
    expected_status: str,
    expected_exit_code: Any,
    expected_finished_at: str,
    expected_account: str,
    expected_allocation_id: int,
    expected_slurm_job_id: str,
    expected_node: str,
) -> dict[str, Any]:
    """Verify one immutable, unwrapped Scheduler task-status response."""
    relative = _relative_manifest_path(evidence.get("path"))
    path = bundle_root / relative
    if path.is_symlink():
        raise RuntimeError(f"{label} status evidence is a symlink")
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_file()
        or not _is_within(resolved, bundle_root)
    ):
        raise RuntimeError(
            f"{label} status evidence escaped its immutable bundle"
        )
    recorded_file_sha = str(
        evidence.get("file_sha256") or ""
    ).strip().lower()
    recorded_canonical_sha = str(
        evidence.get("canonical_sha256") or ""
    ).strip().lower()
    status = _load_json_object(resolved, f"{label} Scheduler status")
    if any(
        key in status
        for key in (
            "file_sha256",
            "canonical_sha256",
            "status_sha256",
            "scheduler_status_sha256",
        )
    ):
        raise RuntimeError(
            f"{label} Scheduler status contains a self-hash field"
        )
    if (
        _sha256(resolved) != recorded_file_sha
        or _canonical_sha256(status) != recorded_canonical_sha
    ):
        raise RuntimeError(f"{label} Scheduler status hash mismatch")

    integer_fields = (
        "id",
        "task_id",
        "allocation_id",
        "assigned_allocation",
    )
    string_fields = (
        "name",
        "status",
        "state",
        "finished_at",
        "account_name",
        "requested_account_name",
        "slurm_job_id",
        "node_name",
        "allocation_node_name",
        "actual_node_name",
        "dedupe_key",
    )
    if any(type(status.get(name)) is not int for name in integer_fields):
        raise RuntimeError(
            f"{label} Scheduler status integer schema mismatch"
        )
    if any(not isinstance(status.get(name), str) for name in string_fields):
        raise RuntimeError(
            f"{label} Scheduler status string schema mismatch"
        )
    if (
        status.get("started_at") is not None
        and not isinstance(status.get("started_at"), str)
    ):
        raise RuntimeError(
            f"{label} Scheduler started_at schema mismatch"
        )
    status_exit_code = status.get("exit_code")
    if (
        status_exit_code is not None
        and type(status_exit_code) is not int
    ):
        raise RuntimeError(
            f"{label} Scheduler exit_code schema mismatch"
        )
    if (
        status["id"] != int(expected_task_id)
        or status["task_id"] != int(expected_task_id)
        or status["status"] != expected_status
        or status["state"] != (
            "succeeded"
            if expected_status == "completed"
            else expected_status
        )
        or status_exit_code != expected_exit_code
        or not status["finished_at"]
        or status["finished_at"] != expected_finished_at
        or status["allocation_id"] != int(expected_allocation_id)
        or status["assigned_allocation"] != int(expected_allocation_id)
        or status["account_name"] != expected_account
        or status["requested_account_name"] != expected_account
        or status["slurm_job_id"] != expected_slurm_job_id
    ):
        raise RuntimeError(
            f"{label} Scheduler execution identity mismatch"
        )
    nodes = [
        status[name]
        for name in (
            "node_name",
            "allocation_node_name",
            "actual_node_name",
        )
        if status[name]
    ]
    if not nodes or any(node != expected_node for node in nodes):
        raise RuntimeError(
            f"{label} Scheduler node identity mismatch: {nodes}"
        )
    return status


def attest_preserved_seal(
    *,
    mirror_root: Path,
    manifest_path: Path,
    source_identity_path: Path,
    params_path: Path,
    project_name: str,
    expected_source_task_id: int,
    expected_manifest_sha256: str,
    expected_source_identity_sha256: str,
    expected_params_sha256: str,
    expected_params_canonical_sha256: str,
    expected_candidate_digest: str,
    expected_profile_sha256: str,
) -> dict[str, Any]:
    """Verify one terminal, immutable, complete preserver publication."""
    root = mirror_root.resolve(strict=True)
    if not root.is_dir() or root.name != "current":
        raise RuntimeError("mirror root must be the atomically published current/")

    if manifest_path.is_symlink() or source_identity_path.is_symlink():
        raise RuntimeError(
            "manifest/source identity must not be symlinks"
        )
    manifest_file = manifest_path.resolve(strict=True)
    identity_file = source_identity_path.resolve(strict=True)
    if (
        manifest_file.parent != root.parent
        or identity_file.parent != root.parent
    ):
        raise RuntimeError(
            "manifest/source identity must be adjacent to sealed current/"
        )
    manifest = _load_json_object(manifest_file, "preserver manifest")
    identity = _load_json_object(identity_file, "source identity")

    declared_manifest_sha = str(
        manifest.get("manifest_sha256") or ""
    ).strip().lower()
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    actual_manifest_sha = _canonical_sha256(unsigned_manifest)
    if (
        len(declared_manifest_sha) != 64
        or declared_manifest_sha != actual_manifest_sha
        or declared_manifest_sha
        != str(expected_manifest_sha256 or "").strip().lower()
    ):
        raise RuntimeError("preserver manifest SHA-256 attestation failed")

    declared_identity_sha = str(
        identity.get("identity_sha256") or ""
    ).strip().lower()
    unsigned_identity = dict(identity)
    unsigned_identity.pop("identity_sha256", None)
    actual_identity_sha = _canonical_sha256(unsigned_identity)
    if (
        len(declared_identity_sha) != 64
        or declared_identity_sha != actual_identity_sha
        or declared_identity_sha
        != str(expected_source_identity_sha256 or "").strip().lower()
    ):
        raise RuntimeError("source identity SHA-256 attestation failed")

    source_task_id = int(manifest.get("source_task_id") or 0)
    if (
        manifest.get("schema_version") != PRESERVER_MANIFEST_SCHEMA
        or identity.get("schema_version") != PRESERVER_IDENTITY_SCHEMA
        or source_task_id != int(expected_source_task_id)
        or int(identity.get("source_task_id") or 0) != source_task_id
        or manifest.get("source_identity_sha256") != declared_identity_sha
    ):
        raise RuntimeError("preserver manifest/source identity binding failed")
    source_account = str(
        identity.get("source_account_name") or ""
    ).strip()
    source_node = str(identity.get("source_node_name") or "").strip()
    source_slurm_job_id = str(
        identity.get("source_slurm_job_id") or ""
    ).strip()
    try:
        source_allocation_id = int(
            identity.get("source_allocation_id") or 0
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "source allocation identity is invalid"
        ) from error
    if (
        not source_account
        or not source_node
        or not source_slurm_job_id
        or source_allocation_id <= 0
    ):
        raise RuntimeError("source Scheduler identity is incomplete")
    if manifest.get("termination_reason") not in ALLOWED_TERMINATION_REASONS:
        raise RuntimeError(
            "preserved workdir is not terminal: "
            f"termination_reason={manifest.get('termination_reason')!r}"
        )
    if (
        manifest.get("aedt_and_aedtresults_adjacent_atomic_publish") is not True
        or manifest.get("source_task_mutated") is not False
        or manifest.get("source_task_cancelled") is not False
    ):
        raise RuntimeError("preserver immutability contract failed")
    recorded_current = Path(
        str(manifest.get("current_directory") or "")
    ).resolve()
    if recorded_current != root:
        raise RuntimeError(
            "preserver current-directory identity mismatch: "
            f"expected={root}, recorded={recorded_current}"
        )

    source_payload = identity.get("source_payload")
    if not isinstance(source_payload, dict):
        raise RuntimeError("source identity payload is unavailable")
    if (
        source_payload.get("fidelity") != "actual_full_geometry"
        or source_payload.get("complete_pipeline") is not True
        or source_payload.get("matrix_enabled") is not True
        or source_payload.get("capacitance_enabled") is not True
        or source_payload.get("loss_enabled") is not True
        or source_payload.get("thermal_enabled") is not True
        or int(source_payload.get("n_explicit_turns") or 0) != 2
        or source_payload.get("automatic_promotion_allowed") is not False
        or source_payload.get("canonical_dataset_mutation_allowed") is not False
    ):
        raise RuntimeError("source identity is not a complete actual-Full task")
    source_payload_sha = _canonical_sha256(source_payload)
    if source_payload_sha != str(
        identity.get("source_payload_sha256") or ""
    ).lower():
        raise RuntimeError("source payload SHA-256 attestation failed")
    stable_binding = {
        "schema_version": "mft-deadline-full-preserver-binding-v1",
        "source_task_id": source_task_id,
        "source_account_name": identity.get("source_account_name"),
        "source_node_name": identity.get("source_node_name"),
        "source_allocation_id": identity.get("source_allocation_id"),
        "source_slurm_job_id": identity.get("source_slurm_job_id"),
        "source_workdir": identity.get("source_workdir"),
        "source_command_sha256": identity.get("source_command_sha256"),
        "source_payload_sha256": identity.get("source_payload_sha256"),
    }
    expected_leaf = (
        f"mft_full_recovery_task_{source_task_id}_"
        f"{_canonical_sha256(stable_binding)[:16]}"
    )
    if root.parent.name != expected_leaf:
        raise RuntimeError(
            "sealed root stable-binding token mismatch: "
            f"expected={expected_leaf!r}, actual={root.parent.name!r}"
        )
    candidate_digest = str(expected_candidate_digest or "").strip().lower()
    profile_sha = str(expected_profile_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_digest):
        raise RuntimeError("expected candidate digest is not SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", profile_sha):
        raise RuntimeError("expected profile digest is not SHA-256")
    if str(
        source_payload.get("physical_candidate_digest") or ""
    ).lower() != candidate_digest:
        raise RuntimeError("source candidate digest mismatch")
    if str(
        source_payload.get("followup_profile_sha256") or ""
    ).lower() != profile_sha:
        raise RuntimeError("source follow-up profile mismatch")
    try:
        replica_of_task_id = int(
            source_payload.get("replica_of_task_id") or 0
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("source replica provenance is invalid") from error
    replica_execution_sha = str(
        source_payload.get("replica_execution_sha256") or ""
    ).strip().lower()
    if replica_of_task_id < 0 or (
        replica_of_task_id > 0
        and not re.fullmatch(r"[0-9a-f]{64}", replica_execution_sha)
    ) or (replica_of_task_id == 0 and replica_execution_sha):
        raise RuntimeError("source replica provenance is incomplete")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("preserver manifest contains no files")
    declared_files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("preserver manifest file entry is invalid")
        relative = _relative_manifest_path(entry.get("path"))
        key = relative.as_posix()
        if key in declared_files:
            raise RuntimeError(f"duplicate manifest file path: {key}")
        declared_files[key] = entry

    actual_files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"sealed current contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            actual_files[relative] = path
    if set(actual_files) != set(declared_files):
        missing = sorted(set(declared_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(declared_files))
        raise RuntimeError(
            f"sealed current file-set mismatch: missing={missing}, extra={extra}"
        )
    if [entry["path"] for entry in entries] != sorted(declared_files):
        raise RuntimeError("preserver manifest file entries are not sorted")
    for relative, path in actual_files.items():
        entry = declared_files[relative]
        try:
            expected_size = int(entry["size_bytes"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                f"invalid manifest size for {relative}"
            ) from error
        if (
            path.stat().st_size != expected_size
            or _sha256(path) != str(entry.get("sha256") or "").lower()
        ):
            raise RuntimeError(
                f"sealed current file hash/size mismatch: {relative}"
            )
    total_bytes = sum(path.stat().st_size for path in actual_files.values())
    if (
        int(manifest.get("file_count") or -1) != len(actual_files)
        or int(manifest.get("total_bytes") or -1) != total_bytes
    ):
        raise RuntimeError("preserver manifest file-count/byte totals mismatch")

    source_result_entry = declared_files.get(
        _SEALED_SOURCE_RESULT_RELATIVE_PATH
    )
    if source_result_entry is None:
        raise RuntimeError(
            "sealed manifest does not contain the exact source result CSV: "
            f"{_SEALED_SOURCE_RESULT_RELATIVE_PATH}"
        )
    source_result_path = actual_files[
        _SEALED_SOURCE_RESULT_RELATIVE_PATH
    ]
    source_result_sha = str(
        source_result_entry.get("sha256") or ""
    ).strip().lower()
    source_result_size = int(source_result_entry["size_bytes"])
    if (
        not re.fullmatch(r"[0-9a-f]{64}", source_result_sha)
        or source_result_path.stat().st_size != source_result_size
        or _sha256(source_result_path) != source_result_sha
    ):
        raise RuntimeError(
            "sealed source result CSV identity mismatch"
        )

    requested_project = str(project_name or "").strip()
    project_entries = [
        relative
        for relative in declared_files
        if Path(relative).suffix.casefold() == ".aedt"
        and Path(relative).stem == requested_project
    ]
    if len(project_entries) != 1:
        raise RuntimeError(
            "sealed manifest does not contain one exact requested project: "
            f"{project_entries}"
        )
    project_relative = Path(project_entries[0])
    results_relative = project_relative.with_name(
        project_relative.stem + ".aedtresults"
    )
    results_prefix = results_relative.as_posix().rstrip("/") + "/"
    if not any(
        relative.startswith(results_prefix) for relative in declared_files
    ):
        raise RuntimeError(
            "sealed manifest contains no adjacent project results payload"
        )
    last_sync = manifest.get("last_sync")
    if (
        not isinstance(last_sync, dict)
        or project_relative.as_posix()
        not in list(last_sync.get("project_paths") or [])
        or results_relative.as_posix()
        not in list(last_sync.get("aedtresults_paths") or [])
    ):
        raise RuntimeError("sealed project is absent from final sync identity")

    if params_path.is_symlink():
        raise RuntimeError("--params must not be a symlink")
    params = params_path.resolve(strict=True)
    if not params.is_file() or not _is_within(
        params, root
    ):
        raise RuntimeError("--params must be a regular file inside sealed current/")
    params_relative = params.relative_to(root).as_posix()
    if params_relative not in declared_files:
        raise RuntimeError("parameter file is absent from sealed manifest")
    expected_params_sha = str(expected_params_sha256 or "").strip().lower()
    expected_params_canonical_sha = str(
        expected_params_canonical_sha256 or ""
    ).strip().lower()
    params_payload = _load_json_object(params, "sealed parameter payload")
    actual_params_canonical_sha = _canonical_sha256(params_payload)
    if (
        len(expected_params_sha) != 64
        or _sha256(params) != expected_params_sha
        or str(declared_files[params_relative].get("sha256") or "").lower()
        != expected_params_sha
    ):
        raise RuntimeError("sealed parameter-file SHA-256 binding failed")
    if (
        len(expected_params_canonical_sha) != 64
        or actual_params_canonical_sha != expected_params_canonical_sha
    ):
        raise RuntimeError("sealed parameter canonical SHA-256 binding failed")

    project_entry = declared_files[project_relative.as_posix()]
    project_aedt_sha = str(project_entry.get("sha256") or "").lower()
    results_inventory = []
    for relative in sorted(declared_files):
        if not relative.startswith(results_prefix):
            continue
        entry = declared_files[relative]
        results_inventory.append({
            "path": relative[len(results_prefix):],
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry.get("sha256") or "").lower(),
        })

    return {
        "source_task_id": source_task_id,
        "manifest_sha256": actual_manifest_sha,
        "source_identity_sha256": actual_identity_sha,
        "termination_reason": manifest["termination_reason"],
        "project_relative_path": project_relative.as_posix(),
        "params_relative_path": params_relative,
        "params_sha256": expected_params_sha,
        "params_canonical_sha256": actual_params_canonical_sha,
        "source_result_relative_path": (
            _SEALED_SOURCE_RESULT_RELATIVE_PATH
        ),
        "source_result_sha256": source_result_sha,
        "source_result_size_bytes": source_result_size,
        "source_payload_sha256": source_payload_sha,
        "source_account_name": source_account,
        "source_node_name": source_node,
        "source_allocation_id": source_allocation_id,
        "source_slurm_job_id": source_slurm_job_id,
        "candidate_digest": candidate_digest,
        "profile_sha256": profile_sha,
        "replica_of_task_id": replica_of_task_id,
        "replica_execution_sha256": replica_execution_sha,
        "project_aedt_sha256": project_aedt_sha,
        "results_inventory_sha256": _canonical_sha256(
            results_inventory
        ),
        "results_file_count": len(results_inventory),
        "file_count": len(actual_files),
        "total_bytes": total_bytes,
    }


def attest_terminal_watch(
    *,
    terminal_watch_path: Path,
    expected_terminal_watch_sha256: str,
    sealed_root: Path,
    seal: dict[str, Any],
    project_name: str,
) -> dict[str, Any]:
    """Require independent Scheduler evidence that source and sidecar ended."""
    terminal_watch_candidate = Path(terminal_watch_path)
    if terminal_watch_candidate.is_symlink():
        raise RuntimeError("terminal-watch result must not be a symlink")
    terminal_watch_file = terminal_watch_candidate.resolve(strict=True)
    if not terminal_watch_file.is_file():
        raise RuntimeError("terminal-watch result is not a regular file")
    bundle_root = terminal_watch_file.parent.resolve(strict=True)
    payload = _load_json_object(
        terminal_watch_file, "terminal-watch result"
    )
    declared_sha = str(
        payload.get("terminal_watch_sha256") or ""
    ).strip().lower()
    unsigned = dict(payload)
    unsigned.pop("terminal_watch_sha256", None)
    actual_sha = _canonical_sha256(unsigned)
    if (
        payload.get("schema_version") != TERMINAL_WATCH_SCHEMA
        or not re.fullmatch(r"[0-9a-f]{64}", declared_sha)
        or declared_sha != actual_sha
        or declared_sha
        != str(expected_terminal_watch_sha256 or "").strip().lower()
    ):
        raise RuntimeError("terminal-watch canonical SHA-256 attestation failed")

    execution = payload.get("execution")
    watch_seal = payload.get("seal")
    watch_identity = payload.get("identity")
    validation = payload.get("validation")
    if not all(
        isinstance(item, dict)
        for item in (execution, watch_seal, watch_identity, validation)
    ):
        raise RuntimeError("terminal-watch result sections are incomplete")
    source_status = execution.get("source_status")
    sidecar_status = execution.get("sidecar_status")
    source_exit_code = execution.get("source_exit_code")
    if (
        not isinstance(source_status, str)
        or not isinstance(sidecar_status, str)
        or (
            source_exit_code is not None
            and type(source_exit_code) is not int
        )
    ):
        raise RuntimeError("Scheduler source/sidecar status schema failed")
    try:
        if type(execution["sidecar_exit_code"]) is not int:
            raise TypeError("sidecar exit code is not an integer")
        sidecar_exit_code = execution["sidecar_exit_code"]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "Scheduler sidecar exit code is unavailable"
        ) from error
    if (
        type(execution.get("source_task_id")) is not int
        or int(execution["source_task_id"])
        != int(seal["source_task_id"])
        or source_status
        not in {"completed", "failed", "cancelled"}
        or not str(execution.get("source_finished_at") or "").strip()
        or type(execution.get("sidecar_task_id")) is not int
        or int(execution["sidecar_task_id"]) <= 0
        or sidecar_status != "completed"
        or sidecar_exit_code != 0
        or not str(execution.get("sidecar_finished_at") or "").strip()
    ):
        raise RuntimeError("Scheduler source/sidecar terminal contract failed")
    status_evidence = {}
    for label in ("source_status_evidence", "sidecar_status_evidence"):
        evidence = execution.get(label)
        if not isinstance(evidence, dict):
            raise RuntimeError(f"terminal-watch {label} is unavailable")
        status_evidence[label] = evidence
    source_status_record = _attest_scheduler_status_evidence(
        evidence=status_evidence["source_status_evidence"],
        bundle_root=bundle_root,
        label="source",
        expected_task_id=int(execution["source_task_id"]),
        expected_status=source_status,
        expected_exit_code=source_exit_code,
        expected_finished_at=str(execution["source_finished_at"]),
        expected_account=seal["source_account_name"],
        expected_allocation_id=int(seal["source_allocation_id"]),
        expected_slurm_job_id=seal["source_slurm_job_id"],
        expected_node=seal["source_node_name"],
    )
    sidecar_status_record = _attest_scheduler_status_evidence(
        evidence=status_evidence["sidecar_status_evidence"],
        bundle_root=bundle_root,
        label="sidecar",
        expected_task_id=int(execution["sidecar_task_id"]),
        expected_status=sidecar_status,
        expected_exit_code=sidecar_exit_code,
        expected_finished_at=str(execution["sidecar_finished_at"]),
        expected_account=seal["source_account_name"],
        expected_allocation_id=int(seal["source_allocation_id"]),
        expected_slurm_job_id=seal["source_slurm_job_id"],
        expected_node=seal["source_node_name"],
    )

    required_validation = (
        "source_terminal",
        "sidecar_completed_zero",
        "one_preserver_result",
        "zero_preserver_fatal",
        "execution_id_bound",
        "manifest_id_bound",
        "source_identity_id_bound",
        "current_is_atomic_leaf",
        "manifest_valid",
        "source_identity_valid",
        "project_and_results_adjacent",
        "params_in_manifest",
        "params_canonical_bound",
        "all_pass",
    )
    failed_checks = [
        name for name in required_validation
        if validation.get(name) is not True
    ]
    if (
        failed_checks
        or payload.get("source_task_mutated") is not False
        or payload.get("sidecar_task_mutated") is not False
    ):
        raise RuntimeError(
            "terminal-watch validation failed: "
            + ", ".join(failed_checks or ["mutation flag"])
        )

    sealed = sealed_root.resolve(strict=True)
    current = sealed / "current"
    manifest_evidence = watch_seal.get("manifest")
    identity_evidence = watch_seal.get("source_identity")
    project_evidence = watch_seal.get("project")
    params_evidence = watch_seal.get("params")
    if not all(
        isinstance(item, dict)
        for item in (
            manifest_evidence,
            identity_evidence,
            project_evidence,
            params_evidence,
        )
    ):
        raise RuntimeError("terminal-watch seal evidence is incomplete")
    if (
        Path(str(watch_seal.get("gpfs_root") or "")).resolve() != sealed
        or Path(
            str(watch_seal.get("current_directory") or "")
        ).resolve() != current.resolve()
        or int(watch_seal.get("file_count") or -1) != int(seal["file_count"])
        or str(
            manifest_evidence.get("recorded_manifest_sha256") or ""
        ).lower() != seal["manifest_sha256"]
        or str(
            identity_evidence.get("recorded_identity_sha256") or ""
        ).lower() != seal["source_identity_sha256"]
    ):
        raise RuntimeError("terminal-watch preserver seal binding failed")
    for evidence, expected_sha, expected_path, label in (
        (
            manifest_evidence,
            seal["manifest_sha256"],
            sealed / "manifest.json",
            "manifest",
        ),
        (
            identity_evidence,
            seal["source_identity_sha256"],
            sealed / "source_identity.json",
            "source identity",
        ),
    ):
        evidence_candidate = Path(str(evidence.get("path") or ""))
        if evidence_candidate.is_symlink():
            raise RuntimeError(
                f"terminal-watch {label} evidence is a symlink"
            )
        evidence_path = evidence_candidate.resolve(strict=True)
        if (
            evidence_path != expected_path.resolve(strict=True)
            or not evidence_path.is_file()
            or _sha256(evidence_path)
            != str(evidence.get("file_sha256") or "").lower()
            or str(evidence.get("canonical_sha256") or "").lower()
            != expected_sha
        ):
            raise RuntimeError(
                f"terminal-watch {label} file evidence mismatch"
            )

    project_relative = str(
        project_evidence.get("relative_path") or ""
    ).replace("\\", "/")
    results_relative = str(
        project_evidence.get("aedtresults_relative_path") or ""
    ).replace("\\", "/")
    expected_results_relative = str(
        Path(seal["project_relative_path"]).with_name(
            project_name + ".aedtresults"
        ).as_posix()
    )
    if (
        project_evidence.get("stem") != project_name
        or project_relative != seal["project_relative_path"]
        or results_relative != expected_results_relative
        or Path(
            str(project_evidence.get("absolute_path") or "")
        ).resolve() != (current / project_relative).resolve()
        or Path(
            str(project_evidence.get("aedtresults_absolute_path") or "")
        ).resolve() != (current / results_relative).resolve()
    ):
        raise RuntimeError("terminal-watch project/results identity mismatch")

    params_relative = str(
        params_evidence.get("relative_path") or ""
    ).replace("\\", "/")
    if (
        params_relative != seal["params_relative_path"]
        or params_relative != "repo/cand.json"
        or Path(
            str(params_evidence.get("absolute_path") or "")
        ).resolve() != (current / params_relative).resolve()
        or str(params_evidence.get("raw_sha256") or "").lower()
        != seal["params_sha256"]
        or str(params_evidence.get("canonical_sha256") or "").lower()
        != seal["params_canonical_sha256"]
    ):
        raise RuntimeError("terminal-watch parameter identity mismatch")
    if (
        str(watch_identity.get("physical_candidate_digest") or "").lower()
        != seal["candidate_digest"]
        or str(watch_identity.get("followup_profile_sha256") or "").lower()
        != seal["profile_sha256"]
        or str(watch_identity.get("source_payload_sha256") or "").lower()
        != seal["source_payload_sha256"]
    ):
        raise RuntimeError("terminal-watch candidate/profile identity mismatch")
    if type(watch_identity.get("replica_of_task_id")) is not int:
        raise RuntimeError("terminal-watch replica provenance schema mismatch")
    watch_replica_of = int(watch_identity["replica_of_task_id"])
    watch_replica_sha = str(
        watch_identity.get("replica_execution_sha256") or ""
    ).strip().lower()
    if (
        watch_replica_of != int(seal["replica_of_task_id"])
        or watch_replica_sha != seal["replica_execution_sha256"]
        or (
            watch_replica_of > 0
            and not re.fullmatch(r"[0-9a-f]{64}", watch_replica_sha)
        )
        or (watch_replica_of == 0 and watch_replica_sha)
    ):
        raise RuntimeError("terminal-watch replica provenance mismatch")

    return {
        "terminal_watch_sha256": actual_sha,
        "source_status": source_status,
        "source_exit_code": execution.get("source_exit_code"),
        "source_finished_at": execution["source_finished_at"],
        "sidecar_task_id": int(execution["sidecar_task_id"]),
        "sidecar_status": sidecar_status,
        "sidecar_exit_code": sidecar_exit_code,
        "sidecar_finished_at": execution["sidecar_finished_at"],
        "source_status_evidence_sha256": _canonical_sha256(
            source_status_record
        ),
        "sidecar_status_evidence_sha256": _canonical_sha256(
            sidecar_status_record
        ),
        "replica_of_task_id": watch_replica_of,
        "replica_execution_sha256": watch_replica_sha,
    }


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def discover_preserved_project(
    mirror_root: Path, project_name: str = ""
) -> tuple[Path, Path]:
    """Return exactly one project and its adjacent results sidecar."""
    root = mirror_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"preserved mirror root is not a directory: {root}")
    requested = str(project_name or "").strip()
    candidates = []
    for aedt_path in sorted(root.rglob("*.aedt")):
        if aedt_path.is_symlink() or not aedt_path.is_file():
            continue
        if requested and aedt_path.stem != requested:
            continue
        results_path = aedt_path.with_name(aedt_path.stem + ".aedtresults")
        if results_path.is_dir() and not results_path.is_symlink():
            candidates.append((aedt_path.resolve(), results_path.resolve()))
    if len(candidates) != 1:
        rendered = [str(item[0]) for item in candidates]
        raise RuntimeError(
            "expected exactly one preserved AEDT project with an adjacent "
            f".aedtresults sidecar, found {len(candidates)}: {rendered}"
        )
    return candidates[0]


def stage_preserved_project(
    aedt_path: Path, results_path: Path, work_dir: Path
) -> tuple[Path, Path]:
    """Atomically stage the immutable pair in a fresh writable workspace."""
    source_aedt_sha = _sha256(aedt_path)
    source_results_inventory = _directory_inventory(results_path)
    destination = work_dir.resolve()
    if destination.exists():
        raise RuntimeError(
            f"recovery workspace must not already exist: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.with_name(
        destination.name + f".incoming.{os.getpid()}"
    )
    if incoming.exists():
        raise RuntimeError(
            f"recovery staging path already exists: {incoming}"
        )
    incoming.mkdir()
    staged_aedt = destination / aedt_path.name
    staged_results = destination / results_path.name
    try:
        shutil.copy2(aedt_path, incoming / aedt_path.name)
        shutil.copytree(
            results_path,
            incoming / results_path.name,
            copy_function=shutil.copy2,
        )
        if (
            _sha256(incoming / aedt_path.name) != source_aedt_sha
            or _directory_inventory(incoming / results_path.name)
            != source_results_inventory
        ):
            raise RuntimeError(
                "staged AEDT project/results integrity verification failed"
            )
        os.replace(incoming, destination)
    except Exception:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    return staged_aedt, staged_results


def _native_project_file(native_project: Any) -> Path:
    """Resolve the exact AEDT file represented by one native project proxy."""
    project_name = str(native_project.GetName() or "").strip()
    project_directory = str(native_project.GetPath() or "").strip()
    if not project_name or not project_directory:
        raise RuntimeError(
            "native AEDT project path identity is incomplete"
        )
    return (
        Path(project_directory) / f"{project_name}.aedt"
    ).resolve(strict=True)


def _exact_native_designs(
    native_project: Any,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    raw_designs = tuple(native_project.GetDesigns() or ())
    design_names = tuple(
        _aedt_design_name(item) for item in raw_designs
    )
    if (
        any(not name for name in design_names)
        or len(design_names) != len(set(design_names))
    ):
        raise RuntimeError(
            "preserved project contains empty or duplicate design names: "
            f"{design_names!r}"
        )
    return design_names, dict(zip(design_names, raw_designs))


def _validate_exact_thermal_design(
    native_project: Any,
    *,
    expected_design_names: set[str],
) -> dict[str, Any]:
    """Require one exact production Icepak design and ThermalSetup."""
    design_names, designs = _exact_native_designs(native_project)
    if (
        len(design_names) != len(expected_design_names)
        or set(design_names) != expected_design_names
    ):
        raise RuntimeError(
            "project does not contain the exact recovery design set: "
            f"{design_names!r}"
        )
    thermal_design = designs.get(_RECOVERY_THERMAL_DESIGN)
    if thermal_design is None:
        raise RuntimeError("project contains no exact icepak_thermal design")
    design_type = str(thermal_design.GetDesignType() or "").strip()
    solution_type = str(thermal_design.GetSolutionType() or "").strip()
    analysis = thermal_design.GetModule("AnalysisSetup")
    setups = tuple(
        str(name) for name in (analysis.GetSetups() or ())
    )
    if (
        design_type != _RECOVERY_THERMAL_DESIGN_TYPE
        or solution_type != _RECOVERY_THERMAL_NATIVE_SOLUTION_TYPE
        or setups != (_RECOVERY_THERMAL_SETUP,)
    ):
        raise RuntimeError(
            "preserved thermal design identity mismatch: "
            f"name={_RECOVERY_THERMAL_DESIGN!r}, "
            f"type={design_type!r}, solution={solution_type!r}, "
            f"setups={setups!r}"
        )
    return {
        "design_name": _RECOVERY_THERMAL_DESIGN,
        "design_type": design_type,
        "solution_type": solution_type,
        "setups": list(setups),
    }


def prepare_staged_thermal_rebuild(
    project: Any,
    *,
    staged_aedt: Path,
    sealed_source_aedt: Path,
    sealed_root: Path,
    expected_source_aedt_sha256: str,
) -> dict[str, Any]:
    """Validate and, if present, remove thermal only from the staged copy.

    A successfully solved thermal design can legitimately be present in a
    preserved post-solve project.  Recovery must rebuild it because the saved
    temperature extraction may be incomplete, but deletion is permitted only
    after proving that the native AEDT proxy points at the fresh staged file.
    """
    staged = Path(staged_aedt).resolve(strict=True)
    source = Path(sealed_source_aedt).resolve(strict=True)
    protected_root = Path(sealed_root).resolve(strict=True)
    if (
        staged == source
        or _is_within(staged, protected_root)
        or not _is_within(source, protected_root)
    ):
        raise RuntimeError(
            "thermal rebuild paths do not prove a distinct staged copy"
        )
    native_project = project.project
    native_file = _native_project_file(native_project)
    if native_file != staged:
        raise RuntimeError(
            "native AEDT project is not bound to the exact staged copy: "
            f"native={native_file}, staged={staged}"
        )
    expected_sha = str(expected_source_aedt_sha256 or "").strip().lower()
    if _sha256(source) != expected_sha:
        raise RuntimeError(
            "sealed source AEDT changed before thermal preparation"
        )

    design_names, _designs = _exact_native_designs(native_project)
    maxwell_set = set(_RECOVERY_MAXWELL_DESIGNS)
    with_thermal_set = maxwell_set | {_RECOVERY_THERMAL_DESIGN}
    lifecycle = {
        "source_design_names": list(design_names),
        "staged_project_path": str(staged),
        "sealed_source_project_path": str(source),
        "native_project_path": str(native_file),
        "native_project_path_bound_to_staged_copy": True,
        "preserved_thermal_present": False,
        "preserved_thermal_validated": False,
        "preserved_thermal_action": "absent_build_new",
        "preserved_thermal_deleted_from_staged_copy": False,
        "sealed_source_mutated": False,
        "sealed_source_aedt_sha256_before": expected_sha,
    }
    if len(design_names) == len(maxwell_set) and set(design_names) == maxwell_set:
        return lifecycle
    if (
        len(design_names) != len(with_thermal_set)
        or set(design_names) != with_thermal_set
    ):
        raise RuntimeError(
            "preserved project is not an exact Full recovery design set: "
            f"{design_names!r}"
        )

    thermal_identity = _validate_exact_thermal_design(
        native_project,
        expected_design_names=with_thermal_set,
    )
    lifecycle.update({
        "preserved_thermal_present": True,
        "preserved_thermal_validated": True,
        "preserved_thermal_identity": thermal_identity,
    })
    native_project.DeleteDesign(_RECOVERY_THERMAL_DESIGN)
    remaining_names, _remaining = _exact_native_designs(native_project)
    if (
        len(remaining_names) != len(maxwell_set)
        or set(remaining_names) != maxwell_set
    ):
        raise RuntimeError(
            "staged thermal deletion did not restore the exact Maxwell set: "
            f"{remaining_names!r}"
        )
    if _sha256(source) != expected_sha:
        raise RuntimeError(
            "sealed source AEDT changed during staged thermal deletion"
        )
    lifecycle.update({
        "preserved_thermal_action": (
            "discarded_from_staged_copy_and_rebuilt"
        ),
        "preserved_thermal_deleted_from_staged_copy": True,
        "post_delete_design_names": list(remaining_names),
        "sealed_source_aedt_sha256_after_staged_delete": expected_sha,
    })
    return lifecycle


def attest_staged_thermal_wrapper(
    thermal_design: Any,
    *,
    staged_aedt: Path,
    staged_results: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    """Prove the rebuilt PyAEDT design writes only into staged storage."""
    expected_project = Path(staged_aedt).resolve(strict=True)
    expected_results = Path(staged_results).resolve(strict=True)
    protected_root = Path(sealed_root).resolve(strict=True)
    project_file = Path(
        str(getattr(thermal_design, "project_file", "") or "")
    ).resolve(strict=True)
    results_directory = Path(
        str(getattr(thermal_design, "results_directory", "") or "")
    ).resolve(strict=True)
    if (
        project_file != expected_project
        or results_directory != expected_results
        or _is_within(project_file, protected_root)
        or _is_within(results_directory, protected_root)
    ):
        raise RuntimeError(
            "rebuilt thermal wrapper escaped the staged project/results pair: "
            f"project={project_file}, results={results_directory}"
        )
    return {
        "project_file": str(project_file),
        "results_directory": str(results_directory),
        "paths_bound_to_staged_copy": True,
    }


def _modeler_object(design: Any, name: str) -> Any:
    modeler = design.modeler
    getter = getattr(modeler, "get_object_from_name", None)
    if callable(getter):
        result = getter(name)
    else:
        result = modeler[name]
    if result is None or result is False:
        raise RuntimeError(f"modeler returned no object for {name!r}")
    return result


def hydrate_full_loss_object_groups(design: Any) -> dict[str, list[str]]:
    """Reconstruct only deterministic groups consumed by loss extraction."""
    names = sorted(
        (str(name) for name in design.modeler.object_names),
        key=_natural_key,
    )

    def select(pattern: str) -> list[Any]:
        matcher = re.compile(pattern)
        return [
            _modeler_object(design, name)
            for name in names
            if matcher.fullmatch(name)
        ]

    groups = {
        "Tx_windings_main": select(r"Tx_main_\d+_\d+"),
        "Tx_windings_side": select(r"Tx_side_\d+_\d+"),
        "Tx_windings_side2": select(r"Tx_side2_\d+_\d+"),
        "Rx_windings_main": select(r"Rx_main_\d+_\d+"),
        "Rx_windings_side": select(r"Rx_side_\d+_\d+"),
        "Rx_windings_side2": select(r"Rx_side2_\d+_\d+"),
        "core_objs": [
            _modeler_object(design, name)
            for name in names
            if _CORE_RE.fullmatch(name)
        ],
        "core_flux_sheets": select(r"core_flux_section_\d+"),
        "core_plates": select(
            r"core_plate_\d+_(?:side_left|center|side_right)"
        ),
        "core_pads": select(
            r"core_plate_pad_\d+_[ab]_(?:side_left|center|side_right)"
        ),
        "wcp_plates": select(r"Tx_main_wcp_\d+_[pn]"),
        "wcp_pads": select(r"Tx_main_wcp_pad_\d+_(?:in|out)_[pn]"),
    }
    for attribute, objects in groups.items():
        setattr(design, attribute, objects)
    design.Tx_windings = (
        groups["Tx_windings_main"]
        + groups["Tx_windings_side"]
        + groups["Tx_windings_side2"]
    )
    design.Rx_windings = (
        groups["Rx_windings_main"]
        + groups["Rx_windings_side"]
        + groups["Rx_windings_side2"]
    )

    required = ("Tx_windings_main", "Rx_windings_main", "core_objs")
    missing = [name for name in required if not groups[name]]
    if missing:
        raise RuntimeError(
            "preserved full-loss design is missing required object groups: "
            + ", ".join(missing)
        )
    return {
        key: [str(item.name) for item in value]
        for key, value in groups.items()
    }


def validate_full_loss_object_groups(
    groups: dict[str, list[str]], df_plus: pd.DataFrame
) -> dict[str, int]:
    """Require the exact current Full-model topology before extraction."""
    n_core_group = int(df_plus["n_core_group"].iloc[0])
    core_plate_on = int(df_plus["core_plate_on"].iloc[0]) != 0
    core_pad_on = (
        core_plate_on
        and float(df_plus["core_plate_pad_t"].iloc[0]) > 0
    )
    wcp_on = int(df_plus["wcp_on"].iloc[0]) != 0
    wcp_pad_on = wcp_on and float(df_plus["wcp_pad_t"].iloc[0]) > 0
    tx_slot_indices = (
        get_tx_y_gaps(df_plus)[1] if wcp_on else []
    )

    expected = {
        "Tx_windings_main": int(df_plus["N1_main"].iloc[0]),
        "Tx_windings_side": int(df_plus["N1_side"].iloc[0]),
        "Tx_windings_side2": int(df_plus["N1_side"].iloc[0]),
        "Rx_windings_main": int(df_plus["N2_main"].iloc[0]),
        "Rx_windings_side": int(df_plus["N2_side"].iloc[0]),
        "Rx_windings_side2": int(df_plus["N2_side"].iloc[0]),
        # The revision-bound Full model uses five wound-core pieces per group
        # and one center-leg flux sheet per group.
        "core_objs": 5 * n_core_group,
        "core_flux_sheets": n_core_group,
        # Every one of n+1 depth interfaces has left/center/right I plates.
        "core_plates": 3 * (n_core_group + 1) if core_plate_on else 0,
        "core_pads": 6 * (n_core_group + 1) if core_pad_on else 0,
        # Each selected Tx gap has one plate on each y side and, when enabled,
        # one inner and one outer pad on each side.
        "wcp_plates": 2 * len(tx_slot_indices) if wcp_on else 0,
        "wcp_pads": 4 * len(tx_slot_indices) if wcp_pad_on else 0,
    }
    actual = {name: len(groups.get(name, [])) for name in expected}
    mismatches = {
        name: {"expected": count, "actual": actual[name]}
        for name, count in expected.items()
        if actual[name] != count
    }
    if mismatches:
        raise RuntimeError(
            "preserved full-loss object-group count mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload, ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
        directory_fd = os.open(
            str(path.parent), os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _cap_result_validation(
    frame: pd.DataFrame, df_plus: pd.DataFrame
) -> tuple[bool, str]:
    """Validate solved capacitance evidence and its required outputs."""
    required = (
        "conv_passes_cap",
        "conv_consecutive_cap",
        "conv_error_pct_cap",
        "conv_delta_pct_cap",
        "C_tx_tx_F",
        "C_rx_rx_F",
        "C_tx_rx_F",
        "f_res_tx_self_Hz",
        "f_res_rx_self_Hz",
        "f_res_interwinding_Hz",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return False, f"capacitance fields are missing: {missing}"
    try:
        values = {
            name: float(frame[name].iloc[0])
            for name in required
        }
        tolerance = float(df_plus["cap_percent_error"].iloc[0])
    except (TypeError, ValueError, OverflowError) as error:
        return False, f"capacitance fields are not numeric: {error}"
    if not all(math.isfinite(value) for value in values.values()):
        return False, "capacitance fields contain non-finite values"
    if (
        tolerance <= 0
        or values["conv_passes_cap"] < 1
        or not values["conv_passes_cap"].is_integer()
        or values["conv_consecutive_cap"] < 1
        or not values["conv_consecutive_cap"].is_integer()
        or values["conv_consecutive_cap"] > values["conv_passes_cap"]
        or values["conv_error_pct_cap"] < 0
        or values["conv_error_pct_cap"] > tolerance
        or values["conv_delta_pct_cap"] < 0
        or values["conv_delta_pct_cap"] > tolerance
    ):
        return False, (
            "capacitance convergence failed: "
            f"tolerance={tolerance}, values={values}"
        )
    positive = (
        "C_tx_tx_F",
        "C_rx_rx_F",
        "C_tx_rx_F",
        "f_res_tx_self_Hz",
        "f_res_rx_self_Hz",
        "f_res_interwinding_Hz",
    )
    if any(values[name] <= 0 for name in positive):
        return False, "capacitance/resonance outputs must be positive"
    return True, "valid"


def _source_values_equal(actual: Any, expected: Any) -> bool:
    """Compare one CSV scalar to its revision-bound parameter scalar."""
    if isinstance(expected, str):
        return not pd.isna(actual) and str(actual) == expected
    try:
        expected_missing = bool(pd.isna(expected))
    except (TypeError, ValueError):
        expected_missing = False
    if expected is None or expected_missing:
        try:
            return actual is None or bool(pd.isna(actual))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError, OverflowError):
            return False
    if pd.api.types.is_number(expected):
        try:
            actual_number = float(actual)
            expected_number = float(expected)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(actual_number)
            and math.isfinite(expected_number)
            and math.isclose(
                actual_number,
                expected_number,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    return not pd.isna(actual) and str(actual) == str(expected)


def _exact_source_integer(
    row: pd.DataFrame, name: str, expected: int
) -> None:
    try:
        value = float(row[name].iloc[0])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
        raise RuntimeError(
            f"sealed source result has no integral {name}"
        ) from error
    if (
        not math.isfinite(value)
        or not value.is_integer()
        or int(value) != int(expected)
    ):
        raise RuntimeError(
            "sealed source result integer contract mismatch: "
            f"{name}={value!r}, expected={expected!r}"
        )


def _exact_rebuilt_integer(
    row: pd.DataFrame, name: str, expected: int
) -> None:
    try:
        value = float(row[name].iloc[0])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
        raise RuntimeError(
            f"rebuilt thermal result has no integral {name}"
        ) from error
    if (
        not math.isfinite(value)
        or not value.is_integer()
        or int(value) != int(expected)
    ):
        raise RuntimeError(
            "rebuilt thermal result integer contract mismatch: "
            f"{name}={value!r}, expected={expected!r}"
        )


def _exact_source_string(
    row: pd.DataFrame, name: str, expected: str
) -> None:
    try:
        value = str(row[name].iloc[0])
    except (KeyError, IndexError) as error:
        raise RuntimeError(
            f"sealed source result has no string {name}"
        ) from error
    if value != expected:
        raise RuntimeError(
            "sealed source result string contract mismatch: "
            f"{name}={value!r}, expected={expected!r}"
        )


def _exact_rebuilt_string(
    row: pd.DataFrame, name: str, expected: str
) -> None:
    try:
        value = str(row[name].iloc[0])
    except (KeyError, IndexError) as error:
        raise RuntimeError(
            f"rebuilt thermal result has no string {name}"
        ) from error
    if value != expected:
        raise RuntimeError(
            "rebuilt thermal result string contract mismatch: "
            f"{name}={value!r}, expected={expected!r}"
        )


def _source_historical_thermal_postsolve_evidence(
    row: pd.DataFrame,
) -> dict[str, Any]:
    """Record optional legacy postsolve fields without inventing evidence."""
    names = tuple(_SOURCE_HISTORICAL_THERMAL_POSTSOLVE_COUNTS)
    present = tuple(name for name in names if name in row.columns)
    if not present:
        return {
            "available": False,
            "availability_reason": "source_columns_absent",
            "expected_column_names": list(names),
            "present_column_names": [],
            "counts": {},
        }
    if len(present) != len(names):
        raise RuntimeError(
            "sealed source historical thermal postsolve columns are "
            "partially present: "
            f"present={list(present)!r}, expected={list(names)!r}"
        )
    for name, expected in _SOURCE_HISTORICAL_THERMAL_POSTSOLVE_COUNTS.items():
        _exact_source_integer(row, name, expected)
    return {
        "available": True,
        "availability_reason": "legacy_source_columns_attested",
        "expected_column_names": list(names),
        "present_column_names": list(names),
        "counts": dict(_SOURCE_HISTORICAL_THERMAL_POSTSOLVE_COUNTS),
    }


def load_sealed_source_result(
    *,
    mirror_root: Path,
    seal: dict[str, Any],
    project_name: str,
    df_plus: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the one manifest-bound, already validated source EM result.

    AEDT's saved RL/C-matrix postprocessor is not reliably replayable after a
    project is reopened.  The source run already persisted those exact values
    before its thermal-only extraction failure.  Reuse is allowed only after
    byte, row, input, solver-revision, convergence, and failure-mode binding.
    """
    root = Path(mirror_root).resolve(strict=True)
    relative = str(seal.get("source_result_relative_path") or "")
    if relative != _SEALED_SOURCE_RESULT_RELATIVE_PATH:
        raise RuntimeError("sealed source result relative-path binding failed")
    candidate = root / _relative_manifest_path(relative)
    if candidate.is_symlink():
        raise RuntimeError("sealed source result CSV must not be a symlink")
    path = candidate.resolve(strict=True)
    if not path.is_file() or not _is_within(path, root):
        raise RuntimeError("sealed source result CSV escaped current/")
    raw = path.read_bytes()
    expected_sha = str(
        seal.get("source_result_sha256") or ""
    ).strip().lower()
    try:
        expected_size = int(seal["source_result_size_bytes"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "sealed source result size binding is invalid"
        ) from error
    if (
        len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        raise RuntimeError(
            "sealed source result CSV changed after manifest attestation"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise RuntimeError(
            "sealed source result CSV is not valid UTF-8"
        ) from error
    header = next(csv.reader(io.StringIO(text, newline="")), [])
    if not header or len(header) != len(set(header)):
        raise RuntimeError(
            "sealed source result CSV has an empty or duplicate header"
        )
    try:
        frame = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception as error:
        raise RuntimeError(
            "sealed source result CSV cannot be parsed"
        ) from error
    if list(frame.columns) != header:
        raise RuntimeError(
            "sealed source result CSV parser/header identity mismatch"
        )
    if "project_name" not in frame.columns:
        raise RuntimeError(
            "sealed source result CSV has no project_name"
        )
    matches = frame[
        frame["project_name"].astype(str) == str(project_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "sealed source result CSV must contain exactly one project row: "
            f"project={project_name!r}, rows={len(matches)}"
        )
    row = matches.iloc[[0]].reset_index(drop=True)

    if (
        not isinstance(df_plus, pd.DataFrame)
        or len(df_plus) != 1
        or len(df_plus.columns) != len(set(df_plus.columns))
    ):
        raise RuntimeError("validated parameter frame is not one unique row")
    mismatches = []
    for name in df_plus.columns:
        if name not in row.columns:
            mismatches.append(f"{name}:missing")
            continue
        actual = row[name].iloc[0]
        expected = df_plus[name].iloc[0]
        if not _source_values_equal(actual, expected):
            mismatches.append(
                f"{name}:source={actual!r},params={expected!r}"
            )
    if mismatches:
        raise RuntimeError(
            "sealed source result does not match bound parameters: "
            + "; ".join(mismatches[:12])
        )

    for name, expected in (
        ("git_hash", _SOURCE_RESULT_GIT_HASH),
        (
            "pyaedt_library_git_hash",
            _SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH,
        ),
        ("em_validity_reason", "valid"),
        (
            "thermal_extraction_failure_reason",
            _SOURCE_RECOVERABLE_THERMAL_FAILURE,
        ),
        ("thermal_dispatch_status", "success"),
        (
            "thermal_pad_material_policy",
            SOURCE_THERMAL_PAD_MATERIAL_POLICY,
        ),
        (
            "thermal_pad_native_readback_contract_version",
            THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION,
        ),
    ):
        _exact_source_string(row, name, expected)
    for name, expected in (
        ("git_dirty", 0),
        ("pyaedt_library_git_dirty", 0),
        ("result_valid_em", 1),
        ("result_valid_thermal", 0),
        ("matrix_solve_attempts", 1),
        ("cap_solve_attempts", 1),
        ("loss_solve_attempts", 1),
        ("thermal_solve_attempts", 1),
        ("thermal_analyze_call_ok", 1),
        ("thermal_convergence_available", 1),
        ("thermal_converged", 1),
        ("thermal_solution_data_available", 1),
        ("thermal_solved", 0),
        ("thermal_extraction_complete", 0),
        ("thermal_pad_native_readback_attested", 1),
    ):
        _exact_source_integer(row, name, expected)
    source_historical_thermal_postsolve = (
        _source_historical_thermal_postsolve_evidence(row)
    )
    for name, expected in _SOURCE_EXTRACTION_BACKENDS.items():
        _exact_source_string(row, name, expected)
    try:
        thermal_iterations = float(row["thermal_iterations"].iloc[0])
        source_tim_k = float(
            row["thermal_pad_conductivity_W_mK"].iloc[0]
        )
        source_native_tim_k = float(
            row[
                "thermal_pad_native_thermal_conductivity_W_mK"
            ].iloc[0]
        )
        source_native_tim_sigma = float(
            row[
                "thermal_pad_native_electrical_conductivity_S_m"
            ].iloc[0]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "sealed source thermal/TIM evidence is not numeric"
        ) from error
    if (
        not math.isfinite(thermal_iterations)
        or thermal_iterations <= 0
        or not math.isclose(
            source_tim_k,
            SOURCE_THERMAL_PAD_CONDUCTIVITY_W_MK,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            source_native_tim_k,
            SOURCE_THERMAL_PAD_CONDUCTIVITY_W_MK,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            source_native_tim_sigma,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "sealed source thermal/TIM evidence contract failed"
        )

    em_valid, em_reason = _em_result_validation(
        row, matrix_on=True, loss_on=True
    )
    cap_valid, cap_reason = _cap_result_validation(row, df_plus)
    if not em_valid or not cap_valid:
        raise RuntimeError(
            "sealed source solved EM evidence is invalid: "
            f"em={em_reason}; cap={cap_reason}"
        )
    row_json = row.to_json(
        orient="records", date_format="iso", double_precision=15
    ).encode("utf-8")
    return row, {
        "relative_path": relative,
        "sha256": expected_sha,
        "size_bytes": expected_size,
        "csv_row_count": int(len(frame)),
        "csv_column_count": int(len(header)),
        "project_row_count": 1,
        "project_row_sha256": hashlib.sha256(row_json).hexdigest(),
        "em_validity_reason": em_reason,
        "cap_validity_reason": cap_reason,
        "source_git_hash": _SOURCE_RESULT_GIT_HASH,
        "source_pyaedt_library_git_hash": (
            _SOURCE_RESULT_PYAEDT_LIBRARY_GIT_HASH
        ),
        "source_thermal_failure_reason": (
            _SOURCE_RECOVERABLE_THERMAL_FAILURE
        ),
        "source_thermal_pad_policy": SOURCE_THERMAL_PAD_MATERIAL_POLICY,
        "source_historical_thermal_postsolve": (
            source_historical_thermal_postsolve
        ),
    }


def _selected_explicit_turn_names(
    names: list[str], count: int
) -> list[str]:
    """Mirror Simulation._select_explicit_turns without AEDT calls."""
    values = list(names)
    if count < 0:
        candidates = values
    elif count == 0:
        candidates = []
    else:
        candidates = values[:count] + values[-count:]
    return list(dict.fromkeys(candidates))


def deterministic_full_loss_object_groups(
    df_plus: pd.DataFrame,
) -> dict[str, list[str]]:
    """Derive the exact Full loss object names from bound parameters only."""
    n_core_group = int(df_plus["n_core_group"].iloc[0])
    core_plate_on = int(df_plus["core_plate_on"].iloc[0]) != 0
    core_pad_on = (
        core_plate_on
        and float(df_plus["core_plate_pad_t"].iloc[0]) > 0
    )
    wcp_on = int(df_plus["wcp_on"].iloc[0]) != 0
    wcp_pad_on = wcp_on and float(df_plus["wcp_pad_t"].iloc[0]) > 0
    tx_slot_indices = (
        get_tx_y_gaps(df_plus)[1] if wcp_on else []
    )

    def turns(prefix: str, count: int) -> list[str]:
        return [f"{prefix}_{index}_0" for index in range(int(count))]

    core_pieces = (
        "leg_left",
        "leg_center",
        "leg_right",
        "yoke_bottom",
        "yoke_top",
    )
    groups = {
        "Tx_windings_main": turns(
            "Tx_main", int(df_plus["N1_main"].iloc[0])
        ),
        "Tx_windings_side": turns(
            "Tx_side", int(df_plus["N1_side"].iloc[0])
        ),
        "Tx_windings_side2": turns(
            "Tx_side2", int(df_plus["N1_side"].iloc[0])
        ),
        "Rx_windings_main": turns(
            "Rx_main", int(df_plus["N2_main"].iloc[0])
        ),
        "Rx_windings_side": turns(
            "Rx_side", int(df_plus["N2_side"].iloc[0])
        ),
        "Rx_windings_side2": turns(
            "Rx_side2", int(df_plus["N2_side"].iloc[0])
        ),
        "core_objs": [
            f"core_{index}_{piece}"
            for index in range(1, n_core_group + 1)
            for piece in core_pieces
        ],
        "core_flux_sheets": [
            f"core_flux_section_{index}"
            for index in range(1, n_core_group + 1)
        ],
        "core_plates": (
            [
                f"core_plate_{interface}_{side}"
                for interface in range(1, n_core_group + 2)
                for side in ("side_left", "center", "side_right")
            ]
            if core_plate_on else []
        ),
        "core_pads": (
            [
                f"core_plate_pad_{interface}_{layer}_{side}"
                for interface in range(1, n_core_group + 2)
                for layer in ("a", "b")
                for side in ("side_left", "center", "side_right")
            ]
            if core_pad_on else []
        ),
        # create_winding_cooling_plates uses the slot only to locate each
        # plate.  Its native object identity is the one-based enumeration
        # order (k + 1), not the geometry gap index itself.
        "wcp_plates": (
            [
                f"Tx_main_wcp_{plate_index}_{side}"
                for plate_index in range(1, len(tx_slot_indices) + 1)
                for side in ("p", "n")
            ]
            if wcp_on else []
        ),
        "wcp_pads": (
            [
                f"Tx_main_wcp_pad_{plate_index}_{layer}_{side}"
                for plate_index in range(1, len(tx_slot_indices) + 1)
                for layer in ("in", "out")
                for side in ("p", "n")
            ]
            if wcp_pad_on else []
        ),
    }
    validate_full_loss_object_groups(groups, df_plus)
    return groups


def _finite_source_loss(row: pd.DataFrame, name: str) -> float:
    try:
        value = float(row[name].iloc[0])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
        raise RuntimeError(
            f"sealed source result is missing thermal loss {name}"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(
            f"sealed source thermal loss is invalid: {name}={value!r}"
        )
    return value


def _require_loss_sum(
    *,
    label: str,
    actual: float,
    expected: float,
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "sealed source thermal loss balance failed: "
            f"{label} actual={actual:.12g}, expected={expected:.12g}"
        )


def hydrate_sealed_loss_evidence(
    source_row: pd.DataFrame,
    object_groups: dict[str, list[str]],
    df_plus: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build exactly the physical loss map consumed by Full Icepak."""
    if not isinstance(source_row, pd.DataFrame) or len(source_row) != 1:
        raise RuntimeError("sealed source loss evidence must be one row")
    n_core_group = int(df_plus["n_core_group"].iloc[0])
    n_explicit = int(df_plus["n_explicit_turns"].iloc[0])
    core_indices = set()
    for name in object_groups.get("core_objs", []):
        match = re.fullmatch(
            r"core_(\d+)(?:_(?:leg_(?:left|center|right)|"
            r"yoke_(?:top|bottom)))?",
            str(name),
        )
        if not match:
            raise RuntimeError(
                f"unexpected Full loss core object name: {name!r}"
            )
        core_indices.add(int(match.group(1)))
    expected_indices = set(range(1, n_core_group + 1))
    if core_indices != expected_indices:
        raise RuntimeError(
            "Full loss core group identity mismatch: "
            f"actual={sorted(core_indices)}, "
            f"expected={sorted(expected_indices)}"
        )

    required = {
        "P_Tx_main_group",
        "P_Rx_main_group",
        *(f"P_core_{index}" for index in sorted(expected_indices)),
        *(
            f"P_turn_{name}"
            for name in object_groups.get("Tx_windings_main", [])
        ),
        *(
            f"P_turn_{name}"
            for name in _selected_explicit_turn_names(
                object_groups.get("Rx_windings_main", []),
                n_explicit,
            )
        ),
        *(
            f"P_{name}"
            for name in (
                object_groups.get("core_plates", [])
                + object_groups.get("wcp_plates", [])
            )
        ),
    }
    if int(df_plus["N2_side"].iloc[0]) > 0:
        required.add("P_Rx_side_group")
        required.update(
            f"P_turn_{name}"
            for name in _selected_explicit_turn_names(
                object_groups.get("Rx_windings_side", []),
                n_explicit,
            )
        )
    loss_map_phys = {
        name: _finite_source_loss(source_row, name)
        for name in sorted(required, key=_natural_key)
    }

    core_sum = sum(
        loss_map_phys[f"P_core_{index}"]
        for index in sorted(expected_indices)
    )
    _require_loss_sum(
        label="core groups",
        actual=core_sum,
        expected=_finite_source_loss(source_row, "P_core_total"),
    )
    winding_sum = (
        loss_map_phys["P_Tx_main_group"]
        + loss_map_phys["P_Rx_main_group"]
        + (
            2.0 * loss_map_phys["P_Rx_side_group"]
            if "P_Rx_side_group" in loss_map_phys else 0.0
        )
    )
    _require_loss_sum(
        label="winding groups",
        actual=winding_sum,
        expected=_finite_source_loss(source_row, "P_winding_total"),
    )
    tx_turn_keys = [
        f"P_turn_{name}"
        for name in object_groups.get("Tx_windings_main", [])
    ]
    _require_loss_sum(
        label="Tx explicit turns",
        actual=sum(loss_map_phys[name] for name in tx_turn_keys),
        expected=loss_map_phys["P_Tx_main_group"],
    )
    core_plate_keys = [
        f"P_{name}" for name in object_groups.get("core_plates", [])
    ]
    wcp_keys = [
        f"P_{name}" for name in object_groups.get("wcp_plates", [])
    ]
    _require_loss_sum(
        label="core plates",
        actual=sum(loss_map_phys[name] for name in core_plate_keys),
        expected=_finite_source_loss(
            source_row, "P_core_plate_total"
        ),
    )
    _require_loss_sum(
        label="winding cooling plates",
        actual=sum(loss_map_phys[name] for name in wcp_keys),
        expected=_finite_source_loss(source_row, "P_wcp_total"),
    )
    return loss_map_phys, {
        "source": "manifest_bound_simulation_results_260706_csv",
        "required_key_count": len(required),
        "required_keys": sorted(required, key=_natural_key),
        "loss_map_sha256": _canonical_sha256(loss_map_phys),
        "core_group_sum_w": core_sum,
        "winding_group_sum_w": winding_sum,
    }


def _overlay_single_row(
    base: pd.DataFrame, update: pd.DataFrame
) -> pd.DataFrame:
    """Replace stale source-stage fields without creating duplicate columns."""
    if (
        not isinstance(base, pd.DataFrame)
        or not isinstance(update, pd.DataFrame)
        or len(base) != 1
        or len(update) != 1
        or len(base.columns) != len(set(base.columns))
        or len(update.columns) != len(set(update.columns))
    ):
        raise RuntimeError("recovery result overlay requires unique one-row frames")
    result = base.copy()
    for name in update.columns:
        result[name] = update[name].iloc[0]
    return result


def _validate_thermal_pad_result(frame: pd.DataFrame) -> dict[str, Any]:
    """Require the rebuilt Icepak result to echo fresh native TIM evidence."""
    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise RuntimeError("rebuilt thermal result is not one row")
    _exact_source_string(
        frame,
        "thermal_pad_material_policy",
        THERMAL_PAD_MATERIAL_POLICY,
    )
    _exact_source_string(
        frame,
        "thermal_pad_native_readback_contract_version",
        THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION,
    )
    _exact_source_integer(
        frame, "thermal_pad_native_readback_attested", 1
    )
    values = {}
    for name in (
        "thermal_pad_conductivity_W_mK",
        "thermal_pad_native_thermal_conductivity_W_mK",
        "thermal_pad_native_electrical_conductivity_S_m",
    ):
        try:
            value = float(frame[name].iloc[0])
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            IndexError,
        ) as error:
            raise RuntimeError(
                f"rebuilt thermal result has no numeric {name}"
            ) from error
        if not math.isfinite(value):
            raise RuntimeError(
                f"rebuilt thermal TIM result is non-finite: {name}"
            )
        values[name] = value
    if (
        not math.isclose(
            values["thermal_pad_conductivity_W_mK"],
            THERMAL_PAD_CONDUCTIVITY_W_MK,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            values[
                "thermal_pad_native_thermal_conductivity_W_mK"
            ],
            THERMAL_PAD_CONDUCTIVITY_W_MK,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            values[
                "thermal_pad_native_electrical_conductivity_S_m"
            ],
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "rebuilt thermal native TIM contract mismatch"
        )
    return {
        "policy": THERMAL_PAD_MATERIAL_POLICY,
        "native_readback_contract_version": (
            THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION
        ),
        "native_readback_attested": True,
        **values,
    }


def _validate_rx_insulation_result(
    frame: pd.DataFrame,
    df_plus: pd.DataFrame,
) -> dict[str, Any]:
    """Attest the candidate's six explicit-Rx inter-turn TIM solids."""
    _exact_source_string(
        frame,
        "thermal_rx_explicit_insulation_model",
        _RX_EXPLICIT_INSULATION_MODEL,
    )
    expected_total = sum(_RX_EXPLICIT_INSULATION_COUNTS.values())
    _exact_source_integer(
        frame,
        "thermal_rx_explicit_insulation_count",
        expected_total,
    )
    for name, expected in (
        (
            "thermal_rx_explicit_insulation_policy",
            RX_EXPLICIT_INSULATION_POLICY,
        ),
        (
            "thermal_rx_explicit_insulation_material",
            RX_EXPLICIT_INSULATION_MATERIAL,
        ),
        (
            "thermal_rx_explicit_insulation_"
            "native_readback_contract_version",
            RX_EXPLICIT_INSULATION_NATIVE_READBACK_CONTRACT_VERSION,
        ),
    ):
        _exact_source_string(frame, name, expected)
    _exact_source_integer(
        frame,
        "thermal_rx_explicit_insulation_native_readback_attested",
        1,
    )
    try:
        raw = frame[
            "thermal_rx_explicit_insulation_counts_json"
        ].iloc[0]
        counts = json.loads(str(raw))
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            "rebuilt thermal Rx insulation counts JSON is invalid"
        ) from error
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_RX_EXPLICIT_INSULATION_COUNTS)
        or any(type(value) is not int for value in counts.values())
        or counts != _RX_EXPLICIT_INSULATION_COUNTS
        or sum(counts.values()) != expected_total
    ):
        raise RuntimeError(
            "rebuilt thermal Rx insulation topology mismatch: "
            f"actual={counts!r}, "
            f"expected={_RX_EXPLICIT_INSULATION_COUNTS!r}"
        )
    try:
        expected_k = float(df_plus["k_ins"].iloc[0])
        native_k = float(
            frame[
                "thermal_rx_explicit_insulation_"
                "native_thermal_conductivity_W_mK"
            ].iloc[0]
        )
        native_sigma = float(
            frame[
                "thermal_rx_explicit_insulation_"
                "native_electrical_conductivity_S_m"
            ].iloc[0]
        )
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise RuntimeError(
            "rebuilt thermal Rx insulation native material evidence "
            "is invalid"
        ) from error
    if (
        not math.isfinite(expected_k)
        or expected_k <= 0
        or not math.isclose(
            native_k, expected_k, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            native_sigma, 0.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise RuntimeError(
            "rebuilt thermal Rx insulation native material mismatch: "
            f"k={native_k!r}, expected_k={expected_k!r}, "
            f"sigma={native_sigma!r}"
        )
    return {
        "model": _RX_EXPLICIT_INSULATION_MODEL,
        "count": expected_total,
        "counts": dict(counts),
        "policy": RX_EXPLICIT_INSULATION_POLICY,
        "material": RX_EXPLICIT_INSULATION_MATERIAL,
        "native_readback_contract_version": (
            RX_EXPLICIT_INSULATION_NATIVE_READBACK_CONTRACT_VERSION
        ),
        "native_readback_attested": True,
        "native_thermal_conductivity_W_mK": native_k,
        "native_electrical_conductivity_S_m": native_sigma,
    }


def _validate_thermal_mesh_result(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Require candidate-exact static, native, and post-solve mesh evidence."""
    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise RuntimeError("rebuilt thermal mesh result is not one row")
    for name, expected in (
        ("thermal_mesh_policy", THERMAL_MESH_POLICY),
        (
            "thermal_mesh_plan_contract_version",
            THERMAL_MESH_PLAN_CONTRACT_VERSION,
        ),
        (
            "thermal_mesh_preflight_contract_version",
            THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
        ),
        (
            "thermal_mesh_preflight_status",
            "passed_standalone_native_premesh",
        ),
    ):
        _exact_rebuilt_string(frame, name, expected)
    for name, expected in {
        **_THERMAL_MESH_EXPECTED_COUNTS,
        "thermal_mesh_native_generation_passed": 1,
        "thermal_mesh_unmeshed_object_count": 0,
    }.items():
        _exact_rebuilt_integer(frame, name, expected)

    try:
        plan_sha256 = str(
            frame["thermal_mesh_plan_sha256"].iloc[0]
        ).strip().lower()
        unmeshed_json = json.loads(str(
            frame["thermal_mesh_unmeshed_objects_json"].iloc[0]
        ))
        postsolve_missing_json = json.loads(str(
            frame[
                "thermal_mesh_postsolve_probe_missing_objects_json"
            ].iloc[0]
        ))
        preflight = json.loads(str(
            frame["thermal_mesh_preflight_json"].iloc[0]
        ))
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            "rebuilt thermal mesh serialized evidence is invalid"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise RuntimeError(
            "rebuilt thermal mesh plan SHA-256 is invalid"
        )
    if unmeshed_json != []:
        raise RuntimeError(
            "rebuilt thermal mesh row reports unmeshed objects"
        )
    if postsolve_missing_json != []:
        raise RuntimeError(
            "rebuilt thermal mesh row reports missing post-solve probes"
        )
    if not isinstance(preflight, dict):
        raise RuntimeError(
            "rebuilt thermal mesh preflight JSON is not an object"
        )
    required_true = (
        "passed",
        "static_contract_passed",
        "generate_mesh_returned",
        "message_scan_complete",
        "analysis_dispatched_after_premesh",
        "native_operation_readback_passed",
        "mesh_artifact_readback_passed",
        "mesh_mapping_coverage_passed",
        "postflight_identity_passed",
        "standalone_idle_barrier_passed",
    )
    failed_flags = [
        name for name in required_true
        if preflight.get(name) is not True
    ]
    if (
        preflight.get("schema")
        != THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION
        or failed_flags
        or preflight.get("required_objects_missing") != []
        or preflight.get("unmeshed_objects") != []
        or preflight.get("native_errors") != []
        or preflight.get("mesh_policy") != THERMAL_MESH_POLICY
        or preflight.get("mesh_plan_sha256") != plan_sha256
        or int(preflight.get("fresh_mesh_artifact_count", 0)) < 1
    ):
        raise RuntimeError(
            "rebuilt thermal native mesh preflight contract mismatch: "
            f"failed_flags={failed_flags!r}"
        )
    mapping_coverage = preflight.get("mesh_mapping_coverage")
    mapping_readbacks = (
        mapping_coverage.get("readbacks")
        if isinstance(mapping_coverage, dict) else None
    )
    if (
        not isinstance(mapping_coverage, dict)
        or mapping_coverage.get("schema")
        != "thermal-grid-mapping-coverage-v1"
        or mapping_coverage.get("passed") is not True
        or mapping_coverage.get("parse_errors") != []
        or int(mapping_coverage.get("global_region_count", 0)) < 1
        or mapping_coverage.get("required_object_count") != 56
        or mapping_coverage.get("mapped_required_object_count") != 56
        or mapping_coverage.get("required_objects_missing") != []
        or mapping_coverage.get("expected_local_region_count") != 8
        or mapping_coverage.get("missing_local_regions") != []
        or mapping_coverage.get("local_regions_without_mesh") != []
        or mapping_coverage.get("uncoupled_local_regions") != []
        or mapping_coverage.get("local_region_objects_missing") != []
        or not isinstance(mapping_readbacks, list)
        or len(mapping_readbacks)
        != mapping_coverage.get("fresh_grid_mapping_count")
        or len(mapping_readbacks) < 9
        or sum(
            isinstance(item, dict)
            and item.get("schema")
            == "thermal-grid-mapping-readback-v1"
            and item.get("parent_region") == 0
            and item.get("has_mesh") is True
            for item in mapping_readbacks
        ) < 1
        or sum(
            isinstance(item, dict)
            and item.get("schema")
            == "thermal-grid-mapping-readback-v1"
            and isinstance(item.get("parent_region"), int)
            and item.get("parent_region") > 0
            and item.get("has_mesh") is True
            and isinstance(item.get("overlapping_mr_face_count"), int)
            and item.get("overlapping_mr_face_count") > 0
            for item in mapping_readbacks
        ) < 8
    ):
        raise RuntimeError(
            "rebuilt thermal native grid-mapping coverage mismatch"
        )
    for key, expected in (
        ("mesh_operation_count", 37),
        ("mesh_assigned_object_count", 93),
        ("object_level_operation_count", 29),
        ("mesh_region_operation_count", 8),
        ("wcp_pad_mesh_region_count", 8),
        ("core_plate_assembly_count", 15),
        ("wcp_assembly_count", 4),
        ("rx_retained_pack_count", 3),
    ):
        if type(preflight.get(key)) is not int \
                or preflight.get(key) != expected:
            raise RuntimeError(
                "rebuilt thermal mesh preflight count mismatch: "
                f"{key}={preflight.get(key)!r}, expected={expected}"
            )
    readback = preflight.get("native_operation_readback")
    operation_readbacks = (
        readback.get("operation_readbacks")
        if isinstance(readback, dict) else None
    )
    if (
        not isinstance(readback, dict)
        or readback.get("missing_operation_names") != []
        or readback.get("required_thin_objects_missing") != []
        or readback.get("expected_operation_count") != 37
        or readback.get("assigned_object_count") != 93
        or readback.get("required_thin_object_count") != 56
        or readback.get("object_level_operation_count") != 29
        or readback.get("mesh_region_operation_count") != 8
        or readback.get("mesh_region_part_readback_passed") is not True
        or not isinstance(operation_readbacks, list)
        or len(operation_readbacks) != 37
        or sum(
            isinstance(item, dict)
            and item.get("operation_type") == "object_level"
            for item in operation_readbacks
        ) != 29
        or sum(
            isinstance(item, dict)
            and item.get("operation_type") == "mesh_region"
            for item in operation_readbacks
        ) != 8
        or any(
            not isinstance(item, dict)
            or item.get("operation_type") != "object_level"
            or item.get("separate_objects") is not True
            for item in operation_readbacks
            if isinstance(item, dict)
            and item.get("operation_type") == "object_level"
        )
        or any(
            item.get("separate_objects") is not None
            or item.get("region_object_count") != 1
            for item in operation_readbacks
            if isinstance(item, dict)
            and item.get("operation_type") == "mesh_region"
        )
    ):
        raise RuntimeError(
            "rebuilt thermal native mesh OO readback mismatch"
        )
    idle = preflight.get("standalone_idle_barrier")
    if (
        not isinstance(idle, dict)
        or idle.get("passed") is not True
        or idle.get("last_running") is not False
        or int(idle.get("idle_observations", 0)) < 2
        or int(idle.get("stable_artifact_observations", 0)) < 2
        or not idle.get("fresh_mesh_artifacts")
    ):
        raise RuntimeError(
            "rebuilt thermal mesh idle/stable-artifact barrier mismatch"
        )
    return {
        "policy": THERMAL_MESH_POLICY,
        "plan_contract_version": THERMAL_MESH_PLAN_CONTRACT_VERSION,
        "preflight_contract_version": (
            THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION
        ),
        "plan_sha256": plan_sha256,
        "counts": dict(_THERMAL_MESH_EXPECTED_COUNTS),
        "native_generation_passed": True,
        "native_operation_readback_passed": True,
        "native_grid_mapping_coverage_passed": True,
        "stable_mesh_artifact_passed": True,
        "postsolve_prior_nine_probe_complete": True,
    }


def _exact_solved_design(
    project: Any, design_name: str, solution_kind: str
) -> tuple[Any, Any]:
    """Bind one existing solved design without creating or renaming designs."""
    if design_name not in {"maxwell_matrix", "maxwell_cap", "maxwell_loss"}:
        raise RuntimeError(f"unsupported recovery design name: {design_name!r}")
    native_project = project.project
    project_name = str(native_project.GetName() or "").strip()
    before_names = tuple(
        _aedt_design_name(item) for item in native_project.GetDesigns()
    )
    matches = []
    for native_design in native_project.GetDesigns():
        if _aedt_design_name(native_design) == design_name:
            matches.append(native_design)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one design named {design_name!r}, found "
            f"{len(matches)}"
        )
    native_design = matches[0]
    design_type = str(native_design.GetDesignType() or "")
    solution_type = str(native_design.GetSolutionType() or "")
    physics_matches = (
        _is_ac_magnetic_solution(solution_type)
        if solution_kind == "ac_magnetic"
        else solution_type.strip().casefold() == "electrostatic"
    )
    if design_type != "Maxwell 3D" or not physics_matches:
        raise RuntimeError(
            "preserved design physics mismatch: "
            f"design={design_name!r}, "
            f"type={design_type!r}, solution={solution_type!r}"
        )
    analysis = native_design.GetModule("AnalysisSetup")
    if tuple(str(name) for name in (analysis.GetSetups() or [])) != ("Setup1",):
        raise RuntimeError("preserved loss design does not contain exact Setup1")
    wrapped = project.create_design(
        name=design_name,
        solver="maxwell3d",
        solution=solution_type,
    )
    after_names = tuple(
        _aedt_design_name(item) for item in native_project.GetDesigns()
    )
    if after_names != before_names:
        raise RuntimeError(
            "binding an existing design mutated the preserved design list: "
            f"before={before_names!r}, after={after_names!r}"
        )
    active_design = native_project.SetActiveDesign(design_name)
    if (
        active_design is None
        or active_design is False
        or _aedt_design_name(active_design) != design_name
    ):
        raise RuntimeError(
            f"failed to activate exact preserved design {design_name!r}"
        )
    app = getattr(wrapped, "solver_instance", None)
    wrapped_project = getattr(app, "oproject", None)
    wrapped_design = getattr(app, "odesign", None)
    if (
        app is None
        or wrapped_project is None
        or wrapped_project is False
        or str(wrapped_project.GetName() or "").strip() != project_name
        or wrapped_design is None
        or wrapped_design is False
        or _aedt_design_name(wrapped_design) != design_name
    ):
        raise RuntimeError(
            "PyAEDT wrapper native project/design identity mismatch: "
            f"project={project_name!r}, design={design_name!r}"
        )
    wrapped_analysis = wrapped_design.GetModule("AnalysisSetup")
    if tuple(
        str(name) for name in (wrapped_analysis.GetSetups() or [])
    ) != ("Setup1",):
        raise RuntimeError(
            f"PyAEDT wrapper for {design_name!r} is not bound to exact Setup1"
        )
    return wrapped, native_design


def recover(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    runtime_evidence = attest_recovery_runtime(
        recovery_revision=args.recovery_revision,
        bundle_sha256=args.bundle_sha256,
    )
    sealed_root_input = Path(args.sealed_root)
    if sealed_root_input.is_symlink():
        raise RuntimeError("sealed root must not be a symlink")
    sealed_root = sealed_root_input.resolve(strict=True)
    current_input = sealed_root / "current"
    if current_input.is_symlink():
        raise RuntimeError("sealed current/ must not be a symlink")
    mirror_root = current_input.resolve(strict=True)
    if list(sealed_root.glob(".incoming-current-*")):
        raise RuntimeError("sealed root contains an incomplete incoming current/")
    params_path = (mirror_root / "repo" / "cand.json").resolve(strict=True)
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if Path(args.work_dir).is_symlink():
        raise RuntimeError("recovery work directory must not be a symlink")
    if _is_within(work_dir, mirror_root) or _is_within(
        output_dir, mirror_root
    ):
        raise RuntimeError(
            "recovery work/output directories must be outside sealed current/"
        )
    if (
        _is_within(work_dir, output_dir)
        or _is_within(output_dir, work_dir)
    ):
        raise RuntimeError(
            "recovery work and output directories must not overlap"
        )
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(
            f"recovery work directory must be fresh or empty: {work_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"recovery output directory must be fresh or empty: {output_dir}"
        )

    seal = attest_preserved_seal(
        mirror_root=mirror_root,
        manifest_path=sealed_root / "manifest.json",
        source_identity_path=sealed_root / "source_identity.json",
        params_path=params_path,
        project_name=args.project_name,
        expected_source_task_id=args.source_task_id,
        expected_manifest_sha256=args.manifest_sha256,
        expected_source_identity_sha256=args.source_identity_sha256,
        expected_params_sha256=args.params_sha256,
        expected_params_canonical_sha256=args.params_canonical_sha256,
        expected_candidate_digest=args.candidate_digest,
        expected_profile_sha256=args.profile_sha256,
    )
    terminal_watch = attest_terminal_watch(
        terminal_watch_path=Path(args.terminal_watch_result),
        expected_terminal_watch_sha256=args.terminal_watch_sha256,
        sealed_root=sealed_root,
        seal=seal,
        project_name=args.project_name,
    )
    seal["terminal_watch"] = terminal_watch
    source_aedt, source_results = discover_preserved_project(
        mirror_root, args.project_name
    )
    if (
        source_aedt.relative_to(mirror_root).as_posix()
        != seal["project_relative_path"]
    ):
        raise RuntimeError("discovered project path does not match sealed identity")

    params = _load_bound_json_object(
        params_path,
        "fixed parameter payload",
        expected_raw_sha256=seal["params_sha256"],
        expected_canonical_sha256=seal["params_canonical_sha256"],
    )
    input_df, physics_revision = _load_fixed_input_parameter(params)
    _, df_plus = validation_check(input_df, strict=True)
    df_plus["physics_data_revision"] = physics_revision
    required_flags = (
        "full_model",
        "matrix_on",
        "cap_on",
        "loss_on",
        "thermal_on",
    )
    disabled = [
        name for name in required_flags
        if int(df_plus[name].iloc[0]) != 1
    ]
    if disabled:
        raise RuntimeError(
            "complete post-solve recovery requires enabled stages: "
            + ", ".join(disabled)
        )
    if (
        int(df_plus["loss_sym_on"].iloc[0]) != 0
        or str(df_plus["thermal_symmetry"].iloc[0]) != "full"
    ):
        raise RuntimeError(
            "complete post-solve recovery requires full loss and thermal models"
        )

    # Reject deadline fan/pad/TIM experiments before source hydration, work
    # directory creation, or AEDT launch.  Historical evidence stays
    # diagnostic and cannot be relabelled as a fixed-boundary Full result.
    fixed_boundary_evidence = attest_fixed_boundary(
        df_plus.iloc[0].to_dict(),
        thermal_pad_conductivity_w_mk=(
            THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
    )
    fixed_boundary_metadata = fixed_boundary_result_metadata(
        fixed_boundary_evidence
    )

    source_result_started = time.monotonic()
    source_result, source_result_evidence = load_sealed_source_result(
        mirror_root=mirror_root,
        seal=seal,
        project_name=args.project_name,
        df_plus=df_plus,
    )
    source_result_load_s = time.monotonic() - source_result_started

    object_groups = deterministic_full_loss_object_groups(df_plus)
    expected_object_group_counts = validate_full_loss_object_groups(
        object_groups, df_plus
    )
    loss_map_phys, loss_evidence = hydrate_sealed_loss_evidence(
        source_result,
        object_groups,
        df_plus,
    )
    sealed_source_results_metadata_sha256 = _directory_metadata_sha256(
        source_results
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fresh_project_dir = work_dir / args.project_name
    fresh_aedt = fresh_project_dir / f"{args.project_name}.aedt"
    fresh_results = fresh_project_dir / (
        f"{args.project_name}.aedtresults"
    )

    desktop = None
    sim = None
    solver_core_policy = None
    result = None
    success_manifest = None
    thermal_lifecycle: dict[str, Any] = {
        "status": "not_prepared",
        "source_project_opened": False,
        "source_results_copied": False,
        "preserved_thermal_present": False,
        "preserved_thermal_validated": False,
        "preserved_thermal_action": (
            "not_opened_fresh_icepak_rebuild"
        ),
        "preserved_thermal_deleted_from_staged_copy": False,
        "sealed_source_mutated": False,
    }
    operation_failure: tuple[BaseException, Any] | None = None
    uncertain_standalone_solver = False
    try:
        desktop = pyDesktop(
            version=args.aedt_version,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        project = desktop.create_project(
            path=str(fresh_project_dir),
            name=args.project_name,
        )
        actual_project_name = str(project.project.GetName() or "").strip()
        if args.project_name and actual_project_name != args.project_name:
            raise RuntimeError(
                "fresh thermal project identity mismatch: "
                f"expected={args.project_name!r}, actual={actual_project_name!r}"
            )
        thermal_lifecycle["status"] = "fresh_project_prepared"
        thermal_lifecycle["fresh_project_directory"] = str(
            fresh_project_dir
        )

        sim = Simulation(desktop=desktop)
        solver_core_policy = attest_recovery_solver_core_policy(sim)
        runtime_evidence.update({
            "thermal_solver_cores": int(sim.NUM_CORE),
            "thermal_solver_tasks": int(sim.NUM_TASK),
            "solver_core_policy": solver_core_policy,
        })
        sim.PROJECT_NAME = actual_project_name
        sim.project_path = str(fresh_project_dir)
        sim.project = project
        sim.input_df = input_df
        sim.df_plus = df_plus
        sim.full_model = True
        sim.loss_em_full = True
        sim.loss_is_sym = False
        sim.solve_attempts = {"matrix": 0, "cap": 0, "loss": 0}
        sim.extraction_attempts = {}
        sim.extraction_backends = {}
        sim.extraction_units = {}
        sim.stage_timings = {}
        sim._remember_native_desktop_handle(desktop)
        # LossAllocator consumes the physical map.  The recovery is Full, so
        # non-core raw and physical values are identical; native core thermal
        # injection deliberately uses only loss_map_phys.
        sim.loss_map_phys = dict(loss_map_phys)
        sim.loss_map = dict(loss_map_phys)
        sim.df_loss_summary = source_result.copy()
        em_frame = source_result.copy()
        em_valid, em_reason = _em_result_validation(
            em_frame, matrix_on=True, loss_on=True
        )
        cap_valid, cap_reason = _cap_result_validation(em_frame, df_plus)
        if not em_valid or not cap_valid:
            raise RuntimeError(
                "recovered solved EM evidence is invalid: "
                f"em={em_reason}; cap={cap_reason}"
            )

        thermal_started = time.monotonic()
        thermal = run_thermal_analysis(sim)
        thermal_s = time.monotonic() - thermal_started
        thermal_pad_evidence = _validate_thermal_pad_result(thermal)
        rx_insulation_evidence = _validate_rx_insulation_result(
            thermal, df_plus
        )
        thermal_mesh_evidence = _validate_thermal_mesh_result(thermal)
        final_thermal_identity = _validate_exact_thermal_design(
            project.project,
            expected_design_names={_RECOVERY_THERMAL_DESIGN},
        )
        final_design_names, _final_designs = _exact_native_designs(
            project.project
        )
        thermal_wrapper_paths = attest_staged_thermal_wrapper(
            sim.design_thermal,
            staged_aedt=fresh_aedt,
            staged_results=fresh_results,
            sealed_root=mirror_root,
        )
        thermal_lifecycle.update({
            "status": "fresh_thermal_complete",
            "rebuilt_thermal_identity": final_thermal_identity,
            "rebuilt_thermal_wrapper": thermal_wrapper_paths,
            "final_design_names": list(final_design_names),
        })
        result = _overlay_single_row(em_frame, thermal)
        for name, value in fixed_boundary_metadata.items():
            result[name] = value
        result["time_thermal"] = thermal_s
        thermal_valid = _thermal_result_is_valid(
            thermal, physics_data_revision=physics_revision
        )
        result["result_valid_em"] = int(em_valid)
        result["em_validity_reason"] = em_reason
        result["result_valid_cap"] = int(cap_valid)
        result["cap_validity_reason"] = cap_reason
        result["result_valid_thermal"] = int(thermal_valid)
        result["postsolve_recovery_schema"] = RECOVERY_SCHEMA
        result["postsolve_recovery_em_source_git_hash"] = (
            _SOURCE_RESULT_GIT_HASH
        )
        result["postsolve_recovery_solver_revision"] = (
            runtime_evidence["recovery_revision"]
        )
        result["postsolve_recovery_package_sha256"] = (
            runtime_evidence["package_sha256"]
        )
        result["postsolve_recovery_thermal_solver_cores"] = int(
            runtime_evidence["thermal_solver_cores"]
        )
        result["postsolve_recovery_solver_core_policy_json"] = json.dumps(
            solver_core_policy,
            sort_keys=True,
            separators=(",", ":"),
        )
        result["postsolve_recovery_pyaedt_library_git_hash"] = (
            runtime_evidence["pyaedt_library_git_hash"]
        )
        result["postsolve_recovery_pyaedt_library_git_dirty"] = int(
            runtime_evidence["pyaedt_library_git_dirty"]
        )
        maxwell_analyze_calls = sum(
            int(sim.solve_attempts.get(label, 0))
            for label in ("matrix", "cap", "loss")
        )
        maxwell_extraction_calls = sum(
            int(sim.extraction_attempts.get(label, 0))
            for label in ("matrix", "cap", "loss")
        )
        result["postsolve_recovery_maxwell_analyze_calls"] = (
            maxwell_analyze_calls
        )
        result["postsolve_recovery_live_maxwell_extraction_calls"] = (
            maxwell_extraction_calls
        )
        result["postsolve_recovery_em_evidence_source"] = (
            "manifest_bound_simulation_results_260706_csv"
        )
        result["postsolve_recovery_source_result_csv_sha256"] = (
            source_result_evidence["sha256"]
        )
        result["postsolve_recovery_source_result_row_sha256"] = (
            source_result_evidence["project_row_sha256"]
        )
        result["postsolve_recovery_source_result_load_s"] = (
            source_result_load_s
        )
        source_postsolve = source_result_evidence[
            "source_historical_thermal_postsolve"
        ]
        source_postsolve_available = (
            source_postsolve.get("available") is True
        )
        result[
            "postsolve_recovery_source_thermal_postsolve_evidence_available"
        ] = int(source_postsolve_available)
        result[
            "postsolve_recovery_source_thermal_postsolve_"
            "evidence_availability_reason"
        ] = source_postsolve["availability_reason"]
        source_counts = source_postsolve.get("counts", {})
        for name in _SOURCE_HISTORICAL_THERMAL_POSTSOLVE_COUNTS:
            result[f"postsolve_recovery_source_{name}"] = (
                int(source_counts[name])
                if source_postsolve_available else None
            )
        result["postsolve_recovery_matrix_extract_s"] = 0.0
        result["postsolve_recovery_cap_extract_s"] = 0.0
        result["postsolve_recovery_loss_extract_s"] = 0.0
        result["postsolve_recovery_thermal_s"] = thermal_s
        result["postsolve_recovery_source_aedt"] = str(source_aedt)
        result["postsolve_recovery_source_aedt_sha256"] = _sha256(source_aedt)
        result["postsolve_recovery_source_task_id"] = seal["source_task_id"]
        result["postsolve_recovery_manifest_sha256"] = (
            seal["manifest_sha256"]
        )
        result["postsolve_recovery_matrix_design"] = (
            "sealed_source_result_csv_no_wrapper"
        )
        result["postsolve_recovery_cap_design"] = (
            "sealed_source_result_csv_no_wrapper"
        )
        result["postsolve_recovery_loss_design"] = (
            "sealed_source_result_csv_no_wrapper"
        )
        result["postsolve_recovery_source_project_opened"] = 0
        result["postsolve_recovery_source_results_copied"] = 0
        result["postsolve_recovery_preserved_thermal_present"] = int(
            thermal_lifecycle["preserved_thermal_present"]
        )
        result["postsolve_recovery_preserved_thermal_validated"] = int(
            thermal_lifecycle["preserved_thermal_validated"]
        )
        result["postsolve_recovery_preserved_thermal_action"] = (
            thermal_lifecycle["preserved_thermal_action"]
        )
        result[
            "postsolve_recovery_preserved_thermal_deleted_from_staged_copy"
        ] = int(
            thermal_lifecycle[
                "preserved_thermal_deleted_from_staged_copy"
            ]
        )
        result["postsolve_recovery_sealed_source_thermal_mutated"] = 0
        result["postsolve_recovery_final_thermal_design_type"] = (
            final_thermal_identity["design_type"]
        )
        result["postsolve_recovery_final_thermal_solution_type"] = (
            final_thermal_identity["solution_type"]
        )
        result["postsolve_recovery_final_thermal_setups_json"] = json.dumps(
            final_thermal_identity["setups"],
            separators=(",", ":"),
        )
        if maxwell_analyze_calls != 0:
            raise RuntimeError(
                "post-solve recovery dispatched an unexpected Maxwell analysis"
            )
        if maxwell_extraction_calls != 0:
            raise RuntimeError(
                "post-solve recovery made an unexpected live Maxwell "
                "extraction call"
            )
        if not thermal_valid:
            _atomic_csv(
                output_dir / "postsolve-recovery-rejected-result.csv", result
            )
            raise RuntimeError(
                "recovered thermal result failed the production validity gate"
            )
        if _sha256(source_aedt) != seal["project_aedt_sha256"]:
            raise RuntimeError(
                "sealed source AEDT changed during fresh thermal recovery"
            )
        thermal_lifecycle[
            "sealed_source_aedt_sha256_before_release"
        ] = seal["project_aedt_sha256"]
        thermal_lifecycle["sealed_source_mutated"] = False

        csv_path = output_dir / "postsolve-recovery-result.csv"
        row = json.loads(result.iloc[0].to_json())
        success_manifest = {
            "schema": RECOVERY_SCHEMA,
            "status": "complete_full_recovery",
            "seal": seal,
            "source": {
                "aedt": str(source_aedt),
                "aedt_sha256": row[
                    "postsolve_recovery_source_aedt_sha256"
                ],
                "aedtresults": str(source_results),
            },
            "thermal_project": {
                "aedt": str(fresh_aedt),
                "aedtresults": str(fresh_results),
                "working_aedt": str(fresh_aedt),
                "working_aedtresults": str(fresh_results),
                "in_place": False,
                "source_project_opened": False,
                "source_results_copied": False,
            },
            "project_name": actual_project_name,
            "design_names": list(final_design_names),
            "thermal_rebuild": dict(thermal_lifecycle),
            "object_groups": object_groups,
            "expected_object_group_counts": expected_object_group_counts,
            "source_result_evidence": source_result_evidence,
            "loss_evidence": loss_evidence,
            "thermal_pad_evidence": thermal_pad_evidence,
            "rx_insulation_evidence": rx_insulation_evidence,
            "thermal_mesh_evidence": thermal_mesh_evidence,
            "fixed_boundary_evidence": fixed_boundary_evidence,
            "recovery_runtime": runtime_evidence,
            "maxwell_analyze_calls": maxwell_analyze_calls,
            "live_maxwell_extraction_calls": maxwell_extraction_calls,
            "source_result_load_s": source_result_load_s,
            "matrix_extract_s": 0.0,
            "cap_extract_s": 0.0,
            "loss_extract_s": 0.0,
            "thermal_s": thermal_s,
            "result_valid_em": True,
            "result_valid_cap": True,
            "result_valid_thermal": True,
            "elapsed_s": time.monotonic() - started,
            "result_csv": str(csv_path),
            "result": row,
        }
    except BaseException as error:
        operation_failure = (error, error.__traceback__)

    release_error = None
    if desktop is not None:
        if sim is not None and bool(
                getattr(sim, "solver_may_be_running", False)):
            # Do not call Desktop release against a native Icepak engine whose
            # completion could not be proved. The scheduler process boundary
            # owns final containment for this explicit failure state.
            logging.error(
                "skipping recovery Desktop release because the standalone "
                "native solver may still be running"
            )
            uncertain_standalone_solver = True
            if operation_failure is None:
                release_error = RuntimeError(
                    "standalone native solver remains uncertain before "
                    "Desktop release"
                )
        else:
            try:
                released = desktop.release_desktop(
                    close_projects=True, close_on_exit=True
                )
                if released is False:
                    raise RuntimeError("AEDT Desktop release returned False")
            except Exception as error:
                release_error = error

    if operation_failure is None and release_error is None:
        try:
            source_sha_after_release = _sha256(source_aedt)
            if source_sha_after_release != seal["project_aedt_sha256"]:
                raise RuntimeError(
                    "sealed source AEDT changed while releasing fresh "
                    "thermal Desktop"
                )
            results_metadata_after_release = _directory_metadata_sha256(
                source_results
            )
            if (
                results_metadata_after_release
                != sealed_source_results_metadata_sha256
            ):
                raise RuntimeError(
                    "sealed source AEDT results changed during fresh "
                    "thermal recovery"
                )
            thermal_lifecycle[
                "sealed_source_aedt_sha256_after_release"
            ] = source_sha_after_release
            thermal_lifecycle[
                "sealed_source_results_metadata_sha256"
            ] = results_metadata_after_release
            thermal_lifecycle["sealed_source_mutated"] = False
            if success_manifest is not None:
                durable_seal = seal_fresh_thermal_project(
                    source_dir=fresh_project_dir,
                    output_dir=output_dir,
                    project_name=actual_project_name,
                    require_gpfs=(os.name != "nt"),
                )
                thermal_lifecycle["durable_project_seal"] = durable_seal
                success_manifest["thermal_project"].update({
                    "aedt": durable_seal["aedt"],
                    "aedtresults": durable_seal["aedtresults"],
                    "durable_seal": durable_seal,
                })
                result["postsolve_recovery_durable_thermal_aedt"] = (
                    durable_seal["aedt"]
                )
                result[
                    "postsolve_recovery_durable_thermal_aedtresults"
                ] = durable_seal["aedtresults"]
                result[
                    "postsolve_recovery_durable_thermal_tree_sha256"
                ] = durable_seal["tree_sha256"]
                result[
                    "postsolve_recovery_durable_thermal_file_count"
                ] = durable_seal["file_count"]
                result[
                    "postsolve_recovery_durable_thermal_total_bytes"
                ] = durable_seal["total_bytes"]
                success_manifest["result"] = json.loads(
                    result.iloc[0].to_json()
                )
                success_manifest["thermal_rebuild"] = dict(
                    thermal_lifecycle
                )
        except Exception as error:
            release_error = error

    if operation_failure is not None or release_error is not None:
        error = (
            operation_failure[0]
            if operation_failure is not None
            else release_error
        )
        if operation_failure is not None and release_error is not None and (
            hasattr(error, "add_note")
        ):
            error.add_note(f"AEDT release also failed: {release_error}")
        failure_manifest = {
            "schema": RECOVERY_SCHEMA,
            "status": "failed",
            "seal": seal,
            "project_name": str(args.project_name),
            "error_type": type(error).__name__,
            "error": str(error),
            "aedt_release_error": (
                str(release_error) if release_error is not None else ""
            ),
            "standalone_solver_may_be_running": (
                uncertain_standalone_solver
            ),
            "thermal_rebuild": thermal_lifecycle,
            "fixed_boundary_evidence": fixed_boundary_evidence,
            "recovery_runtime": runtime_evidence,
            "elapsed_s": time.monotonic() - started,
        }
        failure_artifact_error = None
        try:
            _atomic_json(
                output_dir / "postsolve-recovery-failure.json",
                failure_manifest,
            )
        except BaseException as artifact_error:
            failure_artifact_error = artifact_error
            if not uncertain_standalone_solver:
                raise
        if uncertain_standalone_solver:
            artifact_detail = ""
            if failure_artifact_error is not None:
                artifact_detail = (
                    "; failure artifact write also failed: "
                    f"{type(failure_artifact_error).__name__}: "
                    f"{failure_artifact_error}"
                )
            raise UncertainStandaloneSolverExit(
                "standalone native solver remains uncertain after recovery "
                f"failure: {type(error).__name__}: {error}"
                + artifact_detail
            ) from error
        if operation_failure is not None:
            raise error.with_traceback(operation_failure[1])
        raise error

    if success_manifest is None or result is None:
        raise RuntimeError("recovery finished without a success artifact")
    _atomic_csv(
        Path(success_manifest["result_csv"]),
        result,
    )
    _atomic_json(
        output_dir / "postsolve-recovery-manifest.json", success_manifest
    )
    row = success_manifest["result"]
    print("RESULT_JSON " + json.dumps(row, sort_keys=True), flush=True)
    return success_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse sealed Full matrix/capacitance/loss evidence and run "
            "fresh-project thermal without opening or solving Maxwell"
        )
    )
    parser.add_argument("--sealed-root", required=True)
    parser.add_argument("--source-task-id", required=True, type=int)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-identity-sha256", required=True)
    parser.add_argument("--terminal-watch-result", required=True)
    parser.add_argument("--terminal-watch-sha256", required=True)
    parser.add_argument("--params-sha256", required=True)
    parser.add_argument("--params-canonical-sha256", required=True)
    parser.add_argument("--candidate-digest", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--recovery-revision", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aedt-version", default="2025.2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        recover(args)
    except UncertainStandaloneSolverExit as error:
        try:
            print(
                f"FATAL_CONTAINMENT {error}",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            pass
        try:
            sys.stdout.flush()
        except BaseException:
            pass
        try:
            sys.stderr.flush()
        except BaseException:
            pass
        # PyAEDT registered close_on_exit=True. Normal exception/SystemExit
        # unwinding would run its atexit release against the still-active
        # engine. A hard non-zero exit skips atexit and leaves the SLURM step
        # cgroup to terminate only this task's descendants.
        os._exit(error.exit_code)
        raise RuntimeError("os._exit returned unexpectedly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
