#!/usr/bin/env python3
"""Open one authenticated AEDT snapshot copy without solving or saving."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any, Callable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mft_goal_snapshot_live_aedt as snapshot
from tools import mft_goal_transport_live_aedt_snapshot as transport


SCHEMA = "mft-live-aedt-snapshot-open-validation-v1"
RESULT_SCHEMA = "mft-live-aedt-snapshot-open-validation-result-v1"


class OpenValidationError(RuntimeError):
    """Raised when a saved-project open validation cannot be authenticated."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > snapshot.MAX_MANIFEST_BYTES
    ):
        raise OpenValidationError(f"{label} is not one bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenValidationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise OpenValidationError(f"{label} is not an object")
    return value


def _safe_directory(
    cwd: Path,
    relative: str,
    *,
    expected_uid: int,
    label: str,
) -> Path:
    pure = snapshot._safe_relative_output(relative)
    raw = cwd.joinpath(*pure.parts)
    metadata = raw.lstat()
    if raw.is_symlink() or not raw.is_dir() or metadata.st_uid != expected_uid:
        raise OpenValidationError(f"{label} metadata is unsafe")
    workspace = cwd.resolve(strict=True)
    resolved = raw.resolve(strict=True)
    if workspace == resolved or workspace not in resolved.parents:
        raise OpenValidationError(f"{label} escaped the task workspace")
    return resolved


def _snapshot_contract(
    directory: Path,
    *,
    lane: str,
    source_task_id: int,
    source_slurm_job_id: str,
    source_node: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    manifest_sha256: str,
    manifest_payload_sha256: str,
    expected_uid: int,
) -> tuple[dict[str, Any], Path, dict[str, int]]:
    manifest_path = directory / "manifest.json"
    snapshot._regular_identity(manifest_path, expected_uid=expected_uid)
    if snapshot.sha256_file(manifest_path) != manifest_sha256:
        raise OpenValidationError("snapshot manifest SHA drifted")
    try:
        manifest = transport._validate_seal(
            _read_json(manifest_path, "snapshot manifest"),
            schema=snapshot.SCHEMA,
        )
    except transport.TransportError as exc:
        raise OpenValidationError("snapshot manifest seal is invalid") from exc
    artifact_name = f"{lane}.aedt"
    if (
        manifest.get("payload_sha256") != manifest_payload_sha256
        or manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("production_truth_eligible") is not False
        or manifest.get("solver_result_truth_included") is not False
        or manifest.get("lane") != lane
        or manifest.get("source_task_id") != source_task_id
        or manifest.get("source_slurm_job_id") != source_slurm_job_id
        or manifest.get("source_node") != source_node
        or manifest.get("artifact_filename") != artifact_name
        or manifest.get("artifact_sha256") != artifact_sha256
        or manifest.get("artifact_size_bytes") != artifact_size_bytes
        or manifest.get("source_changed_during_snapshot") is not False
    ):
        raise OpenValidationError("snapshot manifest contract drifted")
    artifact = directory / artifact_name
    identity = snapshot._regular_identity(
        artifact, expected_uid=expected_uid
    )
    if (
        identity["size_bytes"] != artifact_size_bytes
        or snapshot.sha256_file(artifact) != artifact_sha256
    ):
        raise OpenValidationError("snapshot AEDT identity drifted")
    return manifest, artifact, identity


def _copy_authenticated(
    source: Path,
    destination: Path,
    *,
    expected_identity: Mapping[str, int],
    expected_uid: int,
    expected_sha256: str,
) -> dict[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    try:
        with os.fdopen(source_fd, "rb") as reader, destination.open(
            "xb"
        ) as writer:
            opened = snapshot._identity_from_stat(
                os.fstat(reader.fileno()),
                expected_uid=expected_uid,
                path=source,
            )
            if opened != dict(expected_identity):
                raise OpenValidationError(
                    "snapshot AEDT changed before scratch copy"
                )
            while True:
                chunk = reader.read(snapshot.COPY_CHUNK_BYTES)
                if not chunk:
                    break
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            descriptor_after = snapshot._identity_from_stat(
                os.fstat(reader.fileno()),
                expected_uid=expected_uid,
                path=source,
            )
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise
    if (
        descriptor_after != dict(expected_identity)
        or destination.stat().st_size != expected_identity["size_bytes"]
        or snapshot.sha256_file(destination) != expected_sha256
    ):
        raise OpenValidationError("scratch AEDT copy identity drifted")
    return descriptor_after


def _design_name(design: Any) -> str:
    getter = getattr(design, "GetName", None)
    value = getter() if callable(getter) else design
    return str(value or "").split(";")[-1].strip()


def _design_inventory(native_project: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for native_design in tuple(native_project.GetDesigns() or ()):
        name = _design_name(native_design)
        if not name:
            raise OpenValidationError("AEDT project has an empty design name")
        design_type = str(native_design.GetDesignType() or "").strip()
        solution_type = str(native_design.GetSolutionType() or "").strip()
        analysis = native_design.GetModule("AnalysisSetup")
        setups = [
            str(value).strip()
            for value in tuple(analysis.GetSetups() or ())
        ]
        rows.append(
            {
                "name": name,
                "design_type": design_type,
                "solution_type": solution_type,
                "setups": setups,
            }
        )
    names = [row["name"] for row in rows]
    if not rows or len(names) != len(set(names)):
        raise OpenValidationError(
            "AEDT project design inventory is empty or duplicated"
        )
    return rows


def _native_project(loaded: Any, desktop: Any) -> Any:
    native = getattr(loaded, "oproject", None)
    if native is None or native is False:
        project = getattr(loaded, "project", None)
        native = getattr(project, "project", project)
    if native is None or native is False:
        native = desktop.odesktop.GetActiveProject()
    if native is None or native is False:
        raise OpenValidationError("AEDT returned no active native project")
    return native


def _publish_receipt(
    *,
    cwd: Path,
    output_relative_root: str,
    destination_name: str,
    receipt: Mapping[str, Any],
    expected_uid: int,
) -> tuple[Path, str]:
    pure = snapshot._safe_relative_output(output_relative_root)
    workspace = cwd.resolve(strict=True)
    output_root = workspace.joinpath(*pure.parts)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise OpenValidationError("validation output root is unsafe")
    output_root = output_root.resolve(strict=True)
    if workspace == output_root or workspace not in output_root.parents:
        raise OpenValidationError("validation output escaped the workspace")
    if output_root.lstat().st_uid != expected_uid:
        raise OpenValidationError("validation output owner drifted")
    destination = output_root / destination_name
    if destination.exists():
        receipt_path = destination / "validation_receipt.json"
        existing = _read_json(receipt_path, "existing validation receipt")
        try:
            existing = transport._validate_seal(existing, schema=SCHEMA)
        except transport.TransportError as exc:
            raise OpenValidationError(
                "existing validation receipt seal is invalid"
            ) from exc
        if existing != dict(receipt):
            raise OpenValidationError(
                "existing validation receipt identity drifted"
            )
        snapshot._regular_identity(
            receipt_path, expected_uid=expected_uid
        )
        return receipt_path, snapshot.sha256_file(receipt_path)
    staging = output_root / f".{destination_name}.tmp"
    staging.mkdir(mode=0o700)
    try:
        receipt_path = snapshot._atomic_json(
            staging / "validation_receipt.json", receipt
        )
        if os.name == "posix":
            handle = os.open(
                staging,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
        os.chmod(receipt_path, 0o400)
        os.chmod(staging, 0o500)
        os.replace(staging, destination)
        if os.name == "posix":
            handle = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
    except BaseException:
        if staging.exists():
            try:
                os.chmod(staging, 0o700)
            except OSError:
                pass
            shutil.rmtree(staging)
        raise
    published = destination / "validation_receipt.json"
    return published, snapshot.sha256_file(published)


def validate_snapshot_aedt_open(
    *,
    snapshot_relative_directory: str,
    output_relative_root: str,
    lane: str,
    source_task_id: int,
    source_slurm_job_id: str,
    source_node: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    snapshot_manifest_sha256: str,
    snapshot_manifest_payload_sha256: str,
    validator_revision: str,
    aedt_version: str = "2025.2",
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    hostname: str | None = None,
    uid: int | None = None,
    scratch_base: Path | None = None,
    desktop_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    actual_cwd = Path.cwd() if cwd is None else cwd
    actual_hostname = (
        socket.gethostname().split(".", 1)[0] if hostname is None else hostname
    )
    validator_task = snapshot._positive_int(
        env.get("SLURM_SCHED_TASK_ID"), "validator task ID"
    )
    validator_job = snapshot._slurm_job_id(env.get("SLURM_JOB_ID"))
    source_task = snapshot._positive_int(source_task_id, "source task ID")
    source_job = snapshot._slurm_job_id(source_slurm_job_id)
    size = snapshot._positive_int(artifact_size_bytes, "artifact size")
    artifact_sha = snapshot._sha(artifact_sha256, "artifact SHA")
    manifest_sha = snapshot._sha(
        snapshot_manifest_sha256, "snapshot manifest SHA"
    )
    manifest_payload = snapshot._sha(
        snapshot_manifest_payload_sha256,
        "snapshot manifest payload SHA",
    )
    revision = snapshot._git_revision(
        validator_revision, "validator revision"
    )
    if (
        not snapshot.LANE_RE.fullmatch(str(lane or ""))
        or not snapshot.NODE_RE.fullmatch(str(source_node or ""))
        or not snapshot.NODE_RE.fullmatch(actual_hostname)
        or snapshot._positive_int(
            env.get("SLURM_CPUS_PER_TASK"), "Slurm CPU count"
        )
        < 1
    ):
        raise OpenValidationError("AEDT open validation runtime is unsafe")
    if uid is None:
        getuid = getattr(os, "getuid", None)
        if not callable(getuid):
            raise OpenValidationError("AEDT open validation requires POSIX")
        uid = int(getuid())

    directory = _safe_directory(
        actual_cwd,
        snapshot_relative_directory,
        expected_uid=uid,
        label="snapshot directory",
    )
    manifest, source, source_before = _snapshot_contract(
        directory,
        lane=lane,
        source_task_id=source_task,
        source_slurm_job_id=source_job,
        source_node=source_node,
        artifact_sha256=artifact_sha,
        artifact_size_bytes=size,
        manifest_sha256=manifest_sha,
        manifest_payload_sha256=manifest_payload,
        expected_uid=uid,
    )
    production_scratch = scratch_base is None
    base = Path("/enroot") if scratch_base is None else scratch_base
    base = base.resolve(strict=True)
    if production_scratch and base != Path("/enroot"):
        raise OpenValidationError("production scratch base is not /enroot")
    scratch = base / f"mft-aedt-open-validation-t{validator_task}"
    if scratch.exists():
        raise OpenValidationError("AEDT validation scratch already exists")
    scratch.mkdir(mode=0o700)
    staged = scratch / f"{lane}.aedt"
    desktop = None
    opened_project_name: str | None = None
    receipt: dict[str, Any] | None = None
    operation_error: BaseException | None = None
    release_error: BaseException | None = None
    try:
        copied_identity = _copy_authenticated(
            source,
            staged,
            expected_identity=source_before,
            expected_uid=uid,
            expected_sha256=artifact_sha,
        )
        if desktop_factory is None:
            from ansys.aedt.core import Desktop

            desktop_factory = Desktop
        desktop = desktop_factory(
            version=aedt_version,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        loaded = desktop.load_project(str(staged))
        if loaded is None or loaded is False:
            raise OpenValidationError("AEDT could not open the staged project")
        native_project = _native_project(loaded, desktop)
        project_name = str(native_project.GetName() or "").strip()
        project_directory = str(native_project.GetPath() or "").strip()
        if not project_name or not project_directory:
            raise OpenValidationError("AEDT project path identity is empty")
        opened_project_name = project_name
        native_path = (
            Path(project_directory) / f"{project_name}.aedt"
        ).resolve(strict=True)
        if native_path != staged.resolve(strict=True):
            raise OpenValidationError(
                f"AEDT opened a different project: {native_path}"
            )
        designs = _design_inventory(native_project)
        receipt = {
            "schema": SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "solver_result_truth_included": False,
            "saved_project_open_passed": True,
            "saved_results_required": False,
            "lane": lane,
            "source_task_id": source_task,
            "source_slurm_job_id": source_job,
            "source_node": source_node,
            "snapshot_manifest_sha256": manifest_sha,
            "snapshot_manifest_payload_sha256": manifest_payload,
            "candidate_sha256": manifest["candidate_sha256"],
            "fixed_physics_sha256": manifest["fixed_physics_sha256"],
            "artifact_filename": source.name,
            "artifact_size_bytes": size,
            "artifact_sha256": artifact_sha,
            "validator_task_id": validator_task,
            "validator_slurm_job_id": validator_job,
            "validator_node": actual_hostname,
            "validator_revision": revision,
            "aedt_version": aedt_version,
            "project_name_readback": project_name,
            "project_path_readback": str(native_path),
            "design_count": len(designs),
            "designs": designs,
            "automation_calls": {
                "project_load": 1,
                "project_close_without_save": 1,
                "project_save": 0,
                "analysis": 0,
                "design_create_or_edit": 0,
                "geometry_create_or_edit": 0,
                "material_create_or_edit": 0,
                "boundary_create_or_edit": 0,
                "setup_create_or_edit": 0,
            },
            "scratch_copy_identity": copied_identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except BaseException as exc:
        operation_error = exc
    finally:
        if desktop is not None:
            if opened_project_name:
                try:
                    closed = desktop.odesktop.CloseProject(
                        opened_project_name
                    )
                    if closed is False:
                        raise OpenValidationError(
                            "AEDT CloseProject returned False"
                        )
                    if receipt is not None:
                        receipt[
                            "project_close_without_save_attested"
                        ] = True
                except BaseException as exc:
                    release_error = exc
            try:
                released = desktop.release_desktop(
                    close_projects=False,
                    close_on_exit=True,
                )
                if released is False:
                    raise OpenValidationError(
                        "AEDT Desktop release returned False"
                    )
            except BaseException as exc:
                if release_error is None:
                    release_error = exc

    verification_error: BaseException | None = None
    try:
        source_after = snapshot._regular_identity(
            source, expected_uid=uid
        )
        if (
            source_after != source_before
            or snapshot.sha256_file(source) != artifact_sha
            or snapshot.sha256_file(directory / "manifest.json")
            != manifest_sha
        ):
            raise OpenValidationError(
                "snapshot source changed during AEDT open validation"
            )
        if (
            staged.is_symlink()
            or not staged.is_file()
            or staged.stat().st_size != size
            or snapshot.sha256_file(staged) != artifact_sha
        ):
            raise OpenValidationError(
                "scratch AEDT changed during open validation"
            )
        if receipt is not None:
            receipt["source_identity_before"] = source_before
            receipt["source_identity_after"] = source_after
            receipt["source_unchanged"] = True
            receipt["scratch_copy_unchanged"] = True
    except BaseException as exc:
        verification_error = exc
    try:
        shutil.rmtree(scratch)
    except BaseException as exc:
        if verification_error is None:
            verification_error = exc

    for error in (operation_error, release_error, verification_error):
        if error is not None:
            raise OpenValidationError(
                f"AEDT open validation failed: {type(error).__name__}: "
                f"{error}"
            ) from error
    if receipt is None:
        raise OpenValidationError("AEDT open validation produced no receipt")
    receipt["scratch_removed"] = not scratch.exists()
    receipt["payload_sha256"] = snapshot.canonical_sha256(receipt)
    destination_name = (
        f"{lane}-from-t{source_task}-j{source_job}"
        f"-validator-t{validator_task}"
    )
    receipt_path, receipt_sha = _publish_receipt(
        cwd=actual_cwd,
        output_relative_root=output_relative_root,
        destination_name=destination_name,
        receipt=receipt,
        expected_uid=uid,
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "open_validated",
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "lane": lane,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": size,
        "design_count": receipt["design_count"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "payload_sha256": receipt["payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-relative-directory", required=True)
    parser.add_argument("--output-relative-root", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--source-task-id", required=True, type=int)
    parser.add_argument("--source-slurm-job-id", required=True)
    parser.add_argument("--source-node", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--artifact-size-bytes", required=True, type=int)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument(
        "--snapshot-manifest-payload-sha256", required=True
    )
    parser.add_argument("--validator-revision", required=True)
    parser.add_argument("--aedt-version", default="2025.2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_snapshot_aedt_open(
            snapshot_relative_directory=args.snapshot_relative_directory,
            output_relative_root=args.output_relative_root,
            lane=args.lane,
            source_task_id=args.source_task_id,
            source_slurm_job_id=args.source_slurm_job_id,
            source_node=args.source_node,
            artifact_sha256=args.artifact_sha256,
            artifact_size_bytes=args.artifact_size_bytes,
            snapshot_manifest_sha256=args.snapshot_manifest_sha256,
            snapshot_manifest_payload_sha256=(
                args.snapshot_manifest_payload_sha256
            ),
            validator_revision=args.validator_revision,
            aedt_version=args.aedt_version,
        )
    except (
        OSError,
        OpenValidationError,
        snapshot.SnapshotError,
        transport.TransportError,
        ValueError,
    ) as exc:
        print(
            f"LIVE_AEDT_OPEN_VALIDATION_ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
