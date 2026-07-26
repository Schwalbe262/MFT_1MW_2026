from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import mft_goal_snapshot_live_aedt as snapshot
from tools import mft_goal_validate_snapshot_aedt_open as validator


class _Analysis:
    def GetSetups(self):
        return ("Setup1",)


class _Design:
    def __init__(self, name: str):
        self.name = name

    def GetName(self):
        return self.name

    def GetDesignType(self):
        return "Maxwell 3D"

    def GetSolutionType(self):
        return "AC Magnetic"

    def GetModule(self, name: str):
        assert name == "AnalysisSetup"
        return _Analysis()


class _Project:
    def __init__(self, path: Path):
        self.path = path

    def GetName(self):
        return self.path.stem

    def GetPath(self):
        return str(self.path.parent)

    def GetDesigns(self):
        return (_Design("maxwell_matrix"), _Design("maxwell_loss"))


class _Loaded:
    def __init__(self, path: Path):
        self.oproject = _Project(path)


class _Desktop:
    calls = []

    def __init__(self, **kwargs):
        self.odesktop = self
        self.calls.append(("desktop", kwargs))

    def load_project(self, path: str):
        self.calls.append(("load", Path(path).name))
        return _Loaded(Path(path))

    def release_desktop(self, **kwargs):
        self.calls.append(("release", kwargs))
        return True

    def CloseProject(self, name: str):
        self.calls.append(("close", name))
        return True


class _DeletingDesktop(_Desktop):
    def load_project(self, path: str):
        self.loaded_path = Path(path)
        return super().load_project(path)

    def release_desktop(self, **kwargs):
        self.loaded_path.unlink()
        return super().release_desktop(**kwargs)


def _fixture(tmp_path: Path):
    directory = (
        tmp_path
        / "mft_goal_20260726"
        / "live_aedt_snapshots_v1"
        / "full-from-t96307-j829952"
    )
    directory.mkdir(parents=True)
    artifact = directory / "full.aedt"
    artifact.write_bytes(b"saved-project-bytes")
    manifest = {
        "schema": snapshot.SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "solver_result_truth_included": False,
        "lane": "full",
        "source_task_id": 96307,
        "source_slurm_job_id": "829952",
        "source_node": "n116",
        "artifact_filename": "full.aedt",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_size_bytes": artifact.stat().st_size,
        "source_changed_during_snapshot": False,
        "candidate_sha256": "3" * 64,
        "fixed_physics_sha256": "4" * 64,
    }
    manifest["payload_sha256"] = snapshot.canonical_sha256(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return directory, artifact, manifest, manifest_path


def test_validates_open_without_mutating_snapshot(tmp_path: Path) -> None:
    directory, artifact, manifest, manifest_path = _fixture(tmp_path)
    scratch_base = tmp_path / "enroot"
    scratch_base.mkdir()
    _Desktop.calls = []

    result = validator.validate_snapshot_aedt_open(
        snapshot_relative_directory=directory.relative_to(tmp_path).as_posix(),
        output_relative_root=(
            "mft_goal_20260726/aedt_open_validation_v1"
        ),
        lane="full",
        source_task_id=96307,
        source_slurm_job_id="829952",
        source_node="n116",
        artifact_sha256=manifest["artifact_sha256"],
        artifact_size_bytes=manifest["artifact_size_bytes"],
        snapshot_manifest_sha256=snapshot.sha256_file(manifest_path),
        snapshot_manifest_payload_sha256=manifest["payload_sha256"],
        validator_revision="5" * 40,
        environ={
            "SLURM_SCHED_TASK_ID": "96320",
            "SLURM_JOB_ID": "840000",
            "SLURM_CPUS_PER_TASK": "1",
        },
        cwd=tmp_path,
        hostname="n108",
        uid=directory.stat().st_uid,
        scratch_base=scratch_base,
        desktop_factory=_Desktop,
    )

    assert result["status"] == "open_validated"
    assert result["design_count"] == 2
    assert artifact.read_bytes() == b"saved-project-bytes"
    assert not (scratch_base / "mft-aedt-open-validation-t96320").exists()
    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["saved_project_open_passed"] is True
    assert receipt["source_unchanged"] is True
    assert receipt["scratch_copy_unchanged"] is True
    assert receipt["automation_calls"]["analysis"] == 0
    assert receipt["automation_calls"]["project_save"] == 0
    assert [row["name"] for row in receipt["designs"]] == [
        "maxwell_matrix",
        "maxwell_loss",
    ]
    assert [call[0] for call in _Desktop.calls] == [
        "desktop",
        "load",
        "close",
        "release",
    ]


def test_rejects_snapshot_manifest_drift(tmp_path: Path) -> None:
    directory, _artifact, manifest, manifest_path = _fixture(tmp_path)
    scratch_base = tmp_path / "enroot"
    scratch_base.mkdir()

    try:
        validator.validate_snapshot_aedt_open(
            snapshot_relative_directory=(
                directory.relative_to(tmp_path).as_posix()
            ),
            output_relative_root=(
                "mft_goal_20260726/aedt_open_validation_v1"
            ),
            lane="full",
            source_task_id=96307,
            source_slurm_job_id="829952",
            source_node="n116",
            artifact_sha256=manifest["artifact_sha256"],
            artifact_size_bytes=manifest["artifact_size_bytes"],
            snapshot_manifest_sha256="0" * 64,
            snapshot_manifest_payload_sha256=manifest["payload_sha256"],
            validator_revision="5" * 40,
            environ={
                "SLURM_SCHED_TASK_ID": "96320",
                "SLURM_JOB_ID": "840000",
                "SLURM_CPUS_PER_TASK": "1",
            },
            cwd=tmp_path,
            hostname="n108",
            uid=directory.stat().st_uid,
            scratch_base=scratch_base,
            desktop_factory=_Desktop,
        )
    except validator.OpenValidationError as exc:
        assert "manifest SHA drifted" in str(exc)
    else:
        raise AssertionError("manifest drift was accepted")

    assert snapshot.sha256_file(manifest_path) != "0" * 64


def test_rejects_missing_scratch_copy_after_open(tmp_path: Path) -> None:
    directory, _artifact, manifest, manifest_path = _fixture(tmp_path)
    scratch_base = tmp_path / "enroot"
    scratch_base.mkdir()

    try:
        validator.validate_snapshot_aedt_open(
            snapshot_relative_directory=(
                directory.relative_to(tmp_path).as_posix()
            ),
            output_relative_root=(
                "mft_goal_20260726/aedt_open_validation_v1"
            ),
            lane="full",
            source_task_id=96307,
            source_slurm_job_id="829952",
            source_node="n116",
            artifact_sha256=manifest["artifact_sha256"],
            artifact_size_bytes=manifest["artifact_size_bytes"],
            snapshot_manifest_sha256=snapshot.sha256_file(manifest_path),
            snapshot_manifest_payload_sha256=manifest["payload_sha256"],
            validator_revision="5" * 40,
            environ={
                "SLURM_SCHED_TASK_ID": "96320",
                "SLURM_JOB_ID": "840000",
                "SLURM_CPUS_PER_TASK": "1",
            },
            cwd=tmp_path,
            hostname="n108",
            uid=directory.stat().st_uid,
            scratch_base=scratch_base,
            desktop_factory=_DeletingDesktop,
        )
    except validator.OpenValidationError as exc:
        assert "scratch AEDT changed" in str(exc)
    else:
        raise AssertionError("missing scratch AEDT was accepted")
