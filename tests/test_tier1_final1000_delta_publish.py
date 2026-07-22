from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from tools import tier1_corrected_current7_slurm_bundle as bundle
from tools import tier1_corrected_current7_slurm_publish as publisher
from tools import tier1_final1000_delta_publish as delta
from tools import tier1_final1000_multiseed_release as multiseed_release
from tools.tier1_corrected_current7_receipt import canonical_sha256


REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record_bytes(value: bytes) -> dict[str, Any]:
    return {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _record_file(path: Path) -> dict[str, Any]:
    return _record_bytes(path.read_bytes())


def _publication_flags() -> dict[str, bool]:
    return {
        "stage_below_unique_incoming_directory": True,
        "verify_every_file_sha256_before_ready": True,
        "verify_exact_runtime_packages_before_ready": True,
        "write_ready_after_all_verification": True,
        "atomic_rename_incoming_to_content_addressed_bundle": True,
        "make_artifacts_read_only_before_publication": True,
    }


def _standard_plan(
    root: Path,
    manifest: Mapping[str, Any],
    sources: Mapping[str, Path],
    *,
    remote_root: str,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = root / "bundle_manifest.json"
    source_map_path = root / "local_sources.json"
    plan_path = root / "offload_plan.json"
    _write_json(manifest_path, manifest)
    _write_json(source_map_path, {key: str(value) for key, value in sources.items()})
    plan = {
        "schema_version": bundle.PLAN_SCHEMA,
        "bundle_id": manifest["bundle_id"],
        "contract_sha256": manifest["contract_sha256"],
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": bundle.sha256_file(manifest_path),
        "local_sources": str(source_map_path),
        "remote_root": remote_root,
        "remote_bundle": f"{remote_root}/{manifest['bundle_id']}",
        "publication_contract": _publication_flags(),
        "stage_performed": False,
        "submission_performed": False,
    }
    _write_json(plan_path, plan)
    return plan_path, plan


def _sealed_manifest(stable: Mapping[str, Any]) -> dict[str, Any]:
    contract_sha = canonical_sha256(stable)
    return {
        **copy.deepcopy(dict(stable)),
        "bundle_id": f"current7-{contract_sha[:20]}",
        "contract_sha256": contract_sha,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    remote_root = "/gpfs/test/final1000"
    source = tmp_path / "sources"
    large = source / "large-model.bin"
    requirements = source / "requirements.lock"
    parent_code = source / "worker-parent.py"
    child_code = source / "worker-final1000.py"
    stage_profile = source / "stage-profile.json"
    source.mkdir(parents=True)
    large.write_bytes(b"large-immutable-parent-payload")
    requirements.write_bytes(b"numpy==2.2.6\n")
    parent_code.write_bytes(b"print('parent')\n")
    child_code.write_bytes(b"print('final1000')\n")
    stage_profile.write_bytes(b'{"stage":"final-1000-t100"}\n')

    parent_sources = {
        "artifacts/models/large-model.bin": large,
        "artifacts/runtime/requirements.lock": requirements,
        "artifacts/code/tools/worker.py": parent_code,
    }
    parent_files = {
        relative: _record_file(path) for relative, path in parent_sources.items()
    }
    parent_code_inventory = {
        "artifacts/code/tools/worker.py": parent_files["artifacts/code/tools/worker.py"]
    }
    parent_stable = {
        "schema_version": bundle.BUNDLE_SCHEMA,
        "ready_schema_version": bundle.READY_SCHEMA,
        "files": parent_files,
        "code_inventory": parent_code_inventory,
        "code_inventory_sha256": canonical_sha256(parent_code_inventory),
        "relocation": {
            "path": "artifacts/models/large-model.bin",
            "sha256": parent_files["artifacts/models/large-model.bin"]["sha256"],
            "contract_sha256": "a" * 64,
        },
        "runtime": {
            "critical_packages": {"numpy": "2.2.6"},
            "requirements_lock": "artifacts/runtime/requirements.lock",
        },
        "hard_spec": {"T_limit_C": 110.0},
    }
    parent_manifest = _sealed_manifest(parent_stable)
    parent_plan_path, parent_plan = _standard_plan(
        tmp_path / "parent",
        parent_manifest,
        parent_sources,
        remote_root=remote_root,
    )
    ready = {
        "schema_version": bundle.READY_SCHEMA,
        "bundle_id": parent_manifest["bundle_id"],
        "contract_sha256": parent_manifest["contract_sha256"],
        "bundle_manifest_sha256": parent_plan["bundle_manifest_sha256"],
        "file_count": len(parent_files),
        "byte_count": sum(row["size"] for row in parent_files.values()),
        "runtime_verified": True,
        "runtime_packages": {"numpy": "2.2.6"},
        "all_file_sha256_verified": True,
        "every_file_sha256_verified": True,
        "code_inventory_sha256": parent_manifest["code_inventory_sha256"],
        "relocation_contract_sha256": "a" * 64,
        "remote_git_checkout_performed": False,
        "artifacts_read_only": True,
        "runs_mode": "1777",
        "published_at": "2026-07-21T00:00:00+00:00",
    }
    publication = {
        "schema_version": publisher.PUBLICATION_RECEIPT_SCHEMA,
        "bundle_id": parent_manifest["bundle_id"],
        "contract_sha256": parent_manifest["contract_sha256"],
        "bundle_manifest_sha256": parent_plan["bundle_manifest_sha256"],
        "remote_bundle": parent_plan["remote_bundle"],
        "incoming_bundle": f"{remote_root}/.incoming-parent",
        "apply": True,
        "already_ready": False,
        "publication_complete": True,
        "uploaded_files": len(parent_files),
        "resumed_files": 0,
        "remote_write_performed": True,
        "ready": ready,
        "ready_sha256": canonical_sha256(ready),
    }
    publication["receipt_sha256"] = canonical_sha256(publication)
    publication_path = tmp_path / "parent" / "publication_receipt.json"
    _write_json(publication_path, publication)
    identity = delta.ParentIdentity(
        bundle_id=parent_manifest["bundle_id"],
        contract_sha256=parent_manifest["contract_sha256"],
        manifest_sha256=parent_plan["bundle_manifest_sha256"],
        plan_sha256=bundle.sha256_file(parent_plan_path),
        source_map_sha256=bundle.sha256_file(Path(parent_plan["local_sources"])),
        publication_file_sha256=bundle.sha256_file(publication_path),
        remote_bundle=parent_plan["remote_bundle"],
    )

    child_sources = {
        "artifacts/models/large-model.bin": large,
        "artifacts/runtime/requirements.lock": requirements,
        "artifacts/code/tools/worker.py": child_code,
        "artifacts/code/tools/tier1_final1000_stage_profiles.py": stage_profile,
    }
    child_files = {
        relative: _record_file(path) for relative, path in child_sources.items()
    }
    child_code_inventory = {
        relative: child_files[relative]
        for relative in (
            "artifacts/code/tools/worker.py",
            "artifacts/code/tools/tier1_final1000_stage_profiles.py",
        )
    }
    child_stable = {
        **copy.deepcopy(parent_stable),
        "files": child_files,
        "code_inventory": child_code_inventory,
        "code_inventory_sha256": canonical_sha256(child_code_inventory),
        "hard_spec": {
            "T_limit_C": 100.0,
            "size_W_max_mm": 1000.0,
            "size_L_max_mm": 1000.0,
            "size_H_max_mm": 750.0,
            "resonance_max_Hz": 20000.0,
        },
    }
    child_manifest = _sealed_manifest(child_stable)
    candidate_plan_path, candidate_plan = _standard_plan(
        tmp_path / "candidate",
        child_manifest,
        child_sources,
        remote_root=remote_root,
    )
    plan, manifest, receipt = delta.create_delta_plan(
        candidate_plan_path,
        parent_plan_path,
        publication_path,
        tmp_path / "planned",
        identity=identity,
    )
    plan_path = Path(plan["bundle_manifest"]).with_name("offload_plan.json")
    return {
        "plan_path": plan_path,
        "plan": plan,
        "manifest": manifest,
        "receipt": receipt,
        "identity": identity,
        "parent_plan_path": parent_plan_path,
        "parent_plan": parent_plan,
        "parent_manifest": parent_manifest,
        "parent_publication": publication,
        "parent_publication_path": publication_path,
        "parent_sources": parent_sources,
        "candidate_plan_path": candidate_plan_path,
        "candidate_plan": candidate_plan,
        "child_sources": child_sources,
        "child_code": child_code,
        "stage_profile": stage_profile,
    }


def test_multiseed_release_preparation_uses_full_closure_and_never_reuses_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _fixture(tmp_path / "fixture")
    checkout = {
        "revision": "1" * 40,
        "clean": True,
        "science_base_revision": multiseed_release.SCIENCE_BASE_REVISION,
    }
    monkeypatch.setattr(
        multiseed_release,
        "authenticate_checkout",
        lambda _root: copy.deepcopy(checkout),
    )
    closures = multiseed_release.repository_import_closure(REPO)
    assert multiseed_release.MANDATORY_CLOSURE.issubset(closures)
    assert "tools/tier1_final1000_slurm_launch.py" in closures
    assert "tools/tier1_final1000_stage_profiles.py" in closures
    assert "tools/tier1_corrected_current7_slurm_publish.py" in closures

    receipts = []
    for ordinal in range(2):
        output = tmp_path / f"release-{ordinal}"
        receipt = multiseed_release.prepare_stage(
            parent_plan_path=fixture["parent_plan_path"],
            parent_publication_path=fixture["parent_publication_path"],
            code_root=REPO,
            output_root=output,
            created_at="2026-07-22T00:00:00+00:00",
        )
        receipts.append(receipt)
        assert receipt["isolated_import_smoke"]["passed"] is True
        assert receipt["ready_reused"] is False
        assert receipt["remote_write_performed"] is False
        assert receipt["scheduler_submission_performed"] is False
        assert not any(path.name == "READY" for path in output.rglob("*"))
    assert receipts[0]["candidate_bundle_id"] == receipts[1]["candidate_bundle_id"]
    assert receipts[0]["candidate_contract_sha256"] == receipts[1][
        "candidate_contract_sha256"
    ]
    assert receipts[0]["delta_bundle_id"] == receipts[1]["delta_bundle_id"]


def test_multiseed_release_closure_fails_if_transitive_dependency_is_missing(
    tmp_path: Path,
):
    root = tmp_path / "code"
    (root / "tools").mkdir(parents=True)
    for relative in multiseed_release.ROOT_MODULES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="closure is incomplete"):
        multiseed_release.repository_import_closure(root)


class FakeDeltaTransport:
    def __init__(self, method: str = "reflink"):
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()
        self.runtimes: dict[str, dict[str, str]] = {}
        self.sealed: set[str] = set()
        self.promoted: set[str] = set()
        self.events: list[str] = []
        self.method = method
        self.write_count = 0
        self.fail_after: int | None = None
        self.fail_promote_once = False

    def _write(self, event: str) -> None:
        if self.fail_after is not None and self.write_count >= self.fail_after:
            raise RuntimeError("injected delta publication interruption")
        self.events.append(event)
        self.write_count += 1

    def add_parent(self, fixture: Mapping[str, Any]) -> None:
        plan = fixture["parent_plan"]
        manifest = fixture["parent_manifest"]
        root = plan["remote_bundle"]
        self.dirs.add(root)
        self.dirs.add(f"{root}/artifacts/python-site")
        self.files[f"{root}/bundle_manifest.json"] = Path(
            plan["bundle_manifest"]
        ).read_bytes()
        self.files[f"{root}/READY.json"] = delta._json_bytes(
            fixture["parent_publication"]["ready"]
        )
        for relative, source in fixture["parent_sources"].items():
            self.files[f"{root}/{relative}"] = source.read_bytes()
        self.runtimes[root] = dict(manifest["runtime"]["critical_packages"])

    def exists(self, path: str) -> bool:
        return path in self.files or path in self.dirs

    def is_dir(self, path: str) -> bool:
        return path in self.dirs

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        if path in self.dirs:
            return
        self._write(f"mkdir:{mode:o}:{path}")
        self.dirs.add(path)

    def upload_file(self, local: Path, remote: str) -> None:
        self._write(f"upload:{remote}")
        self.files[remote] = local.read_bytes()

    def replace(self, source: str, destination: str) -> None:
        self._write(f"replace:{destination}")
        self.files[destination] = self.files.pop(source)

    def write_control(self, path: str, value: bytes) -> None:
        self._write(f"control:{path}")
        self.files[path] = bytes(value)

    def write_ready(self, path: str, value: bytes) -> None:
        self._write(f"READY:{path}")
        self.files[path] = bytes(value)

    def file_record(self, path: str) -> Mapping[str, Any] | None:
        value = self.files.get(path)
        return None if value is None else _record_bytes(value)

    def choose_clone_method(
        self, source: str, probe_destination: str, expected: Mapping[str, Any]
    ) -> str:
        assert self.file_record(source) == expected
        self._write(f"probe:{self.method}:{probe_destination}")
        return self.method

    def clone_file(self, source: str, destination: str, method: str) -> None:
        assert method in {"reflink", "hardlink"}
        self._write(f"clone:{method}:{destination}")
        self.files[destination] = self.files[source]

    def clone_runtime_tree(self, source: str, destination: str, method: str) -> None:
        assert method in {"reflink", "hardlink"}
        self._write(f"runtime:{method}:{destination}")
        source_root = source.removesuffix("/artifacts/python-site")
        destination_root = destination.removesuffix("/artifacts/python-site")
        self.dirs.add(destination)
        self.runtimes[destination_root] = dict(self.runtimes[source_root])

    def verify_runtime(
        self, root: str, expected_packages: Mapping[str, str]
    ) -> Mapping[str, str]:
        return self.runtimes[root]

    def seal_permissions(self, root: str) -> None:
        self._write(f"seal:{root}")
        self.sealed.add(root)

    def verify_permissions(self, root: str, *, promoted: bool) -> bool:
        return root in self.sealed and (not promoted or root in self.promoted)

    def promote_directory(self, incoming: str, destination: str) -> None:
        if self.fail_promote_once:
            self.fail_promote_once = False
            self.events.append("promote-failed")
            raise RuntimeError("injected lost promotion response")
        self._write(f"promote:{destination}")
        moved_files = {
            destination + path[len(incoming) :]: value
            for path, value in list(self.files.items())
            if path == incoming or path.startswith(incoming + "/")
        }
        for path in list(self.files):
            if path == incoming or path.startswith(incoming + "/"):
                del self.files[path]
        self.files.update(moved_files)
        self.dirs = {
            destination + path[len(incoming) :]
            if path == incoming or path.startswith(incoming + "/")
            else path
            for path in self.dirs
        }
        if incoming in self.runtimes:
            self.runtimes[destination] = self.runtimes.pop(incoming)
        if incoming in self.sealed:
            self.sealed.remove(incoming)
            self.sealed.add(destination)
        self.promoted.add(destination)


class FakeBatchSession:
    def __init__(
        self,
        records: Mapping[str, Mapping[str, Any] | None],
        *,
        corrupt_output: str | None = None,
    ):
        self.records = records
        self.corrupt_output = corrupt_output
        self.commands: list[tuple[str, int]] = []

    def run(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        argv = shlex.split(command)
        assert argv[:2] == ["bash", "-lc"]
        script = argv[2]
        set_line = next(line for line in script.splitlines() if line.startswith("set --"))
        paths = shlex.split(set_line)[2:]
        rows = []
        for index, path in enumerate(paths):
            record = self.records[path]
            if record is None:
                rows.append(f"M\t{index}")
            else:
                rows.append(f"R\t{index}\t{record['size']}\t{record['sha256']}")
        if self.corrupt_output == "omit" and rows:
            rows.pop()
        elif self.corrupt_output == "bad-sha" and rows:
            fields = rows[0].split("\t")
            if fields[0] == "R":
                fields[3] = "not-a-sha"
                rows[0] = "\t".join(fields)
        return SimpleNamespace(exit_code=0, stdout="\n".join(rows) + "\n", stderr="")


def test_ssh_batch_records_are_command_bounded_and_complete():
    paths = [
        f"/gpfs/final bundle/artifacts/code/worker-{index:03d}-with-'quote'.py"
        for index in range(30)
    ]
    expected = {
        path: (
            None
            if index == 17
            else {"size": index + 1, "sha256": f"{index + 1:064x}"}
        )
        for index, path in enumerate(paths)
    }
    session = FakeBatchSession(expected)
    transport = delta.SSHDeltaTransport(session, "")
    transport._MAX_BATCH_COMMAND_BYTES = 1_024

    assert transport.file_records(paths) == expected
    assert len(session.commands) > 1
    assert all(
        len(command.encode("utf-8")) <= transport._MAX_BATCH_COMMAND_BYTES
        for command, _ in session.commands
    )
    assert all(
        timeout == transport._BATCH_RECORD_TIMEOUT_SECONDS
        for _, timeout in session.commands
    )


@pytest.mark.parametrize("corrupt_output", ["omit", "bad-sha"])
def test_ssh_batch_records_fail_closed_on_incomplete_or_invalid_output(
    corrupt_output: str,
):
    path = "/gpfs/final/artifacts/model.bin"
    session = FakeBatchSession(
        {path: {"size": 4, "sha256": "a" * 64}},
        corrupt_output=corrupt_output,
    )
    transport = delta.SSHDeltaTransport(session, "")
    with pytest.raises(RuntimeError, match="remote"):
        transport.file_records([path])


def test_inventory_batch_reader_covers_every_file_on_every_pass():
    files = {
        "artifacts/a.bin": {"size": 1, "sha256": "a" * 64},
        "artifacts/b.bin": {"size": 2, "sha256": "b" * 64},
    }

    class BatchOnlyTransport:
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def file_records(self, paths):
            self.calls.append(tuple(paths))
            return {
                path: files[path.removeprefix("/remote/")] for path in paths
            }

        def file_record(self, path):
            raise AssertionError(f"slow single-file fallback used for {path}")

    transport = BatchOnlyTransport()
    delta._verify_inventory(transport, "/remote", files, "child")
    delta._verify_inventory(transport, "/remote", files, "child")
    expected_paths = (
        "/remote/artifacts/a.bin",
        "/remote/artifacts/b.bin",
    )
    assert transport.calls == [expected_paths, expected_paths]


def test_plan_is_content_addressed_and_classifies_only_small_delta(tmp_path: Path):
    fixture = _fixture(tmp_path)
    plan = fixture["plan"]
    manifest = fixture["manifest"]
    lineage = manifest["delta_publication"]
    loaded_plan, loaded_manifest, *_ = delta.load_delta_plan(
        fixture["plan_path"], required_parent=fixture["identity"]
    )
    assert loaded_plan == plan
    assert loaded_manifest == manifest
    assert plan["bundle_id"] != fixture["candidate_plan"]["bundle_id"]
    assert plan["bundle_id"] == f"current7-{manifest['contract_sha256'][:20]}"
    assert lineage["inherited_file_count"] == 2
    assert lineage["delta_file_count"] == 2
    assert lineage["removed_file_count"] == 0
    assert lineage["runtime_byte_identical_to_parent"] is True
    assert lineage["regular_copy_fallback_allowed"] is False
    assert set(plan["delta_publication"]["payload_upload_paths"]) == {
        "artifacts/code/tools/worker.py",
        "artifacts/code/tools/tier1_final1000_stage_profiles.py",
    }
    publisher.load_bundle_plan(fixture["plan_path"])


@pytest.mark.parametrize("method", ["reflink", "hardlink"])
def test_publish_clones_inherited_and_uploads_only_delta_ready_last(
    tmp_path: Path, method: str
):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport(method)
    remote.add_parent(fixture)
    receipt = delta.publish_delta(
        fixture["plan_path"],
        apply=True,
        transport=remote,
        required_parent=fixture["identity"],
    )
    publisher.validate_publication_receipt(
        receipt, plan=fixture["plan"], manifest=fixture["manifest"]
    )
    assert receipt["server_side_cloned_files"] == 2
    assert receipt["uploaded_files"] == 2
    assert receipt["network_payload_bytes_uploaded"] == sum(
        fixture["manifest"]["files"][relative]["size"]
        for relative in fixture["plan"]["delta_publication"]["payload_upload_paths"]
    )
    uploads = [event for event in remote.events if event.startswith("upload:")]
    assert len(uploads) == 2
    assert all("large-model.bin" not in event for event in uploads)
    assert any(event.startswith(f"runtime:{method}:") for event in remote.events)
    ready_index = next(
        index for index, event in enumerate(remote.events) if event.startswith("READY:")
    )
    assert remote.events[ready_index + 1].startswith("promote:")
    assert not any("regular-copy" in event for event in remote.events)
    final = fixture["plan"]["remote_bundle"]
    for relative, expected in fixture["manifest"]["files"].items():
        assert remote.file_record(f"{final}/{relative}") == expected


def test_dry_run_is_zero_write_and_parent_corruption_fails_before_write(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport()
    remote.add_parent(fixture)
    dry = delta.publish_delta(
        fixture["plan_path"],
        apply=False,
        transport=remote,
        required_parent=fixture["identity"],
    )
    assert dry["publication_complete"] is False
    assert dry["network_payload_bytes_uploaded"] == 0
    assert remote.write_count == 0
    parent_file = (
        fixture["parent_plan"]["remote_bundle"] + "/artifacts/models/large-model.bin"
    )
    remote.files[parent_file] = b"corrupt-parent"
    with pytest.raises(RuntimeError, match="parent file authentication failed"):
        delta.publish_delta(
            fixture["plan_path"],
            apply=True,
            transport=remote,
            required_parent=fixture["identity"],
        )
    assert remote.write_count == 0


def test_local_delta_tamper_and_size_limit_fail_before_remote_write(tmp_path: Path):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport()
    remote.add_parent(fixture)
    fixture["child_code"].write_bytes(b"tampered-after-plan")
    with pytest.raises(RuntimeError, match="delta local source authentication failed"):
        delta.publish_delta(
            fixture["plan_path"],
            apply=True,
            transport=remote,
            required_parent=fixture["identity"],
        )
    assert remote.write_count == 0

    fresh = _fixture(tmp_path / "limit")
    with pytest.raises(RuntimeError, match="small-file limit"):
        delta.create_delta_plan(
            fresh["candidate_plan_path"],
            fresh["parent_plan_path"],
            fresh["parent_publication_path"],
            tmp_path / "rejected",
            identity=fresh["identity"],
            max_delta_file_bytes=1,
        )


def test_interrupted_publication_resumes_exact_files_and_seals_once(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport()
    remote.add_parent(fixture)
    remote.fail_after = 8
    with pytest.raises(RuntimeError, match="interruption"):
        delta.publish_delta(
            fixture["plan_path"],
            apply=True,
            transport=remote,
            required_parent=fixture["identity"],
        )
    assert not any(event.startswith("READY:") for event in remote.events)
    remote.fail_after = None
    receipt = delta.publish_delta(
        fixture["plan_path"],
        apply=True,
        transport=remote,
        required_parent=fixture["identity"],
    )
    assert receipt["publication_complete"] is True
    assert receipt["resumed_files"] > 0
    assert len([event for event in remote.events if event.startswith("READY:")]) == 1


def test_ready_before_lost_promotion_is_adopted_without_rewrite(tmp_path: Path):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport()
    remote.add_parent(fixture)
    remote.fail_promote_once = True
    with pytest.raises(RuntimeError, match="lost promotion"):
        delta.publish_delta(
            fixture["plan_path"],
            apply=True,
            transport=remote,
            required_parent=fixture["identity"],
        )
    ready_count = len([event for event in remote.events if event.startswith("READY:")])
    receipt = delta.publish_delta(
        fixture["plan_path"],
        apply=True,
        transport=remote,
        required_parent=fixture["identity"],
    )
    assert receipt["resumed_files"] == len(fixture["manifest"]["files"])
    assert len([event for event in remote.events if event.startswith("READY:")]) == (
        ready_count
    )


def test_existing_ready_is_reverified_and_incoming_mismatch_fails_closed(
    tmp_path: Path,
):
    fixture = _fixture(tmp_path)
    remote = FakeDeltaTransport()
    remote.add_parent(fixture)
    delta.publish_delta(
        fixture["plan_path"],
        apply=True,
        transport=remote,
        required_parent=fixture["identity"],
    )
    child_file = fixture["plan"]["remote_bundle"] + "/artifacts/models/large-model.bin"
    remote.files[child_file] = b"corrupt-child-after-ready"
    with pytest.raises(RuntimeError, match="child file authentication failed"):
        delta.publish_delta(
            fixture["plan_path"],
            apply=True,
            transport=remote,
            required_parent=fixture["identity"],
        )

    other = _fixture(tmp_path / "incoming")
    remote2 = FakeDeltaTransport()
    remote2.add_parent(other)
    dry = delta.publish_delta(other["plan_path"], required_parent=other["identity"])
    remote2.dirs.add(dry["incoming_bundle"])
    bad = dry["incoming_bundle"] + "/artifacts/models/large-model.bin"
    remote2.files[bad] = b"wrong-incoming-bytes"
    with pytest.raises(RuntimeError, match="incoming inherited file is mismatched"):
        delta.publish_delta(
            other["plan_path"],
            apply=True,
            transport=remote2,
            required_parent=other["identity"],
        )


def test_cli_exposes_explicit_plan_and_dry_run_publish_surface():
    process = subprocess.run(
        [sys.executable, "tools/tier1_final1000_delta_publish.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "{plan,publish}" in process.stdout
    publish_help = subprocess.run(
        [
            sys.executable,
            "tools/tier1_final1000_delta_publish.py",
            "publish",
            "--help",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert publish_help.returncode == 0
    assert "--apply" in publish_help.stdout
