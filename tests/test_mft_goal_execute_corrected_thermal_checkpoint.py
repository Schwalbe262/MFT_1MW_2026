from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from tools import mft_goal_corrected_thermal_submission as submission


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools" / "mft_goal_execute_corrected_thermal_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mft_goal_execute_corrected_thermal_checkpoint", MODULE_PATH
)
assert SPEC and SPEC.loader
executor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = executor
SPEC.loader.exec_module(executor)


def _wcp_aedt_bytes() -> bytes:
    blocks = []
    for name in executor.WCP_SYMMETRY_REGION_NAMES:
        padding = "\n".join(
            [
                f"'{direction}PaddingType'='Absolute Offset'\n"
                f"'{direction}Padding'='2mm'"
                for direction in executor.WCP_PADDING_DIRECTIONS
            ]
        )
        blocks.append(
            "\n".join(
                [
                    "$begin 'GeometryPart'",
                    "$begin 'Attributes'",
                    f"Name='{name}'",
                    "$end 'Attributes'",
                    "$begin 'Operations'",
                    "$begin 'Operation'",
                    "OperationType='SubRegion'",
                    "$begin 'SubRegionParameters'",
                    padding,
                    "$end 'SubRegionParameters'",
                    "$end 'Operation'",
                    "$end 'Operations'",
                    "$end 'GeometryPart'",
                ]
            )
        )
    return ("\n".join(blocks) + "\n").encode("ascii")


def _file_entry(root: Path, path: Path) -> dict:
    stat_result = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat_result.st_size,
        "allocated": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "mode": oct(stat_result.st_mode & 0o777),
        "inode": stat_result.st_ino,
        "device": stat_result.st_dev,
        "sha256": executor.sha256_file(path),
    }


def _checkpoint(root: Path) -> tuple[Path, dict]:
    checkpoint = root / "checkpoint"
    results = (
        checkpoint
        / "simulation.aedtresults"
        / "icepak_thermal.results"
    )
    results.mkdir(parents=True)
    aedt = checkpoint / "simulation.aedt"
    aedt.write_bytes(_wcp_aedt_bytes())
    for family, suffix in (
        ("DV274_Meshes", "_V213.sd"),
        ("DV274_S271_Meshes", "_V0.sd"),
    ):
        for index in (0, 9, 10, 11, 12):
            directory = results / f"{family}{index}{suffix}"
            directory.mkdir()
            (directory / "grid_mapping").write_bytes(
                f"map-{family}-{index}".encode()
            )
            (directory / "grid_output").write_bytes(
                f"grid-{family}-{index}".encode()
            )
    result_root = checkpoint / "simulation.aedtresults"
    (result_root / "ManagedFiles_Design7.asol").write_bytes(b"managed")
    (result_root / "icepak_thermal.asol").write_bytes(b"saved-solution-map")
    (results / "DV274_S271_V0.profile").write_bytes(b"profile-zero")
    (results / "DV274_S271_V275.profile").write_bytes(b"profile-final")
    files = [
        _file_entry(checkpoint, path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    ]
    metadata_sha = "9" * 64
    required_budget = sum(item["size"] for item in files) + 1024 * 1024
    quota_evidence = {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": os.getuid() if hasattr(os, "getuid") else 0,
        "usage_bytes": 1024,
        "soft_limit_bytes": 10 * 1024**4,
        "hard_limit_bytes": 11 * 1024**4,
        "in_doubt_bytes": 0,
        "files_used": 100,
        "files_soft_limit": 1_000_000,
        "files_hard_limit": 1_100_000,
        "files_in_doubt": 0,
        "observed_at_epoch": 1_000.0,
        "source": "gate2:mmlsquota-Y",
    }
    quota_evidence["canonical_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(quota_evidence)
    ).hexdigest()
    quota_evidence["age_seconds_at_validation"] = 1.0
    filesystem_before = {
        "anchor_device": 42,
        "bavail": 20_000_000,
        "block_size": 4096,
        "free_bytes": 20_000_000 * 4096,
        "fsid": 84,
        "readonly": False,
        "required_free_bytes": (
            required_budget + executor.PHYSICAL_FREE_RESERVE_BYTES
        ),
    }
    filesystem_after = {
        **filesystem_before,
        "bavail": 19_000_000,
        "free_bytes": 19_000_000 * 4096,
        "required_free_bytes": executor.PHYSICAL_FREE_RESERVE_BYTES,
    }
    manifest = {
        "schema_version": executor.CHECKPOINT_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "composite_reauthentication_required": True,
        "post_login_quota_reauthentication_required": True,
        "source": "/sealed/source/simulation",
        "destination": str(checkpoint),
        "physics_boundary": {
            "fan_velocity_m_per_s": 1.5,
            "tim_conductivity_w_per_mk": 0.2,
            "thermal_pad_thickness_mm": 2.0,
        },
        "source_provenance": {
            "solver_revision": executor.SOURCE_SOLVER_REVISION,
            "library_revision": executor.PYAEDT_LIBRARY_REVISION,
            "candidate_sha256": executor.SOURCE_CANDIDATE_SHA256,
            "logical_task_id": executor.SOURCE_LOGICAL_TASK_ID,
            "execution_task_id": 96304,
            "slurm_job_id": 824575,
            "allocation_id": 14492,
            "node": "n114",
            "source_project_sha256": executor.sha256_file(aedt),
            "source_static_metadata_sha256": metadata_sha,
            "source_plan_identity_sha256": "8" * 64,
        },
        "quota_before_login_evidence": quota_evidence,
        "filesystem_before": filesystem_before,
        "filesystem_after_copy": filesystem_after,
        "source_snapshot_before": {
            "metadata_sha256": metadata_sha,
            "required_budget_bytes": required_budget,
        },
        "source_snapshot_after": {
            "metadata_sha256": metadata_sha,
            "required_budget_bytes": required_budget,
        },
        "files": files,
    }
    manifest["manifest_payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(manifest)
    ).hexdigest()
    (checkpoint / ".checkpoint_manifest.json").write_bytes(
        executor.canonical_json_bytes(manifest)
    )
    return checkpoint, manifest


def _execution_plan(
    path: Path, checkpoint_authentication: dict
) -> Path:
    storage = {
        "mode": "node_local_scratch_with_gpfs_minimum_retention",
        "scratch_root": str(path.parent / "scratch"),
        "minimum_scratch_working_shadow_bytes": (
            executor.MINIMUM_SCRATCH_WORKING_SHADOW_BYTES
        ),
        "retained_filesystem": "gpfs",
        "retained_root": str(path.parent / "retained"),
        "maximum_minimum_bundle_bytes": (
            executor.MAXIMUM_MINIMUM_BUNDLE_BYTES
        ),
        "minimum_retained_headroom_after_bytes": (
            executor.MINIMUM_RETAINED_HEADROOM_BYTES
        ),
        "minimum_retained_inode_headroom_after": (
            executor.MINIMUM_RETAINED_INODE_HEADROOM
        ),
    }
    gib = 1024**3
    quota = {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": executor.ACCOUNT_UID,
        "name": executor.ACCOUNT_NAME,
        "usage_bytes": 100 * gib,
        "soft_limit_bytes": 400 * gib,
        "hard_limit_bytes": 450 * gib,
        "in_doubt_bytes": 1 * gib,
        "files_used": 1000,
        "files_soft_limit": 100_000,
        "files_hard_limit": 110_000,
        "files_in_doubt": 10,
    }
    authority = {
        "schema": executor.RUNTIME_QUOTA_AUTHORITY_SCHEMA,
        "source": "submission-login:mmlsquota-Y",
        "account_name": executor.ACCOUNT_NAME,
        "account_uid": executor.ACCOUNT_UID,
        "observed_at_epoch": time.time(),
        "maximum_age_seconds": (
            executor.RUNTIME_QUOTA_AUTHORITY_MAX_AGE_SECONDS
        ),
        "conservatism": {
            "maximum_usage_growth_bytes": (
                executor.RUNTIME_QUOTA_DRIFT_RESERVE_BYTES
            ),
            "maximum_file_growth": (
                executor.RUNTIME_QUOTA_DRIFT_RESERVE_INODES
            ),
        },
        "quota": quota,
        "admission": {
            "filesystem": "gpfs",
            "quota_type": "USR",
            "available_before_bytes": 283 * gib,
            "retention_budget_bytes": 4 * gib,
            "headroom_after_bytes": 279 * gib,
            "available_before_inodes": 94_894,
            "retention_budget_inodes": 32,
            "headroom_after_inodes": 94_862,
        },
    }
    authority["payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(authority)
    ).hexdigest()
    revision = executor.EXECUTOR_REQUIRED_ANCESTOR
    core_auth = executor.core_contract_auth_sha256(revision)
    core_policy = {
        "schema": executor.CORE_POLICY_SCHEMA,
        "backend": "standalone",
        "contract_version": executor.CORE_CONTRACT_VERSION,
        "requested_num_cores": executor.CORES,
        "num_tasks": executor.TASKS,
        "required_slurm_cpus_per_task": executor.CORES,
        "solver_revision": revision,
        "auth_sha256": core_auth,
        "environment": {
            executor.CORE_CONTRACT_ENV: executor.CORE_CONTRACT_VERSION,
            executor.CORE_COUNT_ENV: str(executor.CORES),
            executor.CORE_AUTH_ENV: core_auth,
        },
    }
    plan = {
        "schema": executor.EXECUTION_PLAN_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "checkpoint_manifest_sha256": checkpoint_authentication[
            "manifest_sha256"
        ],
        "executor_revision": revision,
        "required_executor_ancestor": executor.EXECUTOR_REQUIRED_ANCESTOR,
        "tool_payload_sha256": executor.sha256_file(MODULE_PATH),
        "dispatch": {
            "cores": 8,
            "tasks": 1,
            "use_auto_settings": False,
        },
        "core_policy": core_policy,
        "runtime_quota_authority": authority,
        "output_storage": storage,
        "symmetry_mesh_region_repair": executor.symmetry_repair_contract(
            checkpoint_authentication["files"][
                next(
                    index
                    for index, row in enumerate(
                        checkpoint_authentication["files"]
                    )
                    if row["path"]
                    == checkpoint_authentication["aedt_relative_path"]
                )
            ]["sha256"]
        ),
    }
    plan["plan_payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(plan)
    ).hexdigest()
    path.write_bytes(executor.canonical_json_bytes(plan))
    return path


def _reseal_checkpoint_manifest(checkpoint: Path, manifest: dict) -> None:
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(manifest)
    ).hexdigest()
    (checkpoint / ".checkpoint_manifest.json").write_bytes(
        executor.canonical_json_bytes(manifest)
    )


def test_authenticates_every_byte_and_keeps_provenance_separate(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)

    result = executor.authenticate_checkpoint(checkpoint)

    assert result["source_provenance"]["solver_revision"] == (
        executor.SOURCE_SOLVER_REVISION
    )
    assert (
        result["source_provenance"]["solver_revision"]
        != executor.EXECUTOR_REQUIRED_ANCESTOR
    )
    assert len(result["files"]) == 25
    assert result["physics_boundary"] == {
        "fan_velocity_m_per_s": 1.5,
        "tim_conductivity_w_per_mk": 0.2,
        "thermal_pad_thickness_mm": 2.0,
    }


def test_authentication_rejects_one_mutated_premesh_byte(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    target = next(checkpoint.rglob("grid_output"))
    original_mtime = target.stat().st_mtime_ns
    target.write_bytes(target.read_bytes() + b"x")
    os.utime(target, ns=(original_mtime, original_mtime))

    with pytest.raises(
        executor.ContinuationError, match="metadata mismatch|hash mismatch"
    ):
        executor.authenticate_checkpoint(checkpoint)


def test_authentication_rejects_source_executor_provenance_collapse(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint(tmp_path)
    manifest["source_provenance"]["solver_revision"] = (
        executor.EXECUTOR_REQUIRED_ANCESTOR
    )
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(manifest)
    ).hexdigest()
    (checkpoint / ".checkpoint_manifest.json").write_bytes(
        executor.canonical_json_bytes(manifest)
    )

    with pytest.raises(executor.ContinuationError, match="not a1e4f"):
        executor.authenticate_checkpoint(checkpoint)


def test_v3_requires_post_login_quota_reauthentication_marker(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint(tmp_path)
    manifest["post_login_quota_reauthentication_required"] = False
    _reseal_checkpoint_manifest(checkpoint, manifest)

    with pytest.raises(
        executor.ContinuationError, match="post-login quota"
    ):
        executor.authenticate_checkpoint(checkpoint)


def test_v3_rejects_tampered_login_quota_canonical_evidence(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint(tmp_path)
    manifest["quota_before_login_evidence"]["usage_bytes"] += 1
    _reseal_checkpoint_manifest(checkpoint, manifest)

    with pytest.raises(
        executor.ContinuationError, match="canonical SHA-256 mismatch"
    ):
        executor.authenticate_checkpoint(checkpoint)


def test_execution_plan_binds_checkpoint_descendant_and_tool_payload(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)

    plan = executor.authenticate_execution_plan(plan_path, authenticated)

    assert plan["required_executor_ancestor"] == (
        executor.EXECUTOR_REQUIRED_ANCESTOR
    )
    assert plan["checkpoint_manifest_sha256"] == authenticated["manifest_sha256"]


def test_execution_plan_rejects_tool_payload_drift(tmp_path: Path) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["tool_payload_sha256"] = "0" * 64
    plan.pop("plan_payload_sha256")
    plan["plan_payload_sha256"] = executor.hashlib.sha256(
        executor.canonical_json_bytes(plan)
    ).hexdigest()
    plan_path.write_bytes(executor.canonical_json_bytes(plan))

    with pytest.raises(executor.ContinuationError, match="tool payload"):
        executor.authenticate_execution_plan(plan_path, authenticated)


def test_clone_is_writable_complete_and_does_not_mutate_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authentication = executor.authenticate_checkpoint(checkpoint)
    source_hashes = {
        item["path"]: executor.sha256_file(
            checkpoint.joinpath(*Path(item["path"]).parts)
        )
        for item in authentication["files"]
    }

    clone = executor.clone_checkpoint(
        authentication, tmp_path / "continuations"
    )

    clone_root = Path(clone["clone_root"])
    assert clone_root.is_dir()
    assert Path(clone["aedt_path"]).is_file()
    assert Path(clone["results_path"]).is_dir()
    for item in authentication["files"]:
        source = checkpoint.joinpath(*Path(item["path"]).parts)
        destination = clone_root.joinpath(*Path(item["path"]).parts)
        assert executor.sha256_file(destination) == item["sha256"]
        assert executor.sha256_file(source) == source_hashes[item["path"]]
        assert destination.stat().st_mode & 0o200
    assert not list((tmp_path / "continuations").glob("*.incoming"))
    assert executor.attest_saved_premesh(
        authentication, clone_root
    )["passed"] is True


def test_exact_symmetry_repair_changes_only_eight_cloned_padding_bytes(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authentication = executor.authenticate_checkpoint(checkpoint)
    clone = executor.clone_checkpoint(
        authentication, tmp_path / "continuations"
    )
    source = checkpoint / "simulation.aedt"
    cloned = Path(clone["aedt_path"])
    source_before = source.read_bytes()
    clone_before = cloned.read_bytes()
    contract = executor.symmetry_repair_contract(
        executor.sha256_file(source)
    )

    evidence = executor.repair_cloned_aedt_symmetry_envelopes(
        authentication=authentication,
        clone=clone,
        contract=contract,
    )

    clone_after = cloned.read_bytes()
    changed = [
        (before, after)
        for before, after in zip(clone_before, clone_after)
        if before != after
    ]
    assert changed == [(ord("2"), ord("0"))] * 8
    assert evidence["changed_byte_count"] == 8
    assert source.read_bytes() == source_before
    assert clone_after.count(b"'+XPadding'='0mm'") == 4
    assert clone_after.count(b"'-ZPadding'='0mm'") == 4
    assert clone_after.count(b"'-XPadding'='2mm'") == 4
    assert clone_after.count(b"'+ZPadding'='2mm'") == 4


def test_symmetry_repair_rejects_nonexact_source_subregion(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint(tmp_path)
    aedt = checkpoint / "simulation.aedt"
    aedt.write_bytes(
        aedt.read_bytes().replace(b"'+XPadding'='2mm'", b"'+XPadding'='1mm'", 1)
    )
    manifest["source_provenance"]["source_project_sha256"] = (
        executor.sha256_file(aedt)
    )
    manifest["files"] = [
        _file_entry(checkpoint, path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != ".checkpoint_manifest.json"
    ]
    _reseal_checkpoint_manifest(checkpoint, manifest)
    authentication = executor.authenticate_checkpoint(checkpoint)
    clone = executor.clone_checkpoint(
        authentication, tmp_path / "continuations"
    )
    with pytest.raises(executor.ContinuationError, match="padding drifted"):
        executor.repair_cloned_aedt_symmetry_envelopes(
            authentication=authentication,
            clone=clone,
            contract=executor.symmetry_repair_contract(
                executor.sha256_file(aedt)
            ),
        )


def test_clone_publish_race_never_clobbers_competing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authentication = executor.authenticate_checkpoint(checkpoint)
    monkeypatch.setattr(
        executor.uuid, "uuid4", lambda: SimpleNamespace(hex="12345678")
    )
    monkeypatch.setattr(
        executor.time,
        "strftime",
        lambda *_args, **_kwargs: "0726T000000Z",
    )
    calls = 0

    def race_then_portable(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            destination.mkdir()
            (destination / "competitor").write_bytes(b"do-not-clobber")
            raise FileExistsError(destination)
        os.rename(source, destination)

    monkeypatch.setattr(executor, "_rename_noreplace", race_then_portable)
    root = tmp_path / "continuations"

    with pytest.raises(FileExistsError):
        executor.clone_checkpoint(authentication, root)

    destination = root / "ct_b7c_0726T000000Z_12345678"
    assert (destination / "competitor").read_bytes() == b"do-not-clobber"
    quarantines = list(root.glob(f"{destination.name}.incomplete.*"))
    assert len(quarantines) == 1


def test_retained_output_quota_admission_accounts_for_clone_and_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)
    plan = executor.authenticate_execution_plan(plan_path, authenticated)
    gib = 1024**3
    monkeypatch.setattr(
        executor.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=100 * gib // 4096,
            f_frsize=4096,
            f_flag=0,
            f_fsid=authenticated["filesystem_evidence"]["after_copy"][
                "fsid"
            ],
        ),
        raising=False,
    )
    result = executor.admit_retained_output_storage(
        authenticated, plan, tmp_path / "scratch"
    )

    assert result["passed"] is True
    assert result["minimum_bundle_budget_bytes"] == 4 * gib
    assert result["quota_source"] == (
        "sealed_fresh_login_node_mmlsquota_authority"
    )
    assert result["retained_physical_required_bytes"] == 54 * gib
    assert result["scratch_required_bytes"] > (
        executor.MINIMUM_SCRATCH_WORKING_SHADOW_BYTES
    )


@pytest.mark.parametrize(
    ("free_bytes", "flags", "fsid_delta", "message"),
    [
        (54 * 1024**3 - 4096, 0, 0, "physical free space"),
        (100 * 1024**3, 1, 0, "read-only"),
        (100 * 1024**3, 0, 1, "filesystem identity"),
    ],
)
def test_retained_output_statvfs_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_bytes: int,
    flags: int,
    fsid_delta: int,
    message: str,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)
    plan = executor.authenticate_execution_plan(plan_path, authenticated)
    expected_fsid = authenticated["filesystem_evidence"]["after_copy"][
        "fsid"
    ]
    monkeypatch.setattr(
        executor.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=free_bytes // 4096,
            f_frsize=4096,
            f_flag=flags,
            f_fsid=expected_fsid + fsid_delta,
        ),
        raising=False,
    )
    with pytest.raises(executor.ContinuationError, match=message):
        executor.admit_retained_output_storage(
            authenticated, plan, tmp_path / "scratch"
        )


def test_submission_and_executor_runtime_quota_contract_match() -> None:
    gib = 1024**3
    retention = {
        "mode": "node_local_scratch_with_gpfs_minimum_retention",
        "scratch_root": "/enroot/mft-corrected-test/output",
        "minimum_scratch_working_shadow_bytes": 256 * gib,
        "retained_filesystem": "gpfs",
        "retained_root": "/gpfs/home1/r1jae262/test",
        "maximum_minimum_bundle_bytes": 4 * gib,
        "minimum_retained_headroom_after_bytes": 8 * gib,
        "minimum_retained_inode_headroom_after": 4096,
    }
    quota = {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": executor.ACCOUNT_UID,
        "name": executor.ACCOUNT_NAME,
        "usage_bytes": 100 * gib,
        "soft_limit_bytes": 400 * gib,
        "hard_limit_bytes": 450 * gib,
        "in_doubt_bytes": 1 * gib,
        "files_used": 1000,
        "files_soft_limit": 100_000,
        "files_hard_limit": 110_000,
        "files_in_doubt": 10,
    }
    observed = time.time()
    authority = submission.build_runtime_quota_authority(
        quota,
        retention=retention,
        now=submission.datetime.fromtimestamp(
            observed, tz=submission.timezone.utc
        ),
    )
    authenticated = executor._authenticate_runtime_quota_authority(
        authority,
        storage=retention,
        now_epoch=observed,
    )
    assert authenticated["payload_sha256"] == authority["payload_sha256"]
    assert authenticated["age_seconds_at_execution"] == pytest.approx(
        0.0, abs=1e-5
    )


def test_authenticated_core_policy_reaches_executor_as_exact_8x1() -> None:
    revision = executor.EXECUTOR_REQUIRED_ANCESTOR
    auth = executor.core_contract_auth_sha256(revision)
    policy = executor._authenticate_core_policy(
        {
            "schema": executor.CORE_POLICY_SCHEMA,
            "backend": "standalone",
            "contract_version": executor.CORE_CONTRACT_VERSION,
            "requested_num_cores": 8,
            "num_tasks": 1,
            "required_slurm_cpus_per_task": 8,
            "solver_revision": revision,
            "auth_sha256": auth,
            "environment": {
                executor.CORE_CONTRACT_ENV: executor.CORE_CONTRACT_VERSION,
                executor.CORE_COUNT_ENV: "8",
                executor.CORE_AUTH_ENV: auth,
            },
        },
        executor_revision=revision,
    )
    result = executor.authenticate_core_environment(
        policy,
        environ={
            **policy["environment"],
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_SCHED_TASK_ID": "96310",
            "SLURM_JOB_ID": "838000",
        },
    )
    assert result["passed"] is True
    assert result["slurm_cpus_per_task"] == 8


def test_core_environment_rejects_default_four_core_fallback() -> None:
    revision = executor.EXECUTOR_REQUIRED_ANCESTOR
    auth = executor.core_contract_auth_sha256(revision)
    policy = executor._authenticate_core_policy(
        {
            "schema": executor.CORE_POLICY_SCHEMA,
            "backend": "standalone",
            "contract_version": executor.CORE_CONTRACT_VERSION,
            "requested_num_cores": 8,
            "num_tasks": 1,
            "required_slurm_cpus_per_task": 8,
            "solver_revision": revision,
            "auth_sha256": auth,
            "environment": {
                executor.CORE_CONTRACT_ENV: executor.CORE_CONTRACT_VERSION,
                executor.CORE_COUNT_ENV: "8",
                executor.CORE_AUTH_ENV: auth,
            },
        },
        executor_revision=revision,
    )
    with pytest.raises(executor.ContinuationError, match="environment mismatch"):
        executor.authenticate_core_environment(
            policy,
            environ={
                "SLURM_CPUS_PER_TASK": "8",
                "SLURM_SCHED_TASK_ID": "96310",
                "SLURM_JOB_ID": "838000",
            },
        )


def test_minimum_retention_keeps_evidence_without_duplicate_premesh(
    tmp_path: Path,
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)
    plan = executor.authenticate_execution_plan(plan_path, authenticated)
    clone = executor.clone_checkpoint(authenticated, tmp_path / "scratch")
    clone_root = Path(clone["clone_root"])
    receipt = clone_root / "corrected_thermal_diagnostic_receipt.json"
    result = clone_root / "corrected_result.json"
    receipt.write_text('{"status":"diagnostic_complete"}', encoding="utf-8")
    result.write_text('{"diagnostic_only":true}', encoding="utf-8")
    monitor = (
        Path(clone["results_path"])
        / "icepak_thermal.results"
        / "DV274_S271_MON0_V0.sd"
    )
    monitor.write_bytes(b"converged-monitor")

    retained = executor.retain_minimum_artifacts(
        authentication=authenticated,
        execution_plan=plan,
        clone=clone,
        receipt_path=receipt,
        corrected_result_path=result,
        convergence={"thermal_monitor_file": str(monitor)},
        corrected_profile_path=(
            Path(clone["results_path"])
            / "icepak_thermal.results"
            / "DV274_S271_V275.profile"
        ),
    )

    destination = Path(retained["destination"])
    assert (destination / "symmetric.aedt").is_file()
    assert (destination / "corrected_result.json").is_file()
    assert (destination / "execution_receipt.json").is_file()
    assert (destination / "manifest.json").is_file()
    assert (destination / ".slurm-scheduler-preserve.json").is_file()
    assert (destination / "retention_receipt.json").is_file()
    assert not list(destination.rglob("grid_mapping"))
    assert not list(destination.rglob("grid_output"))
    assert retained["optional_field_bundle"]["retained"] is False
    assert retained["published_tree"]["tree_verified"] is True
    assert retained["retention_receipt_sha256"] == executor.sha256_file(
        destination / "retention_receipt.json"
    )


@pytest.mark.parametrize("mode", [0o400, 0o500])
def test_retention_protection_metadata_is_fsynced_after_fchmod(
    monkeypatch: pytest.MonkeyPatch, mode: int
) -> None:
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        executor.os,
        "fsync",
        lambda descriptor: events.append(("fsync", descriptor)),
    )
    monkeypatch.setattr(
        executor.os,
        "fchmod",
        lambda descriptor, requested: events.append(
            ("fchmod", descriptor, requested)
        ),
        raising=False,
    )

    executor._fsync_and_protect_descriptor(23, mode, protect=True)

    assert events == [
        ("fsync", 23),
        ("fchmod", 23, mode),
        ("fsync", 23),
    ]


def test_retention_publish_race_never_clobbers_competing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    plan_path = _execution_plan(tmp_path / "plan.json", authenticated)
    plan = executor.authenticate_execution_plan(plan_path, authenticated)
    clone = executor.clone_checkpoint(authenticated, tmp_path / "scratch")
    clone_root = Path(clone["clone_root"])
    receipt = clone_root / "corrected_thermal_diagnostic_receipt.json"
    result = clone_root / "corrected_result.json"
    receipt.write_text('{"status":"diagnostic_complete"}', encoding="utf-8")
    result.write_text('{"diagnostic_only":true}', encoding="utf-8")
    monitor = (
        Path(clone["results_path"])
        / "icepak_thermal.results"
        / "DV274_S271_MON0_V0.sd"
    )
    monitor.write_bytes(b"converged-monitor")
    calls = 0

    def race_then_portable(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            destination.mkdir()
            (destination / "competitor").write_bytes(b"do-not-clobber")
            raise FileExistsError(destination)
        os.rename(source, destination)

    monkeypatch.setattr(executor, "_rename_noreplace", race_then_portable)

    with pytest.raises(FileExistsError):
        executor.retain_minimum_artifacts(
            authentication=authenticated,
            execution_plan=plan,
            clone=clone,
            receipt_path=receipt,
            corrected_result_path=result,
            convergence={"thermal_monitor_file": str(monitor)},
            corrected_profile_path=(
                Path(clone["results_path"])
                / "icepak_thermal.results"
                / "DV274_S271_V275.profile"
            ),
        )

    label = (
        f"b7c-{authenticated['manifest_sha256'][:12]}-"
        f"{plan['executor_revision'][:12]}"
    )
    retained_root = Path(plan["output_storage"]["retained_root"])
    assert (retained_root / label / "competitor").read_bytes() == (
        b"do-not-clobber"
    )
    assert len(list(retained_root.glob(f"{label}.incomplete.*"))) == 1


def test_corrected_profile_must_be_new_or_changed(tmp_path: Path) -> None:
    checkpoint, _manifest = _checkpoint(tmp_path)
    authenticated = executor.authenticate_checkpoint(checkpoint)
    clone = executor.clone_checkpoint(authenticated, tmp_path / "scratch")
    results = Path(clone["results_path"])
    before = executor.snapshot_profiles(results)

    with pytest.raises(executor.ContinuationError, match="no fresh profile"):
        executor.fresh_corrected_profile(results, before)

    profile = (
        results
        / "icepak_thermal.results"
        / "DV274_S271_V275.profile"
    )
    profile.write_bytes(profile.read_bytes() + b"-corrected")
    assert executor.fresh_corrected_profile(results, before) == profile


class _Child:
    def __init__(self, props: dict[str, object]):
        self.props = props

    def GetPropNames(self):
        return list(self.props)

    def GetPropValue(self, name):
        return self.props[name]


class _Boundary:
    def __init__(self, name: str, props: dict[str, object]):
        self.name = name
        self._child_object = _Child(props)


class _Editor:
    _boxes = {
        "Tx_main_0_0": [-500.0, -400.0, -300.0, -450.0, 400.0, 300.0],
        "Rx_main_block_xn": [-450.0, -300.0, -200.0, 450.0, 300.0, 200.0],
        "core_2": [-400.0, -350.0, -250.0, 400.0, 350.0, 250.0],
        "Tx_main_wcp_pad_1": [450.0, -1.0, -100.0, 500.0, 1.0, 100.0],
        "core_plate_pad_1": [-300.0, -1.0, -220.0, 300.0, 1.0, -200.0],
        "mesh_pad_subregion_air": [-550.0, -450.0, -350.0, 550.0, 450.0, 350.0],
        "Region": [-600.0, -500.0, -400.0, 600.0, 500.0, 400.0],
    }
    _materials = {
        "Tx_main_wcp_pad_1": '"thermal_pad"',
        "core_plate_pad_1": '"thermal_pad"',
        "mesh_pad_subregion_air": '"air"',
    }

    def GetObjectsInGroup(self, group):
        if group == "Non Model":
            return ["mesh_pad_subregion_air"]
        assert group == "Solids"
        return list(self._boxes)

    def GetObjectBoundingBox(self, name):
        return self._boxes[name]

    def GetPropertyValue(self, _tab, name, _property):
        return self._materials.get(name, '"copper"')


def test_native_readbacks_attest_fixed_fan_loss_geometry_and_pad_material() -> None:
    boundaries = [
        _Boundary(
            "fan_inlet",
            {
                "X Velocity": "0m_per_sec",
                "Y Velocity": "-1.5m_per_sec",
                "Z Velocity": "0m_per_sec",
            },
        ),
        _Boundary(
            "loss_tx",
            {
                "Total Power": "100W",
                "Objects": ["Tx_main_0_0"],
                "Type": "Solid",
            },
        ),
        _Boundary(
            "loss_rx",
            {
                "Total Power": "80W",
                "Objects": ["Rx_main_block_xn"],
                "Type": "Solid",
            },
        ),
        _Boundary(
            "loss_core",
            {
                "Total Power": "20W",
                "Objects": ["core_2"],
                "Type": "Solid",
            },
        ),
    ]
    ipk = SimpleNamespace(
        boundaries=boundaries,
        oeditor=_Editor(),
        modeler=SimpleNamespace(
            oeditor=_Editor(),
            model_units="mm",
            object_names=[
                "Tx_main_0_0",
                "Rx_main_block_xn",
                "core_2",
                "Tx_main_wcp_pad_1",
                "core_plate_pad_1",
                "mesh_pad_subregion_air",
            ],
        ),
    )

    inventory = executor._native_boundary_inventory(ipk)
    fan = executor._attest_fan(inventory)
    losses = executor._native_loss_assignments(inventory)
    geometry = executor._native_geometry(ipk)

    assert fan["passed"] is True
    assert losses["passed"] is True
    assert losses["total_power_w"] == 200.0
    assert losses["categories"] == {"winding": True, "core": True}
    assert geometry["passed"] is True
    assert geometry["dimensions_mm"] == {
        "width_x": 1000.0,
        "length_y": 800.0,
        "height_z": 600.0,
    }
    assert set(geometry["physical_pad_solids"]) == {
        "Tx_main_wcp_pad_1",
        "core_plate_pad_1",
    }
    assert all(
        row["y_thickness_mm"] == 2.0
        for row in geometry["physical_pad_solids"].values()
    )
    assert "mesh_pad_subregion_air" not in geometry["physical_pad_solids"]


def test_native_fan_and_physical_pad_readback_fail_closed() -> None:
    bad_fan = executor._native_boundary_inventory(
        SimpleNamespace(
            boundaries=[
                _Boundary(
                    "fan_inlet",
                    {
                        "X Velocity": "0m_per_sec",
                        "Y Velocity": "1.5m_per_sec",
                        "Z Velocity": "0m_per_sec",
                    },
                )
            ]
        )
    )
    with pytest.raises(executor.ContinuationError, match="vector drifted"):
        executor._attest_fan(bad_fan)

    editor = _Editor()
    editor._boxes = dict(_Editor._boxes)
    editor._boxes["Tx_main_wcp_pad_1"] = [
        450.0,
        -1.5,
        -100.0,
        500.0,
        1.5,
        100.0,
    ]
    ipk = SimpleNamespace(
        oeditor=editor,
        modeler=SimpleNamespace(oeditor=editor, model_units="mm"),
    )
    with pytest.raises(executor.ContinuationError, match="Y thickness drifted"):
        executor._native_geometry(ipk)


def test_parallel_evidence_requires_acf_one_by_eight_and_fluent_t8_mpi8() -> None:
    evidence = {
        "passed": True,
        "policy": {
            "expected_fluent_processes": 8,
            "expected_num_engines": 1,
            "pyaedt_cores_argument": 8,
            "pyaedt_tasks_argument": 1,
            "pyaedt_use_auto_settings_argument": False,
        },
        "attempts": [
            {
                "passed": True,
                "acf": {
                    "passed": True,
                    "num_cores_readback": 8,
                    "num_engines_readback": 1,
                },
                "process": {
                    "passed": True,
                    "thread_count_readbacks": [8],
                    "nprocs_count_readbacks": [8, 8],
                    "mismatched_process_counts": [],
                },
            }
        ],
    }

    assert executor._attest_parallel_evidence(evidence)["passed"] is True
    evidence["attempts"][0]["process"]["nprocs_count_readbacks"] = [1]
    with pytest.raises(executor.ContinuationError, match="Fluent -t8/MPI8"):
        executor._attest_parallel_evidence(evidence)


def test_executor_source_allows_only_repaired_mesh_not_physics_rebuild() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8")
    forbidden_calls = (
        ".cleanup_solution(",
        ".generate_mesh(",
        "run_thermal_analysis(",
        "_build_geometry(",
        "_assign_losses(",
        "_assign_boundaries(",
        "requests.post(",
        "sbatch ",
    )
    assert all(token not in text for token in forbidden_calls)
    assert "_solve_exact_thermal_setup(sim, ipk, setup)" in text
    assert "generator(THERMAL_SETUP)" in text
    assert '"fresh_native_mesh_generation": 1' in text
    assert '"physical_geometry_create_or_edit": 0' in text
