from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mft_goal_corrected_thermal_submission.py"
SPEC = importlib.util.spec_from_file_location("corrected_submission", MODULE_PATH)
assert SPEC and SPEC.loader
submission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submission
SPEC.loader.exec_module(submission)


NOW = datetime.fromisoformat("2026-07-26T11:00:00+09:00")


def _sha(character: str = "a") -> str:
    return character * 64


def _checkpoint_manifest(path: Path) -> Path:
    project = "simulation_824575_2786652"
    files = [
        {
            "path": f"{project}.aedt",
            "size": 100,
            "allocated": 4096,
            "mtime_ns": 1,
            "mode": "0o100400",
            "sha256": _sha("1"),
        }
    ]
    for name in (
        "ManagedFiles_Design7.asol",
        "icepak_thermal.asol",
        "icepak_thermal.results/DV274_S271_V0.profile",
        "icepak_thermal.results/DV274_S271_V275.profile",
    ):
        files.append(
            {
                "path": f"{project}.aedtresults/{name}",
                "size": 10,
                "allocated": 4096,
                "mtime_ns": 2,
                "mode": "0o100400",
                "sha256": _sha("2"),
            }
        )
    for family, suffix in (
        ("DV274_Meshes", "_V213.sd"),
        ("DV274_S271_Meshes", "_V0.sd"),
    ):
        for index in (0, 9, 10, 11, 12):
            for leaf in ("grid_mapping", "grid_output"):
                files.append(
                    {
                        "path": (
                            f"{project}.aedtresults/icepak_thermal.results/"
                            f"{family}{index}{suffix}/{leaf}"
                        ),
                        "size": 20,
                        "allocated": 4096,
                        "mtime_ns": 3,
                        "mode": "0o100400",
                        "sha256": _sha("3"),
                    }
                )
    assert len(files) == 25
    snapshot = {
        "metadata_sha256": _sha("4"),
        "required_budget_bytes": 26 * 1024**3,
        "required_inodes": 50,
        "included_count": 25,
        "directory_count": 15,
        "logical_bytes": 25 * 1024**3,
        "allocated_bytes": 25 * 1024**3,
    }
    quota_evidence = {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": submission.ACCOUNT_UID,
        "usage_bytes": 20 * 1024**3,
        "soft_limit_bytes": 100 * 1024**3,
        "hard_limit_bytes": 100 * 1024**3,
        "in_doubt_bytes": 0,
        "files_used": 100,
        "files_soft_limit": 100_000,
        "files_hard_limit": 100_000,
        "files_in_doubt": 0,
        "observed_at_epoch": 1_785_000_000.0,
        "source": "gate2:mmlsquota-Y",
    }
    quota_evidence["canonical_sha256"] = submission.canonical_sha256(
        quota_evidence
    )
    quota_evidence["age_seconds_at_validation"] = 1.0
    filesystem_before = {
        "anchor_device": 1,
        "bavail": 100 * 1024**3 // 4096,
        "block_size": 4096,
        "free_bytes": 100 * 1024**3,
        "fsid": 2,
        "readonly": False,
        "required_free_bytes": 76 * 1024**3,
    }
    filesystem_after = {
        **filesystem_before,
        "free_bytes": 70 * 1024**3,
        "bavail": 70 * 1024**3 // 4096,
        "required_free_bytes": 50 * 1024**3,
    }
    value = {
        "schema_version": submission.CHECKPOINT_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "composite_reauthentication_required": True,
        "post_login_quota_reauthentication_required": True,
        "destination": (
            f"{submission.CHECKPOINT_ROOT.as_posix()}/"
            "b7c-static-checkpoint-v1"
        ),
        "source_provenance": {
            "solver_revision": submission.SOURCE_SOLVER_REVISION,
            "library_revision": submission.LIBRARY_REVISION,
            "candidate_sha256": submission.CANDIDATE_SHA256,
            "logical_task_id": submission.SOURCE_LOGICAL_TASK_ID,
            "execution_task_id": submission.SOURCE_EXECUTION_TASK_ID,
            "source_plan_identity_sha256": (
                submission.SOURCE_PLAN_IDENTITY_SHA256
            ),
            "slurm_job_id": 824575,
            "allocation_id": 14492,
            "node": submission.FORBIDDEN_NODE,
            "source_project_sha256": _sha("1"),
            "source_static_metadata_sha256": _sha("4"),
        },
        "physics_boundary": {
            "fan_velocity_m_per_s": 1.5,
            "tim_conductivity_w_per_mk": 0.2,
            "thermal_pad_thickness_mm": 2.0,
        },
        "files": files,
        "source_snapshot_before": snapshot,
        "source_snapshot_after": dict(snapshot),
        "quota_before_login_evidence": quota_evidence,
        "filesystem_before": filesystem_before,
        "filesystem_after_copy": filesystem_after,
    }
    value["manifest_payload_sha256"] = submission.canonical_sha256(value)
    path.write_bytes(submission.canonical_json_bytes(value))
    return path


def _executor_identity() -> dict:
    entries = {}
    for index, relative in enumerate(
        (
            submission.STAGE_ENTRYPOINT,
            submission.EXECUTOR_ENTRYPOINT,
            submission.SUBMISSION_ENTRYPOINT,
        ),
        start=5,
    ):
        entries[relative] = {
            "git_blob_sha1": str(index) * 40,
            "payload_sha256": str(index) * 64,
        }
    return {
        "revision": "b" * 40,
        "required_ancestor": submission.EXECUTOR_REQUIRED_ANCESTOR,
        "tracked_worktree_clean": True,
        "remote_repository": submission.EXECUTOR_REPOSITORY,
        "remote_ref": submission.EXECUTOR_REMOTE_REF,
        "remote_revision_verified": True,
        "entrypoints": entries,
    }


def _plan(tmp_path: Path) -> tuple[dict, Path]:
    checkpoint = _checkpoint_manifest(tmp_path / "checkpoint.json")
    value = submission.build_plan(
        checkpoint_manifest=checkpoint,
        claim_root=tmp_path / "claims",
        executor_identity=_executor_identity(),
        now=NOW,
    )
    path = tmp_path / "plan.json"
    path.write_bytes(submission.canonical_json_bytes(value))
    return value, path


def _license() -> dict:
    return {
        "server_up": True,
        "error": "",
        "admission": {
            "enabled": True,
            "snapshot_valid": True,
            "snapshot_age_seconds": 2.0,
            "snapshot_max_age_seconds": 120.0,
            "blocked_reason": "",
            "features": {
                "electronics_desktop": {
                    "total": 550,
                    "used": 237,
                    "reserve": 24,
                    "admit_headroom": 289,
                }
            },
        },
    }


def _allocation(node: str = "n111", state: str = "mix") -> dict:
    return {
        "id": 1,
        "account_name": "dw16",
        "node_name": node,
        "state": "active",
        "node_pestat_state": state,
        "node_cpu_total": 256,
        "node_cpu_used": 120,
        "node_memory_free_mb": 732536,
        "node_metrics_observed_at": "2026-07-26T02:00:00+00:00",
    }


def _capacity() -> dict:
    return {
        "ready_fit_slots": 0,
        "memory_pressure_state": "ok",
        "preferred_node_relaxed": False,
        "allocations": [],
        "queue_state": "opening",
        "queue_reason": "opening demand pools",
    }


def _quota() -> dict:
    return {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": submission.ACCOUNT_UID,
        "name": "r1jae262",
        "usage_bytes": 20 * 1024**3,
        "soft_limit_bytes": 100 * 1024**3,
        "hard_limit_bytes": 100 * 1024**3,
        "in_doubt_bytes": 0,
        "files_used": 100,
        "files_soft_limit": 100_000,
        "files_hard_limit": 100_000,
        "files_in_doubt": 0,
    }


class FakeScheduler:
    def __init__(self, plan: dict, rows: list[dict] | None = None):
        self.plan = plan
        self.rows = list(rows or [])
        self.post_calls = 0
        self.response_loss = False
        self.stdout: dict[int, str] = {}

    def _page(self, params: dict) -> dict:
        before_id = int(params["before_id"])
        rows = sorted(self.rows, key=lambda row: row["id"], reverse=True)
        if before_id:
            rows = [row for row in rows if row["id"] < before_id]
        total = len(rows)
        page = int(params["page"])
        size = submission.INVENTORY_PAGE_SIZE
        start = (page - 1) * size
        return {
            "filtered_total": total,
            "page": page,
            "page_size": size,
            "page_count": max(1, (total + size - 1) // size),
            "sort_by": "id",
            "sort_order": "desc",
            "filters": {"before_id": before_id},
            "items": rows[start : start + size],
        }

    def _project(self) -> dict:
        active = sum(
            row.get("project") == submission.PROJECT
            and row.get("status") in submission.ACTIVE_STATUSES
            for row in self.rows
        )
        return {
            "name": submission.PROJECT,
            "repos": submission.PROJECT_REPOS,
            "setup": submission.PROJECT_SETUP,
            "entrypoints": submission.PROJECT_ENTRYPOINTS,
            "cleanup_globs": submission.PROJECT_CLEANUP_GLOBS,
            "output_globs": submission.PROJECT_OUTPUT_GLOBS,
            "sim_subdir": "simulation",
            "auto_pull": False,
            "max_active_tasks": 500,
            "aedt_backend": "standalone",
            "logical_active_count": active,
            "deployments": [
                {
                    "account_name": submission.ACCOUNT,
                    "status": "deployed",
                }
            ],
        }

    def get_json(self, path: str, params: dict | None = None):
        if path == "/api/tasks":
            return self._page(params or {})
        if path == f"/api/projects/{submission.PROJECT}":
            return self._project()
        if path == "/api/licenses":
            return _license()
        if path == "/api/allocations":
            return [_allocation()]
        if path == "/api/task-capacity":
            assert params == submission._capacity_params()
            return _capacity()
        if path.startswith("/api/tasks/"):
            task_id = int(path.rsplit("/", 1)[-1])
            return next(row for row in self.rows if row["id"] == task_id)
        raise AssertionError(path)

    def get_text(self, path: str, params: dict | None = None) -> str:
        task_id = int(path.split("/")[3])
        return self.stdout[task_id]

    def post_json(self, path: str, body: dict):
        assert path == "/api/tasks"
        self.post_calls += 1
        assert self.post_calls == 1
        row = {
            **body,
            "id": 100001,
            "task_id": 100001,
            "status": "queued",
            "state": "queued",
            "requested_node_name": "n111",
            "requested_node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "actual_node_name": "",
            "allocation_node_name": "",
        }
        row.pop("command")
        self.rows.append(row)
        if self.response_loss:
            raise submission.SubmissionUncertain("response lost")
        return {"task_id": 100001, "id": 100001}


def test_plan_binds_fresh_executor_and_two_tier_storage(tmp_path: Path) -> None:
    plan, path = _plan(tmp_path)
    loaded = submission.load_plan(path, verify_local_files=False)
    assert loaded == plan
    assert plan["executor"]["revision"] == "b" * 40
    assert plan["submission_profile"]["cpus"] == 8
    assert plan["submission_profile"]["memory_mb"] == 294912
    assert plan["submission_profile"]["node_name"] == "n111"
    assert plan["submission_profile"]["same_node_as_task_id"] == 0
    assert plan["submission_profile"]["timeout_seconds"] == 21600
    storage = plan["execution_contract"]["output_storage"]
    assert storage["mode"] == (
        "node_local_scratch_with_gpfs_minimum_retention"
    )
    assert storage["minimum_scratch_working_shadow_bytes"] == 256 * 1024**3
    assert storage["maximum_minimum_bundle_bytes"] == 4 * 1024**3
    assert "test \"$host\" = 'n111'" in plan["canonical_command"]
    assert "test \"$host\" != 'n114'" in plan["canonical_command"]
    assert "--execution-plan" in plan["canonical_command"]


def test_deadline_relative_timeout_fails_closed() -> None:
    assert submission.calculate_timeout(NOW) == 21600
    with pytest.raises(submission.CorrectedThermalError, match="minimum"):
        submission.calculate_timeout(
            datetime.fromisoformat("2026-07-26T15:00:01+09:00")
        )


def test_complete_paged_inventory_finds_dedupe_only_late_collision(
    tmp_path: Path,
) -> None:
    plan, _path = _plan(tmp_path)
    rows = [
        {
            "id": index,
            "task_id": index,
            "name": f"unrelated-{index}",
            "dedupe_key": f"other-{index}",
            "project": "",
            "status": "completed",
        }
        for index in range(1, 10002)
    ]
    rows[0]["dedupe_key"] = plan["task_identity"]["dedupe_key"]
    scheduler = FakeScheduler(plan, rows)
    inventory = submission.complete_task_inventory(scheduler)
    assert inventory["page_count"] == 2
    with pytest.raises(submission.CorrectedThermalError, match="collision"):
        submission.classify_collisions(
            inventory["rows"],
            task_name=plan["task_identity"]["name"],
            dedupe_key=plan["task_identity"]["dedupe_key"],
        )


def test_node_and_license_gates_reject_relaxation() -> None:
    accepted = submission.validate_node_gate(
        [_allocation()], _capacity(), now=NOW
    )
    assert accepted["queue_state"] == "opening"
    closed_carrier = _allocation()
    closed_carrier["state"] = "closed"
    accepted_closed = submission.validate_node_gate(
        [closed_carrier], _capacity(), now=NOW
    )
    assert accepted_closed["telemetry_carrier_allocation_state"] == "closed"
    bad = _capacity()
    bad["allocations"] = [{"node_name": "n114"}]
    with pytest.raises(submission.CorrectedThermalError, match="n111"):
        submission.validate_node_gate([_allocation()], bad, now=NOW)
    stale = _license()
    stale["admission"]["snapshot_age_seconds"] = 121
    with pytest.raises(submission.CorrectedThermalError, match="unsafe"):
        submission.validate_license_gate(stale)


def test_node_telemetry_rejects_stale_and_future() -> None:
    stale = _allocation()
    stale["node_metrics_observed_at"] = "2026-07-26T01:57:59+00:00"
    with pytest.raises(submission.CorrectedThermalError, match="cannot admit"):
        submission.validate_node_gate([stale], _capacity(), now=NOW)
    future = _allocation()
    future["node_metrics_observed_at"] = "2026-07-26T02:00:06+00:00"
    with pytest.raises(submission.CorrectedThermalError, match="cannot admit"):
        submission.validate_node_gate([future], _capacity(), now=NOW)
    edge = _allocation()
    edge["node_metrics_observed_at"] = "2026-07-26T02:00:05+00:00"
    assert submission.validate_node_gate(
        [edge], _capacity(), now=NOW
    )["node_metrics_age_seconds"] == -5.0


def test_storage_gate_binds_r1_name_and_uid(tmp_path: Path) -> None:
    plan, _path = _plan(tmp_path)
    accepted = submission.validate_storage_gate(_quota(), plan["retention"])
    assert accepted["retention_budget_inodes"] == 32
    wrong_uid = _quota()
    wrong_uid["uid"] = submission.ACCOUNT_UID + 1
    with pytest.raises(submission.CorrectedThermalError, match="identity"):
        submission.validate_storage_gate(wrong_uid, plan["retention"])
    wrong_name = _quota()
    wrong_name["name"] = "another"
    with pytest.raises(submission.CorrectedThermalError, match="identity"):
        submission.validate_storage_gate(wrong_name, plan["retention"])
    quota_command = submission._gpfs_quota_command()
    assert 'test "$u" = \'r1jae262\'' in quota_command
    assert 'test "$n" = \'1455\'' in quota_command
    assert '"$q" -u "$u" -Y gpfs' in quota_command


def test_default_submit_is_get_only(tmp_path: Path) -> None:
    plan, path = _plan(tmp_path)
    scheduler = FakeScheduler(plan)
    result = submission.submit_plan(
        path,
        client=scheduler,
        storage_probe=_quota,
        now=NOW,
        verify_local_files=False,
    )
    assert result["scheduler_submission_performed"] is False
    assert result["scheduler_post_calls"] == 0
    assert scheduler.post_calls == 0


@pytest.mark.parametrize("response_loss", [False, True])
def test_apply_posts_once_and_finalizes_durable_claim(
    tmp_path: Path, response_loss: bool
) -> None:
    plan, path = _plan(tmp_path)
    scheduler = FakeScheduler(plan)
    scheduler.response_loss = response_loss
    result = submission.submit_plan(
        path,
        apply=True,
        client=scheduler,
        storage_probe=_quota,
        now=NOW,
        verify_local_files=False,
        lock_path=tmp_path / "mutation.lock",
    )
    assert scheduler.post_calls == 1
    assert result["scheduler_post_calls"] == 1
    assert result["scheduler_submission_performed"] is True
    assert result["finalized_claim"]["state"] == "finalized"
    assert result["task_id"] == 100001


def _submitted_readback(plan: dict) -> tuple[dict, dict]:
    body = submission.build_submission_body(plan)
    task = {
        **body,
        "id": 100001,
        "task_id": 100001,
        "status": "queued",
        "state": "queued",
        "requested_node_name": "n111",
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "actual_node_name": "",
        "allocation_node_name": "",
    }
    task.pop("command")
    pending = {
        "winner": {
            "task_name": body["name"],
            "dedupe_key": body["dedupe_key"],
        }
    }
    return task, pending


def test_readback_requires_zero_same_node_and_exact_worker(tmp_path: Path) -> None:
    plan, _path = _plan(tmp_path)
    task, pending = _submitted_readback(plan)
    assert (
        submission.validate_task_readback(task, pending, plan=plan)[
            "same_node_as_task_id"
        ]
        == 0
    )
    for field, value in (
        ("same_node_as_task_id", None),
        ("same_node_as_task_id", 96304),
        ("max_workers_per_node", None),
        ("max_workers_per_node", 0),
    ):
        bad = dict(task)
        bad[field] = value
        with pytest.raises(submission.CorrectedThermalError):
            submission.validate_task_readback(bad, pending, plan=plan)


def _marker(plan: dict) -> dict:
    receipt = {
        "schema": "mft-corrected-thermal-checkpoint-execution-v1",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "source_checkpoint_manifest_sha256": plan["checkpoint_manifest"][
            "file_sha256"
        ],
        "source_provenance": {
            "candidate_sha256": submission.CANDIDATE_SHA256,
            "logical_task_id": submission.SOURCE_LOGICAL_TASK_ID,
            "execution_task_id": submission.SOURCE_EXECUTION_TASK_ID,
            "solver_revision": submission.SOURCE_SOLVER_REVISION,
        },
        "executor_provenance": {
            "executor_solver_revision": plan["executor"]["revision"],
            "pyaedt_library_revision": submission.LIBRARY_REVISION,
        },
        "fixed_physics": plan["checkpoint_manifest"]["physics_boundary"],
        "parallel_attestation": {"passed": True},
        "temperatures": {
            "T_max_Tx": 90.0,
            "T_max_Rx_main": 91.0,
            "T_max_Rx_side": 92.0,
            "T_max_core": 110.0,
        },
        "constraint_observation": {
            "winding_max_c": 92.0,
            "winding_limit_c": 100.0,
            "winding_pass": True,
            "core_max_c": 110.0,
            "core_limit_c": 120.0,
            "core_pass": True,
            "all_temperature_constraints_pass": True,
            "diagnostic_only": True,
        },
    }
    return submission.sealed(
        {
            "schema": submission.RETENTION_MARKER_SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "task_name": submission.TASK_NAME,
            "dedupe_key": plan["task_identity"]["dedupe_key"],
            "submission_contract_sha256": plan["contract_digest_sha256"],
            "execution_plan_payload_sha256": plan["execution_contract"][
                "plan_payload_sha256"
            ],
            "retained_destination": "/gpfs/example",
            "retention_manifest_path": "/gpfs/example/manifest.json",
            "retention_manifest_payload_sha256": _sha("8"),
            "retention_manifest_file_sha256": _sha("9"),
            "corrected_receipt_path": "/gpfs/example/execution_receipt.json",
            "corrected_receipt_sha256": _sha("a"),
            "execution_exit_code": 0,
            "verified_files": [],
            "minimum_bundle_bytes": 100,
            "optional_field_bundle": {"retained": False},
            "corrected_receipt": receipt,
        },
        digest_field="payload_sha256",
    )


class CollectorScheduler:
    def __init__(
        self,
        plan: dict,
        marker: dict,
        *,
        terminal_source: bool,
        source_exit_code: int | None = 0,
        corrected_exit_code: int | None = 0,
    ):
        self.plan = plan
        self.marker = marker
        self.terminal_source = terminal_source
        self.source_exit_code = source_exit_code
        self.corrected_exit_code = corrected_exit_code
        self.corrected_id = 100001

    def get_json(self, path: str, params: dict | None = None):
        if path == f"/api/tasks/{self.corrected_id}":
            return {
                "id": self.corrected_id,
                "task_id": self.corrected_id,
                "name": submission.TASK_NAME,
                "dedupe_key": self.plan["task_identity"]["dedupe_key"],
                "project": submission.PROJECT,
                "status": "completed",
                "exit_code": self.corrected_exit_code,
            }
        if path == f"/api/tasks/{submission.SOURCE_EXECUTION_TASK_ID}":
            return {
                "id": submission.SOURCE_EXECUTION_TASK_ID,
                "task_id": submission.SOURCE_EXECUTION_TASK_ID,
                "name": submission.SOURCE_TASK_NAME,
                "dedupe_key": submission.SOURCE_TASK_DEDUPE,
                "project": submission.PROJECT,
                "status": "completed" if self.terminal_source else "running",
                "exit_code": (
                    self.source_exit_code if self.terminal_source else None
                ),
            }
        raise AssertionError(path)

    def get_text(self, path: str, params: dict | None = None) -> str:
        if path.endswith(f"/{self.corrected_id}/stdout"):
            return (
                "CORRECTED_THERMAL_JSON "
                + json.dumps(self.marker, sort_keys=True)
            )
        if path.endswith(
            f"/{submission.SOURCE_EXECUTION_TASK_ID}/stdout"
        ):
            return 'RESULT_JSON {"resonant_frequency_Hz":16000,"T_max_core":999}'
        raise AssertionError(path)


def _maxwell_evidence(plan: dict) -> dict:
    return submission.create_maxwell_source_evidence(
        plan=plan,
        row={
            "resonant_frequency_Hz": 16000.0,
            "loss_total_W": 1000.0,
        },
        read_only_attestation={
            "saved_results_opened_read_only": True,
            "maxwell_analysis_calls": 0,
            "project_write_calls": 0,
            "source_checkpoint_unchanged": True,
            "source_results_manifest_sha256": _sha("c"),
        },
    )


def test_collector_emits_field_level_mixed_provenance(tmp_path: Path) -> None:
    plan, _path = _plan(tmp_path)
    marker = _marker(plan)
    scheduler = CollectorScheduler(plan, marker, terminal_source=False)
    result = submission.collect_mixed_provenance(
        plan=plan,
        corrected_task_id=scheduler.corrected_id,
        client=scheduler,
        maxwell_source_evidence=_maxwell_evidence(plan),
    )
    assert result["canonical"] is False
    assert result["truth_dataset_ingestion_allowed"] is False
    assert result["merged_row"]["resonant_frequency_Hz"] == 16000
    assert result["merged_row"]["T_max_core"] == 110
    assert result["field_sources"]["resonant_frequency_Hz"] == (
        "checkpoint_maxwell_saved_results_readonly"
    )
    assert result["field_sources"]["T_max_core"] == (
        "corrected_thermal_checkpoint_continuation"
    )


def test_collector_fails_on_two_nonthermal_sources(tmp_path: Path) -> None:
    plan, _path = _plan(tmp_path)
    scheduler = CollectorScheduler(plan, _marker(plan), terminal_source=True)
    with pytest.raises(submission.CorrectedThermalError, match="exactly one"):
        submission.collect_mixed_provenance(
            plan=plan,
            corrected_task_id=scheduler.corrected_id,
            client=scheduler,
            maxwell_source_evidence=_maxwell_evidence(plan),
        )


def test_terminal_exit_code_none_is_not_success(tmp_path: Path) -> None:
    plan, _path = _plan(tmp_path)
    source_none = CollectorScheduler(
        plan,
        _marker(plan),
        terminal_source=True,
        source_exit_code=None,
    )
    with pytest.raises(submission.CorrectedThermalError, match="without success"):
        submission._terminal_source(source_none)
    corrected_none = CollectorScheduler(
        plan,
        _marker(plan),
        terminal_source=False,
        corrected_exit_code=None,
    )
    with pytest.raises(submission.CorrectedThermalError, match="terminal-success"):
        submission.collect_mixed_provenance(
            plan=plan,
            corrected_task_id=corrected_none.corrected_id,
            client=corrected_none,
            maxwell_source_evidence=_maxwell_evidence(plan),
        )


def _optional_bundle() -> dict:
    return {
        "retained": False,
        "reason": submission.OPTIONAL_FIELD_BUNDLE_REASON,
        "observed_results_tree_logical_bytes": 123,
        "observed_results_tree_file_count": 4,
        "can_be_retained_only_by_separate_size_and_quota_preflight": True,
    }


def _retention_controls(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    destination = tmp_path / "retained" / "label"
    manifest_path = destination / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"manifest":"bytes"}\n', encoding="utf-8")
    manifest = {
        "control_evidence": dict(submission.RETENTION_CONTROL_EVIDENCE),
        "optional_field_bundle": _optional_bundle(),
    }
    receipt = {
        "schema": "mft-corrected-thermal-minimum-retention-receipt-v1",
        "destination": str(destination),
        "manifest_path": str(manifest_path),
        "manifest_sha256": submission.sha256_file(manifest_path),
        "prune_marker_path": str(
            destination / ".slurm-scheduler-preserve.json"
        ),
        "file_count": 5,
        "minimum_bundle_bytes": 100,
        "optional_field_bundle": _optional_bundle(),
        "published_atomically_with_package": True,
        "manifest_inventory_membership": False,
        "evidence_semantics": submission.RETENTION_EVIDENCE_SEMANTICS,
        "passed": True,
    }
    return manifest, receipt, destination, manifest_path


def test_retention_control_evidence_is_exact(tmp_path: Path) -> None:
    manifest, receipt, destination, manifest_path = _retention_controls(
        tmp_path
    )
    assert submission._validate_retention_control_records(
        manifest=manifest,
        receipt=receipt,
        destination=destination,
        manifest_path=manifest_path,
        minimum_bundle_bytes=100,
    ) == _optional_bundle()
    bad_manifest = dict(manifest)
    bad_manifest["control_evidence"] = {
        **submission.RETENTION_CONTROL_EVIDENCE,
        "extra": "not reviewed",
    }
    with pytest.raises(submission.CorrectedThermalError, match="control"):
        submission._validate_retention_control_records(
            manifest=bad_manifest,
            receipt=receipt,
            destination=destination,
            manifest_path=manifest_path,
            minimum_bundle_bytes=100,
        )
    for field, value in (
        ("published_atomically_with_package", False),
        ("manifest_inventory_membership", True),
        ("prune_marker_path", "/wrong/path"),
    ):
        bad_receipt = dict(receipt)
        bad_receipt[field] = value
        with pytest.raises(submission.CorrectedThermalError, match="receipt"):
            submission._validate_retention_control_records(
                manifest=manifest,
                receipt=bad_receipt,
                destination=destination,
                manifest_path=manifest_path,
                minimum_bundle_bytes=100,
            )
    bad_optional = dict(manifest)
    bad_optional["optional_field_bundle"] = {
        **_optional_bundle(),
        "retained": True,
    }
    with pytest.raises(submission.CorrectedThermalError, match="optional"):
        submission._validate_retention_control_records(
            manifest=bad_optional,
            receipt=receipt,
            destination=destination,
            manifest_path=manifest_path,
            minimum_bundle_bytes=100,
        )


def test_posix_retention_metadata_binds_owner_mode_and_file_link() -> None:
    good_file = SimpleNamespace(
        st_uid=submission.ACCOUNT_UID,
        st_mode=stat.S_IFREG | 0o400,
        st_nlink=1,
    )
    good_directory = SimpleNamespace(
        st_uid=submission.ACCOUNT_UID,
        st_mode=stat.S_IFDIR | 0o500,
        st_nlink=2,
    )
    submission._validate_retained_metadata(
        good_file, directory=False, label="file", posix=True
    )
    submission._validate_retained_metadata(
        good_directory, directory=True, label="directory", posix=True
    )
    for bad in (
        SimpleNamespace(
            st_uid=submission.ACCOUNT_UID + 1,
            st_mode=stat.S_IFREG | 0o400,
            st_nlink=1,
        ),
        SimpleNamespace(
            st_uid=submission.ACCOUNT_UID,
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
        ),
        SimpleNamespace(
            st_uid=submission.ACCOUNT_UID,
            st_mode=stat.S_IFREG | 0o400,
            st_nlink=2,
        ),
    ):
        with pytest.raises(submission.CorrectedThermalError, match="owner"):
            submission._validate_retained_metadata(
                bad, directory=False, label="file", posix=True
            )
