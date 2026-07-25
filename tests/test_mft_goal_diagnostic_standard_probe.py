import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY,
    GOAL_TEMPERATURE_TARGETS,
)
from regression_260707.verify import scheduler_client
from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_strict_al_ingest as strict_al
from tools import tier1_corrected_generation_adapter as adapter


def _production_test_helpers():
    path = Path(__file__).with_name("test_mft_goal_fea_handoff.py")
    spec = importlib.util.spec_from_file_location(
        "_mft_goal_fea_handoff_test_helpers", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Predictor:
    def configure_inference_threads(self, threads):
        return {"threads": threads, "families": ["fake"]}

    def predict_mu_sigma(self, frame):
        return (
            np.full(len(frame), 27.4),
            np.full(len(frame), 0.7),
        )

    def disagreement(self, frame):
        return np.full(len(frame), 1.3)


def _convert_to_near_feasible(result_path: Path):
    result = production._read_json(result_path)
    table_path = result_path.parent / (
        result["artifact_inventory"]["terminal_physical_candidates"]["path"]
    )
    manifest_path = result_path.parent / (
        result["artifact_inventory"][
            "terminal_physical_candidates_manifest"
        ]["path"]
    )
    table = pd.read_csv(table_path)
    physical_values = []
    normalized_values = []
    for physical_json, normalized_json in zip(
        table["physical_G_json"], table["normalized_G_json"]
    ):
        physical = json.loads(physical_json)
        normalized = json.loads(normalized_json)
        physical["Llt_robust_band"] = 0.25
        physical["Llt_ensemble_disagreement"] = 0.2
        normalized["Llt_robust_band"] = 0.5
        normalized["Llt_ensemble_disagreement"] = 0.4
        physical_values.append(
            json.dumps(physical, sort_keys=True, separators=(",", ":"))
        )
        normalized_values.append(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        )
    table["physical_G_json"] = physical_values
    table["normalized_G_json"] = normalized_values
    table["physical_G:Llt_robust_band"] = 0.25
    table["physical_G:Llt_ensemble_disagreement"] = 0.2
    table["normalized_G:Llt_robust_band"] = 0.5
    table["normalized_G:Llt_ensemble_disagreement"] = 0.4
    table["physical_constraint_feasible"] = False
    table["physical_feasible"] = False
    table.to_csv(table_path, index=False)

    manifest = production._read_json(manifest_path)
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("payload_sha256")
    unsigned_manifest["csv"] = {
        "path": table_path.name,
        "sha256": adapter.sha256_file(table_path),
        "size_bytes": table_path.stat().st_size,
    }
    manifest = launch._seal(unsigned_manifest)
    launch._atomic_json(manifest_path, manifest)
    inventory = {
        "terminal_physical_candidates": {
            "path": table_path.name,
            "sha256": adapter.sha256_file(table_path),
            "size_bytes": table_path.stat().st_size,
        },
        "terminal_physical_candidates_manifest": {
            "path": manifest_path.name,
            "sha256": adapter.sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    unsigned_result = dict(result)
    unsigned_result.pop("payload_sha256")
    unsigned_result["physical_feasible_count"] = 0
    unsigned_result["feasible_pareto_count"] = 0
    unsigned_result["terminal_physical_candidates_manifest"] = manifest
    unsigned_result["artifact_inventory"] = inventory
    unsigned_result["artifact_inventory_sha256"] = (
        production.canonical_sha256(inventory)
    )
    launch._atomic_json(result_path, launch._seal(unsigned_result))


def _fixture(tmp_path: Path, monkeypatch):
    claim_root = (tmp_path / "operational-pressure-claims").resolve()
    monkeypatch.setattr(
        probe, "OPERATIONAL_PRESSURE_CLAIM_ROOT", claim_root
    )
    atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=(
            probe.OPERATIONAL_PRESSURE_CLAIM_AUTHORITY_SHA256
        ),
        root_id="1" * 32,
        now="2026-07-25T00:00:00Z",
    )
    helpers = _production_test_helpers()
    bundle, bundle_path, tasks, _task_paths = helpers._bundle(tmp_path)
    result_paths = []
    geometry_by_turns = {}
    for turns in range(5, 9):
        task = next(
            item for item in tasks if item["fixed_primary_turns"] == turns
        )
        result_path, geometry_sha = helpers._write_seed_result(
            tmp_path / "results" / f"n1-{turns}",
            task,
            helpers._decoded_params(turns),
        )
        _convert_to_near_feasible(result_path)
        result_paths.append(result_path)
        geometry_by_turns[turns] = geometry_sha
    generation_identity = {
        "path": str(tmp_path / "generation"),
        "training_run_id": "test-run",
        "dataset_sha256": "a" * 64,
        "evaluation_model_sha256": "b" * 64,
        "train_report": {
            "path": str(tmp_path / "train_report.json"),
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
        "Llt_phys_model": {
            "path": str(tmp_path / "models.pkl"),
            "sha256": "d" * 64,
            "size_bytes": 1,
        },
        "Llt_phys_meta": {
            "path": str(tmp_path / "meta.json"),
            "sha256": "e" * 64,
            "size_bytes": 1,
        },
    }
    monkeypatch.setattr(
        probe,
        "_generation_identity",
        lambda _path, _tasks: copy.deepcopy(generation_identity),
    )
    selection_path = probe.create_selection(
        bundle_manifest_path=bundle_path,
        generation_path=tmp_path / "generation",
        result_paths=result_paths,
        count=4,
        output=tmp_path / "selection",
        predictor=_Predictor(),
    )
    return {
        "bundle": bundle,
        "bundle_path": bundle_path,
        "result_paths": result_paths,
        "geometry_by_turns": geometry_by_turns,
        "selection_path": selection_path,
    }


def _make_plan(tmp_path: Path, fixture):
    return probe.create_plan(
        selection_manifest_path=fixture["selection_path"],
        candidate_physics_sha256=fixture["geometry_by_turns"][6],
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "plan",
        predictor=_Predictor(),
    )


class _FakeScheduler:
    RESULT_VALID = scheduler_client.RESULT_VALID

    def __init__(self):
        self.calls = []
        self.result = None

    def submit_verification(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        guard = kwargs.get("pre_submit_guard")
        if guard is not None:
            guard()
        return 71001

    def get_status(self, _task_id, **_kwargs):
        return "completed"

    def fetch_result(self, *_args, **_kwargs):
        return SimpleNamespace(state=self.RESULT_VALID, result=self.result)

    @staticmethod
    def result_matches_params(result, params, required_keys=None):
        return scheduler_client.result_matches_params(
            result, params, required_keys=required_keys
        )


def _result(plan_path: Path, submission):
    plan, params, selected = probe._load_plan(plan_path)
    profile = production._read_json(
        plan_path.parent / plan["profile"]["path"]
    )
    result = {
        **selected["row_contract"]["decoded_params"],
        **production._effective_params(params, profile),
        "git_hash": plan["solver_revision"],
        "git_dirty": 0,
        "pyaedt_library_git_hash": plan["library_revision"],
        "pyaedt_library_git_dirty": 0,
        "project_name": "diagnostic-symmetric-project",
        "Llt": 13.75,
        "B_max_core": 1.0,
        "f_res_min_tx_rx_only_Hz": 15000.0,
        "thermal_pad_conductivity_W_mK": 0.2,
        "P_winding_total": 3.0,
        "P_Tx_main_group": 1.0,
        "P_Rx_main_group": 1.0,
        "P_Rx_side_total": 1.0,
        "P_core_total": 1.0,
        "P_core_plate_total": 1.0,
        "P_wcp_total": 1.0,
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": submission["core_policy"][
            "contract"
        ],
        "solver_core_opt_in": 1,
        "solver_core_backend": "standalone",
        "solver_num_cores_requested": 8,
        "solver_num_cores_effective": 8,
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": 8,
        "solver_core_slurm_cpus_per_task_readback": "8",
        "solver_core_scheduler_task_id_readback": str(submission["task_id"]),
        "solver_core_slurm_job_id_readback": "81234",
        "solver_core_auth_sha256": submission["core_policy"][
            "auth_sha256"
        ],
        "solver_matrix_hpc_num_cores_readback": 8,
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "9" * 64,
        "solver_core_license_contract": "",
        "solver_core_license_snapshot_sha256": "",
        **{
            target: 99.0 if "core" not in target else 119.0
            for target in GOAL_TEMPERATURE_TARGETS
        },
    }
    return result


def _scheduler_cutover(tmp_path: Path, monkeypatch):
    deployed = tmp_path / "scheduler-deployed"
    deployed.mkdir()
    release = tmp_path / "scheduler-release-manifest.json"
    release.write_bytes(b"reviewed-release")
    launcher = tmp_path / "scheduler-live-launcher.ps1"
    launcher.write_bytes(b"reviewed-launcher")
    release_sha = adapter.sha256_file(release)
    launcher_sha = adapter.sha256_file(launcher)
    monkeypatch.setattr(
        probe, "SCHEDULER_RELEASE_MANIFEST_SHA256", release_sha
    )
    monkeypatch.setattr(
        probe, "SCHEDULER_LIVE_LAUNCHER_SHA256", launcher_sha
    )
    receipt = production._seal(
        {
            "schema_version": probe.SCHEDULER_CUTOVER_SCHEMA,
            "scheduler_url": probe.DIAGNOSTIC_SCHEDULER_URL,
            "candidate_revision": probe.SCHEDULER_MARKER_AWARE_REVISION,
            "candidate_tree": probe.SCHEDULER_MARKER_AWARE_TREE,
            "release_manifest": {
                "path": str(release.resolve()),
                "sha256": release_sha,
            },
            "deployed_path": str(deployed.resolve()),
            "live_launcher_path": str(launcher.resolve()),
            "live_launcher_post_cutover_sha256": launcher_sha,
            "cutover_at": "2026-07-25T00:00:00+00:00",
            "post_health": {
                "status": "ok",
                "captured_at": "2026-07-25T00:00:01+00:00",
            },
            "marker": {
                "filename": scheduler_client.SCHEDULER_PRESERVE_MARKER,
                "schema": scheduler_client.SCHEDULER_PRESERVE_SCHEMA,
                "semantics_verified": True,
            },
            "preflight": {
                "goal_active_count": 0,
                "all_nonterminal_count": 0,
            },
            "rollback_launcher_sha256": "4" * 64,
            "created_at": "2026-07-25T00:00:02+00:00",
        }
    )
    path = tmp_path / "scheduler-cutover.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _strict_scheduler_cutover(tmp_path: Path, monkeypatch):
    backup = tmp_path / "strict-scheduler-backup.db"
    backup.write_bytes(b"strict-db-backup")
    pre = tmp_path / "strict-pre.json"
    immediate = tmp_path / "strict-immediate.json"
    final = tmp_path / "strict-final.json"
    pre.write_text('{"active_count":21}', encoding="utf-8")
    immediate.write_text('{"active_count":21}', encoding="utf-8")
    final.write_text('{"active_count":21}', encoding="utf-8")
    launcher = tmp_path / "strict-live-launcher.cmd"
    launcher.write_bytes(b"strict-reviewed-launcher")
    rollback = tmp_path / "strict-rollback-launcher.cmd"
    rollback.write_bytes(b"legacy-reviewed-launcher")
    launcher_sha = adapter.sha256_file(launcher)
    monkeypatch.setattr(
        probe, "SCHEDULER_STRICT_NODE_LIVE_LAUNCHER", launcher
    )
    monkeypatch.setattr(
        probe, "SCHEDULER_STRICT_NODE_LAUNCHER_SHA256", launcher_sha
    )
    rollback_sha = adapter.sha256_file(rollback)
    monkeypatch.setattr(
        probe,
        "SCHEDULER_STRICT_NODE_ROLLBACK_LAUNCHER_SHA256",
        rollback_sha,
    )
    receipt = {
        "schema_version": probe.SCHEDULER_STRICT_NODE_CUTOVER_SCHEMA,
        "cutover_at": "2026-07-25T11:06:05+09:00",
        "from_commit": probe.SCHEDULER_STRICT_NODE_FROM_REVISION,
        "to_commit": probe.SCHEDULER_STRICT_NODE_REVISION,
        "tree": probe.SCHEDULER_STRICT_NODE_TREE,
        "archive_sha256": "a" * 64,
        "launcher_sha256": launcher_sha,
        "cutover_guard_sha256": "b" * 64,
        "database_migration": "none",
        "configuration_change": "none",
        "database_backup": str(backup.resolve()),
        "database_backup_sha256": adapter.sha256_file(backup),
        "pre_snapshot": str(pre.resolve()),
        "pre_snapshot_sha256": adapter.sha256_file(pre),
        "immediate_snapshot": str(immediate.resolve()),
        "final_snapshot": str(final.resolve()),
        "dynamic_campaign_selection": (
            "every existing task id in inclusive range 96208..96280"
        ),
        "campaign_tasks_preserved": 73,
        "active_tasks_pre": 26,
        "active_tasks_final": 26,
        "allowed_transitions": (
            "running->completed/0|failed/124; "
            "attaching->running|terminal; queued->attaching|running"
        ),
        "protected_cancelled_tasks": [96260, 96276, 96277],
        "allocation_14616_immediate_requested_owned": "64/64",
        "allocation_14619_immediate_requested_owned": "64/64",
        "extra_attach_to_14616_or_14619": False,
        "strict_same_node_cpu_gate": "pass",
        "final_fea_storage_admission_gate": "pass",
        "pressure_episode_preservation": "pass",
        "n114_pressure_episode_preserved": True,
        "database_quick_check": "ok",
        "scheduler_ok": True,
        "scheduler_thread_alive": True,
        "rollback_launcher": str(rollback.resolve()),
        "rollback_launcher_sha256": rollback_sha,
    }
    path = tmp_path / "strict-cutover.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(
        probe,
        "SCHEDULER_STRICT_NODE_CUTOVER_SHA256",
        adapter.sha256_file(path),
    )
    return path


def _live_scheduler_reader(**kwargs):
    if kwargs["endpoint"] == "/api/health":
        return {
            "ok": True,
            "scheduler_ok": True,
            "scheduler_thread_alive": True,
            "scheduler_stalled": False,
            "consecutive_tick_failures": 0,
        }
    if kwargs["endpoint"] == "/api/licenses":
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "server": "1055@test-license",
            "server_up": True,
            "error": "",
            "admission": {
                "enabled": True,
                "snapshot_valid": True,
                "snapshot_age_seconds": 0.0,
                "snapshot_max_age_seconds": 120.0,
                "blocked_reason": "",
                "persistent_cost_by_project": {
                    scheduler_client.MFT_PROJECT: {
                        "electronics_desktop": 1
                    }
                },
                "features": {
                    "electronics_desktop": {"admit_headroom": 4}
                },
            },
        }
    raise AssertionError(kwargs)


def _remote_evidence(submission, result):
    retained = submission["retained_aedt_bundle"]
    artifact = b"diagnostic-aedt"
    marker_payload = {
        **retained["marker_contract"],
        "created_at": "2026-07-25T00:00:00Z",
    }
    marker = production._json_bytes(marker_payload)
    files = [
        {
            "path": "icepak_thermal.results/state.bin",
            "sha256": hashlib.sha256(b"state").hexdigest(),
            "size_bytes": 5,
        }
    ]
    tree_sha = hashlib.sha256(
        json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": probe.RESULTS_MANIFEST_SCHEMA,
        "source_project_name": result["project_name"],
        "source_results_directory_name": (
            f"{result['project_name']}.aedtresults"
        ),
        "retained_results_directory_name": "symmetric.aedtresults",
        "file_count": 1,
        "size_bytes": 5,
        "tree_sha256": tree_sha,
        "files": files,
    }
    manifest_bytes = production._json_bytes(manifest)
    receipt = {
        "schema_version": (
            scheduler_client.RETAINED_AEDT_BUNDLE_RECEIPT_SCHEMA
        ),
        "stage": "standard",
        "dedupe_key": retained["dedupe_key"],
        "parameter_digest": retained["parameter_digest"],
        "solver_revision": submission["solver_revision"],
        "library_revision": submission["library_revision"],
        "profile_sha256": submission["profile_sha256"],
        "artifact_path": retained["artifact_path"],
        "marker_path": retained["marker_path"],
        "retention_required": True,
        "prune_protection_required": True,
        "scheduler_cleanup_exclusion_required": True,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_size_bytes": len(artifact),
        "marker_sha256": hashlib.sha256(marker).hexdigest(),
        "marker_contract_sha256": retained["marker_contract_sha256"],
        "transport_schema_version": retained["transport"][
            "schema_version"
        ],
        "transport_encoding": retained["transport"]["encoding"],
        "transport_chunk_directory": retained["transport"][
            "chunk_directory"
        ],
        "transport_raw_chunk_bytes": retained["transport"][
            "raw_chunk_bytes"
        ],
        "transport_max_encoded_chunk_bytes": retained["transport"][
            "max_encoded_chunk_bytes"
        ],
        "transport_chunk_count": 1,
        "source_project_filename": f"{result['project_name']}.aedt",
        "source_project_name": result["project_name"],
        "results_path": retained["results_path"],
        "results_manifest_path": retained["results_manifest_path"],
        "results_manifest_schema_version": probe.RESULTS_MANIFEST_SCHEMA,
        "results_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "results_tree_sha256": tree_sha,
        "results_file_count": 1,
        "results_size_bytes": 5,
        "source_results_directory_name": (
            f"{result['project_name']}.aedtresults"
        ),
    }
    receipt_bytes = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()

    def metadata_reader(**kwargs):
        if kwargs["relative_path"] == retained["receipt_path"]:
            return receipt_bytes
        if kwargs["relative_path"] == retained["marker_path"]:
            return marker
        raise AssertionError(kwargs)

    def manifest_reader(**kwargs):
        assert kwargs["relative_path"] == retained["results_manifest_path"]
        return manifest_bytes

    return metadata_reader, manifest_reader


def _task_snapshot(submission):
    return {
        "task_id": submission["task_id"],
        "name": submission["task_name"],
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "failure_message": "",
        "slurm_job_id": "81234",
        "allocation_id": 9001,
        "account_name": "test-account",
        "actual_node_name": "n101",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": submission["resources"]["timeout_seconds"],
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": submission["dedupe_key"],
        "remote_cwd": "/gpfs/test",
        "remote_dir": "runs/task-71001",
        "created_at": "2026-07-24 18:23:00",
        "finished_at": "2026-07-25 00:00:00",
    }


def _timeout_task_snapshot(submission, **overrides):
    snapshot = {
        **_task_snapshot(submission),
        "status": "failed",
        "state": "failed",
        "exit_code": 124,
        "failure_message": "task timed out after 14400s",
        "timeout_seconds": 14400,
        "started_at": "2026-07-25 03:20:14",
        "finished_at": "2026-07-25 07:20:45",
    }
    snapshot.update(overrides)
    return snapshot


def _operational_pressure_task_snapshot(submission, **overrides):
    snapshot = {
        **_task_snapshot(submission),
        "status": "failed",
        "state": "failed",
        "exit_code": None,
        "failure_message": probe.OPERATIONAL_PRESSURE_FAILURE_MESSAGE,
        "timeout_seconds": 14400,
        "allocation_id": 14618,
        "account_name": "harry261",
        "actual_node_name": "n109",
        "slurm_job_id": "826817",
        "created_at": "2026-07-24 18:23:00",
        "started_at": "2026-07-25 01:37:27",
        "finished_at": "2026-07-25 04:29:52",
    }
    snapshot.update(overrides)
    return snapshot


def _operational_pressure_events(submission):
    name = submission["task_name"]
    task_id = str(submission["task_id"])
    return list(reversed([
        {
            "id": 127190,
            "created_at": "2026-07-24 22:00:35",
            "kind": "task_requeued",
            "entity_type": "task",
            "entity_id": task_id,
            "account_name": "r1jae262",
            "message": (
                f"task {name} requeued after memory-pressure kill "
                "(attempt 1/3)"
            ),
        },
        {
            "id": 127238,
            "created_at": "2026-07-25 01:36:48",
            "kind": "task_requeued",
            "entity_type": "task",
            "entity_id": task_id,
            "account_name": "jji0930",
            "message": (
                f"task {name} requeued after memory-pressure kill "
                "(attempt 2/3)"
            ),
        },
        {
            "id": 127279,
            "created_at": "2026-07-25 04:29:53",
            "kind": "task_cleanup",
            "entity_type": "task",
            "entity_id": task_id,
            "account_name": "harry261",
            "message": (
                "cleaned mft_campaign-test in slurm_scheduler/runs after "
                f"task {name} ended (failed)"
            ),
        },
    ]))


def _same_allocation_anchor_snapshot(**overrides):
    snapshot = {
        "task_id": 72000,
        "name": "mft-goal-diag-standard-timeout-r1-anchor",
        "status": "running",
        "state": "running",
        "allocation_id": 9002,
        "slurm_job_id": "81300",
        "account_name": "anchor-account",
        "actual_node_name": "n110",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 28800,
        "dedupe_key": "anchor-dedupe",
        "started_at": "2026-07-25 00:05:31",
        "finished_at": None,
    }
    snapshot.update(overrides)
    return snapshot


def _same_allocation_submitted_snapshot(submission, **overrides):
    snapshot = {
        "task_id": submission["task_id"],
        "name": submission["task_name"],
        "status": "running",
        "state": "running",
        "allocation_id": 9002,
        "slurm_job_id": "81300",
        "account_name": "anchor-account",
        "actual_node_name": "n110",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 28800,
        "dedupe_key": submission["dedupe_key"],
        "same_node_as_task_id": 72000,
        "requested_allocation_id": 0,
        "finished_at": None,
    }
    snapshot.update(overrides)
    return snapshot


def _strict_submitted_snapshot(submission, **overrides):
    snapshot = {
        **_same_allocation_submitted_snapshot(submission),
        "node_name": "n110",
        "requested_node_name": "n110",
        "node_name_policy": "strict",
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
        "assigned_allocation": 9002,
        "allocation_node_name": "n110",
        "requested_account_name": "anchor-account",
        "started_at": "2026-07-25 00:06:00",
    }
    snapshot.update(overrides)
    return snapshot


def test_operational_pressure_event_window_proves_complete_lifetime():
    submission = {
        "task_id": 71001,
        "task_name": "mft-goal-diag-standard-pressure",
    }
    lifecycle = _operational_pressure_events(submission)
    generic = [
        {
            "id": 127189 - offset,
            "created_at": "2026-07-24 10:11:14",
            "kind": "allocation_observed",
            "entity_type": "allocation",
            "entity_id": str(offset + 1),
            "account_name": "",
            "message": "allocation observed",
        }
        for offset in range(997)
    ]
    full_window = [*lifecycle, *generic]
    evidence = probe._operational_pressure_events(
        full_window,
        task_id=71001,
        task_name=submission["task_name"],
        task_created_at="2026-07-24 18:23:00",
        task_account_name="harry261",
    )
    window = evidence["event_window"]
    assert window["response_count"] == 1000
    assert window["coverage_method"] == (
        "oldest_event_not_after_task_created_at"
    )
    assert len(window["full_window_sha256"]) == 64
    with pytest.raises(
        production.HandoffContractError,
        match="does not cover the task lifetime",
    ):
        probe._operational_pressure_events(
            [
                *lifecycle,
                *[
                    {
                        **event,
                        "created_at": "2026-07-24 19:00:00",
                    }
                    for event in generic
                ],
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )
    with pytest.raises(
        production.HandoffContractError,
        match="window ordering drifted",
    ):
        probe._operational_pressure_events(
            [
                full_window[0],
                {**full_window[1], "id": full_window[0]["id"]},
                *full_window[2:],
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )
    with pytest.raises(
        production.HandoffContractError,
        match="chronology/account binding drifted",
    ):
        probe._operational_pressure_events(
            [
                {
                    **event,
                    "created_at": (
                        "2026-07-24 18:22:59"
                        if "(attempt 1/3)" in event["message"]
                        else event["created_at"]
                    ),
                }
                for event in lifecycle
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )
    with pytest.raises(
        production.HandoffContractError,
        match="timestamp",
    ):
        probe._operational_pressure_events(
            [
                {
                    **event,
                    "created_at": (
                        "not-a-timestamp"
                        if "(attempt 1/3)" in event["message"]
                        else event["created_at"]
                    ),
                }
                for event in lifecycle
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )
    with pytest.raises(
        production.HandoffContractError,
        match="attempt evidence 1/3",
    ):
        probe._operational_pressure_events(
            [
                {
                    **event,
                    "entity_id": (
                        "99999"
                        if "(attempt 1/3)" in event["message"]
                        else event["entity_id"]
                    ),
                }
                for event in lifecycle
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )
    with pytest.raises(
        production.HandoffContractError,
        match="chronology/account binding drifted",
    ):
        probe._operational_pressure_events(
            [
                {
                    **event,
                    "account_name": (
                        ""
                        if "(attempt 1/3)" in event["message"]
                        else event["account_name"]
                    ),
                }
                for event in lifecycle
            ],
            task_id=71001,
            task_name=submission["task_name"],
            task_created_at="2026-07-24 18:23:00",
            task_account_name="harry261",
        )


def test_same_allocation_submitted_node_canonicalization_is_state_safe():
    submission = {
        "task_id": 71002,
        "task_name": "strict-retry",
        "dedupe_key": "strict-dedupe",
    }
    options = {
        "task_id": 71002,
        "task_name": "strict-retry",
        "dedupe_key": "strict-dedupe",
        "anchor_task_id": 72000,
        "allocation_id": 9002,
        "slurm_job_id": "81300",
        "account_name": "anchor-account",
        "node_name": "n110",
    }
    queued = _same_allocation_submitted_snapshot(
        submission,
        status="queued",
        state="queued",
        allocation_id=None,
        slurm_job_id="",
        actual_node_name="",
    )
    queued.pop("allocation_node_name", None)
    canonical_queued = probe._same_allocation_submitted_task_evidence(
        queued, **options
    )
    assert canonical_queued["actual_node_name"] == ""
    assert (
        probe._same_allocation_submitted_task_evidence(
            canonical_queued, **options
        )
        == canonical_queued
    )
    assigned = _same_allocation_submitted_snapshot(
        submission,
        actual_node_name="",
        allocation_node_name="n110",
    )
    canonical_assigned = probe._same_allocation_submitted_task_evidence(
        assigned, **options
    )
    assert canonical_assigned["actual_node_name"] == "n110"
    with pytest.raises(
        production.HandoffContractError,
        match="same-allocation task binding drifted",
    ):
        probe._same_allocation_submitted_task_evidence(
            {
                **assigned,
                "allocation_node_name": "n109",
            },
            **options,
        )


def test_selection_is_deterministic_diverse_and_mean_band_bound(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    selection, table = probe._load_selection(fixture["selection_path"])
    assert selection["eligible_near_feasible_count"] == 4
    assert sorted(table["diagnostic_primary_turns"].tolist()) == [5, 6, 7, 8]
    assert table["diagnostic_llt_mean_uH"].tolist() == [27.4] * 4
    assert table["diagnostic_only"].all()
    assert not table["production_eligible"].all()
    assert not table["full_submission_allowed"].all()
    assert selection["automatic_promotion"] is False


def test_final_aggregate_plan_reauthentication_allows_search_only_bundle(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    selection, table = probe._load_selection(fixture["selection_path"])
    aggregate_path = tmp_path / "aggregate_manifest.json"
    aggregate = launch._seal(
        {"schema_version": launch.GLOBAL_PARETO_SCHEMA}
    )
    launch._atomic_json(aggregate_path, aggregate)
    result_paths_by_sha256 = {
        str(record["result"]["sha256"]): Path(record["result"]["path"])
        for record in selection["source_results"]
    }
    aggregate_authority = {
        "seed_count": len(result_paths_by_sha256),
        "minimum_seed_count": len(result_paths_by_sha256),
        "result_paths_by_sha256": result_paths_by_sha256,
    }
    selection["source_mode"] = "final_authenticated_aggregate"
    selection["aggregate_source"] = {
        "manifest": production._file_record(aggregate_path),
        "payload_sha256": aggregate["payload_sha256"],
        "seed_count": aggregate_authority["seed_count"],
        "minimum_seed_count": aggregate_authority["minimum_seed_count"],
        "all_bundle_seed_results_reauthenticated": True,
        "global_nds_recomputed": True,
        "input_result_sha256": sorted(result_paths_by_sha256),
    }
    monkeypatch.setattr(
        probe,
        "_load_selection",
        lambda _path: (selection, table),
    )
    observed = {}

    def authenticate_aggregate(**kwargs):
        observed.update(kwargs)
        return aggregate_authority

    monkeypatch.setattr(
        production,
        "_authenticate_aggregate_authority",
        authenticate_aggregate,
    )
    authenticated = probe.authenticate_candidate(
        selection_manifest_path=fixture["selection_path"],
        candidate_physics_sha256=fixture["geometry_by_turns"][6],
        predictor=_Predictor(),
    )
    assert observed["allow_search_only"] is True
    assert (
        authenticated["selection_source"]["source_mode"]
        == "final_authenticated_aggregate"
    )


def test_plan_has_only_standard_and_exact_fixed_physics(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    plan_path = _make_plan(tmp_path, fixture)
    plan, _params, selected = probe._load_plan(plan_path)
    profile = production._read_json(
        plan_path.parent / plan["profile"]["path"]
    )
    assert plan["available_submission_commands"] == ["submit-standard"]
    assert "stages" not in plan
    assert plan["stage"]["full_model"] == 0
    assert plan["stage"]["thermal_symmetry"] == "eighth"
    assert profile["param_overrides"]["fan_velocity"] == 1.5
    assert profile["param_overrides"]["fan_config"] == "dual"
    assert profile["param_overrides"]["core_plate_pad_t"] == 2.0
    assert profile["param_overrides"]["wcp_pad_t"] == 2.0
    assert profile["param_overrides"]["k_ins"] == 0.2
    fixed_expected = selected["row_contract"][
        "fixed_identity_attestation"
    ]["expected"]
    assert all(
        fixed_expected[name] == expected
        for name, expected in FIXED_COOLING_IDENTITY.items()
    )
    retained = plan["stage"]["retained_aedt_bundle"]
    assert (
        plan["stage"]["scheduler_url"]
        == "http://127.0.0.1:8002"
    )
    assert plan["stage"]["retention_run_root"][
        "scheduler_marker_semantics_required"
    ]
    assert retained["artifact_path"].endswith("/symmetric.aedt")
    assert retained["results_path"].endswith("/symmetric.aedtresults")
    assert retained["marker_path"].endswith(
        "/.slurm-scheduler-preserve.json"
    )
    with pytest.raises(SystemExit):
        probe._parser().parse_args(["submit-full"])
    with pytest.raises(
        production.HandoffContractError, match="payload seal"
    ):
        production._load_plan(plan_path)


def test_timeout_retry_is_distinct_fixed_physics_and_requires_failed_124(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_scheduler = _FakeScheduler()
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=original_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan, original_params, _selected = probe._load_plan(
        original_plan_path
    )
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )

    with pytest.raises(
        production.HandoffContractError,
        match="terminal failed/124",
    ):
        probe.create_timeout_retry_plan(
            original_plan_path=original_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / "forbidden-running-retry",
            task_reader=lambda **_kwargs: {
                **_timeout_task_snapshot(original_submission),
                "status": "running",
                "state": "running",
                "exit_code": None,
                "failure_message": "",
                "finished_at": None,
            },
        )
    assert not (tmp_path / "forbidden-running-retry").exists()

    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "retry-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan, retry_params, _retry_selected = probe._load_plan(
        retry_plan_path
    )
    retry_profile = production._read_json(
        retry_plan_path.parent / retry_plan["profile"]["path"]
    )
    original_profile = production._read_json(
        original_plan_path.parent / original_plan["profile"]["path"]
    )
    assert retry_plan["available_submission_commands"] == [
        "submit-timeout-retry"
    ]
    assert retry_plan["retry_of_timeout"]["retry_of_task_id"] == 71001
    assert retry_plan["retry_of_timeout"]["original_task_execution"][
        "exit_code"
    ] == 124
    assert retry_plan["stage"]["resources"] == {
        "cpus": 8,
        "timeout_seconds": 28800,
    }
    assert retry_profile["mem_mb"] == 32768
    assert retry_profile["cpus"] == 8
    assert retry_profile["timeout_seconds"] == 28800
    assert retry_profile["param_overrides"] == original_profile[
        "param_overrides"
    ]
    assert retry_profile["fixed_boundary_contract"] == original_profile[
        "fixed_boundary_contract"
    ]
    assert retry_params == original_params
    assert retry_plan["solver_revision"] == original_plan["solver_revision"]
    assert retry_plan["library_revision"] == original_plan[
        "library_revision"
    ]
    assert (
        retry_plan["stage"]["profile_sha256"]
        != original_plan["stage"]["profile_sha256"]
    )
    assert (
        retry_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        != original_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    )
    assert (
        retry_plan["stage"]["retained_aedt_bundle"]["relative_directory"]
        != original_plan["stage"]["retained_aedt_bundle"][
            "relative_directory"
        ]
    )


def test_timeout_retry_reauthenticates_failure_before_single_submission(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_scheduler = _FakeScheduler()
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=original_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "retry-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )

    blocked_scheduler = _FakeScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="terminal failed/124",
    ):
        probe.submit_timeout_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "forbidden-retry-submission.json",
            scheduler=blocked_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=lambda **_kwargs: _timeout_task_snapshot(
                original_submission,
                exit_code=1,
                failure_message="other failure",
            ),
        )
    assert blocked_scheduler.calls == []
    assert not (tmp_path / "forbidden-retry-submission.json").exists()

    class _RetryScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 71002

    retry_scheduler = _RetryScheduler()
    retry_submission_path = probe.submit_timeout_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "retry-submission.json",
        scheduler=retry_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan = probe._load_plan(retry_plan_path)[0]
    retry_submission = probe._load_submission(
        retry_submission_path, plan=retry_plan
    )
    assert retry_submission["task_id"] == 71002
    assert retry_submission["retry_of_timeout"] == retry_plan[
        "retry_of_timeout"
    ]
    assert len(retry_scheduler.calls) == 1
    submitted_profile = retry_scheduler.calls[0][0][3]
    assert submitted_profile["timeout_seconds"] == 28800
    assert submitted_profile["cpus"] == 8
    assert submitted_profile["mem_mb"] == 32768


def test_timeout_retry_can_seal_exact_same_allocation_binding(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_scheduler = _FakeScheduler()
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=original_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "retry-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan = probe._load_plan(retry_plan_path)[0]
    expected_submission = {
        "task_id": 71002,
        "task_name": retry_plan["stage"]["task_name"],
        "dedupe_key": retry_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
    }

    class _RetryScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 71002

    def task_reader(**kwargs):
        task_id = kwargs["task_id"]
        if task_id == original_submission["task_id"]:
            return _timeout_task_snapshot(original_submission)
        if task_id == 72000:
            return _same_allocation_anchor_snapshot()
        if task_id == 71002:
            return _same_allocation_submitted_snapshot(
                expected_submission
            )
        raise AssertionError(task_id)

    retry_scheduler = _RetryScheduler()
    retry_submission_path = probe.submit_timeout_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "retry-submission.json",
        scheduler=retry_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=task_reader,
        same_node_as_task_id=72000,
        expected_allocation_id=9002,
        expected_slurm_job_id="81300",
        expected_account_name="anchor-account",
        expected_node_name="n110",
    )
    retry_submission = probe._load_submission(
        retry_submission_path, plan=retry_plan
    )
    kwargs = retry_scheduler.calls[0][1]
    assert kwargs["same_node_as_task_id"] == 72000
    assert kwargs["account_name"] == "anchor-account"
    assert kwargs["node_name"] == "n110"
    placement = retry_submission["scheduler_placement_contract"]
    assert placement["schema_version"] == (
        probe.SAME_ALLOCATION_PLACEMENT_SCHEMA
    )
    assert placement["expected_allocation_id"] == 9002
    assert placement["expected_slurm_job_id"] == "81300"
    assert placement["fallback_allocation_allowed"] is False
    assert placement["requested_allocation_id_used"] is False
    assert placement["submitted_task_after_submission"][
        "same_node_as_task_id"
    ] == 72000

    terminal = {
        **_task_snapshot(retry_submission),
        "allocation_id": 9002,
        "slurm_job_id": "81300",
        "account_name": "anchor-account",
        "actual_node_name": "n110",
    }
    assert probe._task_execution_evidence(
        terminal, submission=retry_submission
    )["allocation_id"] == 9002
    with pytest.raises(
        production.HandoffContractError,
        match="escaped its sealed same-allocation",
    ):
        probe._task_execution_evidence(
            {**terminal, "allocation_id": 9003},
            submission=retry_submission,
        )

    unsigned = production._read_json(retry_submission_path)
    unsigned.pop("payload_sha256")
    unsigned["scheduler_placement_contract"][
        "expected_node_name"
    ] = "n109"
    drifted_path = tmp_path / "drifted-retry-submission.json"
    drifted_path.write_text(
        json.dumps(
            production._seal(unsigned),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        production.HandoffContractError,
        match="anchor drifted",
    ):
        probe._load_submission(drifted_path, plan=retry_plan)


def test_same_allocation_retry_fails_closed_on_anchor_or_task_drift(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_scheduler = _FakeScheduler()
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=original_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "retry-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan = probe._load_plan(retry_plan_path)[0]
    expected_submission = {
        "task_id": 71002,
        "task_name": retry_plan["stage"]["task_name"],
        "dedupe_key": retry_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
    }

    class _RetryScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 71002

    blocked_scheduler = _RetryScheduler()

    def failed_anchor_reader(**kwargs):
        if kwargs["task_id"] == original_submission["task_id"]:
            return _timeout_task_snapshot(original_submission)
        return _same_allocation_anchor_snapshot(
            status="failed",
            state="failed",
            finished_at="2026-07-25 01:00:00",
        )

    with pytest.raises(
        production.HandoffContractError,
        match="anchor drifted",
    ):
        probe.submit_timeout_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "failed-anchor-submission.json",
            scheduler=blocked_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=failed_anchor_reader,
            same_node_as_task_id=72000,
            expected_allocation_id=9002,
            expected_slurm_job_id="81300",
            expected_account_name="anchor-account",
            expected_node_name="n110",
        )
    assert blocked_scheduler.calls == []

    def unbound_existing_task_reader(**kwargs):
        task_id = kwargs["task_id"]
        if task_id == original_submission["task_id"]:
            return _timeout_task_snapshot(original_submission)
        if task_id == 72000:
            return _same_allocation_anchor_snapshot()
        if task_id == 71002:
            return _same_allocation_submitted_snapshot(
                expected_submission,
                same_node_as_task_id=0,
            )
        raise AssertionError(task_id)

    duplicate_scheduler = _RetryScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="task binding drifted",
    ):
        probe.submit_timeout_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "unbound-existing-submission.json",
            scheduler=duplicate_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=unbound_existing_task_reader,
            same_node_as_task_id=72000,
            expected_allocation_id=9002,
            expected_slurm_job_id="81300",
            expected_account_name="anchor-account",
            expected_node_name="n110",
        )
    assert len(duplicate_scheduler.calls) == 1
    assert not (tmp_path / "unbound-existing-submission.json").exists()


def test_timeout_retry_strict_r2_authenticates_post_get_and_terminal_binding(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    legacy_cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=legacy_cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=_FakeScheduler(),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    strict_cutover_path = _strict_scheduler_cutover(tmp_path, monkeypatch)
    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "strict-retry-plan",
        strict_node_name="n110",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan = probe._load_plan(retry_plan_path)[0]
    assert "timeout-strict-r2-n110-" in retry_plan["stage"]["task_name"]
    assert retry_plan["scheduler_strict_node_contract"][
        "fallback_allocation_allowed"
    ] is False
    assert retry_plan["scheduler_strict_node_contract"][
        "scheduler_revision"
    ] == probe.SCHEDULER_STRICT_NODE_REVISION
    assert retry_plan["scheduler_strict_node_contract"][
        "scheduler_cutover_receipt_schema"
    ] == "slurm-scheduler-cutover-receipt-v3"
    expected_submission = {
        "task_id": 71002,
        "task_name": retry_plan["stage"]["task_name"],
        "dedupe_key": retry_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
    }
    post_readback = _strict_submitted_snapshot(
        expected_submission,
        status="queued",
        state="queued",
        allocation_id=None,
        assigned_allocation=None,
        slurm_job_id="",
        actual_node_name="",
        allocation_node_name="",
        placement_contract_satisfied=False,
        started_at=None,
    )

    class _StrictScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            guard = kwargs.get("pre_submit_guard")
            if guard is not None:
                guard()
            return {
                "task_id": 71002,
                "submission_source": "post_created",
                "scheduler_mutation_performed": True,
                "api_pre_submission_readback": None,
                "api_post_submission_response": post_readback,
            }

    def task_reader(**kwargs):
        task_id = kwargs["task_id"]
        if task_id == original_submission["task_id"]:
            return _timeout_task_snapshot(original_submission)
        if task_id == 72000:
            return _same_allocation_anchor_snapshot()
        if task_id == 71002:
            return post_readback
        raise AssertionError(task_id)

    strict_scheduler = _StrictScheduler()
    retry_submission_path = probe.submit_timeout_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=strict_cutover_path,
        output=tmp_path / "strict-retry-submission.json",
        scheduler=strict_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=task_reader,
        same_node_as_task_id=72000,
        expected_allocation_id=9002,
        expected_slurm_job_id="81300",
        expected_account_name="anchor-account",
        expected_node_name="n110",
    )
    retry_submission = probe._load_submission(
        retry_submission_path, plan=retry_plan
    )
    kwargs = strict_scheduler.calls[0][1]
    assert kwargs["node_name"] == "n110"
    assert kwargs["node_name_policy"] == "strict"
    assert kwargs["return_submission_evidence"] is True
    assert kwargs["same_node_as_task_id"] == 72000
    strict_receipt = retry_submission["scheduler_strict_node_contract"]
    assert strict_receipt["submission_source"] == "post_created"
    assert strict_receipt["api_post_submission_response"][
        "placement_contract_satisfied"
    ] is False
    assert strict_receipt["api_durable_get_readback"][
        "allocation_node_name"
    ] == ""
    assert retry_submission["scheduler_placement_contract"][
        "submitted_task_after_submission"
    ]["actual_node_name"] == ""
    terminal = _strict_submitted_snapshot(
        expected_submission,
        status="completed",
        state="succeeded",
        exit_code=0,
        failure_message="",
        remote_cwd="/gpfs/test",
        remote_dir="runs/task-71002",
        finished_at="2026-07-25 08:06:00",
    )
    assert probe._task_execution_evidence(
        terminal, submission=retry_submission
    )["actual_node_name"] == "n110"
    with pytest.raises(
        production.HandoffContractError,
        match=(
            "strict-node Scheduler identity drifted|fell back|"
            "escaped its sealed same-allocation"
        ),
    ):
        probe._task_execution_evidence(
            {
                **terminal,
                "allocation_node_name": "n109",
                "actual_node_name": "n109",
            },
            submission=retry_submission,
        )


def test_operational_pressure_retry_is_exact_once_strict_and_revalidates(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    legacy_cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=legacy_cutover_path,
        output=tmp_path / "pressure-original-submission.json",
        scheduler=_FakeScheduler(),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    def task_reader(**_kwargs):
        return _operational_pressure_task_snapshot(
            original_submission
        )

    def event_reader(**_kwargs):
        return _operational_pressure_events(original_submission)
    strict_cutover_path = _strict_scheduler_cutover(
        tmp_path, monkeypatch
    )

    with pytest.raises(
        production.HandoffContractError,
        match="2/3",
    ):
        probe.create_operational_pressure_retry_plan(
            original_plan_path=original_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / "forbidden-pressure-attempt2",
            strict_node_name="n116",
            task_reader=task_reader,
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(original_submission)[:1]
                + _operational_pressure_events(original_submission)[2:]
            ),
        )
    assert not (tmp_path / "forbidden-pressure-attempt2").exists()

    retry_plan_path = probe.create_operational_pressure_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "pressure-retry-plan",
        strict_node_name="n116",
        task_reader=task_reader,
        event_reader=event_reader,
    )
    retry_plan, retry_params, _selected = probe._load_plan(
        retry_plan_path
    )
    alternate_plans = []
    alternate_plan_paths = []
    for suffix, node_name in (
        ("unplaced", ""),
        ("n108", "n108"),
        ("n110", "n110"),
    ):
        alternate_path = probe.create_operational_pressure_retry_plan(
            original_plan_path=original_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / f"pressure-retry-plan-{suffix}",
            strict_node_name=node_name,
            task_reader=task_reader,
            event_reader=event_reader,
        )
        alternate_plan_paths.append(alternate_path)
        alternate_plans.append(probe._load_plan(alternate_path)[0])
    for alternate in alternate_plans:
        assert alternate["stage"]["task_name"] == retry_plan["stage"][
            "task_name"
        ]
        assert alternate["stage"]["workdir"] == retry_plan["stage"][
            "workdir"
        ]
        assert alternate["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ] == retry_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    retry_profile = production._read_json(
        retry_plan_path.parent / retry_plan["profile"]["path"]
    )
    assert retry_plan["available_submission_commands"] == [
        "submit-operational-pressure-retry"
    ]
    assert (
        retry_plan["retry_of_operational_pressure"][
            "retry_of_task_id"
        ]
        == original_submission["task_id"]
    )
    assert (
        retry_plan["retry_of_operational_pressure"][
            "original_task_execution"
        ]["attempt_evidence"]["attempt_count"]
        == 3
    )
    assert len(
        retry_plan["retry_of_operational_pressure"][
            "original_task_execution"
        ]["attempt_evidence"]["task_api_sha256"]
    ) == 64
    assert retry_plan["stage"]["resources"] == {
        "cpus": 8,
        "timeout_seconds": 14400,
    }
    assert (
        f"pressure-r1-t{original_submission['task_id']}-"
        in retry_plan["stage"]["task_name"]
    )
    assert (
        retry_plan["scheduler_strict_node_contract"][
            "task_identity_generation"
        ]
        == probe.OPERATIONAL_PRESSURE_STRICT_TASK_IDENTITY_GENERATION
    )
    assert retry_plan["stage"]["retained_aedt_bundle"]["dedupe_key"] != (
        original_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    )
    assert retry_profile["param_overrides"]["fan_velocity"] == 1.5
    assert retry_profile["param_overrides"]["core_plate_pad_t"] == 2.0
    assert retry_profile["param_overrides"]["wcp_pad_t"] == 2.0
    assert retry_profile["param_overrides"]["k_ins"] == 0.2
    assert retry_params["fan_velocity"] == 1.5
    with pytest.raises(
        production.HandoffContractError,
        match="lifecycle event set drifted",
    ):
        probe.create_operational_pressure_retry_plan(
            original_plan_path=original_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / "forbidden-pressure-extra-event",
            task_reader=task_reader,
            event_reader=lambda **_kwargs: [
                {
                    **_operational_pressure_events(
                        original_submission
                    )[0],
                    "id": 127300,
                    "created_at": "2026-07-25 04:29:54",
                    "message": (
                        f"task {original_submission['task_name']} "
                        "requeued after memory-pressure kill "
                        "(attempt 3/3)"
                    ),
                },
                *_operational_pressure_events(original_submission),
            ],
        )
    with pytest.raises(
        production.HandoffContractError, match="semantics are mixed"
    ):
        probe._plan_retry_kind(
            {
                "retry_of_timeout": {},
                "retry_of_operational_pressure": {},
            }
        )
    with pytest.raises(
        production.HandoffContractError,
        match="bounded direct-or-timeout chain",
    ):
        probe.create_operational_pressure_retry_plan(
            original_plan_path=retry_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / "forbidden-pressure-chain",
            task_reader=task_reader,
            event_reader=event_reader,
        )

    expected_submission = {
        "task_id": 71003,
        "task_name": retry_plan["stage"]["task_name"],
        "dedupe_key": retry_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
    }
    post_readback = {
        "task_id": 71003,
        "name": expected_submission["task_name"],
        "status": "queued",
        "state": "queued",
        "dedupe_key": expected_submission["dedupe_key"],
        "project": scheduler_client.MFT_PROJECT,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 14400,
        "node_name": "n116",
        "requested_node_name": "n116",
        "node_name_policy": "strict",
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": False,
        "allocation_id": None,
        "assigned_allocation": None,
        "allocation_node_name": "",
        "actual_node_name": "",
        "slurm_job_id": "",
        "account_name": None,
        "requested_account_name": "anchor-account",
        "same_node_as_task_id": 72000,
        "requested_allocation_id": 0,
        "started_at": None,
        "finished_at": None,
    }

    class _PressureScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            guard = kwargs.get("pre_submit_guard")
            if guard is not None:
                guard()
            self.calls.append((args, kwargs))
            return {
                "task_id": 71003,
                "submission_source": "post_created",
                "scheduler_mutation_performed": True,
                "api_pre_submission_readback": None,
                "api_post_submission_response": post_readback,
            }

    def retry_task_reader(**kwargs):
        if kwargs["task_id"] == original_submission["task_id"]:
            return _operational_pressure_task_snapshot(
                original_submission
            )
        if kwargs["task_id"] == 72000:
            return _same_allocation_anchor_snapshot(
                actual_node_name="n116",
                timeout_seconds=14400,
            )
        if kwargs["task_id"] == 71003:
            return post_readback
        raise AssertionError(kwargs["task_id"])

    collision_scheduler = _PressureScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="more than one operational-pressure retry sibling",
    ):
        probe.submit_operational_pressure_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=strict_cutover_path,
            output=tmp_path / "forbidden-pressure-siblings.json",
            scheduler=collision_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=retry_task_reader,
            event_reader=event_reader,
            task_list_reader=lambda **_kwargs: [
                post_readback,
                {**post_readback, "task_id": 71004},
            ],
        )
    assert collision_scheduler.calls == []

    blocked_scheduler = _PressureScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="2/3",
    ):
        probe.submit_operational_pressure_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=strict_cutover_path,
            output=tmp_path / "forbidden-pressure-submission.json",
            scheduler=blocked_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=retry_task_reader,
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(original_submission)[:1]
                + _operational_pressure_events(original_submission)[2:]
            ),
        )
    assert blocked_scheduler.calls == []

    scheduler = _PressureScheduler()
    sibling_rows = iter(([], [], [post_readback]))
    submission_path = probe.submit_operational_pressure_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=strict_cutover_path,
        output=tmp_path / "pressure-retry-submission.json",
        scheduler=scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=retry_task_reader,
        event_reader=event_reader,
        task_list_reader=lambda **_kwargs: next(sibling_rows),
        same_node_as_task_id=72000,
        expected_allocation_id=9002,
        expected_slurm_job_id="81300",
        expected_account_name="anchor-account",
        expected_node_name="n116",
    )
    submission = probe._load_submission(
        submission_path, plan=retry_plan
    )
    assert submission["task_id"] == 71003
    assert submission["task_id"] != original_submission["task_id"]
    claim_receipt = submission["operational_pressure_atomic_claim"]
    assert claim_receipt["acquisition_status"] == "fresh_pending"
    assert (
        claim_receipt["fresh_claim_authorized_scheduler_submit_call"]
        is True
    )
    assert claim_receipt["finalized_claim"]["task_id"] == 71003
    assert submission["retry_of_operational_pressure"] == retry_plan[
        "retry_of_operational_pressure"
    ]
    assert "retry_of_timeout" not in submission
    assert scheduler.calls[0][1]["node_name"] == "n116"
    assert scheduler.calls[0][1]["node_name_policy"] == "strict"
    assert scheduler.calls[0][1]["same_node_as_task_id"] == 72000
    assert scheduler.calls[0][1]["account_name"] == "anchor-account"
    assert submission["scheduler_placement_contract"][
        "expected_node_name"
    ] == "n116"
    for index, alternate_path in enumerate(alternate_plan_paths):
        losing_scheduler = _PressureScheduler()
        with pytest.raises(
            production.HandoffContractError,
            match="atomic claim acquisition failed",
        ):
            probe.submit_operational_pressure_retry(
                plan_path=alternate_path,
                scheduler_cutover_receipt_path=(
                    strict_cutover_path
                    if index
                    else legacy_cutover_path
                ),
                output=tmp_path
                / f"forbidden-pressure-placement-{index}.json",
                scheduler=losing_scheduler,
                predictor=_Predictor(),
                live_reader=_live_scheduler_reader,
                task_reader=retry_task_reader,
                event_reader=event_reader,
                task_list_reader=lambda **_kwargs: [post_readback],
            )
        assert losing_scheduler.calls == []
    terminal = {
        **post_readback,
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "failure_message": "",
        "allocation_id": 9002,
        "assigned_allocation": 9002,
        "allocation_node_name": "n116",
        "actual_node_name": "n116",
        "slurm_job_id": "81300",
        "account_name": "anchor-account",
        "placement_contract_satisfied": True,
        "started_at": "2026-07-25 05:00:00",
        "finished_at": "2026-07-25 06:00:00",
        "remote_cwd": "/gpfs/test",
        "remote_dir": "runs/task-71003",
    }
    assert probe._task_execution_evidence(
        terminal, submission=submission
    )["actual_node_name"] == "n116"
    with pytest.raises(
        production.HandoffContractError,
        match="terminal execution evidence drifted",
    ):
        probe._task_execution_evidence(
            {**terminal, "timeout_seconds": 28800},
            submission=submission,
        )
    with pytest.raises(
        production.HandoffContractError,
        match="fell back|escaped|identity drifted",
    ):
        probe._task_execution_evidence(
            {
                **terminal,
                "allocation_node_name": "n109",
                "actual_node_name": "n109",
            },
            submission=submission,
        )


def test_operational_pressure_after_timeout_is_bounded_and_eight_hours(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    base_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    base_submission_path = probe.submit_standard(
        plan_path=base_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "compound-base-submission.json",
        scheduler=_FakeScheduler(),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    base_plan = probe._load_plan(base_plan_path)[0]
    base_submission = probe._load_submission(
        base_submission_path, plan=base_plan
    )
    timeout_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=base_plan_path,
        original_submission_path=base_submission_path,
        output=tmp_path / "compound-timeout-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            base_submission
        ),
    )

    class _TaskIdScheduler(_FakeScheduler):
        def __init__(self, task_id):
            super().__init__()
            self.task_id = task_id

        def submit_verification(self, *args, **kwargs):
            guard = kwargs.get("pre_submit_guard")
            if guard is not None:
                guard()
            self.calls.append((args, kwargs))
            return self.task_id

    timeout_submission_path = probe.submit_timeout_retry(
        plan_path=timeout_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "compound-timeout-submission.json",
        scheduler=_TaskIdScheduler(71002),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            base_submission
        ),
    )
    timeout_plan = probe._load_plan(timeout_plan_path)[0]
    timeout_submission = probe._load_submission(
        timeout_submission_path, plan=timeout_plan
    )

    def pressure_snapshot(submission):
        return _operational_pressure_task_snapshot(
            submission,
            timeout_seconds=28800,
            created_at="2026-07-24 18:23:00",
        )

    compound_plan_path = probe.create_operational_pressure_retry_plan(
        original_plan_path=timeout_plan_path,
        original_submission_path=timeout_submission_path,
        output=tmp_path / "compound-pressure-plan",
        task_reader=lambda **_kwargs: pressure_snapshot(
            timeout_submission
        ),
        event_reader=lambda **_kwargs: _operational_pressure_events(
            timeout_submission
        ),
    )
    compound_plan = probe._load_plan(compound_plan_path)[0]
    compound_record = compound_plan["retry_of_operational_pressure"]
    assert compound_record["immediate_retry_kind"] == "timeout"
    assert compound_record["retry_of_task_id"] == 71002
    assert compound_record["logical_authority_task_id"] == 71001
    assert compound_plan["stage"]["resources"] == {
        "cpus": 8,
        "timeout_seconds": 28800,
    }
    assert "pressure-after-timeout-r1-l71001-" in compound_plan[
        "stage"
    ]["task_name"]
    assert "-i71002-" not in compound_plan["stage"]["task_name"]

    alternate_compound_path = (
        probe.create_operational_pressure_retry_plan(
            original_plan_path=timeout_plan_path,
            original_submission_path=timeout_submission_path,
            output=tmp_path / "compound-pressure-plan-strict",
            strict_node_name="n108",
            task_reader=lambda **_kwargs: pressure_snapshot(
                timeout_submission
            ),
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(timeout_submission)
            ),
        )
    )
    alternate_compound = probe._load_plan(
        alternate_compound_path
    )[0]
    assert alternate_compound["stage"]["task_name"] == compound_plan[
        "stage"
    ]["task_name"]
    assert alternate_compound["stage"]["retained_aedt_bundle"][
        "dedupe_key"
    ] == compound_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]

    sibling = {
        "task_id": 71003,
        "name": compound_plan["stage"]["task_name"],
        "dedupe_key": compound_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
        "status": "queued",
        "state": "queued",
        "project": scheduler_client.MFT_PROJECT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 28800,
        "aedt_backend": "standalone",
        "requested_node_name": "",
        "requested_node_name_policy": "",
        "same_node_as_task_id": 0,
    }
    sibling_rows = iter(([], [], [sibling]))
    compound_scheduler = _TaskIdScheduler(71003)
    compound_submission_path = (
        probe.submit_operational_pressure_retry(
            plan_path=compound_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "compound-pressure-submission.json",
            scheduler=compound_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=lambda **_kwargs: pressure_snapshot(
                timeout_submission
            ),
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(timeout_submission)
            ),
            task_list_reader=lambda **_kwargs: next(sibling_rows),
        )
    )
    compound_submission = probe._load_submission(
        compound_submission_path, plan=compound_plan
    )
    assert compound_submission["task_id"] == 71003
    assert compound_submission["resources"]["timeout_seconds"] == 28800
    assert len(compound_scheduler.calls) == 1
    with pytest.raises(
        production.HandoffContractError,
        match="bounded direct-or-timeout chain",
    ):
        probe.create_operational_pressure_retry_plan(
            original_plan_path=compound_plan_path,
            original_submission_path=compound_submission_path,
            output=tmp_path / "forbidden-compound-depth",
            task_reader=lambda **_kwargs: pressure_snapshot(
                compound_submission
            ),
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(compound_submission)
            ),
        )


def test_operational_pressure_pending_claim_recovers_without_repost(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "pending-original-submission.json",
        scheduler=_FakeScheduler(),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    retry_plan_path = probe.create_operational_pressure_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "pending-pressure-plan",
        task_reader=lambda **_kwargs: (
            _operational_pressure_task_snapshot(original_submission)
        ),
        event_reader=lambda **_kwargs: (
            _operational_pressure_events(original_submission)
        ),
    )
    retry_plan = probe._load_plan(retry_plan_path)[0]
    reference = retry_plan[
        "operational_pressure_atomic_claim_reference"
    ]
    winner = probe._operational_pressure_claim_winner(
        retry_plan_path, retry_plan
    )
    sibling = {
        "task_id": 71005,
        "name": retry_plan["stage"]["task_name"],
        "status": "queued",
        "state": "queued",
        "dedupe_key": retry_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
        "project": scheduler_client.MFT_PROJECT,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 14400,
        "requested_node_name": None,
        "requested_node_name_policy": None,
        "same_node_as_task_id": 0,
    }
    class _IgnoringGuardScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 71005

    ignored_guard_scheduler = _IgnoringGuardScheduler()
    ignored_guard_rows = iter(([], [sibling]))
    with pytest.raises(
        production.HandoffContractError,
        match="locked post-claim pre-POST guard",
    ):
        probe.submit_operational_pressure_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "ignored-guard-submission.json",
            scheduler=ignored_guard_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=lambda **_kwargs: (
                _operational_pressure_task_snapshot(
                    original_submission
                )
            ),
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(original_submission)
            ),
            task_list_reader=lambda **_kwargs: next(
                ignored_guard_rows
            ),
        )
    assert len(ignored_guard_scheduler.calls) == 1
    atomic_claim.validate_pending_claim(
        probe.OPERATIONAL_PRESSURE_CLAIM_ROOT,
        reference,
        expected_winner=winner,
    )
    no_repost_scheduler = _FakeScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="exactly one matching API task and never re-posts",
    ):
        probe.submit_operational_pressure_retry(
            plan_path=retry_plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "pending-zero-sibling.json",
            scheduler=no_repost_scheduler,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
            task_reader=lambda **_kwargs: (
                _operational_pressure_task_snapshot(
                    original_submission
                )
            ),
            event_reader=lambda **_kwargs: (
                _operational_pressure_events(original_submission)
            ),
            task_list_reader=lambda **_kwargs: [],
        )
    assert no_repost_scheduler.calls == []
    recovered_path = probe.submit_operational_pressure_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "pending-recovered-submission.json",
        scheduler=no_repost_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=lambda **_kwargs: (
            _operational_pressure_task_snapshot(original_submission)
        ),
        event_reader=lambda **_kwargs: (
            _operational_pressure_events(original_submission)
        ),
        task_list_reader=lambda **_kwargs: [sibling],
    )
    assert no_repost_scheduler.calls == []
    recovered = probe._load_submission(
        recovered_path, plan=retry_plan
    )
    claim = recovered["operational_pressure_atomic_claim"]
    assert recovered["task_id"] == 71005
    assert claim["acquisition_status"] == "existing_pending"
    assert claim["fresh_claim_authorized_scheduler_submit_call"] is False
    assert claim["recovered_without_scheduler_submit_call"] is True
    assert claim["finalized_claim"]["task_id"] == 71005


def test_strict_retry_rejects_unsafe_node_name_before_plan_write(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=_FakeScheduler(),
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    with pytest.raises(
        production.HandoffContractError, match="unsafe or empty"
    ):
        probe.create_timeout_retry_plan(
            original_plan_path=original_plan_path,
            original_submission_path=original_submission_path,
            output=tmp_path / "unsafe-strict-plan",
            strict_node_name="n110; echo unsafe",
            task_reader=lambda **_kwargs: _timeout_task_snapshot(
                original_submission
            ),
        )
    assert not (tmp_path / "unsafe-strict-plan").exists()


def test_live_active_strict_cutover_receipt_is_pinned_when_present():
    path = Path(
        "C:/Users/peets/slurm_scheduler_runtime/deployment_candidates/"
        "41b3b9393684-strict-cpu-storage-admission-20260725/"
        "cutover_receipt.json"
    )
    if not path.is_file():
        pytest.skip("live active strict-node cutover receipt is host-local")
    receipt, launcher = probe._validate_scheduler_cutover_receipt(
        path,
        verify_live_launcher=True,
        require_strict_node=True,
    )
    assert receipt["candidate_revision"] == (
        probe.SCHEDULER_STRICT_NODE_REVISION
    )
    assert launcher["sha256"] == (
        probe.SCHEDULER_STRICT_NODE_LAUNCHER_SHA256
    )
    assert receipt["pin_generation"] == "scheduler-strict-node-41b-v3"


def test_live_legacy_e542_receipt_remains_collectable_when_present():
    path = Path(
        "C:/Users/peets/slurm_scheduler_runtime/deployment_candidates/"
        "e542c8a6350d-pressure-episode-gate-20260725/"
        "cutover_receipt.json"
    )
    if not path.is_file():
        pytest.skip("live legacy strict-node cutover receipt is host-local")
    legacy_pin = probe._legacy_e542_strict_node_scheduler_pin()
    legacy_contract = probe._strict_node_plan_contract(
        "n110", scheduler_pin=legacy_pin
    )
    receipt, launcher = probe._validate_scheduler_cutover_receipt(
        path,
        verify_live_launcher=False,
        require_strict_node=True,
        strict_node_contract=legacy_contract,
    )
    assert receipt["candidate_revision"] == (
        probe.SCHEDULER_STRICT_NODE_LEGACY_E542_REVISION
    )
    assert receipt["pin_generation"] == "scheduler-strict-node-e542-v2"
    assert launcher is None


def test_strict_generation_registry_is_legacy_read_active_write():
    active_contract = probe._strict_node_plan_contract("n110")
    legacy_pin = probe._legacy_e542_strict_node_scheduler_pin()
    legacy_contract = probe._strict_node_plan_contract(
        "n110", scheduler_pin=legacy_pin
    )
    assert probe._strict_node_scheduler_pin(
        active_contract, require_active=True
    )["pin_generation"] == "scheduler-strict-node-41b-v3"
    assert probe._strict_node_scheduler_pin(legacy_contract)[
        "pin_generation"
    ] == "scheduler-strict-node-e542-v2"
    with pytest.raises(
        production.HandoffContractError,
        match="historical strict-node generation cannot submit",
    ):
        probe._strict_node_scheduler_pin(
            legacy_contract, require_active=True
        )
    mixed = {
        **active_contract,
        "scheduler_cutover_receipt_sha256": legacy_contract[
            "scheduler_cutover_receipt_sha256"
        ],
    }
    with pytest.raises(
        production.HandoffContractError,
        match="placement contract drifted",
    ):
        probe._validate_strict_node_plan_contract(mixed)


def test_live_cross_generation_cutover_substitution_is_rejected_when_present():
    active_path = Path(
        "C:/Users/peets/slurm_scheduler_runtime/deployment_candidates/"
        "41b3b9393684-strict-cpu-storage-admission-20260725/"
        "cutover_receipt.json"
    )
    legacy_path = Path(
        "C:/Users/peets/slurm_scheduler_runtime/deployment_candidates/"
        "e542c8a6350d-pressure-episode-gate-20260725/"
        "cutover_receipt.json"
    )
    if not active_path.is_file() or not legacy_path.is_file():
        pytest.skip("live strict-node cutover receipts are host-local")
    active_contract = probe._strict_node_plan_contract("n110")
    legacy_contract = probe._strict_node_plan_contract(
        "n110",
        scheduler_pin=probe._legacy_e542_strict_node_scheduler_pin(),
    )
    with pytest.raises(
        production.HandoffContractError,
        match="cutover identity drifted",
    ):
        probe._validate_scheduler_cutover_receipt(
            legacy_path,
            verify_live_launcher=False,
            require_strict_node=True,
            strict_node_contract=active_contract,
        )
    with pytest.raises(
        production.HandoffContractError,
        match="cutover identity drifted",
    ):
        probe._validate_scheduler_cutover_receipt(
            active_path,
            verify_live_launcher=False,
            require_strict_node=True,
            strict_node_contract=legacy_contract,
        )


def test_live_legacy_e542_submission_loads_but_cannot_resubmit_after_successor(
    tmp_path,
):
    root = Path(
        "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
        "standard_timeout_retries_strict_r2_e542_n110_260725/"
        "t96213_a4ae16f8a0c3"
    )
    plan_path = root / "plan" / "diagnostic_timeout_retry_plan.json"
    submission_path = root / "diagnostic_timeout_retry_submission.json"
    if not plan_path.is_file() or not submission_path.is_file():
        pytest.skip("live historical e542 submission is host-local")
    plan = probe._load_plan(plan_path)[0]
    submission = probe._load_submission(submission_path, plan=plan)
    assert plan["scheduler_strict_node_contract"][
        "scheduler_revision"
    ] == probe.SCHEDULER_STRICT_NODE_LEGACY_E542_REVISION
    assert submission["task_id"] == 96274
    assert submission["scheduler_live_launcher_identity"]["sha256"] == (
        probe.SCHEDULER_STRICT_NODE_LEGACY_E542_LAUNCHER_SHA256
    )
    scheduler = _FakeScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="historical strict-node generation cannot submit",
    ):
        probe.submit_timeout_retry(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=Path(
                "C:/Users/peets/slurm_scheduler_runtime/"
                "deployment_candidates/"
                "41b3b9393684-strict-cpu-storage-admission-20260725/"
                "cutover_receipt.json"
            ),
            output=tmp_path / "forbidden-legacy-resubmission.json",
            scheduler=scheduler,
        )
    assert scheduler.calls == []
    assert not (tmp_path / "forbidden-legacy-resubmission.json").exists()


def test_live_queued_41b_same_allocation_submissions_round_trip_when_present():
    root = Path(
        "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
        "standard_timeout_retries_strict_r3_41b_260725"
    )
    expected = {
        "t96231_n108_efffb6518d4e": 96281,
        "t96227_n108_7ce2bf976d48": 96282,
        "t96228_n110_90598193e992": 96283,
        "t96223_n110_2a1bb6f2be79": 96284,
    }
    if not all((root / child).is_dir() for child in expected):
        pytest.skip("live queued 41b submissions are host-local")
    for child, task_id in expected.items():
        artifact_root = root / child
        plan = probe._load_plan(
            artifact_root
            / "plan"
            / "diagnostic_timeout_retry_plan.json"
        )[0]
        submission = probe._load_submission(
            artifact_root / "diagnostic_timeout_retry_submission.json",
            plan=plan,
        )
        submitted = submission["scheduler_placement_contract"][
            "submitted_task_after_submission"
        ]
        assert submission["task_id"] == task_id
        assert submitted["status"] == "queued"
        assert submitted["state"] == "queued"
        assert submitted["actual_node_name"] == ""


def test_scheduler_client_strict_opt_in_sends_policy_and_returns_post_evidence(
    monkeypatch,
):
    captured = {}

    class _Response:
        status_code = 201

        @staticmethod
        def json():
            return {
                "id": 73001,
                "node_name": "n110",
                "node_name_policy": "strict",
            }

    monkeypatch.setattr(
        scheduler_client,
        "reconcile_task_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_client,
        "live_project_submission_snapshot",
        lambda *_args, **_kwargs: {"project_submission_slots": 1},
    )

    def post(_url, *, json, timeout):
        assert captured.get("guard_count") == 1
        captured["payload"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(scheduler_client.requests, "post", post)
    result = scheduler_client.submit_verification(
        "strict-client-test",
        "strict_client_test",
        {"N1": 6},
        {"timeout_seconds": 60, "cli_flags": ""},
        solver_revision="a" * 40,
        library_revision="b" * 40,
        required_project_cap=500,
        max_project_active_tasks=500,
        node_name="n110",
        node_name_policy="strict",
        return_submission_evidence=True,
        pre_submit_guard=lambda: captured.update(
            guard_count=captured.get("guard_count", 0) + 1
        ),
        scheduler_url=probe.DIAGNOSTIC_SCHEDULER_URL,
    )
    assert captured["guard_count"] == 1
    assert captured["payload"]["node_name"] == "n110"
    assert captured["payload"]["node_name_policy"] == "strict"
    assert result["task_id"] == 73001
    assert result["submission_source"] == "post_created"
    assert result["api_post_submission_response"][
        "node_name_policy"
    ] == "strict"


def test_mesh_quality_canary_authenticates_nested_preflight_and_ignores_supplementals():
    submission = {
        "task_id": probe.MESH_QUALITY_CANARY_FAILED_TASK_ID,
        "task_name": "source-timeout",
        "dedupe_key": "source-dedupe",
    }
    snapshot = {
        "task_id": probe.MESH_QUALITY_CANARY_FAILED_TASK_ID,
        "name": submission["task_name"],
        "status": "failed",
        "state": "failed",
        "exit_code": 1,
        "failure_message": probe.MESH_QUALITY_CANARY_FAILURE_MESSAGE,
        "slurm_job_id": "826839",
        "allocation_id": 14619,
        "account_name": "jji0930",
        "actual_node_name": "n108",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 28800,
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": submission["dedupe_key"],
        "remote_cwd": "runs",
        "remote_dir": "task-96264",
        "finished_at": "2026-07-25T06:22:41Z",
    }
    result = {
        "thermal_iterations": 0,
        "thermal_convergence_reason": "native_terminal_error",
        "thermal_dispatch_forensic_json": json.dumps({
            "attempts": [{
                "mesh_preflight": {
                    "nested_readback": {"operation_count": 35},
                    "passed": True,
                }
            }]
        }),
    }
    stdout = (
        probe.MESH_QUALITY_CANARY_NATIVE_MESSAGE
        + "\nrx_side_block_mesh_level_ABC123_L_5\n"
        + "RESULT_JSON "
        + json.dumps(result)
    )
    evidence = probe._mesh_quality_failure_evidence(
        snapshot, stdout, submission=submission
    )
    assert evidence["passed_native_premesh_marker_count"] == 1
    assert evidence["zero_iteration_convergence_marker_count"] == 1

    plan = {
        "stage": {
            "task_name": "mft-goal-diag-standard-mesh-canary-r1-"
            "l96225-7a6ccac265d3",
            "retained_aedt_bundle": {"dedupe_key": "canary-dedupe"},
        }
    }
    supplementals = [
        {"task_id": task_id, "name": f"supplemental-{task_id}"}
        for task_id in (96269, 96271, 96272)
    ]
    guard = probe._mesh_quality_canary_sibling_snapshot(
        supplementals, plan=plan
    )
    probe._validate_mesh_quality_canary_sibling_snapshot(
        guard, plan=plan, expected_count=0
    )
    assert guard["rejected_supplemental_task_ids_present"] == [
        96269,
        96271,
        96272,
    ]


def test_mesh_quality_canary_retains_bundle_after_failed_simulation(
    monkeypatch,
):
    captured = {}

    class _Response:
        status_code = 201

        @staticmethod
        def json():
            return {"id": 73002}

    monkeypatch.setattr(
        scheduler_client,
        "retained_aedt_identity",
        lambda *_args, **_kwargs: {"schema_version": "canary"},
    )
    monkeypatch.setattr(
        scheduler_client,
        "_retained_aedt_export_command",
        lambda _retained: "MFT_CANARY_EXPORT=1; ",
    )
    monkeypatch.setattr(
        scheduler_client,
        "_runtime_license_refresh_command",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        scheduler_client,
        "reconcile_task_id",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_client,
        "live_project_submission_snapshot",
        lambda *_args, **_kwargs: {"project_submission_slots": 1},
    )

    def post(_url, *, json, timeout):
        captured["payload"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(scheduler_client.requests, "post", post)
    task_id = scheduler_client.submit_verification(
        "mesh-canary-retention-test",
        "mesh_canary_retention_test",
        {"N1": 6},
        {
            "schema_version": probe.MESH_QUALITY_CANARY_PROFILE_SCHEMA,
            "timeout_seconds": 60,
            "cli_flags": "",
        },
        solver_revision="a" * 40,
        library_revision="b" * 40,
        required_project_cap=500,
        max_project_active_tasks=500,
        scheduler_url=probe.DIAGNOSTIC_SCHEDULER_URL,
    )
    assert task_id == 73002
    command = captured["payload"]["command"]
    assert "simulation_rc=$?" in command
    assert "MFT_CANARY_EXPORT=1" in command
    assert 'if [ "$simulation_rc" -eq 0 ]' not in command


def test_timeout_retry_parser_exposes_complete_same_allocation_contract():
    claim_init = probe._parser().parse_args(
        ["init-operational-pressure-claim-root"]
    )
    assert claim_init.command == (
        "init-operational-pressure-claim-root"
    )
    plan_args = probe._parser().parse_args(
        [
            "plan-timeout-retry",
            "--original-plan",
            "original-plan.json",
            "--original-submission",
            "original-submission.json",
            "--strict-node-name",
            "n110",
            "--output",
            "strict-plan",
        ]
    )
    assert plan_args.strict_node_name == "n110"
    args = probe._parser().parse_args(
        [
            "submit-timeout-retry",
            "--plan",
            "plan.json",
            "--scheduler-cutover-receipt",
            "cutover.json",
            "--same-node-as-task-id",
            "96270",
            "--expected-allocation-id",
            "14616",
            "--expected-slurm-job-id",
            "826363",
            "--expected-account-name",
            "dhj02",
            "--expected-node-name",
            "n110",
            "--output",
            "submission.json",
        ]
    )
    assert args.same_node_as_task_id == 96270
    assert args.expected_allocation_id == 14616
    assert args.expected_slurm_job_id == "826363"
    assert args.expected_account_name == "dhj02"
    assert args.expected_node_name == "n110"
    pressure = probe._parser().parse_args(
        [
            "submit-operational-pressure-retry",
            "--plan",
            "pressure-plan.json",
            "--scheduler-cutover-receipt",
            "cutover.json",
            "--same-node-as-task-id",
            "96301",
            "--expected-allocation-id",
            "14630",
            "--expected-slurm-job-id",
            "826999",
            "--expected-account-name",
            "r1jae262",
            "--expected-node-name",
            "n116",
            "--output",
            "pressure-submission.json",
        ]
    )
    assert pressure.same_node_as_task_id == 96301
    assert pressure.expected_allocation_id == 14630
    assert pressure.expected_slurm_job_id == "826999"
    assert pressure.expected_account_name == "r1jae262"
    assert pressure.expected_node_name == "n116"


def test_operational_pressure_claim_root_cli_is_idempotent_and_local_only(
    tmp_path, monkeypatch, capsys
):
    claim_root = (tmp_path / "cli-pressure-claims").resolve()
    monkeypatch.setattr(
        probe, "OPERATIONAL_PRESSURE_CLAIM_ROOT", claim_root
    )
    scheduler_calls = []
    monkeypatch.setattr(
        scheduler_client,
        "submit_verification",
        lambda *args, **kwargs: scheduler_calls.append((args, kwargs)),
    )
    command = ["init-operational-pressure-claim-root"]
    assert probe.main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert probe.main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["path"] == str(claim_root)
    assert first["claim_root_authority"]["campaign_id"] == (
        "mft-goal-20260726"
    )
    assert scheduler_calls == []


def test_timeout_retry_collection_is_accepted_by_strict_al(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    original_plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    original_scheduler = _FakeScheduler()
    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=original_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    retry_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "retry-plan",
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )

    class _RetryScheduler(_FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 71002

    retry_scheduler = _RetryScheduler()
    retry_submission_path = probe.submit_timeout_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "retry-submission.json",
        scheduler=retry_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=lambda **_kwargs: _timeout_task_snapshot(
            original_submission
        ),
    )
    retry_plan, retry_params, retry_selected = probe._load_plan(
        retry_plan_path
    )
    retry_submission = probe._load_submission(
        retry_submission_path, plan=retry_plan
    )
    retry_scheduler.result = _result(
        retry_plan_path, retry_submission
    )
    metadata_reader, manifest_reader = _remote_evidence(
        retry_submission, retry_scheduler.result
    )
    collection_path = probe.collect_standard(
        plan_path=retry_plan_path,
        submission_path=retry_submission_path,
        output=tmp_path / "retry-collection.json",
        scheduler=retry_scheduler,
        remote_reader=metadata_reader,
        manifest_reader=manifest_reader,
        task_reader=lambda **_kwargs: _task_snapshot(retry_submission),
    )
    monkeypatch.setattr(
        probe,
        "_load_llt_predictor",
        lambda _identity: _Predictor(),
    )

    truth = strict_al.authenticate_collection(collection_path)

    assert truth.adapter_kind == "diagnostic"
    assert truth.collection["task_id"] == 71002
    assert retry_plan["retry_of_timeout"]["retry_of_task_id"] == 71001
    assert truth.source_task_payload_sha256 == (
        retry_selected["task_identity"]["payload_sha256"]
    )
    assert retry_params["fan_velocity"] == 1.5
    assert retry_params["wcp_pad_t"] == 2.0
    assert retry_params["core_plate_pad_t"] == 2.0

    pressure_plan_path = probe.create_operational_pressure_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "strict-al-pressure-plan",
        task_reader=lambda **_kwargs: (
            _operational_pressure_task_snapshot(original_submission)
        ),
        event_reader=lambda **_kwargs: _operational_pressure_events(
            original_submission
        ),
    )
    pressure_plan = probe._load_plan(pressure_plan_path)[0]
    pressure_sibling = {
        "task_id": 71003,
        "name": pressure_plan["stage"]["task_name"],
        "dedupe_key": pressure_plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
        "status": "queued",
        "state": "queued",
        "project": scheduler_client.MFT_PROJECT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 14400,
        "aedt_backend": "standalone",
        "requested_node_name": "",
        "requested_node_name_policy": "",
        "same_node_as_task_id": 0,
    }
    pressure_sibling_rows = iter(([], [], [pressure_sibling]))
    pressure_scheduler = _RetryScheduler()
    pressure_scheduler.submit_verification = (
        lambda *args, **kwargs: (
            kwargs["pre_submit_guard"](),
            pressure_scheduler.calls.append((args, kwargs)),
            71003,
        )[-1]
    )
    pressure_submission_path = probe.submit_operational_pressure_retry(
        plan_path=pressure_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "strict-al-pressure-submission.json",
        scheduler=pressure_scheduler,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
        task_reader=lambda **_kwargs: (
            _operational_pressure_task_snapshot(original_submission)
        ),
        event_reader=lambda **_kwargs: _operational_pressure_events(
            original_submission
        ),
        task_list_reader=lambda **_kwargs: next(
            pressure_sibling_rows
        ),
    )
    pressure_submission = probe._load_submission(
        pressure_submission_path, plan=pressure_plan
    )
    pressure_scheduler.result = _result(
        pressure_plan_path, pressure_submission
    )
    pressure_metadata, pressure_manifest = _remote_evidence(
        pressure_submission, pressure_scheduler.result
    )
    pressure_collection_path = probe.collect_standard(
        plan_path=pressure_plan_path,
        submission_path=pressure_submission_path,
        output=tmp_path / "strict-al-pressure-collection.json",
        scheduler=pressure_scheduler,
        remote_reader=pressure_metadata,
        manifest_reader=pressure_manifest,
        task_reader=lambda **_kwargs: _task_snapshot(
            pressure_submission
        ),
    )
    pressure_truth = strict_al.authenticate_collection(
        pressure_collection_path
    )
    assert pressure_truth.adapter_kind == "diagnostic"
    assert pressure_truth.collection["task_id"] == 71003
    with pytest.raises(
        strict_al.StrictALIngestError,
        match="candidate_physics_sha256 identity is duplicated",
    ):
        strict_al._validate_new_collection_identities(
            [
                {
                    "collection_payload_sha256": truth.collection[
                        "payload_sha256"
                    ],
                    "candidate_physics_sha256": truth.plan[
                        "candidate_physics_sha256"
                    ],
                    "task_id": truth.collection["task_id"],
                    "logical_authority_task_id": (
                        original_submission["task_id"]
                    ),
                    "project_name": "original",
                    "saved_at": "2026-07-25T00:00:00Z",
                },
                {
                    "collection_payload_sha256": (
                        pressure_truth.collection["payload_sha256"]
                    ),
                    "candidate_physics_sha256": pressure_truth.plan[
                        "candidate_physics_sha256"
                    ],
                    "task_id": pressure_truth.collection["task_id"],
                    "logical_authority_task_id": (
                        original_submission["task_id"]
                    ),
                    "project_name": "retry",
                    "saved_at": "2026-07-25T01:00:00Z",
                },
            ]
        )


def test_submit_and_collect_remain_diagnostic_after_actual_pass(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    plan_path = _make_plan(tmp_path, fixture)
    fake = _FakeScheduler()
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    submission_path = probe.submit_standard(
        plan_path=plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "submission.json",
        scheduler=fake,
        predictor=_Predictor(),
        live_reader=_live_scheduler_reader,
    )
    submission = probe._load_submission(
        submission_path, plan=probe._load_plan(plan_path)[0]
    )
    assert len(fake.calls) == 1
    assert (
        fake.calls[0][1]["scheduler_url"]
        == probe.DIAGNOSTIC_SCHEDULER_URL
    )
    assert (
        fake.calls[0][1]["required_project_cap"]
        == probe.GOAL_FEA_PROJECT_CAP
    )
    assert (
        fake.calls[0][1]["max_project_active_tasks"]
        == probe.GOAL_FEA_PROJECT_CAP
    )
    assert submission["scheduler_admission_snapshot"][
        "license_admission_verified"
    ]
    fake.result = _result(plan_path, submission)
    metadata_reader, manifest_reader = _remote_evidence(
        submission, fake.result
    )
    collection_path = probe.collect_standard(
        plan_path=plan_path,
        submission_path=submission_path,
        output=tmp_path / "collection.json",
        scheduler=fake,
        remote_reader=metadata_reader,
        manifest_reader=manifest_reader,
        task_reader=lambda **_kwargs: _task_snapshot(submission),
    )
    collection, *_rest = probe._load_collection(
        collection_path, predictor=_Predictor()
    )
    view = probe.authenticate_collection(
        collection_path, predictor=_Predictor()
    )
    assert set(view) == {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    }
    assert (
        view["schema_version"]
        == probe.AUTHENTICATED_COLLECTION_SCHEMA
    )
    assert collection["goal_physical_spec_passed"] is True
    assert collection["actual_body_probe_temperature_gate_passed"] is True
    assert collection["diagnostic_truth_observation_available"] is True
    assert collection["diagnostic_only"] is True
    assert collection["production_eligible"] is False
    assert collection["automatic_promotion"] is False
    assert collection["full_submission_allowed"] is False
    assert collection["production_package_allowed"] is False
    assert collection["scheduler_task_execution"]["exit_code"] == 0
    truth = collection["truth_evidence"]
    assert truth["actual_physical_Llt_uH"] == 27.5
    assert truth["actual_resonance_Hz"] == 15000.0
    assert truth["actual_exterior_dimensions_mm"]["W"] <= 1200.0
    assert truth["actual_exterior_dimensions_mm"]["L"] <= 1000.0
    assert truth["actual_exterior_dimensions_mm"]["H"] <= 750.0
    assert truth["retained_symmetric_aedtresults"]["file_count"] == 1


def test_scheduler_launcher_identity_accepts_same_resolved_path_alias(
    tmp_path,
):
    launcher = tmp_path / "scheduler-live-launcher.cmd"
    launcher.write_bytes(b"reviewed-launcher")
    alias_parent = tmp_path / "path-alias"
    alias_parent.mkdir()
    alias = alias_parent / ".." / launcher.name
    other = tmp_path / "other-launcher.cmd"
    other.write_bytes(launcher.read_bytes())

    assert probe._same_absolute_regular_file_identity(
        str(alias.absolute()), str(launcher.resolve())
    )
    assert not probe._same_absolute_regular_file_identity(
        str(other.resolve()), str(launcher.resolve())
    )
    assert not probe._same_absolute_regular_file_identity(
        str(tmp_path / "missing.cmd"), str(launcher.resolve())
    )


def test_submit_freshly_reauthenticates_source_before_scheduler_mutation(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    plan_path = _make_plan(tmp_path, fixture)
    source = fixture["result_paths"][1]
    source.chmod(0o666)
    source.write_bytes(source.read_bytes() + b"\n")
    fake = _FakeScheduler()
    with pytest.raises(
        production.HandoffContractError,
        match="source result|result bytes|authentication",
    ):
        probe.submit_standard(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=tmp_path / "missing-cutover.json",
            output=tmp_path / "forbidden.json",
            scheduler=fake,
            predictor=_Predictor(),
        )
    assert fake.calls == []


def test_submit_rejects_launcher_drift_and_blocked_license_before_post(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    plan_path = _make_plan(tmp_path, fixture)
    cutover_path = _scheduler_cutover(tmp_path, monkeypatch)
    cutover = production._read_json(cutover_path)
    launcher = Path(cutover["live_launcher_path"])
    launcher.write_bytes(b"drifted-launcher")
    fake = _FakeScheduler()
    with pytest.raises(
        production.HandoffContractError, match="live launcher"
    ):
        probe.submit_standard(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "launcher-forbidden.json",
            scheduler=fake,
            predictor=_Predictor(),
            live_reader=_live_scheduler_reader,
        )
    assert fake.calls == []

    launcher.write_bytes(b"reviewed-launcher")

    def blocked_reader(**kwargs):
        value = _live_scheduler_reader(**kwargs)
        if kwargs["endpoint"] == "/api/licenses":
            value["admission"]["blocked_reason"] = "capacity exhausted"
        return value

    with pytest.raises(
        production.HandoffContractError, match="license snapshot"
    ):
        probe.submit_standard(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path / "license-forbidden.json",
            scheduler=fake,
            predictor=_Predictor(),
            live_reader=blocked_reader,
        )
    assert fake.calls == []


def test_diagnostic_retention_export_preserves_project_and_results(tmp_path):
    profile, _source = probe._profile_content()
    retained = scheduler_client.retained_aedt_identity(
        "diagnostic-export-test",
        {"x": 1},
        profile,
        "2" * 40,
        "3" * 40,
    )
    command = scheduler_client._retained_aedt_export_command(retained)
    tokens = shlex.split(command)
    script = tokens[tokens.index("-c") + 1]
    marker_contract_json = tokens[tokens.index("-c") + 9]
    receipt_context_json = tokens[tokens.index("-c") + 10]
    source = tmp_path / "mock-project.aedt"
    source.write_bytes(bytes(range(256)) * 100)
    source_results = tmp_path / "mock-project.aedtresults"
    nested = source_results / "icepak_thermal.results"
    nested.mkdir(parents=True)
    (nested / "state.bin").write_bytes(b"state")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            str(tmp_path / retained["artifact_path"]),
            str(tmp_path / retained["receipt_path"]),
            str(tmp_path / retained["marker_path"]),
            str(tmp_path / retained["transport"]["chunk_directory"]),
            str(tmp_path / retained["results_path"]),
            str(tmp_path / retained["results_manifest_path"]),
            marker_contract_json,
            receipt_context_json,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / retained["artifact_path"]).read_bytes() == (
        source.read_bytes()
    )
    assert (
        tmp_path
        / retained["results_path"]
        / "icepak_thermal.results"
        / "state.bin"
    ).read_bytes() == b"state"
    receipt = json.loads(
        (tmp_path / retained["receipt_path"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / retained["results_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["results_file_count"] == 1
    assert receipt["results_tree_sha256"] == manifest["tree_sha256"]
    assert (
        tmp_path / retained["marker_path"]
    ).name == scheduler_client.SCHEDULER_PRESERVE_MARKER


def test_generation_identity_authenticates_only_pinned_llt_artifacts(tmp_path):
    generation = tmp_path / "generation"
    target = generation / "Llt_phys"
    target.mkdir(parents=True)
    model = target / "models.pkl"
    meta = target / "meta.json"
    model.write_bytes(b"model")
    meta.write_text(
        json.dumps(
            {
                "target": "Llt_phys",
                "training_run_id": "run-1",
                "dataset_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "training_run_id": "run-1",
        "dataset_sha256": "a" * 64,
        "artifacts": {
            "Llt_phys/models.pkl": adapter.sha256_file(model),
            "Llt_phys/meta.json": adapter.sha256_file(meta),
        },
    }
    report_path = generation / "train_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    task = {
        "source_identity": {
            "train_report_sha256": adapter.sha256_file(report_path),
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
        }
    }
    identity = probe._generation_identity(generation, [task])
    assert identity["Llt_phys_model"]["sha256"] == adapter.sha256_file(model)
    model.write_bytes(b"drift")
    with pytest.raises(
        production.HandoffContractError, match="artifacts drifted"
    ):
        probe._generation_identity(generation, [task])
