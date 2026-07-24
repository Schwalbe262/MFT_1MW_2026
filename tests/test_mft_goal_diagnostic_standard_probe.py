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
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production
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
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": submission["dedupe_key"],
        "remote_cwd": "/gpfs/test",
        "remote_dir": "runs/task-71001",
        "finished_at": "2026-07-25 00:00:00",
    }


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
