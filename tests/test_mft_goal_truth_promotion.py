import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from module.mft_goal_20260726_contract import (
    GOAL_TEMPERATURE_TARGETS,
)
from regression_260707.optimization import geometry_metrics
from regression_260707.verify import scheduler_client
from tools import mft_goal_diagnostic_standard_probe as diagnostic
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_truth_promotion as promotion


def _diagnostic_helpers():
    path = Path(__file__).with_name(
        "test_mft_goal_diagnostic_standard_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_mft_goal_diagnostic_test_helpers", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _production_helpers():
    path = Path(__file__).with_name("test_mft_goal_fea_handoff.py")
    spec = importlib.util.spec_from_file_location(
        "_mft_goal_production_test_helpers", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Scheduler:
    RESULT_VALID = scheduler_client.RESULT_VALID

    def __init__(self, first_task_id=73001):
        self.calls = []
        self.result = None
        self.next_task_id = first_task_id

    def submit_verification(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        task_id = self.next_task_id
        self.next_task_id += 1
        return task_id

    def get_status(self, _task_id, **_kwargs):
        return "completed"

    def fetch_result(self, *_args, **_kwargs):
        return SimpleNamespace(
            state=self.RESULT_VALID, result=self.result
        )

    @staticmethod
    def result_matches_params(result, params, required_keys=None):
        return scheduler_client.result_matches_params(
            result, params, required_keys=required_keys
        )


def _standard_collections(
    tmp_path: Path,
    monkeypatch,
    *,
    turns=(5, 6, 7),
    result_mutator=None,
):
    helpers = _diagnostic_helpers()
    production_helpers = _production_helpers()
    bundle, bundle_path, tasks, _task_paths = production_helpers._bundle(
        tmp_path
    )
    result_paths = []
    geometry_by_turns = {}
    for primary_turns in range(5, 9):
        task = next(
            item
            for item in tasks
            if item["fixed_primary_turns"] == primary_turns
        )
        decoded = production_helpers._decoded_params(primary_turns)
        decoded["l1"] = 75.0 + 5.0 * (primary_turns - 5)
        result_path, geometry_sha = (
            production_helpers._write_seed_result(
                tmp_path / "results" / f"n1-{primary_turns}",
                task,
                decoded,
            )
        )
        helpers._convert_to_near_feasible(result_path)
        result_paths.append(result_path)
        geometry_by_turns[primary_turns] = geometry_sha
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
        diagnostic,
        "_generation_identity",
        lambda _path, _tasks: copy.deepcopy(generation_identity),
    )
    selection_path = diagnostic.create_selection(
        bundle_manifest_path=bundle_path,
        generation_path=tmp_path / "generation",
        result_paths=result_paths,
        count=4,
        output=tmp_path / "selection",
        predictor=helpers._Predictor(),
    )
    fixture = {
        "bundle": bundle,
        "bundle_path": bundle_path,
        "result_paths": result_paths,
        "geometry_by_turns": geometry_by_turns,
        "selection_path": selection_path,
    }
    cutover_path = helpers._scheduler_cutover(tmp_path, monkeypatch)
    entries = []
    scheduler = _Scheduler()
    for primary_turns in turns:
        plan_path = diagnostic.create_plan(
            selection_manifest_path=fixture["selection_path"],
            candidate_physics_sha256=fixture[
                "geometry_by_turns"
            ][primary_turns],
            solver_revision="2" * 40,
            library_revision="3" * 40,
            output=tmp_path / f"standard-plan-{primary_turns}",
            predictor=helpers._Predictor(),
        )
        submission_path = diagnostic.submit_standard(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=cutover_path,
            output=tmp_path
            / f"standard-submission-{primary_turns}.json",
            scheduler=scheduler,
            predictor=helpers._Predictor(),
            live_reader=helpers._live_scheduler_reader,
        )
        plan = diagnostic._load_plan(plan_path)[0]
        submission = diagnostic._load_submission(
            submission_path, plan=plan
        )
        result = helpers._result(plan_path, submission)
        entries.append(
            {
                "turns": primary_turns,
                "plan_path": plan_path,
                "submission_path": submission_path,
                "submission": submission,
                "result": result,
            }
        )
    by_volume = sorted(
        entries,
        key=lambda entry: geometry_metrics.bounding_box_lit(
            entry["result"]
        )[0],
    )
    for order, entry in enumerate(by_volume):
        winding = 30.0 - 8.0 * order
        entry["result"]["P_winding_total"] = winding
        entry["result"]["P_Tx_main_group"] = winding - 2.0
        entry["result"]["P_Rx_main_group"] = 1.0
        entry["result"]["P_Rx_side_total"] = 1.0
    if result_mutator is not None:
        for entry in entries:
            result_mutator(entry["result"])
    for entry in entries:
        scheduler.result = entry["result"]
        metadata_reader, manifest_reader = helpers._remote_evidence(
            entry["submission"], entry["result"]
        )
        entry["metadata_reader"] = metadata_reader
        entry["manifest_reader"] = manifest_reader
        entry["collection_path"] = diagnostic.collect_standard(
            plan_path=entry["plan_path"],
            submission_path=entry["submission_path"],
            output=tmp_path
            / f"standard-collection-{entry['turns']}.json",
            scheduler=scheduler,
            remote_reader=metadata_reader,
            manifest_reader=manifest_reader,
            task_reader=lambda **_kwargs: helpers._task_snapshot(
                entry["submission"]
            ),
        )
    return {
        "helpers": helpers,
        "fixture": fixture,
        "cutover_path": cutover_path,
        "entries": entries,
    }


def _first_full_plan(plan_set_path: Path) -> Path:
    value = production._read_json(plan_set_path)
    record = value["plans"][0]["plan"]
    path = plan_set_path.parent / record["path"]
    assert production._sha256_file(path) == record["sha256"]
    return path


def _full_result(plan_path: Path, submission, predictor, *, hot=False):
    plan, params, profile, view, _truth = promotion._load_full_plan(
        plan_path, predictor=predictor
    )
    runtime_license_sha = "8" * 64
    runtime_auth = production._core_auth(
        plan["solver_revision"],
        16,
        license_contract=production.FULL_LICENSE_CONTRACT,
        license_snapshot_sha256=runtime_license_sha,
    )
    return {
        **view["selected"]["row_contract"]["decoded_params"],
        **production._effective_params(params, profile),
        "git_hash": plan["solver_revision"],
        "git_dirty": 0,
        "pyaedt_library_git_hash": plan["library_revision"],
        "pyaedt_library_git_dirty": 0,
        "project_name": "truth-promoted-full-project",
        "Llt": 27.5,
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
        "solver_num_cores_requested": 16,
        "solver_num_cores_effective": 16,
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": 16,
        "solver_core_slurm_cpus_per_task_readback": "16",
        "solver_core_scheduler_task_id_readback": str(
            submission["task_id"]
        ),
        "solver_core_slurm_job_id_readback": "91234",
        "solver_core_auth_sha256": runtime_auth,
        "solver_matrix_hpc_num_cores_readback": 16,
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "9" * 64,
        "solver_core_license_contract": production.FULL_LICENSE_CONTRACT,
        "solver_core_license_snapshot_sha256": runtime_license_sha,
        "solver_core_license_checked_at_readback": (
            "2026-07-25T00:00:00+00:00"
        ),
        "solver_core_license_snapshot_age_seconds_readback": 1.0,
        "solver_core_license_headroom_readback_json": json.dumps(
            {
                "anshpc": 48,
                "elec_solve_maxwell": 3,
                "electronics_desktop": 3,
                "electronics3d_gui": 3,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        **{
            target: (
                101.0
                if hot and target == "T_max_Tx"
                else 99.0
                if "core" not in target
                else 119.0
            )
            for target in GOAL_TEMPERATURE_TARGETS
        },
    }


def _full_remote_evidence(submission, result):
    retained = submission["retained_aedt_bundle"]
    artifact = b"full-aedt"
    marker_payload = {
        **retained["marker_contract"],
        "created_at": "2026-07-25T00:00:00Z",
    }
    marker = production._json_bytes(marker_payload)
    state = b"full-state"
    files = [
        {
            "path": "icepak_thermal.results/state.bin",
            "sha256": hashlib.sha256(state).hexdigest(),
            "size_bytes": len(state),
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
        "schema_version": (
            scheduler_client.RETAINED_AEDT_TRUTH_FULL_RESULTS_MANIFEST_SCHEMA
        ),
        "source_project_name": result["project_name"],
        "source_results_directory_name": (
            f"{result['project_name']}.aedtresults"
        ),
        "retained_results_directory_name": PurePosixPath(
            retained["results_path"]
        ).name,
        "file_count": 1,
        "size_bytes": len(state),
        "tree_sha256": tree_sha,
        "files": files,
    }
    manifest_bytes = production._json_bytes(manifest)
    receipt = {
        "schema_version": (
            scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_RECEIPT_SCHEMA
        ),
        "stage": "full",
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
        "results_manifest_schema_version": (
            scheduler_client.RETAINED_AEDT_TRUTH_FULL_RESULTS_MANIFEST_SCHEMA
        ),
        "results_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "results_tree_sha256": tree_sha,
        "results_file_count": 1,
        "results_size_bytes": len(state),
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

    return {
        "artifact": artifact,
        "state": state,
        "receipt": receipt,
        "manifest": manifest,
        "metadata_reader": metadata_reader,
        "manifest_reader": manifest_reader,
    }


def _full_task_snapshot(submission):
    return {
        "task_id": submission["task_id"],
        "name": submission["task_name"],
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "failure_message": "",
        "slurm_job_id": "91234",
        "allocation_id": 9901,
        "account_name": "test-account",
        "actual_node_name": "n201",
        "cpus": 16,
        "memory_mb": 98304,
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": submission["dedupe_key"],
        "remote_cwd": "/gpfs/test-full",
        "remote_dir": "runs/task-full",
        "finished_at": "2026-07-25 00:00:00",
    }


def _combined_reader(standard_entry, full_evidence, *, manifest=False):
    standard_reader = (
        standard_entry["manifest_reader"]
        if manifest
        else standard_entry["metadata_reader"]
    )
    full_reader = (
        full_evidence["manifest_reader"]
        if manifest
        else full_evidence["metadata_reader"]
    )

    def read(**kwargs):
        if kwargs["task_id"] == standard_entry["submission"]["task_id"]:
            return standard_reader(**kwargs)
        return full_reader(**kwargs)

    return read


def test_combined_actual_truth_nds_deduplicates_and_bounds_full_plans(
    tmp_path, monkeypatch
):
    built = _standard_collections(tmp_path, monkeypatch)
    duplicate = tmp_path / "duplicate-diagnostic-collection.json"
    duplicate.write_bytes(
        built["entries"][0]["collection_path"].read_bytes()
    )
    collections = [
        entry["collection_path"] for entry in built["entries"]
    ] + [duplicate]
    manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=collections,
        output=tmp_path / "truth-promotion",
        predictor=built["helpers"]._Predictor(),
    )
    manifest, rows, _by_candidate = promotion._load_truth_manifest(
        manifest_path, predictor=built["helpers"]._Predictor()
    )
    assert manifest["input_collection_count"] == 4
    assert manifest["deduplicated_candidate_count"] == 3
    assert manifest["rank0_count"] == 3
    assert [row["truth_non_dominated_rank"] for row in rows] == [0, 0, 0]
    assert max(row["duplicate_collection_count"] for row in rows) == 2
    assert manifest["surrogate_authority_used"] is False
    assert manifest["surrogate_constraint_relaxed"] is False
    assert manifest["actual_truth_only_promotion"] is True
    assert (
        manifest_path.parent / "truth_validated_pareto_front.csv"
    ).is_file()

    plan_set_path = promotion.create_full_plans(
        truth_manifest_path=manifest_path,
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "full-plans",
        predictor=built["helpers"]._Predictor(),
    )
    plan_set, loaded_plans = promotion._load_full_plan_set(
        plan_set_path, predictor=built["helpers"]._Predictor()
    )
    assert plan_set["selected_plan_count"] == 3
    assert len(loaded_plans) == 3
    assert plan_set["rank0_only"] is True
    assert plan_set["automatic_promotion"] is False
    assert all(
        item["truth_non_dominated_rank"] == 0
        for item in plan_set["plans"]
    )
    with pytest.raises(
        production.HandoffContractError, match="1..3"
    ):
        promotion.create_full_plans(
            truth_manifest_path=manifest_path,
            solver_revision="2" * 40,
            library_revision="3" * 40,
            limit=4,
            output=tmp_path / "forbidden-four-plans",
            predictor=built["helpers"]._Predictor(),
        )
    with pytest.raises(SystemExit):
        diagnostic._parser().parse_args(["submit-full"])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda result: result.__setitem__(
            "f_res_min_tx_rx_only_Hz", 14999.9
        ),
        lambda result: result.__setitem__(
            "Tprobe_Tx_leeward_max", 100.1
        ),
    ],
    ids=["resonance-threshold", "active-probe-temperature"],
)
def test_promotion_rejects_failed_actual_hard_constraint(
    tmp_path, monkeypatch, mutator
):
    built = _standard_collections(
        tmp_path,
        monkeypatch,
        turns=(6,),
        result_mutator=mutator,
    )
    collection = production._read_json(
        built["entries"][0]["collection_path"]
    )
    assert (
        collection["goal_physical_spec_passed"] is False
        or collection["actual_body_probe_temperature_gate_passed"]
        is False
    )
    with pytest.raises(
        production.HandoffContractError,
        match="only passing diagnostic Standard",
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=[
                built["entries"][0]["collection_path"]
            ],
            output=tmp_path / "forbidden-promotion",
            predictor=built["helpers"]._Predictor(),
        )


def test_full_submit_collect_and_self_contained_package(
    tmp_path, monkeypatch
):
    built = _standard_collections(
        tmp_path, monkeypatch, turns=(6,)
    )
    predictor = built["helpers"]._Predictor()
    standard_entry = built["entries"][0]
    manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=[standard_entry["collection_path"]],
        output=tmp_path / "truth-promotion",
        predictor=predictor,
    )
    plan_set_path = promotion.create_full_plans(
        truth_manifest_path=manifest_path,
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "full-plans",
        predictor=predictor,
    )
    full_plan_path = _first_full_plan(plan_set_path)
    production_helpers = _production_helpers()
    license_path = production_helpers._license_snapshot(
        tmp_path / "license.json"
    )
    scheduler = _Scheduler(first_task_id=74001)
    submission_path = promotion.submit_full(
        plan_path=full_plan_path,
        scheduler_cutover_receipt_path=built["cutover_path"],
        license_snapshot_path=license_path,
        output=tmp_path / "full-submission.json",
        scheduler=scheduler,
        predictor=predictor,
        live_reader=built["helpers"]._live_scheduler_reader,
    )
    plan, _params, _profile, _view, standard_truth = (
        promotion._load_full_plan(
            full_plan_path, predictor=predictor
        )
    )
    submission = promotion._load_full_submission(
        submission_path,
        plan=plan,
        standard_truth=standard_truth,
    )
    assert len(scheduler.calls) == 1
    call = scheduler.calls[0][1]
    submitted_profile = scheduler.calls[0][0][3]
    assert call["cpus"] == 16
    assert call["scheduler_url"] == "http://127.0.0.1:8002"
    assert (
        call["required_project_cap"]
        == diagnostic.GOAL_FEA_PROJECT_CAP
    )
    assert (
        call["max_project_active_tasks"]
        == diagnostic.GOAL_FEA_PROJECT_CAP
    )
    assert submitted_profile["param_overrides"]["full_model"] == 1
    assert (
        submitted_profile["param_overrides"]["thermal_symmetry"] == "full"
    )

    valid_result = _full_result(
        full_plan_path, submission, predictor
    )
    scheduler.result = valid_result
    full_evidence = _full_remote_evidence(
        submission, scheduler.result
    )
    full_collection_path = promotion.collect_full(
        plan_path=full_plan_path,
        submission_path=submission_path,
        output=tmp_path / "full-collection.json",
        scheduler=scheduler,
        remote_reader=_combined_reader(
            standard_entry, full_evidence
        ),
        manifest_reader=_combined_reader(
            standard_entry, full_evidence, manifest=True
        ),
        task_reader=lambda **_kwargs: _full_task_snapshot(submission),
        predictor=predictor,
    )
    authenticated = promotion.authenticate_full_collection(
        full_collection_path, predictor=predictor
    )
    full_collection = authenticated["collection"]
    assert full_collection["goal_physical_spec_passed"] is True
    assert full_collection["result_identity"]["full_model"] == 1
    assert (
        full_collection["result_identity"]["thermal_symmetry"] == "full"
    )
    assert (
        full_collection["actual_truth_evidence"][
            "fixed_identity_attestation"
        ]
        == standard_truth["fixed_identity_attestation"]
    )
    mixed_unsigned = copy.deepcopy(full_collection)
    mixed_unsigned.pop("payload_sha256")
    mixed_unsigned["candidate_physics_sha256"] = "f" * 64
    mixed_path = production._write_immutable_json(
        tmp_path / "mixed-candidate-full-collection.json",
        production._seal(mixed_unsigned),
    )
    with pytest.raises(
        production.HandoffContractError,
        match="evidence drifted",
    ):
        promotion.authenticate_full_collection(
            mixed_path, predictor=predictor
        )

    scheduler.result = _full_result(
        full_plan_path, submission, predictor, hot=True
    )
    hot_evidence = _full_remote_evidence(
        submission, scheduler.result
    )
    hot_collection_path = promotion.collect_full(
        plan_path=full_plan_path,
        submission_path=submission_path,
        output=tmp_path / "hot-full-collection.json",
        scheduler=scheduler,
        remote_reader=_combined_reader(
            standard_entry, hot_evidence
        ),
        manifest_reader=_combined_reader(
            standard_entry, hot_evidence, manifest=True
        ),
        task_reader=lambda **_kwargs: _full_task_snapshot(submission),
        predictor=predictor,
    )
    hot_collection = promotion.authenticate_full_collection(
        hot_collection_path, predictor=predictor
    )["collection"]
    assert hot_collection["goal_physical_spec_passed"] is False
    with pytest.raises(
        production.HandoffContractError,
        match="release gates",
    ):
        promotion.package_results(
            full_collection_path=hot_collection_path,
            output=tmp_path / "forbidden-hot-package",
            remote_fetcher=lambda **_kwargs: pytest.fail(
                "hot Full must block before AEDT GET"
            ),
            result_fetcher=lambda **_kwargs: pytest.fail(
                "hot Full must block before results GET"
            ),
            predictor=predictor,
        )

    artifact_bytes = {
        hashlib.sha256(b"diagnostic-aedt").hexdigest(): (
            b"diagnostic-aedt"
        ),
        hashlib.sha256(b"full-aedt").hexdigest(): b"full-aedt",
    }
    result_bytes = {
        hashlib.sha256(b"state").hexdigest(): b"state",
        hashlib.sha256(b"full-state").hexdigest(): b"full-state",
    }

    def artifact_fetcher(**kwargs):
        kwargs["destination"].write_bytes(
            artifact_bytes[kwargs["expected_sha256"]]
        )

    def result_fetcher(**kwargs):
        kwargs["destination"].parent.mkdir(
            parents=True, exist_ok=True
        )
        kwargs["destination"].write_bytes(
            result_bytes[kwargs["expected_sha256"]]
        )

    package_path = promotion.package_results(
        full_collection_path=full_collection_path,
        output=tmp_path / "truth-package",
        remote_fetcher=artifact_fetcher,
        result_fetcher=result_fetcher,
        predictor=predictor,
    )
    package = promotion.authenticate_package(
        package_path, predictor=predictor
    )
    assert Path(
        package["aedt_artifacts"]["symmetric"][
            "local_absolute_path"
        ]
    ).name == "symmetric_model.aedt"
    assert Path(
        package["aedt_artifacts"]["full"]["local_absolute_path"]
    ).name == "full_model.aedt"
    assert (
        package["aedtresults_trees"]["symmetric"]["tree_sha256"]
        == production._read_json(standard_entry["collection_path"])[
            "remote_aedt_bundle_receipt"
        ]["results_tree_sha256"]
    )
    assert (
        package["aedtresults_trees"]["full"]["tree_sha256"]
        == full_evidence["receipt"]["results_tree_sha256"]
    )
    assert (
        package["fixed_cooling_identity"]["fan_velocity_m_s"] == 1.5
    )
    assert (
        package["fixed_cooling_identity"][
            "thermal_pad_conductivity_W_mK"
        ]
        == 0.2
    )
    state_path = (
        package_path.parent
        / "full_model.aedtresults"
        / "icepak_thermal.results"
        / "state.bin"
    )
    state_path.chmod(0o666)
    state_path.write_bytes(b"tampered")
    with pytest.raises(
        production.HandoffContractError,
        match="member bytes|identity",
    ):
        promotion.authenticate_package(package_path, predictor=predictor)


def test_full_submit_fail_closed_before_scheduler_post(
    tmp_path, monkeypatch
):
    built = _standard_collections(
        tmp_path, monkeypatch, turns=(6,)
    )
    predictor = built["helpers"]._Predictor()
    manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=[
            built["entries"][0]["collection_path"]
        ],
        output=tmp_path / "truth-promotion",
        predictor=predictor,
    )
    plans = promotion.create_full_plans(
        truth_manifest_path=manifest_path,
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "full-plans",
        predictor=predictor,
    )
    plan_path = _first_full_plan(plans)
    license_path = _production_helpers()._license_snapshot(
        tmp_path / "license.json"
    )

    def blocked_reader(**kwargs):
        if kwargs["endpoint"] == "/api/health":
            return {
                "ok": False,
                "scheduler_ok": False,
                "scheduler_thread_alive": False,
                "scheduler_stalled": True,
                "consecutive_tick_failures": 1,
            }
        return built["helpers"]._live_scheduler_reader(**kwargs)

    scheduler = _Scheduler()
    with pytest.raises(
        production.HandoffContractError, match="health"
    ):
        promotion.submit_full(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=built["cutover_path"],
            license_snapshot_path=license_path,
            output=tmp_path / "forbidden-health.json",
            scheduler=scheduler,
            predictor=predictor,
            live_reader=blocked_reader,
        )
    assert scheduler.calls == []

    stale = production._read_json(license_path)
    stale["checked_at"] = "2020-01-01T00:00:00+00:00"
    license_path.write_text(
        json.dumps(stale, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(
        production.HandoffContractError, match="stale"
    ):
        promotion.submit_full(
            plan_path=plan_path,
            scheduler_cutover_receipt_path=built["cutover_path"],
            license_snapshot_path=license_path,
            output=tmp_path / "forbidden-license.json",
            scheduler=scheduler,
            predictor=predictor,
            live_reader=built["helpers"]._live_scheduler_reader,
        )
    assert scheduler.calls == []


def test_full_retention_export_is_separate_from_diagnostic_standard(
    tmp_path,
):
    profile, _source = promotion._full_profile()
    retained = scheduler_client.retained_aedt_identity(
        "truth-full-export-test",
        {"x": 1},
        profile,
        "2" * 40,
        "3" * 40,
    )
    assert (
        retained["schema_version"]
        == scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_SCHEMA
    )
    assert retained["stage"] == "full"
    diagnostic_profile, _record = diagnostic._profile_content()
    diagnostic_retained = scheduler_client.retained_aedt_identity(
        "diagnostic-standard-export-test",
        {"x": 1},
        diagnostic_profile,
        "2" * 40,
        "3" * 40,
    )
    assert (
        diagnostic_retained["schema_version"]
        == scheduler_client.RETAINED_AEDT_BUNDLE_SCHEMA
    )
    assert diagnostic_retained["stage"] == "standard"

    command = scheduler_client._retained_aedt_export_command(retained)
    tokens = shlex.split(command)
    script = tokens[tokens.index("-c") + 1]
    marker_contract_json = tokens[tokens.index("-c") + 9]
    receipt_context_json = tokens[tokens.index("-c") + 10]
    source = tmp_path / "mock-full-project.aedt"
    source.write_bytes(b"full-project")
    source_results = tmp_path / "mock-full-project.aedtresults"
    nested = source_results / "icepak_thermal.results"
    nested.mkdir(parents=True)
    (nested / "state.bin").write_bytes(b"full-state")
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
    receipt = json.loads(
        (tmp_path / retained["receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (tmp_path / retained["results_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert (
        receipt["schema_version"]
        == scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_RECEIPT_SCHEMA
    )
    assert (
        manifest["schema_version"]
        == scheduler_client.RETAINED_AEDT_TRUTH_FULL_RESULTS_MANIFEST_SCHEMA
    )
    assert receipt["results_tree_sha256"] == manifest["tree_sha256"]
