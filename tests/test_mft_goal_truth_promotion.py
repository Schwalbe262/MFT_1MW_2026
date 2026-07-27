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
from tools import mft_campaign_atomic_claim as atomic_claim
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
        guard = kwargs.get("pre_submit_guard")
        if guard is not None:
            guard()
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
    claim_root = (tmp_path / "operational-pressure-claims").resolve()
    monkeypatch.setattr(
        diagnostic, "OPERATIONAL_PRESSURE_CLAIM_ROOT", claim_root
    )
    atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=(
            diagnostic.OPERATIONAL_PRESSURE_CLAIM_AUTHORITY_SHA256
        ),
        root_id="3" * 32,
        now="2026-07-25T00:00:00Z",
    )
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
        result.update(
            {
                "thermal_rx_block_interface_contract_version": (
                    "thermal-rx-block-interface-coverage-v1"
                ),
                "thermal_rx_main_interface_coverage_passed": True,
                "thermal_rx_main_unpaired_interfaces": [],
                "thermal_temperature_limiter_triggered": False,
                "thermal_temperature_limiter_max_K": 400.0,
                "thermal_result_scientific_valid": True,
            }
        )
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


def _standard_cohort_24(
    tmp_path: Path,
    monkeypatch,
    *,
    counts=(7, 6, 6, 5),
    result_mutator=None,
):
    if len(counts) != 4 or sum(counts) != promotion.EXACT_COHORT_SIZE:
        raise AssertionError(counts)
    built = _standard_collections(
        tmp_path,
        monkeypatch,
        turns=(5, 6, 7, 8),
        result_mutator=result_mutator,
    )
    helpers = built["helpers"]
    predictor = helpers._Predictor()
    entries = list(built["entries"])
    base_by_turns = {entry["turns"]: entry for entry in entries}
    next_task_id = (
        max(entry["submission"]["task_id"] for entry in entries) + 1
    )
    scheduler = _Scheduler(first_task_id=next_task_id)
    for turns, count in zip((5, 6, 7, 8), counts, strict=True):
        base = base_by_turns[turns]
        for repeat in range(1, count):
            submission_path = diagnostic.submit_standard(
                plan_path=base["plan_path"],
                scheduler_cutover_receipt_path=built["cutover_path"],
                output=(
                    tmp_path
                    / f"standard-submission-{turns}-repeat-{repeat}.json"
                ),
                scheduler=scheduler,
                predictor=predictor,
                live_reader=helpers._live_scheduler_reader,
            )
            plan = diagnostic._load_plan(base["plan_path"])[0]
            submission = diagnostic._load_submission(
                submission_path, plan=plan
            )
            result = copy.deepcopy(base["result"])
            result["solver_core_scheduler_task_id_readback"] = str(
                submission["task_id"]
            )
            scheduler.result = result
            metadata_reader, manifest_reader = helpers._remote_evidence(
                submission, result
            )
            collection_path = diagnostic.collect_standard(
                plan_path=base["plan_path"],
                submission_path=submission_path,
                output=(
                    tmp_path
                    / f"standard-collection-{turns}-repeat-{repeat}.json"
                ),
                scheduler=scheduler,
                remote_reader=metadata_reader,
                manifest_reader=manifest_reader,
                task_reader=lambda _submission=submission, **_kwargs: (
                    helpers._task_snapshot(_submission)
                ),
            )
            entries.append(
                {
                    "turns": turns,
                    "plan_path": base["plan_path"],
                    "submission_path": submission_path,
                    "submission": submission,
                    "result": result,
                    "metadata_reader": metadata_reader,
                    "manifest_reader": manifest_reader,
                    "collection_path": collection_path,
                }
            )
    inventory_path = promotion.create_cohort_inventory(
        standard_submission_paths=[
            entry["submission_path"] for entry in entries
        ],
        output=tmp_path / "standard-cohort-inventory.json",
        predictor=predictor,
    )
    return {
        **built,
        "entries": entries,
        "inventory_path": inventory_path,
        "predictor": predictor,
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
        lambda result: result.__setitem__("T_max_Tx", 100.1),
        lambda result: result.__setitem__("T_max_core", 120.1),
    ],
    ids=[
        "resonance-threshold",
        "active-probe-temperature",
        "winding-body-temperature",
        "core-body-temperature",
    ],
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


def test_promotion_rejects_non_scientific_thermal_result(
    tmp_path, monkeypatch
):
    def invalidate(result):
        result["thermal_rx_main_interface_coverage_passed"] = False
        result["thermal_rx_main_unpaired_interfaces"] = ["153"]
        result["thermal_result_scientific_valid"] = False

    built = _standard_collections(
        tmp_path,
        monkeypatch,
        turns=(6,),
        result_mutator=invalidate,
    )
    with pytest.raises(
        production.HandoffContractError,
        match="thermal scientific truth contract",
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=[
                built["entries"][0]["collection_path"]
            ],
            output=tmp_path / "forbidden-scientific-promotion",
            predictor=built["helpers"]._Predictor(),
        )


def test_truth_v2_classifier_records_size_failure(
    tmp_path, monkeypatch
):
    from regression_260707.verify import finalize

    built = _standard_collections(
        tmp_path, monkeypatch, turns=(6,)
    )
    predictor = built["helpers"]._Predictor()
    collection_path = built["entries"][0]["collection_path"]
    view = diagnostic.authenticate_collection(
        collection_path, predictor=predictor
    )
    _original_volume, original_dimensions = (
        geometry_metrics.bounding_box_lit(view["collection"]["result"])
    )
    dimensions = (
        1200.1,
        float(original_dimensions[1]),
        float(original_dimensions[2]),
    )
    volume = dimensions[0] * dimensions[1] * dimensions[2] * 1e-6

    def oversized(_result):
        return volume, dimensions

    with monkeypatch.context() as context:
        context.setattr(
            promotion.geometry_metrics, "bounding_box_lit", oversized
        )
        context.setattr(finalize, "bounding_box_lit", oversized)
        collection = view["collection"]
        reasons = production._goal_result_reasons(
            collection["result"], view["selected"]
        )
        assert "size_out_of_spec:W" in reasons
        collection["goal_physical_spec_reasons"] = reasons
        collection["goal_physical_spec_passed"] = False
        collection["truth_evidence"]["actual_volume_L"] = volume
        collection["truth_evidence"][
            "actual_exterior_dimensions_mm"
        ] = {
            "W": dimensions[0],
            "L": dimensions[1],
            "H": dimensions[2],
        }
        _truth, status = promotion._actual_standard_observation(
            view, collection_path=collection_path
        )
    assert status["actual_truth_feasible"] is False
    assert "size_out_of_spec:W" in status[
        "promotion_exclusion_reasons"
    ]


def test_truth_v2_exact_24_classifies_every_collection_and_blocks_zero(
    tmp_path, monkeypatch
):
    def mixed_feasibility(result):
        primary_turns = int(float(result["N1_main"])) + int(
            float(result["N1_side"])
        )
        if primary_turns == 7:
            result["f_res_min_tx_rx_only_Hz"] = 14999.9
        if primary_turns == 8:
            result["Tprobe_Tx_leeward_max"] = 100.1

    built = _standard_cohort_24(
        tmp_path,
        monkeypatch,
        counts=(7, 6, 6, 5),
        result_mutator=mixed_feasibility,
    )
    helpers = built["helpers"]
    predictor = built["predictor"]
    collection_paths = [
        entry["collection_path"] for entry in built["entries"]
    ]
    submission_paths = [
        entry["submission_path"] for entry in built["entries"]
    ]
    inventory, cohort_entries = promotion._load_cohort_inventory(
        built["inventory_path"], predictor=predictor
    )
    assert inventory["entry_count"] == 24
    assert inventory["unique_candidate_count"] == 4
    assert len({entry["task_id"] for entry in cohort_entries}) == 24
    with pytest.raises(
        production.HandoffContractError, match="exactly 24"
    ):
        promotion.create_cohort_inventory(
            standard_submission_paths=submission_paths[:23],
            output=tmp_path / "forbidden-cohort-23.json",
            predictor=predictor,
        )
    with pytest.raises(
        production.HandoffContractError, match="exactly 24"
    ):
        promotion.create_cohort_inventory(
            standard_submission_paths=(
                submission_paths + [submission_paths[0]]
            ),
            output=tmp_path / "forbidden-cohort-25.json",
            predictor=predictor,
        )
    with pytest.raises(
        production.HandoffContractError, match="must be unique"
    ):
        promotion.create_cohort_inventory(
            standard_submission_paths=(
                submission_paths[:23] + [submission_paths[0]]
            ),
            output=tmp_path / "forbidden-cohort-duplicate.json",
            predictor=predictor,
        )
    with monkeypatch.context() as context:
        context.setattr(
            diagnostic,
            "authenticate_collection",
            lambda *_args, **_kwargs: {
                "collection": {"task_id": 999999}
            },
        )
        with pytest.raises(
            production.HandoffContractError,
            match="outside or duplicates",
        ):
            promotion._authenticate_cohort_inputs(
                cohort_entries=cohort_entries,
                collection_paths=collection_paths,
                predictor=predictor,
            )

    with pytest.raises(
        production.HandoffContractError, match="exactly 24"
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=collection_paths[:23],
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-missing-v2",
            predictor=predictor,
        )
    with pytest.raises(
        production.HandoffContractError, match="must be unique"
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=(
                collection_paths[:23] + [collection_paths[0]]
            ),
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-duplicate-v2",
            predictor=predictor,
        )
    with pytest.raises(
        production.HandoffContractError, match="exactly 24"
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=(
                collection_paths + [collection_paths[0]]
            ),
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-extra-v2",
            predictor=predictor,
        )

    manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=list(reversed(collection_paths)),
        cohort_inventory_path=built["inventory_path"],
        output=tmp_path / "truth-promotion-v2",
        predictor=predictor,
    )
    manifest, ranked, _by_candidate = promotion._load_truth_manifest(
        manifest_path, predictor=predictor
    )
    assert manifest["schema_version"] == promotion.TRUTH_MANIFEST_SCHEMA_V2
    assert manifest["authenticated_collection_count"] == 24
    assert manifest["feasible_collection_count"] == 13
    assert manifest["infeasible_collection_count"] == 11
    assert len(manifest["classification_rows"]) == 24
    assert len(ranked) == 2
    assert all(row["truth_non_dominated_rank"] == 0 for row in ranked)
    assert {
        row["task_id"]
        for row in manifest["classification_rows"]
        if row["actual_truth_feasible"]
    } == {
        entry["submission"]["task_id"]
        for entry in built["entries"]
        if entry["turns"] in {5, 6}
    }
    assert any(
        "self_resonance_below_minimum"
        in row["promotion_exclusion_reasons"]
        for row in manifest["classification_rows"]
    )
    assert any(
        "temperature_out_of_spec:Tprobe_Tx_leeward_max"
        in row["promotion_exclusion_reasons"]
        for row in manifest["classification_rows"]
    )

    retry_base = built["entries"][-1]
    retry_plan_path = diagnostic.create_timeout_retry_plan(
        original_plan_path=retry_base["plan_path"],
        original_submission_path=retry_base["submission_path"],
        output=tmp_path / "cohort-timeout-retry-plan",
        task_reader=lambda **_kwargs: helpers._timeout_task_snapshot(
            retry_base["submission"]
        ),
    )
    retry_scheduler = _Scheduler(
        first_task_id=max(
            entry["submission"]["task_id"] for entry in built["entries"]
        )
        + 100
    )
    retry_submission_path = diagnostic.submit_timeout_retry(
        plan_path=retry_plan_path,
        scheduler_cutover_receipt_path=built["cutover_path"],
        output=tmp_path / "cohort-timeout-retry-submission.json",
        scheduler=retry_scheduler,
        predictor=predictor,
        live_reader=helpers._live_scheduler_reader,
        task_reader=lambda **_kwargs: helpers._timeout_task_snapshot(
            retry_base["submission"]
        ),
    )
    retry_plan = diagnostic._load_plan(retry_plan_path)[0]
    retry_submission = diagnostic._load_submission(
        retry_submission_path, plan=retry_plan
    )
    retry_result = copy.deepcopy(retry_base["result"])
    retry_result["solver_core_scheduler_task_id_readback"] = str(
        retry_submission["task_id"]
    )
    retry_scheduler.result = retry_result
    retry_metadata_reader, retry_manifest_reader = (
        helpers._remote_evidence(retry_submission, retry_result)
    )
    retry_collection_path = diagnostic.collect_standard(
        plan_path=retry_plan_path,
        submission_path=retry_submission_path,
        output=tmp_path / "cohort-timeout-retry-collection.json",
        scheduler=retry_scheduler,
        remote_reader=retry_metadata_reader,
        manifest_reader=retry_manifest_reader,
        task_reader=lambda **_kwargs: helpers._task_snapshot(
            retry_submission
        ),
    )
    retry_collection_paths = [
        *collection_paths[:-1],
        retry_collection_path,
    ]
    retry_manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=retry_collection_paths,
        cohort_inventory_path=built["inventory_path"],
        output=tmp_path / "truth-promotion-v2-timeout-retry",
        predictor=predictor,
    )
    retry_manifest, retry_ranked, _retry_by_candidate = (
        promotion._load_truth_manifest(
            retry_manifest_path, predictor=predictor
        )
    )
    retry_entry = next(
        entry
        for entry in cohort_entries
        if entry["task_id"] == retry_base["submission"]["task_id"]
    )
    retry_classification = next(
        row
        for row in retry_manifest["classification_rows"]
        if row["cohort_entry_sha256"] == retry_entry["entry_sha256"]
    )
    assert retry_classification["task_id"] == retry_submission["task_id"]
    assert [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in retry_ranked
    ] == [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in ranked
    ]
    with pytest.raises(
        production.HandoffContractError,
        match="outside or duplicates",
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=[
                retry_collection_path,
                *collection_paths[1:],
            ],
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-duplicate-timeout-retry-v2",
            predictor=predictor,
        )

    pressure_plan_path = (
        diagnostic.create_operational_pressure_retry_plan(
            original_plan_path=retry_base["plan_path"],
            original_submission_path=retry_base["submission_path"],
            output=tmp_path / "cohort-pressure-retry-plan",
            task_reader=lambda **_kwargs: (
                helpers._operational_pressure_task_snapshot(
                    retry_base["submission"]
                )
            ),
            event_reader=lambda **_kwargs: (
                helpers._operational_pressure_events(
                    retry_base["submission"]
                )
            ),
        )
    )
    pressure_plan = diagnostic._load_plan(pressure_plan_path)[0]
    pressure_scheduler = _Scheduler(
        first_task_id=max(
            entry["submission"]["task_id"] for entry in built["entries"]
        )
        + 200
    )
    pressure_task_id = pressure_scheduler.next_task_id
    pressure_sibling = {
        "task_id": pressure_task_id,
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
    pressure_submission_path = (
        diagnostic.submit_operational_pressure_retry(
            plan_path=pressure_plan_path,
            scheduler_cutover_receipt_path=built["cutover_path"],
            output=tmp_path / "cohort-pressure-retry-submission.json",
            scheduler=pressure_scheduler,
            predictor=predictor,
            live_reader=helpers._live_scheduler_reader,
            task_reader=lambda **_kwargs: (
                helpers._operational_pressure_task_snapshot(
                    retry_base["submission"]
                )
            ),
            event_reader=lambda **_kwargs: (
                helpers._operational_pressure_events(
                    retry_base["submission"]
                )
            ),
            task_list_reader=lambda **_kwargs: next(
                pressure_sibling_rows
            ),
        )
    )
    pressure_submission = diagnostic._load_submission(
        pressure_submission_path, plan=pressure_plan
    )
    pressure_result = copy.deepcopy(retry_base["result"])
    pressure_result["solver_core_scheduler_task_id_readback"] = str(
        pressure_submission["task_id"]
    )
    pressure_scheduler.result = pressure_result
    pressure_metadata_reader, pressure_manifest_reader = (
        helpers._remote_evidence(pressure_submission, pressure_result)
    )
    pressure_collection_path = diagnostic.collect_standard(
        plan_path=pressure_plan_path,
        submission_path=pressure_submission_path,
        output=tmp_path / "cohort-pressure-retry-collection.json",
        scheduler=pressure_scheduler,
        remote_reader=pressure_metadata_reader,
        manifest_reader=pressure_manifest_reader,
        task_reader=lambda **_kwargs: helpers._task_snapshot(
            pressure_submission
        ),
    )
    pressure_manifest_path = promotion.create_truth_promotion(
        standard_collection_paths=[
            *collection_paths[:-1],
            pressure_collection_path,
        ],
        cohort_inventory_path=built["inventory_path"],
        output=tmp_path / "truth-promotion-v2-pressure-retry",
        predictor=predictor,
    )
    _pressure_manifest, pressure_ranked, _pressure_by_candidate = (
        promotion._load_truth_manifest(
            pressure_manifest_path, predictor=predictor
        )
    )
    assert [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in pressure_ranked
    ] == [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in ranked
    ]
    with pytest.raises(
        production.HandoffContractError,
        match="outside or duplicates",
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=[
                retry_collection_path,
                pressure_collection_path,
                *collection_paths[2:],
            ],
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-mixed-retries-v2",
            predictor=predictor,
        )

    # Direct and compound operational-pressure retries intentionally share
    # one logical claim slot and cannot both win in a campaign.  Exercise
    # compound truth routing under an isolated campaign-root authority.
    compound_claim_root = (
        tmp_path / "compound-operational-pressure-claims"
    ).resolve()
    monkeypatch.setattr(
        diagnostic,
        "OPERATIONAL_PRESSURE_CLAIM_ROOT",
        compound_claim_root,
    )
    atomic_claim.initialize_claim_root(
        compound_claim_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=(
            diagnostic.OPERATIONAL_PRESSURE_CLAIM_AUTHORITY_SHA256
        ),
        root_id="4" * 32,
        now="2026-07-25T00:00:00Z",
    )
    compound_plan_path = (
        diagnostic.create_operational_pressure_retry_plan(
            original_plan_path=retry_plan_path,
            original_submission_path=retry_submission_path,
            output=tmp_path / "cohort-compound-pressure-plan",
            task_reader=lambda **_kwargs: (
                helpers._operational_pressure_task_snapshot(
                    retry_submission,
                    timeout_seconds=28800,
                )
            ),
            event_reader=lambda **_kwargs: (
                helpers._operational_pressure_events(retry_submission)
            ),
        )
    )
    compound_plan = diagnostic._load_plan(compound_plan_path)[0]
    assert compound_plan["retry_of_operational_pressure"][
        "logical_authority_task_id"
    ] == retry_base["submission"]["task_id"]
    assert compound_plan["retry_of_operational_pressure"][
        "retry_of_task_id"
    ] == retry_submission["task_id"]
    assert compound_plan["stage"]["resources"]["timeout_seconds"] == 28800
    compound_scheduler = _Scheduler(
        first_task_id=max(
            entry["submission"]["task_id"] for entry in built["entries"]
        )
        + 300
    )
    compound_task_id = compound_scheduler.next_task_id
    compound_sibling = {
        "task_id": compound_task_id,
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
    compound_sibling_rows = iter(([], [], [compound_sibling]))
    compound_submission_path = (
        diagnostic.submit_operational_pressure_retry(
            plan_path=compound_plan_path,
            scheduler_cutover_receipt_path=built["cutover_path"],
            output=tmp_path / "cohort-compound-pressure-submission.json",
            scheduler=compound_scheduler,
            predictor=predictor,
            live_reader=helpers._live_scheduler_reader,
            task_reader=lambda **_kwargs: (
                helpers._operational_pressure_task_snapshot(
                    retry_submission,
                    timeout_seconds=28800,
                )
            ),
            event_reader=lambda **_kwargs: (
                helpers._operational_pressure_events(retry_submission)
            ),
            task_list_reader=lambda **_kwargs: next(
                compound_sibling_rows
            ),
        )
    )
    compound_submission = diagnostic._load_submission(
        compound_submission_path, plan=compound_plan
    )
    compound_result = copy.deepcopy(retry_base["result"])
    compound_result["solver_core_scheduler_task_id_readback"] = str(
        compound_submission["task_id"]
    )
    compound_scheduler.result = compound_result
    compound_metadata, compound_results_manifest = (
        helpers._remote_evidence(compound_submission, compound_result)
    )
    compound_collection_path = diagnostic.collect_standard(
        plan_path=compound_plan_path,
        submission_path=compound_submission_path,
        output=tmp_path / "cohort-compound-pressure-collection.json",
        scheduler=compound_scheduler,
        remote_reader=compound_metadata,
        manifest_reader=compound_results_manifest,
        task_reader=lambda **_kwargs: helpers._task_snapshot(
            compound_submission
        ),
    )
    compound_truth_path = promotion.create_truth_promotion(
        standard_collection_paths=[
            *collection_paths[:-1],
            compound_collection_path,
        ],
        cohort_inventory_path=built["inventory_path"],
        output=tmp_path / "truth-promotion-v2-compound-pressure",
        predictor=predictor,
    )
    compound_manifest, compound_ranked, _compound_by_candidate = (
        promotion._load_truth_manifest(
            compound_truth_path, predictor=predictor
        )
    )
    compound_classification = next(
        row
        for row in compound_manifest["classification_rows"]
        if row["cohort_entry_sha256"] == retry_entry["entry_sha256"]
    )
    assert compound_classification["task_id"] == compound_task_id
    assert [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in compound_ranked
    ] == [
        (
            row["candidate_physics_sha256"],
            row["truth_non_dominated_rank"],
            row["actual_volume_L"],
            row["actual_total_loss_W"],
        )
        for row in ranked
    ]
    with pytest.raises(
        production.HandoffContractError,
        match="outside or duplicates",
    ):
        promotion.create_truth_promotion(
            standard_collection_paths=[
                retry_collection_path,
                compound_collection_path,
                *collection_paths[2:],
            ],
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-compound-duplicate-v2",
            predictor=predictor,
        )

    first_truth = copy.deepcopy(ranked[0])
    first_classification = next(
        row
        for row in manifest["classification_rows"]
        if row["candidate_physics_sha256"]
        == first_truth["candidate_physics_sha256"]
    )
    conflicting_truth = copy.deepcopy(first_truth)
    conflicting_truth["actual_total_loss_W"] += 0.001
    with pytest.raises(
        production.HandoffContractError,
        match="mixed actual truth",
    ):
        promotion._validate_duplicate_cohort_observations(
            [
                ({}, first_truth, first_classification, {}),
                ({}, conflicting_truth, first_classification, {}),
            ]
        )
    plan_set_path = promotion.create_full_plans(
        truth_manifest_path=manifest_path,
        solver_revision="2" * 40,
        library_revision="3" * 40,
        output=tmp_path / "truth-full-plans-v2",
        predictor=predictor,
    )
    plan_set, plans = promotion._load_full_plan_set(
        plan_set_path, predictor=predictor
    )
    assert plan_set["selected_plan_count"] == len(ranked)
    assert 1 <= len(plans) <= promotion.MAX_FULL_PLANS

    tampered_collection = production._read_json(collection_paths[-1])
    tampered_collection.pop("payload_sha256")
    tampered_collection["goal_physical_spec_passed"] = not (
        tampered_collection["goal_physical_spec_passed"]
    )
    tampered_collection_path = production._write_immutable_json(
        tmp_path / "tampered-collection.json",
        production._seal(tampered_collection),
    )
    with pytest.raises(production.HandoffContractError):
        promotion.create_truth_promotion(
            standard_collection_paths=(
                collection_paths[:-1] + [tampered_collection_path]
            ),
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "forbidden-tampered-v2",
            predictor=predictor,
        )

    tampered_manifest = production._read_json(manifest_path)
    tampered_manifest.pop("payload_sha256")
    classification = tampered_manifest["classification_rows"][0]
    classification.pop("classification_sha256")
    classification["actual_truth_feasible"] = not classification[
        "actual_truth_feasible"
    ]
    classification["classification_sha256"] = (
        production.canonical_sha256(classification)
    )
    tampered_manifest_path = production._write_immutable_json(
        manifest_path.parent / "tampered-truth-manifest.json",
        production._seal(tampered_manifest),
    )
    with pytest.raises(production.HandoffContractError):
        promotion._load_truth_manifest(
            tampered_manifest_path, predictor=predictor
        )

    entry_by_submission = {
        str(Path(entry["submission"]["path"]).resolve()): entry
        for entry in cohort_entries
    }
    last_submission = str(submission_paths[-1].resolve())

    def mixed_revision_entry(path, *, predictor=None):
        del predictor
        entry = copy.deepcopy(
            entry_by_submission[str(path.resolve(strict=True))]
        )
        if str(path.resolve(strict=True)) == last_submission:
            entry["library_revision"] = "4" * 40
            entry.pop("entry_sha256")
            entry["entry_sha256"] = production.canonical_sha256(entry)
        return entry

    with monkeypatch.context() as context:
        context.setattr(
            promotion,
            "_authenticated_submission_entry",
            mixed_revision_entry,
        )
        with pytest.raises(
            production.HandoffContractError,
            match="cannot mix solver/library",
        ):
            promotion.create_cohort_inventory(
                standard_submission_paths=submission_paths,
                output=tmp_path / "forbidden-mixed-revision.json",
                predictor=predictor,
            )

    original_observation = promotion._actual_standard_observation

    def force_infeasible(view, *, collection_path):
        truth, status = original_observation(
            view, collection_path=collection_path
        )
        status = copy.deepcopy(status)
        status["goal_physical_spec_passed"] = False
        status["goal_physical_spec_reasons"] = [
            "test_forced_infeasible"
        ]
        status["actual_truth_feasible"] = False
        status["promotion_exclusion_reasons"] = [
            "test_forced_infeasible"
        ]
        return truth, status

    with monkeypatch.context() as context:
        context.setattr(
            promotion,
            "_actual_standard_observation",
            force_infeasible,
        )
        zero_path = promotion.create_truth_promotion(
            standard_collection_paths=collection_paths,
            cohort_inventory_path=built["inventory_path"],
            output=tmp_path / "truth-promotion-v2-zero",
            predictor=predictor,
        )
        zero_manifest, zero_ranked, _unused = (
            promotion._load_truth_manifest(
                zero_path, predictor=predictor
            )
        )
        assert zero_ranked == []
        assert zero_manifest["feasible_collection_count"] == 0
        assert zero_manifest["rank0_count"] == 0
        assert zero_manifest["full_plan_eligible"] is False
        assert zero_manifest["zero_feasible_audited"] is True
        with pytest.raises(
            production.HandoffContractError,
            match="no rank-0 candidate",
        ):
            promotion.create_full_plans(
                truth_manifest_path=zero_path,
                solver_revision="2" * 40,
                library_revision="3" * 40,
                output=tmp_path / "forbidden-zero-full",
                predictor=predictor,
            )


def test_truth_sorting_includes_the_twenty_fourth_candidate():
    rows = [
        {
            "candidate_physics_sha256": f"{index:064x}",
            "actual_volume_L": float(100 + index),
            "actual_total_loss_W": float(100 + index),
        }
        for index in range(23)
    ]
    last = {
        "candidate_physics_sha256": "f" * 64,
        "actual_volume_L": 1.0,
        "actual_total_loss_W": 1.0,
    }
    ranked = promotion._rank_truth(rows + [last])
    rank0 = [
        row
        for row in ranked
        if row["truth_non_dominated_rank"] == 0
    ]
    assert len(ranked) == 24
    assert [row["candidate_physics_sha256"] for row in rank0] == [
        last["candidate_physics_sha256"]
    ]
    assert max(row["truth_non_dominated_rank"] for row in ranked) == 23


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
