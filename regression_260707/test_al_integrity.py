import copy
import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "verify"))

import al_driver
import scheduler_client
import select_candidates
from module.input_parameter_260706 import KEYS, create_input_parameter
from training.checkpoint_train import filter_valid_training_rows

TEST_REVISION = "a" * 40
TEST_LIBRARY_REVISION = "b" * 40


def complete_candidate_params(**updates):
    safe = {
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0,
        "w2c_w1c_space_y": 40.0,
        "w1c_w2s_space_x": 40.0,
        "w2s_w1s_space_x": 40.0,
        "w1s_w2s_space_y": 40.0,
        "w1s_cs_space_x": 40.0,
        "cs_w1s_space_y": 40.0,
    }
    safe.update(updates)
    row = create_input_parameter(safe).iloc[0]
    # Material-contract constants are explicit solver inputs/results but are
    # intentionally outside the sealed Sobol/campaign identity KEYS.  AL task
    # identity must remain the exact KEYS payload.
    return {key: row[key] for key in KEYS}


def standard_submitted_params(**updates):
    profile = json.loads(
        (HERE / "verify" / "profiles" / "standard.json").read_text(
            encoding="utf-8"
        )
    )
    return scheduler_client.effective_verification_params(
        complete_candidate_params(**updates), profile
    )


def valid_result(**updates):
    result = {
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_solved": 1,
        "thermal_convergence_available": 1,
        "thermal_converged": 1,
        "thermal_extraction_complete": 1,
        "thermal_iterations": 151,
        "thermal_residual_flow_limit": 1e-3,
        "thermal_residual_energy_limit": 1e-7,
        "thermal_residual_continuity": 8e-4,
        "thermal_residual_x_velocity": 4e-4,
        "thermal_residual_y_velocity": 9e-4,
        "thermal_residual_z_velocity": 4e-4,
        "thermal_residual_energy": 4e-9,
        "thermal_rx_model": "homogenized_blocks",
        "thermal_rx_power_balance_ok": 1,
        "thermal_rx_power_balance_group_count": 2,
        "thermal_rx_power_balance_max_abs_w": 0.0,
        "thermal_rx_expected_power_w": 120.0,
        "thermal_rx_assigned_power_w": 120.0,
        "thermal_required_missing_count": 0,
        "thermal_required_group_mask": 15,
        "git_hash": TEST_REVISION,
        "git_dirty": 0,
        "pyaedt_library_git_hash": TEST_LIBRARY_REVISION,
        "pyaedt_library_git_dirty": 0,
        "matrix_solve_attempts": 1,
        "loss_solve_attempts": 1,
        "matrix_extraction_backend": "export_rl_matrix",
        "matrix_conductor_policy": "stranded_no_eddy_no_skin",
        "matrix_winding_stranded_count": 2,
        "matrix_conductor_mesh_operation_count": 0,
        "matrix_plate_eddy_off_readback_count": 5,
        "loss_winding_solid_update_count": 2,
        "loss_winding_mesh_operation_count": 2,
        "loss_conductor_mesh_operation_count": 3,
        "loss_plate_eddy_on_readback_count": 5,
        "Llt": 13.75,
        "l1": 50.0,
        "Ltx": 100.0,
        "Lrx": 100.0,
        "M": 86.25,
        "k": 0.8625,
        "Lmt": 86.25,
        "Lmr": 86.25,
        "Llr": 13.75,
        "full_model": 0,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "matrix_skin_mesh": 0,
        "matrix_percent_error": 1.5,
        "matrix_max_passes": 20,
        "matrix_min_converged": 1,
        "percent_error": 1.5,
        "max_passes": 10,
        "min_converged": 2,
        "freq": 1000.0,
        "V1_rms": 1000.0,
        "I1_rated": 1000.0,
        "I2_rated": 100.0,
        "I2_phase_deg": 0.0,
        "loss_from_copy": 1,
        "P_target": 1_000_000.0,
        "V2_rms": 10_000.0,
        "core_cm": 1.377,
        "core_x": 1.51,
        "core_y": 1.74,
        "core_plate_on": 1,
        "wcp_on": 1,
        "round_corner": 0,
        "plate_temp": 50.0,
        "air_temp": 50.0,
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_k_thermal": 2.0,
        "rx_mesh_mode": "skin",
        "fan_config": "dual",
        "thermal_max_iterations": 250,
        "conductor_temp_C": 80.0,
        "keep_project": 0,
        "B_max_core": 1.1,
        "B_mean_core": 0.8,
        "N2_side": 2,
        "N1_side": 0,
        "conv_passes_matrix": 3,
        "conv_consecutive_matrix": 1,
        "conv_error_pct_matrix": 0.5,
        "conv_delta_pct_matrix": 0.2,
        "conv_passes_loss": 3,
        "conv_consecutive_loss": 2,
        "conv_error_pct_loss": 0.6,
        "conv_delta_pct_loss": 0.3,
        "P_winding_total": 4000.0,
        "P_Tx_main_group": 2400.0,
        "P_Rx_main_group": 1200.0,
        "P_Rx_side_total": 400.0,
        "P_core_total": 2000.0,
        "P_core_plate_total": 500.0,
        "P_wcp_total": 0.0,
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0,
        "w2c_w1c_space_y": 40.0,
        "w1c_w2s_gap_x_actual": 40.0,
        "w1s_cs_space_x": 40.0,
        "cs_w1s_space_y": 40.0,
        "h_gap2": 40.0,
        "T_max_Tx": 90.0,
        "T_max_Rx_main": 91.0,
        "T_max_Rx_side": 92.0,
        "T_max_core": 93.0,
        "Tprobe_Tx_leeward_max": 89.0,
        "Tprobe_Rx_main_leeward_max": 90.0,
        "Tprobe_Rx_side_leeward_max": 91.0,
        "Tprobe_core_center_max": 92.0,
        "Tprobe_core_center_leg_max": 88.0,
        "Tprobe_core_side_leg_max": 90.0,
        "Tprobe_core_top_yoke_max": 92.0,
        "project_name": "simulation-test",
        "saved_at": "2026-07-10 07:00:00",
    }
    result.update(updates)
    if float(result.get("N2_side", 0)) <= 0 and "thermal_required_group_mask" not in updates:
        result["thermal_required_group_mask"] = 11
    return result


def http_response(text):
    response = Mock(text=text, status_code=200)
    response.raise_for_status.return_value = None
    return response


def task_inventory_response(tasks):
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = tasks
    return response


def complete_errors(spec_pass=(1, 1, 1)):
    temperature_targets = tuple(
        al_driver.MANDATORY_SURROGATE_TEMPERATURE_TARGETS
    )
    errors = {
        "dllt_pct": [0.1, 0.2, 0.3],
        "llt_fea": [27.5, 27.5, 27.5],
        "spec_pass": list(spec_pass),
        "temperature_error_expected_count": [len(temperature_targets)] * 3,
        "temperature_error_complete": [1, 1, 1],
        "dloss_pct": [1.0, 2.0, 1.5],
    }
    errors.update({
        f"d_{target}": [1.0, 2.0, 1.5]
        for target in temperature_targets
    })
    return errors


class ThermalTrainingFilterTests(unittest.TestCase):
    def test_training_filter_rejects_mixed_physics_revision_cohorts(self):
        frame = pd.DataFrame({
            "_strict_valid_full": [True, True],
            "Llt": [27.4, 27.6],
            "physics_data_revision": ["revision-a", "revision-b"],
        })

        with self.assertRaisesRegex(
                RuntimeError, "mixes physics_data_revision cohorts"):
            filter_valid_training_rows(frame, "Llt")

    def test_training_filter_records_single_or_legacy_revision_cohort(self):
        explicit = pd.DataFrame({
            "_strict_valid_full": [True, True],
            "Llt": [27.4, 27.6],
            "physics_data_revision": ["revision-a", "revision-a"],
        })
        legacy = explicit.drop(columns=["physics_data_revision"])

        self.assertEqual(
            filter_valid_training_rows(
                explicit, "Llt"
            ).attrs["physics_data_revision_cohort"],
            "revision-a",
        )
        self.assertEqual(
            filter_valid_training_rows(
                legacy, "Llt"
            ).attrs["physics_data_revision_cohort"],
            "legacy_unspecified",
        )

    def test_only_complete_unsaturated_rows_feed_temperature_models(self):
        targets = (
            "Tprobe_core_center_max",
            *al_driver.CORE_REGION_TEMPERATURE_TARGETS,
        )
        for target in targets:
            with self.subTest(target=target):
                frame = pd.DataFrame([
                    valid_result(sample="valid", **{target: 92.0}),
                    valid_result(sample="thermal-invalid", result_valid_thermal=0,
                                 **{target: 93.0}),
                    valid_result(sample="not-solved", thermal_solved=0,
                                 **{target: 94.0}),
                    valid_result(sample="below-cap", **{target: 4699.9}),
                    valid_result(sample="at-cap", **{target: 4700.0}),
                ])

                filtered = filter_valid_training_rows(frame, target)

                self.assertEqual(
                    filtered["sample"].tolist(), ["valid", "below-cap"])
                self.assertEqual(
                    filter_valid_training_rows(frame, "Llt")["sample"].tolist(),
                    ["valid", "below-cap"],
                )


def verification_counts(total=3, valid=3, exhausted=0, pending=0):
    return {
        "round": 1,
        "total": total,
        "valid": valid,
        "exhausted": exhausted,
        "pending": pending,
        "coverage": valid / total if total else 0.0,
        "ingested": valid,
    }


def task_state(task_id=17, stage="WAIT"):
    return {
        "round": 1,
        "stage": stage,
        "task_map": {"0": task_id},
        "solver_git_revision": TEST_REVISION,
        "pyaedt_library_git_revision": TEST_LIBRARY_REVISION,
    }


class SchedulerClientIntegrityTests(unittest.TestCase):
    def setUp(self):
        project_contract = patch.object(
            scheduler_client,
            "require_live_project_mutation_contract",
            return_value={
                "name": scheduler_client.MFT_PROJECT,
                "max_active_tasks":
                    scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
                "auto_pull": False,
            },
        )
        self.require_project_contract = project_contract.start()
        self.addCleanup(project_contract.stop)
        capacity_snapshot = patch.object(
            scheduler_client,
            "live_project_submission_snapshot",
            return_value={
                "project_active": 0,
                "project_submission_slots":
                    scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
            },
        )
        self.capacity_snapshot = capacity_snapshot.start()
        self.addCleanup(capacity_snapshot.stop)

    def test_project_mutation_contract_requires_bounded_cap_and_auto_pull_false(self):
        valid = {
            "name": scheduler_client.MFT_PROJECT,
            "max_active_tasks": scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
            "auto_pull": False,
        }
        self.assertEqual(
            scheduler_client.validate_project_mutation_contract(valid), valid)
        for cap in (1, 137, 299, 300):
            self.assertEqual(
                scheduler_client.validate_project_mutation_contract(
                    {**valid, "max_active_tasks": cap})["max_active_tasks"],
                cap,
            )
        with self.assertRaises(scheduler_client.ProjectContractError):
            scheduler_client.validate_project_mutation_contract(
                {**valid, "max_active_tasks": 299}, expected_cap=300)
        for override in (
            {"max_active_tasks": 0},
            {"max_active_tasks": 301},
            {"max_active_tasks": 400.0},
            {"max_active_tasks": 400.1},
            {"max_active_tasks": "300"},
            {"max_active_tasks": True},
            {"auto_pull": True},
            {"auto_pull": None},
        ):
            with self.subTest(override=override), self.assertRaises(
                    scheduler_client.ProjectContractError):
                scheduler_client.validate_project_mutation_contract(
                    {**valid, **override})

    def test_full_project_contract_fails_closed_on_missing_or_partial_fields(self):
        project = {
            "name": scheduler_client.MFT_PROJECT,
            "max_active_tasks": 275,
            "auto_pull": False,
            "repos": copy.deepcopy(scheduler_client.MFT_PROJECT_REPOS),
            "setup": scheduler_client.MFT_PROJECT_SETUP,
            "entrypoints": copy.deepcopy(
                scheduler_client.MFT_PROJECT_ENTRYPOINTS),
            "cleanup_globs": scheduler_client.MFT_PROJECT_CLEANUP_GLOBS,
            "output_globs": scheduler_client.MFT_PROJECT_OUTPUT_GLOBS,
            "sim_subdir": scheduler_client.MFT_PROJECT_SIM_SUBDIR,
        }
        contract = scheduler_client.validate_project_mutation_contract(
            project, expected_cap=275, require_full=True)
        self.assertEqual(contract["max_active_tasks"], 275)
        for field in (
                "repos", "setup", "entrypoints", "cleanup_globs",
                "output_globs", "sim_subdir"):
            drifted = copy.deepcopy(project)
            drifted.pop(field)
            with self.subTest(field=field), self.assertRaises(
                    scheduler_client.ProjectContractError):
                scheduler_client.validate_project_mutation_contract(
                    drifted, expected_cap=275, require_full=True)

    def test_submit_wrapper_owns_the_common_mutation_lock(self):
        observed = []

        def locked_submit(*_args, **_kwargs):
            observed.append(scheduler_client.campaign_mutation_lock_is_held())
            return 77

        with patch.object(
                scheduler_client, "_submit_verification_locked",
                side_effect=locked_submit) as submit:
            task_id = scheduler_client.submit_verification(
                "candidate-lock", "wd", {}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
            )

        self.assertEqual(task_id, 77)
        self.assertEqual(observed, [True])
        submit.assert_called_once()
        self.assertFalse(scheduler_client.campaign_mutation_lock_is_held())

    def test_absolute_project_capacity_blocks_post(self):
        self.capacity_snapshot.return_value = {
            "project_active": scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS,
            "project_submission_slots": 0,
        }
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), patch.object(
                    scheduler_client.requests, "post") as post:
            with self.assertRaises(scheduler_client.ProjectCapacityError):
                scheduler_client.submit_verification(
                    "candidate-full", "wd", {}, {},
                    solver_revision=TEST_REVISION,
                    library_revision=TEST_LIBRARY_REVISION,
                )

        post.assert_not_called()

    def test_reconcile_accepts_new_and_legacy_active_and_terminal_rows(self):
        for project in (scheduler_client.MFT_PROJECT, ""):
            for status in ("running", "completed"):
                with self.subTest(project=project, status=status):
                    response = task_inventory_response([{
                        "id": 71,
                        "name": "candidate-existing",
                        "dedupe_key": "exact-key",
                        "project": project,
                        "status": status,
                    }])
                    with patch.object(
                            scheduler_client.requests, "get",
                            return_value=response) as get:
                        task_id = scheduler_client.reconcile_task_id(
                            "candidate-existing", "exact-key", attempts=1)
                    self.assertEqual(task_id, 71)
                    self.assertEqual(get.call_args.kwargs["params"], {
                        "limit": 10000,
                        "name_prefix": "candidate-existing",
                    })

    def test_reconcile_rejects_exact_identity_from_another_project(self):
        response = task_inventory_response([{
            "id": 72,
            "name": "candidate-foreign",
            "dedupe_key": "exact-key",
            "project": "IPMSM",
            "status": "running",
        }])
        with patch.object(
                scheduler_client.requests, "get", return_value=response):
            with self.assertRaises(scheduler_client.TaskLookupError):
                scheduler_client.reconcile_task_id(
                    "candidate-foreign", "exact-key", attempts=1)

    def test_submit_exports_fluent_fork_bootstrap_before_solver_launch(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 40}

        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(
                    scheduler_client.requests, "post",
                    return_value=submitted) as post:
            scheduler_client.submit_verification(
                "candidate-mpi", "candidate_workdir", {"x": 1}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
                priority=10,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["priority"], 10)
        command = payload["command"]
        hydra_export = "export I_MPI_HYDRA_BOOTSTRAP=fork;"
        fluent_export = "export FLUENT_MPIRUN_FLAGS='-bootstrap fork';"
        self.assertEqual(command.count(hydra_export), 1)
        self.assertEqual(command.count(fluent_export), 1)
        self.assertLess(
            command.index(hydra_export),
            command.index(fluent_export),
        )
        self.assertLess(
            command.index(fluent_export),
            command.index("python run_simulation_260706.py"),
        )

    def test_submit_forwards_explicit_integer_priority(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 401}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), patch.object(
                    scheduler_client.requests, "post",
                    return_value=submitted) as post:
            task_id = scheduler_client.submit_verification(
                "candidate-priority", "candidate_workdir", {"x": 1}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
                priority=scheduler_client.TEST_TASK_PRIORITY,
            )

        self.assertEqual(task_id, 401)
        self.assertEqual(
            post.call_args.kwargs["json"]["priority"],
            scheduler_client.TEST_TASK_PRIORITY,
        )

    def test_submit_forwards_distinct_node_placement_constraints(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 402}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), patch.object(
                    scheduler_client.requests, "post",
                    return_value=submitted) as post:
            task_id = scheduler_client.submit_verification(
                "candidate-placement", "candidate_workdir", {"x": 1}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
                priority=scheduler_client.TEST_TASK_PRIORITY,
                account_name="account-a",
                node_name="node-101",
                max_workers_per_node=1,
            )

        self.assertEqual(task_id, 402)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["account_name"], "account-a")
        self.assertEqual(payload["node_name"], "node-101")
        self.assertEqual(payload["max_workers_per_node"], 1)
        self.assertFalse(payload.get("exclusive_node", False))

    def test_submit_rejects_non_integer_priority(self):
        for priority in (True, 10.0, "10", None):
            with self.subTest(priority=priority), patch.object(
                    scheduler_client.requests, "post") as post:
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    scheduler_client.submit_verification(
                        "candidate-invalid-priority", "candidate_workdir",
                        {"x": 1}, {},
                        solver_revision=TEST_REVISION,
                        library_revision=TEST_LIBRARY_REVISION,
                        priority=priority,
                    )
            post.assert_not_called()

    def test_dynamic_submit_binds_every_post_to_exact_observed_cap(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 404}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), patch.object(
                scheduler_client.requests, "post", return_value=submitted):
            task_id = scheduler_client.submit_verification(
                "candidate-dynamic-cap", "candidate_workdir", {"x": 1}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
                required_project_cap=275,
            )

        self.assertEqual(task_id, 404)
        self.capacity_snapshot.assert_called_once_with(
            275,
            require_exact_project_cap=True,
            require_full_project=True,
        )

    def test_queued_cancel_uses_status_cas_and_validates_acknowledgement(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"cancelled": [12, 13], "count": 2}
        with patch.object(
                scheduler_client, "campaign_mutation_lock_is_held",
                return_value=True), patch.object(
                scheduler_client.requests, "post", return_value=response) as post:
            result = scheduler_client.cancel_queued_tasks_cas([12, 13])

        self.assertEqual(result, {"cancelled": [12, 13], "count": 2})
        self.assertEqual(post.call_args.kwargs["params"], {
            "task_ids": "12,13",
            "statuses": "queued",
        })
        response.json.return_value = {"cancelled": [99], "count": 1}
        with patch.object(
                scheduler_client, "campaign_mutation_lock_is_held",
                return_value=True), patch.object(
                scheduler_client.requests, "post", return_value=response):
            with self.assertRaises(scheduler_client.ProjectContractError):
                scheduler_client.cancel_queued_tasks_cas([12, 13])

    def test_submit_groups_preflight_and_simulation_before_preserving_exit_code(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 41, "name": "candidate-41"}

        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])) as get, \
                patch.object(scheduler_client.requests, "post", return_value=submitted) as post:
            task_id = scheduler_client.submit_verification(
                "candidate-41", "candidate_workdir", {"x": 1}, {},
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
            )

        self.assertEqual(task_id, 41)
        self.capacity_snapshot.assert_called_once_with(
            scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS)
        self.assertTrue(post.call_args.args[0].endswith("/api/tasks"))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["project"], scheduler_client.MFT_PROJECT)
        self.assertEqual(payload["priority"], 0)
        self.assertRegex(
            payload["dedupe_key"],
            rf"^mft-al:candidate-41:{TEST_REVISION}:{TEST_LIBRARY_REVISION}:[0-9a-f]{{16}}$")
        command = payload["command"]
        parameter_digest = hashlib.sha256(b'{"x":1}').hexdigest()[:16]
        isolated = (
            f"candidate_workdir-s{TEST_REVISION[:12]}-"
            f"l{TEST_LIBRARY_REVISION[:12]}-p{parameter_digest}"
        )
        task_identity = hashlib.sha256(
            payload["dedupe_key"].encode("utf-8")).hexdigest()[:16]
        task_workdir = f"{scheduler_client.SCRATCH_LEAF_PREFIX}{isolated}-t{task_identity}"
        self.assertEqual(payload["remote_cwd"], scheduler_client.GPFS_RUNS_REMOTE_CWD)
        self.assertIn("MFT_GPFS_ROOT=$PWD", command)
        self.assertIn(
            f'MFT_GPFS_WORKDIR="$MFT_GPFS_ROOT/{task_workdir}"', command)
        self.assertIn(f"MFT_NVME_WORKDIR=/enroot/{task_workdir}", command)
        self.assertEqual(payload["cleanup_globs"], task_workdir)
        self.assertNotRegex(payload["cleanup_globs"], r"[/\\*?\[]")
        self.assertIn("findmnt -n -o FSTYPE -T /enroot", command)
        self.assertIn(
            f'"${{MFT_ENROOT_FREE_KB:-0}}" -ge '
            f"{scheduler_client.LOCAL_SCRATCH_MIN_FREE_KB}", command)
        self.assertIn("MFT_WORKDIR=$MFT_NVME_WORKDIR", command)
        self.assertIn("MFT_WORKDIR=$MFT_GPFS_WORKDIR", command)
        self.assertIn("MFT_WORKDIR %s", command)
        self.assertIn(
            "git -C \"${MFT_WORKDIR}/pyaedt_library\" fetch -q origin "
            f"{TEST_LIBRARY_REVISION}", command)
        self.assertIn(
            "git -C \"${MFT_WORKDIR}/pyaedt_library\" checkout -q --detach "
            f"{TEST_LIBRARY_REVISION}", command)
        self.assertIn(f"MFT_LIBRARY_GIT_HASH {TEST_LIBRARY_REVISION}", command)
        self.assertIn(
            f"cd \"${{MFT_WORKDIR}}/repo\" && git fetch -q origin {TEST_REVISION}",
            command)
        self.assertIn(f"git checkout -q --detach {TEST_REVISION}", command)
        self.assertIn(f'test "$(git rev-parse HEAD)" = "{TEST_REVISION}"', command)
        self.assertIn("git diff --quiet HEAD -- && git clean -q -ffd", command)
        self.assertIn("git clean -q -ffdX", command)
        self.assertLess(
            command.index("MFT_LIBRARY_GIT_HASH"),
            command.index("python run_simulation_260706.py"))
        self.assertIn("python run_simulation_260706.py --fixed", command)
        self.assertIn(
            'cleanup() { rm -rf -- "${MFT_NVME_WORKDIR}" "${MFT_GPFS_WORKDIR}" ',
            command)
        self.assertIn("trap cleanup EXIT", command)
        self.assertIn("trap 'exit 143' TERM INT", command)
        self.assertIn(
            f"-mmin +{scheduler_client.LOCAL_SCRATCH_STALE_MINUTES}", command)
        gpfs_sweep = (
            'find "$MFT_GPFS_ROOT" -mindepth 1 -maxdepth 1 -type d '
            '-user "$USER" '
            f"-name {scheduler_client.SCRATCH_LEAF_PATTERN!r} "
            f"-mmin +{scheduler_client.GPFS_SCRATCH_STALE_MINUTES} "
            "-exec rm -rf -- {} +"
        )
        self.assertIn(gpfs_sweep, command)
        self.assertEqual(scheduler_client.GPFS_SCRATCH_STALE_MINUTES, 8 * 60)
        self.assertNotIn(
            'find "$MFT_GPFS_ROOT" -mindepth 1 -maxdepth 1 -type d '
            '-user "$USER" -name \'mft_*\'',
            command,
        )
        self.assertNotIn("rc=$?; rm -rf simulation aedt_temp", command)
        self.assertGreater(
            command.rindex("MFT_LIBRARY_GIT_HASH"),
            command.index("python run_simulation_260706.py"))
        self.assertTrue(command.endswith("exit $simulation_rc )"))
        self.assertEqual(get.call_args.kwargs["params"], {
            "limit": 10000,
            "name_prefix": "candidate-41",
        })

    def test_same_candidate_retries_have_distinct_exact_cleanup_ownership(self):
        submitted = Mock(status_code=201)
        submitted.json.side_effect = [{"id": 51}, {"id": 52}]

        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(
                    scheduler_client.requests, "post",
                    return_value=submitted) as post:
            for name in ("candidate-retry", "candidate-retry-r1"):
                scheduler_client.submit_verification(
                    name, "shared_workdir", {"x": 1}, {},
                    solver_revision=TEST_REVISION,
                    library_revision=TEST_LIBRARY_REVISION,
                )

        payloads = [call.kwargs["json"] for call in post.call_args_list]
        cleanup_globs = [payload["cleanup_globs"] for payload in payloads]
        self.assertEqual(len(set(cleanup_globs)), 2)
        for payload, cleanup_glob in zip(payloads, cleanup_globs):
            self.assertNotRegex(cleanup_glob, r"[/\\*?\[]")
            self.assertRegex(
                cleanup_glob,
                r"^mft_campaign-[A-Za-z0-9_-]+-t[0-9a-f]{16}$",
            )
            self.assertEqual(
                payload["remote_cwd"], scheduler_client.GPFS_RUNS_REMOTE_CWD)
            self.assertIn("MFT_GPFS_ROOT=$PWD", payload["command"])
            self.assertIn(
                f'MFT_GPFS_WORKDIR="$MFT_GPFS_ROOT/{cleanup_glob}"',
                payload["command"])
            self.assertIn(
                f"MFT_NVME_WORKDIR=/enroot/{cleanup_glob}", payload["command"])

    def test_cleanup_basename_is_scheduler_safe_for_hostile_workdir(self):
        submitted = Mock(status_code=201)
        submitted.json.return_value = {"id": 53}

        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(
                    scheduler_client.requests, "post",
                    return_value=submitted) as post:
            scheduler_client.submit_verification(
                "candidate-hostile", "prefix/../shared,*?[case]" * 20,
                {"x": 1}, {}, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION,
            )

        payload = post.call_args.kwargs["json"]
        cleanup_glob = payload["cleanup_globs"]
        self.assertLessEqual(len(cleanup_glob), 198)
        self.assertRegex(
            cleanup_glob,
            r"^mft_campaign-[A-Za-z0-9_-]+-t[0-9a-f]{16}$",
        )
        self.assertNotIn("..", cleanup_glob)
        self.assertIn(
            f'MFT_GPFS_WORKDIR="$MFT_GPFS_ROOT/{cleanup_glob}"',
            payload["command"])

    def test_terminal_exact_identity_is_reconciled_before_post(self):
        first_post = Mock(status_code=201)
        first_post.json.return_value = {"id": 99}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(scheduler_client.requests, "post", return_value=first_post) as post:
            scheduler_client.submit_verification(
                "candidate-terminal", "wd", {}, {}, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
        dedupe_key = post.call_args.kwargs["json"]["dedupe_key"]

        terminal = task_inventory_response([{
            "id": 77,
            "name": "candidate-terminal",
            "dedupe_key": dedupe_key,
            "status": "completed",
        }])
        self.capacity_snapshot.reset_mock()
        with patch.object(scheduler_client.requests, "get", return_value=terminal), \
                patch.object(scheduler_client.requests, "post") as second_post:
            task_id = scheduler_client.submit_verification(
                "candidate-terminal", "wd", {}, {}, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
        self.assertEqual(task_id, 77)
        second_post.assert_not_called()
        self.capacity_snapshot.assert_not_called()

    def test_lookup_uncertainty_prevents_post_and_response_loss_reconciles(self):
        with patch.object(
                scheduler_client.requests, "get",
                side_effect=requests.ConnectionError("inventory offline")), \
                patch.object(scheduler_client.requests, "post") as post, \
                patch.object(scheduler_client.time, "sleep"):
            with self.assertRaises(scheduler_client.TaskLookupError):
                scheduler_client.submit_verification(
                    "candidate-offline", "wd", {}, {}, solver_revision=TEST_REVISION,
                    library_revision=TEST_LIBRARY_REVISION)
        post.assert_not_called()

        first_post = Mock(status_code=201)
        first_post.json.return_value = {"id": 88}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(scheduler_client.requests, "post", return_value=first_post) as post:
            scheduler_client.submit_verification(
                "candidate-lost", "wd", {"x": 1}, {}, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
        dedupe_key = post.call_args.kwargs["json"]["dedupe_key"]
        recovered = task_inventory_response([{
            "id": 88,
            "name": "candidate-lost",
            "dedupe_key": dedupe_key,
            "project": "",
            "status": "running",
        }])
        with patch.object(
                scheduler_client.requests, "get",
                side_effect=[task_inventory_response([]), recovered]), \
                patch.object(
                    scheduler_client.requests, "post",
                    side_effect=requests.Timeout("response lost")):
            task_id = scheduler_client.submit_verification(
                "candidate-lost", "wd", {"x": 1}, {}, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
        self.assertEqual(task_id, 88)

    def test_latest_well_formed_result_is_authoritative(self):
        good = valid_result(sample="good")
        invalid = valid_result(sample="bad", result_valid_thermal=0)
        response = http_response("\n".join([
            "RESULT_JSON " + json.dumps(good),
            "RESULT_JSON {truncated",
            "RESULT_JSON " + json.dumps(invalid),
        ]))

        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(7, attempts=1)

        self.assertEqual(fetched.state, scheduler_client.RESULT_INVALID)
        self.assertEqual(fetched.result, invalid)

        response.text = "\n".join([
            "RESULT_JSON " + json.dumps(good),
            "RESULT_JSON {truncated",
        ])
        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(8, attempts=1)
        self.assertEqual(fetched.state, scheduler_client.RESULT_VALID)
        self.assertEqual(fetched.result, good)

    def test_fetch_distinguishes_missing_and_transport_failure(self):
        with patch.object(
                scheduler_client.requests, "get", return_value=http_response("no rows")):
            fetched = scheduler_client.fetch_result(9, attempts=1)
        self.assertEqual(fetched.state, scheduler_client.RESULT_MISSING)

        with patch.object(
                scheduler_client.requests, "get",
                side_effect=requests.ConnectionError("offline")), \
                patch.object(scheduler_client.time, "sleep") as sleep:
            with self.assertRaises(scheduler_client.ResultFetchError):
                scheduler_client.fetch_result(10, attempts=2, retry_delay=0)
        self.assertEqual(sleep.call_count, 1)

    def test_fetch_requires_result_and_wrapper_library_hashes_to_match(self):
        result = valid_result()
        response = http_response("\n".join([
            f"MFT_LIBRARY_GIT_HASH {TEST_LIBRARY_REVISION}",
            "RESULT_JSON " + json.dumps(result),
        ]))

        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(
                11, attempts=1, expected_revision=TEST_REVISION,
                expected_library_revision=TEST_LIBRARY_REVISION)

        self.assertEqual(fetched.state, scheduler_client.RESULT_VALID)
        self.assertEqual(fetched.result["pyaedt_library_git_hash"], TEST_LIBRARY_REVISION)

        response.text = "\n".join([
            f"MFT_LIBRARY_GIT_HASH {'c' * 40}",
            "RESULT_JSON " + json.dumps(valid_result(
                pyaedt_library_git_hash=TEST_LIBRARY_REVISION)),
        ])
        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(
                12, attempts=1,
                expected_library_revision=TEST_LIBRARY_REVISION)
        self.assertEqual(fetched.state, scheduler_client.RESULT_INVALID)
        self.assertIsNone(fetched.result)

    def test_tail_library_marker_survives_scheduler_stdout_truncation(self):
        result = valid_result()
        response = http_response(
            "x" * 300_000
            + "\nRESULT_JSON " + json.dumps(result)
            + f"\nMFT_LIBRARY_GIT_HASH {TEST_LIBRARY_REVISION}\n"
        )
        response.text = response.text[-262_144:]

        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(
                13, attempts=1, expected_revision=TEST_REVISION,
                expected_library_revision=TEST_LIBRARY_REVISION)

        self.assertEqual(fetched.state, scheduler_client.RESULT_VALID)

    def test_fetch_requests_scheduler_hard_stdout_tail_limit(self):
        response = http_response("\n".join([
            f"MFT_LIBRARY_GIT_HASH {TEST_LIBRARY_REVISION}",
            "RESULT_JSON " + json.dumps(valid_result()),
        ]))
        with patch.object(
                scheduler_client.requests, "get", return_value=response) as get:
            fetched = scheduler_client.fetch_result(
                14, attempts=1, expected_revision=TEST_REVISION,
                expected_library_revision=TEST_LIBRARY_REVISION)

        self.assertEqual(fetched.state, scheduler_client.RESULT_VALID)
        self.assertEqual(
            get.call_args.kwargs["params"]["max_bytes"],
            scheduler_client.MAX_STDOUT_BYTES)

    def test_fetch_rejects_wrapper_result_library_mismatch(self):
        response = http_response("\n".join([
            f"MFT_LIBRARY_GIT_HASH {TEST_LIBRARY_REVISION}",
            "RESULT_JSON " + json.dumps(valid_result(
                pyaedt_library_git_hash="c" * 40)),
        ]))
        with patch.object(scheduler_client.requests, "get", return_value=response):
            fetched = scheduler_client.fetch_result(
                15, attempts=1,
                expected_library_revision=TEST_LIBRARY_REVISION)
        self.assertEqual(fetched.state, scheduler_client.RESULT_INVALID)

    def test_profile_overrides_isolate_mode_and_parameter_payloads(self):
        submitted = Mock(status_code=201)
        submitted.json.side_effect = [{"id": 51}, {"id": 52}]
        profile = {"param_overrides": {"full_model": 0, "matrix_skin_mesh": 0}}
        with patch.object(
                scheduler_client.requests, "get",
                return_value=task_inventory_response([])), \
                patch.object(scheduler_client.requests, "post", return_value=submitted) as post:
            scheduler_client.submit_verification(
                "candidate-a", "shared", {"full_model": 1, "x": 1}, profile,
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
            scheduler_client.submit_verification(
                "candidate-b", "shared", {"full_model": 1, "x": 2}, profile,
                solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)

        commands = [call.kwargs["json"]["command"] for call in post.call_args_list]
        self.assertNotEqual(commands[0], commands[1])
        for command in commands:
            self.assertIn('"full_model":0', command)
            self.assertIn('"matrix_skin_mesh":0', command)
        for call in post.call_args_list:
            self.assertEqual(
                call.kwargs["json"]["timeout_seconds"],
                scheduler_client.DEFAULT_TASK_TIMEOUT_SECONDS)

    def test_profiles_override_candidate_explicit_turn_count(self):
        standard = standard_submitted_params(n_explicit_turns=4)
        fine_profile = json.loads(
            (HERE / "verify" / "profiles" / "fine.json").read_text(
                encoding="utf-8"
            )
        )
        fine = scheduler_client.effective_verification_params(
            complete_candidate_params(n_explicit_turns=4), fine_profile
        )

        self.assertEqual(standard["n_explicit_turns"], 0)
        self.assertEqual(fine["n_explicit_turns"], 2)

    def test_validity_requires_finite_candidate_specific_fields(self):
        self.assertTrue(scheduler_client.is_valid_result(
            valid_result(), expected_revision=TEST_REVISION,
            expected_library_revision=TEST_LIBRARY_REVISION))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(Llt=float("nan"))))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(T_max_core=float("nan"))))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(T_max_Rx_main=4726.85)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(Tprobe_core_center_max=4726.85)))
        for target in al_driver.CORE_REGION_TEMPERATURE_TARGETS:
            with self.subTest(core_region_target=target):
                self.assertFalse(scheduler_client.is_valid_result(
                    valid_result(**{target: 4726.85})))
                missing_region = valid_result()
                missing_region.pop(target)
                self.assertFalse(
                    scheduler_client.is_valid_result(missing_region))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(thermal_solved=0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(thermal_converged=0)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_residual_continuity=2e-3)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_residual_flow_limit=1e-2)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(full_model=2)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(full_model=1)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(loss_sym_on=0)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_symmetry="quarter")))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(conv_error_pct_matrix=1.51)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(conv_error_pct_loss=float("nan"))))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(git_hash="bbbbbbb"), expected_revision=TEST_REVISION))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(pyaedt_library_git_hash="c" * 40),
            expected_library_revision=TEST_LIBRARY_REVISION))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(pyaedt_library_git_dirty=1)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(matrix_on=0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(loss_on=0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(thermal_on=0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(matrix_skin_mesh=1)))
        missing_component = valid_result()
        missing_component.pop("P_Rx_main_group")
        self.assertFalse(scheduler_client.is_valid_result(missing_component))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(P_Tx_main_group=2401.0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(P_target=0)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(matrix_max_passes=1)))
        self.assertFalse(scheduler_client.is_valid_result(valid_result(git_dirty=1)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(matrix_solve_attempts=2)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(matrix_extraction_backend="get_solution_data")))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(matrix_conductor_policy="solid_skin")))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(loss_winding_solid_update_count=1)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_rx_power_balance_ok=0)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_rx_assigned_power_w=119.0)))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(thermal_rx_model="hybrid_explicit")))
        self.assertFalse(scheduler_client.is_valid_result(
            valid_result(n_explicit_turns=2)))

        no_side = valid_result(N2_side=0)
        no_side.pop("T_max_Rx_side")
        no_side.pop("Tprobe_Rx_side_leeward_max")
        self.assertTrue(scheduler_client.is_valid_result(no_side))

        missing_required_side = valid_result()
        missing_required_side.pop("T_max_Rx_side")
        self.assertFalse(scheduler_client.is_valid_result(missing_required_side))


class WaitStateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self._training_patcher = patch.object(
            al_driver, "_assert_training_invariants"
        )
        self.training_invariant = self._training_patcher.start()
        self.addCleanup(self._training_patcher.stop)

    def _context(self, root):
        profile_path = root / "verify" / "profiles" / "standard.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            (HERE / "verify" / "profiles" / "standard.json").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        return patch.multiple(
            al_driver,
            HERE=str(root),
            save_state=Mock(),
            _require_runtime_deployment=Mock(),
        )

    def test_submit_pins_one_solver_revision_in_state_and_task_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rdir = root / "al_rounds" / "round_01"
            rdir.mkdir(parents=True)
            np.save(rdir / "selected_idx.npy", np.array([0]))
            profile = root / "verify" / "profiles" / "standard.json"
            profile.parent.mkdir(parents=True)
            profile.write_text(
                (HERE / "verify" / "profiles" / "standard.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            state = {"round": 1, "stage": "SUBMIT"}
            with patch.object(al_driver, "HERE", str(root)), \
                    patch.object(al_driver, "_assert_training_invariants"), \
                    patch.object(al_driver, "_current_solver_revision", return_value=TEST_REVISION), \
                    patch.object(
                        al_driver, "_current_library_revision",
                        return_value=TEST_LIBRARY_REVISION), \
                    patch.object(
                        al_driver.pd, "read_csv",
                        return_value=pd.DataFrame([complete_candidate_params(l1=50.0)])), \
                    patch.object(al_driver, "save_state") as save, \
                    patch.object(al_driver.time, "sleep"), \
                    patch.object(al_driver, "EXECUTE_SUBMISSIONS", True), \
                    patch.object(al_driver, "_require_runtime_deployment"), \
                    patch.object(
                        scheduler_client, "submit_verification", return_value=17) as submit:
                al_driver.stage_submit(state)

        self.assertEqual(state["solver_git_revision"], TEST_REVISION)
        self.assertEqual(
            state["pyaedt_library_git_revision"], TEST_LIBRARY_REVISION)
        self.assertEqual(
            state["task_records"]["0"]["solver_git_revision"], TEST_REVISION)
        self.assertEqual(
            state["task_records"]["0"]["pyaedt_library_git_revision"],
            TEST_LIBRARY_REVISION)
        self.assertEqual(submit.call_args.kwargs["solver_revision"], TEST_REVISION)
        self.assertEqual(
            submit.call_args.kwargs["library_revision"], TEST_LIBRARY_REVISION)
        self.assertEqual(state["stage"], "WAIT")
        self.assertGreaterEqual(save.call_count, 2)

    def test_active_or_unknown_task_is_never_retried(self):
        for task_status in ("running", "queued", None):
            with self.subTest(task_status=task_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = task_state()
                with self._context(root), \
                        patch.object(al_driver.pd, "read_csv", return_value=pd.DataFrame({"unused": [1]})), \
                        patch.object(scheduler_client, "wait_all", return_value={17: task_status}), \
                        patch.object(scheduler_client, "fetch_result") as fetch, \
                        patch.object(scheduler_client, "submit_verification") as submit:
                    al_driver.stage_wait(state)

                self.assertEqual(state["stage"], "WAIT")
                self.assertEqual(state["task_records"]["0"]["original_id"], 17)
                self.assertIsNone(state["task_records"]["0"]["retry_id"])
                fetch.assert_not_called()
                submit.assert_not_called()

    def test_terminal_invalid_result_retries_once_and_persists_both_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = task_state()
            invalid = scheduler_client.ResultFetch(
                scheduler_client.RESULT_INVALID,
                valid_result(result_valid_thermal=0),
            )
            with self._context(root), \
                    patch.object(
                        al_driver.pd, "read_csv",
                        return_value=pd.DataFrame([complete_candidate_params(l1=50.0)])), \
                    patch.object(scheduler_client, "wait_all", return_value={17: "completed"}), \
                    patch.object(scheduler_client, "fetch_result", return_value=invalid), \
                    patch.object(scheduler_client, "submit_verification", return_value=18) as submit:
                al_driver.stage_wait(state)

            record = state["task_records"]["0"]
            self.assertEqual(record["original_id"], 17)
            self.assertEqual(record["retry_id"], 18)
            self.assertEqual(record["active_id"], 18)
            self.assertEqual(record["attempt"], 1)
            self.assertEqual(state["task_map"], {"0": 18})
            self.assertEqual(state["stage"], "WAIT")
            self.assertEqual(submit.call_args.kwargs["mem_mb"], 65536)

            with self._context(root), \
                    patch.object(
                        al_driver.pd, "read_csv",
                        return_value=pd.DataFrame([complete_candidate_params(l1=50.0)])), \
                    patch.object(scheduler_client, "wait_all", return_value={18: "completed"}), \
                    patch.object(
                        scheduler_client, "fetch_result",
                        return_value=scheduler_client.ResultFetch(
                            scheduler_client.RESULT_VALID,
                            valid_result(**standard_submitted_params(l1=50.0)))), \
                    patch.object(scheduler_client, "submit_verification") as second_submit:
                al_driver.stage_wait(state)

            self.assertEqual(state["stage"], "INGEST")
            self.assertEqual(record["outcome"], "valid")
            self.assertEqual(record["result"]["Llt"], 13.75)
            second_submit.assert_not_called()

    def test_invalid_retry_is_exhausted_without_a_third_submission(self):
        record = al_driver._new_task_record(
            17, solver_revision=TEST_REVISION,
            library_revision=TEST_LIBRARY_REVISION)
        record.update({"retry_id": 18, "active_id": 18, "attempt": 1})
        state = {
            "round": 1,
            "stage": "WAIT",
            "task_map": {"0": 18},
            "task_records": {"0": record},
            "solver_git_revision": TEST_REVISION,
            "pyaedt_library_git_revision": TEST_LIBRARY_REVISION,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._context(root), \
                    patch.object(al_driver.pd, "read_csv", return_value=pd.DataFrame({"unused": [1]})), \
                    patch.object(scheduler_client, "wait_all", return_value={18: "failed"}), \
                    patch.object(
                        scheduler_client, "fetch_result",
                        return_value=scheduler_client.ResultFetch(scheduler_client.RESULT_MISSING)), \
                    patch.object(scheduler_client, "submit_verification") as submit:
                al_driver.stage_wait(state)

        self.assertEqual(record["outcome"], "exhausted")
        self.assertEqual(state["stage"], "INGEST")
        submit.assert_not_called()

    def test_transport_failure_does_not_trigger_solver_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = task_state()
            with self._context(root), \
                    patch.object(al_driver.pd, "read_csv", return_value=pd.DataFrame({"unused": [1]})), \
                    patch.object(scheduler_client, "wait_all", return_value={17: "completed"}), \
                    patch.object(
                        scheduler_client, "fetch_result",
                        side_effect=scheduler_client.ResultFetchError("offline")), \
                    patch.object(scheduler_client, "submit_verification") as submit:
                with self.assertRaisesRegex(RuntimeError, "no tasks were resubmitted"):
                    al_driver.stage_wait(state)

        self.assertEqual(state["stage"], "WAIT")
        self.assertEqual(state["task_records"]["0"]["outcome"], "fetch_error")
        submit.assert_not_called()

    def test_unknown_retry_identity_never_overwrites_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = task_state()
            with self._context(root), \
                    patch.object(al_driver.pd, "read_csv", return_value=pd.DataFrame({"unused": [1]})), \
                    patch.object(scheduler_client, "wait_all", return_value={17: "completed"}), \
                    patch.object(
                        scheduler_client, "fetch_result",
                        return_value=scheduler_client.ResultFetch(scheduler_client.RESULT_MISSING)), \
                    patch.object(scheduler_client, "submit_verification", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "unknown task identity"):
                    al_driver.stage_wait(state)

        record = state["task_records"]["0"]
        self.assertEqual(record["original_id"], 17)
        self.assertEqual(record["active_id"], 17)
        self.assertIsNone(record["retry_id"])
        self.assertEqual(state["task_map"], {"0": 17})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._context(root), \
                    patch.object(scheduler_client, "wait_all") as wait, \
                    patch.object(scheduler_client, "submit_verification") as submit:
                with self.assertRaisesRegex(RuntimeError, "cannot safely resubmit"):
                    al_driver.stage_wait(state)
        wait.assert_not_called()
        submit.assert_not_called()


class IngestIntegrityTests(unittest.TestCase):
    def setUp(self):
        self._training_patcher = patch.object(
            al_driver, "_assert_training_invariants"
        )
        self.training_invariant = self._training_patcher.start()
        self.addCleanup(self._training_patcher.stop)

    def _front(self):
        row = complete_candidate_params(l1=50.0)
        row.update({
            "pred_Llt_phys": 27.5,
            "pred_Tprobe_Tx_leeward_max": 89.5,
            "pred_Tprobe_Rx_main_leeward_max": 90.5,
            "pred_Tprobe_Rx_side_leeward_max": 91.5,
            "pred_Tprobe_core_center_max": 92.5,
            "pred_Tprobe_core_center_leg_max": 88.5,
            "pred_Tprobe_core_side_leg_max": 90.5,
            "pred_Tprobe_core_top_yoke_max": 92.5,
            "pred_P_winding_total": 4000.0,
            "pred_P_core_total": 2000.0,
            "pred_P_core_plate_total": 500.0,
            "pred_P_wcp_total": 0.0,
        })
        return pd.DataFrame([row])

    def _state(self, result=None):
        record = al_driver._new_task_record(
            23, solver_revision=TEST_REVISION,
            library_revision=TEST_LIBRARY_REVISION)
        submitted = standard_submitted_params(l1=50.0)
        record.update({
            "outcome": "valid",
            "result": result or valid_result(**submitted),
            "submitted_params": submitted,
        })
        return {
            "round": 1,
            "stage": "INGEST",
            "task_map": {"0": 23},
            "task_records": {"0": record},
            "solver_git_revision": TEST_REVISION,
            "pyaedt_library_git_revision": TEST_LIBRARY_REVISION,
        }

    def test_ingest_uses_collector_lock_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rdir = root / "al_rounds" / "round_01"
            rdir.mkdir(parents=True)
            from module.input_parameter_260706 import _SOBOL_DIMS

            current_point = np.linspace(0.1, 0.9, len(_SOBOL_DIMS))[None, :]
            np.save(rdir / "pareto_X.npy", current_point)
            # A persisted pre-schema-extension archive must be ignored instead
            # of being broadcast/vstacked into the current unit vectors.
            np.save(root / "al_rounds" / "verified_X.npy", np.array([[0.1, 0.2]]))
            dataset = root / "data" / "dataset" / "train.parquet"
            state = self._state()
            lock_paths = []
            real_file_lock = al_driver.FileLock

            def tracked_lock(path, **kwargs):
                lock_paths.append((path, kwargs))
                return real_file_lock(path, **kwargs)

            with patch.object(al_driver, "HERE", str(root)), \
                    patch.object(al_driver, "DATASET", str(dataset)), \
                    patch.object(al_driver.pd, "read_csv", return_value=self._front()), \
                    patch.object(al_driver, "FileLock", side_effect=tracked_lock), \
                    patch.object(al_driver, "save_state"):
                al_driver.stage_ingest(state)
                state["stage"] = "INGEST"
                al_driver.stage_ingest(state)

            data = pd.read_parquet(dataset)
            verified = np.load(root / "al_rounds" / "verified_X.npy")
            ranks = pd.read_parquet(dataset.parent / "source_ranks.parquet")
            self.assertEqual(len(data), 1)
            self.assertEqual(len(verified), 1)
            self.assertEqual(len(ranks), 1)
            self.assertEqual(
                ranks.loc[0, al_driver.SOURCE_RANK_COLUMN], al_driver.AL_SOURCE_RANK)
            self.assertEqual(lock_paths, [
                (str(dataset) + ".lock", {"timeout": 120}),
                (str(dataset) + ".lock", {"timeout": 120}),
            ])
            self.assertEqual(state["last_errs"]["spec_pass"], [1])
            expected_temperature_count = len(al_driver._required_probe_targets(
                state["task_records"]["0"]["result"]
            ))
            self.assertEqual(
                state["last_errs"]["temperature_error_expected_count"],
                [expected_temperature_count],
            )
            self.assertEqual(
                sum(
                    key.startswith("d_Tprobe")
                    and len(values) == 1
                    for key, values in state["last_errs"].items()
                ),
                expected_temperature_count,
            )
            self.assertEqual(state["last_errs"]["dloss_pct"], [0.0])
            self.assertEqual(state["verification_counts"], {
                "round": 1,
                "total": 1,
                "valid": 1,
                "exhausted": 0,
                "pending": 0,
                "coverage": 1.0,
                "ingested": 1,
            })

    def test_invalid_cached_result_returns_to_wait_without_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rdir = root / "al_rounds" / "round_01"
            rdir.mkdir(parents=True)
            dataset = root / "train.parquet"
            state = self._state(valid_result(result_valid_thermal=0))

            with patch.object(al_driver, "HERE", str(root)), \
                    patch.object(al_driver, "DATASET", str(dataset)), \
                    patch.object(al_driver.pd, "read_csv", return_value=self._front()), \
                    patch.object(al_driver, "save_state"):
                al_driver.stage_ingest(state)

            self.assertFalse(dataset.exists())
            self.assertNotIn("last_errs", state)
            self.assertEqual(state["stage"], "WAIT")
            self.assertEqual(state["task_records"]["0"]["outcome"], "pending")

    def test_ingest_transport_failure_keeps_cached_identity_and_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rdir = root / "al_rounds" / "round_01"
            rdir.mkdir(parents=True)
            record = al_driver._new_task_record(
                23, solver_revision=TEST_REVISION,
                library_revision=TEST_LIBRARY_REVISION)
            state = {
                "round": 1,
                "stage": "INGEST",
                "task_map": {"0": 23},
                "task_records": {"0": record},
                "solver_git_revision": TEST_REVISION,
                "pyaedt_library_git_revision": TEST_LIBRARY_REVISION,
            }
            dataset = root / "train.parquet"
            with patch.object(al_driver, "HERE", str(root)), \
                    patch.object(al_driver, "DATASET", str(dataset)), \
                    patch.object(al_driver.pd, "read_csv", return_value=self._front()), \
                    patch.object(al_driver, "save_state"), \
                    patch.object(
                        scheduler_client, "fetch_result",
                        side_effect=scheduler_client.ResultFetchError("offline")):
                with self.assertRaisesRegex(RuntimeError, "stdout unavailable"):
                    al_driver.stage_ingest(state)

            self.assertEqual(state["stage"], "INGEST")
            self.assertEqual(record["active_id"], 23)
            self.assertEqual(record["outcome"], "fetch_error")
            self.assertFalse(dataset.exists())

    def test_atomic_parquet_failure_preserves_existing_master(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.parquet"
            pd.DataFrame({"value": [1]}).to_parquet(target, index=False)
            with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    al_driver._atomic_write_parquet(pd.DataFrame({"value": [2]}), str(target))
            self.assertEqual(pd.read_parquet(target)["value"].tolist(), [1])

    def test_physical_spec_gate_checks_model_symmetry_all_temperatures_and_flux(self):
        self.assertTrue(al_driver._result_passes_spec(valid_result()))
        self.assertTrue(al_driver._result_passes_spec(
            valid_result(full_model=1, Llt=27.5)))
        self.assertFalse(al_driver._result_passes_spec(
            valid_result(full_model=1, Llt=13.75)))
        self.assertFalse(al_driver._result_passes_spec(valid_result(Llt=float("nan"))))
        self.assertFalse(al_driver._result_passes_spec(valid_result(B_max_core=1.21)))
        self.assertFalse(al_driver._result_passes_spec(valid_result(T_max_Rx_side=100.1)))
        self.assertFalse(al_driver._result_passes_spec(
            valid_result(conv_error_pct_matrix=1.51)))
        self.assertFalse(al_driver._result_passes_spec(
            valid_result(conv_error_pct_loss=float("nan"))))

    def test_al_source_rank_replaces_local_rank_but_preserves_higher_rank(self):
        row = pd.DataFrame([valid_result()])
        local = pd.DataFrame([{
            "project_name": "simulation-test",
            "saved_at": "2026-07-10 07:00:00",
            al_driver.SOURCE_RANK_COLUMN: 40,
        }])
        merged = al_driver._merge_source_ranks(local, row)
        self.assertEqual(merged.loc[0, al_driver.SOURCE_RANK_COLUMN], 50)

        higher = local.copy()
        higher[al_driver.SOURCE_RANK_COLUMN] = 60
        merged = al_driver._merge_source_ranks(higher, row)
        self.assertEqual(merged.loc[0, al_driver.SOURCE_RANK_COLUMN], 60)

        with self.assertRaisesRegex(RuntimeError, "sidecar schema"):
            al_driver._merge_source_ranks(pd.DataFrame({"bad": [1]}), row)

    def test_rank_aware_master_merge_preserves_higher_authority_payload(self):
        incoming = pd.DataFrame([valid_result(payload="al")])
        existing = pd.DataFrame([valid_result(payload="authoritative")])
        ranks = pd.DataFrame([{
            "project_name": "simulation-test",
            "saved_at": "2026-07-10 07:00:00",
            al_driver.SOURCE_RANK_COLUMN: 60,
        }])

        dataset, merged_ranks = al_driver._merge_ranked_dataset(
            existing, incoming, ranks)

        self.assertEqual(dataset.loc[0, "payload"], "authoritative")
        self.assertEqual(
            merged_ranks.loc[0, al_driver.SOURCE_RANK_COLUMN], 60)

    def test_rank_aware_master_merge_rejects_missing_or_malformed_sidecar(self):
        incoming = pd.DataFrame([valid_result(payload="al")])
        existing = pd.DataFrame([valid_result(
            project_name="existing-only", payload="existing")])

        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            al_driver._merge_ranked_dataset(existing, incoming, None)
        with self.assertRaisesRegex(RuntimeError, "sidecar schema"):
            al_driver._merge_ranked_dataset(
                existing, incoming, pd.DataFrame({"bad": [1]}))
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            al_driver._merge_ranked_dataset(
                existing, incoming, pd.DataFrame([{
                    "project_name": "different",
                    "saved_at": "2026-07-10 07:00:00",
                    al_driver.SOURCE_RANK_COLUMN: 40,
                }]))

    def test_rank_merge_recovers_master_first_interrupted_install(self):
        old = pd.DataFrame([valid_result(
            project_name="old", payload="local")])
        old_ranks = pd.DataFrame([{
            "project_name": "old",
            "saved_at": "2026-07-10 07:00:00",
            al_driver.SOURCE_RANK_COLUMN: 40,
        }])
        incoming = pd.DataFrame([valid_result(
            project_name="new-al", payload="al")])
        installed_master, intended_ranks = al_driver._merge_ranked_dataset(
            old, incoming, old_ranks)

        replayed_master, replayed_ranks = al_driver._merge_ranked_dataset(
            installed_master, incoming, old_ranks)

        self.assertEqual(len(replayed_master), 2)
        self.assertEqual(
            replayed_ranks.set_index("project_name").loc[
                "new-al", al_driver.SOURCE_RANK_COLUMN],
            al_driver.AL_SOURCE_RANK)
        self.assertEqual(
            intended_ranks.sort_values("project_name").reset_index(drop=True).to_dict("records"),
            replayed_ranks.sort_values("project_name").reset_index(drop=True).to_dict("records"))
        with self.assertRaisesRegex(RuntimeError, "payload differs"):
            al_driver._merge_ranked_dataset(
                installed_master, incoming.assign(payload="changed"), old_ranks)

    def test_dataset_transaction_installs_master_before_rank_sidecar(self):
        dataset = pd.DataFrame([valid_result()])
        ranks = pd.DataFrame([{
            "project_name": "simulation-test",
            "saved_at": "2026-07-10 07:00:00",
            al_driver.SOURCE_RANK_COLUMN: 50,
        }])
        with patch.object(
                al_driver, "_stage_parquet",
                side_effect=["dataset.tmp", "ranks.tmp"]), patch.object(
                    al_driver, "DATASET", "train.parquet"), patch.object(
                        al_driver.os, "replace") as replace:
            al_driver._install_dataset_transaction(
                dataset, ranks, "source_ranks.parquet")

        self.assertEqual(replace.call_args_list, [
            unittest.mock.call("dataset.tmp", "train.parquet"),
            unittest.mock.call("ranks.tmp", "source_ranks.parquet"),
        ])

    def test_selector_fills_documented_k33_batch(self):
        rng = np.random.default_rng(7)
        X = rng.random((49, 6))
        F = rng.random((49, 2))
        G = -rng.random((49, 2))
        sigma = rng.random(49)

        picked = select_candidates.select(X, F, G, sigma)

        self.assertEqual(len(picked), al_driver.SPEC["K"])
        self.assertEqual(len(set(picked)), al_driver.SPEC["K"])

    def test_collector_cannot_replace_rank_50_al_row_with_local_part(self):
        from regression_260707.campaign import collect_wave

        old = pd.DataFrame([valid_result(sample_weight=3.0)])
        ranks = pd.DataFrame([{
            "project_name": "simulation-test",
            "saved_at": "2026-07-10 07:00:00",
            collect_wave.SOURCE_RANK_COLUMN: al_driver.AL_SOURCE_RANK,
        }])
        ranked_old = collect_wave._attach_source_ranks(
            old, ranks, ["project_name", "saved_at"])
        incoming = old.copy()
        incoming["sample_weight"] = 1.0
        incoming[collect_wave.SOURCE_RANK_COLUMN] = collect_wave.SOURCE_RANK_LOCAL_PART
        selected = collect_wave.select_new_unique_rows(
            incoming, ranked_old, ["project_name", "saved_at"])
        self.assertTrue(selected.empty)


class ActiveLearningGateTests(unittest.TestCase):
    def setUp(self):
        self._training_patcher = patch.object(
            al_driver, "_assert_training_invariants"
        )
        self.training_invariant = self._training_patcher.start()
        self.addCleanup(self._training_patcher.stop)

    def test_check_cannot_agree_with_incomplete_temperature_coverage(self):
        errors = complete_errors()
        errors.pop("d_Tprobe_Rx_main_leeward_max")
        state = {
            "round": 1, "q_mult": 1.0, "history": [], "last_errs": errors,
            "verification_counts": verification_counts(),
        }

        al_driver.stage_check(state)

        history = state["history"][-1]
        self.assertFalse(history["temperature_error_coverage_complete"])
        self.assertEqual(state["stage"], "TRAIN")
        self.assertEqual(state["round"], 2)

    def test_check_cannot_converge_when_actual_spec_fails(self):
        state = {
            "round": 1,
            "q_mult": 1.0,
            "history": [],
            "last_errs": complete_errors(spec_pass=(1, 1, 0)),
            "verification_counts": verification_counts(),
        }

        al_driver.stage_check(state)

        self.assertEqual(state["history"][-1]["fea_full_spec_pass"], 2)
        self.assertEqual(state["stage"], "TRAIN")

    def test_check_converges_with_complete_agreement_and_three_full_spec_passes(self):
        errors = complete_errors()
        repeat = al_driver.SPEC["K"] // 3
        remainder = al_driver.SPEC["K"] % 3
        errors = {
            key: values * repeat + values[:remainder]
            for key, values in errors.items()
        }
        errors["spec_pass"] = [1, 1, 1] + [0] * (al_driver.SPEC["K"] - 3)
        state = {
            "round": 1,
            "q_mult": 1.0,
            "history": [],
            "last_errs": errors,
            "verification_counts": verification_counts(
                total=al_driver.SPEC["K"], valid=al_driver.SPEC["K"]),
        }

        al_driver.stage_check(state)

        history = state["history"][-1]
        temperature_target_count = len(
            al_driver.MANDATORY_SURROGATE_TEMPERATURE_TARGETS
        )
        self.assertEqual(
            history["temperature_error_count"],
            temperature_target_count * al_driver.SPEC["K"],
        )
        self.assertEqual(
            history["temperature_error_expected_count"],
            temperature_target_count * al_driver.SPEC["K"],
        )
        self.assertTrue(history["temperature_error_coverage_complete"])
        self.assertTrue(history["loss_error_coverage_complete"])
        self.assertTrue(history["verification_coverage_ok"])
        self.assertTrue(history["verification_rows_complete"])
        self.assertEqual(history["fea_full_spec_pass"], 3)
        self.assertEqual(state["stage"], "TRAIN")
        self.assertTrue(state["post_convergence_retrain_done"])
        self.assertEqual(state["round"], 2)

    def test_hard_cap_is_not_reported_as_verified_done(self):
        state = {
            "round": al_driver.SPEC["max_rounds"],
            "q_mult": 1.0,
            "history": [],
            "last_errs": complete_errors(spec_pass=(0, 0, 0)),
            "verification_counts": {
                **verification_counts(),
                "round": al_driver.SPEC["max_rounds"],
            },
        }

        al_driver.stage_check(state)

        self.assertEqual(state["stage"], "HARD_CAP")

    def test_three_valid_of_thirty_three_cannot_finish(self):
        state = {
            "round": 1,
            "q_mult": 1.0,
            "history": [],
            "last_errs": complete_errors(),
            "verification_counts": verification_counts(
                total=33, valid=3, exhausted=30),
        }

        al_driver.stage_check(state)

        self.assertFalse(state["history"][-1]["verification_coverage_ok"])
        self.assertEqual(state["stage"], "TRAIN")

    def test_claimed_coverage_without_all_valid_rows_cannot_finish(self):
        state = {
            "round": 1,
            "q_mult": 1.0,
            "history": [],
            "last_errs": complete_errors(),
            "verification_counts": verification_counts(
                total=33, valid=24, exhausted=9),
        }

        al_driver.stage_check(state)

        history = state["history"][-1]
        self.assertTrue(history["verification_coverage_ok"])
        self.assertFalse(history["verification_rows_complete"])
        self.assertEqual(state["stage"], "TRAIN")

    def test_loss_agreement_requires_complete_finite_rows_and_max_bound(self):
        for dloss in ([1.0, 2.0], [1.0, 2.0, 6.0]):
            with self.subTest(dloss=dloss):
                errors = complete_errors()
                errors["dloss_pct"] = dloss
                state = {
                    "round": 1,
                    "q_mult": 1.0,
                    "history": [],
                    "last_errs": errors,
                    "verification_counts": verification_counts(),
                }
                al_driver.stage_check(state)
                self.assertEqual(state["stage"], "TRAIN")


if __name__ == "__main__":
    unittest.main()
