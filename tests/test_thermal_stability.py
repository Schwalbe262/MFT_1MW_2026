import json
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from module import thermal_260706 as thermal


class _Object:
    def __init__(
        self,
        name,
        volume=1.0,
        bounding_box=(0.0, 0.0, 0.0, 100.0, 1.0, 100.0),
    ):
        self.name = name
        self.is3d = True
        self.volume = volume
        self.bounding_box = list(bounding_box)


class _NoLiveDimensionObject:
    """Object identity whose editor-backed dimension must never be queried."""

    def __init__(self, name, volume=1.0):
        self.name = name
        self.volume = volume

    @property
    def is3d(self):
        raise AssertionError("post-solve Object3d.is3d query is forbidden")


class _Material:
    pass


class _Materials:
    def __init__(self):
        self.material_keys = {}

    def add_material(self, name):
        material = _Material()
        self.material_keys[name] = material
        return material

    def __getitem__(self, name):
        return self.material_keys[name]


class _Boundary:
    def __init__(self, props=None, update_result=True, post_update_props=None):
        self.props = props or {}
        self.auto_update = True
        self.update_result = update_result
        self.post_update_props = post_update_props or {}
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        self.props.update(self.post_update_props)
        return self.update_result


class _DesignHandle:
    def __init__(self, name):
        self._name = name
        self.analyze_calls = []

    def GetName(self):
        return self._name

    def GetDesignType(self):
        return "Icepak"

    def GetModule(self, name):
        if name == "AnalysisSetup":
            return SimpleNamespace(GetSetups=lambda: ["ThermalSetup"])
        return SimpleNamespace()

    def Analyze(self, setup_name, blocking):
        self.analyze_calls.append((setup_name, blocking))
        return 0


class _ProjectHandle:
    def __init__(self, name="thermal_test"):
        self._name = name
        self.active_calls = 0
        self.last_design = None

    def GetName(self):
        return self._name

    def SetActiveDesign(self, name):
        self.active_calls += 1
        self.last_design = _DesignHandle(name)
        return self.last_design


class _SingleActivationProjectHandle(_ProjectHandle):
    """Models a native project proxy that becomes stale while Analyze runs."""

    def SetActiveDesign(self, name):
        if self.active_calls:
            raise RuntimeError("stale q7 project proxy reused after native solve")
        return super().SetActiveDesign(name)


class _Setup:
    def __init__(self, update_result=True):
        self.name = "ThermalSetup"
        self.props = {}
        self.update_result = update_result
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        return self.update_result


class _Face:
    def __init__(self, face_id):
        self.id = face_id


class _Region:
    def __init__(self):
        self.top_face_x = _Face(1)
        self.bottom_face_x = _Face(2)
        self.top_face_y = _Face(3)
        self.bottom_face_y = _Face(4)
        self.top_face_z = _Face(5)
        self.bottom_face_z = _Face(6)


class _BoundaryIcepak:
    def __init__(
        self,
        fail_boundary=None,
        fixed_temperature=None,
        fixed_update_result=True,
        fixed_post_update_props=None,
    ):
        self.fail_boundary = fail_boundary
        self.fixed_temperature = fixed_temperature
        self.fixed_update_result = fixed_update_result
        self.fixed_post_update_props = fixed_post_update_props
        self.source_calls = []
        self.source_boundaries = []
        self.ambient_calls = []
        self.region = _Region()
        self.modeler = SimpleNamespace(
            object_names=[],
            delete=Mock(return_value=True),
            create_air_region=Mock(return_value=self.region),
        )

    def _result(self, name):
        return False if name == self.fail_boundary else _Boundary()

    def assign_source(self, **kwargs):
        self.source_calls.append(kwargs)
        if kwargs["boundary_name"] == self.fail_boundary:
            return False
        temperature = self.fixed_temperature or kwargs["assignment_value"]
        boundary = _Boundary({
            "Objects": list(kwargs["assignment"]),
            "Thermal Condition": kwargs["thermal_condition"],
            "Temperature": temperature,
        }, self.fixed_update_result, self.fixed_post_update_props)
        self.source_boundaries.append(boundary)
        return boundary

    def set_ambient_temp(self, value):
        self.ambient_calls.append(value)

    def assign_symmetry_wall(self, **kwargs):
        return self._result(kwargs["boundary_name"])

    def assign_velocity_free_opening(self, **kwargs):
        return self._result(kwargs["boundary_name"])

    def assign_pressure_free_opening(self, **kwargs):
        return self._result(kwargs["boundary_name"])


class _FieldSummary:
    def __init__(self, icepak):
        self._icepak = icepak
        self._names = []

    def add_calculation(self, _entity, _geometry, name, _quantity):
        self._names.append(name)

    def _next_rows(self):
        index = self._icepak.field_summary_calls
        self._icepak.field_summary_calls += 1
        response = self._icepak.field_summary_responses[min(index, len(self._icepak.field_summary_responses) - 1)]
        if response is False:
            return False
        rows = []
        for name in self._names:
            if name not in response:
                continue
            maximum, mean = response[name]
            rows.append({
                "Entity": "Object",
                "Quantity": "Temperature",
                "Geometry Name": name,
                "Max": maximum,
                "Mean": mean,
            })
        return rows

    def get_field_summary_data(self, **_kwargs):
        rows = self._next_rows()
        if rows is False:
            return False
        return pd.DataFrame(rows)

    def export_csv(self, output_file, setup=None):
        rows = self._next_rows()
        self._icepak.field_summary_export_call = (output_file, setup)
        if rows is False:
            return False
        columns = ["Entity", "Quantity", "Geometry Name", "Max", "Mean"]
        body = [",".join(columns)]
        body.extend(
            ",".join(str(row.get(column, "")) for column in columns)
            for row in rows
        )
        self._icepak.field_summary_export_text = (
            "Field Summary\nProject\nDesign\nSolution\n"
            + "\n".join(body)
            + "\n"
        )
        return True


class _Icepak:
    def __init__(
        self, analyze_result, field_summary_responses, setup_result=True,
        setup_update_result=True, scalar_responses=None,
    ):
        self.analyze_result = analyze_result
        self.analyze_calls = 0
        self.field_summary_calls = 0
        self.field_summary_responses = field_summary_responses
        self.field_summary_export_call = None
        self.field_summary_export_text = ""
        self.scalar_responses = scalar_responses or {}
        self.scalar_calls = []
        self.setup_result = setup_result
        self.setup_update_result = setup_update_result
        self.design_name = "icepak_thermal"
        self.setup_names = ["ThermalSetup"]
        self.existing_analysis_sweeps = ["ThermalSetup : SteadyState"]
        self.mesh = SimpleNamespace(assign_mesh_level=Mock())
        self.post = SimpleNamespace(
            create_field_summary=lambda: _FieldSummary(self),
            get_scalar_field_value=self._get_scalar_field_value,
        )
        self._oproject = _ProjectHandle()
        self.odesktop = SimpleNamespace(
            AreThereSimulationsRunning=lambda: False,
            GetMessages=lambda *_args: [],
        )
        self._odesign = _DesignHandle("stale_design")
        self.design_solutions = SimpleNamespace(_odesign=self._odesign)

    @property
    def oproject(self):
        # PyAEDT exposes the native project through its refreshed private
        # handle; model that indirection so a postflight rebind is observable.
        return self._oproject

    def _get_scalar_field_value(self, _quantity, **kwargs):
        key = (kwargs["object_name"], kwargs["scalar_function"])
        self.scalar_calls.append((key, kwargs["object_type"], kwargs["solution"]))
        value = self.scalar_responses.get(key)
        if isinstance(value, Exception):
            raise value
        return value

    def create_setup(self, name):
        if not self.setup_result:
            return False
        self.setup = _Setup(self.setup_update_result)
        self.setup.name = name
        return self.setup

    def analyze(self, **_kwargs):
        index = self.analyze_calls
        self.analyze_calls += 1
        result = self.analyze_result
        if isinstance(result, (list, tuple)):
            result = result[min(index, len(result) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


class _DesignWrapper:
    def __init__(self, solver):
        self.solver_instance = solver

    def __getattr__(self, name):
        return getattr(self.solver_instance, name)


class ThermalStabilityTest(unittest.TestCase):
    def test_blocking_solve_logs_visible_start_and_end_markers(self):
        with patch.dict(
            os.environ, {"MFT_AEDT_SOLVE_HEARTBEAT_SECONDS": "5"}
        ), self.assertLogs(level="WARNING") as captured:
            with thermal._blocking_solve_log_heartbeat(
                "project-a", "icepak_thermal", "ThermalSetup"
            ):
                pass

        output = "\n".join(captured.output)
        self.assertIn("pooled blocking solve started", output)
        self.assertIn("project=project-a", output)
        self.assertIn("pooled blocking solve ended", output)

    @staticmethod
    def _strict_parallel_core_policy(total=16):
        return {
            "opt_in": True,
            "backend": "standalone",
            "requested_num_cores": total,
            "effective_num_cores": total,
            "num_tasks": 1,
            "slurm_cpus_per_task_readback": total,
            "affinity_count_readback": total,
        }

    @staticmethod
    def _attestor_process(
        pid,
        *,
        uid=1001,
        ppid=1,
        create_time=1.0,
        argv=None,
        name="python",
    ):
        process = Mock()
        process.pid = pid
        process.uids.return_value = SimpleNamespace(
            real=uid,
            effective=uid,
        )
        process.ppid.return_value = ppid
        process.create_time.return_value = create_time
        process.cmdline.return_value = list(argv or [name])
        process.name.return_value = name
        return process

    @staticmethod
    def _attestor_cgroup(job_id, step_id, leaf="task_0"):
        return (
            "0::/slurm/uid_1001/"
            f"job_{job_id}/step_{step_id}/{leaf}\n"
        )

    def test_standalone_icepak_maps_allocation_to_total_fluent_processes(self):
        sim = SimpleNamespace(
            NUM_CORE=16,
            NUM_TASK=1,
            solver_core_policy=self._strict_parallel_core_policy(),
        )

        policy = thermal._standalone_thermal_parallel_policy(sim)

        self.assertEqual(policy["pyaedt_cores_argument"], 16)
        self.assertEqual(policy["pyaedt_tasks_argument"], 1)
        self.assertFalse(policy["pyaedt_use_auto_settings_argument"])
        self.assertEqual(policy["expected_fluent_processes"], 16)
        self.assertEqual(policy["expected_num_engines"], 1)
        self.assertEqual(policy["maxwell_num_engines_unchanged"], 1)
        self.assertEqual(sim.NUM_TASK, 1)

    def test_standalone_icepak_rejects_maxwell_engine_count_change(self):
        sim = SimpleNamespace(
            NUM_CORE=16,
            NUM_TASK=16,
            solver_core_policy=self._strict_parallel_core_policy(),
        )

        with self.assertRaisesRegex(
                RuntimeError, "one-engine Maxwell contract"):
            thermal._standalone_thermal_parallel_policy(sim)

    def test_fluent_command_parser_reads_t_and_rps_counts(self):
        cortex = (
            "/ansys_inc/v252/fluent/bin/fluent 3ddp -t16 "
            "-r25.2.0 -driver null"
        )
        rps = (
            "/ansys_inc/v252/fluent/bin/cortex "
            "nprocs_string='16' mode=3ddp"
        )

        self.assertEqual(
            thermal._fluent_process_counts(cortex),
            {"thread_counts": [16], "nprocs_counts": []},
        )
        self.assertEqual(
            thermal._fluent_process_counts(rps),
            {"thread_counts": [], "nprocs_counts": [16]},
        )

    def test_exact_icepak_acf_readback_requires_one_engine_sixteen_cores(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "pyaedt_config.acf")
            before = {"path": str(path), "exists": False}
            path.write_text(
                "\n".join([
                    "$begin 'DSOConfig'",
                    "ConfigName='pyaedt_config'",
                    "DesignType='Icepak'",
                    "MachineName='localhost'",
                    "NumEngines=1",
                    "NumCores=16",
                    "NumGPUs=0",
                    "UseAutoSettings=False",
                    "$end 'DSOConfig'",
                    "",
                ]),
                encoding="utf-8",
            )
            native_ipk = SimpleNamespace(working_directory=root)
            policy = {
                "expected_fluent_processes": 16,
                "expected_num_engines": 1,
            }

            evidence = thermal._validated_thermal_hpc_acf(
                native_ipk, policy, before
            )

            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["num_engines_readback"], 1)
            self.assertEqual(evidence["num_cores_readback"], 16)

            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "NumEngines=1", "NumEngines=16"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    RuntimeError, "NumEngines"):
                thermal._validated_thermal_hpc_acf(
                    native_ipk, policy, before
                )

    def test_process_attestor_rejects_any_nonmatching_t_count(self):
        attestor = thermal._StandaloneThermalProcessAttestor(16)
        attestor._scan_count = 1
        attestor._successful_scan_count = 1
        attestor._records = {
            "wrong": {
                "pid": 20,
                "ppid": 10,
                "create_time": 1.0,
                "name": "3ddp_host",
                "argv": ["3ddp_host", "-t1"],
                "commandline": "3ddp_host -t1",
                "thread_counts": [1],
                "nprocs_counts": [],
            },
        }

        with self.assertRaisesRegex(
                RuntimeError, "process count mismatch"):
            attestor.finish()

    def test_process_attestor_scope_fails_closed_without_exact_slurm_step(self):
        attestor = thermal._StandaloneThermalProcessAttestor(
            8,
            root_pid=10,
        )
        with patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "",
                "SLURM_STEP_ID": "",
                "SLURM_STEPID": "",
            },
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "requires SLURM_JOB_ID"):
            attestor.start()

        with patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "838192", "SLURM_STEP_ID": "0"},
            clear=False,
        ), patch.object(
            thermal.os,
            "geteuid",
            return_value=1001,
            create=True,
        ), patch.object(
            thermal,
            "_read_process_cgroup",
            return_value="0::/user.slice/session.scope\n",
        ), self.assertRaisesRegex(RuntimeError, "exact Slurm step cgroup"):
            attestor.start()

    def test_process_attestor_includes_reparented_exact_step_only(self):
        root = self._attestor_process(
            10,
            ppid=9,
            create_time=10.0,
        )
        old_fluent = self._attestor_process(
            20,
            create_time=20.0,
            argv=["fluent", "3ddp", "-t8"],
            name="fluent",
        )
        reused_pid_fluent = self._attestor_process(
            20,
            create_time=21.0,
            argv=["fluent", "3ddp", "-t8"],
            name="fluent",
        )
        runtime_without_count = self._attestor_process(
            21,
            create_time=21.5,
            argv=["cortex", "3ddp"],
            name="cortex",
        )
        other_step = self._attestor_process(
            22,
            create_time=22.0,
            argv=["fluent", "3ddp", "-t8"],
            name="fluent",
        )
        other_job = self._attestor_process(
            23,
            create_time=23.0,
            argv=["fluent", "3ddp", "-t8"],
            name="fluent",
        )
        other_uid = self._attestor_process(
            24,
            uid=1002,
            create_time=24.0,
            argv=["fluent", "3ddp", "-t8"],
            name="fluent",
        )
        cgroups = {
            10: self._attestor_cgroup("838192", "0"),
            20: self._attestor_cgroup("838192", "0", "task_1"),
            21: self._attestor_cgroup("838192", "0", "task_1"),
            22: self._attestor_cgroup("838192", "1"),
            23: self._attestor_cgroup("838193", "0"),
            24: self._attestor_cgroup("838192", "0"),
        }
        attestor = thermal._StandaloneThermalProcessAttestor(
            8,
            root_pid=10,
        )
        with patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "838192", "SLURM_STEP_ID": "0"},
            clear=False,
        ), patch.object(
            thermal.os,
            "geteuid",
            return_value=1001,
            create=True,
        ), patch.object(
            thermal,
            "_read_process_cgroup",
            side_effect=lambda pid: cgroups[int(pid)],
        ), patch(
            "psutil.process_iter",
            side_effect=[
                [root, old_fluent],
                [
                    root,
                    reused_pid_fluent,
                    runtime_without_count,
                    other_step,
                    other_job,
                    other_uid,
                ],
            ],
        ):
            attestor._scope = attestor._capture_scope()
            baseline = attestor._scoped_processes()
            attestor._baseline = {
                attestor._identity(record): record
                for record in baseline.values()
            }
            attestor._scan_once()
            evidence = attestor.finish()

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["scope"]["slurm_job_id"], "838192")
        self.assertEqual(evidence["scope"]["slurm_step_id"], "0")
        self.assertEqual(
            {record["pid"] for record in evidence["runtime_commands"]},
            {20, 21},
        )
        self.assertEqual(
            [record["pid"] for record in evidence["explicit_commands"]],
            [20],
        )
        self.assertEqual(evidence["commands"], evidence["explicit_commands"])
        self.assertEqual(evidence["thread_count_readbacks"], [8])
        self.assertEqual(evidence["runtime_commands"][0]["ppid"], 1)

    def test_strict_standalone_dispatch_requests_and_attests_t16(self):
        from module import aedt_pool_adapter

        sim = SimpleNamespace(
            NUM_CORE=16,
            NUM_TASK=1,
            solver_core_policy=self._strict_parallel_core_policy(),
            solver_may_be_running=False,
            _attest_cached_native_desktop=Mock(),
            save_project=Mock(),
        )
        native_ipk = SimpleNamespace(analyze=Mock(return_value=True))
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        convergence = self._convergence()
        convergence["_thermal_completion_poll"] = {
            "desktop_attested": True,
            "last_running": False,
            "timed_out": False,
            "outcome": "terminal_fresh_monitor",
        }
        process_evidence = {
            "schema": "thermal-fluent-process-command-attestation-v1",
            "passed": True,
            "thread_count_readbacks": [16, 16],
            "nprocs_count_readbacks": [16],
        }
        acf_evidence = {
            "schema": "thermal-icepak-hpc-acf-readback-v1",
            "passed": True,
            "num_cores_readback": 16,
            "num_engines_readback": 1,
        }
        attestor = Mock()
        attestor.start.return_value = attestor
        attestor.finish.return_value = process_evidence

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled",
                return_value=False), patch.object(
                    thermal, "_standalone_thermal_completion_timeout",
                    return_value=5.0), patch.object(
                    thermal, "_prepare_thermal_dispatch",
                    return_value=preflight), patch.object(
                    thermal, "_snapshot_thermal_monitors",
                    return_value={}), patch.object(
                    thermal, "_thermal_desktop_handle",
                    return_value=SimpleNamespace(
                        GetMessages=Mock(return_value=[])
                    )), patch.object(
                    thermal, "_poll_thermal_dispatch_evidence",
                    return_value=(convergence, False, "")), patch.object(
                    thermal, "_thermal_hpc_acf_snapshot",
                    return_value={
                        "path": "/tmp/pyaedt_config.acf",
                        "exists": False,
                    }), patch.object(
                    thermal, "_validated_thermal_hpc_acf",
                    return_value=acf_evidence), patch.object(
                    thermal, "_StandaloneThermalProcessAttestor",
                    return_value=attestor):
            result = thermal._solve_exact_thermal_setup(
                sim, SimpleNamespace(), SimpleNamespace()
            )

        native_ipk.analyze.assert_called_once_with(
            setup="ThermalSetup",
            blocking=True,
            cores=16,
            tasks=1,
            gpus=0,
            use_auto_settings=False,
        )
        attestor.start.assert_called_once_with()
        attestor.finish.assert_called_once_with()
        self.assertTrue(result["analyze_call_ok"])
        self.assertTrue(sim.thermal_parallel_evidence["passed"])
        forensic = json.loads(result["forensic_json"])
        parallel = forensic["attempts"][0]["parallel_attestation"]
        self.assertTrue(parallel["passed"])
        self.assertEqual(
            parallel["process"]["thread_count_readbacks"], [16, 16]
        )

    def test_pooled_field_summary_uses_attested_shared_export(self):
        from module import aedt_pool_adapter

        field_summary = SimpleNamespace(
            export_csv=Mock(return_value=True),
            get_field_summary_data=Mock(
                side_effect=AssertionError("node-local tempfile is forbidden")
            ),
        )
        provenance = {
            "transport": "pooled_shared_results",
            "parent": "/gpfs/task/.aedtresults",
        }
        sim = SimpleNamespace(
            _new_aedt_export_target=Mock(return_value=(
                "/gpfs/task/.aedtresults/thermal-summary.txt",
                provenance,
            )),
            _read_attested_aedt_export=Mock(return_value=(
                "Field Summary\nProject\nDesign\nSolution\n"
                "Entity,Quantity,Geometry Name,Min,Max,Mean,Stdev,Total\n"
                "Object,Temperature,core_1,40.0,91.5,75.25,2.0,300.0\n"
            )),
            _remove_attested_aedt_export=Mock(),
        )

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=True):
            frame = thermal._field_summary_data_frame(
                sim, field_summary, "ThermalSetup : SteadyState"
            )

        self.assertEqual(frame.loc[0, "Geometry Name"], "core_1")
        self.assertEqual(frame.loc[0, "Max"], 91.5)
        self.assertEqual(frame.loc[0, "Mean"], 75.25)
        field_summary.get_field_summary_data.assert_not_called()
        field_summary.export_csv.assert_called_once_with(
            "/gpfs/task/.aedtresults/thermal-summary.txt",
            setup="ThermalSetup : SteadyState",
        )
        sim._read_attested_aedt_export.assert_called_once_with(
            "/gpfs/task/.aedtresults/thermal-summary.txt",
            provenance,
            unittest.mock.ANY,
            "thermal_field_summary",
        )
        sim._remove_attested_aedt_export.assert_called_once_with(
            "/gpfs/task/.aedtresults/thermal-summary.txt",
            provenance,
            "thermal_field_summary",
        )

    def test_pooled_field_summary_cleans_export_after_failure(self):
        from module import aedt_pool_adapter

        provenance = {"transport": "pooled_shared_results"}
        sim = SimpleNamespace(
            _new_aedt_export_target=Mock(return_value=(
                "/gpfs/task/.aedtresults/thermal-summary.txt",
                provenance,
            )),
            _read_attested_aedt_export=Mock(),
            _remove_attested_aedt_export=Mock(),
        )
        field_summary = SimpleNamespace(export_csv=Mock(return_value=False))

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=True):
            with self.assertRaisesRegex(
                    RuntimeError, "field-summary export returned False"):
                thermal._field_summary_data_frame(
                    sim, field_summary, "ThermalSetup : SteadyState"
                )

        sim._read_attested_aedt_export.assert_not_called()
        sim._remove_attested_aedt_export.assert_called_once_with(
            "/gpfs/task/.aedtresults/thermal-summary.txt",
            provenance,
            "thermal_field_summary",
        )

    def test_standalone_field_summary_keeps_pyaedt_data_path(self):
        from module import aedt_pool_adapter

        expected = pd.DataFrame([{"Geometry Name": "core_1", "Max": 91.5}])
        field_summary = SimpleNamespace(
            get_field_summary_data=Mock(return_value=expected)
        )
        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=False):
            actual = thermal._field_summary_data_frame(
                SimpleNamespace(),
                field_summary,
                "ThermalSetup : SteadyState",
            )

        self.assertIs(actual, expected)
        field_summary.get_field_summary_data.assert_called_once_with(
            setup="ThermalSetup : SteadyState", pandas_output=True
        )

    def _completion_poll(
            self, telemetry, running, *, timeout_s=20.0,
            idle_monitor_grace_s=4.0, poll_s=2.0,
            desktop_attestor=None):
        from module import aedt_pool_adapter

        now = [100.0]
        sleeps = []

        def sleep_and_advance(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        desktop = SimpleNamespace(
            AreThereSimulationsRunning=Mock(side_effect=list(running))
        )
        if desktop_attestor is None:
            def desktop_attestor(target):
                return target

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled",
                return_value=False), patch.object(
                    thermal, "_thermal_convergence_telemetry",
                    side_effect=list(telemetry)):
            result = thermal._poll_thermal_dispatch_evidence(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
                {},
                timeout_s=timeout_s,
                poll_s=poll_s,
                clock=lambda: now[0],
                sleeper=sleep_and_advance,
                completion_barrier=True,
                idle_monitor_grace_s=idle_monitor_grace_s,
                desktop=desktop,
                desktop_attestor=desktop_attestor,
            )
        return result, sleeps, desktop

    def test_standalone_completion_waits_true_unknown_false_for_delayed_monitor(
            self):
        (convergence, running, error), sleeps, desktop = self._completion_poll(
            [
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._convergence(),
            ],
            [True, RuntimeError("transient proxy gap"), False, False],
        )

        self.assertEqual(convergence["thermal_convergence_reason"], "converged")
        poll = convergence["_thermal_completion_poll"]
        self.assertEqual(poll["outcome"], "terminal_fresh_monitor")
        self.assertFalse(poll["timed_out"])
        self.assertFalse(running)
        self.assertEqual(error, "")
        self.assertEqual(sleeps, [2.0, 2.0, 2.0])
        self.assertEqual(
            [item["running"] for item in poll["running_transitions"]],
            [True, None, False],
        )
        self.assertEqual(
            desktop.AreThereSimulationsRunning.call_count, 4
        )

    def test_standalone_completion_accepts_monitor_during_idle_grace(self):
        (convergence, running, _error), sleeps, _desktop = self._completion_poll(
            [
                self._monitor_failure("monitor_missing"),
                self._convergence(),
            ],
            [False, False],
        )

        self.assertEqual(convergence["thermal_convergence_reason"], "converged")
        self.assertFalse(running)
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(
            convergence["_thermal_completion_poll"]["outcome"],
            "terminal_fresh_monitor",
        )

    def test_standalone_completion_never_exits_on_monitor_while_running_true(
            self):
        (convergence, running, _error), sleeps, desktop = self._completion_poll(
            [
                self._convergence(),
                self._convergence(),
                self._convergence(),
            ],
            [True, True, False],
        )

        self.assertEqual(convergence["thermal_convergence_reason"], "converged")
        self.assertFalse(running)
        self.assertEqual(sleeps, [2.0, 2.0])
        self.assertEqual(
            convergence["_thermal_completion_poll"]["outcome"],
            "terminal_fresh_monitor",
        )
        self.assertEqual(
            desktop.AreThereSimulationsRunning.call_count, 3
        )

    def test_standalone_completion_monitor_does_not_authorize_unknown_state(
            self):
        unknown = RuntimeError("native running state unavailable")
        (convergence, running, error), sleeps, desktop = self._completion_poll(
            [
                self._convergence(),
                self._convergence(),
                self._convergence(),
                self._convergence(),
            ],
            [unknown, unknown, unknown, unknown],
            timeout_s=5.0,
        )

        poll = convergence["_thermal_completion_poll"]
        self.assertIsNone(running)
        self.assertIn("native running state unavailable", error)
        self.assertEqual(poll["outcome"], "completion_timeout")
        self.assertTrue(poll["timed_out"])
        self.assertEqual(sleeps, [2.0, 2.0, 1.0])
        self.assertEqual(
            desktop.AreThereSimulationsRunning.call_count, 4
        )

    def test_standalone_completion_final_attestation_failure_blocks_idle(
            self):
        missing = self._monitor_failure("monitor_missing")
        attestor = Mock(side_effect=[
            SimpleNamespace(),
            RuntimeError("final endpoint identity mismatch"),
            RuntimeError("final endpoint identity mismatch"),
            RuntimeError("final endpoint identity mismatch"),
        ])
        (convergence, running, _error), sleeps, _desktop = (
            self._completion_poll(
                [missing, self._convergence(), self._convergence(),
                 self._convergence()],
                [True, False, False, False],
                desktop_attestor=attestor,
            )
        )

        poll = convergence["_thermal_completion_poll"]
        self.assertFalse(running)
        self.assertFalse(poll["desktop_attested"])
        self.assertIn(
            "final endpoint identity mismatch",
            poll["desktop_attestation_error"],
        )
        self.assertEqual(
            poll["outcome"], "idle_monitor_grace_expired"
        )
        self.assertEqual(attestor.call_count, 4)
        self.assertEqual(sleeps, [2.0, 2.0, 2.0])

    def test_standalone_completion_rejects_missing_monitor_after_idle_grace(
            self):
        missing = self._monitor_failure("monitor_missing")
        (convergence, running, _error), sleeps, desktop = self._completion_poll(
            [missing, missing, missing],
            [False, False, False],
            idle_monitor_grace_s=3.0,
        )

        poll = convergence["_thermal_completion_poll"]
        self.assertEqual(
            convergence["thermal_convergence_reason"], "monitor_missing"
        )
        self.assertFalse(running)
        self.assertEqual(poll["outcome"], "idle_monitor_grace_expired")
        self.assertFalse(poll["timed_out"])
        self.assertEqual(poll["elapsed_s"], 3.0)
        self.assertEqual(sleeps, [2.0, 1.0])
        self.assertEqual(
            desktop.AreThereSimulationsRunning.call_count, 3
        )

    def test_standalone_completion_times_out_without_releasing_running_engine(
            self):
        missing = self._monitor_failure("monitor_missing")
        (convergence, running, _error), sleeps, desktop = self._completion_poll(
            [missing, missing, missing, missing],
            [True, True, True, True],
            timeout_s=5.0,
            idle_monitor_grace_s=3.0,
        )

        poll = convergence["_thermal_completion_poll"]
        self.assertTrue(running)
        self.assertEqual(poll["outcome"], "completion_timeout")
        self.assertTrue(poll["timed_out"])
        self.assertEqual(poll["elapsed_s"], 5.0)
        self.assertEqual(sleeps, [2.0, 2.0, 1.0])
        self.assertEqual(
            desktop.AreThereSimulationsRunning.call_count, 4
        )

    def test_exact_standalone_timeout_keeps_uncertainty_and_avoids_aedt_calls(
            self):
        from module import aedt_pool_adapter

        sim = SimpleNamespace(
            NUM_CORE=16,
            solver_may_be_running=False,
            _attest_cached_native_desktop=Mock(),
            save_project=Mock(
                side_effect=AssertionError("unsafe save_project call")
            ),
        )
        native_ipk = SimpleNamespace(analyze=Mock(return_value=True))
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        convergence = self._monitor_failure("monitor_missing")
        convergence["_thermal_completion_poll"] = {
            "desktop_attested": True,
            "last_running": True,
            "timed_out": True,
            "outcome": "completion_timeout",
        }
        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled",
                return_value=False), patch.object(
                    thermal, "_standalone_thermal_completion_timeout",
                    return_value=5.0), patch.object(
                    thermal, "_prepare_thermal_dispatch",
                    return_value=preflight), patch.object(
                    thermal, "_snapshot_thermal_monitors",
                    return_value={}), patch.object(
                    thermal, "_thermal_desktop_handle",
                    return_value=SimpleNamespace(
                        GetMessages=Mock(return_value=[])
                    )), patch.object(
                    thermal, "_poll_thermal_dispatch_evidence",
                    return_value=(convergence, True, "")), patch.object(
                    thermal, "_bounded_thermal_model_context",
                    side_effect=AssertionError("unsafe model-context call")), \
                patch.object(
                    thermal, "_bounded_thermal_messages",
                    side_effect=AssertionError("unsafe message call")):
            with self.assertRaisesRegex(
                    RuntimeError, "did not prove exact Desktop idle"):
                thermal._solve_exact_thermal_setup(
                    sim, SimpleNamespace(), SimpleNamespace()
                )

        self.assertTrue(sim.solver_may_be_running)
        sim.save_project.assert_not_called()

    def test_exact_standalone_attested_idle_clears_uncertainty(self):
        from module import aedt_pool_adapter

        sim = SimpleNamespace(
            NUM_CORE=16,
            solver_may_be_running=False,
            _attest_cached_native_desktop=Mock(),
            save_project=Mock(),
        )
        native_ipk = SimpleNamespace(analyze=Mock(return_value=True))
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        convergence = self._convergence()
        convergence["_thermal_completion_poll"] = {
            "desktop_attested": True,
            "last_running": False,
            "timed_out": False,
            "outcome": "terminal_fresh_monitor",
        }
        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled",
                return_value=False), patch.object(
                    thermal, "_standalone_thermal_completion_timeout",
                    return_value=5.0), patch.object(
                    thermal, "_prepare_thermal_dispatch",
                    return_value=preflight), patch.object(
                    thermal, "_snapshot_thermal_monitors",
                    return_value={}), patch.object(
                    thermal, "_thermal_desktop_handle",
                    return_value=SimpleNamespace(
                        GetMessages=Mock(return_value=[])
                    )), patch.object(
                    thermal, "_poll_thermal_dispatch_evidence",
                    return_value=(convergence, False, "")):
            result = thermal._solve_exact_thermal_setup(
                sim, SimpleNamespace(), SimpleNamespace()
            )

        self.assertFalse(sim.solver_may_be_running)
        self.assertTrue(result["analyze_call_ok"])
        sim.save_project.assert_called_once_with()

    def test_pooled_field_summary_busy_polls_suspend_outer_session_lock(self):
        from module import aedt_pool_adapter

        states = iter([True, "true", False])
        lock_depth = [0]
        events = []

        @contextmanager
        def guard():
            lock_depth[0] += 1
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")
                lock_depth[0] -= 1

        @contextmanager
        def native_window():
            events.append("outer-suspended")
            yield
            events.append("outer-restored")

        def running():
            events.append(("running", lock_depth[0]))
            return next(states)

        desktop = SimpleNamespace(AreThereSimulationsRunning=Mock(side_effect=running))
        native_ipk = SimpleNamespace(odesktop=desktop)
        sim = SimpleNamespace(
            aedt_native_solve_window=native_window,
            aedt_automation_transaction=guard,
        )
        now = [100.0]

        def sleep_and_advance(seconds):
            self.assertEqual(lock_depth[0], 0)
            with guard():
                events.append("sibling-automation")
            now[0] += seconds

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=True), \
                patch.object(
                    thermal, "_prepare_thermal_dispatch",
                    return_value={"native_ipk": native_ipk},
                ):
            with thermal._pooled_field_summary_window(
                    sim, native_ipk, SimpleNamespace(),
                    timeout_s=20.0, poll_s=2.0,
                    clock=lambda: now[0], sleeper=sleep_and_advance,
            ) as yielded:
                self.assertIs(yielded, native_ipk)
                self.assertEqual(lock_depth[0], 1)
                events.append("field-export")

        self.assertEqual(now[0], 104.0)
        self.assertEqual(desktop.AreThereSimulationsRunning.call_count, 3)
        self.assertEqual(events.count("sibling-automation"), 2)
        idle_index = events.index(("running", 1), events.index(("running", 1)) + 1)
        idle_index = events.index(("running", 1), idle_index + 1)
        export_index = events.index("field-export")
        self.assertNotIn("lock-exit", events[idle_index:export_index])
        self.assertEqual(lock_depth[0], 0)

    def test_pooled_field_summary_window_timeout_releases_every_busy_poll(self):
        from module import aedt_pool_adapter

        desktop = SimpleNamespace(AreThereSimulationsRunning=Mock(return_value=True))
        lock_depth = [0]

        @contextmanager
        def guard():
            lock_depth[0] += 1
            try:
                yield
            finally:
                lock_depth[0] -= 1

        native_ipk = SimpleNamespace(odesktop=desktop)
        sim = SimpleNamespace(
            aedt_native_solve_window=lambda: nullcontext(),
            aedt_automation_transaction=guard,
        )
        now = [100.0]

        def sleep_and_advance(seconds):
            self.assertEqual(lock_depth[0], 0)
            now[0] += seconds

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=True), \
                patch.object(
                    thermal, "_prepare_thermal_dispatch",
                    return_value={"native_ipk": native_ipk},
                ):
            with self.assertRaisesRegex(
                    RuntimeError, "timed out waiting for pooled AEDT"):
                with thermal._pooled_field_summary_window(
                        sim, native_ipk, SimpleNamespace(),
                        timeout_s=3.0, poll_s=2.0,
                        clock=lambda: now[0], sleeper=sleep_and_advance,
                ):
                    self.fail("busy pooled Desktop must not yield export window")

        self.assertEqual(now[0], 103.0)
        self.assertEqual(lock_depth[0], 0)

    def test_standalone_field_summary_window_is_a_lock_free_null_context(self):
        from module import aedt_pool_adapter

        native_ipk = SimpleNamespace()
        sim = SimpleNamespace(
            aedt_native_solve_window=Mock(
                side_effect=AssertionError("standalone must not suspend a lock")
            ),
            aedt_automation_transaction=Mock(
                side_effect=AssertionError("standalone must not acquire a lock")
            ),
        )

        with patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=False):
            with thermal._pooled_field_summary_window(
                    sim, native_ipk, SimpleNamespace()) as yielded:
                self.assertIs(yielded, native_ipk)

        sim.aedt_native_solve_window.assert_not_called()
        sim.aedt_automation_transaction.assert_not_called()

    def test_native_pipeline_barrier_suspends_every_outer_lock_depth(self):
        from module import aedt_pool_adapter

        lock_depth = [2]
        events = []

        @contextmanager
        def native_window():
            saved = lock_depth[0]
            self.assertGreater(saved, 0)
            lock_depth[0] = 0
            events.append("suspended")
            try:
                yield
            finally:
                lock_depth[0] = saved
                events.append("restored")

        def wait_for_cohort():
            self.assertEqual(lock_depth[0], 0)
            events.append("cohort-complete")
            return {"native_pipeline_barrier_granted": True}

        sim = SimpleNamespace(
            aedt_native_solve_window=native_window,
            wait_for_pooled_native_pipeline=wait_for_cohort,
        )
        with patch.object(
            aedt_pool_adapter, "pooled_backend_enabled", return_value=True
        ):
            status = thermal._wait_for_pooled_native_pipeline(sim)

        self.assertTrue(status["native_pipeline_barrier_granted"])
        self.assertEqual(events, ["suspended", "cohort-complete", "restored"])
        self.assertEqual(lock_depth[0], 2)

    @staticmethod
    def _convergence(converged=True):
        return {
            "thermal_convergence_available": 1,
            "thermal_converged": 1 if converged else 0,
            "thermal_iterations": 151 if converged else 142,
            "thermal_residual_continuity": 7.9912e-4 if converged else 1.0657e18,
            "thermal_residual_x_velocity": 3.6686e-4 if converged else 1.1056e-1,
            "thermal_residual_y_velocity": 9.8308e-4 if converged else 5.6684e-2,
            "thermal_residual_z_velocity": 3.9535e-4 if converged else 1.4627e-1,
            "thermal_residual_energy": 4.3936e-9 if converged else 1.2163e-1,
            "thermal_residual_flow_limit": 1e-3,
            "thermal_residual_energy_limit": 1e-7,
            "thermal_convergence_reason": "converged" if converged else "residual_threshold",
            "thermal_monitor_file": "monitor.sd",
        }

    @staticmethod
    def _monitor_failure(reason):
        return {
            "thermal_convergence_available": 0,
            "thermal_converged": 0,
            "thermal_iterations": 0,
            "thermal_residual_continuity": float("nan"),
            "thermal_residual_x_velocity": float("nan"),
            "thermal_residual_y_velocity": float("nan"),
            "thermal_residual_z_velocity": float("nan"),
            "thermal_residual_energy": float("nan"),
            "thermal_residual_flow_limit": float("nan"),
            "thermal_residual_energy_limit": float("nan"),
            "thermal_convergence_reason": reason,
            "thermal_monitor_file": "",
        }

    def _run(
        self,
        analyze_result,
        responses,
        include_side=False,
        n1_side=0,
        probe_names=(),
        setup_result=True,
        setup_update_result=True,
        convergence=None,
        tx_count=1,
        scalar_responses=None,
        core_k_anisotropic=1,
        pooled=False,
        rebind_sequence=None,
        object_factory=None,
        probe_factory=None,
    ):
        ipk = _Icepak(
            analyze_result,
            responses,
            setup_result=setup_result,
            setup_update_result=setup_update_result,
            scalar_responses=scalar_responses,
        )
        wrapper = _DesignWrapper(ipk)
        project = SimpleNamespace(create_design=lambda **_kwargs: wrapper)
        current_native_project = {"value": None}
        if rebind_sequence is None:
            default_native_project = _ProjectHandle("thermal_test")
            known_native_projects = [default_native_project]

            def next_native_project():
                current_native_project["value"] = default_native_project
                return default_native_project
        else:
            known_native_projects = list(rebind_sequence)
            native_projects = iter(known_native_projects)

            def next_native_project():
                value = next(native_projects)
                current_native_project["value"] = value
                return value

        rebind_project = Mock(side_effect=next_native_project)

        def set_active_project(project_name):
            self.assertEqual(project_name, "thermal_test")
            value = current_native_project["value"]
            self.assertIsNotNone(value)
            return value

        def scoped_messages(project_name, design_name, severity):
            self.assertEqual(
                (project_name, design_name),
                ("thermal_test", "icepak_thermal"),
            )
            if severity == 2:
                return []
            values = ["pre-dispatch thermal diagnostic"]
            dispatched = any(
                getattr(project_value, "last_design", None) is not None
                and project_value.last_design.analyze_calls
                for project_value in known_native_projects
            )
            if dispatched:
                values.append(
                    "Normal completion of simulation on server: thermal-node"
                )
            return values

        pooled_desktop = SimpleNamespace(
            AreThereSimulationsRunning=lambda: False,
            GetMessages=scoped_messages,
            SetActiveProject=Mock(side_effect=set_active_project),
        )
        ipk.odesktop = pooled_desktop
        native_window = Mock(return_value=nullcontext())
        native_pipeline_barrier = Mock()
        sim = SimpleNamespace(
            project=project,
            _rebind_native_project_for_design_creation=rebind_project,
            _attest_cached_native_desktop=(
                lambda desktop, **_kwargs: desktop
            ),
            df_plus=pd.DataFrame({
                "thermal_symmetry": ["eighth"],
                "thermal_max_iterations": [100],
                "N1_side": [n1_side],
                "N2_side": [1 if include_side else 0],
                "n_explicit_turns": [1],
                "core_k_anisotropic": [core_k_anisotropic],
                "core_k_thermal": [2.0],
                "core_lamination_factor": [0.85],
                "core_k_alloy": [9.0],
                "core_k_interlayer": [0.2],
            }),
            input_df=pd.DataFrame([{}]),
            NUM_CORE=4,
            PROJECT_NAME="thermal_test",
            save_project=Mock(),
            _ensure_pooled_shared_results_directory=Mock(),
            _new_aedt_export_target=Mock(return_value=(
                "/shared/results/mft_thermal_field_summary_export.txt",
                {"transport": "pooled_shared_results"},
            )),
            _read_attested_aedt_export=Mock(
                side_effect=lambda *_args: ipk.field_summary_export_text
            ),
            _remove_attested_aedt_export=Mock(),
            aedt_native_solve_window=native_window,
            aedt_automation_transaction=lambda: nullcontext(),
            wait_for_pooled_native_pipeline=native_pipeline_barrier,
            _native_desktop_handle=Mock(return_value=pooled_desktop),
            stage_timings={},
            solver_may_be_running=False,
        )
        object_factory = object_factory or _Object
        probe_factory = probe_factory or (
            lambda name: SimpleNamespace(name=name, is3d=False)
        )
        side = [object_factory("Rx_side_0")] if include_side else []
        objects = {
            "Tx": [object_factory(f"Tx_main_{index}") for index in range(tx_count)],
            "Rx_main_explicit": [object_factory("Rx_main_0")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": side,
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core": [object_factory("core_1")],
            "wcp_pads": [],
            "core_pads": [],
        }
        probe_sheets = [probe_factory(name) for name in probe_names]

        def record_mock_power_balance(_ipk, target_sim, _objects, **_kwargs):
            target_sim.thermal_rx_model = "hybrid_explicit"
            target_sim.thermal_rx_power_balance = [{
                "group": "P_Rx_main_group",
                "name_hint": "Rx_main_0",
                "expected_w": 0.0,
                "assigned_w": 0.0,
            }]
            # The real loss allocator always emits the native-core transport
            # contract. This helper replaces that allocator, so provide the
            # same neutral legacy telemetry explicitly.
            target_sim.thermal_core_loss_contract_version = "legacy_test"
            target_sim.thermal_core_loss_source = "legacy_test"
            target_sim.thermal_core_loss_correction_factor = 1.0
            target_sim.thermal_core_expected_injected_w = 0.0
            target_sim.thermal_core_requested_wrapper_echo_w = 0.0
            target_sim.thermal_core_native_readback_w = 0.0
            target_sim.thermal_core_restore_factor = 1.0
            target_sim.thermal_core_native_restored_full_w = 0.0
            target_sim.thermal_core_full_expected_margin_adjusted_w = 0.0
            target_sim.thermal_core_native_restored_rel_error = 0.0
            target_sim.thermal_core_native_readback_count = 0
            target_sim.thermal_core_power_balance_abs_error_w = 0.0
            target_sim.thermal_core_power_balance_rel_error = 0.0
            return {}

        with ExitStack() as stack:
            from module import aedt_pool_adapter

            stack.enter_context(patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=pooled
            ))
            stack.enter_context(patch.object(thermal, "set_design_variables"))
            stack.enter_context(patch.object(
                thermal,
                "_create_thermal_materials",
                return_value=(
                    1.0,
                    1.0,
                    {
                        "thermal_conductivity_W_mK": 0.2,
                        "electrical_conductivity_S_m": 0.0,
                    },
                ),
            ))
            stack.enter_context(patch.object(thermal, "_build_geometry", return_value=objects))
            stack.enter_context(patch.object(
                thermal, "_create_probe_sheets", return_value=probe_sheets))
            stack.enter_context(patch.object(
                thermal, "_assign_losses", side_effect=record_mock_power_balance))
            stack.enter_context(patch.object(thermal, "_assign_boundaries"))
            mesh_plan = {
                "schema": thermal.THERMAL_MESH_PLAN_CONTRACT_VERSION,
                "policy": thermal.THERMAL_MESH_POLICY,
                "plan_sha256": "a" * 64,
                "operations": [{
                    "name": "test_mesh",
                    "category": "test",
                    "operation_type": "object_level",
                    "level": 3,
                    "objects": ["Rx_main_0"],
                    "shared_region": False,
                    "separate_objects": True,
                    "actual_operation_names": ["test_mesh_L_3"],
                }],
                "operation_count": 1,
                "assigned_object_count": 1,
                "required_thin_object_count": 0,
                "required_thin_objects": [],
                "required_objects_missing": [],
                "core_plate_assembly_count": 0,
                "wcp_assembly_count": 0,
                "rx_retained_pack_count": 0,
                "rx_block_shared_pack_count": 0,
                "rx_block_shared_packs": [],
                "rx_main_block_objects": [],
                "shared_operation_count": 0,
                "object_level_operation_count": 1,
                "mesh_region_operation_count": 0,
                "wcp_pad_mesh_region_count": 0,
                "separate_object_operation_count": 1,
            }
            stack.enter_context(patch.object(
                thermal, "_assign_thermal_mesh",
                return_value=mesh_plan,
            ))
            stack.enter_context(patch.object(
                thermal,
                "_generate_and_attest_thermal_mesh",
                return_value={
                    "schema": (
                        thermal.THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION
                    ),
                    "status": "passed_standalone_native_premesh",
                    "passed": True,
                    "static_contract_passed": True,
                    "generate_mesh_returned": True,
                    "message_scan_complete": True,
                    "analysis_dispatched_after_premesh": False,
                    "required_objects_missing": [],
                    "unmeshed_objects": [],
                    "mesh_artifact_readback_passed": True,
                    "mesh_mapping_coverage_passed": True,
                    "native_operation_readback_passed": True,
                    "standalone_idle_barrier_passed": True,
                    "postflight_identity_passed": True,
                    "mesh_plan_sha256": "a" * 64,
                },
            ))
            stack.enter_context(patch.object(
                thermal,
                "_snapshot_thermal_rx_interface_cases",
                return_value={},
            ))
            stack.enter_context(patch.object(
                thermal,
                "_thermal_rx_block_interface_coverage",
                return_value={
                    "schema": (
                        thermal.THERMAL_RX_BLOCK_INTERFACE_CONTRACT_VERSION
                    ),
                    "passed": True,
                    "unpaired_interfaces": [],
                },
            ))
            if isinstance(convergence, (list, tuple)):
                convergence_values = list(convergence)
                convergence_index = {"value": 0}

                def next_convergence(*_args, **_kwargs):
                    index = min(
                        convergence_index["value"],
                        len(convergence_values) - 1,
                    )
                    convergence_index["value"] += 1
                    return convergence_values[index]

                telemetry = stack.enter_context(patch.object(
                    thermal,
                    "_thermal_convergence_telemetry",
                    side_effect=next_convergence,
                ))
            else:
                telemetry = stack.enter_context(patch.object(
                    thermal,
                    "_thermal_convergence_telemetry",
                    return_value=convergence or self._convergence(),
                ))

            def poll_once(target_sim, target_ipk, target_setup, snapshot, **_kwargs):
                value = dict(thermal._thermal_convergence_telemetry(
                    target_sim,
                    target_ipk,
                    target_setup,
                    attempts=1,
                    monitor_snapshot=snapshot,
                ))
                value["_thermal_completion_poll"] = {
                    "desktop_attested": True,
                    "last_running": False,
                    "timed_out": False,
                }
                return value, False, ""

            stack.enter_context(patch.object(
                thermal,
                "_poll_thermal_dispatch_evidence",
                side_effect=poll_once,
            ))
            sleeper = stack.enter_context(patch("time.sleep"))
            result = thermal.run_thermal_analysis(sim)
        ipk.telemetry_mock = telemetry
        ipk.sleep_mock = sleeper
        ipk.rebind_project_mock = rebind_project
        ipk.native_window_mock = native_window
        ipk.native_pipeline_barrier_mock = native_pipeline_barrier
        self.assertGreaterEqual(rebind_project.call_count, 2)
        return ipk, result.iloc[0]

    @staticmethod
    def _loss_objects():
        return {
            "Tx": [_Object("Tx_main_0")],
            "Rx_main_explicit": [],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core": [],
        }

    @staticmethod
    def _boundary_sim():
        return SimpleNamespace(df_plus=pd.DataFrame({
            "plate_temp": [45.0],
            "air_temp": [50.0],
            "fan_velocity": [1.5],
            "fan_config": ["dual"],
        }))

    @staticmethod
    def _core_k_frame(anisotropic=1, legacy=2.0):
        return pd.DataFrame({
            "core_k_anisotropic": [anisotropic],
            "core_k_thermal": [legacy],
            "core_lamination_factor": [0.85],
            "core_k_alloy": [9.0],
            "core_k_interlayer": [0.2],
            "k_ins": [0.2],
            "cw2": [0.665],
            "gap2": [0.339],
        })

    def test_wound_core_rule_of_mixtures_and_piece_orientation(self):
        k_inplane, k_throughstack = (
            thermal._derive_wound_core_conductivity(0.85, 9.0, 0.2)
        )
        self.assertAlmostEqual(k_inplane, 7.68)
        self.assertAlmostEqual(k_throughstack, 1.1842105263157894)

        for name in (
                "core_1_leg_left", "core_2_leg_center",
                "core_3_leg_right", "core_1_leg_center_bottom",
                "core_1_leg_center_top"):
            self.assertEqual(
                thermal._core_thermal_material_for_piece(name),
                "core_amorphous_thermal_leg",
            )
        for name in ("core_1_yoke_top", "core_3_yoke_bottom"):
            self.assertEqual(
                thermal._core_thermal_material_for_piece(name),
                "core_amorphous_thermal_yoke",
            )
        with self.assertRaisesRegex(ValueError, "unrecognized segmented"):
            thermal._core_thermal_material_for_piece("core_1")

        materials = _Materials()
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "0.2W_per_mK",
                "conductivity": "0S_per_m",
            },
        ):
            thermal._create_thermal_materials(
                SimpleNamespace(materials=materials), self._core_k_frame()
            )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal_leg"
            ].thermal_conductivity,
            [k_throughstack, k_inplane, k_inplane],
        )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal_yoke"
            ].thermal_conductivity,
            [k_inplane, k_inplane, k_throughstack],
        )
        self.assertNotIn("core_amorphous_thermal", materials.material_keys)
        self.assertEqual(
            materials.material_keys["thermal_pad"].thermal_conductivity,
            0.2,
        )
        self.assertEqual(
            thermal.THERMAL_PAD_MATERIAL_POLICY,
            "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
            "electrically_insulating_v1",
        )
        self.assertEqual(
            thermal._thermal_pad_result_metadata({
                "thermal_conductivity_W_mK": 0.2,
                "electrical_conductivity_S_m": 0.0,
            }),
            {
                "thermal_pad_conductivity_W_mK": [0.2],
                "thermal_pad_material_policy": [
                    "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
                    "electrically_insulating_v1"
                ],
                "thermal_pad_native_readback_contract_version": [
                    "thermal-pad-native-material-readback-v1"
                ],
                "thermal_pad_native_readback_attested": [1],
                "thermal_pad_native_thermal_conductivity_W_mK": [0.2],
                "thermal_pad_native_electrical_conductivity_S_m": [0.0],
            },
        )

    def test_existing_thermal_pad_is_overwritten_and_native_attested(self):
        materials = _Materials()
        stale = materials.add_material("thermal_pad")
        stale.conductivity = 123.0
        stale.thermal_conductivity = 3.0
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "0.2W_per_mK",
                "conductivity": "0S_per_m",
            },
        ):
            _, _, readback = thermal._create_thermal_materials(
                SimpleNamespace(materials=materials), self._core_k_frame()
            )
        self.assertEqual(stale.conductivity, 0)
        self.assertEqual(stale.thermal_conductivity, 0.2)
        self.assertEqual(readback, {
            "thermal_conductivity_W_mK": 0.2,
            "electrical_conductivity_S_m": 0.0,
        })

    def test_thermal_pad_native_readback_rejects_stale_value(self):
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "3W_per_mK",
                "conductivity": "0S_per_m",
            },
        ), self.assertRaisesRegex(
            RuntimeError, "thermal_conductivity mismatch"
        ):
            thermal._thermal_pad_native_readback(_Materials())

    def test_explicit_rx_insulation_gap_indices_cover_only_retained_pairs(self):
        self.assertEqual(thermal._explicit_rx_gap_indices(1, 2), [])
        self.assertEqual(thermal._explicit_rx_gap_indices(6, 0), [])
        self.assertEqual(thermal._explicit_rx_gap_indices(6, 1), [])
        self.assertEqual(thermal._explicit_rx_gap_indices(6, 2), [0, 4])
        self.assertEqual(thermal._explicit_rx_gap_indices(6, 3), [0, 1, 2, 3, 4])
        self.assertEqual(thermal._explicit_rx_gap_indices(6, -1), [0, 1, 2, 3, 4])

        frame = pd.DataFrame({"N2_main": [6], "N2_side": [5]})
        self.assertEqual(
            thermal._expected_rx_insulation_counts(frame, 2, "full"),
            {
                "Rx_main_insulation": 2,
                "Rx_side_insulation": 2,
                "Rx_side2_insulation": 2,
            },
        )
        self.assertEqual(
            thermal._expected_rx_insulation_counts(frame, 2, "eighth"),
            {
                "Rx_main_insulation": 2,
                "Rx_side_insulation": 2,
                "Rx_side2_insulation": 0,
            },
        )

    def test_explicit_rx_insulation_tiles_exact_candidate_gaps(self):
        frame = pd.DataFrame({
            "N2_main": [6],
            "cw2": [1.0],
            "gap2": [0.25],
            "sl2_main_x": [10.0],
            "sl2_main_y": [8.0],
        })
        created = []

        def create_polyline(**kwargs):
            created.append(kwargs)
            return _Object(kwargs["name"])

        ipk = SimpleNamespace(modeler=SimpleNamespace(
            create_polyline=create_polyline
        ))
        result = thermal._build_explicit_rx_insulation(
            ipk, frame, "main", "Rx_main", -20.0, 2, 300.0
        )

        self.assertEqual(
            [item.name for item in result],
            ["Rx_main_insulation_gap_0", "Rx_main_insulation_gap_4"],
        )
        self.assertEqual(
            [item["material"] for item in created],
            [thermal.RX_EXPLICIT_INSULATION_MATERIAL] * 2,
        )
        self.assertEqual(
            [item["xsection_width"] for item in created], [0.25, 0.25]
        )
        self.assertEqual(
            [item["xsection_height"] for item in created], [300.0, 300.0]
        )
        self.assertEqual(
            created[0]["points"],
            [
                ["6.125mm + -20.0mm", "5.125mm", "0mm"],
                ["-6.125mm + -20.0mm", "5.125mm", "0mm"],
                ["-6.125mm + -20.0mm", "-5.125mm", "0mm"],
                ["6.125mm + -20.0mm", "-5.125mm", "0mm"],
                ["6.125mm + -20.0mm", "5.125mm", "0mm"],
            ],
        )
        self.assertEqual(
            created[1]["points"][0],
            ["11.125mm + -20.0mm", "10.125mm", "0mm"],
        )

    def test_explicit_rx_insulation_material_is_native_attested_only_when_used(self):
        frame = self._core_k_frame()
        frame["n_explicit_turns"] = 2
        frame["N2_main"] = 6
        frame["N2_side"] = 5
        materials = _Materials()
        stale = materials.add_material(
            thermal.RX_EXPLICIT_INSULATION_MATERIAL
        )
        stale.conductivity = 99.0
        stale.thermal_conductivity = 99.0

        def raw_material(_materials, name):
            if name == thermal.RX_EXPLICIT_INSULATION_MATERIAL:
                return {
                    "thermal_conductivity": "0.2W_per_mK",
                    "conductivity": "0S_per_m",
                }
            self.assertEqual(name, "thermal_pad")
            return {
                "thermal_conductivity": "0.2W_per_mK",
                "conductivity": "0S_per_m",
            }

        with patch.object(
            thermal, "_raw_aedt_material_props", side_effect=raw_material
        ) as native_readback:
            _, _, readback = thermal._create_thermal_materials(
                SimpleNamespace(materials=materials), frame
            )

        self.assertEqual(stale.conductivity, 0)
        self.assertEqual(stale.thermal_conductivity, 0.2)
        self.assertEqual(
            readback["rx_explicit_insulation"],
            {
                "thermal_conductivity_W_mK": 0.2,
                "electrical_conductivity_S_m": 0.0,
            },
        )
        self.assertEqual(
            [item.args[1] for item in native_readback.call_args_list],
            [thermal.RX_EXPLICIT_INSULATION_MATERIAL, "thermal_pad"],
        )

        standard = frame.copy()
        standard["n_explicit_turns"] = 0
        standard_materials = _Materials()
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "0.2W_per_mK",
                "conductivity": "0S_per_m",
            },
        ) as standard_readback:
            _, _, standard_result = thermal._create_thermal_materials(
                SimpleNamespace(materials=standard_materials), standard
            )
        self.assertNotIn(
            thermal.RX_EXPLICIT_INSULATION_MATERIAL,
            standard_materials.material_keys,
        )
        self.assertNotIn("rx_explicit_insulation", standard_result)
        standard_readback.assert_called_once_with(
            standard_materials, "thermal_pad"
        )

    def test_explicit_rx_insulation_native_mismatch_fails_closed(self):
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "0.1W_per_mK",
                "conductivity": "0S_per_m",
            },
        ), self.assertRaisesRegex(
            RuntimeError, "winding_insulation thermal_conductivity mismatch"
        ):
            thermal._rx_insulation_native_readback(_Materials(), 0.2)

    def test_explicit_rx_insulation_result_metadata_requires_native_evidence(self):
        counts = {
            "Rx_main_insulation": 2,
            "Rx_side_insulation": 2,
            "Rx_side2_insulation": 2,
        }
        material_readback = {
            "rx_explicit_insulation": {
                "thermal_conductivity_W_mK": 0.2,
                "electrical_conductivity_S_m": 0.0,
            }
        }
        metadata = thermal._rx_insulation_result_metadata(
            material_readback, counts
        )
        self.assertEqual(
            metadata["thermal_rx_explicit_insulation_count"], [6]
        )
        self.assertEqual(
            metadata[
                "thermal_rx_explicit_insulation_native_readback_attested"
            ],
            [1],
        )
        self.assertEqual(
            metadata[
                "thermal_rx_explicit_insulation_native_thermal_conductivity_W_mK"
            ],
            [0.2],
        )
        self.assertEqual(
            metadata["thermal_rx_explicit_insulation_counts_json"],
            [
                '{"Rx_main_insulation":2,"Rx_side2_insulation":2,'
                '"Rx_side_insulation":2}'
            ],
        )
        with self.assertRaisesRegex(
            RuntimeError, "without native material readback"
        ):
            thermal._rx_insulation_result_metadata({}, counts)

        unused = thermal._rx_insulation_result_metadata({}, {})
        self.assertEqual(
            unused["thermal_rx_explicit_insulation_model"],
            ["not_required_no_adjacent_explicit_foils_v1"],
        )
        self.assertEqual(
            unused[
                "thermal_rx_explicit_insulation_native_readback_attested"
            ],
            [0],
        )

    def test_legacy_core_material_path_remains_scalar(self):
        frame = self._core_k_frame(anisotropic=0, legacy=2.75)
        # These anchors are irrelevant when the explicit legacy opt-out is set.
        frame["core_k_alloy"] = -1.0
        frame["core_k_interlayer"] = -1.0
        contract = thermal._core_thermal_conductivity_contract(frame)
        self.assertEqual(contract["thermal_core_conductivity_model"], "isotropic_legacy")
        self.assertEqual(contract["thermal_core_k_inplane"], 2.75)
        self.assertEqual(contract["thermal_core_k_throughstack"], 2.75)

        materials = _Materials()
        with patch.object(
            thermal,
            "_raw_aedt_material_props",
            return_value={
                "thermal_conductivity": "0.2W_per_mK",
                "conductivity": "0S_per_m",
            },
        ):
            thermal._create_thermal_materials(
                SimpleNamespace(materials=materials), frame
            )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal"
            ].thermal_conductivity,
            2.75,
        )
        self.assertNotIn(
            "core_amorphous_thermal_leg", materials.material_keys
        )
        self.assertNotIn(
            "core_amorphous_thermal_yoke", materials.material_keys
        )

    def test_thermal_geometry_segments_only_the_anisotropic_core(self):
        def core_call(anisotropic):
            frame = pd.DataFrame({
                "core_k_anisotropic": [anisotropic],
                "core_k_thermal": [2.0],
                "core_lamination_factor": [0.85],
                "core_k_alloy": [9.0],
                "core_k_interlayer": [0.2],
                "n_explicit_turns": [1],
                "l1": [89.0],
                "l2": [236.5],
                "nwh1": [284.5],
                "nwh2": [284.5],
                "n_core_group": [1],
                "core_plate_on": [0],
                "core_plate_pad_t": [0.0],
                "wcp_on": [0],
                "wcp_pad_t": [0.0],
                "N1_main": [1],
                "N2_main": [1],
                "N2_side": [0],
                "nwl1_main": [100.0],
                "wff1_main": [0.5],
                "sl1_main_x": [100.0],
                "sl1_main_y": [100.0],
            })
            core = _Object("core_1")
            tx = _Object("Tx_main_0")
            rx = _Object("Rx_main_0")
            ipk = SimpleNamespace(modeler=SimpleNamespace(
                object_names=[core.name, tx.name, rx.name]
            ))
            sim = SimpleNamespace(df_plus=frame)
            with patch.object(
                    thermal, "create_core",
                    return_value=([core], [], [])) as create_core_mock, \
                    patch.object(
                        thermal, "get_tx_y_gaps", return_value=([], [])
                    ), \
                    patch.object(
                        thermal, "create_coil",
                        return_value=([tx], None, 1.0, None, None, None),
                    ), \
                    patch.object(
                        thermal, "_build_rx_group",
                        return_value=([rx], []),
                    ):
                thermal._build_geometry(ipk, sim, mode="full")
            return create_core_mock.call_args.kwargs

        anisotropic = core_call(1)
        self.assertIs(anisotropic["segmented_lamination"], True)
        self.assertEqual(
            anisotropic["core_material_leg"],
            "core_amorphous_thermal_leg",
        )
        self.assertEqual(
            anisotropic["core_material_yoke"],
            "core_amorphous_thermal_yoke",
        )

        legacy = core_call(0)
        self.assertEqual(legacy["core_material"], "core_amorphous_thermal")
        self.assertNotIn("segmented_lamination", legacy)
        self.assertNotIn("core_material_leg", legacy)
        self.assertNotIn("core_material_yoke", legacy)

    def test_symmetry_split_preserves_every_expected_insulation_solid(self):
        frame = pd.DataFrame({
            "core_k_anisotropic": [0],
            "core_k_thermal": [2.0],
            "core_lamination_factor": [0.85],
            "core_k_alloy": [9.0],
            "core_k_interlayer": [0.2],
            "n_explicit_turns": [2],
            "l1": [89.0],
            "l2": [236.5],
            "nwh1": [284.5],
            "nwh2": [284.5],
            "n_core_group": [1],
            "core_plate_on": [0],
            "core_plate_pad_t": [0.0],
            "wcp_on": [0],
            "wcp_pad_t": [0.0],
            "N1_main": [1],
            "N2_main": [4],
            "N2_side": [0],
            "nwl1_main": [100.0],
            "wff1_main": [0.5],
            "sl1_main_x": [100.0],
            "sl1_main_y": [100.0],
        })
        core = _Object("core_1")
        tx = _Object("Tx_main_0")
        rx = _Object("Rx_main_0")
        insulation = [
            _Object(f"Rx_main_insulation_gap_{index}")
            for index in range(3)
        ]

        def build_rx_group(
                _ipk, _df, _prefix, _name, _offset, _n_explicit, _height,
                insulation_sink=None):
            insulation_sink.extend(insulation)
            return [rx], []

        def run(drop_insulation=False):
            names = [
                core.name, tx.name, rx.name,
                *(item.name for item in insulation),
            ]
            split_calls = []

            def split(**kwargs):
                split_calls.append(kwargs)
                if (
                    drop_insulation
                    and kwargs["plane"] == "YZ"
                    and insulation[-1].name in names
                ):
                    names.remove(insulation[-1].name)
                return True

            ipk = SimpleNamespace(modeler=SimpleNamespace(
                object_names=names,
                split=split,
            ))
            sim = SimpleNamespace(df_plus=frame)
            with patch.object(
                    thermal, "create_core",
                    return_value=([core], [], [])), \
                    patch.object(
                        thermal, "get_tx_y_gaps", return_value=([], [])
                    ), \
                    patch.object(
                        thermal, "create_coil",
                        return_value=([tx], None, 1.0, None, None, None),
                    ), \
                    patch.object(
                        thermal, "_build_rx_group",
                        side_effect=build_rx_group,
                    ):
                result = thermal._build_geometry(
                    ipk, sim, mode="eighth"
                )
            return result, split_calls

        result, split_calls = run()
        self.assertEqual(
            [item.name for item in result["Rx_main_insulation"]],
            [item.name for item in insulation],
        )
        self.assertEqual(
            [item["plane"] for item in split_calls], ["XY", "XZ", "YZ"]
        )

        with self.assertRaisesRegex(
            RuntimeError, "insulation topology mismatch"
        ):
            run(drop_insulation=True)

    def test_thermal_payload_tags_anisotropic_and_legacy_core_models(self):
        response = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        _ipk, anisotropic = self._run(None, [response])
        self.assertEqual(
            anisotropic["thermal_core_conductivity_model"],
            "anisotropic_wound_rule_of_mixtures_v1",
        )
        self.assertAlmostEqual(anisotropic["thermal_core_k_inplane"], 7.68)
        self.assertAlmostEqual(
            anisotropic["thermal_core_k_throughstack"],
            1.1842105263157894,
        )

        _ipk, legacy = self._run(
            None, [response], core_k_anisotropic=0
        )
        self.assertEqual(
            legacy["thermal_core_conductivity_model"], "isotropic_legacy"
        )
        self.assertEqual(legacy["thermal_core_k_inplane"], 2.0)
        self.assertEqual(legacy["thermal_core_k_throughstack"], 2.0)

    @staticmethod
    def _probe_frame(n_group):
        # A compact, fully derived geometry row exercises the real probe
        # placement formulas without opening AEDT.
        from module.input_parameter_260706 import (
            create_input_parameter,
            get_drawing_default_params,
            validation_check,
        )

        params = get_drawing_default_params()
        params["n_core_group"] = n_group
        ok, frame = validation_check(
            create_input_parameter(params), strict=True
        )
        if not ok:
            raise AssertionError("probe test fixture did not validate")
        return frame

    def test_core_probe_depth_uses_core_not_central_plate(self):
        odd = self._probe_frame(3)
        self.assertEqual(thermal._core_probe_y_positions(odd, "eighth"), [0.0])

        even = self._probe_frame(4)
        stack_t = (
            float(even["core_plate_t"].iloc[0])
            + 2.0 * float(even["core_plate_pad_t"].iloc[0])
        )
        expected = 0.5 * (
            float(even["core_depth_each"].iloc[0]) + stack_t
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "eighth"), [expected]
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "quarter"), [expected]
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "full"),
            [-expected, expected],
        )

    def test_probe_sheets_cover_center_and_outer_side_legs(self):
        frame = self._probe_frame(4)
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, eighth=True, mode="eighth"
        )
        names = {sheet.name for sheet in sheets}
        self.assertIn("Tprobe_core_center_leg", names)
        self.assertIn("Tprobe_core_side_leg", names)
        self.assertIn("Tprobe_core_top_yoke", names)
        self.assertIn("Tprobe_Rx_side_side", names)
        self.assertIn("Tprobe_Rx_side1_inner", names)
        self.assertNotIn("Tprobe_Rx_side2_side", names)
        by_name = {call["name"]: call for call in calls}
        center = by_name["Tprobe_core_center_leg"]
        side = by_name["Tprobe_core_side_leg"]
        top_yoke = by_name["Tprobe_core_top_yoke"]
        expected_y = thermal._core_probe_y_positions(frame, "eighth")[0]
        self.assertAlmostEqual(float(center["origin"][1][:-2]), expected_y)
        self.assertAlmostEqual(float(side["origin"][1][:-2]), expected_y)
        self.assertAlmostEqual(float(top_yoke["origin"][1][:-2]), expected_y)
        l1 = float(frame["l1"].iloc[0])
        l2 = float(frame["l2"].iloc[0])
        self.assertAlmostEqual(
            float(side["origin"][0][:-2]),
            -(2.0 * l1 + l2) + 0.05 * l1,
        )
        self.assertAlmostEqual(
            float(top_yoke["origin"][2][:-2]),
            float(frame["h1"].iloc[0]) / 2.0 + 0.05 * l1,
        )
        # Negative-x side winding: outward is the more-negative radial pack;
        # inward is its +x mirror toward the transformer centre.
        outward = by_name["Tprobe_Rx_side_side"]
        inward = by_name["Tprobe_Rx_side1_inner"]
        self.assertLess(
            float(outward["origin"][0][:-2]),
            float(inward["origin"][0][:-2]),
        )
        self.assertEqual(outward["sizes"], inward["sizes"])

    def test_full_probe_sheets_mirror_inner_and_outer_side_faces(self):
        frame = self._probe_frame(4)
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, mode="full"
        )
        names = {sheet.name for sheet in sheets}
        self.assertTrue({
            "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
            "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
        }.issubset(names))
        by_name = {call["name"]: call for call in calls}
        left_outer = float(
            by_name["Tprobe_Rx_side_side"]["origin"][0][:-2]
        )
        left_inner = float(
            by_name["Tprobe_Rx_side1_inner"]["origin"][0][:-2]
        )
        right_outer = float(
            by_name["Tprobe_Rx_side2_side"]["origin"][0][:-2]
        )
        right_inner = float(
            by_name["Tprobe_Rx_side2_inner"]["origin"][0][:-2]
        )
        self.assertLess(left_outer, left_inner)
        self.assertGreater(right_outer, right_inner)
        left_ranges = thermal._rx_side_face_x_ranges(
            frame, -(float(frame["l1"].iloc[0]) * 1.5
                     + float(frame["l2"].iloc[0]))
        )
        right_ranges = thermal._rx_side_face_x_ranges(
            frame, float(frame["l1"].iloc[0]) * 1.5
            + float(frame["l2"].iloc[0])
        )
        for relation in ("outward", "inward"):
            self.assertAlmostEqual(
                left_ranges[relation][0], -right_ranges[relation][1]
            )
            self.assertAlmostEqual(
                left_ranges[relation][1], -right_ranges[relation][0]
            )

    def test_invalid_probe_span_is_recorded_before_aedt_creation(self):
        frame = self._probe_frame(4)
        frame.loc[:, "l2"] = 5.0
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs["name"])
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, eighth=True, mode="eighth"
        )

        self.assertNotIn("Tprobe_core_top_yoke", calls)
        self.assertIn("Tprobe_core_top_yoke", sheets.expected_names)
        failure = next(
            item for item in sheets.failures
            if item["probe"] == "Tprobe_core_top_yoke"
        )
        self.assertEqual(failure["stage"], "geometry")
        self.assertEqual(failure["reason"], "invalid_rectangle")

    def test_core_probe_aggregates_center_and_side_legs(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_core_center_leg": (84.0, 78.0),
            "Tprobe_core_side_leg": (89.0, 81.0),
            "Tprobe_core_top_yoke": (92.0, 85.0),
        }
        _ipk, row = self._run(
            None,
            [complete],
            probe_names=[
                "Tprobe_core_center_leg",
                "Tprobe_core_side_leg",
                "Tprobe_core_top_yoke",
            ],
        )
        self.assertEqual(row["Tprobe_core_center_leg_max"], 84.0)
        self.assertEqual(row["Tprobe_core_side_leg_max"], 89.0)
        self.assertEqual(row["Tprobe_core_top_yoke_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_mean"], 85.0)
        self.assertEqual(row["thermal_extraction_complete"], 1)

    def test_even_full_core_probe_selects_hottest_depth_and_leg(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_core_center_leg_neg": (83.0, 77.0),
            "Tprobe_core_center_leg_pos": (86.0, 79.0),
            "Tprobe_core_side_leg_neg": (90.0, 82.0),
            "Tprobe_core_side_leg_pos": (88.0, 80.0),
            "Tprobe_core_top_yoke_neg": (92.0, 84.0),
            "Tprobe_core_top_yoke_pos": (91.0, 83.0),
        }
        _ipk, row = self._run(
            None,
            [complete],
            probe_names=[
                "Tprobe_core_center_leg_neg",
                "Tprobe_core_center_leg_pos",
                "Tprobe_core_side_leg_neg",
                "Tprobe_core_side_leg_pos",
                "Tprobe_core_top_yoke_neg",
                "Tprobe_core_top_yoke_pos",
            ],
        )
        self.assertEqual(row["Tprobe_core_center_leg_max"], 86.0)
        self.assertEqual(row["Tprobe_core_side_leg_max"], 90.0)
        self.assertEqual(row["Tprobe_core_top_yoke_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_mean"], 84.0)

    def test_none_and_false_analyze_returns_are_validated_from_data(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        for analyze_result in (None, False):
            with self.subTest(analyze_result=analyze_result):
                ipk, row = self._run(analyze_result, [complete])
                self.assertEqual(ipk.analyze_calls, 1)
                self.assertEqual(row["thermal_solved"], 1)
                self.assertEqual(row["thermal_extraction_complete"], 1)
                self.assertEqual(row["thermal_required_group_mask"], 11)
                self.assertEqual(row["thermal_required_group_count"], 3)
                self.assertTrue(math.isnan(row["T_max_Rx_side"]))
                self.assertEqual(ipk.oproject.active_calls, 3)
                self.assertIs(ipk._odesign, ipk.oproject.last_design)
                self.assertIs(ipk.design_solutions._odesign, ipk.oproject.last_design)
                self.assertFalse(
                    ipk.setup.props["Solution Initialization - Use Model Based Flow Initialization"]
                )
                self.assertEqual(ipk.setup.props["Under-relaxation - Pressure"], "0.7")
                self.assertFalse(
                    ipk.setup.props["Sequential Solve of Flow and Energy Equations"]
                )
                self.assertEqual(ipk.setup.props["Convergence Criteria - Flow"], "0.001")
                self.assertEqual(ipk.setup.props["Convergence Criteria - Energy"], "1e-07")

    def test_pooled_q7_stale_solve_project_is_rebound_before_extraction(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        initial_project = _ProjectHandle("thermal_test")
        solve_project = _SingleActivationProjectHandle("thermal_test")
        postflight_project = _ProjectHandle("thermal_test")
        extraction_project = _ProjectHandle("thermal_test")
        field_summary_project = _ProjectHandle("thermal_test")

        ipk, row = self._run(
            None,
            [complete],
            pooled=True,
            rebind_sequence=[
                initial_project,
                solve_project,
                postflight_project,
                extraction_project,
                field_summary_project,
            ],
        )

        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(ipk.rebind_project_mock.call_count, 5)
        ipk.native_pipeline_barrier_mock.assert_called_once_with()
        self.assertGreaterEqual(ipk.native_window_mock.call_count, 3)
        self.assertEqual(solve_project.active_calls, 1)
        self.assertEqual(
            solve_project.last_design.analyze_calls,
            [("ThermalSetup", True)],
        )
        self.assertIs(ipk.oproject, field_summary_project)
        self.assertEqual(postflight_project.active_calls, 1)
        self.assertGreaterEqual(extraction_project.active_calls, 1)
        self.assertGreaterEqual(field_summary_project.active_calls, 2)

    def test_post_solve_extraction_never_queries_object_dimension_proxy(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_Tx_side": (86.0, 74.0),
        }

        _ipk, row = self._run(
            None,
            [complete],
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
            object_factory=_NoLiveDimensionObject,
            probe_factory=_NoLiveDimensionObject,
        )

        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["Tprobe_Tx_side_max"], 86.0)
        self.assertEqual(row["T_max_core"], 91.0)

    def test_post_solve_rebind_failure_is_explicit_and_fail_closed(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "thermal post-solve exact project/design rebind failed",
        ):
            self._run(
                None,
                [complete],
                rebind_sequence=[
                    _ProjectHandle("thermal_test"),
                    _ProjectHandle("thermal_test"),
                    RuntimeError("q7 exact project is no longer available"),
                ],
            )

    def test_false_with_missing_monitor_retries_once_and_accepts_fresh_convergence(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            [False, None],
            [complete],
            convergence=[
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._convergence(),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 2)
        self.assertEqual(row["thermal_solve_attempts"], 2)
        self.assertEqual(row["thermal_analyze_call_ok"], 1)
        self.assertEqual(row["thermal_analyze_return_false"], 1)
        self.assertEqual(row["thermal_solved"], 1)
        ipk.sleep_mock.assert_not_called()
        self.assertEqual(ipk.telemetry_mock.call_count, 3)

    def test_exception_with_malformed_monitor_does_not_double_dispatch(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            [RuntimeError("AnalyzeAll failed"), None],
            [complete],
            convergence=[
                self._monitor_failure("monitor_malformed"),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(row["thermal_solve_attempts"], 1)
        self.assertEqual(row["thermal_analyze_call_ok"], 0)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_convergence_reason"], "monitor_malformed")
        ipk.sleep_mock.assert_not_called()

    def test_retry_is_capped_at_two_attempts(self):
        ipk, row = self._run(
            [False, False],
            [{}],
            convergence=[
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_malformed"),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 2)
        self.assertEqual(ipk.telemetry_mock.call_count, 3)
        self.assertEqual(row["thermal_solve_attempts"], 2)
        self.assertEqual(row["thermal_solved"], 0)
        ipk.sleep_mock.assert_not_called()

    def test_successful_invocation_with_missing_monitor_does_not_retry(self):
        ipk, row = self._run(
            None,
            [{}],
            convergence=self._monitor_failure("monitor_missing"),
        )
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(row["thermal_solve_attempts"], 1)
        self.assertEqual(row["thermal_convergence_reason"], "monitor_missing")
        self.assertEqual(row["thermal_solved"], 0)
        ipk.sleep_mock.assert_not_called()

    def test_unconverged_residuals_skip_field_summary_and_fail_gate(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(None, [complete], convergence=self._convergence(False))
        self.assertEqual(ipk.field_summary_calls, 0)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_converged"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_convergence_reason"], "residual_threshold")
        self.assertEqual(row["thermal_solve_attempts"], 1)
        ipk.sleep_mock.assert_not_called()

    def test_partial_field_summary_preserves_values_but_fails_gate(self):
        partial = {
            "Tx_main_0": (81.0, 70.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(None, [partial, partial, partial])
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_missing_count"], 2)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertEqual(row["T_max_Tx"], 81.0)
        self.assertEqual(row["T_max_core"], 91.0)
        self.assertTrue(math.isnan(row["T_max_Rx_main"]))
        self.assertEqual(row["thermal_calculator_attempts"], 0)
        self.assertEqual(ipk.oproject.active_calls, 5)

    def test_one_missing_modeled_tx_turn_invalidates_tx_group(self):
        missing_second_tx = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [missing_second_tx] * 3,
            tx_count=2,
        )

        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertTrue(math.isnan(row["T_max_Tx"]))
        self.assertEqual(row["T_max_Tx_main_0"], 81.0)
        self.assertTrue(math.isnan(row["T_max_Tx_main_1"]))

    def test_missing_tx_side_probe_is_optional_only_without_tx_side_turns(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups],
            n1_side=0,
            probe_names=["Tprobe_Tx_side"],
        )
        self.assertEqual(ipk.field_summary_calls, 1)
        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["thermal_missing_count"], 2)
        self.assertTrue(math.isnan(row["Tprobe_Tx_side_max"]))
        self.assertTrue(math.isnan(row["Tprobe_Tx_side_mean"]))

        _ipk, row = self._run(
            None,
            [complete_groups],
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
        )
        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_missing_count"], 2)

    def test_missing_probe_uses_bounded_saved_field_scalar_fallback(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups] * 3,
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
            scalar_responses={
                ("Tprobe_Tx_side", "Maximum"): 86.0,
                ("Tprobe_Tx_side", "Mean"): 74.0,
            },
        )

        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_calculator_attempts"], 2)
        self.assertEqual(len(ipk.scalar_calls), 2)
        self.assertTrue(all(call[1] == "surface" for call in ipk.scalar_calls))
        self.assertEqual(row["Tprobe_Tx_side_max"], 86.0)
        self.assertEqual(row["Tprobe_Tx_side_mean"], 74.0)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["thermal_probe_failure_count"], 0)
        self.assertEqual(json.loads(row["thermal_probe_failures_json"]), [])

    def test_failed_probe_fallback_is_structured_and_stays_quarantinable(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups] * 3,
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
        )

        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_extraction_failure_reason"],
                         "required_probe_temperature_missing")
        self.assertEqual(row["thermal_probe_failure_count"], 1)
        [failure] = json.loads(row["thermal_probe_failures_json"])
        self.assertEqual(failure["probe"], "Tprobe_Tx_side")
        self.assertEqual(failure["stage"], "extraction")
        self.assertEqual(failure["reason"], "saved_field_fallback_exhausted")
        self.assertEqual(
            failure["columns"],
            ["Tprobe_Tx_side_max", "Tprobe_Tx_side_mean"],
        )
        self.assertEqual(len(ipk.scalar_calls), 2)

    def test_report_failure_does_not_launch_another_solve(self):
        ipk, row = self._run(None, [False, False, False])
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_solution_data_available"], 0)
        self.assertEqual(row["thermal_missing_count"], 6)

    def test_side_requirement_comes_from_input_not_surviving_objects(self):
        missing_side = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        _ipk, row = self._run(None, [missing_side] * 3, include_side=True)
        self.assertEqual(row["thermal_required_group_mask"], 15)
        self.assertEqual(row["thermal_required_group_count"], 4)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertEqual(row["thermal_solved"], 0)

        complete = dict(missing_side, Rx_side_0=(89.0, 73.0))
        _ipk, row = self._run(None, [complete], include_side=True)
        self.assertEqual(row["thermal_required_group_mask"], 15)
        self.assertEqual(row["thermal_required_missing_count"], 0)
        self.assertEqual(row["thermal_solved"], 1)

    def test_setup_creation_and_update_fail_hard(self):
        with self.assertRaisesRegex(RuntimeError, "create_setup returned no ThermalSetup"):
            self._run(None, [{}], setup_result=False)
        with self.assertRaisesRegex(RuntimeError, "ThermalSetup update returned False"):
            self._run(None, [{}], setup_update_result=False)

    def test_required_loss_key_missing_fails_hard(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={},
        )
        with self.assertRaisesRegex(KeyError, "required thermal loss key missing: P_turn_Tx_main_0"):
            thermal._assign_losses(SimpleNamespace(), sim, self._loss_objects(), mode="full")

    def test_solid_block_false_return_is_not_recorded(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )
        ipk = SimpleNamespace(assign_solid_block=Mock(return_value=False))
        with self.assertRaisesRegex(RuntimeError, "solid block source for Tx_main_0 returned no boundary"):
            thermal._assign_losses(ipk, sim, self._loss_objects(), mode="full")
        self.assertFalse(hasattr(sim, "thermal_injected"))

    def test_solid_block_props_are_validated_before_recording(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )
        wrong = _Boundary({
            "Block Type": "Solid",
            "Objects": ["Tx_main_0"],
            "Total Power": "0W",
        })
        ipk = SimpleNamespace(assign_solid_block=Mock(return_value=wrong))
        with self.assertRaisesRegex(RuntimeError, "property 'Total Power' mismatch"):
            thermal._assign_losses(ipk, sim, self._loss_objects(), mode="full")
        self.assertFalse(hasattr(sim, "thermal_injected"))

    def test_solid_block_success_records_injected_loss(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            self._loss_objects(),
            mode="full",
        )
        self.assertEqual(injected, {"Tx_main_0": 12.5})
        self.assertEqual(sim.thermal_injected, injected)

    def test_explicit_insulation_never_receives_em_loss(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={
                "P_turn_Tx_main_0": 12.5,
                "P_Rx_main_group": 0.0,
            },
        )
        objects = self._loss_objects()
        objects.update({
            "Rx_main_insulation": [
                _Object("Rx_main_insulation_gap_0")
            ],
            "Rx_side_insulation": [
                _Object("Rx_side_insulation_gap_0")
            ],
            "Rx_side2_insulation": [
                _Object("Rx_side2_insulation_gap_0")
            ],
        })

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="full",
        )
        self.assertEqual(injected, {"Tx_main_0": 12.5})
        self.assertFalse(
            any("insulation" in name for name in injected)
        )

    def test_blocks_only_rx_side_still_receives_group_loss(self):
        side_block = _Object("Rx_side_block", volume=4.0)
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["Rx_side_blocks"] = [side_block]
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_Rx_main_group": 0.0, "P_Rx_side_group": 25.0},
        )

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="full",
        )
        self.assertEqual(injected, {"Rx_side_block": 25.0})

    def test_single_rx_turn_receives_exact_group_loss_without_turn_report(self):
        side_turn = _Object("Rx_side_0_0", volume=4.0)
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["Rx_side_explicit"] = [side_turn]
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0], "n_explicit_turns": [0]}),
            loss_map_phys={"P_Rx_main_group": 0.0, "P_Rx_side_group": 25.0},
        )

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="full",
        )

        self.assertEqual(injected, {"Rx_side_0_0": 25.0})
        self.assertEqual(sim.thermal_rx_power_balance[-1]["assigned_w"], 25.0)
        self.assertEqual(sim.thermal_rx_model, "homogenized_blocks")

    def test_segmented_core_loss_is_volume_weighted_once_per_group(self):
        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        frame = pd.DataFrame({
            "n_core_group": [1],
            "w1": [8.0],
            "core_plate_t": [0.0],
            "core_plate_pad_t": [0.0],
            "n_explicit_turns": [0],
        })
        pieces = [
            _Object("core_1_leg_left", volume=1.0),
            _Object("core_1_leg_center", volume=1.0),
            _Object("core_1_yoke_top", volume=2.0),
        ]
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["core"] = pieces
        sim = SimpleNamespace(
            df_plus=frame,
            loss_map_phys={"P_Rx_main_group": 0.0, "P_core_1": 80.0},
        )
        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="eighth",
        )
        self.assertEqual(
            injected,
            {
                "core_1_leg_left": 2.5,
                "core_1_leg_center": 2.5,
                "core_1_yoke_top": 5.0,
            },
        )
        self.assertEqual(sum(injected.values()), 10.0)
        self.assertEqual(sim.thermal_core_expected_injected_w, 10.0)

        # The unsegmented branch must not require a volume and must retain the
        # exact pre-extension source assignment.
        legacy_objects = self._loss_objects()
        legacy_objects["Tx"] = []
        legacy_objects["core"] = [SimpleNamespace(name="core_1")]
        legacy_sim = SimpleNamespace(
            df_plus=frame,
            loss_map_phys={"P_Rx_main_group": 0.0, "P_core_1": 80.0},
        )
        legacy_injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            legacy_sim,
            legacy_objects,
            mode="eighth",
        )
        self.assertEqual(legacy_injected, {"core_1": 10.0})

    def test_fixed_temperature_uses_supported_condition_and_props(self):
        ipk = _BoundaryIcepak()
        objs = {
            "core_plates": [_Object("core_plate")],
            "wcp_plates": [_Object("wcp_plate")],
        }
        thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")
        self.assertEqual(len(ipk.source_calls), 1)
        call = ipk.source_calls[0]
        self.assertEqual(call["thermal_condition"], "Temperature")
        self.assertEqual(call["assignment_value"], "45.0cel")
        boundary = ipk.source_boundaries[0]
        self.assertEqual(boundary.update_calls, 1)
        self.assertEqual(boundary.props["Thermal Condition"], "Fixed Temperature")
        self.assertEqual(boundary.props["Temperature"], "45.0cel")
        self.assertTrue(boundary.auto_update)
        self.assertEqual(ipk.ambient_calls, [50.0])

    def test_fixed_temperature_props_mismatch_fails_hard(self):
        ipk = _BoundaryIcepak(fixed_post_update_props={"Temperature": "AmbientTemp"})
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "property 'Temperature' mismatch"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_fixed_temperature_update_false_fails_hard(self):
        ipk = _BoundaryIcepak(fixed_update_result=False)
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "fixed temperature source.*update failed"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_fixed_temperature_false_return_fails_hard(self):
        ipk = _BoundaryIcepak(fail_boundary="cold_plates_fixed_T")
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "fixed temperature source.*returned no boundary"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_any_opening_or_symmetry_false_return_fails_hard(self):
        objs = {"core_plates": [], "wcp_plates": []}
        for boundary_name in ("sym_x0", "outlet_z"):
            with self.subTest(boundary_name=boundary_name):
                ipk = _BoundaryIcepak(fail_boundary=boundary_name)
                with self.assertRaisesRegex(
                    RuntimeError, f"thermal boundary {boundary_name} returned no boundary"
                ):
                    thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_required_geometry_validates_full_side_groups_separately(self):
        base = {
            "core": [_Object("core")],
            "Tx": [_Object("Tx")],
            "Rx_main_explicit": [_Object("Rx_main")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [_Object("Rx_side")],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [_Object("Rx_side2")],
            "Rx_side2_blocks": [],
        }
        thermal._require_thermal_geometry(base, "full", 1)

        missing_side2 = dict(base, Rx_side2_explicit=[])
        with self.assertRaisesRegex(RuntimeError, "Rx_side2"):
            thermal._require_thermal_geometry(missing_side2, "full", 1)

        missing_side = dict(base, Rx_side_explicit=[])
        with self.assertRaisesRegex(RuntimeError, "Rx_side"):
            thermal._require_thermal_geometry(missing_side, "full", 1)

        thermal._require_thermal_geometry(missing_side2, "eighth", 1)

    def test_required_geometry_includes_enabled_cooling_hardware(self):
        base = {
            "core": [_Object("core")],
            "Tx": [_Object("Tx")],
            "Rx_main_explicit": [_Object("Rx_main")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core_plates": [],
            "core_pads": [],
            "wcp_plates": [],
            "wcp_pads": [],
        }
        for keyword, expected in (
            ("require_core_plates", "core_plates"),
            ("require_core_pads", "core_pads"),
            ("require_wcp_plates", "wcp_plates"),
            ("require_wcp_pads", "wcp_pads"),
        ):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(RuntimeError, expected):
                thermal._require_thermal_geometry(base, "eighth", 0, **{keyword: True})

    def test_wcp_pad_mesh_region_padding_clips_contacts_and_symmetry_planes(self):
        full = thermal._wcp_pad_mesh_region_padding_mm("full")
        quarter = thermal._wcp_pad_mesh_region_padding_mm("quarter")
        eighth = thermal._wcp_pad_mesh_region_padding_mm("eighth")

        self.assertEqual(
            list(full.values()),
            [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
        )
        self.assertEqual(
            list(quarter.values()),
            [0.0, 2.0, 0.0, 0.0, 2.0, 2.0],
        )
        self.assertEqual(
            list(eighth.values()),
            [0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        )
        for padding in (full, quarter, eighth):
            self.assertTrue(
                any(
                    padding[direction] > 0.0
                    for direction in ("+X", "-X", "+Z", "-Z")
                )
            )
        with self.assertRaisesRegex(ValueError, "unsupported thermal symmetry"):
            thermal._wcp_pad_mesh_region_padding_mm("unknown")

    def test_candidate_thermal_mesh_is_local_object_separated_and_exactly_covered(self):
        operations = []
        mesh_regions = []

        def assign_mesh_level(levels, name):
            level = next(iter(set(levels.values())))
            operation = SimpleNamespace(
                name=name,
                props={
                    "Command": "AssignMeshLevel",
                    "Objects": list(levels),
                    "Level": str(level),
                },
                auto_update=True,
                update=Mock(return_value=True),
            )
            operations.append(operation)
            mesh.meshoperations.append(operation)
            return [name]

        def assign_mesh_region(assignment, level, name):
            self.assertEqual(len(assignment), 1)
            object_name = str(assignment[0])
            subregion = SimpleNamespace(
                name=f"{name}_subregion",
                parts={object_name: _Object(object_name)},
            )
            region = SimpleNamespace(
                name=name,
                enable=True,
                settings={"MeshRegionResolution": int(level)},
                manual_settings=False,
                assignment=subregion,
                update=Mock(return_value=True),
            )
            mesh_regions.append(region)
            return region

        mesh = SimpleNamespace(
            assign_mesh_level=Mock(side_effect=assign_mesh_level),
            assign_mesh_region=Mock(side_effect=assign_mesh_region),
            meshoperations=[],
        )
        core_plates = []
        core_pads = []
        for index in range(5):
            for side in ("side_left", "center", "side_right"):
                core_plates.append(_Object(
                    f"core_plate_{index}_{side}"
                ))
                core_pads.extend([
                    _Object(f"core_plate_pad_{index}_a_{side}"),
                    _Object(f"core_plate_pad_{index}_b_{side}"),
                ])
        wcp_plates = []
        wcp_pads = []
        for index in (1, 2):
            for side in ("p", "n"):
                wcp_plates.append(_Object(
                    f"Tx_main_wcp_{index}_{side}"
                ))
                wcp_pads.extend([
                    _Object(
                        f"Tx_main_wcp_pad_{index}_in_{side}"
                    ),
                    _Object(
                        f"Tx_main_wcp_pad_{index}_out_{side}"
                    ),
                ])
        objs = {
            "core_plates": core_plates,
            "core_pads": core_pads,
            "wcp_plates": wcp_plates,
            "wcp_pads": wcp_pads,
            "Tx": [_Object(f"Tx_main_{i}") for i in range(6)],
            "Rx_main_blocks": [
                _Object(f"Rx_main_block_{i}") for i in range(4)
            ],
            "Rx_side_blocks": [
                _Object(f"Rx_side_block_{i}") for i in range(4)
            ],
            "Rx_side2_blocks": [
                _Object(f"Rx_side2_block_{i}") for i in range(4)
            ],
            "Rx_main_explicit": [
                _Object(f"Rx_main_{i}_0") for i in (0, 1, 34, 35)
            ],
            "Rx_side_explicit": [
                _Object(f"Rx_side_{i}_0") for i in (0, 1, 22, 23)
            ],
            "Rx_side2_explicit": [
                _Object(f"Rx_side2_{i}_0") for i in (0, 1, 22, 23)
            ],
            "Rx_main_insulation": [
                _Object("Rx_main_insulation_gap_0"),
                _Object("Rx_main_insulation_gap_34"),
            ],
            "Rx_side_insulation": [
                _Object("Rx_side_insulation_gap_0"),
                _Object("Rx_side_insulation_gap_22"),
            ],
            "Rx_side2_insulation": [
                _Object("Rx_side2_insulation_gap_0"),
                _Object("Rx_side2_insulation_gap_22"),
            ],
        }

        plan = thermal._assign_thermal_mesh(
            SimpleNamespace(
                mesh=mesh,
                modeler=SimpleNamespace(model_units="mm"),
            ),
            objs,
            mode="eighth",
        )

        self.assertEqual(plan["core_plate_assembly_count"], 15)
        self.assertEqual(plan["wcp_assembly_count"], 4)
        self.assertEqual(plan["rx_retained_pack_count"], 3)
        self.assertEqual(plan["operation_count"], 37)
        self.assertEqual(plan["assigned_object_count"], 93)
        self.assertEqual(plan["required_thin_object_count"], 56)
        self.assertEqual(plan["required_objects_missing"], [])
        self.assertEqual(plan["shared_operation_count"], 3)
        self.assertEqual(plan["rx_block_shared_pack_count"], 3)
        self.assertEqual(plan["separate_object_operation_count"], 26)
        self.assertEqual(plan["object_level_operation_count"], 29)
        self.assertEqual(plan["mesh_region_operation_count"], 8)
        self.assertEqual(plan["wcp_pad_mesh_region_count"], 8)
        by_name = {item["name"]: item for item in plan["operations"]}
        self.assertIs(
            by_name["rx_main_block_mesh_level"]["separate_objects"],
            False,
        )
        self.assertIs(
            by_name["rx_side_block_mesh_level"]["separate_objects"],
            False,
        )
        self.assertIs(
            by_name["rx_side2_block_mesh_level"]["separate_objects"],
            False,
        )
        self.assertNotIn("pad_mesh_level", by_name)
        self.assertNotIn("rx_mesh_level", by_name)
        self.assertEqual(
            by_name["rx_main_retained_pack_mesh_level"]["level"], 5
        )
        self.assertEqual(
            by_name["rx_side_retained_pack_mesh_level"]["level"], 5
        )
        self.assertEqual(
            by_name["rx_side2_retained_pack_mesh_level"]["level"], 5
        )
        self.assertEqual(
            by_name["rx_main_insulation_mesh_level"]["level"], 4
        )
        self.assertEqual(
            by_name["wcp_assembly_mesh_level_1_p"]["level"], 5
        )
        self.assertEqual(
            by_name["wcp_assembly_mesh_level_1_p"]["objects"],
            ["Tx_main_wcp_1_p"],
        )
        self.assertEqual(
            by_name["wcp_pad_mesh_region_1_in_p"]["objects"],
            ["Tx_main_wcp_pad_1_in_p"],
        )
        self.assertEqual(
            by_name[
                "wcp_pad_mesh_region_1_in_p"
            ]["native_region_object_name"],
            "wcp_pad_mesh_region_1_in_p_subregion",
        )
        self.assertEqual(
            by_name[
                "core_plate_assembly_mesh_level_0_center"
            ]["level"],
            2,
        )
        for operation in operations:
            operation.update.assert_called_once_with()
            self.assertNotIn("Command", operation.props)
            expected_separate = operation.name not in {
                "rx_main_block_mesh_level",
                "rx_side_block_mesh_level",
                "rx_side2_block_mesh_level",
            }
            self.assertIs(
                operation.props["Mesh Object(s) Separately Enabled"],
                expected_separate,
            )
        self.assertEqual(len(mesh_regions), 8)
        for region in mesh_regions:
            region.update.assert_called_once_with()
            self.assertFalse(region.manual_settings)
            self.assertEqual(
                region.settings["MeshRegionResolution"], 5
            )
            self.assertEqual(
                region.assignment.padding_types,
                ["Absolute Offset"] * 6,
            )
            self.assertEqual(
                region.assignment.padding_values,
                ["0mm", "2mm", "0mm", "0mm", "2mm", "0mm"],
            )
        self.assertEqual(plan["symmetry_mode"], "eighth")
        self.assertEqual(
            list(plan["wcp_pad_padding_by_direction_mm"].values()),
            [0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        )

    def test_explicit_insulation_requires_retained_copper(self):
        objs = {
            "core_plates": [],
            "core_pads": [],
            "wcp_plates": [],
            "wcp_pads": [],
            "Tx": [],
            "Rx_main_blocks": [],
            "Rx_side_blocks": [],
            "Rx_side2_blocks": [],
            "Rx_main_explicit": [],
            "Rx_side_explicit": [],
            "Rx_side2_explicit": [],
            "Rx_main_insulation": [_Object("Rx_main_insulation_gap_0")],
            "Rx_side_insulation": [],
            "Rx_side2_insulation": [],
        }
        mesh = SimpleNamespace(
            assign_mesh_level=Mock(),
            meshoperations=[],
        )
        with self.assertRaisesRegex(
            RuntimeError, "exists without retained copper"
        ):
            thermal._assign_thermal_mesh(
                SimpleNamespace(mesh=mesh), objs
            )

    def test_thermal_mesh_failures_are_not_silenced(self):
        empty = {
            "core_plates": [],
            "wcp_pads": [],
            "wcp_plates": [],
            "core_pads": [],
            "Tx": [],
            "Rx_main_blocks": [],
            "Rx_side_blocks": [],
            "Rx_side2_blocks": [],
            "Rx_main_explicit": [_Object("Rx_main_0")],
            "Rx_side_explicit": [],
            "Rx_side2_explicit": [],
            "Rx_main_insulation": [],
            "Rx_side_insulation": [],
            "Rx_side2_insulation": [],
        }
        mesh = SimpleNamespace(
            assign_mesh_level=Mock(return_value=[]),
            meshoperations=[],
        )
        with self.assertRaisesRegex(
            RuntimeError, "rx_main_retained_pack_mesh_level assignment"
        ):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        failed_rx_op = SimpleNamespace(
            name="rx_main_single", props={}, auto_update=True, update=Mock(return_value=False))
        mesh.assign_mesh_level.return_value = ["rx_main_single"]
        mesh.meshoperations = [failed_rx_op]
        with self.assertRaisesRegex(
                RuntimeError, "rx_main_retained_pack_mesh_level mesh operation update failed"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        incomplete_wcp = dict(
            empty,
            Rx_main_explicit=[],
            wcp_pads=[_Object("Tx_main_wcp_pad_1_in_p")],
            wcp_plates=[_Object("Tx_main_wcp_1_p")],
        )
        with self.assertRaisesRegex(
            RuntimeError, "incomplete wcp assembly"
        ):
            thermal._assign_thermal_mesh(
                SimpleNamespace(mesh=mesh), incomplete_wcp
            )

    @staticmethod
    def _minimal_mesh_plan():
        return {
            "schema": thermal.THERMAL_MESH_PLAN_CONTRACT_VERSION,
            "policy": thermal.THERMAL_MESH_POLICY,
            "plan_sha256": "b" * 64,
            "operations": [{
                "name": "thin_mesh",
                "category": "test",
                "operation_type": "object_level",
                "level": 5,
                "objects": ["pad_a", "pad_b"],
                "shared_region": False,
                "separate_objects": True,
                "actual_operation_names": ["thin_mesh_L_5"],
            }],
            "operation_count": 1,
            "assigned_object_count": 2,
            "required_thin_object_count": 2,
            "required_thin_objects": ["pad_a", "pad_b"],
            "required_objects_missing": [],
            "core_plate_assembly_count": 0,
            "wcp_assembly_count": 1,
            "wcp_pad_mesh_region_count": 0,
            "rx_retained_pack_count": 0,
            "rx_block_shared_pack_count": 0,
            "rx_block_shared_packs": [],
            "rx_main_block_objects": [],
            "shared_operation_count": 0,
            "object_level_operation_count": 1,
            "mesh_region_operation_count": 0,
            "separate_object_operation_count": 1,
        }

    def test_native_mesh_readback_resolves_ids_and_requires_separate_object_props(self):
        props = {
            "Assignment": [101, 102],
            "Level": "5",
            "Mesh Object(s) Separately Enabled": True,
        }
        operation_child = SimpleNamespace(
            GetPropNames=lambda: list(props),
            GetPropValue=lambda name: props[name],
        )
        mesh_child = SimpleNamespace(
            GetChildNames=lambda: ["thin_mesh_L_5"],
            GetChildObject=lambda _name: operation_child,
        )
        native_design = SimpleNamespace(
            GetChildObject=lambda name: (
                mesh_child if name == "Mesh" else None
            )
        )
        editor = SimpleNamespace(
            GetObjectNameByID=lambda object_id: {
                101: "pad_a",
                102: "pad_b",
            }[object_id]
        )

        evidence = thermal._native_thermal_mesh_operation_readback(
            native_design,
            self._minimal_mesh_plan(),
            native_editor=editor,
        )
        self.assertEqual(evidence["assigned_object_count"], 2)
        self.assertEqual(evidence["required_thin_object_count"], 2)
        self.assertEqual(evidence["required_thin_objects_missing"], [])
        self.assertEqual(
            evidence["operation_readbacks"][0]["level_schema"], "Level"
        )
        self.assertIsNone(
            evidence["operation_readbacks"][0]["incr_level"]
        )

        props["Mesh Object(s) Separately Enabled"] = False
        with self.assertRaisesRegex(
            RuntimeError, "object-separation readback mismatch"
        ):
            thermal._native_thermal_mesh_operation_readback(
                native_design,
                self._minimal_mesh_plan(),
                native_editor=editor,
            )

    @staticmethod
    def _mesh_region_plan():
        return {
            "schema": thermal.THERMAL_MESH_PLAN_CONTRACT_VERSION,
            "policy": thermal.THERMAL_MESH_POLICY,
            "plan_sha256": "c" * 64,
            "operations": [{
                "name": "wcp_pad_mesh_region_1_in_p",
                "category": "wcp_pad_region",
                "operation_type": "mesh_region",
                "level": 5,
                "objects": ["Tx_main_wcp_pad_1_in_p"],
                "shared_region": False,
                "separate_objects": None,
                "actual_operation_names": [
                    "wcp_pad_mesh_region_1_in_p"
                ],
                "native_region_object_name": "SubRegionPad1",
            }],
            "operation_count": 1,
            "assigned_object_count": 1,
            "required_thin_object_count": 1,
            "required_thin_objects": ["Tx_main_wcp_pad_1_in_p"],
            "required_objects_missing": [],
            "core_plate_assembly_count": 0,
            "wcp_assembly_count": 1,
            "wcp_pad_mesh_region_count": 1,
            "rx_retained_pack_count": 0,
            "rx_block_shared_pack_count": 0,
            "rx_block_shared_packs": [],
            "rx_main_block_objects": [],
            "shared_operation_count": 0,
            "object_level_operation_count": 0,
            "mesh_region_operation_count": 1,
            "separate_object_operation_count": 0,
        }

    @staticmethod
    def _grid_mapping_text(
        region_name, parent_region, object_domains, overlap_faces=()
    ):
        object_blocks = []
        for index, (name, domains) in enumerate(object_domains.items()):
            values = ",".join(map(str, domains))
            object_blocks.extend([
                f"$begin 'Object{index}'",
                f"Name='{name}'",
                f"Domains[{len(domains)}:{values}]",
                f"$end 'Object{index}'",
            ])
        overlaps = ",".join(map(str, overlap_faces))
        return "\n".join([
            "$begin 'MeshRegion'",
            f"Name='{region_name}'",
            f"ParentRegion={parent_region}",
            "HasMesh=true",
            "$begin 'Objects'",
            f"Count={len(object_domains)}",
            *object_blocks,
            "$end 'Objects'",
            f"OverlappingMRFaces[{len(overlap_faces)}:{overlaps}]",
            "$end 'MeshRegion'",
            "",
        ])

    @staticmethod
    def _write_grid_mapping_artifact(root, name, text):
        directory = Path(root, name)
        directory.mkdir()
        mapping = directory / "grid_mapping"
        output = directory / "grid_output"
        mapping.write_text(text, encoding="utf-8")
        output.write_bytes(b"native-grid")
        mapping_signature = thermal._thermal_monitor_signature(mapping)
        return {
            "directory": str(directory),
            "name": name,
            "grid_output_size": output.stat().st_size,
            "grid_output_sha256_sample": (
                thermal._thermal_monitor_signature(output)[2]
            ),
            "grid_mapping_size": mapping_signature[0],
            "grid_mapping_sha256_sample": mapping_signature[2],
            "mtime_ns": mapping_signature[1],
        }

    def test_grid_mapping_coverage_accepts_coupled_pad_region(self):
        plan = self._mesh_region_plan()
        with tempfile.TemporaryDirectory() as tmp:
            global_artifact = self._write_grid_mapping_artifact(
                tmp,
                "DV1_Meshes0_V0.sd",
                self._grid_mapping_text(
                    "GlobalRegion", 0, {"Fluid": (1,)}
                ),
            )
            local_artifact = self._write_grid_mapping_artifact(
                tmp,
                "DV1_Meshes1_V0.sd",
                self._grid_mapping_text(
                    "wcp_pad_mesh_region_1_in_p",
                    6,
                    {"Tx_main_wcp_pad_1_in_p": (17,)},
                    overlap_faces=(101, 102),
                ),
            )

            evidence = thermal._thermal_mesh_mapping_coverage(
                plan, [global_artifact, local_artifact]
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["required_objects_missing"], [])
        self.assertEqual(evidence["uncoupled_local_regions"], [])
        local = next(
            item for item in evidence["readbacks"]
            if item["region_name"] == "wcp_pad_mesh_region_1_in_p"
        )
        self.assertEqual(local["overlapping_mr_face_count"], 2)

    def test_grid_mapping_coverage_rejects_disconnected_pad_region(self):
        plan = self._mesh_region_plan()
        with tempfile.TemporaryDirectory() as tmp:
            global_artifact = self._write_grid_mapping_artifact(
                tmp,
                "DV1_Meshes0_V0.sd",
                self._grid_mapping_text(
                    "GlobalRegion", 0, {"Fluid": (1,)}
                ),
            )
            disconnected = self._write_grid_mapping_artifact(
                tmp,
                "DV1_Meshes1_V0.sd",
                self._grid_mapping_text(
                    "wcp_pad_mesh_region_1_in_p",
                    6,
                    {"Tx_main_wcp_pad_1_in_p": (17,)},
                    overlap_faces=(),
                ),
            )

            evidence = thermal._thermal_mesh_mapping_coverage(
                plan, [global_artifact, disconnected]
            )

        self.assertFalse(evidence["passed"])
        self.assertEqual(
            evidence["uncoupled_local_regions"],
            ["wcp_pad_mesh_region_1_in_p"],
        )
        self.assertEqual(evidence["required_objects_missing"], [])

    def test_native_mesh_region_readback_requires_exact_part_level_and_enable(self):
        operation_props = {
            "Assignment": ["SubRegionPad1"],
            "Mesh Resolution": "5",
            "Enabled": True,
            "Enclosing Geometries": "Tx_main_wcp_pad_1_in_p",
        }
        operation_child = SimpleNamespace(
            GetPropNames=lambda: list(operation_props),
            GetPropValue=lambda name: operation_props[name],
        )
        mesh_child = SimpleNamespace(
            GetChildNames=lambda: ["wcp_pad_mesh_region_1_in_p"],
            GetChildObject=lambda _name: operation_child,
        )
        history_props = {
            "Part Names": ["Tx_main_wcp_pad_1_in_p"],
        }
        history_child = SimpleNamespace(
            GetPropNames=lambda: list(history_props),
            GetPropValue=lambda name: history_props[name],
        )
        region_child = SimpleNamespace(
            GetChildNames=lambda: ["CreateSubRegion1"],
            GetChildObject=lambda _name: history_child,
        )
        native_design = SimpleNamespace(
            GetChildObject=lambda name: (
                mesh_child if name == "Mesh" else None
            )
        )
        editor = SimpleNamespace(
            GetChildObject=lambda name: (
                region_child if name == "SubRegionPad1" else None
            ),
        )

        evidence = thermal._native_thermal_mesh_operation_readback(
            native_design,
            self._mesh_region_plan(),
            native_editor=editor,
        )
        self.assertEqual(evidence["assigned_object_count"], 1)
        self.assertEqual(evidence["object_level_operation_count"], 0)
        self.assertEqual(evidence["mesh_region_operation_count"], 1)
        self.assertTrue(evidence["mesh_region_part_readback_passed"])
        self.assertEqual(
            evidence["operation_readbacks"][0]["region_object_name"],
            "SubRegionPad1",
        )
        self.assertEqual(
            evidence["operation_readbacks"][0]["region_object_count"], 1
        )

        for key, bad_value, pattern in (
            ("Mesh Resolution", "4", "level readback mismatch"),
            ("Enabled", False, "mesh region is disabled"),
            (
                "Enclosing Geometries",
                "wrong_pad",
                "enclosing-geometry mismatch",
            ),
        ):
            original = operation_props[key]
            operation_props[key] = bad_value
            with self.assertRaisesRegex(RuntimeError, pattern):
                thermal._native_thermal_mesh_operation_readback(
                    native_design,
                    self._mesh_region_plan(),
                    native_editor=editor,
                )
            operation_props[key] = original

        history_props["Part Names"] = ["wrong_pad"]
        with self.assertRaisesRegex(
            RuntimeError, "mesh-region part readback mismatch"
        ):
            thermal._native_thermal_mesh_operation_readback(
                native_design,
                self._mesh_region_plan(),
                native_editor=editor,
            )

    def _native_mesh_range_readback(self, props):
        operation_child = SimpleNamespace(
            GetPropNames=lambda: list(props),
            GetPropValue=lambda name: props[name],
        )
        mesh_child = SimpleNamespace(
            GetChildNames=lambda: ["thin_mesh_L_5"],
            GetChildObject=lambda _name: operation_child,
        )
        native_design = SimpleNamespace(
            GetChildObject=lambda name: (
                mesh_child if name == "Mesh" else None
            )
        )
        return thermal._native_thermal_mesh_operation_readback(
            native_design, self._minimal_mesh_plan()
        )

    def test_native_mesh_readback_accepts_live_min_max_incr_schema(self):
        evidence = self._native_mesh_range_readback({
            "Assignment": ["pad_a", "pad_b"],
            "MinLevel": "5",
            "MaxLevel": 5,
            "IncrLevel": "0",
            "Mesh Object(s) Separately Enabled": True,
        })

        operation = evidence["operation_readbacks"][0]
        self.assertEqual(
            operation["level_schema"], "MinLevel/MaxLevel/IncrLevel"
        )
        self.assertEqual(operation["level"], 5)
        self.assertEqual(operation["min_level"], 5)
        self.assertEqual(operation["max_level"], 5)
        self.assertEqual(operation["incr_level"], 0)

    def test_native_mesh_readback_rejects_min_max_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError, "level range readback mismatch"
        ):
            self._native_mesh_range_readback({
                "Assignment": ["pad_a", "pad_b"],
                "MinLevel": "5",
                "MaxLevel": "4",
                "IncrLevel": "0",
                "Mesh Object(s) Separately Enabled": True,
            })

    def test_native_mesh_readback_rejects_incomplete_live_level_schema(self):
        with self.assertRaisesRegex(
            RuntimeError, "lacks one complete level schema"
        ):
            self._native_mesh_range_readback({
                "Assignment": ["pad_a", "pad_b"],
                "MinLevel": "5",
                "IncrLevel": "0",
                "Mesh Object(s) Separately Enabled": True,
            })

    def test_native_mesh_readback_rejects_bad_increment(self):
        for bad_increment in ("not-an-integer", "1", "0.5", True):
            with self.subTest(incr_level=bad_increment):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "integer property readback is invalid|"
                    "increment readback mismatch",
                ):
                    self._native_mesh_range_readback({
                        "Assignment": ["pad_a", "pad_b"],
                        "MinLevel": "5",
                        "MaxLevel": "5",
                        "IncrLevel": bad_increment,
                        "Mesh Object(s) Separately Enabled": True,
                    })

    def test_prior_nine_postsolve_mesh_probe_gate_is_exact_and_finite(self):
        wcp_pads = [
            _Object(f"Tx_main_wcp_pad_{index}_{layer}_{side}")
            for index in (1, 2)
            for side in ("p", "n")
            for layer in ("in", "out")
        ]
        objs = {
            "wcp_pads": wcp_pads,
            "Rx_side2_explicit": [
                _Object("Rx_side2_0_0"),
                _Object("Rx_side2_22_0"),
                _Object("Rx_side2_23_0"),
            ],
        }
        names = thermal._thermal_mesh_postsolve_probe_object_names(objs)
        self.assertEqual(len(names), 9)
        self.assertIn("Rx_side2_22_0", names)
        self.assertNotIn("Rx_side2_23_0", names)
        temperatures = {
            column: 50.0
            for name in names
            for column in (f"T_mean_{name}", f"T_max_{name}")
        }

        complete = thermal._thermal_mesh_postsolve_probe_status(
            names, temperatures
        )
        self.assertEqual(complete, {
            "object_count": 9,
            "missing_count": 0,
            "missing_objects": [],
            "complete": True,
        })

        temperatures.pop("T_max_Tx_main_wcp_pad_1_in_p")
        missing = thermal._thermal_mesh_postsolve_probe_status(
            names, temperatures
        )
        self.assertEqual(missing["missing_count"], 1)
        self.assertEqual(
            missing["missing_objects"],
            ["Tx_main_wcp_pad_1_in_p"],
        )
        self.assertFalse(missing["complete"])

    def test_mesh_idle_barrier_survives_delayed_start_and_stable_grid(self):
        now = [0.0]
        states = iter([False, True, False, False])
        artifact = [{
            "directory": "mesh",
            "name": "DV1_Meshes0_V0.sd",
            "grid_output_size": 10,
            "grid_output_sha256_sample": "a",
            "grid_mapping_size": 5,
            "grid_mapping_sha256_sample": "b",
            "mtime_ns": 1,
        }]
        artifact_reads = iter([[], artifact, artifact])

        def sleep_and_advance(seconds):
            now[0] += seconds

        with patch.object(
            thermal,
            "_attest_standalone_mesh_desktop",
            side_effect=lambda *_args: next(states),
        ), patch.object(
            thermal,
            "_fresh_thermal_mesh_artifacts",
            side_effect=lambda *_args: next(artifact_reads),
        ):
            evidence = thermal._wait_for_standalone_mesh_idle(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
                {},
                require_fresh_artifact=True,
                timeout_s=10.0,
                poll_s=1.0,
                stable_artifact_s=1.0,
                clock=lambda: now[0],
                sleeper=sleep_and_advance,
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["last_running"], False)
        self.assertEqual(evidence["stable_artifact_observations"], 2)
        self.assertTrue(any(
            item["running"] is True
            for item in evidence["running_transitions"]
        ))

    def test_mesh_completion_timeout_keeps_uncertainty_and_blocks_scan(self):
        from module import aedt_pool_adapter

        plan = self._minimal_mesh_plan()
        generator = Mock(return_value=True)
        native_ipk = SimpleNamespace(
            mesh=SimpleNamespace(generate_mesh=generator),
            modeler=SimpleNamespace(oeditor=None),
        )
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "setups": ["ThermalSetup"],
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        sim = SimpleNamespace(solver_may_be_running=False)
        advance = Mock(
            side_effect=AssertionError(
                "unsafe exact message scan before idle"
            )
        )
        with patch.object(
            aedt_pool_adapter, "pooled_backend_enabled",
            return_value=False,
        ), patch.object(
            thermal, "_prepare_thermal_dispatch",
            return_value=preflight,
        ), patch.object(
            thermal, "_native_thermal_mesh_operation_readback",
            return_value={
                "missing_operation_names": [],
                "required_thin_objects_missing": [],
            },
        ), patch.object(
            thermal, "_snapshot_thermal_mesh_artifacts",
            return_value={},
        ), patch.object(
            thermal, "_thermal_desktop_handle",
            return_value=SimpleNamespace(),
        ), patch.object(
            thermal, "capture_scoped_message_cursor",
            return_value=object(),
        ), patch.object(
            thermal, "_attest_standalone_mesh_desktop",
            return_value=False,
        ), patch.object(
            thermal, "_wait_for_standalone_mesh_idle",
            return_value={"passed": False, "last_running": None},
        ), patch.object(
            thermal, "advance_scoped_message_cursor", advance
        ):
            with self.assertRaisesRegex(
                RuntimeError, "completion remains uncertain"
            ):
                thermal._generate_and_attest_thermal_mesh(
                    sim, SimpleNamespace(), _Setup(), plan
                )

        self.assertTrue(sim.solver_may_be_running)
        advance.assert_not_called()

    def test_generate_mesh_unmeshed_error_fails_after_safe_idle(self):
        from module import aedt_pool_adapter

        plan = self._minimal_mesh_plan()
        native_ipk = SimpleNamespace(
            mesh=SimpleNamespace(generate_mesh=Mock(return_value=True)),
            modeler=SimpleNamespace(oeditor=None),
        )
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "setups": ["ThermalSetup"],
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        update = SimpleNamespace(
            new_messages=(
                "'pad_a': Object does not have mesh",
                "Failed to run solver",
            ),
            new_errors=("Simulation completed with execution error",),
            fatal_messages=("Simulation completed with execution error",),
        )
        sim = SimpleNamespace(solver_may_be_running=False)
        with patch.object(
            aedt_pool_adapter, "pooled_backend_enabled",
            return_value=False,
        ), patch.object(
            thermal, "_prepare_thermal_dispatch",
            side_effect=[preflight, preflight],
        ), patch.object(
            thermal, "_native_thermal_mesh_operation_readback",
            return_value={
                "missing_operation_names": [],
                "required_thin_objects_missing": [],
            },
        ), patch.object(
            thermal, "_snapshot_thermal_mesh_artifacts",
            return_value={},
        ), patch.object(
            thermal, "_thermal_desktop_handle",
            return_value=SimpleNamespace(),
        ), patch.object(
            thermal, "capture_scoped_message_cursor",
            return_value=object(),
        ), patch.object(
            thermal, "_attest_standalone_mesh_desktop",
            return_value=False,
        ), patch.object(
            thermal, "_wait_for_standalone_mesh_idle",
            return_value={
                "passed": True,
                "fresh_mesh_artifacts": [{"name": "mesh"}],
            },
        ), patch.object(
            thermal, "advance_scoped_message_cursor",
            return_value=update,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "native thermal mesh preflight failed"
            ):
                thermal._generate_and_attest_thermal_mesh(
                    sim, SimpleNamespace(), _Setup(), plan
                )

        self.assertFalse(sim.solver_may_be_running)

    def test_mesh_stats_export_seals_native_quality_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def export_mesh_stats(setup_name, output_file=None):
                self.assertEqual(setup_name, "ThermalSetup")
                target = Path(output_file)
                target.write_text(
                    "Skewness\nElement Volume\nFace Alignment\n",
                    encoding="utf-8",
                )
                return str(target)

            evidence = thermal._export_thermal_mesh_stats(
                SimpleNamespace(
                    results_directory=str(root),
                    export_mesh_stats=export_mesh_stats,
                ),
                "ThermalSetup",
            )

        self.assertTrue(evidence["exported"])
        self.assertEqual(
            evidence["quality_metrics_reported"],
            ["element_volume", "face_alignment", "skewness"],
        )
        self.assertFalse(evidence["mesh_quality_checks_modified"])
        self.assertFalse(
            evidence["mesh_quality_check_disable_requested"]
        )

    def test_setup_control_readback_fails_on_native_drift(self):
        props = {
            "Flow Regime": "Turbulent",
            "Convergence Criteria - Max Iterations": 250,
            "Convergence Criteria - Flow": "0.001",
            "Convergence Criteria - Energy": "1e-7",
            "Solution Initialization - Use Model Based Flow Initialization": False,
            "Under-relaxation - Pressure": "0.7",
            "Sequential Solve of Flow and Energy Equations": False,
            "Include Gravity": False,
        }
        setup = SimpleNamespace(props=props)
        native = SimpleNamespace(
            GetPropNames=lambda: list(props),
            GetPropValue=lambda name: props[name],
        )
        evidence = thermal._thermal_setup_control_readback(setup, native)
        self.assertTrue(evidence["wrapper_passed"])
        self.assertTrue(evidence["native_complete"])

        drifted = dict(props)
        drifted["Under-relaxation - Pressure"] = "0.6"
        with self.assertRaisesRegex(
            RuntimeError, "native ThermalSetup control drifted"
        ):
            thermal._thermal_setup_control_readback(
                setup,
                SimpleNamespace(
                    GetPropNames=lambda: list(drifted),
                    GetPropValue=lambda name: drifted[name],
                ),
            )

    def test_converged_analyze_rejects_fresh_unmeshed_warning_no_retry(self):
        from module import aedt_pool_adapter

        sim = SimpleNamespace(
            NUM_CORE=4,
            solver_may_be_running=False,
            _attest_cached_native_desktop=lambda desktop, **_kwargs: desktop,
            save_project=Mock(),
            thermal_mesh_preflight={
                "schema": thermal.THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
                "passed": True,
                "mesh_plan_sha256": "b" * 64,
                "mesh_artifact_readback_passed": True,
                "mesh_mapping_coverage_passed": True,
            },
        )
        native_ipk = SimpleNamespace(analyze=Mock(return_value=True))
        preflight = {
            "project": "thermal_test",
            "design": "icepak_thermal",
            "native_ipk": native_ipk,
            "native_design": SimpleNamespace(),
        }
        convergence = self._convergence()
        convergence["_thermal_completion_poll"] = {
            "desktop_attested": True,
            "last_running": False,
            "timed_out": False,
        }
        update = SimpleNamespace(
            new_messages=(
                "'Rx_side2_22_0': Object does not have mesh",
            ),
            new_errors=(),
            fatal_messages=(),
        )
        with patch.object(
            aedt_pool_adapter, "pooled_backend_enabled",
            return_value=False,
        ), patch.object(
            thermal, "_standalone_thermal_completion_timeout",
            return_value=5.0,
        ), patch.object(
            thermal, "_prepare_thermal_dispatch",
            return_value=preflight,
        ), patch.object(
            thermal, "_snapshot_thermal_monitors",
            return_value={},
        ), patch.object(
            thermal, "_thermal_desktop_handle",
            return_value=SimpleNamespace(),
        ), patch.object(
            thermal, "capture_scoped_message_cursor",
            return_value=object(),
        ), patch.object(
            thermal, "advance_scoped_message_cursor",
            return_value=update,
        ), patch.object(
            thermal, "_poll_thermal_dispatch_evidence",
            return_value=(convergence, False, ""),
        ), patch.object(
            thermal, "_bounded_thermal_model_context",
            return_value={},
        ):
            result = thermal._solve_exact_thermal_setup(
                sim, SimpleNamespace(), _Setup()
            )

        self.assertEqual(native_ipk.analyze.call_count, 1)
        self.assertEqual(result["solve_attempts"], 1)
        self.assertFalse(result["analyze_call_ok"])
        self.assertEqual(
            result["convergence"]["thermal_convergence_reason"],
            "native_unmeshed_objects",
        )

    def test_native_residual_parser_accepts_428_and_rejects_429(self):
        stable = (
            "1.5100000000000000e+02 Continuity(7.9911999999999995e-04)"
            "XVelocity(3.6685999999999999e-04)YVelocity(9.8308000000000011e-04)"
            "ZVelocity(3.9534999999999999e-04)Energy(4.3936000000000002e-09)\n"
        )
        divergent = (
            "1.4200000000000000e+02 Continuity(1.0657000000000000e+18)"
            "XVelocity(1.1056000000000001e-01)YVelocity(5.6683999999999998e-02)"
            "ZVelocity(1.4627000000000001e-01)Energy(1.2163000000000000e-01)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            stable_path = Path(tmp, "stable.sd")
            failed_path = Path(tmp, "failed.sd")
            stable_path.write_text(stable, encoding="utf-8")
            failed_path.write_text(divergent, encoding="utf-8")
            self.assertTrue(thermal._parse_thermal_residual_monitor(stable_path)["converged"])
            parsed = thermal._parse_thermal_residual_monitor(failed_path)
            self.assertFalse(parsed["converged"])
            self.assertEqual(parsed["iteration"], 142)
            self.assertEqual(parsed["values"]["Continuity"], 1.0657e18)

            invalid_tail = Path(tmp, "invalid_tail.sd")
            invalid_tail.write_text(
                stable
                + "152 Continuity(nan)XVelocity(1e-4)YVelocity(1e-4)"
                  "ZVelocity(1e-4)Energy(1e-9)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                thermal._parse_thermal_residual_monitor(invalid_tail)

            truncated_tail = Path(tmp, "truncated_tail.sd")
            truncated_tail.write_text(
                stable + "152 Continuity(8e-4)XVelocity(4e-4)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                thermal._parse_thermal_residual_monitor(truncated_tail)

            duplicate_tail = Path(tmp, "duplicate_tail.sd")
            duplicate_tail.write_text(
                stable
                + "152 Continuity(8e-4)Continuity(7e-4)XVelocity(4e-4)"
                  "YVelocity(9e-4)ZVelocity(4e-4)Energy(4e-9)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                thermal._parse_thermal_residual_monitor(duplicate_tail)

    def test_convergence_reader_uses_latest_history_and_ignores_solution_monitor(self):
        stable = (
            "151 Continuity(7e-4)XVelocity(3e-4)YVelocity(9e-4)"
            "ZVelocity(3e-4)Energy(4e-9)\n"
        )
        failed = (
            "142 Continuity(1e18)XVelocity(1e-1)YVelocity(1e-1)"
            "ZVelocity(1e-1)Energy(1e-1)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            old = design / "DV1_S1_MON0_V0.sd"
            newest = design / "DV1_S2_MON0_V0.sd"
            ignored = design / "DV1_SOL3_MON0_V0.sd"
            old.write_text(stable, encoding="utf-8")
            newest.write_text(failed, encoding="utf-8")
            ignored.write_text(stable, encoding="utf-8")
            os.utime(old, (1, 1))
            os.utime(newest, (2, 2))
            os.utime(ignored, (3, 3))
            ipk = SimpleNamespace(
                design_name="icepak_thermal",
                results_directory=str(root),
            )
            setup = SimpleNamespace(props={
                "Convergence Criteria - Flow": "0.001",
                "Convergence Criteria - Energy": "1e-07",
            })
            result = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(result["thermal_convergence_available"], 1)
            self.assertEqual(result["thermal_converged"], 0)
            self.assertEqual(result["thermal_monitor_file"], newest.name)

    def test_convergence_reader_rejects_monitors_older_than_solve_start(self):
        stable = (
            "151 Continuity(7e-4)XVelocity(3e-4)YVelocity(9e-4)"
            "ZVelocity(3e-4)Energy(4e-9)\n"
        )
        setup = SimpleNamespace(props={
            "Convergence Criteria - Flow": "0.001",
            "Convergence Criteria - Energy": "1e-07",
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            monitor = design / "DV1_S1_MON0_V0.sd"
            monitor.write_text(stable, encoding="utf-8")
            old_ns = 1_000_000_000
            solve_start_ns = 2_000_000_000
            os.utime(monitor, ns=(old_ns, old_ns))
            ipk = SimpleNamespace(
                design_name="icepak_thermal", results_directory=str(root),
            )

            stale = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup,
                attempts=1, not_before_ns=solve_start_ns,
            )
            self.assertEqual(stale["thermal_convergence_reason"], "monitor_missing")
            self.assertEqual(stale["thermal_monitor_file"], "")

            fresh_ns = 3_000_000_000
            os.utime(monitor, ns=(fresh_ns, fresh_ns))
            fresh = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup,
                attempts=1, not_before_ns=solve_start_ns,
            )
            self.assertEqual(fresh["thermal_convergence_reason"], "converged")
            self.assertEqual(fresh["thermal_monitor_file"], monitor.name)

    def test_missing_or_malformed_residual_monitor_fails_closed(self):
        setup = SimpleNamespace(props={})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            ipk = SimpleNamespace(design_name="icepak_thermal", results_directory=str(root))
            missing = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(missing["thermal_converged"], 0)
            self.assertEqual(missing["thermal_convergence_reason"], "monitor_missing")
            (design / "DV1_S1_MON0_V0.sd").write_text("not residual data", encoding="utf-8")
            malformed = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(malformed["thermal_converged"], 0)
            self.assertEqual(malformed["thermal_convergence_reason"], "monitor_malformed")

    def test_split_requires_truthy_result_and_live_retained_object(self):
        obj = _Object("core_1")
        modeler = SimpleNamespace(split=Mock(return_value=False), object_names=[obj.name])
        with self.assertRaisesRegex(RuntimeError, "thermal geometry split failed"):
            thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")

        modeler = SimpleNamespace(split=Mock(return_value=[obj.name]), object_names=[])
        with self.assertRaisesRegex(RuntimeError, "retained no live objects"):
            thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")

        modeler = SimpleNamespace(split=Mock(return_value=[obj.name]), object_names=[obj.name])
        alive = thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")
        self.assertEqual(alive, [obj])


if __name__ == "__main__":
    unittest.main()
