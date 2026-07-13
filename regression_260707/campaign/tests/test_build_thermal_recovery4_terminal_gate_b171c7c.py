import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _build_thermal_recovery4_terminal_gate_b171c7c as builder  # noqa: E402


class Recovery4TerminalGateBuilderTests(unittest.TestCase):
    def setUp(self):
        self.plan_tasks = []
        self.submitted = []
        self.metadata = []
        self.results = []
        for index, task_id in enumerate(builder.RECOVERY_IDS):
            ordinal = index + 1
            name = f"recovery-{ordinal}"
            dedupe = f"dedupe-{ordinal}"
            source_task_id = (27794, 27928, 27880, 27758)[index]
            self.plan_tasks.append({
                "ordinal": ordinal,
                "name": name,
                "dedupe_key": dedupe,
                "source_task_id": source_task_id,
                "effective_params": {"candidate": ordinal},
            })
            self.submitted.append({
                "ordinal": ordinal,
                "task_id": task_id,
                "source_task_id": source_task_id,
                "name": name,
                "dedupe_key": dedupe,
            })
            self.metadata.append({
                "id": task_id,
                "name": name,
                "dedupe_key": dedupe,
                "project": builder.production.EXPECTED_RESOURCES["project"],
                "status": "completed",
                "cpus": 4,
                "memory_mb": 65536,
                "gpus": 0,
                "timeout_seconds": 14400,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "scheduling_profile": "fea_bursty",
                "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
                "priority": 100,
            })
            monitor = f"monitor-{task_id}.csv"
            forensic = {
                "schema": "thermal-dispatch-forensic-v1",
                "attempts": [{
                    "attempt": 1,
                    "dispatch_status": "success",
                    "native_running": False,
                    "monitor_reason": "converged",
                    "monitor_file": monitor,
                    "identity": {
                        "design": "icepak_thermal",
                        "design_type": "Icepak",
                        "setups": ["ThermalSetup"],
                        "wrapper_setups": ["ThermalSetup"],
                    },
                }],
                "final_convergence": {
                    "available": 1,
                    "converged": 1,
                    "reason": "converged",
                    "monitor_file": monitor,
                },
            }
            self.results.append({
                "candidate": ordinal,
                "thermal_solve_attempts": 1,
                "thermal_dispatch_status": "success",
                "thermal_analyze_call_ok": 1,
                "thermal_solution_data_available": 1,
                "thermal_monitor_file": monitor,
                "thermal_convergence_reason": "converged",
                "thermal_dispatch_forensic_json": json.dumps(forensic),
            })
        self.plan = {
            "profile": {"param_overrides": {"thermal_on": 1}},
            "tasks": self.plan_tasks,
        }
        self.submission = {"tasks": self.submitted}
        self.static_proof = {
            "solver_revision": builder.SOLVER,
            "source_path": builder.THERMAL_SOURCE_PATH,
            "source_sha256": "1" * 64,
            "function": builder.THERMAL_SOLVE_FUNCTION,
            "function_sha256": "2" * 64,
            "entrypoint": "ThermalSetup",
            "native_analyze_call_count": 1,
            "analyze_all_call_count": 0,
            "proof_kind": "exact-git-blob-python-ast",
        }

    def _build(self, output, *, metadata=None, results=None):
        metadata = self.metadata if metadata is None else metadata
        results = self.results if results is None else results
        fetched = [
            builder.scheduler_client.ResultFetch(
                builder.scheduler_client.RESULT_VALID, copy.deepcopy(result),
            )
            for result in results
        ]
        patches = (
            mock.patch.object(builder.production, "_load_recovery",
                              return_value=(self.plan, self.submission)),
            mock.patch.object(builder.production, "_task_detail",
                              side_effect=copy.deepcopy(metadata)),
            mock.patch.object(builder, "_static_dispatch_proof",
                              return_value=self.static_proof),
            mock.patch.object(builder.scheduler_client, "fetch_result",
                              side_effect=fetched),
            mock.patch.object(builder.scheduler_client, "is_valid_result",
                              return_value=True),
            mock.patch.object(builder.scheduler_client, "result_matches_params",
                              return_value=True),
            mock.patch.object(builder.rapid_campaign, "thermal_saturation_columns",
                              return_value=[]),
        )
        entered = [patch.start() for patch in patches]
        try:
            gate = builder.build_gate(output)
        finally:
            for patch in reversed(patches):
                patch.stop()
        return gate, entered

    def test_builds_exact_validate_gate_schema_and_fetches_each_result_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "terminal-gate.json"
            gate, mocks = self._build(output)
            saved = json.loads(output.read_text(encoding="utf-8"))

        load, detail, static, fetch, strict, params, saturation = mocks
        load.assert_called_once_with()
        self.assertEqual(
            [call.args[0] for call in detail.call_args_list],
            list(builder.RECOVERY_IDS),
        )
        self.assertEqual(
            [call.args[0] for call in fetch.call_args_list],
            list(builder.RECOVERY_IDS),
        )
        self.assertTrue(all(call.kwargs["attempts"] == 1
                            for call in fetch.call_args_list))
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(strict.call_count, 4)
        self.assertEqual(params.call_count, 4)
        self.assertEqual(saturation.call_count, 4)
        static.assert_called_once_with()
        self.assertEqual(saved, gate)
        self.assertEqual(gate["strict_valid_count"], 4)
        self.assertEqual(gate["scheduler_mutation_count"], 0)
        self.assertEqual(gate["tasks"][0]["thermal_dispatch"][
            "analyze_all_call_count"], 0)
        self.assertTrue(gate["tasks"][3]["known_good_nonregression"])
        unsigned = dict(gate)
        seal = unsigned.pop("gate_sha256")
        self.assertEqual(seal, builder.production._sha(unsigned))
        builder.production._validate_gate(gate, seal, self.submission)

    def test_noncompleted_task_aborts_before_static_proof_or_any_result_fetch(self):
        metadata = copy.deepcopy(self.metadata)
        metadata[2]["status"] = "running"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "terminal-gate.json"
            with mock.patch.object(builder.production, "_load_recovery",
                                   return_value=(self.plan, self.submission)), \
                    mock.patch.object(builder.production, "_task_detail",
                                      side_effect=metadata) as detail, \
                    mock.patch.object(builder, "_static_dispatch_proof") as static, \
                    mock.patch.object(builder.scheduler_client,
                                      "fetch_result") as fetch:
                with self.assertRaisesRegex(RuntimeError, "all exact tasks completed"):
                    builder.build_gate(output)
            self.assertFalse(output.exists())
        self.assertEqual(detail.call_count, 4)
        static.assert_not_called()
        fetch.assert_not_called()

    def test_metadata_identity_drift_aborts_before_result_fetch(self):
        metadata = copy.deepcopy(self.metadata)
        metadata[0]["dedupe_key"] = "wrong"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "terminal-gate.json"
            with mock.patch.object(builder.production, "_load_recovery",
                                   return_value=(self.plan, self.submission)), \
                    mock.patch.object(builder.production, "_task_detail",
                                      side_effect=metadata), \
                    mock.patch.object(builder.scheduler_client,
                                      "fetch_result") as fetch:
                with self.assertRaisesRegex(RuntimeError, "metadata drifted"):
                    builder.build_gate(output)
            self.assertFalse(output.exists())
        fetch.assert_not_called()

    def test_existing_output_fails_before_any_external_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "terminal-gate.json"
            output.write_text("owned", encoding="utf-8")
            with mock.patch.object(builder.production, "_load_recovery") as load, \
                    mock.patch.object(builder.production, "_task_detail") as detail, \
                    mock.patch.object(builder.scheduler_client,
                                      "fetch_result") as fetch:
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    builder.build_gate(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "owned")
        load.assert_not_called()
        detail.assert_not_called()
        fetch.assert_not_called()

    def test_atomic_writer_uses_windows_no_replace_rename_when_links_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "terminal-gate.json"
            with mock.patch.object(builder.os, "link",
                                   side_effect=OSError("unsupported")), \
                    mock.patch.object(builder.os, "name", "nt"):
                builder._atomic_create_json(output, {"sealed": True})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"sealed": True},
            )

    def test_strict_result_rejects_invalid_params_and_saturation(self):
        fetched = builder.scheduler_client.ResultFetch(
            builder.scheduler_client.RESULT_VALID, self.results[0],
        )
        with mock.patch.object(builder.scheduler_client, "is_valid_result",
                               return_value=True), \
                mock.patch.object(builder.scheduler_client, "result_matches_params",
                                  return_value=False), \
                mock.patch.object(builder.rapid_campaign,
                                  "thermal_saturation_columns", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "effective params drifted"):
                builder._strict_result(
                    fetched, self.plan_tasks[0], builder.RECOVERY_IDS[0],
                    {"thermal_on": 1},
                )
        with mock.patch.object(builder.scheduler_client, "is_valid_result",
                               return_value=True), \
                mock.patch.object(builder.scheduler_client, "result_matches_params",
                                  return_value=True), \
                mock.patch.object(builder.rapid_campaign,
                                  "thermal_saturation_columns",
                                  return_value=["T_max_core"]):
            with self.assertRaisesRegex(RuntimeError, "thermal saturation"):
                builder._strict_result(
                    fetched, self.plan_tasks[0], builder.RECOVERY_IDS[0],
                    {"thermal_on": 1},
                )

    def test_failed_source_forensic_fail_closed_mutations(self):
        mutations = (
            ("schema", lambda value: value.update(schema="wrong")),
            ("three attempts", lambda value: value["attempts"].extend(
                copy.deepcopy(value["attempts"]) * 2)),
            ("setup identity", lambda value: value["attempts"][0]["identity"].update(
                setups=["AnalyzeAll"])),
            ("final failure", lambda value: value["attempts"][-1].update(
                dispatch_status="false")),
            ("stale monitor", lambda value: value["final_convergence"].update(
                monitor_file="")),
            ("not converged", lambda value: value["final_convergence"].update(
                converged=0)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                result = copy.deepcopy(self.results[0])
                forensic = json.loads(result["thermal_dispatch_forensic_json"])
                mutate(forensic)
                result["thermal_dispatch_forensic_json"] = json.dumps(forensic)
                with self.assertRaises(RuntimeError):
                    builder._failed_source_dispatch(
                        result, self.static_proof, builder.RECOVERY_IDS[0],
                    )

    def test_result_analyze_all_count_must_be_zero(self):
        result = copy.deepcopy(self.results[0])
        result["thermal_analyze_all_call_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "not integer zero"):
            builder._failed_source_dispatch(
                result, self.static_proof, builder.RECOVERY_IDS[0],
            )

    def test_static_proof_requires_exact_setup_call_and_no_analyze_all(self):
        good = '''
_THERMAL_SETUP_NAME = "ThermalSetup"
def _solve_exact_thermal_setup(sim, ipk, setup,
                               setup_name=_THERMAL_SETUP_NAME):
    native_ipk = ipk
    native_ipk.analyze(setup=setup_name, cores=sim.NUM_CORE, blocking=True)
'''
        proof = builder._static_dispatch_proof(good)
        self.assertEqual(proof["entrypoint"], "ThermalSetup")
        self.assertEqual(proof["analyze_all_call_count"], 0)
        bad = good + "\ndef hidden(ipk):\n    ipk.AnalyzeAll()\n"
        with self.assertRaisesRegex(RuntimeError, "zero AnalyzeAll"):
            builder._static_dispatch_proof(bad)
        wrong_setup = good.replace("setup=setup_name", "setup='OtherSetup'")
        with self.assertRaisesRegex(RuntimeError, "keyword dispatch drifted"):
            builder._static_dispatch_proof(wrong_setup)


if __name__ == "__main__":
    unittest.main()
