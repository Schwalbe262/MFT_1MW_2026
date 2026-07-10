import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import feeder  # noqa: E402
import local_gate  # noqa: E402
import pinned_pilot  # noqa: E402

SOLVER = "a" * 40
LIBRARY = "b" * 40


def gate_results():
    return [
        {
            "project_name": f"simulation-{index}",
            "matrix_extraction_backend": "export_rl_matrix",
            "matrix_solve_attempts": 1,
            "loss_solve_attempts": 1,
        }
        for index in range(3)
    ]


class FakeStreamingProcess:
    def __init__(self, lines):
        self.pid = 12345
        self._lines = iter(lines)
        self.stdout = self
        self.consumed = 0
        self.wait_called = False

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._lines)
        self.consumed += 1
        return line

    def poll(self):
        return None

    def wait(self):
        self.wait_called = True
        return 0


class LocalGateManifestTests(unittest.TestCase):
    def test_campaign_and_al_entrypoints_import_from_their_own_cwds(self):
        regression_root = CAMPAIGN_DIR.parent
        commands = [
            ([sys.executable, "local_gate.py", "--help"], CAMPAIGN_DIR),
            ([sys.executable, "al_driver.py", "--help"], regression_root),
        ]
        for command, cwd in commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    command, cwd=cwd, capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_valid_three_result_manifest_opens_p02_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            tag = pinned_pilot.local_gate_tag(SOLVER, LIBRARY)
            path = Path(directory) / f"{tag}.json"
            path.write_text(json.dumps({
                "tag": tag,
                "solver_revision": SOLVER,
                "library_revision": LIBRARY,
                "sample_count": 3,
                "passed": True,
                "results": gate_results(),
            }), encoding="utf-8")
            with mock.patch.object(
                    pinned_pilot.scheduler_client, "is_valid_result",
                    return_value=True) as validate:
                result = pinned_pilot.validate_local_gate(
                    SOLVER, LIBRARY, manifest_dir=directory)

        self.assertEqual(result["manifest"], str(path))
        self.assertEqual(validate.call_count, 3)
        for call in validate.call_args_list:
            self.assertEqual(call.kwargs["expected_revision"], SOLVER)
            self.assertEqual(call.kwargs["expected_library_revision"], LIBRARY)
            self.assertEqual(call.kwargs["expected_profile"]["keep_project"], 1)

    def test_local_gate_rejects_duplicate_projects_and_wrong_backend(self):
        results = gate_results()
        results[1]["project_name"] = results[0]["project_name"]
        with mock.patch.object(
                local_gate.scheduler_client, "is_valid_result", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "duplicate project"):
                local_gate._validate_results(results, SOLVER, LIBRARY)

    def test_existing_valid_manifest_is_reused_without_starting_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local3.json"
            path.write_text(json.dumps({"passed": True}), encoding="utf-8")
            with mock.patch.object(
                    local_gate.al_driver, "_current_solver_revision", return_value=SOLVER), \
                    mock.patch.object(
                        local_gate.al_driver, "_current_library_revision", return_value=LIBRARY), \
                    mock.patch.object(local_gate, "_manifest_path", return_value=path), \
                    mock.patch.object(
                        local_gate.pinned_pilot, "validate_local_gate",
                        return_value={"manifest": str(path), "tag": "local3", "projects": []}), \
                    mock.patch.object(local_gate.subprocess, "Popen") as popen, \
                    mock.patch("builtins.print"):
                manifest = local_gate.run_gate()

        self.assertEqual(manifest, {"passed": True})
        popen.assert_not_called()

        results = gate_results()
        results[0]["matrix_extraction_backend"] = "get_solution_data"
        with mock.patch.object(
                local_gate.scheduler_client, "is_valid_result", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "export_rl_matrix"):
                local_gate._validate_results(results, SOLVER, LIBRARY)

    def test_local_command_is_exact_and_consecutive(self):
        joined = " ".join(local_gate.EXACT_ARGS)
        self.assertIn("--count 3", joined)
        self.assertIn("--require-consecutive", joined)
        assignments = {}
        for index, token in enumerate(local_gate.EXACT_ARGS[:-1]):
            if token == "--set":
                key, value = local_gate.EXACT_ARGS[index + 1].split("=", 1)
                assignments[key] = value
        expected = dict(pinned_pilot.scheduler_client.STANDARD_PROFILE_CONTRACT)
        expected["keep_project"] = 1
        self.assertEqual(set(assignments), set(expected))
        for key, value in expected.items():
            if isinstance(value, str):
                self.assertEqual(assignments[key], value)
            else:
                self.assertEqual(float(assignments[key]), float(value))

        standard_path = (
            CAMPAIGN_DIR.parent / "verify" / "profiles" / "standard.json")
        standard = json.loads(standard_path.read_text(encoding="utf-8"))
        self.assertEqual(
            standard["param_overrides"],
            pinned_pilot.scheduler_client.STANDARD_PROFILE_CONTRACT)

    def test_gate_cleanup_terminates_captured_python_process_tree(self):
        options = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(
            [sys.executable, "-c", (
                "import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                "print(p.pid,flush=True); time.sleep(60)"
            )],
            stdout=subprocess.PIPE, text=True, **options,
        )
        try:
            child_pid = int(process.stdout.readline().strip())
            captured = {}
            local_gate._capture_process_tree(process.pid, captured)
            self.assertIn(child_pid, captured)
            local_gate._terminate_captured_processes(captured, wait_seconds=2)
            process.wait(timeout=2)
            import psutil
            self.assertFalse(psutil.pid_exists(child_pid))
        finally:
            if process.poll() is None:
                process.kill()
            if process.stdout is not None:
                process.stdout.close()

    def test_malformed_result_fails_before_later_output_is_consumed(self):
        cases = [
            ("RESULT_JSON {bad\n", "malformed RESULT_JSON"),
            ("RESULT_JSON []\n", "non-object RESULT_JSON"),
        ]
        for first_line, expected_error in cases:
            with self.subTest(first_line=first_line):
                process = FakeStreamingProcess([
                    first_line,
                    'RESULT_JSON {"project_name":"must-not-be-read"}\n',
                ])
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with mock.patch.object(
                            local_gate.al_driver, "_current_solver_revision",
                            return_value=SOLVER), mock.patch.object(
                            local_gate.al_driver, "_current_library_revision",
                            return_value=LIBRARY), mock.patch.object(
                            local_gate, "_manifest_path",
                            return_value=root / "local3.json"), mock.patch.object(
                            local_gate, "REGRESSION_ROOT", root), mock.patch.object(
                            local_gate, "REPO_ROOT", root), mock.patch.object(
                            local_gate.subprocess, "Popen", return_value=process), \
                            mock.patch.object(
                                local_gate, "_capture_process_tree"), \
                            mock.patch.object(
                                local_gate, "_terminate_captured_processes") as cleanup, \
                            mock.patch("builtins.print"):
                        with self.assertRaisesRegex(RuntimeError, expected_error):
                            local_gate.run_gate(force=True)

                self.assertEqual(process.consumed, 1)
                self.assertFalse(process.wait_called)
                cleanup.assert_called_once()


class FleetGateTests(unittest.TestCase):
    def test_nonzero_feeder_requires_completed_p08_gate(self):
        argv = [
            "feeder.py", "--once", "--target", "1", "--buffer", "0",
            "--solver-revision", SOLVER, "--library-revision", LIBRARY,
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                feeder.al_driver, "_current_solver_revision", return_value=SOLVER), \
                mock.patch.object(
                    feeder.al_driver, "_current_library_revision", return_value=LIBRARY), \
                mock.patch.object(feeder, "validate_p08_completion") as validate, \
                mock.patch.object(feeder, "step", return_value=False) as step:
            feeder.main()

        validate.assert_called_once_with(SOLVER, LIBRARY, seed=260710)
        step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
