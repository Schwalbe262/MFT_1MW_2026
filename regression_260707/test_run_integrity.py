import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from run_simulation_260706 import (
    Simulation,
    _finalize_run_cleanup,
    _git_provenance,
    _project_delete_policy,
    main,
)
from module.source_contract import SOLVER_REVISION_PATHS


class ResultPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.previous_cwd = os.getcwd()
        self.temp_directory = tempfile.TemporaryDirectory()
        os.chdir(self.temp_directory.name)

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.temp_directory.cleanup()

    @staticmethod
    def _simulation():
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        return simulation

    @staticmethod
    def _fake_parquet(frame, path, index=False):
        del frame, index
        Path(path).write_text("complete", encoding="utf-8")

    def test_parquet_is_atomic_and_schema_mismatch_preserves_csv(self):
        events = []
        csv_path = Path("results.csv")
        original_csv = "legacy_column\n123\n"
        csv_path.write_text(original_csv, encoding="utf-8")
        real_replace = os.replace

        def write_parquet(frame, path, index=False):
            events.append("parquet_write")
            self._fake_parquet(frame, path, index=index)

        def atomic_replace(source, destination):
            events.append("parquet_replace")
            real_replace(source, destination)

        class RecordingLock:
            def __init__(self, _path):
                pass

            def __enter__(self):
                events.append("csv_lock")

            def __exit__(self, *_args):
                return False

        with (
            patch.object(pd.DataFrame, "to_parquet", write_parquet),
            patch("run_simulation_260706.os.replace", atomic_replace),
            patch("run_simulation_260706.FileLock", RecordingLock),
        ):
            self._simulation().save_results_to_csv(
                pd.DataFrame([{"new_column": 7}]), str(csv_path)
            )

        self.assertEqual(events, ["parquet_write", "parquet_replace", "csv_lock"])
        self.assertEqual(csv_path.read_text(encoding="utf-8"), original_csv)
        self.assertEqual(len(list(Path("results_parts_260706").glob("*.parquet"))), 1)
        self.assertEqual(list(Path("results_parts_260706").glob("*.tmp-*")), [])
        self.assertEqual(list(Path(".").glob("results_old_*.csv")), [])

    def test_matching_csv_schema_appends_without_overwriting_parts(self):
        simulation = self._simulation()
        frame = pd.DataFrame([{"value": 7}])

        with patch.object(pd.DataFrame, "to_parquet", self._fake_parquet):
            simulation.save_results_to_csv(frame, "results.csv")
            simulation.save_results_to_csv(frame, "results.csv")

        saved = pd.read_csv("results.csv")
        self.assertEqual(saved["value"].tolist(), [7, 7])
        self.assertEqual(len(list(Path("results_parts_260706").glob("*.parquet"))), 2)

    def test_failed_parquet_write_removes_temp_and_keeps_csv_fallback(self):
        def fail_after_partial_write(_frame, path, index=False):
            del index
            Path(path).write_text("partial", encoding="utf-8")
            raise OSError("disk write failed")

        with patch.object(pd.DataFrame, "to_parquet", fail_after_partial_write):
            self._simulation().save_results_to_csv(
                pd.DataFrame([{"value": 9}]), "results.csv"
            )

        self.assertEqual(pd.read_csv("results.csv")["value"].tolist(), [9])
        self.assertEqual(list(Path("results_parts_260706").glob("*.parquet")), [])
        self.assertEqual(list(Path("results_parts_260706").glob("*.tmp-*")), [])

    def test_parquet_failure_and_csv_mismatch_use_jsonl_fallback(self):
        Path("results.csv").write_text("legacy\n1\n", encoding="utf-8")

        with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("no parquet")):
            self._simulation().save_results_to_csv(
                pd.DataFrame([{"value": 11}]), "results.csv"
            )

        fallback = Path("results_fallback_260706.jsonl")
        self.assertTrue(fallback.is_file())
        self.assertIn('"value":11', fallback.read_text(encoding="utf-8"))
        self.assertEqual(Path("results.csv").read_text(encoding="utf-8"), "legacy\n1\n")


class FinalCleanupTests(unittest.TestCase):
    def test_project_delete_policy_is_available_before_validation(self):
        keep = pd.DataFrame({"keep_project": [1]})
        discard = pd.DataFrame({"keep_project": [0]})
        self.assertFalse(_project_delete_policy(keep, fixed_mode=False))
        self.assertTrue(_project_delete_policy(discard, fixed_mode=True))
        self.assertFalse(_project_delete_policy(discard, fixed_mode=True, hold=True))
        self.assertFalse(_project_delete_policy(discard, fixed_mode=True, model_only=True))

    def test_disposable_project_is_deleted_after_descendant_termination(self):
        events = []
        simulation = SimpleNamespace(
            delete_project_folder=lambda **kwargs: events.append(("delete", kwargs))
        )

        with patch(
            "run_simulation_260706._terminate_spawned_descendants",
            side_effect=lambda *_args: events.append(("terminate", {})),
        ):
            _finalize_run_cleanup(
                {}, {123: (1, 1.0)}, sim=simulation, delete_project=True
            )

        self.assertEqual(events[0], ("terminate", {}))
        self.assertEqual(events[1], ("delete", {"max_attempts": 3, "wait_s": 1}))

    def test_hold_preserves_process_and_project(self):
        simulation = SimpleNamespace(
            delete_project_folder=lambda **_kwargs: self.fail("project was deleted")
        )
        with patch("run_simulation_260706._terminate_spawned_descendants") as terminate:
            _finalize_run_cleanup(
                {}, {}, sim=simulation, held=True, delete_project=True
            )
        terminate.assert_not_called()

    def test_keep_project_still_terminates_descendants_without_deleting(self):
        simulation = SimpleNamespace(
            delete_project_folder=lambda **_kwargs: self.fail("project was deleted")
        )
        with patch("run_simulation_260706._terminate_spawned_descendants") as terminate:
            _finalize_run_cleanup(
                {}, {}, sim=simulation, held=False, delete_project=False
            )
        terminate.assert_called_once_with({}, {})


class FixedCliCompletionTests(unittest.TestCase):
    @staticmethod
    def _args(model_only=False, fixed=True, hold=False, count=None,
              require_consecutive=False):
        return SimpleNamespace(
            headless=False,
            golden=False,
            fixed=fixed,
            params=None,
            model_only=model_only,
            round_corner=None,
            full=False,
            matrix_on=None,
            loss_on=None,
            thermal_on=None,
            set_overrides=[],
            hold=hold,
            count=count,
            require_consecutive=require_consecutive,
        )

    def test_invalid_fixed_result_exits_nonzero(self):
        with (
            patch("run_simulation_260706.parse_args", return_value=self._args()),
            patch("run_simulation_260706.run_one_loop", return_value=False),
            patch("run_simulation_260706.os._exit") as exit_process,
        ):
            main()

        exit_process.assert_called_once_with(1)
    def test_model_only_fixed_run_can_exit_zero_without_result_row(self):
        with (
            patch(
                "run_simulation_260706.parse_args",
                return_value=self._args(model_only=True),
            ),
            patch("run_simulation_260706.run_one_loop", return_value=None),
            patch("run_simulation_260706.os._exit") as exit_process,
        ):
            main()

        exit_process.assert_called_once_with(0)

    def test_random_model_only_forwards_hold_for_gui_inspection(self):
        args = self._args(model_only=True, fixed=False, hold=True)
        with (
            patch("run_simulation_260706.parse_args", return_value=args),
            patch("run_simulation_260706.run_one_loop", return_value=True) as run,
        ):
            main()

        run.assert_called_once_with(param=None, model_only=True, hold=True)

    def test_consecutive_random_gate_aborts_on_first_failed_sample(self):
        args = self._args(
            fixed=False, count=3, require_consecutive=True)
        with (
            patch("run_simulation_260706.parse_args", return_value=args),
            patch("run_simulation_260706.run_one_loop", return_value=False) as run,
            patch("run_simulation_260706.time.sleep"),
            patch(
                "run_simulation_260706.os._exit", side_effect=SystemExit
            ) as exit_process,
        ):
            with self.assertRaises(SystemExit):
                main()

        run.assert_called_once()
        exit_process.assert_called_once_with(1)


class SourceProvenanceTests(unittest.TestCase):
    def test_dirty_check_is_limited_to_shared_solver_contract_paths(self):
        with patch(
                "subprocess.check_output",
                side_effect=["a" * 40 + "\n", ""]) as check_output:
            revision, dirty = _git_provenance()

        self.assertEqual((revision, dirty), ("a" * 40, 0))
        status_command = check_output.call_args_list[1].args[0]
        self.assertIn("--untracked-files=all", status_command)
        self.assertEqual(
            tuple(status_command[status_command.index("--") + 1:]),
            SOLVER_REVISION_PATHS)


if __name__ == "__main__":
    unittest.main()
