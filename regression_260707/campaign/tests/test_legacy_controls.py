import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
RELAUNCH = CAMPAIGN_DIR / "relaunch.sh"
sys.path.insert(0, str(CAMPAIGN_DIR))

import submit_wave  # noqa: E402


class LegacySubmissionTests(unittest.TestCase):
    def test_main_rejects_every_argument_shape_without_post(self):
        argument_sets = (
            ["submit_wave.py"],
            ["submit_wave.py", "--help"],
            ["submit_wave.py", "--wave", "1", "--tasks", "400"],
            ["submit_wave.py", "--wave", "0", "--pilot"],
        )
        with mock.patch.object(submit_wave.requests, "post") as post:
            for argv in argument_sets:
                with self.subTest(argv=argv), mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit):
                        submit_wave.main()
            post.assert_not_called()

    def test_submit_helper_is_also_fail_closed(self):
        with mock.patch.object(submit_wave.requests, "post") as post:
            with self.assertRaises(RuntimeError):
                submit_wave.submit("name", "workdir", "--headless")
            post.assert_not_called()


class RelaunchPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        cls.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")

    def run_relaunch(self, *args):
        if not self.bash:
            self.skipTest("bash is unavailable")
        return subprocess.run(
            [self.bash, RELAUNCH.name, *args],
            cwd=CAMPAIGN_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_shell_syntax(self):
        if not self.bash:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [self.bash, "-n", RELAUNCH.name],
            cwd=CAMPAIGN_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_inputs_fail_before_process_stop(self):
        cases = (
            (("bad",), "target must be"),
            (("1", "bad"), "buffer must be"),
            (("1", "0"), "full solver and library revisions"),
            (("0", "0"), "full solver and library revisions"),
            (("1", "0", "abc", "def"), "solver revision must be"),
            (("1", "0", "a" * 40), "full solver and library revisions"),
            (("1", "0", "a" * 40, "b" * 40, "extra"), "Usage:"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.run_relaunch(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("campaign loops stopped", result.stdout)

    def test_all_preflight_validation_precedes_process_stop(self):
        text = RELAUNCH.read_text(encoding="utf-8")
        stop_position = text.index("manage_campaign_loops.ps1")
        for marker in (
            "target must be a non-negative integer",
            "solver revision must be a full 40-character SHA",
            "al_driver._current_solver_revision()",
            "al_driver._current_library_revision()",
            "Scheduler health check",
        ):
            self.assertLess(text.index(marker), stop_position, marker)


if __name__ == "__main__":
    unittest.main()
