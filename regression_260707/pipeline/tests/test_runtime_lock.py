import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from regression_260707.pipeline.__main__ import (
    SINGLETON_EXIT_CODE,
    _run_role_locked,
)
from regression_260707.pipeline.runtime_lock import (
    AlreadyRunningError,
    RoleInstanceLock,
)


class RoleInstanceLockTests(unittest.TestCase):
    def test_same_role_is_exclusive_but_roles_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            with RoleInstanceLock(
                directory,
                "controller",
                {"solver_revision": "a" * 40},
            ) as first:
                owner = json.loads(first.path.read_text(encoding="utf-8"))
                self.assertEqual(owner["role"], "controller")
                self.assertEqual(owner["solver_revision"], "a" * 40)
                with self.assertRaises(AlreadyRunningError) as raised:
                    RoleInstanceLock(directory, "controller").acquire()
                self.assertEqual(raised.exception.path, first.path)
                self.assertEqual(
                    json.loads(raised.exception.owner)["pid"], owner["pid"]
                )
                with RoleInstanceLock(directory, "supervisor"):
                    pass

            with RoleInstanceLock(directory, "controller"):
                pass

    def test_stale_file_is_reused_instead_of_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locks" / "supervisor.lock"
            path.parent.mkdir(parents=True)
            path.write_text("stale owner metadata\n", encoding="utf-8")
            with RoleInstanceLock(directory, "supervisor") as acquired:
                self.assertEqual(acquired.path, path)
            owner = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(owner["role"], "supervisor")
            self.assertTrue(path.is_file())

    def test_unknown_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported singleton role"):
                RoleInstanceLock(directory, "worker")

    def test_cli_wrapper_fails_with_stable_exit_code_without_running_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = mock.Mock()
            with RoleInstanceLock(directory, "controller"):
                with self.assertRaises(SystemExit) as raised, mock.patch(
                    "sys.stderr", new_callable=io.StringIO
                ) as stderr:
                    _run_role_locked(
                        Path(directory), "controller", {}, callback
                    )
            self.assertEqual(raised.exception.code, SINGLETON_EXIT_CODE)
            callback.assert_not_called()
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["role"], "controller")
            self.assertEqual(payload["exit_code"], SINGLETON_EXIT_CODE)


if __name__ == "__main__":
    unittest.main()
