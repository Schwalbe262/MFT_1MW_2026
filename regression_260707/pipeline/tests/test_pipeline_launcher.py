import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
MFT_RUNTIME_ROOT = PIPELINE_ROOT.parent
LAUNCHER = PIPELINE_ROOT / "start_pipeline_role.ps1"


class PipelineLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.powershell = shutil.which("powershell") or shutil.which("powershell.exe")

    def _invoke(self, *arguments):
        if not self.powershell:
            self.skipTest("Windows PowerShell is unavailable")
        return subprocess.run(
            [
                self.powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                *map(str, arguments),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @staticmethod
    def _write_reviewed_config(root: Path) -> tuple[Path, str]:
        library = root / "library"
        library.mkdir()
        config = root / "reviewed-verification.json"
        config.write_text(
            json.dumps({
                "standard": {
                    "adapter": "mft_scheduler_v1",
                    "execute": True,
                    "library_root": str(library),
                },
                "fine": {
                    "adapter": "mft_scheduler_v1",
                    "execute": True,
                    "library_root": str(library),
                },
            }),
            encoding="utf-8",
        )
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        return config, digest

    def test_validate_only_seals_config_and_builds_external_runtime_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, digest = self._write_reviewed_config(root)
            state = root / "state"
            result = self._invoke(
                "-Role", "Controller",
                "-SolverRevision", "A" * 40,
                "-LibraryRevision", "B" * 40,
                "-VerificationConfig", config,
                "-ReviewedVerificationConfigSha256", digest.upper(),
                "-PipelineRuntimeRoot", state,
                "-MftRuntimeRoot", MFT_RUNTIME_ROOT,
                "-Python", sys.executable,
                "-ValidateOnly",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            contract = json.loads(result.stdout.lstrip("\ufeff"))
            self.assertEqual(contract["solver_revision"], "a" * 40)
            self.assertEqual(contract["library_revision"], "b" * 40)
            self.assertEqual(contract["pipeline_runtime_root"], str(state.resolve()))
            self.assertEqual(
                contract["singleton_lock"],
                str(state.resolve() / "locks" / "controller.lock"),
            )
            sealed = Path(contract["verification_config"])
            self.assertTrue(sealed.is_file())
            self.assertEqual(hashlib.sha256(sealed.read_bytes()).hexdigest(), digest)
            self.assertIn("--pipeline-root", contract["arguments"])
            self.assertIn(str(state.resolve()), contract["arguments"])
            self.assertFalse((state / "jobs.sqlite3").exists())

    def test_review_hash_mismatch_fails_before_any_python_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._write_reviewed_config(root)
            result = self._invoke(
                "-Role", "Supervisor",
                "-SolverRevision", "a" * 40,
                "-LibraryRevision", "b" * 40,
                "-VerificationConfig", config,
                "-ReviewedVerificationConfigSha256", "c" * 64,
                "-PipelineRuntimeRoot", root / "state",
                "-MftRuntimeRoot", MFT_RUNTIME_ROOT,
                "-Python", sys.executable,
                "-ValidateOnly",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "verification config does not match the reviewed SHA256",
                result.stderr,
            )
            self.assertFalse((root / "state" / "jobs.sqlite3").exists())

    def test_source_tree_cannot_be_used_as_pipeline_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, digest = self._write_reviewed_config(root)
            result = self._invoke(
                "-Role", "Supervisor",
                "-SolverRevision", "a" * 40,
                "-LibraryRevision", "b" * 40,
                "-VerificationConfig", config,
                "-ReviewedVerificationConfigSha256", digest,
                "-PipelineRuntimeRoot", MFT_RUNTIME_ROOT / "pipeline_runtime",
                "-MftRuntimeRoot", MFT_RUNTIME_ROOT,
                "-Python", sys.executable,
                "-ValidateOnly",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be outside the MFT source/runtime tree", result.stderr)


if __name__ == "__main__":
    unittest.main()
