import subprocess
import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock


import sys


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import deployment_gate  # noqa: E402
import feeder  # noqa: E402
import rapid_campaign  # noqa: E402


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40


class DeploymentGateTests(unittest.TestCase):
    @mock.patch("deployment_gate.subprocess.check_output")
    def test_revision_must_be_an_exact_advertised_branch_head(self, check_output):
        check_output.return_value = (
            f"{SOLVER_REVISION}\trefs/heads/stabilize/mft-sim-260710\n"
            f"{LIBRARY_REVISION}\trefs/heads/pinned-library\n"
            f"{'c' * 40}\trefs/tags/not-a-head\n"
        )

        accepted = deployment_gate.require_advertised_revision(
            ".", SOLVER_REVISION, "solver"
        )
        self.assertEqual(
            accepted["refs"], ["refs/heads/stabilize/mft-sim-260710"]
        )
        with self.assertRaisesRegex(RuntimeError, "not an advertised origin branch head"):
            deployment_gate.require_advertised_revision(".", "d" * 40, "solver")
        with self.assertRaisesRegex(RuntimeError, "must be a full SHA"):
            deployment_gate.require_advertised_revision(".", "a" * 12, "solver")

    @mock.patch("deployment_gate.subprocess.check_output")
    def test_remote_query_failure_is_fail_closed(self, check_output):
        check_output.side_effect = subprocess.CalledProcessError(
            128, ["git", "ls-remote"], output="origin unavailable"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            deployment_gate.require_advertised_revision(
                ".", SOLVER_REVISION, "solver"
            )

    def test_execute_validates_both_repositories_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    rapid_campaign.deployment_gate, "validate_deployment"
                ) as validate, \
                mock.patch.object(
                    rapid_campaign, "_validate_pinned_local_revisions",
                    return_value=(SOLVER_REVISION, LIBRARY_REVISION),
                ), \
                mock.patch.object(
                    rapid_campaign, "candidate_supply_audit"
                ) as candidate_audit:
            validate.side_effect = RuntimeError("revision is not deployed")
            with self.assertRaisesRegex(RuntimeError, "revision is not deployed"):
                rapid_campaign.run_once(
                    SOLVER_REVISION,
                    LIBRARY_REVISION,
                    library_root=tmp,
                    state_path=Path(tmp, "state.json"),
                    execute=True,
                )
            validate.assert_called_once_with(
                rapid_campaign.REPO_ROOT,
                SOLVER_REVISION,
                tmp,
                LIBRARY_REVISION,
            )
            candidate_audit.assert_not_called()

    def test_direct_feeder_uses_the_same_remote_gate(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"MFT_PYAEDT_LIBRARY_ROOT": tmp}
        ), mock.patch.object(
            feeder.deployment_gate, "validate_deployment", return_value={}
        ) as validate:
            feeder._require_deployed_revisions(
                SOLVER_REVISION, LIBRARY_REVISION
            )
        validate.assert_called_once_with(
            feeder.REPO_ROOT, SOLVER_REVISION, tmp, LIBRARY_REVISION
        )


if __name__ == "__main__":
    unittest.main()
