import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _continuous_refill_b171c7c as controller  # noqa: E402


class MatureRefillPolicyTests(unittest.TestCase):
    def test_sample_health_findings_are_alerts_and_refill_continues(self):
        outcomes = []
        for task_id in range(1, 31):
            outcomes.append({
                "task_id": task_id,
                "status": "failed",
                "state": "invalid",
                "reason": "task_failed",
                "saturation_columns": [],
                "error_fingerprint": None,
                "error_message": "Icepak residual did not converge",
            })
        outcomes[0].update({
            "status": "completed",
            "reason": "solver_revision_mismatch",
            "saturation_columns": ["Tprobe_core_max"],
        })
        health = {"terminal": 30, "valid": 0, "valid_rate": 0.0}

        alerts, reasons, waits = controller._mature_production_policy(
            {"outcomes": outcomes}, outcomes, health, {"pinned": False})

        self.assertEqual(reasons, [])
        self.assertEqual(waits, [])
        self.assertIn("thermal_saturation_detected:1", alerts)
        self.assertIn("revision_mismatch_detected:1", alerts)
        self.assertIn("completed_strict_invalid:1", alerts)
        self.assertIn("recent_valid_rate_below_70pct:0.000", alerts)
        self.assertIn("fleet_valid_rate_below_90pct:0.000", alerts)
        self.assertIn("strict_collector_not_pinned_to_b171", alerts)
        self.assertEqual(
            controller._maintained_pool_action(
                target_active=300,
                logical_active=299,
                target_reached=False,
                reasons=reasons,
                wait_reasons=waits,
            ),
            "refill_300",
        )

    def test_malformed_health_evidence_still_fails_closed(self):
        malformed = [{"task_id": 1}]
        with self.assertRaises(KeyError):
            controller._mature_production_policy(
                {"outcomes": malformed},
                malformed,
                {"terminal": 1, "valid": 0, "valid_rate": 0.0},
                {"pinned": True},
            )

    def test_local_recovery_evidence_tamper_is_rejected(self):
        payload = json.loads(
            controller.LOCAL_RECOVERY_EVIDENCE_PATH.read_text(
                encoding="utf-8"))
        payload["reviewed_contract"]["thermal_iterations"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-local-recovery.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                    controller, "LOCAL_RECOVERY_EVIDENCE_PATH", path):
                with self.assertRaisesRegex(
                        RuntimeError, "contract drifted"):
                    controller._local_recovery_evidence()

    def test_rejected_submission_evidence_tamper_is_rejected(self):
        payload = json.loads(
            controller.REJECTED_CANCELLATION_EVIDENCE_PATH.read_text(
                encoding="utf-8"))
        payload["reviewed_contract"]["task_id"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-cancellation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                    controller, "REJECTED_CANCELLATION_EVIDENCE_PATH", path):
                with self.assertRaisesRegex(
                        RuntimeError, "evidence drifted"):
                    controller._rejected_submission_seal()


if __name__ == "__main__":
    unittest.main()
