from __future__ import annotations

import json
from pathlib import Path
import re
import tarfile
import unittest


ROOT = Path(__file__).resolve().parent / "artifacts"


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class IsolationArtifactTests(unittest.TestCase):
    def test_primary_process_and_license_overlap_was_real(self):
        verdict = read_json("primary_verdict_732549.json")
        checks = verdict["checks"]
        self.assertTrue(checks["same_desktop"])
        self.assertTrue(checks["two_projects"])
        self.assertTrue(checks["desktop_checkout_exactly_one_during_overlap"])
        self.assertTrue(checks["solver_checkouts_at_least_two_during_overlap"])
        self.assertTrue(checks["two_solver_processes_during_overlap"])
        self.assertTrue(checks["A_solver_left_after_sigterm"])
        self.assertTrue(checks["B_solver_survived_after_sigterm"])
        self.assertFalse(checks["B_convergence_exported"])

        processes = read_json("process_AB_overlap_732549.json")
        by_pid = {item["pid"]: item for item in processes}
        self.assertTrue(by_pid[1279633]["desktop_descendant"])
        self.assertTrue(by_pid[1279899]["desktop_descendant"])
        self.assertEqual(by_pid[1279633]["comm"], "MAXWELLCOMENGIN")
        self.assertEqual(by_pid[1279899]["comm"], "MAXWELLCOMENGIN")

    def test_reopen_probe_pass_was_a_false_positive(self):
        automated = read_json("reopen_verdict_732554.json")
        corrected = read_json("corrected_audit_20260713.json")
        self.assertTrue(automated["fresh_reopen_pass"])
        self.assertFalse(corrected["corrected_pass"])
        self.assertIn(
            "not_sufficient_solution_validity_condition",
            corrected["solver_pid_survival_classification"],
        )

        convergence = (ROOT / "B_convergence_fresh_732554.prop").read_text(
            encoding="utf-8", errors="replace"
        )
        completed = re.search(
            r"^Completed\s*:\s*(\S+)", convergence, re.MULTILINE
        )
        self.assertIsNotNone(completed)
        self.assertEqual(completed.group(1), "N/A")
        self.assertEqual(
            [line for line in convergence.splitlines() if re.match(r"^\s*\d+\|", line)],
            [],
        )

    def test_saved_sibling_has_no_field_solution_payload(self):
        with tarfile.open(ROOT / "pilot_B_bundle_732549.tar.gz", "r:gz") as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
        self.assertIn("pilot_B.aedtresults/DesignB.asol", names)
        field_suffixes = {".fld", ".plo", ".sat", ".ngmesh", ".adp", ".profile"}
        self.assertFalse(
            [name for name in names if Path(name).suffix.casefold() in field_suffixes]
        )

    def test_final_policy_rejects_pool_activation(self):
        summary = read_json("experiment_summary_20260713.json")
        verdict = summary["verdict"]
        fresh = summary["fresh_reopen_evidence"]
        self.assertEqual(verdict["fault_isolation_contract"], "REJECT")
        self.assertFalse(verdict["live_pool_activation_authorized_by_this_experiment"])
        self.assertEqual(fresh["convergence_completed"], "N/A")
        self.assertEqual(fresh["convergence_data_rows"], 0)
        self.assertFalse(fresh["field_solution_payload_present"])
        self.assertFalse(fresh["fresh_reopen_pass"])

    def test_followup_priority_and_terminal_state_are_recorded(self):
        slurm = (ROOT / "reopen_slurm_job_732554.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        self.assertIn("Priority=4294210729", slurm)
        summary = read_json("experiment_summary_20260713.json")
        job = next(item for item in summary["job_transitions"] if item["job_id"] == 732554)
        self.assertEqual(job["terminal_state"], "COMPLETED")
        self.assertEqual(job["exit_code"], "0:0")


if __name__ == "__main__":
    unittest.main()
