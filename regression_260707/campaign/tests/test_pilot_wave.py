import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))
sys.path.insert(0, str(REPO_ROOT))

import pilot_wave  # noqa: E402
from module.input_parameter_260706 import KEYS, get_drawing_default_params  # noqa: E402


SOLVER_REVISION = "a" * 40
TRUSTED_REVISION = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
DATA_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
TARGETS = (
    "Llt_phys",
    "k",
    "P_winding_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
)


def _legacy_row(index, llt, *, strict=True, revision=TRUSTED_REVISION):
    defaults = get_drawing_default_params()
    row = {key: defaults[key] for key in KEYS}
    row.update({
        "full_model": 0,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "git_hash": revision,
        "pyaedt_library_git_hash": LIBRARY_REVISION,
        "git_dirty": 0,
        "pyaedt_library_git_dirty": 0,
        "task_id": 10_000 + index,
        "task_name": f"legacy-{index}",
        "project_name": f"legacy_project_{index}",
        "Llt": float(llt),
        "k": 0.95 + index / 1000.0,
        "P_winding_total": 1000.0 + index,
        "P_Tx_main_group": 600.0 + index,
        "P_Rx_main_group": 300.0 + index,
        "P_Rx_side_total": 100.0 + index,
        "_test_strict": bool(strict),
    })
    return row


def _fake_annotate(frame, *_args, **_kwargs):
    out = frame.copy()
    out["_strict_valid_em"] = out["_test_strict"].astype(bool)
    out["_strict_valid_thermal"] = out["_test_strict"].astype(bool)
    out["_strict_valid_full"] = out["_test_strict"].astype(bool)
    out["_strict_invalid_reasons"] = out["_test_strict"].map(
        {True: "", False: "thermal_flag:test"}
    )
    return out


def _manifest_entries(manifest):
    entries = manifest.get("entries", manifest.get("tasks"))
    if not isinstance(entries, list):
        raise AssertionError("pilot manifest has no entries/tasks list")
    return entries


def _entry_kind(entry):
    return entry.get("kind", entry.get("entry_type", entry.get("type")))


def _source_targets(entry):
    targets = entry.get("source_targets", entry.get("legacy_targets"))
    if not isinstance(targets, dict):
        raise AssertionError("legacy replay entry has no source target mapping")
    return targets


def _call_build_manifest(dataset):
    return pilot_wave.build_manifest(
        solver_revision=SOLVER_REVISION,
        library_revision=LIBRARY_REVISION,
        data_revision=DATA_REVISION,
        core_lamination_factor=0.85,
        legacy_dataset=dataset,
        fresh_count=2,
        replay_count=3,
        seed=260713,
    )


class PilotManifestTests(unittest.TestCase):
    def test_synthetic_parquet_replays_strict_trusted_rows_across_llt_range(self):
        rows = [
            _legacy_row(0, 5.0),
            _legacy_row(1, 10.0),
            _legacy_row(2, 15.0, strict=False),
            _legacy_row(3, 20.0),
            _legacy_row(4, 25.0),
            _legacy_row(5, 30.0),
            _legacy_row(6, 100.0, revision="c" * 40),
        ]
        candidates = iter([
            {**{key: rows[0][key] for key in KEYS}, "N1_main": 7},
            {**{key: rows[0][key] for key in KEYS}, "N1_main": 8},
        ])

        annotate_owner = getattr(pilot_wave, "quality_contract", pilot_wave)
        sampler_owner = getattr(pilot_wave, "pinned_pilot", pilot_wave)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "legacy.parquet"
            pd.DataFrame(rows).to_parquet(dataset, index=False)
            with mock.patch.object(
                    annotate_owner, "annotate_validity",
                    side_effect=_fake_annotate), mock.patch.object(
                        sampler_owner, "cursor_after_valid_candidates",
                        return_value=0), mock.patch.object(
                        sampler_owner, "next_valid_candidate",
                        side_effect=lambda cursor, seed: (
                            cursor + 1, cursor, next(candidates))):
                manifest = _call_build_manifest(dataset)

        entries = _manifest_entries(manifest)
        fresh = [entry for entry in entries if _entry_kind(entry) == "fresh"]
        replay = [
            entry for entry in entries
            if _entry_kind(entry) in ("replay", "legacy_replay")
        ]
        self.assertEqual(len(fresh), 2)
        self.assertEqual(len(replay), 3)
        self.assertEqual(len({entry["replay_of"] for entry in replay}), 3)
        self.assertTrue({entry["replay_of"] for entry in replay}.issubset(
            {10_000, 10_001, 10_003, 10_004, 10_005}
        ))

        replay_llt = [_source_targets(entry)["Llt_phys"] for entry in replay]
        self.assertTrue(all(set(_source_targets(entry)) == set(TARGETS)
                            for entry in replay))
        # Every trusted row is eighth symmetry, so the physical value is 2*Llt.
        self.assertTrue(all(value in {10.0, 20.0, 40.0, 50.0, 60.0}
                            for value in replay_llt))
        self.assertLessEqual(min(replay_llt), 20.0)
        self.assertGreaterEqual(max(replay_llt), 50.0)
        self.assertNotIn(30.0, replay_llt)  # quarantined source row
        self.assertNotIn(200.0, replay_llt)  # wrong solver revision

        for entry in entries:
            self.assertEqual(entry["params"]["core_lamination_factor"], 0.85)
            self.assertEqual(
                entry["params"]["physics_data_revision"], DATA_REVISION
            )
            self.assertNotIn("replay_of", entry["params"])
        self.assertEqual(manifest["solver_revision"], SOLVER_REVISION)
        self.assertEqual(manifest["library_revision"], LIBRARY_REVISION)
        self.assertEqual(manifest["data_revision"], DATA_REVISION)
        pilot_wave.validate_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "yield threshold"):
            pilot_wave.validate_manifest({**manifest, "yield_threshold": 0.50})


class PilotComparisonTests(unittest.TestCase):
    def test_relative_error_handles_boundaries_and_zero_reference(self):
        self.assertAlmostEqual(pilot_wave.relative_error(100.5, 100.0), 0.005)
        self.assertAlmostEqual(pilot_wave.relative_error(97.0, 100.0), 0.03)
        self.assertEqual(pilot_wave.relative_error(0.0, 0.0), 0.0)
        self.assertTrue(math.isinf(pilot_wave.relative_error(1.0, 0.0)))

    def test_paired_target_gates_are_inclusive_and_aggregate_per_target(self):
        source = {
            "Llt_phys": 100.0,
            "k": 0.8,
            "P_winding_total": 100.0,
            "P_Tx_main_group": 100.0,
            "P_Rx_main_group": 100.0,
            "P_Rx_side_total": 0.0,
        }
        manifest = {
            "entries": [
                {
                    "kind": "legacy_replay", "name": "replay-0",
                    "task_id": 501, "replay_of": 10001,
                    "source_targets": dict(source),
                },
                {
                    "kind": "legacy_replay", "name": "replay-1",
                    "task_id": 502, "replay_of": 10002,
                    "source_targets": dict(source),
                },
            ]
        }
        results = pd.DataFrame([
            {
                "task_id": 501, "task_name": "replay-0",
                "full_model": 0, "Llt": 50.25,
                "k": 0.804,  # exactly 0.5%
                "P_winding_total": 103.0,
                "P_Tx_main_group": 97.0,
                "P_Rx_main_group": 103.0,
                "P_Rx_side_total": 0.0,
            },
            {
                "task_id": 502, "task_name": "replay-1",
                "full_model": 0, "Llt": 50.0,
                "k": 0.8040008,  # just beyond 0.5%; only k must fail
                "P_winding_total": 100.0,
                "P_Tx_main_group": 100.0,
                "P_Rx_main_group": 100.0,
                "P_Rx_side_total": 0.0,
            },
        ])

        report = pilot_wave.build_salvage_report(manifest, results)

        self.assertEqual(len(report["pairs"]), 2)
        self.assertTrue(report["targets"]["Llt_phys"]["passed"])
        self.assertFalse(report["targets"]["k"]["passed"])
        for target in TARGETS[2:]:
            self.assertTrue(report["targets"][target]["passed"])
        self.assertFalse(report["passed"])

    def test_paired_gate_rejects_task_identity_drift(self):
        source = {target: 1.0 for target in TARGETS}
        manifest = {
            "entries": [{
                "kind": "legacy_replay", "name": "replay-0",
                "task_id": 501, "replay_of": 10001,
                "source_targets": source,
            }]
        }
        result = {
            "task_id": 501, "task_name": "wrong-task",
            "full_model": 1, "Llt": 1.0,
            **{target: 1.0 for target in TARGETS if target != "Llt_phys"},
        }

        report = pilot_wave.build_salvage_report(
            manifest, pd.DataFrame([result])
        )

        self.assertIn("identity:task_name", report["pairs"][0]["match_error"])
        self.assertTrue(all(
            not gate["passed"] for gate in report["targets"].values()
        ))
        self.assertFalse(report["passed"])


class PilotYieldReportTests(unittest.TestCase):
    def test_yield_uses_all_planned_entries_and_counts_each_reason(self):
        manifest = {
            "entries": [
                {
                    "entry_id": f"fresh-{index:03d}", "kind": "fresh",
                    "task_id": 1000 + index,
                }
                for index in range(20)
            ],
            "yield_threshold": 0.85,
            "submission": {
                "executed": True, "completed_at": "2026-07-13T00:00:00Z"
            },
        }
        validity = [
            {
                "entry_id": entry["entry_id"],
                "kind": "fresh",
                "strict_full": index < 17,
                "reasons": [] if index < 17 else [
                    "thermal_flag:test", "thermal_convergence:test"
                ],
            }
            for index, entry in enumerate(manifest["entries"])
        ]
        salvage = {"passed": True}
        with mock.patch.object(
                pilot_wave, "_entry_validity", return_value=validity), \
                mock.patch.object(
                    pilot_wave, "build_salvage_report", return_value=salvage):
            report = pilot_wave.build_report(manifest, pd.DataFrame())

        self.assertEqual(report["yield"]["strict_full"], 17)
        self.assertEqual(report["yield"]["planned"], 20)
        self.assertEqual(report["yield"]["yield"], 0.85)
        self.assertTrue(report["yield"]["passed"])
        self.assertEqual(
            report["yield"]["quarantine_reasons"],
            {"thermal_convergence:test": 3, "thermal_flag:test": 3},
        )
        self.assertTrue(report["passed"])

        partial = {
            **manifest,
            "submission": {"executed": True, "completed_at": None},
        }
        with mock.patch.object(
                pilot_wave, "_entry_validity", return_value=validity), \
                mock.patch.object(
                    pilot_wave, "build_salvage_report", return_value=salvage):
            partial_report = pilot_wave.build_report(partial, pd.DataFrame())
        self.assertFalse(partial_report["submission_complete"])
        self.assertFalse(partial_report["passed"])


if __name__ == "__main__":
    unittest.main()
