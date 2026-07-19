from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tools.tier1_resonance_feedback import (
    CONFORMAL_HALF_WIDTH_CONTRACT,
    HARD_SPEC,
    SIDECAR_SCHEMA,
    TARGETS,
    TargetPredictor,
    _q90_half_width,
    authenticate_v6,
    compose_audits,
    derive_resonance_screen,
    monitor_searches,
    write_audit,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _LinearModel:
    n_jobs = 1

    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, frame):
        return frame["x"].to_numpy(dtype=float) + self.offset


class Tier1FeedbackTests(unittest.TestCase):
    def _runtime(self, root: Path):
        solver = "a" * 40
        library = "b" * 40
        candidates = []
        for index in (1, 2):
            digest = f"{index:064x}"
            candidates.append({
                "candidate_id": f"res20-{index:03d}",
                "candidate_digest": digest,
                "task_name": f"task-{index}",
                "scheduler_dedupe_key": f"dedupe-{index}",
                "effective_params": {
                    "N1_main": 8, "N1_side": 0,
                    "N2_main": 80, "N2_side": 0,
                },
            })
        plan = {
            "plan_id": "plan",
            "production_eligible": False,
            "automatic_model_or_candidate_promotion": False,
            "solver_contract": {
                "solver_revision": solver,
                "library_revision": library,
            },
            "candidates": candidates,
        }
        plan_path = root / "plan.json"
        _write_json(plan_path, plan)
        plan_sha = _sha(plan_path)
        states = {}
        for index, candidate in enumerate(candidates, start=1):
            result = {
                "candidate_id": candidate["candidate_id"],
                "candidate_digest": candidate["candidate_digest"],
                "task_id": 100 + index,
                "source_plan_sha256": plan_sha,
                "measurement_verified": True,
                "production_eligible": False,
                "fea_or_production_approval_granted": False,
                "solver_revision": solver,
                "library_revision": library,
                "Llt_phys": 34.0 + index,
                "k": 0.995,
                "C_tx_tx_F": 1.7e-8,
                "C_rx_rx_F": 4.0e-10,
                "C_tx_rx_F": 2.5e-10,
                "f_res_tx_self_Hz": 20_000.0,
                "f_res_rx_self_Hz": 13_000.0,
                "f_res_interwinding_Hz": 1_600_000.0,
            }
            result_path = root / "results" / f"{candidate['candidate_id']}.json"
            _write_json(result_path, result)
            states[candidate["candidate_digest"]] = {
                "candidate_id": candidate["candidate_id"],
                "state": "measurement_verified",
                "measurement_verified": True,
                "production_eligible": False,
                "task_id": 100 + index,
                "task_name": candidate["task_name"],
                "scheduler_dedupe_key": candidate["scheduler_dedupe_key"],
                "task_status": "completed",
                "result_reference": {
                    "path": str(result_path),
                    "sha256": _sha(result_path),
                    "size": result_path.stat().st_size,
                },
            }
        state = {
            "plan_sha256": plan_sha,
            "production_eligible": False,
            "candidates": states,
        }
        _write_json(root / "state.json", state)
        return candidates, states

    def test_frozen_cohort_authenticates_and_never_becomes_strict_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, _states = self._runtime(root)
            rows, audit = authenticate_v6(
                root,
                expected_verified=1,
                excluded_candidate_ids=[candidates[1]["candidate_id"]],
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema_version"], SIDECAR_SCHEMA)
        self.assertTrue(rows[0]["measurement_verified"])
        self.assertFalse(rows[0]["strict_full_eligible"])
        self.assertFalse(rows[0]["loss_thermal_eligible"])
        self.assertFalse(rows[0]["production_eligible"])
        self.assertEqual(tuple(rows[0]["eligible_targets"]), TARGETS)
        self.assertEqual(audit["duplicate_counts"]["task_id"], 0)

    def test_tampered_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _candidates, states = self._runtime(root)
            status = next(iter(states.values()))
            result_path = Path(status["result_reference"]["path"])
            result = json.loads(result_path.read_text())
            result["k"] = 0.9
            _write_json(result_path, result)
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                authenticate_v6(root, expected_verified=2)

    def test_delta_sidecar_authenticates_frozen_parent_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "runtime")
            candidates, _states = self._runtime(root)
            frozen_output = Path(temporary, "frozen")
            write_audit(
                root,
                frozen_output,
                expected_verified=1,
                excluded_candidate_ids=[candidates[1]["candidate_id"]],
            )
            combined_output = Path(temporary, "combined")
            combined = write_audit(
                root,
                combined_output,
                expected_verified=2,
                delta_from_audit=frozen_output / "audit.json",
            )
        self.assertEqual(combined["combined_cohort"]["row_count"], 2)
        self.assertEqual(combined["delta_sidecar"]["row_count"], 1)
        self.assertEqual(
            combined["delta_sidecar"]["candidate_ids"],
            [candidates[1]["candidate_id"]],
        )

    def test_separate_runtime_composition_authenticates_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_runtime = Path(temporary, "base-runtime")
            delta_runtime = Path(temporary, "delta-runtime")
            base_candidates, _ = self._runtime(base_runtime)
            delta_candidates, _ = self._runtime(delta_runtime)

            # Make the second runtime a genuinely independent scheduler/result
            # identity while preserving the exact solver/library contract.
            delta_plan_path = delta_runtime / "plan.json"
            delta_plan = json.loads(delta_plan_path.read_text())
            candidate = delta_plan["candidates"][0]
            candidate["candidate_id"] = "res20-065"
            candidate["candidate_digest"] = "f" * 64
            candidate["task_name"] = "task-50961"
            candidate["scheduler_dedupe_key"] = "dedupe-50961"
            _write_json(delta_plan_path, delta_plan)
            plan_sha = _sha(delta_plan_path)
            old_digest = delta_candidates[0]["candidate_digest"]
            state_path = delta_runtime / "state.json"
            state = json.loads(state_path.read_text())
            status = state["candidates"].pop(old_digest)
            status.update({
                "candidate_id": "res20-065",
                "task_id": 50961,
                "task_name": "task-50961",
                "scheduler_dedupe_key": "dedupe-50961",
            })
            result_path = Path(status["result_reference"]["path"])
            result = json.loads(result_path.read_text())
            result.update({
                "candidate_id": "res20-065",
                "candidate_digest": "f" * 64,
                "task_id": 50961,
                "source_plan_sha256": plan_sha,
            })
            new_result_path = result_path.with_name("res20-065.json")
            _write_json(new_result_path, result)
            status["result_reference"] = {
                "path": str(new_result_path),
                "sha256": _sha(new_result_path),
                "size": new_result_path.stat().st_size,
            }
            state["plan_sha256"] = plan_sha
            state["candidates"] = {"f" * 64: status}
            _write_json(state_path, state)

            base_output = Path(temporary, "base-audit")
            delta_output = Path(temporary, "delta-audit")
            write_audit(
                base_runtime,
                base_output,
                expected_verified=1,
                excluded_candidate_ids=[base_candidates[1]["candidate_id"]],
            )
            write_audit(
                delta_runtime,
                delta_output,
                expected_verified=1,
                excluded_candidate_ids=[delta_candidates[1]["candidate_id"]],
            )
            composed = compose_audits(
                audit_paths=[base_output / "audit.json", delta_output / "audit.json"],
                output=Path(temporary, "composed"),
            )
            self.assertEqual(composed["composition"]["source_row_counts"], [1, 1])
            self.assertEqual(composed["composition"]["combined_row_count"], 2)
            self.assertFalse(composed["production_eligible"])

            with self.assertRaisesRegex(RuntimeError, "identity duplicates"):
                compose_audits(
                    audit_paths=[
                        base_output / "audit.json",
                        Path(temporary, "composed", "audit.json"),
                    ],
                    output=Path(temporary, "invalid"),
                )

    def test_predictor_attests_conformal_half_width(self):
        bundle = {
            "schema_version": "mft-tier1-target-model-v1",
            "features": ["x"],
            "transform": None,
            "models": [
                ("lightgbm", _LinearModel(-0.1)),
                ("lightgbm", _LinearModel(0.1)),
            ],
            "absolute_q90_half_width": 0.25,
        }
        predictor = TargetPredictor(bundle)
        mu, half = predictor.predict_mu_sigma(pd.DataFrame({"x": [1.0]}))
        self.assertAlmostEqual(mu[0], 1.0)
        self.assertGreaterEqual(half[0], 0.25)
        self.assertEqual(
            predictor.predict_mu_sigma_contract,
            CONFORMAL_HALF_WIDTH_CONTRACT,
        )

    def test_target_conformal_q90_has_empirical_held_out_coverage(self):
        errors = np.linspace(0.0, 1.0, 64)
        half_width, coverage = _q90_half_width(errors)
        self.assertGreaterEqual(coverage, 0.9)
        self.assertEqual(half_width, np.quantile(errors, 0.9, method="higher"))
        with self.assertRaisesRegex(RuntimeError, "at least 30"):
            _q90_half_width(errors[:29])

    def test_versioned_110c_15khz_hard_contract(self):
        self.assertEqual(HARD_SPEC["primary_conductor_thickness_mm"], 5.0)
        self.assertEqual(HARD_SPEC["n_core_group_max"], 4)
        self.assertEqual(HARD_SPEC["insulation_min_mm"], 40.0)
        self.assertEqual(HARD_SPEC["B_limit_T"], 1.2)
        self.assertEqual(HARD_SPEC["T_limit_C"], 110.0)
        self.assertEqual(HARD_SPEC["resonance_min_Hz"], 15_000.0)
        self.assertEqual(HARD_SPEC["magnetizing_inductance_factor"], 0.5)
        self.assertEqual(
            (
                HARD_SPEC["size_W_max_mm"], HARD_SPEC["size_L_max_mm"],
                HARD_SPEC["size_H_max_mm"],
            ),
            (1_200.0, 1_200.0, 750.0),
        )
        screen = derive_resonance_screen(
            {
                "Llt_phys": 35.0, "k": 0.995,
                "C_tx_tx_F": 1.7e-8, "C_rx_rx_F": 4e-10,
                "C_tx_rx_F": 2.5e-10,
            },
            {"N1_main": 8, "N1_side": 0, "N2_main": 80, "N2_side": 0},
        )
        self.assertTrue(np.isfinite(screen["f_res_min_screen_Hz"]))
        self.assertEqual(
            screen["f_res_min_screen_Hz"],
            min(
                screen["f_res_tx_screen_Hz"],
                screen["f_res_rx_screen_Hz"],
                screen["f_res_interwinding_screen_Hz"],
            ),
        )

    def test_monitor_attests_completed_seed_and_no_fea_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_output = root / "seed-11"
            _write_json(seed_output / "result.json", {
                "seed": 11,
                "completed_generations": 241,
                "feasible_pareto_count": 3,
                "fea_submission_approved": False,
            })
            launch_path = root / "launch.json"
            _write_json(launch_path, {
                "schema_version": "mft-tier1-corrected-search-launch-v1",
                "model_manifest_sha256": "a" * 64,
                "nsga_code_revision": "b" * 40,
                "population": 160,
                "max_generations": 240,
                "jobs": [{
                    "seed": 11,
                    "pid": 999_999_999,
                    "command": ["python", "search-seed"],
                    "output": str(seed_output),
                }],
            })
            identity_status_path = root / "identity-status.json"
            _write_json(identity_status_path, {
                "launch_sha256": _sha(launch_path),
                "jobs": [{
                    "seed": 11,
                    "process_alive": True,
                    "command_identity_verified": True,
                }],
            })
            status = monitor_searches(
                launch_path=launch_path,
                output=root / "status.json",
                once=True,
                identity_status_path=identity_status_path,
            )
        self.assertEqual(status["state_counts"], {"completed": 1})
        self.assertEqual(status["jobs"][0]["completed_generations"], 241)
        self.assertTrue(status["jobs"][0]["launch_command_identity_verified"])
        self.assertIsNone(status["jobs"][0]["runtime_command_identity_verified"])
        self.assertTrue(status["jobs"][0]["terminal_result_verified"])
        self.assertFalse(status["submission_performed"])
        self.assertFalse(status["fea_submission_approved"])


if __name__ == "__main__":
    unittest.main()
