import hashlib
import json
from pathlib import Path

from regression_260707.monitoring.readers import (
    ArtifactService,
    SEALED_SUCCESSOR_COHORT_ID,
    SEALED_SUCCESSOR_CONSTRAINT_VERSION,
    SEALED_SUCCESSOR_CONSTRAINTS,
    SEALED_SUCCESSOR_HANDOFF_ENV,
    SEALED_SUCCESSOR_HANDOFF_SCHEMA,
    SEALED_SUCCESSOR_HANDOFF_SHA_ENV,
    SEALED_SUCCESSOR_REPLAY_COMMIT,
    SEALED_SUCCESSOR_TEMPERATURE_TARGETS,
)


class _NoNsgaScheduler:
    def mft_pipeline_status(self):
        return {"available": False}


class _NoContinuousPipeline:
    def snapshot(self):
        return {"available": False, "warnings": []}


def _canonical_sha(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_fixture(tmp_path: Path) -> dict[str, Path | str]:
    deployment_root = tmp_path / "deployments" / SEALED_SUCCESSOR_COHORT_ID
    bundle_root = deployment_root / "bundle"
    candidate_path = (
        bundle_root
        / "artifacts"
        / "code"
        / "regression_260707"
        / "optimization"
        / "t120_exact_candidate_discovery_v3.json"
    )
    candidate_relative = candidate_path.relative_to(bundle_root).as_posix()
    source_model_sha = "1" * 64
    deployment_model_sha = "2" * 64
    stable_identity_sha = "3" * 64
    temperature_contract_sha = "4" * 64
    warm_start_sha = "5" * 64
    code_revision = "6" * 40
    constraints = {name: -1.0 for name in SEALED_SUCCESSOR_CONSTRAINTS}
    constraints.update({
        "Llt_robust_band": -0.05,
        "half_magnetizing_resonance_minimum": -1000.0,
        "decoded_space_shrink": 0.0,
        "minimum_physical_insulation": 0.0,
        "core_group_manufacturability_limit": 0.0,
        "exterior_height_limit": 0.0,
    })
    temperatures = {
        name: 100.0 - position
        for position, name in enumerate(SEALED_SUCCESSOR_TEMPERATURE_TARGETS)
    }
    candidate = {
        "schema_version": "mft-nsga2-t120-exact-candidate-discovery-v3",
        "created_at": "2026-07-18T17:38:30+09:00",
        "production_eligible": False,
        "live_switch_performed": False,
        "fea_submission_performed": False,
        "provenance": {
            "composite_model_manifest_sha256": deployment_model_sha,
            "source_model_manifest_sha256": source_model_sha,
        },
        "decoded_geometry": {
            "N1_main": 6,
            "N2_side": 21,
            "l1_mm": 72,
            "l2_mm": 357.5,
            "h1_mm": 606,
            "w1_mm": 598,
            "n_core_group": 4,
            "cw1_mm": 5.0,
            "gap1_mm": 3.0,
            "cw2_mm": 1.947,
            "gap2_mm": 0.607,
            "core_plate_t_mm": 19.6,
            "core_plate_pad_t_mm": 2.0,
            "wcp_t_mm": 23.4,
            "wcp_pad_t_mm": 2.0,
            "wcp_len_pct": 72.49,
        },
        "exterior_box_mm": [1000.0, 1000.0, 700.0],
        "total_positive_violation": 0.0,
        "constraints_G": constraints,
        "robust_temperature_C": temperatures,
    }
    candidate_sha = _write_json(candidate_path, candidate)

    hard_spec = {
        "B_limit_T": 1.2,
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 120.0,
        "insulation_min_mm": 40.0,
        "magnetizing_inductance_factor": 0.5,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "q_sigma": 1.0,
        "resonance_min_Hz": 10000.0,
        "size_H_max_mm": 750.0,
        "size_L_max_mm": 1200.0,
        "size_W_max_mm": 1200.0,
        "uncertainty_contract": "q90_conformal_half_width_physical_v1",
    }
    hard_spec_sha = _canonical_sha(hard_spec)
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = {
        "schema_version": "mft-tier1-slurm-deployment-v1",
        "attestation_schema_version": "mft-tier1-slurm-temperature-attestation-v3",
        "task_schema_version": "mft-tier1-slurm-seed-task-v3",
        "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
        "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "nsga_code_revision": code_revision,
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "source_model_manifest_sha256": source_model_sha,
        "deployment_model_manifest_sha256": deployment_model_sha,
        "stable_identity_sha256": stable_identity_sha,
        "temperature_constraint_contract_sha256": temperature_contract_sha,
        "warm_start_sha256": warm_start_sha,
        "files": {candidate_relative: {"sha256": candidate_sha}},
        "high_value_sha256": {candidate_relative: candidate_sha},
    }
    manifest_sha = _write_json(manifest_path, manifest)

    deployment_path = deployment_root / "deployment_plan.json"
    deployment = {
        "schema_version": "mft-tier1-slurm-deployment-v1",
        "attestation_schema_version": "mft-tier1-slurm-temperature-attestation-v3",
        "task_schema_version": "mft-tier1-slurm-seed-task-v3",
        "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
        "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "nsga_code_revision": code_revision,
        "production_eligible": False,
        "fea_submission_approved": False,
        "bundle_manifest": str(manifest_path.resolve()),
        "bundle_manifest_sha256": manifest_sha,
        "local_bundle": str(bundle_root.resolve()),
        "source_model_manifest_sha256": source_model_sha,
        "deployment_model_manifest_sha256": deployment_model_sha,
        "stable_identity_sha256": stable_identity_sha,
        "temperature_constraint_contract_sha256": temperature_contract_sha,
        "warm_start_sha256": warm_start_sha,
    }
    deployment_sha = _write_json(deployment_path, deployment)

    replay_path = tmp_path / "attestation" / "replay.json"
    replay = {
        "schema_version": "mft-nsga2-t120-independent-bundle-replay-v3",
        "completed_at": "2026-07-18T17:46:35+09:00",
        "production_eligible": False,
        "live_switch_performed": False,
        "fea_submission_performed": False,
        "successor_bundle": {
            "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
            "bundle_manifest_sha256": manifest_sha,
            "nsga_code_revision": code_revision,
            "all_manifest_file_hashes_passed": True,
            "all_high_value_hashes_passed": True,
            "source_model_manifest_sha256": source_model_sha,
            "deployment_model_manifest_sha256": deployment_model_sha,
            "stable_identity_sha256": stable_identity_sha,
            "warm_start_sha256": warm_start_sha,
        },
        "append_only_discovery_seal": {
            "sha256": candidate_sha,
            "sha256_before_replay": candidate_sha,
            "sha256_after_replay": candidate_sha,
            "size_before_equals_after": True,
            "mtime_before_equals_after": True,
            "sha256_before_equals_after": True,
        },
        "independent_replay": {
            "repair_bit_exact": True,
            "decoder_valid": True,
            "hard_geometry_joint_pass": True,
            "feature_parity_pass": True,
            "all_hard_pass": True,
            "total_positive_violation": 0.0,
            "Llt_mu_uH": 27.5,
            "Llt_q90_half_width_uH": 0.5,
            "Llt_robust_G_uH": -0.05,
            "objective_volume_liters": 700.0,
            "objective_total_loss_W": 6000.0,
            "exterior_box_mm": [1000.0, 1000.0, 700.0],
            "resonance_screen_Hz": 11000.0,
            "constraint_G": constraints,
            "robust_temperature_C": temperatures,
        },
        "release_decision": {
            "immutable_successor_bundle_ready": True,
            "independent_replay_passed": True,
            "automatic_live_switch_allowed": False,
            "automatic_fea_submission_allowed": False,
            "root_approval_required": True,
        },
    }
    replay_sha = _write_json(replay_path, replay)

    handoff_path = tmp_path / "handoff" / "handoff.json"
    handoff = {
        "schema_version": SEALED_SUCCESSOR_HANDOFF_SCHEMA,
        "evidence_id": "t120-exact-test",
        "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
        "deployment_plan": {
            "path": str(deployment_path.resolve()), "sha256": deployment_sha,
        },
        "bundle_manifest": {
            "path": str(manifest_path.resolve()), "sha256": manifest_sha,
        },
        "replay_attestation": {
            "path": str(replay_path.resolve()),
            "sha256": replay_sha,
            "commit": SEALED_SUCCESSOR_REPLAY_COMMIT,
        },
        "candidate": {
            "path": str(candidate_path.resolve()), "sha256": candidate_sha,
        },
        "all_hard_pass": True,
        "total_positive_violation": 0.0,
        "worst_temperature_C": 100.0,
        "Llt_robust_G_uH": -0.05,
        "production_submission_enabled": False,
        "fea_submission_enabled": False,
        "scheduler_task_id": None,
        "live_pointer_switch_performed": False,
    }
    handoff_sha = _write_json(handoff_path, handoff)
    return {
        "handoff_path": handoff_path,
        "handoff_sha": handoff_sha,
        "candidate_path": candidate_path,
    }


def _service(tmp_path: Path) -> ArtifactService:
    return ArtifactService(
        tmp_path,
        scheduler=_NoNsgaScheduler(),
        continuous_pipeline=_NoContinuousPipeline(),
        record_runtime=False,
    )


def test_sealed_successor_is_separate_from_canonical_candidates(
    tmp_path, monkeypatch,
):
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_ENV, str(fixture["handoff_path"]))
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_SHA_ENV, str(fixture["handoff_sha"]))

    payload = _service(tmp_path).nsga2()

    assert payload["candidate_count"] == 0
    assert payload["valid_candidate_count"] == 0
    assert payload["candidates"] == []
    successor = payload["sealed_successor"]
    assert successor["source_kind"] == "sealed_successor_evidence"
    assert successor["status"] == "fea_pending"
    assert successor["integrity_verified"] is True
    assert successor["canonical_candidate"] is False
    assert successor["terminal_result"] is False
    assert successor["production"] is False
    assert successor["fea"] is False
    assert successor["fea_status"] == "pending"
    assert len(successor["candidate"]["robust_temperatures_C"]) == 11
    assert set(successor["candidate"]["constraints_G"]) == set(
        SEALED_SUCCESSOR_CONSTRAINTS
    )


def test_sealed_successor_child_tamper_fails_closed_without_stale_design(
    tmp_path, monkeypatch,
):
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_ENV, str(fixture["handoff_path"]))
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_SHA_ENV, str(fixture["handoff_sha"]))
    service = _service(tmp_path)
    assert service.nsga2()["sealed_successor"]["available"] is True

    candidate_path = fixture["candidate_path"]
    assert isinstance(candidate_path, Path)
    candidate_path.write_bytes(candidate_path.read_bytes() + b"\n")
    payload = service.nsga2()

    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    rejected = payload["sealed_successor"]
    assert rejected["available"] is False
    assert rejected["integrity_verified"] is False
    assert rejected["status"] == "rejected"
    assert "candidate" not in rejected
    assert "SHA-256 mismatch" in rejected["warning"]


def test_sealed_successor_root_hash_mismatch_fails_before_child_disclosure(
    tmp_path, monkeypatch,
):
    fixture = _sealed_fixture(tmp_path)
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_ENV, str(fixture["handoff_path"]))
    monkeypatch.setenv(SEALED_SUCCESSOR_HANDOFF_SHA_ENV, "0" * 64)

    rejected = _service(tmp_path).nsga2()["sealed_successor"]

    assert rejected["available"] is False
    assert rejected["integrity_verified"] is False
    assert rejected["status"] == "rejected"
    assert "candidate" not in rejected
    assert "handoff SHA-256 mismatch" in rejected["warning"]
