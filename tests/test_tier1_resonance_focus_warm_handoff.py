from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import tier1_resonance_focus_warm_handoff as handoff


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _candidate(role: str, index: int) -> dict:
    params = {"fixture_role": role, "fixture_index": index}
    constraints = {name: -1.0 for name in handoff.EXPECTED_CONSTRAINT_NAMES}
    if role == "normalized":
        constraints[handoff.RESONANCE_CONSTRAINT] = 100.0 + index
    else:
        constraints[handoff.RESONANCE_CONSTRAINT] = -10.0 - index
        constraints["Llt_robust_band"] = 0.01 * (index + 1)
    return {
        "rank": index + 1,
        "terminal_population_index": index,
        "decoded_params": params,
        "decoded_params_sha256": handoff._json_sha(params),
        "constraint_G": constraints,
        "target_predictions": {"Llt_phys": 27.5},
        "target_conformal_half_width": {"Llt_phys": 0.4},
        "derived_resonance": {"f_res_min_screen_Hz": 15_000.0},
        "production_eligible": False,
        "fea_submission_approved": False,
        "eligible_for_submission": False,
    }


def _source(
    root: Path, role: str, count: int, seed: int,
    model_sha: str = (
        "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
    ),
) -> None:
    root = root.resolve()
    cohort = f"fixture-{role}-cohort"
    source_sha = (
        "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
    )
    bundle_sha = ("6" if role == "main" else "7" if role == "deep" else "8") * 64
    candidates = [_candidate(role, index) for index in range(count)]
    result = {
        "schema_version": handoff.SEARCH_SCHEMA,
        "seed": seed,
        "model_manifest_sha256": model_sha,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec": handoff.HARD_SPEC,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "temperature_constraint_contract_sha256": (
            handoff.TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        ),
        "constraint_names": list(handoff.EXPECTED_CONSTRAINT_NAMES),
        "constraint_minimum_G": {
            name: min(row["constraint_G"][name] for row in candidates)
            for name in handoff.EXPECTED_CONSTRAINT_NAMES
        },
        "terminal_population_best_constraint_G": candidates[0]["constraint_G"],
        "completed_generations": 600,
        "feasible_pareto_count": 0,
        "candidates": [],
        "next_target_fea_batch_plan": {
            "candidates": candidates,
            "submission_performed": False,
            "current_candidates_eligible_for_submission": False,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
    }
    record_dir = root / "cohorts" / cohort / "results" / f"task-{seed}"
    result_path = record_dir / "result.json"
    result_sha = _write_json(result_path, result)
    seed_status = {
        "schema_version": handoff.SEED_STATUS_SCHEMA,
        "state": "completed",
        "exit_code": 0,
        "seed": seed,
        "task_id": str(seed),
        "cohort_id": cohort,
        "bundle_manifest_sha256": bundle_sha,
        "result_sha256": result_sha,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "temperature_constraint_contract_sha256": (
            handoff.TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        ),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
    }
    seed_status_path = record_dir / "seed_status.json"
    seed_status_sha = _write_json(seed_status_path, seed_status)
    terminal = {
        "schema_version": handoff.TERMINAL_SCHEMA,
        "authenticated": True,
        "terminal_state": "completed",
        "scheduler_state": "completed",
        "scheduler_exit_code": 0,
        "cohort_id": cohort,
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "source_model_manifest_sha256": source_sha,
        "temperature_constraint_contract_sha256": (
            handoff.TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        ),
        "task_id": seed,
        "seed": seed,
        "remote_status": {
            "local_path": str(seed_status_path),
            "sha256": seed_status_sha,
        },
        "result": {"local_path": str(result_path), "sha256": result_sha},
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
    }
    identity = {
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": model_sha,
        "temperature_constraint_contract_sha256": (
            handoff.TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        ),
    }
    search_profile = None if role == "main" else {
        "namespace": handoff.ROLE_NAMESPACES[role]
    }
    updated_at = "2026-07-19T00:00:00+00:00"
    status = {
        "schema_version": handoff.STATUS_SCHEMA,
        "updated_at": updated_at,
        "cohort_id": cohort,
        **identity,
        "hard_spec": handoff.HARD_SPEC,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
        "search_profile": search_profile,
        "healthy": True,
        "error": None,
        "terminal_results": [terminal],
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    status_path = root / "cohorts" / cohort / "status.json"
    status_sha = _write_json(status_path, status)
    current = {
        "cohort_id": cohort,
        **identity,
        "hard_spec": handoff.HARD_SPEC,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
        "bundle_manifest_sha256": bundle_sha,
    }
    pointer = {
        "schema_version": handoff.POINTER_SCHEMA,
        "current": current,
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
    }
    pointer_path = root / "canonical" / "model_pointer.json"
    pointer_sha = _write_json(pointer_path, pointer)
    index = {
        "schema_version": handoff.INDEX_SCHEMA,
        "updated_at": updated_at,
        "active_cohort_id": cohort,
        **identity,
        "hard_spec": handoff.HARD_SPEC,
        "path_containment_root": str(root),
        "model_pointer": {
            "path": str(pointer_path),
            "sha256": pointer_sha,
            "schema_version": handoff.POINTER_SCHEMA,
        },
        "status": {
            "path": str(status_path),
            "sha256": status_sha,
            "schema_version": handoff.STATUS_SCHEMA,
        },
    }
    _write_json(root / "canonical" / "index.json", index)


def _fixtures(tmp_path: Path):
    roots = {role: tmp_path / role for role in ("main", "deep", "normalized")}
    _source(roots["main"], "main", 20, 1001)
    _source(roots["deep"], "deep", 20, 2001)
    _source(roots["normalized"], "normalized", 40, 3001)
    return roots


def _patch_decoder(monkeypatch):
    dims = tuple((f"x{index}", 0.0, 1.0) for index in range(25))
    monkeypatch.setattr(
        handoff, "_load_sobol_schema", lambda _root: (dims, 1, 10, {}),
    )

    def decode(params, *_args):
        role_offset = {"main": 0.0, "deep": 0.3, "normalized": 0.6}[
            params["fixture_role"]
        ]
        base = role_offset + params["fixture_index"] / 1000.0
        return np.mod(base + np.arange(25) / 97.0, 1.0)

    monkeypatch.setattr(handoff, "decoded_to_unit", decode)


def _deployment_contract_probe(warm_path: Path):
    code = (
        "from pathlib import Path; "
        "from tools import tier1_resonance_focus_slurm_rolling as entry; "
        f"print(entry.rolling.validate_resonance_focus_warm_contract(Path({str(warm_path)!r})))"
    )
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=handoff.REPO,
        env=environment, capture_output=True, text=True, check=False,
    )


def test_builds_deterministic_exact_64x25_50_50_pool(monkeypatch, tmp_path):
    roots = _fixtures(tmp_path)
    _patch_decoder(monkeypatch)
    first = handoff.build_handoff(
        main_root=roots["main"], deep_root=roots["deep"],
        normalized_root=roots["normalized"], nsga_code_root=tmp_path,
        output=tmp_path / "output-a",
    )
    second = handoff.build_handoff(
        main_root=roots["main"], deep_root=roots["deep"],
        normalized_root=roots["normalized"], nsga_code_root=tmp_path,
        output=tmp_path / "output-b",
    )

    warm = np.load(tmp_path / "output-a" / "next_warm_start.npy")
    assert warm.shape == (64, 25)
    assert first["warm_start"]["sha256"] == second["warm_start"]["sha256"]
    assert first["category_counts"] == {
        "all_except_resonance": 32,
        "resonance_pass_near": 32,
    }
    categories = [row["category"] for row in first["selected_provenance"]]
    assert categories[::2] == ["all_except_resonance"] * 32
    assert categories[1::2] == ["resonance_pass_near"] * 32
    resonance_roles = {
        row["source_role"] for row in first["selected_provenance"]
        if row["category"] == "resonance_pass_near"
    }
    assert resonance_roles == {"main", "deep"}
    assert all(
        len(source["terminal_result_set_sha256"]) == 64
        for source in first["source_evidence"].values()
    )
    assert first["optimizer_resonance_scale_Hz"] == 250.0
    assert first["scheduler_write_performed"] is False
    assert first["fea_submission_performed"] is False


def test_result_byte_drift_fails_closed(monkeypatch, tmp_path):
    roots = _fixtures(tmp_path)
    _patch_decoder(monkeypatch)
    result_path = next((roots["normalized"] / "cohorts").rglob("result.json"))
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="result .* SHA mismatch"):
        handoff.build_handoff(
            main_root=roots["main"], deep_root=roots["deep"],
            normalized_root=roots["normalized"], nsga_code_root=tmp_path,
            output=tmp_path / "output",
        )


def test_category_shortage_never_relaxes_selection(monkeypatch, tmp_path):
    roots = _fixtures(tmp_path)
    _source(roots["normalized"], "normalized", 31, 3001)
    _patch_decoder(monkeypatch)
    with pytest.raises(RuntimeError, match="candidate shortage: need 32, have 31"):
        handoff.build_handoff(
            main_root=roots["main"], deep_root=roots["deep"],
            normalized_root=roots["normalized"], nsga_code_root=tmp_path,
            output=tmp_path / "output",
        )


def test_cross_source_model_generation_mismatch_fails_closed(
    monkeypatch, tmp_path,
):
    roots = _fixtures(tmp_path)
    _source(roots["deep"], "deep", 20, 2001, model_sha="9" * 64)
    _patch_decoder(monkeypatch)
    with pytest.raises(RuntimeError, match="result generation mismatch"):
        handoff.build_handoff(
            main_root=roots["main"], deep_root=roots["deep"],
            normalized_root=roots["normalized"], nsga_code_root=tmp_path,
            output=tmp_path / "output",
        )


def test_deployment_reauthenticates_warm_contract(monkeypatch, tmp_path):
    roots = _fixtures(tmp_path)
    _patch_decoder(monkeypatch)
    output = tmp_path / "output"
    handoff.build_handoff(
        main_root=roots["main"], deep_root=roots["deep"],
        normalized_root=roots["normalized"], nsga_code_root=tmp_path,
        output=output,
    )
    warm_path = output / "next_warm_start.npy"
    accepted = _deployment_contract_probe(warm_path)
    assert accepted.returncode == 0, accepted.stderr

    contract_path = output / "resonance_focus_warm_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(contract)
    tampered["category_counts"]["all_except_resonance"] = 31
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm handoff contract mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["selected_provenance"][0]["source_role"] = "main"
    tampered["selected_provenance_sha256"] = handoff._json_sha(
        tampered["selected_provenance"]
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm provenance mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["selected_provenance"][1]["resonance_G_Hz"] = 1.0
    tampered["selected_provenance_sha256"] = handoff._json_sha(
        tampered["selected_provenance"]
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm provenance mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["source_evidence"]["deep"]["generation_identity"][
        "deployment_model_manifest_sha256"
    ] = "9" * 64
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm provenance mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["selected_provenance"][0]["result_sha256"] = "a" * 64
    tampered["selected_provenance_sha256"] = handoff._json_sha(
        tampered["selected_provenance"]
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm provenance mismatch" in rejected.stderr
