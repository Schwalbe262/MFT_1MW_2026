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

from tools import tier1_thermal_crossover_warm_handoff as handoff


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _candidate(role: str, index: int) -> dict:
    params = {"fixture_role": role, "fixture_index": index}
    constraints = {
        name: -1.0 for name in handoff.sealed.EXPECTED_CONSTRAINT_NAMES
    }
    if role == "resfocus":
        for offset, name in enumerate(handoff.CORE_THERMAL_CONSTRAINTS):
            constraints[name] = 10.0 + 0.1 * index + offset
    else:
        constraints["Llt_robust_band"] = 0.1 + index / 1000.0
    return {
        "rank": index + 1,
        "terminal_population_index": index,
        "decoded_params": params,
        "decoded_params_sha256": handoff._json_sha(params),
        "constraint_G": constraints,
        "target_predictions": {"Llt_phys": 27.5},
        "target_conformal_half_width": {"Llt_phys": 0.4},
        "derived_resonance": {"f_res_min_screen_Hz": 15_500.0},
        "production_eligible": False,
        "fea_submission_approved": False,
        "eligible_for_submission": False,
    }


def _source(role: str, count: int = 48) -> dict:
    result_sha = hashlib.sha256(f"result:{role}".encode()).hexdigest()
    status_sha = hashlib.sha256(f"status:{role}".encode()).hexdigest()
    result = {
        "next_target_fea_batch_plan": {
            "candidates": [_candidate(role, index) for index in range(count)]
        }
    }
    reference = {
        "role": role,
        "cohort_id": f"fixture-{role}-cohort",
        "task_id": 1000 + handoff.SOURCE_ROLES.index(role),
        "seed": 2000 + handoff.SOURCE_ROLES.index(role),
        "seed_status_path": f"/{role}/seed_status.json",
        "seed_status_sha256": status_sha,
        "result_path": f"/{role}/result.json",
        "result_sha256": result_sha,
    }
    profile = None
    controller_sha = hashlib.sha256(f"controller:{role}".encode()).hexdigest()
    if role == "resfocus":
        controller_sha = handoff.HOTFIX_RESONANCE_CONTROLLER_SHA256
        profile = {
            "namespace": "resonance-focus250-p320-g600-warm64-v1",
            "variant": "resonance-focus-250hz-v1",
            "optimizer_resonance_scale_Hz": 250.0,
            "population": 320,
            "max_generations": 600,
            "inference_threads": 8,
            "rolling_target": 32,
            "seed_start": 1_907_192_000,
            "task_priority": -7,
        }
    evidence = {
        "role": role,
        "requested_root": f"/{role}",
        "containment_root": f"/{role}",
        "cohort_id": reference["cohort_id"],
        "bundle_manifest_sha256": "6" * 64,
        "controller_source_sha256": controller_sha,
        "search_profile": profile,
        "generation_identity": handoff.EXPECTED_GENERATION_IDENTITY,
        "terminal_result_count": 1,
        "terminal_result_sha256": [result_sha],
        "terminal_seed_status_sha256": [status_sha],
        "terminal_result_set_sha256": handoff._json_sha([result_sha]),
        "terminal_seed_status_set_sha256": handoff._json_sha([status_sha]),
    }
    return {"evidence": evidence, "results": [(result, reference)]}


def _patch_sources_and_decoder(monkeypatch):
    sources = {role: _source(role) for role in handoff.SOURCE_ROLES}
    monkeypatch.setattr(
        handoff,
        "authenticate_source",
        lambda _root, role: copy.deepcopy(sources[role]),
    )
    dims = tuple((f"x{index}", 0.0, 1.0) for index in range(25))
    monkeypatch.setattr(
        handoff, "_load_sobol_schema", lambda _root: (dims, 1, 10, {}),
    )

    def decode(params, *_args):
        offsets = {
            "main": 0.0,
            "deep": 0.2,
            "normalized": 0.4,
            "resfocus": 0.6,
        }
        base = offsets[params["fixture_role"]] + (
            params["fixture_index"] / 1000.0
        )
        return np.mod(base + np.arange(25) / 97.0, 1.0)

    monkeypatch.setattr(handoff, "decoded_to_unit", decode)


def _build(monkeypatch, tmp_path: Path, name="output") -> dict:
    _patch_sources_and_decoder(monkeypatch)
    return handoff.build_handoff(
        main_root=tmp_path / "main",
        deep_root=tmp_path / "deep",
        normalized_root=tmp_path / "normalized",
        resonance_focus_root=tmp_path / "resfocus",
        nsga_code_root=tmp_path,
        output=tmp_path / name,
    )


def _deployment_contract_probe(warm_path: Path, *, balanced4c: bool = False):
    entrypoint = (
        "tier1_thermal_balanced_crossover_slurm_rolling"
        if balanced4c
        else "tier1_thermal_crossover_slurm_rolling"
    )
    code = (
        "from pathlib import Path; "
        f"from tools import {entrypoint} as entry; "
        "print(entry.rolling.validate_thermal_crossover_warm_contract("
        f"Path({str(warm_path)!r})))"
    )
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=handoff.REPO,
        env=environment, capture_output=True, text=True, check=False,
    )


def test_builds_exact_authenticated_balanced_64x25_crossover(
    monkeypatch, tmp_path,
):
    first = _build(monkeypatch, tmp_path, "output-a")
    second = handoff.build_handoff(
        main_root=tmp_path / "main",
        deep_root=tmp_path / "deep",
        normalized_root=tmp_path / "normalized",
        resonance_focus_root=tmp_path / "resfocus",
        nsga_code_root=tmp_path,
        output=tmp_path / "output-b",
    )
    warm = np.load(tmp_path / "output-a" / "next_warm_start.npy")
    assert warm.shape == (64, 25)
    assert first["warm_start"]["sha256"] == second["warm_start"]["sha256"]
    assert first["category_counts"] == {
        "target_complete_branch": 32,
        "cool_core_diverse_branch": 32,
    }
    assert first["exact_target_complete_count"] == 32
    assert first["exact_cool_core_count"] == 32
    provenance = first["selected_provenance"]
    assert [row["category"] for row in provenance[::2]] == (
        ["target_complete_branch"] * 32
    )
    assert [row["category"] for row in provenance[1::2]] == (
        ["cool_core_diverse_branch"] * 32
    )
    assert all(row["target_complete"] for row in provenance[::2])
    assert all(row["cool_core_complete"] for row in provenance[1::2])
    assert {row["source_role"] for row in provenance} == set(
        handoff.SOURCE_ROLES
    )
    assert {row["source_role"] for row in provenance[1::2]} == {
        "main", "deep", "normalized",
    }
    assert len({row["decoded_params_sha256"] for row in provenance}) == 64
    assert first["authoritative_terminal_G"] == "physical_unscaled"
    assert first["optimizer_core_thermal_scale_C"] == 1.0
    assert first["optimizer_resonance_scale_Hz"] == 250.0
    assert first["acquisition_ranking_contract"]["active"] is True
    assert first["acquisition_ranking_contract"][
        "hard_constraint_mutation"
    ] is False
    assert first["acquisition_ranking_contract"][
        "soft_axis_targets_mm"
    ] == {"exterior_width": 1_000.0, "exterior_length": 1_000.0}
    assert first["scheduler_write_performed"] is False
    assert first["fea_submission_approved"] is False
    assert first["aedt_used"] is False
    assert first["automatic_promotion_allowed"] is False


def test_resfocus_hotfix_identity_drift_is_rejected(monkeypatch, tmp_path):
    source = _source("resfocus")
    source["evidence"]["controller_source_sha256"] = "0" * 64
    monkeypatch.setattr(
        handoff.sealed, "authenticate_source", lambda *_args: source,
    )
    with pytest.raises(RuntimeError, match="zero-score hotfix cohort"):
        handoff.authenticate_source(tmp_path, "resfocus")


def test_target_complete_shortage_fails_closed(monkeypatch, tmp_path):
    sources = {role: _source(role) for role in handoff.SOURCE_ROLES}
    candidates = sources["resfocus"]["results"][0][0][
        "next_target_fea_batch_plan"
    ]["candidates"]
    for candidate in candidates[23:]:
        candidate["constraint_G"]["Llt_robust_band"] = 1.0
    monkeypatch.setattr(
        handoff,
        "authenticate_source",
        lambda _root, role: copy.deepcopy(sources[role]),
    )
    dims = tuple((f"x{index}", 0.0, 1.0) for index in range(25))
    monkeypatch.setattr(
        handoff, "_load_sobol_schema", lambda _root: (dims, 1, 10, {}),
    )
    monkeypatch.setattr(
        handoff,
        "decoded_to_unit",
        lambda params, *_args: np.mod(
            params["fixture_index"] / 1000.0 + np.arange(25) / 97.0,
            1.0,
        ),
    )
    with pytest.raises(RuntimeError, match="exact target-complete.*shortage"):
        handoff.build_handoff(
            main_root=tmp_path / "main",
            deep_root=tmp_path / "deep",
            normalized_root=tmp_path / "normalized",
            resonance_focus_root=tmp_path / "resfocus",
            nsga_code_root=tmp_path,
            output=tmp_path / "output",
        )


def test_deployment_reauthenticates_contract_and_rejects_tampering(
    monkeypatch, tmp_path,
):
    _build(monkeypatch, tmp_path)
    output = tmp_path / "output"
    warm_path = output / "next_warm_start.npy"
    contract_path = output / handoff.CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    accepted = _deployment_contract_probe(warm_path)
    assert accepted.returncode == 0, accepted.stderr
    balanced_accepted = _deployment_contract_probe(
        warm_path,
        balanced4c=True,
    )
    assert balanced_accepted.returncode == 0, balanced_accepted.stderr

    tampered = copy.deepcopy(contract)
    tampered["optimizer_core_thermal_scale_C"] = 4.0
    tampered["acquisition_ranking_contract"] = (
        handoff.thermal_crossover_acquisition_contract(
            handoff.HARD_SPEC,
            4.0,
        )
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path, balanced4c=True)
    assert rejected.returncode != 0
    assert "warm handoff contract mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["optimizer_core_thermal_constraints"].append(
        "temperature_robust_limit:T_max_Tx"
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm handoff contract mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["selected_provenance"][1]["cool_core_complete"] = False
    tampered["selected_provenance_sha256"] = handoff._json_sha(
        tampered["selected_provenance"]
    )
    _write_json(contract_path, tampered)
    rejected = _deployment_contract_probe(warm_path)
    assert rejected.returncode != 0
    assert "warm provenance mismatch" in rejected.stderr

    tampered = copy.deepcopy(contract)
    tampered["source_evidence"]["resfocus"][
        "controller_source_sha256"
    ] = "0" * 64
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
