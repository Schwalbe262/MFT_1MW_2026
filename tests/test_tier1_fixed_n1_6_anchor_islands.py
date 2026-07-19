from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import tier1_fixed_n1_6_anchor_islands_warm_handoff as islands


def _constraints() -> dict[str, float]:
    return {
        name: -1.0 for name in islands.sealed.EXPECTED_CONSTRAINT_NAMES
    }


def _row(
    index: int, *, role: str = "normalized", resonance: float = 2_100.0,
    llt: float = -0.01, thermal: float = -0.1,
    support: float = -1.0, digest_params: dict | None = None,
) -> dict:
    params = digest_params or {"N1": 6, "index": index}
    values = _constraints()
    values[islands.RESONANCE_CONSTRAINT] = resonance
    values[islands.LLT_CONSTRAINT] = llt
    for name in islands.THERMAL_CONSTRAINTS:
        values[name] = thermal
    values[islands.SUPPORT_CONSTRAINTS[0]] = support
    return {
        "role": role,
        "decoded_params": params,
        "decoded_params_sha256": islands._json_sha(params),
        "constraint_G": values,
        "coordinate": np.asarray([
            ((index * (position + 3)) % 997) / 997.0
            for position in range(islands.WARM_SHAPE[1])
        ], dtype=float),
        "reference": {
            "cohort_id": "cohort",
            "task_id": 10_000 + index,
            "seed": 20_000 + index,
            "result_sha256": f"{30_000 + index:064x}",
            "seed_status_sha256": f"{40_000 + index:064x}",
        },
    }


def _classify(rows: list[dict], monkeypatch) -> list[dict]:
    monkeypatch.setattr(
        islands.sealed, "_candidate_rows", lambda _source: rows,
    )
    return islands._prepare_rows({})


def test_anchor_contract_is_exactly_4_plus_five_times_12_and_fail_closed():
    assert islands.SOURCE_ROLES == (
        "main", "deep", "normalized", "resfocus", "thermal1",
        "thermal4", "fixed5", "fixed6", "bridge6",
    )
    assert set(islands.ROLE_NAMESPACES) == set(islands.SOURCE_ROLES)
    assert set(islands.ROLE_CONTROLLER_SHA256) == set(islands.SOURCE_ROLES)
    assert set(islands.ROLE_PROFILE_SHA256) == set(islands.SOURCE_ROLES)
    assert islands.RESONANCE_ANCHOR_QUOTA == 4
    assert islands.STAIRCASE_CAPS_HZ == (2_000.0, 1_500.0, 1_000.0, 500.0, 0.0)
    assert islands.STAIRCASE_QUOTA_PER_CAP == 12
    assert (
        islands.RESONANCE_ANCHOR_QUOTA
        + len(islands.STAIRCASE_CAPS_HZ)
        * islands.STAIRCASE_QUOTA_PER_CAP
        == islands.WARM_SHAPE[0]
    )
    assert islands.DECODED_SHA_DUPLICATE_CAP == 1
    assert islands.HARD_SPEC == {
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 110.0,
        "B_limit_T": 1.2,
        "insulation_min_mm": 40.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "resonance_min_Hz": 15_000.0,
        "magnetizing_inductance_factor": 0.5,
        "size_W_max_mm": 1_200.0,
        "size_L_max_mm": 1_200.0,
        "size_H_max_mm": 750.0,
        "q_sigma": 1.0,
        "uncertainty_contract": "q90_conformal_half_width_physical_v1",
    }


def test_physical_anchor_eligibility_preserves_support_and_exact_thresholds(
    monkeypatch,
):
    rows = [
        _row(0, role="fixed6", resonance=150.0, llt=0.05, thermal=9.0),
        _row(1, role="fixed6", resonance=150.0001, llt=0.05, thermal=9.0),
        _row(2, role="fixed6", resonance=0.0, llt=0.0501, thermal=9.0),
        _row(3, resonance=2_100.0, llt=0.0, thermal=0.0),
        _row(4, resonance=2_100.0, llt=0.0, thermal=0.001),
        _row(5, resonance=2_100.0, llt=0.0, thermal=0.0, support=0.001),
    ]
    classified = _classify(rows, monkeypatch)
    assert classified[0]["resonance_llt_anchor_eligible"] is True
    assert classified[1]["resonance_llt_anchor_eligible"] is False
    assert classified[2]["resonance_llt_anchor_eligible"] is False
    assert classified[3]["thermal_llt_anchor_eligible"] is True
    assert classified[4]["thermal_llt_anchor_eligible"] is False
    assert classified[5]["thermal_llt_anchor_eligible"] is False
    assert classified[0]["thermal_max_G_C"] == 9.0
    assert classified[0]["thermal_positive_sum_C"] == pytest.approx(
        9.0 * len(islands.THERMAL_CONSTRAINTS)
    )


def test_selection_enforces_global_decoded_sha_cap_and_staircase_quotas(
    monkeypatch,
):
    resonance = _classify([
        _row(
            index, role="fixed6", resonance=-float(index), llt=0.0,
            thermal=10.0 + index,
        )
        for index in range(4)
    ], monkeypatch)
    thermal_rows = _classify([
        _row(
            100 + index,
            role=("normalized", "thermal1", "resfocus")[index % 3],
            resonance=2_050.0 + index,
            llt=-0.01,
            thermal=-0.1,
        )
        for index in range(90)
    ], monkeypatch)
    # The same decoded SHA with a worse reference must never consume quota.
    duplicate = dict(thermal_rows[0])
    duplicate["reference"] = {
        **duplicate["reference"],
        "seed": 999_999,
        "result_sha256": "f" * 64,
    }
    thermal_rows.append(duplicate)

    a_selected = islands._select_diverse(
        resonance,
        islands.RESONANCE_ANCHOR_QUOTA,
        rank_key=islands._resonance_anchor_rank,
        required_roles=("fixed6",),
    )
    seen = {row["decoded_params_sha256"] for row in a_selected}
    selected_by_cap = {}
    for index, cap in enumerate(islands.STAIRCASE_CAPS_HZ):
        selected_by_cap[cap] = islands._select_diverse(
            thermal_rows,
            islands.STAIRCASE_QUOTA_PER_CAP,
            rank_key=islands._staircase_rank(cap),
            excluded_sha=seen,
            required_roles=(
                ("normalized", "thermal1", "resfocus") if index == 0 else ()
            ),
        )
        seen.update(
            row["decoded_params_sha256"] for row in selected_by_cap[cap]
        )
    assert len(a_selected) == 4
    assert all(len(rows) == 12 for rows in selected_by_cap.values())
    assert len(seen) == 64
    assert {
        row["role"] for row in selected_by_cap[2_000.0]
    }.issuperset({"normalized", "thermal1", "resfocus"})


def test_authentication_pins_status_profile_and_controller_bytes(
    tmp_path: Path, monkeypatch,
):
    role = "deep"
    profile = {
        "namespace": islands.ROLE_NAMESPACES[role],
        "population": 320,
    }
    # Use a local exact digest to isolate behavior from the live profile value.
    monkeypatch.setitem(
        islands.ROLE_PROFILE_SHA256, role, islands._json_sha(profile),
    )
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"search_profile": profile}), encoding="utf-8")
    evidence = {
        "controller_source_sha256": islands.ROLE_CONTROLLER_SHA256[role],
        "status": {
            "path": str(status),
            "sha256": islands.sealed._sha_file(status),
        },
    }
    monkeypatch.setattr(
        islands.sealed, "authenticate_source",
        lambda _root, _role: {"evidence": evidence, "results": []},
    )
    source = islands._authenticate(tmp_path, role)
    assert source["evidence"]["status_search_profile"] == profile
    assert source["evidence"]["status_search_profile_sha256"] == islands._json_sha(
        profile
    )

    status.write_text(
        json.dumps({"search_profile": {**profile, "population": 321}}),
        encoding="utf-8",
    )
    evidence["status"]["sha256"] = islands.sealed._sha_file(status)
    with pytest.raises(RuntimeError, match="controller/search-profile seal"):
        islands._authenticate(tmp_path, role)


def test_capture_retries_when_authenticated_head_moves(
    tmp_path: Path, monkeypatch,
):
    calls = 0
    terminal = {
        "terminal_result_sha256": ["1" * 64],
        "terminal_seed_status_sha256": ["2" * 64],
        "terminal_snapshot_count": 1,
    }

    def write(path: Path, payload: dict) -> str:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return islands.sealed._sha_file(path)

    def fake_authenticate(_root: Path, role: str) -> dict:
        nonlocal calls
        calls += 1
        evidence = {
            "role": role,
            "terminal_result_sha256": terminal["terminal_result_sha256"],
            "terminal_seed_status_sha256": terminal[
                "terminal_seed_status_sha256"
            ],
            "terminal_result_set_sha256": islands._json_sha(
                terminal["terminal_result_sha256"]
            ),
            "terminal_seed_status_set_sha256": islands._json_sha(
                terminal["terminal_seed_status_sha256"]
            ),
        }
        for kind in islands.SOURCE_SNAPSHOT_KINDS:
            path = tmp_path / f"{kind}.json"
            digest = write(path, {"attempt": calls, "kind": kind})
            evidence[kind] = {"path": str(path), "sha256": digest}
        if calls == 1:
            write(tmp_path / "index.json", {"attempt": 99, "kind": "index"})
        return {"evidence": evidence, "results": []}

    monkeypatch.setattr(islands, "_authenticate", fake_authenticate)
    monkeypatch.setattr(
        islands.bridge,
        "_terminal_snapshot_evidence",
        lambda _status, _role: terminal,
    )
    captured = islands._capture_authenticated_source(tmp_path, "normalized")
    assert calls == 2
    assert json.loads(captured["_snapshot_payloads"]["index"])["attempt"] == 2
