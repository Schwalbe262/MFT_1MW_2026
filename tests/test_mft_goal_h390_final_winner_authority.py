import json
from pathlib import Path

import pytest

from tools import mft_goal_h390_final_winner_authority as authority


CANDIDATE_SHA = "8b284925997689369e14b71bb5d19625c5d99a2494cfdd175a03c0255268ea79"


def _write_sealed(path: Path, payload: dict) -> Path:
    value = dict(payload)
    value["payload_sha256"] = authority._sha(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    fea = {
        "schema_version": authority.FEA_SCHEMA,
        "complete": True,
        "ranked_records": [
            {
                "task_id": 98019,
                "mode": "matrix_turngraded_cap_loss_thermal",
                "exact_gap_task": True,
                "candidate_sha256": CANDIDATE_SHA,
                "contract_valid": True,
                "geometry_pass": True,
                "exact_Lm_pass": True,
                "thermal_pass": True,
                "W_mm": 1055.8,
                "L_mm": 888.4,
                "H_mm": 650.0,
                "N1": 6,
                "N2": 60,
                "N2_main": 35,
                "N2_side": 25,
                "gap_mm": 0.4448,
                "Lm_primary_full_mH": 1.99859,
                "Lm_primary_relative_error": 0.000705,
                "T_max_Tx_C": 99.0,
                "T_max_Rx_C": 105.0,
                "T_max_core_C": 121.0,
                "Tx_loss_W": 2319.9,
                "Rx_loss_W": 758.0,
                "core_loss_W": 2306.6,
                "result_json_sha256": "1" * 64,
                "stdout_sha256": "2" * 64,
            }
        ],
    }
    pair = {
        "schema_version": authority.PAIR_SCHEMA,
        "candidate_physics_sha256": CANDIDATE_SHA,
        "physical_identity_sha256": "3" * 64,
        "all_contracts_passed": True,
        "pair_resonance_pass_15kHz": True,
        "minimum_resonance_Hz": 15122.253,
        "tx": {
            "terminal_capacitance_F": 7.455392e-9,
            "self_inductance_H": 0.002,
            "resonance_Hz": 41216.384,
            "task_id": 98021,
        },
        "rx": {
            "terminal_capacitance_F": 5.53831e-10,
            "self_inductance_H": 0.2,
            "resonance_Hz": 15122.253,
            "task_id": 98020,
        },
    }
    params = tmp_path / "params.json"
    params.write_text('{"core_center_gap_mm":0.4448}', encoding="utf-8")
    return (
        _write_sealed(tmp_path / "fea.json", fea),
        _write_sealed(tmp_path / "pair.json", pair),
        params,
    )


def test_separates_original_and_relaxed_thermal_limits(tmp_path):
    fea, pair, params = _sources(tmp_path)
    result = authority.build_authority(
        fea_collection_path=fea,
        electrical_pair_path=pair,
        params_path=params,
    )
    assert result["common_pass"] is True
    assert result["final_pass_original_strict"] is False
    assert result["final_pass_latest_relaxed"] is True
    strict = result["thermal_verdicts"][
        "original_strict_Tx100_Rx100_core120"
    ]
    assert strict["component_pass"] == {
        "Tx_C": True,
        "Rx_C": False,
        "core_C": False,
    }
    assert result["electrical_high_accuracy"]["limiting_winding"] == "Rx"


def test_rejects_electrical_thermal_candidate_mismatch(tmp_path):
    fea, pair, params = _sources(tmp_path)
    pair_value = json.loads(pair.read_text(encoding="utf-8"))
    pair_value.pop("payload_sha256")
    pair_value["candidate_physics_sha256"] = "f" * 64
    _write_sealed(pair, pair_value)
    with pytest.raises(authority.AuthorityError, match="identity mismatch"):
        authority.build_authority(
            fea_collection_path=fea,
            electrical_pair_path=pair,
            params_path=params,
        )
