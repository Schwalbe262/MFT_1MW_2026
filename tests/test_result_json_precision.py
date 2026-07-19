import json
import math

import pandas as pd

from run_simulation_260706 import (
    PANDAS_JSON_DOUBLE_PRECISION,
    _series_to_json,
)


CAPACITANCE_FIELDS = (
    "C_tx_tx_raw_F",
    "C_rx_rx_raw_F",
    "C_tx_rx_signed_raw_F",
    "C_tx_rx_raw_F",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_signed_F",
    "C_tx_rx_F",
)


def test_result_json_preserves_all_sub_nf_capacitance_fields():
    expected = {
        "C_tx_tx_raw_F": 1.856789012345678e-10,
        "C_rx_rx_raw_F": 8.721234567890123e-10,
        "C_tx_rx_signed_raw_F": -3.456789012345678e-10,
        "C_tx_rx_raw_F": 3.456789012345678e-10,
        "C_tx_tx_F": 7.654321098765432e-10,
        "C_rx_rx_F": 9.876543210123456e-10,
        "C_tx_rx_signed_F": -6.543210987654321e-10,
        "C_tx_rx_F": 6.543210987654321e-10,
    }

    payload = json.loads(_series_to_json(pd.Series(expected)))

    assert PANDAS_JSON_DOUBLE_PRECISION == 15
    assert set(CAPACITANCE_FIELDS).issubset(payload)
    for field in CAPACITANCE_FIELDS:
        # pandas JSON precision is decimal-place based.  At 15 places, values
        # in the operational sub-nF range are retained to <5e-16 F instead of
        # being rounded to 1e-10 F steps by pandas' default precision of 10.
        assert math.isclose(
            payload[field], expected[field], rel_tol=5e-6, abs_tol=5e-16
        ), field

    assert len({payload[field] for field in CAPACITANCE_FIELDS}) > 4
    assert payload["C_tx_rx_signed_raw_F"] < 0.0
    assert payload["C_tx_rx_signed_F"] < 0.0


def test_result_json_precision_does_not_regress_ordinary_metrics_or_types():
    expected = {
        "Llt": 27.51234567890123,
        "P_total": 5959.612345678901,
        "Tmax": 111.3123456789012,
        "N1": 5,
        "converged": True,
        "project_name": "mft-cap-precision",
    }

    payload = json.loads(_series_to_json(pd.Series(expected)))

    for field in ("Llt", "P_total", "Tmax"):
        assert math.isclose(payload[field], expected[field], rel_tol=1e-14)
    assert payload["N1"] == 5
    assert payload["converged"] is True
    assert payload["project_name"] == "mft-cap-precision"


def test_fallback_and_failed_sample_date_format_keeps_high_precision():
    timestamp = pd.Timestamp("2026-07-19T12:34:56.123456Z")
    row = pd.Series({
        "C_rx_rx_F": 2.721234567890123e-10,
        "created_at": timestamp,
    })

    payload = json.loads(_series_to_json(row, date_format="iso"))

    assert math.isclose(
        payload["C_rx_rx_F"], row["C_rx_rx_F"], rel_tol=5e-6, abs_tol=5e-16
    )
    assert payload["created_at"].startswith("2026-07-19T12:34:56.123")
