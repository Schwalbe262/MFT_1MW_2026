from __future__ import annotations

from tools import mft_goal_primary_temp_aggressive_neighborhood as subject


def _row(index: int, *, ratio: float, core: float, secondary: float):
    return {
        "physical_geometry_sha256": f"{index:064x}",
        "mechanism_proxies": {
            "primary_dc_resistance_proxy_ratio": ratio,
            "core_effective_area_ratio": 0.95,
        },
        "secondary_winding_robust_max_C": secondary,
        "core_robust_max_C": core,
        "corrected_Crx_transfer_estimate_F": 0.5e-9,
        "fixed_lm2mh_replay": {"fmin_Hz": 16_000.0},
    }


def test_selection_prioritizes_physical_resistance_after_core_gate():
    rows = [
        _row(
            index,
            ratio=0.932 + index * 0.0001,
            core=119.0,
            secondary=104.0,
        )
        for index in range(20)
    ]
    rows.append(_row(100, ratio=0.90, core=120.1, secondary=90.0))
    rows.append(_row(101, ratio=0.941, core=110.0, secondary=90.0))

    selected = subject._select(rows)

    assert len(selected) == subject.COUNT == 12
    assert [row["physical_geometry_sha256"] for row in selected] == [
        f"{index:064x}" for index in range(12)
    ]
    assert all(row["core_robust_max_C"] <= 120.0 for row in selected)
    assert all(
        row["mechanism_proxies"][
            "primary_dc_resistance_proxy_ratio"
        ]
        <= 0.94
        for row in selected
    )


def test_selection_returns_empty_when_eight_core_safe_lanes_do_not_exist():
    rows = [
        _row(
            index,
            ratio=0.935,
            core=119.0 if index < 7 else 121.0,
            secondary=104.0,
        )
        for index in range(12)
    ]
    assert subject._select(rows) == []


def test_tool_has_no_apply_or_submit_entry_point():
    options = {
        option
        for action in subject._parser()._actions
        for option in action.option_strings
    }
    assert "--apply" not in options
    assert not hasattr(subject, "submit")
