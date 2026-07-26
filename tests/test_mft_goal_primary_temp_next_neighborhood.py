from __future__ import annotations

import copy

import pytest

from tools import mft_goal_primary_temp_next_neighborhood as subject


def _fixed_decoded() -> dict[str, object]:
    return {
        "N1": 6,
        "N2": 60,
        "N1_main": 6,
        "N1_side": 0,
        "cw1": 5.0,
        "gap1": 1.6,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
        "core_center_gap_mm": 0.0,
    }


def _screened_rows() -> list[dict[str, object]]:
    rows = []
    index = 0
    for (
        structural_family,
        l1_delta,
        _l2_delta,
        _width_growth,
    ) in subject.STRUCTURAL_STENCIL:
        for plate_family, _plate_t, _w1_reduction, _contact in (
            subject.PLATE_PATH_STENCIL
        ):
            for height_delta in subject.PRIMARY_HEIGHT_DELTAS_MM:
                index += 1
                rows.append(
                    {
                        "physical_geometry_sha256": f"{index:064x}",
                        "mutations": {
                            "structural_family": structural_family,
                            "plate_family": plate_family,
                            "l1_core_section_delta_mm": l1_delta,
                            "nwh1_conductor_height_delta_mm": height_delta,
                        },
                        "fixed_lm2mh_replay": {"fmin_Hz": 16_000.0},
                        "corrected_Crx_transfer_estimate_F": 0.5e-9,
                        "primary_improvement_C": 1.0 + 0.01 * index,
                        "primary_winding_robust_max_C": 102.0 - 0.01 * index,
                        "secondary_winding_robust_max_C": 104.0,
                        "core_robust_max_C": 119.0 + 0.001 * index,
                        "P_total_screen_W": 8_000.0 + index,
                    }
                )
    return rows


def test_fixed_controls_reject_any_5t_boundary_escape():
    subject._fixed_controls(_fixed_decoded())
    escaped = copy.deepcopy(_fixed_decoded())
    escaped["cw1"] = 5.1
    with pytest.raises(subject.NeighborhoodError, match="cw1"):
        subject._fixed_controls(escaped)


def test_selection_is_twelve_unique_diversified_dry_run_candidates():
    selected = subject._select(_screened_rows())
    identities = {
        str(row["physical_geometry_sha256"]) for row in selected
    }
    roles = [str(row["selection_role"]) for row in selected]

    assert len(selected) == subject.COUNT == 12
    assert len(identities) == 12
    assert roles.count("balanced_width_for_shorter_y_path") == 3
    assert roles.count("shorter_y_path") == 3
    assert roles.count("maximum_short_y_path") == 3
    assert roles.count("plate_spreading_contact_hedge") == 3


def test_tool_has_no_apply_or_submit_entry_point():
    options = {
        option
        for action in subject._parser()._actions
        for option in action.option_strings
    }
    assert "--apply" not in options
    assert not hasattr(subject, "submit")
