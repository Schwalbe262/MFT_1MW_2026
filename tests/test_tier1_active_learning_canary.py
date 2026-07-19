from __future__ import annotations

import pytest

from tools.tier1_active_learning_canary import (
    _assert_no_base_overlap,
    _next_candidate_ordinal,
    _select_measurement_candidate,
)
from tools.tier1_terminal_followup import TERMINAL_ANALYSIS_SCHEMA


def _analysis() -> dict:
    params = {"N1": 8, "cw1": 5.0}
    from tools.tier1_resonance_feedback import _json_sha

    return {
        "schema_version": TERMINAL_ANALYSIS_SCHEMA,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "fea_submission_approved": False,
        "active_learning_measurement_submission_performed": False,
        "production_validity_and_measurement_eligibility_are_separate": True,
        "first_active_learning_measurement_candidate": {
            "decoded_params": params,
            "decoded_params_sha256": _json_sha(params),
            "measurement_eligible": True,
            "measurement_geometry_passed": True,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
    }


def test_measurement_candidate_never_inherits_production_validity():
    selected = _select_measurement_candidate(_analysis())
    assert selected["measurement_eligible"] is True
    assert selected["production_eligible"] is False

    tampered = _analysis()
    tampered["first_active_learning_measurement_candidate"][
        "production_eligible"
    ] = True
    with pytest.raises(RuntimeError, match="measurement candidate is invalid"):
        _select_measurement_candidate(tampered)


def test_candidate_id_and_scheduler_identity_dedupe_fail_closed():
    base = [{
        "candidate_id": "res20-064",
        "candidate_digest": "a" * 64,
        "task_name": "old-task",
        "scheduler_dedupe_key": "old-dedupe",
    }]
    assert _next_candidate_ordinal(base) == 65
    fresh = {
        "candidate_id": "res20-065",
        "candidate_digest": "b" * 64,
        "task_name": "new-task",
        "scheduler_dedupe_key": "new-dedupe",
    }
    _assert_no_base_overlap(fresh, base)
    duplicate = dict(fresh, scheduler_dedupe_key="old-dedupe")
    with pytest.raises(RuntimeError, match="overlaps base scheduler_dedupe_key"):
        _assert_no_base_overlap(duplicate, base)
