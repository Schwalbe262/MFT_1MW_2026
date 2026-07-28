from regression_260707.monitoring.app import nsga_progress_summary
from regression_260707.monitoring.readers import (
    CURRENT7_HARD_SPEC,
    CURRENT7_TEMPERATURE_TARGETS,
)


def _current7_payload() -> dict[str, object]:
    search = {
        "configured": True,
        "available": True,
        "integrity_verified": True,
        "healthy": True,
        "pointer_verified": True,
        "constraint_contract_verified": True,
        "authority_eligible": True,
        "freshness_required": True,
        "freshness_ok": True,
        "freshness_age_seconds": 4.0,
        "harvest_observed_at": "2026-07-20 06:40:04",
        "status_event_at": "2026-07-20T06:40:00+00:00",
        "source": "C:/runtime/current7-index.json",
        "cohort_id": "current7-test",
        "bundle_id": "current7-bundle",
        "bundle_manifest_sha256": "a" * 64,
        "index_file_sha256": "b" * 64,
        "snapshot_file_sha256": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "constraint_identity_sha256": "e" * 64,
        "hard_spec_sha256": "f" * 64,
        "hard_constraint_contract_sha256": "1" * 64,
        "temperature_contract_sha256": "2" * 64,
        "constraint_version": (
            "1200x1200x750-res15k-t110-core4-cw1-5-lmhalf"
        ),
        "constraints": dict(CURRENT7_HARD_SPEC),
        "constraint_names": ["Llt_robust_band"],
        "temperature_targets": list(CURRENT7_TEMPERATURE_TARGETS),
        "search_count": 36,
        "running_count": 36,
        "queued_count": 0,
        "attaching_count": 0,
        "active_plus_queued": 36,
        "completed_count": 0,
        "failed_count": 0,
        "cancelled_count": 0,
        "timeout_count": 0,
        "authenticated_terminal_seed_count": 0,
        "refused_terminal_count": 0,
        "feasible_pareto_count": 0,
        "candidate_preview_count": 0,
        "candidate_preview_limit": 256,
        "candidate_preview_truncated": False,
        "near_feasible_count": 0,
        "state_counts": {"running": 36},
        "snapshot_attempts": 1,
        "updated_at": "2026-07-20T06:40:00+00:00",
        "warnings": [],
    }
    return {
        "available": True,
        "status": "running",
        "source_kind": "continuous_nsga_with_current7",
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "search_authority": {
            "configured": True,
            "kind": "current7",
            "available": True,
            "integrity_verified": True,
            "legacy_role": "archive",
        },
        "tier1_current7_search": search,
        "tier1_feedback_search": {
            "available": True,
            "integrity_verified": False,
            "stale": True,
            "archive_state": "stale",
            "updated_at": "2026-07-19T08:45:46+00:00",
        },
    }


def test_current7_progress_is_authoritative_while_legacy_is_stale_archive():
    progress = nsga_progress_summary(_current7_payload())

    assert progress["available"] is True
    assert progress["integrity_verified"] is True
    assert progress["authority_kind"] == "current7"
    assert progress["authority_identity_sha256"] == "a" * 64
    assert progress["authority_index_file_sha256"] == "b" * 64
    assert progress["authority_snapshot_sha256"] == "c" * 64
    assert progress["authority_snapshot_identity_sha256"] == "d" * 64
    assert progress["constraint_identity_sha256"] == "e" * 64
    assert progress["harvest_observed_at"] == "2026-07-20 06:40:04"
    assert progress["status_event_at"] == "2026-07-20T06:40:00+00:00"
    assert progress["coherent_snapshot"] is True
    assert progress["search_count"] == 36
    assert progress["terminal_results_verified"] == 0
    assert progress["temperature_constraint_contract"]["target_count"] == 7
    assert progress["legacy_archive"]["archive_state"] == "stale"


def test_configured_bad_current7_fails_closed_without_legacy_fallback():
    payload = _current7_payload()
    payload["available"] = False
    current7 = payload["tier1_current7_search"]
    assert isinstance(current7, dict)
    current7["integrity_verified"] = False
    authority = payload["search_authority"]
    assert isinstance(authority, dict)
    authority["available"] = False
    authority["integrity_verified"] = False

    progress = nsga_progress_summary(payload)

    assert progress["available"] is False
    assert progress["integrity_verified"] is False
    assert progress["authority_kind"] == "current7"
    assert progress["legacy_archive"]["available"] is True
    assert "failed closed" in progress["error"]


def _condition_archive_payload() -> dict[str, object]:
    searches = []
    for position in range(4):
        constraints = dict(CURRENT7_HARD_SPEC)
        constraints.update({
            "size_W_max_mm": 1_200.0 - (position * 50.0),
            "size_L_max_mm": 1_200.0 - (position * 50.0),
            "T_limit_C": 125.0 - (position * 7.5),
        })
        searches.append({
            "configured": True,
            "available": True,
            "integrity_verified": True,
            "healthy": False,
            "pointer_verified": True,
            "constraint_contract_verified": True,
            "authority_eligible": False,
            "display_only": True,
            "read_only": True,
            "gui_launch_eligible": False,
            "freshness_required": False,
            "freshness_ok": True,
            "source": f"C:/runtime/condition-{position}/current7-index.json",
            "cohort_id": f"condition-{position}",
            "bundle_manifest_sha256": f"{position + 3:x}" * 64,
            "hard_spec_sha256": f"{position + 7:x}" * 64,
            "constraint_version": f"condition-{position}",
            "constraints": constraints,
            "temperature_targets": list(CURRENT7_TEMPERATURE_TARGETS),
            "search_count": 3,
            "running_count": 0,
            "queued_count": 0,
            "attaching_count": 0,
            "active_plus_queued": 0,
            "completed_count": 2,
            "failed_count": 1,
            "cancelled_count": 0,
            "timeout_count": 0,
            "authenticated_terminal_seed_count": 1,
            "refused_terminal_count": 1,
            "feasible_pareto_count": 0,
            "candidate_preview_count": 0,
            "candidate_preview_limit": 256,
            "candidate_preview_truncated": False,
            "near_feasible_count": 1,
            "state_counts": {"completed": 2, "failed": 1},
            "snapshot_attempts": 1,
            "updated_at": f"2026-07-23T00:0{position}:00+00:00",
            "warnings": ["authenticated refusal retained for display"],
            "lanes": [],
        })
    return {
        "available": True,
        "status": "completed",
        "source_kind": "current7_condition_archive",
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "search_authority": {
            "configured": False,
            "kind": "condition_archive",
            "available": False,
            "integrity_verified": False,
            "legacy_role": "archive",
        },
        "tier1_feedback_search": {
            "available": True,
            "integrity_verified": False,
            "stale": True,
            "authority_role": "archive",
            "archived": True,
            "archive_state": "stale",
        },
        "tier1_current7_search": {
            "configured": False,
            "source": None,
        },
        "tier1_current7_condition_searches": searches,
        "near_feasible_preview": [{
            "id": "near-final",
            "valid_pareto": False,
            "production_eligible": False,
            "fea_submission_approved": False,
            "gui_launch_eligible": False,
        }],
        "near_feasible_preview_count": 1,
    }


def test_condition_archive_progress_aggregates_four_read_only_generations():
    payload = _condition_archive_payload()
    searches = payload["tier1_current7_condition_searches"]
    assert isinstance(searches, list)
    payload["tier1_current7_condition_searches"] = list(reversed(searches))

    progress = nsga_progress_summary(payload)

    assert progress["available"] is True
    assert progress["integrity_verified"] is True
    assert progress["source_kind"] == "current7_condition_archive"
    assert progress["authority_kind"] == "none"
    assert progress["condition_count"] == 4
    assert progress["constraint_version"] == "condition-3"
    assert progress["search_count"] == 12
    assert progress["completed_count"] == 8
    assert progress["failed_count"] == 4
    assert progress["terminal_results_verified"] == 4
    assert progress["refused_terminal_count"] == 4
    assert progress["near_feasible_count"] == 4
    assert progress["near_feasible_preview_count"] == 1
    assert progress["near_feasible_preview_truncated"] is True
    assert progress["candidate_count"] == 0
    assert progress["counters_consistent"] is True
    assert progress["all_configured_sources_verified"] is True
    assert progress["error"] is None


def test_condition_archive_progress_fails_closed_on_non_display_source():
    payload = _condition_archive_payload()
    searches = payload["tier1_current7_condition_searches"]
    assert isinstance(searches, list)
    assert isinstance(searches[2], dict)
    searches[2]["display_only"] = False

    progress = nsga_progress_summary(payload)

    assert progress["available"] is False
    assert progress["integrity_verified"] is False
    assert progress["rejected_source_count"] == 1
    assert "failed integrity validation" in progress["error"]


def test_condition_archive_progress_rejects_legacy_authority_boundary():
    payload = _condition_archive_payload()
    authority = payload["search_authority"]
    assert isinstance(authority, dict)
    authority.update({
        "kind": "legacy",
        "available": True,
        "integrity_verified": True,
        "source": "legacy.json",
        "legacy_role": "primary",
    })

    progress = nsga_progress_summary(payload)

    assert progress["available"] is False
    assert progress["integrity_verified"] is False
    assert "failed integrity validation" in progress["error"]
