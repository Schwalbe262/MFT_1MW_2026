from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "mft_goal_old_g0_parallel_capacity_no_submit_20260725.json"
)


def _load() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def _payload_sha256(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_audit_is_canonically_sealed_and_post_zero() -> None:
    value = _load()
    assert value["payload_sha256"] == _payload_sha256(value)
    decision = value["decision"]
    assert decision["action"] == "do_not_submit_additional_old_g0_nsga"
    assert decision["allowed"] is False
    assert decision["additional_seed_range"] is None
    assert decision["new_bundle_prepared"] is False
    assert decision["new_aggregate_prepared"] is False
    assert decision["scheduler_http_methods_used"] == ["GET"]
    assert decision["scheduler_get_calls_performed"] is True
    assert decision["scheduler_get_call_count_claimed"] is False
    assert decision["scheduler_post_calls"] == 0
    assert decision["scheduler_patch_calls"] == 0
    assert decision["scheduler_delete_calls"] == 0
    assert decision["scheduler_cancel_calls"] == 0


def test_reserved_active_learning_seeds_and_generations_are_isolated() -> None:
    isolation = _load()["seed_and_generation_isolation"]
    assert isolation["existing_old_g0_seed_interval"] == {
        "start": 2607262000,
        "end": 2607262511,
        "count": 512,
    }
    assert isolation["reserved_active_learning_seed_interval"] == {
        "start": 2607263000,
        "end": 2607263511,
        "count": 512,
        "used_by_this_audit": False,
        "reservation_preserved": True,
    }
    assert isolation["new_old_g0_seed_interval"] is None
    assert isolation["old_g0_and_new_generation_results_mixed"] is False
    assert isolation["existing_old_g0_aggregate_modified"] is False


def test_no_go_is_grounded_in_completed_science_and_live_capacity() -> None:
    value = _load()
    science = value["old_g0_science"]
    assert science["seed_count"] == 512
    assert science["terminal_row_count"] == 163_840
    assert science["deduplicated_physical_geometry_count"] == 133_563
    assert science["hard_feasible_count"] == 0
    assert science["production_pareto_count"] == 0
    assert science["quality_gate_passed"] is False
    assert science["quality_failed_target_count"] == 15
    assert science["search_only_proposal"] is True
    assert science["production_eligible"] is False
    assert science["automatic_promotion_allowed"] is False
    assert science["standard_candidate_count"] == 12
    assert science["standard_candidate_n1_counts"] == {
        "5": 0,
        "6": 12,
        "7": 0,
        "8": 0,
    }
    blockers = science["standard_candidate_positive_constraint_frequency"]
    assert blockers["Llt_robust_band"] == 12
    assert blockers["Llt_ensemble_disagreement"] == 12

    live = value["live_scheduler_snapshot"]
    assert live["active_standard_fea_task_count"] == 4
    assert live["active_standard_fea_requested_cpus"] == 32
    assert live["old_g0_standard_capacity"]["ready_fit_slots"] == 0
    assert live["full_fast_lane_capacity"]["ready_fit_slots"] == 0
    assert live["full_fast_lane_capacity"]["request"] == {
        "cpus": 16,
        "memory_mib": 98304,
        "timeout_seconds": 43200,
        "scheduling_profile": "fea",
        "aedt_backend": "standalone",
        "priority": 100,
        "full_model": 1,
        "thermal_symmetry": "full",
    }


def test_fixed_physics_and_scheduler_project_are_untouched() -> None:
    fixed = _load()["fixed_contract"]
    assert fixed["size_limits_mm"] == {"W": 1200.0, "L": 1000.0, "H": 750.0}
    assert fixed["resonance_minimum_hz"] == 15000.0
    assert fixed["temperature_limits_c"] == {
        "winding": 100.0,
        "core": 120.0,
    }
    assert fixed["fan_velocity_m_s"] == 1.5
    assert fixed["tim_and_insulation_conductivity_w_mk"] == 0.2
    assert fixed["wcp_pad_thickness_mm"] == 2.0
    assert fixed["core_plate_pad_thickness_mm"] == 2.0
    assert fixed["physics_modified_by_this_audit"] is False
    assert fixed["scheduler_project_modified_by_this_audit"] is False
