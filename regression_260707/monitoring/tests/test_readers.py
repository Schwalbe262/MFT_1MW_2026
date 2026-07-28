import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

import regression_260707.monitoring.readers as readers_module
from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707.model_targets import (
    CORE_REGION_TEMPERATURE_TARGETS,
    SURROGATE_TEMPERATURE_TARGETS,
)
from regression_260707.monitoring.readers import (
    ArtifactService,
    CURRENT_PHYSICS_DATA_REVISION,
    ReadResult,
    RefillControllerReader,
    RuntimeRecorder,
    SafeArtifactCache,
    SchedulerReader,
    SimulationPolicyConflict,
    TARGET_META,
    TEMPERATURE_TARGETS,
    _campaign_frame_summary,
    _display_text_has_mojibake,
    _normalized_display_text,
    _simulation_timing_summary,
    _zero_aware_percentage_metrics,
)


OLDER_SOLVER_REVISION = "a" * 40
NEWER_SOLVER_REVISION = "b" * 40


def _current7_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _publish_current7_runtime(
    root: Path, *, terminal: bool = True,
    legacy_empty_contract: bool = False,
    scaled: bool = False,
    updated_at: str | None = None,
    harvest_observed_at: str | None = None,
    provenance_drift: str | None = None,
    constraint_version_override: str | None = None,
    hard_spec_override: dict[str, object] | None = None,
    constraint_names_override: list[str] | None = None,
    candidate_overrides: dict[str, object] | None = None,
    include_authenticated_record: bool | None = None,
    refusal_overrides: list[dict[str, object]] | None = None,
    mixed_bundle_projection: bool = False,
) -> dict[str, object]:
    """Publish one synthetic secondary index using the harvester wire format."""
    canonical = root / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    legacy = canonical / "index.json"
    legacy.write_bytes(b"legacy-all11-sentinel\n")
    bundle_id = "current7-test-1234567890ab"
    bundle_manifest_sha = "a" * 64
    predecessor_bundle_id = "current7-predecessor-abcdef123456"
    predecessor_manifest_sha = "b" * 64
    authenticated_terminal = (
        terminal
        if include_authenticated_record is None
        else include_authenticated_record
    )
    effective_refusal_overrides = refusal_overrides or (
        [{}] if mixed_bundle_projection else []
    )
    refusals: list[dict[str, object]] = [
        {
            "task_id": 7002 if mixed_bundle_projection else 7001,
            "bundle_id": (
                predecessor_bundle_id if mixed_bundle_projection else bundle_id
            ),
            "seed": 102 if mixed_bundle_projection else 101,
            "reason": "RuntimeError:synthetic authenticated refusal",
            **overrides,
        }
        for overrides in effective_refusal_overrides
    ]
    constraint_version = constraint_version_override or (
        "1200x1200x750-res15k-t110-core4-cw1-5-lmhalf"
    )
    hard_spec = dict(
        hard_spec_override or readers_module.CURRENT7_HARD_SPEC
    )
    hard_sha = readers_module._canonical_json_sha256(hard_spec)
    temperatures = list(readers_module.CURRENT7_TEMPERATURE_TARGETS)
    constraints = list(
        constraint_names_override or readers_module.CURRENT7_CONSTRAINT_NAMES
    )
    publish_contract = terminal or not legacy_empty_contract
    identity_values = {
        "constraint_version": constraint_version if publish_contract else None,
        "hard_spec": hard_spec if publish_contract else None,
        "hard_spec_sha256": hard_sha if publish_contract else None,
        "hard_constraint_contract_sha256": (
            "c" * 64 if publish_contract else None
        ),
        "temperature_contract_sha256": (
            "d" * 64 if publish_contract else None
        ),
        "constraint_names": constraints if publish_contract else [],
        "temperature_targets": temperatures,
    }
    decoded = {
        "N1_main": 5,
        "N1_side": 0,
        "N2_main": 3,
        "N2_side": 0,
        "n_core_group": 3,
        "cw1": 5.0,
    }
    means = {
        "Llt_phys": 27.5,
        "B_mean_core": 0.82,
        "C_tx_tx_F": 1.2e-9,
        "C_rx_rx_F": 1.3e-9,
        "C_tx_rx_F": 0.8e-9,
        **{target: 98.0 for target in temperatures},
    }
    half_widths = {
        "Llt_phys": 0.1,
        "B_mean_core": 0.02,
        "C_tx_tx_F": 0.05e-9,
        "C_rx_rx_F": 0.05e-9,
        "C_tx_rx_F": 0.04e-9,
        **{target: 2.0 for target in temperatures},
    }
    candidate = {
        "candidate_id": "terminal-0000",
        "terminal_population_index": 0,
        "decoded_params": decoded,
        "size_W_mm": 1_050.0,
        "size_L_mm": 1_040.0,
        "size_H_mm": 700.0,
        "volume_L": 764.4,
        "total_loss_W": 5_700.0,
        "predicted_total_loss_W": 5_700.0,
        "predicted_winding_loss_W": 3_000.0,
        "predicted_core_loss_W": 2_000.0,
        "predicted_core_plate_loss_W": 400.0,
        "predicted_winding_cold_plate_loss_W": 300.0,
        "predicted_Tx_main_winding_loss_W": 1_700.0,
        "predicted_Rx_main_winding_loss_W": 1_000.0,
        "predicted_Rx_side_winding_loss_W": 300.0,
        "pred_Llt_phys": 27.5,
        "B_design_analytic_T": 1.0,
        "pred_B_mean_core": 0.82,
        "minimum_realized_insulation_mm": 45.0,
        "pred_f_res_tx_screen_Hz": 16_200.0,
        "pred_f_res_rx_screen_Hz": 16_100.0,
        "pred_f_res_min_screen_Hz": 16_100.0,
        "pred_f_res_interwinding_screen_Hz": 31_000.0,
        "surrogate_mean_predictions": means,
        "surrogate_q90_conformal_half_widths": half_widths,
        "physical_constraint_G": {name: -1.0 for name in constraints},
        "physical_feasible": True,
        "feasible": True,
        "total_positive_violation": 0.0,
        "source_bundle_id": (
            "wrong-bundle" if provenance_drift == "candidate" else bundle_id
        ),
        "source_seed": 101,
        "source_candidate_position": 0,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    candidate.update(candidate_overrides or {})
    candidate["candidate_identity_sha256"] = (
        readers_module._canonical_json_sha256(decoded)
    )
    aggregate = {
        "schema_version": readers_module.CURRENT7_AGGREGATE_SCHEMA,
        "authenticated_seed_count": 1 if authenticated_terminal else 0,
        "candidate_count": 1 if authenticated_terminal else 0,
        "feasible_candidate_count": 1 if authenticated_terminal else 0,
        "pareto_count": 1 if authenticated_terminal else 0,
        "pareto_candidates": [candidate] if authenticated_terminal else [],
        "pareto_preview_truncated": False,
        "least_violation_candidate": (
            candidate if authenticated_terminal else None
        ),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        **identity_values,
    }
    task = {
        "task_id": 7001,
        "name": "mft-current7-test-101",
        "seed": 101,
        "bundle_id": (
            "wrong-bundle" if provenance_drift == "task" else bundle_id
        ),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "status": (
            "unknown-state"
            if provenance_drift == "state"
            else ("completed" if terminal else "running")
        ),
    }
    tasks = [task]
    if mixed_bundle_projection:
        tasks.append({
            "task_id": 7002,
            "name": "mft-current7-predecessor-test-102",
            "seed": 102,
            "bundle_id": predecessor_bundle_id,
            "bundle_manifest_sha256": predecessor_manifest_sha,
            "status": "completed",
        })
    scale_policy_sha256 = "e" * 64 if scaled else None
    scale_summary = None
    if scaled:
        task["scale_policy_sha256"] = scale_policy_sha256
        scaled_resources = {
            "aedt_backend": "standalone",
            "aedt_used": False,
            "cpus_per_seed_task": 8,
            "max_workers_per_node": 8,
            "memory_mb_per_seed_task": 65_536,
            "priority": 0,
        }
        unsigned_scale_summary = {
            "schema_version": readers_module.CURRENT7_SCALE_SUMMARY_SCHEMA,
            "bundle_id": bundle_id,
            "policy_sha256": scale_policy_sha256,
            "effective_active_total": 80,
            "island_active_quotas": {
                "island-a": 53,
                "island-b": 27,
            },
            "max_workers_per_node": 8,
            "scaled_resource_contract": scaled_resources,
            "scaled_resource_contract_sha256": (
                readers_module._canonical_json_sha256(scaled_resources)
            ),
            "base_resource_contract_sha256": "f" * 64,
        }
        scale_summary = {
            **unsigned_scale_summary,
            "summary_sha256": readers_module._canonical_json_sha256(
                unsigned_scale_summary
            ),
        }
    records = []
    if authenticated_terminal:
        record = {
            "schema_version": (
                "mft-tier1-current7-content-addressed-seed-record-v1"
            ),
            "bundle_id": (
                "wrong-bundle"
                if provenance_drift == "terminal" else bundle_id
            ),
            "seed": 101,
            "authenticated": True,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
            **{
                key: identity_values[key]
                for key in (
                    "constraint_version",
                    "hard_spec_sha256",
                    "hard_constraint_contract_sha256",
                    "temperature_contract_sha256",
                )
            },
        }
        record["record_sha256"] = readers_module._canonical_json_sha256(record)
        records.append(record)
    snapshot_identity = {
        "schema_version": readers_module.CURRENT7_SNAPSHOT_IDENTITY_SCHEMA,
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": bundle_manifest_sha,
        "task_schema_version": "mft-tier1-current7-slurm-seed-task-v1",
        "result_schema_version": "mft-tier1-current7-search-seed-v1",
        "inventory_sha256": readers_module._canonical_json_sha256(tasks),
        "terminal_record_sha256": sorted(
            record["record_sha256"] for record in records
        ),
        "refusals_sha256": readers_module._canonical_json_sha256(refusals),
        "aggregate_sha256": readers_module._canonical_json_sha256(aggregate),
    }
    if scaled:
        snapshot_identity.update({
            "scale_policy_sha256": scale_policy_sha256,
            "scale_summary_sha256": scale_summary["summary_sha256"],
        })
    snapshot_sha = readers_module._canonical_json_sha256(snapshot_identity)
    cohort_id = f"current7-test-{snapshot_sha[:16]}"
    compatibility = {
        "schema_version": readers_module.CURRENT7_COMPATIBILITY_SCHEMA,
        "monitor_generation": "84ff69f-cap-parity-overlay-20260719",
        "compatible": False,
        "activation_allowed": False,
        "legacy_canonical_index_write_allowed": False,
        "reasons": ["requires-current7-reader"],
        "required_action": "deploy_new_immutable_8010_current7_reader",
    }
    published_at = updated_at or datetime.now(timezone.utc).replace(
        tzinfo=None
    ).strftime("%Y-%m-%d %H:%M:%S")
    projection_fields: dict[str, object] = {}
    if mixed_bundle_projection:
        source_bundle_cohorts = [
            {
                "bundle_id": predecessor_bundle_id,
                "bundle_manifest_sha256": predecessor_manifest_sha,
                "cohort_id": "predecessor-test",
                "launch_plan_sha256": "1" * 64,
                "manifest_contract_sha256": "2" * 64,
                "publication_receipt_sha256": "3" * 64,
                "ready_sha256": "4" * 64,
                "remote_bundle": f"/gpfs/test/{predecessor_bundle_id}",
                "resource_policy_ids": ["legacy-8c-test"],
                "role": "predecessor",
                "stage_spec_sha256": hard_sha,
            },
            {
                "bundle_id": bundle_id,
                "bundle_manifest_sha256": bundle_manifest_sha,
                "cohort_id": "successor-test",
                "launch_plan_sha256": "5" * 64,
                "manifest_contract_sha256": "6" * 64,
                "publication_receipt_sha256": "7" * 64,
                "ready_sha256": "8" * 64,
                "remote_bundle": f"/gpfs/test/{bundle_id}",
                "resource_policy_ids": ["successor-4c-test"],
                "role": "successor",
                "stage_spec_sha256": hard_sha,
            },
        ]
        projection_fields = {
            "projection_anchor_bundle_id": bundle_id,
            "source_bundle_cohorts": source_bundle_cohorts,
            "source_bundle_cohorts_sha256": (
                readers_module._canonical_json_sha256(source_bundle_cohorts)
            ),
            "mixed_bundle_projection": True,
            "mixed_resource_policy_projection": True,
        }
    state_counts: dict[str, int] = {}
    for item in tasks:
        item_status = str(item["status"])
        state_counts[item_status] = state_counts.get(item_status, 0) + 1
    status = {
        "schema_version": readers_module.CURRENT7_STATUS_SCHEMA,
        "cohort_id": cohort_id,
        "snapshot_identity": snapshot_identity,
        "snapshot_sha256": snapshot_sha,
        "updated_at": published_at,
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": bundle_manifest_sha,
        "task_schema_version": snapshot_identity["task_schema_version"],
        "result_schema_version": snapshot_identity["result_schema_version"],
        **identity_values,
        "scheduler_task_count": len(tasks),
        "state_counts": dict(sorted(state_counts.items())),
        "latest_tasks": tasks,
        "authenticated_terminal_seed_count": len(records),
        "refused_terminal_count": len(refusals),
        "refusals": refusals,
        "terminal_results": records,
        "aggregate": aggregate,
        "legacy_8010_compatibility": compatibility,
        "healthy": not refusals,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
        **projection_fields,
    }
    if scaled:
        status.update({
            "scale_policy_sha256": scale_policy_sha256,
            "scale_summary": scale_summary,
        })
    cohort = root / "cohorts" / cohort_id
    cohort.mkdir(parents=True, exist_ok=True)
    status_path = cohort / "status.json"
    compatibility_path = cohort / "compatibility.json"
    status_bytes = _current7_json_bytes(status)
    compatibility_bytes = _current7_json_bytes(compatibility)
    status_path.write_bytes(status_bytes)
    compatibility_path.write_bytes(compatibility_bytes)
    index = {
        "schema_version": readers_module.CURRENT7_INDEX_SCHEMA,
        "active_cohort_id": cohort_id,
        "snapshot_sha256": snapshot_sha,
        "updated_at": status["updated_at"],
        "status_event_at": status["updated_at"],
        "harvest_observed_at": (
            harvest_observed_at
            or datetime.now(timezone.utc).replace(tzinfo=None).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": bundle_manifest_sha,
        **identity_values,
        "path_containment_root": str(root.resolve()),
        "status": {
            "path": str(status_path.resolve()),
            "schema_version": readers_module.CURRENT7_STATUS_SCHEMA,
            "sha256": hashlib.sha256(status_bytes).hexdigest(),
        },
        "compatibility": {
            "path": str(compatibility_path.resolve()),
            "schema_version": readers_module.CURRENT7_COMPATIBILITY_SCHEMA,
            "sha256": hashlib.sha256(compatibility_bytes).hexdigest(),
            "legacy_84ff69f_compatible": False,
        },
        "legacy_canonical_index_path": str(legacy.resolve()),
        "legacy_canonical_index_touched": False,
        "activation_allowed": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
        **projection_fields,
    }
    if mixed_bundle_projection:
        index["condition_display_only"] = True
    if scaled:
        index.update({
            "scale_policy_sha256": scale_policy_sha256,
            "scale_summary": scale_summary,
        })
    index_path = canonical / "current7-index.json"
    index_path.write_bytes(_current7_json_bytes(index))
    return {
        "index_path": index_path,
        "status_path": status_path,
        "legacy_path": legacy,
        "anchor_bundle_id": bundle_id,
        "predecessor_bundle_id": predecessor_bundle_id,
    }


def _isolate_continuous_nsga_sources(monkeypatch, service: ArtifactService):
    monkeypatch.setattr(service, "_local_nsga_ui_status", lambda: ([], []))
    monkeypatch.setattr(
        service,
        "_slurm_nsga_offload_diagnostics",
        lambda: {"available": False, "warnings": []},
    )
    monkeypatch.setattr(
        service,
        "_combined_tier1_feedback_nsga_diagnostics",
        lambda: {
            "available": False,
            "integrity_verified": False,
            "candidate_rows": [],
            "near_candidate_rows": [],
            "lanes": [],
            "stale": True,
            "warnings": ["Tier-1 rolling status is stale"],
        },
    )
    monkeypatch.setattr(service.scheduler, "mft_pipeline_status", lambda: {})


def _reseal_current7_runtime(published: dict[str, object]) -> None:
    """Recompute every existing synthetic status/index seal after a mutation."""

    status_path = Path(published["status_path"])
    index_path = Path(published["index_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    identity = status["snapshot_identity"]
    identity["inventory_sha256"] = readers_module._canonical_json_sha256(
        status["latest_tasks"]
    )
    identity["terminal_record_sha256"] = sorted(
        record["record_sha256"] for record in status["terminal_results"]
    )
    identity["refusals_sha256"] = readers_module._canonical_json_sha256(
        status["refusals"]
    )
    identity["aggregate_sha256"] = readers_module._canonical_json_sha256(
        status["aggregate"]
    )
    snapshot_sha = readers_module._canonical_json_sha256(identity)
    status["snapshot_sha256"] = snapshot_sha
    index["snapshot_sha256"] = snapshot_sha
    status_bytes = _current7_json_bytes(status)
    status_path.write_bytes(status_bytes)
    index["status"]["sha256"] = hashlib.sha256(status_bytes).hexdigest()
    index_path.write_bytes(_current7_json_bytes(index))


def test_current7_legacy_status_limit_is_exactly_64_mib(tmp_path):
    limit = readers_module.CURRENT7_STATUS_MAX_BYTES
    assert limit == 64 * 1024 * 1024
    path = tmp_path / "status.json"
    prefix = b'{"padding":"'
    suffix = b'"}'

    path.write_bytes(prefix + b"x" * (limit - len(prefix) - len(suffix)) + suffix)
    value, raw = readers_module._bounded_json_bytes(path, limit)
    assert len(raw) == limit
    assert len(value["padding"]) == limit - len(prefix) - len(suffix)

    path.write_bytes(
        prefix + b"x" * (limit + 1 - len(prefix) - len(suffix)) + suffix
    )
    with pytest.raises(
        ValueError,
        match=rf"status\.json exceeds {limit} byte safety limit",
    ):
        readers_module._bounded_json_bytes(path, limit)


def test_current7_secondary_index_projects_exact_seven_temperature_pareto(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(tmp_path / "current7")
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    diagnostics = service._tier1_current7_diagnostics()
    assert diagnostics["integrity_verified"] is True
    assert diagnostics["healthy"] is True
    assert diagnostics["constraint_contract_verified"] is True
    assert diagnostics["authority_eligible"] is True
    assert len(diagnostics["constraint_identity_sha256"]) == 64
    assert len(diagnostics["index_file_sha256"]) == 64
    assert len(diagnostics["snapshot_file_sha256"]) == 64
    assert diagnostics["feasible_pareto_count"] == 1
    assert diagnostics["temperature_targets"] == list(
        readers_module.CURRENT7_TEMPERATURE_TARGETS
    )
    _isolate_continuous_nsga_sources(monkeypatch, service)
    monkeypatch.setattr(
        service.scheduler,
        "mft_pipeline_status",
        lambda: pytest.fail("current7 authority must not wait on legacy API"),
    )
    nsga = service._continuous_nsga2()
    assert nsga["status"] == "completed"
    assert nsga["selected_model_id"].startswith("tier1-current7:")
    assert nsga["candidate_count"] == 1
    assert len(nsga["pareto_generations"]) == 1
    assert nsga["pareto_generations"][0]["active"] is True
    assert nsga["pareto_generations"][0]["gui_launch_eligible"] is False
    assert nsga["selected_generation_id"] == nsga["pareto_generations"][0]["id"]
    assert "Tier-1 rolling status is stale" in (
        nsga["tier1_feedback_search"]["warnings"]
    )
    assert "Tier-1 rolling status is stale" not in nsga["warnings"]
    candidate = nsga["candidates"][0]
    assert candidate["spec_status"] == "pass"
    assert candidate["gui_launch_eligible"] is False
    assert candidate["report"]["size_W_mm"] == 1_050.0
    assert candidate["report"]["pred_core_loss_W"] == 2_000.0
    assert candidate["report"]["pred_secondary_winding_loss_W"] == 1_300.0
    assert candidate["constraints"]["resonance"]["pass"] is True
    assert published["legacy_path"].read_bytes() == b"legacy-all11-sentinel\n"


def _final_goal_current7_contract() -> tuple[dict[str, object], list[str]]:
    hard_spec = dict(readers_module.CURRENT7_HARD_SPEC)
    hard_spec.update({
        "T_limit_C": 100.0,
        "resonance_max_Hz": 20_000.0,
        "size_W_max_mm": 1_000.0,
        "size_L_max_mm": 1_000.0,
        "size_H_max_mm": 750.0,
    })
    constraints = list(readers_module.CURRENT7_CONSTRAINT_NAMES)
    minimum_position = constraints.index(
        readers_module.CURRENT7_RESONANCE_MINIMUM_CONSTRAINT
    )
    constraints.insert(
        minimum_position + 1,
        readers_module.CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT,
    )
    return hard_spec, constraints


def test_current7_condition_index_adds_read_only_final_goal_tab_and_detail(
    tmp_path, monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "current7-primary")
    hard_spec, constraint_names = _final_goal_current7_contract()
    condition = _publish_current7_runtime(
        tmp_path / "current7-final-goal",
        constraint_version_override=(
            "1000x1000x750-resmax20k-t100-core4-cw1-5-lmhalf"
        ),
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        candidate_overrides={
            "size_W_mm": 950.0,
            "size_L_mm": 940.0,
            "size_H_mm": 700.0,
            "volume_L": 625.1,
            "pred_f_res_tx_screen_Hz": 18_200.0,
            "pred_f_res_rx_screen_Hz": 18_100.0,
            "pred_f_res_min_screen_Hz": 18_100.0,
            "physical_constraint_G": {
                name: -1.0 for name in constraint_names
            },
        },
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"])
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(condition["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)

    payload = service._continuous_nsga2()

    assert payload["search_authority"]["kind"] == "current7"
    assert payload["source"] == str(primary["index_path"].resolve())
    assert payload["constraints"] == readers_module.CURRENT7_HARD_SPEC
    assert payload["candidate_count"] == 1
    assert len(payload["pareto_generations"]) == 2
    assert sum(
        generation["active"] is True
        for generation in payload["pareto_generations"]
    ) == 1
    final_goal = next(
        generation for generation in payload["pareto_generations"]
        if generation.get("source_kind") == "current7_condition_index"
    )
    assert final_goal["active"] is False
    assert final_goal["read_only"] is True
    assert final_goal["authority_eligible"] is False
    assert final_goal["gui_launch_eligible"] is False
    assert final_goal["resonance_contract"] == {
        "direction": "band",
        "minimum_key": "resonance_min_Hz",
        "maximum_key": "resonance_max_Hz",
        "minimum_constraint": "half_magnetizing_resonance_minimum",
        "maximum_constraint": "half_magnetizing_resonance_maximum",
        "minimum_operator": ">=",
        "maximum_operator": "<",
    }
    assert "1000×1000×750 mm" in final_goal["label"]
    assert "공진 15 kHz ≤ f < 20 kHz" in final_goal["label"]
    assert "100°C" in final_goal["label"]

    detail = service.nsga2_generation(final_goal["id"])

    assert detail["available"] is True
    assert detail["integrity_verified"] is True
    assert detail["result_scope"] == "read_only_condition_generation"
    assert detail["constraints"]["resonance_min_Hz"] == 15_000.0
    assert detail["constraints"]["resonance_max_Hz"] == 20_000.0
    assert detail["valid_candidate_count"] == 1
    candidate = detail["candidates"][0]
    assert candidate["spec_status"] == "pass"
    assert candidate["constraints"]["resonance"]["operator"] == "band"
    assert candidate["constraints"]["resonance"]["direction"] == "band"
    assert candidate["constraints"]["resonance"]["limit"] == [
        15_000.0, 20_000.0,
    ]
    assert candidate["constraints"]["resonance"]["minimum_operator"] == ">="
    assert candidate["constraints"]["resonance"]["maximum_operator"] == "<"
    assert candidate["constraints"]["resonance"]["value"] == 18_100.0
    assert candidate["gui_launch_eligible"] is False
    assert candidate["production_eligible"] is False
    assert candidate["fea_submission_approved"] is False


def test_condition_indexes_without_primary_form_read_only_archive(
    tmp_path, monkeypatch,
):
    base_spec, constraint_names = _final_goal_current7_contract()
    stage_specs = (
        ("entry", 1_200.0, 125.0),
        ("bridge", 1_150.0, 115.0),
        ("close", 1_075.0, 107.5),
        ("final", 1_000.0, 100.0),
    )
    published = []
    for stage, size_limit, temperature_limit in stage_specs:
        hard_spec = dict(base_spec)
        hard_spec.update({
            "size_W_max_mm": size_limit,
            "size_L_max_mm": size_limit,
            "T_limit_C": temperature_limit,
        })
        published.append(
            _publish_current7_runtime(
                tmp_path / f"current7-{stage}",
                constraint_version_override=(
                    f"{stage}-{size_limit:g}-t{temperature_limit:g}"
                ),
                hard_spec_override=hard_spec,
                constraint_names_override=constraint_names,
                candidate_overrides={
                    "size_W_mm": size_limit - 1.0,
                    "size_L_mm": size_limit - 1.0,
                    "size_H_mm": 700.0,
                    "pred_f_res_tx_screen_Hz": 18_200.0,
                    "pred_f_res_rx_screen_Hz": 18_100.0,
                    "pred_f_res_min_screen_Hz": 18_100.0,
                    "physical_constraint_G": {
                        name: -1.0 for name in constraint_names
                    },
                },
            )
        )
    monkeypatch.delenv(readers_module.CURRENT7_INDEX_ENV, raising=False)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        os.pathsep.join(
            str(item["index_path"]) for item in reversed(published)
        ),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    monkeypatch.setattr(
        service.scheduler,
        "mft_pipeline_status",
        lambda: pytest.fail(
            "condition archive must not depend on the scheduler MFT endpoint"
        ),
    )

    payload = service._continuous_nsga2()

    assert payload["available"] is True
    assert payload["source_kind"] == "current7_condition_archive"
    assert payload["status"] == "completed"
    assert payload["search_authority"] == {
        "configured": False,
        "kind": "condition_archive",
        "available": False,
        "integrity_verified": False,
        "source": None,
        "legacy_role": "archive",
        "warning": "read-only condition archive; no execution authority",
    }
    assert payload["candidate_count"] == 0
    assert payload["valid_candidate_count"] == 0
    assert payload["lanes"] == []
    assert isinstance(payload["near_feasible_preview"], list)
    assert len(payload["pareto_generations"]) == 4
    assert all(
        generation["read_only"] is True
        and generation["authority_eligible"] is False
        and generation["gui_launch_eligible"] is False
        and generation["active"] is False
        for generation in payload["pareto_generations"]
    )
    selected = next(
        generation for generation in payload["pareto_generations"]
        if generation["id"] == payload["selected_generation_id"]
    )
    assert selected["constraint_version"] == "final-1000-t100"
    assert payload["constraint_version"] == "final-1000-t100"
    assert payload["constraints"]["size_W_max_mm"] == 1_000.0
    assert payload["constraints"]["T_limit_C"] == 100.0
    assert payload["tier1_feedback_search"]["archived"] is True
    assert payload["tier1_feedback_search"]["authority_role"] == "archive"
    assert "읽기 전용" in payload["note"]

    detail = service.nsga2_generation(selected["id"])
    assert detail["available"] is True
    assert detail["result_scope"] == "read_only_condition_generation"
    assert detail["valid_candidate_count"] == 1


@pytest.mark.parametrize("valid_condition_count", [0, 1])
def test_condition_archive_configuration_failure_never_revives_legacy(
    tmp_path, monkeypatch, valid_condition_count,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    configured = [
        _publish_current7_runtime(
            tmp_path / f"condition-{position}",
            constraint_version_override=f"condition-{position}",
            hard_spec_override=hard_spec,
            constraint_names_override=constraint_names,
        )
        for position in range(valid_condition_count + 1)
    ]
    Path(configured[-1]["index_path"]).write_text(
        "{invalid condition index", encoding="utf-8"
    )
    monkeypatch.delenv(readers_module.CURRENT7_INDEX_ENV, raising=False)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        os.pathsep.join(str(item["index_path"]) for item in configured),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    monkeypatch.setattr(
        service,
        "_combined_tier1_feedback_nsga_diagnostics",
        lambda: {
            "available": True,
            "integrity_verified": True,
            "candidate_rows": [],
            "near_candidate_rows": [],
            "lanes": [],
            "stale": False,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        service.scheduler,
        "mft_pipeline_status",
        lambda: pytest.fail(
            "configured condition archive must not revive the legacy API"
        ),
    )

    payload = service._continuous_nsga2()

    assert payload["available"] is False
    assert payload["source_kind"] == "current7_condition_archive"
    assert payload["search_authority"]["kind"] == "condition_archive"
    assert payload["search_authority"]["available"] is False
    assert payload["search_authority"]["integrity_verified"] is False
    assert payload["tier1_feedback_search"]["archived"] is True
    assert payload["tier1_feedback_search"]["authority_role"] == "archive"
    assert payload["candidate_count"] == 0
    assert len(payload["pareto_generations"]) == valid_condition_count
    assert len(payload["tier1_current7_condition_searches"]) == (
        valid_condition_count + 1
    )
    assert payload["tier1_current7_condition_searches"][-1][
        "integrity_verified"
    ] is False


def test_condition_archive_over_limit_configuration_fails_closed(
    tmp_path, monkeypatch,
):
    configured_paths = [
        tmp_path / f"condition-{position}.json"
        for position in range(
            readers_module.CURRENT7_CONDITION_INDEX_LIMIT + 1
        )
    ]
    monkeypatch.delenv(readers_module.CURRENT7_INDEX_ENV, raising=False)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        os.pathsep.join(str(path) for path in configured_paths),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    monkeypatch.setattr(
        service,
        "_combined_tier1_feedback_nsga_diagnostics",
        lambda: {
            "available": True,
            "integrity_verified": True,
            "candidate_rows": [],
            "near_candidate_rows": [],
            "lanes": [],
            "stale": False,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        service.scheduler,
        "mft_pipeline_status",
        lambda: pytest.fail(
            "rejected condition configuration must not revive legacy"
        ),
    )

    payload = service._continuous_nsga2()

    assert payload["available"] is False
    assert payload["source_kind"] == "current7_condition_archive"
    assert payload["search_authority"]["kind"] == "condition_archive"
    assert payload["candidate_count"] == 0
    assert payload["tier1_current7_condition_searches"] == []
    assert any(
        "exceeds the bounded limit" in warning
        for warning in payload["warnings"]
    )


def test_current7_condition_authenticated_refusal_keeps_read_only_tab_visible(
    tmp_path, monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "current7-primary")
    hard_spec, constraint_names = _final_goal_current7_contract()
    condition = _publish_current7_runtime(
        tmp_path / "current7-final-goal-refusal",
        constraint_version_override=(
            "1000x1000x750-resmax20k-t100-core4-cw1-5-lmhalf"
        ),
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        include_authenticated_record=False,
        refusal_overrides=[{}],
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"])
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(condition["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["available"] is True
    assert diagnostics["integrity_verified"] is True
    assert diagnostics["healthy"] is False
    assert diagnostics["authority_eligible"] is False
    assert diagnostics["completed_count"] == 1
    assert diagnostics["authenticated_terminal_seed_count"] == 0
    assert diagnostics["refused_terminal_count"] == 1
    assert diagnostics["warnings"] == [
        "current7 refused 1 terminal artifact(s)"
    ]

    generations = service._current7_condition_generation_records(
        [diagnostics]
    )
    assert len(generations) == 1
    generation = generations[0]
    assert generation["selectable"] is True
    assert generation["state"] == "read_only_with_refusals"
    assert generation["source_healthy"] is False
    assert generation["authenticated_terminal_seed_count"] == 0
    assert generation["refused_terminal_count"] == 1

    detail = service.nsga2_generation(generation["id"])
    assert detail["available"] is True
    assert detail["integrity_verified"] is True
    assert detail["candidate_count"] == 0
    assert detail["generation"]["refused_terminal_count"] == 1
    assert detail["tier1_current7_search"]["refused_terminal_count"] == 1


def test_current7_four_stage_mixed_bundle_projection_is_read_only_visible(
    tmp_path, monkeypatch,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = [
        _publish_current7_runtime(
            tmp_path / f"current7-mixed-{stage}",
            constraint_version_override=f"{stage}-mixed-current7-test",
            hard_spec_override=hard_spec,
            constraint_names_override=constraint_names,
            mixed_bundle_projection=True,
        )
        for stage in ("entry", "bridge", "close", "final")
    ]
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        os.pathsep.join(str(item["index_path"]) for item in published),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()

    assert len(diagnostics) == 4
    assert all(item["integrity_verified"] is True for item in diagnostics)
    assert all(item["authority_eligible"] is False for item in diagnostics)
    assert all(item["mixed_bundle_projection"] is True for item in diagnostics)
    assert all(item["source_bundle_count"] == 2 for item in diagnostics)
    assert all(
        len(item["source_bundle_cohorts_sha256"]) == 64
        for item in diagnostics
    )
    assert all(item["search_count"] == 2 for item in diagnostics)
    assert all(item["completed_count"] == 2 for item in diagnostics)
    assert all(
        item["authenticated_terminal_seed_count"] == 1
        for item in diagnostics
    )
    assert all(item["refused_terminal_count"] == 1 for item in diagnostics)
    generations = service._current7_condition_generation_records(diagnostics)
    assert len(generations) == 4
    assert all(item["read_only"] is True for item in generations)
    assert all(item["gui_launch_eligible"] is False for item in generations)
    assert all(item["state"] == "read_only_with_refusals" for item in generations)


def test_current7_mixed_bundle_projection_deduplicates_terminal_tasks_by_triple(
    tmp_path, monkeypatch,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = _publish_current7_runtime(
        tmp_path / "current7-mixed-triple",
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        mixed_bundle_projection=True,
    )
    status_path = Path(published["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    # Physical IDs may be reused by a different sealed bundle cohort.  The
    # authenticated identity is the complete bundle/task/seed triple.
    status["latest_tasks"][1]["task_id"] = 7001
    status["latest_tasks"][1]["seed"] = 101
    status["refusals"][0]["task_id"] = 7001
    status["refusals"][0]["seed"] = 101
    status_path.write_bytes(_current7_json_bytes(status))
    _reseal_current7_runtime(published)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(published["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["completed_count"] == 2
    assert diagnostics["authenticated_terminal_seed_count"] == 1
    assert diagnostics["refused_terminal_count"] == 1


def test_current7_mixed_bundle_projection_rejects_conflicting_task_id_aliases(
    tmp_path, monkeypatch,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = _publish_current7_runtime(
        tmp_path / "current7-mixed-task-id-alias-drift",
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        mixed_bundle_projection=True,
    )
    status_path = Path(published["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert "id" not in status["latest_tasks"][0]
    status["latest_tasks"][0]["id"] = (
        status["latest_tasks"][0]["task_id"] + 1
    )
    status_path.write_bytes(_current7_json_bytes(status))
    _reseal_current7_runtime(published)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(published["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "task id/task_id aliases are inconsistent" in warning
        for warning in diagnostics["warnings"]
    )


@pytest.mark.parametrize(
    ("identity_location", "field", "value"),
    (
        ("task", "task_id", 7001.5),
        ("task", "task_id", True),
        ("task", "seed", 101.5),
        ("refusal", "task_id", 7002.5),
        ("refusal", "seed", 102.5),
        ("record", "seed", 101.5),
        ("candidate", "source_seed", 101.5),
    ),
)
def test_current7_mixed_bundle_projection_rejects_coerced_identity_numbers(
    tmp_path, monkeypatch, identity_location, field, value,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = _publish_current7_runtime(
        tmp_path / f"current7-mixed-coerced-{identity_location}-{field}",
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        mixed_bundle_projection=True,
    )
    status_path = Path(published["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if identity_location == "task":
        status["latest_tasks"][0][field] = value
    elif identity_location == "refusal":
        status["refusals"][0][field] = value
    elif identity_location == "record":
        record = status["terminal_results"][0]
        record[field] = value
        unsigned = {
            key: item for key, item in record.items()
            if key != "record_sha256"
        }
        record["record_sha256"] = readers_module._canonical_json_sha256(
            unsigned
        )
    else:
        status["aggregate"]["pareto_candidates"][0][field] = value
        status["aggregate"]["least_violation_candidate"][field] = value
    status_path.write_bytes(_current7_json_bytes(status))
    _reseal_current7_runtime(published)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(published["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False


def test_current7_mixed_bundle_projection_rejects_resealed_unknown_refusal_bundle(
    tmp_path, monkeypatch,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = _publish_current7_runtime(
        tmp_path / "current7-mixed-unknown-refusal",
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        mixed_bundle_projection=True,
    )
    status_path = Path(published["status_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["refusals"][0]["bundle_id"] = "current7-not-in-sealed-cohorts"
    status_path.write_bytes(_current7_json_bytes(status))
    _reseal_current7_runtime(published)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(published["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "refusal bundle is not in sealed source cohorts" in warning
        for warning in diagnostics["warnings"]
    )


def test_current7_mixed_bundle_projection_rejects_source_cohort_sha_drift(
    tmp_path, monkeypatch,
):
    hard_spec, constraint_names = _final_goal_current7_contract()
    published = _publish_current7_runtime(
        tmp_path / "current7-mixed-cohort-sha-drift",
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        mixed_bundle_projection=True,
    )
    status_path = Path(published["status_path"])
    index_path = Path(published["index_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    status["source_bundle_cohorts"][0]["remote_bundle"] += "-drift"
    index["source_bundle_cohorts"] = status["source_bundle_cohorts"]
    status_bytes = _current7_json_bytes(status)
    status_path.write_bytes(status_bytes)
    index["status"]["sha256"] = hashlib.sha256(status_bytes).hexdigest()
    index_path.write_bytes(_current7_json_bytes(index))
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(index_path),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "source bundle cohort SHA is invalid" in warning
        for warning in diagnostics["warnings"]
    )


def test_current7_primary_authenticated_refusal_never_grants_authority(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-primary-refusal",
        include_authenticated_record=False,
        refusal_overrides=[{}],
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["available"] is True
    assert diagnostics["integrity_verified"] is True
    assert diagnostics["healthy"] is False
    assert diagnostics["authority_eligible"] is False
    assert diagnostics["completed_count"] == 1
    assert diagnostics["authenticated_terminal_seed_count"] == 0
    assert diagnostics["refused_terminal_count"] == 1


def test_current7_condition_terminal_outcome_partition_mismatch_fails_closed(
    tmp_path, monkeypatch,
):
    condition = _publish_current7_runtime(
        tmp_path / "current7-missing-terminal-outcome",
        include_authenticated_record=False,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(condition["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "terminal outcome partition does not match completed tasks" in warning
        for warning in diagnostics["warnings"]
    )


def test_current7_condition_terminal_record_refusal_overlap_fails_closed(
    tmp_path, monkeypatch,
):
    condition = _publish_current7_runtime(
        tmp_path / "current7-overlapping-terminal-outcome",
        refusal_overrides=[{}],
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(condition["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "terminal outcome partition overlaps for seed(s): 101" in warning
        for warning in diagnostics["warnings"]
    )


@pytest.mark.parametrize(
    "refusal_override",
    [
        {"seed": 999},
        {"task_id": 7999},
    ],
)
def test_current7_condition_unknown_refusal_identity_fails_closed(
    tmp_path, monkeypatch, refusal_override,
):
    condition = _publish_current7_runtime(
        tmp_path / "current7-unknown-refusal",
        include_authenticated_record=False,
        refusal_overrides=[refusal_override],
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(condition["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["integrity_verified"] is False
    assert any(
        "refusal references an unknown or non-completed task identity" in warning
        for warning in diagnostics["warnings"]
    )


def test_current7_condition_index_resonance_direction_drift_fails_closed(
    tmp_path, monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "current7-primary")
    hard_spec, _ = _final_goal_current7_contract()
    rejected = _publish_current7_runtime(
        tmp_path / "current7-condition-rejected",
        constraint_version_override=(
            "1000x1000x750-resmax20k-t100-core4-cw1-5-lmhalf"
        ),
        hard_spec_override=hard_spec,
        # Deliberately retain the minimum constraint name with a maximum key.
        constraint_names_override=list(readers_module.CURRENT7_CONSTRAINT_NAMES),
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"])
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(rejected["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)

    payload = service._continuous_nsga2()

    assert payload["search_authority"]["available"] is True
    assert len(payload["pareto_generations"]) == 1
    assert payload["pareto_generations"][0]["active"] is True
    assert payload["tier1_current7_condition_searches"][0][
        "integrity_verified"
    ] is False
    assert any(
        "constraint names/resonance direction drifted" in warning
        for warning in payload["warnings"]
    )


def test_current7_running_zero_terminal_is_visible_without_old_index_mutation(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-running", terminal=False
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    diagnostics = service._tier1_current7_diagnostics()
    assert diagnostics["integrity_verified"] is True
    assert diagnostics["constraint_contract_verified"] is True
    assert diagnostics["authority_eligible"] is True
    assert diagnostics["running_count"] == 1
    assert diagnostics["candidate_rows"] == []
    _isolate_continuous_nsga_sources(monkeypatch, service)
    nsga = service._continuous_nsga2()
    assert nsga["status"] == "running"
    assert nsga["available"] is True
    assert nsga["search_authority"]["kind"] == "current7"
    assert nsga["search_authority"]["integrity_verified"] is True
    assert nsga["candidate_count"] == 0
    assert nsga["current7_active_seed_workers"] == 1
    assert published["legacy_path"].read_bytes() == b"legacy-all11-sentinel\n"


def test_current7_active_stale_pointer_fails_closed(tmp_path, monkeypatch):
    stale_at = (
        datetime.now(timezone.utc) - timedelta(minutes=10)
    ).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    published = _publish_current7_runtime(
        tmp_path / "current7-active-stale",
        terminal=False,
        harvest_observed_at=stale_at,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["freshness_required"] is True
    assert diagnostics["freshness_ok"] is False
    assert diagnostics["authority_eligible"] is False
    assert "authority was rejected" in diagnostics["warnings"][0]
    _isolate_continuous_nsga_sources(monkeypatch, service)
    payload = service._continuous_nsga2()
    assert payload["available"] is False
    assert payload["current7_active_seed_workers"] == 0
    assert payload["tier1_active_seed_workers"] == 0
    assert payload["active_seed_workers"] == 0
    assert not any(
        lane.get("source_kind") == "tier1_current7_secondary_index"
        for lane in payload["lanes"]
    )


def test_current7_active_fresh_harvest_accepts_old_task_event(
    tmp_path, monkeypatch,
):
    old_event_at = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    published = _publish_current7_runtime(
        tmp_path / "current7-active-quiet",
        terminal=False,
        updated_at=old_event_at,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["freshness_required"] is True
    assert diagnostics["freshness_ok"] is True
    assert diagnostics["authority_eligible"] is True
    assert diagnostics["status_event_at"] == old_event_at
    assert diagnostics["harvest_observed_at"] != old_event_at


def test_current7_cache_reuses_authenticated_snapshot_across_heartbeat(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-heartbeat-cache", terminal=False
    )
    index_path = published["index_path"]
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(index_path))
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    first = service._tier1_current7_diagnostics()
    first["warnings"].append("caller mutation must not enter cache")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    heartbeat = (
        datetime.now(timezone.utc) + timedelta(seconds=1)
    ).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    index["harvest_observed_at"] = heartbeat
    index_path.write_bytes(_current7_json_bytes(index))
    reference_paths = {
        Path(index[name]["path"]).resolve()
        for name in ("status", "compatibility")
    }
    original_reader = readers_module._bounded_json_bytes

    def reject_reference_reread(path, maximum):
        if Path(path).resolve() in reference_paths:
            raise AssertionError("immutable current7 reference was reread")
        return original_reader(path, maximum)

    monkeypatch.setattr(
        readers_module, "_bounded_json_bytes", reject_reference_reread
    )

    second = service._tier1_current7_diagnostics()

    assert second["integrity_verified"] is True
    assert second["authority_eligible"] is True
    assert second["harvest_observed_at"] == heartbeat
    assert second["index_file_sha256"] != first["index_file_sha256"]
    assert second["snapshot_file_sha256"] == first["snapshot_file_sha256"]
    assert "caller mutation must not enter cache" not in second["warnings"]


def test_current7_scaled_snapshot_replays_scale_identity(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-scaled", terminal=False, scaled=True
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["authority_eligible"] is True
    assert diagnostics["scale_policy_sha256"] == "e" * 64
    assert diagnostics["scale_summary"]["effective_active_total"] == 80
    assert diagnostics["scale_summary"]["max_workers_per_node"] == 8


@pytest.mark.parametrize("tamper", ["summary", "policy"])
def test_current7_scaled_snapshot_rejects_scale_tamper(
    tmp_path, monkeypatch, tamper,
):
    published = _publish_current7_runtime(
        tmp_path / f"current7-scaled-{tamper}",
        terminal=False,
        scaled=True,
    )
    status_path = published["status_path"]
    index_path = published["index_path"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if tamper == "summary":
        status["scale_summary"]["effective_active_total"] = 81
        index["scale_summary"] = status["scale_summary"]
    else:
        status["scale_policy_sha256"] = "0" * 64
        index["scale_policy_sha256"] = "0" * 64
    status_bytes = _current7_json_bytes(status)
    status_path.write_bytes(status_bytes)
    index["status"]["sha256"] = hashlib.sha256(status_bytes).hexdigest()
    index_path.write_bytes(_current7_json_bytes(index))
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(index_path))
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is False
    assert diagnostics["authority_eligible"] is False
    assert "scale" in diagnostics["warnings"][0]


def test_current7_cache_invalidates_on_new_status_snapshot(
    tmp_path, monkeypatch,
):
    root = tmp_path / "current7-snapshot-cache"
    published = _publish_current7_runtime(root, terminal=False)
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    first = service._tier1_current7_diagnostics()

    _publish_current7_runtime(root, terminal=True)
    second = service._tier1_current7_diagnostics()

    assert first["running_count"] == 1
    assert second["running_count"] == 0
    assert second["completed_count"] == 1
    assert second["snapshot_file_sha256"] != first["snapshot_file_sha256"]


def test_current7_cache_reauthenticates_changed_status_file(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-cache-tamper", terminal=False
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    assert service._tier1_current7_diagnostics()["integrity_verified"] is True
    status_path = published["status_path"]
    status_path.write_bytes(status_path.read_bytes() + b" ")

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is False
    assert "hash mismatch" in diagnostics["warnings"][0]


def test_current7_legacy_archive_cache_tracks_pointer_and_reference_stat(
    tmp_path, monkeypatch,
):
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    archive = tmp_path / "archive"
    status_path = archive / "status.json"
    model_path = archive / "model.json"
    pointer_path = archive / "index.json"
    archive.mkdir()
    status_path.write_bytes(b"{}")
    model_path.write_bytes(b"{}")
    pointer_path.write_bytes(_current7_json_bytes({
        "status": {"path": str(status_path), "sha256": "a" * 64},
        "model_pointer": {"path": str(model_path), "sha256": "b" * 64},
    }))
    service._tier1_nsga_pointer = pointer_path
    service._tier1_nsga_supplemental_roots = ()
    calls = 0

    def archived_projection():
        nonlocal calls
        calls += 1
        return {"available": False, "warnings": []}

    monkeypatch.setattr(
        service, "_combined_tier1_feedback_nsga_diagnostics",
        archived_projection,
    )

    first = service._archived_tier1_feedback_nsga_diagnostics()
    first["warnings"].append("caller mutation")
    second = service._archived_tier1_feedback_nsga_diagnostics()
    status_path.write_bytes(b"{ }\n")
    third = service._archived_tier1_feedback_nsga_diagnostics()

    assert calls == 2
    assert second["warnings"] == []
    assert third["warnings"] == []


def test_current7_status_event_pointer_mismatch_fails_closed(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-event-drift", terminal=False
    )
    index_path = published["index_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["status_event_at"] = "2000-01-01 00:00:00"
    index_path.write_bytes(_current7_json_bytes(index))
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(index_path))
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is False
    assert diagnostics["authority_eligible"] is False
    assert "status event identity" in diagnostics["warnings"][0]


def test_current7_terminal_only_snapshot_does_not_require_heartbeat(
    tmp_path, monkeypatch,
):
    sealed_at = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    published = _publish_current7_runtime(
        tmp_path / "current7-terminal-sealed",
        terminal=True,
        updated_at=sealed_at,
        harvest_observed_at=sealed_at,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["freshness_required"] is False
    assert diagnostics["freshness_ok"] is True
    assert diagnostics["authority_eligible"] is True


@pytest.mark.parametrize(
    "provenance_drift", ["terminal", "task", "candidate", "state"]
)
def test_current7_provenance_drift_fails_closed(
    tmp_path, monkeypatch, provenance_drift,
):
    published = _publish_current7_runtime(
        tmp_path / f"current7-{provenance_drift}-drift",
        provenance_drift=provenance_drift,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_diagnostics()

    assert diagnostics["available"] is True
    assert diagnostics["integrity_verified"] is False
    assert diagnostics["authority_eligible"] is False
    assert "provenance" in diagnostics["warnings"][0] or (
        "task identity" in diagnostics["warnings"][0]
    )


def test_current7_legacy_empty_running_contract_is_diagnostic_only(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-running-legacy-empty",
        terminal=False,
        legacy_empty_contract=True,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    diagnostics = service._tier1_current7_diagnostics()
    assert diagnostics["integrity_verified"] is True
    assert diagnostics["constraint_contract_verified"] is False
    assert diagnostics["authority_eligible"] is False
    _isolate_continuous_nsga_sources(monkeypatch, service)
    nsga = service._continuous_nsga2()
    assert nsga["available"] is False
    assert nsga["search_authority"]["kind"] == "current7"
    assert nsga["search_authority"]["integrity_verified"] is False
    assert any("no legacy fallback" in item for item in nsga["warnings"])


def test_configured_bad_current7_never_selects_scheduler_fallback(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(
        tmp_path / "current7-no-fallback",
        terminal=False,
        legacy_empty_contract=True,
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    monkeypatch.setattr(
        service.scheduler,
        "mft_pipeline_status",
        lambda: {
            "available": True,
            "nsga": {
                "active_seed_workers": 0,
                "lanes": [],
                "pareto_results": [{
                    "lane": "stale-scheduler-result",
                    "scope": "current_model",
                    "run_id": "legacy-run",
                    "model_id": "legacy-model",
                    "embedded_points": True,
                    "points": [{"volume_L": 700.0, "total_loss_W": 5_000.0}],
                }],
            },
        },
    )

    nsga = service._continuous_nsga2()

    assert nsga["available"] is False
    assert nsga["candidate_count"] == 0
    assert nsga["candidates"] == []
    assert nsga["near_feasible_preview"] == []
    archived = next(
        item for item in nsga["result_sets"]
        if item["lane"] == "stale-scheduler-result"
    )
    assert archived["stale"] is True
    assert archived["selected"] is False


def test_current7_secondary_index_fails_closed_on_status_tamper(
    tmp_path, monkeypatch,
):
    published = _publish_current7_runtime(tmp_path / "current7-tamper")
    monkeypatch.setenv(
        readers_module.CURRENT7_INDEX_ENV, str(published["index_path"])
    )
    status_path = published["status_path"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["aggregate"]["pareto_candidates"][0]["total_loss_W"] = 1.0
    status_path.write_bytes(_current7_json_bytes(status))
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    diagnostics = service._tier1_current7_diagnostics()
    assert diagnostics["available"] is True
    assert diagnostics["integrity_verified"] is False
    assert diagnostics["candidate_rows"] == []
    assert "hash mismatch" in diagnostics["warnings"][0]
    assert published["legacy_path"].read_bytes() == b"legacy-all11-sentinel\n"


def _blocker_hpo_v2_status(now: datetime) -> dict[str, object]:
    jobs = {
        "C_tx_tx_F/extra_trees": {
            "target": "C_tx_tx_F",
            "family": "extra_trees",
            "total": 20,
            "complete": 4,
            "running": 1,
            "failed": 0,
            "status": "running",
            "last_ordinal": 4,
        },
        "C_tx_tx_F/random_forest": {
            "target": "C_tx_tx_F",
            "family": "random_forest",
            "total": 20,
            "complete": 5,
            "running": 0,
            "failed": 0,
            "status": "running",
            "last_ordinal": 4,
        },
    }
    return {
        "schema_version": readers_module.BLOCKER_HPO_V2_STATUS_SCHEMA,
        "phase": "optimizing",
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "selected_cumulative_trials_per_job": 20,
        "pid": 1234,
        "host": "test-host",
        "started_at": (now - timedelta(minutes=2)).isoformat(),
        "heartbeat_at": now.isoformat(),
        "activity_at": now.isoformat(),
        "preflight": {
            "ready": True,
            "strict_full_rows": 6_151,
            "job_count": 2,
            "planned_model_fits_from_clean_studies": 200,
            "implementation_sha256": "c" * 64,
        },
        "trials": {
            "total": 40,
            "complete": 9,
            "running": 1,
            "failed": 0,
        },
        "jobs": jobs,
        "result": None,
        "error": None,
        "production_eligible": False,
        "production_model_eligible": False,
        "fea_submission_approved": False,
        "promotion_approved": False,
    }


def test_blocker_hpo_v2_status_uses_explicit_fresh_heartbeat(
    tmp_path, monkeypatch,
):
    now = datetime.now().astimezone()
    path = tmp_path / "hpo-v2-status.json"
    path.write_bytes(_current7_json_bytes(_blocker_hpo_v2_status(now)))
    monkeypatch.setenv(readers_module.BLOCKER_HPO_V2_STATUS_ENV, str(path))
    service = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )
    status = service._blocker_hpo_v2_diagnostics()
    assert status["available"] is True
    assert status["validated_running"] is True
    assert status["selected_hpo_target_count"] == 2
    assert status["trial_complete"] == 9
    assert status["trial_running"] == 1
    assert status["observed_strict_full_rows"] == 6_151


def test_blocker_hpo_v2_status_rejects_stale_or_authoritative_claims(
    tmp_path, monkeypatch,
):
    now = datetime.now().astimezone()
    path = tmp_path / "hpo-v2-status.json"
    stale = _blocker_hpo_v2_status(now - timedelta(minutes=5))
    path.write_bytes(_current7_json_bytes(stale))
    monkeypatch.setenv(readers_module.BLOCKER_HPO_V2_STATUS_ENV, str(path))
    service = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )
    status = service._blocker_hpo_v2_diagnostics()
    assert status["available"] is True
    assert status["validated_running"] is False
    assert "stale" in status["warnings"][0]
    sealed_preflight = _blocker_hpo_v2_status(now - timedelta(days=1))
    sealed_preflight["phase"] = "preflight_complete"
    path.write_bytes(_current7_json_bytes(sealed_preflight))
    preflight_status = service._blocker_hpo_v2_diagnostics()
    assert preflight_status["state"] == "preflight_complete"
    assert preflight_status["validated_running"] is False
    assert preflight_status["warnings"] == []
    asserted = _blocker_hpo_v2_status(now)
    asserted["promotion_approved"] = True
    path.write_bytes(_current7_json_bytes(asserted))
    rejected = service._blocker_hpo_v2_diagnostics()
    assert rejected["state"] == "invalid"
    assert "promotion authority" in rejected["warnings"][0]


def _publish_minimal_tier1_rolling_runtime(
    root: Path,
    *,
    now: datetime,
    cohort_id: str,
    task_id: int,
    seed: int,
    namespace: str | None,
    code_revision: str = "3" * 40,
) -> dict[str, object]:
    canonical = root / "canonical"
    cohort = root / "cohorts" / cohort_id
    deployment = root / "deployments" / cohort_id / "bundle"
    canonical.mkdir(parents=True)
    cohort.mkdir(parents=True)
    deployment.mkdir(parents=True)
    index_path = canonical / "index.json"
    model_path = canonical / "model_pointer.json"
    status_path = cohort / "status.json"
    manifest_path = deployment / "bundle_manifest.json"
    constraint_version = (
        "mft-tier1-envelope-1200x1200x750-res15k-"
        "t110all11-core4-5t-lmhalf-v4"
    )
    hard_spec = {
        "size_W_max_mm": 1200.0,
        "size_L_max_mm": 1200.0,
        "size_H_max_mm": 750.0,
        "resonance_min_Hz": 15_000.0,
        "T_limit_C": 110.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "magnetizing_inductance_factor": 0.5,
    }
    hard_spec_sha = hashlib.sha256(json.dumps(
        hard_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    source_sha = "1" * 64
    deployment_sha = "2" * 64
    temperature_targets = [
        "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
        *TEMPERATURE_TARGETS,
    ]
    temperature_contract = {
        "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
        "robust_upper_bound_C": 110.0,
        "target_count": len(temperature_targets),
        "targets": temperature_targets,
    }
    temperature_sha = hashlib.sha256(json.dumps(
        temperature_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    attestation_schema = "mft-tier1-slurm-temperature-attestation-v3"
    task_schema = "mft-tier1-slurm-seed-task-v3"
    manifest = {
        "schema_version": "mft-tier1-slurm-deployment-v1",
        "bundle_id": f"tier1-{cohort_id}",
        "cohort_id": cohort_id,
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": deployment_sha,
        "nsga_code_revision": code_revision,
        "attestation_schema_version": attestation_schema,
        "task_schema_version": task_schema,
        "temperature_constraint_contract": temperature_contract,
        "temperature_constraint_contract_sha256": temperature_sha,
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "controller_source_sha256": "4" * 64,
        "files": {},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    model_pointer = {
        "schema_version": "mft-tier1-slurm-model-pointer-v1",
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "current": {
            "cohort_id": cohort_id,
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
    }
    model_path.write_text(json.dumps(model_pointer), encoding="utf-8")
    status = {
        "schema_version": "mft-tier1-slurm-rolling-status-v1",
        "updated_at": now.isoformat(),
        "freshness_deadline_seconds": 30,
        "healthy": True,
        "error": None,
        "cohort_id": cohort_id,
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "bundle_manifest_sha256": manifest_sha,
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": deployment_sha,
        "nsga_code_revision": code_revision,
        "attestation_schema_version": attestation_schema,
        "task_schema_version": task_schema,
        "temperature_constraint_contract": temperature_contract,
        "temperature_constraint_contract_sha256": temperature_sha,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "controller": {"pid": task_id + 10_000, "mode": "rolling-worker"},
        "latest_tasks": [{
            "task_id": task_id,
            "seed": seed,
            "status": "running",
            "name": f"tier1-{namespace or 'main'}-{seed}",
            "cpus": 8,
            "memory_mb": 32_768,
            "account_name": "test",
            "slurm_job_id": str(700_000 + task_id),
        }],
        "scheduler_task_count": 1,
        "state_counts": {"running": 1},
        "rolling": {
            "enabled": True,
            "stop_condition": "explicit_operator_stop_only",
            "terminal_completion_triggers_refill": True,
            "seed_identity_count": 1,
            "active_plus_queued": 1,
            "target_active_plus_queued": 1,
            "duplicate_seed_count": 0,
        },
        "terminal_results": [],
        "aggregate_pareto": {
            "schema_version": "mft-tier1-slurm-aggregate-pareto-v1",
            "authenticated_terminal_count": 0,
            "source_candidate_count": 0,
            "unique_candidate_count": 0,
            "pareto_count": 0,
            "candidate_preview_count": 0,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": False,
            "candidates": [],
            "near_feasible_count": 0,
            "near_feasible": [],
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
        },
    }
    if namespace:
        status["search_profile"] = {
            "schema_version": "mft-tier1-nsga-search-profile-v1",
            "namespace": namespace,
        }
    status_path.write_text(json.dumps(status), encoding="utf-8")
    index = {
        "schema_version": "mft-tier1-slurm-rolling-index-v1",
        "active_cohort_id": cohort_id,
        "path_containment_root": str(root),
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": deployment_sha,
        "attestation_schema_version": attestation_schema,
        "task_schema_version": task_schema,
        "temperature_constraint_contract": temperature_contract,
        "temperature_constraint_contract_sha256": temperature_sha,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "model_pointer": {
            "path": str(model_path),
            "schema_version": "mft-tier1-slurm-model-pointer-v1",
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        },
        "status": {
            "path": str(status_path),
            "schema_version": "mft-tier1-slurm-rolling-status-v1",
            "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        },
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return {
        "index_path": index_path,
        "status_path": status_path,
        "manifest_path": manifest_path,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "constraint_version": constraint_version,
        "temperature_contract": temperature_contract,
        "source_sha": source_sha,
        "deployment_sha": deployment_sha,
        "code_revision": code_revision,
    }


def test_current_physics_revision_comes_from_repo_contract():
    assert CURRENT_PHYSICS_DATA_REVISION == PHYSICS_DATA_REVISION


def test_tier1_nine_supplemental_runtimes_are_authenticated_and_accepted(
    tmp_path, monkeypatch
):
    now = datetime.fromisoformat("2026-07-19T05:00:00+09:00")
    main = _publish_minimal_tier1_rolling_runtime(
        tmp_path / "main",
        now=now,
        cohort_id="main-cohort",
        task_id=101,
        seed=1_807_180_001,
        namespace=None,
    )
    supplemental_roots = []
    for position in range(9):
        root = tmp_path / f"supplemental-{position}"
        supplemental_roots.append(root)
        _publish_minimal_tier1_rolling_runtime(
            root,
            now=now,
            cohort_id=f"supplemental-cohort-{position}",
            task_id=201 + position,
            seed=1_907_190_001 + position,
            namespace=f"supplemental-lane-{position}",
        )
    monkeypatch.setenv("MFT_TIER1_ROLLING_INDEX", str(main["index_path"]))
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS",
        os.pathsep.join(str(root) for root in supplemental_roots),
    )
    monkeypatch.delenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOT_LIMIT", raising=False
    )

    combined = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )._combined_tier1_feedback_nsga_diagnostics()

    assert combined["source_count"] == 10
    assert combined["supplemental_source_count"] == 9
    assert combined["configured_source_count"] == 10
    assert combined["rejected_source_count"] == 0
    assert combined["all_configured_sources_verified"] is True
    assert not any(
        "supplemental root limit" in warning
        for warning in combined["warnings"]
    )


def test_tier1_supplemental_root_default_limit_rejects_excess(
    tmp_path, monkeypatch
):
    limit = readers_module.TIER1_SUPPLEMENTAL_ROOT_LIMIT
    roots = [tmp_path / f"root-{position}" for position in range(limit + 1)]
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS",
        os.pathsep.join(str(root) for root in roots),
    )
    monkeypatch.delenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOT_LIMIT", raising=False
    )

    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    assert limit >= 16
    assert service._tier1_nsga_supplemental_root_limit == limit
    assert len(service._tier1_nsga_supplemental_roots) == limit
    assert service._tier1_nsga_supplemental_roots == tuple(
        root.absolute() for root in roots[:limit]
    )
    assert service._tier1_nsga_supplemental_root_warnings == (
        "Tier-1 supplemental root limit exceeded; extra roots were rejected "
        f"(limit={limit})",
    )


def test_tier1_supplemental_root_limit_override_is_bounded(
    tmp_path, monkeypatch
):
    requested_limit = 20
    roots = [
        tmp_path / f"configured-root-{position}"
        for position in range(requested_limit + 1)
    ]
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS",
        os.pathsep.join(str(root) for root in roots),
    )
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOT_LIMIT", str(requested_limit)
    )

    configured = ArtifactService(
        tmp_path / "configured", record_runtime=False
    )

    assert configured._tier1_nsga_supplemental_root_limit == requested_limit
    assert len(configured._tier1_nsga_supplemental_roots) == requested_limit
    assert configured._tier1_nsga_supplemental_root_warnings == (
        "Tier-1 supplemental root limit exceeded; extra roots were rejected "
        f"(limit={requested_limit})",
    )

    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOT_LIMIT",
        str(readers_module.TIER1_SUPPLEMENTAL_ROOT_LIMIT_CEILING + 1),
    )
    bounded = ArtifactService(tmp_path / "bounded", record_runtime=False)
    assert bounded._tier1_nsga_supplemental_root_limit == (
        readers_module.TIER1_SUPPLEMENTAL_ROOT_LIMIT
    )
    assert len(bounded._tier1_nsga_supplemental_roots) == (
        readers_module.TIER1_SUPPLEMENTAL_ROOT_LIMIT
    )
    assert any(
        "outside the bounded range" in warning
        for warning in bounded._tier1_nsga_supplemental_root_warnings
    )


def test_tier1_supplemental_runtimes_are_authenticated_and_merged(
    tmp_path, monkeypatch
):
    now = datetime.fromisoformat("2026-07-19T05:00:00+09:00")
    main_root = tmp_path / "main"
    deep_root = tmp_path / "deep"
    normalized_root = tmp_path / "normalized"
    main = _publish_minimal_tier1_rolling_runtime(
        main_root,
        now=now,
        cohort_id="main-cohort",
        task_id=101,
        seed=1_807_180_001,
        namespace=None,
    )
    _publish_minimal_tier1_rolling_runtime(
        deep_root,
        now=now,
        cohort_id="deep-cohort",
        task_id=201,
        seed=1_907_190_001,
        namespace="deep-p320-g600-v1",
    )
    _publish_minimal_tier1_rolling_runtime(
        normalized_root,
        now=now,
        cohort_id="normalized-cohort",
        task_id=301,
        seed=1_907_290_001,
        namespace="normalized-coordinate-v1",
    )
    monkeypatch.setenv("MFT_TIER1_ROLLING_INDEX", str(main["index_path"]))
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS",
        os.pathsep.join((str(deep_root), str(normalized_root))),
    )
    service = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )

    combined = service._combined_tier1_feedback_nsga_diagnostics()

    assert combined["available"] is True, combined["warnings"]
    assert combined["integrity_verified"] is True
    assert combined["source_count"] == 3
    assert combined["supplemental_source_count"] == 2
    assert combined["configured_source_count"] == 3
    assert combined["rejected_source_count"] == 0
    assert combined["all_configured_sources_verified"] is True
    assert combined["search_count"] == 3
    assert combined["running_count"] == 3
    assert combined["active_plus_queued"] == 3
    assert len(combined["lanes"]) == 3
    assert {lane["source_label"] for lane in combined["lanes"]} == {
        "main", "deep-p320-g600-v1", "normalized-coordinate-v1",
    }
    assert all(lane["alive"] is True for lane in combined["lanes"])
    assert [item["accepted"] for item in combined["sources"]] == [
        True, True, True,
    ]
    assert combined["sources"][0]["bundle_manifest_verified"] is None
    assert all(
        item["bundle_manifest_verified"] is True
        for item in combined["sources"][1:]
    )

    service.scheduler = mock.Mock()
    service.scheduler.mft_pipeline_status.return_value = {"available": False}
    view = service._continuous_nsga2()
    assert view is not None
    assert view["tier1_active_seed_workers"] == 3
    assert view["pareto_generations"][0]["source_count"] == 3
    assert set(view["pareto_generations"][0]["cohort_ids"]) == {
        "main-cohort", "deep-cohort", "normalized-cohort",
    }


def test_tier1_supplemental_failures_never_replace_or_taint_primary(
    tmp_path, monkeypatch
):
    now = datetime.fromisoformat("2026-07-19T05:00:00+09:00")
    main_root = tmp_path / "main"
    tampered_root = tmp_path / "tampered"
    other_generation_root = tmp_path / "other-generation"
    main = _publish_minimal_tier1_rolling_runtime(
        main_root,
        now=now,
        cohort_id="main-cohort",
        task_id=101,
        seed=1_807_180_001,
        namespace=None,
    )
    tampered = _publish_minimal_tier1_rolling_runtime(
        tampered_root,
        now=now,
        cohort_id="tampered-cohort",
        task_id=201,
        seed=1_907_190_001,
        namespace="tampered",
    )
    _publish_minimal_tier1_rolling_runtime(
        other_generation_root,
        now=now,
        cohort_id="other-cohort",
        task_id=301,
        seed=1_907_290_001,
        namespace="other-code",
        code_revision="9" * 40,
    )
    Path(tampered["manifest_path"]).write_text(
        Path(tampered["manifest_path"]).read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    monkeypatch.setenv("MFT_TIER1_ROLLING_INDEX", str(main["index_path"]))
    monkeypatch.setenv(
        "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS",
        os.pathsep.join((str(tampered_root), str(other_generation_root))),
    )
    service = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )

    combined = service._combined_tier1_feedback_nsga_diagnostics()

    assert combined["available"] is True
    assert combined["integrity_verified"] is True
    assert combined["source_count"] == 1
    assert combined["search_count"] == 1
    assert combined["running_count"] == 1
    assert combined["rejected_source_count"] == 2
    assert combined["all_configured_sources_verified"] is False
    assert all(
        item["accepted"] is False for item in combined["sources"][1:]
    )
    assert any(
        "bundle manifest identity mismatch" in warning
        for warning in combined["warnings"]
    )
    assert any(
        "generation/model identity does not match primary" in warning
        for warning in combined["warnings"]
    )

    Path(main["status_path"]).write_text(
        Path(main["status_path"]).read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    no_primary = ArtifactService(
        tmp_path / "regression", clock=lambda: now, record_runtime=False
    )._combined_tier1_feedback_nsga_diagnostics()
    assert no_primary["available"] is False
    assert no_primary["source_count"] == 0
    assert no_primary["supplemental_source_count"] == 0
    assert all(item["accepted"] is False for item in no_primary["sources"])


def test_tier1_supplemental_preview_is_bounded_and_source_fair(
    tmp_path, monkeypatch
):
    service = ArtifactService(tmp_path, record_runtime=False)
    service._tier1_nsga_supplemental_roots = (tmp_path / "deep",)
    hard_spec = {"T_limit_C": 110.0, "resonance_min_Hz": 15_000.0}
    hard_sha = hashlib.sha256(json.dumps(
        hard_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    contract = {"robust_upper_bound_C": 110.0, "targets": ["T_max_Tx"]}

    def source(
        *, role, label, cohort, seed, task_id, pareto_count, preview_count,
        near_count, near_preview_count,
    ):
        return {
            "available": True,
            "integrity_verified": True,
            "pointer_verified": True,
            "source_role": role,
            "source_label": label,
            "cohort_id": cohort,
            "constraint_version": "t110-v4",
            "constraints": hard_spec,
            "hard_spec_sha256": hard_sha,
            "model_manifest_sha256": "1" * 64,
            "deployment_model_manifest_sha256": "2" * 64,
            "nsga_code_revision": "3" * 40,
            "temperature_constraint_contract": contract,
            "search_count": 1,
            "running_count": 1,
            "queued_count": 0,
            "attaching_count": 0,
            "completed_count": 1,
            "terminal_results_verified": 1,
            "failed_terminal_results_verified": 0,
            "active_plus_queued": 1,
            "feasible_pareto_count": pareto_count,
            "candidate_preview_count": preview_count,
            "near_feasible_count": near_count,
            "state_counts": {"running": 1},
            "rolling": {"target_active_plus_queued": 1},
            "lanes": [{
                "task_id": task_id,
                "seed": seed,
                "state": "running",
                "lane": f"old-{label}",
            }],
            "candidate_rows": [{
                "task_id": task_id,
                "seed": seed,
                "candidate": {
                    "index": position,
                    "volume_L": float(position + 1),
                    "total_loss_W": float(10_000 - position),
                },
            } for position in range(preview_count)],
            "near_candidate_rows": [{
                "task_id": task_id,
                "seed": seed,
                "candidate": {
                    "terminal_population_index": position,
                    "target_acquisition_score": float(position),
                    "production_eligible": False,
                    "fea_submission_approved": False,
                    "eligible_for_submission": False,
                },
            } for position in range(near_preview_count)],
            "warnings": [],
        }

    primary = source(
        role="primary", label="main", cohort="main", seed=100,
        task_id=1, pareto_count=200, preview_count=128,
        near_count=100, near_preview_count=64,
    )
    deep = source(
        role="supplemental", label="deep", cohort="deep", seed=200,
        task_id=2, pareto_count=2, preview_count=2,
        near_count=2, near_preview_count=2,
    )
    monkeypatch.setattr(
        service, "_tier1_feedback_nsga_diagnostics", lambda: primary
    )
    monkeypatch.setattr(
        service, "_tier1_supplemental_nsga_diagnostics", lambda: [deep]
    )

    combined = service._combined_tier1_feedback_nsga_diagnostics()

    assert combined["feasible_pareto_count"] == 202
    assert combined["candidate_preview_count"] == 128
    assert combined["candidate_preview_limit"] == 128
    assert combined["candidate_preview_truncated"] is True
    assert {row["source_label"] for row in combined["candidate_rows"]} == {
        "main", "deep",
    }
    assert [
        row["source_label"] for row in combined["candidate_rows"][:4]
    ] == ["main", "deep", "main", "deep"]
    assert combined["near_feasible_count"] == 102
    assert combined["near_feasible_preview_count"] == 64
    assert combined["near_feasible_preview_truncated"] is True
    assert [
        row["source_label"] for row in combined["near_candidate_rows"][:4]
    ] == ["main", "deep", "main", "deep"]


def test_tier1_condition_generation_inventory_groups_versions_and_defers_history(
        tmp_path, monkeypatch):
    cohort_root = tmp_path / "tier1-generations"
    cohort_root.mkdir()

    def hard_spec_sha(spec):
        return hashlib.sha256(json.dumps(
            spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def publish(
            cohort_id, version, hard_spec, *, updated_at, healthy,
            pareto_count=0, completed_count=0):
        cohort = cohort_root / cohort_id
        cohort.mkdir()
        payload = {
            "schema_version": "mft-tier1-slurm-rolling-status-v1",
            "updated_at": updated_at,
            "healthy": healthy,
            "cohort_id": cohort_id,
            "constraint_version": version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha(hard_spec),
            "bundle_manifest_sha256": "b" * 64,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "scheduler_task_count": completed_count + 2,
            "state_counts": {"completed": completed_count, "running": 2},
            "terminal_results": [
                {"task_id": position + 1}
                for position in range(completed_count)
            ],
            "aggregate_pareto": {
                "pareto_count": pareto_count,
                "candidate_preview_total": pareto_count,
                # Inventory must not copy these rows into the dashboard payload.
                "candidates": [
                    {"terminal_population_index": position}
                    for position in range(min(pareto_count, 2))
                ],
            },
        }
        (cohort / "status.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    hard_spec_v3 = {
        "size_W_max_mm": 1200.0,
        "size_L_max_mm": 1200.0,
        "size_H_max_mm": 750.0,
        "resonance_min_Hz": 10_000.0,
        "T_limit_C": 120.0,
        "n_core_group_max": 4,
        "cw1_fixed_mm": 5.0,
    }
    hard_spec_v1 = {
        key: value for key, value in hard_spec_v3.items()
        if key != "T_limit_C"
    }
    version_v1 = "mft-tier1-envelope-1200-res10k-v1"
    version_v2 = "mft-tier1-envelope-1200-res10k-t120-v2"
    version_v3 = "mft-tier1-envelope-1200-res10k-t120-v3"
    publish(
        "current-v3", version_v3, hard_spec_v3,
        updated_at="2026-07-19T02:00:00+09:00",
        healthy=True, pareto_count=239, completed_count=12,
    )
    # Two statuses share the v2 generation identity.  The healthy completed
    # snapshot wins even though an incomplete duplicate advertises more rows.
    publish(
        "v2-incomplete-duplicate", version_v2, hard_spec_v3,
        updated_at="2026-07-18T23:59:00+09:00",
        healthy=False, pareto_count=99, completed_count=20,
    )
    publish(
        "archived-v2", version_v2, hard_spec_v3,
        updated_at="2026-07-18T23:58:00+09:00",
        healthy=True, pareto_count=0, completed_count=1_000,
    )
    publish(
        "incomplete-v1", version_v1, hard_spec_v1,
        updated_at="2026-07-17T23:00:00+09:00",
        healthy=False, pareto_count=0, completed_count=30,
    )

    monkeypatch.setenv("MFT_TIER1_NSGA_COHORT_ROOT", str(cohort_root))
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    active = {
        "integrity_verified": True,
        "constraint_version": version_v3,
        "constraints": hard_spec_v3,
        "hard_spec_sha256": hard_spec_sha(hard_spec_v3),
        "cohort_id": "current-v3",
        "search_count": 14,
        "completed_count": 12,
        "state_counts": {"completed": 12, "running": 2},
        "updated_at": "2026-07-19T02:00:00+09:00",
    }

    records = service._tier1_generation_records(
        active,
        active_candidate_count=239,
        active_display_count=128,
    )

    assert len(records) == 3
    current, archived, incomplete = records
    assert current["active"] is True
    assert current["constraint_version"] == version_v3
    assert current["candidate_count"] == 239
    assert current["display_candidate_count"] == 128
    assert archived["constraint_version"] == version_v2
    assert archived["cohort_id"] == "archived-v2"
    assert archived["state"] == "archived"
    assert archived["candidate_count"] == 0
    assert archived["updated_at"] == "2026-07-18T23:58:00.000001+09:00"
    assert current["updated_at"] == "2026-07-19T02:00:00+09:00"
    assert incomplete["constraint_version"] == version_v1
    assert incomplete["state"] == "incomplete"
    assert incomplete["selectable"] is False
    assert incomplete["warning"]
    assert current["id"] != archived["id"]
    assert all(
        record["candidate_endpoint"]
        == f"/api/nsga2/generations/{record['id']}"
        for record in records
    )

    public_records = [
        service._public_tier1_generation(record) for record in records
    ]
    assert all("candidates" not in record for record in public_records)
    assert all(
        not any(key.startswith("_") for key in record)
        for record in public_records
    )
    assert public_records[0]["gui_launch_eligible"] is True
    assert public_records[1]["gui_launch_eligible"] is False
    assert public_records[1]["detail_integrity_pending"] is True

    live_gate_failed = {
        **active,
        "pointer_verified": True,
        "integrity_verified": False,
    }
    failed_live_records = service._tier1_generation_records(
        live_gate_failed,
        active_candidate_count=0,
        active_display_count=0,
    )
    assert all(
        record["cohort_id"] != "current-v3"
        for record in failed_live_records
    )

    service._tier1_rolling_index_configured = True
    assert service._tier1_generation_records({
        **live_gate_failed,
        "pointer_verified": False,
    }) == []
    service._tier1_rolling_index_configured = False

    malformed_status = json.loads(
        (cohort_root / "archived-v2" / "status.json").read_text(encoding="utf-8")
    )
    malformed_status["hard_spec_sha256"] += "trailing-junk"
    (cohort_root / "archived-v2" / "status.json").write_text(
        json.dumps(malformed_status), encoding="utf-8"
    )
    malformed_records = service._tier1_generation_records(
        active,
        active_candidate_count=239,
        active_display_count=128,
    )
    assert all(record["cohort_id"] != "archived-v2" for record in malformed_records)

    # Restore the authenticated status for the detail-path assertions below.
    publish_payload = malformed_status
    publish_payload["hard_spec_sha256"] = hard_spec_sha(hard_spec_v3)
    (cohort_root / "archived-v2" / "status.json").write_text(
        json.dumps(publish_payload), encoding="utf-8"
    )

    # A self-declared empty generation must still take the full archived
    # terminal/result/aggregate authentication path before the UI calls it
    # verified.
    current_payload = {
        "available": True,
        "display_candidate_count": 128,
        "candidates": [],
        "pareto_generations": public_records,
        "selected_generation_id": current["id"],
        "tier1_feedback_search": active,
    }
    monkeypatch.setattr(service, "_continuous_nsga2", lambda: current_payload)
    historical_calls = []

    def authenticated_empty(record):
        historical_calls.append(record["id"])
        return {
            "available": True,
            "integrity_verified": True,
            "historical_snapshot": True,
            "source": str(record["_path"]),
            "updated_at": record["updated_at"],
            "constraint_version": record["constraint_version"],
            "constraints": dict(record["hard_spec"]),
            "hard_spec_sha256": record["hard_spec_sha256"],
            "model_manifest_sha256": "c" * 64,
            "candidate_rows": [],
            "search_count": record["search_count"],
            "completed_count": record["completed_count"],
            "feasible_pareto_count": 0,
            "warnings": [],
        }

    monkeypatch.setattr(
        service, "_historical_tier1_diagnostics", authenticated_empty
    )
    archived_payload = service.nsga2_generation(archived["id"])
    assert historical_calls == [archived["id"]]
    assert archived_payload["available"] is True
    assert archived_payload["integrity_verified"] is True
    assert archived_payload["selected_generation_id"] == archived["id"]
    assert archived_payload["candidate_count"] == 0
    assert archived_payload["candidates"] == []
    assert archived_payload["generation"]["gui_launch_eligible"] is False
    assert archived_payload["constraints"] == hard_spec_v3

    incomplete_payload = service.nsga2_generation(incomplete["id"])
    assert incomplete_payload["available"] is False
    assert incomplete_payload["integrity_verified"] is False
    assert incomplete_payload["candidates"] == []
    assert incomplete_payload["generation"]["selectable"] is False
    assert incomplete_payload["warning"] == incomplete["warning"]
    assert not any(
        "identity mismatch" in warning
        for warning in incomplete_payload["warnings"]
    )

    invalid = service.nsga2_generation("../../outside")
    assert invalid["available"] is False
    assert invalid["candidates"] == []


def test_archived_tier1_snapshot_accepts_sealed_once_controller(tmp_path):
    del tmp_path
    assert readers_module._tier1_controller_mode_verified(
        {"mode": "rolling-worker"}, historical_snapshot=False,
    )
    assert not readers_module._tier1_controller_mode_verified(
        {"mode": "once"}, historical_snapshot=False,
    )
    assert readers_module._tier1_controller_mode_verified(
        {"mode": "once"}, historical_snapshot=True,
    )
    assert not readers_module._tier1_controller_mode_verified(
        {"mode": "unknown"}, historical_snapshot=True,
    )
    assert readers_module._tier1_staleness_warning(
        freshness_ok=False, historical_snapshot=False,
    ) == "Tier-1 rolling status is stale"
    assert readers_module._tier1_staleness_warning(
        freshness_ok=False, historical_snapshot=True,
    ) is None


def test_display_text_normalization_falls_back_on_mojibake():
    assert _display_text_has_mojibake("활성 코호트 1234567890") is False
    assert _display_text_has_mojibake("理쒖쟻?ㅺ퀎") is True
    assert _normalized_display_text(
        "理쒖쟻?ㅺ퀎", "Active cohort 1234567890"
    ) == "Active cohort 1234567890"


def _install_checkpoint_fixture(
        campaign_root: Path, *, metrics_hash_valid: bool = True,
        parity_profile_sha: str = "profile-sha",
        target: str = "Llt_phys",
        capacitance_recovery: dict | None = None) -> tuple[Path, Path]:
    pointer = campaign_root / "training" / "registry" / "current.json"
    pointer.unlink()
    run_root = campaign_root / "training" / "checkpoint_runs" / "current"
    metrics_path = run_root / "checkpoint_metrics" / "threshold_000500_attempt_000001.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "schema_version": 1,
        "completed_at": "2026-07-11T02:40:00+09:00",
        "checkpoint": 500,
        "dataset_sha256": "snapshot-sha",
        "profile_sha256": "profile-sha",
        "strict_full_rows": 100,
        "metrics": [
            {
                "time": "2026-07-11 02:40:00", "target": target,
                "n": 100, "r2": .95, "rmse": .15, "mape_pct": 1.0,
                "p90_ape_pct": 2.0, "slice": "global",
            },
            {
                "time": "2026-07-11 02:40:00", "target": target,
                "n": 100, "r2": .90, "rmse": .25, "mape_pct": 1.5,
                "p90_ape_pct": 3.0, "slice": "Llt20-40",
            },
        ],
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if not metrics_hash_valid:
        metrics_hash = "0" * 64
    state = {
        "schema_version": 2,
        "completed": [{
            "threshold": 500,
            "actual_strict_full_rows": 100,
            "snapshot_sha256": "snapshot-sha",
            "metrics_result": str(metrics_path),
            "metrics_result_sha256": metrics_hash,
            "profile_sha256": "profile-sha",
            "activation_minimum_strict_full_rows": 3000,
            "completed_at": "2026-07-11T02:41:00+09:00",
            "kind": "metrics_only",
        }],
        "identity": {
            "solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "library_revision": "c" * 40,
            "profile_sha256": "profile-sha",
            "activation_minimum_strict_full_rows": 3000,
        },
    }
    state_path = run_root / "checkpoint_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    parity_path = metrics_path.with_suffix(".parity.json")
    actual_values = (
        [2.34567890123456e-10, 2.45678901234567e-10]
        if target.startswith("C_") else [27.0, 28.0]
    )
    predicted_values = (
        [2.35567890123456e-10, 2.44678901234567e-10]
        if target.startswith("C_") else [27.1, 27.9]
    )
    parity_payload = {
        "schema_version": 1,
        "artifact_type": "checkpoint_cv_oof_parity",
        "checkpoint": 500,
        "dataset_sha256": "snapshot-sha",
        "profile_sha256": parity_profile_sha,
        "strict_full_rows": 100,
        "prediction_kind": "out_of_fold",
        "cv": {"n_splits": 5, "shuffle": True, "seed": 42},
        "max_pairs_per_target": 400,
        "targets": {
            target: {
                "n": 100,
                "sample_count": 2,
                "sampling": {"method": "evenly_spaced_position", "limit": 400},
                "pairs": [
                    {"row_position": 0, "row_index": 10, "actual": actual_values[0], "predicted": predicted_values[0]},
                    {"row_position": 99, "row_index": 20, "actual": actual_values[1], "predicted": predicted_values[1]},
                ],
            },
        },
    }
    if capacitance_recovery is not None:
        parity_payload["capacitance_recovery"] = capacitance_recovery
    parity_path.write_text(json.dumps(parity_payload), encoding="utf-8")
    return metrics_path, parity_path


def _write_capacitance_parity_overlay(
        campaign_root: Path, *, mutate=None) -> tuple[Path, str]:
    targets = {}
    for offset, target in enumerate((
        "C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F",
    ), start=1):
        actual = [
            (offset + 0.12345678901234) * 1e-10,
            (offset + 0.23456789012345) * 1e-10,
        ]
        targets[target] = {
            "n": 100,
            "sample_count": 2,
            "sampling": {
                "method": "evenly_spaced_position", "limit": 2000,
            },
            "pairs": [
                {
                    "row_position": 0, "row_index": 100 + offset,
                    "actual": actual[0], "predicted": actual[0] * 1.01,
                },
                {
                    "row_position": 99, "row_index": 200 + offset,
                    "actual": actual[1], "predicted": actual[1] * .99,
                },
            ],
        }
    # A deliberately malformed non-C target proves that the overlay parser
    # never lets this dedicated evidence source replace unrelated parity.
    targets["Llt_phys"] = {"pairs": "must-not-be-read"}
    payload = {
        "schema_version": 1,
        "artifact_type": "checkpoint_cv_oof_parity",
        "completed_at": "2026-07-19T15:23:21+09:00",
        "checkpoint": 6151,
        "strict_full_rows": 6151,
        "dataset_sha256": "d" * 64,
        "profile_sha256": "e" * 64,
        "prediction_kind": "out_of_fold",
        "cv": {"n_splits": 5, "shuffle": True, "seed": 42},
        "max_pairs_per_target": 2000,
        "capacitance_recovery": {
            "contract": "mft-capacitance-lc-inverse-v1",
            "status": "applied",
            "recovered_row_count": 100,
        },
        "targets": targets,
    }
    if mutate is not None:
        mutate(payload)
    path = campaign_root / "overlay" / "checkpoint_006151.parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _service_after_overlay_env(campaign_root: Path, artifact_service):
    return ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        refill_controller=artifact_service.refill_controller,
        continuous_pipeline=artifact_service.continuous_pipeline,
        clock=artifact_service.clock,
        record_runtime=False,
    )


def test_data_counts_quality_throughput_and_revision(artifact_service):
    data = artifact_service.data()
    assert data["raw_total_rows"] == 2
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 1
    assert data["em_valid_rows"] == 1
    assert data["thermal_valid_rows"] == 1
    assert data["complete_rows"] == 1
    assert data["throughput_1h"] == 1
    assert data["added_24h"] == 1
    assert data["collector"]["no_data_tasks"] == 1
    assert data["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert data["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert data["pinned_library_revision"] == "c" * 40
    assert data["training_cohort"] == {
        "available": True,
        "count_basis": "exact_solver_library_strict_full",
        "source_kind": "campaign_training_fallback",
        "raw_rows": 2,
        "strict_em_rows": 2,
        "strict_full_rows": 1,
        "solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
        "library_revision": "c" * 40,
        "source_dataset_generation": None,
        "updated_at": "2026-07-11T02:30:00+09:00",
    }
    assert data["rows_not_latest_revision"] == 1
    assert data["rows_not_current_physics_revision"] == 0
    assert data["count_basis"] == "physics_revision_strict_full"
    assert data["member_git_hash_shorts"] == [
        "754923c", "bbbbbbb",
    ]
    assert data["eta_3000"] is not None
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["cohort_basis"] == "active_identity"
    assert timing["cohort_filter"] == {
        "git_hash": "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a",
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert timing["active_cohort"]["status"] == "active"
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["window_limit_rows"] == 100
    assert timing["stages"]["matrix"]["mean_seconds"] == 300.0
    assert timing["stages"]["total"]["mean_seconds"] == 3_000.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 0


def test_data_revision_aggregate_does_not_reset_with_zero_pinned_status(
        campaign_root, artifact_service):
    manifest_path = Path(campaign_root, "data", "dataset", "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 436
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    strict_path = Path(campaign_root, "training", "strict_data_status.json")
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["strict_em_rows"] = 0
    strict["strict_full_rows"] = 0
    strict_path.write_text(json.dumps(strict), encoding="utf-8")

    service = ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        continuous_pipeline=artifact_service.continuous_pipeline,
        clock=artifact_service.clock,
        record_runtime=False,
    )
    data = service.data()

    assert data["raw_total_rows"] == 436
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 1
    assert data["em_valid_rows"] == 1
    assert data["thermal_valid_rows"] == 1
    assert data["training_cohort"]["strict_full_rows"] == 0
    assert data["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert data["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"

    dashboard = service.dashboard(record=False)
    assert dashboard["data"]["total_rows"] == 1
    assert dashboard["models"]["current_data_count"] == 0


def test_data_prefers_pipeline_canonical_checkpoint_without_restart(
    campaign_root, artifact_service, tmp_path
):
    pipeline_root = tmp_path / "pipeline"
    canonical = pipeline_root / "canonical_checkpoint"
    canonical.mkdir(parents=True)
    strict_path = canonical / "strict_data_status.json"
    strict_path.write_text(json.dumps({
        "time": "2026-07-11T02:59:00+09:00",
        "raw_rows": 12,
        "strict_em_rows": 11,
        "strict_full_rows": 9,
        "state_identity": {
            "solver_revision": "d" * 40,
            "library_revision": "e" * 40,
        },
    }), encoding="utf-8")

    class CanonicalPipeline:
        root = pipeline_root

        @staticmethod
        def snapshot():
            return {
                "training": {
                    "latest_eligible_cohort": {
                        "available": True, "strict_full_rows": 10,
                    },
                    "latest_completed_training_snapshot": {
                        "available": True, "strict_full_rows": 9,
                    },
                    "latest_completed_job": {"id": 42, "state": "succeeded"},
                },
                "active_model": {
                    "verified": True, "production_active": False,
                    "state": "awaiting_activation",
                },
                "warnings": [],
            }

    service = ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        refill_controller=artifact_service.refill_controller,
        continuous_pipeline=CanonicalPipeline(),
        clock=artifact_service.clock,
        record_runtime=False,
    )
    first = service.data()
    assert first["training_cohort"]["strict_full_rows"] == 9
    assert first["training_cohort"]["source_kind"] == (
        "pipeline_canonical_checkpoint"
    )
    assert first["latest_eligible_cohort"]["strict_full_rows"] == 10
    assert first["latest_completed_training"]["job"]["id"] == 42
    assert first["active_model"]["production_active"] is False

    updated = json.loads(strict_path.read_text(encoding="utf-8"))
    updated.update(raw_rows=14, strict_em_rows=13, strict_full_rows=12)
    strict_path.write_text(json.dumps(updated), encoding="utf-8")
    second = service.data()
    assert second["training_cohort"]["strict_full_rows"] == 12

    strict_path.write_text("{", encoding="utf-8")
    corrupt = service.data()
    assert corrupt["training_cohort"]["available"] is False
    assert corrupt["training_cohort"]["strict_full_rows"] is None


def test_safe_artifact_cache_fail_closed_does_not_reuse_old_json(tmp_path):
    cache = SafeArtifactCache()
    path = tmp_path / "pointer.json"
    path.write_text(json.dumps({"generation": "new"}), encoding="utf-8")
    assert cache.json(path, {}, fail_closed=True).value == {
        "generation": "new"
    }
    path.write_text("{", encoding="utf-8")
    failed = cache.json(path, {}, fail_closed=True)
    assert failed.value == {}
    assert failed.warning


def test_models_include_planned_missing_targets_and_metrics(artifact_service):
    payload = artifact_service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    assert lookup["Llt_phys"]["r2"] == .91
    assert lookup["Llt_phys"]["trained"] is True
    assert lookup["Llt_phys"]["evaluated"] is True
    assert lookup["Llt_phys"]["evaluation_kind"] == "active_registry"
    assert lookup["Llt_phys"]["parity_available"] is False
    assert lookup["P_winding_total"]["status"] == "attention"
    assert lookup["Tprobe_Tx_leeward_max"]["status"] == "not_trained"
    assert lookup["Llt_phys"]["history"][-1]["n"] == 100


def test_models_use_only_sha_authorized_checkpoint_metrics_and_parity(
        campaign_root, artifact_service):
    metrics_path, parity_path = _install_checkpoint_fixture(campaign_root)

    payload = artifact_service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    model = lookup["Llt_phys"]

    assert payload["trained_count"] == 0
    assert payload["evaluated_count"] == 1
    assert payload["latest_checkpoint"] == 500
    assert payload["activation_minimum_strict_full_rows"] == 3000
    assert payload["checkpoint_evaluated_at"] == "2026-07-11T02:40:00+09:00"
    assert payload["activation_state"] == "preactivation_checkpoint"
    assert payload["source_kind"] == "checkpoint_cv"
    assert Path(payload["checkpoint_source"]).resolve() == metrics_path.resolve()
    assert not any("model pointer is unavailable" in warning
                   for warning in payload["warnings"])
    assert "배포 모델로 취급하지 않습니다" in payload["quality_note"]

    assert model["status"] == "checkpoint"
    assert model["trained"] is False
    assert model["evaluated"] is True
    assert model["deployable"] is False
    assert model["evaluation_kind"] == "checkpoint_cv"
    assert model["checkpoint"] == 500
    assert model["n_used"] == 100
    assert model["r2"] == .95
    assert model["rmse"] == .15
    assert model["mape_pct"] == 1.0
    assert model["parity_available"] is True
    assert model["parity_sample_count"] == 2
    assert model["parity_provenance"] is None
    assert Path(model["parity_source"]).resolve() == parity_path.resolve()
    assert "pairs" not in model

    parity = artifact_service.model_parity("Llt_phys")
    assert parity["available"] is True
    assert parity["checkpoint"] == 500
    assert parity["n"] == 100
    assert parity["sample_count"] == 2
    assert parity["pairs"] == [
        {"row_position": 0, "row_index": 10, "actual": 27.0, "predicted": 27.1},
        {"row_position": 99, "row_index": 20, "actual": 28.0, "predicted": 27.9},
    ]

    dashboard = artifact_service.dashboard(record=False)
    model_stage = next(
        stage for stage in dashboard["status"]["stages"]
        if stage["key"] == "models"
    )
    assert model_stage["state"] == "waiting"
    assert f"checkpoint 500 CV 1/{payload['target_count']}" in model_stage["detail"]
    assert "활성화 1/3,000" in model_stage["detail"]
    assert not any(
        warning.startswith("미학습 모델:")
        for warning in dashboard["status"]["warnings"]
    )


@pytest.mark.parametrize("capacitance_recovery", [
    None,
    {},
    {"status": "applied", "recovered_row_count": 100},
    {
        "contract": "mft-capacitance-lc-inverse-v1",
        "recovered_row_count": 100,
    },
    {
        "contract": "mft-capacitance-lc-inverse-v1",
        "status": "applied",
    },
])
def test_capacitance_checkpoint_parity_without_complete_recovery_provenance_is_hidden(
        campaign_root, artifact_service, capacitance_recovery):
    _install_checkpoint_fixture(
        campaign_root,
        target="C_rx_rx_F",
        capacitance_recovery=capacitance_recovery,
    )

    payload = artifact_service.models(current_data_count=100)
    model = next(
        item for item in payload["models"]
        if item["target"] == "C_rx_rx_F"
    )

    assert model["evaluated"] is True
    assert model["status"] == "invalid_provenance"
    assert model["parity_available"] is False
    assert model["parity_sample_count"] == 0
    assert model["parity_source"] is None
    assert model["parity_provenance"] == {
        "required": True,
        "valid": False,
        "status": "legacy_quantized_labels",
        "message": "legacy quantized labels / 교정 재학습 대기",
        "contract": (
            capacitance_recovery.get("contract")
            if isinstance(capacitance_recovery, dict) else None
        ),
        "expected_contract": "mft-capacitance-lc-inverse-v1",
        "recovery_status": (
            capacitance_recovery.get("status")
            if isinstance(capacitance_recovery, dict) else None
        ),
        "recovered_row_count": (
            capacitance_recovery.get("recovered_row_count")
            if isinstance(capacitance_recovery, dict) else None
        ),
        "unique_actual_count": 2,
    }
    assert any(
        "C_rx_rx_F parity provenance invalid: "
        "legacy quantized labels / 교정 재학습 대기" in warning
        for warning in payload["warnings"]
    )

    parity = artifact_service.model_parity("C_rx_rx_F")
    assert parity["available"] is False
    assert parity["pairs"] == []
    assert parity["parity_provenance"] == model["parity_provenance"]


def test_corrected_capacitance_checkpoint_parity_is_auto_enabled_with_provenance(
        campaign_root, artifact_service):
    recovery = {
        "contract": "mft-capacitance-lc-inverse-v1",
        "status": "applied",
        "recovered_row_count": 121,
        "max_observed_abs_delta_F": 4.9e-11,
    }
    _, parity_path = _install_checkpoint_fixture(
        campaign_root,
        target="C_rx_rx_F",
        capacitance_recovery=recovery,
    )

    payload = artifact_service.models(current_data_count=100)
    model = next(
        item for item in payload["models"]
        if item["target"] == "C_rx_rx_F"
    )

    assert model["status"] == "checkpoint"
    assert model["parity_available"] is True
    assert model["parity_sample_count"] == 2
    assert Path(model["parity_source"]).resolve() == parity_path.resolve()
    assert model["parity_provenance"] == {
        "required": True,
        "valid": True,
        "status": "corrected",
        "message": "LC 역산 교정 labels",
        "contract": "mft-capacitance-lc-inverse-v1",
        "expected_contract": "mft-capacitance-lc-inverse-v1",
        "recovery_status": "applied",
        "recovered_row_count": 121,
        "unique_actual_count": 2,
    }

    parity = artifact_service.model_parity("C_rx_rx_F")
    assert parity["available"] is True
    assert parity["parity_provenance"] == model["parity_provenance"]
    assert parity["metadata"]["capacitance_recovery"] == {
        "contract": "mft-capacitance-lc-inverse-v1",
        "status": "applied",
        "recovered_row_count": 121,
    }
    assert [pair["actual"] for pair in parity["pairs"]] == [
        2.34567890123456e-10,
        2.45678901234567e-10,
    ]


def test_authenticated_capacitance_overlay_replaces_only_legacy_c_parity(
        campaign_root, artifact_service, monkeypatch):
    _install_checkpoint_fixture(campaign_root, target="C_rx_rx_F")
    overlay_path, overlay_sha = _write_capacitance_parity_overlay(campaign_root)
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_ARTIFACT_ENV, str(overlay_path)
    )
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_SHA256_ENV, overlay_sha.upper()
    )
    service = _service_after_overlay_env(campaign_root, artifact_service)

    payload = service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    model = lookup["C_rx_rx_F"]

    assert model["status"] == "parity_overlay"
    assert model["evaluation_kind"] == "checkpoint_cv"
    assert model["checkpoint"] == 500
    assert model["parity_checkpoint"] == 6151
    assert model["parity_source_kind"] == "authenticated_overlay"
    assert model["parity_available"] is True
    assert model["parity_sample_count"] == 2
    assert Path(model["parity_source"]).resolve() == overlay_path.resolve()
    assert model["parity_provenance"] == {
        "required": True,
        "valid": True,
        "status": "corrected",
        "message": "LC 역산 교정 labels",
        "contract": "mft-capacitance-lc-inverse-v1",
        "expected_contract": "mft-capacitance-lc-inverse-v1",
        "recovery_status": "applied",
        "recovered_row_count": 100,
        "unique_actual_count": 2,
        "source_kind": "authenticated_overlay",
        "overlay_path": str(overlay_path.absolute()),
        "overlay_sha256": overlay_sha,
        "checkpoint": 6151,
        "completed_at": "2026-07-19T15:23:21+09:00",
    }
    assert not any(
        "C_rx_rx_F parity provenance invalid" in warning
        for warning in payload["warnings"]
    )
    # The malformed Llt payload embedded in the dedicated overlay is ignored.
    assert lookup["Llt_phys"]["evaluated"] is False
    assert lookup["Llt_phys"]["parity_available"] is False
    assert lookup["Llt_phys"]["parity_provenance"] is None

    parity = service.model_parity("C_rx_rx_F")
    assert parity["available"] is True
    assert parity["checkpoint"] == 6151
    assert parity["source_kind"] == "authenticated_overlay"
    assert parity["evaluated_at"] == "2026-07-19T15:23:21+09:00"
    assert Path(parity["source"]).resolve() == overlay_path.resolve()
    assert parity["parity_provenance"] == model["parity_provenance"]
    assert parity["metadata"]["authenticated_overlay"] == {
        "path": str(overlay_path.absolute()),
        "sha256": overlay_sha,
        "checkpoint": 6151,
        "completed_at": "2026-07-19T15:23:21+09:00",
    }
    assert parity["metadata"]["capacitance_recovery"] == {
        "contract": "mft-capacitance-lc-inverse-v1",
        "status": "applied",
        "recovered_row_count": 100,
    }
    assert parity["pairs"][0]["actual"] == pytest.approx(
        2.12345678901234e-10
    )


def test_capacitance_overlay_sha_mismatch_keeps_legacy_warning(
        campaign_root, artifact_service, monkeypatch):
    _install_checkpoint_fixture(campaign_root, target="C_rx_rx_F")
    overlay_path, _ = _write_capacitance_parity_overlay(campaign_root)
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_ARTIFACT_ENV, str(overlay_path)
    )
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_SHA256_ENV, "0" * 64
    )
    service = _service_after_overlay_env(campaign_root, artifact_service)

    payload = service.models(current_data_count=100)
    model = next(
        item for item in payload["models"]
        if item["target"] == "C_rx_rx_F"
    )

    assert model["status"] == "invalid_provenance"
    assert model["parity_available"] is False
    assert model["parity_source"] is None
    assert any(
        "capacitance parity overlay SHA-256 mismatch" in warning
        for warning in payload["warnings"]
    )
    assert any(
        "legacy quantized labels / 교정 재학습 대기" in warning
        for warning in payload["warnings"]
    )


@pytest.mark.parametrize(("case", "mutate", "warning"), [
    (
        "artifact-type",
        lambda payload: payload.update(artifact_type="not-parity"),
        "overlay contract mismatch",
    ),
    (
        "recovery-contract",
        lambda payload: payload["capacitance_recovery"].update(
            contract="legacy-contract"
        ),
        "overlay contract mismatch",
    ),
    (
        "recovery-status",
        lambda payload: payload["capacitance_recovery"].update(
            status="audit_unavailable"
        ),
        "overlay contract mismatch",
    ),
    (
        "recovered-count",
        lambda payload: payload["capacitance_recovery"].update(
            recovered_row_count=99
        ),
        "overlay target metadata is malformed",
    ),
    (
        "missing-cap-target",
        lambda payload: payload["targets"].pop("C_tx_rx_F"),
        "overlay target is missing",
    ),
])
def test_capacitance_overlay_contract_failures_keep_legacy_parity_hidden(
        campaign_root, artifact_service, monkeypatch, case, mutate, warning):
    del case
    _install_checkpoint_fixture(campaign_root, target="C_rx_rx_F")
    overlay_path, overlay_sha = _write_capacitance_parity_overlay(
        campaign_root, mutate=mutate
    )
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_ARTIFACT_ENV, str(overlay_path)
    )
    monkeypatch.setenv(
        readers_module.CAPACITANCE_PARITY_SHA256_ENV, overlay_sha
    )
    service = _service_after_overlay_env(campaign_root, artifact_service)

    payload = service.models(current_data_count=100)
    model = next(
        item for item in payload["models"]
        if item["target"] == "C_rx_rx_F"
    )

    assert model["status"] == "invalid_provenance"
    assert model["parity_available"] is False
    assert any(warning in item for item in payload["warnings"])
    assert any(
        "legacy quantized labels / 교정 재학습 대기" in item
        for item in payload["warnings"]
    )


def test_models_reject_checkpoint_with_bad_metrics_hash(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root, metrics_hash_valid=False)

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")

    # learning_curve.csv still exists, but it is history only and cannot make
    # an unauthorised checkpoint appear in the table.
    assert model["history"]
    assert model["trained"] is False
    assert model["evaluated"] is False
    assert model["status"] == "not_trained"
    assert payload["evaluated_count"] == 0
    assert payload["latest_checkpoint"] is None
    assert any("checkpoint metrics hash mismatch" in warning
               for warning in payload["warnings"])


def test_bad_parity_identity_keeps_metrics_but_returns_empty_parity(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root, parity_profile_sha="wrong-profile")

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")

    assert model["status"] == "checkpoint"
    assert model["evaluated"] is True
    assert model["r2"] == .95
    assert model["parity_available"] is False
    assert model["parity_sample_count"] == 0
    assert any("checkpoint parity identity mismatch" in warning
               for warning in payload["warnings"])
    parity = artifact_service.model_parity("Llt_phys")
    assert parity["available"] is False
    assert parity["pairs"] == []
    assert any("checkpoint parity identity mismatch" in warning
               for warning in parity["warnings"])


def test_checkpoint_state_must_match_current_strict_solver_library_identity(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root)
    state_path = (
        campaign_root / "training" / "checkpoint_runs" / "current" /
        "checkpoint_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["identity"]["solver_revision"] = "f" * 40
    state_path.write_text(json.dumps(state), encoding="utf-8")

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")
    assert payload["latest_checkpoint"] is None
    assert payload["evaluated_count"] == 0
    assert model["evaluated"] is False


def test_missing_active_pointer_warns_once_activation_floor_is_reached(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root)

    payload = artifact_service.models(current_data_count=3000)

    assert payload["activation_state"] == "activation_due"
    assert any("accepted schema-v2 model pointer is unavailable" in warning
               for warning in payload["warnings"])


def test_core_region_temperature_models_and_predictions_are_independent(
        artifact_service):
    assert TEMPERATURE_TARGETS == SURROGATE_TEMPERATURE_TARGETS
    assert tuple(TEMPERATURE_TARGETS[-4:]) == (
        "Tprobe_core_center_max",
        *CORE_REGION_TEMPERATURE_TARGETS,
    )

    model_payload = artifact_service.models(current_data_count=100)
    model_lookup = {item["target"]: item for item in model_payload["models"]}
    expected_labels = {
        "Tprobe_core_center_max": "코어 최대 온도(3영역 최대)",
        "Tprobe_core_center_leg_max": "코어 중앙 레그 최대 온도",
        "Tprobe_core_side_leg_max": "코어 사이드 레그 최대 온도",
        "Tprobe_core_top_yoke_max": "코어 상부 요크 최대 온도",
    }
    for target, label in expected_labels.items():
        assert TARGET_META[target]["label"] == label
        assert model_lookup[target]["label"] == label
        assert model_lookup[target]["status"] == "not_trained"

    predictions = {
        target: 80.0 + index
        for index, target in enumerate(TEMPERATURE_TARGETS)
    }
    predictions["Tprobe_core_center_max"] = 95.0
    predictions["Tprobe_core_center_leg_max"] = 91.0
    predictions["Tprobe_core_side_leg_max"] = 98.0
    predictions["Tprobe_core_top_yoke_max"] = 101.0
    candidate = artifact_service._candidate(
        {f"pred_{target}": value for target, value in predictions.items()},
        round_number=2,
        index=0,
    )

    assert {
        key: value
        for key, value in candidate["pred_temperatures_C"].items()
        if value is not None
    } == predictions
    assert all(
        candidate["pred_temperatures_C"][target] is None
        for target in ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core")
    )
    assert candidate["pred_max_temperature_C"] == 101.0
    assert candidate["constraints"]["temperature"]["pass"] is False


def test_nsga_and_verification_are_joined(artifact_service):
    nsga = artifact_service.nsga2()
    assert nsga["round"] == 2
    assert nsga["candidate_count"] == 2
    assert nsga["summary"]["min_volume_L"] == 500
    assert nsga["comparison"]["min_volume_change_L"] == -100
    assert nsga["candidates"][0]["id"] == "r02-0000"
    assert nsga["candidates"][0]["spec_status"] == "unknown"  # temperature models are absent
    assert nsga["candidates"][0]["constraints"]["bfield"]["pass"] is True
    assert nsga["candidates"][0]["B_design_analytic_T"] == 1.0
    assert nsga["candidates"][0]["diagnostic_pred_B_max_core"] == 2.7
    assert nsga["candidates"][0]["report"]["cw1_conductor_thickness_mm"] == 5

    verification = artifact_service.verification(nsga)
    assert verification["counts"]["coverage"] == 1.0
    assert verification["standard_candidates"][0]["evaluation"]["computed_status"] == "pass"
    assert verification["standard_candidates"][0]["evaluation"]["timing_seconds"] == {
        "matrix": 353.31,
        "loss": 1720.78,
        "icepak": 1039.83,
        "total": 3113.92,
    }
    assert verification["final"]["status"] == "pass"
    assert verification["final"]["evaluation"]["checks"]["full_model"]["pass"] is True
    assert verification["final"]["evaluation"]["timing_seconds"]["total"] == 3113.92


def test_verification_ignores_unindexed_deadline_hit_candidate(artifact_service):
    nsga = artifact_service.nsga2()
    nsga["candidates"].insert(
        0,
        {
            "id": "deadline-hit-full-em-standard-pass",
            "source_kind": "deadline_full_em_hit",
        },
    )

    verification = artifact_service.verification(nsga)

    assert verification["counts"]["coverage"] == 1.0
    assert verification["standard_candidates"][0]["candidate_id"] == "r02-0000"


def test_fea_timings_fail_closed_without_nonnegative_finite_result_fields(artifact_service):
    evaluation = artifact_service._evaluate_fea({
        "time_matrix": "12.5",
        "time_loss": -1,
        "time_thermal": "nan",
        "time": True,
    })
    assert evaluation["timing_seconds"] == {
        "matrix": 12.5,
        "loss": None,
        "icepak": None,
        "total": None,
    }

    missing = artifact_service._evaluate_fea({})
    assert missing["timing_seconds"] == {
        "matrix": None,
        "loss": None,
        "icepak": None,
        "total": None,
    }


def test_simulation_timing_summary_uses_newest_sha_for_current_physics():
    older = {
        "git_hash": OLDER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    newer = {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    summary = _simulation_timing_summary([
        {
            **older, "saved_at": "2026-07-11 00:00:00",
            "time_matrix": "100", "time_thermal": "100",
        },
        {
            "git_hash": "legacy", "physics_data_revision": "legacy",
            "saved_at": "2026-07-11 05:00:00", "thermal_on": 0,
            "time_matrix": "999", "time_thermal": "3",
        },
        {
            **newer, "git_hash": NEWER_SOLVER_REVISION.upper(),
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": "400", "time_thermal": "400",
        },
        {
            **newer, "saved_at": "2026-07-11 01:00:00",
            "time_matrix": "200", "time_thermal": "200",
        },
        {
            **older,
            "saved_at": "2026-07-11 03:00:00",
            "time_matrix": "300", "time_thermal": "300",
        },
    ], limit=2)

    assert summary["cohort_basis"] == "active_identity"
    assert summary["cohort_label"] == "활성 코호트 bbbbbbbbbb"
    assert summary["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert summary["active_cohort"]["available"] is True
    assert summary["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert summary["cohort_rows"] == 2
    assert summary["window_rows"] == 2
    assert summary["stages"]["matrix"]["sample_count"] == 2
    assert summary["stages"]["matrix"]["mean_seconds"] == 300.0
    assert summary["stages"]["matrix"]["median_seconds"] == 300.0
    assert summary["stages"]["icepak"]["mean_seconds"] == 300.0
    assert summary["stages"]["loss"]["sample_count"] == 0
    assert summary["stages"]["loss"]["mean_seconds"] is None


def test_simulation_timing_summary_does_not_fall_back_to_legacy_rows():
    summary = _simulation_timing_summary([
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": "2026-07-11 03:00:00", "thermal_on": 0,
            "time_matrix": 1, "time_loss": 2,
            "time_thermal": 3, "time": 6,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": "legacy_unspecified",
            "time_thermal": 3,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": "previous-physics-revision",
            "time_thermal": 3,
        },
    ])

    assert summary["available"] is False
    assert summary["cohort_filter"] == {
        "git_hash": None,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert "현재 revision 데이터 없음" in summary["cohort_label"]
    assert CURRENT_PHYSICS_DATA_REVISION in summary["cohort_label"]
    assert summary["cohort_rows"] == 0
    assert summary["window_rows"] == 0
    assert all(
        stage["sample_count"] == 0
        and stage["mean_seconds"] is None
        and stage["median_seconds"] is None
        for stage in summary["stages"].values()
    )


def test_simulation_timing_summary_uses_cap_solve_plus_extraction_only_when_on():
    base = {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    summary = _simulation_timing_summary([
        {
            **base, "saved_at": "2026-07-11 04:00:00", "cap_on": 1,
            "cap_solve_time_s": 10, "cap_extraction_time_s": 2,
            "time": 100,
        },
        {
            **base, "saved_at": "2026-07-11 03:00:00", "cap_on": "true",
            "cap_solve_time_s": 20, "cap_extraction_time_s": 4,
            "time": 110,
        },
        {
            **base, "saved_at": "2026-07-11 02:00:00", "cap_on": 0,
            "cap_solve_time_s": 999, "cap_extraction_time_s": 999,
            "time": 120,
        },
        {
            **base, "saved_at": "2026-07-11 01:00:00", "cap_on": 1,
            "cap_solve_time_s": 30,
            "time": 130,
        },
    ])

    stage = summary["stages"]["electrostatic"]
    assert stage["source_fields"] == [
        "cap_solve_time_s", "cap_extraction_time_s",
    ]
    assert stage["sample_count"] == 2
    assert stage["mean_seconds"] == pytest.approx(18.0)
    assert stage["median_seconds"] == pytest.approx(18.0)
    assert summary["stages"]["total"]["sample_count"] == 4


def test_simulation_timing_summary_tolerates_missing_columns():
    frame = pd.DataFrame([
        {"saved_at": "2026-07-11 03:00:00", "time_thermal": 3},
        {"git_hash": OLDER_SOLVER_REVISION, "time_matrix": 100},
        {
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "time_loss": 200,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
        },
    ])

    summary = _simulation_timing_summary(frame)

    assert summary["available"] is False
    assert summary["cohort_rows"] == 1
    assert summary["window_rows"] == 1
    assert all(stage["sample_count"] == 0
               for stage in summary["stages"].values())


def test_active_cohort_degrades_when_recency_or_newest_hash_is_missing():
    undated = _simulation_timing_summary([{
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "time_matrix": 100,
    }])
    assert undated["active_cohort"]["available"] is False
    assert undated["cohort_rows"] == 0

    newest_hash_missing = _simulation_timing_summary([
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 03:00:00",
            "time_matrix": 100,
        },
        {
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": 200,
        },
    ])
    assert newest_hash_missing["active_cohort"]["available"] is False
    assert newest_hash_missing["cohort_rows"] == 0


def test_simulation_timing_summary_degrades_if_physics_import_is_unavailable():
    summary = _simulation_timing_summary(
        [{
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": 100,
        }],
        current_physics_revision=None,
    )

    assert summary["available"] is False
    assert summary["active_cohort"]["status"] == "physics_revision_unavailable"
    assert summary["cohort_filter"] == {
        "git_hash": None,
        "physics_data_revision": None,
    }
    assert "import 실패" in summary["cohort_label"]
    assert summary["cohort_rows"] == 0


def test_zero_aware_percentage_metrics_exclude_structural_zero_targets():
    metrics = _zero_aware_percentage_metrics(
        [0.0, 10.0, 20.0],
        [5.0, 11.0, 18.0],
    )

    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["p90_ape_pct"] == pytest.approx(10.0)
    assert metrics["mape_n"] == 2
    assert metrics["mape_excluded_zero_count"] == 1
    assert metrics["mape_valid_pair_count"] == 3
    assert metrics["mape_zero_abs_tolerance"] == 1e-9

    all_zero = _zero_aware_percentage_metrics([0.0], [123.0])
    assert all_zero["mape_pct"] is None
    assert all_zero["p90_ape_pct"] is None
    assert all_zero["mape_n"] == 0
    assert all_zero["mape_excluded_zero_count"] == 1


def test_campaign_frame_summary_scopes_all_panels_to_newest_rolling_sha():
    from .conftest import FIXED_NOW

    current = {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "saved_at": (FIXED_NOW - timedelta(minutes=20)).isoformat(),
        "thermal_core_conductivity_model": (
            "anisotropic_wound_rule_of_mixtures_v1"
        ),
        "thermal_core_k_inplane": 18.0,
        "thermal_core_k_throughstack": 2.0,
        "core_lamination_factor": 0.85,
    }
    frame = pd.DataFrame([
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 1e-9,
            "C_rx_rx_F": 4e-9,
            "C_tx_rx_F": 0.2e-9,
            "f_res_tx_self_Hz": 100_000.0,
            "f_res_rx_self_Hz": 200_000.0,
            "f_res_interwinding_Hz": 50_000.0,
            "winding_flux_linkage_readback_status": "available",
            "winding_flux_linkage_readback_applicable": 1,
            "winding_flux_linkage_readback_available": 1,
            "winding_flux_linkage_readback_passed": 1,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 3e-9,
            "C_rx_rx_F": 8e-9,
            "C_tx_rx_F": 0.6e-9,
            "f_res_tx_self_Hz": 300_000.0,
            "f_res_rx_self_Hz": 400_000.0,
            "f_res_interwinding_Hz": 150_000.0,
            "winding_flux_linkage_readback_status": "unavailable",
            "winding_flux_linkage_readback_applicable": 1,
            "winding_flux_linkage_readback_available": 0,
            "winding_flux_linkage_readback_passed": 0,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 0,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": "thermal:required_group_missing",
            "cap_on": 1,
            "winding_flux_linkage_readback_status": "available",
            "winding_flux_linkage_readback_available": 1,
        },
        {
            **current,
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": "em:matrix_invalid",
            "cap_on": 1,
            "winding_flux_linkage_readback_status": "unavailable",
            "winding_flux_linkage_readback_available": 0,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=30)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 99e-9,
            "C_rx_rx_F": 99e-9,
            "C_tx_rx_F": 99e-9,
            "f_res_tx_self_Hz": 9_900_000.0,
            "f_res_rx_self_Hz": 9_900_000.0,
            "f_res_interwinding_Hz": 9_900_000.0,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 99.0,
            "thermal_core_k_throughstack": 99.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": (
                "untrusted_provenance:solver_revision_mismatch"
            ),
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 10.0,
            "thermal_core_k_throughstack": 10.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 10.0,
            "thermal_core_k_throughstack": 10.0,
        },
    ])
    summary = _campaign_frame_summary(frame, FIXED_NOW)

    assert summary["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert summary["active_cohort"]["physics_data_revision"] == (
        CURRENT_PHYSICS_DATA_REVISION
    )
    assert len(summary["cohorts"]) == 3
    cohorts = {
        (item["git_hash"], item["physics_data_revision"]): item
        for item in summary["cohorts"]
    }
    cohort = cohorts[(NEWER_SOLVER_REVISION, CURRENT_PHYSICS_DATA_REVISION)]
    assert cohort == {
        "git_hash": NEWER_SOLVER_REVISION,
        "git_hash_short": NEWER_SOLVER_REVISION[:10],
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "latest_saved_at": (FIXED_NOW - timedelta(minutes=20)).isoformat(),
        "active": True,
        "current": True,
        "raw_rows": 5,
        "strict_em_rows": 4,
        "strict_full_rows": 3,
        "growth_rate_per_hour": 3.0,
    }
    older = cohorts[(OLDER_SOLVER_REVISION, CURRENT_PHYSICS_DATA_REVISION)]
    assert older["active"] is False
    assert older["current"] is False
    assert older["raw_rows"] == 1
    assert older["strict_em_rows"] == 1
    assert older["strict_full_rows"] == 1
    legacy = cohorts[(
        "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
        "legacy_unspecified",
    )]
    assert legacy["active"] is False
    assert legacy["current"] is False
    assert legacy["raw_rows"] == 2
    assert legacy["strict_em_rows"] == 0
    assert legacy["strict_full_rows"] == 0
    assert [item["git_hash"] for item in summary["cohorts"]] == [
        NEWER_SOLVER_REVISION,
        "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
        OLDER_SOLVER_REVISION,
    ]
    aggregate = summary["physics_revision_aggregate"]
    assert aggregate["physics_data_revision"] == CURRENT_PHYSICS_DATA_REVISION
    assert aggregate["raw_rows"] == 6
    assert aggregate["strict_em_rows"] == 5
    assert aggregate["strict_full_rows"] == 4
    assert aggregate["growth_rate_per_hour"] == 4.0
    assert aggregate["member_git_hashes"] == [
        NEWER_SOLVER_REVISION, OLDER_SOLVER_REVISION,
    ]
    assert aggregate["member_git_hash_shorts"] == ["bbbbbbb", "aaaaaaa"]

    electrostatic = summary["electrostatic"]
    assert electrostatic["cohort_basis"] == "active_strict_full"
    assert electrostatic["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert electrostatic["cohort_rows"] == 3
    assert electrostatic["cap_stage_present_rows"] == 2
    assert electrostatic["cap_stage_absent_rows"] == 1
    assert electrostatic["cap_stage_unknown_rows"] == 0
    expected_capacitance = {
        "tx_tx": ("C_tx_tx_F", 1.0, 2.0, 3.0),
        "rx_rx": ("C_rx_rx_F", 4.0, 6.0, 8.0),
        "tx_rx": ("C_tx_rx_F", 0.2, 0.4, 0.6),
    }
    for key, (source, minimum, median, maximum) in expected_capacitance.items():
        metric = electrostatic["capacitance"][key]
        assert metric["source_column"] == source
        assert metric["sample_count"] == 2
        assert metric["min_nF"] == pytest.approx(minimum)
        assert metric["median_nF"] == pytest.approx(median)
        assert metric["max_nF"] == pytest.approx(maximum)
    expected_resonance = {
        "tx_self": ("f_res_tx_self_Hz", 100.0, 200.0, 300.0),
        "rx_self": ("f_res_rx_self_Hz", 200.0, 300.0, 400.0),
        "interwinding": (
            "f_res_interwinding_Hz", 50.0, 100.0, 150.0
        ),
    }
    for key, (source, minimum, median, maximum) in expected_resonance.items():
        metric = electrostatic["resonance"][key]
        assert metric["source_column"] == source
        assert metric["sample_count"] == 2
        assert metric["min_kHz"] == pytest.approx(minimum)
        assert metric["median_kHz"] == pytest.approx(median)
        assert metric["max_kHz"] == pytest.approx(maximum)

    thermal = {
        item["model"]: item for item in summary["thermal_models"]["models"]
    }
    assert summary["thermal_models"]["cohort_basis"] == "active_identity"
    assert summary["thermal_models"]["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert summary["thermal_models"]["total_rows"] == 5
    assert summary["thermal_models"]["tagged_rows"] == 5
    assert summary["thermal_models"]["missing_rows"] == 0
    assert thermal["anisotropic_wound_rule_of_mixtures_v1"]["count"] == 5
    assert "isotropic_legacy" not in thermal
    assert thermal["anisotropic_wound_rule_of_mixtures_v1"][
        "thermal_core_k_inplane"
    ]["median"] == pytest.approx(18.0)

    quarantine = summary["quarantine"]
    assert quarantine["current"]["rows"] == 2
    assert quarantine["legacy"]["rows"] == 2
    current_reasons = {
        item["reason"]: item["count"]
        for item in quarantine["current"]["reasons"]
    }
    legacy_reasons = {
        item["reason"]: item["count"]
        for item in quarantine["legacy"]["reasons"]
    }
    assert current_reasons == {
        "em:matrix_invalid": 1,
        "thermal:required_group_missing": 1,
    }
    assert not any("solver_revision_mismatch" in reason
                   for reason in current_reasons)
    assert legacy_reasons[
        "untrusted_provenance:solver_revision_mismatch"
    ] == 2

    metadata = summary["current_cohort_metadata"]
    assert metadata["core_lamination_factor"] == {
        "source_column": "core_lamination_factor",
        "sample_count": 5,
        "min": 0.85,
        "median": 0.85,
        "max": 0.85,
    }
    readback = metadata["winding_flux_linkage_readback"]
    assert readback["cohort_rows"] == 5
    assert readback["available_rows"] == 2
    assert readback["unavailable_rows"] == 2
    assert readback["missing_rows"] == 1
    assert {item["status"]: item["count"] for item in readback["statuses"]} == {
        "available": 2,
        "missing": 1,
        "unavailable": 2,
    }


def test_campaign_cohorts_sort_inactive_rows_by_latest_saved_row():
    from .conftest import FIXED_NOW

    recent_revision = "c" * 40
    undated_revision = "d" * 40
    inactive_second = FIXED_NOW - timedelta(minutes=10)
    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=1)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=100)
            ).isoformat(),
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=200)
            ).isoformat(),
        },
        {
            "git_hash": recent_revision,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=800)
            ).isoformat(),
        },
        {
            "git_hash": undated_revision,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        },
    ])

    cohorts = _campaign_frame_summary(frame, FIXED_NOW)["cohorts"]

    assert [item["git_hash"] for item in cohorts] == [
        NEWER_SOLVER_REVISION,
        recent_revision,
        OLDER_SOLVER_REVISION,
        undated_revision,
    ]
    assert cohorts[1]["latest_saved_at"] == (
        inactive_second + timedelta(microseconds=800)
    ).isoformat()
    assert cohorts[2]["raw_rows"] == 2
    assert cohorts[2]["latest_saved_at"] == (
        inactive_second + timedelta(microseconds=200)
    ).isoformat()
    assert cohorts[3]["latest_saved_at"] is None


def test_campaign_frame_summary_tolerates_missing_columns_and_uses_flags():
    from .conftest import FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=2)).isoformat(),
            "result_valid_em": 1,
            "result_valid_thermal": 1,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=1)).isoformat(),
        },
    ])

    summary = _campaign_frame_summary(frame, FIXED_NOW)

    cohort = summary["cohorts"][0]
    assert cohort["active"] is True
    assert cohort["raw_rows"] == 2
    assert cohort["strict_em_rows"] == 1
    assert cohort["strict_full_rows"] == 1
    assert cohort["growth_rate_per_hour"] == 1.0
    electrostatic = summary["electrostatic"]
    assert electrostatic["available"] is False
    assert electrostatic["cohort_rows"] == 1
    assert electrostatic["cap_stage_present_rows"] == 0
    assert electrostatic["cap_stage_absent_rows"] == 0
    assert electrostatic["cap_stage_unknown_rows"] == 1
    for metric in (
        *electrostatic["capacitance"].values(),
        *electrostatic["resonance"].values(),
    ):
        assert metric["sample_count"] == 0
        assert all(value is None for key, value in metric.items()
                   if key.startswith(("min_", "median_", "max_")))
    thermal = summary["thermal_models"]
    assert thermal["available"] is False
    assert thermal["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert thermal["total_rows"] == 2
    assert thermal["tagged_rows"] == 0
    assert thermal["missing_rows"] == 2
    assert thermal["models"] == []
    assert summary["current_cohort_metadata"]["core_lamination_factor"][
        "sample_count"
    ] == 0
    assert summary["current_cohort_metadata"][
        "winding_flux_linkage_readback"
    ]["missing_rows"] == 2


def test_campaign_audit_is_per_sha_and_preserves_malformed_rows(tmp_path):
    frame = pd.DataFrame({
        "git_hash": [
            NEWER_SOLVER_REVISION, None, OLDER_SOLVER_REVISION, "bad-sha",
        ],
        "physics_data_revision": [CURRENT_PHYSICS_DATA_REVISION] * 4,
    }, index=[7, 7, 3, 9])
    calls = []

    def fake_annotate(
            cohort, *, expected_solver_revision,
            expected_library_revision):
        calls.append((
            tuple(cohort["git_hash"]), expected_solver_revision,
            expected_library_revision,
        ))
        audited = cohort.copy()
        valid = expected_solver_revision is not None
        audited["_strict_valid_em"] = valid
        audited["_strict_valid_thermal"] = valid
        audited["_strict_valid_full"] = valid
        audited["_strict_invalid_reasons"] = "" if valid else "bad-provenance"
        return audited

    service = ArtifactService(tmp_path, record_runtime=False)
    with mock.patch(
        "regression_260707.quality_contract.annotate_validity",
        side_effect=fake_annotate,
    ):
        audited, warning = service._audited_campaign_frame(
            ReadResult(frame, "memory.parquet", True),
            NEWER_SOLVER_REVISION,
            "d" * 40,
        )

    assert warning is None
    assert audited.index.tolist() == [7, 7, 3, 9]
    assert audited["git_hash"].tolist()[:1] == [NEWER_SOLVER_REVISION]
    assert pd.isna(audited["git_hash"].iloc[1])
    assert audited["git_hash"].tolist()[2:] == [OLDER_SOLVER_REVISION, "bad-sha"]
    assert audited["_strict_valid_full"].tolist() == [True, False, True, False]
    assert calls == [
        ((NEWER_SOLVER_REVISION,), NEWER_SOLVER_REVISION, "d" * 40),
        ((None,), None, None),
        ((OLDER_SOLVER_REVISION,), OLDER_SOLVER_REVISION, None),
        (("bad-sha",), None, None),
    ]


def test_artifact_service_data_reads_lossless_campaign_parquet(campaign_root):
    from .conftest import DummyScheduler, FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": FIXED_NOW.isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 2e-9,
            "C_rx_rx_F": 4e-9,
            "C_tx_rx_F": 0.5e-9,
            "f_res_tx_self_Hz": 100_000.0,
            "f_res_rx_self_Hz": 200_000.0,
            "f_res_interwinding_Hz": 50_000.0,
            "thermal_core_conductivity_model": (
                "anisotropic_wound_rule_of_mixtures_v1"
            ),
            "thermal_core_k_inplane": 18.0,
            "thermal_core_k_throughstack": 2.0,
            "time_matrix": 480.0,
            "time_loss": 1_800.0,
            "cap_solve_time_s": 40.0,
            "cap_extraction_time_s": 5.0,
            "time_thermal": 900.0,
            "time": 3_180.0,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 0,
            "time_matrix": 300.0,
            "time_loss": 1_000.0,
            "time_thermal": 800.0,
            "time": 2_100.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_on": 0,
            "time_matrix": 1.0,
            "time_loss": 2.0,
            "time_thermal": 3.0,
            "time": 6.0,
        },
        {
            "git_hash": "c" * 40,
            "physics_data_revision": "previous-physics-revision",
            "saved_at": (FIXED_NOW - timedelta(hours=3)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
        },
    ])
    manifest_path = campaign_root / "data" / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parquet_path = campaign_root / "data" / "dataset" / "train.parquet"
    frame.to_parquet(parquet_path, index=False)
    history_path = (
        campaign_root / "monitoring" / "runtime" / "monitor_history.jsonl"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "time": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
        "data": {"cohorts": [{
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "strict_full_rows": 0,
        }]},
    }) + "\n", encoding="utf-8")
    service = ArtifactService(
        campaign_root,
        scheduler=DummyScheduler(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )

    audit_calls = []

    def audit_by_declared_sha(
            cohort, *, expected_solver_revision,
            expected_library_revision):
        declared = {
            str(value).strip().lower() for value in cohort["git_hash"]
        }
        assert declared == {expected_solver_revision}
        assert expected_library_revision is None
        audit_calls.append((expected_solver_revision, tuple(cohort.index)))
        audited = cohort.copy()
        valid = audited["physics_data_revision"].ne("legacy_unspecified")
        audited["_strict_valid_em"] = valid
        audited["_strict_valid_thermal"] = valid
        audited["_strict_valid_full"] = valid
        audited["_strict_invalid_reasons"] = [
            "" if item else "test:legacy" for item in valid
        ]
        return audited

    with mock.patch(
        "regression_260707.quality_contract.annotate_validity",
        side_effect=audit_by_declared_sha,
    ):
        data = service.data()

    assert audit_calls == [
        (NEWER_SOLVER_REVISION, (0,)),
        (OLDER_SOLVER_REVISION, (1,)),
        ("b171c7ce5f7a018be6a575a32b1a1f5b7caa980c", (2,)),
        ("c" * 40, (3,)),
    ]

    assert data["source"]["campaign_rows"] == str(parquet_path)
    assert data["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert data["cohorts"][0]["active"] is True
    assert data["cohorts"][0]["current"] is True
    assert data["cohorts"][0]["raw_rows"] == 1
    assert data["cohorts"][0]["strict_full_rows"] == 1
    assert data["count_basis"] == "physics_revision_strict_full"
    assert data["raw_total_rows"] == 4
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 2
    assert data["em_valid_rows"] == 2
    assert data["throughput_1h"] == 1
    assert data["member_git_hashes"] == [
        NEWER_SOLVER_REVISION, OLDER_SOLVER_REVISION,
    ]
    assert data["member_git_hash_shorts"] == ["bbbbbbb", "aaaaaaa"]
    older_cohort = next(
        item for item in data["cohorts"]
        if item["git_hash"] == OLDER_SOLVER_REVISION
    )
    # Old zero-based runtime snapshots must not turn a rolled SHA's existing
    # rows into false +/h growth.  saved_at is the authoritative basis.
    assert older_cohort["growth_rate_per_hour"] == 0
    assert data["rows_not_current_physics_revision"] == 2
    assert data["eta_3000"] is not None
    assert data["electrostatic"]["cap_stage_present_rows"] == 1
    assert data["electrostatic"]["capacitance"]["tx_tx"][
        "median_nF"
    ] == pytest.approx(2.0)
    assert data["thermal_models"]["total_rows"] == 1
    assert data["thermal_models"]["tagged_rows"] == 1
    assert data["quarantine"]["legacy"]["rows"] == 1
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["stages"]["matrix"]["mean_seconds"] == 480.0
    assert timing["stages"]["electrostatic"]["mean_seconds"] == 45.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 1
    assert timing["stages"]["icepak"]["mean_seconds"] == 900.0
    assert timing["stages"]["total"]["mean_seconds"] == 3_180.0


def test_runtime_recorder_recovers_from_winerror5_without_temp_collision(
        tmp_path, artifact_service):
    recorder = RuntimeRecorder(tmp_path / "runtime", min_interval_seconds=0)
    dashboard = artifact_service.dashboard(record=False)
    denied = PermissionError(13, "RaiDrive rename denied")
    denied.winerror = 5

    with mock.patch(
            "regression_260707.monitoring.readers.os.replace",
            side_effect=denied) as replace, mock.patch(
            "regression_260707.monitoring.readers.time.sleep"):
        recorder.record(dashboard)

    assert replace.call_count == 5
    snapshot = json.loads(recorder.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    history = recorder.history()["entries"]
    assert history[-1]["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    assert history[-1]["data"]["pinned_library_revision"] == dashboard["data"]["pinned_library_revision"]
    assert list(recorder.directory.glob(".monitor_snapshot.json.*.tmp")) == []


def test_runtime_snapshot_failure_warns_but_history_still_appends(
        campaign_root, artifact_service):
    service = ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        refill_controller=artifact_service.refill_controller,
        clock=artifact_service.clock,
        record_runtime=True,
    )
    with mock.patch.object(
            service.recorder, "_write_snapshot",
            side_effect=PermissionError(13, "snapshot blocked")):
        dashboard = service.dashboard()

    assert any("snapshot write failed" in warning for warning in dashboard["status"]["warnings"])
    history = service.recorder.history()["entries"]
    assert history[-1]["time"] == dashboard["generated_at"]
    assert history[-1]["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    assert history[-1]["data"]["pinned_library_revision"] == dashboard["data"]["pinned_library_revision"]


def test_final_display_fails_closed_on_missing_or_negative_wcp_loss(
        artifact_service, campaign_root):
    artifact = json.loads(
        Path(campaign_root, "verify", "results", "final_verification.json")
        .read_text(encoding="utf-8")
    )
    result = dict(artifact["result"])
    result["P_wcp_total"] = -1.0
    negative = artifact_service._evaluate_fea(result, require_full_model=True)
    assert negative["computed_status"] == "fail"
    assert negative["checks"]["loss_components"]["pass"] is False

    result.pop("P_wcp_total")
    missing = artifact_service._evaluate_fea(result, require_full_model=True)
    assert missing["computed_status"] == "unknown"
    assert missing["checks"]["loss_components"]["pass"] is None


def test_corrupt_json_retains_last_good_value(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"stage": "WAIT"}', encoding="utf-8")
    cache = SafeArtifactCache()
    first = cache.json(path, {})
    assert first.value["stage"] == "WAIT"
    path.write_text("{broken", encoding="utf-8")
    second = cache.json(path, {})
    assert second.value["stage"] == "WAIT"
    assert "마지막 정상" in second.warning or "읽기 실패" in second.warning


def test_parquet_reader_retains_last_good_synthetic_frame(tmp_path):
    path = tmp_path / "train.parquet"
    expected = pd.DataFrame([{
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "C_tx_tx_F": 1e-9,
    }])
    expected.to_parquet(path, index=False)
    cache = SafeArtifactCache()

    first = cache.parquet(path)
    assert first.exists is True
    assert first.warning is None
    assert first.value.to_dict("records") == expected.to_dict("records")

    path.write_bytes(b"partial parquet")
    second = cache.parquet(path)
    assert second.exists is True
    assert second.value.to_dict("records") == expected.to_dict("records")
    assert second.warning is not None


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_scheduler_uses_aggregate_summary_and_exact_project_status():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), timeout))
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({"name_prefix": "mft", "total": 9, "statuses": {"running": 4, "completed": 3, "failed": 2}})
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 510,
            "desired_simulations": 500,
            "effective_simulations": 480,
            "validated_concurrency_limit": 500,
            "min_desired_simulations": 0,
            "max_desired_simulations": 500,
            "policy_revision": 12,
            "scale_down_mode": "drain",
            "queued_count": 7,
            "attaching_count": 2,
            "executing_count": 4,
            "solving_count": 3,
            "logical_active_count": 13,
            "updated_at": "2026-07-13T01:00:00+09:00",
        })

    result = SchedulerReader(base_url="http://example.test", opener=opener, ttl=0).snapshot()
    assert result["connected"] is True
    assert result["running"] == 4
    assert result["total"] == 9
    assert result["parallel_target"] == 500
    assert result["effective_simulations"] == 480
    assert result["validated_concurrency_limit"] == 500
    assert result["policy_revision"] == 12
    assert result["live_queued"] == 7
    assert result["live_attaching"] == 2
    assert result["live_active"] == 4
    assert result["live_solving"] == 3
    assert result["logical_active"] == 6
    assert result["control_enabled"] is True
    assert calls[0][1] == "GET"
    assert "/api/tasks/summary?" in calls[0][0]
    assert "/api/tasks?" not in calls[0][0]
    assert calls[1][1] == "GET"
    assert calls[1][0].endswith("/api/projects/MFT_1MW_2026v1")
    assert calls[2][1] == "GET"
    assert calls[2][0].endswith(
        "/api/projects/MFT_1MW_2026v1/simulation-policy"
    )


def test_scheduler_legacy_cap_above_300_remains_visible_but_not_mutable():
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({"total": 9, "statuses": {"running": 4}})
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 510,
            "queued_count": 3,
            "attaching_count": 2,
            "executing_count": 4,
        })

    result = SchedulerReader(
        base_url="http://legacy.test", opener=opener, ttl=0
    ).snapshot()

    assert result["connected"] is True
    assert result["legacy_project_cap"] == 510
    assert result["logical_active"] == 6
    assert result["policy_supported"] is False
    assert result["control_enabled"] is False
    assert result["parallel_target"] is None
    assert "durable simulation-policy" in result["control_gate_reason"]


def test_scheduler_policy_allows_bounded_repair_of_legacy_desired_510():
    control = SchedulerReader(base_url="http://repair.test")._simulation_policy_control({
        "project": "MFT_1MW_2026v1",
        "desired_simulations": 510,
        "effective_simulations": 500,
        "validated_concurrency_limit": 510,
        "min_desired_simulations": 0,
        "max_desired_simulations": 500,
        "policy_revision": 4,
        "scale_down_mode": "drain",
        "queued_count": 10,
        "attaching_count": 2,
        "active_count": 6,
        "solving_count": 4,
        "logical_active_count": 6,
    })

    assert control["desired_simulations"] == 510
    assert control["parallel_target_max"] == 500
    assert control["live_active"] == 4
    assert control["logical_active"] == 6
    assert control["control_enabled"] is True


def test_scheduler_normalizes_aedt_pool_and_license_attach_status():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 9,
                "statuses": {"running": 4, "completed": 5},
            })
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 300,
                "queued_count": 7,
                "attaching_count": 2,
                "executing_count": 4,
                "logical_active_count": 13,
            })
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 2,
                    "target_project_concurrency": 16,
                    "projects_per_aedt": 2,
                },
                "plan": {
                    "idle_session_count": 2,
                    "hard_session_count": 3,
                    "warm_spare_status_reason": "target satisfied",
                },
                "sessions": [
                    {"id": 1, "state": "ready"},
                    {"id": 2, "state": "ready"},
                    {"id": 3, "state": "starting"},
                ],
                "leases": [
                    {"id": 11, "state": "active"},
                    {"id": 12, "state": "queued"},
                    {"id": 13, "state": "releasing"},
                ],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "label": "Electronics Desktop",
                    "total": 550,
                    "used": 344,
                }],
                # A conflicting fallback proves the complete feature list wins.
                "in_use": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 12,
                }],
                "error": "",
                "admission": {"snapshot_valid": True},
            })
        if "/api/tasks?" in request.full_url and "_aedt_pool_hosts" in request.full_url:
            return FakeResponse([
                {
                    "task_id": 701, "name": "bundle-a-host",
                    "project": "_aedt_pool_hosts", "status": "running",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 702, "name": "bundle-b-host",
                    "project": "_aedt_pool_hosts", "status": "attaching",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 703, "name": "bundle-c-host",
                    "project": "_aedt_pool_hosts", "status": "queued",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 704, "name": "finished-host",
                    "project": "_aedt_pool_hosts", "status": "completed",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 705, "name": "unrelated-host",
                    "project": "other", "status": "running",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 706, "name": "central-session-host",
                    "project": "_aedt_pool_hosts", "status": "running",
                    "entrypoint": "slurm_scheduler.aedt_session_host",
                },
            ])
        raise AssertionError(request.full_url)

    result = SchedulerReader(
        base_url="http://example.test",
        opener=opener,
        timeout=2.0,
        optional_timeout=0.25,
        pool_timeout=4.0,
        ttl=0,
    ).snapshot()

    assert result["connected"] is True
    attach = result["aedt_attach"]
    assert attach["available"] is True
    assert attach["state"] == "operational"
    assert attach["errors"] == []
    assert attach["pool"] == {
        "available": True,
        "enabled": True,
        "adapter_ready": True,
        "validation_passed": True,
        "operational": True,
        "max_sessions": 8,
        "min_idle_sessions": 2,
        "idle_sessions": 2,
        "hard_sessions": 3,
        "warm_spare_deficit": None,
        "warm_spare_start_needed": None,
        "session_record_count": 3,
        "lease_record_count": 3,
        "live_leases": 3,
        "queued_leases": 1,
        "ready_sessions": 2,
        "busy_sessions": 0,
        "session_states": {"ready": 2, "starting": 1},
        "lease_states": {"active": 1, "queued": 1, "releasing": 1},
        "warm_spare_reason": "target satisfied",
        "error": None,
    }
    assert attach["license"] == {
        "available": True,
        "feature": "electronics_desktop",
        "label": "Electronics Desktop",
        "used": 344,
        "total": 550,
        "snapshot_valid": True,
        "checked_at": "2026-07-13T02:00:00+09:00",
        "error": None,
    }
    assert attach["node_local"] == {
        "available": True,
        "project": "_aedt_pool_hosts",
        "active_host_tasks": 3,
        "statuses": {"attaching": 1, "queued": 1, "running": 1},
        "bundle_count": 3,
        "bundle_ids": ["bundle-a", "bundle-b", "bundle-c"],
        "expected_projects": None,
        "hosts": [
            {
                "task_id": 701, "name": "bundle-a-host",
                "status": "running", "bundle_id": "bundle-a",
            },
            {
                "task_id": 702, "name": "bundle-b-host",
                "status": "attaching", "bundle_id": "bundle-b",
            },
            {
                "task_id": 703, "name": "bundle-c-host",
                "status": "queued", "bundle_id": "bundle-c",
            },
        ],
        "error": None,
    }
    optional_calls = {
        url: timeout for url, timeout in calls
        if url.endswith(("/api/aedt-pool", "/api/licenses"))
    }
    assert optional_calls == {
        "http://example.test/api/aedt-pool": 4.0,
        "http://example.test/api/licenses": 0.25,
    }


@pytest.mark.parametrize("failed_endpoint", ["pool", "license"])
def test_scheduler_optional_attach_endpoint_failure_is_section_local(
        failed_endpoint):
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 4,
                "statuses": {"running": 4},
            })
        if request.full_url.endswith(
                "/api/projects/MFT_1MW_2026v1/simulation-policy"):
            return FakeResponse({
                "project": "MFT_1MW_2026v1",
                "desired_simulations": 300,
                "effective_simulations": 300,
                "validated_concurrency_limit": 500,
                "min_desired_simulations": 0,
                "max_desired_simulations": 500,
                "policy_revision": 3,
                "scale_down_mode": "drain",
                "queued_count": 0,
                "attaching_count": 0,
                "active_count": 4,
                "solving_count": 4,
                "logical_active_count": 4,
                "control_enabled": True,
            })
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 300,
                "desired_simulations": 300,
                "effective_simulations": 300,
                "validated_concurrency_limit": 500,
                "min_desired_simulations": 0,
                "max_desired_simulations": 500,
                "policy_revision": 3,
                "queued_count": 0,
                "attaching_count": 0,
                "executing_count": 4,
                "solving_count": 4,
                "logical_active_count": 4,
            })
        if request.full_url.endswith("/api/aedt-pool"):
            if failed_endpoint == "pool":
                raise URLError("optional endpoint timed out")
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 0,
                },
                "plan": {
                    "idle_session_count": 0,
                    "hard_session_count": 0,
                },
                "sessions": [],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            if failed_endpoint == "license":
                return FakeResponse({"features": "malformed"})
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 0,
                }],
                "admission": {"snapshot_valid": True},
            })
        raise AssertionError(request.full_url)

    result = SchedulerReader(
        base_url="http://optional.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    ).snapshot()

    # Optional diagnostics never erase the authoritative summary/project state.
    assert result["connected"] is True
    assert result["control_enabled"] is True
    assert result["running"] == 4
    assert result["parallel_target"] == 300
    attach = result["aedt_attach"]
    assert len(attach["errors"]) == 1
    assert attach["node_local"]["available"] is False
    assert "/api/tasks?" in attach["node_local"]["error"]
    if failed_endpoint == "pool":
        assert attach["state"] == "pool_unavailable"
        assert attach["pool"]["available"] is False
        assert "/api/aedt-pool" in attach["pool"]["error"]
        assert attach["license"]["available"] is True
        assert attach["license"]["used"] == 0
    else:
        assert attach["state"] == "partial"
        assert attach["pool"]["available"] is True
        assert attach["license"]["available"] is False
        assert "/api/licenses" in attach["license"]["error"]


def test_scheduler_aedt_attach_reports_warm_spare_shortfall():
    def opener(request, timeout):
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 2,
                },
                "plan": {
                    "idle_session_count": 0,
                    "hard_session_count": 1,
                    "warm_spare_deficit": 2,
                    "warm_spare_start_needed": 2,
                    "warm_spare_status_reason": (
                        "warm-spare session startup is in progress"
                    ),
                    "state_counts": {"starting": 1},
                    "lease_counts": {},
                },
                # Historical terminal rows must not become pool capacity.
                "sessions": [
                    *[{"id": index, "state": "closed"} for index in range(20)],
                    {"id": 21, "state": "starting"},
                ],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 344,
                }],
                "admission": {"snapshot_valid": True},
            })
        raise AssertionError(request.full_url)

    attach = SchedulerReader(
        base_url="http://warm-spare.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    )._aedt_attach_snapshot()

    assert attach["state"] == "warming"
    assert attach["pool"]["hard_sessions"] == 1
    assert attach["pool"]["max_sessions"] == 8
    assert attach["pool"]["session_record_count"] == 21
    assert attach["pool"]["warm_spare_deficit"] == 2


def test_scheduler_aedt_attach_marks_stale_license_snapshot_degraded():
    def opener(request, timeout):
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 1,
                },
                "plan": {
                    "idle_session_count": 1,
                    "hard_session_count": 1,
                    "state_counts": {"ready": 1},
                    "lease_counts": {},
                },
                "sessions": [{"id": 1, "state": "ready"}],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 344,
                }],
                "error": "showing the last good snapshot",
                "admission": {"snapshot_valid": False},
            })
        raise AssertionError(request.full_url)

    attach = SchedulerReader(
        base_url="http://stale-license.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    )._aedt_attach_snapshot()

    assert attach["state"] == "degraded"
    assert attach["license"]["available"] is True
    assert attach["license"]["used"] == 344
    assert attach["license"]["snapshot_valid"] is False
    assert attach["errors"] == ["showing the last good snapshot"]


def test_scheduler_aedt_attach_keeps_last_good_values_on_transient_timeout():
    calls = {"pool": 0, "license": 0}

    def opener(request, timeout):
        if request.full_url.endswith("/api/aedt-pool"):
            calls["pool"] += 1
            if calls["pool"] > 1:
                raise URLError("transient pool timeout")
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 10,
                    "min_idle_aedt_sessions": 1,
                },
                "plan": {
                    "idle_session_count": 2,
                    "hard_session_count": 3,
                    "state_counts": {"ready": 2, "busy": 1},
                    "lease_counts": {"active": 3},
                },
                "sessions": [],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            calls["license"] += 1
            if calls["license"] > 1:
                raise URLError("transient license timeout")
            return FakeResponse({
                "checked_at": "2026-07-18T09:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 21,
                }],
                "admission": {"snapshot_valid": True},
            })
        if "/api/tasks?" in request.full_url:
            return FakeResponse([])
        raise AssertionError(request.full_url)

    reader = SchedulerReader(
        base_url="http://last-good.test",
        opener=opener,
        optional_timeout=0.1,
        optional_stale_ttl=300,
        ttl=0,
    )
    first = reader._aedt_attach_snapshot()
    second = reader._aedt_attach_snapshot()

    assert first["state"] == "operational"
    assert second["state"] == "degraded"
    assert second["pool"]["available"] is True
    assert second["pool"]["idle_sessions"] == 2
    assert second["pool"]["stale"] is True
    assert second["license"]["available"] is True
    assert second["license"]["used"] == 21
    assert second["license"]["stale"] is True
    assert len(second["errors"]) == 2


def test_scheduler_snapshot_coalesces_concurrent_dashboard_refreshes():
    calls = {"summary": 0}

    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            calls["summary"] += 1
            time.sleep(0.05)
            return FakeResponse({"total": 1, "statuses": {"running": 1}})
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 1,
                "queued_count": 0,
                "attaching_count": 0,
                "executing_count": 1,
            })
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": False,
                    "adapter_ready": False,
                    "validation_passed": False,
                    "operational": False,
                    "max_aedt_sessions": 0,
                    "min_idle_aedt_sessions": 0,
                },
                "plan": {"idle_session_count": 0, "hard_session_count": 0},
                "sessions": [],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop", "total": 550, "used": 0,
                }],
                "admission": {"snapshot_valid": True},
            })
        if "/api/tasks?" in request.full_url:
            return FakeResponse([])
        raise AssertionError(request.full_url)

    reader = SchedulerReader(
        base_url="http://single-flight.test",
        opener=opener,
        optional_timeout=0.2,
        ttl=60,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = list(executor.map(lambda _: reader.snapshot(), range(8)))

    assert calls["summary"] == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)


def test_status_shows_validated_experimental_hpo_without_claiming_approval(
        tmp_path):
    service = ArtifactService(tmp_path, record_runtime=False)
    status = service._status(
        data={
            "total_rows": 5477,
            "throughput_1h": 1,
            "stalled": False,
        },
        models={
            "activation_state": "unavailable",
            "trained_count": 0,
            "missing_count": 1,
            "target_count": 1,
            "models": [{"trained": False, "label": "누설 인덕턴스"}],
            "warnings": ["accepted schema-v2 model pointer is unavailable"],
        },
        nsga={
            "status": "waiting",
            "available": False,
            "candidate_count": 0,
            "round": None,
            "warnings": [],
        },
        verification={
            "stage": "NOT_STARTED",
            "counts": {"total": 0, "pending": 0, "valid": 0},
            "final": {"status": "waiting"},
            "warnings": [],
        },
        scheduler={"connected": True, "running": 1, "pending": 0},
        continuous_pipeline={
            "experimental_shadow_training": {
                "validated_running": True,
                "wave_phase": "experimental_hpo",
                "observed_strict_full_rows": 5098,
                "selected_hpo_target_count": 8,
                "completed_hpo_target_count": 4,
                "eligibility": "FEA-NOT-APPROVED",
            }
        },
    )

    model_stage = next(item for item in status["stages"] if item["key"] == "models")
    assert model_stage == {
        "key": "models",
        "label": "모델 학습",
        "state": "active",
        "detail": "후보 HPO 학습 중 · 5,098행 · 4/8 targets · 운영 미승인",
    }
    assert "accepted schema-v2 model pointer is unavailable" in status["warnings"]
    assert any(
        warning.startswith("운영 승인 모델 미등록:")
        for warning in status["warnings"]
    )


def test_status_describes_continuous_nsga_without_round_none(tmp_path):
    service = ArtifactService(tmp_path, record_runtime=False)
    status = service._status(
        data={"total_rows": 5098, "throughput_1h": 1, "stalled": False},
        models={
            "activation_state": "unavailable",
            "trained_count": 0,
            "missing_count": 0,
            "target_count": 0,
            "models": [],
            "warnings": [],
        },
        nsga={
            "status": "running",
            "available": True,
            "al_stage": "CONTINUOUS",
            "round": None,
            "active_seed_workers": 8,
            "current_model_completed_runs": 70,
            "candidate_count": 17,
            "valid_candidate_count": 0,
            "warnings": [],
        },
        verification={
            "stage": "NOT_STARTED",
            "counts": {"total": 0, "pending": 0, "valid": 0},
            "final": {"status": "waiting"},
            "warnings": [],
        },
        scheduler={"connected": True, "running": 1, "pending": 0},
        continuous_pipeline={},
    )

    nsga_stage = next(item for item in status["stages"] if item["key"] == "nsga2")
    assert nsga_stage["state"] == "active"
    assert nsga_stage["detail"] == "continuous · 8 workers · 70 runs · valid 0/17"
    assert "round None" not in nsga_stage["detail"]


def test_tier1_feedback_search_is_visible_only_after_terminal_hash_verification(
    tmp_path, monkeypatch
):
    search_root = tmp_path / "tier1" / "nsga"
    result_dir = search_root / "seed-26071801"
    result_path = result_dir / "result.json"
    model_sha = "1" * 64
    code_revision = "2" * 40
    result_payload = {
        "schema_version": "mft-tier1-corrected-search-seed-v1",
        "seed": 26071801,
        "model_manifest_sha256": model_sha,
        "nsga_code_revision": code_revision,
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "candidates": [{
            "index": 0,
            "volume_L": 740.0,
            "total_loss_W": 12_500.0,
            "decoded_params": {
                "n_core_group": 4,
                "cw1": 5.0,
                "B_design_analytic_T": 1.0,
            },
            "predictions": {
                "Llt_phys": 27.5,
                "B_mean_core": 0.8,
                "B_max_core": 1.0,
            },
            "derived_resonance": {"pred_f_res_min_screen_Hz": 21_000.0},
            "production_eligible": False,
        }],
        "next_target_fea_batch_plan": {
            "candidate_count": 3,
            "submission_performed": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    status_path = search_root / "status.json"
    status_path.write_text(json.dumps({
        "schema_version": "mft-tier1-corrected-search-monitor-v1",
        "updated_at": "2026-07-18T13:30:00+09:00",
        "model_manifest_sha256": model_sha,
        "nsga_code_revision": code_revision,
        "launch_sha256": "3" * 64,
        "population": 160,
        "max_generations": 240,
        "production_eligible": False,
        "fea_submission_approved": False,
        "submission_performed": False,
        "jobs": [
            {
                "seed": 26071801,
                "state": "completed",
                "pid": 101,
                "process_alive": True,
                "launch_command_identity_verified": True,
                "runtime_command_identity_verified": True,
                "terminal_result_verified": True,
                "completed_generations": 109,
                "feasible_pareto_count": 1,
                "stderr_size": 0,
                "output": str(result_dir),
                "result_sha256": result_sha,
            },
            {
                "seed": 26071802,
                "state": "running",
                "pid": 102,
                "process_alive": True,
                "launch_command_identity_verified": True,
                "runtime_command_identity_verified": True,
                "terminal_result_verified": None,
                "completed_generations": None,
                "feasible_pareto_count": None,
                "stderr_size": 0,
                "output": str(search_root / "seed-26071802"),
                "result_sha256": None,
            },
        ],
    }), encoding="utf-8")
    monkeypatch.setenv("MFT_TIER1_NSGA_STATUS", str(status_path))

    service = ArtifactService(
        tmp_path,
        clock=lambda: datetime.fromisoformat(
            "2026-07-18T13:30:30+09:00"
        ),
        record_runtime=False,
    )
    payload = service.nsga2()

    tier1 = payload["tier1_feedback_search"]
    assert tier1["integrity_verified"] is True
    assert tier1["completed_count"] == 1
    assert tier1["running_count"] == 1
    assert tier1["feasible_pareto_count"] == 1
    assert tier1["terminal_results_verified"] == 1
    assert tier1["lanes"][0]["process_reap_pending"] is True
    assert tier1["next_target_plan_count"] == 3
    assert tier1["production_eligible"] is False
    assert tier1["fea_submission_approved"] is False
    assert payload["tier1_active_seed_workers"] == 1
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["embedded_authenticated_result"] is True
    assert payload["candidates"][0]["production_eligible"] is False
    assert "1/2개 seed 완료" in payload["note"]

    result_path.write_text("{}", encoding="utf-8")
    tampered = ArtifactService(
        tmp_path,
        clock=lambda: datetime.fromisoformat(
            "2026-07-18T13:30:30+09:00"
        ),
        record_runtime=False,
    ).nsga2()
    assert tampered["tier1_feedback_search"]["integrity_verified"] is False
    assert tampered["candidate_count"] == 0
    assert any(
        "result hash mismatch" in warning
        for warning in tampered["warnings"]
    )


def test_tier1_stable_pointer_switches_constraint_generation_without_mixing(
    tmp_path, monkeypatch
):
    root = tmp_path / "nsga"
    root.mkdir()
    pointer_path = root / "latest.json"
    now = datetime.fromisoformat("2026-07-18T14:00:00+09:00")

    def publish(name, version, constraints):
        status_path = root / name
        payload = {
            "schema_version": "mft-tier1-corrected-search-monitor-v1",
            "updated_at": now.isoformat(),
            "constraint_version": version,
            "constraints": constraints,
            "model_manifest_sha256": "1" * 64,
            "nsga_code_revision": "2" * 40,
            "launch_sha256": "3" * 64,
            "production_eligible": False,
            "fea_submission_approved": False,
            "submission_performed": False,
            "jobs": [{
                "seed": 1,
                "state": "failed",
                "pid": 111,
                "process_alive": False,
                "launch_command_identity_verified": True,
                "runtime_command_identity_verified": False,
                "terminal_result_verified": False,
                "stderr_size": 0,
            }],
        }
        status_path.write_text(json.dumps(payload), encoding="utf-8")
        pointer_path.write_text(json.dumps({
            "schema_version": "mft-tier1-nsga-pointer-v1",
            "status": status_path.name,
            "status_sha256": hashlib.sha256(
                status_path.read_bytes()
            ).hexdigest(),
            "constraint_version": version,
            "constraints": constraints,
        }), encoding="utf-8")

    old_constraints = {
        "max_dimensions_mm": [1000, 1000, 750],
        "min_resonance_hz": 20_000,
    }
    new_constraints = {
        "max_dimensions_mm": [1200, 1200, 750],
        "min_resonance_hz": 10_000,
    }
    publish("status-20k.json", "mft-20k-v1", old_constraints)
    monkeypatch.setenv("MFT_TIER1_NSGA_POINTER", str(pointer_path))
    monkeypatch.delenv("MFT_TIER1_NSGA_STATUS", raising=False)
    service = ArtifactService(
        tmp_path, clock=lambda: now, record_runtime=False
    )
    old = service._tier1_feedback_nsga_diagnostics()
    assert old["pointer_verified"] is True
    assert old["constraint_version"] == "mft-20k-v1"
    assert old["constraints"]["min_resonance_hz"] == 20_000

    publish("status-10k.json", "mft-1200x1200x750-10k-v1", new_constraints)
    current = service._tier1_feedback_nsga_diagnostics()
    assert current["pointer_verified"] is True
    assert current["constraint_version"] == "mft-1200x1200x750-10k-v1"
    assert current["constraints"] == new_constraints
    assert "20k" not in current["constraint_version"]


def test_tier1_running_pointer_status_becomes_unavailable_when_stale(
    tmp_path, monkeypatch
):
    root = tmp_path / "nsga"
    root.mkdir()
    now = datetime.fromisoformat("2026-07-18T14:00:00+09:00")
    constraints = {
        "max_dimensions_mm": [1200, 1200, 750],
        "min_resonance_hz": 10_000,
    }
    status_path = root / "status.json"
    status_path.write_text(json.dumps({
        "schema_version": "mft-tier1-corrected-search-monitor-v1",
        "updated_at": (now - timedelta(seconds=181)).isoformat(),
        "constraint_version": "mft-1200x1200x750-10k-v1",
        "constraints": constraints,
        "model_manifest_sha256": "1" * 64,
        "nsga_code_revision": "2" * 40,
        "launch_sha256": "3" * 64,
        "production_eligible": False,
        "fea_submission_approved": False,
        "submission_performed": False,
        "jobs": [{"seed": 1, "state": "running"}],
    }), encoding="utf-8")
    pointer_path = root / "latest.json"
    pointer_path.write_text(json.dumps({
        "schema_version": "mft-tier1-nsga-pointer-v1",
        "status": status_path.name,
        "status_sha256": hashlib.sha256(
            status_path.read_bytes()
        ).hexdigest(),
        "constraint_version": "mft-1200x1200x750-10k-v1",
        "constraints": constraints,
    }), encoding="utf-8")
    monkeypatch.setenv("MFT_TIER1_NSGA_POINTER", str(pointer_path))
    monkeypatch.delenv("MFT_TIER1_NSGA_STATUS", raising=False)

    status = ArtifactService(
        tmp_path, clock=lambda: now, record_runtime=False
    )._tier1_feedback_nsga_diagnostics()
    assert status["available"] is False
    assert status["stale"] is True
    assert status["integrity_verified"] is False


def test_tier1_rolling_candidate_preview_contract_is_bounded_and_fail_closed():
    candidates = [{"rank": index} for index in range(128)]
    explicit = readers_module._rolling_candidate_preview_contract(
        {
            "pareto_count": 151,
            "candidate_preview_count": 128,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": True,
        },
        candidates,
    )
    assert explicit == {
        "safe": True,
        "explicit": True,
        "total_count": 151,
        "preview_count": 128,
        "preview_limit": 128,
        "preview_truncated": True,
    }
    assert readers_module._rolling_candidate_preview_contract(
        {"pareto_count": 128}, candidates
    )["safe"] is True
    assert readers_module._rolling_candidate_preview_contract(
        {"pareto_count": 129}, [*candidates, {"rank": 128}]
    )["safe"] is False

    for drift in (
        {"candidate_preview_count": 127},
        {"candidate_preview_limit": 127},
        {"candidate_preview_truncated": False},
    ):
        aggregate = {
            "pareto_count": 151,
            "candidate_preview_count": 128,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": True,
            **drift,
        }
        assert readers_module._rolling_candidate_preview_contract(
            aggregate, candidates
        )["safe"] is False
    assert readers_module._rolling_candidate_preview_contract(
        {
            "pareto_count": 151,
            "candidate_preview_count": 128,
        },
        candidates,
    )["safe"] is False


def test_authenticated_tier1_candidate_uses_active_robust_hard_spec(tmp_path):
    hard_spec = {
        "B_limit_T": 1.2,
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 120.0,
        "insulation_min_mm": 40.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "resonance_min_Hz": 10_000.0,
        "size_W_max_mm": 1200.0,
        "size_L_max_mm": 1200.0,
        "size_H_max_mm": 750.0,
    }
    decoded = {
        "n_core_group": 4,
        "N2_side": 1,
        "cw1": 5.0,
        "B_design_analytic_T": 0.9,
        "size_W_mm": 1180.0,
        "size_L_mm": 1170.0,
        "size_H_mm": 730.0,
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 42.0,
        "w2c_w1c_space_x": 41.0,
        "w2c_w1c_space_y": 43.0,
    }
    predictions = {
        "Llt_phys": 27.5,
        **{target: 110.0 for target in (
            "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
            *TEMPERATURE_TARGETS,
        )},
    }
    half_width = {
        "Llt_phys": 0.2,
        **{target: 5.0 for target in (
            "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
            *TEMPERATURE_TARGETS,
        )},
    }
    source_candidate = {
        "decoded_params": decoded,
        "predictions": predictions,
        "conformal_half_width": half_width,
        "derived_resonance": {"f_res_min_screen_Hz": 10_500.0},
    }
    row = {
        **decoded,
        **{f"pred_{key}": value for key, value in predictions.items()},
        "volume_L": 800.0,
        "total_loss_W": 6000.0,
    }
    service = ArtifactService(tmp_path, record_runtime=False)

    candidate = service._authenticated_tier1_candidate(
        row,
        0,
        0,
        source_candidate=source_candidate,
        hard_spec=hard_spec,
        constraint_version="t120-res10k-size1200-v1",
    )

    assert candidate["spec_status"] == "pass"
    assert candidate["constraint_source"] == (
        "authenticated_tier1_robust_hard_spec"
    )
    assert candidate["constraints"]["temperature"]["limit"] == 120.0
    assert candidate["constraints"]["temperature"]["value"] == 115.0
    assert candidate["constraints"]["resonance"]["limit"] == 10_000.0
    assert candidate["constraints"]["size_width"]["limit"] == 1200.0
    assert candidate["constraints"]["primary_thickness"]["pass"] is True

    source_candidate["predictions"]["T_max_Tx"] = 116.0
    failed = service._authenticated_tier1_candidate(
        row,
        0,
        0,
        source_candidate=source_candidate,
        hard_spec=hard_spec,
        constraint_version="t120-res10k-size1200-v1",
    )
    assert failed["constraints"]["temperature"]["value"] == 121.0
    assert failed["constraints"]["temperature"]["pass"] is False
    assert failed["spec_status"] == "fail"


def test_candidate_dimension_aliases_populate_report_and_size_constraints(
    tmp_path,
):
    service = ArtifactService(tmp_path, record_runtime=False)

    candidate = service._candidate(
        {
            "exterior_size_WxLxH_mm": "900 x 800 x 700 mm",
            "exterior_footprint_cm2": 7200.0,
        },
        0,
        0,
    )

    assert candidate["report"]["size_W_mm"] == 900.0
    assert candidate["report"]["size_L_mm"] == 800.0
    assert candidate["report"]["size_H_mm"] == 700.0
    assert candidate["report"]["footprint_cm2"] == 7200.0
    assert candidate["constraints"]["size_width"]["value"] == 900.0
    assert candidate["constraints"]["size_length"]["value"] == 800.0
    assert candidate["constraints"]["size_height"]["value"] == 700.0


def test_authenticated_tier1_candidate_replays_exact_exterior_dimensions(
    tmp_path,
):
    hard_spec = {
        "B_limit_T": 1.2,
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 120.0,
        "insulation_min_mm": 40.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "resonance_min_Hz": 10_000.0,
        "size_W_max_mm": 1200.0,
        "size_L_max_mm": 1200.0,
        "size_H_max_mm": 750.0,
    }
    decoded = {
        "N2_side": 23,
        "n_core_group": 4,
        "cw1": 5.0,
        "l1": 68.0,
        "l2": 357.5,
        "h1": 540.0,
        "w1": 523.0,
        "sl1_main_x": 474.366,
        "nwl1_main": 53.0,
        "sl1_main_y": 861.366,
        "nwb1_main_y": 71.8,
        "sl2_side_x": 148.0,
        "sl2_side_y": 606.0,
        "nwl2_side": 54.799,
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0,
        "w2c_w1c_space_y": 40.0,
    }
    predictions = {
        "Llt_phys": 27.52018052801716,
        **{
            target: 100.0
            for target in (
                "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
                *TEMPERATURE_TARGETS,
            )
        },
    }
    half_width = {
        "Llt_phys": 0.5054864202323373,
        **{
            target: 5.0
            for target in (
                "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
                *TEMPERATURE_TARGETS,
            )
        },
    }
    service = ArtifactService(tmp_path, record_runtime=False)

    candidate = service._authenticated_tier1_candidate(
        {
            "volume_L": 799.330106311568,
            "total_loss_W": 5959.578902363387,
        },
        0,
        1,
        source_candidate={
            "decoded_params": decoded,
            "predictions": predictions,
            "conformal_half_width": half_width,
            "derived_resonance": {"f_res_min_screen_Hz": 10_462.965131545465},
        },
        hard_spec=hard_spec,
        constraint_version=(
            "mft-tier1-envelope-1200x1200x750-res10k-t120all11-core4-5t-"
            "lmhalf-v3"
        ),
    )

    assert candidate["report"]["size_W_mm"] == pytest.approx(1176.598)
    assert candidate["report"]["size_L_mm"] == pytest.approx(1004.966)
    assert candidate["report"]["size_H_mm"] == pytest.approx(676.0)
    assert candidate["report"]["footprint_cm2"] == pytest.approx(
        11824.40985668
    )
    assert (
        candidate["report"]["size_W_mm"]
        * candidate["report"]["size_L_mm"]
        * candidate["report"]["size_H_mm"]
        * 1e-6
    ) == pytest.approx(799.330106311568)
    assert candidate["constraints"]["size_width"] == {
        "value": pytest.approx(1176.598),
        "limit": 1200.0,
        "margin": pytest.approx(23.402),
        "pass": True,
        "evidence": "independent_display_replay",
    }
    assert candidate["constraints"]["size_length"] == {
        "value": pytest.approx(1004.966),
        "limit": 1200.0,
        "margin": pytest.approx(195.034),
        "pass": True,
        "evidence": "independent_display_replay",
    }
    assert candidate["constraints"]["size_height"] == {
        "value": pytest.approx(676.0),
        "limit": 750.0,
        "margin": pytest.approx(74.0),
        "pass": True,
        "evidence": "independent_display_replay",
    }

    mismatched = service._authenticated_tier1_candidate(
        {
            "volume_L": 800.0,
            "total_loss_W": 5959.578902363387,
        },
        0,
        1,
        source_candidate={
            "decoded_params": decoded,
            "predictions": predictions,
            "conformal_half_width": half_width,
            "derived_resonance": {"f_res_min_screen_Hz": 10_462.965131545465},
        },
        hard_spec=hard_spec,
        constraint_version=(
            "mft-tier1-envelope-1200x1200x750-res10k-t120all11-core4-5t-"
            "lmhalf-v3"
        ),
    )
    assert "size_W_mm" not in mismatched["report"]
    assert "size_L_mm" not in mismatched["report"]
    assert "size_H_mm" not in mismatched["report"]
    assert "footprint_cm2" not in mismatched["report"]
    assert mismatched["constraints"]["size_width"] == {
        "value": None,
        "limit": 1200.0,
        "margin": None,
        "pass": True,
        "evidence": "authenticated_tier1_feasible_pareto",
    }


def test_tier1_rolling_index_updates_in_place_and_fails_closed_on_tamper(
    tmp_path, monkeypatch
):
    now = datetime.fromisoformat("2026-07-18T14:00:00+09:00")
    root = tmp_path / "rolling"
    canonical = root / "canonical"
    cohort = root / "cohorts" / "cohort-1"
    canonical.mkdir(parents=True)
    cohort.mkdir(parents=True)
    index_path = canonical / "index.json"
    model_path = canonical / "model_pointer.json"
    status_path = cohort / "status.json"
    constraint_version = "mft-1200x1200x750-res10k-t120-v2"
    hard_spec = {
        "size_W_max_mm": 1200.0,
        "size_L_max_mm": 1200.0,
        "size_H_max_mm": 750.0,
        "resonance_min_Hz": 10_000.0,
        "T_limit_C": 120.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "magnetizing_inductance_factor": 0.5,
    }
    hard_spec_sha = hashlib.sha256(json.dumps(
        hard_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    source_sha = "1" * 64
    deployment_sha = "2" * 64
    tier1_temperature_targets = [
        "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
        *TEMPERATURE_TARGETS,
    ]
    temperature_contract = {
        "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
        "robust_upper_bound_C": 120.0,
        "side_winding_conditional_targets": [
            "T_max_Rx_side", "Tprobe_Rx_side_leeward_max",
        ],
        "source": "revision_175_SURROGATE_TEMPERATURE_TARGETS",
        "target_count": len(tier1_temperature_targets),
        "targets": tier1_temperature_targets,
        "unconditional_target_count": len(tier1_temperature_targets) - 2,
    }
    model_pointer = {
        "schema_version": "mft-tier1-slurm-model-pointer-v1",
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "current": {
            "cohort_id": "cohort-1",
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
    }
    model_path.write_text(json.dumps(model_pointer), encoding="utf-8")

    def task(task_id, seed, state):
        return {
            "task_id": task_id,
            "seed": seed,
            "status": state,
            "name": f"rolling-{seed}",
            "cpus": 8,
            "memory_mb": 32768,
            "account_name": "test",
            "slurm_job_id": str(700_000 + task_id),
        }

    status = {
        "schema_version": "mft-tier1-slurm-rolling-status-v1",
        "updated_at": now.isoformat(),
        "freshness_deadline_seconds": 20,
        "healthy": True,
        "error": None,
        "cohort_id": "cohort-1",
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": deployment_sha,
        "nsga_code_revision": "3" * 40,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "controller": {"pid": 1234, "mode": "rolling-worker"},
        "temperature_constraint_contract": temperature_contract,
        "latest_tasks": [task(101, 1, "running"), task(102, 2, "queued")],
        "scheduler_task_count": 2,
        "state_counts": {"queued": 1, "running": 1},
        "rolling": {
            "enabled": True,
            "stop_condition": "explicit_operator_stop_only",
            "terminal_completion_triggers_refill": True,
            "seed_identity_count": 2,
            "active_plus_queued": 2,
            "duplicate_seed_count": 0,
        },
        "terminal_results": [],
        "aggregate_pareto": {
            "schema_version": "mft-tier1-slurm-aggregate-pareto-v1",
            "authenticated_terminal_count": 0,
            "source_candidate_count": 0,
            "unique_candidate_count": 0,
                "pareto_count": 0,
                "candidates": [],
                "near_feasible_count": 0,
                "near_feasible": [],
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
        },
    }

    def publish():
        status_path.write_text(json.dumps(status), encoding="utf-8")
        index = {
            "schema_version": "mft-tier1-slurm-rolling-index-v1",
            "active_cohort_id": "cohort-1",
            "path_containment_root": str(root),
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "temperature_constraint_contract": temperature_contract,
            "model_pointer": {
                "path": str(model_path),
                "schema_version": "mft-tier1-slurm-model-pointer-v1",
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "status": {
                "path": str(status_path),
                "schema_version": "mft-tier1-slurm-rolling-status-v1",
                "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
            },
        }
        index_path.write_text(json.dumps(index), encoding="utf-8")

    publish()
    monkeypatch.setenv("MFT_TIER1_ROLLING_INDEX", str(index_path))
    monkeypatch.delenv("MFT_TIER1_NSGA_POINTER", raising=False)
    monkeypatch.delenv("MFT_TIER1_NSGA_STATUS", raising=False)
    service = ArtifactService(
        tmp_path, clock=lambda: now, record_runtime=False
    )
    first = service._tier1_feedback_nsga_diagnostics()
    assert first["available"] is True, first["warnings"]
    assert first["integrity_verified"] is True
    assert first["running_count"] == 1
    assert first["queued_count"] == 1
    assert first["constraints"]["T_limit_C"] == 120.0
    status["aggregate_pareto"].update({
        "candidate_preview_count": 0,
        "candidate_preview_limit": 128,
        "candidate_preview_truncated": False,
    })
    publish()
    explicit_preview = service._tier1_feedback_nsga_diagnostics()
    assert explicit_preview["available"] is True
    assert explicit_preview["candidate_preview_count"] == 0
    assert explicit_preview["candidate_preview_limit"] == 128
    assert explicit_preview["candidate_preview_truncated"] is False
    assert explicit_preview["candidate_preview_contract_explicit"] is True
    status["aggregate_pareto"]["candidates"] = {}
    publish()
    malformed_inventory = service._tier1_feedback_nsga_diagnostics()
    assert malformed_inventory["available"] is False
    assert any(
        "candidate inventory is not a list" in warning
        for warning in malformed_inventory["warnings"]
    )
    status["aggregate_pareto"]["candidates"] = []
    status["aggregate_pareto"]["near_feasible"] = {}
    publish()
    malformed_near_inventory = service._tier1_feedback_nsga_diagnostics()
    assert malformed_near_inventory["available"] is False
    assert any(
        "near-feasible inventory is not a list" in warning
        for warning in malformed_near_inventory["warnings"]
    )
    status["aggregate_pareto"]["near_feasible"] = []
    publish()

    original_bounded_read = readers_module._bounded_json_bytes
    index_reads = 0

    def publish_during_first_index_read(path, max_bytes):
        nonlocal index_reads
        parsed = original_bounded_read(path, max_bytes)
        if path == index_path and index_reads == 0:
            index_reads += 1
            status["latest_tasks"] = [
                task(101, 1, "running"), task(102, 2, "running")
            ]
            status["state_counts"] = {"running": 2}
            publish()
        return parsed

    monkeypatch.setattr(
        readers_module, "_bounded_json_bytes", publish_during_first_index_read
    )
    second = service._tier1_feedback_nsga_diagnostics()
    assert second["available"] is True
    assert second["coherent_snapshot_verified"] is True
    assert second["coherent_snapshot_attempts"] == 2
    assert second["running_count"] == 2
    assert second["queued_count"] == 0

    monkeypatch.setattr(
        readers_module, "_bounded_json_bytes", original_bounded_read
    )
    status_path.write_text(
        status_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    tampered = service._tier1_feedback_nsga_diagnostics()
    assert tampered["available"] is False
    assert tampered["pointer_verified"] is False
    assert any("hash mismatch" in item for item in tampered["warnings"])


def test_tier1_rolling_authenticates_bounded_near_feasible_preview(
    tmp_path, monkeypatch
):
    now = datetime.fromisoformat("2026-07-18T14:00:00+09:00")
    root = tmp_path / "rolling-near"
    canonical = root / "canonical"
    cohort = root / "cohorts" / "cohort-near"
    harvest = cohort / "harvest"
    canonical.mkdir(parents=True)
    harvest.mkdir(parents=True)
    index_path = canonical / "index.json"
    model_path = canonical / "model_pointer.json"
    status_path = cohort / "status.json"
    constraint_version = "mft-res10k-t120-all11-v1"
    hard_spec = {
        "T_limit_C": 120.0,
        "resonance_min_Hz": 10_000.0,
    }
    hard_spec_sha = hashlib.sha256(json.dumps(
        hard_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    source_sha = "1" * 64
    deployment_sha = "2" * 64
    code_revision = "3" * 40
    temperature_targets = [
        "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
        *TEMPERATURE_TARGETS,
    ]
    temperature_contract = {
        "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
        "robust_upper_bound_C": 120.0,
        "target_count": len(temperature_targets),
        "targets": temperature_targets,
    }
    model_path.write_text(json.dumps({
        "schema_version": "mft-tier1-slurm-model-pointer-v1",
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
        "current": {
            "cohort_id": "cohort-near",
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
        },
    }), encoding="utf-8")

    terminal_results = []
    near_candidates = []
    for offset in range(65):
        task_id = 1_000 + offset
        seed = 2_000 + offset
        target_candidate = {
            "rank": 1,
            "terminal_population_index": offset,
            "target_acquisition_score": float(offset),
            "decoded_params": {"l1": 70 + offset},
            "production_eligible": False,
            "fea_submission_approved": False,
            "eligible_for_submission": False,
        }
        target_candidates = [target_candidate]
        if offset == 0:
            target_candidates.extend({
                **target_candidate,
                "terminal_population_index": 10_000 + extra,
                "target_acquisition_score": extra / 10.0,
                "decoded_params": {"l1": 10_000 + extra},
            } for extra in range(1, 4))
        result_payload = {
            "schema_version": "mft-tier1-corrected-search-seed-v1",
            "seed": seed,
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "nsga_code_revision": code_revision,
            "model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
            "automatic_promotion_allowed": False,
            "completed_generations": 2,
            "feasible_pareto_count": 0,
            "candidates": [],
            "next_target_fea_batch_plan": {
                "schema_version": "mft-tier1-next-target-fea-batch-plan-v1",
                "candidate_count": len(target_candidates),
                "candidates": target_candidates,
                "production_eligible": False,
                "fea_submission_approved": False,
                "submission_performed": False,
                "current_candidates_eligible_for_submission": False,
                "scheduler_cap_verified_before_submission": False,
            },
        }
        result_path = harvest / f"result-{task_id}.json"
        result_path.write_text(json.dumps(result_payload), encoding="utf-8")
        result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
        remote_payload = {
            "schema_version": "mft-tier1-slurm-seed-status-v1",
            "task_id": task_id,
            "seed": seed,
            "state": "completed",
            "exit_code": 0,
            "result_sha256": result_sha,
            "cohort_id": "cohort-near",
            "nsga_code_revision": code_revision,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
        remote_path = harvest / f"remote-{task_id}.json"
        remote_path.write_text(json.dumps(remote_payload), encoding="utf-8")
        remote_sha = hashlib.sha256(remote_path.read_bytes()).hexdigest()
        terminal_results.append({
            "schema_version": "mft-tier1-slurm-terminal-result-v1",
            "authenticated": True,
            "task_id": task_id,
            "seed": seed,
            "cohort_id": "cohort-near",
            "scheduler_state": "completed",
            "scheduler_exit_code": 0,
            "terminal_state": "completed",
            "constraint_version": constraint_version,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "completed_generations": 2,
            "feasible_pareto_count": 0,
            "candidate_count": 0,
            "remote_status": {
                "state": "completed",
                "local_path": str(remote_path),
                "sha256": remote_sha,
            },
            "result": {
                "local_path": str(result_path),
                "sha256": result_sha,
                "bytes": result_path.stat().st_size,
            },
        })
        near_candidates.extend({
            **candidate,
            "source_task_id": task_id,
            "source_seed": seed,
        } for candidate in target_candidates[:3])

    status = {
        "schema_version": "mft-tier1-slurm-rolling-status-v1",
        "updated_at": now.isoformat(),
        "freshness_deadline_seconds": 20,
        "healthy": True,
        "error": None,
        "cohort_id": "cohort-near",
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "source_model_manifest_sha256": source_sha,
        "deployment_model_manifest_sha256": deployment_sha,
        "nsga_code_revision": code_revision,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "controller": {"pid": 1234, "mode": "rolling-worker"},
        "temperature_constraint_contract": temperature_contract,
        "latest_tasks": [{
            "task_id": 999,
            "seed": 999,
            "status": "running",
            "name": "rolling-active",
            "cpus": 8,
            "memory_mb": 32768,
        }],
        "scheduler_task_count": 1,
        "state_counts": {"running": 1},
        "rolling": {
            "enabled": True,
            "stop_condition": "explicit_operator_stop_only",
            "terminal_completion_triggers_refill": True,
            "seed_identity_count": 1,
            "active_plus_queued": 1,
            "duplicate_seed_count": 0,
        },
        "terminal_results": terminal_results,
        "aggregate_pareto": {
            "schema_version": "mft-tier1-slurm-aggregate-pareto-v1",
            "authenticated_terminal_count": 65,
            "source_candidate_count": 0,
            "unique_candidate_count": 65,
            "pareto_count": 0,
            "candidates": [],
            "near_feasible_count": 67,
            "near_feasible": near_candidates[:64],
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
        },
    }

    def publish():
        status_path.write_text(json.dumps(status), encoding="utf-8")
        index_path.write_text(json.dumps({
            "schema_version": "mft-tier1-slurm-rolling-index-v1",
            "active_cohort_id": "cohort-near",
            "path_containment_root": str(root),
            "constraint_version": constraint_version,
            "hard_spec": hard_spec,
            "hard_spec_sha256": hard_spec_sha,
            "source_model_manifest_sha256": source_sha,
            "deployment_model_manifest_sha256": deployment_sha,
            "temperature_constraint_contract": temperature_contract,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "model_pointer": {
                "path": str(model_path),
                "schema_version": "mft-tier1-slurm-model-pointer-v1",
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "status": {
                "path": str(status_path),
                "schema_version": "mft-tier1-slurm-rolling-status-v1",
                "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
            },
        }), encoding="utf-8")

    publish()
    monkeypatch.setenv("MFT_TIER1_ROLLING_INDEX", str(index_path))
    monkeypatch.delenv("MFT_TIER1_NSGA_POINTER", raising=False)
    monkeypatch.delenv("MFT_TIER1_NSGA_STATUS", raising=False)
    service = ArtifactService(tmp_path, clock=lambda: now, record_runtime=False)
    verified = service._tier1_feedback_nsga_diagnostics()
    assert verified["available"] is True, verified["warnings"]
    assert verified["integrity_verified"] is True
    assert verified["near_feasible_count"] == 67
    assert verified["near_feasible_preview_count"] == 64
    assert verified["near_feasible_preview_truncated"] is True
    assert len(verified["near_candidate_rows"]) == 64
    assert verified["near_candidate_rows"][0]["task_id"] == 1_000
    assert verified["near_candidate_rows"][0]["seed"] == 2_000
    assert (
        verified["near_candidate_rows"][0]["candidate"]
        ["eligible_for_submission"]
        is False
    )

    # A seed process may fail while the rolling controller remains healthy.
    # The failed terminal must be hash-authenticated and counted, but it must
    # never contribute a result, Pareto row, or acquisition candidate.
    failed_terminal = status["terminal_results"][-1]
    failed_result = failed_terminal.pop("result")
    failed_terminal.pop("completed_generations")
    failed_terminal.pop("feasible_pareto_count")
    failed_terminal.pop("candidate_count")
    failure = "optimizer exit code 1"
    failed_remote_path = Path(
        failed_terminal["remote_status"]["local_path"]
    )
    failed_remote = json.loads(failed_remote_path.read_text(encoding="utf-8"))
    failed_remote.update({
        "state": "failed",
        "exit_code": 1,
        "failure": failure,
    })
    failed_remote.pop("result_sha256")
    failed_remote_path.write_text(json.dumps(failed_remote), encoding="utf-8")
    failed_terminal.update({
        "scheduler_state": "failed",
        "scheduler_exit_code": 1,
        "terminal_state": "failed",
        "failure": failure,
    })
    failed_terminal["remote_status"].update({
        "state": "failed",
        "failure": failure,
        "sha256": hashlib.sha256(
            failed_remote_path.read_bytes()
        ).hexdigest(),
    })
    status["aggregate_pareto"]["near_feasible_count"] = 66
    publish()
    verified_with_failure = service._tier1_feedback_nsga_diagnostics()
    assert verified_with_failure["available"] is True, (
        verified_with_failure["warnings"]
    )
    assert verified_with_failure["terminal_results_verified"] == 65
    assert verified_with_failure["failed_terminal_results_verified"] == 1
    assert verified_with_failure["near_feasible_count"] == 66

    failed_terminal["result"] = failed_result
    publish()
    failed_with_result = service._tier1_feedback_nsga_diagnostics()
    assert failed_with_result["available"] is False
    assert any(
        "failed terminal unexpectedly declares a result" in warning
        for warning in failed_with_result["warnings"]
    )
    failed_terminal.pop("result")

    failed_terminal["remote_status"]["failure"] = "different failure"
    publish()
    mismatched_failure = service._tier1_feedback_nsga_diagnostics()
    assert mismatched_failure["available"] is False
    assert any(
        "terminal identity mismatch" in warning
        for warning in mismatched_failure["warnings"]
    )
    failed_terminal["remote_status"]["failure"] = failure

    status["aggregate_pareto"]["near_feasible"][0][
        "target_acquisition_score"
    ] = -1.0
    publish()
    unauthenticated = service._tier1_feedback_nsga_diagnostics()
    assert unauthenticated["available"] is False
    assert unauthenticated["near_candidate_rows"] == []
    assert any(
        "near candidate 0 is unauthenticated" in warning
        for warning in unauthenticated["warnings"]
    )

    near_candidates[0]["target_acquisition_score"] = 0.0
    status["aggregate_pareto"]["near_feasible"] = near_candidates[:63]
    publish()
    short_preview = service._tier1_feedback_nsga_diagnostics()
    assert short_preview["available"] is False
    assert any(
        "aggregate Pareto identity mismatch" in warning
        for warning in short_preview["warnings"]
    )


def test_tier1_near_feasible_preview_is_compact_read_only_and_physical(
    tmp_path,
):
    service = ArtifactService(tmp_path, record_runtime=False)
    diagnostics = {
        "integrity_verified": True,
        "constraints": {
            "B_limit_T": 1.2,
            "Llt_target_uH": 27.5,
            "Llt_tol_uH": 0.55,
            "T_limit_C": 110.0,
            "resonance_min_Hz": 15_000.0,
            "size_W_max_mm": 1_200.0,
            "size_L_max_mm": 1_200.0,
            "size_H_max_mm": 750.0,
        },
        "near_candidate_rows": [{
            "source_role": "supplemental",
            "source_label": "bridge",
            "source_cohort_id": "cohort-bridge",
            "task_id": 59_477,
            "seed": 1_907_197_023,
            "result_sha256": "a" * 64,
            "candidate": {
                "terminal_population_index": 7,
                "target_acquisition_score": 0.25,
                "decoded_params_sha256": "b" * 64,
                "decoded_params": {"n_core_group": 4, "cw1": 5.0},
                "target_predictions": {"Llt_phys": 26.77886},
                "target_conformal_half_width": {"Llt_phys": 0.50549},
                "derived_resonance": {"f_res_min_screen_Hz": 13_563.26},
                "constraint_G": {
                    "Llt_robust_band": 0.67663,
                    "half_magnetizing_resonance_minimum": 1_436.74,
                    "temperature_robust_limit:T_max_Tx": 5.1197,
                    "temperature_robust_limit:T_max_core": -0.24359,
                    "analytical_flux_density_limit": -0.43579,
                    "exterior_width_limit": -0.322,
                    "exterior_length_limit": -90.842,
                    "exterior_height_limit": -42.0,
                },
                "production_eligible": False,
                "fea_submission_approved": False,
                "eligible_for_submission": False,
            },
        }],
    }

    preview = service._tier1_near_feasible_preview(diagnostics)

    assert len(preview) == 1
    candidate = preview[0]
    assert candidate["id"].startswith("near-bridge-task-59477")
    assert candidate["size_W_mm"] == pytest.approx(1_199.678)
    assert candidate["size_L_mm"] == pytest.approx(1_109.158)
    assert candidate["size_H_mm"] == pytest.approx(708.0)
    assert candidate["volume_L"] == pytest.approx(
        1_199.678 * 1_109.158 * 708.0 / 1_000_000.0
    )
    assert candidate["Llt_robust_lower_uH"] == pytest.approx(26.27337)
    assert candidate["Llt_robust_upper_uH"] == pytest.approx(27.28435)
    assert candidate["f_res_min_screen_Hz"] == pytest.approx(13_563.26)
    assert candidate["robust_max_temperature_C"] == pytest.approx(115.1197)
    assert candidate["B_design_analytic_T"] == pytest.approx(0.76421)
    assert candidate["violation_count"] == 3
    assert candidate["valid_pareto"] is False
    assert candidate["production_eligible"] is False
    assert candidate["fea_submission_approved"] is False
    assert candidate["gui_launch_eligible"] is False


def test_scheduler_simulation_policy_uses_versioned_drain_patch_and_exact_readback():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data, timeout))
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "desired_simulations": 500,
            "effective_simulations": 275,
            "validated_concurrency_limit": 500,
            "min_desired_simulations": 0,
            "max_desired_simulations": 500,
            "policy_revision": 18,
            "scale_down_mode": "drain",
            "queued_count": 200,
            "attaching_count": 5,
            "active_count": 70,
            "solving_count": 65,
            "updated_at": "2026-07-13T01:05:00+09:00",
            "repos": [{"url": "must-not-be-sent"}],
            "setup": "must-not-be-sent",
            "entrypoints": [{"path": "must-not-be-sent"}],
        })

    reader = SchedulerReader(base_url="http://example.test", opener=opener, ttl=60)
    result = reader.set_simulation_policy(500, expected_revision=17)

    assert result["parallel_target"] == 500
    assert result["effective_simulations"] == 275
    assert result["logical_active"] == 75
    assert len(calls) == 1
    assert calls[0][0].endswith("/api/projects/MFT_1MW_2026v1/simulation-policy")
    assert calls[0][1] == "PATCH"
    assert json.loads(calls[0][2].decode("utf-8")) == {
        "desired_simulations": 500,
        "expected_revision": 17,
        "scale_down_mode": "drain",
    }


def test_scheduler_simulation_policy_rejects_invalid_value_before_request():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise AssertionError("invalid target must not reach scheduler")

    reader = SchedulerReader(base_url="http://example.test", opener=opener)
    for invalid in (601, -1, 1.5, True, "300"):
        with pytest.raises(ValueError):
            reader.set_simulation_policy(invalid, expected_revision=1)
    with pytest.raises(ValueError):
        reader.set_simulation_policy(300, expected_revision=None)
    assert calls == []


def test_scheduler_simulation_policy_maps_revision_conflict():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 409, "conflict", {}, None)

    reader = SchedulerReader(base_url="http://example.test", opener=opener)
    with pytest.raises(SimulationPolicyConflict):
        reader.set_simulation_policy(500, expected_revision=17)


def test_scheduler_disables_control_when_runtime_lacks_exact_live_count_fields():
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 9,
                "statuses": {"queued": 3, "attaching": 2, "running": 4},
            })
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 300,
        })

    result = SchedulerReader(
        base_url="http://old-runtime.test", opener=opener, ttl=0
    ).snapshot()

    assert result["connected"] is True
    assert result["control_enabled"] is False
    assert result["parallel_target"] is None
    assert result["logical_active"] == 6
    assert "live counts" in result["project_error"]


@pytest.mark.parametrize(
    "state_payload",
    (
        {"policy": {"target": 275}},
        {"generation": {"identity": {"project_concurrency_target": 275}}},
    ),
    ids=("policy-target", "production-generation-identity"),
)
def test_refill_controller_reader_returns_latest_tick_and_state_target(
    tmp_path, monkeypatch, state_payload
):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            (
                json.dumps({"action": "older_tick"}),
                json.dumps({
                    "action": "rolling_refill_complete",
                    "active_project_tasks_before": 271,
                    "accepted_or_reconciled_count": 4,
                    "generation": {"id": "restart-v3-1234567890abcdef"},
                }),
                "",
            )
        ),
        encoding="utf-8",
    )
    tick_timestamp = 1_784_000_000
    os.utime(log_path, (tick_timestamp, tick_timestamp))
    monkeypatch.setenv("MFT_CONTROLLER_STATE_PATH", str(state_path))
    monkeypatch.setenv("MFT_CONTROLLER_LOG_PATH", str(log_path))

    result = RefillControllerReader().snapshot()

    assert result == {
        "available": True,
        "last_tick_at": result["last_tick_at"],
        "action": "rolling_refill_complete",
        "active_project_tasks_before": 271,
        "accepted_or_reconciled_count": 4,
        "generation_id": "restart-v3-1234567890abcdef",
        "concurrency_target": 275,
    }
    assert datetime.fromisoformat(result["last_tick_at"]).timestamp() == tick_timestamp


def test_refill_controller_reader_missing_files_is_unavailable(tmp_path):
    result = RefillControllerReader(
        tmp_path / "missing-state.json", tmp_path / "missing.log"
    ).snapshot()

    assert result == {"available": False}


def test_refill_controller_reader_rejects_malformed_last_log_line(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    log_path.write_text(
        json.dumps({"action": "no_refill_needed"}) + "\n{malformed\n\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_rejects_malformed_state(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text("{malformed", encoding="utf-8")
    log_path.write_text(
        json.dumps({"action": "no_refill_needed"}) + "\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_requires_tick_action(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    log_path.write_text(
        json.dumps({"active_project_tasks_before": 300}) + "\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_accepts_tick_larger_than_64_kib(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    tick_line = json.dumps({
        "action": "pooled_bundle_pending",
        "active_project_tasks_before": 298,
        "accepted_or_reconciled_count": 0,
        "padding": "x" * 70_000,
    })
    assert 64 * 1024 < len(tick_line.encode("utf-8")) < 128 * 1024
    log_path.write_text(
        json.dumps({"action": "older_tick"}) + "\n" + tick_line + "\n",
        encoding="utf-8",
    )

    result = RefillControllerReader(state_path, log_path).snapshot()

    assert result["available"] is True
    assert result["action"] == "pooled_bundle_pending"
    assert result["active_project_tasks_before"] == 298
    assert result["accepted_or_reconciled_count"] == 0


def test_dashboard_survives_missing_optional_artifacts(tmp_path):
    from regression_260707.monitoring.readers import ArtifactService
    from .conftest import DummyRefillController, DummyScheduler, FIXED_NOW

    service = ArtifactService(
        tmp_path / "empty",
        scheduler=DummyScheduler(),
        refill_controller=DummyRefillController(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )
    payload = service.dashboard()
    assert payload["data"]["total_rows"] == 0
    assert payload["models"]["trained_count"] == 0
    assert payload["nsga2"]["available"] is False
    assert payload["verification"]["final"]["status"] == "waiting"
    timing = payload["data"]["simulation_timing"]
    assert timing["available"] is False
    assert timing["window_rows"] == 0
    assert all(
        stage["sample_count"] == 0
        and stage["mean_seconds"] is None
        and stage["median_seconds"] is None
        for stage in timing["stages"].values()
    )
