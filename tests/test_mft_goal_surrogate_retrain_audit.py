from __future__ import annotations

import json
from pathlib import Path

from module.mft_goal_20260726_contract import (
    GOAL_G0_MODEL_TARGETS,
    fixed_identity_expectations,
)
from tools import mft_goal_strict_al_ingest as strict_al
from tools import mft_goal_surrogate_retrain_audit as audit


def _documentary_result() -> dict:
    result = {
        **fixed_identity_expectations(),
        **strict_al.STANDARD_MODE,
        "physics_data_revision": audit.EXPECTED_PHYSICS_DATA_REVISION,
        "N1": 6,
        **{target: 1.0 for target in GOAL_G0_MODEL_TARGETS},
    }
    return {
        "schema_version": "mft-deadline-design-fea-result-v1",
        "task_id": 92_015,
        "result": result,
    }


def test_documentary_result_is_never_promoted(tmp_path: Path):
    payload = _documentary_result()
    path = tmp_path / "standard-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    record = audit.classify_documentary_result(
        path,
        payload,
        physics_data_revision=audit.EXPECTED_PHYSICS_DATA_REVISION,
    )

    assert record is not None
    assert record["fixed_identity_exact"] is True
    assert record["retained_standard_mode_exact"] is True
    assert record["all_25_targets_finite"] is True
    assert record["strict_collection_admissible"] is False
    assert "unsupported_collection_schema" in record["rejection_codes"]
    assert (
        "authenticated_source_task_lineage_absent"
        in record["rejection_codes"]
    )


def test_historical_keep_project_zero_and_missing_pad_metadata_are_reported(
    tmp_path: Path,
):
    payload = _documentary_result()
    del payload["result"]["thermal_pad_conductivity_W_mK"]
    payload["result"]["keep_project"] = 0
    path = tmp_path / "standard-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    record = audit.classify_documentary_result(
        path,
        payload,
        physics_data_revision=audit.EXPECTED_PHYSICS_DATA_REVISION,
    )

    assert record is not None
    assert record["fixed_identity_exact"] is False
    assert (
        "thermal_pad_conductivity_W_mK"
        in record["fixed_identity_mismatch_fields"]
    )
    assert record["retained_standard_mode_exact"] is False
    assert "keep_project" in record["standard_mode_mismatch_fields"]


def test_quality_summary_fails_closed_on_21_of_25_targets(tmp_path: Path):
    missing = {
        "T_max_Tx",
        "T_max_Rx_main",
        "T_max_Rx_side",
        "T_max_core",
    }
    targets = {
        target: {"passed": True}
        for target in GOAL_G0_MODEL_TARGETS
        if target not in missing
    }
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "dataset_sha256": audit.DEFAULT_BASE_SHA256,
                "strict_full_rows": audit.DEFAULT_BASE_ROWS,
                "targets": targets,
                "capacitance_recovery": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    summary = audit.summarize_quality_status(
        path, expected_dataset_sha256=audit.DEFAULT_BASE_SHA256
    )

    assert summary["observed_target_count"] == 21
    assert summary["expected_target_count"] == 25
    assert set(summary["missing_targets"]) == missing
    assert summary["reported_passed"] is True
    assert summary["goal_25_target_quality_passed"] is False


def test_discovery_is_bounded_to_collection_and_result_names(tmp_path: Path):
    result = tmp_path / "standard-result.json"
    result.write_text("{}", encoding="utf-8")
    collection = tmp_path / "collection.json"
    collection.write_text("{}", encoding="utf-8")
    ignored = tmp_path / "plan.json"
    ignored.write_text("{}", encoding="utf-8")

    paths, facts = audit.discover_json_artifacts([tmp_path])

    assert paths == [collection, result]
    assert facts["candidate_json_files"] == 2
