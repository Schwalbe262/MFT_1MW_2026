import json
from pathlib import Path
import re

from regression_260707.quality_contract import (
    PHYSICS_EQUIVALENT_SOLVER_REVISIONS,
)


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "q22_physics_compatibility.json"
)


def test_q22_manifest_matches_exact_directional_quality_contract():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pins = manifest["pins"]
    contract = manifest["quality_contract"]

    expected = pins["existing_training_cohort_solver_revision"]
    approved = set(contract["approved_actual_solver_revisions_added_for_q22"])
    assert contract["expected_solver_revision"] == expected
    assert approved == {
        pins["proven_runtime_solver_revision"],
        pins["campaign_solver_revision"],
    }
    assert approved.issubset(PHYSICS_EQUIVALENT_SOLVER_REVISIONS[expected])
    assert contract["directional"] is True
    assert contract["accept_arbitrary_descendants"] is False
    assert contract["accept_abbreviated_sha"] is False
    for revision in (
        expected,
        *approved,
        pins["pyaedt_library_revision"],
        pins["scheduler_package_commit"],
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_q22_manifest_records_three_project_proof_and_identical_surface():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    evidence = manifest["q21b_runtime_evidence"]
    surface = manifest["runtime_surface_attestation"]

    assert evidence["session_id"] == 536
    assert [task["task_id"] for task in evidence["tasks"]] == [
        41796, 41797, 41798,
    ]
    assert all(
        task["terminal_status"] == "completed" and task["exit_code"] == 0
        for task in evidence["tasks"]
    )
    assert surface["base_revision"] == manifest["pins"][
        "proven_runtime_solver_revision"
    ]
    assert surface["candidate_revision"] == manifest["pins"][
        "campaign_solver_revision"
    ]
    paths = {
        item["path"]: item
        for item in surface["required_identical_objects"]
    }
    assert set(paths) == {
        "run_simulation_260706.py",
        "module",
        "regression_260707/campaign/feeder.py",
        "regression_260707/verify/profiles/standard.json",
    }
    assert all(
        item["base_object_sha1"] == item["candidate_object_sha1"]
        for item in paths.values()
    )
    assert {
        item["path"] for item in surface["complete_name_status_diff"]
    } == {
        "regression_260707/campaign/collect_wave.py",
        "regression_260707/quality_contract.py",
        "regression_260707/test_pipeline_completion.py",
        "regression_260707/verify/scheduler_client.py",
    }
