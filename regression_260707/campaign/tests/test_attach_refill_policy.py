from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ROOT = CAMPAIGN_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (CAMPAIGN_ROOT, REGRESSION_ROOT, VERIFY_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import attach_refill_policy as policy_module
import attach_aware_refill_controller as controller_module
import scheduler_client


SHA = {
    "solver": "1" * 40,
    "library": "2" * 40,
    "selector": "3" * 40,
    "controller": "4" * 40,
    "validation": "5" * 40,
    "runtime": "6" * 40,
    "canary": "7" * 40,
    "validation_scheduler": "8" * 40,
    "timeout_validation_scheduler": "9" * 40,
}


def provenance(**changes) -> policy_module.RevisionProvenance:
    values = {
        "solver_revision": SHA["solver"],
        "library_revision": SHA["library"],
        "data_contract_revision": "mft1mw-1k101-native-lamination-kf0p85-v3",
        "physics_data_revision": "mft1mw-1k101-native-lamination-kf0p85-v3",
        "core_lamination_factor": 0.85,
        "scheduler_selector_revision": SHA["selector"],
        "scheduler_runtime_revision": SHA["runtime"],
        "controller_base_revision": SHA["controller"],
        "attach_canary_revision": SHA["canary"],
        "attach_validation_revision": SHA["validation"],
        "attach_validation_scheduler_revision": SHA["validation_scheduler"],
        "attach_timeout_validation_scheduler_revision": (
            SHA["timeout_validation_scheduler"]
        ),
    }
    values.update(changes)
    return policy_module.RevisionProvenance(**values)


def attach_policy(**changes) -> policy_module.AttachRefillPolicy:
    values = {
        "primary_backend": "pooled",
        "project_concurrency_target": 300,
        "max_aedt_sessions": 150,
        "projects_per_aedt": 2,
        "validated_projects_per_aedt": 2,
        "provenance": provenance(),
        "pooled_fraction": 1.0,
    }
    values.update(changes)
    return policy_module.AttachRefillPolicy(**values)


def candidates(count: int) -> list[policy_module.ProjectCandidate]:
    return [
        policy_module.ProjectCandidate(
            name=f"mft-next-{index:04d}",
            params_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
        )
        for index in range(count)
    ]


def test_pooled_target_300_requires_enough_aedt_sessions():
    with pytest.raises(ValueError, match="cannot cover"):
        attach_policy(max_aedt_sessions=149)


def test_projects_per_aedt_is_generic_but_evidence_bounded():
    future = attach_policy(
        max_aedt_sessions=100,
        projects_per_aedt=3,
        validated_projects_per_aedt=3,
    )
    assert future.max_pooled_projects == 300
    assert policy_module.desired_aedt_sessions(300, 3) == 100

    with pytest.raises(ValueError, match="validation evidence"):
        attach_policy(
            max_aedt_sessions=100,
            projects_per_aedt=3,
            validated_projects_per_aedt=2,
        )


def test_provenance_physics_identity_is_validated_and_changes_digest():
    base = provenance()
    assert base.as_dict()["physics_data_revision"] == (
        "mft1mw-1k101-native-lamination-kf0p85-v3"
    )
    assert base.as_dict()["core_lamination_factor"] == 0.85
    assert provenance(
        physics_data_revision="mft1mw-1k101-native-lamination-kf0p85-v4"
    ).digest != base.digest
    assert provenance(core_lamination_factor=0.70).digest != base.digest

    for invalid in (True, 0.0, -0.1, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="core_lamination_factor"):
            provenance(core_lamination_factor=invalid)
    with pytest.raises(ValueError, match="physics_data_revision"):
        provenance(physics_data_revision=260713)
    with pytest.raises(ValueError, match="physics_data_revision"):
        provenance(physics_data_revision="revision with spaces")


def test_pin_candidate_params_copies_and_overrides_both_physics_fields():
    source = {
        "N1_main": 12,
        "core_lamination_factor": 0.70,
        "physics_data_revision": "stale-revision",
    }
    pinned = policy_module.pin_candidate_params(source, provenance())

    assert pinned is not source
    assert source["core_lamination_factor"] == 0.70
    assert source["physics_data_revision"] == "stale-revision"
    assert pinned["core_lamination_factor"] == 0.85
    assert pinned["physics_data_revision"] == (
        "mft1mw-1k101-native-lamination-kf0p85-v3"
    )


def test_pooled_fraction_is_bounded_normalized_and_part_of_policy_digest():
    standalone_only = attach_policy(pooled_fraction=0)
    pooled = attach_policy(pooled_fraction=1)

    assert standalone_only.pooled_fraction == 0.0
    assert pooled.pooled_fraction == 1.0
    assert standalone_only.digest != pooled.digest
    for invalid in (True, -0.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="pooled_fraction"):
            attach_policy(pooled_fraction=invalid)


def test_production_policy_pins_restart_v3_with_zero_pooled_fraction():
    path = CAMPAIGN_ROOT / "attach_refill_policy_canary_n2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance_payload = payload.pop("provenance")
    payload.pop("schema")
    policy = policy_module.AttachRefillPolicy(
        provenance=policy_module.RevisionProvenance(**provenance_payload),
        **payload,
    )

    assert policy.primary_backend == "pooled"
    assert policy.pooled_fraction == 0.0
    assert policy.project_concurrency_target == 300
    assert policy.projects_per_aedt == 2
    assert policy.validated_projects_per_aedt == 2
    assert policy.provenance.solver_revision == (
        "4f585b0540dbe3b2828f991024fdb9f1f2d23b8b"
    )
    assert policy.provenance.library_revision == (
        "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
    )
    assert policy.provenance.physics_data_revision == (
        "mft1mw-1k101-native-lamination-kf0p85-v3"
    )
    assert policy.provenance.data_contract_revision == (
        policy.provenance.physics_data_revision
    )
    assert policy.provenance.core_lamination_factor == 0.85


def test_bundle_expected_rows_equals_logical_projects_not_desktop_count():
    policy = attach_policy()
    bundles = policy_module.make_refill_bundles(candidates(5), policy)

    assert [bundle.expected_rows for bundle in bundles] == [2, 2, 1]
    assert sum(bundle.expected_rows for bundle in bundles) == 5
    assert all(bundle.backend == "pooled" for bundle in bundles)
    assert all(bundle.scheduling_profile == "fea_bursty" for bundle in bundles)


def test_standalone_and_pooled_both_use_fea_bursty_independently():
    policy = attach_policy(primary_backend="standalone", max_aedt_sessions=1)
    bundles = policy_module.make_refill_bundles(candidates(3), policy)

    assert len(bundles) == 3
    assert all(bundle.backend == "standalone" for bundle in bundles)
    assert all(bundle.expected_rows == 1 for bundle in bundles)
    assert all(bundle.scheduling_profile == "fea_bursty" for bundle in bundles)


def test_failed_pooled_bundle_falls_back_only_for_missing_rows_without_cancel():
    policy = attach_policy()
    bundle = policy_module.make_refill_bundles(candidates(2), policy)[0]
    decision = policy_module.reconcile_failed_bundle(
        bundle,
        task_ids=[101, 102],
        task_statuses={101: "completed", 102: "failed"},
        accepted_row_task_ids=[101],
        policy=policy,
    )

    assert decision["action"] == "submit_standalone_fallback"
    assert decision["fallback_backend"] == "standalone"
    assert decision["fallback_expected_rows"] == 1
    assert decision["cancel_task_ids"] == []
    assert decision["affects_other_bundles"] is False


def test_running_sibling_prevents_early_fallback():
    policy = attach_policy()
    bundle = policy_module.make_refill_bundles(candidates(2), policy)[0]
    decision = policy_module.reconcile_failed_bundle(
        bundle,
        task_ids=[101, 102],
        task_statuses={101: "failed", 102: "running"},
        accepted_row_task_ids=[],
        policy=policy,
    )

    assert decision["terminal"] is False
    assert decision["action"] == "wait"
    assert decision["cancel_task_ids"] == []


def test_task_options_include_target_aware_provenance_and_generic_n():
    policy = attach_policy(
        max_aedt_sessions=100,
        projects_per_aedt=3,
        validated_projects_per_aedt=3,
    )
    bundle = policy_module.make_refill_bundles(candidates(3), policy)[0]
    options = policy_module.task_submission_options(
        bundle, policy, candidate_index=1
    )

    assert options["aedt_backend"] == "pooled"
    assert options["scheduling_profile"] == "fea_bursty"
    assert options["expected_rows"] == 1
    assert options["bundle_expected_rows"] == 3
    assert options["dedupe_scope"] == provenance().digest
    assert options["submission_env"]["MFT_AEDT_SHARED_CANARY"] == "1"
    assert options["submission_env"]["MFT_PROJECTS_PER_AEDT"] == "3"
    assert (
        options["submission_env"]["MFT_DATA_CONTRACT_REVISION"]
        == "mft1mw-1k101-native-lamination-kf0p85-v3"
    )
    assert options["submission_env"]["MFT_PHYSICS_DATA_REVISION"] == (
        "mft1mw-1k101-native-lamination-kf0p85-v3"
    )
    assert options["submission_env"]["MFT_CORE_LAMINATION_FACTOR"] == "0.85"


class _Response:
    status_code = 201

    def json(self):
        return {"task_id": 901}


def test_scheduler_submission_emits_backend_and_scoped_provenance():
    profile = {"param_overrides": {}, "cli_flags": "--thermal", "timeout_seconds": 99}
    captured = {}

    def post(_url, *, json, timeout):
        captured.update(json)
        assert timeout == 20
        return _Response()

    with (
        mock.patch.object(scheduler_client, "campaign_mutation_lock_is_held", return_value=True),
        mock.patch.object(scheduler_client, "reconcile_task_id", return_value=None),
        mock.patch.object(
            scheduler_client,
            "live_project_submission_snapshot",
            return_value={"project_submission_slots": 1},
        ),
        mock.patch.object(scheduler_client.requests, "post", side_effect=post),
    ):
        task_id = scheduler_client._submit_verification_locked(
            "mft-next-0001",
            "mft-work",
            {"x": 1},
            profile,
            solver_revision=SHA["solver"],
            library_revision=SHA["library"],
            aedt_backend="pooled",
            scheduling_profile="fea_bursty",
            submission_env={
                "MFT_AEDT_SHARED_CANARY": "1",
                "MFT_DATA_CONTRACT_REVISION": "strict-v2",
            },
            dedupe_scope="a" * 64,
        )

    assert task_id == 901
    assert captured["aedt_backend"] == "pooled"
    assert captured["scheduling_profile"] == "fea_bursty"
    assert captured["dedupe_key"].endswith(":scope-" + "a" * 64)
    assert "export MFT_AEDT_SHARED_CANARY=1" in captured["command"]
    assert "MFT_SUBMISSION_PROVENANCE" in captured["command"]


def test_scoped_dedupe_changes_only_when_provenance_changes():
    profile = {"param_overrides": {}}
    plain = scheduler_client.verification_dedupe_key(
        "x", {"p": 1}, profile, SHA["solver"], SHA["library"]
    )
    scoped = scheduler_client.verification_dedupe_key(
        "x", {"p": 1}, profile, SHA["solver"], SHA["library"],
        dedupe_scope="f" * 64,
    )

    assert scoped != plain
    assert scoped.startswith(plain)


def test_scheduler_parameter_dedupe_changes_with_each_physics_pin():
    profile = {"param_overrides": {}}
    base_provenance = provenance()
    revision_provenance = provenance(
        physics_data_revision="mft1mw-1k101-native-lamination-kf0p85-v4"
    )
    kf_provenance = provenance(core_lamination_factor=0.70)

    def dedupe(item):
        params = policy_module.pin_candidate_params({"N1_main": 12}, item)
        return scheduler_client.verification_dedupe_key(
            "mft-next-physics", params, profile, SHA["solver"], SHA["library"]
        )

    assert len({
        dedupe(base_provenance),
        dedupe(revision_provenance),
        dedupe(kf_provenance),
    }) == 3


def ready_pool_status(policy):
    return {
        "enabled": True,
        "operational": True,
        "validation_passed": True,
        "max_aedt_sessions": policy.max_aedt_sessions,
        "target_project_concurrency": policy.project_concurrency_target,
        "projects_per_aedt": policy.projects_per_aedt,
    }


def test_coordinator_refills_project_deficit_not_desktop_deficit():
    policy = attach_policy()
    plan = controller_module.AttachAwareRefillCoordinator(policy).plan_cycle(
        active_project_tasks=294,
        candidates=candidates(6),
        pool_status=ready_pool_status(policy),
    )

    assert plan["logical_project_deficit"] == 6
    assert plan["scheduler_task_count"] == 6
    assert plan["expected_rows"] == 6
    assert plan["desired_aedt_sessions"] == 3
    assert plan["bundle_expected_rows"] == [2, 2, 2]
    assert plan["selected_backend"] == "pooled"
    assert plan["cancel_task_ids"] == []
    assert plan["mass_cancel_authorized"] is False


def test_fractional_pool_admission_does_not_starve_one_slot_cycles():
    policy = attach_policy(pooled_fraction=0.25)
    selected = [
        controller_module.AttachAwareRefillCoordinator(policy).plan_cycle(
            active_project_tasks=299,
            candidates=[candidate],
            pool_status=ready_pool_status(policy),
        )["selected_backend"]
        for candidate in candidates(64)
    ]

    assert "pooled" in selected
    assert "standalone" in selected


def test_pool_gate_failure_uses_existing_standalone_path_without_waiting():
    policy = attach_policy()
    status = ready_pool_status(policy)
    status["operational"] = False
    plan = controller_module.AttachAwareRefillCoordinator(policy).plan_cycle(
        active_project_tasks=298,
        candidates=candidates(2),
        pool_status=status,
    )

    assert plan["selected_backend"] == "standalone"
    assert plan["backend_reason"] == "pool_unavailable_standalone_fallback"
    assert plan["desired_aedt_sessions"] == 0
    assert plan["bundle_expected_rows"] == [1, 1]


def test_failed_bundle_fallback_manifest_is_standalone_and_scoped_to_missing():
    policy = attach_policy()
    coordinator = controller_module.AttachAwareRefillCoordinator(policy)
    bundle = policy_module.make_refill_bundles(candidates(2), policy)[0]
    result = coordinator.plan_failed_bundle_fallback(
        bundle,
        task_ids=[201, 202],
        task_statuses={201: "completed", 202: "failed"},
        accepted_row_task_ids=[201],
    )

    assert result["action"] == "submit_standalone_fallback"
    assert len(result["fallback_bundles"]) == 1
    assert result["fallback_bundles"][0]["backend"] == "standalone"
    assert result["fallback_bundles"][0]["fallback_of"] == bundle.bundle_id
    assert result["fallback_bundles"][0]["expected_rows"] == 1
    assert (
        result["fallback_bundles"][0]["candidates"][0]["name"]
        != bundle.candidates[1].name
    )
    assert (
        result["fallback_bundles"][0]["candidates"][0]["params_sha256"]
        == bundle.candidates[1].params_sha256
    )
    assert result["fallback_submission_options"][0]["aedt_backend"] == "standalone"
    assert result["cancel_task_ids"] == []
