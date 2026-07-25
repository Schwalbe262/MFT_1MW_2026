from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import mft_goal_fea_handoff as production
from tools import mft_goal_safe_refill as refill
from tools import mft_goal_terminal_success_watcher as watcher


def _candidate(logical: int = 96223) -> dict[str, Any]:
    return {
        "logical_authority_task_id": logical,
        "retry_generation": "timeout12h-r3",
        "candidate_physics_sha256": "a" * 64,
        "candidate_stem": "a" * 12,
        "plan": {"path": "source.json", "sha256": "b" * 64, "size_bytes": 1},
        "plan_payload_sha256": "c" * 64,
        "task_name": f"mft-timeout12h-l{logical}-{'a' * 12}",
        "dedupe_key": "sealed-dedupe",
        "account_name": refill.TARGET_ACCOUNT,
        "prospective_grid_gb": 20.0,
        "fresh_grid_output_bytes": 20 * 1024**3,
        "claim_key": "sealed-claim",
        "claim_directory": "absent-claim",
        "fixed_identity_attestation_sha256": refill.FIXED_IDENTITY_SHA256,
    }


def _plan(root: Path) -> dict[str, Any]:
    return refill._sealed(
        {
            "schema_version": refill.PLAN_SCHEMA,
            "output_root": str(root),
            "scheduler_url": refill.SCHEDULER_URL,
            "watcher_plan": {
                "path": str(root / "watch_plan.json"),
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
            "cutover_receipt": {
                "path": str(root / "cutover.json"),
                "sha256": refill.TARGET_CUTOVER_SHA256,
                "size_bytes": 1,
            },
            "watcher_extension_directory": str(root / "extensions"),
            "candidates": [_candidate()],
        }
    )


def _capacity(ready: int = 1) -> dict[str, Any]:
    return {
        "ready_fit_slots": ready,
        "memory_pressure_state": "ok",
        "queue_state": "ready" if ready else "opening",
    }


def _storage(effective_free: float = 60.0) -> dict[str, Any]:
    return {
        "schema_version": "mft-goal-safe-refill-gpfs-probe-v1",
        "limiting_quota": {
            "raw_free_gb": effective_free + 2.0,
            "in_doubt_gb": 2.0,
            "effective_free_gb": effective_free,
        },
    }


def _evaluation(selected: int | None, eligible: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "mft-goal-safe-refill-gate-evaluation-v1",
        "selected_logical_authority_task_id": selected,
        "candidates": [
            {
                **_candidate(),
                "eligible": eligible,
                "gates": {"all": eligible},
            }
        ],
    }


def test_evaluate_requires_account_zero_capacity_and_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refill, "_claim_state", lambda _candidate: "unsubmitted")
    active = {
        "task_id": 7,
        "name": "other-fea",
        "dedupe_key": "other",
        "account_name": refill.TARGET_ACCOUNT,
        "aedt_backend": "standalone",
    }
    result = refill._evaluate(
        _plan(tmp_path),
        task_reader=lambda **_kwargs: [active],
        capacity_reader=lambda **_kwargs: _capacity(ready=0),
        storage_reader=lambda _plan: _storage(effective_free=25.0),
        validate_cutover=False,
    )
    row = result["candidates"][0]
    assert row["eligible"] is False
    assert row["gates"]["account_active_fea_zero"] is False
    assert row["gates"]["scheduler_ready_fit_slots_positive"] is False
    assert row["gates"]["fresh_gpfs_envelope_passed"] is False


def test_evaluate_passes_only_with_all_five_live_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refill, "_claim_state", lambda _candidate: "unsubmitted")
    result = refill._evaluate(
        _plan(tmp_path),
        task_reader=lambda **_kwargs: [],
        capacity_reader=lambda **_kwargs: _capacity(ready=1),
        storage_reader=lambda _plan: _storage(effective_free=31.0),
        validate_cutover=False,
    )
    row = result["candidates"][0]
    assert row["eligible"] is True
    assert result["selected_logical_authority_task_id"] == 96223
    assert (
        row["fresh_gpfs_remaining_after_in_doubt_prospective_floor_gb"]
        == pytest.approx(1.0)
    )


def _patch_cycle_authorities(
    monkeypatch: pytest.MonkeyPatch, plan: dict[str, Any]
) -> None:
    monkeypatch.setattr(refill, "load_plan", lambda _path: plan)
    monkeypatch.setattr(
        refill,
        "authenticate_extension_authority",
        lambda *_args, **_kwargs: {"authorized": True},
    )


def test_dry_run_never_calls_submitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = production._write_immutable_json(
        tmp_path / "safe_refill_plan.json", {"plan": True}
    )
    authority_path = production._write_immutable_json(
        tmp_path / "authority.json", {"authority": True}
    )
    plan = _plan(tmp_path)
    _patch_cycle_authorities(monkeypatch, plan)
    monkeypatch.setattr(
        refill, "_evaluate", lambda *_args, **_kwargs: _evaluation(96223)
    )
    called = {"submit": 0}

    def submitter(**_kwargs: Any) -> Path:
        called["submit"] += 1
        raise AssertionError("dry-run must never submit")

    state = refill.cycle(
        refill_plan_path=plan_path,
        extension_authority_path=authority_path,
        authorize_submit=False,
        submitter=submitter,
    )
    assert state["action"] == "dry_run_would_submit"
    assert state["scheduler_post_calls_this_cycle"] == 0
    assert called["submit"] == 0


def test_live_cycle_revalidates_and_posts_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = production._write_immutable_json(
        tmp_path / "safe_refill_plan.json", {"plan": True}
    )
    authority_path = production._write_immutable_json(
        tmp_path / "authority.json", {"authority": True}
    )
    source_path = production._write_immutable_json(
        tmp_path / "source.json", {"source": True}
    )
    plan = _plan(tmp_path)
    plan["candidates"][0]["plan"] = production._file_record(source_path)
    _patch_cycle_authorities(monkeypatch, plan)
    calls = {"evaluate": 0, "submit": 0, "guard": 0}

    def evaluate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls["evaluate"] += 1
        return _evaluation(96223)

    def submitter(
        *, output: Path, additional_pre_submit_guard: Any, **_kwargs: Any
    ) -> Path:
        additional_pre_submit_guard()
        calls["guard"] += 1
        calls["submit"] += 1
        return production._write_immutable_json(output, {"submission": True})

    monkeypatch.setattr(refill, "_evaluate", evaluate)
    monkeypatch.setattr(
        refill.timeout12h,
        "_load_plan",
        lambda _path: ({"plan": True}, {}, {}, {}),
    )
    monkeypatch.setattr(
        refill.timeout12h,
        "load_submission_for_probe",
        lambda *_args, **_kwargs: {
            "scheduler_strict_node_contract": {
                "scheduler_mutation_performed": True
            }
        },
    )

    def ensure_extension(**kwargs: Any) -> Path:
        path = kwargs["paths"]["extension"]
        return production._write_immutable_json(path, {"extension": True})

    monkeypatch.setattr(refill, "_ensure_extension", ensure_extension)
    state = refill.cycle(
        refill_plan_path=plan_path,
        extension_authority_path=authority_path,
        authorize_submit=True,
        submitter=submitter,
    )
    assert calls == {"evaluate": 3, "submit": 1, "guard": 1}
    assert state["scheduler_post_calls_this_cycle"] == 1
    assert state["scheduler_cancel_calls_this_cycle"] == 0


def test_immediate_guard_failure_prevents_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = production._write_immutable_json(
        tmp_path / "safe_refill_plan.json", {"plan": True}
    )
    authority_path = production._write_immutable_json(
        tmp_path / "authority.json", {"authority": True}
    )
    source_path = production._write_immutable_json(
        tmp_path / "source.json", {"source": True}
    )
    plan = _plan(tmp_path)
    plan["candidates"][0]["plan"] = production._file_record(source_path)
    _patch_cycle_authorities(monkeypatch, plan)
    evaluations = iter(
        [_evaluation(96223), _evaluation(96223), _evaluation(None, False)]
    )
    monkeypatch.setattr(
        refill, "_evaluate", lambda *_args, **_kwargs: next(evaluations)
    )
    posts = {"count": 0}

    def submitter(*, additional_pre_submit_guard: Any, **_kwargs: Any) -> Path:
        additional_pre_submit_guard()
        posts["count"] += 1
        raise AssertionError("guard failure must happen before POST")

    with pytest.raises(refill.SafeRefillError, match="pre-POST"):
        refill.cycle(
            refill_plan_path=plan_path,
            extension_authority_path=authority_path,
            authorize_submit=True,
            submitter=submitter,
        )
    assert posts["count"] == 0


def test_restart_recovers_claim_without_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = production._write_immutable_json(
        tmp_path / "safe_refill_plan.json", {"plan": True}
    )
    authority_path = production._write_immutable_json(
        tmp_path / "authority.json", {"authority": True}
    )
    source_path = production._write_immutable_json(
        tmp_path / "source.json", {"source": True}
    )
    plan = _plan(tmp_path)
    plan["candidates"][0]["plan"] = production._file_record(source_path)
    _patch_cycle_authorities(monkeypatch, plan)
    paths = refill._slot_paths(plan, plan["candidates"][0])
    production._write_immutable_json(
        paths["intent"],
        refill._sealed(
            {
                "schema_version": refill.INTENT_SCHEMA,
                "safe_refill_plan": production._file_record(plan_path),
                "logical_authority_task_id": 96223,
                "candidate_physics_sha256": "a" * 64,
                "initial_gate_evaluation_sha256": "b" * 64,
                "fresh_gate_evaluation_sha256": "c" * 64,
                "maximum_post_calls": 1,
                "scheduler_cancel_allowed": False,
                "authorized_at_utc": refill._now(),
            }
        ),
    )
    monkeypatch.setattr(refill, "_claim_state", lambda _candidate: "claimed")
    monkeypatch.setattr(
        refill, "_evaluate", lambda *_args, **_kwargs: _evaluation(None, False)
    )
    calls = {"recover": 0}

    def recover(
        *, output: Path, additional_pre_submit_guard: Any, **_kwargs: Any
    ) -> Path:
        # Existing-claim recovery must never invoke this POST guard.
        assert callable(additional_pre_submit_guard)
        calls["recover"] += 1
        return production._write_immutable_json(output, {"submission": True})

    monkeypatch.setattr(
        refill.timeout12h,
        "_load_plan",
        lambda _path: ({"plan": True}, {}, {}, {}),
    )
    monkeypatch.setattr(
        refill.timeout12h,
        "load_submission_for_probe",
        lambda *_args, **_kwargs: {
            "scheduler_strict_node_contract": {
                "scheduler_mutation_performed": False
            }
        },
    )

    def ensure_extension(**kwargs: Any) -> Path:
        return production._write_immutable_json(
            kwargs["paths"]["extension"], {"extension": True}
        )

    monkeypatch.setattr(refill, "_ensure_extension", ensure_extension)
    state = refill.cycle(
        refill_plan_path=plan_path,
        extension_authority_path=authority_path,
        authorize_submit=True,
        submitter=recover,
    )
    assert calls["recover"] == 1
    assert state["action"] == "claim_recovered_and_watcher_extended"
    assert state["scheduler_post_calls_this_cycle"] == 0


def test_watcher_merges_reauthenticated_dynamic_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension_dir = tmp_path / "extensions"
    extension_dir.mkdir()
    (extension_dir / "l96223.json").write_text("{}", encoding="utf-8")
    extension = {
        "logical_authority_task_id": 96223,
        "execution_task_id": 97000,
    }
    fake_refill = SimpleNamespace(
        authenticate_extension_authority=lambda *_args, **_kwargs: {
            "extension_directory": str(extension_dir)
        },
        authenticate_watcher_extension_receipt=lambda *_args, **_kwargs: (
            extension
        ),
    )
    monkeypatch.setattr(
        watcher.importlib, "import_module", lambda _name: fake_refill
    )
    authority = production._write_immutable_json(
        tmp_path / "authority.json", {"authority": True}
    )
    base = {
        "slots": [
            {
                "logical_authority_task_id": 96209,
                "execution_task_id": 96294,
            }
        ],
        "exact_logical_slot_count": 1,
    }
    merged = watcher._plan_with_authorized_extensions(
        base,
        watch_plan_path=tmp_path / "watch_plan.json",
        extension_authority_path=authority,
    )
    assert merged["exact_logical_slot_count"] == 2
    assert merged["authorized_extension_count"] == 1
    assert [row["logical_authority_task_id"] for row in merged["slots"]] == [
        96209,
        96223,
    ]
