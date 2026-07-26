from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_postdeadline_ui_updater as updater


OBSERVED = "2026-07-26T18:20:00+09:00"


def _item(item_id: str, state: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": f"title {item_id}",
        "detail": f"detail {item_id}",
        "state": state,
        "updated_at": "2026-07-26T18:00:00+09:00",
        "progress_pct": 1,
        "evidence": ["preserved"],
    }


def _status() -> dict[str, Any]:
    current = [_item(spec.card_id, "in_progress") for spec in updater.TASK_SPECS]
    current.extend(
        [
            {
                **_item("fea-handoff", "in_progress"),
                "title": (
                    "SLURM · ALLOCATION JOBS 4 · SUBMITTED 120 · "
                    "RUNNING 4 · QUEUED 0 · COLLECTIONS 0"
                ),
            },
            _item("parallel-workstreams", "in_progress"),
        ]
    )
    return {
        "schema_version": updater.STATUS_SCHEMA,
        "goal_id": "mft-goal-20260726",
        "goal": "goal",
        "owner": "Codex",
        "generated_at": "2026-07-26T18:00:00+09:00",
        "deadline_at": "2026-07-26T18:00:00+09:00",
        "original_deadline_missed": True,
        "summary": "stale lifecycle summary that must be refreshed",
        "current": current,
        "completed": [_item("completed-static", "completed")],
        "attention": [
            _item("pareto-truth-boundary", "attention"),
            _item("attention-static", "blocked"),
        ],
        "unknown_top_level": {"preserve": True},
    }


def _task(
    spec: updater.TaskSpec,
    *,
    state: str = "running",
    allocation_id: int | None = 100,
    slurm_job_id: str = "200",
    failure_message: str = "",
) -> dict[str, Any]:
    if spec.expected_allocation_id is not None:
        allocation_id = spec.expected_allocation_id
    if spec.expected_slurm_job_id is not None:
        slurm_job_id = spec.expected_slurm_job_id
    actual_node = spec.requested_node if allocation_id else ""
    return {
        "id": spec.task_id,
        "task_id": spec.task_id,
        "name": spec.task_name,
        "state": state,
        "status": state,
        "queue_state": state,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "max_workers_per_node": spec.max_workers_per_node or 8,
        "same_node_as_task_id": 0,
        "requested_node_name": spec.requested_node,
        "node_name": spec.requested_node,
        "node_name_policy": "strict",
        "actual_node_name": actual_node,
        "allocation_node_name": actual_node,
        "placement_contract_satisfied": state != "queued",
        "preferred_node_relaxed": False,
        "allocation_id": allocation_id,
        "slurm_job_id": slurm_job_id,
        "account_name": spec.requested_account or "",
        "requested_account_name": spec.requested_account or "",
        "created_at": "2026-07-26 09:00:00",
        "started_at": "2026-07-26 09:01:00" if allocation_id else None,
        "finished_at": None,
        "exit_code": None,
        "failure_message": failure_message,
    }


def _mixed_tasks() -> dict[int, dict[str, Any]]:
    specs = updater.TASK_SPECS
    tasks = {
        specs[0].task_id: updater._validate_task(
            specs[0], _task(specs[0], allocation_id=101)
        ),
        specs[1].task_id: updater._validate_task(
            specs[1],
            _task(
                specs[1],
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        ),
        specs[2].task_id: updater._validate_task(
            specs[2],
            _task(specs[2], state="succeeded", allocation_id=103),
        ),
        specs[3].task_id: updater._validate_task(
            specs[3],
            _task(
                specs[3],
                state="failed",
                allocation_id=104,
                failure_message="task timed out",
            ),
        ),
        specs[4].task_id: updater._validate_task(
            specs[4],
            _task(specs[4], allocation_id=105),
        ),
        specs[5].task_id: updater._validate_task(
            specs[5],
            _task(
                specs[5],
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        ),
    }
    for spec in specs[6:]:
        tasks[spec.task_id] = updater._validate_task(
            spec,
            _task(
                spec,
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        )
    return tasks


def _reader_from(tasks: dict[int, dict[str, Any]]):
    def reader(_scheduler_url: str, task_id: int) -> dict[str, Any]:
        return copy.deepcopy(tasks[task_id])

    return reader


def _thermal_state(*, watcher_state: str = "running") -> dict[str, Any]:
    return updater._sealed(
        {
            "schema": updater.THERMAL_BRIDGE_STATE_SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "updated_at_utc": "2026-07-26T10:20:00+00:00",
            "heartbeat_interval_seconds": 60,
            "watcher_pid": 40984,
            "tool_path": (
                r"C:\w\mft-goal-20260726"
                r"\tools\mft_goal_corrected_thermal_transport_bridge.py"
            ),
            "tool_sha256": "a" * 64,
            "source_task_id": 96324,
            "source_task_name": updater.TASK_SPECS[0].task_name,
            "source_task_state": (
                "succeeded" if watcher_state == "collected" else "running"
            ),
            "watcher_state": watcher_state,
            "stage": (
                "thermal_artifact_materialized"
                if watcher_state == "collected"
                else "waiting_source_terminal"
            ),
            "bridge_task_id": 97001 if watcher_state == "collected" else None,
            "artifact_collected": watcher_state == "collected",
            "orchestration_root": (
                r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
                r"\postdeadline_thermal_transport_bridge_v1"
            ),
            "orchestration_root_exists": watcher_state == "collected",
            "orchestration_plan_exists": watcher_state == "collected",
            "scheduler_get_calls_total": 17,
            "scheduler_post_calls_total": 1 if watcher_state == "collected" else 0,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "failure": None,
        }
    )


def _continuation_state(
    *,
    status: str = "pending_standard_collections",
) -> dict[str, Any]:
    submitted = status == "full_submitted_pending_actual_result"
    return updater._sealed(
        {
            "schema_version": updater.STANDARD_FULL_CONTINUATION_STATE_SCHEMA,
            "observed_at_utc": "2026-07-26T10:21:00+00:00",
            "original_deadline_missed": True,
            "postdeadline": True,
            "canonical": False,
            "production_eligible": False,
            "scientific_pass_claimed": False,
            "full_result_available": False,
            "full_actual_constraints_passed": False,
            "promotion_completed": False,
            "status": status,
            "source_postsuccess_state": "postsuccess-state.json",
            "plan": {"path": "plan.json"} if submitted else None,
            "attempt_ledger": {"path": "attempt.json"} if submitted else None,
            "submission_receipt": (
                {"path": "receipt.json"} if submitted else None
            ),
            "full_task_id": 97001 if submitted else None,
            "maximum_scheduler_posts": 1,
            "scheduler_post_attempts_consumed": 1 if submitted else 0,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
        }
    )


def test_merge_preserves_protected_truth_and_seals_lifecycle_only() -> None:
    source = _status()
    completed = copy.deepcopy(source["completed"])
    attention = copy.deepcopy(source["attention"])
    parallel = copy.deepcopy(source["current"][-1])

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    sync = updater.validate_status_sync(merged)

    assert merged["generated_at"] == OBSERVED
    assert merged["completed"] == completed
    assert merged["attention"] == attention
    assert "18:20 KST" in merged["summary"]
    assert "running2 · queued5 · terminal2" in merged["summary"]
    assert "physical feasible0" in merged["summary"]
    assert "scientific/production PASS가 아닙니다" in merged["summary"]
    assert merged["current"][-1] != parallel
    assert merged["current"][-1]["id"] == "parallel-workstreams"
    assert (
        "RUNNING 2 · QUEUED 5 · ALLOCATION JOBS 2"
        in merged["current"][-1]["title"]
    )
    assert (
        "RISK task96328/allocation14620 FORCE 07-27 04:07:51 KST"
        in merged["current"][-1]["title"]
    )
    assert any(
        "RISK task96328 allocation14620" in value
        for value in merged["current"][-1]["evidence"]
    )
    policy = next(
        item
        for item in merged["current"]
        if item["id"] == updater.SELECTION_POLICY_CARD_ID
    )
    assert (
        policy["title"]
        == "DESIGN SELECTION | SYMMETRY/STANDARD PRIMARY | "
        "TERMINAL 1/7 | AUTO FULL OFF"
    )
    assert any(
        "primary candidate-selection gate=authenticated symmetric/Standard FEA"
        in value
        for value in policy["evidence"]
    )
    assert any(
        "task96326 lifecycle=SUCCEEDED / role=diagnostic reference only"
        in value
        for value in policy["evidence"]
    )
    assert any(
        "final explicit Full validation candidate cap=1" in value
        for value in policy["evidence"]
    )
    assert merged["unknown_top_level"] == {"preserve": True}
    assert sync["allocation_jobs_active"] == 2
    assert sync["running"] == 2
    assert sync["queued"] == 5
    assert sync["submitted_total"] == 125
    assert sync["collections_preserved"] == 0
    assert sync["scheduler_methods_used"] == ["GET"]
    assert sync["scientific_pass_generated"] is False
    assert sync["managed_task_ids"] == [
        96324,
        96325,
        96326,
        96327,
        96328,
        96329,
        96330,
        96331,
        96332,
    ]

    success = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[2].card_id
    )
    failed = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[3].card_id
    )
    assert "COLLECTION/PASS PENDING" in success["title"]
    assert any(
        "collection_authenticated=false" in value for value in success["evidence"]
    )
    assert "task timed out" in failed["detail"]
    assert "과학적 infeasibility" in failed["detail"]

    official = next(
        item
        for item in merged["current"]
        if item["id"] == "postdeadline-standard-official6-96328"
    )
    assert "task96328 RUNNING" in official["title"]
    assert "FORCE-CANCEL RISK 07-27 04:07:51 KST" in official["title"]
    assert any(
        "force boundary leads by 4h01m56s" in value
        for value in official["evidence"]
    )
    assert any(
        "hard residual guarantee=false" in value
        for value in official["evidence"]
    )
    assert any(
        "42942394a873e40181f9074f2625807224239b291bcf2f4cf2ccacde50b11edc"
        in value
        for value in official["evidence"]
    )
    official8 = next(
        item
        for item in merged["current"]
        if item["id"] == "postdeadline-standard-official8-96329"
    )
    assert "task96329 QUEUED" in official8["title"]
    assert any(
        "5319a8a4dceb27082b91fc6badd540221298eeb9f324e2fa30e76e529313d3ec"
        in value
        for value in official8["evidence"]
    )
    for task_id, order, node, receipt_sha in (
        (
            96330,
            1,
            "n110",
            "d71987f49b062132fdf90a358bcc819a2cf9fd2e74c76fecad04e22ea9329233",
        ),
        (
            96331,
            12,
            "n112",
            "2e207daa33165e8921816e56beb2458d1dcd8515d6cb0b8d5a25cb0cf3cde532",
        ),
        (
            96332,
            5,
            "n115",
            "4275d9e8e004f3f9f57d9483ca9e7b8a48d410778f39a80848c95c396ed2171c",
        ),
    ):
        card = next(
            item
            for item in merged["current"]
            if item["id"]
            == f"postdeadline-standard-official{order}-{task_id}"
        )
        assert f"task{task_id} QUEUED" in card["title"]
        assert node in card["title"]
        assert any("search_only=true" in value for value in card["evidence"])
        assert any(receipt_sha in value for value in card["evidence"])

    handoff = next(item for item in merged["current"] if item["id"] == "fea-handoff")
    assert (
        handoff["title"] == "SLURM · ALLOCATION JOBS 2 · SUBMITTED 125 · "
        "RUNNING 2 · QUEUED 5 · COLLECTIONS 0"
    )
    for item in merged["current"]:
        assert len(item["title"]) <= 160
        assert len(item["detail"]) <= 1_200
        assert len(item["evidence"]) <= 12
        assert all(len(value) <= 500 for value in item["evidence"])


def test_merge_upserts_missing_official_task_card_before_parallel() -> None:
    source = _status()
    source["current"] = [
        item
        for item in source["current"]
        if item["id"] != "postdeadline-standard-official6-96328"
    ]

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged["current"]]

    assert ids.count("postdeadline-standard-official6-96328") == 1
    assert ids.index("postdeadline-standard-official6-96328") < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged)


def test_merge_removes_stale_automatic_continuation_card_when_disarmed() -> None:
    source = _status()
    source["current"].insert(
        -1,
        _item(updater.LEGACY_CONTINUATION_CARD_ID, "in_progress"),
    )

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged["current"]]

    assert updater.LEGACY_CONTINUATION_CARD_ID not in ids
    assert ids.count(updater.SELECTION_POLICY_CARD_ID) == 1
    assert ids.index(updater.SELECTION_POLICY_CARD_ID) < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged)


def test_synchronize_once_is_atomic_and_identity_failure_keeps_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codex-work-status.json"
    path.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    raw_tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    sync = updater.synchronize_once(
        status_file=path,
        task_reader=_reader_from(raw_tasks),
        observed_at=OBSERVED,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert updater.validate_status_sync(payload) == sync
    assert not list(tmp_path.glob("*.tmp"))

    before = path.read_bytes()
    drifted = copy.deepcopy(raw_tasks)
    drifted[96325]["name"] = "wrong-task-name"
    with pytest.raises(updater.UpdaterError, match="name drifted"):
        updater.synchronize_once(
            status_file=path,
            task_reader=_reader_from(drifted),
            observed_at=OBSERVED,
        )
    assert path.read_bytes() == before


def test_merge_adds_authenticated_codex_automation_cards(tmp_path: Path) -> None:
    postsuccess = updater._sealed(
        {
            "schema_version": updater.POSTSUCCESS_STATE_SCHEMA,
            "status": "pending_standard_collections",
            "collection_count": 0,
            "pending_count": 2,
            "terminal_failure_count": 0,
            "diagnostic_only": True,
            "production_eligible": False,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "surrogate_retraining_performed": False,
            "production_pareto_emitted": False,
        }
    )
    postsuccess_path = tmp_path / "postsuccess-state.json"
    postsuccess_path.write_text(
        json.dumps(postsuccess, ensure_ascii=False),
        encoding="utf-8",
    )
    thermal_path = tmp_path / "thermal-watch-state.json"
    thermal_path.write_text(
        json.dumps(_thermal_state(), ensure_ascii=False),
        encoding="utf-8",
    )
    continuation_path = tmp_path / "standard-full-continuation-state.json"
    continuation_path.write_text(
        json.dumps(_continuation_state(), ensure_ascii=False),
        encoding="utf-8",
    )
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    pending = updater._sealed(
        {
            "schema": updater.FINAL_GATE_PENDING_SCHEMA,
            "status": "pending",
            "pending_reasons": [
                "full collector state is watching",
                "thermal collector state is watching",
            ],
            "final_package_created": False,
            "full_aedt_promoted": False,
            "symmetric_aedt_promoted": False,
        }
    )
    (gate_root / "pending_manifest.json").write_text(
        json.dumps(pending, ensure_ascii=False),
        encoding="utf-8",
    )

    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
        postsuccess_state_file=postsuccess_path,
        thermal_bridge_state_file=thermal_path,
        standard_full_continuation_state_file=continuation_path,
        final_gate_root=gate_root,
    )

    by_id = {item["id"]: item for item in merged["current"]}
    assert "COLLECTIONS 0 · PENDING 2" in by_id[
        "codex-standard-postsuccess-pipeline"
    ]["title"]
    assert by_id["codex-final-aedt-package-gate"]["title"].endswith(
        "PENDING"
    )
    thermal = by_id["codex-thermal-artifact-handoff"]
    assert thermal["title"] == (
        "CODEX AUTO · THERMAL ARTIFACT HANDOFF · RUNNING"
    )
    assert any(
        "source task96324 state=running" in item
        for item in thermal["evidence"]
    )
    assert any(
        "Scheduler POST count=0" in item for item in thermal["evidence"]
    )
    assert any(
        "scientific claim=false" in item for item in thermal["evidence"]
    )
    continuation = by_id["codex-standard-full-continuation"]
    assert continuation["title"].endswith("PENDING_STANDARD_COLLECTIONS")
    assert any(
        "Scheduler POST attempts consumed=0/1" in item
        for item in continuation["evidence"]
    )
    assert any(
        "scientific PASS=false" in item
        for item in continuation["evidence"]
    )
    policy = by_id[updater.SELECTION_POLICY_CARD_ID]
    assert "AUTO FULL OFF" in policy["title"]
    assert any(
        "automatic Standard-to-Full per candidate=false" in item
        for item in policy["evidence"]
    )
    postsuccess_card = by_id["codex-standard-postsuccess-pipeline"]
    assert any(
        "Standard selection lanes=7" in item
        and "task96332" in item
        for item in postsuccess_card["evidence"]
    )
    assert any(
        "full AEDT promoted=false" in item
        for item in by_id["codex-final-aedt-package-gate"]["evidence"]
    )
    assert merged["current"][-1]["id"] == "parallel-workstreams"
    updater.validate_status_sync(merged)

    postsuccess["pending_count"] = 1
    postsuccess_path.write_text(
        json.dumps(postsuccess, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.merge_status(
            _status(),
            _mixed_tasks(),
            observed_at=OBSERVED,
            postsuccess_state_file=postsuccess_path,
            thermal_bridge_state_file=thermal_path,
            standard_full_continuation_state_file=continuation_path,
            final_gate_root=gate_root,
        )


def test_thermal_bridge_state_absence_and_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }
    missing = tmp_path / "missing-thermal-state.json"
    before = status_file.read_bytes()

    with pytest.raises(FileNotFoundError):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            thermal_bridge_state_file=missing,
        )
    assert status_file.read_bytes() == before

    thermal_path = tmp_path / "thermal-state.json"
    state = _thermal_state()
    state["scheduler_post_calls_total"] = 1
    thermal_path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            thermal_bridge_state_file=thermal_path,
        )
    assert status_file.read_bytes() == before


def test_standard_full_continuation_state_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }
    continuation_path = tmp_path / "continuation-state.json"
    state = _continuation_state()
    state["scientific_pass_claimed"] = True
    continuation_path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )
    before = status_file.read_bytes()

    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            standard_full_continuation_state_file=continuation_path,
        )
    assert status_file.read_bytes() == before


class _Response(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def test_scheduler_reader_uses_bounded_get(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = updater.TASK_SPECS[0]
    observed: dict[str, Any] = {}

    def urlopen(request: Any, *, timeout: float) -> _Response:
        observed["method"] = request.get_method()
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response(json.dumps(_task(spec)).encode("utf-8"))

    monkeypatch.setattr(updater.urllib.request, "urlopen", urlopen)
    result = updater._get_scheduler_task(
        "http://127.0.0.1:8002",
        spec.task_id,
        timeout_seconds=3.0,
    )

    assert result["task_id"] == spec.task_id
    assert observed == {
        "method": "GET",
        "url": f"http://127.0.0.1:8002/api/tasks/{spec.task_id}",
        "timeout": 3.0,
    }


@pytest.mark.parametrize(
    "spec",
    tuple(
        spec
        for spec in updater.TASK_SPECS
        if spec.max_workers_per_node == 1
    ),
)
def test_official_task_max_workers_is_fail_closed(
    spec: updater.TaskSpec,
) -> None:
    assert spec.task_id in {96328, 96329, 96330, 96331, 96332}
    task = _task(spec)
    task["max_workers_per_node"] = 2

    with pytest.raises(updater.UpdaterError, match="max_workers_per_node drifted"):
        updater._validate_task(spec, task)


def test_corrected_allocation_force_risk_identity_is_fail_closed() -> None:
    spec = next(item for item in updater.TASK_SPECS if item.task_id == 96328)
    task = _task(spec)
    task["allocation_id"] = 14621

    with pytest.raises(
        updater.UpdaterError,
        match="corrected allocation identity drifted",
    ):
        updater._validate_task(spec, task)


def test_run_once_writes_sealed_pid_and_log(tmp_path: Path) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    pid_file = tmp_path / "updater.pid.json"
    log_file = tmp_path / "updater.jsonl"
    lock_file = tmp_path / "updater.lock"
    tasks = {
        spec.task_id: _task(spec, allocation_id=300 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    result = updater.run_updater(
        status_file=status_file,
        scheduler_url=updater.DEFAULT_SCHEDULER_URL,
        once=True,
        interval_seconds=60,
        pid_file=pid_file,
        log_file=log_file,
        lock_file=lock_file,
        task_reader=_reader_from(tasks),
    )

    assert result is not None
    pid = json.loads(pid_file.read_text(encoding="utf-8"))
    updater.validate_seal(pid, updater.PID_SCHEMA)
    assert pid["scheduler_methods_allowed"] == ["GET"]
    assert pid["scheduler_mutation_performed"] is False
    events = [
        json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "updater_started",
        "cycle_completed",
    ]
    for event in events:
        updater.validate_seal(event, updater.LOG_SCHEMA)


def test_cli_modes_and_default_interval() -> None:
    parser = updater._parser()
    assert parser.parse_args(["--once"]).interval_seconds == 60
    parsed = parser.parse_args(
        [
            "--watch",
            "--thermal-bridge-state-file",
            "thermal.json",
            "--standard-full-continuation-state-file",
            "continuation.json",
        ]
    )
    assert parsed.watch is True
    assert parsed.thermal_bridge_state_file == Path("thermal.json")
    assert parsed.standard_full_continuation_state_file == Path(
        "continuation.json"
    )
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--once", "--watch"])
