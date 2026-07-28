from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official_standard_hedge_submit as submitter


OBSERVED = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def test_historical_scheduler_payload_compatibility_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = submitter.CANDIDATES[0]
    fresh = {
        "command": (
            "before; "
            f"{submitter.PYAEDT_LIBRARY_BINDING_COMMAND}"
            "after;"
        ),
        "dedupe_key": "sealed-dedupe",
        "cpus": 8,
    }
    sealed = copy.deepcopy(fresh)
    sealed["command"] = sealed["command"].replace(
        submitter.PYAEDT_LIBRARY_BINDING_COMMAND,
        "",
        1,
    )
    sealed_sha256 = submitter.payload_sha256(sealed)
    monkeypatch.setattr(
        submitter,
        "HISTORICAL_SCHEDULER_PAYLOAD_SHA256_BY_CANDIDATE",
        {spec.candidate_sha256: sealed_sha256},
    )
    plan = {
        "scheduler_payload": sealed,
        "scheduler_payload_sha256": sealed_sha256,
    }

    assert submitter._scheduler_payload_matches_reviewed_lane(
        spec,
        plan,
        fresh,
    )

    drifted = copy.deepcopy(fresh)
    drifted["command"] += " true;"
    assert not submitter._scheduler_payload_matches_reviewed_lane(
        spec,
        plan,
        drifted,
    )

    unsealed = copy.deepcopy(plan)
    unsealed["scheduler_payload_sha256"] = "0" * 64
    assert not submitter._scheduler_payload_matches_reviewed_lane(
        spec,
        unsealed,
        fresh,
    )

    assert not submitter._scheduler_payload_matches_reviewed_lane(
        submitter.CANDIDATES[1],
        plan,
        fresh,
    )


def test_launcher_history_accepts_only_the_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = tmp_path / "historical.cmd"
    historical.write_text("historical launcher", encoding="utf-8")
    live_config = tmp_path / "app.yaml"
    live_config.write_text("config", encoding="utf-8")
    successor_root = tmp_path / "successor"
    successor_config = successor_root / "slurm_scheduler" / "config.py"
    successor_scheduler = (
        successor_root / "slurm_scheduler" / "scheduler.py"
    )
    successor_config.parent.mkdir(parents=True)
    successor_config.write_text("successor config", encoding="utf-8")
    successor_scheduler.write_text("successor scheduler", encoding="utf-8")
    current = tmp_path / "live.cmd"
    current.write_text(
        " ".join(
            (
                "successor-commit",
                str(successor_root),
                str(live_config),
                "8002",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(submitter, "LIVE_LAUNCHER", current)
    monkeypatch.setattr(submitter, "LIVE_CONFIG", live_config)
    monkeypatch.setattr(
        submitter,
        "LIVE_LAUNCHER_SHA256",
        submitter.sha256_file(historical),
    )
    monkeypatch.setattr(
        submitter,
        "LIVE_LAUNCHER_SIZE_BYTES",
        historical.stat().st_size,
    )
    monkeypatch.setattr(
        submitter,
        "HISTORICAL_LAUNCHER_ARCHIVE",
        historical,
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_LIVE_LAUNCHER_SHA256",
        submitter.sha256_file(current),
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_COMMIT",
        "successor-commit",
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_ROOT",
        successor_root,
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_CONFIG_PY",
        successor_config,
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_CONFIG_PY_SHA256",
        submitter.sha256_file(successor_config),
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_SCHEDULER_PY",
        successor_scheduler,
    )
    monkeypatch.setattr(
        submitter,
        "CURRENT_DEPLOYED_SCHEDULER_PY_SHA256",
        submitter.sha256_file(successor_scheduler),
    )

    record, text = submitter._authenticate_launcher_history()
    assert record == {
        "path": str(current.resolve()),
        "sha256": submitter.sha256_file(historical),
        "size_bytes": historical.stat().st_size,
    }
    assert text == "historical launcher"

    current.write_text(
        current.read_text(encoding="utf-8") + " arbitrary-drift",
        encoding="utf-8",
    )
    with pytest.raises(
        submitter.PostdeadlineContractError,
        match="launcher bytes drifted",
    ):
        submitter._authenticate_launcher_history()


class Lock:
    def __enter__(self) -> Lock:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class BatchScheduler:
    def __init__(self) -> None:
        self.tasks: dict[int, dict[str, Any]] = {}
        self.posts: list[str] = []
        self.next_id = 97001

    def poster(
        self, _url: str, payload: Mapping[str, Any]
    ) -> tuple[int, dict[str, int], None]:
        name = str(payload["name"])
        spec = next(
            item for item in submitter.CANDIDATES if item.task_name == name
        )
        attempt = (
            submitter.OUTPUT_ROOT
            / spec.lane_name
            / submitter.ATTEMPT_NAME
        )
        assert attempt.is_file()
        task_id = self.next_id
        self.next_id += 1
        self.posts.append(name)
        self.tasks[task_id] = readback(spec, payload, task_id)
        return 201, {"task_id": task_id}, None

    def reader(
        self, path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        if path.startswith("/api/tasks/"):
            return self.tasks[int(path.rsplit("/", 1)[1])]
        if path == "/api/tasks":
            name_prefix = dict(query or []).get("name_prefix")
            return [
                task
                for task in self.tasks.values()
                if task["name"].startswith(str(name_prefix or ""))
            ]
        raise AssertionError(f"unexpected GET {path} {query}")


def readback(
    spec: submitter.CandidateSpec,
    payload: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "name": spec.task_name,
        "dedupe_key": payload["dedupe_key"],
        "project": submitter.PROJECT,
        "requested_account_name": spec.account_name,
        "requested_node_name": spec.node_name,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": submitter.CPUS,
        "memory_mb": submitter.MEMORY_MB,
        "timeout_seconds": submitter.SCHEDULER_SECONDS,
        "max_workers_per_node": submitter.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "status": "queued",
        "preferred_node_relaxed": False,
        "node_name_policy": "strict",
        "account_name": "",
        "node_name": "",
    }


def batch_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    list[dict[str, Any]],
]:
    output = tmp_path / "submit"
    monkeypatch.setattr(submitter, "OUTPUT_ROOT", output)
    manifest_path = tmp_path / "prepare_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {"payload_sha256": "a" * 64}
    lanes: list[dict[str, Any]] = []
    for spec in submitter.CANDIDATES:
        lane_dir = tmp_path / "plans" / spec.lane_name
        lane_dir.mkdir(parents=True)
        payload = {
            "name": spec.task_name,
            "dedupe_key": f"dedupe:{spec.selection_order}",
        }
        plan = {
            "payload_sha256": f"{spec.selection_order:064x}",
            "scheduler_payload_sha256": submitter.payload_sha256(payload),
            "dedupe_key": payload["dedupe_key"],
            "scheduler_payload": payload,
        }
        plan_path = submitter.write_immutable_json(
            lane_dir / "plan.json", plan
        )
        lanes.append(
            {
                "spec": spec,
                "plan": plan,
                "plan_path": plan_path,
                "lane_nonce": f"{spec.selection_order:064x}",
            }
        )
    return output, manifest_path, manifest, lanes


def fake_gate(lane: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
    return submitter.sealed(
        {
            "schema_version": submitter.LIVE_GATE_SCHEMA,
            "selection_order": lane["spec"].selection_order,
            "scheduler_post_calls": 0,
        }
    )


def run_batch(
    output: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    lanes: list[dict[str, Any]],
    scheduler: BatchScheduler,
) -> Path:
    return submitter.submit_batch(
        authorize_post=submitter.AUTHORIZATION,
        manifest_path=manifest_path,
        output_root=output,
        reader=scheduler.reader,
        poster=scheduler.poster,
        observed_at=OBSERVED,
        lock_factory=Lock,
        package_authenticator=lambda _path: (manifest, lanes),
    )


def test_success_contract_writes_three_complete_lanes_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest_path, manifest, lanes = batch_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(submitter, "live_submission_gate", fake_gate)
    scheduler = BatchScheduler()
    final_path = run_batch(
        output, manifest_path, manifest, lanes, scheduler
    )
    assert len(scheduler.posts) == 3
    final = submitter.validate_seal(
        submitter.read_json(final_path), submitter.BATCH_FINAL_SCHEMA
    )
    aggregate_path = output / submitter.AGGREGATE_SUBMIT_MANIFEST_NAME
    aggregate = submitter.validate_seal(
        submitter.read_json(aggregate_path),
        submitter.AGGREGATE_SUBMIT_MANIFEST_SCHEMA,
    )
    assert aggregate["task_ids"] == [97001, 97002, 97003]
    assert aggregate["selection_orders"] == [1, 12, 5]
    assert aggregate["get_only_collector_ready"] is True
    assert final["aggregate_submit_manifest"]["sha256"] == (
        submitter.sha256_file(aggregate_path)
    )
    for spec in submitter.CANDIDATES:
        lane_root = output / spec.lane_name
        for name in (
            submitter.LANE_INTENT_NAME,
            submitter.ATTEMPT_NAME,
            submitter.RECEIPT_NAME,
            submitter.LANE_FINAL_NAME,
        ):
            assert (lane_root / name).is_file()
        attempt = submitter.validate_seal(
            submitter.read_json(lane_root / submitter.ATTEMPT_NAME),
            submitter.ATTEMPT_SCHEMA,
        )
        assert attempt["post_call_consumed_before_network"] is True


def test_replay_authenticates_existing_tasks_without_another_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest_path, manifest, lanes = batch_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(submitter, "live_submission_gate", fake_gate)
    scheduler = BatchScheduler()
    first = run_batch(output, manifest_path, manifest, lanes, scheduler)
    second = run_batch(output, manifest_path, manifest, lanes, scheduler)
    assert first == second
    assert len(scheduler.posts) == 3


def test_partial_progress_resumes_only_unconsumed_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest_path, manifest, lanes = batch_fixture(
        tmp_path, monkeypatch
    )
    scheduler = BatchScheduler()
    fail_twelve = True

    def partial_gate(
        lane: Mapping[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        if fail_twelve and lane["spec"].selection_order == 12:
            raise submitter.PostdeadlineContractError("injected gate drift")
        return fake_gate(lane)

    monkeypatch.setattr(submitter, "live_submission_gate", partial_gate)
    with pytest.raises(
        submitter.PostdeadlineContractError,
        match="submission incomplete",
    ):
        run_batch(output, manifest_path, manifest, lanes, scheduler)
    assert len(scheduler.posts) == 2
    fail_twelve = False
    run_batch(output, manifest_path, manifest, lanes, scheduler)
    assert len(scheduler.posts) == 3
    assert scheduler.posts.count(submitter.CANDIDATES[0].task_name) == 1
    assert scheduler.posts.count(submitter.CANDIDATES[1].task_name) == 1
    assert scheduler.posts.count(submitter.CANDIDATES[2].task_name) == 1


def test_ambiguous_outcome_consumes_each_lane_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest_path, manifest, lanes = batch_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(submitter, "live_submission_gate", fake_gate)
    scheduler = BatchScheduler()
    calls = 0

    def ambiguous(
        _url: str, _payload: Mapping[str, Any]
    ) -> tuple[None, None, str]:
        nonlocal calls
        calls += 1
        return None, None, "connection reset"

    kwargs = {
        "authorize_post": submitter.AUTHORIZATION,
        "manifest_path": manifest_path,
        "output_root": output,
        "reader": scheduler.reader,
        "poster": ambiguous,
        "observed_at": OBSERVED,
        "lock_factory": Lock,
        "package_authenticator": lambda _path: (manifest, lanes),
    }
    with pytest.raises(submitter.PostdeadlineContractError):
        submitter.submit_batch(**kwargs)
    assert calls == 3
    for spec in submitter.CANDIDATES:
        lane_root = output / spec.lane_name
        assert (lane_root / submitter.ATTEMPT_NAME).is_file()
        assert (lane_root / submitter.AMBIGUOUS_NAME).is_file()
    with pytest.raises(submitter.PostdeadlineContractError):
        submitter.submit_batch(**kwargs)
    assert calls == 3


def test_node_gate_rejects_live_exact_account_node_allocation() -> None:
    row = {
        "id": 1,
        "account_name": "dhj02",
        "node_name": "n110",
        "state": "closed",
        "node_pestat_state": "idle",
        "node_cpu_total": 256,
        "node_cpu_used": 0,
        "node_memory_total_mb": 1_000_000,
        "node_memory_free_mb": 700_000,
        "node_metrics_observed_at": "2026-07-26 12:00:00",
    }
    evidence = submitter._node_gate(
        [row],
        account_name="dhj02",
        node_name="n110",
        observed_at=OBSERVED,
    )
    assert evidence["exact_account_node_live_allocation_count"] == 0
    row["state"] = "active"
    with pytest.raises(
        submitter.PostdeadlineContractError,
        match="node capacity gate failed",
    ):
        submitter._node_gate(
            [row],
            account_name="dhj02",
            node_name="n110",
            observed_at=OBSERVED,
        )


def test_manifest_hash_tamper_fails_before_lane_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "prepare_manifest.json"
    manifest.write_text('{"sealed":true}\n', encoding="utf-8")
    monkeypatch.setattr(submitter, "PREPARE_MANIFEST", manifest)
    monkeypatch.setattr(
        submitter,
        "PREPARE_MANIFEST_SHA256",
        submitter.sha256_file(manifest),
    )
    manifest.write_text('{"sealed":false}\n', encoding="utf-8")
    with pytest.raises(
        submitter.PostdeadlineContractError,
        match="manifest bytes drifted",
    ):
        submitter.authenticate_prepare_package(
            manifest,
            reauthenticate=lambda _spec: pytest.fail(
                "lane authentication must not run"
            ),
        )


def test_authorization_rejected_before_loading_or_posting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest_path, _manifest, _lanes = batch_fixture(
        tmp_path, monkeypatch
    )
    calls = 0

    def package(_path: Path) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError

    with pytest.raises(
        submitter.PostdeadlineContractError,
        match="authorization is absent",
    ):
        submitter.submit_batch(
            authorize_post="wrong",
            manifest_path=manifest_path,
            output_root=output,
            package_authenticator=package,
        )
    assert calls == 0
