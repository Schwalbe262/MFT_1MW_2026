from __future__ import annotations

import copy
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools import tier1_final1000_multiseed_consumer as consumer
from tools import tier1_final1000_multiseed_contract as contract
from tools import tier1_final1000_multiseed_monitor as monitor
from tools import tier1_final1000_multiseed_status as compact
from tools import tier1_final1000_stage_profiles as profiles


FIXED_TIME = "2026-07-22T05:00:00+00:00"


class EmptyScheduler:
    get_count = 0
    batch_get_count = 0
    task_get_count = 0

    def list_tasks(self, *, name_prefix: str, limit: int):
        assert name_prefix == consumer.TASK_NAME_PREFIX
        assert limit == consumer.SCHEDULER_BATCH_LIMIT
        return []

    def get_task(self, task_id: int):
        raise AssertionError(f"unexpected task detail read: {task_id}")


class NoRemoteReads:
    def stat(self, account_name: str, path: str):
        raise AssertionError(f"unexpected remote stat: {account_name}:{path}")

    def read_bytes(self, account_name: str, path: str, *, maximum_bytes: int):
        raise AssertionError(f"unexpected remote read: {account_name}:{path}")

    def close(self):
        return None


def _inputs() -> consumer.ConsumerInputs:
    plan_sha = "a" * 64
    bindings = {}
    for stage in profiles.STAGES:
        hard_spec = {"stage_id": stage.stage_id}
        bindings[stage.stage_id] = {
            "plan": {
                "bundle_id": f"bundle-{stage.stage_id}",
                "bundle_manifest_sha256": "c" * 64,
            },
            "manifest": {
                "contract_sha256": "d" * 64,
                "constraint_version": "fixture-v1",
                "hard_spec": hard_spec,
                "hard_spec_sha256": consumer.canonical_sha256(hard_spec),
                "hard_constraint_contract_sha256": "e" * 64,
                "temperature_contract_sha256": "f" * 64,
                "constraint_names": ["fixture-constraint"],
                "temperature_targets": ["fixture-temperature"],
            },
        }
    return consumer.ConsumerInputs(
        plan={"launch_plan_sha256": plan_sha},
        state={"state_sha256": "b" * 64, "entries": []},
        cohorts_by_plan_sha={plan_sha: {"bindings": bindings}},
    )


def _seed_live_parents(runtime: Path) -> dict[str, bytes]:
    pointers = {}
    for stage in profiles.STAGES:
        root = runtime / "conditions" / stage.stage_id / "canonical"
        snapshot = compact.build_compact_snapshot(
            stage_id=stage.stage_id,
            physical_lanes=[],
            seed_records=[],
            frontend_static={
                "cohort_id": f"live-v1-{stage.stage_id}",
                "source_bundle_cohorts": [
                    {"cohort_id": f"live-{index}"} for index in range(4)
                ],
            },
            updated_at=FIXED_TIME,
        )
        assert (
            compact.publish_compact_snapshot(
                root, *snapshot, pointer_name=consumer.POINTER_NAME
            )
            > 0
        )
        pointer = root / consumer.POINTER_NAME
        pointers[stage.stage_id] = pointer.read_bytes()
    return pointers


def _consume(
    runtime: Path,
    output: Path,
    *,
    apply: bool,
    observed_at: str = FIXED_TIME,
):
    return consumer.consume_once(
        _inputs(),
        scheduler=EmptyScheduler(),
        remote=NoRemoteReads(),
        runtime=runtime,
        output_root=output,
        protected_current7_index=Path(
            "C:/isolated-current7-primary/canonical/current7-index.json"
        ),
        apply=apply,
        publish_mode="shadow",
        observed_at=observed_at,
    )


def test_default_dry_run_writes_no_files_and_apply_is_restart_idempotent(
    tmp_path: Path,
):
    runtime = tmp_path / "live"
    output = tmp_path / "shadow"
    parent_pointers = _seed_live_parents(runtime)

    preview = _consume(runtime, output, apply=False)
    assert preview["local_write_count"] == 0
    assert preview["scheduler_mutation_count"] == 0
    assert preview["remote_write_count"] == 0
    assert not output.exists()
    assert {
        stage_id: (
            runtime / "conditions" / stage_id / "canonical" / consumer.POINTER_NAME
        ).read_bytes()
        for stage_id in parent_pointers
    } == parent_pointers

    first = _consume(runtime, output, apply=True)
    assert first["local_write_count"] > 0
    assert all(
        item["schema_version"] == contract.COMPACT_INDEX_SCHEMA
        for item in first["condition_inventory"]["indexes"]
    )
    for item in first["condition_inventory"]["indexes"]:
        page = monitor.adapt_condition_index(Path(item["path"]))
        assert page["source_bundle_cohorts"] == [
            {"cohort_id": f"live-{index}"} for index in range(4)
        ]
        assert page["virtual_scheduler_task_ids_created"] is False

    restarted = _consume(runtime, output, apply=True)
    assert restarted["local_write_count"] == 0
    assert restarted["condition_inventory"] == first["condition_inventory"]


def test_capability_receipt_is_exact_and_freshness_bounded(tmp_path: Path):
    runtime = tmp_path / "live"
    output = tmp_path / "shadow"
    _seed_live_parents(runtime)
    result = _consume(runtime, output, apply=True)
    receipt_path = output / "canonical" / "capability.json"
    with consumer.WriterLease(output / "canonical" / "test-writer.lock") as lease:
        receipt = consumer.build_capability_receipt(
            result=result,
            inventory_path=Path(result["condition_inventory_path"]),
            poll_seconds=60,
            freshness_deadline_seconds=120,
            runtime_root=runtime,
            writer_lease=lease,
        )
        consumer._atomic_json(receipt_path, receipt)
        published = datetime.fromisoformat(receipt["published_at"])
        checked = consumer.require_consumer_capability(
            receipt_path,
            now=published + timedelta(seconds=119),
            required_publish_mode="shadow",
        )
        assert checked["capabilities"] == consumer.CAPABILITIES
        with pytest.raises(RuntimeError, match="capability/freshness"):
            consumer.require_consumer_capability(
                receipt_path,
                now=published + timedelta(seconds=121),
                required_publish_mode="shadow",
            )

        # The production driver defaults to canonical and must never accept a
        # perfectly fresh shadow-canary receipt.
        with pytest.raises(RuntimeError, match="capability/freshness"):
            consumer.require_consumer_capability(receipt_path, now=published)

        tampered = copy.deepcopy(receipt)
        tampered["publish_mode"] = "canonical"
        tampered["receipt_sha256"] = consumer.canonical_sha256(
            {key: item for key, item in tampered.items() if key != "receipt_sha256"}
        )
        consumer._atomic_json(receipt_path, tampered)
        with pytest.raises(RuntimeError, match="capability/freshness"):
            consumer.require_consumer_capability(receipt_path, now=published)
        consumer._atomic_json(receipt_path, receipt)

        index_path = Path(receipt["condition_indexes"][0]["path"])
        index_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="advanced"):
            consumer.require_consumer_capability(
                receipt_path,
                now=published,
                required_publish_mode="shadow",
            )


def _minimal_v1_index(stage_id: str) -> dict:
    return {
        "schema_version": consumer.CURRENT7_INDEX_SCHEMA,
        "final_goal_stage_id": stage_id,
        "identity": f"final-{stage_id}",
    }


def test_v1_cutover_handoff_requires_stopped_writer_and_stable_four_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stop_file = tmp_path / "STOP_OLD_HARVESTER"
    stop_file.write_text("stop\n", encoding="utf-8")
    indexes = {}
    for stage in profiles.STAGES:
        path = tmp_path / stage.stage_id / consumer.POINTER_NAME
        path.parent.mkdir(parents=True)
        path.write_bytes(contract.json_bytes(_minimal_v1_index(stage.stage_id)))
        indexes[stage.stage_id] = path

    monkeypatch.setattr(consumer, "_pid_is_running", lambda _pid: False)
    receipt = consumer.build_v1_handoff_receipt(
        old_pid=44816,
        stop_file=stop_file,
        condition_indexes=indexes,
        observed_at=FIXED_TIME,
    )
    assert (
        consumer.validate_v1_handoff_receipt(receipt, require_live_pointer_match=True)[
            "old_writer_exited"
        ]
        is True
    )

    indexes[profiles.STAGES[0].stage_id].write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="advanced after handoff"):
        consumer.validate_v1_handoff_receipt(receipt, require_live_pointer_match=True)

    monkeypatch.setattr(consumer, "_pid_is_running", lambda _pid: True)
    with pytest.raises(RuntimeError, match="still running"):
        consumer.build_v1_handoff_receipt(
            old_pid=44816,
            stop_file=stop_file,
            condition_indexes=indexes,
        )


def test_canonical_capability_seals_handoff_containment_and_active_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    live_parent = tmp_path / "live-parent"
    canonical = tmp_path / "canonical-runtime"
    _seed_live_parents(live_parent)
    result = _consume(live_parent, canonical, apply=True)
    result["publish_mode"] = "canonical"

    stop_file = tmp_path / "STOP_OLD_V1"
    stop_file.write_text("stop\n", encoding="utf-8")
    archived_indexes = {}
    for stage in profiles.STAGES:
        path = tmp_path / "v1-final" / f"{stage.stage_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contract.json_bytes(_minimal_v1_index(stage.stage_id)))
        archived_indexes[stage.stage_id] = path
    original_pid_check = consumer._pid_is_running
    monkeypatch.setattr(
        consumer,
        "_pid_is_running",
        lambda pid: False if pid == 44816 else original_pid_check(pid),
    )
    handoff = consumer.build_v1_handoff_receipt(
        old_pid=44816,
        stop_file=stop_file,
        condition_indexes=archived_indexes,
        observed_at=FIXED_TIME,
    )
    handoff_path = tmp_path / "v1-handoff.json"
    consumer._atomic_json(handoff_path, handoff)

    receipt_path = canonical / "canonical" / "capability.json"
    with consumer.WriterLease(
        canonical / "canonical" / "canonical-writer.lock"
    ) as lease:
        receipt = consumer.build_capability_receipt(
            result=result,
            inventory_path=Path(result["condition_inventory_path"]),
            poll_seconds=30,
            freshness_deadline_seconds=120,
            runtime_root=canonical,
            writer_lease=lease,
            handoff_receipt_path=handoff_path,
        )
        consumer._atomic_json(receipt_path, receipt)
        checked = consumer.require_consumer_capability(
            receipt_path,
            now=datetime.fromisoformat(receipt["published_at"]),
        )
        assert checked["publish_mode"] == "canonical"
        assert checked["v1_handoff"]["old_writer_exited"] is True
        assert checked["writer_lease"]["run_id"] == lease.run_id


def test_writer_lease_refuses_a_concurrent_consumer_and_seals_clean_stop(
    tmp_path: Path,
):
    lease_path = tmp_path / "writer.lock"
    with consumer.WriterLease(lease_path) as first:
        with pytest.raises(RuntimeError, match="owns the writer lease"):
            with consumer.WriterLease(lease_path):
                pass
        capability = tmp_path / "capability.json"
        capability.write_text("{}\n", encoding="utf-8")
        stopped = consumer.build_stop_handoff(
            run_id=first.run_id, capability_receipt_path=capability
        )
    assert consumer.validate_stop_handoff(stopped)["clean_shutdown"] is True


def test_identity_failure_before_apply_keeps_all_last_good_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = tmp_path / "live"
    output = tmp_path / "shadow"
    _seed_live_parents(runtime)
    _consume(runtime, output, apply=True)
    pointers = {
        stage.stage_id: output
        / "conditions"
        / stage.stage_id
        / "canonical"
        / consumer.POINTER_NAME
        for stage in profiles.STAGES
    }
    before = {stage_id: path.read_bytes() for stage_id, path in pointers.items()}
    original = consumer.build_compact_snapshot
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic manifest identity divergence")
        return original(*args, **kwargs)

    monkeypatch.setattr(consumer, "build_compact_snapshot", fail_second)
    with pytest.raises(RuntimeError, match="identity divergence"):
        _consume(
            runtime,
            output,
            apply=True,
            observed_at="2026-07-22T05:01:00+00:00",
        )
    assert {
        stage_id: path.read_bytes() for stage_id, path in pointers.items()
    } == before


def test_cli_dry_run_never_constructs_writer_lease_or_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "dry-output"
    monkeypatch.setattr(consumer, "load_inputs", lambda *args, **kwargs: _inputs())
    monkeypatch.setattr(
        consumer,
        "Final1000ReadOnlySchedulerApi",
        lambda *_args, **_kwargs: EmptyScheduler(),
    )
    monkeypatch.setattr(
        consumer, "AccountSftpReader", lambda *_args, **_kwargs: NoRemoteReads()
    )
    monkeypatch.setattr(
        consumer,
        "consume_once",
        lambda *args, **kwargs: {
            "schema_version": consumer.CONSUMER_SCHEMA,
            "local_write_count": 0,
        },
    )

    class ForbiddenLease:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run constructed a writer lease")

    monkeypatch.setattr(consumer, "WriterLease", ForbiddenLease)
    assert (
        consumer.main(
            [
                "run",
                "--launch-plan",
                str(tmp_path / "plan.json"),
                "--bindings",
                str(tmp_path / "bindings.json"),
                "--controller-state",
                str(tmp_path / "state.json"),
                "--runtime",
                str(tmp_path / "runtime"),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert not output.exists()


def test_seal_handoff_preview_is_write_free_without_explicit_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop\n", encoding="utf-8")
    arguments = [
        "seal-v1-handoff",
        "--old-pid",
        "44816",
        "--stop-file",
        str(stop_file),
        "--output",
        str(tmp_path / "handoff.json"),
    ]
    for stage in profiles.STAGES:
        path = tmp_path / f"{stage.stage_id}.json"
        path.write_bytes(contract.json_bytes(_minimal_v1_index(stage.stage_id)))
        arguments.extend(["--condition-index", f"{stage.stage_id}={path}"])
    monkeypatch.setattr(consumer, "_pid_is_running", lambda _pid: False)
    assert consumer.main(arguments) == 0
    assert not (tmp_path / "handoff.json").exists()


def test_watch_retry_classifier_never_hides_identity_or_schema_failures():
    assert consumer._transient_transport_error(TimeoutError()) is True
    assert (
        consumer._transient_transport_error(
            RuntimeError("manifest identity divergence")
        )
        is False
    )
    assert consumer._transient_transport_error(ValueError("schema drift")) is False
