from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse

import pytest

from tests.test_tier1_final1000_stage_launch import _rendered_plan
from tools import tier1_corrected_current7_slurm_harvest as current_harvest
from tools import tier1_corrected_current7_slurm_seed_runner as seed_runner
from tools import tier1_final1000_slurm_controller as controller
from tools import tier1_final1000_slurm_harvest as harvest
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


REPO = Path(__file__).resolve().parents[1]


class Scheduler:
    def __init__(self, tasks, *, batch_omit_ids=()):
        self.tasks = {int(task["id"]): copy.deepcopy(task) for task in tasks}
        self.batch_omit_ids = {int(value) for value in batch_omit_ids}
        self.get_count = 0
        self.batch_get_count = 0
        self.task_get_count = 0

    def list_tasks(self, *, name_prefix, limit):
        assert name_prefix == controller.TASK_NAME_PREFIX
        assert limit == harvest.SCHEDULER_BATCH_LIMIT
        self.get_count += 1
        self.batch_get_count += 1
        return [
            copy.deepcopy(task)
            for task_id, task in sorted(self.tasks.items(), reverse=True)
            if task_id not in self.batch_omit_ids
            and str(task.get("name") or "").startswith(name_prefix)
        ][:limit]

    def get_task(self, task_id):
        self.get_count += 1
        self.task_get_count += 1
        value = self.tasks.get(int(task_id))
        return copy.deepcopy(value) if value is not None else None


class NoRemoteReads:
    def stat(self, _account_name, _path):
        raise AssertionError("running-only projection must not read SFTP")

    def read_bytes(self, _account_name, _path, *, maximum_bytes):
        del maximum_bytes
        raise AssertionError("running-only projection must not read SFTP")


def _state_and_tasks(plan, *, scheduler_state="running", all_entries=False):
    state = controller._initial_state(plan)
    plan_index = controller._task_index(plan)
    templates = controller._task_templates(plan)
    tasks = []
    task_id = 91_000
    seen_stages = set()
    for entry in state["entries"]:
        if not all_entries and entry["stage_id"] in seen_stages:
            continue
        seen_stages.add(entry["stage_id"])
        task_id += 1
        entry["task_id"] = task_id
        entry["state"] = (
            scheduler_state
            if scheduler_state in controller.ACTIVE_STATES | controller.TERMINAL_STATES
            else "running"
        )
        expected = controller._task_for_entry(
            entry, plan_index=plan_index, templates=templates
        )
        tasks.append(
            {
                "id": task_id,
                "name": expected["name"],
                "dedupe_key": expected["dedupe_key"],
                "status": scheduler_state,
                "account_name": "account-a",
                "slurm_job_id": str(task_id),
                "exit_code": 0 if scheduler_state == "completed" else None,
                "created_at": "2026-07-21T01:00:00+00:00",
                "started_at": "2026-07-21T01:00:01+00:00",
                "finished_at": (
                    "2026-07-21T01:10:00+00:00"
                    if scheduler_state == "completed"
                    else None
                ),
            }
        )
    state = controller._seal_state(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    return controller._validate_state(state, plan), tasks


def _bindings(plan):
    values = {}
    temperature_targets = list(seed_runner.CURRENT_TEMPERATURE_TARGETS)
    for stage in profiles.STAGES:
        hard_spec, constraint_names = launch.optimizer_stage_contract(stage)
        summary = plan["stage_bindings"][stage.stage_id]
        manifest = {
            "contract_sha256": "c" * 64,
            "task_schema_version": "mft-tier1-current7-slurm-seed-task-v1",
            "status_schema_version": current_harvest.STATUS_SCHEMA,
            "constraint_version": (
                f"final1000-{stage.stage_id}-res15to20k-n1-6-v1"
            ),
            "hard_spec": hard_spec,
            "hard_spec_sha256": launch.canonical_sha256(hard_spec),
            "hard_constraint_contract_sha256": "d" * 64,
            "temperature_contract_sha256": "e" * 64,
            "constraint_names": constraint_names,
            "temperature_targets": temperature_targets,
            "search_execution": {
                "result_schema_version": current_harvest.RESULT_SCHEMA,
                "result_filename": "result.json",
            },
        }
        values[stage.stage_id] = {
            "plan": {
                "bundle_id": summary["bundle_id"],
                "bundle_manifest_sha256": summary["bundle_manifest_sha256"],
                "remote_bundle": summary["remote_bundle"],
            },
            "manifest": manifest,
        }
    return values


def test_apply_publishes_four_isolated_condition_indexes_and_never_primary(
    tmp_path, monkeypatch
):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    bindings = _bindings(plan)
    runtime = tmp_path / "final1000-runtime"
    state_path = runtime / "controller" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    launch_path = tmp_path / "launch.json"
    bindings_path = tmp_path / "bindings.json"
    launch_path.write_text("{}", encoding="utf-8")
    bindings_path.write_text("{}", encoding="utf-8")
    current7_root = tmp_path / "current7-primary"
    protected = current7_root / "canonical" / "current7-index.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"current7-primary-sentinel\n")
    monkeypatch.setattr(
        harvest,
        "_validate_inputs",
        lambda *_args: (plan, bindings, state),
    )

    result = harvest.harvest_once(
        launch_path,
        bindings_path,
        state_path,
        scheduler=Scheduler(tasks),
        remote=NoRemoteReads(),
        runtime=runtime,
        protected_current7_index=protected,
        apply=True,
        observed_at="2026-07-21 01:02:03",
    )

    assert result["scheduler_mutation_count"] == 0
    assert result["remote_write_count"] == 0
    assert result["scheduler_get_count"] == 1
    assert result["scheduler_batch_get_count"] == 1
    assert result["scheduler_task_get_count"] == 0
    assert protected.read_bytes() == b"current7-primary-sentinel\n"
    indexes = result["condition_inventory"]["indexes"]
    assert [item["stage_id"] for item in indexes] == [
        stage.stage_id for stage in profiles.STAGES
    ]
    assert result["condition_inventory"]["ui_environment"]["value"] == (
        os.pathsep.join(item["path"] for item in indexes)
    )
    for item, stage in zip(indexes, profiles.STAGES, strict=True):
        path = Path(item["path"])
        assert path == (
            runtime
            / "conditions"
            / stage.stage_id
            / "canonical"
            / "current7-index.json"
        ).resolve()
        index = json.loads(path.read_text(encoding="utf-8"))
        assert index["schema_version"] == current_harvest.CURRENT7_INDEX_SCHEMA
        assert index["final_goal_stage_id"] == stage.stage_id
        assert index["hard_spec"]["resonance_min_Hz"] == 15_000.0
        assert index["hard_spec"]["resonance_max_Hz"] == 20_000.0
        assert index["harvest_observed_at"] == "2026-07-21 01:02:03"
        assert index["condition_display_only"] is True


def test_terminal_scheduler_envelope_uses_detail_get_then_caches(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, scheduler_state="completed")
    runtime = tmp_path / "final1000-runtime"
    scheduler = Scheduler(tasks)
    by_stage, pending, hits = harvest._inventory_tasks(
        plan, state, scheduler=scheduler, runtime=runtime
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 5
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 4
    assert hits == 0
    assert len(pending) == 4
    for path, payload in pending:
        current_harvest._atomic_replace(path, payload)

    scheduler.tasks.clear()
    by_stage, pending, hits = harvest._inventory_tasks(
        plan, state, scheduler=scheduler, runtime=runtime
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 5
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 4
    assert hits == 4
    assert pending == []


def test_500_running_tasks_use_one_campaign_inventory_get(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, all_entries=True)
    assert len(tasks) == 500
    scheduler = Scheduler(tasks)

    by_stage, pending, hits = harvest._inventory_tasks(
        plan,
        state,
        scheduler=scheduler,
        runtime=tmp_path / "final1000-runtime",
    )

    assert sum(len(items) for items in by_stage.values()) == 500
    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0
    assert pending == []
    assert hits == 0


def test_task_missing_from_capped_batch_uses_one_fail_closed_detail_get(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    omitted_id = int(tasks[0]["id"])
    scheduler = Scheduler(tasks, batch_omit_ids={omitted_id})

    by_stage, pending, hits = harvest._inventory_tasks(
        plan,
        state,
        scheduler=scheduler,
        runtime=tmp_path / "final1000-runtime",
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 2
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 1
    assert pending == []
    assert hits == 0


def test_batch_inventory_identity_mismatch_fails_closed(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    tasks[0]["dedupe_key"] = "foreign-dedupe"
    scheduler = Scheduler(tasks)

    with pytest.raises(RuntimeError, match="inventory identity changed"):
        harvest._inventory_tasks(
            plan,
            state,
            scheduler=scheduler,
            runtime=tmp_path / "final1000-runtime",
        )

    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0


def test_terminal_batch_row_requires_matching_detail_evidence(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, scheduler_state="completed")

    class ChangedTerminalScheduler(Scheduler):
        def get_task(self, task_id):
            value = super().get_task(task_id)
            assert value is not None
            value["status"] = "failed"
            return value

    scheduler = ChangedTerminalScheduler(tasks)

    with pytest.raises(RuntimeError, match="terminal scheduler evidence changed"):
        harvest._inventory_tasks(
            plan,
            state,
            scheduler=scheduler,
            runtime=tmp_path / "final1000-runtime",
        )

    assert scheduler.get_count == 2
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 1


def test_read_only_api_uses_filtered_max_limit_inventory_get(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b"[]"

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(harvest.urllib.request, "urlopen", fake_urlopen)
    scheduler = harvest.Final1000ReadOnlySchedulerApi(
        "http://scheduler:8002", timeout=7
    )

    assert scheduler.list_tasks(
        name_prefix=controller.TASK_NAME_PREFIX,
        limit=harvest.SCHEDULER_BATCH_LIMIT,
    ) == []

    assert len(requests) == 1
    request, timeout = requests[0]
    parsed = urllib.parse.urlparse(request.full_url)
    assert request.get_method() == "GET"
    assert parsed.path == "/api/tasks"
    assert urllib.parse.parse_qs(parsed.query) == {
        "name_prefix": [controller.TASK_NAME_PREFIX],
        "limit": [str(harvest.SCHEDULER_BATCH_LIMIT)],
    }
    assert timeout == 7
    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0


def test_runtime_overlap_with_current7_primary_fails_before_writes(tmp_path):
    protected = tmp_path / "current7" / "canonical" / "current7-index.json"
    runtime = tmp_path / "current7" / "conditions" / "final1000"
    with pytest.raises(RuntimeError, match="overlaps"):
        harvest._assert_runtime_isolated(runtime, protected)


def test_cli_exposes_watch_apply_but_no_scheduler_mutation_surface():
    process = subprocess.run(
        [sys.executable, "tools/tier1_final1000_slurm_harvest.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    help_text = process.stdout.lower()
    assert "--watch" in help_text
    assert "--apply" in help_text
    assert "--scheduler-url" in help_text
    assert "--submit" not in help_text
    assert "--cancel" not in help_text
    assert "--close" not in help_text
