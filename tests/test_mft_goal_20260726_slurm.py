from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import mft_goal_20260726_slurm as slurm


def _submission_fixture():
    plan = {
        "bundle_id": "mft-goal-" + "a" * 24,
        "remote_bundle": "/gpfs/tmp_cpu2/mft_goal_20260726/deployment",
        "relocation_sources": {},
    }
    tasks = {}
    entries = []
    for index in range(slurm.EXPECTED_CAMPAIGN_TASKS):
        seed = 2_607_262_000 + index
        turns = 5 + index % 4
        payload_sha = f"{index + 1:064x}"
        task = {
            "seed": seed,
            "fixed_primary_turns": turns,
            "payload_sha256": payload_sha,
        }
        plan["relocation_sources"][seed] = (
            "artifacts/campaign/relocations/"
            f"seed-{seed}-n1-{turns}.json"
        )
        tasks[payload_sha] = task
        entries.append(
            {
                "task_id": 95_684 + index,
                "deduped": False,
                "name": f"mft-goal-nsga-s{seed}-n1-{turns}",
                "task_payload_sha256": payload_sha,
                "seed": seed,
                "fixed_primary_turns": turns,
                "submitted_at": "2026-07-24T14:00:00+00:00",
            }
        )
    ledger = {
        "schema_version": slurm.SUBMISSION_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "scheduler_url": slurm.DEFAULT_SCHEDULER_URL,
        "submissions": entries,
    }
    return plan, tasks, ledger


def _result_fixture(*, task_id=95_684):
    seed = 2_607_262_000
    turns = 5
    task = {
        "seed": seed,
        "fixed_primary_turns": turns,
        "payload_sha256": "1" * 64,
        "stage_spec_sha256": slurm.goal.GOAL_STAGE_SPEC_SHA256,
        "hard_constraint_contract_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "evaluation_model_sha256": "4" * 64,
        "search_only_proposal": True,
    }
    submission = {
        "task_id": task_id,
        "seed": seed,
        "fixed_primary_turns": turns,
        "task_payload_sha256": task["payload_sha256"],
        "task": task,
    }
    csv_bytes = b"terminal_population_index\n0\n"
    manifest = slurm._sealed(
        {
            "schema_version": (
                slurm.goal.preflight.GOAL_TERMINAL_TABLE_SCHEMA
            ),
            "goal_contract_required": True,
            "row_count": slurm.goal.POPULATION,
            "terminal_population_index_min": 0,
            "terminal_population_index_max": slurm.goal.POPULATION - 1,
            "columns": ["terminal_population_index"],
            "required_identity_columns": ["terminal_population_index"],
            "csv": {
                "path": "terminal_physical_candidates.csv",
                "sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "size_bytes": len(csv_bytes),
            },
            "source_identity": {
                "seed": seed,
                "task_id": str(task_id),
                "bundle_id": task["payload_sha256"],
                "island_id": f"n1-{turns}",
            },
            "stage_spec_sha256": slurm.goal.GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                slurm.goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            "one_row_per_terminal_individual": True,
            "physical_deduplication_key": "physical_geometry_sha256",
            "global_pareto_provenance_ready": True,
        }
    )
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    inventory = {
        "terminal_physical_candidates": {
            "path": "terminal_physical_candidates.csv",
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "size_bytes": len(csv_bytes),
        },
        "terminal_physical_candidates_manifest": {
            "path": "terminal_physical_candidates.manifest.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": len(manifest_bytes),
        },
    }
    result = slurm._sealed(
        {
            "schema_version": slurm.goal.SEARCH_RESULT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": slurm.goal.GOAL_CONTRACT_SCHEMA,
            "task_payload_sha256": task["payload_sha256"],
            "seed": seed,
            "fixed_primary_turns": turns,
            "population": slurm.goal.POPULATION,
            "generations": slurm.goal.GENERATIONS,
            "evaluated_generations": slurm.goal.GENERATIONS,
            "completed_generations": (
                slurm.goal.EXPECTED_ALGORITHM_N_GEN_COUNTER
            ),
            "terminal_population_count": slurm.goal.POPULATION,
            "stage_spec_sha256": task["stage_spec_sha256"],
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            "temperature_contract_sha256": (
                slurm.goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "dataset_sha256": task["dataset_sha256"],
            "evaluation_model_sha256": task[
                "evaluation_model_sha256"
            ],
            "cooling_contract_sha256": (
                slurm.goal.FIXED_COOLING_IDENTITY_SHA256
            ),
            "operating_point_sha256": (
                slurm.goal.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "search_only_proposal": True,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
            "fea_submission_performed": False,
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": slurm.canonical_sha256(inventory),
            "terminal_physical_candidates_manifest": manifest,
        }
    )
    result_bytes = (
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    return submission, result_bytes, csv_bytes, manifest_bytes


def test_scheduler_payload_is_one_seed_max_resource_task() -> None:
    plan = {
        "bundle_id": "mft-goal-" + "a" * 24,
        "remote_bundle": "/gpfs/tmp_cpu2/mft_goal_20260726/deployment",
        "relocation_sources": {
            2607262000: (
                "artifacts/campaign/relocations/"
                "seed-2607262000-n1-5.json"
            )
        },
    }
    task = {
        "seed": 2607262000,
        "fixed_primary_turns": 5,
        "payload_sha256": "b" * 64,
    }

    payload = slurm.scheduler_payload(plan=plan, task=task)

    assert payload["cpus"] == 8
    assert payload["memory_mb"] == 65_536
    assert payload["timeout_seconds"] == 14_400
    assert payload["max_workers_per_node"] == 8
    assert payload["gpus"] == 0
    assert payload["payload_json"] == task
    assert "project" not in payload
    assert "artifacts/code/tools/mft_goal_20260726_launch.py" in payload[
        "command"
    ]
    assert 'output="runs/task-${SLURM_SCHED_TASK_ID:' in payload["command"]
    assert '--output "$output"' in payload["command"]
    assert "Scheduler" not in payload["remote_cwd"]


def test_safe_relative_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        slurm._safe_relative("../outside", "test")
    with pytest.raises(RuntimeError, match="unsafe"):
        slurm._safe_relative("/absolute", "test")
    with pytest.raises(RuntimeError, match="unsafe"):
        slurm._safe_relative(r"..\windows-escape", "test")
    assert slurm._safe_relative(
        "artifacts/code/tools/runner.py", "test"
    ) == "artifacts/code/tools/runner.py"


def test_atomic_relocation_seal_round_trip(tmp_path: Path) -> None:
    value = slurm._sealed(
        {
            "schema_version": slurm.RELOCATION_SCHEMA,
            "task_payload_sha256": "a" * 64,
            "paths": {
                "generation": "/gpfs/generation",
                "candidate": "/gpfs/candidate.json",
                "quality_status": "/gpfs/quality.json",
                "code_root": "/gpfs/code",
                "dataset": "/gpfs/data.parquet",
                "profile": "/gpfs/profile.json",
            },
            "code_manifest_path": "/gpfs/code_manifest.json",
            "source_absolute_paths_are_documentary_only": True,
            "remote_git_checkout_required": False,
        }
    )
    path = tmp_path / "relocation.json"
    slurm._atomic_json(path, value)

    observed = slurm._read_json(path)
    assert slurm._validate_seal(
        observed, slurm.RELOCATION_SCHEMA
    ) == observed
    observed["paths"]["dataset"] = "/gpfs/changed.parquet"
    with pytest.raises(RuntimeError, match="seal mismatch"):
        slurm._validate_seal(observed, slurm.RELOCATION_SCHEMA)


def test_submission_ledger_is_exact_512_task_seed_payload_cover() -> None:
    plan, tasks, ledger = _submission_fixture()

    authenticated = slurm._authenticate_submission_entries(
        plan=plan,
        ledger=ledger,
        expected_tasks=tasks,
        scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
    )

    assert len(authenticated) == slurm.EXPECTED_CAMPAIGN_TASKS
    assert len({item["task_id"] for item in authenticated}) == 512
    assert len({item["seed"] for item in authenticated}) == 512
    assert len(
        {item["task_payload_sha256"] for item in authenticated}
    ) == 512


def test_submission_ledger_rejects_duplicate_and_incomplete_cover() -> None:
    plan, tasks, ledger = _submission_fixture()
    duplicate = copy.deepcopy(ledger)
    duplicate["submissions"][1]["task_id"] = duplicate["submissions"][0][
        "task_id"
    ]
    with pytest.raises(RuntimeError, match="duplicate"):
        slurm._authenticate_submission_entries(
            plan=plan,
            ledger=duplicate,
            expected_tasks=tasks,
            scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
        )

    incomplete = copy.deepcopy(ledger)
    incomplete["submissions"].pop()
    with pytest.raises(RuntimeError, match="exactly 512"):
        slurm._authenticate_submission_entries(
            plan=plan,
            ledger=incomplete,
            expected_tasks=tasks,
            scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
        )


def _scheduler_status(plan, submission, *, status, exit_code, account="r1"):
    expected = slurm.scheduler_payload(
        plan=plan,
        task=submission["task"],
        priority=10,
    )
    return {
        "id": submission["task_id"],
        "task_id": submission["task_id"],
        "status": status,
        "exit_code": exit_code,
        "account_name": account,
        "actual_node_name": "node-1",
        "project": "",
        **{
            key: expected[key]
            for key in (
                "name",
                "remote_cwd",
                "required_capability",
                "env_profile",
                "cpus",
                "memory_mb",
                "scheduling_profile",
                "aedt_backend",
                "gpus",
                "priority",
                "timeout_seconds",
                "dedupe_key",
                "max_workers_per_node",
            )
        },
    }


def test_status_query_classifies_incomplete_and_failed_without_harvest(
    monkeypatch,
) -> None:
    plan, tasks, ledger = _submission_fixture()
    authenticated = slurm._authenticate_submission_entries(
        plan=plan,
        ledger=ledger,
        expected_tasks=tasks,
        scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
    )
    submission = authenticated[0]

    queued = _scheduler_status(
        plan,
        submission,
        status="queued",
        exit_code=None,
        account="",
    )
    monkeypatch.setattr(slurm, "_api_json_with_retry", lambda *_: queued)
    row = slurm._query_one_status(
        submission=submission,
        plan=plan,
        scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
        retries=1,
    )
    assert row["state_class"] == "incomplete"
    assert row["harvested"] is False
    assert row["error"] is None

    failed = _scheduler_status(
        plan,
        submission,
        status="failed",
        exit_code=19,
    )
    monkeypatch.setattr(slurm, "_api_json_with_retry", lambda *_: failed)
    row = slurm._query_one_status(
        submission=submission,
        plan=plan,
        scheduler_url=slurm.DEFAULT_SCHEDULER_URL,
        retries=1,
    )
    assert row["state_class"] == "terminal_failure"
    assert "exit_code=19" in row["error"]
    assert row["harvested"] is False


def test_result_header_rejects_tamper_and_path_traversal() -> None:
    submission, payload, _csv, _manifest = _result_fixture()
    result, records = slurm._validate_result_header(
        payload=payload,
        submission=submission,
    )
    assert result["task_payload_sha256"] == submission[
        "task_payload_sha256"
    ]
    assert set(records) == set(slurm.TERMINAL_ARTIFACT_ROLES)

    tampered = bytearray(payload)
    tampered[payload.index(b"mft-goal-20260726")] = ord("x")
    with pytest.raises(RuntimeError, match="seal mismatch"):
        slurm._validate_result_header(
            payload=bytes(tampered),
            submission=submission,
        )

    value = json.loads(payload)
    value.pop("payload_sha256")
    value["artifact_inventory"][
        "terminal_physical_candidates"
    ]["path"] = "../outside.csv"
    value["artifact_inventory_sha256"] = slurm.canonical_sha256(
        value["artifact_inventory"]
    )
    escaped = slurm._sealed(value)
    escaped_bytes = json.dumps(escaped, sort_keys=True).encode()
    with pytest.raises(RuntimeError, match="unsafe"):
        slurm._validate_result_header(
            payload=escaped_bytes,
            submission=submission,
        )


def test_only_completed_exit_zero_reaches_harvest(tmp_path: Path) -> None:
    submission, _payload, _csv, _manifest = _result_fixture()
    context = {
        "scheduler_url": slurm.DEFAULT_SCHEDULER_URL,
        "results_root": tmp_path,
        "plan": {
            "bundle_id": "mft-goal-" + "a" * 24,
            "remote_bundle": "/gpfs/goal",
        },
    }
    row = {
        "task_id": submission["task_id"],
        "scheduler_status": "running",
        "exit_code": None,
        "state_class": "incomplete",
    }
    with pytest.raises(RuntimeError, match="completed exit-0"):
        slurm._harvest_one_completed_task(
            context=context,
            submission=submission,
            row=row,
            connection=object(),
            retries=1,
        )


def test_resume_reuses_verified_local_terminal_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    submission, payload, csv_bytes, manifest_bytes = _result_fixture()
    task_id = submission["task_id"]
    task_dir = tmp_path / f"task-{task_id}"
    task_dir.mkdir()
    (task_dir / "terminal_physical_candidates.csv").write_bytes(csv_bytes)
    (
        task_dir / "terminal_physical_candidates.manifest.json"
    ).write_bytes(manifest_bytes)
    monkeypatch.setattr(
        slurm,
        "_fetch_result_bytes",
        lambda **_kwargs: payload,
    )
    monkeypatch.setattr(
        slurm.goal,
        "_validated_seed_table",
        lambda *_args, **_kwargs: ({}, object()),
    )

    class NoRemoteConnection:
        def get(self):
            raise AssertionError(
                "verified resume must not redownload terminal artifacts"
            )

    context = {
        "scheduler_url": slurm.DEFAULT_SCHEDULER_URL,
        "results_root": tmp_path,
        "plan": {
            "bundle_id": "mft-goal-" + "a" * 24,
            "remote_bundle": "/gpfs/goal",
        },
    }
    row = {
        "task_id": task_id,
        "scheduler_status": "completed",
        "exit_code": 0,
        "state_class": "completed_exit_zero",
        "scheduler_task_identity_sha256": "5" * 64,
        "harvested": False,
        "error": None,
    }
    harvested = slurm._harvest_one_completed_task(
        context=context,
        submission=submission,
        row=row,
        connection=NoRemoteConnection(),
        retries=1,
    )

    assert harvested["harvested"] is True
    assert all(
        not record["downloaded"]
        for record in harvested["terminal_artifacts"].values()
    )
    assert (task_dir / "result.json").is_file()
    receipt = slurm._read_json(task_dir / "harvest_receipt.json")
    slurm._validate_seal(
        receipt,
        slurm.TASK_HARVEST_RECEIPT_SCHEMA,
    )


def test_result_api_cap_falls_back_to_contained_sftp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    submission, payload, csv_bytes, manifest_bytes = _result_fixture()
    padding = b" " * (
        slurm.RESULT_JSON_MAX_BYTES - len(payload) + 100
    )
    full_payload = payload + padding
    api_tail = full_payload[-slurm.RESULT_JSON_MAX_BYTES :]
    task_id = submission["task_id"]
    task_dir = tmp_path / f"task-{task_id}"
    task_dir.mkdir()
    (task_dir / "terminal_physical_candidates.csv").write_bytes(csv_bytes)
    (
        task_dir / "terminal_physical_candidates.manifest.json"
    ).write_bytes(manifest_bytes)
    monkeypatch.setattr(
        slurm,
        "_fetch_result_bytes",
        lambda **_kwargs: api_tail,
    )
    identity = {
        "bytes": len(full_payload),
        "sha256": hashlib.sha256(full_payload).hexdigest(),
    }
    monkeypatch.setattr(
        slurm,
        "_contained_remote_identity",
        lambda *_args, **_kwargs: identity,
    )

    def fake_download(
        _connection,
        _remote_file,
        destination,
        _retries,
        **_kwargs,
    ):
        Path(destination).write_bytes(full_payload)
        return {
            "bytes": len(full_payload),
            "sha256": identity["sha256"],
            "transport": "direct_sftp",
            "attempts": 1,
        }

    monkeypatch.setattr(
        slurm.transport,
        "_download_verified_sftp",
        fake_download,
    )
    monkeypatch.setattr(
        slurm.goal,
        "_validated_seed_table",
        lambda *_args, **_kwargs: ({}, object()),
    )

    class FakeConnection:
        def get(self):
            return object()

    context = {
        "scheduler_url": slurm.DEFAULT_SCHEDULER_URL,
        "results_root": tmp_path,
        "plan": {
            "bundle_id": "mft-goal-" + "a" * 24,
            "remote_bundle": "/gpfs/goal",
        },
    }
    row = {
        "task_id": task_id,
        "scheduler_status": "completed",
        "exit_code": 0,
        "state_class": "completed_exit_zero",
        "scheduler_task_identity_sha256": "5" * 64,
        "harvested": False,
        "error": None,
    }
    harvested = slurm._harvest_one_completed_task(
        context=context,
        submission=submission,
        row=row,
        connection=FakeConnection(),
        retries=1,
    )

    assert harvested["harvested"] is True
    receipt = slurm._read_json(task_dir / "harvest_receipt.json")
    assert receipt["result"]["transport"] == (
        "scheduler_remote_file_probe_then_contained_sftp"
    )
    assert receipt["result"]["remote_sha256"] == identity["sha256"]


def test_harvest_cli_help_exposes_bounded_read_only_controls(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        slurm._parser().parse_args(["harvest", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--max-polls" in help_text
    assert "--poll-interval-seconds" in help_text
    assert "--require-ready" in help_text
    assert "--scheduler-url" in help_text
