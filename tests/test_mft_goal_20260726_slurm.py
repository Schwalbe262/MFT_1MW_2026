from __future__ import annotations

from pathlib import Path

import pytest

from tools import mft_goal_20260726_slurm as slurm


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
