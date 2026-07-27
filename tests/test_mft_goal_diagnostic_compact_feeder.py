from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from tools import mft_goal_diagnostic_compact_feeder as feeder
from tools import mft_goal_diagnostic_compact_slurm as offload


def _fixture(tmp_path: Path):
    root = tmp_path / "feeder"
    root.mkdir()
    (root / "claims").mkdir()
    start = feeder.scout.CONTINUATION_SEED_START
    authority = feeder._sealed(
        {
            "output_root": str(root),
            "continuation_seed_start": start,
            "continuation_seed_end_inclusive": start + 99,
            "continuation_seed_count": 100,
            "allowed_accounts": list(feeder.ALLOWED_ACCOUNTS),
            "base_tasks": [
                {
                    "seed": 2_607_264_100 + index,
                    "task_id": index + 1,
                    "name": f"base-{index}",
                    "dedupe_key": f"base-dedupe-{index}",
                }
                for index in range(100)
            ],
            "recovery_tasks": [
                {
                    "seed": 2_607_264_153 + index,
                    "task_id": 101 + index,
                    "name": f"recovery-{index}",
                    "dedupe_key": f"recovery-dedupe-{index}",
                    "requested_account": feeder.ALLOWED_ACCOUNTS[
                        index % len(feeder.ALLOWED_ACCOUNTS)
                    ],
                }
                for index in range(47)
            ],
            "baseline_attempt_count": 147,
        }
    )
    identity = "a" * 24
    plan = {
        "bundle_id": f"mft-goal-diag-compact-{identity}",
        "contract_sha256": "b" * 64,
        "diagnostic_plan_sha256": "c" * 64,
        "remote_bundle": f"/remote/{identity}",
        "scheduler_priority": offload.SCHEDULER_PRIORITY,
        "relocation_sources": {
            str(seed): (
                "artifacts/campaign/relocations/"
                f"seed-{seed}-n1-6.json"
            )
            for seed in range(start, start + 100)
        },
    }
    tasks = [
        {
            "seed": seed,
            "fixed_primary_turns": 6,
            "payload_sha256": f"{ordinal:064x}",
        }
        for ordinal, seed in enumerate(range(start, start + 100), 1)
    ]
    return authority, plan, tasks


def _scheduler_row(
    task_id: int,
    *,
    name: str,
    dedupe_key: str,
    status: str,
    account: str = "",
) -> dict[str, Any]:
    return {
        "id": task_id,
        "task_id": task_id,
        "name": name,
        "dedupe_key": dedupe_key,
        "status": status,
        "state": "succeeded" if status == "completed" else status,
        "requested_account_name": account,
    }


class FakeScheduler:
    def __init__(
        self,
        *,
        base: dict[int, dict[str, Any]],
        extension: dict[int, dict[str, Any]] | None = None,
    ):
        self.base = base
        self.extension = extension if extension is not None else {}
        self.post_count = 0
        self.get_count = 0

    def list_namespace_tasks(self, _prefix: str):
        self.get_count += 1
        return list(self.extension.values())

    def get_task(self, task_id: int):
        self.get_count += 1
        if task_id in self.base:
            return dict(self.base[task_id])
        for row in self.extension.values():
            if row["id"] == task_id:
                return dict(row)
        raise AssertionError(task_id)

    def submit_task(self, payload: Mapping[str, Any]):
        self.post_count += 1
        seed = int(payload["payload_json"]["seed"])
        task_id = 10_000 + seed
        assert payload["account_name"] in feeder.ALLOWED_ACCOUNTS
        self.extension[seed] = _scheduler_row(
            task_id,
            name=str(payload["name"]),
            dedupe_key=str(payload["dedupe_key"]),
            status="queued",
            account=str(payload["account_name"]),
        )
        return {"task_id": task_id, "deduped": False}


def test_pinned_payload_round_robins_only_healthy_accounts(
    tmp_path: Path,
) -> None:
    authority, plan, tasks = _fixture(tmp_path)
    payloads = [
        feeder.pinned_scheduler_payload(
            authority=authority, plan=plan, task=task
        )
        for task in tasks[:6]
    ]
    assert [row["account_name"] for row in payloads] == [
        "dhj02",
        "jji0930",
        "r1jae262",
        "dw16",
        "wjddn5916",
        "dhj02",
    ]
    assert all(
        row["dedupe_key"].startswith(feeder.DEDUPE_PREFIX)
        for row in payloads
    )
    assert all("harry261" not in row.values() for row in payloads)
    assert all("exec python -W ignore " in row["command"] for row in payloads)


def test_cycle_fills_only_deficit_and_restart_does_not_duplicate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    authority, plan, tasks = _fixture(tmp_path)
    monkeypatch.setattr(
        feeder,
        "load_authority",
        lambda _path: (authority, plan, tasks),
    )
    base = {
        row["task_id"]: _scheduler_row(
            row["task_id"],
            name=row["name"],
            dedupe_key=row["dedupe_key"],
            status="running" if index < 51 else "failed",
        )
        for index, row in enumerate(authority["base_tasks"])
    }
    base.update(
        {
            row["task_id"]: _scheduler_row(
                row["task_id"],
                name=row["name"],
                dedupe_key=row["dedupe_key"],
                status="queued",
                account=row["requested_account"],
            )
            for row in authority["recovery_tasks"]
        }
    )
    shared_extension: dict[int, dict[str, Any]] = {}
    first_client = FakeScheduler(
        base=base, extension=shared_extension
    )
    first = feeder.run_cycle(
        authority_path=tmp_path / "unused.json",
        receipt_out=tmp_path / "cycle-1.json",
        apply=True,
        scheduler=first_client,
    )
    assert first["active_before"] == 98
    assert first["recovery_active_count"] == 47
    assert first["selected_seed_count"] == 2
    assert first_client.post_count == 2
    assert first["final_promotion_allowed"] is False
    assert first["equal_three_leg_air_gap_FEA_required"] is True

    second_client = FakeScheduler(
        base=base, extension=shared_extension
    )
    second = feeder.run_cycle(
        authority_path=tmp_path / "unused.json",
        receipt_out=tmp_path / "cycle-2.json",
        apply=True,
        scheduler=second_client,
    )
    assert second["active_before"] == 100
    assert second["selected_seed_count"] == 0
    assert second_client.post_count == 0
    assert len(shared_extension) == 2


def test_live_53_plus_47_baseline_dry_cycle_selects_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    authority, plan, tasks = _fixture(tmp_path)
    monkeypatch.setattr(
        feeder,
        "load_authority",
        lambda _path: (authority, plan, tasks),
    )
    baseline: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(authority["base_tasks"]):
        baseline[row["task_id"]] = _scheduler_row(
            row["task_id"],
            name=row["name"],
            dedupe_key=row["dedupe_key"],
            status="running" if index < 53 else "failed",
        )
    for row in authority["recovery_tasks"]:
        baseline[row["task_id"]] = _scheduler_row(
            row["task_id"],
            name=row["name"],
            dedupe_key=row["dedupe_key"],
            status="queued",
            account=row["requested_account"],
        )
    client = FakeScheduler(base=baseline)
    cycle = feeder.run_cycle(
        authority_path=tmp_path / "unused.json",
        receipt_out=tmp_path / "dry.json",
        apply=False,
        scheduler=client,
    )
    assert cycle["base_active_count"] == 53
    assert cycle["recovery_active_count"] == 47
    assert cycle["active_before"] == 100
    assert cycle["selected_seeds"] == []
    assert client.post_count == 0


def test_cycle_receipts_resume_at_next_monotonic_index(
    tmp_path: Path,
) -> None:
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    authority_sha = "f" * 64
    value = feeder._sealed(
        {
            "schema_version": feeder.CYCLE_SCHEMA,
            "feeder_authority_payload_sha256": authority_sha,
        }
    )
    feeder._atomic_json(cycles / "cycle-00000001.json", value)
    assert (
        feeder._next_cycle_index(
            cycles, authority_payload_sha256=authority_sha
        )
        == 2
    )
    feeder._atomic_json(cycles / "cycle-00000003.json", value)
    with pytest.raises(RuntimeError, match="not contiguous"):
        feeder._next_cycle_index(
            cycles, authority_payload_sha256=authority_sha
        )
