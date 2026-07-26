from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import mft_goal_diagnostic_final_handoff_v2 as handoff


def _sealed_terminal(root: Path, *, state: str, status: str, exit_code):
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    stdout.write_text("terminal output\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    value = {
        "schema": handoff.TERMINAL_SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "candidate_sha256": handoff.CANDIDATE_SHA256,
        "fixed_physics_sha256": handoff.FIXED_PHYSICS_SHA256,
        "diagnostic_temperature_observation_available": False,
        "scheduler_access": {
            "methods_used": ["GET"],
            "mutation_performed": False,
        },
        "task": {
            "task_id": 96313,
            "name": (
                "mft-goal-corrected-thermal-l96230-b7c30cb70b95-native-r5-n111"
            ),
            "account_name": "r1jae262",
            "actual_node_name": "n111",
            "assigned_allocation": 14641,
            "slurm_job_id": "838708",
            "cpus": 8,
            "memory_mb": 294912,
            "timeout_seconds": 6600,
            "state": state,
            "status": status,
            "exit_code": exit_code,
        },
        "streams": {
            "stdout": handoff._record(stdout, relative_to=root),
            "stderr": handoff._record(stderr, relative_to=root),
        },
    }
    value["payload_sha256"] = handoff.canonical_sha256(value)
    path = root / "terminal_evidence.json"
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _mock_scheduler(
    monkeypatch: pytest.MonkeyPatch, terminal_path: Path
) -> None:
    value = json.loads(terminal_path.read_text(encoding="utf-8"))
    task = dict(value["task"])
    task["id"] = task.pop("task_id")
    root = terminal_path.parent
    streams = {
        name: (root / record["path"]).read_bytes()
        for name, record in value["streams"].items()
    }
    monkeypatch.setattr(
        handoff,
        "_scheduler_authority",
        lambda _url: (task, streams),
    )


def test_pending_terminal_evidence_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign" / "terminal"
    root.mkdir(parents=True)
    path = _sealed_terminal(
        root, state="running", status="running", exit_code=None
    )
    _mock_scheduler(monkeypatch, path)

    with pytest.raises(handoff.HandoffError, match="publication refused"):
        handoff.validate_terminal_evidence(
            path, tmp_path / "campaign", scheduler_url="http://127.0.0.1:8002"
        )


def test_terminal_evidence_detects_stream_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    root = campaign / "terminal"
    root.mkdir(parents=True)
    path = _sealed_terminal(
        root, state="failed", status="failed", exit_code=124
    )
    _mock_scheduler(monkeypatch, path)
    (root / "stdout.log").write_text("drift\n", encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="stream evidence drifted"):
        handoff.validate_terminal_evidence(
            path, campaign, scheduler_url="http://127.0.0.1:8002"
        )


@pytest.mark.parametrize(
    ("state", "status", "exit_code"),
    [
        ("succeeded", "completed", False),
        ("succeeded", "failed", 0),
        ("failed", "failed", None),
        ("failed", "completed", 124),
        ("cancelled", "failed", 1),
    ],
)
def test_terminal_union_rejects_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    status: str,
    exit_code,
) -> None:
    campaign = tmp_path / "campaign"
    root = campaign / "terminal"
    root.mkdir(parents=True)
    path = _sealed_terminal(
        root, state=state, status=status, exit_code=exit_code
    )
    _mock_scheduler(monkeypatch, path)

    with pytest.raises(handoff.HandoffError, match="terminal union"):
        handoff.validate_terminal_evidence(
            path, campaign, scheduler_url="http://127.0.0.1:8002"
        )


def test_terminal_evidence_must_match_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    root = campaign / "terminal"
    root.mkdir(parents=True)
    path = _sealed_terminal(
        root, state="failed", status="failed", exit_code=124
    )
    _mock_scheduler(monkeypatch, path)
    original = handoff._scheduler_authority

    def drifted(url: str):
        task, streams = original(url)
        task = dict(task)
        task["exit_code"] = 1
        return task, streams

    monkeypatch.setattr(handoff, "_scheduler_authority", drifted)
    with pytest.raises(handoff.HandoffError, match="live Scheduler GET"):
        handoff.validate_terminal_evidence(
            path, campaign, scheduler_url="http://127.0.0.1:8002"
        )


def test_no_replace_publish_and_existing_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    terminal_root = campaign / "terminal"
    terminal_root.mkdir(parents=True)
    terminal_path = _sealed_terminal(
        terminal_root, state="failed", status="failed", exit_code=124
    )
    _mock_scheduler(monkeypatch, terminal_path)
    terminal = handoff.validate_terminal_evidence(
        terminal_path,
        campaign,
        scheduler_url="http://127.0.0.1:8002",
    )
    reference = {
        "root": "authority",
        "sha256": "1" * 64,
        "non_authoritative_alternate_standard": {
            "path": "alternate.csv",
            "sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(handoff, "_validate_pareto", lambda _base: reference)
    monkeypatch.setattr(
        handoff, "_validate_model_package", lambda _base: reference
    )
    monkeypatch.setattr(
        handoff, "_validate_open_collection", lambda _base: reference
    )
    monkeypatch.setattr(
        handoff, "_validate_full_failure", lambda _base: reference
    )

    first = handoff.publish_handoff(
        base_root=campaign,
        terminal_evidence_path=terminal_path,
        scheduler_url="http://127.0.0.1:8002",
        output_name="handoff-v2",
    )
    second = handoff.publish_handoff(
        base_root=campaign,
        terminal_evidence_path=terminal_path,
        scheduler_url="http://127.0.0.1:8002",
        output_name="handoff-v2",
    )

    assert first["status"] == "published"
    assert second["status"] == "already_published"
    assert terminal["terminal_branch"] == "failure"
    assert sorted(
        path.name for path in (campaign / "handoff-v2").iterdir()
    ) == [
        "DIAGNOSTIC_ONLY_DO_NOT_PROMOTE.txt",
        "handoff_manifest.json",
        "handoff_seal.json",
    ]
    (campaign / "handoff-v2" / "extra").mkdir()
    with pytest.raises(handoff.HandoffError, match="non-regular entry"):
        handoff.publish_handoff(
            base_root=campaign,
            terminal_evidence_path=terminal_path,
            scheduler_url="http://127.0.0.1:8002",
            output_name="handoff-v2",
        )
