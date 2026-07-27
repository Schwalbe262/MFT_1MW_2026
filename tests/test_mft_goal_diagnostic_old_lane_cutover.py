from __future__ import annotations

import copy
import html
import json
from pathlib import Path
import threading

import pytest

from tools import mft_goal_diagnostic_old_lane_cutover as cutover


class FakeScheduler:
    def __init__(
        self,
        tasks: dict[int, dict],
        payloads: dict[int, dict],
        *,
        race_to_terminal: set[int] | None = None,
    ) -> None:
        self.tasks = copy.deepcopy(tasks)
        self.payloads = copy.deepcopy(payloads)
        self.race_to_terminal = set(race_to_terminal or ())
        self.get_count = 0
        self.html_get_count = 0
        self.cancel_count = 0
        self.cancel_calls: list[tuple[int, str]] = []
        self._lock = threading.Lock()

    def get_task(self, task_id: int) -> dict:
        with self._lock:
            self.get_count += 1
            return copy.deepcopy(self.tasks[task_id])

    def get_task_html(self, task_id: int) -> str:
        with self._lock:
            self.html_get_count += 1
            task = self.tasks[task_id]
            payload = self.payloads[task_id]
            return (
                "<html><body>"
                f"<h1>Task {task_id}: {html.escape(task['name'])}</h1>"
                "<table>"
                "<tr><th>Remote CWD</th>"
                f'<td class="mono-cell">{html.escape(task["remote_cwd"])}</td>'
                "</tr>"
                '<tr><th>Payload JSON</th><td class="mono-cell"><pre>'
                f"{html.escape(json.dumps(payload, sort_keys=True))}"
                "</pre></td></tr>"
                "</table></body></html>"
            )

    def cancel_task(self, task_id: int, *, expected_status: str) -> dict:
        assert task_id in cutover.OLD_TASK_IDS
        assert expected_status in cutover.CANCELLABLE_STATUSES
        with self._lock:
            if task_id in self.race_to_terminal:
                self.race_to_terminal.remove(task_id)
                self.tasks[task_id]["status"] = "completed"
                raise cutover.ConditionalCancelRejected("status raced")
            if self.tasks[task_id]["status"] != expected_status:
                raise cutover.ConditionalCancelRejected("status changed")
            self.cancel_count += 1
            self.cancel_calls.append((task_id, expected_status))
            self.tasks[task_id]["status"] = "cancelled"
            return {"task_id": task_id, "status": "cancelled"}


def _sealed(value: dict) -> dict:
    return cutover._seal(value)


def _old_profile(monkeypatch: pytest.MonkeyPatch) -> dict:
    profile = _sealed(
        {
            "schema_version": "test-old-profile",
            "fixed_secondary_interturn_gap_mm": 0.35,
            "fixed_primary_turns": 6,
            "fixed_secondary_turns": 60,
            "authorized_seed_start": cutover.OLD_SEED_START,
            "authorized_seed_end_inclusive": cutover.OLD_AUTHORIZED_SEED_END,
        }
    )
    monkeypatch.setattr(cutover, "OLD_PROFILE_SHA256", profile["payload_sha256"])
    return profile


def _old_payload(seed: int, profile: dict) -> dict:
    return _sealed(
        {
            "schema_version": "test-old-task",
            "seed": seed,
            "fixed_primary_turns": 6,
            "manufacturing_search_profile_payload_sha256": profile["payload_sha256"],
            "manufacturing_search_profile": copy.deepcopy(profile),
            "activation": {
                "aligned_compact_bank_proof": {
                    "search_profile_payload_sha256": profile["payload_sha256"],
                    "fixed_secondary_interturn_gap_mm": 0.35,
                    "all_rows_gap2_exactly_fixed": True,
                }
            },
        }
    )


def _new_profile() -> dict:
    return _sealed(
        {
            "schema_version": "test-new-profile",
            "fixed_primary_turns": 6,
            "fixed_secondary_turns": 60,
            "authorized_seed_start": cutover.NEW_SEED_START,
            "authorized_seed_end_inclusive": cutover.NEW_SEED_END,
            "secondary_interturn_gap_search_mm": {
                "minimum": 0.35,
                "maximum": 2.0,
                "step": 0.001,
            },
            "secondary_conductor_thickness_search_mm": {
                "minimum": 0.3,
                "maximum": 1.0,
            },
            "scheduler_priority": 100,
        }
    )


def _new_payload(seed: int, profile: dict) -> dict:
    return _sealed(
        {
            "schema_version": "test-new-task",
            "seed": seed,
            "manufacturing_search_profile_payload_sha256": profile["payload_sha256"],
            "manufacturing_search_profile": copy.deepcopy(profile),
            "activation": {
                "aligned_compact_bank_proof": {
                    "search_profile_payload_sha256": profile["payload_sha256"],
                    "secondary_gap_mode": "bounded_gap2_cw2",
                    "all_rows_gap2_within_allowed_band": True,
                    "all_rows_cw2_within_allowed_band": True,
                }
            },
        }
    )


def _inventory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    old_statuses: dict[int, str] | None = None,
) -> tuple[FakeScheduler, dict]:
    old_profile = _old_profile(monkeypatch)
    new_profile = _new_profile()
    old_statuses = old_statuses or {}
    tasks: dict[int, dict] = {}
    payloads: dict[int, dict] = {}
    for task_id in cutover.OLD_TASK_IDS:
        seed = cutover.OLD_SEED_START + task_id - cutover.OLD_TASK_ID_START
        name = f"{cutover.OLD_BUNDLE_NAME}-s{seed}-n1-6"
        tasks[task_id] = {
            "task_id": task_id,
            "name": name,
            "status": old_statuses.get(task_id, "completed"),
            "priority": cutover.OLD_PRIORITY,
            "dedupe_key": (cutover.OLD_DEDUPE_PREFIX + f"{task_id:064x}"[-64:]),
            "remote_cwd": f"/remote/account/{cutover.OLD_BUNDLE_NAME}",
        }
        payloads[task_id] = _old_payload(seed, old_profile)
    for ordinal, seed in enumerate(
        range(cutover.NEW_SEED_START, cutover.NEW_SEED_END + 1)
    ):
        task_id = 98000 + ordinal
        bundle = "mft-goal-diag-variable-gap-exact100-test"
        name = f"{bundle}-s{seed}-n1-6"
        tasks[task_id] = {
            "task_id": task_id,
            "name": name,
            "status": "queued",
            "priority": cutover.NEW_PRIORITY,
            "dedupe_key": f"new:{seed}",
            "remote_cwd": f"/remote/account/{bundle}",
        }
        payloads[task_id] = _new_payload(seed, new_profile)
    return FakeScheduler(tasks, payloads), new_profile


def _write_new_receipt(path: Path, scheduler: FakeScheduler) -> dict:
    rows = []
    for ordinal, seed in enumerate(
        range(cutover.NEW_SEED_START, cutover.NEW_SEED_END + 1)
    ):
        task_id = 98000 + ordinal
        task = scheduler.tasks[task_id]
        rows.append(
            {
                "seed": seed,
                "task_id": task_id,
                "name": task["name"],
                "dedupe_key": task["dedupe_key"],
                "status": task["status"],
            }
        )
    value = _sealed(
        {
            "schema_version": cutover.NEW_RECEIPT_SCHEMA,
            "apply": True,
            "bundle_id": "mft-goal-diag-variable-gap-exact100-test",
            "task_count": 100,
            "absent_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "tasks": rows,
        }
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    old_statuses: dict[int, str] | None = None,
    race_to_terminal: set[int] | None = None,
) -> tuple[FakeScheduler, Path, Path]:
    scheduler, new_profile = _inventory(monkeypatch, old_statuses=old_statuses)
    scheduler.race_to_terminal = set(race_to_terminal or ())
    receipt_path = tmp_path / "new_receipt.json"
    evidence_path = tmp_path / "new_readback.json"
    _write_new_receipt(receipt_path, scheduler)
    cutover.attest_new_readback(
        new_receipt_path=receipt_path,
        new_profile_sha256=new_profile["payload_sha256"],
        evidence_out=evidence_path,
        scheduler=scheduler,
    )
    return scheduler, receipt_path, evidence_path


def test_old_audit_authenticates_exact82_and_never_mutates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, _profile = _inventory(
        monkeypatch,
        old_statuses={
            cutover.OLD_TASK_ID_START: "running",
            cutover.OLD_TASK_ID_START + 1: "queued",
        },
    )

    audit = cutover.audit_old(scheduler)

    assert len(audit["rows"]) == 82
    assert audit["cancellable_task_ids"] == [97297, 97298]
    assert audit["terminal_preserved_task_ids"] == list(range(97299, 97379))
    assert scheduler.get_count == 82
    assert scheduler.html_get_count == 82
    assert scheduler.cancel_count == 0


def test_old_audit_profile_drift_fails_before_any_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, _profile = _inventory(monkeypatch)
    scheduler.payloads[97297]["manufacturing_search_profile_payload_sha256"] = "f" * 64
    unsigned = dict(scheduler.payloads[97297])
    unsigned.pop("payload_sha256")
    scheduler.payloads[97297]["payload_sha256"] = cutover._sha256_value(unsigned)

    with pytest.raises(cutover.CutoverError, match="identity drifted"):
        cutover.audit_old(scheduler)

    assert scheduler.cancel_count == 0


def test_attest_new_binds_sealed_receipt_and_live_exact100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, profile = _inventory(monkeypatch)
    receipt_path = tmp_path / "receipt.json"
    evidence_path = tmp_path / "evidence.json"
    receipt = _write_new_receipt(receipt_path, scheduler)

    evidence = cutover.attest_new_readback(
        new_receipt_path=receipt_path,
        new_profile_sha256=profile["payload_sha256"],
        evidence_out=evidence_path,
        scheduler=scheduler,
    )

    assert evidence["new_receipt_payload_sha256"] == receipt["payload_sha256"]
    assert evidence["task_count"] == 100
    assert len(evidence["tasks"]) == 100
    assert evidence["scheduler_cancel_count"] == 0
    assert evidence_path.is_file()


def test_cutover_default_is_audit_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, receipt_path, evidence_path = _authority(
        tmp_path,
        monkeypatch,
        old_statuses={97297: "running", 97298: "queued"},
    )
    before_cancel_count = scheduler.cancel_count

    result = cutover.run_cutover(
        new_receipt_path=receipt_path,
        new_readback_evidence_path=evidence_path,
        output_root=tmp_path / "cutover",
        scheduler=scheduler,
    )

    assert result["schema_version"] == cutover.CUTOVER_AUDIT_SCHEMA
    assert result["apply"] is False
    assert result["selected_conditional_cancel_task_ids"] == [97297, 97298]
    assert scheduler.cancel_count == before_cancel_count == 0
    assert not (tmp_path / "cutover" / "pre_cutover_receipt.json").exists()


def test_apply_rejects_tampered_readback_before_scheduler_reads_or_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, receipt_path, evidence_path = _authority(
        tmp_path, monkeypatch, old_statuses={97297: "running"}
    )
    value = json.loads(evidence_path.read_text("utf-8"))
    value.pop("payload_sha256")
    value["new_receipt_file_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(_sealed(value)), encoding="utf-8")
    get_before = scheduler.get_count

    with pytest.raises(cutover.CutoverError, match="evidence mismatch"):
        cutover.run_cutover(
            new_receipt_path=receipt_path,
            new_readback_evidence_path=evidence_path,
            output_root=tmp_path / "cutover",
            apply=True,
            scheduler=scheduler,
        )

    assert scheduler.get_count == get_before
    assert scheduler.cancel_count == 0


def test_apply_cancels_only_fresh_nonterminal_rows_with_cas_and_seals_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, receipt_path, evidence_path = _authority(
        tmp_path,
        monkeypatch,
        old_statuses={
            97297: "running",
            97298: "queued",
            97299: "completed",
        },
    )
    output = tmp_path / "cutover"

    result = cutover.run_cutover(
        new_receipt_path=receipt_path,
        new_readback_evidence_path=evidence_path,
        output_root=output,
        apply=True,
        scheduler=scheduler,
    )

    assert result["schema_version"] == cutover.POST_CUTOVER_SCHEMA
    assert scheduler.cancel_calls == [(97297, "running"), (97298, "queued")]
    assert scheduler.tasks[97299]["status"] == "completed"
    assert result["conditional_cancelled_task_ids"] == [97297, 97298]
    assert result["out_of_range_cancel_count"] == 0
    assert result["terminal_cancel_count"] == 0
    assert (output / "pre_cutover_receipt.json").is_file()
    assert (output / "post_cutover_receipt.json").is_file()

    replay = cutover.run_cutover(
        new_receipt_path=receipt_path,
        new_readback_evidence_path=evidence_path,
        output_root=output,
        apply=True,
        scheduler=scheduler,
    )
    assert replay == result
    assert scheduler.cancel_calls == [(97297, "running"), (97298, "queued")]


def test_apply_preserves_task_that_becomes_terminal_before_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, receipt_path, evidence_path = _authority(
        tmp_path,
        monkeypatch,
        old_statuses={97297: "running"},
        race_to_terminal={97297},
    )

    result = cutover.run_cutover(
        new_receipt_path=receipt_path,
        new_readback_evidence_path=evidence_path,
        output_root=tmp_path / "cutover",
        apply=True,
        scheduler=scheduler,
    )

    action = next(row for row in result["actions"] if row["task_id"] == 97297)
    assert action["result"] == "terminal_preserved_after_cas_rejection"
    assert action["cancel_post_performed"] is False
    assert scheduler.tasks[97297]["status"] == "completed"
    assert scheduler.cancel_count == 0
