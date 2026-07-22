from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import tier1_final1000_multiseed_controller as controller


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "tier1_final1000_live_v3_chain.json"
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_json(path: Path) -> tuple[bytes, dict]:
    payload = path.read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    return payload, value


def test_exact_live_v3_chain_upgrades_without_replaying_current_science_template():
    descriptor = _json(FIXTURE)
    root = Path(descriptor["runtime_root"])
    state_path = root / descriptor["source_state"]
    if not state_path.is_file():
        pytest.skip("exact operator-owned live-v3 fixture is unavailable")
    snapshot = descriptor["captured_snapshot"]
    source_payload, source_state = _read_json(state_path)
    source_copy = copy.deepcopy(source_state)
    plan_relatives = [descriptor["source_plan"], *descriptor["ancestor_plans"]]
    plan_reads = [_read_json(root / relative) for relative in plan_relatives]
    source_plan = plan_reads[0][1]
    ancestors = [item[1] for item in plan_reads[1:]]

    for relative, (payload, plan) in zip(plan_relatives, plan_reads, strict=True):
        expected = snapshot["plans"][relative]
        assert hashlib.sha256(payload).hexdigest() == expected["file_sha256"]
        assert plan["launch_plan_sha256"] == expected["launch_plan_sha256"]

    captured_source = snapshot["source"]
    source_file_sha256 = hashlib.sha256(source_payload).hexdigest()
    assert int(source_state["revision"]) >= int(captured_source["revision"])
    assert len(source_state["entries"]) >= int(captured_source["entry_count"])
    exact_captured_snapshot = source_file_sha256 == captured_source["file_sha256"]
    if exact_captured_snapshot:
        assert source_state["state_sha256"] == captured_source["state_sha256"]
        assert int(source_state["revision"]) == int(captured_source["revision"])
        assert len(source_state["entries"]) == int(captured_source["entry_count"])

    upgraded = controller.upgrade_v1_state(
        source_state,
        source_plan,
        source_plan=source_plan,
        ancestor_plans=ancestors,
        batch_length=4,
    )
    restarted = controller.validate_state(upgraded, source_plan)
    assert restarted == upgraded
    assert controller.validate_state(json.loads(json.dumps(upgraded)), source_plan) == upgraded
    assert source_state == source_copy
    assert len(upgraded["entries"]) == len(source_state["entries"])
    assert controller.active_physical_count(upgraded) == 500
    assert set(upgraded["harvest_cohorts"]) == set(
        descriptor["expected_harvest_cohort_ids"]
    )
    assert set(
        entry["source_fixed_generations"] for entry in upgraded["entries"]
    ) == set(descriptor["expected_generation_counts"])
    by_cohort = Counter(
        entry["source_harvest_cohort_id"] for entry in upgraded["entries"]
    )
    assert set(by_cohort) == set(descriptor["expected_harvest_cohort_ids"])
    by_cohort_generation = Counter(
        (
            entry["source_harvest_cohort_id"],
            int(entry["source_fixed_generations"]),
        )
        for entry in upgraded["entries"]
    )
    for identity, captured_count in snapshot["upgrade"][
        "entry_counts_by_cohort_and_generations"
    ].items():
        cohort_id, generations = identity.rsplit(":", 1)
        assert by_cohort_generation[(cohort_id, int(generations))] >= int(
            captured_count
        )
    if exact_captured_snapshot:
        assert upgraded["state_sha256"] == snapshot["upgrade"]["state_sha256"]
        assert len(upgraded["entries"]) == snapshot["upgrade"]["entry_count"]
    for entry in upgraded["entries"]:
        assert (
            entry["task_envelope"]["payload_json"]["max_generations"]
            == entry["source_fixed_generations"]
        )
        assert (
            upgraded["harvest_cohorts"][entry["source_harvest_cohort_id"]][
                "launch_plan_sha256"
            ]
            == entry["source_launch_plan_sha256"]
        )
