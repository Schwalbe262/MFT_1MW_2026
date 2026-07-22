from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tools import tier1_final1000_paired_resource_comparator as comparator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "docs"
    / "evidence"
    / "tier1_final1000_paired_resource_comparator_87052_20260723.json"
)
PACKAGE_PATH = (
    ROOT
    / "docs"
    / "evidence"
    / "tier1_final1000_paired_resource_comparator_87052_package_20260723.json"
)


def _config() -> dict[str, Any]:
    return comparator.validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _package() -> dict[str, Any]:
    return comparator.validate_package(
        json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    )


def test_exact_source_pair_and_current_capacity_are_sealed() -> None:
    package = _package()
    anchor = package["anchor"]
    assert anchor["task_id"] == 87052
    assert anchor["seed"] == 2257498123
    assert anchor["source_parent_task_sha256"] == comparator.SOURCE_PARENT_TASK_SHA256
    assert anchor["source_child_task_sha256"] == comparator.SOURCE_CHILD_TASK_SHA256
    assert package["capacity"]["eligible_allocation_count"] == 41
    assert package["capacity"]["exact_fit_slots"] == {"2": 719, "3": 567, "4": 459}
    assert package["capacity"]["allocation_local"] is True
    assert package["decision_policy"]["per_seed_latency_gate"] is None
    assert package["decision_policy"]["per_seed_latency_override_allowed"] is False
    assert package["batch_rollout_policy"]["target_formula"] == "exact_fit_slots[selected_cpus]"
    assert package["batch_rollout_policy"]["recompute_capacity_each_controller_cycle"] is True
    assert package["batch_rollout_policy"]["account_and_node_limits_are_hard"] is True
    assert package["batch_rollout_policy"]["resource2_rollout_currently_allowed"] is False


def test_companions_change_only_explicit_resource_and_namespace_fields() -> None:
    package = _package()
    source = package["anchor"]["child_task"]
    expected_envelope_changes = {
        "/name",
        "/command",
        "/cpus",
        "/dedupe_key",
        "/payload_json/inference_threads",
        "/payload_json/scheduler_cpus",
    }
    dedupes: set[str] = set()
    for cpus in comparator.COMPANION_CPUS:
        task = package["companions"][str(cpus)]
        assert comparator._changed_paths(source, task) == expected_envelope_changes
        assert comparator._changed_paths(
            source["payload_json"], task["payload_json"]
        ) == {"/inference_threads", "/scheduler_cpus"}
        assert comparator._payload_without_resource_fields(
            source["payload_json"]
        ) == comparator._payload_without_resource_fields(task["payload_json"])
        assert task["cpus"] == cpus
        assert task["memory_mb"] == 28672
        assert task["max_workers_per_node"] == 32
        assert task["payload_json"]["inference_threads"] == cpus
        assert task["payload_json"]["scheduler_cpus"] == cpus
        assert task["dedupe_key"].startswith(
            f"{comparator.COMPARATOR_DEDUPE_PREFIX}c{cpus}:"
        )
        assert task["dedupe_key"] not in dedupes
        dedupes.add(task["dedupe_key"])
        for variable in comparator.THREAD_VARIABLES:
            assert f"export {variable}={cpus}" in task["command"]
        assert f" {cpus} {comparator.MEMORY_MB * 1024**2} " in task["command"]
        assert "tier1_corrected_current7_slurm_seed_runner.py" in task["command"]
        assert "tier1_final1000_multiseed_phase_b_runner.py" not in task["command"]


def test_package_rebuilds_without_external_or_timestamp_state() -> None:
    package = _package()
    source = {
        "parent_task": package["anchor"]["parent_task"],
        "child_task": package["anchor"]["child_task"],
    }
    assert comparator._assemble_package(config=_config(), source=source) == package


def test_embedded_remote_wrapper_is_valid_python() -> None:
    compile(comparator._INLINE_WRAPPER, "<paired-resource-wrapper>", "exec")


def test_science_fingerprint_ignores_only_resource_binding() -> None:
    result4 = {
        "payload_sha256": "a" * 64,
        "optimizer_pid": 44,
        "inference_threads": 4,
        "scheduler_cpus": 4,
        "terminal": {
            "optimizer_inference_threads": 4,
            "optimizer_binding": {
                "threads_per_model": 4,
                "family_threads": {"xgboost": 4},
                "policy": "same",
            },
            "pareto_sha256": "b" * 64,
        },
    }
    result2 = copy.deepcopy(result4)
    result2.update(
        payload_sha256="c" * 64,
        optimizer_pid=99,
        inference_threads=2,
        scheduler_cpus=2,
    )
    binding = result2["terminal"]["optimizer_binding"]
    result2["terminal"]["optimizer_inference_threads"] = 2
    binding["threads_per_model"] = 2
    binding["family_threads"] = {"xgboost": 2}
    assert comparator.science_fingerprint(result2) == comparator.science_fingerprint(result4)
    result2["terminal"]["pareto_sha256"] = "d" * 64
    assert comparator.science_fingerprint(result2) != comparator.science_fingerprint(result4)


def _summary(cpus: int, wall: float, fingerprint: str = "f" * 64) -> dict[str, Any]:
    return {
        "cpus": cpus,
        "wall_time_seconds": wall,
        "all_resource_gates_passed": True,
        "science_fingerprint": {"projection_sha256": fingerprint},
    }


def test_decision_uses_slot_weighted_throughput_without_latency_override() -> None:
    package = _package()
    decision = comparator.decide_throughput(
        capacity=package["capacity"],
        summaries={
            "2": _summary(2, 130.0),
            "3": _summary(3, 100.0),
            "4": _summary(4, 100.0),
        },
    )
    assert decision["selected_cpus"] == 3
    assert decision["selected_exact_active_target"] == 567
    assert decision["resource_change_recommended"] is True
    assert decision["metrics"]["3"]["slot_weighted_throughput_ratio_vs_4cpu"] == pytest.approx(567 / 459)
    assert decision["per_seed_latency_gate_applied"] is False
    assert decision["per_seed_latency_override_applied"] is False
    assert decision["automatic_promotion_performed"] is False


def test_science_mismatch_retains_anchor_even_if_candidate_is_faster() -> None:
    package = _package()
    decision = comparator.decide_throughput(
        capacity=package["capacity"],
        summaries={
            "2": _summary(2, 50.0, "2" * 64),
            "3": _summary(3, 50.0, "3" * 64),
            "4": _summary(4, 100.0, "4" * 64),
        },
    )
    assert decision["science_fingerprints_all_match"] is False
    assert decision["selected_cpus"] == 4
    assert decision["resource_change_recommended"] is False


def test_failed_87052_makes_every_submission_path_launch_forbidden() -> None:
    package = _package()

    class NeverCalled:
        get_count = 0
        post_count = 0

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - assertion aid
            raise AssertionError(f"Scheduler was called through {name}")

    scheduler = NeverCalled()
    with pytest.raises(RuntimeError, match="launch forbidden"):
        comparator._submission_outcome(
            package=package,
            approval={},
            anchor_terminal={},
            scheduler=scheduler,
            cpus=2,
            apply=False,
            receipt_out=None,
        )
    assert scheduler.get_count == 0
    assert scheduler.post_count == 0


class _Response:
    def __init__(self, value: Any) -> None:
        self.raw = json.dumps(value).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return io.BytesIO(self.raw).read(limit)


@pytest.mark.parametrize("cpus", comparator.COMPANION_CPUS)
def test_bound_client_allows_at_most_one_post_per_companion(
    monkeypatch: pytest.MonkeyPatch, cpus: int
) -> None:
    task = _package()["companions"][str(cpus)]
    calls = 0

    def fake_urlopen(_request: Any, timeout: float) -> _Response:
        nonlocal calls
        assert timeout == 30.0
        calls += 1
        return _Response({**task, "id": 90000 + cpus, "status": "queued"})

    monkeypatch.setattr(comparator.urllib.request, "urlopen", fake_urlopen)
    client = comparator.ComparatorSchedulerClient("http://scheduler", expected=task)
    assert client.submit_task(task)["id"] == 90000 + cpus
    assert client.post_count == 1
    with pytest.raises(RuntimeError, match="already attempted"):
        client.submit_task(task)
    assert calls == 1


def test_mutated_capacity_or_launch_policy_is_rejected() -> None:
    package = _package()
    broken = copy.deepcopy(package)
    broken["capacity"]["exact_fit_slots"]["3"] += 1
    with pytest.raises(RuntimeError):
        comparator.validate_package(broken)
    broken = copy.deepcopy(package)
    broken["launch_policy"]["launch_allowed"] = True
    unsigned = {key: item for key, item in broken.items() if key != "package_sha256"}
    broken["package_sha256"] = comparator.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError):
        comparator.validate_package(broken)
