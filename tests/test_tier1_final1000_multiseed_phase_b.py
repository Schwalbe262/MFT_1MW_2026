from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import types
from typing import Any

import pytest

from tests.test_tier1_final1000_multiseed_phase_a import (
    FakePhaseAScheduler,
    PassingGateReader,
    _child_tasks,
    _consumer_capability,
    _monitoring_capability,
    _reach_prepared_cutover,
    _rendered_plan,
    _running_v1_with_scheduler,
)
from tools import tier1_corrected_current7_slurm_seed_runner as single_runner
from tools import tier1_corrected_generation_preflight as generation_preflight
from tools import tier1_final1000_multiseed_contract as phase_a
from tools import tier1_final1000_multiseed_harvest as harvest
from tools import tier1_final1000_multiseed_phase_b_contract as contract
from tools import tier1_final1000_multiseed_phase_b_runner as runner
from tools import tier1_final1000_multiseed_phase_b_evidence as evidence
from tools import tier1_final1000_multiseed_phase_b_smoke_release as smoke_release
from tools import tier1_final1000_multiseed_phase_b_successor as successor
from tools import tier1_final1000_multiseed_driver as phase_a_driver
from tools import tier1_final1000_slurm_launch as final1000_launch
from tools import tier1_final1000_stage_profiles as profiles


FAKE_SEED_RUNNER = r"""from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--bundle-root", type=Path, required=True)
parser.add_argument("--payload", type=Path, required=True)
parser.add_argument("--payload-root")
parser.add_argument("--payload-sha256", required=True)
parser.add_argument("--heartbeat-seconds")
args = parser.parse_args()
payload = json.loads(args.payload.read_text(encoding="utf-8"))
seed = int(payload["seed"])
task_id = os.environ["SLURM_SCHED_TASK_ID"]
print(f"phase-b seed {seed} stdout", flush=True)
output = args.bundle_root / "runs" / f"task-{task_id}" / f"seed-{seed}"
output.mkdir(parents=True, exist_ok=True)
(output / "runtime_env.json").write_text(
    json.dumps(
        {
            name: os.environ[name]
            for name in (
                "TMPDIR",
                "TMP",
                "TEMP",
                "JOBLIB_TEMP_FOLDER",
                "XDG_CACHE_HOME",
            )
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
for name in ("TMPDIR", "JOBLIB_TEMP_FOLDER", "XDG_CACHE_HOME"):
    (Path(os.environ[name]) / f"{seed}.scratch").write_text(name, encoding="ascii")
wall_started = time.monotonic()
cpu_started = time.process_time()
(output / "started.txt").write_text(str(time.time()), encoding="ascii")
sleep_by_seed = json.loads(os.environ.get("PHASE_B_TEST_SLEEP_BY_SEED", "{}"))
time.sleep(
    float(sleep_by_seed.get(str(seed), os.environ.get("PHASE_B_TEST_SLEEP", "0.1")))
)
failure_seed = os.environ.get("PHASE_B_TEST_FAILURE_SEED")
status = {
    "schema_version": "mft-tier1-current7-slurm-seed-status-v1",
    "task_id": task_id,
    "seed": seed,
    "payload_sha256": args.payload_sha256,
    "terminal": True,
    "phase": "terminal",
    "phase_b_test_cpuset": os.environ["MFT_FINAL1000_PHASE_B_CHILD_CPUSET"],
    "observed_peak_rss_bytes": int(
        os.environ.get("PHASE_B_TEST_OBSERVED_PEAK_RSS_BYTES", str(1024**3))
    ),
}
child_wall = max(0.0, time.monotonic() - wall_started)
child_cpu = max(0.0, time.process_time() - cpu_started)
child_capacity = child_wall * 4
if os.environ.get("PHASE_B_TEST_OMIT_CHILD_TELEMETRY") != "1":
    status["phase_b_child_resource_telemetry"] = {
        "schema_version": "mft-tier1-final1000-phase-b-child-cpu-telemetry-v1",
        "available": True,
        "measurement": "resource.getrusage(self+children)-delta",
        "child_cpus": 4,
        "wall_time_seconds": child_wall,
        "process_tree_cpu_seconds": child_cpu,
        "cpu_capacity_seconds": child_capacity,
        "cpu_utilization_fraction": child_cpu / child_capacity if child_capacity else 0.0,
    }
if os.environ.get("PHASE_B_TEST_OMIT_SEMLOCK_ATTESTATION") != "1":
    status["phase_b_semlock_stress_sha256"] = "e" * 64
exit_code = 0
if failure_seed is not None and seed == int(failure_seed):
    print("synthetic seed-local optimizer failure", file=sys.stderr, flush=True)
    status.update(
        state="failed",
        loaded_model_count=8,
        exit_code=17,
        failure="synthetic seed-local optimizer failure",
    )
    exit_code = 17
else:
    result = (json.dumps({"seed": seed}, sort_keys=True) + "\n").encode()
    (output / "result.json").write_bytes(result)
    status.update(
        state="completed",
        loaded_model_count=8,
        exit_code=0,
        result_sha256=hashlib.sha256(result).hexdigest(),
    )
(output / "legacy_seed_status.json").write_text(
    json.dumps(status, sort_keys=True), encoding="utf-8"
)
exit_override_seed = os.environ.get("PHASE_B_TEST_EXIT_OVERRIDE_SEED")
if exit_override_seed is not None and seed == int(exit_override_seed):
    exit_code = 23
(output / "finished.txt").write_text(str(time.time()), encoding="ascii")
raise SystemExit(exit_code)
"""


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    length: int = 4,
    task_id: int = 99400,
    internal_deadline_seconds: int = 82_800,
    cleanup_reserve_seconds: int = 1_800,
    minimum_child_start_budget_seconds: int = 7_200,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    task = contract.build_concurrent_batch_task(
        _child_tasks(length),
        internal_deadline_seconds=internal_deadline_seconds,
        cleanup_reserve_seconds=cleanup_reserve_seconds,
        minimum_child_start_budget_seconds=minimum_child_start_budget_seconds,
        model_load_stagger_seconds=5,
    )
    bundle = tmp_path / "bundle"
    code = bundle / "artifacts" / "code" / "tools"
    code.mkdir(parents=True)
    (code / "tier1_corrected_current7_slurm_seed_runner.py").write_text(
        FAKE_SEED_RUNNER, encoding="utf-8"
    )
    payload_root = tmp_path / "scheduler" / "runs"
    payload_path = payload_root / "request-1" / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(phase_a.json_bytes(task["payload_json"]))
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", str(task_id))
    return bundle, payload_root, payload_path, task


def _scaled_clock(scale: float):
    real_started = time.monotonic()
    return lambda: (time.monotonic() - real_started) * scale


class FakeFullInventoryShadowProvider:
    def __init__(self, total: int = 13_005):
        high_watermark = 42_000
        task_ids = list(range(high_watermark, high_watermark - total, -1))
        self.rows = [{"id": task_id} for task_id in task_ids]
        self.post_count = 0
        self.inventory_get_count = max(1, (total + 9_999) // 10_000) + 1
        unsigned = {
            "schema_version": "mft-tier1-final1000-inventory-snapshot-v1",
            "high_watermark_task_id": high_watermark,
            "before_id": high_watermark + 1,
            "filtered_total": total,
            "page_size": 10_000,
            "page_count": max(1, (total + 9_999) // 10_000),
            "task_ids_sha256": phase_a.canonical_sha256(task_ids),
            "server_snapshot_revision": "fixture-revision-1",
        }
        self.inventory_snapshot_receipt = {
            **unsigned,
            "sha256": phase_a.canonical_sha256(unsigned),
        }

    def list_namespace_tasks(self) -> list[dict[str, int]]:
        return copy.deepcopy(self.rows)


def _full_inventory_shadow(total: int = 13_005) -> dict[str, Any]:
    return successor.attest_full_inventory_shadow(
        FakeFullInventoryShadowProvider(total),
        helper_commit=successor.FULL_INVENTORY_REQUIREMENTS["required_helper_commit"],
        helper_file_sha256=successor.FULL_INVENTORY_REQUIREMENTS[
            "required_helper_sha256"
        ],
    )


def _run(
    bundle: Path,
    payload_root: Path,
    payload_path: Path,
    task: dict[str, Any],
    *,
    monotonic=time.monotonic,
) -> int:
    return runner.run(
        bundle,
        payload_path,
        payload_root,
        phase_a.canonical_sha256(task["payload_json"]),
        heartbeat_seconds=0.01,
        termination_grace_seconds=0.2,
        monotonic=monotonic,
        available_cpus=range(int(task["cpus"])),
    )


def _bounded_cgroup_snapshot(*, limit_bytes: int) -> dict[str, Any]:
    return {
        "cgroup_available": True,
        "cgroup_version": 2,
        "cgroup_path": "/slurm/job-test/step-test",
        "cgroup_measurement": "linux-cgroup-v2-memory-files",
        "cgroup_unavailable_reason": None,
        "cgroup_current_bytes": 2 * 1024**3,
        "cgroup_peak_bytes": 8 * 1024**3,
        "cgroup_limit_bytes": limit_bytes,
        "cgroup_limit_unbounded": False,
    }


def test_parent_memory_contract_is_sealed_and_unavailable_or_unbounded_fails_closed():
    children = [{"ordinal": 0, "seed": 101}, {"ordinal": 1, "seed": 102}]
    finished = {
        0: {"legacy": {"observed_peak_rss_bytes": 3 * 1024**3}},
        1: {"legacy": {"observed_peak_rss_bytes": 4 * 1024**3}},
    }
    parent_request = 2 * contract.CHILD_MEMORY_MB * 1024**2
    bounded = runner._build_parent_memory_evidence(
        task_id="99001",
        manifest_sha256="a" * 64,
        terminal_status={"status_sha256": "b" * 64},
        children=children,
        finished=finished,
        cgroup_memory=_bounded_cgroup_snapshot(limit_bytes=parent_request),
    )
    assert contract.validate_parent_memory_evidence(bounded) == bounded
    assert bounded["child_requested_memory_bytes"] == 28 * 1024**3
    assert bounded["parent_requested_memory_bytes"] == parent_request
    assert bounded["child_peak_rss_available_count"] == 2
    assert bounded["child_peak_rss_sum_bytes"] == 7 * 1024**3
    assert bounded["child_peak_rss_max_bytes"] == 4 * 1024**3
    assert bounded["total_child_peak_rss_within_parent_request"] is True
    assert bounded["cgroup_peak_within_limit"] is True
    assert bounded["cgroup_limit_covers_parent_request"] is True
    assert bounded["safety_passed"] is True

    bad_gate = {
        key: value for key, value in bounded.items() if key != "evidence_sha256"
    }
    bad_gate["safety_passed"] = False
    with pytest.raises(RuntimeError, match="parent memory evidence"):
        contract.seal_parent_memory_evidence(bad_gate)

    unbounded_snapshot = _bounded_cgroup_snapshot(limit_bytes=parent_request)
    unbounded_snapshot.update(
        cgroup_limit_bytes=None,
        cgroup_limit_unbounded=True,
    )
    unbounded = runner._build_parent_memory_evidence(
        task_id="99001",
        manifest_sha256="a" * 64,
        terminal_status={"status_sha256": "b" * 64},
        children=children,
        finished=finished,
        cgroup_memory=unbounded_snapshot,
    )
    assert contract.validate_parent_memory_evidence(unbounded) == unbounded
    assert unbounded["cgroup_available"] is True
    assert unbounded["cgroup_limit_unbounded"] is True
    assert unbounded["cgroup_peak_within_limit"] is False
    assert unbounded["cgroup_limit_covers_parent_request"] is False
    assert unbounded["safety_passed"] is False

    unavailable = runner._build_parent_memory_evidence(
        task_id="99001",
        manifest_sha256="a" * 64,
        terminal_status={"status_sha256": "b" * 64},
        children=children,
        finished=finished,
        cgroup_memory=runner._unavailable_cgroup_memory("non-linux-platform"),
    )
    assert contract.validate_parent_memory_evidence(unavailable) == unavailable
    assert unavailable["cgroup_available"] is False
    assert unavailable["safety_passed"] is False

    incomplete_children = runner._build_parent_memory_evidence(
        task_id="99001",
        manifest_sha256="a" * 64,
        terminal_status={"status_sha256": "b" * 64},
        children=children,
        finished={0: finished[0]},
        cgroup_memory=_bounded_cgroup_snapshot(limit_bytes=parent_request),
    )
    assert incomplete_children["child_peak_rss_available_count"] == 1
    assert incomplete_children[
        "total_child_peak_rss_within_parent_request"
    ] is False
    assert incomplete_children["safety_passed"] is False


def test_linux_cgroup_v2_and_v1_memory_files_are_collected_safely(tmp_path: Path):
    proc = tmp_path / "self.cgroup"
    cgroup_root = tmp_path / "cgroup"
    v2 = cgroup_root / "slurm" / "job-1"
    v2.mkdir(parents=True)
    proc.write_text("0::/slurm/job-1\n", encoding="ascii")
    (v2 / "memory.current").write_text("100\n", encoding="ascii")
    (v2 / "memory.peak").write_text("250\n", encoding="ascii")
    (v2 / "memory.max").write_text("500\n", encoding="ascii")
    assert runner._collect_parent_cgroup_memory(
        platform_name="linux",
        proc_self_cgroup=proc,
        cgroup_root=cgroup_root,
    ) == {
        "cgroup_available": True,
        "cgroup_version": 2,
        "cgroup_path": "/slurm/job-1",
        "cgroup_measurement": "linux-cgroup-v2-memory-files",
        "cgroup_unavailable_reason": None,
        "cgroup_current_bytes": 100,
        "cgroup_peak_bytes": 250,
        "cgroup_limit_bytes": 500,
        "cgroup_limit_unbounded": False,
    }
    (v2 / "memory.max").write_text("max\n", encoding="ascii")
    unbounded = runner._collect_parent_cgroup_memory(
        platform_name="linux",
        proc_self_cgroup=proc,
        cgroup_root=cgroup_root,
    )
    assert unbounded["cgroup_available"] is True
    assert unbounded["cgroup_limit_bytes"] is None
    assert unbounded["cgroup_limit_unbounded"] is True

    v1 = cgroup_root / "memory" / "slurm" / "job-2"
    v1.mkdir(parents=True)
    proc.write_text("7:cpu:/slurm/job-2\n9:memory:/slurm/job-2\n", encoding="ascii")
    (v1 / "memory.usage_in_bytes").write_text("125\n", encoding="ascii")
    (v1 / "memory.max_usage_in_bytes").write_text("275\n", encoding="ascii")
    (v1 / "memory.limit_in_bytes").write_text("550\n", encoding="ascii")
    v1_evidence = runner._collect_parent_cgroup_memory(
        platform_name="linux",
        proc_self_cgroup=proc,
        cgroup_root=cgroup_root,
    )
    assert v1_evidence["cgroup_available"] is True
    assert v1_evidence["cgroup_version"] == 1
    assert v1_evidence["cgroup_path"] == "/slurm/job-2"
    assert v1_evidence["cgroup_current_bytes"] == 125
    assert v1_evidence["cgroup_peak_bytes"] == 275
    assert v1_evidence["cgroup_limit_bytes"] == 550

    proc.write_text("0::/../../escape\n", encoding="ascii")
    escaped = runner._collect_parent_cgroup_memory(
        platform_name="linux",
        proc_self_cgroup=proc,
        cgroup_root=cgroup_root,
    )
    assert escaped["cgroup_available"] is False
    assert escaped["cgroup_path"] is None


def test_exact_variable_parent_envelopes_and_deterministic_seals():
    expected = {lanes: (lanes * 4, lanes * 28_672) for lanes in range(1, 9)}
    for lanes, envelope in expected.items():
        first = contract.build_concurrent_batch_task(_child_tasks(lanes))
        second = contract.build_concurrent_batch_task(_child_tasks(lanes))
        assert first == second
        assert (first["cpus"], first["memory_mb"]) == envelope
        assert first["payload_json"]["parent_resource_policy"]["cpus"] == envelope[0]
        assert (
            first["payload_json"]["parent_resource_policy"]["memory_mb"] == envelope[1]
        )
        assert {
            child["task"]["cpus"] for child in first["payload_json"]["children"]
        } == {4}
        assert {
            child["task"]["memory_mb"] for child in first["payload_json"]["children"]
        } == {28_672}
        assert first["payload_json"]["model_context_reuse"] is False
        assert first["payload_json"]["rng_context_reuse"] is False
        assert first["payload_json"]["model_load_stagger_seconds"] == 5

    # Required packing shapes consume common 44/48/60-CPU allocations exactly.
    assert 32 + contract.build_concurrent_batch_task(_child_tasks(3))["cpus"] == 44
    assert 32 + contract.build_concurrent_batch_task(_child_tasks(4))["cpus"] == 48
    assert 32 + contract.build_concurrent_batch_task(_child_tasks(7))["cpus"] == 60

    tampered = contract.build_concurrent_batch_task(_child_tasks(8))
    tampered["memory_mb"] -= 1
    with pytest.raises(RuntimeError, match="Scheduler task seal mismatch"):
        contract.validate_batch_task(tampered)


def test_remote_smoke_release_renders_authenticated_1_4_8_lane_tasks():
    plan = _rendered_plan()
    required_hashes = {
        contract.INFERENCE_SAFETY["required_helper_path"]: contract.INFERENCE_SAFETY[
            "required_helper_sha256"
        ]
    }
    records = {
        f"artifacts/code/{relative}": {
            "sha256": required_hashes.get(relative, f"{index + 1:064x}"),
            "size": index + 10,
        }
        for index, relative in enumerate(sorted(contract.REMOTE_CODE_FILES))
    }
    required_unsigned = {
        "schema_version": "mft-tier1-required-runtime-code-inventory-v1",
        "required_files": sorted(contract.REMOTE_CODE_FILES),
        "code_records": records,
        "code_records_sha256": phase_a.canonical_sha256(records),
        "required_sha256": required_hashes,
    }
    stable_manifest = {
        "schema_version": "mft-tier1-current7-slurm-bundle-v1",
        "files": copy.deepcopy(records),
        "code_inventory": copy.deepcopy(records),
        "code_inventory_sha256": phase_a.canonical_sha256(records),
        "required_runtime_code": {
            **required_unsigned,
            "sha256": phase_a.canonical_sha256(required_unsigned),
        },
        "relocation": {"contract_sha256": "d" * 64},
        "runtime": {"critical_packages": {"numpy": "test"}},
    }
    contract_sha = phase_a.canonical_sha256(stable_manifest)
    manifest = {
        **stable_manifest,
        "bundle_id": f"current7-{contract_sha[:20]}",
        "contract_sha256": contract_sha,
    }
    selected_stage = profiles.STAGES[0]
    for wave in ("canaries", "ramp"):
        for task in plan["task_waves"][wave]:
            payload = task["payload_json"]
            if payload["final_goal_stage_id"] != selected_stage.stage_id:
                continue
            payload["bundle_id"] = manifest["bundle_id"]
            payload_sha = phase_a.canonical_sha256(payload)
            task["command"] = final1000_launch._replace_payload_sha(
                task["command"], payload_sha
            )
            task["remote_cwd"] = f"/remote/{manifest['bundle_id']}"
            resources = {
                key: task[key]
                for key in (
                    "cpus",
                    "memory_mb",
                    "scheduling_profile",
                    "aedt_backend",
                    "gpus",
                    "priority",
                    "timeout_seconds",
                    "max_workers_per_node",
                )
            }
            task["dedupe_key"] = "mft-tier1-final1000:" + phase_a.canonical_sha256(
                {
                    "goal": "final1000",
                    "stage_profile_sha256": profiles.stage_profile(selected_stage)[
                        "sha256"
                    ],
                    "bundle_id": manifest["bundle_id"],
                    "payload": payload,
                    "resources": resources,
                }
            )
            final1000_launch.validate_task(task)
    binding = plan["stage_bindings"][selected_stage.stage_id]
    binding["bundle_id"] = manifest["bundle_id"]
    binding["remote_bundle"] = f"/remote/{manifest['bundle_id']}"
    binding["ready"]["bundle_id"] = manifest["bundle_id"]
    binding["ready_sha256"] = phase_a.canonical_sha256(binding["ready"])
    plan_unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = phase_a.canonical_sha256(plan_unsigned)
    plan = final1000_launch.validate_launch_plan(plan)
    ready = {
        "schema_version": "mft-tier1-current7-slurm-ready-v1",
        "bundle_id": manifest["bundle_id"],
        "contract_sha256": contract_sha,
        "bundle_manifest_sha256": "a" * 64,
        "file_count": len(records),
        "byte_count": sum(record["size"] for record in records.values()),
        "every_file_sha256_verified": True,
        "all_file_sha256_verified": True,
        "runtime_verified": True,
        "runtime_packages": {"numpy": "test"},
        "code_inventory_sha256": manifest["code_inventory_sha256"],
        "relocation_contract_sha256": "d" * 64,
        "remote_git_checkout_performed": False,
        "artifacts_read_only": True,
        "runs_mode": "1777",
        "published_at": "2026-07-22T00:00:00+09:00",
    }

    release = smoke_release.build_smoke_release(
        plan,
        manifest,
        ready,
        manifest_file_sha256="a" * 64,
    )

    assert release["lane_shapes"] == [1, 4, 8]
    assert [task["cpus"] for task in release["tasks"]] == [4, 16, 32]
    assert [task["memory_mb"] for task in release["tasks"]] == [
        28_672,
        114_688,
        229_376,
    ]
    children = [
        child
        for task in release["tasks"]
        for child in task["payload_json"]["children"]
    ]
    assert len(children) == 13
    assert len({child["seed"] for child in children}) == 13
    assert len({child["logical_dedupe_key"] for child in children}) == 13
    assert release["four_lane_is_mandatory_first_production_canary"] is True
    assert release["smoke_ready"] is True
    assert release["scheduler_write_performed"] is False
    assert release["submission_performed"] is False

    tampered = copy.deepcopy(manifest)
    tampered["code_inventory"].pop(next(iter(records)))
    with pytest.raises(RuntimeError, match="content-address|code inventory"):
        smoke_release.build_smoke_release(
            plan,
            tampered,
            ready,
            manifest_file_sha256="a" * 64,
        )
    for missing_ready_field in (
        "contract_sha256",
        "file_count",
        "byte_count",
        "all_file_sha256_verified",
        "artifacts_read_only",
        "runs_mode",
        "runtime_packages",
    ):
        incomplete_ready = copy.deepcopy(ready)
        incomplete_ready.pop(missing_ready_field)
        with pytest.raises(RuntimeError, match="READY seal mismatch"):
            smoke_release.build_smoke_release(
                plan,
                manifest,
                incomplete_ready,
                manifest_file_sha256="a" * 64,
            )


def test_dry_run_repack_covers_exact_500_without_mutation():
    first = evidence.build_dry_run_evidence(
        _rendered_plan(), source_file_sha256="a" * 64
    )
    second = evidence.build_dry_run_evidence(
        _rendered_plan(), source_file_sha256="a" * 64
    )
    assert first == second
    assert first["logical_seed_count"] == 500
    assert first["running_logical_count"] == 447
    assert first["queued_logical_count"] == 53
    assert first["physical_parent_count"] == 108
    assert first["running_parent_count"] == 100
    assert first["queued_parent_count"] == 8
    assert first["shape_parent_counts"] == {
        "1": 44,
        "4": 14,
        "8": 50,
    }
    assert first["scheduler_task_envelope_reduction"] == 392
    assert first["account_maxjobs_allocation_slot_reduction"] == 0
    assert first["live_allocation_logical_capacity"] == 447
    assert first["exact_500_running_capacity_gap"] == 53
    assert first["packing_increases_allocation_pool_capacity"] is False
    assert first["packing_releases_account_maxjobs_allocation_slots"] is False
    assert first["packing_effect_scope"] == (
        "task-attach-control-plane-and-owned-lane-use-v1"
    )
    assert first["stage_logical_quotas"] == {
        profiles.STAGES[0].stage_id: 300,
        profiles.STAGES[1].stage_id: 150,
        profiles.STAGES[2].stage_id: 40,
        profiles.STAGES[3].stage_id: 10,
    }
    assert first["queued_stage_shapes"][profiles.STAGES[0].stage_id] == {
        "1": 1,
        "4": 1,
        "8": 6,
    }
    assert first["packing_shapes"] == [8, 4, 1]
    assert first["unauthenticated_production_tail_shapes"] == [2, 3, 5, 6, 7]
    assert first["pure_eight_only_allowed"] is False
    assert first["global_intent_order"] == "lane-count-descending-v1"
    assert first["unavailable_shape_intent_action"] == (
        "skip-and-continue-lower-shapes-with-fresh-capacity-get-v1"
    )
    assert first["head_of_line_blocking_allowed"] is False
    assert first["live_inventory_driven"] is True
    assert first["capacity_loss_lanes"] == 0
    assert first["actual_capacity_aware_submitter_implementation_present"] is False
    assert first["authenticated_remote_smoke_shapes"] == []
    assert first["shape_capacity_recheck"]["endpoint_path"] == "/api/task-capacity"
    assert [
        item["lane_count"] for item in first["shape_capacity_recheck"]["queries"]
    ] == [8, 4, 1]
    assert all(
        item["partition"] == "auto"
        for item in first["shape_capacity_recheck"]["queries"]
    )
    assert (
        first["shape_capacity_recheck"][
            "scheduler_response_snapshot_revision_verified"
        ]
        is False
    )
    assert (
        first["shape_capacity_recheck"]["atomic_capacity_reservation_supported"]
        is False
    )
    assert first["lane_count_independent_science_identity"] is True
    assert first["capacity_snapshot_recheck_required"] is True
    assert first["static_eight_first_pack_production_eligible"] is False
    assert first["dispatch_admission_policy"] == {
        "schema_version": "mft-tier1-final1000-phase-b-dispatch-admission-v1",
        "logical_child_cpus": 4,
        "logical_child_memory_mb": 28_672,
        "production_packing_shapes": [8, 4, 1],
        "packing_order": "global-lane-count-descending-v1",
        "unavailable_shape_intent_action": (
            "skip-and-continue-lower-shapes-with-fresh-capacity-get-v1"
        ),
        "head_of_line_blocking_allowed": False,
        "pure_eight_only_allowed": False,
        "unauthenticated_tail_shapes": [2, 3, 5, 6, 7],
        "default_model_load_stagger_seconds": 5.0,
        "eight_lane_dispatch_fill_seconds": 35.0,
        "scheduler_ready_lane_reserve": 0,
        "scheduler_ready_lane_policy": "natural-terminal-vacancy-only-v1",
        "speculative_ready_lane_submission_allowed": False,
        "empty_node_or_capacity_prediction_allowed": False,
        "capacity_snapshot_recheck_before_activation_required": True,
        "shape_capacity_endpoint_path": "/api/task-capacity",
        "shape_capacity_recheck_before_each_post_required": True,
        "shape_capacity_snapshot_fence_required": True,
        "capacity_snapshot_drift_action": "rerender-or-fail-closed",
        "persistent_pool_allocations_shared_by_scheduler_tasks": True,
        "packing_effect_scope": "task-attach-control-plane-and-owned-lane-use-v1",
        "allocation_pool_capacity_increase_claimed": False,
        "account_maxjobs_allocation_slot_reduction_claimed": False,
        "exact_500_running_requires": (
            "live-allocation-pool-growth-or-rightsize-or-authenticated-resource-change"
        ),
    }
    assert first["scheduler_write_performed"] is False
    assert first["submission_performed"] is False
    assert first["remote_write_performed"] is False
    assert first["fea_submission_performed"] is False
    assert first["aedt_used"] is False

    published = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "evidence"
            / "tier1_final1000_multiseed_phase_b_dry_run_20260722.json"
        ).read_text(encoding="utf-8")
    )
    published_unsigned = {
        key: value for key, value in published.items() if key != "evidence_sha256"
    }
    assert published["evidence_sha256"] == phase_a.canonical_sha256(published_unsigned)
    # This immutable file is the archived active39 snapshot.  Its seal remains
    # valid, but it must not be rebuilt as though it represented live capacity.
    assert published["running_logical_count"] == 431
    assert published["queued_logical_count"] == 69
    integration = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "evidence"
            / "tier1_final1000_multiseed_phase_b_integration_20260722.json"
        ).read_text(encoding="utf-8")
    )
    integration_unsigned = {
        key: value for key, value in integration.items() if key != "evidence_sha256"
    }
    assert integration["evidence_sha256"] == phase_a.canonical_sha256(
        integration_unsigned
    )
    assert integration["repeated_predict_stress"]["predict_call_count"] == 1280

    prerelease = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "evidence"
            / "tier1_final1000_multiseed_phase_b_smoke_prerelease_20260723.json"
        ).read_text(encoding="utf-8")
    )
    prerelease_unsigned = {
        key: value for key, value in prerelease.items() if key != "evidence_sha256"
    }
    assert prerelease["evidence_sha256"] == phase_a.canonical_sha256(
        prerelease_unsigned
    )
    assert prerelease["assessment"] == "blocked-fail-closed"
    assert prerelease["production_eligible"] is False
    assert prerelease["smoke_release_emitted"] is False
    assert prerelease["smoke_task_shapes_emitted"] == []
    assert prerelease["ready_marker_present"] is False
    assert prerelease["successor_plan"]["launch_eligible"] is False
    assert prerelease["successor_plan"]["bundle_bindings_complete"] is False
    assert prerelease["successor_plan"]["production_logical_seed_count"] == 500
    assert len(prerelease["bundle_assessments"]) == 8
    assert {
        item["stage_id"] for item in prerelease["bundle_assessments"]
    } == {stage.stage_id for stage in profiles.STAGES}
    assert {
        item["kind"] for item in prerelease["bundle_assessments"]
    } == {"topology64-base", "delta-over-safe"}
    assert all(
        item["required_runtime_code_present"] is False
        for item in prerelease["bundle_assessments"]
    )
    assert prerelease["required_runtime_code_contract"]["required_file_count"] == 19
    assert (
        prerelease["required_runtime_code_contract"][
            "assessed_manifest_required_file_count_present"
        ]
        == 9
    )
    assert len(
        prerelease["required_runtime_code_contract"]["missing_required_files"]
    ) == 10
    assert prerelease["scheduler_write_performed"] is False
    assert prerelease["submission_performed"] is False
    assert prerelease["remote_write_performed"] is False


def test_cpu_efficiency_benchmark_is_finite_paired_and_non_mutating():
    design = successor.build_cpu_efficiency_benchmark_design(
        bundle_manifest_sha256="a" * 64,
        model_inventory_sha256="b" * 64,
        node_identity="same-node-benchmark-fixture",
        seed_start=90_000,
    )
    assert design["total_logical_run_count"] == 58
    assert design["single_child_cpu_scaling"]["cpu_counts"] == [4, 6, 8]
    assert design["single_child_cpu_scaling"]["logical_run_count"] == 18
    assert design["concurrent_density"]["logical_run_count"] == 40
    assert design["concurrent_density"]["configurations"][0]["waves_for_ten_seeds"] == [
        8,
        2,
    ]
    assert (
        design["concurrent_density"]["configurations"][1]["oversubscription_ratio"]
        == 1.25
    )
    assert design["production_child_cpus_before_benchmark"] == 4
    assert design["production_child_cpus_after_design"] == 4
    assert design["automatic_resource_policy_change_allowed"] is False
    assert design["scheduler_write_performed"] is False
    assert design["submission_performed"] is False
    unsigned = {
        key: value for key, value in design.items() if key != "benchmark_design_sha256"
    }
    assert design["benchmark_design_sha256"] == phase_a.canonical_sha256(unsigned)


def test_capacity_aware_placement_exact_shapes_stage_boundaries_and_drift():
    plan = _rendered_plan()
    inventory = successor.current_empty_pool_inventory()
    next_seeds = {
        stage.stage_id: stage.seed_start + 10_000 for stage in profiles.STAGES
    }
    rendered = successor.build_placement_plan(
        plan,
        next_seed_by_stage=next_seeds,
        allocation_inventory=inventory,
    )
    tasks = rendered.pop("parent_tasks")
    assert successor.validate_placement_plan(rendered, parent_tasks=tasks) == rendered
    assert len(tasks) == 108
    assert rendered["logical_seed_count"] == 500
    assert rendered["running_logical_count"] == 447
    assert rendered["queued_logical_count"] == 53
    assert rendered["live_allocation_logical_capacity"] == 447
    assert rendered["exact_500_running_capacity_gap"] == 53
    assert rendered["packing_increases_allocation_pool_capacity"] is False
    assert rendered["packing_releases_account_maxjobs_allocation_slots"] is False
    assert (
        "live allocation pool capacity is below exact-500 running target"
        in rendered["production_blockers"]
    )
    assert rendered["shape_parent_counts"] == {
        "1": 44,
        "4": 14,
        "8": 50,
    }
    assert rendered["packing_shapes"] == [8, 4, 1]
    assert rendered["unauthenticated_production_tail_shapes"] == [2, 3, 5, 6, 7]
    assert rendered["pure_eight_only_allowed"] is False
    assert rendered["capacity_loss_lanes"] == 0
    assert rendered["actual_capacity_aware_submitter_implementation_present"] is False
    assert rendered["authenticated_remote_smoke_shapes"] == []
    assert [summary["lane_count"] for summary in rendered["parent_summaries"]] == (
        sorted(
            (summary["lane_count"] for summary in rendered["parent_summaries"]),
            reverse=True,
        )
    )
    assert {
        summary["lane_count"]
        for summary in rendered["parent_summaries"]
        if summary["scheduler_disposition"] == "running-capacity"
    } == {8, 4, 1}
    assert rendered["running_stage_shapes"] == {
        profiles.STAGES[0].stage_id: {"1": 39, "4": 12, "8": 20},
        profiles.STAGES[1].stage_id: {"1": 2, "4": 1, "8": 18},
        profiles.STAGES[2].stage_id: {"8": 5},
        profiles.STAGES[3].stage_id: {"1": 2, "8": 1},
    }
    assert rendered["queued_stage_shapes"] == {
        profiles.STAGES[0].stage_id: {"1": 1, "4": 1, "8": 6},
        profiles.STAGES[1].stage_id: {},
        profiles.STAGES[2].stage_id: {},
        profiles.STAGES[3].stage_id: {},
    }
    assert (
        rendered["logical_child_inventory_sha256"]
        == rendered["one_lane_repack_logical_child_inventory_sha256"]
    )
    assert all(
        len(
            {
                child["task"]["payload_json"]["final_goal_stage_id"]
                for child in task["payload_json"]["children"]
            }
        )
        == 1
        for task in tasks
    )
    assert all(
        task["cpus"] == task["payload_json"]["batch_length"] * 4
        and task["memory_mb"] == task["payload_json"]["batch_length"] * 28_672
        for task in tasks
    )
    successor.require_inventory_identity(rendered, inventory)

    def reseal_placement(value: dict[str, Any]) -> None:
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "placement_plan_sha256"
        }
        value["placement_plan_sha256"] = phase_a.canonical_sha256(unsigned)

    forged_allocation = copy.deepcopy(rendered)
    running_summary = next(
        summary
        for summary in forged_allocation["parent_summaries"]
        if summary["scheduler_disposition"] == "running-capacity"
    )
    running_summary["allocation_id"] = "forged-allocation"
    reseal_placement(forged_allocation)
    with pytest.raises(RuntimeError, match="allocation binding drifted"):
        successor.validate_placement_plan(forged_allocation, parent_tasks=tasks)

    swapped_disposition = copy.deepcopy(rendered)
    same_shape_running = next(
        summary
        for summary in swapped_disposition["parent_summaries"]
        if summary["scheduler_disposition"] == "running-capacity"
        and summary["stage_id"] == profiles.STAGES[0].stage_id
        and summary["lane_count"] == 8
    )
    same_shape_queued = next(
        summary
        for summary in swapped_disposition["parent_summaries"]
        if summary["scheduler_disposition"] == "queued-capacity"
        and summary["stage_id"] == profiles.STAGES[0].stage_id
        and summary["lane_count"] == 8
    )
    same_shape_running["scheduler_disposition"] = "queued-capacity"
    same_shape_queued["scheduler_disposition"] = "running-capacity"
    reseal_placement(swapped_disposition)
    with pytest.raises(RuntimeError, match="allocation binding drifted"):
        successor.validate_placement_plan(swapped_disposition, parent_tasks=tasks)

    forged_task_binding = copy.deepcopy(rendered)
    forged_task_binding["parent_summaries"][0]["seed_start"] += 1
    reseal_placement(forged_task_binding)
    with pytest.raises(RuntimeError, match="task inventory drifted"):
        successor.validate_placement_plan(forged_task_binding, parent_tasks=tasks)

    drifted = successor.current_empty_pool_inventory(
        observed_at="2026-07-22T00:00:01+09:00"
    )
    with pytest.raises(RuntimeError, match="inventory drifted"):
        successor.require_inventory_identity(rendered, drifted)


def test_live_capacity_decomposition_is_exact_largest_first_8_4_1():
    allowed = (2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16)
    for capacity in allowed:
        inventory = successor.build_allocation_inventory(
            [
                {
                    "allocation_id": f"capacity-{capacity}",
                    "usable_lane_capacity": capacity,
                    "state": "active",
                }
            ],
            observed_at="2026-07-23T01:20:00+09:00",
        )
        slots = successor.decompose_allocation_inventory(inventory)
        shapes = [slot["lane_count"] for slot in slots]
        assert shapes == sorted(shapes, reverse=True)
        assert set(shapes) <= {8, 4, 1}
        assert sum(shapes) == capacity

    latest = successor.current_empty_pool_inventory()
    latest_slots = successor.decompose_allocation_inventory(latest)
    latest_shapes = Counter(slot["lane_count"] for slot in latest_slots)
    assert latest["allocation_count"] == 40
    assert latest["usable_lane_count"] == 447
    assert latest_shapes == Counter({8: 44, 4: 13, 1: 43})

    with pytest.raises(RuntimeError, match="inventory row is invalid"):
        successor.build_allocation_inventory(
            [
                {
                    "allocation_id": "draining-must-not-count",
                    "usable_lane_capacity": 16,
                    "state": "draining",
                }
            ],
            observed_at="2026-07-23T01:20:00+09:00",
        )


def test_phase_a_batch4_migration_waits_without_intents(tmp_path: Path):
    plan = _rendered_plan()
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    reader = PassingGateReader(scheduler)
    initial = phase_a_driver.initial_driver_state(predecessor, plan, plan, capability)
    batch1, _ = phase_a_driver.cycle(
        initial, plan, capability, scheduler, reader, apply=True
    )
    for dedupe in batch1["gate_lanes"]["batch1"]:
        scheduler.rows[dedupe]["status"] = "completed"
    batch4, _ = phase_a_driver.cycle(
        batch1, plan, capability, scheduler, reader, apply=True
    )
    assert batch4["phase"] == "batch4"
    migration = successor.prepare_migration_state(batch4, plan, capability)
    assert migration["phase"] == "awaiting_phase_a_refill"
    assert migration["source_driver_phase"] == "batch4"
    assert migration["phase_a_adapter"] is None
    assert migration["placement_plan"] is None
    assert migration["authority_released"] is False
    assert migration["mutation_authority"] == {
        "protocol": "single-owner-phase-a-phase-b-handoff-v1",
        "active_owner_kind": "phase-a-supervisor",
        "active_owner_identity": "pid:55304",
        "active_owner_count": 1,
        "simultaneous_owners_allowed": False,
        "phase_a_authority_relinquished": False,
        "phase_b_authority_granted": False,
        "phase_b_authority_lease_sha256": None,
        "phase_b_scheduler_mutation_enabled": False,
    }
    assert migration["scheduler_submission_intents"] == []
    assert scheduler.post_count == 8


def test_refill_handoff_requires_stable_exit_consumer_capability_and_inventory(
    tmp_path: Path,
):
    plan, capability, scheduler, reader, _awaiting, prepared = _reach_prepared_cutover(
        tmp_path
    )
    with _consumer_capability(tmp_path, plan, prepared["controller_state"]) as (
        capability_path,
        _receipt,
    ):
        refill, intents = phase_a_driver.cycle(
            prepared,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=capability_path,
        )
    assert refill["phase"] == "refill"
    inventory = successor.current_empty_pool_inventory()
    migration = successor.prepare_migration_state(
        refill,
        plan,
        capability,
        allocation_inventory=inventory,
    )
    assert migration["phase"] == "awaiting_phase_a_supervisor_stop"
    assert migration["phase_a_adapter"]["cancellation_performed"] is False
    assert migration["placement_plan"]["logical_seed_count"] == 500
    assert migration["scheduler_submission_intents"] == []
    assert all(task["payload_json"]["batch_length"] == 4 for task in intents)

    stop = phase_a_driver.write_supervisor_stop(
        tmp_path / "STOP-PHASE-A-SUPERVISOR.json",
        "reviewed Phase-B authority handoff",
    )
    exit_receipt = successor.build_phase_a_exit_receipt(
        first_stable_driver_read=refill,
        second_stable_driver_read=copy.deepcopy(refill),
        plan=plan,
        monitoring_capability=capability,
        supervisor_stop_receipt=stop,
        supervisor_pid=successor.EXPECTED_PHASE_A_SUPERVISOR_PID,
        process_exited=True,
    )
    inventory_shadow = _full_inventory_shadow()
    shadow_only_consumer = successor.build_phase_b_consumer_capability(
        implementation_revision="dry-shadow",
        test_evidence_sha256="a" * 64,
        shadow_validation_sha256=inventory_shadow["attestation_sha256"],
        full_inventory_shadow_attestation=inventory_shadow,
        activation_authorized=False,
    )
    with pytest.raises(RuntimeError, match="consumer capability"):
        successor.release_migration_authority(
            migration,
            supervisor_stop_receipt=stop,
            phase_a_exit_receipt=exit_receipt,
            phase_b_consumer_capability=shadow_only_consumer,
            current_allocation_inventory=inventory,
        )
    authorized_consumer = successor.build_phase_b_consumer_capability(
        implementation_revision="reviewed-release",
        test_evidence_sha256="c" * 64,
        shadow_validation_sha256=inventory_shadow["attestation_sha256"],
        full_inventory_shadow_attestation=inventory_shadow,
        activation_authorized=True,
    )
    with pytest.raises(RuntimeError, match="inventory drifted"):
        successor.release_migration_authority(
            migration,
            supervisor_stop_receipt=stop,
            phase_a_exit_receipt=exit_receipt,
            phase_b_consumer_capability=authorized_consumer,
            current_allocation_inventory=successor.current_empty_pool_inventory(
                observed_at="2026-07-22T00:00:01+09:00"
            ),
        )
    released = successor.release_migration_authority(
        migration,
        supervisor_stop_receipt=stop,
        phase_a_exit_receipt=exit_receipt,
        phase_b_consumer_capability=authorized_consumer,
        current_allocation_inventory=inventory,
    )
    assert released["phase"] == "ready_for_authorized_activation"
    assert released["authority_released"] is True
    assert released["mutation_authority"]["active_owner_count"] == 1
    assert (
        released["mutation_authority"]["active_owner_kind"]
        == "phase-b-successor-controller"
    )
    assert released["mutation_authority"]["phase_a_authority_relinquished"] is True
    assert released["mutation_authority"]["simultaneous_owners_allowed"] is False
    assert released["mutation_authority"]["phase_b_scheduler_mutation_enabled"] is False
    assert released["scheduler_submission_intents"] == []
    assert released["scheduler_write_performed"] is False
    assert released["cancellation_performed"] is False
    conflicting = copy.deepcopy(released)
    conflicting["mutation_authority"]["active_owner_count"] = 2
    conflicting_unsigned = {
        key: value for key, value in conflicting.items() if key != "state_sha256"
    }
    conflicting["state_sha256"] = phase_a.canonical_sha256(conflicting_unsigned)
    with pytest.raises(RuntimeError, match="single mutation authority"):
        successor.validate_migration_state(conflicting)


def test_consumer_capability_refuses_single_10000_inventory_or_short_shadow():
    with pytest.raises(RuntimeError, match="inventory snapshot"):
        short_shadow = _full_inventory_shadow(10_000)
        successor.build_phase_b_consumer_capability(
            implementation_revision="short-inventory",
            test_evidence_sha256="a" * 64,
            shadow_validation_sha256=short_shadow["attestation_sha256"],
            full_inventory_shadow_attestation=short_shadow,
            activation_authorized=True,
        )
    inventory_shadow = _full_inventory_shadow()
    capability = successor.build_phase_b_consumer_capability(
        implementation_revision="paged-13k-shadow",
        test_evidence_sha256="d" * 64,
        shadow_validation_sha256=inventory_shadow["attestation_sha256"],
        full_inventory_shadow_attestation=inventory_shadow,
        activation_authorized=True,
    )
    assert capability["full_inventory_requirements"]["cursor"] == "before_id"
    assert (
        capability["full_inventory_requirements"]["required_helper_commit"]
        == "dafb4509d52f4abcacb68c7bacfc850468d7dc1f"
    )
    assert capability["single_limit_10000_inventory_used"] is False
    assert capability["full_inventory_shadow_attestation"]["row_count"] == 13_005
    assert (
        capability["full_inventory_shadow_attestation"]["scheduler_post_count_after"]
        == 0
    )
    assert successor.validate_phase_b_consumer_capability(capability) == capability


def test_real_four_child_concurrency_isolated_journals_and_prefix_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(tmp_path, monkeypatch)
    parent_memory_bytes = int(task["memory_mb"]) * 1024**2
    monkeypatch.setattr(
        runner,
        "_collect_parent_cgroup_memory",
        lambda: _bounded_cgroup_snapshot(limit_bytes=parent_memory_bytes),
    )
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "0.35")
    shared_sentinel = tmp_path / "shared-tmp-sentinel"
    shared_sentinel.write_text("must survive", encoding="ascii")
    real_started = time.monotonic()
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(200),
        )
        == 0
    )
    wall = time.monotonic() - real_started
    # Sequential execution would require at least 1.4 seconds.
    assert wall < 0.9
    run_root = bundle / "runs" / "task-99400"
    seeds = [child["seed"] for child in task["payload_json"]["children"]]
    starts = [
        float((run_root / f"seed-{seed}" / "started.txt").read_text()) for seed in seeds
    ]
    finishes = [
        float((run_root / f"seed-{seed}" / "finished.txt").read_text())
        for seed in seeds
    ]
    assert max(starts) < min(finishes)
    receipts = [
        contract.validate_child_receipt(
            json.loads((run_root / f"seed-{seed}" / "seed_status.json").read_text()),
            manifest=json.loads((run_root / "batch_manifest.json").read_text()),
        )
        for seed in seeds
    ]
    assert [receipt["ordinal"] for receipt in receipts] == list(range(4))
    cpu_sets = [set(receipt["cpu_set"]) for receipt in receipts]
    assert all(len(cpu_set) == 4 for cpu_set in cpu_sets)
    assert len(set().union(*cpu_sets)) == 16
    for receipt in receipts:
        child_telemetry = receipt["child_resource_telemetry"]
        assert child_telemetry["available"] is True
        assert child_telemetry["child_cpus"] == 4
        assert child_telemetry["cpu_capacity_seconds"] == pytest.approx(
            child_telemetry["wall_time_seconds"] * 4
        )
        assert child_telemetry["cpu_utilization_fraction"] == pytest.approx(
            child_telemetry["process_tree_cpu_seconds"]
            / child_telemetry["cpu_capacity_seconds"]
        )
        child_dir = run_root / f"seed-{receipt['seed']}"
        for label in ("stdout", "stderr"):
            stream = child_dir / f"child_{label}.log"
            assert stream.is_file()
            assert receipt[f"{label}_relative_path"] == (
                f"seed-{receipt['seed']}/child_{label}.log"
            )
            assert receipt[f"{label}_size_bytes"] == stream.stat().st_size
            assert receipt[f"{label}_sha256"] == hashlib.sha256(
                stream.read_bytes()
            ).hexdigest()
            assert not (child_dir / f"child_{label}.log.active").exists()
        assert receipt["stdio_capture_complete"] is True
    status = contract.validate_task_status(
        json.loads((run_root / "task_status.json").read_text())
    )
    assert status["state"] == "completed"
    assert status["sealed_child_count"] == 4
    assert status["physical_scheduler_task_count"] == 1
    assert status["virtual_scheduler_task_ids_created"] is False
    memory_path = run_root / contract.PARENT_MEMORY_EVIDENCE_FILENAME
    memory_evidence = contract.validate_parent_memory_evidence(
        json.loads(memory_path.read_text(encoding="utf-8"))
    )
    observations = [
        {
            "ordinal": receipt["ordinal"],
            "seed": receipt["seed"],
            "legacy_status_sha256": receipt["legacy_status_sha256"],
            "observed_peak_rss_bytes": receipt["legacy_status"][
                "observed_peak_rss_bytes"
            ],
        }
        for receipt in receipts
    ]
    assert memory_evidence["task_id"] == "99400"
    assert memory_evidence["manifest_sha256"] == json.loads(
        (run_root / "batch_manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]
    assert memory_evidence["task_status_sha256"] == status["status_sha256"]
    assert memory_evidence["logical_child_count"] == 4
    assert memory_evidence["parent_requested_memory_bytes"] == parent_memory_bytes
    assert memory_evidence["child_peak_rss_available_count"] == 4
    assert memory_evidence["child_peak_rss_sum_bytes"] == 4 * 1024**3
    assert memory_evidence["child_peak_rss_max_bytes"] == 1024**3
    assert memory_evidence["child_peak_rss_observations_sha256"] == (
        phase_a.canonical_sha256(observations)
    )
    assert memory_evidence["safety_passed"] is True
    assert memory_evidence["fea_submission_performed"] is False
    assert memory_evidence["aedt_used"] is False
    assert not list(run_root.glob(f".{memory_path.name}*.tmp"))
    telemetry = status["resource_telemetry"]
    assert telemetry["parent_requested_cpus"] == 16
    assert telemetry["logical_child_cpus"] == 4
    assert telemetry["logical_child_count"] == 4
    assert telemetry["configured_model_load_stagger_seconds"] == 5
    assert telemetry["maximum_planned_dispatch_fill_seconds"] == 15
    assert 0 <= telemetry["dispatch_fill_seconds"] < wall
    assert telemetry["scheduler_ready_lane_reserve"] == 0
    assert telemetry["scheduler_admission_policy"] == "natural-terminal-vacancy-only-v1"
    assert telemetry["logical_child_cpu_capacity_seconds"] == pytest.approx(
        telemetry["sum_child_wall_time_seconds"] * 4
    )
    assert telemetry["scheduler_envelope_idle_capacity_seconds"] == pytest.approx(
        max(
            0,
            telemetry["requested_cpu_capacity_seconds"]
            - telemetry["logical_child_cpu_capacity_seconds"],
        )
    )
    if telemetry["available"]:
        assert telemetry["logical_child_cpu_utilization_fraction"] == pytest.approx(
            telemetry["aggregate_child_cpu_seconds"]
            / telemetry["logical_child_cpu_capacity_seconds"]
        )
        assert telemetry["reported_child_cpu_available_count"] == 4
        assert telemetry["reported_child_cpu_utilization_fraction"] == pytest.approx(
            telemetry["reported_child_process_tree_cpu_seconds"]
            / telemetry["reported_child_cpu_capacity_seconds"]
        )
        assert telemetry["reaped_minus_reported_child_cpu_seconds"] == pytest.approx(
            telemetry["aggregate_child_cpu_seconds"]
            - telemetry["reported_child_process_tree_cpu_seconds"]
        )
    assert telemetry["future_comparison_child_cpu_counts"] == [1, 2, 4]
    assert telemetry["automatic_child_cpu_reduction_allowed"] is False
    assert (
        task["payload_json"]["cpu_telemetry"]["automatic_child_cpu_reduction_allowed"]
        is False
    )
    runtime_roots = []
    for seed in seeds:
        child_dir = run_root / f"seed-{seed}"
        runtime_environment = json.loads((child_dir / "runtime_env.json").read_text())
        assert runtime_environment["TMPDIR"] == runtime_environment["TMP"]
        assert runtime_environment["TMPDIR"] == runtime_environment["TEMP"]
        runtime_roots.append(str(Path(runtime_environment["TMPDIR"]).parent))
        assert not (child_dir / "runtime").exists()
    assert len(set(runtime_roots)) == 4
    assert shared_sentinel.read_text(encoding="ascii") == "must survive"


def test_one_seed_local_failure_does_not_stop_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, task_id=99401
    )
    seeds = [child["seed"] for child in task["payload_json"]["children"]]
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "0.08")
    monkeypatch.setenv("PHASE_B_TEST_FAILURE_SEED", str(seeds[1]))
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(300),
        )
        == runner.CHILD_FAILURE_EXIT_CODE
    )
    run_root = bundle / "runs" / "task-99401"
    receipts = [
        json.loads((run_root / f"seed-{seed}" / "seed_status.json").read_text())
        for seed in seeds
    ]
    assert [receipt["state"] for receipt in receipts].count("failed") == 1
    assert [receipt["state"] for receipt in receipts].count("completed") == 3
    status = json.loads((run_root / "task_status.json").read_text())
    assert status["state"] == "completed_with_failures"
    assert status["sealed_child_count"] == 4
    failed = next(receipt for receipt in receipts if receipt["state"] == "failed")
    stderr_path = run_root / failed["stderr_relative_path"]
    assert "synthetic seed-local optimizer failure" in stderr_path.read_text()
    assert failed["stderr_sha256"] == hashlib.sha256(stderr_path.read_bytes()).hexdigest()


def test_out_of_order_completion_stays_hidden_until_prefix_and_restart_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, task_id=99405
    )
    seeds = [child["seed"] for child in task["payload_json"]["children"]]
    monkeypatch.setenv(
        "PHASE_B_TEST_SLEEP_BY_SEED",
        json.dumps(
            {
                str(seeds[0]): 0.45,
                str(seeds[1]): 0.05,
                str(seeds[2]): 0.05,
                str(seeds[3]): 0.05,
            }
        ),
    )
    result: list[int] = []
    worker = threading.Thread(
        target=lambda: result.append(
            _run(
                bundle,
                payload_root,
                payload_path,
                task,
                monotonic=_scaled_clock(200),
            )
        )
    )
    worker.start()
    run_root = bundle / "runs" / "task-99405"
    early_finished = run_root / f"seed-{seeds[3]}" / "finished.txt"
    deadline = time.monotonic() + 3
    while not early_finished.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert early_finished.is_file()
    # Ordinal 3 is physically done, but its immutable receipt is not public
    # before slow ordinal 0 closes the contiguous prefix.
    assert not (run_root / f"seed-{seeds[3]}" / "seed_status.json").exists()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert result == [0]
    assert all(
        (run_root / f"seed-{seed}" / "seed_status.json").is_file() for seed in seeds
    )
    with pytest.raises(RuntimeError, match="new physical parent identity"):
        _run(bundle, payload_root, payload_path, task)


def test_nonzero_process_exit_cannot_be_hidden_by_completed_child_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, length=1, task_id=99409
    )
    seed = int(task["payload_json"]["children"][0]["seed"])
    monkeypatch.setenv("PHASE_B_TEST_EXIT_OVERRIDE_SEED", str(seed))

    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(300),
        )
        == runner.LANE_FATAL_EXIT_CODE
    )
    run_root = bundle / "runs" / "task-99409"
    status = json.loads((run_root / "task_status.json").read_text())
    receipt = json.loads(
        (run_root / f"seed-{seed}" / "seed_status.json").read_text()
    )
    assert status["state"] == "failed"
    assert receipt["state"] == "refused"
    assert "exit-code mismatch" in receipt["failure"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal contract")
def test_signal_stops_all_children_and_same_parent_restart_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, task_id=99402
    )
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "5")

    def interrupt() -> None:
        time.sleep(0.15)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=interrupt)
    sender.start()
    try:
        assert (
            _run(
                bundle,
                payload_root,
                payload_path,
                task,
                monotonic=_scaled_clock(300),
            )
            == runner.STOPPED_EXIT_CODE
        )
    finally:
        sender.join(timeout=2)
    run_root = bundle / "runs" / "task-99402"
    status = contract.validate_task_status(
        json.loads((run_root / "task_status.json").read_text())
    )
    assert status["state"] == "stopped"
    assert status["stop_requested"] is True
    assert not status["active_children"]
    with pytest.raises(RuntimeError, match="new physical parent identity"):
        _run(bundle, payload_root, payload_path, task)


def test_deadline_kills_children_and_seals_deadline_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path,
        monkeypatch,
        task_id=99403,
        internal_deadline_seconds=100,
        cleanup_reserve_seconds=10,
        minimum_child_start_budget_seconds=50,
    )
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "5")
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(1_000),
        )
        == runner.STOPPED_EXIT_CODE
    )
    status = contract.validate_task_status(
        json.loads((bundle / "runs" / "task-99403" / "task_status.json").read_text())
    )
    assert status["state"] == "deadline"
    assert str(status["stop_reason"]).startswith("internal_deadline")


def test_child_affinity_contract_enforces_exact_subset_without_nested_srun():
    current = {2, 3, 4, 5, 6, 7}
    writes: list[set[int]] = []

    def get(_pid: int) -> set[int]:
        return set(current)

    def put(_pid: int, cpus: set[int]) -> None:
        writes.append(set(cpus))
        current.clear()
        current.update(cpus)

    observed = single_runner.apply_phase_b_child_affinity(
        {
            single_runner.PHASE_B_CHILD_MARKER_ENV: single_runner.PHASE_B_CHILD_MARKER,
            single_runner.PHASE_B_CHILD_CPUSET_ENV: "2,3,4,5",
        },
        platform_name="posix",
        affinity_getter=get,
        affinity_setter=put,
    )
    assert observed == (2, 3, 4, 5)
    assert writes == [{2, 3, 4, 5}]
    assert contract.CPU_ISOLATION["nested_srun_used"] is False
    with pytest.raises(RuntimeError, match="escaped"):
        single_runner.apply_phase_b_child_affinity(
            {
                single_runner.PHASE_B_CHILD_MARKER_ENV: single_runner.PHASE_B_CHILD_MARKER,
                single_runner.PHASE_B_CHILD_CPUSET_ENV: "2,3,4,99",
            },
            platform_name="posix",
            affinity_getter=lambda _pid: {2, 3, 4, 5},
            affinity_setter=lambda _pid, _cpus: None,
        )


def test_repeated_predict_stress_forbids_joblib_threadpool_and_semlock():
    import numpy as np

    class SafePredictor:
        def __init__(self):
            self.fitted = types.SimpleNamespace(n_jobs=1)
            self.bundle = {"models": [("extratrees", self.fitted)]}
            self.calls = 0

        def predict_mu_sigma(self, frame, conformal=True):
            assert conformal is True
            self.calls += 1
            return np.ones(len(frame)), np.full(len(frame), 0.1)

    models = {
        target: SafePredictor()
        for target in generation_preflight.CURRENT_REQUIRED_MODEL_TARGETS
    }
    fake_parallel_backends = types.SimpleNamespace(ThreadPool=object)
    stress = generation_preflight.attest_semlock_free_repeated_predict(
        models,
        [{"x": 1.0}],
        repeats=64,
        parallel_backends_module=fake_parallel_backends,
    )
    assert stress["predict_call_count"] == 64 * len(models)
    assert stress["joblib_thread_pool_construction_attempt_count"] == 0
    assert stress["multiprocessing_semlock_construction_attempt_count"] == 0
    assert all(predictor.calls == 64 for predictor in models.values())
    binding = {
        "threads_per_model": 4,
        "target_count": len(models),
        "model_count": len(models),
        "families": ["extratrees"],
        "family_threads": {"extratrees": 1},
        "semaphore_free_families": ["extratrees"],
        "semaphore_free_sklearn_forest": True,
        "policy": single_runner.PHASE_B_INFERENCE_POLICY,
    }
    release_unsigned = {
        "required_release_commit": (single_runner.PHASE_B_SEMLOCK_SAFE_RELEASE_COMMIT),
        "smoke_schema": "mft-tier1-semlock-safe-inference-smoke-v1",
        "helper_relative_path": "tools/tier1_semlock_safe_inference_smoke.py",
        "helper_sha256": single_runner.PHASE_B_SEMLOCK_SAFE_HELPER_SHA256,
        "semaphore_free_sklearn_families": ["extratrees", "randomforest"],
        "helper_code_inventory_authenticated": True,
        "tmp_isolation_claimed_as_enospc_fix": False,
    }
    release = {
        **release_unsigned,
        "sha256": phase_a.canonical_sha256(release_unsigned),
    }
    assert (
        single_runner.validate_phase_b_semlock_preflight(
            {
                "inference_binding": binding,
                "semlock_free_repeated_predict": stress,
                "semlock_safe_inference_release": release,
            }
        )
        == stress
    )

    class UnsafePredictor(SafePredictor):
        def predict_mu_sigma(self, frame, conformal=True):
            fake_parallel_backends.ThreadPool(2)
            return super().predict_mu_sigma(frame, conformal=conformal)

    unsafe = dict(models)
    unsafe[next(iter(unsafe))] = UnsafePredictor()
    with pytest.raises(RuntimeError, match="ThreadPool construction is forbidden"):
        generation_preflight.attest_semlock_free_repeated_predict(
            unsafe,
            [{"x": 1.0}],
            repeats=2,
            parallel_backends_module=fake_parallel_backends,
        )


def test_phase_b_parent_refuses_completed_child_without_semlock_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, length=1, task_id=99406
    )
    monkeypatch.setenv("PHASE_B_TEST_OMIT_SEMLOCK_ATTESTATION", "1")
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(300),
        )
        == runner.LANE_FATAL_EXIT_CODE
    )
    status = json.loads(
        (bundle / "runs" / "task-99406" / "task_status.json").read_text()
    )
    assert status["state"] == "failed"
    receipt = next((bundle / "runs" / "task-99406").glob("seed-*/seed_status.json"))
    assert "SemLock stress attestation" in json.loads(receipt.read_text())["failure"]


def test_phase_b_parent_refuses_completed_child_without_cpu_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, length=1, task_id=99407
    )
    monkeypatch.setenv("PHASE_B_TEST_OMIT_CHILD_TELEMETRY", "1")
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(300),
        )
        == runner.LANE_FATAL_EXIT_CODE
    )
    receipt = next((bundle / "runs" / "task-99407").glob("seed-*/seed_status.json"))
    sealed = json.loads(receipt.read_text())
    assert sealed["state"] == "refused"
    assert "child CPU telemetry mismatch" in sealed["failure"]
    assert sealed["child_resource_telemetry"]["available"] is False


class MemoryRemote:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def stat(self, account_name: str, path: str):
        from tools.tier1_corrected_current7_slurm_harvest import RemoteFileStat

        payload = self.files[path]
        return RemoteFileStat(size=len(payload), mtime=1, mode=0o444)

    def read_bytes(self, account_name: str, path: str, *, maximum_bytes: int):
        payload = self.files[path]
        assert len(payload) <= maximum_bytes
        return payload


def test_terminal_receipt_before_cursor_recovery_is_complete_bound_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path, monkeypatch, task_id=99408
    )
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "0.01")
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(300),
        )
        == 0
    )
    run_root = bundle / "runs" / "task-99408"
    remote_root = task["remote_cwd"]
    batch_manifest = json.loads((run_root / "batch_manifest.json").read_text())
    final_status = json.loads((run_root / "task_status.json").read_text())

    # Model the narrow crash: ordinal 0's immutable receipt reached disk, but
    # the parent died after Scheduler terminalized and before cursor commit.
    stale_status = contract.seal_task_status(
        {
            **final_status,
            "state": "running",
            "active_children": [],
            "launched_child_count": 1,
            "finished_child_count": 1,
            "sealed_child_count": 0,
            "completed_child_count": 0,
            "failed_child_count": 0,
            "finished_at": None,
            "resource_telemetry": None,
        }
    )
    child = task["payload_json"]["children"][0]
    seed = int(child["seed"])
    receipt = json.loads(
        (run_root / f"seed-{seed}" / "seed_status.json").read_text()
    )
    result_value = {
        "seed": seed,
        "completed_generations": 1,
        "feasible_pareto_count": 0,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    result_payload = phase_a.json_bytes(result_value)
    receipt["result_sha256"] = hashlib.sha256(result_payload).hexdigest()
    receipt = contract.seal_child_receipt(receipt)
    receipt_payload = phase_a.json_bytes(receipt)
    receipt_path = (
        f"{remote_root}/runs/task-99408/seed-{seed}/seed_status.json"
    )
    result_path = f"{remote_root}/runs/task-99408/seed-{seed}/result.json"
    files = {
        f"{remote_root}/runs/task-99408/batch_manifest.json": phase_a.json_bytes(
            batch_manifest
        ),
        f"{remote_root}/runs/task-99408/task_status.json": phase_a.json_bytes(
            stale_status
        ),
        receipt_path: receipt_payload,
        result_path: result_payload,
    }
    receipt_scan = []
    for ordinal, manifest_child in enumerate(batch_manifest["ordered_children"]):
        child_seed = int(manifest_child["seed"])
        path = (
            f"{remote_root}/runs/task-99408/seed-{child_seed}/seed_status.json"
        )
        if ordinal == 0:
            receipt_scan.append(
                {
                    "ordinal": ordinal,
                    "seed": child_seed,
                    "present": True,
                    "remote_path": path,
                    "receipt": copy.deepcopy(receipt),
                    "remote_receipt_sha256": hashlib.sha256(
                        receipt_payload
                    ).hexdigest(),
                    "remote_receipt_size": len(receipt_payload),
                    "remote_mode": 0o444,
                    "stable_stat_count": 3,
                    "stable_byte_read_count": 2,
                    "absence_confirmed": False,
                }
            )
        else:
            receipt_scan.append(
                {
                    "ordinal": ordinal,
                    "seed": child_seed,
                    "present": False,
                    "remote_path": path,
                    "receipt": None,
                    "remote_receipt_sha256": None,
                    "remote_receipt_size": None,
                    "remote_mode": None,
                    "stable_stat_count": 2,
                    "stable_byte_read_count": 0,
                    "absence_confirmed": True,
                }
            )
    watcher_capability_sha256 = "a" * 64
    watcher_revision_sha256 = "b" * 64
    recovery = contract.build_terminal_receipt_recovery_attestation(
        parent_task=task,
        manifest=batch_manifest,
        task_status=stale_status,
        task_id=99408,
        scheduler_state="completed",
        receipt_scan=receipt_scan,
        watcher_capability_sha256=watcher_capability_sha256,
        watcher_revision_sha256=watcher_revision_sha256,
        scheduler_get_count=1,
    )
    base_item = {
        "task_id": 99408,
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "account_name": "fixture",
        "status": "completed",
        "exit_code": 0,
        "parent_task": copy.deepcopy(task),
        "scheduler_task": copy.deepcopy(task),
        "phase_b_watcher_capability_sha256": watcher_capability_sha256,
        "phase_b_watcher_revision_sha256": watcher_revision_sha256,
    }
    plan = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "bundle_manifest_sha256": task["payload_json"][
            "bundle_manifest_sha256"
        ],
        "remote_bundle": remote_root,
    }
    bundle_manifest = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "search_execution": {"result_filename": "result.json"},
    }
    monkeypatch.setattr(
        harvest,
        "_harvest_result_artifacts",
        lambda **_kwargs: ({}, {}, []),
    )

    # No recovery proof preserves the stale, empty cursor prefix.
    assert (
        harvest.harvest_batch_lane(
            base_item,
            plan=plan,
            manifest=bundle_manifest,
            remote=MemoryRemote(copy.deepcopy(files)),
            result_validator=lambda *_args, **_kwargs: None,
        )
        == []
    )
    recovered_item = {
        **base_item,
        "phase_b_terminal_receipt_recovery_attestation": recovery,
    }
    observations = harvest.harvest_batch_lane(
        recovered_item,
        plan=plan,
        manifest=bundle_manifest,
        remote=MemoryRemote(copy.deepcopy(files)),
        result_validator=lambda *_args, **_kwargs: None,
    )
    assert len(observations) == 1
    assert observations[0]["seed"] == seed
    assert observations[0]["terminal_state"] == "completed"
    assert observations[0]["result_object"]["sha256"] == receipt[
        "result_sha256"
    ]
    assert recovery["scheduler_access"] == "GET-only"
    assert recovery["scheduler_post_count"] == 0
    assert recovery["remote_access"] == "read-only"
    assert recovery["remote_write_count"] == 0

    def reseal(value: dict[str, Any]) -> dict[str, Any]:
        unsigned = {
            key: item for key, item in value.items() if key != "attestation_sha256"
        }
        value["attestation_sha256"] = phase_a.canonical_sha256(unsigned)
        return value

    invalid_recoveries = []
    incomplete = copy.deepcopy(recovery)
    incomplete["receipt_scan"].pop()
    invalid_recoveries.append(reseal(incomplete))
    tampered_receipt = copy.deepcopy(recovery)
    tampered_receipt["receipt_scan"][0]["receipt"]["failure"] = "tampered"
    invalid_recoveries.append(reseal(tampered_receipt))
    parent_mismatch = copy.deepcopy(recovery)
    parent_mismatch["parent_task_sha256"] = "c" * 64
    invalid_recoveries.append(reseal(parent_mismatch))
    scheduler_write = copy.deepcopy(recovery)
    scheduler_write["scheduler_post_count"] = 1
    invalid_recoveries.append(reseal(scheduler_write))
    for invalid in invalid_recoveries:
        with pytest.raises(RuntimeError, match="terminal recovery|terminal receipt"):
            harvest.harvest_batch_lane(
                {
                    **base_item,
                    "phase_b_terminal_receipt_recovery_attestation": invalid,
                },
                plan=plan,
                manifest=bundle_manifest,
                remote=MemoryRemote(copy.deepcopy(files)),
                result_validator=lambda *_args, **_kwargs: None,
            )

    with pytest.raises(RuntimeError, match="attestation seal mismatch"):
        harvest.harvest_batch_lane(
            {
                **recovered_item,
                "phase_b_watcher_revision_sha256": "d" * 64,
            },
            plan=plan,
            manifest=bundle_manifest,
            remote=MemoryRemote(copy.deepcopy(files)),
            result_validator=lambda *_args, **_kwargs: None,
        )
    with pytest.raises(RuntimeError, match="attestation seal mismatch"):
        harvest.harvest_batch_lane(
            {**recovered_item, "status": "running"},
            plan=plan,
            manifest=bundle_manifest,
            remote=MemoryRemote(copy.deepcopy(files)),
            result_validator=lambda *_args, **_kwargs: None,
        )

    # Even a self-consistent attestation cannot substitute different remote
    # immutable bytes: the harvester re-reads and hashes the actual receipt.
    wrong_hash = copy.deepcopy(recovery)
    wrong_hash["receipt_scan"][0]["remote_receipt_sha256"] = "f" * 64
    wrong_hash = reseal(wrong_hash)
    with pytest.raises(RuntimeError, match="differs from remote receipt bytes"):
        harvest.harvest_batch_lane(
            {
                **base_item,
                "phase_b_terminal_receipt_recovery_attestation": wrong_hash,
            },
            plan=plan,
            manifest=bundle_manifest,
            remote=MemoryRemote(copy.deepcopy(files)),
            result_validator=lambda *_args, **_kwargs: None,
        )

    # The status object is an exact input to the recovery seal, not merely a
    # numeric cursor copied into it.
    changed_status = contract.seal_task_status(
        {**stale_status, "updated_at": "2026-07-23T01:02:03+09:00"}
    )
    changed_files = copy.deepcopy(files)
    changed_files[
        f"{remote_root}/runs/task-99408/task_status.json"
    ] = phase_a.json_bytes(changed_status)
    with pytest.raises(RuntimeError, match="parent/journal binding"):
        harvest.harvest_batch_lane(
            recovered_item,
            plan=plan,
            manifest=bundle_manifest,
            remote=MemoryRemote(changed_files),
            result_validator=lambda *_args, **_kwargs: None,
        )


def test_stopped_parent_terminal_prefix_replays_without_virtual_task_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare(
        tmp_path,
        monkeypatch,
        task_id=99404,
        internal_deadline_seconds=100,
        cleanup_reserve_seconds=10,
        minimum_child_start_budget_seconds=50,
    )
    monkeypatch.setenv("PHASE_B_TEST_SLEEP", "5")
    assert (
        _run(
            bundle,
            payload_root,
            payload_path,
            task,
            monotonic=_scaled_clock(1_000),
        )
        == runner.STOPPED_EXIT_CODE
    )
    run_root = bundle / "runs" / "task-99404"
    remote_root = task["remote_cwd"]
    files = {
        f"{remote_root}/runs/task-99404/batch_manifest.json": (
            run_root / "batch_manifest.json"
        ).read_bytes(),
        f"{remote_root}/runs/task-99404/task_status.json": (
            run_root / "task_status.json"
        ).read_bytes(),
    }
    for receipt in run_root.glob("seed-*/seed_status.json"):
        files[
            f"{remote_root}/runs/task-99404/{receipt.parent.name}/seed_status.json"
        ] = receipt.read_bytes()
    item = {
        "task_id": 99404,
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "account_name": "fixture",
        "status": "failed",
        "exit_code": runner.STOPPED_EXIT_CODE,
        "parent_task": copy.deepcopy(task),
        "scheduler_task": copy.deepcopy(task),
    }
    plan = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "bundle_manifest_sha256": task["payload_json"]["bundle_manifest_sha256"],
        "remote_bundle": remote_root,
    }
    bundle_manifest = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "search_execution": {"result_filename": "result.json"},
    }
    replay = harvest.harvest_mixed_inventory(
        [item], plan=plan, manifest=bundle_manifest, remote=MemoryRemote(files)
    )
    assert replay["physical_task_count"] == 1
    assert replay["authenticated_seed_count"] >= 1
    assert replay["virtual_scheduler_task_ids_created"] is False
    assert {record["physical_parent_task_id"] for record in replay["records"]} == {
        99404
    }
