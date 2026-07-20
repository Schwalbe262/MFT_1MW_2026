from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import tier1_corrected_current7_slurm_harvest as harvest


def _bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _payload_builder(plan, _manifest, *, seed, priority=None):
    del priority
    payload = {
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "seed": int(seed),
        "lane": {"island_id": f"island-{seed}", "fixed_primary_turns": 5},
    }
    return {
        "name": f"mft-t1c7-test-{seed}",
        "dedupe_key": f"mft-tier1-current7:test-{seed}",
        "payload_json": payload,
    }


def _contracts(seeds=(101,)):
    plan = {
        "bundle_id": "current7-1234567890abcdefabcd",
        "bundle_manifest_sha256": "a" * 64,
        "remote_bundle": "/gpfs/tmp_cpu2/current7-test",
    }
    manifest = {
        "contract_sha256": "b" * 64,
        "task_schema_version": "mft-tier1-current7-slurm-seed-task-v1",
        "status_schema_version": harvest.STATUS_SCHEMA,
        "temperature_targets": [f"temperature-{index}" for index in range(7)],
        "search_execution": {
            "result_schema_version": harvest.RESULT_SCHEMA,
            "result_filename": "result.json",
        },
    }
    submissions = {}
    for offset, seed in enumerate(seeds, start=1):
        expected = _payload_builder(plan, manifest, seed=seed)
        submissions[expected["dedupe_key"]] = {
            "dedupe_key": expected["dedupe_key"],
            "task_id": 7000 + offset,
            "name": expected["name"],
            "seed": seed,
            "island_id": f"island-{seed}",
            "wave": "canary",
        }
    state = {
        "schema_version": harvest.CONTROLLER_STATE_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "priority_override": None,
        "submissions": submissions,
    }
    state["state_sha256"] = harvest.canonical_sha256(state)
    return plan, manifest, state


class Scheduler:
    def __init__(self, tasks):
        self.tasks = {int(task["id"]): copy.deepcopy(task) for task in tasks}
        self.get_count = 0

    def get_task(self, task_id):
        self.get_count += 1
        return copy.deepcopy(self.tasks.get(int(task_id)))


class Remote:
    def __init__(self, files, *, unstable=()):
        self.files = {key: bytes(value) for key, value in files.items()}
        self.unstable = set(unstable)
        self.read_counts = {}
        self.stat_count = 0

    def stat(self, account_name, path):
        self.stat_count += 1
        value = self.files[(account_name, path)]
        return harvest.RemoteFileStat(size=len(value), mtime=1234, mode=0o444)

    def read_bytes(self, account_name, path, *, maximum_bytes):
        key = (account_name, path)
        self.read_counts[key] = self.read_counts.get(key, 0) + 1
        value = self.files[key]
        assert len(value) <= maximum_bytes
        if key in self.unstable and self.read_counts[key] == 2:
            return value[:-1] + (b" " if value[-1:] != b" " else b"\n")
        return value


def _task_and_files(plan, manifest, state, *, terminal="completed", candidates=None):
    submission = next(iter(state["submissions"].values()))
    task_id = submission["task_id"]
    seed = submission["seed"]
    expected = _payload_builder(plan, manifest, seed=seed)
    hard_spec = {
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 110.0,
        "B_limit_T": 1.2,
        "insulation_min_mm": 40.0,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "resonance_min_Hz": 15_000.0,
        "size_W_max_mm": 1_200.0,
        "size_L_max_mm": 1_200.0,
        "size_H_max_mm": 750.0,
    }
    supplied_candidates = candidates or []
    default_least = {
        "decoded_params": {"fallback": 1},
        "volume_L": 999.0,
        "total_loss_W": 9999.0,
        "physical_constraint_G": {"resonance": 0.1},
        "physical_feasible": False,
    }
    pareto_candidates = [
        copy.deepcopy(candidate)
        for candidate in supplied_candidates
        if candidate.get("physical_feasible", candidate.get("feasible")) is True
    ]
    least_candidates = [
        copy.deepcopy(supplied_candidates[0] if supplied_candidates else default_least)
    ]
    artifact_payloads = {
        "pareto_candidates": _bytes(
            {
                "schema_version": harvest.PARETO_CANDIDATES_SCHEMA,
                "authoritative_constraints": "terminal_unscaled_physical_replay",
                "candidate_count": len(pareto_candidates),
                "candidates": pareto_candidates,
                "production_eligible": False,
                "fea_submission_performed": False,
                "automatic_promotion_allowed": False,
            }
        ),
        "least_violation_candidates": _bytes(
            {
                "schema_version": harvest.LEAST_VIOLATION_CANDIDATES_SCHEMA,
                "ranking": "minimum_sum_positive_optimizer_normalized_G",
                "candidate_count": 1,
                "candidates": least_candidates,
                "production_eligible": False,
                "fea_submission_performed": False,
                "automatic_promotion_allowed": False,
            }
        ),
        "infeasibility_report": _bytes(
            {
                "schema_version": harvest.INFEASIBILITY_REPORT_SCHEMA,
                "authoritative_constraints": "terminal_unscaled_physical_replay",
                "population_size": max(1, len(supplied_candidates)),
                "physical_feasible_count": len(pareto_candidates),
                "production_eligible": False,
                "fea_submission_performed": False,
                "automatic_promotion_allowed": False,
            }
        ),
    }
    filenames = {
        "pareto_X": "pareto_X.npy",
        "pareto_F": "pareto_F.npy",
        "pareto_G_physical": "pareto_G_physical.npy",
        "pareto_front": "pareto_front.csv",
        "pareto_candidates": "pareto_candidates.json",
        "terminal_X": "terminal_X.npy",
        "terminal_F": "terminal_F.npy",
        "terminal_G_optimizer": "terminal_G_optimizer.npy",
        "terminal_G_physical": "terminal_G_physical.npy",
        "least_violation_X": "least_violation_X.npy",
        "least_violation_F": "least_violation_F.npy",
        "least_violation_G_physical": "least_violation_G_physical.npy",
        "least_violation_front": "least_violation.csv",
        "least_violation_candidates": "least_violation_candidates.json",
        "infeasibility_report": "infeasibility_report.json",
    }
    for name in harvest.REQUIRED_SEARCH_ARTIFACTS:
        artifact_payloads.setdefault(name, f"sealed-{name}\n".encode())
    artifact_inventory = {
        name: {
            "path": filenames[name],
            "sha256": hashlib.sha256(artifact_payloads[name]).hexdigest(),
            "size_bytes": len(artifact_payloads[name]),
        }
        for name in sorted(harvest.REQUIRED_SEARCH_ARTIFACTS)
    }
    result = {
        "schema_version": harvest.RESULT_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "seed": seed,
        "completed_generations": 201,
        "constraint_version": (
            "1200x1200x750-res15k-t110-core4-cw1-5-lmhalf"
        ),
        "hard_spec": hard_spec,
        "stage_spec_sha256": harvest.canonical_sha256(hard_spec),
        "hard_constraint_contract_sha256": "c" * 64,
        "temperature_contract_sha256": "d" * 64,
        "constraint_names": ["Llt_robust_band", "exterior_width_limit"],
        "temperature_targets": list(manifest["temperature_targets"]),
        "terminal_population_count": max(1, len(supplied_candidates)),
        "physical_feasible_count": len(pareto_candidates),
        "feasible_pareto_count": len(pareto_candidates),
        "least_violation_count": 1,
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": harvest.canonical_sha256(artifact_inventory),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    result_bytes = _bytes(result)
    status = {
        "schema_version": harvest.STATUS_SCHEMA,
        "terminal": terminal in {"completed", "failed", "cancelled"},
        "state": terminal,
        "task_id": str(task_id),
        "seed": seed,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "payload_sha256": harvest.canonical_sha256(expected["payload_json"]),
        "island_id": submission["island_id"],
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    status_path = harvest._remote_child(
        plan["remote_bundle"], "runs", f"task-{task_id}", "seed_status.json"
    )
    result_path = harvest._remote_child(
        plan["remote_bundle"],
        "runs",
        f"task-{task_id}",
        f"seed-{seed}",
        "result.json",
    )
    task = {
        "id": task_id,
        "name": expected["name"],
        "dedupe_key": expected["dedupe_key"],
        "status": "completed" if terminal == "completed" else "failed",
        "account_name": "account-a",
        "slurm_job_id": "8001",
        "exit_code": 0 if terminal == "completed" else 1,
        "created_at": "2026-07-20T01:00:00+00:00",
        "started_at": "2026-07-20T01:01:00+00:00",
        "finished_at": "2026-07-20T01:02:00+00:00",
    }
    files = {
        ("account-a", status_path): _bytes(status),
        ("account-a", result_path): result_bytes,
    }
    output_root = harvest._remote_child(
        plan["remote_bundle"], "runs", f"task-{task_id}", f"seed-{seed}"
    )
    for name, payload in artifact_payloads.items():
        files[("account-a", harvest._remote_child(output_root, filenames[name]))] = payload
    return task, status, result, files


def _noop_validator(_value, **_kwargs):
    return None


def test_default_dry_run_skips_nonterminal_and_writes_zero(tmp_path):
    plan, manifest, state = _contracts()
    submission = next(iter(state["submissions"].values()))
    expected = _payload_builder(plan, manifest, seed=submission["seed"])
    task = {
        "id": submission["task_id"],
        "name": expected["name"],
        "dedupe_key": expected["dedupe_key"],
        "status": "running",
        "account_name": "account-a",
    }
    remote = Remote({})
    runtime = tmp_path / "runtime"
    result = harvest.harvest_snapshot(
        plan=plan,
        manifest=manifest,
        controller_state=state,
        scheduler=Scheduler([task]),
        remote=remote,
        runtime=runtime,
        payload_builder=_payload_builder,
        result_validator=_noop_validator,
    )
    assert result["apply"] is False
    assert result["local_write_count"] == 0
    assert result["scheduler_mutation_count"] == 0
    assert result["remote_write_count"] == 0
    assert result["nonterminal_skipped_count"] == 1
    assert remote.stat_count == 0
    assert not runtime.exists()


def test_apply_is_additive_content_addressed_and_idempotent(tmp_path):
    plan, manifest, state = _contracts()
    candidates = [
        {
            "decoded_params": {"x": 1},
            "volume_L": 800.0,
            "total_loss_W": 5900.0,
            "physical_constraint_G": {"resonance": -0.2},
            "physical_feasible": True,
        }
    ]
    task, _status, result_value, files = _task_and_files(
        plan, manifest, state, candidates=candidates
    )
    runtime = tmp_path / "runtime"
    legacy = runtime / "canonical" / "index.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-all11-sentinel\n")
    arguments = dict(
        plan=plan,
        manifest=manifest,
        controller_state=state,
        scheduler=Scheduler([task]),
        remote=Remote(files),
        runtime=runtime,
        apply=True,
        payload_builder=_payload_builder,
        result_validator=_noop_validator,
    )
    first = harvest.harvest_snapshot(**arguments)
    assert first["authenticated_seed_count"] == 1
    assert first["refused_terminal_count"] == 0
    assert first["aggregate"]["feasible_candidate_count"] == 1
    assert first["aggregate"]["pareto_count"] == 1
    assert first["aggregate"]["hard_spec"]["resonance_min_Hz"] == 15_000.0
    assert first["aggregate"]["constraint_version"].startswith("1200x1200")
    assert first["local_write_count"] == 21
    assert legacy.read_bytes() == b"legacy-all11-sentinel\n"
    current7_index = json.loads(
        (runtime / "canonical" / "current7-index.json").read_text()
    )
    assert current7_index["schema_version"] == harvest.CURRENT7_INDEX_SCHEMA
    assert current7_index["legacy_canonical_index_touched"] is False
    result_sha = hashlib.sha256(_bytes(result_value)).hexdigest()
    assert (
        runtime
        / "current7-cache"
        / "objects"
        / "sha256"
        / result_sha[:2]
        / f"{result_sha}.json"
    ).is_file()

    second = harvest.harvest_snapshot(**arguments)
    assert second["cohort_id"] == first["cohort_id"]
    assert second["local_write_count"] == 0
    assert legacy.read_bytes() == b"legacy-all11-sentinel\n"


@pytest.mark.parametrize(
    "failure_mode", ["partial", "tampered", "unstable", "artifact_tamper"]
)
def test_partial_tampered_or_unstable_remote_is_refused_without_writes(
    tmp_path, failure_mode
):
    plan, manifest, state = _contracts()
    task, status, _result, files = _task_and_files(plan, manifest, state)
    status_key = next(key for key in files if key[1].endswith("seed_status.json"))
    result_key = next(key for key in files if key[1].endswith("result.json"))
    unstable = ()
    if failure_mode == "partial":
        status["terminal"] = False
        files[status_key] = _bytes(status)
    elif failure_mode == "tampered":
        status["result_sha256"] = "0" * 64
        files[status_key] = _bytes(status)
    else:
        if failure_mode == "unstable":
            unstable = (result_key,)
        else:
            artifact_key = next(
                key for key in files if key[1].endswith("terminal_X.npy")
            )
            original = files[artifact_key]
            files[artifact_key] = b"X" + original[1:]
    runtime = tmp_path / "runtime"
    result = harvest.harvest_snapshot(
        plan=plan,
        manifest=manifest,
        controller_state=state,
        scheduler=Scheduler([task]),
        remote=Remote(files, unstable=unstable),
        runtime=runtime,
        payload_builder=_payload_builder,
        result_validator=_noop_validator,
    )
    assert result["refused_terminal_count"] == 1
    assert result["authenticated_seed_count"] == 0
    assert result["local_write_count"] == 0
    assert not runtime.exists()


def _observation(task_id, status_sha, result_sha, result_bytes):
    result_value = {
        "candidates": [],
        "constraint_version": "current7-test",
        "stage_spec_sha256": "a" * 64,
        "hard_constraint_contract_sha256": "b" * 64,
        "temperature_contract_sha256": "c" * 64,
    }
    return {
        "bundle_id": "current7-a",
        "seed": 101,
        "island_id": "island-a",
        "task_id": task_id,
        "scheduler_state": "completed",
        "terminal_state": "completed",
        "status_object": {"sha256": status_sha, "size": 2},
        "result_object": {"sha256": result_sha, "size": len(result_bytes)},
        "completed_generations": 201,
        "feasible_pareto_count": 0,
        "_status_bytes": b"{}",
        "_result_bytes": result_bytes,
        "artifact_objects": {},
        "_artifact_payloads": {},
        "_candidates": [],
        "_result": result_value,
    }


def test_duplicate_bundle_seed_is_idempotent_but_divergence_is_refused():
    result = b'{"result":1}\n'
    digest = hashlib.sha256(result).hexdigest()
    status_digest = hashlib.sha256(b"{}").hexdigest()
    records = harvest.deduplicate_observations(
        [
            _observation(1, status_digest, digest, result),
            _observation(2, status_digest, digest, result),
        ]
    )
    assert len(records) == 1
    assert records[0]["task_ids"] == [1, 2]
    divergent = b'{"result":2}\n'
    with pytest.raises(RuntimeError, match="divergent terminal results"):
        harvest.deduplicate_observations(
            [
                _observation(1, status_digest, digest, result),
                _observation(
                    2,
                    status_digest,
                    hashlib.sha256(divergent).hexdigest(),
                    divergent,
                ),
            ]
        )


def test_aggregate_reports_feasible_pareto_and_least_violation():
    record = {
        "bundle_id": "current7-a",
        "seed": 11,
        "_result": {
            "candidates": [
                {
                    "decoded_params": {"x": 1},
                    "volume_L": 10,
                    "total_loss_W": 100,
                    "G": [-1],
                    "feasible": True,
                },
                {
                    "decoded_params": {"x": 2},
                    "volume_L": 12,
                    "total_loss_W": 90,
                    "G": [-1],
                    "feasible": True,
                },
                {
                    "decoded_params": {"x": 3},
                    "volume_L": 12,
                    "total_loss_W": 110,
                    "G": [-1],
                    "feasible": True,
                },
                {
                    "decoded_params": {"x": 4},
                    "volume_L": 5,
                    "total_loss_W": 50,
                    "G": [0.2, -1],
                    "feasible": False,
                },
            ]
        },
    }
    aggregate = harvest.aggregate_candidates([record])
    assert aggregate["feasible_candidate_count"] == 3
    assert aggregate["pareto_count"] == 2
    assert [row["decoded_params"]["x"] for row in aggregate["pareto_candidates"]] == [
        1,
        2,
    ]
    assert aggregate["least_violation_candidate"]["decoded_params"]["x"] == 1

    record["_result"]["candidates"] = [
        {
            "decoded_params": {"x": 8},
            "volume_L": 8,
            "total_loss_W": 80,
            "G": [0.3],
            "feasible": False,
        },
        {
            "decoded_params": {"x": 9},
            "volume_L": 9,
            "total_loss_W": 70,
            "G": [0.1],
            "feasible": False,
        },
    ]
    aggregate = harvest.aggregate_candidates([record])
    assert aggregate["pareto_count"] == 0
    assert aggregate["least_violation_candidate"]["decoded_params"]["x"] == 9


def test_legacy_8010_compatibility_is_explicitly_false():
    _plan, manifest, _state = _contracts()
    receipt = harvest.legacy_8010_compatibility(manifest)
    assert receipt["monitor_generation"] == harvest.LEGACY_MONITOR_GENERATION
    assert receipt["compatible"] is False
    assert receipt["activation_allowed"] is False
    assert receipt["legacy_canonical_index_write_allowed"] is False
    assert "temperature_target_count_7_not_all11" in receipt["reasons"]


def test_snapshot_and_terminal_record_seals_are_reader_recomputable(tmp_path):
    plan, manifest, state = _contracts()
    task, _status, _result, files = _task_and_files(plan, manifest, state)
    runtime = tmp_path / "runtime"
    result = harvest.harvest_snapshot(
        plan=plan,
        manifest=manifest,
        controller_state=state,
        scheduler=Scheduler([task]),
        remote=Remote(files),
        runtime=runtime,
        apply=True,
        payload_builder=_payload_builder,
        result_validator=_noop_validator,
    )
    status = json.loads(
        (Path(result["publication"]["cohort_path"]) / "status.json").read_text()
    )
    assert status["snapshot_sha256"] == harvest.canonical_sha256(
        status["snapshot_identity"]
    )
    for record in status["terminal_results"]:
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        assert record["record_sha256"] == harvest.canonical_sha256(unsigned)
