from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_al_slurm_train as al_train


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _admitted_dataset(tmp_path: Path, revision: str = "a" * 40):
    root = tmp_path / "dataset"
    root.mkdir()
    dataset = root / "strict_al.parquet"
    dataset.write_bytes(b"strict-al-fixture")
    dataset_sha = al_train._sha256_file(dataset)
    unsigned = {
        "schema_version": al_train.strict_al.MANIFEST_SCHEMA,
        "repository": {"revision": revision},
        "goal_targets": list(al_train.TARGETS),
        "authenticated_standard_collections": [
            {
                "thermal_mesh_policy": (
                    al_train.strict_al.REQUIRED_THERMAL_MESH_POLICY
                ),
                "thermal_mesh_plan_contract_version": (
                    al_train.strict_al.REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
                ),
            }
            for _index in range(8)
        ],
        "authenticated_thermal_mesh_truth": {
            "thermal_mesh_policy": (
                al_train.strict_al.REQUIRED_THERMAL_MESH_POLICY
            ),
            "thermal_mesh_plan_contract_version": (
                al_train.strict_al.REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
            ),
            "authenticated_row_count": 8,
            "every_authenticated_row_exact_B7_v8": True,
        },
        "output_dataset": {
            "path": dataset.name,
            "sha256": dataset_sha,
            "size_bytes": dataset.stat().st_size,
            "row_count": 6159,
        },
        "retraining_admission": {
            "allowed": True,
            "reasons": [],
            "strict_new_rows": 8,
            "unique_complete_geometries": 8,
            "unique_source_tasks": 8,
            "targeted_strata": [6],
            "all_25_targets_required_per_new_row": True,
        },
        "canonical_source_mutated": False,
        "scheduler_mutation_performed": False,
        "physics_override_performed": False,
        "quality_threshold_relaxation_performed": False,
        "new_model_generation_required": True,
        "old_generation_result_mixing_allowed": False,
        "next_campaign_contract": {
            "seed_start": al_train.strict_al.NEXT_CAMPAIGN_SEED_START,
            "seed_end_inclusive": al_train.strict_al.NEXT_CAMPAIGN_SEED_END,
            "seed_count": al_train.strict_al.NEXT_CAMPAIGN_SEED_COUNT,
            "all_four_N1_strata_required": True,
            "single_dataset_sha256_required": dataset_sha,
            "single_model_generation_required": True,
            "old_generation_result_mixing_allowed": False,
        },
    }
    manifest = dict(unsigned)
    manifest["payload_sha256"] = canonical_sha256(unsigned)
    manifest_path = _write_json(root / "manifest.json", manifest)
    return manifest_path, dataset, manifest


def _task():
    unsigned = {
        "schema_version": al_train.TASK_SCHEMA,
        "campaign_id": al_train.CAMPAIGN_ID,
        "deployment_identity_sha256": "1" * 64,
        "bundle_id": "mft-goal-al-train-" + "1" * 24,
        "code": {
            "revision": "a" * 40,
            "inventory_sha256": "2" * 64,
            "source_marker_sha256": "3" * 64,
        },
        "dataset": {
            "path": "artifacts/input/strict_al.parquet",
            "sha256": "4" * 64,
            "size_bytes": 100,
            "manifest_path": "artifacts/input/manifest.json",
            "manifest_sha256": "5" * 64,
            "manifest_payload_sha256": "6" * 64,
            "row_count": 6159,
            "source_dataset_generation": "goal-al-" + "6" * 16,
        },
        "profile": {
            "path": "artifacts/code/" + al_train.PROFILE_RELATIVE,
            "sha256": "7" * 64,
            "size_bytes": 100,
            "canonical_sha256": "8" * 64,
        },
        "quality_thresholds": {
            "path": "artifacts/code/" + al_train.THRESHOLDS_RELATIVE,
            "sha256": "9" * 64,
            "size_bytes": 100,
        },
        "training": {
            "targets": list(al_train.TARGETS),
            "targets_sha256": al_train.TARGETS_SHA256,
            "model_threads": al_train.MODEL_THREADS,
            "target_workers": al_train.TARGET_WORKERS,
            "max_model_thread_budget": (
                al_train.MAX_MODEL_THREAD_BUDGET
            ),
            "hyperparameter_tuning_performed": False,
            "training_invocation_count": 1,
        },
        "resources": {
            "cpus": al_train.CPUS,
            "memory_mb": al_train.MEMORY_MB,
            "timeout_seconds": al_train.TIMEOUT_SECONDS,
            "max_workers_per_node": al_train.MAX_WORKERS_PER_NODE,
            "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1",
        },
        "next_campaign": {
            "seed_start": 2607263000,
            "seed_end_inclusive": 2607263511,
            "seed_count": 512,
        },
        "fixed_cooling_identity_sha256": "a" * 64,
        "fixed_operating_identity_sha256": "b" * 64,
        "canonical_dataset_mutation_allowed": False,
        "quality_threshold_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
        "scheduler_project_code_included": False,
    }
    return al_train._sealed(unsigned)


def test_dataset_manifest_requires_exact_admission_and_code(tmp_path):
    manifest_path, dataset, manifest = _admitted_dataset(tmp_path)
    authenticated, observed = al_train._validate_dataset_manifest(
        manifest_path,
        expected_code_revision="a" * 40,
    )
    assert observed == dataset.resolve()
    assert authenticated["retraining_admission"]["allowed"] is True

    rejected = copy.deepcopy(manifest)
    rejected.pop("payload_sha256")
    rejected["retraining_admission"]["allowed"] = False
    rejected["retraining_admission"]["reasons"] = ["too_few"]
    rejected["payload_sha256"] = canonical_sha256(rejected)
    _write_json(manifest_path, rejected)
    with pytest.raises(
        al_train.ALTrainingError,
        match="admission",
    ):
        al_train._validate_dataset_manifest(
            manifest_path,
            expected_code_revision="a" * 40,
        )


def test_dataset_manifest_rejects_revision_and_seed_interval(tmp_path):
    manifest_path, _dataset, manifest = _admitted_dataset(tmp_path)
    with pytest.raises(
        al_train.ALTrainingError,
        match="exact training code revision",
    ):
        al_train._validate_dataset_manifest(
            manifest_path,
            expected_code_revision="b" * 40,
        )

    changed = copy.deepcopy(manifest)
    changed.pop("payload_sha256")
    changed["next_campaign_contract"]["seed_start"] += 1
    changed["payload_sha256"] = canonical_sha256(changed)
    _write_json(manifest_path, changed)
    with pytest.raises(
        al_train.ALTrainingError,
        match="next-campaign",
    ):
        al_train._validate_dataset_manifest(
            manifest_path,
            expected_code_revision="a" * 40,
        )


def test_plan_seals_admitted_dataset_and_exact_training_budget(
    monkeypatch,
    tmp_path,
):
    revision = "a" * 40
    manifest_path, dataset, manifest = _admitted_dataset(
        tmp_path,
        revision=revision,
    )
    code = tmp_path / "code"
    profile = code / al_train.PROFILE_RELATIVE
    thresholds = code / al_train.THRESHOLDS_RELATIVE
    worker = code / al_train.TOOL_RELATIVE
    _write_json(profile, {"schema_version": 1})
    _write_json(thresholds, {"targets": {}})
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("# worker fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        al_train.adapter,
        "authenticate_code_root",
        lambda root, expected: {
            "path": str(root.resolve()),
            "revision": expected,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        al_train,
        "_code_sources",
        lambda _root: {
            "artifacts/code/" + al_train.TOOL_RELATIVE: worker,
            "artifacts/code/" + al_train.PROFILE_RELATIVE: profile,
            "artifacts/code/" + al_train.THRESHOLDS_RELATIVE: thresholds,
        },
    )
    plan, deployment, task = al_train.build_plan(
        dataset_manifest_path=manifest_path,
        code_root=code,
        expected_code_revision=revision,
        local_root=tmp_path / "plans",
        remote_root="/gpfs/test/al",
    )
    assert task["dataset"]["sha256"] == al_train._sha256_file(dataset)
    assert (
        task["dataset"]["manifest_payload_sha256"]
        == manifest["payload_sha256"]
    )
    assert task["training"]["targets"] == list(al_train.TARGETS)
    assert task["training"]["target_workers"] == 4
    assert task["training"]["model_threads"] == 2
    assert task["training"]["max_model_thread_budget"] == 8
    assert task["resources"]["cpus"] == 8
    assert deployment["task_payload_sha256"] == task["payload_sha256"]
    assert plan["remote_bundle"].startswith("/gpfs/test/al/")
    assert Path(plan["local_plan_dir"]).is_dir()


def test_scheduler_payload_is_exactly_one_eight_cpu_training_task():
    task = _task()
    plan = {
        "bundle_id": task["bundle_id"],
        "task_payload_sha256": task["payload_sha256"],
        "remote_bundle": (
            "/gpfs/tmp_cpu2/mft_goal_20260726/al_training/"
            + task["bundle_id"]
        ),
    }
    payload = al_train.scheduler_payload(
        plan=plan,
        task=task,
        priority=77,
    )
    assert payload["cpus"] == 8
    assert payload["memory_mb"] == 65_536
    assert payload["scheduling_profile"] == "standard"
    assert payload["gpus"] == 0
    assert payload["priority"] == 77
    assert payload["max_workers_per_node"] == 1
    assert payload["payload_json"] == task
    assert "mft_goal_al_slurm_train.py execute" in payload["command"]
    assert "SLURM_SCHEDULER_PAYLOAD_PATH" in payload["command"]
    assert payload == al_train.scheduler_payload(
        plan=plan,
        task=task,
        priority=77,
    )


def test_submit_dry_run_never_contacts_scheduler(monkeypatch, tmp_path):
    task = _task()
    plan = {
        "bundle_id": task["bundle_id"],
        "task_payload_sha256": task["payload_sha256"],
        "remote_bundle": "/gpfs/example/" + task["bundle_id"],
    }
    monkeypatch.setattr(
        al_train,
        "load_plan",
        lambda _path: (plan, {}, task),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run contacted Scheduler")

    monkeypatch.setattr(al_train.transport, "_api_json", forbidden)
    monkeypatch.setattr(al_train.transport, "_account", forbidden)
    output = al_train.submit(
        plan_path=tmp_path / "plan.json",
        apply=False,
    )
    assert output["apply"] is False
    assert output["submission"] is None
    assert output["scheduler_request"]["cpus"] == 8


def test_execute_rejects_non_eight_cpu_contract_before_training(
    monkeypatch,
    tmp_path,
):
    payload = _write_json(tmp_path / "task.json", _task())
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setenv("SLURM_SCHEDULER_TASK_CPUS", "8")
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", "123")
    with pytest.raises(
        al_train.ALTrainingError,
        match="exactly eight",
    ):
        al_train.execute(
            payload_path=payload,
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_scheduler_collection_rejects_nonterminal_task():
    task = _task()
    plan = {
        "bundle_id": task["bundle_id"],
        "task_payload_sha256": task["payload_sha256"],
        "remote_bundle": "/gpfs/example/" + task["bundle_id"],
    }
    expected = al_train.scheduler_payload(
        plan=plan,
        task=task,
        priority=100,
    )
    submission = {
        "task_id": 123,
        "priority": 100,
    }
    scheduler = {
        "id": 123,
        "task_id": 123,
        "name": expected["name"],
        "status": "running",
        "exit_code": None,
        "account_name": "r1",
        **{
            key: expected[key]
            for key in (
                "remote_cwd",
                "cpus",
                "memory_mb",
                "scheduling_profile",
                "aedt_backend",
                "gpus",
                "priority",
                "timeout_seconds",
                "dedupe_key",
                "max_workers_per_node",
                "required_capability",
                "env_profile",
            )
        },
    }
    with pytest.raises(
        al_train.ALTrainingError,
        match="completed exact",
    ):
        al_train._validated_scheduler_task(
            scheduler,
            plan=plan,
            task=task,
            submission=submission,
        )


def test_result_contract_binds_quality_exit_and_artifact_count():
    task = _task()
    generation = "registry/generations/20260725T010101-example"
    output_inventory = {
        role: {
            "path": (
                generation + "/train_report.json"
                if role == "train_report"
                else f"{role}.json"
                if role in {"candidate", "quality_status"}
                else f"{role}.log"
            ),
            "sha256": "a" * 64,
            "size_bytes": 1,
        }
        for role in (
            "candidate",
            "quality_status",
            "train_report",
            "train_stdout",
            "train_stderr",
            "quality_stdout",
            "quality_stderr",
        )
    }
    artifacts = {}
    for index in range(len(al_train.TARGETS) * 2):
        relative = f"{generation}/target-{index}/models.pkl"
        artifacts[relative] = {
            "path": relative,
            "sha256": f"{index:064x}"[-64:],
            "size_bytes": index + 1,
        }
    result = al_train._sealed(
        {
            "schema_version": al_train.RESULT_SCHEMA,
            "campaign_id": al_train.CAMPAIGN_ID,
            "bundle_id": task["bundle_id"],
            "task_payload_sha256": task["payload_sha256"],
            "scheduler_task_id": 123,
            "training_invocation_count": 1,
            "training_exit_code": 0,
            "quality_exit_code": 2,
            "quality_passed": False,
            "search_only_proposal": True,
            "generation_relative": generation,
            "documentary_generation_path": "/gpfs/example/" + generation,
            "dataset_sha256": task["dataset"]["sha256"],
            "dataset_manifest_payload_sha256": task["dataset"][
                "manifest_payload_sha256"
            ],
            "code_revision": task["code"]["revision"],
            "targets": list(al_train.TARGETS),
            "targets_sha256": al_train.TARGETS_SHA256,
            "slurm_cpu_contract": {
                "SLURM_CPUS_PER_TASK": 8,
                "SLURM_SCHEDULER_TASK_CPUS": 8,
            },
            "output_inventory": output_inventory,
            "generation_artifacts": artifacts,
            "canonical_dataset_mutated": False,
            "quality_threshold_relaxation_performed": False,
            "scheduler_mutation_performed": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    assert al_train._validate_result(
        result,
        task=task,
        scheduler_task_id=123,
    )["search_only_proposal"] is True
    broken = copy.deepcopy(result)
    broken.pop("payload_sha256")
    broken["quality_exit_code"] = 0
    broken["payload_sha256"] = canonical_sha256(broken)
    with pytest.raises(
        al_train.ALTrainingError,
        match="result contract",
    ):
        al_train._validate_result(
            broken,
            task=task,
            scheduler_task_id=123,
        )

    wrong_cpu = copy.deepcopy(result)
    wrong_cpu.pop("payload_sha256")
    wrong_cpu["slurm_cpu_contract"]["SLURM_CPUS_PER_TASK"] = 7
    wrong_cpu["payload_sha256"] = canonical_sha256(wrong_cpu)
    with pytest.raises(
        al_train.ALTrainingError,
        match="result contract",
    ):
        al_train._validate_result(
            wrong_cpu,
            task=task,
            scheduler_task_id=123,
        )
