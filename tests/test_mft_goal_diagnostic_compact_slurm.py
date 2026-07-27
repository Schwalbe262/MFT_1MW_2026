from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_diagnostic_compact_slurm as offload


def _synthetic_tasks() -> list[dict[str, Any]]:
    activation = {
        "payload_sha256": "1" * 64,
        "compact_search_contract_sha256": "2" * 64,
        "compact_coordinate_bank_sha256": "3" * 64,
        "source_quality_passed": False,
    }
    identity = {
        "dataset_sha256": "4" * 64,
        "evaluation_model_sha256": "5" * 64,
        "quality_status_sha256": "6" * 64,
    }
    source = {
        "generation": "/source/generation",
        "candidate": "/source/candidate.json",
        "quality_status": "/source/quality.json",
        "code_root": "/source/code",
        "dataset": "/source/dataset.parquet",
        "profile": "/source/profile.json",
        "expected_code_revision": "7" * 40,
    }
    tasks = []
    for ordinal, seed in enumerate(offload.EXACT_SEEDS):
        authorization = {
            "authorization_mode": (
                offload.preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC
            ),
            "source_activation_payload_sha256": activation[
                "payload_sha256"
            ],
            "seed": seed,
            "fixed_primary_turns": offload.FIXED_PRIMARY_TURNS,
            "compact_search_contract_sha256": activation[
                "compact_search_contract_sha256"
            ],
            "compact_coordinate_bank_sha256": activation[
                "compact_coordinate_bank_sha256"
            ],
            "dataset_sha256": identity["dataset_sha256"],
            "evaluation_model_sha256": identity[
                "evaluation_model_sha256"
            ],
            "quality_status_sha256": identity["quality_status_sha256"],
            "source_quality_passed": False,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
        }
        tasks.append(
            {
                "seed": seed,
                "fixed_primary_turns": offload.FIXED_PRIMARY_TURNS,
                "activation": copy.deepcopy(activation),
                "source_identity": copy.deepcopy(identity),
                "source": copy.deepcopy(source),
                "code_manifest_payload_sha256": "8" * 64,
                "code_inventory_sha256": "9" * 64,
                "compact_run_authorization": authorization,
                "payload_sha256": canonical_sha256(
                    {"ordinal": ordinal, "seed": seed}
                ),
                "screening_only": True,
                "production_eligible": False,
                "final_design_claim_allowed": False,
                "fresh512_activation_evidence": False,
            }
        )
    return tasks


@pytest.fixture
def permissive_task_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        offload.scout,
        "_validate_task",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        offload.scout,
        "_validate_activation",
        lambda value: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        offload.preflight,
        "validate_goal_compact_run_authorization_seal",
        lambda value: copy.deepcopy(dict(value)),
    )


def test_exact32_inventory_is_diagnostic_only(
    permissive_task_validation: None,
) -> None:
    tasks, common = offload._exact_task_inventory(_synthetic_tasks())
    assert [task["seed"] for task in tasks] == list(offload.EXACT_SEEDS)
    assert common["screening_only"] is True
    assert common["production_eligible"] is False
    assert common["final_design_claim_allowed"] is False
    assert common["automatic_promotion_allowed"] is False
    assert common["fresh512_activation_evidence"] is False


@pytest.mark.parametrize(
    "mutation",
    ["count", "seed", "bank", "quality", "model", "source"],
)
def test_exact32_inventory_rejects_mixed_authority(
    permissive_task_validation: None,
    mutation: str,
) -> None:
    tasks = _synthetic_tasks()
    if mutation == "count":
        tasks.pop()
    elif mutation == "seed":
        tasks[-1]["seed"] += 1
        tasks[-1]["compact_run_authorization"]["seed"] += 1
    elif mutation == "bank":
        tasks[-1]["compact_run_authorization"][
            "compact_coordinate_bank_sha256"
        ] = "a" * 64
    elif mutation == "quality":
        tasks[-1]["source_identity"]["quality_status_sha256"] = "a" * 64
    elif mutation == "model":
        tasks[-1]["source_identity"]["evaluation_model_sha256"] = "a" * 64
    else:
        tasks[-1]["source"]["candidate"] = "/different/candidate.json"
    with pytest.raises(RuntimeError):
        offload._exact_task_inventory(tasks)


def test_inventory_rejects_unauthenticated_or_tampered_task(
    permissive_task_validation: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _synthetic_tasks()

    def reject_authorization(value: dict[str, Any]) -> dict[str, Any]:
        if value["seed"] == offload.EXACT_SEEDS[-1]:
            raise RuntimeError("unauthenticated")
        return copy.deepcopy(value)

    monkeypatch.setattr(
        offload.preflight,
        "validate_goal_compact_run_authorization_seal",
        reject_authorization,
    )
    with pytest.raises(RuntimeError, match="unauthenticated"):
        offload._exact_task_inventory(tasks)

    monkeypatch.setattr(
        offload.scout,
        "_validate_task",
        lambda value: (
            (_ for _ in ()).throw(RuntimeError("tampered"))
            if value["seed"] == offload.EXACT_SEEDS[-1]
            else copy.deepcopy(value)
        ),
    )
    with pytest.raises(RuntimeError, match="tampered"):
        offload._exact_task_inventory(tasks)


def _submission_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    bundle_id = "mft-goal-diag-compact-" + "a" * 24
    tasks = _synthetic_tasks()
    plan = {
        "bundle_id": bundle_id,
        "contract_sha256": "b" * 64,
        "diagnostic_plan_sha256": "c" * 64,
        "bundle_manifest_sha256": "d" * 64,
        "remote_bundle": f"/remote/{bundle_id}",
        "local_plan_dir": str(plan_dir),
        "scheduler_claim_root": str(plan_dir / "scheduler-claims"),
        "scheduler_priority": offload.SCHEDULER_PRIORITY,
        "relocation_sources": {
            str(seed): (
                "artifacts/campaign/relocations/"
                f"seed-{seed}-n1-{offload.FIXED_PRIMARY_TURNS}.json"
            )
            for seed in offload.EXACT_SEEDS
        },
    }
    authority = offload._initialize_claim_root(plan=plan, tasks=tasks)
    authentication = {
        "sha256": "e" * 64,
        "scheduler_claim_root_sha256": authority["sha256"],
    }
    ready = {
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "runtime_verified": True,
    }
    plan_path = tmp_path / "offload-plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        offload,
        "authenticate_plan",
        lambda _path: (plan, {}, tasks, authentication),
    )
    return plan, tasks, authentication, ready, plan_path


class FakeScheduler:
    def __init__(self, rows: dict[int, dict[str, Any]] | None = None):
        self.rows = copy.deepcopy(rows or {})
        self.post_count = 0
        self.get_count = 0
        self.next_id = max(self.rows, default=10_000) + 1

    def list_namespace_tasks(self, name_prefix: str) -> list[dict[str, Any]]:
        self.get_count += 1
        return [
            copy.deepcopy(row)
            for row in self.rows.values()
            if row["name"].startswith(name_prefix)
        ]

    def submit_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.post_count += 1
        task_id = self.next_id
        self.next_id += 1
        self.rows[task_id] = {
            "id": task_id,
            "task_id": task_id,
            "status": "queued",
            "state": "queued",
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "task_json": copy.deepcopy(payload),
        }
        return {"id": task_id, "task_id": task_id, "deduped": False}

    def get_task(self, task_id: int) -> dict[str, Any]:
        self.get_count += 1
        return copy.deepcopy(self.rows[task_id])


def test_scheduler_client_exhausts_exact_namespace_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = offload.SchedulerClient("http://scheduler.invalid")
    prefix = offload.TASK_NAME_PREFIX + "a" * 24 + "-"
    calls: list[str] = []

    def request(path: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(path)
        page = len(calls)
        task_id = page
        return {
            "filtered_total": 2,
            "page": page,
            "page_size": 10_000,
            "page_count": 2,
            "has_previous": page > 1,
            "has_next": page < 2,
            "sort_by": "id",
            "sort_order": "asc",
            "items": [
                {
                    "id": task_id,
                    "name": f"{prefix}s{task_id}",
                    "dedupe_key": f"{offload.DEDUPE_PREFIX}{task_id}",
                }
            ],
        }

    monkeypatch.setattr(client, "_request", request)
    rows = client.list_namespace_tasks(prefix)
    assert [row["id"] for row in rows] == [1, 2]
    assert len(calls) == 2


def test_scheduler_payload_binds_full_diagnostic_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, tasks, _auth, _ready, _path = _submission_fixture(
        tmp_path, monkeypatch
    )
    payload = offload.scheduler_payload(plan=plan, task=tasks[0])
    assert "mft_goal_diagnostic_compact_scout.py execute" in payload["command"]
    assert "mft_goal_20260726_slurm.py" not in payload["command"]
    assert payload["cpus"] == offload.CPUS_PER_TASK
    assert payload["gpus"] == 0
    changed = copy.deepcopy(plan)
    changed["remote_bundle"] = "/different/remote"
    assert (
        offload.scheduler_payload(plan=changed, task=tasks[0])["dedupe_key"]
        != payload["dedupe_key"]
    )
    with pytest.raises(RuntimeError, match="priority"):
        offload.scheduler_payload(plan=plan, task=tasks[0], priority=99)


def test_dry_run_and_apply_omission_never_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _tasks, _auth, ready, plan_path = _submission_fixture(
        tmp_path, monkeypatch
    )
    scheduler = FakeScheduler()
    preview = offload.submit(
        plan_path=plan_path,
        client=scheduler,
        remote_ready=ready,
    )
    assert preview["schema_version"] == offload.DRY_RUN_SCHEMA
    assert preview["absent_count"] == offload.EXACT_TASK_COUNT
    assert scheduler.post_count == 0
    assert list((Path(plan["scheduler_claim_root"]) / "claims").iterdir()) == []

    with pytest.raises(RuntimeError, match="receipt-out"):
        offload.submit(
            plan_path=plan_path,
            client=scheduler,
            remote_ready=ready,
            apply=True,
        )
    assert scheduler.post_count == 0
    assert list((Path(plan["scheduler_claim_root"]) / "claims").iterdir()) == []


def test_apply_posts_exact100_get_verifies_and_replays_immutably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _tasks, authentication, ready, plan_path = _submission_fixture(
        tmp_path, monkeypatch
    )
    receipt_path = tmp_path / "receipt.json"
    scheduler = FakeScheduler()
    receipt = offload.submit(
        plan_path=plan_path,
        client=scheduler,
        remote_ready=ready,
        receipt_out=receipt_path,
        apply=True,
    )
    assert scheduler.post_count == offload.EXACT_TASK_COUNT
    assert scheduler.get_count >= offload.EXACT_TASK_COUNT + 1
    assert receipt["first_clean_run_exact100_scheduler_posts"] is True
    assert receipt["campaign_authorized_post_count"] == offload.EXACT_TASK_COUNT
    assert receipt["automatic_promotion_allowed"] is False
    assert receipt["authentication_sha256"] == authentication["sha256"]
    claim_dirs = list(
        (Path(plan["scheduler_claim_root"]) / "claims").iterdir()
    )
    assert len(claim_dirs) == offload.EXACT_TASK_COUNT
    assert all((path / "finalized.json").is_file() for path in claim_dirs)

    replay = FakeScheduler(scheduler.rows)
    observed = offload.submit(
        plan_path=plan_path,
        client=replay,
        remote_ready=ready,
        receipt_out=receipt_path,
        apply=True,
    )
    assert observed == receipt
    assert replay.post_count == 0
    assert replay.get_count >= offload.EXACT_TASK_COUNT + 1
    with pytest.raises(RuntimeError, match="already exists"):
        offload._immutable_json(receipt_path, receipt)


def test_pending_claim_without_scheduler_task_forbids_repost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, tasks, _auth, ready, plan_path = _submission_fixture(
        tmp_path, monkeypatch
    )
    payload = offload.scheduler_payload(plan=plan, task=tasks[0])
    state, _pending, _finalized = offload._acquire_task_claim(
        plan=plan,
        tasks=tasks,
        payload=payload,
    )
    assert state == "fresh_pending"
    scheduler = FakeScheduler()
    with pytest.raises(RuntimeError, match="re-POST is forbidden"):
        offload.submit(
            plan_path=plan_path,
            client=scheduler,
            remote_ready=ready,
            receipt_out=tmp_path / "receipt.json",
            apply=True,
        )
    assert scheduler.post_count == 0


def test_receipt_rejects_contradictory_sealed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, tasks, authentication, ready, plan_path = _submission_fixture(
        tmp_path, monkeypatch
    )
    scheduler = FakeScheduler()
    receipt = offload.submit(
        plan_path=plan_path,
        client=scheduler,
        remote_ready=ready,
        receipt_out=tmp_path / "receipt.json",
        apply=True,
    )
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "sha256"
    }
    unsigned["apply"] = False
    unsigned["absent_count"] = offload.EXACT_TASK_COUNT
    unsigned["scheduler_post_count"] = 999
    tampered = offload._seal(unsigned)
    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            offload.scheduler_payload(plan=plan, task=task)
            for task in tasks
        )
    }
    with pytest.raises(RuntimeError):
        offload._validate_receipt(
            tampered,
            plan=plan,
            expected=expected,
            authentication=authentication,
            ready=ready,
            scheduler_url=offload.DEFAULT_SCHEDULER_URL,
        )


def _write_source(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "sha256": offload.adapter.sha256_file(path),
        "size": path.stat().st_size,
    }


def _authentication_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    plan_dir = tmp_path / "authenticated-plan"
    plan_dir.mkdir()
    sources_dir = tmp_path / "sources"
    records = {
        "model": _write_source(sources_dir / "model.pkl", b"model"),
        "code": _write_source(sources_dir / "tool.py", b"print('ok')\n"),
        "report": _write_source(sources_dir / "train_report.json", b"{}\n"),
        "candidate": _write_source(sources_dir / "candidate.json", b"{}\n"),
        "quality": _write_source(sources_dir / "quality.json", b"{}\n"),
        "dataset": _write_source(sources_dir / "dataset.parquet", b"dataset"),
        "profile": _write_source(sources_dir / "profile.json", b"{}\n"),
    }
    generation = "generation-a"
    generation_relative = (
        f"artifacts/g0/registry/generations/{generation}"
    )
    paths = {
        f"{generation_relative}/model.pkl": sources_dir / "model.pkl",
        f"{generation_relative}/train_report.json": (
            sources_dir / "train_report.json"
        ),
        "artifacts/code/tool.py": sources_dir / "tool.py",
        "artifacts/g0/candidate.json": sources_dir / "candidate.json",
        "artifacts/g0/quality_status.json": sources_dir / "quality.json",
        "artifacts/g0/dataset/train.parquet": sources_dir / "dataset.parquet",
        "artifacts/g0/profile.json": sources_dir / "profile.json",
    }
    kinds = {
        f"{generation_relative}/model.pkl": "model_artifact",
        f"{generation_relative}/train_report.json": "generation_report",
        "artifacts/code/tool.py": "diagnostic_code",
        "artifacts/g0/candidate.json": "generation_candidate",
        "artifacts/g0/quality_status.json": "quality_status",
        "artifacts/g0/dataset/train.parquet": "training_dataset",
        "artifacts/g0/profile.json": "training_profile",
    }
    files = {
        relative: {
            **records[
                {
                    f"{generation_relative}/model.pkl": "model",
                    f"{generation_relative}/train_report.json": "report",
                    "artifacts/code/tool.py": "code",
                    "artifacts/g0/candidate.json": "candidate",
                    "artifacts/g0/quality_status.json": "quality",
                    "artifacts/g0/dataset/train.parquet": "dataset",
                    "artifacts/g0/profile.json": "profile",
                }[relative]
            ],
            "kind": kinds[relative],
        }
        for relative in paths
    }
    identity = {
        "train_report_sha256": records["report"]["sha256"],
        "candidate_sha256": records["candidate"]["sha256"],
        "quality_status_sha256": records["quality"]["sha256"],
        "dataset_sha256": records["dataset"]["sha256"],
        "profile_sha256": "1" * 64,
        "evaluation_model_sha256": canonical_sha256(
            {"model.pkl": records["model"]["sha256"]}
        ),
        "code_revision": "2" * 40,
    }
    bundle_id = "mft-goal-diag-compact-" + "b" * 24
    remote_root = "/remote/diagnostic"
    remote_bundle = f"{remote_root}/{bundle_id}"
    relocations = {
        str(seed): (
            "artifacts/campaign/relocations/"
            f"seed-{seed}-n1-{offload.FIXED_PRIMARY_TURNS}.json"
        )
        for seed in offload.EXACT_SEEDS
    }
    code_inventory = {
        "artifacts/code/tool.py": {
            "sha256": records["code"]["sha256"],
            "size": records["code"]["size"],
        }
    }
    deployment_stable = {
        "schema_version": offload.DEPLOYMENT_SCHEMA,
        "campaign_id": offload.scout.CAMPAIGN_ID,
        "diagnostic_bundle_payload_sha256": "3" * 64,
        "activation_payload_sha256": "4" * 64,
        "code_manifest_payload_sha256": "5" * 64,
        "code_inventory_sha256": canonical_sha256(code_inventory),
        "code_revision": identity["code_revision"],
        "source_identity": identity,
        "source_paths": {
            "generation": (
                f"{remote_bundle}/{generation_relative}"
            ),
            "candidate": f"{remote_bundle}/artifacts/g0/candidate.json",
            "quality_status": (
                f"{remote_bundle}/artifacts/g0/quality_status.json"
            ),
            "code_root": f"{remote_bundle}/artifacts/code",
            "dataset": (
                f"{remote_bundle}/artifacts/g0/dataset/train.parquet"
            ),
            "profile": f"{remote_bundle}/artifacts/g0/profile.json",
            "code_manifest": (
                f"{remote_bundle}/artifacts/campaign/code_manifest.json"
            ),
        },
        "relocations": relocations,
        "files": files,
        "runtime": {},
        "execution_contract": {
            "task_count": offload.EXACT_TASK_COUNT,
            "seed_start": offload.EXACT_SEED_START,
            "seed_end_inclusive": offload.EXACT_SEEDS[-1],
            "fixed_primary_turns": offload.FIXED_PRIMARY_TURNS,
            "population": offload.scout.POPULATION,
            "generations": offload.scout.GENERATIONS,
            "inference_threads": offload.scout.INFERENCE_THREADS,
            "one_seed_per_scheduler_task": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "fresh512_activation_evidence": False,
            "aedt_used": False,
            "gpus": 0,
        },
        "source_checkout_authentication": {
            "revision": identity["code_revision"],
            "clean": True,
            "detached_head": True,
            "code_manifest_payload_sha256": "5" * 64,
            "code_inventory_sha256": canonical_sha256(code_inventory),
        },
        "clean_source_authenticated_during_prepare": True,
        "detached_source_authenticated_before_plan": True,
        "full_model_artifact_hashes_authenticated": True,
        "scheduler_project_source_included": False,
        "scheduler_service_modified": False,
    }
    deployment = {
        **deployment_stable,
        "bundle_id": bundle_id,
        "contract_sha256": canonical_sha256(deployment_stable),
    }
    tasks = _synthetic_tasks()
    plan = {
        "schema_version": offload.transport.PLAN_SCHEMA,
        "diagnostic_schema_version": offload.PLAN_SCHEMA,
        "bundle_id": bundle_id,
        "contract_sha256": deployment["contract_sha256"],
        "bundle_manifest_sha256": "6" * 64,
        "local_plan_dir": str(plan_dir),
        "remote_root": remote_root,
        "remote_bundle": remote_bundle,
        "goal_bundle_root": str(tmp_path),
        "goal_bundle_payload_sha256": "3" * 64,
        "task_count": offload.EXACT_TASK_COUNT,
        "relocation_sources": relocations,
        "scheduler_claim_root": str(plan_dir / "scheduler-claims"),
        "scheduler_priority": offload.SCHEDULER_PRIORITY,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "automatic_promotion_allowed": False,
        "fresh512_activation_evidence": False,
        "recommended_resources": {
            "cpus": offload.CPUS_PER_TASK,
            "memory_mb": offload.MEMORY_MB_PER_TASK,
            "timeout_seconds": offload.TIMEOUT_SECONDS,
            "max_workers_per_node": offload.MAX_WORKERS_PER_NODE,
            "maximum_parallel_tasks": offload.EXACT_TASK_COUNT,
        },
    }
    plan["diagnostic_plan_sha256"] = canonical_sha256(plan)
    offload._initialize_claim_root(plan=plan, tasks=tasks)
    source_map = {relative: str(path) for relative, path in paths.items()}
    bundle = {"payload_sha256": "3" * 64}
    activation = {"payload_sha256": "4" * 64}
    code_manifest = {
        "payload_sha256": "5" * 64,
        "code_revision": identity["code_revision"],
    }
    common = {
        "source_identity": identity,
        "compact_search_contract_sha256": "7" * 64,
        "compact_coordinate_bank_sha256": "8" * 64,
        "source_quality_passed": False,
    }
    monkeypatch.setattr(
        offload.transport,
        "load_plan",
        lambda _path: (plan, deployment, source_map),
    )
    monkeypatch.setattr(
        offload,
        "_bundle_tasks",
        lambda _root: (
            bundle,
            {},
            activation,
            code_manifest,
            [],
            tasks,
            common,
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    return plan_path, paths, plan


def test_authenticate_plan_rehashes_model_and_source_and_seals_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, paths, plan = _authentication_fixture(tmp_path, monkeypatch)
    _plan, _deployment, tasks, evidence = offload.authenticate_plan(plan_path)
    assert len(tasks) == offload.EXACT_TASK_COUNT
    assert evidence["full_model_artifact_hashes_reauthenticated"] is True
    assert evidence["automatic_promotion_allowed"] is False

    model_path = next(
        path for relative, path in paths.items() if relative.endswith("model.pkl")
    )
    model_path.write_bytes(b"tampered-model")
    with pytest.raises(RuntimeError, match="source changed"):
        offload.authenticate_plan(plan_path)

    model_path.write_bytes(b"model")
    original_remote = plan["remote_bundle"]
    plan["remote_bundle"] = "/tampered/remote"
    with pytest.raises(RuntimeError, match="plan authority"):
        offload.authenticate_plan(plan_path)
    plan["remote_bundle"] = original_remote
