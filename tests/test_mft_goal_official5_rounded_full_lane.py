from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from module.input_parameter_260706 import ALL_INPUT_KEYS
from regression_260707.verify import scheduler_client
from tools import mft_goal_official5_rounded_full_collector as collector
from tools import mft_goal_official5_rounded_full_prepare as prepare
from tools import mft_goal_official5_rounded_full_submit as submitter


REVISION = prepare.EXPECTED_SOURCE_SOLVER_REVISION
LIBRARY = "e" * 40
NOW = datetime(2026, 7, 26, 16, 0, tzinfo=timezone.utc)


class Lock:
    def __enter__(self) -> Lock:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _params() -> dict[str, Any]:
    return {key: 0 for key in ALL_INPUT_KEYS}


def _retained(dedupe: str = "rounded-full-dedupe") -> dict[str, Any]:
    root = "goal-fea-retained/0123456789abcdef"
    return {
        "stage": "full",
        "dedupe_key": dedupe,
        "solver_revision": REVISION,
        "library_revision": LIBRARY,
        "profile_sha256": "f" * 64,
        "parameter_digest": "0123456789abcdef",
        "relative_directory": root,
        "artifact_path": f"{root}/full_model.aedt",
        "receipt_path": f"{root}/full_model.aedt.receipt.json",
        "marker_path": f"{root}/.slurm-scheduler-preserve.json",
        "results_path": f"{root}/full_model.aedtresults",
        "results_manifest_path": (
            f"{root}/full_model.aedtresults.manifest.json"
        ),
        "retention_required": True,
        "prune_protection_required": True,
        "transport": {
            "chunk_directory": f"{root}/full_model.aedt.chunks",
        },
    }


def _payload(
    lane: prepare.StrictLane,
    dedupe: str = "rounded-full-dedupe",
) -> dict[str, Any]:
    return {
        "name": prepare.TASK_NAME,
        "project": prepare.PROJECT,
        "account_name": lane.account_name,
        "node_name": lane.node_name,
        "node_name_policy": "strict",
        "cpus": prepare.CPUS,
        "memory_mb": prepare.MEMORY_MB,
        "max_workers_per_node": prepare.MAX_WORKERS_PER_NODE,
        "timeout_seconds": prepare.SCHEDULER_TIMEOUT_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "dedupe_key": dedupe,
        "command": (
            f"git checkout {REVISION}; "
            "python tools/mft_runtime_license_snapshot.py "
            '--output-dir "$MFT_WORKDIR/.goal-license-runtime" '
            f"--solver-revision {REVISION} && "
            f"export {scheduler_client.RUNTIME_LICENSE_REFRESH_ENV}=1; "
            "timeout --signal=TERM --kill-after=300s 79200s "
            "python run_simulation_260706.py --fixed --thermal "
            "--headless --full --params cand.json; simulation_rc=$?;"
        ),
    }


def test_reviewed_profile_is_accepted_and_retains_full_bundle() -> None:
    profile = prepare._load_full_profile()  # noqa: SLF001
    retained = scheduler_client.retained_aedt_identity(
        "rounded-full-test",
        _params(),
        profile,
        REVISION,
        LIBRARY,
    )
    assert profile["param_overrides"]["full_model"] == 1
    assert profile["param_overrides"]["thermal_symmetry"] == "full"
    assert profile["param_overrides"]["round_corner"] == 1
    assert profile["param_overrides"]["corner_radius"] == 10.0
    assert profile["param_overrides"]["corner_segments"] == 4
    assert profile["fixed_boundary_contract"] == {
        **prepare.FIXED_BOUNDARY,
        "core_plate_on": 1,
        "wcp_on": 1,
    }
    assert retained["stage"] == "full"
    assert retained["artifact_path"].endswith("/full_model.aedt")
    assert retained["results_path"].endswith("/full_model.aedtresults")


def test_full_payload_audit_enforces_resources_and_retention() -> None:
    lane = prepare.StrictLane("dhj02", "n116")
    profile = prepare._load_full_profile()  # noqa: SLF001
    core = {
        "contract": production_full_contract(),
        "requested_num_cores": 16,
        "runtime_license_refresh_required": True,
    }
    retained = _retained()
    payload = prepare._inject_diagnostic_checkpoint(  # noqa: SLF001
        _payload(lane), retained, solver_revision=REVISION
    )
    prepare._validate_full_payload(  # noqa: SLF001
        payload=payload,
        retained=retained,
        core_evidence=core,
        params=_params(),
        profile=profile,
        solver_revision=REVISION,
        lane=lane,
    )
    command = payload["command"]
    assert command.index("--full --model-only") < command.index(
        "timeout --signal=TERM --kill-after=300s 79200s"
    )
    assert "MFT_ROUNDED_FULL_CHECKPOINT_CREATED " in command
    drift = copy.deepcopy(payload)
    drift["memory_mb"] = 1
    with pytest.raises(prepare.ContractError):
        prepare._validate_full_payload(  # noqa: SLF001
            payload=drift,
            retained=_retained(),
            core_evidence=core,
            params=_params(),
            profile=profile,
            solver_revision=REVISION,
            lane=lane,
        )


def test_generated_checkpoint_transport_is_atomic_and_non_promotable(
    tmp_path: Path,
) -> None:
    lane = prepare.StrictLane("dhj02", "n116")
    retained = _retained()
    payload = _payload(lane)
    checkpoint = prepare.diagnostic_checkpoint_contract(
        payload=payload,
        retained=retained,
        solver_revision=REVISION,
    )
    source = tmp_path / "source.aedt"
    source.write_bytes(b"real rounded geometry and setup")
    root = tmp_path / "goal-fea-checkpoints" / "test"
    artifact = root / "full_model_geometry_setup_checkpoint.aedt"
    receipt = root / "full_model_geometry_setup_checkpoint.receipt.json"
    marker = root / scheduler_client.SCHEDULER_PRESERVE_MARKER
    chunks = root / "full_model_geometry_setup_checkpoint.aedt.chunks"
    checkpoint = {
        **checkpoint,
        "relative_directory": "goal-fea-checkpoints/test",
        "artifact_path": (
            "goal-fea-checkpoints/test/"
            "full_model_geometry_setup_checkpoint.aedt"
        ),
        "receipt_path": (
            "goal-fea-checkpoints/test/"
            "full_model_geometry_setup_checkpoint.receipt.json"
        ),
        "marker_path": (
            "goal-fea-checkpoints/test/"
            f"{scheduler_client.SCHEDULER_PRESERVE_MARKER}"
        ),
        "chunk_directory": (
            "goal-fea-checkpoints/test/"
            "full_model_geometry_setup_checkpoint.aedt.chunks"
        ),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            prepare._checkpoint_python(),  # noqa: SLF001
            str(source),
            str(artifact),
            str(receipt),
            str(marker),
            str(chunks),
            json.dumps(
                checkpoint,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(
        "MFT_ROUNDED_FULL_CHECKPOINT_CREATED "
    )
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    assert receipt_value["scientific_pass"] is False
    assert receipt_value["thermal_pass"] is False
    assert receipt_value["production_promotion_eligible"] is False
    assert receipt_value["terminal_success_required_for_collection"] is False
    assert marker_value["schema"] == (
        scheduler_client.SCHEDULER_PRESERVE_SCHEMA
    )
    assert set(marker_value) == {
        "schema",
        "preserve",
        "created_at",
        "reason",
        "owner",
    }
    assert artifact.read_bytes() == source.read_bytes()
    assert (chunks / "00000000.b64").is_file()


def production_full_contract() -> str:
    return prepare.production.FULL_CORE_CONTRACT


def test_license_freshness_is_bounded(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    value = {
        "schema": "mft-aedt-license-headroom-snapshot-v1",
        "server_up": True,
        "server": "1055@172.16.10.81",
        "checked_at": (NOW - timedelta(seconds=30)).isoformat(),
        "features": {
            "anshpc": {"total": 900, "used": 0},
            "elec_solve_maxwell": {"total": 550, "used": 0},
            "electronics_desktop": {"total": 550, "used": 0},
            "electronics3d_gui": {"total": 550, "used": 0},
        },
    }
    snapshot.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    evidence = prepare.validate_license_freshness(
        snapshot, observed_at=NOW
    )
    assert evidence["fresh"] is True
    assert evidence["age_seconds"] == 30
    with pytest.raises(prepare.ContractError):
        prepare.validate_license_freshness(
            snapshot,
            observed_at=NOW
            + timedelta(
                seconds=prepare.MAX_LICENSE_SNAPSHOT_AGE_SECONDS + 31
            ),
        )


def test_dry_run_never_has_submission_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_plan = tmp_path / "rounded_plan.json"
    source_plan.write_text("{}", encoding="utf-8")
    license_path = tmp_path / "snapshot.json"
    license_path.write_text("{}", encoding="utf-8")
    lane = prepare.StrictLane("dhj02", "n116")
    profile = prepare._load_full_profile()  # noqa: SLF001
    payload = _payload(lane)
    retained = _retained()
    core = {
        "contract": production_full_contract(),
        "requested_num_cores": 16,
        "runtime_license_refresh_required": True,
    }
    monkeypatch.setattr(
        prepare,
        "derive_payload",
        lambda **_kwargs: (
            _params(),
            profile,
            payload,
            retained,
            core,
        ),
    )
    monkeypatch.setattr(
        prepare.rounded,
        "load_plan",
        lambda _path: {
            "solver_revision": REVISION,
            "library_revision": LIBRARY,
        },
    )
    output = tmp_path / "dry.json"
    prepare.dry_run(
        standard_plan_path=source_plan,
        lane=lane,
        license_snapshot_path=license_path,
        output=output,
        observed_at=NOW,
    )
    value = prepare.validate_seal(
        prepare.read_json(output), prepare.DRY_RUN_SCHEMA
    )
    assert value["dry_run_only"] is True
    assert value["submission_authority"] is False
    assert value["scheduler_get_calls"] == 0
    assert value["scheduler_post_calls"] == 0
    assert value["scheduler_submission_performed"] is False
    assert value["output_contract"]["required_local_collection"][
        "artifact_filename"
    ] == "full_model.aedt"


def test_standard_success_rejects_any_task_other_than_96340(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prepare.rounded_collect,
        "load_contract",
        lambda **_kwargs: {
            "task_id": 1,
            "candidate_physics_sha256": prepare.SOURCE_CANDIDATE_SHA256,
            "solver_revision": REVISION,
        },
    )
    with pytest.raises(
        prepare.ContractError, match="not task96340"
    ):
        prepare.authenticate_standard_success(
            standard_plan_path=tmp_path / "plan",
            standard_final_path=tmp_path / "final",
            standard_collection_root=tmp_path / "collection",
        )


def test_exact_once_submit_second_call_does_not_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = prepare.StrictLane("dhj02", "n116")
    payload = _payload(lane)
    plan_path = tmp_path / prepare.PLAN_NAME
    plan_path.write_text("{}", encoding="utf-8")
    plan = {
        "payload_sha256": "a" * 64,
        "scheduler_payload": payload,
        "target_lane": {
            "account_name": lane.account_name,
            "node_name": lane.node_name,
        },
    }
    license_path = tmp_path / "snapshot.json"
    license_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(prepare, "load_plan", lambda _path: plan)
    monkeypatch.setattr(
        submitter,
        "_reauthenticate_source",
        lambda *_args: {"source_node_name": "n113"},
    )
    monkeypatch.setattr(
        submitter, "_license_path", lambda _plan: license_path
    )
    monkeypatch.setattr(
        prepare,
        "validate_license_freshness",
        lambda *_args, **_kwargs: {"fresh": True},
    )
    monkeypatch.setattr(
        prepare.continuation,
        "_lane_preflight",
        lambda **_kwargs: {"separate_strict_node_lane": True},
    )
    post_calls = 0

    def post_once(_url: str, posted: dict[str, Any]) -> tuple[int, Any]:
        nonlocal post_calls
        assert posted == payload
        post_calls += 1
        return 201, {"task_id": 99_999}

    readback = {
        "task_id": 99_999,
        "name": prepare.TASK_NAME,
        "dedupe_key": payload["dedupe_key"],
        "project": prepare.PROJECT,
        "requested_account_name": lane.account_name,
        "requested_node_name": lane.node_name,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": 0,
        "requested_allocation_id": 0,
        "cpus": prepare.CPUS,
        "memory_mb": prepare.MEMORY_MB,
        "timeout_seconds": prepare.SCHEDULER_TIMEOUT_SECONDS,
        "max_workers_per_node": prepare.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    kwargs = {
        "plan_path": plan_path,
        "authorization": submitter.POST_AUTHORIZATION,
        "reader": lambda _path, _query: copy.deepcopy(readback),
        "post_once": post_once,
        "lock_factory": Lock,
        "observed_at": NOW,
    }
    receipt = submitter.submit(**kwargs)
    assert receipt.is_file()
    assert post_calls == 1
    assert submitter.submit(**kwargs) == receipt
    assert post_calls == 1


def test_full_collector_active_poll_is_get_only() -> None:
    lane = {"account_name": "dhj02", "node_name": "n116"}
    contract = {
        "task_id": 99_999,
        "task_name": prepare.TASK_NAME,
        "dedupe_key": "rounded-full-dedupe",
        "target_lane": lane,
    }
    task = {
        "task_id": 99_999,
        "name": prepare.TASK_NAME,
        "dedupe_key": "rounded-full-dedupe",
        "project": prepare.PROJECT,
        "requested_account_name": "dhj02",
        "requested_node_name": "n116",
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": 0,
        "requested_allocation_id": 0,
        "cpus": prepare.CPUS,
        "memory_mb": prepare.MEMORY_MB,
        "timeout_seconds": prepare.SCHEDULER_TIMEOUT_SECONDS,
        "max_workers_per_node": prepare.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "status": "running",
        "account_name": "dhj02",
        "actual_node_name": "n116",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
    }
    result = collector.collect(
        contract=contract,
        output=Path("unused"),
        reader=lambda _path, _query: copy.deepcopy(task),
    )
    assert result == {
        "event": "rounded_full_task_active",
        "task_id": 99_999,
        "status": "running",
        "scheduler_get_only": True,
        "scheduler_post_calls": 0,
    }


def test_timeout_checkpoint_collection_is_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b"rounded full geometry setup checkpoint"
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {
        "artifact_path": (
            "goal-fea-checkpoints/x/"
            "full_model_geometry_setup_checkpoint.aedt"
        ),
        "artifact_sha256": digest,
        "artifact_size_bytes": len(raw),
        "transport_chunk_directory": "goal-fea-checkpoints/x/chunks",
        "transport_raw_chunk_bytes": 768_000,
        "transport_max_encoded_chunk_bytes": 1_024_000,
        "transport_chunk_count": 1,
    }
    contract = {
        "task_id": 99_999,
        "task_name": prepare.TASK_NAME,
        "scheduler_url": prepare.SCHEDULER_URL,
        "solver_revision": REVISION,
        "library_revision": LIBRARY,
        "checkpoint": {
            "rounding_policy": prepare.ROUNDING_POLICY,
            "fixed_boundary": prepare.FIXED_BOUNDARY,
        },
    }
    monkeypatch.setattr(
        collector,
        "_checkpoint_metadata",
        lambda *_args, **_kwargs: (
            receipt,
            {"diagnostic_only": True},
            b"receipt",
            b"marker",
        ),
    )

    def fetcher(**kwargs: Any) -> None:
        assert kwargs["task_id"] == 99_999
        kwargs["destination"].write_bytes(raw)

    output = tmp_path / "checkpoint"
    result = collector.collect_diagnostic_checkpoint(
        contract=contract,
        output=output,
        remote_fetcher=fetcher,
    )
    collection = prepare.validate_seal(
        prepare.read_json(output / "collection_receipt.json"),
        collector.CHECKPOINT_COLLECTION_SCHEMA,
    )
    assert result["scientific_pass"] is False
    assert result["thermal_pass"] is False
    assert result["production_promotion_eligible"] is False
    assert collection["scheduler_terminal_success_required"] is False
    assert collection["scientific_result_available"] is False
    assert collection["production_package_eligible"] is False
    assert collection["diagnostic_only"] is True
    assert (
        output / "full_model_geometry_setup_checkpoint.aedt"
    ).read_bytes() == raw


def test_failed_full_task_attempts_checkpoint_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = {"account_name": "dhj02", "node_name": "n116"}
    contract = {
        "task_id": 99_999,
        "task_name": prepare.TASK_NAME,
        "dedupe_key": "rounded-full-dedupe",
        "target_lane": lane,
    }
    task = {
        "task_id": 99_999,
        "name": prepare.TASK_NAME,
        "dedupe_key": "rounded-full-dedupe",
        "project": prepare.PROJECT,
        "requested_account_name": "dhj02",
        "requested_node_name": "n116",
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": 0,
        "requested_allocation_id": 0,
        "cpus": prepare.CPUS,
        "memory_mb": prepare.MEMORY_MB,
        "timeout_seconds": prepare.SCHEDULER_TIMEOUT_SECONDS,
        "max_workers_per_node": prepare.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "status": "timed_out",
        "state": "failed",
        "exit_code": 124,
        "account_name": "dhj02",
        "actual_node_name": "n116",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
    }
    checkpoint = {
        "event": "rounded_full_checkpoint_collected",
        "scientific_pass": False,
        "thermal_pass": False,
        "production_promotion_eligible": False,
    }
    monkeypatch.setattr(
        collector,
        "collect_diagnostic_checkpoint",
        lambda **_kwargs: copy.deepcopy(checkpoint),
    )
    result = collector.collect(
        contract=contract,
        output=tmp_path / "terminal",
        reader=lambda _path, _query: copy.deepcopy(task),
    )
    assert result["diagnostic_checkpoint"] == checkpoint
    assert result["scientific_pass"] is False
    assert result["production_promotion_eligible"] is False
    failure = prepare.validate_seal(
        prepare.read_json(tmp_path / "terminal_failure.json"),
        collector.FAILURE_SCHEMA,
    )
    assert failure["diagnostic_checkpoint"] == checkpoint
    assert failure["scientific_pass"] is False
    assert failure["thermal_pass"] is False
    assert failure["production_promotion_eligible"] is False
