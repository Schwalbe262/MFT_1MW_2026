from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_postdeadline_ui_updater as updater
from tools import mft_goal_local_symmetric_selection_watch as local_watch


OBSERVED = "2026-07-26T18:20:00+09:00"


def _item(item_id: str, state: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": f"title {item_id}",
        "detail": f"detail {item_id}",
        "state": state,
        "updated_at": "2026-07-26T18:00:00+09:00",
        "progress_pct": 1,
        "evidence": ["preserved"],
    }


def _status() -> dict[str, Any]:
    current = [_item(spec.card_id, "in_progress") for spec in updater.TASK_SPECS]
    current.extend(
        [
            {
                **_item("fea-handoff", "in_progress"),
                "title": (
                    "SLURM · ALLOCATION JOBS 4 · SUBMITTED 120 · "
                    "RUNNING 4 · QUEUED 0 · COLLECTIONS 0"
                ),
            },
            _item("parallel-workstreams", "in_progress"),
        ]
    )
    return {
        "schema_version": updater.STATUS_SCHEMA,
        "goal_id": "mft-goal-20260726",
        "goal": "goal",
        "owner": "Codex",
        "generated_at": "2026-07-26T18:00:00+09:00",
        "deadline_at": "2026-07-26T18:00:00+09:00",
        "original_deadline_missed": True,
        "summary": "stale lifecycle summary that must be refreshed",
        "current": current,
        "completed": [_item("completed-static", "completed")],
        "attention": [
            _item("pareto-truth-boundary", "attention"),
            _item("attention-static", "blocked"),
        ],
        "unknown_top_level": {"preserve": True},
    }


def _task(
    spec: updater.TaskSpec,
    *,
    state: str = "running",
    allocation_id: int | None = 100,
    slurm_job_id: str = "200",
    failure_message: str = "",
) -> dict[str, Any]:
    if spec.expected_allocation_id is not None:
        allocation_id = spec.expected_allocation_id
    if spec.expected_slurm_job_id is not None:
        slurm_job_id = spec.expected_slurm_job_id
    actual_node = spec.requested_node if allocation_id else ""
    return {
        "id": spec.task_id,
        "task_id": spec.task_id,
        "name": spec.task_name,
        "state": state,
        "status": state,
        "queue_state": state,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "max_workers_per_node": spec.max_workers_per_node or 8,
        "same_node_as_task_id": spec.expected_same_node_as_task_id,
        "requested_allocation_id": 0,
        "requested_node_name": spec.requested_node,
        "node_name": spec.requested_node,
        "node_name_policy": "strict",
        "actual_node_name": actual_node,
        "allocation_node_name": actual_node,
        "placement_contract_satisfied": state != "queued",
        "preferred_node_relaxed": False,
        "allocation_id": allocation_id,
        "slurm_job_id": slurm_job_id,
        "account_name": spec.requested_account or "",
        "requested_account_name": spec.requested_account or "",
        "created_at": "2026-07-26 09:00:00",
        "started_at": "2026-07-26 09:01:00" if allocation_id else None,
        "finished_at": None,
        "exit_code": None,
        "failure_message": failure_message,
    }


def _aux_task(
    spec: updater.AuxiliaryTaskSpec,
    *,
    state: str = "running",
    allocation_id: int | None = 14655,
    slurm_job_id: str = "842568",
    node: str = "n107",
    account: str = "harry261",
    failure_message: str = "",
) -> dict[str, Any]:
    actual_node = node if allocation_id else ""
    return {
        "id": spec.task_id,
        "task_id": spec.task_id,
        "name": spec.task_name,
        "state": state,
        "status": state,
        "queue_state": state,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "max_workers_per_node": spec.max_workers_per_node,
        "actual_node_name": actual_node,
        "allocation_node_name": actual_node,
        "placement_contract_satisfied": bool(actual_node),
        "allocation_id": allocation_id,
        "slurm_job_id": slurm_job_id,
        "account_name": account if allocation_id else "",
        "created_at": "2026-07-26 18:00:00",
        "started_at": "2026-07-26 18:01:00" if allocation_id else None,
        "finished_at": None,
        "exit_code": None,
        "failure_message": failure_message,
    }


def _mixed_auxiliary_tasks() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for spec in updater.AUTHORITATIVE_AUXILIARY_TASK_SPECS:
        values: dict[str, Any] = {}
        if spec.task_id == 96395:
            values = {
                "state": "failed",
                "failure_message": (
                    "RuntimeError: authenticated standard warm role has no "
                    "hard-feasible design"
                ),
            }
        elif spec.task_id == 96396:
            values = {
                "state": "cancelled",
                "node": "n111",
                "account": "r1jae262",
                "allocation_id": 14651,
                "slurm_job_id": "840787",
            }
        elif spec.task_id == 96397:
            values = {"state": "succeeded"}
        elif spec.task_id == 96414:
            values = {
                "state": "cancelled",
                "node": "n112",
                "account": "r1jae262",
                "allocation_id": 14649,
                "slurm_job_id": "840584",
            }
        elif spec.task_id == 96415:
            values = {
                "node": "n111",
                "account": "r1jae262",
                "allocation_id": 14651,
                "slurm_job_id": "840787",
            }
        elif spec.task_id == 96743:
            values = {
                "node": "n110",
                "account": "dhj02",
                "allocation_id": 14648,
                "slurm_job_id": "840585",
            }
        elif (
            spec in updater.TARGET_AXIS_TASK_SPECS
            or spec in updater.FRESH_SPLITTEMP_TASK_SPECS
        ):
            values = {
                "state": "queued",
                "allocation_id": None,
                "slurm_job_id": "",
                "node": "",
                "account": "",
            }
        result[spec.task_id] = updater._validate_auxiliary_task(
            spec,
            _aux_task(spec, **values),
        )
    return result


def _mixed_tasks() -> dict[int, dict[str, Any]]:
    specs = updater.TASK_SPECS
    tasks = {
        specs[0].task_id: updater._validate_task(
            specs[0], _task(specs[0], allocation_id=101)
        ),
        specs[1].task_id: updater._validate_task(
            specs[1],
            _task(
                specs[1],
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        ),
        specs[2].task_id: updater._validate_task(
            specs[2],
            _task(specs[2], state="succeeded", allocation_id=103),
        ),
        specs[3].task_id: updater._validate_task(
            specs[3],
            _task(
                specs[3],
                state="failed",
                allocation_id=104,
                failure_message="task timed out",
            ),
        ),
        specs[4].task_id: updater._validate_task(
            specs[4],
            _task(specs[4], allocation_id=105),
        ),
        specs[5].task_id: updater._validate_task(
            specs[5],
            _task(
                specs[5],
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        ),
    }
    for spec in specs[6:]:
        tasks[spec.task_id] = updater._validate_task(
            spec,
            _task(
                spec,
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        )
    return tasks


def _current_fast_lane_tasks() -> dict[int, dict[str, Any]]:
    tasks = {
        spec.task_id: updater._validate_task(spec, _task(spec))
        for spec in updater.TASK_SPECS
    }
    failed_spec = next(spec for spec in updater.TASK_SPECS if spec.task_id == 96337)
    tasks[96337] = updater._validate_task(
        failed_spec,
        _task(
            failed_spec,
            state="failed",
            failure_message="standalone core opt-in authentication digest mismatch",
        ),
    )
    hedge_spec = next(
        spec
        for spec in updater.TASK_SPECS
        if spec.task_id == updater.ROUNDED_TIMEOUT_HEDGE_TASK_ID
    )
    tasks[updater.ROUNDED_TIMEOUT_HEDGE_TASK_ID] = updater._validate_task(
        hedge_spec,
        _task(
            hedge_spec,
            state="queued",
            allocation_id=None,
            slurm_job_id="",
        ),
    )
    return tasks


def _reader_from(tasks: dict[int, dict[str, Any]]):
    def reader(_scheduler_url: str, task_id: int) -> dict[str, Any]:
        return copy.deepcopy(tasks[task_id])

    return reader


def test_task96324_card_reports_primary_solver_failure_before_receipt_error() -> None:
    spec = updater.TASK_SPECS[0]
    task = updater._validate_task(
        spec,
        _task(
            spec,
            state="failed",
            allocation_id=14644,
            slurm_job_id="839461",
            failure_message=(
                "ValueError: Out of range float values are not JSON compliant: nan"
            ),
        ),
    )

    card = updater._task_card(spec, task, OBSERVED)

    assert "Icepak native ThermalSetup execution error" in card["detail"]
    assert "no Fluent process or temperature result" in card["detail"]
    assert any(
        value.startswith("failure_message=ValueError:") for value in card["evidence"]
    )
    assert any(
        value.startswith("authenticated terminal root cause=Icepak native")
        for value in card["evidence"]
    )


def test_corrected_official5_fast_lane_replaces_failed_pre_em_attempt() -> None:
    failed_spec = next(spec for spec in updater.TASK_SPECS if spec.task_id == 96337)
    current_spec = next(spec for spec in updater.TASK_SPECS if spec.task_id == 96338)
    failed = updater._validate_task(
        failed_spec,
        _task(
            failed_spec,
            state="failed",
            failure_message="standalone core opt-in authentication digest mismatch",
        ),
    )
    current = updater._validate_task(current_spec, _task(current_spec))

    failed_card = updater._task_card(failed_spec, failed, OBSERVED)
    current_card = updater._task_card(current_spec, current, OBSERVED)

    assert "task96337 TERMINAL FAILED" in failed_card["title"]
    assert "Pre-EM AEDT startup failed" in failed_card["detail"]
    assert any(
        "selection_lane_effective=false" in value
        and "superseded_by_task96338=true" in value
        for value in failed_card["evidence"]
    )
    assert "task96338 RUNNING" in current_card["title"]
    assert any(
        value == "strict node placement contract / same_node_as_task_id=96332"
        for value in current_card["evidence"]
    )
    assert any(
        "selection_lane_effective=true" in value
        and "failover_for_task96332=true" in value
        for value in current_card["evidence"]
    )
    assert any(
        "0731cf22e3f78f93da943cc9102b7d96a2719bfbe1ca65e782789b2d98bc30dd" in value
        for value in current_card["evidence"]
    )


def test_rounded_final_running_card_separates_operational_submission_history() -> None:
    spec = next(
        spec for spec in updater.TASK_SPECS if spec.task_id == updater.ROUNDED_FINAL_TASK_ID
    )
    task = updater._validate_task(spec, _task(spec, state="running"))

    card = updater._task_card(spec, task, OBSERVED)

    assert "최종 rounded Standard 대칭 FEA ThermalSetup 실행 중" in card["title"]
    assert "task96340 RUNNING" in card["title"]
    assert card["title"].startswith(
        "CODEX | SUPERSEDED/NON-FINAL: PRIMARY 5T MISMATCH"
    )
    assert card["progress_pct"] == 0
    assert "scientific PASS가 없습니다" in card["detail"]
    assert any(
        "superseded rounded lifecycle=task96340 RUNNING" in value
        for value in card["evidence"]
    )
    assert any(
        "current selection/scientific eligibility=false" in value
        for value in card["evidence"]
    )
    assert any(
        "v1 operational submission only=HTTP422" in value
        and "task not created" in value
        for value in card["evidence"]
    )
    assert any(
        "v2 operational submission only=task96339 FAILED pre-solver" in value
        and "scientific failure count unchanged" in value
        for value in card["evidence"]
    )
    assert any(
        "collection_authenticated=false" in value
        and "scientific_pass_generated=false" in value
        for value in card["evidence"]
    )


def test_rounded_pipeline_keeps_cancelled_helper_outside_scientific_counts() -> None:
    card = updater._rounded_final_pipeline_card(
        _current_fast_lane_tasks(),
        OBSERVED,
    )

    assert "THERMAL RUNNING" in card["title"]
    assert "HEDGE QUEUED" in card["title"]
    assert "ARCHIVED INVALID CANDIDATE #5" in card["title"]
    assert "task96340/96342 CANCELLED" in card["title"]
    assert "NOT IN AXIS-v6" in card["title"]
    assert any(
        "task96341 CANCELLED" in value
        and "attach=false" in value
        and "solver_contact=false" in value
        and "scientific_failure=false" in value
        for value in card["evidence"]
    )
    assert any(
        f"{updater.ROUNDED_SNAPSHOT_SIZE_BYTES:,}B" in value
        and updater.ROUNDED_SNAPSHOT_SHA256 in value
        for value in card["evidence"]
    )
    assert any(
        f"commit={updater.ROUNDED_FULL_PREPARE_COMMIT}" in value
        and "POST0" in value
        for value in card["evidence"]
    )
    assert any(
        f"commit={updater.ROUNDED_PACKAGE_GATE_COMMIT}" in value
        and "package publish=false" in value
        for value in card["evidence"]
    )
    assert any(
        f"layout QA={updater.ROUNDED_DRAWING_QA_PASSED}/"
        f"{updater.ROUNDED_DRAWING_QA_PASSED} PASS" in value
        and "specification validity=false" in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_SLIDES}-slide PPTX" in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_SLIDES}-page PDF" in value
        for value in card["evidence"]
    )
    assert any(
        "drawing publication=false" in value
        and "blocked by primary 5T mismatch" in value
        and "task96340 result cannot release drawing" in value
        for value in card["evidence"]
    )
    assert any(
        "corrected verification=standard/unrounded symmetric" in value
        and "rounded verification=false" in value
        and "candidate target ETA=1-2h" in value
        for value in card["evidence"]
    )
    assert any(
        f"commit={updater.ROUNDED_BOUNDED_CORRECTION_COMMIT}" in value
        and "prepare-only" in value
        and "Scheduler POST0" in value
        and "submit=false" in value
        for value in card["evidence"]
    )
    assert any(
        "actual scientific PASS=0" in value
        and "actual production PASS=0" in value
        for value in card["evidence"]
    )


def test_authoritative_axis_v6_and_reference_baseline_are_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui_root = tmp_path / "reference-gui"
    run_root = gui_root / "simulation" / "simulation1"
    run_root.mkdir(parents=True)
    (run_root / "simulation1.aedt").write_bytes(b"aedt")
    (run_root / "convergence_matrix.txt").write_text("matrix", encoding="utf-8")
    (run_root / "convergence_cap.txt").write_text("cap", encoding="utf-8")
    (gui_root / "local_thermal_retry_stdout.log").write_text(
        'SOLVER_CORE_DISPATCH_JSON {"stage":"loss"}\n'
        'SOLVER_CORE_DISPATCH_JSON {"stage":"thermal"}\n',
        encoding="utf-8",
    )
    (gui_root / "local_thermal_retry_stderr.log").write_text("", encoding="utf-8")
    monkeypatch.setattr(updater, "_pid_exists", lambda pid: pid in {34080, 44520})

    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
        auxiliary_tasks=_mixed_auxiliary_tasks(),
        reference_gui_root=gui_root,
    )
    sync = updater.validate_status_sync(merged)
    ids = [item["id"] for item in merged["current"]]

    assert ids[:6] == [
        updater.PRIMARY_5T_RECOVERY_CARD_ID,
        updater.EXACT_N1_6_CORRECTED_GUI_FEA_CARD_ID,
        updater.EXACT_N1_6_GUI_FEA_CARD_ID,
        updater.TARGET_AXIS_CARD_ID,
        updater.REFERENCE_BASELINE_CARD_ID,
        updater.AXIS_V6_CARD_ID,
    ]
    target = merged["current"][3]
    assert "W1200/L1000 NSGA-II" in target["title"]
    assert "FINAL528" in target["title"]
    assert "FRESH OK0 RUN0 Q512 FAIL0" in target["title"]
    assert "SYM96743 RUNNING" in target["title"]
    assert any(
        "fresh task ranges=96485-96740 + 96756-97011" in value
        and "seeds=16+512=528" in value
        for value in target["evidence"]
    )
    assert any(
        "primary winding<=100C" in value
        and "secondary winding<=120C" in value
        and "core<=120C" in value
        for value in target["evidence"]
    )
    assert sum(
        value.startswith("fresh N1=") and "seeds=128" in value
        for value in target["evidence"]
    ) == 4
    axis = merged["current"][5]
    assert "SUPERSEDED HISTORICAL WRONG AXIS" in axis["title"]
    assert "RUNNING 15" in axis["title"]
    assert "SUCCEEDED 1" in axis["title"]
    assert any(
        "task96397 exit0" in value
        and "terminal rows=320" in value
        and "physical feasible=0" in value
        and "single-seed provisional only" in value
        for value in axis["evidence"]
    )
    assert any(
        "requested total=128CPU + 1TiB" in value
        and "population=320" in value
        and "generations=80" in value
        for value in axis["evidence"]
    )
    reference = merged["current"][4]
    assert "THERMAL MESH RUNNING" in reference["title"]
    assert "REMOTE96415 DIRECT RUNNING n111/j840787" in reference["title"]
    assert "LOCAL" in reference["title"]
    assert "PID44520 ACTIVE" in reference["title"]
    assert any(
        "Lm=7.682399mH" in value
        and "k=0.995902386" in value
        and "Llk=63.348112uH" in value
        for value in reference["evidence"]
    )
    assert any(
        "f_rx=6.525010kHz LIMITING" in value
        and "raw Lm=7.682399mH" in value
        and "raw historical only" in value
        for value in reference["evidence"]
    )
    assert any(
        "Lm=2.000mH by air gap" in value
        and "limiting resonance=12.788kHz" in value
        and "15kHz pass=false" in value
        for value in reference["evidence"]
    )
    assert any(
        "loss running=false" in value
        and "thermal dispatched=true" in value
        and "thermal running=true" in value
        for value in reference["evidence"]
    )
    assert sync["authoritative_task_ids"] == [
        spec.task_id for spec in updater.AUTHORITATIVE_AUXILIARY_TASK_SPECS
    ]
    assert sync["axis_v6"]["running"] == 15
    assert sync["axis_v6"]["succeeded"] == 1
    assert sync["axis_v6"]["superseded_historical"] is True
    assert sync["axis_v6"]["scientific_pass_generated"] is False
    assert sync["axis_v6"]["global_nds_generated"] is True
    assert sync["axis_v6"]["fixed_lm_global_nds_generated"] is True
    assert sync["axis_v6"]["resonance_screening_contract"]["lm_mH"] == 2.0
    assert sync["target_axis"]["width_drawing_x_max_mm"] == 1_200.0
    assert sync["target_axis"]["length_perpendicular_y_max_mm"] == 1_000.0
    assert sync["target_axis"]["submission_state"] == "submitted"
    assert sync["target_axis"]["queued"] == 528
    assert sync["target_axis"]["fresh_queued"] == 512
    assert sync["target_axis"]["expected_seed_count"] == 528
    assert sync["target_axis"]["expected_raw_terminal_rows"] == 168_960
    assert sync["target_axis"]["fresh_task_ranges"] == [
        [96485, 96740],
        [96756, 97011],
    ]
    assert sync["target_axis"]["task_ids"] == [
        *range(96416, 96432),
        *range(96485, 96741),
        *range(96756, 97012),
    ]
    assert sync["target_axis"]["temperature_gate_C"] == {
        "primary_winding_max": 100.0,
        "secondary_winding_max": 120.0,
        "core_max": 120.0,
    }
    assert sync["target_axis"]["final_symmetric_retry"]["task_id"] == 96743
    assert sync["target_axis"]["final_symmetric_retry"]["state"] == "running"
    assert len(
        json.dumps(merged, ensure_ascii=False).encode("utf-8")
    ) < 256 * 1024


def test_old_exact_n1_6_gui_fea_card_preserves_truth_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        updater,
        "_windows_process_snapshot",
        lambda: {
            updater.EXACT_N1_6_GUI_CONTROLLER_PID: (1, "python.exe"),
            updater.EXACT_N1_6_GUI_AEDT_PID: (
                updater.EXACT_N1_6_GUI_CONTROLLER_PID,
                "ansysedt.exe",
            ),
        },
    )
    monkeypatch.setattr(
        updater,
        "_pid_exists",
        lambda pid: pid
        in {
            updater.EXACT_N1_6_GUI_CONTROLLER_PID,
            updater.EXACT_N1_6_GUI_AEDT_PID,
        },
    )

    card = updater._exact_n1_6_gui_fea_card(OBSERVED)

    assert card["id"] == updater.EXACT_N1_6_GUI_FEA_CARD_ID
    assert len(card["title"]) <= 160
    assert "DIAGNOSTIC THERMAL INVALID" in card["title"]
    assert "OLD LOCAL EXACT 6/60" in card["title"]
    assert "interf153/150 WALL + 5000K" in card["title"]
    assert "NOT DESIGN FAILURE" in card["title"]
    assert "CANARY97041" not in card["title"]
    assert card["state"] == "in_progress"
    assert any(
        updater.EXACT_N1_6_GUI_GEOMETRY_SHA256 in value
        and "source task=96622" in value
        for value in card["evidence"]
    )
    assert any(
        "production_eligible=false" in value
        and "scientific_valid=false" in value
        for value in card["evidence"]
    )
    assert any(
        "unpaired interfaces=interf153,interf150" in value
        and "Rx_main_block_xn,Rx_main_block_yp" in value
        for value in card["evidence"]
    )
    assert any(
        "limiter triggered=true" in value
        and "limiter=5000 K" in value
        for value in card["evidence"]
    )


def test_corrected_exact_n1_6_gui_card_reports_native_stage_without_thermal_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "simulation" / "simulation1"
    run_root.mkdir(parents=True)
    project = run_root / "simulation1.aedt"
    project.write_bytes(b"aedt")
    (tmp_path / "corrected_gui_stdout.log").write_text(
        'SOLVER_CORE_DISPATCH_JSON {"stage":"matrix"}\n'
        "PyAEDT INFO: Solving design setup Setup1\n"
        "PyAEDT INFO: Design setup Setup1 solved correctly in 77s\n"
        'SOLVER_CORE_DISPATCH_JSON {"stage":"cap"}\n'
        "PyAEDT INFO: Solving design setup Setup1\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "mft-corrected-visible-gui-launch-receipt-v1",
        "controller_pid": updater.EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID,
        "aedt_pid": updater.EXACT_N1_6_CORRECTED_GUI_AEDT_PID,
        "grpc_port": updater.EXACT_N1_6_CORRECTED_GUI_GRPC_PORT,
        "grpc_port_listen_owner_pid": (
            updater.EXACT_N1_6_CORRECTED_GUI_AEDT_PID
        ),
        "project_path": str(project.resolve()),
        "canonical_geometry_sha256": (
            updater.EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256
        ),
        "model": "symmetric_eighth_nonrounded",
        "turns_primary": 6,
        "turns_secondary_total": 60,
        "solver_core_contract": "default-four-core-cap-v1",
        "solver_core_affinity_readback": 4,
    }
    (tmp_path / "corrected_gui_launch_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "_windows_process_snapshot",
        lambda: {
            updater.EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID: (
                1,
                "python.exe",
            ),
            updater.EXACT_N1_6_CORRECTED_GUI_AEDT_PID: (
                updater.EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID,
                "ansysedt.exe",
            ),
        },
    )
    monkeypatch.setattr(
        updater,
        "_pid_exists",
        lambda pid: pid
        in {
            updater.EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID,
            updater.EXACT_N1_6_CORRECTED_GUI_AEDT_PID,
        },
    )

    card = updater._corrected_n1_6_gui_fea_card(
        OBSERVED,
        corrected_canary_task={
            "state": "running",
            "actual_node_name": "n114",
            "slurm_job_id": "844341",
            "allocation_id": 14713,
        },
        root=tmp_path,
    )

    assert card["id"] == updater.EXACT_N1_6_CORRECTED_GUI_FEA_CARD_ID
    assert len(card["title"]) <= 160
    assert "CORRECTED VISIBLE GUI" in card["title"]
    assert "EXACT 6/60" in card["title"]
    assert "CAP NATIVE RUNNING" in card["title"]
    assert "4-CORE" in card["title"]
    assert "CANARY97041 RUNNING n114/j844341" in card["title"]
    assert card["state"] == "in_progress"
    assert any(
        "receipt_verified=true" in value
        and "gRPC port 64321" in value
        for value in card["evidence"]
    )
    assert any(
        "native solver readback=CAP RUNNING" in value
        and "completed stages=matrix" in value
        and "local solver cores=4" in value
        for value in card["evidence"]
    )
    assert any(
        "local thermal dispatch observed=false" in value
        and "Rx native preflight marker observed=false" in value
        and "temperatures=pending" in value
        for value in card["evidence"]
    )
    assert any(
        "task97041=RUNNING" in value
        and "node=n114" in value
        and "Slurm job=844341" in value
        for value in card["evidence"]
    )
    assert any(
        "scientific_valid=pending" in value
        and "production_eligible=false" in value
        for value in card["evidence"]
    )


def test_corrected_gui_stage_recognizes_direct_thermal_analyze(
    tmp_path: Path,
) -> None:
    (tmp_path / "corrected_gui_stdout.log").write_text(
        'SOLVER_CORE_DISPATCH_JSON {"stage":"matrix"}\n'
        "PyAEDT INFO: Solving design setup Setup1\n"
        "PyAEDT INFO: Design setup Setup1 solved correctly in 77s\n"
        'SOLVER_CORE_DISPATCH_JSON {"stage":"cap"}\n'
        "PyAEDT INFO: Solving design setup Setup1\n"
        "PyAEDT INFO: Design setup Setup1 solved correctly in 24s\n"
        'SOLVER_CORE_DISPATCH_JSON {"stage":"loss"}\n'
        "PyAEDT INFO: Solving design setup Setup1\n"
        "PyAEDT INFO: Design setup Setup1 solved correctly in 120s\n"
        'THERMAL_RX_INTERFACE_PREFLIGHT_JSON={"passed":true}\n'
        "PyAEDT INFO: Solving design setup ThermalSetup\n",
        encoding="utf-8",
    )

    stage = updater._corrected_n1_6_gui_solver_stage(tmp_path)

    assert stage == {
        "stage": "thermal",
        "state": "running",
        "completed_stages": ["matrix", "cap", "loss"],
        "thermal_dispatch_observed": True,
        "thermal_preflight_observed": True,
        "stdout_available": True,
    }


def test_turn_graded_cap_card_reports_parallel_pair_campaign() -> None:
    tasks = {}
    for task_id in updater.TURN_GRADED_CAP_TASK_IDS:
        candidate_index = (
            0 if task_id < 97_068 else 1 + (task_id - 97_068) // 2
        )
        active_winding = "Tx" if task_id % 2 == 0 else "Rx"
        state = "running" if task_id < 97_093 else "queued"
        tasks[task_id] = {
            "task_id": task_id,
            "candidate_index": candidate_index,
            "active_winding": active_winding,
            "state": state,
            "actual_node_name": "n109" if state == "running" else "",
            "slurm_job_id": str(800_000 + task_id) if state == "running" else "",
            "allocation_id": 14_700 if state == "running" else None,
            "exit_code": None,
        }
    submission_state = {
        "canary_receipt_payload_sha256": "a" * 64,
        "acquisition_receipt_payload_sha256": "b" * 64,
    }

    card = updater._turn_graded_cap_card(tasks, submission_state, OBSERVED)

    assert card["id"] == updater.TURN_GRADED_CAP_CARD_ID
    assert len(card["title"]) <= 160
    assert "25 GEOM/50 SOLVES" in card["title"]
    assert "RUN27 QUEUE23 OK0 FAIL0" in card["title"]
    assert "6/60 Tx97066 RUNNING Rx97067 RUNNING" in card["title"]
    assert card["state"] == "in_progress"
    assert any(
        "task97066 Tx RUNNING" in value
        and "task97067 Rx RUNNING" in value
        for value in card["evidence"]
    )
    assert any(
        "bulk task range=97068-97115" in value
        and "paired geometries=24" in value
        for value in card["evidence"]
    )
    assert any(
        "Scheduler method=GET only" in value
        and "scheduler repository modified=false" in value
        for value in card["evidence"]
    )


def test_clean_library_thermal_card_reports_parallel_replay() -> None:
    tasks = {
        task_id: {
            "task_id": task_id,
            "source_task_id": task_id - 74,
            "state": "running",
            "actual_node_name": f"n{108 + task_id % 4}",
            "slurm_job_id": str(900_000 + task_id),
            "allocation_id": 15_000 + task_id % 4,
            "exit_code": None,
        }
        for task_id in updater.CLEAN_LIBRARY_THERMAL_TASK_IDS
    }
    submission_state = {
        "submission_payload_sha256": (
            updater.CLEAN_LIBRARY_THERMAL_SUBMISSION_PAYLOAD_SHA256
        ),
        "plan_payload_sha256": (
            updater.CLEAN_LIBRARY_THERMAL_PLAN_PAYLOAD_SHA256
        ),
        "runtime_provenance": {
            "payload_sha256": "a" * 64,
            "library_hash_count": 24,
        },
    }

    card = updater._clean_library_thermal_card(
        tasks,
        submission_state,
        OBSERVED,
    )

    assert card["id"] == updater.CLEAN_LIBRARY_THERMAL_CARD_ID
    assert len(card["title"]) <= 160
    assert "CLEAN-LIB THERMAL24" in card["title"]
    assert "RUN24 QUEUE0 OK0 FAIL0" in card["title"]
    assert "PROVENANCE 24/24" in card["title"]
    assert "SCI-VALID PENDING" in card["title"]
    assert card["state"] == "in_progress"
    assert any(
        "tasks=97116-97139" in value
        and "source geometries=97042-97065" in value
        for value in card["evidence"]
    )
    assert any(
        "active requested total=192CPU+1572864MB" in value
        for value in card["evidence"]
    )
    assert any(
        "runtime library-root/hash provenance=24/24" in value
        for value in card["evidence"]
    )
    assert any(
        "authenticated thermal rows=0" in value
        and "production PASS=0" in value
        for value in card["evidence"]
    )


def test_lastmile_card_reports_parallel_submission_without_production_pass(
) -> None:
    tasks = {}
    for spec in updater.LASTMILE_TASK_SPECS:
        attaching = spec.rank <= 6
        raw = {
            "id": spec.task_id,
            "task_id": spec.task_id,
            "name": spec.task_name,
            "dedupe_key": spec.dedupe_key,
            "project": "MFT_1MW_2026v1",
            "priority": updater.LASTMILE_PRIORITY,
            "cpus": updater.LASTMILE_CPUS,
            "memory_mb": updater.LASTMILE_MEMORY_MB,
            "timeout_seconds": updater.LASTMILE_TIMEOUT_SECONDS,
            "max_workers_per_node": 1,
            "aedt_backend": "standalone",
            "scheduling_profile": "fea_bursty",
            "state": "attaching" if attaching else "queued",
            "status": "attaching" if attaching else "queued",
            "allocation_id": 14710 if attaching else None,
            "slurm_job_id": "843930" if attaching else "",
            "actual_node_name": "n107" if attaching else "",
            "placement_contract_satisfied": False,
            "created_at": "2026-07-26 22:23:00",
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "failure_message": "",
        }
        tasks[spec.task_id] = updater._validate_lastmile_task(spec, raw)

    card = updater._lastmile_acquisition_card(
        tasks,
        {"receipt": {"payload_sha256": updater.LASTMILE_RECEIPT_PAYLOAD_SHA256}},
        OBSERVED,
    )

    assert card["id"] == updater.LASTMILE_ACQUISITION_CARD_ID
    assert len(card["title"]) <= 160
    assert "8/8 SUBMITTED" in card["title"]
    assert "RUN0 ATTACH6 QUEUE2 OK0 FAIL0" in card["title"]
    assert "ACQUISITION-ONLY" in card["title"]
    assert any(
        "tasks=97033-97040" in value
        and "total=64CPU+512GiB" in value
        and "active nodes=n107" in value
        for value in card["evidence"]
    )
    assert any(
        "automatic_submission_recommended=false" in value
        and "production_eligible=false" in value
        and "actual scientific PASS=false" in value
        and "actual production PASS=false" in value
        for value in card["evidence"]
    )


def test_final528_card_keeps_empty_production_front_separate_from_diagnostics(
) -> None:
    collector = {
        "global_nds_final": True,
        "final_files_written": True,
        "successful_terminal_seed_count": 528,
        "raw_terminal_row_count": 168_960,
        "geometry_deduplicated_candidate_count": 2_800,
        "global_screening_feasible_count": 0,
        "partial_screening_pareto_count": 0,
        "partial_conditional_nonthermal_pareto_count": 0,
        "partial_minimum_violation_objective_front_count": 73,
        "fea_acquisition_candidate_count": 12,
        "payload_sha256": "a" * 64,
        "pareto_manifest_payload_sha256": "b" * 64,
    }

    card = updater._target_axis_card(
        _mixed_auxiliary_tasks(),
        observed_at=OBSERVED,
        collector_status=collector,
    )

    assert "NDS COMPLETE" in card["title"]
    assert "HARD-FEASIBLE 0" in card["title"]
    assert "PRODUCTION FRONT 0" in card["title"]
    assert "empty Front means no production candidate" in card["detail"]
    assert "diagnostic only" in card["detail"]
    assert any(
        "production hard-feasible volume-loss Pareto=0" in value
        and "minimum-violation 3-objective diagnostic Front=73" in value
        and "never feasible/production" in value
        for value in card["evidence"]
    )
    assert any(
        "collector payload sha256=" + "a" * 64 in value
        and "Pareto manifest payload sha256=" + "b" * 64 in value
        for value in card["evidence"]
    )


def test_final_collector_loader_requires_matching_sealed_pareto_manifest(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "collector_status.json"
    manifest_path = tmp_path / "global_pareto_manifest.json"
    manifest = {
        "schema_version": updater.TARGET_AXIS_PARETO_MANIFEST_SCHEMA,
        "campaign_id": updater.TARGET_AXIS_CAMPAIGN,
        "aggregate_hard_spec_sha256": updater.TARGET_AXIS_HARD_SHA256,
        "source_seed_count": 528,
        "source_raw_terminal_row_count": 168_960,
        "geometry_deduplicated_candidate_count": 2_800,
        "global_screening_feasible_count": 0,
        "global_pareto_count": 0,
        "conditional_nonthermal_pareto_count": 0,
        "minimum_violation_objective_front_count": 73,
        "fea_acquisition_candidate_count": 12,
        "global_non_dominated_sorting_complete": True,
        "screening_only": True,
        "production_eligible": False,
    }
    manifest["payload_sha256"] = updater.canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    status = {
        "schema_version": updater.TARGET_AXIS_COLLECTOR_STATE_SCHEMA,
        "campaign_id": updater.TARGET_AXIS_CAMPAIGN,
        "aggregate_hard_spec_sha256": updater.TARGET_AXIS_HARD_SHA256,
        "expected_seed_count": 528,
        "expected_legacy_seed_count": 16,
        "expected_fresh_seed_count": 512,
        "expected_raw_terminal_row_count": 168_960,
        "classification": "screening-only",
        "production_eligible": False,
        "global_nds_final": True,
        "final_files_written": True,
        "successful_terminal_seed_count": 528,
        "raw_terminal_row_count": 168_960,
        "geometry_deduplicated_candidate_count": 2_800,
        "global_screening_feasible_count": 0,
        "partial_screening_pareto_count": 0,
        "partial_conditional_nonthermal_pareto_count": 0,
        "partial_minimum_violation_objective_front_count": 73,
        "fea_acquisition_candidate_count": 12,
        "status_counts": {"completed": 528},
        "pareto_manifest_payload_sha256": manifest["payload_sha256"],
    }
    status["payload_sha256"] = updater.canonical_sha256(status)
    status_path.write_text(
        json.dumps(status, ensure_ascii=False), encoding="utf-8"
    )

    observed = updater._target_axis_collector_state(status_path)

    assert observed is not None
    assert observed["global_nds_final"] is True
    manifest["minimum_violation_objective_front_count"] = 74
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = updater.canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(updater.UpdaterError, match="Pareto manifest drifted"):
        updater._target_axis_collector_state(status_path)


def test_fresh_splittemp_task_topology_and_turn_strata_are_exact() -> None:
    specs = updater.FRESH_SPLITTEMP_TASK_SPECS

    assert len(specs) == 512
    assert [spec.task_id for spec in specs] == [
        *range(96485, 96741),
        *range(96756, 97012),
    ]
    assert [spec.seed for spec in specs] == list(
        range(2707277000, 2707277512)
    )
    assert 96743 not in {spec.task_id for spec in specs}
    assert {
        turns: sum(spec.primary_turns == turns for spec in specs)
        for turns in (5, 6, 7, 8)
    } == {5: 128, 6: 128, 7: 128, 8: 128}
    assert updater.FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_name == (
        "mft-final-sym-gap-2b2138a99445-g00860423-r1"
    )


def test_reference_terminal_marker_overrides_live_process_heuristic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui_root = tmp_path / "reference-gui"
    run_root = gui_root / "simulation" / "simulation1"
    run_root.mkdir(parents=True)
    (run_root / "simulation1.aedt").write_bytes(b"aedt")
    (run_root / "convergence_matrix.txt").write_text("matrix", encoding="utf-8")
    (run_root / "convergence_cap.txt").write_text("cap", encoding="utf-8")
    (gui_root / "local_thermal_retry_stdout.log").write_text(
        'SOLVER_CORE_DISPATCH_JSON {"stage":"thermal"}\n'
        "Solving design setup ThermalSetup\n",
        encoding="utf-8",
    )
    (gui_root / "local_thermal_retry_stderr.log").write_text("", encoding="utf-8")
    (gui_root / "thermal_failure.json").write_text(
        json.dumps(
            {
                "schema": updater.REFERENCE_THERMAL_TERMINAL_SCHEMA,
                "sealed": True,
                "terminal": True,
                "status": "failed",
                "state": "failed",
                "stage": "thermal_fluent_case_read",
                "thermal_solved": False,
                "temperature_results_available": False,
                "failure_class": "native_fluent_case_read_journal_interrupt",
                "failure_message": "no valid temperature fields",
                "retry": {
                    "status": "prepared",
                    "parameter": "thermal_rx_side_block_mesh_level",
                    "target_value": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(updater, "_pid_exists", lambda pid: pid in {34080, 44520})
    monkeypatch.setattr(
        updater,
        "_descendant_processes",
        lambda _pid: [(50000, "fluent.exe")],
    )

    local = updater._reference_local_stage(gui_root)
    card = updater._reference_baseline_card(
        _mixed_auxiliary_tasks(),
        OBSERVED,
        gui_root,
        local_stage=local,
    )

    assert local["thermal_failed"] is True
    assert local["thermal_running"] is False
    assert local["fluent_running"] is False
    assert local["fluent_pids"] == []
    assert "THERMAL FAILED" in card["title"]
    assert "RETRY PREPARED" in card["title"]
    assert "PID44520 ACTIVE" in card["title"]
    assert any(
        "terminal marker priority=true" in value
        and "native_fluent_case_read_journal_interrupt" in value
        and "thermal_rx_side_block_mesh_level->4" in value
        for value in card["evidence"]
    )

    (gui_root / "thermal_mesh_l4_retry_failure.json").write_text(
        json.dumps(
            {
                "schema": updater.REFERENCE_THERMAL_RETRY_TERMINAL_SCHEMA,
                "sealed": True,
                "terminal": True,
                "status": "failed",
                "state": "failed",
                "task_id": 96432,
                "result_valid_thermal": False,
                "temperature_results_available": False,
                "failure_class": (
                    "direct_analyze_mesh_quality_canary_"
                    "control_flow_incompatibility"
                ),
                "failure_message": (
                    "symmetry direct-analyze cannot bypass a requested "
                    "mesh-quality canary"
                ),
            }
        ),
        encoding="utf-8",
    )
    retry_failed_local = updater._reference_local_stage(gui_root)
    retry_failed_card = updater._reference_baseline_card(
        _mixed_auxiliary_tasks(),
        OBSERVED,
        gui_root,
        local_stage=retry_failed_local,
    )

    assert retry_failed_local["thermal_retry_failed"] is True
    assert "L4 RETRY CONTROL-FLOW FAILED" in retry_failed_card["title"]
    assert any(
        "retry task96432="
        "direct_analyze_mesh_quality_canary_control_flow_incompatibility"
        in value
        for value in retry_failed_card["evidence"]
    )


def test_axis_v6_raw_completion_is_historical_screening_only() -> None:
    tasks = _mixed_auxiliary_tasks()
    for spec in updater.AXIS_V6_TASK_SPECS:
        tasks[spec.task_id] = updater._validate_auxiliary_task(
            spec,
            _aux_task(spec, state="succeeded"),
        )

    card = updater._axis_v6_card(tasks, OBSERVED)

    assert "SUPERSEDED HISTORICAL WRONG AXIS" in card["title"]
    assert "AXIS-v6 16/16 SUCCEEDED" in card["title"]
    assert "NEW-AXIS FEASIBLE 0 · PF 0" in card["title"]
    assert card["progress_pct"] == 100
    assert "5,120 terminal rows" in card["detail"]
    assert "4,683 unique geometries" in card["detail"]
    assert any(
        "geometry pass raw=217" in value
        and "unique=210" in value
        and "mean resonance pass=true" in value
        for value in card["evidence"]
    )
    assert any(
        "compact thermal surrogate extrapolation invalid=true" in value
        and "reported minimum=302.67C" in value
        for value in card["evidence"]
    )
    assert any(
        "fixed-Lm2mH classification=screening-only" in value
        and "production eligible=false" in value
        for value in card["evidence"]
    )


def test_single_rounded_timeout_hedge_is_operational_only() -> None:
    spec = next(
        spec
        for spec in updater.TASK_SPECS
        if spec.task_id == updater.ROUNDED_TIMEOUT_HEDGE_TASK_ID
    )
    task = updater._validate_task(
        spec,
        _task(
            spec,
            state="queued",
            allocation_id=None,
            slurm_job_id="",
        ),
    )

    card = updater._task_card(spec, task, OBSERVED)

    assert "TIMEOUT HEDGE QUEUED · NO ALLOCATION YET" in card["title"]
    assert "task96342 QUEUED" in card["title"]
    assert "SUPERSEDED/NON-FINAL: PRIMARY 5T MISMATCH" in card["title"]
    assert "exact same rounded B5 candidate and physics" in card["detail"]
    assert "not a new design" in card["detail"]
    assert updater.ROUNDED_TIMEOUT_HEDGE_POST_AT_KST in card["detail"]
    assert "Task96340 remains lifecycle-visible" in card["detail"]
    assert "no longer authoritative for the corrected design" in card["detail"]
    assert card["progress_pct"] == 0
    assert any(
        "superseded_by_primary_5T_constraint=true" in value
        and "scientific_pass_eligible=false" in value
        for value in card["evidence"]
    )
    assert any(
        "Scheduler GET task96342 QUEUED" in value
        and "allocationnone" in value
        and "Slurmnone" in value
        for value in card["evidence"]
    )
    assert any(
        "source task96340 lifecycle retained=true" in value
        and "corrected rounded validation lane=false" in value
        and "standard/unrounded symmetric" in value
        for value in card["evidence"]
    )
    assert any(
        updater.ROUNDED_TIMEOUT_HEDGE_CANDIDATE_SHA256 in value
        and "same rounded B5 candidate=true" in value
        for value in card["evidence"]
    )
    assert any(
        updater.ROUNDED_TIMEOUT_HEDGE_PHYSICS_SHA256 in value
        and "same physics contract as task96340=true" in value
        for value in card["evidence"]
    )
    assert any(
        "single Scheduler POST=1" in value
        and "HTTP201" in value
        and updater.ROUNDED_TIMEOUT_HEDGE_POST_AT_KST in value
        and "repeat POST=false" in value
        for value in card["evidence"]
    )
    assert any(
        "account=harry261" in value
        and "requested node=n107" in value
        and "max_workers_per_node=1" in value
        and "new allocation required=true" in value
        for value in card["evidence"]
    )
    assert any(
        "authenticated GET-only collector=active" in value
        and "Scheduler methods=GET" in value
        and "collector POST calls=0" in value
        for value in card["evidence"]
    )
    assert any(
        "classification=operational timeout hedge" in value
        and "new design=false" in value
        and "scientific candidate count unchanged=true" in value
        for value in card["evidence"]
    )
    assert any(
        "actual scientific PASS=0" in value
        and "collection_authenticated=false" in value
        for value in card["evidence"]
    )


def _thermal_state(*, watcher_state: str = "running") -> dict[str, Any]:
    return updater._sealed(
        {
            "schema": updater.THERMAL_BRIDGE_STATE_SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "updated_at_utc": "2026-07-26T10:20:00+00:00",
            "heartbeat_interval_seconds": 60,
            "watcher_pid": 40984,
            "tool_path": (
                r"C:\w\mft-goal-20260726"
                r"\tools\mft_goal_corrected_thermal_transport_bridge.py"
            ),
            "tool_sha256": "a" * 64,
            "source_task_id": 96324,
            "source_task_name": updater.TASK_SPECS[0].task_name,
            "source_task_state": (
                "succeeded" if watcher_state == "collected" else "running"
            ),
            "watcher_state": watcher_state,
            "stage": (
                "thermal_artifact_materialized"
                if watcher_state == "collected"
                else "waiting_source_terminal"
            ),
            "bridge_task_id": 97001 if watcher_state == "collected" else None,
            "artifact_collected": watcher_state == "collected",
            "orchestration_root": (
                r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
                r"\postdeadline_thermal_transport_bridge_v1"
            ),
            "orchestration_root_exists": watcher_state == "collected",
            "orchestration_plan_exists": watcher_state == "collected",
            "scheduler_get_calls_total": 17,
            "scheduler_post_calls_total": 1 if watcher_state == "collected" else 0,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "failure": None,
        }
    )


def _continuation_state(
    *,
    status: str = "pending_standard_collections",
) -> dict[str, Any]:
    submitted = status == "full_submitted_pending_actual_result"
    return updater._sealed(
        {
            "schema_version": updater.STANDARD_FULL_CONTINUATION_STATE_SCHEMA,
            "observed_at_utc": "2026-07-26T10:21:00+00:00",
            "original_deadline_missed": True,
            "postdeadline": True,
            "canonical": False,
            "production_eligible": False,
            "scientific_pass_claimed": False,
            "full_result_available": False,
            "full_actual_constraints_passed": False,
            "promotion_completed": False,
            "status": status,
            "source_postsuccess_state": "postsuccess-state.json",
            "plan": {"path": "plan.json"} if submitted else None,
            "attempt_ledger": {"path": "attempt.json"} if submitted else None,
            "submission_receipt": ({"path": "receipt.json"} if submitted else None),
            "full_task_id": 97001 if submitted else None,
            "maximum_scheduler_posts": 1,
            "scheduler_post_attempts_consumed": 1 if submitted else 0,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
        }
    )


def _local_selection_state(
    *,
    status: str = "partial_measured_waiting",
    authenticated: int = 2,
    pending: int = 4,
    failures: int = 1,
    failure_task_ids: tuple[int, ...] | None = None,
    selected: dict[str, Any] | None = None,
    prepared: int = 0,
    layout: str = "v6",
) -> dict[str, Any]:
    if layout not in {"v5", "v6"}:
        raise ValueError("unsupported test layout")
    expected_task_ids = list(
        updater.STANDARD_SELECTION_TASK_IDS
        if layout == "v6"
        else updater.LEGACY_STANDARD_SELECTION_TASK_IDS
    )
    payload: dict[str, Any] = {
        "schema_version": (updater.LOCAL_SYMMETRIC_SELECTION_STATE_SCHEMA),
        "diagnostic_only": True,
        "search_only": True,
        "canonical": False,
        "production_eligible": False,
        "original_deadline_missed": True,
        "symmetric_model_primary": True,
        "prepare_only": True,
        "scheduler_methods_used": [],
        "scheduler_mutation_performed": False,
        "scheduler_submission_performed": False,
        "scheduler_cancel_performed": False,
        "scheduler_restart_performed": False,
        "automatic_full_trigger": False,
        "automatic_full_continuation": False,
        "status": status,
        "watch_complete": status
        in {
            "selected_symmetric_hard_pass",
            "local_prepare_only_batch_ready",
            "terminal_no_passing_or_small_local_correction",
        },
        "expected_task_ids": expected_task_ids,
        "authenticated_observation_count": authenticated,
        "pending_count": pending,
        "terminal_failure_count": failures,
        "selected_symmetric_result": selected,
        "finite_local_budget": {
            "current_round": 0,
            "max_rounds": 2,
            "max_candidates_per_round": 3,
            "max_candidates_total": 6,
            "candidate_count_this_round": prepared,
        },
        "terminal_failures_are_physics_observations": False,
        "pending_allows_local_candidate_generation": False,
        "full_model_started_by_watcher": False,
    }
    if layout == "v6":
        effective_statuses = {task_id: "pending" for task_id in expected_task_ids}
        authenticated_ids: list[int] = []
        if selected is not None:
            authenticated_ids.append(int(selected["task_id"]))
        authenticated_ids.extend(
            task_id for task_id in expected_task_ids if task_id not in authenticated_ids
        )
        for task_id in authenticated_ids[:authenticated]:
            effective_statuses[task_id] = "collection_ready"
        failed_ids = (
            list(failure_task_ids)
            if failure_task_ids is not None
            else [
                task_id
                for task_id in reversed(expected_task_ids)
                if effective_statuses[task_id] == "pending"
            ][:failures]
        )
        if (
            len(failed_ids) != failures
            or len(set(failed_ids)) != failures
            or any(
                task_id not in expected_task_ids
                or effective_statuses[task_id] != "pending"
                for task_id in failed_ids
            )
        ):
            raise ValueError("invalid effective failure task ids")
        for task_id in failed_ids:
            effective_statuses[task_id] = "terminal_failure"
        lanes = []
        for task_id in updater.STANDARD_SELECTION_LIFECYCLE_TASK_IDS:
            selection_effective = task_id in updater.STANDARD_SELECTION_TASK_IDS
            if task_id == 96332:
                lane_status = "pending"
            elif task_id == 96337:
                lane_status = "terminal_failure"
            else:
                lane_status = effective_statuses[task_id]
            lanes.append(
                {
                    "task_id": task_id,
                    "effective_status": lane_status,
                    "selection_effective": selection_effective,
                }
            )
        payload.update(
            {
                "lifecycle_task_ids": list(
                    updater.STANDARD_SELECTION_LIFECYCLE_TASK_IDS
                ),
                "selection_superseded_task_ids": list(
                    updater.STANDARD_SELECTION_SUPERSEDED_TASK_IDS
                ),
                "effective_lane_count": 7,
                "lifecycle_lane_count": 9,
                "lanes": lanes,
            }
        )
    return updater._sealed(payload)


def test_merge_preserves_protected_truth_and_seals_lifecycle_only() -> None:
    source = _status()
    completed = copy.deepcopy(source["completed"])
    attention = copy.deepcopy(source["attention"])
    parallel = copy.deepcopy(source["current"][-1])

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    sync = updater.validate_status_sync(merged)

    assert merged["generated_at"] == OBSERVED
    assert merged["completed"] == completed
    assert merged["attention"] == attention
    assert "18:20 KST" in merged["summary"]
    assert "running2 · queued10 · terminal2" in merged["summary"]
    assert "physical feasible0" in merged["summary"]
    assert "actual scientific PASS=0" in merged["summary"]
    assert "scientific/production PASS가 아닙니다" in merged["summary"]
    assert merged["current"][-1] != parallel
    assert merged["current"][-1]["id"] == "parallel-workstreams"
    assert (
        "RUNNING 2 · QUEUED 10 · ALLOCATION JOBS 3"
        in merged["current"][-1]["title"]
    )
    assert (
        "RISK task96328/allocation14620 FORCE 07-27 04:07:51 KST"
        in merged["current"][-1]["title"]
    )
    assert any(
        "RISK task96328 allocation14620" in value
        for value in merged["current"][-1]["evidence"]
    )
    policy = next(
        item
        for item in merged["current"]
        if item["id"] == updater.SELECTION_POLICY_CARD_ID
    )
    assert (
        policy["title"] == "DESIGN SELECTION | SYMMETRY/STANDARD PRIMARY | "
        "TERMINAL 1/7 | AUTO FULL OFF"
    )
    assert any(
        "primary candidate-selection gate=authenticated symmetric/Standard FEA" in value
        for value in policy["evidence"]
    )
    assert any(
        "task96326 lifecycle=SUCCEEDED / role=diagnostic reference only" in value
        for value in policy["evidence"]
    )
    assert any(
        "final explicit Full validation candidate cap=1" in value
        for value in policy["evidence"]
    )
    lane_evidence = next(
        value
        for value in policy["evidence"]
        if value.startswith("Standard selection lanes=7")
    )
    assert "task96333" in lane_evidence
    assert "task96329" not in lane_evidence
    assert "task96338" in lane_evidence
    assert "task96332" not in lane_evidence
    assert any(
        "effective official#8 selection lane=task96333" in value
        and "task96329 superseded but lifecycle-visible" in value
        for value in policy["evidence"]
    )
    assert any(
        "effective official#5 selection lane=task96338" in value
        and "task96332 superseded but lifecycle-visible" in value
        and "task96337 pre-EM operational failure only" in value
        for value in policy["evidence"]
    )
    assert any(
        "actual scientific PASS=0" in value and "actual production PASS=0" in value
        for value in policy["evidence"]
    )
    drawing = next(
        item
        for item in merged["current"]
        if item["id"] == updater.FINAL_DRAWING_CARD_ID
    )
    assert drawing["title"] == (
        "CODEX | DRAWING INVALID | PRIMARY 5T MISMATCH | "
        "DRAFT SUPERSEDED | FINAL RELEASE OFF"
    )
    assert drawing["state"] == "in_progress"
    assert drawing["progress_pct"] == 0
    assert any(
        "drawing validity=false" in value
        and "primary reference=5.0/1.6mm" in value
        and "draft model=1.13/4.6mm" in value
        for value in drawing["evidence"]
    )
    assert any(
        "설계도면260706.pdf pages=9" in value
        and "설계도면260706.pptx slides=9" in value
        and "960x540pt / 16:9" in value
        for value in drawing["evidence"]
    )
    assert any(
        updater.DRAWING_REFERENCE_PDF_SHA256 in value
        and "source unchanged=true" in value
        for value in drawing["evidence"]
    )
    assert any(
        updater.DRAWING_REFERENCE_PPTX_SHA256 in value
        and "source unchanged=true" in value
        for value in drawing["evidence"]
    )
    assert any(
        "round_corner=True=rounded-rectangle racetrack" in value
        and "straight spans + concentric corner arcs" in value
        and "circular coil=false" in value
        for value in drawing["evidence"]
    )
    assert any(
        f"drawing views exported={updater.ROUNDED_DRAWING_VIEW_COUNT} PNG" in value
        and updater.ROUNDED_DRAWING_VIEWS_MANIFEST_SHA256 in value
        and updater.ROUNDED_DRAWING_SUPPLEMENTAL_MANIFEST_SHA256 in value
        for value in drawing["evidence"]
    )
    assert any(
        f"drawing readiness QA={updater.ROUNDED_DRAWING_QA_PASSED}/"
        f"{updater.ROUNDED_DRAWING_QA_PASSED} layout-only PASS" in value
        and "specification validity=false" in value
        and updater.ROUNDED_DRAWING_READINESS_MANIFEST_SHA256 in value
        and updater.ROUNDED_DRAWING_QA_SHA256 in value
        for value in drawing["evidence"]
    )
    assert any(
        updater.ROUNDED_DRAWING_DRAFT_PPTX_SHA256 in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_PPTX_BYTES:,}B" in value
        for value in drawing["evidence"]
    )
    assert any(
        updater.ROUNDED_DRAWING_DRAFT_PDF_SHA256 in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_PDF_BYTES:,}B" in value
        for value in drawing["evidence"]
    )
    assert any(
        "Z: publication=false" in value
        and "final PPTX claimed=false" in value
        and "final PDF claimed=false" in value
        and "final deliverable claimed=false" in value
        for value in drawing["evidence"]
    )
    assert any(
        "task96340 input superseded=true" in value
        and "task96340 result cannot release drawing" in value
        and "corrected 5T design selection and verification required" in value
        for value in drawing["evidence"]
    )
    assert any(
        f"commit={updater.ROUNDED_BOUNDED_CORRECTION_COMMIT}" in value
        and "prepared=true" in value
        and "Scheduler POST0" in value
        and "submitted=false" in value
        and "5T eligible=false" in value
        for value in drawing["evidence"]
    )
    recovery = merged["current"][0]
    assert recovery["id"] == updater.PRIMARY_5T_RECOVERY_CARD_ID
    assert "W=x≤1200" in recovery["title"]
    assert "L=y≤1000" in recovery["title"]
    assert "회전·축교환 금지" in recovery["title"]
    assert recovery["progress_pct"] == 25
    assert any(
        "verification lane=standard/unrounded symmetric" in value
        and "rounded verification=false" in value
        for value in recovery["evidence"]
    )
    assert any(
        "superseded optimization=axis-v6 tasks96397-96412" in value
        and "wrong W<=1000,L<=1200 axis" in value
        for value in recovery["evidence"]
    )
    assert any(
        "current target optimization=W<=1200,L<=1000,H<=750" in value
        and "submission pending" in value
        for value in recovery["evidence"]
    )
    assert any(
        "air-gap tuned Lm=2.000mH" in value
        and "Ltx=Lm+Llt_phys" in value
        and "Lrx=Ltx*(N2/N1)^2" in value
        for value in recovery["evidence"]
    )
    assert merged["unknown_top_level"] == {"preserve": True}
    assert sync["allocation_jobs_active"] == 3
    assert sync["running"] == 2
    assert sync["queued"] == 10
    assert sync["submitted_total"] == 130
    assert sync["collections_preserved"] == 0
    assert sync["scheduler_methods_used"] == ["GET"]
    assert sync["scientific_pass_generated"] is False
    assert sync["managed_task_ids"] == [
        96324,
        96325,
        96326,
        96327,
        96328,
        96329,
        96330,
        96331,
        96332,
        96333,
        96337,
        96338,
        96340,
        96342,
    ]
    assert sync["operational_history"]["task96341"] == {
        "state": "cancelled",
        "attached": False,
        "started": False,
        "slurm_job_id": "",
        "solver_contact": False,
        "scientific_failure": False,
        "included_in_scientific_effective_counts": False,
    }
    assert sync["operational_history"]["task96342"] == {
        "state": "queued",
        "allocation_id": None,
        "slurm_job_id": "",
        "source_task_id": 96340,
        "same_candidate_and_physics": True,
        "scheduler_post_calls": 1,
        "collector_methods": ["GET"],
        "collector_scheduler_mutation": False,
        "operational_timeout_hedge": True,
        "new_design": False,
        "scientific_pass": False,
        "included_in_scientific_effective_counts": False,
    }

    success = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[2].card_id
    )
    failed = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[3].card_id
    )
    assert "COLLECTION/PASS PENDING" in success["title"]
    assert any(
        "collection_authenticated=false" in value for value in success["evidence"]
    )
    assert "task timed out" in failed["detail"]
    assert "과학적 infeasibility" in failed["detail"]

    official = next(
        item
        for item in merged["current"]
        if item["id"] == "postdeadline-standard-official6-96328"
    )
    assert "task96328 RUNNING" in official["title"]
    assert "FORCE-CANCEL RISK 07-27 04:07:51 KST" in official["title"]
    assert any(
        "force boundary leads by 4h01m56s" in value for value in official["evidence"]
    )
    assert any(
        "hard residual guarantee=false" in value for value in official["evidence"]
    )
    assert any(
        "42942394a873e40181f9074f2625807224239b291bcf2f4cf2ccacde50b11edc" in value
        for value in official["evidence"]
    )
    official8 = next(
        item
        for item in merged["current"]
        if item["id"] == "postdeadline-standard-official8-96329"
    )
    assert "task96329 QUEUED" in official8["title"]
    assert any(
        "5319a8a4dceb27082b91fc6badd540221298eeb9f324e2fa30e76e529313d3ec" in value
        for value in official8["evidence"]
    )
    assert any(
        "selection_lane_effective=false" in value
        and "superseded_by_task96333=true" in value
        for value in official8["evidence"]
    )
    failover = next(
        item
        for item in merged["current"]
        if item["id"] == "postdeadline-standard-official8-failover-96333"
    )
    assert "task96333 QUEUED" in failover["title"]
    assert "n111" in failover["title"]
    assert any(
        "requested account=r1jae262" in value
        and "requested node=n111" in value
        and "node policy=strict" in value
        for value in failover["evidence"]
    )
    assert any(
        "cpus8 / memory98304MB / scheduler timeout45300s" in value
        for value in failover["evidence"]
    )
    assert any(
        "selection_lane_effective=true" in value
        and "failover_for_task96329=true" in value
        for value in failover["evidence"]
    )
    assert any(
        "8990b3f339ce36f66d4ecf5fdbbd111889d55b25d860d00e3d98e6508101180c" in value
        for value in failover["evidence"]
    )
    rounded = next(
        item
        for item in merged["current"]
        if item["id"] == "final-rounded-standard-symmetric-96340"
    )
    assert "최종 rounded Standard 대칭 FEA 대기 중" in rounded["title"]
    assert "task96340 QUEUED" in rounded["title"]
    assert "v1/v2 제출 실패는 solver 이전 운영 이력" in rounded["detail"]
    assert any(
        "v1 operational submission only=HTTP422" in value
        and "scientific failure count unchanged" in value
        for value in rounded["evidence"]
    )
    assert any(
        "v2 operational submission only=task96339 FAILED pre-solver" in value
        and "no AEDT solve" in value
        for value in rounded["evidence"]
    )
    assert any(
        "v3 effective compute=task96340" in value
        and "not competing scientific candidates" in value
        for value in rounded["evidence"]
    )
    pipeline = next(
        item
        for item in merged["current"]
        if item["id"] == updater.ROUNDED_FINAL_PIPELINE_CARD_ID
    )
    assert "STANDARD QUEUED" in pipeline["title"]
    assert "HEDGE QUEUED" in pipeline["title"]
    assert "ARCHIVED INVALID CANDIDATE #5" in pipeline["title"]
    assert "task96340/96342 CANCELLED" in pipeline["title"]
    assert any(
        "task96341 CANCELLED" in value
        and "scientific_failure=false" in value
        and "excluded from scientific/effective counts" in value
        for value in pipeline["evidence"]
    )
    assert any(
        updater.ROUNDED_SNAPSHOT_SHA256 in value
        and f"{updater.ROUNDED_SNAPSHOT_SIZE_BYTES:,}B" in value
        for value in pipeline["evidence"]
    )
    assert any(
        "one-shot gate + pre-solve geometry/setup checkpoint prepared" in value
        and "Scheduler GET0 POST0" in value
        for value in pipeline["evidence"]
    )
    assert any(
        "package publish=false" in value
        and "scientific package allowed=false" in value
        for value in pipeline["evidence"]
    )
    assert any(
        f"drawing views exported={updater.ROUNDED_DRAWING_VIEW_COUNT} PNG" in value
        and "source project save=false" in value
        for value in pipeline["evidence"]
    )
    assert any(
        f"layout QA={updater.ROUNDED_DRAWING_QA_PASSED}/"
        f"{updater.ROUNDED_DRAWING_QA_PASSED} PASS" in value
        and "specification validity=false" in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_SLIDES}-slide PPTX" in value
        and f"{updater.ROUNDED_DRAWING_DRAFT_SLIDES}-page PDF" in value
        for value in pipeline["evidence"]
    )
    assert any(
        f"commit={updater.ROUNDED_BOUNDED_CORRECTION_COMMIT}" in value
        and "Scheduler POST0" in value
        and "submit=false" in value
        for value in pipeline["evidence"]
    )
    for task_id, order, node, receipt_sha in (
        (
            96330,
            1,
            "n110",
            "d71987f49b062132fdf90a358bcc819a2cf9fd2e74c76fecad04e22ea9329233",
        ),
        (
            96331,
            12,
            "n112",
            "2e207daa33165e8921816e56beb2458d1dcd8515d6cb0b8d5a25cb0cf3cde532",
        ),
        (
            96332,
            5,
            "n115",
            "4275d9e8e004f3f9f57d9483ca9e7b8a48d410778f39a80848c95c396ed2171c",
        ),
    ):
        card = next(
            item
            for item in merged["current"]
            if item["id"] == f"postdeadline-standard-official{order}-{task_id}"
        )
        assert f"task{task_id} QUEUED" in card["title"]
        assert node in card["title"]
        assert any("search_only=true" in value for value in card["evidence"])
        assert any(receipt_sha in value for value in card["evidence"])

    handoff = next(item for item in merged["current"] if item["id"] == "fea-handoff")
    assert (
        handoff["title"] == "SLURM · ALLOCATION JOBS 3 · SUBMITTED 130 · "
        "RUNNING 2 · QUEUED 10 · COLLECTIONS 0"
    )
    for item in merged["current"]:
        assert len(item["title"]) <= 160
        assert len(item["detail"]) <= 1_200
        assert len(item["evidence"]) <= 12
        assert all(len(value) <= 500 for value in item["evidence"])


def test_merge_upserts_missing_official_task_card_before_parallel() -> None:
    source = _status()
    source["current"] = [
        item
        for item in source["current"]
        if item["id"] != "postdeadline-standard-official6-96328"
    ]

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged["current"]]

    assert ids.count("postdeadline-standard-official6-96328") == 1
    assert ids.index("postdeadline-standard-official6-96328") < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged)


def test_final_drawing_card_is_unique_and_before_parallel() -> None:
    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    merged_again = updater.merge_status(
        merged,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged_again["current"]]

    assert ids.count(updater.FINAL_DRAWING_CARD_ID) == 1
    assert ids.index(updater.FINAL_DRAWING_CARD_ID) < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged_again)


def test_compact_design_status_card_is_truthful_and_fail_closed() -> None:
    card = updater._compact_design_status_card(OBSERVED)

    assert card["id"] == updater.COMPACT_DESIGN_STATUS_CARD_ID
    assert len(card["title"]) <= 160
    assert card["state"] == "in_progress"
    assert card["progress_pct"] == 45
    assert len(card["evidence"]) <= 12
    assert "FIXED-LM PATH TEST71 PASS" in card["title"]
    assert "AUDIT FIX2 PENDING" in card["title"]
    assert "FRESH512/SCOUT POST0" in card["title"]
    assert "not proof that a compact design is impossible" in card["detail"]
    assert "authenticated compact activation token" in card["detail"]
    assert "fixed-Lm wrapper must not alter legacy/nonreserved runs" in card[
        "detail"
    ]
    assert any(
        "commit bc63ec5" in item and "tests 71 passed" in item
        for item in card["evidence"]
    )
    assert any(
        "code audit blockers=2" in item
        and "Runner activation binding missing" in item
        and "scope leak" in item
        and "Scheduler POST0" in item
        for item in card["evidence"]
    )
    assert any(
        "Ltx(0.002H+Llt_phys)" in item
        and "Lrx=Ltx*(N2/N1)^2" in item
        and "air-gap synthesis still required" in item
        for item in card["evidence"]
    )
    assert any(
        "A=W1160..1170,L975..1000" in item
        and "B=W1170..1200,L960..975" in item
        and "C=W1160..1170,L960..975" in item
        for item in card["evidence"]
    )
    assert any(
        "N1=6/7/8 exact A/B/C/H" in item
        and "N1=5 exact A/B/H" in item
        for item in card["evidence"]
    )
    assert any(
        "every 5 generations" in item
        and "invalid_or_near_feasible_fallback=false" in item
        for item in card["evidence"]
    )
    assert any(
        "strict B7/v8 dataset" in item
        and "current real activation closed" in item
        and "scout also blocked" in item
        for item in card["evidence"]
    )
    assert any(
        "commit ccdaa7d" in item and "trigger_allowed=false" in item
        for item in card["evidence"]
    )
    assert any(
        "slice=1454 geometries" in item
        and "W<=1170mm OR L<=975mm" in item
        and "volume<830.95994977L" in item
        for item in card["evidence"]
    )
    assert any(
        "hard-feasible=0" in item
        and "proof_of_impossibility=false" in item
        and "audit-only" in item
        for item in card["evidence"]
    )
    assert any(
        "axis policy=no_axis_swap" in item
        and "all hard constraints retained" in item
        for item in card["evidence"]
    )


def test_merge_upserts_one_compact_design_card_before_parallel() -> None:
    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    merged_again = updater.merge_status(
        merged,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged_again["current"]]

    assert ids.count(updater.COMPACT_DESIGN_STATUS_CARD_ID) == 1
    assert ids.index(updater.COMPACT_DESIGN_STATUS_CARD_ID) < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged_again)


def _write_rx_main_l5_canary_evidence(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fixed_cooling = copy.deepcopy(
        updater.RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING
    )
    params = {
        **fixed_cooling,
        "thermal_max_iterations": 1,
    }
    payload = {
        "name": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_NAME,
        "dedupe_key": "sealed-rx-main-l5-canary",
        "command": (
            "run exact B7/v8 native earliest canary "
            f"{updater.RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256}"
        ),
        "cpus": updater.RX_MAIN_L5_NATIVE_CANARY_CPUS,
        "memory_mb": updater.RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB,
        "timeout_seconds": updater.RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS,
        "max_workers_per_node": 1,
    }
    params_path = root / "params.json"
    payload_path = root / "dry_run_payload.json"
    params_path.write_text(
        json.dumps(params, indent=2) + "\n",
        encoding="utf-8",
    )
    payload_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    params_file_sha256 = updater._file_sha256(params_path)
    payload_file_sha256 = updater._file_sha256(payload_path)
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256",
        params_file_sha256,
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_PAYLOAD_SHA256",
        payload_file_sha256,
    )
    promotion_gate = {
        "design_scientific_promotion_forbidden_for_one_iteration_canary": True,
        "interface_fix_canary_requires": {
            "contract": "thermal-rx-block-interface-coverage-v1",
            "mesh_plan_contract": "thermal-mesh-plan-v8",
            "mesh_policy": (
                "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
            ),
            "missing_fluid_coupling": [],
            "missing_rx_main_solids": [],
            "no_unpaired_log_marker": True,
            "rx_main_adjacency_passed": True,
            "unpaired_interfaces": [],
        },
    }
    command_sha256 = updater.hashlib.sha256(
        payload["command"].encode("utf-8")
    ).hexdigest()
    identity = {
        "task_name": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_NAME,
        "geometry_sha256": updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256,
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
    }
    pre_submit = {
        "schema": (
            "mft-goal-rx-main-l5-native-earliest-canary-pre-submit-v1"
        ),
        **identity,
        "dedupe_key": payload["dedupe_key"],
        "scheduler_post_calls": 0,
        "fixed_cooling": fixed_cooling,
        "payload_file_sha256": payload_file_sha256,
        "params_file_sha256": params_file_sha256,
        "payload_canonical_sha256": updater.canonical_sha256(payload),
        "params_canonical_sha256": updater.canonical_sha256(params),
        "command_sha256": command_sha256,
        "promotion_gate": promotion_gate,
    }
    pre_submit_path = root / "pre_submit_receipt.json"
    pre_submit_path.write_text(
        json.dumps(pre_submit, indent=2) + "\n",
        encoding="utf-8",
    )
    pre_submit_file_sha256 = updater._file_sha256(pre_submit_path)
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_PRE_SUBMIT_SHA256",
        pre_submit_file_sha256,
    )
    receipt = {
        "schema": (
            "mft-goal-rx-main-l5-native-earliest-canary-submission-v1"
        ),
        **identity,
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_ID,
        "dedupe_key": payload["dedupe_key"],
        "submission_source": "post_created",
        "scheduler_post_calls": 1,
        "scheduler_mutation_performed": True,
        "existing_tasks_or_gui_mutated": False,
        "thermal_max_iterations": 1,
        "fixed_cooling": fixed_cooling,
        "interface_contract": "thermal-rx-block-interface-coverage-v1",
        "mesh_plan_contract": "thermal-mesh-plan-v8",
        "mesh_policy": (
            "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        ),
        "pre_submit_receipt_file_sha256": pre_submit_file_sha256,
        "payload_canonical_sha256": updater.canonical_sha256(payload),
        "params_canonical_sha256": updater.canonical_sha256(params),
        "command_sha256": command_sha256,
        "promotion_gate": promotion_gate,
        "task_readback_sha256": "c" * 64,
        "status": "queued",
    }
    receipt_path = root / "submission_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_RECEIPT_SHA256",
        updater._file_sha256(receipt_path),
    )
    failure_message = (
        "RuntimeError: AEDT desktop startup failed after 3 attempts: "
        "RuntimeError: standalone core opt-in authentication digest mismatch"
    )
    monitor = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA,
        "scheduler_mutation_performed": False,
        "updated_at_utc": OBSERVED,
        "task": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_ID,
            "name": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_NAME,
            "status": "failed",
            "exit_code": 1,
            "failure_message": failure_message,
            "actual_node_name": "n113",
            "slurm_job_id": "845126",
        },
    }
    monitor["payload_sha256"] = updater._rx_main_l5_monitor_payload_sha256(
        monitor
    )
    monitor_path = root / "monitor_state.json"
    monitor_path.write_text(json.dumps(monitor), encoding="utf-8")
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SHA256",
        updater._file_sha256(monitor_path),
    )
    gate = {
        "schema": "mft-goal-rx-main-l5-terminal-gate-v1",
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_ID,
        "task_status": "failed",
        "scheduler_mutation_performed": False,
        "physical_geometry_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        ),
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
        "interface_fix_canary_passed": False,
        "scientific_design_promotion_passed": False,
        "thermal_result_scientific_valid": False,
        "one_iteration_canary_not_a_design_temperature_result": True,
        "result_json_sha256": "",
        "coverage": None,
        "preflight": None,
        "checks": {
            "terminal_status": True,
            "result_json_present": False,
            "coverage_json_present": False,
        },
        "strict_result_fields": {
            "thermal_result_scientific_valid": None,
            "thermal_rx_main_interface_coverage_passed": None,
        },
    }
    gate["payload_sha256"] = updater._rx_main_l5_monitor_payload_sha256(gate)
    gate_path = root / "terminal_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_SHA256",
        updater._file_sha256(gate_path),
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_PAYLOAD_SHA256",
        gate["payload_sha256"],
    )
    predecessor = {
        "id": updater.RX_MAIN_L5_NATIVE_CANARY_TASK_ID,
        "status": "failed",
        "exit_code": 1,
        "classification": (
            "operational_pre_solver_core_auth_failure_no_scientific_result"
        ),
        "terminal_gate_payload_sha256": gate["payload_sha256"],
    }
    successor_root = root / "successor"
    successor_root.mkdir()
    successor_params_path = successor_root / "params.json"
    successor_params_path.write_text(
        json.dumps(params, indent=2) + "\n",
        encoding="utf-8",
    )
    successor_payload = {
        "name": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
        "command": (
            "run corrected B7/v8 native earliest canary "
            f"{updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256}"
        ),
        "cpus": updater.RX_MAIN_L5_NATIVE_CANARY_CPUS,
        "memory_mb": updater.RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB,
        "timeout_seconds": updater.RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS,
    }
    successor_payload_path = successor_root / "dry_run_payload.json"
    successor_payload_path.write_text(
        json.dumps(successor_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_PAYLOAD_SHA256",
        updater._file_sha256(successor_payload_path),
    )
    unchanged = {
        "cpus": updater.RX_MAIN_L5_NATIVE_CANARY_CPUS,
        "memory_mb": updater.RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB,
        "timeout_seconds": updater.RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS,
        "thermal_max_iterations": 1,
        "fixed_cooling": fixed_cooling,
        "mesh_plan_contract": "thermal-mesh-plan-v8",
        "mesh_policy": (
            "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        ),
    }
    lineage = {
        "schema": "mft-goal-rx-main-l5-corrected-successor-lineage-v1",
        "scheduler_post_calls": 0,
        "computed_core_auth_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
        ),
        "geometry_sha256": updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256,
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
        "successor_task_name": (
            updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME
        ),
        "predecessor": predecessor,
        "checks": {"all_lineage_checks": True},
        "unchanged_identity": unchanged,
    }
    lineage_path = successor_root / "lineage_pre_submit.json"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256",
        updater._file_sha256(lineage_path),
    )
    successor_receipt = {
        "schema": "mft-goal-rx-main-l5-corrected-successor-submission-v1",
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
        "task_name": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
        "submission_source": "post_created",
        "scheduler_post_calls": 1,
        "scheduler_mutation_performed": True,
        "existing_tasks_or_gui_mutated": False,
        "computed_core_auth_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
        ),
        "lineage_pre_submit_file_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256
        ),
        "geometry_sha256": updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256,
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
        "fixed_cooling": fixed_cooling,
        "thermal_max_iterations": 1,
        "predecessor": predecessor,
        "status": "queued",
    }
    successor_receipt_path = successor_root / "submission_receipt.json"
    successor_receipt_path.write_text(
        json.dumps(successor_receipt),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_RECEIPT_SHA256",
        updater._file_sha256(successor_receipt_path),
    )
    return successor_root


def _write_rx_main_l5_operational_chain_evidence(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    successor_root = _write_rx_main_l5_canary_evidence(root, monkeypatch)
    successor_monitor = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA,
        "scheduler_mutation_performed": False,
        "updated_at_utc": OBSERVED,
        "task": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
            "name": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
            "status": "failed",
            "exit_code": 1,
            "failure_message": "result extraction failed",
            "actual_node_name": "n110",
            "slurm_job_id": updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID,
            "started_at": "2026-07-27 02:19:33",
            "finished_at": "2026-07-27 02:42:57",
        },
    }
    successor_monitor["payload_sha256"] = (
        updater._rx_main_l5_monitor_payload_sha256(successor_monitor)
    )
    successor_process = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_PROCESS_SCHEMA,
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
        "allowed_http_method": "GET",
        "scheduler_mutation_performed": False,
        "post_cancel_retry_forbidden": True,
    }
    successor_process["payload_sha256"] = (
        updater._rx_main_l5_monitor_payload_sha256(successor_process)
    )
    (successor_root / "monitor_state.json").write_text(
        json.dumps(successor_monitor),
        encoding="utf-8",
    )
    (successor_root / "monitor_process.json").write_text(
        json.dumps(successor_process),
        encoding="utf-8",
    )
    stdout_sha256 = "1" * 64
    stderr_sha256 = "2" * 64
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDOUT_SHA256",
        stdout_sha256,
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDERR_SHA256",
        stderr_sha256,
    )
    terminal_gate = {
        "schema": "mft-goal-rx-main-l5-terminal-gate-v1",
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
        "task_status": "failed",
        "scheduler_mutation_performed": False,
        "physical_geometry_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        ),
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
        "slurm_job_id": updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "interface_fix_canary_passed": False,
        "scientific_design_promotion_passed": False,
        "thermal_result_scientific_valid": False,
        "one_iteration_canary_not_a_design_temperature_result": True,
        "result_json_sha256": "",
        "coverage": None,
        "preflight": None,
        "checks": {
            "terminal_status": True,
            "result_json_present": False,
            "coverage_json_present": False,
            "no_unpaired_log_marker": True,
        },
        "strict_result_fields": {
            "thermal_result_scientific_valid": None,
            "thermal_rx_main_interface_coverage_passed": None,
        },
    }
    terminal_gate["payload_sha256"] = (
        updater._rx_main_l5_monitor_payload_sha256(terminal_gate)
    )
    terminal_gate_path = successor_root / "terminal_gate.json"
    terminal_gate_path.write_text(
        json.dumps(terminal_gate),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_PAYLOAD_SHA256",
        terminal_gate["payload_sha256"],
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_GATE_SHA256",
        updater._file_sha256(terminal_gate_path),
    )

    hedge_root = root / "hedge"
    hedge_root.mkdir()
    params = {
        **copy.deepcopy(updater.RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING),
        "thermal_max_iterations": 1,
    }
    hedge_params_path = hedge_root / "params.json"
    hedge_params_path.write_text(json.dumps(params, indent=2) + "\n")
    payload = {
        "name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME,
        "command": (
            "MFT_EXCLUSIVE_PLACEMENT_FAIL; "
            "MFT_EXCLUSIVE_PLACEMENT_JSON; exit 86; "
            f"{updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256}"
        ),
        "cpus": updater.RX_MAIN_L5_NATIVE_CANARY_CPUS,
        "memory_mb": updater.RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB,
        "timeout_seconds": updater.RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS,
        "node_name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE,
        "node_name_policy": "strict",
        "exclusive_node": True,
        "max_workers_per_node": 1,
    }
    payload_path = hedge_root / "dry_run_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_HEDGE_PAYLOAD_SHA256",
        updater._file_sha256(payload_path),
    )
    pre_submit = {
        "schema": "mft-goal-rx-main-l5-isolated-hedge-pre-submit-v1",
        "scheduler_post_calls": 0,
        "source_task_id": (
            updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
        ),
        "expected_node": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE,
        "node_allocations": [],
        "node_tasks": [],
        "submission_policy": "exactly_one_post_no_retry_no_cancel",
    }
    pre_path = hedge_root / "pre_submit_receipt.json"
    pre_path.write_text(json.dumps(pre_submit))
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_HEDGE_PRE_SUBMIT_SHA256",
        updater._file_sha256(pre_path),
    )
    post_attempt = {
        "schema": "mft-goal-rx-main-l5-isolated-hedge-post-attempt-v1",
        "scheduler_post_calls": 1,
        "retry_forbidden": True,
    }
    post_path = hedge_root / "post_attempt_receipt.json"
    post_path.write_text(json.dumps(post_attempt))
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_HEDGE_POST_ATTEMPT_SHA256",
        updater._file_sha256(post_path),
    )
    receipt = {
        "schema": "mft-goal-rx-main-l5-isolated-hedge-submission-v1",
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
        "task_name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME,
        "scheduler_post_calls": 1,
        "scheduler_mutation_performed": True,
        "retry_cancel_forbidden": True,
        "strict_node_placement": True,
        "requested_node_name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE,
        "node_name_policy": "strict",
        "exclusive_node_response": None,
        "runtime_fail_close_marker_required": "MFT_EXCLUSIVE_PLACEMENT_JSON",
        "geometry_sha256": (
            updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        ),
        "solver_revision": updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
        "core_auth_sha256": updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256,
        "cpus": updater.RX_MAIN_L5_NATIVE_CANARY_CPUS,
        "memory_mb": updater.RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB,
        "max_workers_per_node": 1,
    }
    receipt_path = hedge_root / "submission_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_HEDGE_RECEIPT_SHA256",
        updater._file_sha256(receipt_path),
    )
    hedge_monitor = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA,
        "scheduler_mutation_performed": False,
        "updated_at_utc": OBSERVED,
        "task": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
            "name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME,
            "status": "queued",
            "exit_code": None,
            "actual_node_name": "",
            "slurm_job_id": "",
        },
    }
    hedge_monitor["payload_sha256"] = (
        updater._rx_main_l5_monitor_payload_sha256(hedge_monitor)
    )
    hedge_process = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_PROCESS_SCHEMA,
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
        "allowed_http_method": "GET",
        "scheduler_mutation_performed": False,
        "post_cancel_retry_forbidden": True,
    }
    hedge_process["payload_sha256"] = (
        updater._rx_main_l5_monitor_payload_sha256(hedge_process)
    )
    (hedge_root / "monitor_state.json").write_text(
        json.dumps(hedge_monitor)
    )
    (hedge_root / "monitor_process.json").write_text(
        json.dumps(hedge_process)
    )
    shared_started_at = "2026-07-27 02:19:33"
    placement_state = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_STATE_SCHEMA,
        "scheduler_mutation_performed": False,
        "updated_at_utc": OBSERVED,
        "allocation": None,
        "predecessor": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
            "status": "failed",
            "exit_code": 1,
            "actual_node_name": "n110",
            "allocation_id": (
                updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID
            ),
            "slurm_job_id": updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID,
            "started_at": shared_started_at,
        },
        "simultaneous": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_SIMULTANEOUS_TASK_ID,
            "status": "failed",
            "actual_node_name": "n110",
            "allocation_id": (
                updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID
            ),
            "slurm_job_id": updater.RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID,
            "started_at": shared_started_at,
        },
        "task": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
            "name": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME,
            "status": "queued",
            "requested_node_name": (
                updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            ),
            "node_name_policy": "strict",
            "strict_node_placement": True,
            "actual_node_name": "",
            "allocation_id": 0,
            "slurm_job_id": "",
            "placement_contract_satisfied": False,
        },
    }
    placement_state["payload_sha256"] = updater.hashlib.sha256(
        json.dumps(
            placement_state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    placement_process = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_PROCESS_SCHEMA,
        "allowed_http_method": "GET",
        "scheduler_mutation_performed": False,
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
        "predecessor_id": (
            updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
        ),
        "simultaneous_id": (
            updater.RX_MAIN_L5_NATIVE_CANARY_SIMULTANEOUS_TASK_ID
        ),
    }
    placement_process["payload_sha256"] = updater.hashlib.sha256(
        json.dumps(
            placement_process,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    (hedge_root / "placement_lineage_state.json").write_text(
        json.dumps(placement_state)
    )
    (hedge_root / "placement_lineage_monitor_process.json").write_text(
        json.dumps(placement_process)
    )
    return successor_root, hedge_root


def test_rx_main_l5_operational_chain_card_is_stage_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root, hedge_root = (
        _write_rx_main_l5_operational_chain_evidence(
            tmp_path,
            monkeypatch,
        )
    )
    monkeypatch.setattr(
        updater,
        "_rx_main_l5_strict_n107_receipt",
        lambda _root: ({"scheduler_post_calls": 1}, "authenticated"),
    )
    monkeypatch.setattr(
        updater,
        "_rx_main_l5_strict_n107_terminal",
        lambda _root: (
            {
                "state": "failed",
                "exit_code": 86,
                "node_name": "n107",
                "slurm_job_id": "845454",
            },
            "authenticated",
        ),
    )
    monkeypatch.setattr(
        updater,
        "_rx_main_l5_strict_n107_successor_receipt",
        lambda _root: ({"scheduler_post_calls": 1}, "authenticated"),
    )
    monkeypatch.setattr(
        updater,
        "_rx_main_l5_strict_n107_successor_live",
        lambda _root: (
            {
                "state": "running",
                "allocation_id": 14718,
                "slurm_job_id": "845487",
                "temp_free_kb": 721_310_408,
                "temp_min_free_kb": 20_971_520,
                "observed_at": OBSERVED,
            },
            "authenticated",
        ),
    )

    card = updater._rx_main_l5_operational_chain_card(
        OBSERVED,
        root=tmp_path,
        successor_root=successor_root,
        hedge_root=hedge_root,
    )

    assert len(card["title"]) <= 160
    assert "97141 MATRIX-OK/SESSION-FAIL" in card["title"]
    assert "97142 EXCL QUEUED" in card["title"]
    assert "97143 GUARD-FALSE-NEG" in card["title"]
    assert "97144 n107 GUARD+TEMP PASS RUNNING" in card["title"]
    assert card["state"] == "in_progress"
    assert card["progress_pct"] == 65
    assert any(
        "Matrix Setup1 solved 3s" in item
        and "result-extraction collapse" in item
        for item in card["evidence"]
    )
    assert any(
        "task97121 started same second" in item
        and "allocation14648/job840585" in item
        for item in card["evidence"]
    )
    assert any(
        "operational pre-thermal failure" in item
        and "candidate design invalidated=false" in item
        for item in card["evidence"]
    )
    assert any(
        "task97142 receipt authenticated=true" in item
        and "strict n107" in item
        and "exclusive_node=true" in item
        for item in card["evidence"]
    )
    assert any(
        "task97142 GET monitor authenticated=true" in item
        and "state=queued" in item
        and "placement lineage authenticated=true" in item
        for item in card["evidence"]
    )
    assert any(
        "task97143 receipt authenticated=true" in item
        and "terminal=FAILED exit86" in item
        and "runtime guard false-negative=true" in item
        and "design invalidated=false" in item
        for item in card["evidence"]
    )
    assert any(
        "task97144 receipt authenticated=true" in item
        and "state=running" in item
        and "allocation=14718" in item
        and "task-local temp PASS" in item
        for item in card["evidence"]
    )


def test_rx_main_l5_native_canary_card_uses_receipt_without_live_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root = _write_rx_main_l5_canary_evidence(
        tmp_path, monkeypatch
    )

    card = updater._rx_main_l5_native_canary_card(
        OBSERVED,
        root=tmp_path,
        successor_root=successor_root,
    )

    assert card["id"] == updater.RX_MAIN_L5_NATIVE_CANARY_CARD_ID
    assert len(card["title"]) <= 160
    assert card["state"] == "in_progress"
    assert len(card["evidence"]) <= 12
    assert "97140 PRE-SOLVER FAILED" in card["title"]
    assert "97141 SUBMITTED 1 POST" in card["title"]
    assert "LIVE STATE UNCLAIMED" in card["title"]
    assert "QUEUED" not in card["title"]
    assert any(
        "operational pre-solver core-auth failure" in item
        for item in card["evidence"]
    )
    assert any(
        "mesh generated=false" in item
        and "FEA invoked=false" in item
        and "scientific result=false" in item
        and "design failure=false" in item
        for item in card["evidence"]
    )
    assert any(
        updater.RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256 in item
        and updater.RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256 in item
        for item in card["evidence"]
    )
    assert any(
        "8CPU/65536MiB/43200s (12h)" in item
        for item in card["evidence"]
    )
    assert any(
        "thermal_max_iterations=1" in item for item in card["evidence"]
    )
    assert any(
        updater.RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256 in item
        and updater.RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION in item
        for item in card["evidence"]
    )
    assert any(
        updater.RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION in item
        for item in card["evidence"]
    )
    assert any(
        "fixed cooling unchanged=true" in item for item in card["evidence"]
    )
    assert any(
        "missing_rx_main_solids=[]" in item
        and "missing_fluid_coupling=[]" in item
        and "rx_main_adjacency_passed=true" in item
        for item in card["evidence"]
    )
    assert any(
        "unpaired_interfaces=[]" in item
        and "no_unpaired_log_marker=true" in item
        for item in card["evidence"]
    )
    assert any(
        "interface pass claimed=false" in item
        and "design scientific promotion=false" in item
        and "production promotion=false" in item
        for item in card["evidence"]
    )


def test_rx_main_l5_native_canary_rejects_unsealed_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root = _write_rx_main_l5_canary_evidence(
        tmp_path, monkeypatch
    )
    (successor_root / "monitor_state.json").write_text(
        json.dumps({"state": "running"}),
        encoding="utf-8",
    )

    card = updater._rx_main_l5_native_canary_card(
        OBSERVED,
        root=tmp_path,
        successor_root=successor_root,
    )

    assert "MONITOR INVALID" in card["title"]
    assert "GET RUNNING" not in card["title"]
    assert any(
        "GET monitor authenticated=false" in item
        and "live state=unclaimed" in item
        for item in card["evidence"]
    )


def test_rx_main_l5_native_canary_accepts_sealed_get_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root = _write_rx_main_l5_canary_evidence(
        tmp_path, monkeypatch
    )
    monitor = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA,
        "scheduler_mutation_performed": False,
        "updated_at_utc": OBSERVED,
        "task": {
            "id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
            "name": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
            "status": "running",
            "actual_node_name": "n110",
            "slurm_job_id": "840585",
        },
    }
    monitor["payload_sha256"] = updater._rx_main_l5_monitor_payload_sha256(
        monitor
    )
    process = {
        "schema": updater.RX_MAIN_L5_NATIVE_CANARY_MONITOR_PROCESS_SCHEMA,
        "task_id": updater.RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
        "allowed_http_method": "GET",
        "scheduler_mutation_performed": False,
        "post_cancel_retry_forbidden": True,
    }
    process["payload_sha256"] = updater._rx_main_l5_monitor_payload_sha256(
        process
    )
    (successor_root / "monitor_state.json").write_text(
        json.dumps(monitor),
        encoding="utf-8",
    )
    (successor_root / "monitor_process.json").write_text(
        json.dumps(process),
        encoding="utf-8",
    )

    card = updater._rx_main_l5_native_canary_card(
        OBSERVED,
        root=tmp_path,
        successor_root=successor_root,
    )

    assert "GET RUNNING n110/j840585" in card["title"]
    assert any(
        "GET monitor authenticated=true" in item
        and "state=running" in item
        and "node=n110" in item
        and "job=840585" in item
        for item in card["evidence"]
    )
    assert any(
        "interface pass claimed=false" in item
        and "production promotion=false" in item
        for item in card["evidence"]
    )


def test_rx_main_l5_native_canary_receipt_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root = _write_rx_main_l5_canary_evidence(
        tmp_path, monkeypatch
    )
    receipt_path = tmp_path / "submission_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scheduler_post_calls"] = 2
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    card = updater._rx_main_l5_native_canary_card(
        OBSERVED,
        root=tmp_path,
        successor_root=successor_root,
    )

    assert "LINEAGE INVALID" in card["title"]
    assert "SUBMITTED" not in card["title"]
    assert any(
        "lineage_authentication=invalid_fail_closed" == item
        for item in card["evidence"]
    )
    assert any(
        "design promotion=false" in item
        and "production promotion=false" in item
        for item in card["evidence"]
    )


def test_merge_upserts_one_rx_main_l5_native_canary_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor_root = _write_rx_main_l5_canary_evidence(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        updater,
        "RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_ROOT",
        successor_root,
    )

    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    merged_again = updater.merge_status(
        merged,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged_again["current"]]

    assert ids.count(updater.RX_MAIN_L5_NATIVE_CANARY_CARD_ID) == 1
    assert ids.index(updater.RX_MAIN_L5_NATIVE_CANARY_CARD_ID) < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged_again)


def test_merge_removes_stale_automatic_continuation_card_when_disarmed() -> None:
    source = _status()
    source["current"].insert(
        -1,
        _item(updater.LEGACY_CONTINUATION_CARD_ID, "in_progress"),
    )

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    ids = [item["id"] for item in merged["current"]]

    assert updater.LEGACY_CONTINUATION_CARD_ID not in ids
    assert ids.count(updater.SELECTION_POLICY_CARD_ID) == 1
    assert ids.index(updater.SELECTION_POLICY_CARD_ID) < ids.index(
        "parallel-workstreams"
    )
    updater.validate_status_sync(merged)


def test_synchronize_once_is_atomic_and_identity_failure_keeps_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codex-work-status.json"
    path.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    raw_tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    sync = updater.synchronize_once(
        status_file=path,
        task_reader=_reader_from(raw_tasks),
        observed_at=OBSERVED,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert updater.validate_status_sync(payload) == sync
    assert not list(tmp_path.glob("*.tmp"))

    before = path.read_bytes()
    drifted = copy.deepcopy(raw_tasks)
    drifted[96325]["name"] = "wrong-task-name"
    with pytest.raises(updater.UpdaterError, match="name drifted"):
        updater.synchronize_once(
            status_file=path,
            task_reader=_reader_from(drifted),
            observed_at=OBSERVED,
        )
    assert path.read_bytes() == before


def test_merge_adds_authenticated_codex_automation_cards(tmp_path: Path) -> None:
    postsuccess = updater._sealed(
        {
            "schema_version": updater.POSTSUCCESS_STATE_SCHEMA,
            "status": "pending_standard_collections",
            "collection_count": 0,
            "pending_count": 7,
            "terminal_failure_count": 0,
            "expected_lane_count": 7,
            "lifecycle_lane_count": 9,
            "effective_task_ids": list(updater.STANDARD_SELECTION_TASK_IDS),
            "lifecycle_task_ids": list(updater.STANDARD_SELECTION_LIFECYCLE_TASK_IDS),
            "selection_superseded_task_ids": list(
                updater.STANDARD_SELECTION_SUPERSEDED_TASK_IDS
            ),
            "lifecycle_collection_count": 0,
            "lifecycle_pending_count": 8,
            "lifecycle_terminal_failure_count": 1,
            "lanes": [
                {
                    "task_id": task_id,
                    "selection_effective": (
                        task_id in updater.STANDARD_SELECTION_TASK_IDS
                    ),
                    "status": ("terminal_failure" if task_id == 96337 else "pending"),
                }
                for task_id in updater.STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            ],
            "diagnostic_only": True,
            "production_eligible": False,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "surrogate_retraining_performed": False,
            "production_pareto_emitted": False,
        }
    )
    postsuccess_path = tmp_path / "postsuccess-state.json"
    postsuccess_path.write_text(
        json.dumps(postsuccess, ensure_ascii=False),
        encoding="utf-8",
    )
    thermal_path = tmp_path / "thermal-watch-state.json"
    thermal_path.write_text(
        json.dumps(_thermal_state(), ensure_ascii=False),
        encoding="utf-8",
    )
    continuation_path = tmp_path / "standard-full-continuation-state.json"
    continuation_path.write_text(
        json.dumps(_continuation_state(), ensure_ascii=False),
        encoding="utf-8",
    )
    gate_root = tmp_path / "gate"
    gate_root.mkdir()
    pending = updater._sealed(
        {
            "schema": updater.FINAL_GATE_PENDING_SCHEMA,
            "status": "pending",
            "pending_reasons": [
                "full collector state is watching",
                "thermal collector state is watching",
            ],
            "final_package_created": False,
            "full_aedt_promoted": False,
            "symmetric_aedt_promoted": False,
        }
    )
    (gate_root / "pending_manifest.json").write_text(
        json.dumps(pending, ensure_ascii=False),
        encoding="utf-8",
    )

    merged = updater.merge_status(
        _status(),
        _current_fast_lane_tasks(),
        observed_at=OBSERVED,
        postsuccess_state_file=postsuccess_path,
        thermal_bridge_state_file=thermal_path,
        standard_full_continuation_state_file=continuation_path,
        final_gate_root=gate_root,
    )

    by_id = {item["id"]: item for item in merged["current"]}
    postsuccess_card = by_id["codex-standard-postsuccess-pipeline"]
    assert "AUTH 0 | PENDING 7 | ACTUAL PASS 0" in postsuccess_card["title"]
    assert any(
        "selection-effective authenticated=0/7 / pending=7 / "
        "operational terminal failures=0" in item
        for item in postsuccess_card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=0/9 / pending=8 / "
        "operational terminal failures=1" in item
        for item in postsuccess_card["evidence"]
    )
    assert any(
        "lifecycle mode=v6-sealed-effective-plus-live-GET-lifecycle" in item
        for item in postsuccess_card["evidence"]
    )
    assert by_id["codex-final-aedt-package-gate"]["title"].endswith("PENDING")
    thermal = by_id["codex-thermal-artifact-handoff"]
    assert thermal["title"] == ("CODEX AUTO · THERMAL ARTIFACT HANDOFF · RUNNING")
    assert any("source task96324 state=running" in item for item in thermal["evidence"])
    assert any("Scheduler POST count=0" in item for item in thermal["evidence"])
    assert any("scientific claim=false" in item for item in thermal["evidence"])
    continuation = by_id["codex-standard-full-continuation"]
    assert continuation["title"].endswith("PENDING_STANDARD_COLLECTIONS")
    assert any(
        "Scheduler POST attempts consumed=0/1" in item
        for item in continuation["evidence"]
    )
    assert any("scientific PASS=false" in item for item in continuation["evidence"])
    policy = by_id[updater.SELECTION_POLICY_CARD_ID]
    assert "AUTO FULL OFF" in policy["title"]
    assert any(
        "automatic Standard-to-Full per candidate=false" in item
        for item in policy["evidence"]
    )
    postsuccess_card = by_id["codex-standard-postsuccess-pipeline"]
    assert any(
        "Standard selection lanes=7" in item
        and "task96338" in item
        and "task96332" not in item
        for item in postsuccess_card["evidence"]
    )
    assert any(
        "full AEDT promoted=false" in item
        for item in by_id["codex-final-aedt-package-gate"]["evidence"]
    )
    assert merged["current"][-1]["id"] == "parallel-workstreams"
    updater.validate_status_sync(merged)

    postsuccess["pending_count"] = 1
    postsuccess_path.write_text(
        json.dumps(postsuccess, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.merge_status(
            _status(),
            _mixed_tasks(),
            observed_at=OBSERVED,
            postsuccess_state_file=postsuccess_path,
            thermal_bridge_state_file=thermal_path,
            standard_full_continuation_state_file=continuation_path,
            final_gate_root=gate_root,
        )


def test_local_symmetric_selection_card_shows_sealed_selection_and_budget(
    tmp_path: Path,
) -> None:
    selected = {
        "task_id": 96331,
        "candidate_physics_sha256": "c" * 64,
        "minimum_normalized_actual_margin": 0.0125,
        "actual_total_loss_W": 5319.5,
        "actual_volume_L": 803.2,
    }
    state = _local_selection_state(
        status="selected_symmetric_hard_pass",
        authenticated=2,
        pending=5,
        failures=0,
        selected=selected,
    )
    path = tmp_path / "local-selection-state.json"
    path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )

    merged = updater.merge_status(
        _status(),
        _current_fast_lane_tasks(),
        observed_at=OBSERVED,
        local_symmetric_selection_state_file=path,
    )
    card = next(
        item
        for item in merged["current"]
        if item["id"] == updater.LOCAL_SYMMETRIC_SELECTION_CARD_ID
    )

    assert "SELECTED task96331" in card["title"]
    assert "AUTH 2/7" in card["title"]
    assert "PENDING 5" in card["title"]
    assert "ACTUAL PASS 0" in card["title"]
    assert "AUTO FULL OFF" in card["title"]
    assert any(
        "selection-effective authenticated=2/7 / pending=5 / "
        "operational terminal failures=0" in item
        for item in card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=2/9 / pending=6 / "
        "operational terminal failures=1" in item
        for item in card["evidence"]
    )
    assert any(
        "lifecycle mode=v6-sealed-effective-plus-live-GET-lifecycle" in item
        for item in card["evidence"]
    )
    assert any(
        "selected symmetric result=task96331 cccccccccccc" in item
        and "loss=5319.500W" in item
        and "volume=803.200L" in item
        for item in card["evidence"]
    )
    assert any(
        "local bounded budget=round 1/2 / prepared 0/3 / total cap 6" in item
        for item in card["evidence"]
    )
    assert any(
        item == "AUTO FULL OFF / Scheduler mutation=false / methods=[]"
        for item in card["evidence"]
    )
    assert merged["current"][-1]["id"] == "parallel-workstreams"
    updater.validate_status_sync(merged)


def test_effective_and_lifecycle_failures_are_reported_separately(
    tmp_path: Path,
) -> None:
    local_state = _local_selection_state(
        status="partial_measured_waiting",
        authenticated=0,
        pending=6,
        failures=1,
        failure_task_ids=(96328,),
    )
    local_path = tmp_path / "local-selection-state.json"
    local_path.write_text(
        json.dumps(local_state, ensure_ascii=False),
        encoding="utf-8",
    )
    tasks = _current_fast_lane_tasks()

    local_card = updater._local_symmetric_selection_card(
        local_path,
        OBSERVED,
        tasks,
    )

    assert "PENDING 6" in local_card["title"]
    assert any(
        "selection-effective authenticated=0/7 / pending=6 / "
        "operational terminal failures=1" in item
        for item in local_card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=0/9 / pending=7 / "
        "operational terminal failures=2" in item
        for item in local_card["evidence"]
    )

    postsuccess = updater._sealed(
        {
            "schema_version": updater.POSTSUCCESS_STATE_SCHEMA,
            "status": "pending_standard_collections",
            "collection_count": 0,
            "pending_count": 6,
            "terminal_failure_count": 1,
            "expected_lane_count": 7,
            "lifecycle_lane_count": 9,
            "effective_task_ids": list(updater.STANDARD_SELECTION_TASK_IDS),
            "lifecycle_task_ids": list(
                updater.STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            ),
            "selection_superseded_task_ids": list(
                updater.STANDARD_SELECTION_SUPERSEDED_TASK_IDS
            ),
            "lifecycle_collection_count": 0,
            "lifecycle_pending_count": 7,
            "lifecycle_terminal_failure_count": 2,
            "lanes": [
                {
                    "task_id": lane["task_id"],
                    "selection_effective": lane["selection_effective"],
                    "status": lane["effective_status"],
                }
                for lane in local_state["lanes"]
            ],
            "diagnostic_only": True,
            "production_eligible": False,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "surrogate_retraining_performed": False,
            "production_pareto_emitted": False,
        }
    )
    postsuccess_path = tmp_path / "postsuccess-state.json"
    postsuccess_path.write_text(
        json.dumps(postsuccess, ensure_ascii=False),
        encoding="utf-8",
    )

    pipeline_card = updater._postsuccess_card(
        postsuccess_path,
        OBSERVED,
        tasks,
    )

    assert "PENDING 6" in pipeline_card["title"]
    assert any(
        "selection-effective authenticated=0/7 / pending=6 / "
        "operational terminal failures=1" in item
        for item in pipeline_card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=0/9 / pending=7 / "
        "operational terminal failures=2" in item
        for item in pipeline_card["evidence"]
    )


def test_local_symmetric_selection_v5_fallback_overlays_new_lifecycle_once(
    tmp_path: Path,
) -> None:
    state = _local_selection_state(
        status="awaiting_symmetric_results",
        authenticated=0,
        pending=7,
        failures=0,
        layout="v5",
    )
    path = tmp_path / "legacy-local-selection-state.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tasks = _mixed_tasks()
    failed_spec = next(spec for spec in updater.TASK_SPECS if spec.task_id == 96337)
    current_spec = next(spec for spec in updater.TASK_SPECS if spec.task_id == 96338)
    tasks[96337] = updater._validate_task(
        failed_spec,
        _task(
            failed_spec,
            state="failed",
            failure_message="standalone core opt-in authentication digest mismatch",
        ),
    )
    tasks[96338] = updater._validate_task(current_spec, _task(current_spec))

    card = updater._local_symmetric_selection_card(path, OBSERVED, tasks)

    assert "AUTH 0/7" in card["title"]
    assert "PENDING 7" in card["title"]
    assert any(
        "selection-effective authenticated=0/7 / pending=7 / "
        "operational terminal failures=0" in item
        for item in card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=0/9 / pending=8 / "
        "operational terminal failures=1" in item
        for item in card["evidence"]
    )
    assert any(
        "lifecycle mode=v5-overlay-fallback" in item for item in card["evidence"]
    )


def test_local_symmetric_selection_card_shows_bounded_local_batch(
    tmp_path: Path,
) -> None:
    state = _local_selection_state(
        status="local_prepare_only_batch_ready",
        authenticated=6,
        pending=0,
        failures=1,
        prepared=3,
    )
    path = tmp_path / "local-selection-state.json"
    path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )

    card = updater._local_symmetric_selection_card(path, OBSERVED)

    assert "LOCAL BATCH 3/3 READY" in card["title"]
    assert "AUTH 6/7" in card["title"]
    assert "PENDING 0" in card["title"]
    assert any(
        "selection-effective authenticated=6/7 / pending=0 / "
        "operational terminal failures=1" in item
        for item in card["evidence"]
    )
    assert any(
        "lifecycle attempts=9 / authenticated=6/9 / pending=1 / "
        "operational terminal failures=2" in item
        for item in card["evidence"]
    )
    assert any(
        "prepared 3/3" in item and "prepare-only" in item for item in card["evidence"]
    )


def test_local_symmetric_selection_missing_and_tamper_are_fail_safe(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-local-selection-state.json"
    missing_card = updater._local_symmetric_selection_card(missing, OBSERVED)
    assert "STATE MISSING" in missing_card["title"]
    assert "AUTO FULL OFF" in missing_card["title"]
    assert any(
        item == "selected symmetric result=none claimed"
        for item in missing_card["evidence"]
    )
    assert any(
        item == "AUTO FULL OFF / Scheduler mutation=false"
        for item in missing_card["evidence"]
    )

    path = tmp_path / "tampered-local-selection-state.json"
    state = _local_selection_state()
    state["selected_symmetric_result"] = {
        "task_id": 99999,
        "candidate_physics_sha256": "forged",
    }
    path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )

    merged = updater.merge_status(
        _status(),
        _mixed_tasks(),
        observed_at=OBSERVED,
        local_symmetric_selection_state_file=path,
    )
    card = next(
        item
        for item in merged["current"]
        if item["id"] == updater.LOCAL_SYMMETRIC_SELECTION_CARD_ID
    )
    assert "STATE INVALID" in card["title"]
    assert not any("99999" in item for item in card["evidence"])
    updater.validate_status_sync(merged)


def test_local_symmetric_selection_accepts_producer_sealed_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "local-watch-output"
    produced = local_watch.process_cycle(
        source_state_path=tmp_path / "missing-source-state.json",
        aggregate_manifest=tmp_path / "missing-aggregate.json",
        output_root=output,
    )
    assert produced["status"] == "blocked_fail_closed"

    card = updater._local_symmetric_selection_card(output / "state.json", OBSERVED)

    assert "FAIL-CLOSED" in card["title"]
    assert "STATE INVALID" not in card["title"]
    assert "AUTO FULL OFF" in card["title"]
    assert any(
        item == "AUTO FULL OFF / Scheduler mutation=false / methods=[]"
        for item in card["evidence"]
    )


def test_thermal_bridge_state_absence_and_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }
    missing = tmp_path / "missing-thermal-state.json"
    before = status_file.read_bytes()

    with pytest.raises(FileNotFoundError):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            thermal_bridge_state_file=missing,
        )
    assert status_file.read_bytes() == before

    thermal_path = tmp_path / "thermal-state.json"
    state = _thermal_state()
    state["scheduler_post_calls_total"] = 1
    thermal_path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            thermal_bridge_state_file=thermal_path,
        )
    assert status_file.read_bytes() == before


def test_standard_full_continuation_state_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }
    continuation_path = tmp_path / "continuation-state.json"
    state = _continuation_state()
    state["scientific_pass_claimed"] = True
    continuation_path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )
    before = status_file.read_bytes()

    with pytest.raises(updater.UpdaterError, match="seal drifted"):
        updater.synchronize_once(
            status_file=status_file,
            task_reader=_reader_from(tasks),
            observed_at=OBSERVED,
            standard_full_continuation_state_file=continuation_path,
        )
    assert status_file.read_bytes() == before


class _Response(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def test_scheduler_reader_uses_bounded_get(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = updater.TASK_SPECS[0]
    observed: dict[str, Any] = {}

    def urlopen(request: Any, *, timeout: float) -> _Response:
        observed["method"] = request.get_method()
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response(json.dumps(_task(spec)).encode("utf-8"))

    monkeypatch.setattr(updater.urllib.request, "urlopen", urlopen)
    result = updater._get_scheduler_task(
        "http://127.0.0.1:8002",
        spec.task_id,
        timeout_seconds=3.0,
    )

    assert result["task_id"] == spec.task_id
    assert observed == {
        "method": "GET",
        "url": f"http://127.0.0.1:8002/api/tasks/{spec.task_id}",
        "timeout": 3.0,
    }


@pytest.mark.parametrize(
    "spec",
    tuple(spec for spec in updater.TASK_SPECS if spec.max_workers_per_node == 1),
)
def test_official_task_max_workers_is_fail_closed(
    spec: updater.TaskSpec,
) -> None:
    assert spec.task_id in {96328, 96329, 96330, 96331, 96332, 96333, 96342}
    task = _task(spec)
    task["max_workers_per_node"] = 2

    with pytest.raises(updater.UpdaterError, match="max_workers_per_node drifted"):
        updater._validate_task(spec, task)


def test_corrected_allocation_force_risk_identity_is_fail_closed() -> None:
    spec = next(item for item in updater.TASK_SPECS if item.task_id == 96328)
    task = _task(spec)
    task["allocation_id"] = 14621

    with pytest.raises(
        updater.UpdaterError,
        match="corrected allocation identity drifted",
    ):
        updater._validate_task(spec, task)


def test_superseded_rounded_lane_accepts_scheduler_queue_reset() -> None:
    spec = next(
        item
        for item in updater.TASK_SPECS
        if item.task_id == updater.ROUNDED_FINAL_TASK_ID
    )
    task = _task(spec, state="queued", allocation_id=None, slurm_job_id="")
    task["allocation_id"] = None
    task["slurm_job_id"] = ""

    result = updater._validate_task(spec, task)

    assert result["state"] == "queued"
    assert result["allocation_id"] is None
    assert result["slurm_job_id"] == ""


def test_timeout_hedge_new_allocation_requirement_is_fail_closed() -> None:
    spec = next(
        item
        for item in updater.TASK_SPECS
        if item.task_id == updater.ROUNDED_TIMEOUT_HEDGE_TASK_ID
    )
    task = _task(spec, state="queued", allocation_id=None, slurm_job_id="")
    task["requested_allocation_id"] = 14620

    with pytest.raises(
        updater.UpdaterError,
        match="new-allocation requirement drifted",
    ):
        updater._validate_task(spec, task)


def test_run_once_writes_sealed_pid_and_log(tmp_path: Path) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    pid_file = tmp_path / "updater.pid.json"
    log_file = tmp_path / "updater.jsonl"
    lock_file = tmp_path / "updater.lock"
    tasks = {
        spec.task_id: _task(spec, allocation_id=300 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    result = updater.run_updater(
        status_file=status_file,
        scheduler_url=updater.DEFAULT_SCHEDULER_URL,
        once=True,
        interval_seconds=60,
        pid_file=pid_file,
        log_file=log_file,
        lock_file=lock_file,
        task_reader=_reader_from(tasks),
    )

    assert result is not None
    pid = json.loads(pid_file.read_text(encoding="utf-8"))
    updater.validate_seal(pid, updater.PID_SCHEMA)
    assert pid["scheduler_methods_allowed"] == ["GET"]
    assert pid["scheduler_mutation_performed"] is False
    events = [
        json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "updater_started",
        "cycle_completed",
    ]
    for event in events:
        updater.validate_seal(event, updater.LOG_SCHEMA)


def test_cli_modes_and_default_interval() -> None:
    parser = updater._parser()
    defaults = parser.parse_args(["--once"])
    assert defaults.interval_seconds == 60
    assert (
        defaults.local_symmetric_selection_state_file
        == updater.DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    )
    parsed = parser.parse_args(
        [
            "--watch",
            "--thermal-bridge-state-file",
            "thermal.json",
            "--local-symmetric-selection-state-file",
            "local-selection.json",
            "--standard-full-continuation-state-file",
            "continuation.json",
        ]
    )
    assert parsed.watch is True
    assert parsed.thermal_bridge_state_file == Path("thermal.json")
    assert parsed.local_symmetric_selection_state_file == Path("local-selection.json")
    assert parsed.standard_full_continuation_state_file == Path("continuation.json")
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--once", "--watch"])
