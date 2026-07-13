import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from regression_260707.monitoring.readers import ArtifactService


KST = timezone(timedelta(hours=9))
FIXED_NOW = datetime(2026, 7, 11, 3, 0, tzinfo=KST)


class DummyScheduler:
    def __init__(self):
        self.parallel_target = 300

    def snapshot(self):
        return {
            "connected": True,
            "url": "http://scheduler.invalid",
            "read_only": False,
            "control_enabled": True,
            "task_prefix": "mft",
            "project": "MFT_1MW_2026v1",
            "parallel_target": self.parallel_target,
            "parallel_target_min": 1,
            "parallel_target_max": 300,
            "live_queued": 3,
            "live_attaching": 2,
            "live_running": 4,
            "logical_active": 9,
            "total": 12,
            "running": 4,
            "pending": 3,
            "completed": 3,
            "failed": 1,
            "cancelled": 1,
            "other": 0,
            "statuses": {"running": 4, "pending": 3, "completed": 3, "failed": 1, "cancelled": 1},
            "error": None,
            "updated_at": FIXED_NOW.isoformat(),
        }

    def set_parallel_target(self, target):
        self.parallel_target = target
        return {
            "project": "MFT_1MW_2026v1",
            "parallel_target": target,
            "parallel_target_min": 1,
            "parallel_target_max": 300,
            "live_queued": 3,
            "live_attaching": 2,
            "live_running": 4,
            "logical_active": 9,
            "project_updated_at": FIXED_NOW.isoformat(),
        }


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def campaign_root(tmp_path):
    root = tmp_path / "regression_260707"
    dataset = root / "data" / "dataset"
    write_json(dataset / "manifest.json", {
        "updated": "260711_023000_000000", "total_rows": 2, "new_rows": 1,
        "new_unique_rows": 1, "git_hashes": [
            "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a", "b" * 40,
        ],
    })
    write_csv(dataset / "train_io.csv", [
        {
            "train_io_schema_version": "3",
            "saved_at": "2026-07-11 02:30:00", "result_valid_em": "1", "result_valid_thermal": "1",
            "time_matrix": "300", "time_loss": "1700", "time_thermal": "1000", "time": "3000",
            "git_hash": "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a", "project_name": "one",
        },
        {
            "train_io_schema_version": "3",
            "saved_at": "2026-07-11 01:30:00", "result_valid_em": "1", "result_valid_thermal": "0",
            "time_matrix": "600", "time_loss": "1800", "time_thermal": "1200", "time": "3600",
            "git_hash": "b" * 40, "project_name": "two",
        },
    ])
    write_json(dataset / "collect_cache.json", {"harvested": [1, 2], "nodata": [3], "local_parts": ["a.parquet"]})
    write_json(root / "training" / "strict_data_status.json", {
        "time": "2026-07-11T02:30:00+09:00",
        "raw_rows": 2,
        "strict_em_rows": 2,
        "strict_full_rows": 1,
        "state_identity": {
            "solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "library_revision": "c" * 40,
        },
    })
    history_path = root / "monitoring" / "runtime" / "monitor_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("\n".join([
        json.dumps({
            "time": "2026-07-11T00:30:00+09:00",
            "data": {
                "count_basis": "pinned_strict_full", "total_rows": 51,
                "pinned_solver_revision": "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a",
                "pinned_library_revision": "c" * 40,
            },
        }),
        json.dumps({
            "time": "2026-07-11T01:00:00+09:00",
            "data": {"count_basis": "pinned_strict_full", "total_rows": 51},
        }),
        json.dumps({
            "time": "2026-07-11T01:30:00+09:00",
            "data": {
                "count_basis": "pinned_strict_full", "total_rows": 0,
                "pinned_solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
                "pinned_library_revision": "c" * 40,
            },
        }),
        json.dumps({
            "time": "2026-07-11T02:30:00+09:00",
            "data": {
                "count_basis": "pinned_strict_full", "total_rows": 1,
                "pinned_solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
                "pinned_library_revision": "c" * 40,
            },
        }),
    ]) + "\n", encoding="utf-8")

    registry = root / "training" / "registry"
    generation = registry / "generations" / "run1"
    write_json(generation / "train_report.json", {
        "time": "2026-07-11T02:00:00+09:00",
        "training_run_id": "run1",
        "dataset_sha256": "dataset-sha",
        "profile_sha256": "profile-sha",
        "strict_full_rows": 100,
        "report": {
            "Llt_phys": {"n_train": 80, "n_holdout": 20, "r2": .91, "rmse": .2, "mape_pct": 1.2, "p90_ape_pct": 2.1, "q90_conformal": 1.1},
            "P_winding_total": {"n_train": 80, "n_holdout": 20, "r2": .72, "rmse": 120, "mape_pct": 22, "p90_ape_pct": 30, "q90_conformal": 2.2},
        },
    })
    report_sha256 = hashlib.sha256(
        (generation / "train_report.json").read_bytes()
    ).hexdigest()
    write_json(generation / "quality_gate.json", {
        "passed": True, "training_run_id": "run1",
        "dataset_sha256": "dataset-sha",
        "profile_sha256": "profile-sha",
        "generation": "generations/run1",
        "generation_report_sha256": report_sha256,
    })
    gate_sha256 = hashlib.sha256(
        (generation / "quality_gate.json").read_bytes()
    ).hexdigest()
    write_json(registry / "current.json", {
        "schema_version": 2, "training_run_id": "run1",
        "generation": "generations/run1",
        "dataset_sha256": "dataset-sha",
        "profile_sha256": "profile-sha",
        "strict_full_rows": 100,
        "generation_report_sha256": report_sha256,
        "quality_gate_sha256": gate_sha256,
    })
    write_json(generation / "Llt_phys" / "meta.json", {"trained_at": "2026-07-11T02:00:00+09:00"})
    write_csv(root / "training" / "learning_curve.csv", [
        {"time": "2026-07-10 10:00:00", "target": "Llt_phys", "n": "60", "r2": ".85", "rmse": ".4", "mape_pct": "2", "p90_ape_pct": "3", "slice": "global"},
        {"time": "2026-07-11 02:00:00", "target": "Llt_phys", "n": "100", "r2": ".91", "rmse": ".2", "mape_pct": "1.2", "p90_ape_pct": "2.1", "slice": "global"},
    ])

    front_fields = {
        "N1_main": 7, "N2_main": 35, "N2_side": 25, "l1": 60, "l2": 350, "h1": 420, "w1": 400,
        "n_core_group": 4, "core_plate_t": 20, "cw1": 5, "gap1": 3, "cw2": 1, "gap2": 1,
        "nwh1": 350, "nwh2": 300, "cc_w2c_space_x": 42, "cc_w2c_space_y": 42,
        "w2c_w1c_space_x": 41, "w2c_w1c_space_y": 41, "w1c_w2s_gap_x_actual": 43,
        "w1s_cs_space_x": 44, "cs_w1s_space_y": 45, "h_gap2": 46,
        "pred_Llt_phys": 27.5, "sigma_Llt_phys": .1,
        "B_design_analytic_T": 1.0, "pred_B_mean_core": .8,
        "pred_B_max_core": 2.7,
        "size_W_mm": 900, "size_L_mm": 700, "size_H_mm": 500,
        "size_WxLxH_mm": "900.0 × 700.0 × 500.0",
        "footprint_cm2": 6300, "turns_primary": 7,
        "turns_secondary_center": 35, "turns_secondary_side": 25,
        "cw1_conductor_thickness_mm": 5, "cw2_conductor_thickness_mm": 1,
        "core_depth_each_mm": 70, "wcp_len_pct": 50, "wcp_len_x_mm": 140,
        "core_cold_plate_thickness_mm": 20,
        "core_thermal_pad_thickness_mm": 2,
        "winding_cold_plate_thickness_mm": 20,
        "winding_thermal_pad_thickness_mm": 2,
        "pred_primary_winding_loss_W": 2000,
        "pred_secondary_center_winding_loss_W": 1200,
        "pred_secondary_side_winding_loss_W": 800,
        "pred_secondary_winding_loss_W": 2000,
        "pred_total_winding_loss_W": 4000,
        "pred_core_loss_W": 2000, "pred_core_cold_plate_loss_W": 500,
        "pred_winding_cold_plate_loss_W": 250,
        "rated_power_W": 1_000_000, "pred_efficiency_pct": 99.2,
        "Ae_effective_m2": .05, "core_lamination_factor": .85,
    }
    write_csv(root / "al_rounds" / "round_00" / "pareto_front.csv", [
        {**front_fields, "volume_L": 600, "total_loss_W": 7000},
    ])
    write_csv(root / "al_rounds" / "round_02" / "pareto_front.csv", [
        {**front_fields, "volume_L": 500, "total_loss_W": 8000},
        {**front_fields, "volume_L": 550, "total_loss_W": 6500, "pred_Llt_phys": 27.7},
    ])
    result = {
        **front_fields,
        "full_model": 0, "Llt": 13.75, "B_max_core": 1.0, "T_max_Tx": 90,
        "T_max_Rx_main": 91, "T_max_Rx_side": 92, "T_max_core": 88,
        "conv_error_pct_matrix": .5, "conv_error_pct_loss": .6,
        "P_winding_total": 4000, "P_core_total": 2000,
        "P_core_plate_total": 500, "P_wcp_total": 250,
        "time_matrix": 353.31, "time_loss": 1720.78,
        "time_thermal": 1039.83, "time": 3113.92,
        "git_hash": "a" * 40, "pyaedt_library_git_hash": "c" * 40,
    }
    write_json(root / "al_rounds" / "state.json", {
        "round": 2, "stage": "WAIT",
        "verification_counts": {"total": 1, "valid": 1, "pending": 0, "exhausted": 0, "ingested": 1},
        "task_records": {"0": {"active_id": 123, "last_status": "completed", "outcome": "valid", "result": result}},
        "history": [],
    })
    fine_result = {**result, "full_model": 1, "Llt_phys": 27.5, "volume_L": 500}
    write_json(root / "verify" / "results" / "final_verification.json", {
        "candidate_id": "r02-0000", "task_id": 999, "profile": "fine", "passed": True,
        "updated_at": "2026-07-11T02:50:00+09:00", "result": fine_result,
    })
    return root


@pytest.fixture
def artifact_service(campaign_root):
    return ArtifactService(
        campaign_root,
        scheduler=DummyScheduler(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )
