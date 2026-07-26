"""Prepare an eight-lane primary-temperature last-mile FEA hedge.

The source is the authenticated rank-1 N1=6/N2=60 candidate from the current
targeted symmetric batch.  Cooling boundaries, TIM, primary foil thickness and
primary interturn gap remain fixed.  The stencil spends unused envelope margin
on primary foil height and exchanges shorter Y-path length for thicker
unchanged-material winding cooling plates.

Preparation is local-only.  The emitted plan reuses the targeted batch submit
and collect commands, and carries a conservative ``automatic_submission_recommended``
screen.  It never submits by itself.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
from typing import Any
import warnings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_targeted_symmetric_fea_batch as targeted  # noqa: E402


SCHEMA = "mft-goal-primary-temperature-lastmile-screen-v1"
DEFAULT_SOURCE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_cooler_thermal_hedge_v1\batch_plan.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_lastmile_v1"
)
SOURCE_GEOMETRY_SHA256 = (
    "656b673929bbf8b5bcef01fa74338c5d9033bac732308d68c24807e506c03776"
)
COUNT = 8
PRIORITY = 90


class LastMileError(RuntimeError):
    """The last-mile source, prediction, or execution contract drifted."""


def _candidate_pool(
    base: dict[str, Any],
    existing_geometry_sha256: set[str],
) -> list[dict[str, Any]]:
    warnings.filterwarnings(
        "ignore",
        message="DataFrame is highly fragmented",
        category=Warning,
    )
    # Re-evaluate through validation below; this direct decoded lookup only
    # establishes the exact 79/80% contact-length targets.
    _ok, base_frame = validation_check(
        create_input_parameter(base), strict=False
    )
    if not _ok:
        raise LastMileError("source rank-1 params are not valid")
    base_decoded = base_frame.iloc[0].to_dict()
    reference_plate_length = float(base_decoded["wcp_len_ref_x"])

    # Each (plate thickness, w1 reduction) pair keeps L <= 1000 mm.  Reducing
    # w1 shortens primary/secondary mean turn length while making room for a
    # thicker aluminum winding plate; neither fan nor TIM is changed.
    plate_path_pairs = (
        (float(base["wcp_t"]), 0.0),
        (float(base["wcp_t"]), 4.0),
        (float(base["wcp_t"]), 8.0),
        (float(base["wcp_t"]), 12.0),
        (18.0, 8.0),
        (18.0, 12.0),
        (20.0, 16.0),
        (20.0, 20.0),
        (22.0, 24.0),
        (22.0, 28.0),
        (24.0, 32.0),
        (26.0, 40.0),
    )
    baseline_contact_fraction = (
        float(base["wcp_len_x"]) / reference_plate_length
    )
    contact_fractions = (
        baseline_contact_fraction,
        0.79,
        0.80,
    )
    raw: list[dict[str, Any]] = []
    seen: set[str] = set()
    for height_delta in (4.0, 6.0):
        for plate_thickness, w1_reduction in plate_path_pairs:
            for contact_fraction in contact_fractions:
                params = dict(base)
                params.update(
                    {
                        "h1": round(float(base["h1"]) + height_delta, 1),
                        "nwh1": round(
                            float(base["nwh1"]) + height_delta, 1
                        ),
                        "w1": round(
                            float(base["w1"]) - w1_reduction, 1
                        ),
                        "wcp_t": plate_thickness,
                        "wcp_len_x": round(
                            reference_plate_length * contact_fraction, 1
                        ),
                    }
                )
                try:
                    valid, decoded_frame = validation_check(
                        create_input_parameter(params), strict=False
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if not valid:
                    continue
                decoded = {
                    key: targeted._builtin(value)  # noqa: SLF001
                    for key, value in decoded_frame.iloc[0].to_dict().items()
                }
                volume, dimensions = geometry_metrics.bounding_box_lit(
                    decoded
                )
                if any(
                    float(value) > targeted.SIZE_LIMITS_MM[axis] + 1e-9
                    for axis, value in zip(("W", "L", "H"), dimensions)
                ):
                    continue
                if (
                    int(decoded["N1"]) != 6
                    or int(decoded["N2"]) != 60
                    or float(decoded["cw1"]) != 5.0
                    or float(decoded["gap1"]) != 1.6
                    or float(decoded["fan_velocity"]) != 1.5
                    or str(decoded["fan_config"]) != "dual"
                    or float(decoded["core_plate_pad_t"]) != 2.0
                    or float(decoded["wcp_pad_t"]) != 2.0
                    or float(decoded["k_ins"]) != 0.2
                ):
                    raise LastMileError("fixed topology/cooling/TIM escaped")
                geometry_sha = targeted._geometry_sha(decoded)  # noqa: SLF001
                projected = {
                    key: targeted._builtin(decoded[key])  # noqa: SLF001
                    for key in sorted(ALL_INPUT_KEYS)
                }
                params_sha = targeted._sha(projected)  # noqa: SLF001
                if (
                    geometry_sha in existing_geometry_sha256
                    or geometry_sha in seen
                ):
                    continue
                seen.add(geometry_sha)
                raw.append(
                    {
                        "decoded": decoded,
                        "params": projected,
                        "params_sha256": params_sha,
                        "physical_geometry_sha256": geometry_sha,
                        "dimensions": [float(item) for item in dimensions],
                        "volume_L": float(volume),
                        "mutations": {
                            "primary_height_extension_mm": height_delta,
                            "w1_path_reduction_mm": w1_reduction,
                            "wcp_t_mm": plate_thickness,
                            "wcp_len_x_mm": projected["wcp_len_x"],
                            "wcp_contact_fraction": contact_fraction,
                        },
                    }
                )
    if len(raw) < COUNT:
        raise LastMileError(
            f"only {len(raw)} distinct valid last-mile candidates exist"
        )
    return raw


def prepare(
    *,
    source_plan_path: Path,
    output: Path,
    priority: int = PRIORITY,
) -> Path:
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or not 0 <= priority <= 100
    ):
        raise LastMileError("priority must be an integer in 0..100")
    source_plan, source_root, profile = targeted._load_plan(  # noqa: SLF001
        source_plan_path
    )
    source_lanes = [
        lane
        for lane in source_plan["lanes"]
        if lane["candidate"]["physical_geometry_sha256"]
        == SOURCE_GEOMETRY_SHA256
    ]
    if len(source_lanes) != 1:
        raise LastMileError("exact rank-1 source geometry is not unique")
    source_lane = source_lanes[0]
    if int(source_lane["rank"]) != 1:
        raise LastMileError("expected source geometry is not rank 1")
    base_path = (
        source_root / source_lane["params"]["path"]
    ).resolve(strict=True)
    base = targeted._read_json(base_path)  # noqa: SLF001
    existing = {
        lane["candidate"]["physical_geometry_sha256"]
        for lane in source_plan["lanes"]
    }
    raw = _candidate_pool(base, existing)

    import pandas as pd

    frame = pd.DataFrame([item["decoded"] for item in raw])
    means, upper, model_identity = targeted._n1_6_cooler_predictions(  # noqa: SLF001
        frame
    )
    calibration_ratio, calibration = (
        targeted._turn_graded_crx_calibration()  # noqa: SLF001
    )
    baseline = source_lane["candidate"]
    baseline_primary = float(
        baseline["screening_primary_winding_max_C"]
    )
    baseline_secondary = float(
        baseline["screening_secondary_winding_max_C"]
    )
    baseline_core = float(baseline["screening_core_max_C"])

    screened = []
    primary_improvers = []
    margin_preservers = []
    for index, item in enumerate(raw):
        primary = max(
            float(upper[name][index])
            for name in targeted.PRIMARY_WINDING_TEMPERATURES
        )
        secondary = max(
            float(upper[name][index])
            for name in targeted.SECONDARY_WINDING_TEMPERATURES
        )
        core = max(
            float(upper[name][index])
            for name in targeted.CORE_TEMPERATURES
        )
        legacy_crx = float(means["C_rx_rx_F"][index])
        corrected_crx = legacy_crx * calibration_ratio
        replay = targeted._fixed_lm_resonance(  # noqa: SLF001
            llt_uH=float(means["Llt_phys"][index]),
            c_tx_F=float(means["C_tx_tx_F"][index]),
            c_rx_F=corrected_crx,
            c_inter_F=float(means["C_tx_rx_F"][index]),
            n1=6,
            n2=60,
        )
        loss = (
            float(means["P_winding_total"][index])
            + float(means["P_core_total"][index])
        )
        summary = {
            **item,
            "primary_winding_robust_max_C": primary,
            "secondary_winding_robust_max_C": secondary,
            "core_robust_max_C": core,
            "primary_improvement_C": baseline_primary - primary,
            "corrected_Crx_transfer_estimate_F": corrected_crx,
            "fixed_lm2mh_replay": replay,
            "P_Tx_main_group_W": float(
                means["P_Tx_main_group"][index]
            ),
            "P_winding_total_W": float(
                means["P_winding_total"][index]
            ),
            "P_core_total_W": float(means["P_core_total"][index]),
            "P_total_screen_W": loss,
            "preserves_baseline_secondary_core_robust_margins": bool(
                secondary <= baseline_secondary + 1e-9
                and core <= baseline_core + 1e-9
            ),
        }
        screened.append(summary)
        # The broader set is still useful as a local-only contingency plan.
        # Automatic submission remains disabled unless every selected lane
        # also preserves the baseline secondary/core robust margins.
        if (
            replay["fmin_Hz"] >= targeted.RESONANCE_MIN_HZ
            and corrected_crx <= 0.55e-9
            and primary < baseline_primary
        ):
            primary_improvers.append(summary)
            if summary[
                "preserves_baseline_secondary_core_robust_margins"
            ]:
                margin_preservers.append(summary)
    if len(primary_improvers) < COUNT:
        raise LastMileError(
            "fewer than eight last-mile candidates improve primary temperature"
        )

    # Prefer exact margin preservers.  If fewer than eight exist, complete the
    # local-only plan with the smallest secondary/core regression instead of
    # blindly selecting the coldest-primary but overheated-core variants.
    selected = sorted(
        primary_improvers,
        key=lambda row: (
            not row[
                "preserves_baseline_secondary_core_robust_margins"
            ],
            max(
                0.0,
                row["secondary_winding_robust_max_C"]
                - baseline_secondary,
            ),
            max(0.0, row["core_robust_max_C"] - baseline_core),
            row["primary_winding_robust_max_C"],
            row["core_robust_max_C"],
            row["P_total_screen_W"],
            row["physical_geometry_sha256"],
        ),
    )[:COUNT]
    selected_primary = [
        float(row["primary_winding_robust_max_C"]) for row in selected
    ]
    safe_to_submit = bool(
        all(
            row["preserves_baseline_secondary_core_robust_margins"]
            for row in selected
        )
        and
        min(selected_primary) <= 100.5
        and statistics.median(selected_primary) <= 101.5
        and max(selected_primary) <= baseline_primary - 0.5
    )

    destination = output.resolve()
    if destination.exists():
        raise LastMileError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    profile_path = targeted._atomic_json(  # noqa: SLF001
        destination / "goal_standard.json", profile
    )
    lanes = []
    for rank, item in enumerate(selected, start=1):
        geometry_sha = item["physical_geometry_sha256"]
        stem = geometry_sha[:12]
        params_path = targeted._atomic_json(  # noqa: SLF001
            destination / "params" / f"rank-{rank:02d}-{stem}.json",
            item["params"],
        )
        name = f"mft-goal-txlast-r{rank:02d}-{stem}"
        identity = scheduler_client.verification_submission_identity(
            name,
            item["params"],
            profile,
            source_plan["solver_revision"],
            source_plan["library_revision"],
        )
        retained = scheduler_client.retained_aedt_identity(
            name,
            item["params"],
            profile,
            source_plan["solver_revision"],
            source_plan["library_revision"],
        )
        screen = {
            **item["fixed_lm2mh_replay"],
            "C_tx_tx_F": float(
                means["C_tx_tx_F"][
                    next(
                        index
                        for index, raw_item in enumerate(raw)
                        if raw_item["physical_geometry_sha256"] == geometry_sha
                    )
                ]
            ),
            "C_rx_rx_F": item["corrected_Crx_transfer_estimate_F"],
            "C_rx_rx_corrected_turn_graded_transfer_estimate_F": item[
                "corrected_Crx_transfer_estimate_F"
            ],
            "primary_winding_robust_max_C": item[
                "primary_winding_robust_max_C"
            ],
            "secondary_winding_robust_max_C": item[
                "secondary_winding_robust_max_C"
            ],
            "core_robust_max_C": item["core_robust_max_C"],
            "P_Tx_main_group_W": item["P_Tx_main_group_W"],
            "P_winding_total_W": item["P_winding_total_W"],
            "P_core_total_W": item["P_core_total_W"],
            "P_total_screen_W": item["P_total_screen_W"],
        }
        lanes.append(
            {
                "rank": rank,
                "selection_role": "minimum_primary_temperature_lastmile",
                "candidate": {
                    "physical_geometry_sha256": geometry_sha,
                    "params_sha256": item["params_sha256"],
                    "effective_params_sha256": targeted._sha(  # noqa: SLF001
                        identity["merged"]
                    ),
                    "raw_row_sha256": targeted._sha(  # noqa: SLF001
                        {
                            "source_geometry": SOURCE_GEOMETRY_SHA256,
                            "params": item["params_sha256"],
                            "screen": screen,
                        }
                    ),
                    "source": copy.deepcopy(
                        source_lane["candidate"]["source"]
                    ),
                    "dimensions_mm": {
                        "W_drawing_x": item["dimensions"][0],
                        "L_perpendicular_y": item["dimensions"][1],
                        "H": item["dimensions"][2],
                    },
                    "objective_volume_L": item["volume_L"],
                    "objective_total_loss_W": item["P_total_screen_W"],
                    "normalized_constraint_violation_l2": max(
                        0.0,
                        (
                            item["primary_winding_robust_max_C"] - 100.0
                        )
                        / 10.0,
                    ),
                    "fixed_lm2mh_screen": copy.deepcopy(
                        item["fixed_lm2mh_replay"]
                    ),
                    "surrogate_screen": screen,
                    "turn_graded_Crx_calibration": copy.deepcopy(calibration),
                    "screening_primary_winding_max_C": item[
                        "primary_winding_robust_max_C"
                    ],
                    "screening_secondary_winding_max_C": item[
                        "secondary_winding_robust_max_C"
                    ],
                    "screening_winding_max_C": max(
                        item["primary_winding_robust_max_C"],
                        item["secondary_winding_robust_max_C"],
                    ),
                    "screening_core_max_C": item["core_robust_max_C"],
                    "neighborhood_mutations": copy.deepcopy(
                        item["mutations"]
                    ),
                    "surrogate_model_identity": {
                        key: copy.deepcopy(model_identity[key])
                        for key in (
                            "generation",
                            "train_report_sha256",
                            "dataset_sha256",
                            "evaluation_model_sha256",
                        )
                    },
                },
                "neighborhood_vector_normalized": [
                    item["mutations"]["primary_height_extension_mm"] / 6.0,
                    item["mutations"]["w1_path_reduction_mm"] / 40.0,
                    item["mutations"]["wcp_t_mm"] / 26.0,
                    item["mutations"]["wcp_contact_fraction"],
                ],
                "params": targeted._file_record(  # noqa: SLF001
                    params_path, relative_to=destination
                ),
                "params_sha256": item["params_sha256"],
                "scheduler": {
                    "project": scheduler_client.MFT_PROJECT,
                    "name": name,
                    "workdir": f"mft_goal_txlast_r{rank:02d}_{stem}",
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": targeted._sha(  # noqa: SLF001
                        identity["merged"]
                    ),
                    "cpus": targeted.CPUS,
                    "memory_mb": targeted.MEMORY_MB,
                    "timeout_seconds": targeted.TIMEOUT_SECONDS,
                    "max_workers_per_node": targeted.MAX_WORKERS_PER_NODE,
                    "priority": priority,
                    "aedt_backend": "standalone",
                    "environment": targeted._core_environment(  # noqa: SLF001
                        source_plan["solver_revision"]
                    ),
                    "retained_aedt": retained,
                    "retained_aedt_required": False,
                },
            }
        )

    plan = targeted._seal(  # noqa: SLF001
        {
            "schema_version": targeted.BATCH_PLAN_SCHEMA,
            "campaign_id": targeted.BATCH_CAMPAIGN_ID,
            "created_at_utc": targeted._now(),  # noqa: SLF001
            "authority": copy.deepcopy(source_plan["authority"]),
            "hard_spec": copy.deepcopy(source_plan["hard_spec"]),
            "air_gap_attestation": copy.deepcopy(
                source_plan["air_gap_attestation"]
            ),
            "selection": {
                "source_batch_plan": targeted._file_record(  # noqa: SLF001
                    source_plan_path
                ),
                "source_batch_payload_sha256": source_plan["payload_sha256"],
                "source_rank": 1,
                "source_geometry_sha256": SOURCE_GEOMETRY_SHA256,
                "selected_candidate_count": COUNT,
                "minimum_count": targeted.MIN_BATCH,
                "maximum_count": targeted.MAX_BATCH,
                "stencil": (
                    "primary height +4/+6mm; shorter Y path exchanged for "
                    "baseline/thicker unchanged-material winding plate; "
                    "plate contact length baseline/79/80%; cooling/TIM fixed"
                ),
                "baseline_primary_robust_max_C": baseline_primary,
                "baseline_secondary_robust_max_C": baseline_secondary,
                "baseline_core_robust_max_C": baseline_core,
                "selected_primary_robust_max_C": selected_primary,
                "margin_preserving_candidate_count": len(
                    margin_preservers
                ),
                "automatic_submission_recommended": safe_to_submit,
                "automatic_submission_rule": (
                    "best<=100.5C; median<=101.5C; worst improves >=0.5C; "
                    "all preserve baseline secondary/core robust margins"
                ),
                "existing_16_geometry_hashes_excluded": True,
            },
            "solver_revision": source_plan["solver_revision"],
            "library_revision": source_plan["library_revision"],
            "profile": targeted._file_record(  # noqa: SLF001
                profile_path, relative_to=destination
            ),
            "profile_canonical_sha256": targeted._sha(profile),  # noqa: SLF001
            "scheduler_priority": priority,
            "lanes": lanes,
            "submission_ready": True,
            "explicit_apply_required": True,
            "automatic_submission_enabled": False,
            "scheduler_repository_modified": False,
            "scheduler_project_configuration_modified": False,
            "scheduler_submission_performed": False,
            "classification": "symmetric-unrounded-FEA-acquisition-only",
            "hard_label": (
                "core_center_gap_mm=0; pre-gap acquisition only; "
                "Lm=2mH is not physically attested"
            ),
            "production_eligible": False,
        }
    )
    plan_path = targeted._atomic_json(  # noqa: SLF001
        destination / "batch_plan.json", plan
    )
    dry_run_path = targeted.dry_run(
        plan_path=plan_path, output=destination / "dry_run.json"
    )
    report_rows = [
        {
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key not in {"decoded", "params"}
        }
        for row in sorted(
            screened,
            key=lambda item: (
                item["primary_winding_robust_max_C"],
                item["physical_geometry_sha256"],
            ),
        )
    ]
    targeted._atomic_json(  # noqa: SLF001
        destination / "screen_report.json",
        targeted._seal(  # noqa: SLF001
            {
                "schema_version": SCHEMA,
                "created_at_utc": targeted._now(),  # noqa: SLF001
                "source_plan": targeted._file_record(source_plan_path),  # noqa: SLF001
                "source_plan_payload_sha256": source_plan["payload_sha256"],
                "plan": targeted._file_record(plan_path),  # noqa: SLF001
                "plan_payload_sha256": plan["payload_sha256"],
                "dry_run": targeted._file_record(dry_run_path),  # noqa: SLF001
                "candidate_pool_count": len(screened),
                "primary_improver_count": len(primary_improvers),
                "margin_preserving_candidate_count": len(
                    margin_preservers
                ),
                "selected_count": len(selected),
                "automatic_submission_recommended": safe_to_submit,
                "selected_geometry_sha256": [
                    item["physical_geometry_sha256"] for item in selected
                ],
                "rows": report_rows,
                "scheduler_mutation_performed": False,
            }
        ),
    )
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--priority", type=int, default=PRIORITY)
    return parser


def main() -> int:
    args = _parser().parse_args()
    path = prepare(
        source_plan_path=args.source_plan,
        output=args.output,
        priority=args.priority,
    )
    print(json.dumps({"status": "ok", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
