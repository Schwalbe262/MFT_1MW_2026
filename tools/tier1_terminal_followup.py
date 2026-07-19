"""Fail-closed terminal analysis for authenticated Tier-1 NSGA searches.

This command is deliberately read-only with respect to the scheduler.  It
authenticates every terminal seed result, cross-checks feasible Pareto rows
against the same terminal population's exact ``constraint_G`` evidence, and
either identifies canary-eligible designs or prepares a repaired-on-load warm
start for the next four seeds.  It never submits FEA and never promotes a
model or design to production.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tier1_resonance_feedback import (  # noqa: E402
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    SEARCH_SCHEMA,
    _atomic_bytes,
    _atomic_json,
    _json_sha,
    _now,
    _read_json,
    _sha256,
    optimizer_constraint_normalization,
)


TERMINAL_ANALYSIS_SCHEMA = "mft-tier1-terminal-followup-v1"
WARM_CONTRACT_SCHEMA = "mft-tier1-next-warm-start-v1"
CONSTRAINT_TOLERANCE = 1e-9
MEASUREMENT_GEOMETRY_CONSTRAINTS = (
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "minimum_physical_insulation",
    "core_group_manufacturability_limit",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)


def _git_revision(root: Path) -> str:
    root = root.resolve()
    process = subprocess.run(
        [
            "git", "-c", f"safe.directory={root.as_posix()}",
            "-C", str(root), "rev-parse", "HEAD",
        ],
        check=True, capture_output=True, text=True,
    )
    return process.stdout.strip()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{label} is not finite") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _load_sobol_schema(nsga_code_root: Path):
    nsga_code_root = nsga_code_root.resolve()
    if _git_revision(nsga_code_root) != EXPECTED_NSGA_REVISION:
        raise RuntimeError("NSGA code revision drifted")
    regression = nsga_code_root / "regression_260707"
    for path in (str(nsga_code_root), str(regression)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from module.input_parameter_260706 import (
        N1_MAX_TURNS,
        N1_MIN_TURNS,
        _SOBOL_DIMS,
        get_drawing_default_params,
    )

    return (
        tuple(_SOBOL_DIMS), int(N1_MIN_TURNS), int(N1_MAX_TURNS),
        get_drawing_default_params(),
    )


def _integer_unit(target: int, minimum: int, maximum: int) -> float:
    scale = maximum - minimum + 0.9999
    return float(np.clip((target - minimum + 0.5) / scale, 0.0, 1.0))


def decoded_to_unit(
    params: dict, sobol_dims: tuple, n1_min: int, n1_max: int,
    defaults: dict,
) -> np.ndarray:
    """Invert the documented decoder sufficiently for repaired warm starts."""
    n1 = int(params["N1"])
    n2 = int(params["N2"])
    n2_side = int(params["N2_side"])
    l1 = _finite(params["l1"], "l1")
    l2 = _finite(params["l2"], "l2")
    h1 = _finite(params["h1"], "h1")
    w1 = _finite(params["w1"], "w1")
    core_plate_t = _finite(params["core_plate_t"], "core_plate_t")
    core_pad_t = _finite(
        params.get("core_plate_pad_t", defaults["core_plate_pad_t"]),
        "core_plate_pad_t",
    )
    core_stack_t = core_plate_t + 2.0 * core_pad_t
    depth_min = _finite(
        params.get("core_depth_min", defaults["core_depth_min"]),
        "core_depth_min",
    )
    depth_max = _finite(
        params.get("core_depth_max", defaults["core_depth_max"]),
        "core_depth_max",
    )
    n_min = max(1, math.ceil((w1 - core_stack_t) / (depth_max + core_stack_t)))
    n_max = max(n_min, math.floor((w1 - core_stack_t) / (depth_min + core_stack_t)))
    groups = int(params["n_core_group"])
    group_unit = _integer_unit(groups, n_min, n_max)
    spaces = sum(
        _finite(params[name], name)
        for name in (
            "cc_w2c_space_x", "w2c_w1c_space_x",
            "w1c_w2s_space_x", "w1s_cs_space_x",
        )
    )
    budget = l2 - spaces
    n1_gaps = max(int(params["N1_main"]) - 1, 0)
    n1_gaps += max(int(params["N1_side"]) - 1, 0)
    primary_pack = n1 * _finite(params["cw1"], "cw1")
    primary_pack += n1_gaps * _finite(params["gap1"], "gap1")
    f1_split = primary_pack / budget if budget > 0.0 else 0.5
    physical = {
        "u_N1": _integer_unit(n1, n1_min, n1_max),
        "u_N1_side": 0.0,
        "u_N2_side": float(np.clip((n2_side + 0.1) / (0.8 * n2), 0.0, 1.0)),
        "l1": l1,
        "total_length": 2.0 * l2 + 4.0 * l1,
        "total_height": h1 + 2.0 * l1,
        "w1": w1,
        "u_ngroup": group_unit,
        "wcp_t": _finite(params["wcp_t"], "wcp_t"),
        "core_plate_t": core_plate_t,
        "f1_split": f1_split,
        "gap1": _finite(params["gap1"], "gap1"),
        "gap2": _finite(params["gap2"], "gap2"),
        "wh1": _finite(params["nwh1"], "nwh1") / h1,
        "wh2": _finite(params["nwh2"], "nwh2") / h1,
        "wcp_len_pct": _finite(params["wcp_len_pct"], "wcp_len_pct"),
    }
    for name in (
        "cc_w2c_space_x", "cc_w2c_space_y",
        "w2c_w1c_space_x", "w2c_w1c_space_y",
        "w1c_w2s_space_x", "w1s_cs_space_x", "cs_w1s_space_y",
        "w2s_w1s_space_x", "w1s_w2s_space_y",
    ):
        physical[name] = _finite(params[name], name)
    coordinates = []
    for name, lower, upper in sobol_dims:
        value = _finite(physical[name], name)
        coordinates.append(float(np.clip((value - lower) / (upper - lower), 0.0, 1.0)))
    return np.asarray(coordinates, dtype=float)


def _validate_result(result: dict, launch: dict, seed: int) -> None:
    if (
        result.get("schema_version") != SEARCH_SCHEMA
        or int(result.get("seed", -1)) != seed
        or result.get("model_manifest_sha256")
        != launch.get("model_manifest_sha256")
        or result.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or result.get("hard_spec") != HARD_SPEC
        or result.get("production_eligible") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError(f"terminal search authentication failed: seed {seed}")
    plan = result.get("next_target_fea_batch_plan")
    if (
        not isinstance(plan, dict)
        or plan.get("submission_performed") is not False
        or plan.get("current_candidates_eligible_for_submission") is not False
        or plan.get("production_eligible") is not False
        or plan.get("fea_submission_approved") is not False
    ):
        raise RuntimeError(f"terminal next-plan fail-closed contract drifted: {seed}")


def _positive_violation(constraints: dict) -> tuple[float, int, float]:
    values = [_finite(value, name) for name, value in sorted(constraints.items())]
    positive = [max(value, 0.0) for value in values]
    return sum(positive), sum(value > CONSTRAINT_TOLERANCE for value in values), max(positive)


def _normalized_positive_violation(constraints: dict) -> tuple[float, float]:
    names = tuple(str(name) for name in constraints)
    contract = optimizer_constraint_normalization(names, HARD_SPEC)
    positive = {
        name: max(_finite(constraints[name], name), 0.0)
        / float(contract["scales"][name])
        for name in names
    }
    return sum(positive.values()), max(positive.values(), default=0.0)


def _measurement_metadata(row: dict) -> dict:
    """Separate active-learning eligibility from production feasibility."""
    constraints = row["constraint_G"]
    missing = [name for name in MEASUREMENT_GEOMETRY_CONSTRAINTS if name not in constraints]
    if missing:
        raise RuntimeError(f"measurement geometry constraints missing: {missing}")
    geometry_violation = max(
        max(_finite(constraints[name], name), 0.0)
        for name in MEASUREMENT_GEOMETRY_CONSTRAINTS
    )
    predictions = row.get("target_predictions")
    uncertainty = row.get("target_conformal_half_width")
    resonance = row.get("derived_resonance")
    if not all(isinstance(value, dict) for value in (predictions, uncertainty, resonance)):
        raise RuntimeError("measurement acquisition target evidence is missing")
    llt_mean = _finite(predictions.get("Llt_phys"), "measurement Llt mean")
    llt_half_width = _finite(
        uncertainty.get("Llt_phys"), "measurement Llt half width"
    )
    resonance_mean = _finite(
        resonance.get("f_res_min_screen_Hz"), "measurement resonance mean"
    )
    llt_mean_error = abs(llt_mean - float(HARD_SPEC["Llt_target_uH"]))
    resonance_mean_violation = max(
        float(HARD_SPEC["resonance_min_Hz"]) - resonance_mean, 0.0
    )
    density_violation = max(
        _finite(constraints.get("strict_full_density_support", 0.0), "density G"),
        0.0,
    )
    # Lower is better.  Geometry remains a hard measurement gate.  Mean Llt
    # and resonance drive target proximity, while a bounded uncertainty reward
    # prefers points with greater expected information gain.  This score never
    # changes production feasibility.
    acquisition_score = (
        llt_mean_error
        + resonance_mean_violation / 1_000.0
        + 0.1 * density_violation
        - 0.05 * min(llt_half_width, 10.0)
    )
    return {
        "measurement_geometry_passed": (
            geometry_violation <= CONSTRAINT_TOLERANCE
        ),
        "measurement_geometry_max_positive_G": geometry_violation,
        "measurement_Llt_mean_error_uH": llt_mean_error,
        "measurement_Llt_conformal_half_width_uH": llt_half_width,
        "measurement_resonance_mean_Hz": resonance_mean,
        "measurement_resonance_mean_violation_Hz": resonance_mean_violation,
        "measurement_density_positive_G": density_violation,
        "measurement_acquisition_score": acquisition_score,
        "measurement_eligible": geometry_violation <= CONSTRAINT_TOLERANCE,
        "measurement_purpose": (
            "target-specific Llt/k/capacitance uncertainty reduction"
        ),
    }


def _select_diverse_warm(rows: list[dict], coordinates: np.ndarray, limit: int):
    if not rows:
        return [], np.empty((0, coordinates.shape[1]), dtype=float)
    limit = min(limit, len(rows))
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index]["normalized_total_positive_constraint_violation"],
            rows[index]["total_positive_constraint_violation"],
            rows[index]["constraint_violation_count"],
            rows[index]["decoded_params_sha256"],
        ),
    )
    selected = [order[0]]
    while len(selected) < limit:
        remaining = [index for index in order if index not in selected]
        if not remaining:
            break
        distances = np.asarray([
            min(float(np.linalg.norm(coordinates[index] - coordinates[chosen])) for chosen in selected)
            for index in remaining
        ])
        violations = np.asarray([
            rows[index]["normalized_total_positive_constraint_violation"]
            for index in remaining
        ])
        denominator = max(float(np.max(violations)), 1e-12)
        score = distances - 0.1 * violations / denominator
        selected.append(remaining[int(np.argmax(score))])
    return [rows[index] for index in selected], coordinates[selected]


def analyze_terminal(
    *, launch_path: Path, status_path: Path, nsga_code_root: Path,
    output: Path, warm_limit: int = 64,
) -> dict:
    launch_path = launch_path.resolve()
    status_path = status_path.resolve()
    output = output.resolve()
    launch = _read_json(launch_path)
    status = _read_json(status_path)
    if (
        status.get("launch_sha256") != _sha256(launch_path)
        or status.get("model_manifest_sha256")
        != launch.get("model_manifest_sha256")
        or status.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
    ):
        raise RuntimeError("launch/status authentication failed")
    states = {job.get("state") for job in status.get("jobs", [])}
    if states != {"completed"} or len(status["jobs"]) != len(launch["jobs"]):
        raise RuntimeError("all launched seeds must be terminal-completed")
    status_by_seed = {int(job["seed"]): job for job in status["jobs"]}
    constraint_minima: dict[str, list[float]] = defaultdict(list)
    near_by_digest: dict[str, dict] = {}
    fully_valid_by_digest: dict[str, dict] = {}
    result_references = []
    for launched in launch["jobs"]:
        seed = int(launched["seed"])
        result_path = Path(launched["output"]).resolve() / "result.json"
        result_sha = _sha256(result_path)
        status_job = status_by_seed[seed]
        if status_job.get("result_sha256") != result_sha:
            raise RuntimeError(f"monitor result SHA mismatch: {seed}")
        result = _read_json(result_path)
        _validate_result(result, launch, seed)
        result_references.append({
            "seed": seed, "path": str(result_path), "sha256": result_sha,
            "completed_generations": int(result["completed_generations"]),
            "feasible_pareto_count": int(result["feasible_pareto_count"]),
        })
        for name, value in result["constraint_minimum_G"].items():
            constraint_minima[name].append(_finite(value, name))
        next_rows = result["next_target_fea_batch_plan"]["candidates"]
        next_by_digest = {}
        for row in next_rows:
            digest = str(row.get("decoded_params_sha256") or "")
            if digest != _json_sha(row.get("decoded_params")):
                raise RuntimeError(f"decoded parameter SHA mismatch: {seed}")
            if digest in next_by_digest:
                raise RuntimeError(f"duplicate next-plan geometry: {seed}")
            total, count, maximum = _positive_violation(row["constraint_G"])
            normalized_total, normalized_maximum = (
                _normalized_positive_violation(row["constraint_G"])
            )
            normalized = {
                **row,
                "source_seed": seed,
                "total_positive_constraint_violation": total,
                "constraint_violation_count": count,
                "maximum_positive_constraint_violation": maximum,
                "normalized_total_positive_constraint_violation": (
                    normalized_total
                ),
                "normalized_maximum_positive_constraint_violation": (
                    normalized_maximum
                ),
                "canary_eligible": False,
                "production_eligible": False,
                "fea_submission_approved": False,
            }
            normalized.update(_measurement_metadata(normalized))
            next_by_digest[digest] = normalized
            incumbent = near_by_digest.get(digest)
            if incumbent is None or (
                normalized_total, total, count, seed
            ) < (
                incumbent["normalized_total_positive_constraint_violation"],
                incumbent["total_positive_constraint_violation"],
                incumbent["constraint_violation_count"],
                incumbent["source_seed"],
            ):
                near_by_digest[digest] = normalized
        pareto_rows = result.get("candidates") or []
        if len(pareto_rows) != int(result["feasible_pareto_count"]):
            raise RuntimeError(f"feasible Pareto count drifted: {seed}")
        matched = 0
        for candidate in pareto_rows:
            digest = _json_sha(candidate["decoded_params"])
            evidence = next_by_digest.get(digest)
            if evidence is None:
                continue
            if any(
                _finite(value, name) > CONSTRAINT_TOLERANCE
                for name, value in evidence["constraint_G"].items()
            ):
                raise RuntimeError(f"Pareto candidate has positive constraint: {seed}")
            matched += 1
            eligible = {
                **evidence,
                "pareto_index": int(candidate["index"]),
                "volume_L": _finite(candidate["volume_L"], "volume_L"),
                "total_loss_W": _finite(candidate["total_loss_W"], "total_loss_W"),
                "predictions": candidate["predictions"],
                "conformal_half_width": candidate["conformal_half_width"],
                "derived_resonance": candidate["derived_resonance"],
                "canary_eligible": True,
                "production_eligible": False,
                "fea_submission_approved": False,
            }
            incumbent = fully_valid_by_digest.get(digest)
            if incumbent is None or (seed, int(candidate["index"])) < (
                incumbent["source_seed"], incumbent["pareto_index"],
            ):
                fully_valid_by_digest[digest] = eligible
        if pareto_rows and matched == 0:
            raise RuntimeError(f"no Pareto row has terminal constraint evidence: {seed}")

    near_rows = sorted(
        near_by_digest.values(),
        key=lambda row: (
            row["normalized_total_positive_constraint_violation"],
            row["total_positive_constraint_violation"],
            row["constraint_violation_count"],
            row["decoded_params_sha256"],
        ),
    )
    valid_rows = sorted(
        fully_valid_by_digest.values(),
        key=lambda row: (
            row["total_loss_W"], row["volume_L"],
            row["source_seed"], row["decoded_params_sha256"],
        ),
    )
    measurement_rows = sorted(
        (row for row in near_rows if row["measurement_eligible"]),
        key=lambda row: (
            row["measurement_acquisition_score"],
            -row["measurement_Llt_conformal_half_width_uH"],
            row["source_seed"], row["decoded_params_sha256"],
        ),
    )
    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(nsga_code_root)
    coordinates = np.vstack([
        decoded_to_unit(row["decoded_params"], sobol_dims, n1_min, n1_max, defaults)
        for row in near_rows
    ]) if near_rows else np.empty((0, len(sobol_dims)), dtype=float)
    warm_rows, inverse_warm_coordinates = _select_diverse_warm(
        near_rows, coordinates, int(warm_limit),
    )
    inherited_coordinates = np.empty((0, len(sobol_dims)), dtype=float)
    inherited_reference = None
    source_warm = launch.get("warm_start")
    if source_warm is not None:
        source_warm_path = Path(source_warm["path"]).resolve()
        if _sha256(source_warm_path) != source_warm.get("sha256"):
            raise RuntimeError("source warm-start SHA drifted")
        inherited_coordinates = np.asarray(
            np.load(source_warm_path, allow_pickle=False), dtype=float,
        )
        if (
            inherited_coordinates.ndim != 2
            or inherited_coordinates.shape[1] != len(sobol_dims)
            or not np.isfinite(inherited_coordinates).all()
        ):
            raise RuntimeError("source warm-start coordinate schema drifted")
        inherited_reference = {
            "path": str(source_warm_path),
            "sha256": source_warm["sha256"],
            "coordinate_count": len(inherited_coordinates),
        }
    combined_coordinates = np.vstack([
        inverse_warm_coordinates, inherited_coordinates,
    ])
    unique_indices = []
    seen_coordinates = set()
    for index, coordinate in enumerate(combined_coordinates):
        identity = tuple(np.round(coordinate, 12))
        if identity in seen_coordinates:
            continue
        seen_coordinates.add(identity)
        unique_indices.append(index)
    warm_coordinates = combined_coordinates[unique_indices[:int(warm_limit)]]
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, warm_coordinates, allow_pickle=False)
    _atomic_bytes(warm_path, buffer.getvalue())
    next_seed_start = max(int(job["seed"]) for job in launch["jobs"]) + 1
    next_seeds = list(range(next_seed_start, next_seed_start + 4))
    warm_contract = {
        "schema_version": WARM_CONTRACT_SCHEMA,
        "created_at": _now(),
        "warm_start": {
            "path": str(warm_path), "sha256": _sha256(warm_path),
            "shape": list(warm_coordinates.shape),
            "coordinate_contract": "inverse_decoded_then_current_problem_repair_v1",
        },
        "inherited_warm_start": inherited_reference,
        "inverse_near_candidate_coordinate_count": len(inverse_warm_coordinates),
        "source_launch_sha256": _sha256(launch_path),
        "source_status_sha256": _sha256(status_path),
        "source_result_sha256": [row["sha256"] for row in result_references],
        "selected_decoded_params_sha256": [
            row["decoded_params_sha256"] for row in warm_rows
        ],
        "optimizer_constraint_normalization": (
            optimizer_constraint_normalization(
                tuple(near_rows[0]["constraint_G"]), HARD_SPEC,
            ) if near_rows else None
        ),
        "prepared_next_seeds": next_seeds,
        "population": int(launch["population"]),
        "max_generations": int(launch["max_generations"]),
        "launch_performed": False,
        "production_eligible": False,
        "fea_submission_approved": False,
    }
    _atomic_json(output / "next_warm_start_contract.json", warm_contract)
    llt_half_widths = [
        _finite(row["target_conformal_half_width"]["Llt_phys"], "Llt half width")
        for row in near_rows
        if isinstance(row.get("target_conformal_half_width"), dict)
        and row["target_conformal_half_width"].get("Llt_phys") is not None
    ]
    minimum_llt_half_width = min(llt_half_widths) if llt_half_widths else None
    llt_uncertainty_floor_G = (
        minimum_llt_half_width - float(HARD_SPEC["Llt_tol_uH"])
        if minimum_llt_half_width is not None else None
    )
    llt_robust_band_structurally_infeasible = (
        llt_uncertainty_floor_G is not None
        and llt_uncertainty_floor_G > CONSTRAINT_TOLERANCE
    )
    analysis = {
        "schema_version": TERMINAL_ANALYSIS_SCHEMA,
        "created_at": _now(),
        "launch_sha256": _sha256(launch_path),
        "status_sha256": _sha256(status_path),
        "model_manifest_sha256": launch["model_manifest_sha256"],
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "result_references": result_references,
        "constraint_minimum_G_across_seeds": {
            name: min(values) for name, values in sorted(constraint_minima.items())
        },
        "unique_near_candidate_count": len(near_rows),
        "near_candidates": near_rows[:64],
        "fully_valid_unique_count": len(valid_rows),
        "fully_valid_candidates": valid_rows,
        "first_canary_candidate": valid_rows[0] if valid_rows else None,
        "canary_submission_performed": False,
        "active_learning_measurement_unique_count": len(measurement_rows),
        "active_learning_measurement_candidates": measurement_rows[:32],
        "first_active_learning_measurement_candidate": (
            measurement_rows[0] if measurement_rows else None
        ),
        "active_learning_measurement_submission_performed": False,
        "active_learning_measurement_requires_authenticated_scheduler_dedupe": True,
        "production_validity_and_measurement_eligibility_are_separate": True,
        "reachability_flags": {
            "minimum_observed_Llt_conformal_half_width_uH": minimum_llt_half_width,
            "Llt_uncertainty_only_minimum_possible_G": llt_uncertainty_floor_G,
            "Llt_robust_band_structurally_infeasible_under_current_scalar_half_width": (
                llt_robust_band_structurally_infeasible
            ),
            "next_batch_requires_model_or_uncertainty_contract_repair": (
                len(valid_rows) == 0 and llt_robust_band_structurally_infeasible
            ),
            "next_batch_launch_recommended": (
                len(valid_rows) == 0 and not llt_robust_band_structurally_infeasible
            ),
        },
        "next_warm_start_contract": {
            "path": str(output / "next_warm_start_contract.json"),
            "sha256": _sha256(output / "next_warm_start_contract.json"),
        },
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "fea_submission_approved": False,
    }
    _atomic_json(output / "terminal_analysis.json", analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--nsga-code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-limit", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.warm_limit <= 256:
        raise SystemExit("--warm-limit must be between 1 and 256")
    result = analyze_terminal(
        launch_path=args.launch, status_path=args.status,
        nsga_code_root=args.nsga_code_root, output=args.output,
        warm_limit=args.warm_limit,
    )
    print({
        "fully_valid_unique_count": result["fully_valid_unique_count"],
        "unique_near_candidate_count": result["unique_near_candidate_count"],
        "output": str(args.output.resolve()),
    })


if __name__ == "__main__":
    main()
