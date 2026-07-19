"""Build the sealed 64-point Tier-1 core-thermal crossover warm pool.

The command is read-only with respect to all source canonicals.  It writes
only below ``--output`` and never contacts the scheduler, submits FEA, starts
AEDT, publishes a pointer, or promotes a candidate/model.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_resonance_focus_warm_handoff as sealed  # noqa: E402
from tools.tier1_resonance_feedback import (  # noqa: E402
    CONSTRAINT_VERSION,
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    TEMPERATURE_CONSTRAINT_CONTRACT_SHA256,
    _json_sha,
    optimizer_constraint_normalization,
    thermal_crossover_acquisition_contract,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    _load_sobol_schema,
    decoded_to_unit,
)


SCHEMA = "mft-tier1-thermal-crossover-warm-handoff-v1"
CONTRACT_FILENAME = "thermal_crossover_warm_contract.json"
WARM_SHAPE = (64, 25)
BRANCH_COUNT = 32
MIN_EXACT_TARGET_COMPLETE = 24
TOLERANCE = 1e-9
HOTFIX_RESONANCE_CONTROLLER_SHA256 = (
    "5097c476f983f6460861ff7a53aaef4472fc67eda0924006db3d4429fd27052a"
)
CORE_THERMAL_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
TARGET_COMPLETION_CONSTRAINTS = tuple(
    name for name in sealed.EXPECTED_CONSTRAINT_NAMES
    if not name.startswith("temperature_robust_limit:")
)
SOURCE_ROLES = ("main", "deep", "normalized", "resfocus")
DEFAULT_MAIN = sealed.DEFAULT_MAIN
DEFAULT_DEEP = sealed.DEFAULT_DEEP
DEFAULT_NORMALIZED = sealed.DEFAULT_NORMALIZED
DEFAULT_RESONANCE_FOCUS = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\t1r250_260719"
)
DEFAULT_CODE = sealed.DEFAULT_CODE
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-thermal-crossover-balanced-v1"
)
EXPECTED_GENERATION_IDENTITY = {
    "constraint_version": CONSTRAINT_VERSION,
    "hard_spec_sha256": _json_sha(HARD_SPEC),
    "source_model_manifest_sha256": (
        "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
    ),
    "deployment_model_manifest_sha256": (
        "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
    ),
    "temperature_constraint_contract_sha256": (
        TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
    ),
    "hard_spec": HARD_SPEC,
    "nsga_code_revision": EXPECTED_NSGA_REVISION,
}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not finite") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def authenticate_source(root: Path, role: str) -> dict:
    source = sealed.authenticate_source(root, role)
    if role == "resfocus":
        evidence = source["evidence"]
        profile = evidence.get("search_profile") or {}
        if (
            evidence.get("controller_source_sha256")
            != HOTFIX_RESONANCE_CONTROLLER_SHA256
            or profile.get("namespace")
            != "resonance-focus250-p320-g600-warm64-v1"
            or profile.get("variant") != "resonance-focus-250hz-v1"
            or profile.get("optimizer_resonance_scale_Hz") != 250.0
            or profile.get("population") != 320
            or profile.get("max_generations") != 600
            or profile.get("inference_threads") != 8
            or profile.get("rolling_target") != 32
            or profile.get("seed_start") != 1_907_192_000
            or profile.get("task_priority") != -7
        ):
            raise RuntimeError(
                "resfocus source is not the authenticated zero-score hotfix cohort"
            )
    return source


def _prepare_rows(source: dict) -> list[dict]:
    default_normalization = optimizer_constraint_normalization(
        sealed.EXPECTED_CONSTRAINT_NAMES,
        HARD_SPEC,
        resonance_scale_hz=250.0,
    )
    rows = sealed._candidate_rows(source)
    for row in rows:
        values = row["constraint_G"]
        target_score = sum(
            max(_finite(values[name], name), 0.0)
            / default_normalization["scales"][name]
            for name in TARGET_COMPLETION_CONSTRAINTS
        )
        core_values = {
            name: _finite(values[name], name)
            for name in CORE_THERMAL_CONSTRAINTS
        }
        core_positive = [max(value, 0.0) for value in core_values.values()]
        row.update({
            "target_complete": all(
                values[name] <= TOLERANCE
                for name in TARGET_COMPLETION_CONSTRAINTS
            ),
            "target_normalized_positive_violation": float(target_score),
            "core_thermal_G_C": core_values,
            "core_thermal_max_positive_violation_C": float(
                max(core_positive)
            ),
            "core_thermal_sum_positive_violation_C": float(
                sum(core_positive)
            ),
            "cool_core_complete": all(
                value <= TOLERANCE for value in core_values.values()
            ),
        })
    return rows


def _deduplicate(rows: list[dict], rank_key) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in rows:
        digest = row["decoded_params_sha256"]
        incumbent = unique.get(digest)
        if incumbent is None or rank_key(row) < rank_key(incumbent):
            unique[digest] = row
    return sorted(unique.values(), key=rank_key)


def _select_diverse(
    rows: list[dict], count: int, *, rank_key, required_roles=(),
) -> list[dict]:
    rows = _deduplicate(rows, rank_key)
    if len(rows) < count:
        raise RuntimeError(
            f"candidate shortage: need {count}, have {len(rows)}"
        )
    selected: list[int] = []
    for role in required_roles:
        options = [
            index for index, row in enumerate(rows)
            if row["role"] == role
        ]
        if not options:
            raise RuntimeError(f"candidate source role is absent: {role}")
        selected.append(options[0])
    if not selected:
        selected.append(0)
    selected = list(dict.fromkeys(selected))
    quality = np.asarray(
        [float(index) for index in range(len(rows))], dtype=float
    )
    quality /= max(float(len(rows) - 1), 1.0)
    while len(selected) < count:
        remaining = [
            index for index in range(len(rows)) if index not in selected
        ]
        scored = []
        for index in remaining:
            distance = min(
                float(np.linalg.norm(
                    rows[index]["coordinate"] - rows[chosen]["coordinate"]
                ))
                for chosen in selected
            )
            scored.append((distance - 0.05 * quality[index], -index, index))
        selected.append(max(scored)[2])
    return [rows[index] for index in selected]


def _target_rank(row: dict) -> tuple:
    return (
        0 if row["target_complete"] else 1,
        row["target_normalized_positive_violation"],
        row["core_thermal_sum_positive_violation_C"],
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _cool_rank(row: dict) -> tuple:
    return (
        0 if row["cool_core_complete"] else 1,
        row["core_thermal_max_positive_violation_C"],
        row["core_thermal_sum_positive_violation_C"],
        row["target_normalized_positive_violation"],
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _source_evidence(source: dict) -> dict:
    return source["evidence"]


def build_handoff(
    *, main_root: Path, deep_root: Path, normalized_root: Path,
    resonance_focus_root: Path, nsga_code_root: Path, output: Path,
) -> dict:
    roots = {
        "main": main_root,
        "deep": deep_root,
        "normalized": normalized_root,
        "resfocus": resonance_focus_root,
    }
    sources = {
        role: authenticate_source(roots[role], role) for role in SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if (
        not all(identity == identities[0] for identity in identities[1:])
        or identities[0] != EXPECTED_GENERATION_IDENTITY
    ):
        raise RuntimeError("four-source result generation mismatch")

    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(
        nsga_code_root
    )
    if len(sobol_dims) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    rows = []
    for source in sources.values():
        rows.extend(_prepare_rows(source))
    for row in rows:
        row["coordinate"] = decoded_to_unit(
            row["decoded_params"], sobol_dims, n1_min, n1_max, defaults,
        )
        if row["coordinate"].shape != (WARM_SHAPE[1],):
            raise RuntimeError("decoded warm coordinate shape mismatch")
        if not np.isfinite(row["coordinate"]).all():
            raise RuntimeError("decoded warm coordinate is non-finite")

    exact_target = _deduplicate(
        [row for row in rows if row["target_complete"]], _target_rank,
    )
    if len(exact_target) < MIN_EXACT_TARGET_COMPLETE:
        raise RuntimeError(
            "authenticated exact target-complete candidate shortage: "
            f"need {MIN_EXACT_TARGET_COMPLETE}, have {len(exact_target)}"
        )
    exact_selected = _select_diverse(
        exact_target,
        min(BRANCH_COUNT, len(exact_target)),
        rank_key=_target_rank,
        required_roles=("resfocus",),
    )
    target_selected = list(exact_selected)
    if len(target_selected) < BRANCH_COUNT:
        selected_ids = {
            row["decoded_params_sha256"] for row in target_selected
        }
        near_rows = [
            row for row in rows
            if row["decoded_params_sha256"] not in selected_ids
        ]
        target_selected.extend(_select_diverse(
            near_rows,
            BRANCH_COUNT - len(target_selected),
            rank_key=_target_rank,
        ))
    target_ids = {
        row["decoded_params_sha256"] for row in target_selected
    }
    cool_rows = [
        row for row in rows
        if row["cool_core_complete"]
        and row["decoded_params_sha256"] not in target_ids
    ]
    cool_selected = _select_diverse(
        cool_rows,
        BRANCH_COUNT,
        rank_key=_cool_rank,
        required_roles=("main", "deep", "normalized"),
    )

    selected = []
    for index in range(BRANCH_COUNT):
        selected.append(("target_complete_branch", target_selected[index]))
        selected.append(("cool_core_diverse_branch", cool_selected[index]))
    coordinates = np.vstack([
        row["coordinate"] for _category, row in selected
    ])
    if coordinates.shape != WARM_SHAPE:
        raise RuntimeError(f"warm pool shape mismatch: {coordinates.shape}")
    coordinate_ids = [tuple(np.round(row, 12)) for row in coordinates]
    geometry_ids = [
        row["decoded_params_sha256"] for _category, row in selected
    ]
    if len(set(coordinate_ids)) != WARM_SHAPE[0]:
        raise RuntimeError("cross-branch warm coordinates are not unique")
    if len(set(geometry_ids)) != WARM_SHAPE[0]:
        raise RuntimeError("cross-branch warm geometries are not unique")
    selected_roles = {row["role"] for _category, row in selected}
    if selected_roles != set(SOURCE_ROLES):
        raise RuntimeError("warm pool does not cross all four source roles")

    output = output.resolve()
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    sealed._atomic_bytes(warm_path, buffer.getvalue())
    provenance = []
    for warm_index, (category, row) in enumerate(selected):
        reference = row["reference"]
        coordinate_sha = _json_sha([
            float(value) for value in row["coordinate"]
        ])
        provenance.append({
            "warm_index": warm_index,
            "category": category,
            "source_role": row["role"],
            "cohort_id": reference["cohort_id"],
            "task_id": reference["task_id"],
            "seed": reference["seed"],
            "seed_status_sha256": reference["seed_status_sha256"],
            "result_sha256": reference["result_sha256"],
            "decoded_params_sha256": row["decoded_params_sha256"],
            "unit_coordinate_sha256": coordinate_sha,
            "target_complete": row["target_complete"],
            "target_normalized_positive_violation": row[
                "target_normalized_positive_violation"
            ],
            "cool_core_complete": row["cool_core_complete"],
            "core_thermal_G_C": row["core_thermal_G_C"],
            "core_thermal_max_positive_violation_C": row[
                "core_thermal_max_positive_violation_C"
            ],
            "core_thermal_sum_positive_violation_C": row[
                "core_thermal_sum_positive_violation_C"
            ],
        })
    exact_target_count = sum(
        row["target_complete"] for row in target_selected
    )
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": (
            "32_target_complete_pressure_at_least24_exact_plus_"
            "32_exact_cool_core_three_source_diverse_interleaved_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": sealed._sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "decoded_to_unit_then_current_problem_repair_v1"
            ),
        },
        "category_counts": {
            "target_complete_branch": BRANCH_COUNT,
            "cool_core_diverse_branch": BRANCH_COUNT,
        },
        "exact_target_complete_count": exact_target_count,
        "minimum_exact_target_complete_count": MIN_EXACT_TARGET_COMPLETE,
        "exact_cool_core_count": sum(
            row["cool_core_complete"] for row in cool_selected
        ),
        "selected_source_roles": sorted(selected_roles),
        "source_evidence": {
            role: _source_evidence(source)
            for role, source in sources.items()
        },
        "selected_provenance": provenance,
        "selected_provenance_sha256": _json_sha(provenance),
        "hard_spec": HARD_SPEC,
        "hard_spec_sha256": _json_sha(HARD_SPEC),
        "constraint_version": CONSTRAINT_VERSION,
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "authoritative_terminal_G": "physical_unscaled",
        "optimizer_scale_scope": (
            "search_pressure_only_physical_G_unchanged"
        ),
        "optimizer_resonance_scale_Hz": 250.0,
        "optimizer_core_thermal_scale_C": 1.0,
        "optimizer_core_thermal_constraints": list(
            CORE_THERMAL_CONSTRAINTS
        ),
        "acquisition_ranking_contract": (
            thermal_crossover_acquisition_contract(HARD_SPEC, 1.0)
        ),
        "rolling_target": 32,
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "seed_start": 1_907_193_000,
        "task_priority": -6,
        "launch_performed": False,
        "scheduler_write_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    contract_path = output / CONTRACT_FILENAME
    sealed._atomic_json(contract_path, contract)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sealed._sha_file(contract_path),
        **contract,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--main-root", type=Path, default=DEFAULT_MAIN)
    result.add_argument("--deep-root", type=Path, default=DEFAULT_DEEP)
    result.add_argument(
        "--normalized-root", type=Path, default=DEFAULT_NORMALIZED
    )
    result.add_argument(
        "--resonance-focus-root",
        type=Path,
        default=DEFAULT_RESONANCE_FOCUS,
    )
    result.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> int:
    args = parser().parse_args()
    result = build_handoff(
        main_root=args.main_root,
        deep_root=args.deep_root,
        normalized_root=args.normalized_root,
        resonance_focus_root=args.resonance_focus_root,
        nsga_code_root=args.nsga_code_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
