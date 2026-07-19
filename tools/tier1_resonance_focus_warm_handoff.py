"""Build an authenticated 50/50 warm pool for resonance-focused Tier-1 NSGA.

This command only reads the three rolling search canonicals and writes below
an explicit output directory.  It never calls the scheduler, submits FEA,
starts AEDT, publishes a pointer, or promotes a design/model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tier1_resonance_feedback import (  # noqa: E402
    CONSTRAINT_VERSION,
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    SEARCH_SCHEMA,
    TEMPERATURE_CONSTRAINT_CONTRACT_SHA256,
    _json_sha,
    optimizer_constraint_normalization,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    _load_sobol_schema,
    decoded_to_unit,
)


SCHEMA = "mft-tier1-resonance-focus-warm-handoff-v1"
INDEX_SCHEMA = "mft-tier1-slurm-rolling-index-v1"
POINTER_SCHEMA = "mft-tier1-slurm-model-pointer-v1"
STATUS_SCHEMA = "mft-tier1-slurm-rolling-status-v1"
TERMINAL_SCHEMA = "mft-tier1-slurm-terminal-result-v1"
SEED_STATUS_SCHEMA = "mft-tier1-slurm-seed-status-v1"
RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
TOLERANCE = 1e-9
NORMALIZED_COUNT = 32
RESONANCE_PASS_COUNT = 32
WARM_SHAPE = (64, 25)
ROLE_NAMESPACES = {
    "main": None,
    "deep": "deep-p320-g600-v1",
    "normalized": "normalized-p320-g600-warm64-v1",
    "resfocus": "resonance-focus250-p320-g600-warm64-v1",
}
DEFAULT_MAIN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling"
)
DEFAULT_DEEP = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_deep_t110_res15k_260719"
)
DEFAULT_NORMALIZED = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_normalized_t110_res15k_260719"
)
DEFAULT_CODE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_code\7c832f7f78f9"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-resonance-focus-50x50-v1"
)
EXPECTED_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    "temperature_robust_limit:T_max_Tx",
    "temperature_robust_limit:T_max_Rx_main",
    "temperature_robust_limit:T_max_Rx_side",
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_Tx_leeward_max",
    "temperature_robust_limit:Tprobe_Rx_main_leeward_max",
    "temperature_robust_limit:Tprobe_Rx_side_leeward_max",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_center_leg_max",
    "temperature_robust_limit:Tprobe_core_side_leg_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "minimum_physical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
    "core_group_manufacturability_limit",
    RESONANCE_CONSTRAINT,
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(path: Path) -> tuple[dict, bytes, str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return value, payload, _sha_bytes(payload)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not finite")
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not finite") from exc
    if not math.isfinite(output):
        raise RuntimeError(f"{label} is not finite")
    return output


def _contained(root: Path, value: Any, label: str) -> Path:
    root = root.resolve(strict=True)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} has no path")
    path = Path(value).resolve(strict=True)
    if path != root and root not in path.parents:
        raise RuntimeError(f"{label} escaped canonical containment: {path}")
    return path


def _read_sealed_reference(
    root: Path, reference: Any, *, schema: str, label: str,
) -> tuple[dict, Path, str]:
    if not isinstance(reference, dict):
        raise RuntimeError(f"{label} reference is missing")
    if reference.get("schema_version") != schema:
        raise RuntimeError(f"{label} reference schema mismatch")
    path = _contained(root, reference.get("path"), label)
    value, _payload, digest = _read_json_bytes(path)
    if digest != reference.get("sha256"):
        raise RuntimeError(f"{label} reference SHA mismatch")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"{label} object schema mismatch")
    return value, path, digest


def _read_terminal_reference(
    root: Path, reference: Any, *, schema: str, label: str,
) -> tuple[dict, Path, str]:
    if not isinstance(reference, dict):
        raise RuntimeError(f"{label} reference is missing")
    path = _contained(root, reference.get("local_path"), label)
    value, _payload, digest = _read_json_bytes(path)
    if digest != reference.get("sha256"):
        raise RuntimeError(f"{label} reference SHA mismatch")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"{label} object schema mismatch")
    return value, path, digest


def _identity_fields(value: dict) -> dict:
    return {
        key: value.get(key)
        for key in (
            "constraint_version",
            "hard_spec_sha256",
            "source_model_manifest_sha256",
            "deployment_model_manifest_sha256",
            "temperature_constraint_contract_sha256",
        )
    }


def _validate_source_identity(
    role: str, index: dict, pointer: dict, status: dict,
) -> None:
    current = pointer.get("current")
    if not isinstance(current, dict):
        raise RuntimeError(f"{role} pointer has no current cohort")
    expected_namespace = ROLE_NAMESPACES[role]
    namespace = (status.get("search_profile") or {}).get("namespace")
    if expected_namespace is None:
        # The main 256-lane cohort predates the named search profile.
        if namespace not in {None, ""}:
            raise RuntimeError(f"{role} unexpectedly has a sidecar namespace")
    elif namespace != expected_namespace:
        raise RuntimeError(f"{role} search namespace mismatch: {namespace}")
    if (
        index.get("active_cohort_id") != current.get("cohort_id")
        or status.get("cohort_id") != current.get("cohort_id")
        or index.get("updated_at") != status.get("updated_at")
        or index.get("hard_spec") != HARD_SPEC
        or status.get("hard_spec") != HARD_SPEC
        or current.get("hard_spec") != HARD_SPEC
        or _json_sha(HARD_SPEC) != current.get("hard_spec_sha256")
        or index.get("constraint_version") != CONSTRAINT_VERSION
        or status.get("constraint_version") != CONSTRAINT_VERSION
        or current.get("constraint_version") != CONSTRAINT_VERSION
        or current.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or status.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or status.get("temperature_constraint_contract_sha256")
        != TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        or status.get("healthy") is not True
        or status.get("error") not in {None, ""}
        or status.get("production_eligible") is not False
        or status.get("fea_submission_approved") is not False
        or status.get("fea_submission_performed") is not False
        or status.get("aedt_used") is not False
    ):
        raise RuntimeError(f"{role} canonical identity/fail-closed seal mismatch")
    identities = (_identity_fields(index), _identity_fields(status), _identity_fields(current))
    if not (identities[0] == identities[1] == identities[2]):
        raise RuntimeError(f"{role} index/status/pointer identity mismatch")


def _validate_result(
    *, role: str, root: Path, current: dict, terminal: dict,
) -> tuple[dict, dict]:
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("authenticated") is not True
        or terminal.get("terminal_state") != "completed"
        or terminal.get("scheduler_state") != "completed"
        or int(terminal.get("scheduler_exit_code", -1)) != 0
        or terminal.get("cohort_id") != current["cohort_id"]
        or terminal.get("constraint_version") != CONSTRAINT_VERSION
        or terminal.get("hard_spec_sha256") != _json_sha(HARD_SPEC)
        or terminal.get("source_model_manifest_sha256")
        != current["source_model_manifest_sha256"]
        or terminal.get("temperature_constraint_contract_sha256")
        != TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        or terminal.get("production_eligible") is not False
        or terminal.get("fea_submission_approved") is not False
        or terminal.get("fea_submission_performed") is not False
    ):
        raise RuntimeError(f"{role} terminal result record seal mismatch")
    seed = int(terminal["seed"])
    task_id = str(terminal["task_id"])
    seed_status, status_path, status_sha = _read_terminal_reference(
        root, terminal.get("remote_status"),
        schema=SEED_STATUS_SCHEMA, label=f"{role} seed status {task_id}",
    )
    result, result_path, result_sha = _read_terminal_reference(
        root, terminal.get("result"),
        schema=SEARCH_SCHEMA, label=f"{role} result {task_id}",
    )
    if (
        seed_status.get("state") != "completed"
        or int(seed_status.get("exit_code", -1)) != 0
        or int(seed_status.get("seed", -1)) != seed
        or str(seed_status.get("task_id")) != task_id
        or seed_status.get("cohort_id") != current["cohort_id"]
        or seed_status.get("bundle_manifest_sha256")
        != current["bundle_manifest_sha256"]
        or seed_status.get("result_sha256") != result_sha
        or seed_status.get("hard_spec_sha256") != _json_sha(HARD_SPEC)
        or seed_status.get("temperature_constraint_contract_sha256")
        != TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        or seed_status.get("production_eligible") is not False
        or seed_status.get("fea_submission_approved") is not False
        or seed_status.get("fea_submission_performed") is not False
        or result.get("schema_version") != SEARCH_SCHEMA
        or int(result.get("seed", -1)) != seed
        or result.get("model_manifest_sha256")
        != current["deployment_model_manifest_sha256"]
        or result.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or result.get("constraint_version") != CONSTRAINT_VERSION
        or result.get("hard_spec") != HARD_SPEC
        or result.get("hard_spec_sha256") != _json_sha(HARD_SPEC)
        or result.get("temperature_constraint_contract_sha256")
        != TEMPERATURE_CONSTRAINT_CONTRACT_SHA256
        or tuple(result.get("constraint_names") or ())
        != EXPECTED_CONSTRAINT_NAMES
        or result.get("production_eligible") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError(f"{role} seed-status/result identity mismatch: {task_id}")
    plan = result.get("next_target_fea_batch_plan")
    if (
        not isinstance(plan, dict)
        or plan.get("submission_performed") is not False
        or plan.get("current_candidates_eligible_for_submission") is not False
        or plan.get("production_eligible") is not False
        or plan.get("fea_submission_approved") is not False
    ):
        raise RuntimeError(f"{role} result next-plan is not fail closed: {task_id}")
    reference = {
        "role": role,
        "cohort_id": current["cohort_id"],
        "task_id": int(task_id),
        "seed": seed,
        "seed_status_path": str(status_path),
        "seed_status_sha256": status_sha,
        "result_path": str(result_path),
        "result_sha256": result_sha,
    }
    return result, reference


def authenticate_source(root: Path, role: str) -> dict:
    if role not in ROLE_NAMESPACES:
        raise RuntimeError(f"unknown source role: {role}")
    requested_root = root.resolve(strict=True)
    index_path = requested_root / "canonical" / "index.json"
    index, _bytes, index_sha = _read_json_bytes(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise RuntimeError(f"{role} canonical index schema mismatch")
    containment = Path(str(index.get("path_containment_root") or "")).resolve(
        strict=True
    )
    if requested_root.resolve() != containment:
        # Junction aliases are valid only when both resolve to the same target.
        raise RuntimeError(f"{role} canonical containment root mismatch")
    pointer, pointer_path, pointer_sha = _read_sealed_reference(
        containment, index.get("model_pointer"),
        schema=POINTER_SCHEMA, label=f"{role} model pointer",
    )
    status, status_path, status_sha = _read_sealed_reference(
        containment, index.get("status"),
        schema=STATUS_SCHEMA, label=f"{role} rolling status",
    )
    _validate_source_identity(role, index, pointer, status)
    current = pointer["current"]
    terminal_results = status.get("terminal_results")
    if not isinstance(terminal_results, list) or not terminal_results:
        raise RuntimeError(f"{role} has no sealed terminal results")
    result_rows = []
    seen_task_ids: set[int] = set()
    seen_seeds: set[int] = set()
    for terminal in terminal_results:
        result, reference = _validate_result(
            role=role, root=containment, current=current, terminal=terminal,
        )
        if reference["task_id"] in seen_task_ids or reference["seed"] in seen_seeds:
            raise RuntimeError(f"{role} duplicate terminal task/seed identity")
        seen_task_ids.add(reference["task_id"])
        seen_seeds.add(reference["seed"])
        result_rows.append((result, reference))
    evidence = {
        "role": role,
        "requested_root": str(requested_root),
        "containment_root": str(containment),
        "index": {"path": str(index_path), "sha256": index_sha},
        "pointer": {"path": str(pointer_path), "sha256": pointer_sha},
        "status": {"path": str(status_path), "sha256": status_sha},
        "cohort_id": current["cohort_id"],
        "bundle_manifest_sha256": current["bundle_manifest_sha256"],
        "controller_source_sha256": current.get("controller_source_sha256"),
        "search_profile": current.get("search_profile"),
        "generation_identity": {
            **_identity_fields(current),
            "hard_spec": current["hard_spec"],
            "nsga_code_revision": current["nsga_code_revision"],
        },
        "terminal_result_count": len(result_rows),
        "terminal_result_sha256": sorted(
            reference["result_sha256"] for _result, reference in result_rows
        ),
        "terminal_seed_status_sha256": sorted(
            reference["seed_status_sha256"]
            for _result, reference in result_rows
        ),
    }
    evidence["terminal_result_set_sha256"] = _json_sha(
        evidence["terminal_result_sha256"]
    )
    evidence["terminal_seed_status_set_sha256"] = _json_sha(
        evidence["terminal_seed_status_sha256"]
    )
    return {"evidence": evidence, "results": result_rows}


def _candidate_rows(source: dict) -> list[dict]:
    role = source["evidence"]["role"]
    normalization = optimizer_constraint_normalization(
        EXPECTED_CONSTRAINT_NAMES, HARD_SPEC,
    )
    rows: dict[str, dict] = {}
    for result, reference in source["results"]:
        candidates = result["next_target_fea_batch_plan"].get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError(f"{role} result candidate list is missing")
        result_seen: set[str] = set()
        for candidate in candidates:
            params = candidate.get("decoded_params")
            digest = str(candidate.get("decoded_params_sha256") or "")
            if not isinstance(params, dict) or digest != _json_sha(params):
                raise RuntimeError(f"{role} decoded parameter SHA mismatch")
            if digest in result_seen:
                raise RuntimeError(f"{role} duplicate geometry within result")
            result_seen.add(digest)
            constraints = candidate.get("constraint_G")
            if (
                not isinstance(constraints, dict)
                or set(constraints) != set(EXPECTED_CONSTRAINT_NAMES)
                or len(constraints) != len(EXPECTED_CONSTRAINT_NAMES)
            ):
                raise RuntimeError(f"{role} candidate constraint schema mismatch")
            values = {
                name: _finite(constraints[name], f"{role}:{digest}:{name}")
                for name in EXPECTED_CONSTRAINT_NAMES
            }
            other_score = sum(
                max(values[name], 0.0) / normalization["scales"][name]
                for name in EXPECTED_CONSTRAINT_NAMES
                if name != RESONANCE_CONSTRAINT
            )
            row = {
                "role": role,
                "decoded_params": params,
                "decoded_params_sha256": digest,
                "constraint_G": values,
                "resonance_G_Hz": values[RESONANCE_CONSTRAINT],
                "other_normalized_positive_violation": other_score,
                "reference": reference,
            }
            rank = (
                max(values[RESONANCE_CONSTRAINT], 0.0),
                other_score,
                reference["seed"],
                reference["result_sha256"],
            )
            incumbent = rows.get(digest)
            if incumbent is None or rank < incumbent["_rank"]:
                row["_rank"] = rank
                rows[digest] = row
    return sorted(rows.values(), key=lambda row: (*row["_rank"], row["decoded_params_sha256"]))


def _select_diverse(rows: list[dict], count: int, required_roles=()) -> list[dict]:
    if len(rows) < count:
        raise RuntimeError(f"candidate shortage: need {count}, have {len(rows)}")
    selected: list[int] = []
    for role in required_roles:
        options = [index for index, row in enumerate(rows) if row["role"] == role]
        if not options:
            raise RuntimeError(f"candidate source role is absent: {role}")
        selected.append(options[0])
    if not selected:
        selected.append(0)
    selected = list(dict.fromkeys(selected))
    quality = np.asarray([sum(row["_rank"][:2]) for row in rows], dtype=float)
    q_span = max(float(np.max(quality) - np.min(quality)), 1e-12)
    quality = (quality - float(np.min(quality))) / q_span
    while len(selected) < count:
        remaining = [index for index in range(len(rows)) if index not in selected]
        scored = []
        for index in remaining:
            distance = min(
                float(np.linalg.norm(rows[index]["coordinate"] - rows[chosen]["coordinate"]))
                for chosen in selected
            )
            scored.append((distance - 0.05 * quality[index], -index, index))
        selected.append(max(scored)[2])
    return [rows[index] for index in selected]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_bytes(
        path,
        (json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ) + "\n").encode("utf-8"),
    )


def build_handoff(
    *, main_root: Path, deep_root: Path, normalized_root: Path,
    nsga_code_root: Path, output: Path,
) -> dict:
    sources = {
        "main": authenticate_source(main_root, "main"),
        "deep": authenticate_source(deep_root, "deep"),
        "normalized": authenticate_source(normalized_root, "normalized"),
    }
    identity_fields = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if not all(identity == identity_fields[0] for identity in identity_fields[1:]):
        raise RuntimeError("main/deep/normalized result generation mismatch")

    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(nsga_code_root)
    if len(sobol_dims) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    all_rows = {role: _candidate_rows(source) for role, source in sources.items()}
    for rows in all_rows.values():
        for row in rows:
            row["coordinate"] = decoded_to_unit(
                row["decoded_params"], sobol_dims, n1_min, n1_max, defaults,
            )
            if row["coordinate"].shape != (WARM_SHAPE[1],):
                raise RuntimeError("decoded warm coordinate shape mismatch")
            if not np.isfinite(row["coordinate"]).all():
                raise RuntimeError("decoded warm coordinate is non-finite")

    normalized_rows = [
        row for row in all_rows["normalized"]
        if all(
            value <= TOLERANCE
            for name, value in row["constraint_G"].items()
            if name != RESONANCE_CONSTRAINT
        )
    ]
    resonance_rows = [
        row for role in ("main", "deep") for row in all_rows[role]
        if row["resonance_G_Hz"] <= TOLERANCE
    ]
    normalized_selected = _select_diverse(
        normalized_rows, NORMALIZED_COUNT,
    )
    resonance_selected = _select_diverse(
        resonance_rows, RESONANCE_PASS_COUNT, required_roles=("main", "deep"),
    )
    selected = []
    for index in range(max(len(normalized_selected), len(resonance_selected))):
        if index < len(normalized_selected):
            selected.append(("all_except_resonance", normalized_selected[index]))
        if index < len(resonance_selected):
            selected.append(("resonance_pass_near", resonance_selected[index]))
    coordinates = np.vstack([row["coordinate"] for _category, row in selected])
    if coordinates.shape != WARM_SHAPE:
        raise RuntimeError(f"warm pool shape mismatch: {coordinates.shape}")
    coordinate_identities = [tuple(np.round(row, 12)) for row in coordinates]
    if len(set(coordinate_identities)) != WARM_SHAPE[0]:
        raise RuntimeError("cross-category warm coordinates are not unique")
    geometry_ids = [row["decoded_params_sha256"] for _category, row in selected]
    if len(set(geometry_ids)) != WARM_SHAPE[0]:
        raise RuntimeError("cross-category warm geometries are not unique")

    output = output.resolve()
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    _atomic_bytes(warm_path, buffer.getvalue())
    selected_provenance = []
    for warm_index, (category, row) in enumerate(selected):
        selected_provenance.append({
            "warm_index": warm_index,
            "category": category,
            "source_role": row["role"],
            "cohort_id": row["reference"]["cohort_id"],
            "task_id": row["reference"]["task_id"],
            "seed": row["reference"]["seed"],
            "seed_status_sha256": row["reference"]["seed_status_sha256"],
            "result_sha256": row["reference"]["result_sha256"],
            "decoded_params_sha256": row["decoded_params_sha256"],
            "resonance_G_Hz": row["resonance_G_Hz"],
            "other_normalized_positive_violation": row[
                "other_normalized_positive_violation"
            ],
        })
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": (
            "32_normalized_all_physical_constraints_except_resonance_pass_"
            "plus_32_main_deep_resonance_pass_near_diverse_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": _sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "inverse_decoded_then_current_problem_repair_v1"
            ),
        },
        "category_counts": {
            "all_except_resonance": len(normalized_selected),
            "resonance_pass_near": len(resonance_selected),
        },
        "source_evidence": {
            role: source["evidence"] for role, source in sources.items()
        },
        "selected_provenance": selected_provenance,
        "selected_provenance_sha256": _json_sha(selected_provenance),
        "hard_spec": HARD_SPEC,
        "hard_spec_sha256": _json_sha(HARD_SPEC),
        "constraint_version": CONSTRAINT_VERSION,
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "optimizer_resonance_scale_Hz": 250.0,
        "optimizer_scale_scope": "search_pressure_only_physical_G_unchanged",
        "rolling_target": 32,
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "seed_start": 1_907_192_000,
        "task_priority": -7,
        "launch_performed": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    contract_path = output / "resonance_focus_warm_contract.json"
    _atomic_json(contract_path, contract)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": _sha_file(contract_path),
        **contract,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--main-root", type=Path, default=DEFAULT_MAIN)
    result.add_argument("--deep-root", type=Path, default=DEFAULT_DEEP)
    result.add_argument("--normalized-root", type=Path, default=DEFAULT_NORMALIZED)
    result.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> int:
    args = parser().parse_args()
    result = build_handoff(
        main_root=args.main_root,
        deep_root=args.deep_root,
        normalized_root=args.normalized_root,
        nsga_code_root=args.nsga_code_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
