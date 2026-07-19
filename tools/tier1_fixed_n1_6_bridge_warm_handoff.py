"""Build a sealed low-temperature/resonance bridge warm pool for fixed N1=6.

The command reads four authenticated Tier-1 rolling canonicals and writes only
below ``--output``.  It never contacts the scheduler, submits FEA, starts AEDT,
publishes a pointer, or promotes a model/design.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_resonance_focus_warm_handoff as sealed  # noqa: E402
from tools.tier1_resonance_feedback import (  # noqa: E402
    CONSTRAINT_VERSION,
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    _json_sha,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    _git_revision,
    _load_sobol_schema,
    decoded_to_unit,
)


SCHEMA = "mft-tier1-fixed-n1-6-bridge-warm-handoff-v1"
POST_REPAIR_PREFLIGHT_SCHEMA = (
    "mft-tier1-fixed-n1-6-bridge-post-repair-preflight-v1"
)
CONTRACT_FILENAME = "fixed_n1_6_bridge_warm_contract.json"
ORTHOGONAL_SCHEMA = "mft-tier1-fixed-n1-6-orthogonal-warm-handoff-v1"
ORTHOGONAL_POST_REPAIR_PREFLIGHT_SCHEMA = (
    "mft-tier1-fixed-n1-6-orthogonal-post-repair-preflight-v1"
)
ORTHOGONAL_CONTRACT_FILENAME = (
    "fixed_n1_6_orthogonal_warm_contract.json"
)
WARM_SHAPE = (64, 25)
THERMAL_COUNT_PER_SOURCE = 16
FIXED6_RESONANCE_COUNT = 24
RESFOCUS_RESONANCE_COUNT = 8
ORTHOGONAL_THERMAL_LLT_COUNT_PER_SOURCE = 12
ORTHOGONAL_BRIDGE_COUNT = 37
ORTHOGONAL_RESONANCE_LLT_COUNT = 3
ORTHOGONAL_BRIDGE_MINIMUM_TERMINALS = 32
TOLERANCE = 1e-9
SOURCE_ROLES = ("normalized", "thermal", "fixed6", "resfocus")
ORTHOGONAL_SOURCE_ROLES = ("normalized", "thermal", "bridge", "fixed6")
SOURCE_SNAPSHOT_KINDS = ("index", "pointer", "status")
SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS = 8
ROLE_NAMESPACES = {
    "normalized": "normalized-p320-g600-warm64-v1",
    "thermal": "thermal-crossover-core1c-res250-p320-g600-warm64-v1",
    "fixed6": "fixed-n1-6-resonance-llt-core4c-res100-p320-g600-warm64-v1",
    "resfocus": "resonance-focus250-p320-g600-warm64-v1",
    "bridge": (
        "fixed-n1-6-thermal-bridge-core2c-llt0p55-res150-"
        "p320-g600-warm64-v1"
    ),
}
REQUIRED_ANCHOR_TASKS = {
    "normalized": 57059,
    "thermal": 57919,
    "fixed6": 58486,
    "resfocus": 57505,
}
ORTHOGONAL_REQUIRED_ANCHOR_TASKS = {
    "normalized": 57059,
    "thermal": 59279,
    "fixed6": 59035,
}
THERMAL_CONSTRAINTS = tuple(
    name for name in sealed.EXPECTED_CONSTRAINT_NAMES
    if name.startswith("temperature_robust_limit:")
)
CORE_THERMAL_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
LLT_CONSTRAINTS = ("Llt_robust_band", "Llt_ensemble_disagreement")
DEFAULT_NORMALIZED = sealed.DEFAULT_NORMALIZED
DEFAULT_THERMAL = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_thermal_crossover_t110_res15k_260719"
)
DEFAULT_FIXED6 = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_n1_6_resonance_llt_t110_res15k_260719"
)
DEFAULT_RESFOCUS = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\t1r250_260719"
)
DEFAULT_BRIDGE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_n1_6_thermal_bridge_t110_res15k_260719"
)
DEFAULT_CODE = sealed.DEFAULT_CODE
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-6-thermal-bridge-v1"
)
DEFAULT_ORTHOGONAL_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-6-orthogonal-bridge-v1"
)
NSGA_DECODER_RELATIVE_PATH = "module/input_parameter_260706.py"


def _finite(value, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not finite") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _verify_nsga_code_root(nsga_code_root: Path) -> dict:
    nsga_code_root = nsga_code_root.resolve(strict=True)
    revision = _git_revision(nsga_code_root)
    if revision != EXPECTED_NSGA_REVISION:
        raise RuntimeError(
            "NSGA code revision does not match EXPECTED_NSGA_REVISION"
        )
    status = subprocess.run(
        [
            "git", "-c", f"safe.directory={nsga_code_root.as_posix()}",
            "-C", str(nsga_code_root), "status", "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("NSGA code root is not clean")
    decoder = (nsga_code_root / NSGA_DECODER_RELATIVE_PATH).resolve(strict=True)
    if nsga_code_root not in decoder.parents:
        raise RuntimeError("NSGA decoder escaped code root")
    return {
        "revision": revision,
        "clean": True,
        "decoder_relative_path": NSGA_DECODER_RELATIVE_PATH,
        "decoder_sha256": sealed._sha_file(decoder),
    }


def _authenticate(root: Path, role: str) -> dict:
    previous = sealed.ROLE_NAMESPACES.get(role)
    had_previous = role in sealed.ROLE_NAMESPACES
    sealed.ROLE_NAMESPACES[role] = ROLE_NAMESPACES[role]
    try:
        return sealed.authenticate_source(root, role)
    finally:
        if had_previous:
            sealed.ROLE_NAMESPACES[role] = previous
        else:
            sealed.ROLE_NAMESPACES.pop(role, None)


def _terminal_snapshot_evidence(status: dict, role: str) -> dict:
    terminal_results = status.get("terminal_results")
    if not isinstance(terminal_results, list) or not terminal_results:
        raise RuntimeError(f"{role} status snapshot has no terminal results")
    references = []
    for terminal in terminal_results:
        if not isinstance(terminal, dict):
            raise RuntimeError(f"{role} status terminal record is invalid")
        result = terminal.get("result")
        seed_status = terminal.get("remote_status")
        if not isinstance(result, dict) or not isinstance(seed_status, dict):
            raise RuntimeError(f"{role} status terminal reference is missing")
        result_sha = result.get("sha256")
        seed_status_sha = seed_status.get("sha256")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (result_sha, seed_status_sha)
        ):
            raise RuntimeError(f"{role} status terminal SHA is invalid")
        if (
            terminal.get("authenticated") is not True
            or terminal.get("terminal_state") != "completed"
            or terminal.get("scheduler_state") != "completed"
            or int(terminal.get("scheduler_exit_code", -1)) != 0
        ):
            raise RuntimeError(
                f"{role} status snapshot contains unauthenticated terminal row"
            )
        references.append({
            "task_id": int(terminal["task_id"]),
            "seed": int(terminal["seed"]),
            "result_sha256": result_sha,
            "seed_status_sha256": seed_status_sha,
        })
    if len({item["task_id"] for item in references}) != len(references):
        raise RuntimeError(f"{role} status snapshot has duplicate task IDs")
    if len({item["seed"] for item in references}) != len(references):
        raise RuntimeError(f"{role} status snapshot has duplicate seeds")
    result_order = [item["result_sha256"] for item in references]
    status_order = [item["seed_status_sha256"] for item in references]
    reference_set = sorted(
        references,
        key=lambda item: (
            item["task_id"], item["seed"], item["result_sha256"],
            item["seed_status_sha256"],
        ),
    )
    return {
        "terminal_snapshot_count": len(references),
        "terminal_result_order_sha256": _json_sha(result_order),
        "terminal_seed_status_order_sha256": _json_sha(status_order),
        "terminal_reference_order_sha256": _json_sha(references),
        "terminal_reference_set_sha256": _json_sha(reference_set),
        "terminal_result_sha256": sorted(result_order),
        "terminal_seed_status_sha256": sorted(status_order),
    }


def _capture_authenticated_source(root: Path, role: str) -> dict:
    """Authenticate and immediately capture the exact canonical-head bytes."""

    last_error = None
    for _attempt in range(SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS):
        source = _authenticate(root, role)
        evidence = source["evidence"]
        payloads = {}
        try:
            for kind in SOURCE_SNAPSHOT_KINDS:
                source_reference = evidence.get(kind)
                if not isinstance(source_reference, dict):
                    raise RuntimeError(f"{role} {kind} source evidence is missing")
                source_path = Path(
                    str(source_reference.get("path") or "")
                ).resolve(strict=True)
                payload = source_path.read_bytes()
                digest = sealed._sha_bytes(payload)
                if digest != source_reference.get("sha256"):
                    raise RuntimeError(
                        f"{role} {kind} changed after authentication"
                    )
                parsed = json.loads(payload.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError(
                        f"{role} {kind} snapshot is not an object"
                    )
                payloads[kind] = payload
        except (
            OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError,
        ) as exc:
            last_error = exc
            continue
        status = json.loads(payloads["status"].decode("utf-8"))
        terminal_evidence = _terminal_snapshot_evidence(status, role)
        if (
            terminal_evidence["terminal_result_sha256"]
            != evidence.get("terminal_result_sha256")
            or terminal_evidence["terminal_seed_status_sha256"]
            != evidence.get("terminal_seed_status_sha256")
            or _json_sha(terminal_evidence["terminal_result_sha256"])
            != evidence.get("terminal_result_set_sha256")
            or _json_sha(terminal_evidence["terminal_seed_status_sha256"])
            != evidence.get("terminal_seed_status_set_sha256")
        ):
            last_error = RuntimeError(
                f"{role} captured terminal references drifted"
            )
            continue
        evidence.update({
            key: value for key, value in terminal_evidence.items()
            if key not in {
                "terminal_result_sha256", "terminal_seed_status_sha256",
            }
        })
        source["_snapshot_payloads"] = payloads
        return source
    raise RuntimeError(
        f"{role} canonical head did not remain stable for authenticated capture"
    ) from last_error


def _seal_source_snapshots(
    output: Path, sources: dict[str, dict],
) -> dict[str, dict]:
    """Persist already-captured canonical heads as content-addressed files."""

    snapshots = {}
    for role in sources:
        evidence = sources[role]["evidence"]
        payloads = sources[role].get("_snapshot_payloads")
        if not isinstance(payloads, dict):
            raise RuntimeError(f"{role} source snapshot bytes were not captured")
        role_snapshots = {}
        for kind in SOURCE_SNAPSHOT_KINDS:
            source_reference = evidence.get(kind)
            if not isinstance(source_reference, dict):
                raise RuntimeError(f"{role} {kind} source evidence is missing")
            payload = payloads.get(kind)
            if not isinstance(payload, bytes):
                raise RuntimeError(f"{role} {kind} snapshot bytes are missing")
            digest = sealed._sha_bytes(payload)
            if digest != source_reference.get("sha256"):
                raise RuntimeError(f"{role} {kind} captured SHA drifted")
            relative = (
                Path("source_snapshots") / role / f"{kind}-{digest}.json"
            )
            destination = output / relative
            if destination.exists():
                existing = destination.read_bytes()
                if existing != payload:
                    raise RuntimeError(
                        f"content-addressed {role} {kind} snapshot collision"
                    )
            else:
                sealed._atomic_bytes(destination, payload)
            role_snapshots[kind] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
                "source_sha256": source_reference["sha256"],
            }
        snapshots[role] = role_snapshots
    return snapshots


def _seal_selected_artifacts(
    output: Path, selected: list[tuple[str, dict]],
) -> list[dict]:
    """Seal the selected rows' immutable result and seed-status evidence."""

    artifacts = []
    for warm_index, (_category, row) in enumerate(selected):
        reference = row["reference"]
        sealed_references = {}
        parsed = {}
        for kind, path_key, sha_key in (
            ("result", "result_path", "result_sha256"),
            ("seed_status", "seed_status_path", "seed_status_sha256"),
        ):
            source_path = Path(str(reference.get(path_key) or "")).resolve(
                strict=True
            )
            payload = source_path.read_bytes()
            digest = sealed._sha_bytes(payload)
            if digest != reference.get(sha_key):
                raise RuntimeError(
                    f"selected {kind} changed after source authentication"
                )
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"selected {kind} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"selected {kind} is not a JSON object")
            relative = (
                Path("selected_evidence") / kind / f"{kind}-{digest}.json"
            )
            destination = output / relative
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise RuntimeError(
                        f"content-addressed selected {kind} collision"
                    )
            else:
                sealed._atomic_bytes(destination, payload)
            sealed_references[kind] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
            }
            parsed[kind] = value
        result = parsed["result"]
        seed_status = parsed["seed_status"]
        candidates = (
            result.get("next_target_fea_batch_plan") or {}
        ).get("candidates")
        matching = [
            candidate for candidate in candidates or []
            if isinstance(candidate, dict)
            and candidate.get("decoded_params_sha256")
            == row["decoded_params_sha256"]
        ]
        if (
            result.get("schema_version") != sealed.SEARCH_SCHEMA
            or int(result.get("seed", -1)) != int(reference["seed"])
            or len(matching) != 1
            or matching[0].get("decoded_params") != row["decoded_params"]
            or _json_sha(matching[0].get("decoded_params"))
            != row["decoded_params_sha256"]
            or seed_status.get("schema_version") != sealed.SEED_STATUS_SCHEMA
            or seed_status.get("state") != "completed"
            or int(seed_status.get("exit_code", -1)) != 0
            or int(seed_status.get("seed", -1)) != int(reference["seed"])
            or str(seed_status.get("task_id")) != str(reference["task_id"])
            or seed_status.get("cohort_id") != reference["cohort_id"]
            or seed_status.get("result_sha256")
            != reference["result_sha256"]
        ):
            raise RuntimeError("selected result/seed-status identity mismatch")
        artifacts.append({
            "warm_index": warm_index,
            "source_role": row["role"],
            "cohort_id": reference["cohort_id"],
            "task_id": reference["task_id"],
            "seed": reference["seed"],
            **sealed_references,
        })
    return artifacts


def _prepare_rows(source: dict) -> list[dict]:
    rows = sealed._candidate_rows(source)
    for row in rows:
        constraints = row["constraint_G"]
        params = row["decoded_params"]
        row.update({
            "original_N1": int(params["N1"]),
            "all_thermal_pass": all(
                _finite(constraints[name], name) <= TOLERANCE
                for name in THERMAL_CONSTRAINTS
            ),
            "resonance_pass": (
                _finite(constraints[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT)
                <= TOLERANCE
            ),
            "Llt_pass": all(
                _finite(constraints[name], name) <= TOLERANCE
                for name in LLT_CONSTRAINTS
            ),
            "orthogonal_other_pass": all(
                _finite(constraints[name], name) <= TOLERANCE
                for name in sealed.EXPECTED_CONSTRAINT_NAMES
                if name not in {
                    *THERMAL_CONSTRAINTS,
                    *LLT_CONSTRAINTS,
                    RESONANCE_CONSTRAINT,
                }
            ),
            "thermal_positive_sum_C": float(sum(
                max(_finite(constraints[name], name), 0.0)
                for name in THERMAL_CONSTRAINTS
            )),
            "thermal_positive_max_C": float(max(
                max(_finite(constraints[name], name), 0.0)
                for name in THERMAL_CONSTRAINTS
            )),
            "core_positive_sum_C": float(sum(
                max(_finite(constraints[name], name), 0.0)
                for name in CORE_THERMAL_CONSTRAINTS
            )),
            "core_positive_max_C": float(max(
                max(_finite(constraints[name], name), 0.0)
                for name in CORE_THERMAL_CONSTRAINTS
            )),
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


def _thermal_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    return (
        max(_finite(values[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT), 0.0)
        / 150.0,
        max(_finite(values["Llt_robust_band"], "Llt_robust_band"), 0.0)
        / float(HARD_SPEC["Llt_tol_uH"]),
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _resonance_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    return (
        row["core_positive_max_C"] / 2.0,
        row["core_positive_sum_C"] / 2.0,
        max(_finite(values["Llt_robust_band"], "Llt_robust_band"), 0.0)
        / float(HARD_SPEC["Llt_tol_uH"]),
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _orthogonal_thermal_llt_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    return (
        max(_finite(values[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT), 0.0)
        / 150.0,
        row["thermal_positive_sum_C"] / 2.0,
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _orthogonal_bridge_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    normalized_positive = (
        max(_finite(values["Llt_robust_band"], "Llt_robust_band"), 0.0)
        / 0.25
        + max(_finite(
            values["Llt_ensemble_disagreement"],
            "Llt_ensemble_disagreement",
        ), 0.0) / 0.5
        + max(_finite(
            values[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT,
        ), 0.0) / 150.0
        + row["thermal_positive_sum_C"] / 2.0
    )
    return (
        normalized_positive,
        row["thermal_positive_max_C"] / 2.0,
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _orthogonal_resonance_llt_rank(row: dict) -> tuple:
    return (
        row["thermal_positive_sum_C"] / 2.0,
        row["thermal_positive_max_C"] / 2.0,
        row["reference"]["seed"],
        row["reference"]["result_sha256"],
        row["decoded_params_sha256"],
    )


def _orthogonal_bridge_diversity_evidence(rows: list[dict]) -> dict:
    """Seal minimum parameter and physical-constraint diversity for bridge rows."""

    if len(rows) != ORTHOGONAL_BRIDGE_COUNT:
        raise RuntimeError("orthogonal bridge diversity received wrong quota")
    coordinates = np.vstack([row["coordinate"] for row in rows])
    if coordinates.shape != (ORTHOGONAL_BRIDGE_COUNT, WARM_SHAPE[1]):
        raise RuntimeError("orthogonal bridge coordinate shape mismatch")
    pairwise = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest = pairwise.min(axis=1)
    spans = np.ptp(coordinates, axis=0)
    constraint_matrix = np.asarray([
        [
            _finite(row["constraint_G"][name], name)
            for name in sealed.EXPECTED_CONSTRAINT_NAMES
        ]
        for row in rows
    ], dtype=float)
    constraint_span = np.ptp(constraint_matrix, axis=0)
    scale = np.where(constraint_span > 1e-12, constraint_span, 1.0)
    normalized_constraints = constraint_matrix / scale
    constraint_pairwise = np.linalg.norm(
        normalized_constraints[:, None, :]
        - normalized_constraints[None, :, :],
        axis=2,
    )
    np.fill_diagonal(constraint_pairwise, np.inf)
    constraint_fingerprints = {
        _json_sha([float(round(value, 9)) for value in row])
        for row in constraint_matrix
    }
    evidence = {
        "schema_version": "mft-tier1-orthogonal-bridge-diversity-v1",
        "selected_count": len(rows),
        "decoded_params_sha256_unique_count": len({
            row["decoded_params_sha256"] for row in rows
        }),
        "unit_parameter_minimum_pairwise_l2": float(nearest.min()),
        "unit_parameter_median_nearest_l2": float(np.median(nearest)),
        "unit_parameter_axis_span_ge_0p02_count": int((spans >= 0.02).sum()),
        "unit_parameter_centered_rank": int(np.linalg.matrix_rank(
            coordinates - coordinates.mean(axis=0), tol=1e-9,
        )),
        "constraint_G_rounded_1e_9_unique_count": len(
            constraint_fingerprints
        ),
        "constraint_G_normalized_minimum_pairwise_l2": float(
            constraint_pairwise.min()
        ),
        "constraint_G_centered_rank": int(np.linalg.matrix_rank(
            normalized_constraints - normalized_constraints.mean(axis=0),
            tol=1e-9,
        )),
        "gates": {
            "decoded_sha_all_unique": True,
            "unit_parameter_minimum_pairwise_l2_ge": 0.001,
            "unit_parameter_median_nearest_l2_ge": 0.005,
            "unit_parameter_axis_span_ge_0p02_minimum_count": 8,
            "unit_parameter_centered_rank_minimum": 8,
            "constraint_G_rounded_1e_9_minimum_unique_count": 37,
            "constraint_G_normalized_minimum_pairwise_l2_ge": 0.01,
            "constraint_G_centered_rank_minimum": 8,
        },
        "passed": True,
    }
    if (
        evidence["decoded_params_sha256_unique_count"] != len(rows)
        or evidence["unit_parameter_minimum_pairwise_l2"] < 0.001
        or evidence["unit_parameter_median_nearest_l2"] < 0.005
        or evidence["unit_parameter_axis_span_ge_0p02_count"] < 8
        or evidence["unit_parameter_centered_rank"] < 8
        or evidence["constraint_G_rounded_1e_9_unique_count"] < 37
        or evidence["constraint_G_normalized_minimum_pairwise_l2"] < 0.01
        or evidence["constraint_G_centered_rank"] < 8
    ):
        evidence["passed"] = False
        raise RuntimeError(
            "orthogonal bridge parameter/constraint-G diversity gate failed: "
            f"{json.dumps(evidence, sort_keys=True)}"
        )
    evidence["sha256"] = _json_sha(evidence)
    return evidence


def _select_diverse(
    rows: list[dict], count: int, *, anchor_task: int, rank_key,
) -> list[dict]:
    rows = _deduplicate(rows, rank_key)
    anchors = [
        index for index, row in enumerate(rows)
        if int(row["reference"]["task_id"]) == int(anchor_task)
    ]
    if not anchors:
        raise RuntimeError(
            f"required anchor task {anchor_task} has no eligible candidate"
        )
    if len(rows) < count:
        raise RuntimeError(f"candidate shortage: need {count}, have {len(rows)}")
    selected = [anchors[0]]
    quality = np.arange(len(rows), dtype=float)
    quality /= max(float(len(rows) - 1), 1.0)
    while len(selected) < count:
        remaining = [index for index in range(len(rows)) if index not in selected]
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


def build_handoff(
    *, normalized_root: Path, thermal_root: Path, fixed6_root: Path,
    resonance_focus_root: Path, nsga_code_root: Path, output: Path,
) -> dict:
    nsga_code_root = nsga_code_root.resolve(strict=True)
    code_identity = _verify_nsga_code_root(nsga_code_root)
    nsga_code_revision = code_identity["revision"]
    roots = {
        "normalized": normalized_root,
        "thermal": thermal_root,
        "fixed6": fixed6_root,
        "resfocus": resonance_focus_root,
    }
    sources = {
        role: _capture_authenticated_source(roots[role], role)
        for role in SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if not all(identity == identities[0] for identity in identities[1:]):
        raise RuntimeError("bridge source generation identity mismatch")

    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(
        nsga_code_root
    )
    if len(sobol_dims) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    rows_by_role = {}
    for role, source in sources.items():
        rows = _prepare_rows(source)
        for row in rows:
            row["coordinate"] = decoded_to_unit(
                row["decoded_params"], sobol_dims, n1_min, n1_max, defaults,
            )
            if (
                row["coordinate"].shape != (WARM_SHAPE[1],)
                or not np.isfinite(row["coordinate"]).all()
                or (row["coordinate"] < 0.0).any()
                or (row["coordinate"] > 1.0).any()
            ):
                raise RuntimeError("decoded bridge coordinate is invalid")
        rows_by_role[role] = rows

    selected_by_role = {
        "normalized": _select_diverse(
            [
                row for row in rows_by_role["normalized"]
                if row["all_thermal_pass"] and row["original_N1"] == 6
            ],
            THERMAL_COUNT_PER_SOURCE,
            anchor_task=REQUIRED_ANCHOR_TASKS["normalized"],
            rank_key=_thermal_rank,
        ),
        "thermal": _select_diverse(
            [
                row for row in rows_by_role["thermal"]
                if row["all_thermal_pass"] and row["original_N1"] == 6
            ],
            THERMAL_COUNT_PER_SOURCE,
            anchor_task=REQUIRED_ANCHOR_TASKS["thermal"],
            rank_key=_thermal_rank,
        ),
        "fixed6": _select_diverse(
            [
                row for row in rows_by_role["fixed6"]
                if row["resonance_pass"] and row["original_N1"] == 6
            ],
            FIXED6_RESONANCE_COUNT,
            anchor_task=REQUIRED_ANCHOR_TASKS["fixed6"],
            rank_key=_resonance_rank,
        ),
        "resfocus": _select_diverse(
            [row for row in rows_by_role["resfocus"] if row["resonance_pass"]],
            RESFOCUS_RESONANCE_COUNT,
            anchor_task=REQUIRED_ANCHOR_TASKS["resfocus"],
            rank_key=_resonance_rank,
        ),
    }
    thermal_rows = []
    for index in range(THERMAL_COUNT_PER_SOURCE):
        thermal_rows.append(selected_by_role["normalized"][index])
        thermal_rows.append(selected_by_role["thermal"][index])
    resonance_rows = []
    for index in range(FIXED6_RESONANCE_COUNT):
        resonance_rows.append(selected_by_role["fixed6"][index])
        if index < RESFOCUS_RESONANCE_COUNT:
            resonance_rows.append(selected_by_role["resfocus"][index])
    resonance_rows = resonance_rows[:32]
    selected = []
    for index in range(32):
        selected.append(("low_temperature_branch", thermal_rows[index]))
        selected.append(("resonance_pass_branch", resonance_rows[index]))

    coordinates = np.vstack([row["coordinate"] for _category, row in selected])
    if coordinates.shape != WARM_SHAPE:
        raise RuntimeError(f"warm pool shape mismatch: {coordinates.shape}")
    if len({tuple(np.round(row, 12)) for row in coordinates}) != 64:
        raise RuntimeError("bridge warm coordinates are not unique")
    if len({row["decoded_params_sha256"] for _category, row in selected}) != 64:
        raise RuntimeError("bridge warm geometries are not unique")

    output = output.resolve()
    source_snapshots = _seal_source_snapshots(output, sources)
    selected_artifacts = _seal_selected_artifacts(output, selected)
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    sealed._atomic_bytes(warm_path, buffer.getvalue())
    provenance = []
    for warm_index, (category, row) in enumerate(selected):
        reference = row["reference"]
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
            "unit_coordinate_sha256": _json_sha([
                float(value) for value in row["coordinate"]
            ]),
            "original_N1": row["original_N1"],
            "fixed_N1_repair_required": row["original_N1"] != 6,
            "all_thermal_pass": row["all_thermal_pass"],
            "resonance_pass": row["resonance_pass"],
            "Llt_robust_G_uH": _finite(
                row["constraint_G"]["Llt_robust_band"], "Llt"
            ),
            "resonance_G_Hz": _finite(
                row["constraint_G"][RESONANCE_CONSTRAINT], "resonance"
            ),
            "core_positive_max_C": row["core_positive_max_C"],
            "core_positive_sum_C": row["core_positive_sum_C"],
        })
    anchors = {
        role: next(
            item for item in provenance
            if item["source_role"] == role
            and item["task_id"] == REQUIRED_ANCHOR_TASKS[role]
        )
        for role in SOURCE_ROLES
    }
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": (
            "32_low_temperature_n1_6_normalized16_thermal16_plus_"
            "32_resonance_pass_fixed6_24_resfocus8_interleaved_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": sealed._sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "authenticated_decoded_to_unit_then_fixed_n1_6_repair_v1"
            ),
        },
        "category_counts": {
            "low_temperature_branch": 32,
            "resonance_pass_branch": 32,
        },
        "source_role_counts": {
            role: sum(item["source_role"] == role for item in provenance)
            for role in SOURCE_ROLES
        },
        "branch_source_role_counts": {
            "low_temperature_branch": {
                "normalized": 16,
                "thermal": 16,
            },
            "resonance_pass_branch": {
                "fixed6": 24,
                "resfocus": 8,
            },
        },
        "required_anchor_tasks": REQUIRED_ANCHOR_TASKS,
        "anchors": anchors,
        "source_evidence": {
            role: source["evidence"] for role, source in sources.items()
        },
        "source_snapshots": source_snapshots,
        "selected_artifact_snapshots": selected_artifacts,
        "selected_artifact_snapshots_sha256": _json_sha(selected_artifacts),
        "selected_provenance": provenance,
        "selected_provenance_sha256": _json_sha(provenance),
        "hard_spec": HARD_SPEC,
        "hard_spec_sha256": _json_sha(HARD_SPEC),
        "constraint_version": CONSTRAINT_VERSION,
        "constraint_names": list(sealed.EXPECTED_CONSTRAINT_NAMES),
        "constraint_names_sha256": _json_sha(
            list(sealed.EXPECTED_CONSTRAINT_NAMES)
        ),
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "nsga_code_root_revision": nsga_code_revision,
        "nsga_code_root_revision_verified": True,
        "nsga_code_root_clean_verified": code_identity["clean"],
        "nsga_decoder_relative_path": code_identity["decoder_relative_path"],
        "nsga_decoder_source_sha256": code_identity["decoder_sha256"],
        "fixed_primary_turns": 6,
        "fixed_primary_turns_scope": (
            "runner_repair_after_authenticated_inverse_coordinate_handoff"
        ),
        "authoritative_terminal_G": "physical_unscaled",
        "optimizer_scale_scope": (
            "search_pressure_only_physical_G_unchanged"
        ),
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": 2.0,
        "optimizer_Llt_scale_uH": float(HARD_SPEC["Llt_tol_uH"]),
        "rolling_target": 32,
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "seed_start": 1_907_197_000,
        "task_priority": -2,
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


def build_orthogonal_handoff(
    *, normalized_root: Path, thermal_root: Path, bridge_root: Path,
    fixed6_root: Path, nsga_code_root: Path, output: Path,
) -> dict:
    """Build a sealed three-way handoff after the live bridge has matured."""

    nsga_code_root = nsga_code_root.resolve(strict=True)
    code_identity = _verify_nsga_code_root(nsga_code_root)
    roots = {
        "normalized": normalized_root,
        "thermal": thermal_root,
        "bridge": bridge_root,
        "fixed6": fixed6_root,
    }
    sources = {
        role: _capture_authenticated_source(roots[role], role)
        for role in ORTHOGONAL_SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if not all(identity == identities[0] for identity in identities[1:]):
        raise RuntimeError("orthogonal source generation identity mismatch")
    bridge_terminal_count = int(
        sources["bridge"]["evidence"]["terminal_result_count"]
    )
    if bridge_terminal_count < ORTHOGONAL_BRIDGE_MINIMUM_TERMINALS:
        raise RuntimeError(
            "orthogonal handoff requires at least "
            f"{ORTHOGONAL_BRIDGE_MINIMUM_TERMINALS} authenticated bridge "
            f"terminals; observed {bridge_terminal_count}"
        )

    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(
        nsga_code_root
    )
    if len(sobol_dims) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    rows_by_role = {}
    for role, source in sources.items():
        rows = _prepare_rows(source)
        for row in rows:
            row["coordinate"] = decoded_to_unit(
                row["decoded_params"], sobol_dims, n1_min, n1_max, defaults,
            )
            if (
                row["coordinate"].shape != (WARM_SHAPE[1],)
                or not np.isfinite(row["coordinate"]).all()
                or (row["coordinate"] < 0.0).any()
                or (row["coordinate"] > 1.0).any()
            ):
                raise RuntimeError("decoded orthogonal coordinate is invalid")
        rows_by_role[role] = rows

    branch_candidate_supply_counts = {
        "normalized_thermal_llt_pass_unique": len({
            row["decoded_params_sha256"]
            for row in rows_by_role["normalized"]
            if row["original_N1"] == 6
            and row["all_thermal_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        }),
        "thermal_thermal_llt_pass_unique": len({
            row["decoded_params_sha256"]
            for row in rows_by_role["thermal"]
            if row["original_N1"] == 6
            and row["all_thermal_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        }),
        "bridge_near_unique": len({
            row["decoded_params_sha256"]
            for row in rows_by_role["bridge"]
            if row["original_N1"] == 6 and row["orthogonal_other_pass"]
        }),
        "fixed6_resonance_llt_pass_unique": len({
            row["decoded_params_sha256"]
            for row in rows_by_role["fixed6"]
            if row["original_N1"] == 6
            and row["resonance_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        }),
    }

    selected_by_role: dict[str, list[dict]] = {}
    excluded: set[str] = set()

    def select(
        role: str, eligible: list[dict], count: int, *,
        rank_key, anchor_task: int | None = None,
    ) -> list[dict]:
        eligible = [
            row for row in eligible
            if row["decoded_params_sha256"] not in excluded
        ]
        if anchor_task is None:
            ranked = _deduplicate(eligible, rank_key)
            if len(ranked) < count:
                raise RuntimeError(
                    f"{role} candidate shortage: need {count}, have "
                    f"{len(ranked)}"
                )
            anchor_task = int(ranked[0]["reference"]["task_id"])
        chosen = _select_diverse(
            eligible, count, anchor_task=anchor_task, rank_key=rank_key,
        )
        excluded.update(row["decoded_params_sha256"] for row in chosen)
        selected_by_role[role] = chosen
        return chosen

    normalized_selected = select(
        "normalized",
        [
            row for row in rows_by_role["normalized"]
            if row["original_N1"] == 6
            and row["all_thermal_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        ],
        ORTHOGONAL_THERMAL_LLT_COUNT_PER_SOURCE,
        rank_key=_orthogonal_thermal_llt_rank,
        anchor_task=ORTHOGONAL_REQUIRED_ANCHOR_TASKS["normalized"],
    )
    thermal_selected = select(
        "thermal",
        [
            row for row in rows_by_role["thermal"]
            if row["original_N1"] == 6
            and row["all_thermal_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        ],
        ORTHOGONAL_THERMAL_LLT_COUNT_PER_SOURCE,
        rank_key=_orthogonal_thermal_llt_rank,
        anchor_task=ORTHOGONAL_REQUIRED_ANCHOR_TASKS["thermal"],
    )
    bridge_selected = select(
        "bridge",
        [
            row for row in rows_by_role["bridge"]
            if row["original_N1"] == 6 and row["orthogonal_other_pass"]
        ],
        ORTHOGONAL_BRIDGE_COUNT,
        rank_key=_orthogonal_bridge_rank,
    )
    bridge_diversity = _orthogonal_bridge_diversity_evidence(
        bridge_selected
    )
    fixed6_selected = select(
        "fixed6",
        [
            row for row in rows_by_role["fixed6"]
            if row["original_N1"] == 6
            and row["resonance_pass"]
            and row["Llt_pass"]
            and row["orthogonal_other_pass"]
        ],
        ORTHOGONAL_RESONANCE_LLT_COUNT,
        rank_key=_orthogonal_resonance_llt_rank,
        anchor_task=ORTHOGONAL_REQUIRED_ANCHOR_TASKS["fixed6"],
    )
    fixed6_resonance_llt_supply = _deduplicate([
        row for row in rows_by_role["fixed6"]
        if row["original_N1"] == 6
        and row["resonance_pass"]
        and row["Llt_pass"]
        and row["orthogonal_other_pass"]
    ], _orthogonal_resonance_llt_rank)
    if (
        len(fixed6_resonance_llt_supply) != ORTHOGONAL_RESONANCE_LLT_COUNT
        or {
            row["decoded_params_sha256"] for row in fixed6_selected
        } != {
            row["decoded_params_sha256"]
            for row in fixed6_resonance_llt_supply
        }
    ):
        raise RuntimeError(
            "orthogonal handoff did not preserve all three authenticated "
            "fixed6 resonance+Llt pass points"
        )
    fixed6_resonance_llt_preservation = {
        "authenticated_unique_count": len(fixed6_resonance_llt_supply),
        "selected_unique_count": len(fixed6_selected),
        "selected_task_ids": sorted(
            int(row["reference"]["task_id"]) for row in fixed6_selected
        ),
        "selected_decoded_params_sha256": sorted(
            row["decoded_params_sha256"] for row in fixed6_selected
        ),
        "required_anchor_task_id": ORTHOGONAL_REQUIRED_ANCHOR_TASKS[
            "fixed6"
        ],
        "required_anchor_preserved": True,
        "all_authenticated_unique_points_preserved": True,
    }
    fixed6_resonance_llt_preservation["sha256"] = _json_sha(
        fixed6_resonance_llt_preservation
    )

    thermal_llt_rows = []
    for index in range(ORTHOGONAL_THERMAL_LLT_COUNT_PER_SOURCE):
        thermal_llt_rows.extend((
            normalized_selected[index], thermal_selected[index],
        ))
    branches = {
        "thermal_llt_pass_branch": thermal_llt_rows,
        "bridge_near_branch": bridge_selected,
        "resonance_llt_pass_branch": fixed6_selected,
    }
    selected = []
    for index in range(max(len(rows) for rows in branches.values())):
        for category, rows in branches.items():
            if index < len(rows):
                selected.append((category, rows[index]))
    coordinates = np.vstack([row["coordinate"] for _category, row in selected])
    if coordinates.shape != WARM_SHAPE:
        raise RuntimeError(f"orthogonal warm pool shape mismatch: {coordinates.shape}")
    if len({tuple(np.round(row, 12)) for row in coordinates}) != 64:
        raise RuntimeError("orthogonal warm coordinates are not unique")
    if len({row["decoded_params_sha256"] for _category, row in selected}) != 64:
        raise RuntimeError("orthogonal warm geometries are not unique")

    output = output.resolve()
    source_snapshots = _seal_source_snapshots(output, sources)
    selected_artifacts = _seal_selected_artifacts(output, selected)
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    sealed._atomic_bytes(warm_path, buffer.getvalue())
    provenance = []
    for warm_index, (category, row) in enumerate(selected):
        reference = row["reference"]
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
            "unit_coordinate_sha256": _json_sha([
                float(value) for value in row["coordinate"]
            ]),
            "original_N1": row["original_N1"],
            "fixed_N1_repair_required": False,
            "all_thermal_pass": row["all_thermal_pass"],
            "Llt_pass": row["Llt_pass"],
            "resonance_pass": row["resonance_pass"],
            "orthogonal_other_pass": row["orthogonal_other_pass"],
            "Llt_robust_G_uH": _finite(
                row["constraint_G"]["Llt_robust_band"], "Llt"
            ),
            "Llt_disagreement_G_uH": _finite(
                row["constraint_G"]["Llt_ensemble_disagreement"],
                "Llt disagreement",
            ),
            "resonance_G_Hz": _finite(
                row["constraint_G"][RESONANCE_CONSTRAINT], "resonance"
            ),
            "thermal_positive_max_C": row["thermal_positive_max_C"],
            "thermal_positive_sum_C": row["thermal_positive_sum_C"],
        })
    anchors = {
        role: next(
            item for item in provenance
            if item["source_role"] == role
            and item["task_id"] == task_id
        )
        for role, task_id in ORTHOGONAL_REQUIRED_ANCHOR_TASKS.items()
    }
    bridge_anchor = next(
        item for item in provenance if item["source_role"] == "bridge"
    )
    contract = {
        "schema_version": ORTHOGONAL_SCHEMA,
        "selection_contract": (
            "24_thermal_and_llt_pass_normalized12_thermal12_plus_"
            "37_latest_bridge_near_plus_3_resonance_and_llt_pass_"
            "fixed6_interleaved_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": sealed._sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "authenticated_decoded_to_unit_then_fixed_n1_6_repair_v1"
            ),
        },
        "category_counts": {
            category: len(rows) for category, rows in branches.items()
        },
        "source_role_counts": {
            role: sum(item["source_role"] == role for item in provenance)
            for role in ORTHOGONAL_SOURCE_ROLES
        },
        "branch_source_role_counts": {
            "thermal_llt_pass_branch": {"normalized": 12, "thermal": 12},
            "bridge_near_branch": {"bridge": 37},
            "resonance_llt_pass_branch": {"fixed6": 3},
        },
        "branch_candidate_supply_counts": branch_candidate_supply_counts,
        "bridge_diversity_evidence": bridge_diversity,
        "fixed6_resonance_llt_preservation": (
            fixed6_resonance_llt_preservation
        ),
        "resonance_llt_supply_policy": (
            "seal_all_3_authenticated_unique_fixed6_pass_points_then_fill_"
            "remaining_pool_with_latest_bridge_near_points_v1"
        ),
        "quota_revision_evidence": {
            "initial_planned_counts": {
                "thermal_llt_pass_branch": 24,
                "bridge_near_branch": 16,
                "resonance_llt_pass_branch": 24,
            },
            "authenticated_recount_fixed6_resonance_llt_pass_unique": (
                branch_candidate_supply_counts[
                    "fixed6_resonance_llt_pass_unique"
                ]
            ),
            "authenticated_recount_bridge_near_unique": (
                branch_candidate_supply_counts["bridge_near_unique"]
            ),
            "revised_counts": {
                "thermal_llt_pass_branch": 24,
                "bridge_near_branch": 37,
                "resonance_llt_pass_branch": 3,
            },
            "reason": (
                "captured_fixed6_source_has_only_3_authenticated_unique_"
                "simultaneous_resonance_and_Llt_pass_points; preserve_all_3_"
                "and_fill_21_point_shortfall_from_mature_bridge_source"
            ),
            "decoded_sha_uniqueness_preserved_across_all_branches": True,
        },
        "required_anchor_tasks": ORTHOGONAL_REQUIRED_ANCHOR_TASKS,
        "anchors": anchors,
        "bridge_anchor": bridge_anchor,
        "bridge_minimum_terminal_contract": {
            "minimum_authenticated_terminal_count": (
                ORTHOGONAL_BRIDGE_MINIMUM_TERMINALS
            ),
            "observed_authenticated_terminal_count": bridge_terminal_count,
            "captured_status_sha256": sources["bridge"]["evidence"][
                "status"
            ]["sha256"],
            "latest_authenticated_head_required": True,
            "satisfied": True,
        },
        "source_evidence": {
            role: source["evidence"] for role, source in sources.items()
        },
        "source_snapshots": source_snapshots,
        "selected_artifact_snapshots": selected_artifacts,
        "selected_artifact_snapshots_sha256": _json_sha(selected_artifacts),
        "selected_provenance": provenance,
        "selected_provenance_sha256": _json_sha(provenance),
        "hard_spec": HARD_SPEC,
        "hard_spec_sha256": _json_sha(HARD_SPEC),
        "constraint_version": CONSTRAINT_VERSION,
        "constraint_names": list(sealed.EXPECTED_CONSTRAINT_NAMES),
        "constraint_names_sha256": _json_sha(
            list(sealed.EXPECTED_CONSTRAINT_NAMES)
        ),
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "nsga_code_root_revision": code_identity["revision"],
        "nsga_code_root_revision_verified": True,
        "nsga_code_root_clean_verified": code_identity["clean"],
        "nsga_decoder_relative_path": code_identity["decoder_relative_path"],
        "nsga_decoder_source_sha256": code_identity["decoder_sha256"],
        "fixed_primary_turns": 6,
        "fixed_primary_turns_scope": (
            "runner_repair_after_authenticated_inverse_coordinate_handoff"
        ),
        "authoritative_terminal_G": "physical_unscaled",
        "optimizer_scale_scope": (
            "search_pressure_only_physical_G_unchanged"
        ),
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_Llt_disagreement_scale_uH": 0.5,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "optimizer_all_active_thermal_constraints": list(
            THERMAL_CONSTRAINTS
        ),
        "optimizer_all_active_thermal_side_activation": (
            "finite_N2_side_gt_0_else_physical_negative_BIG"
        ),
        "soft_axis_pressure_scope": "disabled_until_stage_feasible",
        "acquisition_ranking_contract": None,
        "hard_constraint_mutation": False,
        "rolling_target": 32,
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "seed_start": 1_907_198_000,
        "task_priority": -1,
        "launch_performed": False,
        "scheduler_write_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    contract_path = output / ORTHOGONAL_CONTRACT_FILENAME
    sealed._atomic_json(contract_path, contract)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sealed._sha_file(contract_path),
        **contract,
    }


def _post_repair_preflight_summary(contract: dict, result: dict) -> dict:
    warm = result.get("warm_start")
    audit = (warm or {}).get("fixed_primary_turns_warm_audit")
    repair = result.get("fixed_primary_turns_contract")
    warm_repair = (warm or {}).get("fixed_primary_turns_repair")
    initialization = result.get("initialization_audit")
    warm_filter = (initialization or {}).get("warm_filter")
    normalization = result.get("optimizer_constraint_normalization")
    acquisition = result.get("acquisition_ranking_contract")
    source_model_shas = {
        evidence["generation_identity"]["source_model_manifest_sha256"]
        for evidence in contract["source_evidence"].values()
    }
    temperature_shas = {
        evidence["generation_identity"][
            "temperature_constraint_contract_sha256"
        ]
        for evidence in contract["source_evidence"].values()
    }
    repair_unsigned = dict(repair or {})
    repair_sha = repair_unsigned.pop("sha256", None)
    normalization_unsigned = dict(normalization or {})
    normalization_sha = normalization_unsigned.pop("sha256", None)
    core_names = {
        "temperature_robust_limit:T_max_core",
        "temperature_robust_limit:Tprobe_core_center_max",
        "temperature_robust_limit:Tprobe_core_top_yoke_max",
    }
    scale_overrides = (normalization or {}).get(
        "explicit_optimizer_only_scale_overrides"
    )
    if (
        result.get("schema_version") != sealed.SEARCH_SCHEMA
        or int(result.get("population", -1)) != 64
        or int(result.get("max_generations", -1)) != 1
        or int(result.get("completed_generations", -1)) < 1
        or result.get("nsga_code_revision") != contract["nsga_code_revision"]
        or result.get("constraint_version") != contract["constraint_version"]
        or result.get("hard_spec") != contract["hard_spec"]
        or result.get("hard_spec_sha256") != contract["hard_spec_sha256"]
        or result.get("model_manifest_sha256") not in source_model_shas
        or len(source_model_shas) != 1
        or result.get("temperature_constraint_contract_sha256")
        not in temperature_shas
        or len(temperature_shas) != 1
        or tuple(result.get("constraint_names") or ())
        != tuple(contract["constraint_names"])
        or result.get("fixed_primary_turns") != 6
        or result.get("terminal_population_fixed_primary_turns_verified")
        is not True
        or result.get("terminal_population_primary_turn_values") != [6]
        or not isinstance(warm, dict)
        or warm.get("sha256") != contract["warm_start"]["sha256"]
        or warm.get("coordinate_count") != 64
        or warm.get("dimension_count") != 25
        or warm.get("post_repair_coordinate_count") != 64
        or warm.get("repaired_by_current_simultaneous_hard_constraint_problem")
        is not True
        or not isinstance(repair, dict)
        or repair != warm_repair
        or repair.get("schema_version")
        != "mft-tier1-fixed-primary-turns-repair-v1"
        or repair.get("fixed_primary_turns") != 6
        or repair.get("coordinate_index") != 0
        or repair.get("hard_constraint_mutation") is not False
        or repair.get("objective_mutation") is not False
        or repair.get("warm_start_post_repair_minimum_unique_count") != 32
        or _json_sha(repair_unsigned) != repair_sha
        or not isinstance(audit, dict)
        or audit.get("schema_version")
        != "mft-tier1-fixed-primary-turns-warm-audit-v1"
        or audit.get("source_coordinate_count") != 64
        or audit.get("post_repair_coordinate_shape") != [64, 25]
        or audit.get("post_repair_coordinate_dtype") != "float64"
        or audit.get("post_repair_decoded_unique_count") != 64
        or audit.get("post_repair_hard_feasible_count") != 64
        or audit.get("post_repair_rejected_count") != 0
        or audit.get("minimum_unique_count") != 32
        or audit.get("fixed_primary_turns") != 6
        or audit.get("fixed_primary_turn_coordinate_verified") is not True
        or audit.get("physical_geometry_dedupe_performed") is not True
        or audit.get("source_sha_verified_before_repair") is not True
        or not isinstance(
            audit.get("post_repair_coordinates_sha256"), str
        )
        or len(audit["post_repair_coordinates_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in audit["post_repair_coordinates_sha256"]
        )
        or not isinstance(warm_filter, dict)
        or warm_filter != {
            "input_count": 64,
            "hard_feasible_count": 64,
            "decoded_unique_count": 64,
            "rejected_count": 0,
        }
        or not isinstance(normalization, dict)
        or normalization.get("schema_version")
        != "mft-tier1-optimizer-constraint-normalization-v1"
        or normalization.get("constraint_order") != contract["constraint_names"]
        or normalization.get("authoritative_terminal_G")
        != "physical_unscaled"
        or normalization.get("scales", {}).get("Llt_robust_band") != 0.55
        or normalization.get("scales", {}).get(
            "half_magnetizing_resonance_minimum"
        ) != 150.0
        or not isinstance(scale_overrides, dict)
        or set(scale_overrides) != core_names
        or set(scale_overrides.values()) != {2.0}
        or _json_sha(normalization_unsigned) != normalization_sha
        or not isinstance(acquisition, dict)
        or acquisition.get("active") is not True
        or acquisition.get("hard_constraint_mutation") is not False
        or acquisition.get("core_thermal_positive_G_scale_C") != 2.0
        or any(
            result.get(field) is not False
            for field in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed", "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("post-repair preflight contract mismatch")
    return {
        "schema_version": POST_REPAIR_PREFLIGHT_SCHEMA,
        "seed": int(result["seed"]),
        "population": 64,
        "max_generations": 1,
        "completed_generations": int(result["completed_generations"]),
        "source_warm_sha256": warm["sha256"],
        "post_repair_coordinates_sha256": audit[
            "post_repair_coordinates_sha256"
        ],
        "post_repair_coordinate_shape": [64, 25],
        "post_repair_decoded_unique_count": 64,
        "post_repair_hard_feasible_count": 64,
        "post_repair_rejected_count": 0,
        "fixed_primary_turns": 6,
        "optimizer_Llt_scale_uH": 0.55,
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": 2.0,
        "terminal_population_fixed_primary_turns_verified": True,
    }


def seal_post_repair_preflight(
    contract_path: Path, result_path: Path,
    nsga_code_root: Path = DEFAULT_CODE,
) -> dict:
    contract_path = contract_path.resolve(strict=True)
    result_path = result_path.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA:
        raise RuntimeError("bridge handoff contract schema mismatch")
    expected_names = list(sealed.EXPECTED_CONSTRAINT_NAMES)
    if contract.get("constraint_names") is None:
        # Contracts created immediately before this trust-boundary field was
        # introduced can be upgraded without rebuilding/changing their warm SHA.
        contract["constraint_names"] = expected_names
        contract["constraint_names_sha256"] = _json_sha(expected_names)
    elif (
        contract.get("constraint_names") != expected_names
        or contract.get("constraint_names_sha256") != _json_sha(expected_names)
    ):
        raise RuntimeError("bridge constraint-name contract mismatch")
    code_identity = _verify_nsga_code_root(nsga_code_root)
    code_fields = {
        "nsga_code_root_revision": code_identity["revision"],
        "nsga_code_root_revision_verified": True,
        "nsga_code_root_clean_verified": code_identity["clean"],
        "nsga_decoder_relative_path": code_identity["decoder_relative_path"],
        "nsga_decoder_source_sha256": code_identity["decoder_sha256"],
    }
    for key, value in code_fields.items():
        existing = contract.get(key)
        if existing is not None and existing != value:
            raise RuntimeError("bridge NSGA code identity mismatch")
        contract[key] = value
    payload = result_path.read_bytes()
    digest = sealed._sha_bytes(payload)
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("post-repair preflight is not valid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("post-repair preflight is not a JSON object")
    summary = _post_repair_preflight_summary(contract, result)
    relative = (
        Path("post_repair_preflight") / f"result-{digest}.json"
    )
    destination = contract_path.parent / relative
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError("post-repair preflight SHA collision")
    else:
        sealed._atomic_bytes(destination, payload)
    contract["post_repair_preflight"] = {
        "schema_version": POST_REPAIR_PREFLIGHT_SCHEMA,
        "result": {
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(payload),
        },
        "verified_summary": summary,
        "verified_summary_sha256": _json_sha(summary),
        "required_for_deployment": True,
        "verified": True,
    }
    sealed._atomic_json(contract_path, contract)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sealed._sha_file(contract_path),
        "preflight_result_sha256": digest,
        "summary": summary,
    }


def _orthogonal_post_repair_preflight_summary(
    contract: dict, result: dict,
) -> dict:
    """Fail closed on the orthogonal lane's repaired 64-point smoke run."""

    warm = result.get("warm_start")
    audit = (warm or {}).get("fixed_primary_turns_warm_audit")
    repair = result.get("fixed_primary_turns_contract")
    warm_repair = (warm or {}).get("fixed_primary_turns_repair")
    initialization = result.get("initialization_audit")
    warm_filter = (initialization or {}).get("warm_filter")
    normalization = result.get("optimizer_constraint_normalization")
    source_model_shas = {
        evidence["generation_identity"]["source_model_manifest_sha256"]
        for evidence in contract["source_evidence"].values()
    }
    temperature_shas = {
        evidence["generation_identity"][
            "temperature_constraint_contract_sha256"
        ]
        for evidence in contract["source_evidence"].values()
    }
    repair_unsigned = dict(repair or {})
    repair_sha = repair_unsigned.pop("sha256", None)
    normalization_unsigned = dict(normalization or {})
    normalization_sha = normalization_unsigned.pop("sha256", None)
    expected_overrides = {
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        **{name: 2.0 for name in THERMAL_CONSTRAINTS},
    }
    scales = (normalization or {}).get("scales")
    if (
        contract.get("schema_version") != ORTHOGONAL_SCHEMA
        or result.get("schema_version") != sealed.SEARCH_SCHEMA
        or int(result.get("population", -1)) != 64
        or int(result.get("max_generations", -1)) != 1
        or int(result.get("completed_generations", -1)) < 1
        or result.get("nsga_code_revision") != contract["nsga_code_revision"]
        or result.get("constraint_version") != contract["constraint_version"]
        or result.get("hard_spec") != contract["hard_spec"]
        or result.get("hard_spec_sha256") != contract["hard_spec_sha256"]
        or len(source_model_shas) != 1
        or result.get("model_manifest_sha256") not in source_model_shas
        or len(temperature_shas) != 1
        or result.get("temperature_constraint_contract_sha256")
        not in temperature_shas
        or tuple(result.get("constraint_names") or ())
        != tuple(contract["constraint_names"])
        or result.get("fixed_primary_turns") != 6
        or result.get("terminal_population_fixed_primary_turns_verified")
        is not True
        or result.get("terminal_population_primary_turn_values") != [6]
        or result.get("optimizer_resonance_scale_Hz") != 150.0
        or result.get("optimizer_Llt_scale_uH") != 0.25
        or result.get("optimizer_all_active_thermal_scale_C") != 2.0
        or result.get("optimizer_core_thermal_scale_C") is not None
        or result.get("acquisition_ranking_contract") is not None
        or not isinstance(warm, dict)
        or warm.get("sha256") != contract["warm_start"]["sha256"]
        or warm.get("coordinate_count") != 64
        or warm.get("dimension_count") != 25
        or warm.get("post_repair_coordinate_count") != 64
        or warm.get("repaired_by_current_simultaneous_hard_constraint_problem")
        is not True
        or not isinstance(repair, dict)
        or repair != warm_repair
        or repair.get("schema_version")
        != "mft-tier1-fixed-primary-turns-repair-v1"
        or repair.get("fixed_primary_turns") != 6
        or repair.get("coordinate_index") != 0
        or repair.get("hard_constraint_mutation") is not False
        or repair.get("objective_mutation") is not False
        or repair.get("warm_start_post_repair_minimum_unique_count") != 32
        or _json_sha(repair_unsigned) != repair_sha
        or not isinstance(audit, dict)
        or audit.get("schema_version")
        != "mft-tier1-fixed-primary-turns-warm-audit-v1"
        or audit.get("source_coordinate_count") != 64
        or audit.get("post_repair_coordinate_shape") != [64, 25]
        or audit.get("post_repair_coordinate_dtype") != "float64"
        or audit.get("post_repair_decoded_unique_count") != 64
        or audit.get("post_repair_hard_feasible_count") != 64
        or audit.get("post_repair_rejected_count") != 0
        or audit.get("minimum_unique_count") != 32
        or audit.get("fixed_primary_turns") != 6
        or audit.get("fixed_primary_turn_coordinate_verified") is not True
        or audit.get("physical_geometry_dedupe_performed") is not True
        or audit.get("source_sha_verified_before_repair") is not True
        or not isinstance(audit.get("post_repair_coordinates_sha256"), str)
        or len(audit["post_repair_coordinates_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in audit["post_repair_coordinates_sha256"]
        )
        or warm_filter != {
            "input_count": 64,
            "hard_feasible_count": 64,
            "decoded_unique_count": 64,
            "rejected_count": 0,
        }
        or not isinstance(normalization, dict)
        or normalization.get("schema_version")
        != "mft-tier1-optimizer-constraint-normalization-v1"
        or normalization.get("constraint_order") != contract["constraint_names"]
        or normalization.get("authoritative_terminal_G")
        != "physical_unscaled"
        or normalization.get("optimizer_Llt_scale_uH") != 0.25
        or normalization.get("optimizer_all_active_thermal_scale_C") != 2.0
        or normalization.get("optimizer_all_active_thermal_constraints")
        != list(THERMAL_CONSTRAINTS)
        or normalization.get("all_active_thermal_side_activation")
        != "finite_N2_side_gt_0_else_physical_negative_BIG"
        or normalization.get("explicit_optimizer_only_scale_overrides")
        != expected_overrides
        or not isinstance(scales, dict)
        or scales.get("Llt_robust_band") != 0.25
        or scales.get("Llt_ensemble_disagreement") != 0.5
        or any(scales.get(name) != 2.0 for name in THERMAL_CONSTRAINTS)
        or _json_sha(normalization_unsigned) != normalization_sha
        or any(
            result.get(field) is not False
            for field in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed", "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("orthogonal post-repair preflight contract mismatch")
    return {
        "schema_version": ORTHOGONAL_POST_REPAIR_PREFLIGHT_SCHEMA,
        "seed": int(result["seed"]),
        "population": 64,
        "max_generations": 1,
        "completed_generations": int(result["completed_generations"]),
        "source_warm_sha256": warm["sha256"],
        "post_repair_coordinates_sha256": audit[
            "post_repair_coordinates_sha256"
        ],
        "post_repair_coordinate_shape": [64, 25],
        "post_repair_decoded_unique_count": 64,
        "post_repair_hard_feasible_count": 64,
        "post_repair_rejected_count": 0,
        "fixed_primary_turns": 6,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_Llt_disagreement_scale_uH": 0.5,
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "optimizer_all_active_thermal_constraint_count": len(
            THERMAL_CONSTRAINTS
        ),
        "soft_axis_pressure_scope": "disabled_until_stage_feasible",
        "terminal_population_fixed_primary_turns_verified": True,
    }


def seal_orthogonal_post_repair_preflight(
    contract_path: Path, result_path: Path,
    nsga_code_root: Path = DEFAULT_CODE,
) -> dict:
    """Attach an authenticated orthogonal one-generation preflight result."""

    contract_path = contract_path.resolve(strict=True)
    result_path = result_path.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != ORTHOGONAL_SCHEMA:
        raise RuntimeError("orthogonal handoff contract schema mismatch")
    expected_names = list(sealed.EXPECTED_CONSTRAINT_NAMES)
    if (
        contract.get("constraint_names") != expected_names
        or contract.get("constraint_names_sha256") != _json_sha(expected_names)
    ):
        raise RuntimeError("orthogonal constraint-name contract mismatch")
    code_identity = _verify_nsga_code_root(nsga_code_root)
    for key, value in {
        "nsga_code_root_revision": code_identity["revision"],
        "nsga_code_root_revision_verified": True,
        "nsga_code_root_clean_verified": code_identity["clean"],
        "nsga_decoder_relative_path": code_identity["decoder_relative_path"],
        "nsga_decoder_source_sha256": code_identity["decoder_sha256"],
    }.items():
        if contract.get(key) != value:
            raise RuntimeError("orthogonal NSGA code identity mismatch")
    payload = result_path.read_bytes()
    digest = sealed._sha_bytes(payload)
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("orthogonal preflight is not valid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("orthogonal preflight is not a JSON object")
    summary = _orthogonal_post_repair_preflight_summary(contract, result)
    relative = Path("post_repair_preflight") / f"result-{digest}.json"
    destination = contract_path.parent / relative
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError("orthogonal preflight SHA collision")
    else:
        sealed._atomic_bytes(destination, payload)
    contract["post_repair_preflight"] = {
        "schema_version": ORTHOGONAL_POST_REPAIR_PREFLIGHT_SCHEMA,
        "result": {
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(payload),
        },
        "verified_summary": summary,
        "verified_summary_sha256": _json_sha(summary),
        "required_for_deployment": True,
        "verified": True,
    }
    sealed._atomic_json(contract_path, contract)
    return {
        "contract_path": str(contract_path),
        "contract_sha256": sealed._sha_file(contract_path),
        "preflight_result_sha256": digest,
        "summary": summary,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--normalized-root", type=Path, default=DEFAULT_NORMALIZED
    )
    result.add_argument("--thermal-root", type=Path, default=DEFAULT_THERMAL)
    result.add_argument("--fixed6-root", type=Path, default=DEFAULT_FIXED6)
    result.add_argument(
        "--resonance-focus-root", type=Path, default=DEFAULT_RESFOCUS
    )
    result.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--seal-post-repair-preflight-result", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.seal_post_repair_preflight_result is not None:
        result = seal_post_repair_preflight(
            args.output / CONTRACT_FILENAME,
            args.seal_post_repair_preflight_result,
            args.nsga_code_root,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    result = build_handoff(
        normalized_root=args.normalized_root,
        thermal_root=args.thermal_root,
        fixed6_root=args.fixed6_root,
        resonance_focus_root=args.resonance_focus_root,
        nsga_code_root=args.nsga_code_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
