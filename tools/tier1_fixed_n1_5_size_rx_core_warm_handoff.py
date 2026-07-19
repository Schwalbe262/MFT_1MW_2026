"""Build a sealed fixed-N1=5 size/Rx/core Tier-1 warm handoff.

The handoff has two equal, interleaved islands:

* 32 authenticated points that already pass width, all applicable Rx robust
  temperature constraints, 15 kHz resonance, and the robust Llt band; and
* 32 authenticated core-temperature bridge points that preserve width, Rx,
  resonance, and non-core temperature constraints while retaining the
  original (unrelaxed) Llt constraint for optimizer repair.

This command writes only below ``--output``.  It never contacts Slurm, submits
FEA, starts AEDT, publishes a pointer, or promotes a design/model.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_fixed_n1_6_bridge_warm_handoff as shared  # noqa: E402
from tools import tier1_resonance_focus_warm_handoff as sealed  # noqa: E402
from tools.tier1_resonance_feedback import (  # noqa: E402
    CONSTRAINT_VERSION,
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    _json_sha,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    _load_sobol_schema,
    decoded_to_unit,
)


SCHEMA = "mft-tier1-fixed-n1-5-size-rx-core-warm-handoff-v1"
POST_REPAIR_PREFLIGHT_SCHEMA = (
    "mft-tier1-fixed-n1-5-size-rx-core-post-repair-preflight-v1"
)
CONTRACT_FILENAME = "fixed_n1_5_size_rx_core_warm_contract.json"
WARM_SHAPE = (64, 25)
PRESERVED_ISLAND_COUNT = 32
CORE_BRIDGE_ISLAND_COUNT = 32
DECODED_SHA_DUPLICATE_CAP = 1
FIXED_PRIMARY_TURNS = 5
TOLERANCE = 1e-9
CORE_BRIDGE_MAX_LLT_G_UH = 10.5
SOURCE_ROLES = ("resfocus", "deep")
SOURCE_SNAPSHOT_KINDS = shared.SOURCE_SNAPSHOT_KINDS
SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS = shared.SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS
ROLE_NAMESPACES = {
    "resfocus": "resonance-focus250-p320-g600-warm64-v1",
    "deep": "deep-p320-g600-v1",
}
REQUIRED_ANCHOR_TASKS = {"resfocus": 56699, "deep": 56286}
EXPECTED_ANCHOR_IDENTITIES = {
    "resfocus": {
        "task_id": 56699,
        "seed": 1_907_192_020,
        "result_sha256": (
            "1886a5feb5f08d4b167381536cf99ffab4fc43f5cbd1cdd60dd57bb2d3691fc0"
        ),
        "decoded_params_sha256": (
            "46877de4752bbe135601e152b3c27c8c28199edbe8643ffbf56eda0925bf26cb"
        ),
        "constraint_G_sha256": (
            "37bdc3c89a625626b9c842fc847ad163d6684a2ef6ddf4396cfa7542e7e6aa36"
        ),
    },
    "deep": {
        "task_id": 56286,
        "seed": 1_907_190_096,
        "result_sha256": (
            "e3ec3b00dac09b6c650f0dabf0e2e74f387da462a98e6a8a8ee0fc8cfaa6d335"
        ),
        "decoded_params_sha256": (
            "c2ece2eac33a1e900471c35faae81ad4b6b98c15f3d3a322746ef93400dbaaaf"
        ),
        "constraint_G_sha256": (
            "ac53f1f03c42e2e7f74d72b7792597756526558eafc70129dddbcc98cc0f2324"
        ),
    },
}
RX_THERMAL_CONSTRAINTS = (
    "temperature_robust_limit:T_max_Rx_main",
    "temperature_robust_limit:T_max_Rx_side",
    "temperature_robust_limit:Tprobe_Rx_main_leeward_max",
    "temperature_robust_limit:Tprobe_Rx_side_leeward_max",
)
CORE_THERMAL_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_center_leg_max",
    "temperature_robust_limit:Tprobe_core_side_leg_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
NON_CORE_THERMAL_CONSTRAINTS = tuple(
    name for name in sealed.EXPECTED_CONSTRAINT_NAMES
    if name.startswith("temperature_robust_limit:")
    and name not in CORE_THERMAL_CONSTRAINTS
)
RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
PRESERVED_BASE_CONSTRAINTS = (
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "minimum_physical_insulation",
    "Llt_ensemble_disagreement",
    "core_group_manufacturability_limit",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
DEFAULT_RESFOCUS = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\t1r250_260719"
)
DEFAULT_DEEP = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_deep_t110_res15k_260719"
)
DEFAULT_CODE = sealed.DEFAULT_CODE
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-5-size-rx-core-island-v1"
)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contained(root: Path, relative: object, label: str) -> Path:
    root = root.resolve(strict=True)
    value = Path(str(relative or ""))
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise RuntimeError(f"{label} path is not contained")
    path = (root / value).resolve(strict=True)
    if path != root and root not in path.parents:
        raise RuntimeError(f"{label} escaped handoff")
    return path


def _authenticate(root: Path, role: str) -> dict:
    if role not in ROLE_NAMESPACES:
        raise RuntimeError(f"unknown source role: {role}")
    previous = sealed.ROLE_NAMESPACES.get(role)
    existed = role in sealed.ROLE_NAMESPACES
    sealed.ROLE_NAMESPACES[role] = ROLE_NAMESPACES[role]
    try:
        return sealed.authenticate_source(root, role)
    finally:
        if existed:
            sealed.ROLE_NAMESPACES[role] = previous
        else:
            sealed.ROLE_NAMESPACES.pop(role, None)


def _capture_authenticated_source(root: Path, role: str) -> dict:
    """Authenticate then capture one stable canonical head byte-for-byte."""

    last_error = None
    for _attempt in range(SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS):
        source = _authenticate(root, role)
        evidence = source["evidence"]
        payloads = {}
        try:
            for kind in SOURCE_SNAPSHOT_KINDS:
                reference = evidence.get(kind)
                if not isinstance(reference, dict):
                    raise RuntimeError(f"{role} {kind} evidence is missing")
                path = Path(str(reference.get("path") or "")).resolve(
                    strict=True
                )
                payload = path.read_bytes()
                if sealed._sha_bytes(payload) != reference.get("sha256"):
                    raise RuntimeError(f"{role} {kind} changed after auth")
                value = json.loads(payload.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RuntimeError(f"{role} {kind} is not an object")
                payloads[kind] = payload
        except (
            OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError,
        ) as exc:
            last_error = exc
            continue
        status = json.loads(payloads["status"].decode("utf-8"))
        terminal = shared._terminal_snapshot_evidence(status, role)
        if (
            terminal["terminal_result_sha256"]
            != evidence.get("terminal_result_sha256")
            or terminal["terminal_seed_status_sha256"]
            != evidence.get("terminal_seed_status_sha256")
            or _json_sha(terminal["terminal_result_sha256"])
            != evidence.get("terminal_result_set_sha256")
            or _json_sha(terminal["terminal_seed_status_sha256"])
            != evidence.get("terminal_seed_status_set_sha256")
        ):
            last_error = RuntimeError(f"{role} terminal head drifted")
            continue
        evidence.update({
            key: value for key, value in terminal.items()
            if key not in {
                "terminal_result_sha256", "terminal_seed_status_sha256",
            }
        })
        source["_snapshot_payloads"] = payloads
        return source
    raise RuntimeError(
        f"{role} canonical head did not remain stable for capture"
    ) from last_error


def _seal_source_snapshots(
    output: Path, sources: dict[str, dict],
) -> dict[str, dict]:
    snapshots = {}
    for role in SOURCE_ROLES:
        evidence = sources[role]["evidence"]
        payloads = sources[role].get("_snapshot_payloads")
        if not isinstance(payloads, dict):
            raise RuntimeError(f"{role} source bytes were not captured")
        role_snapshots = {}
        for kind in SOURCE_SNAPSHOT_KINDS:
            payload = payloads.get(kind)
            source_reference = evidence.get(kind)
            if not isinstance(payload, bytes) or not isinstance(
                source_reference, dict
            ):
                raise RuntimeError(f"{role} {kind} capture is missing")
            digest = sealed._sha_bytes(payload)
            if digest != source_reference.get("sha256"):
                raise RuntimeError(f"{role} {kind} capture SHA drifted")
            relative = Path("source_snapshots") / role / f"{kind}-{digest}.json"
            destination = output / relative
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise RuntimeError("content-addressed snapshot collision")
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


def _topology_pass(params: dict) -> bool:
    try:
        return bool(
            int(params["N1"]) == FIXED_PRIMARY_TURNS
            and int(params["N1_main"]) == FIXED_PRIMARY_TURNS
            and int(params["N1_side"]) == 0
            and int(params["N2"]) == 50
            and int(params["N2_main"]) + int(params["N2_side"]) == 50
            and int(params["N2_side"]) > 0
            and int(params["n_core_group"]) <= int(
                HARD_SPEC["n_core_group_max"]
            )
            and float(params["cw1"])
            == float(HARD_SPEC["primary_conductor_thickness_mm"])
            and int(params["core_plate_on"]) == 1
            and int(params["wcp_on"]) == 1
            and int(params["cap_on"]) == 1
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _prepare_rows(source: dict, sobol: tuple) -> list[dict]:
    dimensions, n1_min, n1_max, defaults = sobol
    rows = sealed._candidate_rows(source)
    for row in rows:
        params = row["decoded_params"]
        values = row["constraint_G"]
        coordinate = decoded_to_unit(
            params, dimensions, n1_min, n1_max, defaults,
        )
        if (
            coordinate.shape != (WARM_SHAPE[1],)
            or not np.isfinite(coordinate).all()
            or (coordinate < 0.0).any()
            or (coordinate > 1.0).any()
        ):
            raise RuntimeError("decoded warm coordinate is invalid")
        row.update({
            "coordinate": coordinate,
            "topology_pass": _topology_pass(params),
            "width_mm": (
                float(HARD_SPEC["size_W_max_mm"])
                + _finite(values["exterior_width_limit"], "width G")
            ),
            "length_mm": (
                float(HARD_SPEC["size_L_max_mm"])
                + _finite(values["exterior_length_limit"], "length G")
            ),
            "height_mm": (
                float(HARD_SPEC["size_H_max_mm"])
                + _finite(values["exterior_height_limit"], "height G")
            ),
            "rx_max_G_C": max(
                _finite(values[name], name) for name in RX_THERMAL_CONSTRAINTS
            ),
            "non_core_thermal_max_G_C": max(
                _finite(values[name], name)
                for name in NON_CORE_THERMAL_CONSTRAINTS
            ),
            "core_max_G_C": max(
                _finite(values[name], name)
                for name in CORE_THERMAL_CONSTRAINTS
            ),
            "core_positive_sum_C": sum(
                max(_finite(values[name], name), 0.0)
                for name in CORE_THERMAL_CONSTRAINTS
            ),
            "constraint_G_sha256": _json_sha(values),
        })
    return rows


def _base_preserved(row: dict) -> bool:
    values = row["constraint_G"]
    return bool(
        row["topology_pass"]
        and all(
            _finite(values[name], name) <= TOLERANCE
            for name in PRESERVED_BASE_CONSTRAINTS
        )
        and row["rx_max_G_C"] <= TOLERANCE
        and _finite(values[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT)
        <= TOLERANCE
    )


def _preserved_island_eligible(row: dict) -> bool:
    return bool(
        _base_preserved(row)
        and _finite(row["constraint_G"]["strict_full_density_support"], "density")
        <= TOLERANCE
        and _finite(row["constraint_G"]["Llt_robust_band"], "Llt")
        <= TOLERANCE
    )


def _core_bridge_eligible(row: dict) -> bool:
    return bool(
        _base_preserved(row)
        and row["non_core_thermal_max_G_C"] <= TOLERANCE
        and _finite(row["constraint_G"]["Llt_robust_band"], "Llt")
        <= CORE_BRIDGE_MAX_LLT_G_UH
    )


def _preserved_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    size_score = (
        max(row["width_mm"] - 1_000.0, 0.0) / 100.0
        + max(row["length_mm"] - 1_000.0, 0.0) / 100.0
        + max(row["height_mm"] - 700.0, 0.0) / 100.0
    )
    return (
        size_score,
        max(row["core_max_G_C"], 0.0) / 2.0,
        -min(row["rx_max_G_C"], 0.0) / 20.0,
        abs(_finite(values["Llt_robust_band"], "Llt")),
        row["reference"]["seed"],
        row["decoded_params_sha256"],
    )


def _core_bridge_rank(row: dict) -> tuple:
    values = row["constraint_G"]
    score = (
        max(row["core_max_G_C"], 0.0) / 1.0
        + row["core_positive_sum_C"] / 2.0
        + max(_finite(values["Llt_robust_band"], "Llt"), 0.0)
        / float(HARD_SPEC["Llt_tol_uH"])
        + max(_finite(values["strict_full_density_support"], "density"), 0.0)
        + max(row["width_mm"] - 1_000.0, 0.0) / 100.0
        + max(row["length_mm"] - 1_000.0, 0.0) / 100.0
    )
    return (
        score,
        max(row["core_max_G_C"], 0.0),
        max(_finite(values["Llt_robust_band"], "Llt"), 0.0),
        row["reference"]["seed"],
        row["decoded_params_sha256"],
    )


def _select_diverse(
    rows: list[dict], count: int, *, anchor_task: int, rank_key,
    excluded_sha256: set[str] | None = None,
) -> list[dict]:
    excluded_sha256 = excluded_sha256 or set()
    unique = {}
    for row in rows:
        digest = row["decoded_params_sha256"]
        if digest in excluded_sha256:
            continue
        incumbent = unique.get(digest)
        if incumbent is None or rank_key(row) < rank_key(incumbent):
            unique[digest] = row
    ordered = sorted(unique.values(), key=rank_key)
    anchors = [
        index for index, row in enumerate(ordered)
        if int(row["reference"]["task_id"]) == int(anchor_task)
    ]
    if not anchors:
        raise RuntimeError(
            f"required anchor task {anchor_task} has no eligible candidate"
        )
    if len(ordered) < count:
        raise RuntimeError(f"candidate shortage: need {count}, have {len(ordered)}")
    # Keep the search in the ranked island while allowing coordinate diversity.
    ordered = ordered[: max(count * 4, anchors[0] + 1)]
    anchors = [
        index for index, row in enumerate(ordered)
        if int(row["reference"]["task_id"]) == int(anchor_task)
    ]
    selected = [anchors[0]]
    quality = np.arange(len(ordered), dtype=float)
    quality /= max(float(len(ordered) - 1), 1.0)
    while len(selected) < count:
        remaining = [
            index for index in range(len(ordered)) if index not in selected
        ]
        scored = []
        for index in remaining:
            distance = min(
                float(np.linalg.norm(
                    ordered[index]["coordinate"]
                    - ordered[chosen]["coordinate"]
                ))
                for chosen in selected
            )
            scored.append((distance - 0.05 * quality[index], -index, index))
        selected.append(max(scored)[2])
    return [ordered[index] for index in selected]


def _anchor_summary(row: dict) -> dict:
    values = row["constraint_G"]
    return {
        "source_role": row["role"],
        "task_id": int(row["reference"]["task_id"]),
        "seed": int(row["reference"]["seed"]),
        "result_sha256": row["reference"]["result_sha256"],
        "decoded_params_sha256": row["decoded_params_sha256"],
        "constraint_G_sha256": row["constraint_G_sha256"],
        "N1": int(row["decoded_params"]["N1"]),
        "n_core_group": int(row["decoded_params"]["n_core_group"]),
        "cw1_mm": float(row["decoded_params"]["cw1"]),
        "width_mm": row["width_mm"],
        "length_mm": row["length_mm"],
        "height_mm": row["height_mm"],
        "resonance_min_Hz": (
            float(HARD_SPEC["resonance_min_Hz"])
            - float(values[RESONANCE_CONSTRAINT])
        ),
        "Llt_robust_G_uH": float(values["Llt_robust_band"]),
        "Rx_all11_related_max_G_C": row["rx_max_G_C"],
        "core_max_G_C": row["core_max_G_C"],
    }


def _verify_required_anchor(rows: list[dict], role: str) -> dict:
    expected = EXPECTED_ANCHOR_IDENTITIES[role]
    matches = [
        row for row in rows
        if int(row["reference"]["task_id"]) == expected["task_id"]
        and row["decoded_params_sha256"] == expected["decoded_params_sha256"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{role} required anchor identity is absent/ambiguous")
    row = matches[0]
    observed = {
        "task_id": int(row["reference"]["task_id"]),
        "seed": int(row["reference"]["seed"]),
        "result_sha256": row["reference"]["result_sha256"],
        "decoded_params_sha256": row["decoded_params_sha256"],
        "constraint_G_sha256": row["constraint_G_sha256"],
    }
    if observed != expected or not _topology_pass(row["decoded_params"]):
        raise RuntimeError(f"{role} required anchor SHA/physics identity mismatch")
    return row


def audit_required_anchors(
    *, resonance_focus_root: Path = DEFAULT_RESFOCUS,
    deep_root: Path = DEFAULT_DEEP,
    nsga_code_root: Path = DEFAULT_CODE,
) -> dict:
    """Read-only, current-canonical authentication of the two pinned anchors."""

    code_identity = shared._verify_nsga_code_root(nsga_code_root)
    sobol = _load_sobol_schema(nsga_code_root)
    roots = {"resfocus": resonance_focus_root, "deep": deep_root}
    sources = {
        role: _authenticate(roots[role], role) for role in SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if identities[0] != identities[1]:
        raise RuntimeError("anchor source generation identity mismatch")
    anchors = {}
    for role, source in sources.items():
        rows = _prepare_rows(source, sobol)
        anchors[role] = _anchor_summary(_verify_required_anchor(rows, role))
    return {
        "schema_version": "mft-tier1-fixed-n1-5-anchor-audit-v1",
        "authenticated": True,
        "generation_identity": identities[0],
        "anchors": anchors,
        "nsga_code_identity": code_identity,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }


def build_handoff(
    *, resonance_focus_root: Path, deep_root: Path,
    nsga_code_root: Path, output: Path,
) -> dict:
    nsga_code_root = nsga_code_root.resolve(strict=True)
    code_identity = shared._verify_nsga_code_root(nsga_code_root)
    sobol = _load_sobol_schema(nsga_code_root)
    if len(sobol[0]) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    roots = {"resfocus": resonance_focus_root, "deep": deep_root}
    sources = {
        role: _capture_authenticated_source(roots[role], role)
        for role in SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if identities[0] != identities[1]:
        raise RuntimeError("warm source generation identity mismatch")

    rows_by_role = {
        role: _prepare_rows(source, sobol)
        for role, source in sources.items()
    }
    for role in SOURCE_ROLES:
        _verify_required_anchor(rows_by_role[role], role)
    all_rows = [row for rows in rows_by_role.values() for row in rows]
    preserved = _select_diverse(
        [row for row in all_rows if _preserved_island_eligible(row)],
        PRESERVED_ISLAND_COUNT,
        anchor_task=REQUIRED_ANCHOR_TASKS["resfocus"],
        rank_key=_preserved_rank,
    )
    preserved_sha = {row["decoded_params_sha256"] for row in preserved}
    core_bridge = _select_diverse(
        [row for row in all_rows if _core_bridge_eligible(row)],
        CORE_BRIDGE_ISLAND_COUNT,
        anchor_task=REQUIRED_ANCHOR_TASKS["deep"],
        rank_key=_core_bridge_rank,
        excluded_sha256=preserved_sha,
    )
    selected = []
    for index in range(32):
        selected.append(("size_rx_resonance_llt_preserved", preserved[index]))
        selected.append(("core_temperature_bridge", core_bridge[index]))

    coordinates = np.vstack([row["coordinate"] for _kind, row in selected])
    decoded_shas = [
        row["decoded_params_sha256"] for _kind, row in selected
    ]
    if coordinates.shape != WARM_SHAPE or coordinates.dtype != np.float64:
        raise RuntimeError(f"warm pool shape/dtype mismatch: {coordinates.shape}")
    if len(set(decoded_shas)) != len(decoded_shas):
        raise RuntimeError("decoded SHA duplicate cap exceeded")
    coordinate_shas = {
        _json_sha([float(value) for value in coordinate])
        for coordinate in coordinates
    }
    if len(coordinate_shas) != WARM_SHAPE[0]:
        raise RuntimeError("warm coordinates are not unique")
    if not all(_base_preserved(row) for _kind, row in selected):
        raise RuntimeError("selected warm row escaped width/Rx/resonance preserve")

    output = output.resolve()
    source_snapshots = _seal_source_snapshots(output, sources)
    selected_artifacts = shared._seal_selected_artifacts(output, selected)
    warm_path = output / "next_warm_start.npy"
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    sealed._atomic_bytes(warm_path, buffer.getvalue())
    provenance = []
    for warm_index, (category, row) in enumerate(selected):
        reference = row["reference"]
        values = row["constraint_G"]
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
            "constraint_G_sha256": row["constraint_G_sha256"],
            "unit_coordinate_sha256": _json_sha([
                float(value) for value in row["coordinate"]
            ]),
            "N1": int(row["decoded_params"]["N1"]),
            "N1_main": int(row["decoded_params"]["N1_main"]),
            "N1_side": int(row["decoded_params"]["N1_side"]),
            "N2": int(row["decoded_params"]["N2"]),
            "N2_main": int(row["decoded_params"]["N2_main"]),
            "N2_side": int(row["decoded_params"]["N2_side"]),
            "n_core_group": int(row["decoded_params"]["n_core_group"]),
            "cw1_mm": float(row["decoded_params"]["cw1"]),
            "width_mm": row["width_mm"],
            "length_mm": row["length_mm"],
            "height_mm": row["height_mm"],
            "width_G_mm": float(values["exterior_width_limit"]),
            "Rx_all11_related_max_G_C": row["rx_max_G_C"],
            "non_core_thermal_max_G_C": row["non_core_thermal_max_G_C"],
            "core_max_G_C": row["core_max_G_C"],
            "Llt_robust_G_uH": float(values["Llt_robust_band"]),
            "resonance_G_Hz": float(values[RESONANCE_CONSTRAINT]),
            "strict_full_density_G": float(
                values["strict_full_density_support"]
            ),
        })
    anchor_provenance = {
        role: next(
            item for item in provenance
            if item["source_role"] == role
            and item["task_id"] == REQUIRED_ANCHOR_TASKS[role]
            and item["decoded_params_sha256"]
            == EXPECTED_ANCHOR_IDENTITIES[role]["decoded_params_sha256"]
        )
        for role in SOURCE_ROLES
    }
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": (
            "32_size_rx_resonance_llt_preserved_plus_"
            "32_core_temperature_bridge_interleaved_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": sealed._sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "authenticated_decoded_to_unit_then_fixed_n1_5_repair_v1"
            ),
        },
        "category_counts": {
            "size_rx_resonance_llt_preserved": 32,
            "core_temperature_bridge": 32,
        },
        "source_role_counts": {
            role: sum(item["source_role"] == role for item in provenance)
            for role in SOURCE_ROLES
        },
        "required_anchor_tasks": REQUIRED_ANCHOR_TASKS,
        "expected_anchor_identities": EXPECTED_ANCHOR_IDENTITIES,
        "anchors": anchor_provenance,
        "source_evidence": {
            role: source["evidence"] for role, source in sources.items()
        },
        "source_snapshots": source_snapshots,
        "selected_artifact_snapshots": selected_artifacts,
        "selected_artifact_snapshots_sha256": _json_sha(selected_artifacts),
        "selected_provenance": provenance,
        "selected_provenance_sha256": _json_sha(provenance),
        "decoded_sha_duplicate_cap": DECODED_SHA_DUPLICATE_CAP,
        "selected_decoded_sha256_set_sha256": _json_sha(
            sorted(decoded_shas)
        ),
        "warm_source_preserve_contract": {
            "all_selected_width_G_le_zero": True,
            "all_selected_Rx_related_all11_G_le_zero": True,
            "all_selected_resonance_G_le_zero": True,
            "preserved_island_Llt_robust_G_le_zero": True,
            "core_bridge_Llt_G_max_uH": CORE_BRIDGE_MAX_LLT_G_UH,
            "core_bridge_Llt_hard_constraint_relaxed": False,
            "authoritative_terminal_G": "physical_unscaled",
        },
        "topology_contract": {
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "fixed_primary_turns_every_offspring_repair": True,
            "warm_N1_main": FIXED_PRIMARY_TURNS,
            "warm_N1_side": 0,
            "warm_N2_total": 50,
            "warm_N2_side_positive": True,
            "warm_core_and_winding_coldplates_on": True,
            "warm_capacitance_model_on": True,
        },
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
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "fixed_primary_turns_scope": (
            "runner_repair_after_authenticated_inverse_coordinate_handoff"
        ),
        "authoritative_terminal_G": "physical_unscaled",
        "optimizer_scale_scope": (
            "search_pressure_only_physical_G_unchanged"
        ),
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": 1.0,
        "optimizer_Llt_scale_uH": float(HARD_SPEC["Llt_tol_uH"]),
        "soft_axis_target_W_mm": 1_000.0,
        "soft_axis_target_L_mm": 1_000.0,
        "rolling_target": 32,
        "population": 320,
        "max_generations": 600,
        "inference_threads": 8,
        "seed_start": 1_907_199_000,
        "task_priority": -1,
        "post_repair_preflight": None,
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


def _post_repair_preflight_summary(contract: dict, result: dict) -> dict:
    warm = result.get("warm_start")
    audit = (warm or {}).get("fixed_primary_turns_warm_audit")
    repair = result.get("fixed_primary_turns_contract")
    initialization = result.get("initialization_audit") or {}
    warm_filter = initialization.get("warm_filter")
    normalization = result.get("optimizer_constraint_normalization")
    acquisition = result.get("acquisition_ranking_contract")
    repair_unsigned = dict(repair or {})
    repair_sha = repair_unsigned.pop("sha256", None)
    normalization_unsigned = dict(normalization or {})
    normalization_sha = normalization_unsigned.pop("sha256", None)
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
    expected_core = {
        "temperature_robust_limit:T_max_core",
        "temperature_robust_limit:Tprobe_core_center_max",
        "temperature_robust_limit:Tprobe_core_top_yoke_max",
    }
    overrides = (normalization or {}).get(
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
        or result.get("constraint_names") != contract["constraint_names"]
        or result.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or result.get("terminal_population_fixed_primary_turns_verified")
        is not True
        or result.get("terminal_population_primary_turn_values") != [5]
        or not isinstance(warm, dict)
        or warm.get("sha256") != contract["warm_start"]["sha256"]
        or warm.get("coordinate_count") != 64
        or warm.get("dimension_count") != 25
        or warm.get("post_repair_coordinate_count") != 64
        or warm.get("repaired_by_current_simultaneous_hard_constraint_problem")
        is not True
        or not isinstance(repair, dict)
        or repair != warm.get("fixed_primary_turns_repair")
        or repair.get("schema_version")
        != "mft-tier1-fixed-primary-turns-repair-v1"
        or repair.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
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
        or audit.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or audit.get("fixed_primary_turn_coordinate_verified") is not True
        or audit.get("physical_geometry_dedupe_performed") is not True
        or audit.get("source_sha_verified_before_repair") is not True
        or not _is_sha256(audit.get("post_repair_coordinates_sha256"))
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
        or normalization.get("scales", {}).get(RESONANCE_CONSTRAINT) != 150.0
        or not isinstance(overrides, dict)
        or set(overrides) != expected_core
        or set(overrides.values()) != {1.0}
        or _json_sha(normalization_unsigned) != normalization_sha
        or not isinstance(acquisition, dict)
        or acquisition.get("active") is not True
        or acquisition.get("hard_constraint_mutation") is not False
        or acquisition.get("authoritative_terminal_G")
        != "physical_unscaled_unchanged"
        or acquisition.get("core_thermal_positive_G_scale_C") != 1.0
        or acquisition.get("soft_axis_targets_mm") != {
            "exterior_width": 1_000.0,
            "exterior_length": 1_000.0,
        }
        or any(
            result.get(field) is not False
            for field in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed", "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("N1=5 post-repair preflight contract mismatch")
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
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "optimizer_Llt_scale_uH": 0.55,
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": 1.0,
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
        raise RuntimeError("N1=5 handoff contract schema mismatch")
    code_identity = shared._verify_nsga_code_root(nsga_code_root)
    if (
        code_identity["revision"] != contract.get("nsga_code_root_revision")
        or code_identity["decoder_sha256"]
        != contract.get("nsga_decoder_source_sha256")
    ):
        raise RuntimeError("N1=5 handoff NSGA code identity mismatch")
    payload = result_path.read_bytes()
    digest = sealed._sha_bytes(payload)
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("post-repair preflight is not valid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("post-repair preflight is not an object")
    summary = _post_repair_preflight_summary(contract, result)
    relative = Path("post_repair_preflight") / f"result-{digest}.json"
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


def _validate_source_snapshots(contract_root: Path, contract: dict) -> dict:
    sources = contract.get("source_evidence")
    snapshots = contract.get("source_snapshots")
    if (
        not isinstance(sources, dict)
        or set(sources) != set(SOURCE_ROLES)
        or not isinstance(snapshots, dict)
        or set(snapshots) != set(SOURCE_ROLES)
    ):
        raise RuntimeError("N1=5 source inventory mismatch")
    terminal_sets = {}
    for role in SOURCE_ROLES:
        role_snapshots = snapshots[role]
        evidence = sources[role]
        if set(role_snapshots) != set(SOURCE_SNAPSHOT_KINDS):
            raise RuntimeError(f"{role} snapshot inventory mismatch")
        loaded = {}
        for kind in SOURCE_SNAPSHOT_KINDS:
            reference = role_snapshots[kind]
            path = _contained(
                contract_root, reference.get("path"), f"{role} {kind}",
            )
            payload = path.read_bytes()
            digest = sealed._sha_bytes(payload)
            if (
                digest != reference.get("sha256")
                or digest != reference.get("source_sha256")
                or len(payload) != int(reference.get("size_bytes", -1))
                or digest != (evidence.get(kind) or {}).get("sha256")
            ):
                raise RuntimeError(f"{role} {kind} snapshot byte mismatch")
            loaded[kind] = json.loads(payload.decode("utf-8"))
        previous = sealed.ROLE_NAMESPACES.get(role)
        existed = role in sealed.ROLE_NAMESPACES
        sealed.ROLE_NAMESPACES[role] = ROLE_NAMESPACES[role]
        try:
            sealed._validate_source_identity(
                role, loaded["index"], loaded["pointer"], loaded["status"],
            )
        finally:
            if existed:
                sealed.ROLE_NAMESPACES[role] = previous
            else:
                sealed.ROLE_NAMESPACES.pop(role, None)
        current = loaded["pointer"]["current"]
        if (
            current.get("cohort_id") != evidence.get("cohort_id")
            or current.get("bundle_manifest_sha256")
            != evidence.get("bundle_manifest_sha256")
            or current.get("search_profile") != evidence.get("search_profile")
            or current.get("controller_source_sha256")
            != evidence.get("controller_source_sha256")
            or evidence.get("generation_identity") != {
                key: current.get(key) for key in (
                    "constraint_version", "hard_spec_sha256",
                    "source_model_manifest_sha256",
                    "deployment_model_manifest_sha256",
                    "temperature_constraint_contract_sha256",
                    "hard_spec", "nsga_code_revision",
                )
            }
        ):
            raise RuntimeError(f"{role} snapshot generation identity mismatch")
        terminal = shared._terminal_snapshot_evidence(
            loaded["status"], role,
        )
        for key, value in terminal.items():
            if key in {
                "terminal_result_sha256", "terminal_seed_status_sha256",
            }:
                if sorted(value) != sorted(evidence.get(key) or []):
                    raise RuntimeError(f"{role} terminal SHA set mismatch")
            elif evidence.get(key) != value:
                raise RuntimeError(f"{role} terminal snapshot identity mismatch")
        terminal_sets[role] = {
            (
                int(item["task_id"]), int(item["seed"]),
                item["result"]["sha256"], item["remote_status"]["sha256"],
            )
            for item in loaded["status"]["terminal_results"]
        }
    identities = [
        sources[role]["generation_identity"] for role in SOURCE_ROLES
    ]
    if identities[0] != identities[1]:
        raise RuntimeError("N1=5 source generation identity mismatch")
    return terminal_sets


def validate_contract(
    warm_start: Path, nsga_code_root: Path | None = None,
) -> tuple[Path, str]:
    """Fail closed unless all source, row, warm, and preflight seals agree."""

    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(CONTRACT_FILENAME).resolve(strict=True)
    contract_root = contract_path.parent
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    code_identity = shared._verify_nsga_code_root(
        (nsga_code_root or DEFAULT_CODE).resolve(strict=True)
    )
    expected_hard_spec = {
        "B_limit_T": 1.2,
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 110.0,
        "insulation_min_mm": 40.0,
        "magnetizing_inductance_factor": 0.5,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "q_sigma": 1.0,
        "resonance_min_Hz": 15_000.0,
        "size_H_max_mm": 750.0,
        "size_L_max_mm": 1_200.0,
        "size_W_max_mm": 1_200.0,
        "uncertainty_contract": "q90_conformal_half_width_physical_v1",
    }
    warm_meta = contract.get("warm_start")
    selected = contract.get("selected_provenance")
    artifacts = contract.get("selected_artifact_snapshots")
    preflight = contract.get("post_repair_preflight")
    if (
        contract.get("schema_version") != SCHEMA
        or contract.get("selection_contract")
        != (
            "32_size_rx_resonance_llt_preserved_plus_"
            "32_core_temperature_bridge_interleaved_v1"
        )
        or contract.get("hard_spec") != expected_hard_spec
        or contract.get("hard_spec") != HARD_SPEC
        or _json_sha(contract.get("hard_spec"))
        != contract.get("hard_spec_sha256")
        or contract.get("constraint_version") != CONSTRAINT_VERSION
        or contract.get("constraint_names")
        != list(sealed.EXPECTED_CONSTRAINT_NAMES)
        or _json_sha(contract.get("constraint_names"))
        != contract.get("constraint_names_sha256")
        or contract.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or contract.get("nsga_code_root_revision") != code_identity["revision"]
        or contract.get("nsga_decoder_source_sha256")
        != code_identity["decoder_sha256"]
        or contract.get("nsga_code_root_clean_verified") is not True
        or contract.get("fixed_primary_turns") != 5
        or contract.get("optimizer_resonance_scale_Hz") != 150.0
        or contract.get("optimizer_core_thermal_scale_C") != 1.0
        or contract.get("optimizer_Llt_scale_uH") != 0.55
        or contract.get("category_counts") != {
            "size_rx_resonance_llt_preserved": 32,
            "core_temperature_bridge": 32,
        }
        or contract.get("required_anchor_tasks") != REQUIRED_ANCHOR_TASKS
        or contract.get("expected_anchor_identities")
        != EXPECTED_ANCHOR_IDENTITIES
        or contract.get("decoded_sha_duplicate_cap") != 1
        or contract.get("rolling_target") != 32
        or contract.get("population") != 320
        or contract.get("max_generations") != 600
        or contract.get("inference_threads") != 8
        or contract.get("seed_start") != 1_907_199_000
        or contract.get("task_priority") != -1
        or not isinstance(warm_meta, dict)
        or warm_meta.get("filename") != warm_start.name
        or warm_meta.get("sha256") != sealed._sha_file(warm_start)
        or warm_meta.get("shape") != [64, 25]
        or warm_meta.get("dtype") != "float64"
        or not isinstance(selected, list)
        or len(selected) != 64
        or _json_sha(selected) != contract.get("selected_provenance_sha256")
        or not isinstance(artifacts, list)
        or len(artifacts) != 64
        or _json_sha(artifacts)
        != contract.get("selected_artifact_snapshots_sha256")
        or any(
            contract.get(field) is not False for field in (
                "launch_performed", "scheduler_write_performed",
                "fea_submission_approved", "fea_submission_performed",
                "aedt_used", "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("fixed-N1=5 size/Rx/core contract mismatch")
    coordinates = np.load(warm_start, allow_pickle=False)
    coordinate_shas = [
        _json_sha([float(value) for value in coordinates[index]])
        for index in range(len(coordinates))
    ] if (
        isinstance(coordinates, np.ndarray)
        and coordinates.shape == WARM_SHAPE
        and coordinates.dtype == np.float64
        and np.isfinite(coordinates).all()
        and (coordinates >= 0.0).all()
        and (coordinates <= 1.0).all()
    ) else []
    if (
        len(coordinate_shas) != 64
        or len(set(coordinate_shas)) != 64
        or any(
            item.get("warm_index") != index
            or item.get("unit_coordinate_sha256") != coordinate_shas[index]
            for index, item in enumerate(selected)
        )
    ):
        raise RuntimeError("N1=5 warm coordinate/provenance mismatch")

    terminal_sets = _validate_source_snapshots(contract_root, contract)
    artifact_by_index = {
        int(item["warm_index"]): item for item in artifacts
    }
    sobol = _load_sobol_schema(nsga_code_root or DEFAULT_CODE)
    decoded_shas = []
    for index, item in enumerate(selected):
        artifact = artifact_by_index.get(index)
        if not isinstance(artifact, dict):
            raise RuntimeError("selected artifact index mismatch")
        parsed = {}
        for kind in ("result", "seed_status"):
            reference = artifact.get(kind)
            path = _contained(
                contract_root, (reference or {}).get("path"),
                f"selected {kind}",
            )
            payload = path.read_bytes()
            digest = sealed._sha_bytes(payload)
            if (
                digest != reference.get("sha256")
                or len(payload) != int(reference.get("size_bytes", -1))
            ):
                raise RuntimeError("selected artifact byte seal mismatch")
            parsed[kind] = json.loads(payload.decode("utf-8"))
        result = parsed["result"]
        status = parsed["seed_status"]
        candidates = (
            result.get("next_target_fea_batch_plan") or {}
        ).get("candidates")
        matches = [
            candidate for candidate in candidates or []
            if candidate.get("decoded_params_sha256")
            == item.get("decoded_params_sha256")
        ]
        if len(matches) != 1:
            raise RuntimeError("selected decoded identity is absent/ambiguous")
        candidate = matches[0]
        params = candidate.get("decoded_params")
        values = candidate.get("constraint_G")
        reference_tuple = (
            int(item["task_id"]), int(item["seed"]),
            item["result_sha256"], item["seed_status_sha256"],
        )
        if (
            artifact.get("source_role") != item.get("source_role")
            or artifact.get("task_id") != item.get("task_id")
            or artifact.get("seed") != item.get("seed")
            or artifact["result"]["sha256"] != item.get("result_sha256")
            or artifact["seed_status"]["sha256"]
            != item.get("seed_status_sha256")
            or reference_tuple not in terminal_sets[item["source_role"]]
            or status.get("state") != "completed"
            or int(status.get("exit_code", -1)) != 0
            or int(status.get("seed", -1)) != int(item["seed"])
            or str(status.get("task_id")) != str(item["task_id"])
            or status.get("result_sha256") != item["result_sha256"]
            or not isinstance(params, dict)
            or _json_sha(params) != item.get("decoded_params_sha256")
            or not isinstance(values, dict)
            or set(values) != set(sealed.EXPECTED_CONSTRAINT_NAMES)
            or _json_sha(values) != item.get("constraint_G_sha256")
            or not _topology_pass(params)
        ):
            raise RuntimeError("selected artifact semantic mismatch")
        coordinate = decoded_to_unit(params, *sobol)
        if _json_sha([float(value) for value in coordinate]) != coordinate_shas[index]:
            raise RuntimeError("selected decoded-to-unit mismatch")
        prepared = {
            "decoded_params": params,
            "constraint_G": values,
            "topology_pass": True,
            "width_mm": 1_200.0 + float(values["exterior_width_limit"]),
            "length_mm": 1_200.0 + float(values["exterior_length_limit"]),
            "height_mm": 750.0 + float(values["exterior_height_limit"]),
            "rx_max_G_C": max(float(values[name]) for name in RX_THERMAL_CONSTRAINTS),
            "non_core_thermal_max_G_C": max(
                float(values[name]) for name in NON_CORE_THERMAL_CONSTRAINTS
            ),
        }
        category = item.get("category")
        if (
            not _base_preserved(prepared)
            or float(item.get("width_G_mm")) > TOLERANCE
            or float(item.get("Rx_all11_related_max_G_C")) > TOLERANCE
            or float(item.get("resonance_G_Hz")) > TOLERANCE
            or (
                category == "size_rx_resonance_llt_preserved"
                and float(values["Llt_robust_band"]) > TOLERANCE
            )
            or (
                category == "core_temperature_bridge"
                and (
                    prepared["non_core_thermal_max_G_C"] > TOLERANCE
                    or float(values["Llt_robust_band"])
                    > CORE_BRIDGE_MAX_LLT_G_UH
                )
            )
            or category not in {
                "size_rx_resonance_llt_preserved",
                "core_temperature_bridge",
            }
        ):
            raise RuntimeError("selected island physical preserve mismatch")
        decoded_shas.append(item["decoded_params_sha256"])
    if (
        len(set(decoded_shas)) != 64
        or _json_sha(sorted(decoded_shas))
        != contract.get("selected_decoded_sha256_set_sha256")
        or [item["category"] for item in selected] != [
            "size_rx_resonance_llt_preserved" if index % 2 == 0
            else "core_temperature_bridge"
            for index in range(64)
        ]
    ):
        raise RuntimeError("N1=5 island quota/duplicate mismatch")
    for role, identity in EXPECTED_ANCHOR_IDENTITIES.items():
        anchor = contract.get("anchors", {}).get(role)
        if (
            anchor not in selected
            or anchor.get("task_id") != identity["task_id"]
            or anchor.get("seed") != identity["seed"]
            or anchor.get("result_sha256") != identity["result_sha256"]
            or anchor.get("decoded_params_sha256")
            != identity["decoded_params_sha256"]
            or anchor.get("constraint_G_sha256")
            != identity["constraint_G_sha256"]
        ):
            raise RuntimeError("N1=5 required anchor provenance mismatch")

    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != POST_REPAIR_PREFLIGHT_SCHEMA
        or preflight.get("required_for_deployment") is not True
        or preflight.get("verified") is not True
    ):
        raise RuntimeError("N1=5 post-repair preflight is not sealed")
    reference = preflight.get("result")
    preflight_path = _contained(
        contract_root, (reference or {}).get("path"), "post-repair preflight",
    )
    payload = preflight_path.read_bytes()
    if (
        sealed._sha_bytes(payload) != reference.get("sha256")
        or len(payload) != int(reference.get("size_bytes", -1))
    ):
        raise RuntimeError("N1=5 post-repair preflight byte mismatch")
    summary = _post_repair_preflight_summary(
        contract, json.loads(payload.decode("utf-8")),
    )
    if (
        summary != preflight.get("verified_summary")
        or _json_sha(summary) != preflight.get("verified_summary_sha256")
    ):
        raise RuntimeError("N1=5 post-repair preflight summary mismatch")
    return contract_path, sealed._sha_file(contract_path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-anchors")
    audit.add_argument("--resonance-focus-root", type=Path, default=DEFAULT_RESFOCUS)
    audit.add_argument("--deep-root", type=Path, default=DEFAULT_DEEP)
    audit.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    build = subparsers.add_parser("build")
    build.add_argument("--resonance-focus-root", type=Path, default=DEFAULT_RESFOCUS)
    build.add_argument("--deep-root", type=Path, default=DEFAULT_DEEP)
    build.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    seal = subparsers.add_parser("seal-preflight")
    seal.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    seal.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    seal.add_argument("--result", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "audit-anchors":
        value = audit_required_anchors(
            resonance_focus_root=args.resonance_focus_root,
            deep_root=args.deep_root,
            nsga_code_root=args.nsga_code_root,
        )
    elif args.command == "build":
        value = build_handoff(
            resonance_focus_root=args.resonance_focus_root,
            deep_root=args.deep_root,
            nsga_code_root=args.nsga_code_root,
            output=args.output,
        )
    elif args.command == "seal-preflight":
        value = seal_post_repair_preflight(
            args.output / CONTRACT_FILENAME,
            args.result,
            args.nsga_code_root,
        )
    else:
        path, digest = validate_contract(
            args.output / "next_warm_start.npy", args.nsga_code_root,
        )
        value = {"contract_path": str(path), "contract_sha256": digest}
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
