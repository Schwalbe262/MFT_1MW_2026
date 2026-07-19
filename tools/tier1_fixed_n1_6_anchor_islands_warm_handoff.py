"""Build a sealed two-island warm pool for the fixed-N1=6 crossover search.

The source side is deliberately read-only.  Nine live Tier-1 canonicals are
captured and authenticated byte-for-byte, while every file written by this
command is contained below ``--output``.  The command never calls Slurm,
submits FEA, starts AEDT, publishes a model pointer, or promotes a design.

The warm pool keeps two physically meaningful fronts alive:

* ``resonance_llt_anchor`` preserves resonance ``G <= +150 Hz`` and robust
  leakage-inductance ``G <= +0.05 uH`` while the optimizer attacks all eleven
  robust temperature constraints.
* ``thermal_llt_staircase_*`` preserves all eleven thermal constraints and
  robust leakage inductance at physical ``G <= 0`` while five optimizer-only
  resonance caps are explored from ``+2000`` down to ``0 Hz``.

All values recorded here are authoritative, unscaled physical ``G`` values.
Optimizer-only allowances/scales are a separate, fail-closed runtime contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import tier1_fixed_n1_6_bridge_warm_handoff as bridge  # noqa: E402
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


SCHEMA = "mft-tier1-fixed-n1-6-anchor-islands-warm-handoff-v1"
CONTRACT_FILENAME = "fixed_n1_6_anchor_islands_warm_contract.json"
WARM_SHAPE = (64, 25)
TOLERANCE = 1e-9
RESONANCE_EPSILON_HZ = 150.0
LLT_ANCHOR_EPSILON_UH = 0.05
RESONANCE_ANCHOR_QUOTA = 4
STAIRCASE_CAPS_HZ = (2_000.0, 1_500.0, 1_000.0, 500.0, 0.0)
STAIRCASE_QUOTA_PER_CAP = 12
DECODED_SHA_DUPLICATE_CAP = 1
SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS = 8
SOURCE_SNAPSHOT_KINDS = ("index", "pointer", "status")

SOURCE_ROLES = (
    "main",
    "deep",
    "normalized",
    "resfocus",
    "thermal1",
    "thermal4",
    "fixed5",
    "fixed6",
    "bridge6",
)
ROLE_NAMESPACES = {
    "main": None,
    "deep": "deep-p320-g600-v1",
    "normalized": "normalized-p320-g600-warm64-v1",
    "resfocus": "resonance-focus250-p320-g600-warm64-v1",
    "thermal1": "thermal-crossover-core1c-res250-p320-g600-warm64-v1",
    "thermal4": "thermal-crossover-balanced4c-res250-p320-g600-warm64-v1",
    "fixed5": "fixed-n1-5-thermal-llt-core1c-res250-p320-g600-warm64-v1",
    "fixed6": "fixed-n1-6-resonance-llt-core4c-res100-p320-g600-warm64-v1",
    "bridge6": (
        "fixed-n1-6-thermal-bridge-core2c-llt0p55-res150-"
        "p320-g600-warm64-v1"
    ),
}
ROLE_CONTROLLER_SHA256 = {
    "main": "22914cd2ad0d0b184595f0fb0ad335943f3641d7123c11cf0f03c0b23c1d3c89",
    "deep": "b9b0ac7f0340b097533ba109d7d30c0337f6107a916d1db3f8f02f4b1b1c26f0",
    "normalized": "0b48bcf01e691be0ee217700f173d6cf42f80902f5457251684bf19ac0de7e86",
    "resfocus": "5097c476f983f6460861ff7a53aaef4472fc67eda0924006db3d4429fd27052a",
    "thermal1": "ffc451375af92ac13910f59c9dd6888a0e1d595e6767ba41f6927cee3bccc71d",
    "thermal4": "c0b9691d4dadd699d684b0fa92a2133485719b89c8c6462b45b54a38a61c9e18",
    "fixed5": "fbc1c784d06e2a7b3fc4b60fab0db665e8707fafb658451fd7a4e22a6762f2a2",
    "fixed6": "fbc1c784d06e2a7b3fc4b60fab0db665e8707fafb658451fd7a4e22a6762f2a2",
    "bridge6": "313c017e4372cff09b0f7429e0b86456261f28577139215e56784f55eda9b878",
}
ROLE_PROFILE_SHA256 = {
    "main": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
    "deep": "fb3606d3a27aefc8d9937fe654a604bbdc227a00516c64a1a601dcadb6bb6f3f",
    "normalized": "134ecec74586f2c807cc1f3f3bc38959be524baf4040a83b59b04a7eedc15049",
    "resfocus": "b62a33362f8c16e757da8beda5e41d2319be52bb780dba767f37a0d7051fee46",
    "thermal1": "0c349508e14c78811c67be2b1dbcb1daeeeedbaeaba0ae18279d21ff1d7e5ec3",
    "thermal4": "cca6b8585549a3f63c02e529f054991a223502bbdb98349cb8270a41f5c27e39",
    "fixed5": "8ad06c3c336d4f121360b981bf42baa0bf9e7626de0076d2bfa3abbeacad132c",
    "fixed6": "56792a1bfe789c93068ac9422edb07cc9960bd60d9908fa9dac1962e0bc51974",
    "bridge6": "52b7ca05a056ba6f4083bc52b1a23a8cf3b427125661550369fbd83877d9f102",
}

DEFAULT_ROOTS = {
    "main": Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling"
    ),
    "deep": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_deep_t110_res15k_260719"
    ),
    "normalized": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_normalized_t110_res15k_260719"
    ),
    "resfocus": Path(r"C:\Users\peets\slurm_scheduler_runtime\t1r250_260719"),
    "thermal1": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_thermal_crossover_t110_res15k_260719"
    ),
    "thermal4": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_thermal_balanced4c_t110_res15k_260719"
    ),
    "fixed5": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_5_thermal_llt_t110_res15k_260719"
    ),
    "fixed6": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_6_resonance_llt_t110_res15k_260719"
    ),
    "bridge6": Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_6_thermal_bridge_t110_res15k_260719"
    ),
}
DEFAULT_CODE = sealed.DEFAULT_CODE
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-6-anchor-islands-v1"
)

THERMAL_CONSTRAINTS = tuple(
    name for name in sealed.EXPECTED_CONSTRAINT_NAMES
    if name.startswith("temperature_robust_limit:")
)
RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
LLT_CONSTRAINT = "Llt_robust_band"
ANCHOR_CONSTRAINTS = set(THERMAL_CONSTRAINTS) | {
    RESONANCE_CONSTRAINT, LLT_CONSTRAINT,
}
SUPPORT_CONSTRAINTS = tuple(
    name for name in sealed.EXPECTED_CONSTRAINT_NAMES
    if name not in ANCHOR_CONSTRAINTS
)


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


def _authenticate(root: Path, role: str) -> dict:
    if role not in SOURCE_ROLES:
        raise RuntimeError(f"unknown anchor-island source role: {role}")
    previous = sealed.ROLE_NAMESPACES.get(role)
    had_previous = role in sealed.ROLE_NAMESPACES
    sealed.ROLE_NAMESPACES[role] = ROLE_NAMESPACES[role]
    try:
        source = sealed.authenticate_source(root, role)
    finally:
        if had_previous:
            sealed.ROLE_NAMESPACES[role] = previous
        else:
            sealed.ROLE_NAMESPACES.pop(role, None)
    evidence = source.get("evidence") or {}
    status_reference = evidence.get("status")
    if not isinstance(status_reference, dict):
        raise RuntimeError(f"{role} authenticated status reference is missing")
    status_path = Path(str(status_reference.get("path") or "")).resolve(
        strict=True
    )
    status_payload = status_path.read_bytes()
    if sealed._sha_bytes(status_payload) != status_reference.get("sha256"):
        raise RuntimeError(f"{role} status changed after authentication")
    try:
        status = json.loads(status_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{role} authenticated status is invalid") from exc
    if (
        evidence.get("controller_source_sha256")
        != ROLE_CONTROLLER_SHA256[role]
        or _json_sha(status.get("search_profile"))
        != ROLE_PROFILE_SHA256[role]
    ):
        raise RuntimeError(f"{role} controller/search-profile seal mismatch")
    evidence["status_search_profile"] = status.get("search_profile")
    evidence["status_search_profile_sha256"] = _json_sha(
        status.get("search_profile")
    )
    return source


def _capture_authenticated_source(root: Path, role: str) -> dict:
    """Authenticate one rolling head and retain the exact authenticated bytes."""

    last_error: BaseException | None = None
    for _attempt in range(SOURCE_SNAPSHOT_CAPTURE_ATTEMPTS):
        source = _authenticate(root, role)
        evidence = source["evidence"]
        payloads: dict[str, bytes] = {}
        try:
            for kind in SOURCE_SNAPSHOT_KINDS:
                reference = evidence.get(kind)
                if not isinstance(reference, dict):
                    raise RuntimeError(f"{role} {kind} source evidence is missing")
                path = Path(str(reference.get("path") or "")).resolve(strict=True)
                payload = path.read_bytes()
                if sealed._sha_bytes(payload) != reference.get("sha256"):
                    raise RuntimeError(f"{role} {kind} changed after authentication")
                parsed = json.loads(payload.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"{role} {kind} snapshot is not an object")
                payloads[kind] = payload
            status = json.loads(payloads["status"].decode("utf-8"))
            terminal = bridge._terminal_snapshot_evidence(status, role)
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
                raise RuntimeError(f"{role} captured terminal references drifted")
        except (
            OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError,
        ) as exc:
            last_error = exc
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
        f"{role} canonical head did not remain stable for authenticated capture"
    ) from last_error


def _seal_source_snapshots(output: Path, sources: dict[str, dict]) -> dict:
    snapshots = {}
    for role in SOURCE_ROLES:
        evidence = sources[role]["evidence"]
        payloads = sources[role].get("_snapshot_payloads")
        if not isinstance(payloads, dict):
            raise RuntimeError(f"{role} source snapshot bytes were not captured")
        role_snapshots = {}
        for kind in SOURCE_SNAPSHOT_KINDS:
            payload = payloads.get(kind)
            reference = evidence.get(kind)
            if not isinstance(payload, bytes) or not isinstance(reference, dict):
                raise RuntimeError(f"{role} {kind} snapshot evidence is missing")
            digest = sealed._sha_bytes(payload)
            if digest != reference.get("sha256"):
                raise RuntimeError(f"{role} {kind} captured SHA drifted")
            relative = Path("source_snapshots") / role / f"{kind}-{digest}.json"
            destination = output / relative
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise RuntimeError(f"content-addressed {role} {kind} collision")
            else:
                sealed._atomic_bytes(destination, payload)
            role_snapshots[kind] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
                "source_sha256": reference["sha256"],
            }
        snapshots[role] = role_snapshots
    return snapshots


def _original_n1(params: dict) -> int:
    value = params.get("N1")
    if value is None:
        value = _finite(params.get("N1_main"), "N1_main") + _finite(
            params.get("N1_side"), "N1_side"
        )
    value = _finite(value, "N1")
    if not value.is_integer():
        raise RuntimeError("decoded N1 is not integral")
    return int(value)


def _prepare_rows(source: dict) -> list[dict]:
    rows = sealed._candidate_rows(source)
    for row in rows:
        values = row["constraint_G"]
        thermal = {
            name: _finite(values[name], name) for name in THERMAL_CONSTRAINTS
        }
        support = {
            name: _finite(values[name], name) for name in SUPPORT_CONSTRAINTS
        }
        resonance = _finite(values[RESONANCE_CONSTRAINT], RESONANCE_CONSTRAINT)
        llt = _finite(values[LLT_CONSTRAINT], LLT_CONSTRAINT)
        support_pass = all(value <= TOLERANCE for value in support.values())
        row.update({
            "original_N1": _original_n1(row["decoded_params"]),
            "support_pass": support_pass,
            "all_thermal_pass": all(
                value <= TOLERANCE for value in thermal.values()
            ),
            "thermal_max_G_C": float(max(thermal.values())),
            "thermal_positive_sum_C": float(sum(
                max(value, 0.0) for value in thermal.values()
            )),
            "resonance_G_Hz": resonance,
            "Llt_robust_G_uH": llt,
            "resonance_llt_anchor_eligible": bool(
                support_pass
                and resonance <= RESONANCE_EPSILON_HZ + TOLERANCE
                and llt <= LLT_ANCHOR_EPSILON_UH + TOLERANCE
            ),
            "thermal_llt_anchor_eligible": bool(
                support_pass
                and all(value <= TOLERANCE for value in thermal.values())
                and llt <= TOLERANCE
            ),
        })
    return rows


def _rank_identity(row: dict) -> tuple:
    reference = row["reference"]
    return (
        int(reference["seed"]),
        str(reference["result_sha256"]),
        str(row["decoded_params_sha256"]),
    )


def _deduplicate(rows: list[dict], rank_key: Callable[[dict], tuple]) -> list[dict]:
    unique: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        digest = str(row.get("decoded_params_sha256") or "")
        if len(digest) != 64 or _json_sha(row.get("decoded_params")) != digest:
            raise RuntimeError("candidate decoded SHA contract mismatch")
        counts[digest] += 1
        incumbent = unique.get(digest)
        if incumbent is None or rank_key(row) < rank_key(incumbent):
            unique[digest] = row
    if any(count < 1 for count in counts.values()):  # pragma: no cover - defensive
        raise RuntimeError("decoded SHA occurrence accounting failed")
    return sorted(unique.values(), key=rank_key)


def _resonance_anchor_rank(row: dict) -> tuple:
    return (
        max(row["thermal_max_G_C"], 0.0),
        row["thermal_positive_sum_C"],
        max(row["resonance_G_Hz"], 0.0) / RESONANCE_EPSILON_HZ,
        max(row["Llt_robust_G_uH"], 0.0) / LLT_ANCHOR_EPSILON_UH,
        *_rank_identity(row),
    )


def _staircase_rank(cap_hz: float) -> Callable[[dict], tuple]:
    def rank(row: dict) -> tuple:
        return (
            max(row["resonance_G_Hz"] - cap_hz, 0.0) / 150.0,
            max(row["resonance_G_Hz"], 0.0) / 150.0,
            -min(row["thermal_max_G_C"], 0.0) / 2.0,
            -min(row["Llt_robust_G_uH"], 0.0) / 0.05,
            *_rank_identity(row),
        )
    return rank


def _select_diverse(
    rows: list[dict], count: int, *, rank_key: Callable[[dict], tuple],
    excluded_sha: set[str] | None = None, required_roles: tuple[str, ...] = (),
) -> list[dict]:
    excluded_sha = set(excluded_sha or ())
    rows = [
        row for row in _deduplicate(rows, rank_key)
        if row["decoded_params_sha256"] not in excluded_sha
    ]
    if len(rows) < count:
        raise RuntimeError(f"candidate shortage: need {count}, have {len(rows)}")
    selected: list[int] = []
    for role in required_roles:
        match = next(
            (index for index, row in enumerate(rows) if row["role"] == role),
            None,
        )
        if match is None:
            raise RuntimeError(f"required anchor source role is absent: {role}")
        selected.append(match)
    if not selected:
        selected.append(0)
    selected = list(dict.fromkeys(selected))
    quality = np.arange(len(rows), dtype=float)
    quality /= max(float(len(rows) - 1), 1.0)
    while len(selected) < count:
        remaining = [index for index in range(len(rows)) if index not in selected]
        scored = []
        for index in remaining:
            distance = min(float(np.linalg.norm(
                rows[index]["coordinate"] - rows[chosen]["coordinate"]
            )) for chosen in selected)
            scored.append((distance - 0.05 * quality[index], -index, index))
        selected.append(max(scored)[2])
    return [rows[index] for index in selected]


def _seal_selected_artifacts(
    output: Path, selected: list[tuple[str, dict, float | None]],
) -> list[dict]:
    # Reuse the already exhaustive result/seed-status semantic verifier.
    bridge_selected = [
        (category, row) for category, row, _cap_hz in selected
    ]
    return bridge._seal_selected_artifacts(output, bridge_selected)


def build_handoff(
    *, roots: dict[str, Path], nsga_code_root: Path, output: Path,
) -> dict:
    if set(roots) != set(SOURCE_ROLES):
        raise RuntimeError("anchor-island source root inventory mismatch")
    code_identity = bridge._verify_nsga_code_root(
        nsga_code_root.resolve(strict=True)
    )
    sources = {
        role: _capture_authenticated_source(roots[role], role)
        for role in SOURCE_ROLES
    }
    identities = [
        source["evidence"]["generation_identity"]
        for source in sources.values()
    ]
    if not all(identity == identities[0] for identity in identities[1:]):
        raise RuntimeError("nine-source generation identity mismatch")

    sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(
        nsga_code_root.resolve(strict=True)
    )
    if len(sobol_dims) != WARM_SHAPE[1]:
        raise RuntimeError("warm coordinate dimension is not exactly 25")
    rows = []
    for role in SOURCE_ROLES:
        role_rows = _prepare_rows(sources[role])
        for row in role_rows:
            row["coordinate"] = decoded_to_unit(
                row["decoded_params"], sobol_dims, n1_min, n1_max, defaults,
            )
            if (
                row["coordinate"].shape != (WARM_SHAPE[1],)
                or not np.isfinite(row["coordinate"]).all()
                or (row["coordinate"] < 0.0).any()
                or (row["coordinate"] > 1.0).any()
            ):
                raise RuntimeError("decoded anchor coordinate is invalid")
            if row["original_N1"] == 6:
                rows.append(row)

    resonance_rows = [
        row for row in rows if row["resonance_llt_anchor_eligible"]
    ]
    resonance_selected = _select_diverse(
        resonance_rows,
        RESONANCE_ANCHOR_QUOTA,
        rank_key=_resonance_anchor_rank,
        required_roles=("fixed6",),
    )
    selected_sha = {
        row["decoded_params_sha256"] for row in resonance_selected
    }
    staircase_rows = [
        row for row in rows if row["thermal_llt_anchor_eligible"]
    ]
    staircase_selected: dict[float, list[dict]] = {}
    for index, cap_hz in enumerate(STAIRCASE_CAPS_HZ):
        selected = _select_diverse(
            staircase_rows,
            STAIRCASE_QUOTA_PER_CAP,
            rank_key=_staircase_rank(cap_hz),
            excluded_sha=selected_sha,
            required_roles=(
                ("normalized", "thermal1", "resfocus") if index == 0 else ()
            ),
        )
        staircase_selected[cap_hz] = selected
        selected_sha.update(row["decoded_params_sha256"] for row in selected)

    selected: list[tuple[str, dict, float | None]] = [
        ("resonance_llt_anchor", row, None) for row in resonance_selected
    ]
    for offset in range(STAIRCASE_QUOTA_PER_CAP):
        for cap_hz in STAIRCASE_CAPS_HZ:
            selected.append((
                f"thermal_llt_staircase_{int(cap_hz)}hz",
                staircase_selected[cap_hz][offset],
                cap_hz,
            ))
    if len(selected) != WARM_SHAPE[0]:
        raise RuntimeError("anchor-island selection size mismatch")
    coordinates = np.vstack([row["coordinate"] for _category, row, _cap in selected])
    geometry_counts = Counter(
        row["decoded_params_sha256"] for _category, row, _cap in selected
    )
    if coordinates.shape != WARM_SHAPE:
        raise RuntimeError(f"warm pool shape mismatch: {coordinates.shape}")
    if len({tuple(np.round(row, 12)) for row in coordinates}) != WARM_SHAPE[0]:
        raise RuntimeError("anchor-island warm coordinates are not unique")
    if max(geometry_counts.values(), default=0) > DECODED_SHA_DUPLICATE_CAP:
        raise RuntimeError("anchor-island decoded SHA duplicate cap exceeded")

    output = output.resolve()
    source_snapshots = _seal_source_snapshots(output, sources)
    selected_artifacts = _seal_selected_artifacts(output, selected)
    buffer = io.BytesIO()
    np.save(buffer, coordinates, allow_pickle=False)
    warm_path = output / "next_warm_start.npy"
    sealed._atomic_bytes(warm_path, buffer.getvalue())

    provenance = []
    for warm_index, (category, row, cap_hz) in enumerate(selected):
        reference = row["reference"]
        provenance.append({
            "warm_index": warm_index,
            "category": category,
            "source_role": row["role"],
            "cohort_id": reference["cohort_id"],
            "task_id": reference["task_id"],
            "seed": reference["seed"],
            "result_sha256": reference["result_sha256"],
            "seed_status_sha256": reference["seed_status_sha256"],
            "decoded_params_sha256": row["decoded_params_sha256"],
            "unit_coordinate_sha256": _json_sha([
                float(value) for value in row["coordinate"]
            ]),
            "original_N1": row["original_N1"],
            "support_pass": row["support_pass"],
            "all_thermal_pass": row["all_thermal_pass"],
            "resonance_llt_anchor_eligible": row[
                "resonance_llt_anchor_eligible"
            ],
            "thermal_llt_anchor_eligible": row[
                "thermal_llt_anchor_eligible"
            ],
            "resonance_G_Hz": row["resonance_G_Hz"],
            "Llt_robust_G_uH": row["Llt_robust_G_uH"],
            "thermal_max_G_C": row["thermal_max_G_C"],
            "thermal_positive_sum_C": row["thermal_positive_sum_C"],
            "optimizer_resonance_cap_Hz": cap_hz,
            "optimizer_resonance_cap_excess_Hz": (
                max(row["resonance_G_Hz"] - cap_hz, 0.0)
                if cap_hz is not None else None
            ),
        })
    category_counts = dict(sorted(Counter(
        item["category"] for item in provenance
    ).items()))
    expected_category_counts = {
        "resonance_llt_anchor": RESONANCE_ANCHOR_QUOTA,
        **{
            f"thermal_llt_staircase_{int(cap)}hz": STAIRCASE_QUOTA_PER_CAP
            for cap in STAIRCASE_CAPS_HZ
        },
    }
    if category_counts != dict(sorted(expected_category_counts.items())):
        raise RuntimeError("anchor-island category quota mismatch")
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": (
            "n1_6_support_pass_resllt_eps150hz_0p05uh_quota4_plus_"
            "thermal11_llt_pass_five_resonance_cap_strata_quota12_each_v1"
        ),
        "warm_start": {
            "filename": warm_path.name,
            "sha256": sealed._sha_file(warm_path),
            "shape": list(coordinates.shape),
            "dtype": str(coordinates.dtype),
            "coordinate_contract": (
                "authenticated_n1_6_decoded_to_unit_then_current_repair_v1"
            ),
        },
        "island_quotas": {
            "resonance_llt_anchor": RESONANCE_ANCHOR_QUOTA,
            "thermal_llt_staircase_total": (
                len(STAIRCASE_CAPS_HZ) * STAIRCASE_QUOTA_PER_CAP
            ),
            "thermal_llt_staircase_per_cap": STAIRCASE_QUOTA_PER_CAP,
        },
        "category_counts": category_counts,
        "staircase_caps_Hz": list(STAIRCASE_CAPS_HZ),
        "resonance_anchor_epsilon_Hz": RESONANCE_EPSILON_HZ,
        "Llt_anchor_epsilon_uH": LLT_ANCHOR_EPSILON_UH,
        "decoded_params_sha256_duplicate_cap": DECODED_SHA_DUPLICATE_CAP,
        "selected_decoded_params_sha256_count": len(geometry_counts),
        "selected_source_role_counts": dict(sorted(Counter(
            item["source_role"] for item in provenance
        ).items())),
        "source_roles_authenticated": list(SOURCE_ROLES),
        "source_role_count": len(SOURCE_ROLES),
        "source_evidence": {
            role: sources[role]["evidence"] for role in SOURCE_ROLES
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
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "optimizer_allowance_scope": (
            "optimizer_only_physical_terminal_G_unchanged"
        ),
        "soft_axis_pressure_enabled": False,
        "soft_axis_pressure_contract": None,
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


def _contained_evidence(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} path is missing")
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise RuntimeError(f"{label} escaped handoff containment")
    return path


def validate_handoff_contract(
    warm_start: Path, nsga_code_root: Path | None = None,
) -> tuple[Path, str]:
    """Re-authenticate the immutable handoff before any deployment prepare."""

    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(CONTRACT_FILENAME).resolve(strict=True)
    root = contract_path.parent.resolve(strict=True)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("anchor-island handoff contract is invalid JSON") from exc
    expected_categories = {
        "resonance_llt_anchor": RESONANCE_ANCHOR_QUOTA,
        **{
            f"thermal_llt_staircase_{int(cap)}hz": STAIRCASE_QUOTA_PER_CAP
            for cap in STAIRCASE_CAPS_HZ
        },
    }
    expected_quotas = {
        "resonance_llt_anchor": RESONANCE_ANCHOR_QUOTA,
        "thermal_llt_staircase_total": (
            len(STAIRCASE_CAPS_HZ) * STAIRCASE_QUOTA_PER_CAP
        ),
        "thermal_llt_staircase_per_cap": STAIRCASE_QUOTA_PER_CAP,
    }
    warm_contract = contract.get("warm_start") or {}
    if (
        contract.get("schema_version") != SCHEMA
        or warm_contract.get("filename") != warm_start.name
        or warm_contract.get("sha256") != sealed._sha_file(warm_start)
        or warm_contract.get("shape") != list(WARM_SHAPE)
        or warm_contract.get("dtype") != "float64"
        or contract.get("island_quotas") != expected_quotas
        or contract.get("category_counts")
        != dict(sorted(expected_categories.items()))
        or contract.get("staircase_caps_Hz") != list(STAIRCASE_CAPS_HZ)
        or contract.get("resonance_anchor_epsilon_Hz")
        != RESONANCE_EPSILON_HZ
        or contract.get("Llt_anchor_epsilon_uH")
        != LLT_ANCHOR_EPSILON_UH
        or contract.get("decoded_params_sha256_duplicate_cap")
        != DECODED_SHA_DUPLICATE_CAP
        or contract.get("selected_decoded_params_sha256_count")
        != WARM_SHAPE[0]
        or contract.get("source_roles_authenticated") != list(SOURCE_ROLES)
        or contract.get("source_role_count") != len(SOURCE_ROLES)
        or contract.get("hard_spec") != HARD_SPEC
        or contract.get("hard_spec_sha256") != _json_sha(HARD_SPEC)
        or contract.get("constraint_version") != CONSTRAINT_VERSION
        or contract.get("constraint_names")
        != list(sealed.EXPECTED_CONSTRAINT_NAMES)
        or contract.get("constraint_names_sha256")
        != _json_sha(list(sealed.EXPECTED_CONSTRAINT_NAMES))
        or contract.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or contract.get("fixed_primary_turns") != 6
        or contract.get("authoritative_terminal_G")
        != "physical_unscaled_unchanged"
        or contract.get("soft_axis_pressure_enabled") is not False
        or contract.get("soft_axis_pressure_contract") is not None
        or any(
            contract.get(field) is not False
            for field in (
                "launch_performed", "scheduler_write_performed",
                "fea_submission_approved", "fea_submission_performed",
                "aedt_used", "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("anchor-island handoff top-level contract mismatch")

    try:
        with warm_start.open("rb") as stream:
            coordinates = np.asarray(np.load(stream, allow_pickle=False), dtype=float)
    except (OSError, ValueError) as exc:
        raise RuntimeError("anchor-island warm coordinates are invalid") from exc
    if (
        coordinates.shape != WARM_SHAPE
        or not np.isfinite(coordinates).all()
        or (coordinates < 0.0).any()
        or (coordinates > 1.0).any()
        or len({tuple(np.round(row, 12)) for row in coordinates})
        != WARM_SHAPE[0]
    ):
        raise RuntimeError("anchor-island warm coordinate contract mismatch")

    source_evidence = contract.get("source_evidence")
    source_snapshots = contract.get("source_snapshots")
    if (
        not isinstance(source_evidence, dict)
        or set(source_evidence) != set(SOURCE_ROLES)
        or not isinstance(source_snapshots, dict)
        or set(source_snapshots) != set(SOURCE_ROLES)
    ):
        raise RuntimeError("anchor-island nine-source evidence is incomplete")
    identities = []
    for role in SOURCE_ROLES:
        evidence = source_evidence[role]
        snapshots = source_snapshots[role]
        if (
            evidence.get("role") != role
            or evidence.get("controller_source_sha256")
            != ROLE_CONTROLLER_SHA256[role]
            or evidence.get("status_search_profile_sha256")
            != ROLE_PROFILE_SHA256[role]
            or _json_sha(evidence.get("status_search_profile"))
            != ROLE_PROFILE_SHA256[role]
            or not isinstance(evidence.get("terminal_result_count"), int)
            or evidence["terminal_result_count"] < 1
            or not isinstance(snapshots, dict)
            or set(snapshots) != set(SOURCE_SNAPSHOT_KINDS)
        ):
            raise RuntimeError(f"{role} source identity contract mismatch")
        identities.append(evidence.get("generation_identity"))
        for kind in SOURCE_SNAPSHOT_KINDS:
            item = snapshots[kind]
            path = _contained_evidence(root, item.get("path"), f"{role} {kind}")
            payload = path.read_bytes()
            digest = sealed._sha_bytes(payload)
            if (
                digest != item.get("sha256")
                or digest != item.get("source_sha256")
                or len(payload) != int(item.get("size_bytes", -1))
                or digest != (evidence.get(kind) or {}).get("sha256")
            ):
                raise RuntimeError(f"{role} {kind} snapshot SHA mismatch")
    if not all(identity == identities[0] for identity in identities[1:]):
        raise RuntimeError("anchor-island source generation identities differ")

    provenance = contract.get("selected_provenance")
    artifacts = contract.get("selected_artifact_snapshots")
    if (
        not isinstance(provenance, list)
        or len(provenance) != WARM_SHAPE[0]
        or _json_sha(provenance) != contract.get("selected_provenance_sha256")
        or not isinstance(artifacts, list)
        or len(artifacts) != WARM_SHAPE[0]
        or _json_sha(artifacts)
        != contract.get("selected_artifact_snapshots_sha256")
    ):
        raise RuntimeError("anchor-island selected evidence seal mismatch")
    decoded: list[dict] = []
    decoded_shas: list[str] = []
    actual_categories = Counter()
    actual_roles = Counter()
    for index, (item, artifact) in enumerate(zip(provenance, artifacts)):
        category = item.get("category")
        role = item.get("source_role")
        if (
            item.get("warm_index") != index
            or artifact.get("warm_index") != index
            or category not in expected_categories
            or role not in SOURCE_ROLES
            or artifact.get("source_role") != role
            or artifact.get("cohort_id") != item.get("cohort_id")
            or int(artifact.get("task_id", -1))
            != int(item.get("task_id", -2))
            or int(artifact.get("seed", -1)) != int(item.get("seed", -2))
            or item.get("original_N1") != 6
            or item.get("support_pass") is not True
            or not isinstance(item.get("decoded_params_sha256"), str)
            or not isinstance(item.get("unit_coordinate_sha256"), str)
        ):
            raise RuntimeError("anchor-island selected provenance mismatch")
        resonance = _finite(item.get("resonance_G_Hz"), "resonance G")
        llt = _finite(item.get("Llt_robust_G_uH"), "Llt G")
        _finite(item.get("thermal_max_G_C"), "thermal max G")
        _finite(item.get("thermal_positive_sum_C"), "thermal sum G")
        if category == "resonance_llt_anchor":
            semantic_ok = bool(
                item.get("resonance_llt_anchor_eligible") is True
                and resonance <= RESONANCE_EPSILON_HZ + TOLERANCE
                and llt <= LLT_ANCHOR_EPSILON_UH + TOLERANCE
                and item.get("optimizer_resonance_cap_Hz") is None
                and item.get("optimizer_resonance_cap_excess_Hz") is None
            )
        else:
            cap = next(
                value for value in STAIRCASE_CAPS_HZ
                if category == f"thermal_llt_staircase_{int(value)}hz"
            )
            semantic_ok = bool(
                item.get("thermal_llt_anchor_eligible") is True
                and item.get("all_thermal_pass") is True
                and llt <= TOLERANCE
                and item.get("optimizer_resonance_cap_Hz") == cap
                and math.isclose(
                    _finite(
                        item.get("optimizer_resonance_cap_excess_Hz"),
                        "resonance cap excess",
                    ),
                    max(resonance - cap, 0.0),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        if not semantic_ok:
            raise RuntimeError("anchor-island physical anchor semantics mismatch")
        parsed = {}
        for kind in ("result", "seed_status"):
            reference = artifact.get(kind)
            if not isinstance(reference, dict):
                raise RuntimeError("anchor-island selected artifact is missing")
            path = _contained_evidence(
                root, reference.get("path"), f"selected {kind}"
            )
            payload = path.read_bytes()
            if (
                sealed._sha_bytes(payload) != reference.get("sha256")
                or len(payload) != int(reference.get("size_bytes", -1))
            ):
                raise RuntimeError("anchor-island selected artifact SHA mismatch")
            parsed[kind] = json.loads(payload.decode("utf-8"))
        result = parsed["result"]
        seed_status = parsed["seed_status"]
        candidates = (result.get("next_target_fea_batch_plan") or {}).get(
            "candidates"
        )
        matching = [
            candidate for candidate in candidates or []
            if isinstance(candidate, dict)
            and candidate.get("decoded_params_sha256")
            == item["decoded_params_sha256"]
        ]
        if (
            len(matching) != 1
            or _json_sha(matching[0].get("decoded_params"))
            != item["decoded_params_sha256"]
            or seed_status.get("state") != "completed"
            or int(seed_status.get("exit_code", -1)) != 0
            or int(seed_status.get("seed", -1)) != int(item["seed"])
            or str(seed_status.get("task_id")) != str(item["task_id"])
            or seed_status.get("result_sha256") != item["result_sha256"]
        ):
            raise RuntimeError("anchor-island selected result semantics mismatch")
        decoded.append(matching[0]["decoded_params"])
        decoded_shas.append(item["decoded_params_sha256"])
        actual_categories[category] += 1
        actual_roles[role] += 1
    if (
        dict(sorted(actual_categories.items()))
        != dict(sorted(expected_categories.items()))
        or dict(sorted(actual_roles.items()))
        != contract.get("selected_source_role_counts")
        or max(Counter(decoded_shas).values(), default=0)
        > DECODED_SHA_DUPLICATE_CAP
    ):
        raise RuntimeError("anchor-island quota or duplicate contract mismatch")

    if nsga_code_root is not None:
        code_identity = bridge._verify_nsga_code_root(
            nsga_code_root.resolve(strict=True)
        )
        if (
            code_identity["revision"] != contract.get("nsga_code_root_revision")
            or code_identity["clean"] is not True
            or code_identity["decoder_relative_path"]
            != contract.get("nsga_decoder_relative_path")
            or code_identity["decoder_sha256"]
            != contract.get("nsga_decoder_source_sha256")
        ):
            raise RuntimeError("anchor-island NSGA code identity mismatch")
        sobol_dims, n1_min, n1_max, defaults = _load_sobol_schema(
            nsga_code_root.resolve(strict=True)
        )
        decoded_coordinates = np.vstack([
            decoded_to_unit(params, sobol_dims, n1_min, n1_max, defaults)
            for params in decoded
        ])
        if not np.allclose(
            decoded_coordinates, coordinates, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("anchor-island decoded coordinate replay mismatch")
    return contract_path, sealed._sha_file(contract_path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for role in SOURCE_ROLES:
        result.add_argument(
            f"--{role.replace('_', '-')}-root",
            dest=f"{role}_root",
            type=Path,
            default=DEFAULT_ROOTS[role],
        )
    result.add_argument("--nsga-code-root", type=Path, default=DEFAULT_CODE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> int:
    args = parser().parse_args()
    roots = {
        role: getattr(args, f"{role}_root") for role in SOURCE_ROLES
    }
    result = build_handoff(
        roots=roots,
        nsga_code_root=args.nsga_code_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
