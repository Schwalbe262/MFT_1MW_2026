"""Build an authenticated 32+32 N1=5 deep-crossover warm pool.

The pool interleaves two already sealed sources: the fixed-N1=5 size/Rx/core
island and the balanced thermal crossover.  Source trees are captured below
the output root, and the resulting bytes can be validated without consulting
their mutable original locations.  This command never contacts Slurm or AEDT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tier1_fixed_n1_5_size_rx_core_warm_handoff import (  # noqa: E402
    validate_contract as validate_size_contract,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    _load_sobol_schema,
    decoded_to_unit,
)


SCHEMA = "mft-tier1-fixed-n1-5-deep-crossover-warm-handoff-v1"
CONTRACT_FILENAME = "fixed_n1_5_deep_crossover_warm_contract.json"
WARM_FILENAME = "next_warm_start.npy"
WARM_SHAPE = (64, 25)
SOURCE_COUNT = 32
FIXED_PRIMARY_TURNS = 5
SOURCE_NAMES = ("size_rx_core", "thermal_crossover")
DEFAULT_SIZE_SOURCE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-5-size-rx-core-island-v1\next_warm_start.npy"
)
DEFAULT_THERMAL_SOURCE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-thermal-crossover-balanced-v1\next_warm_start.npy"
)
DEFAULT_CODE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_code"
    r"\7c832f7f78f9"
)
DEFAULT_ANCHOR_RECORD = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\t1r250_260719\cohorts"
    r"\res15k-41497047d1-1a53578a25-431c948c36\results"
    r"\task-59961\record.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
    r"\t110-res15k-n1-5-deep-crossover-32x32-v3"
)
ANCHOR_TASK_ID = 59961
ANCHOR_SEED = 1_907_192_471
ANCHOR_COHORT_ID = "res15k-41497047d1-1a53578a25-431c948c36"
ANCHOR_RECORD_SHA256 = (
    "6ea8851113661179bb0debf012b738a8e24dfa17c5928ccf207bcbeeb7d2086a"
)
ANCHOR_RESULT_SHA256 = (
    "320204056c17e04ed6b233fd884417e00f271b3fb010f7d72ce48b7d8f9104b2"
)
ANCHOR_STATUS_SHA256 = (
    "e57a7a567b8b76ba8b548817d82219d4638ae35fff81f58f621b4236b9b31ce8"
)
ANCHOR_DECODED_PARAMS_SHA256 = (
    "3607a3e75d079aa201f61bcb54b3b615e819326d1de1d7cdccdbe309d1f19da1"
)
ANCHOR_CONSTRAINT_G_SHA256 = (
    "40c1790057cab91a90bba0d44c6f2eada9f3366696dcd62acc95d407b9684e68"
)
ANCHOR_INVERSE_COORDINATE_SHA256 = (
    "e36e8f36947600bf80165f60a1a27706d870983774c93244cf7fb9154f58f8e4"
)
ANCHOR_DECODE_IDENTITY_FIELDS = (
    "N1_main", "N1_side", "N2_main", "N2_side", "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t", "wcp_t",
    "wcp_pad_t", "wcp_len_x", "cw1", "gap1", "cw2", "gap2", "nwh1",
    "nwh2", "cc_w2c_space_x", "cc_w2c_space_y", "w2c_w1c_space_x",
    "w2c_w1c_space_y", "w1c_w2s_space_x", "w2s_w1s_space_x",
    "w1s_w2s_space_y", "w1s_cs_space_x", "cs_w1s_space_y",
)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _load_warm(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.shape != WARM_SHAPE or not np.isfinite(values).all():
        raise RuntimeError(f"warm coordinate shape/finite mismatch: {path}")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError(f"warm coordinate escaped unit interval: {path}")
    return values


def _diverse_indices(values: np.ndarray, count: int = SOURCE_COUNT) -> list[int]:
    """Deterministic max-min subset with a centroid-extreme first point."""

    if values.ndim != 2 or not 0 < count <= len(values):
        raise RuntimeError("invalid diversity input")
    centroid = values.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(values - centroid, axis=1)))
    selected = [first]
    remaining = set(range(len(values))) - {first}
    while len(selected) < count:
        ranked = []
        for index in remaining:
            distance = min(
                float(np.linalg.norm(values[index] - values[chosen]))
                for chosen in selected
            )
            ranked.append((-distance, index))
        _negative_distance, chosen = min(ranked)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _validate_source_contracts(
    size_warm: Path, thermal_warm: Path, code_root: Path,
) -> dict[str, dict[str, str]]:
    size_contract, size_contract_sha = validate_size_contract(
        size_warm, code_root,
    )
    # Imported lazily to avoid making the handoff builder part of the rolling
    # controller's module initialization graph.
    from tools.tier1_slurm_rolling import (  # noqa: PLC0415
        validate_thermal_crossover_warm_contract,
    )

    thermal_reference = validate_thermal_crossover_warm_contract(
        thermal_warm, allow_external=True,
    )
    if thermal_reference is None:
        raise RuntimeError("thermal crossover source validation was skipped")
    thermal_contract, thermal_contract_sha = thermal_reference
    return {
        "size_rx_core": {
            "warm_sha256": _sha(size_warm),
            "contract_filename": size_contract.name,
            "contract_sha256": size_contract_sha,
        },
        "thermal_crossover": {
            "warm_sha256": _sha(thermal_warm),
            "contract_filename": thermal_contract.name,
            "contract_sha256": thermal_contract_sha,
        },
    }


def _authenticated_anchor(
    record_path: Path, code_root: Path,
) -> tuple[list[float], dict, dict[str, bytes]]:
    record_path = record_path.resolve(strict=True)
    root = record_path.parent.resolve(strict=True)
    result_path = (root / "result.json").resolve(strict=True)
    status_path = (root / "seed_status.json").resolve(strict=True)
    payloads = {
        "record.json": record_path.read_bytes(),
        "result.json": result_path.read_bytes(),
        "seed_status.json": status_path.read_bytes(),
    }
    shas = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }
    if shas != {
        "record.json": ANCHOR_RECORD_SHA256,
        "result.json": ANCHOR_RESULT_SHA256,
        "seed_status.json": ANCHOR_STATUS_SHA256,
    }:
        raise RuntimeError("authenticated N1=5 anchor bytes drifted")
    record = json.loads(payloads["record.json"].decode("utf-8"))
    result = json.loads(payloads["result.json"].decode("utf-8"))
    status = json.loads(payloads["seed_status.json"].decode("utf-8"))
    candidates = (result.get("next_target_fea_batch_plan") or {}).get(
        "candidates"
    )
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("authenticated N1=5 anchor candidate is missing")
    candidate = candidates[0]
    params = candidate.get("decoded_params")
    constraints = candidate.get("constraint_G")
    core_positive = {
        "temperature_robust_limit:T_max_core",
        "temperature_robust_limit:Tprobe_core_center_max",
        "temperature_robust_limit:Tprobe_core_top_yoke_max",
    }
    if (
        record.get("task_id") != ANCHOR_TASK_ID
        or record.get("seed") != ANCHOR_SEED
        or record.get("authenticated") is not True
        or record.get("terminal_state") != "completed"
        or (record.get("result") or {}).get("sha256")
        != ANCHOR_RESULT_SHA256
        or (record.get("remote_status") or {}).get("sha256")
        != ANCHOR_STATUS_SHA256
        or str(status.get("task_id")) != str(ANCHOR_TASK_ID)
        or status.get("seed") != ANCHOR_SEED
        or status.get("cohort_id") != ANCHOR_COHORT_ID
        or status.get("state") != "completed"
        or status.get("result_sha256") != ANCHOR_RESULT_SHA256
        or result.get("seed") != ANCHOR_SEED
        or result.get("schema_version")
        != "mft-tier1-corrected-search-seed-v1"
        or result.get("nsga_code_revision")
        != "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        or candidate.get("terminal_population_index") != 0
        or candidate.get("decoded_params_sha256")
        != ANCHOR_DECODED_PARAMS_SHA256
        or not isinstance(params, dict)
        or _canonical_sha(params) != ANCHOR_DECODED_PARAMS_SHA256
        or not isinstance(constraints, dict)
        or _canonical_sha(constraints) != ANCHOR_CONSTRAINT_G_SHA256
        or params.get("N1") != FIXED_PRIMARY_TURNS
        or params.get("N1_main") + params.get("N1_side")
        != FIXED_PRIMARY_TURNS
        or any(
            float(value) > 1e-9
            for name, value in constraints.items() if name not in core_positive
        )
        or any(float(constraints[name]) <= 0.0 for name in core_positive)
    ):
        raise RuntimeError("authenticated N1=5 anchor semantics drifted")
    dimensions, n1_min, n1_max, defaults = _load_sobol_schema(code_root)
    coordinate = [
        float(value) for value in decoded_to_unit(
            params, dimensions, n1_min, n1_max, defaults,
        )
    ]
    if (
        len(coordinate) != WARM_SHAPE[1]
        or not np.isfinite(coordinate).all()
        or _canonical_sha(coordinate) != ANCHOR_INVERSE_COORDINATE_SHA256
    ):
        raise RuntimeError("authenticated N1=5 anchor coordinate is invalid")
    from module.input_parameter_260706 import (  # noqa: PLC0415
        decode_unit_sample,
        unit_to_dims,
    )

    decoded_replay = decode_unit_sample(
        unit_to_dims(coordinate), allow_space_shrink=False, space_min=40.0,
        fixed_cw1_mm=5.0,
    )
    decoded_replay.update({"core_plate_pad_t": 2.0, "wcp_pad_t": 2.0})
    forward_differences = {
        name: [float(params[name]), float(decoded_replay[name])]
        for name in ANCHOR_DECODE_IDENTITY_FIELDS
        if float(decoded_replay[name]) != float(params[name])
    }
    if forward_differences != {
        "cw2": [1.99, 2.0],
        "w1c_w2s_space_x": [32.4, 32.0],
        "wcp_len_x": [375.9, 376.3],
    }:
        raise RuntimeError("authenticated N1=5 inverse forward decode drifted")
    evidence = {
        "task_id": ANCHOR_TASK_ID,
        "seed": ANCHOR_SEED,
        "cohort_id": ANCHOR_COHORT_ID,
        "captured_root": "authenticated_anchor",
        "files": shas,
        "decoded_params_sha256": ANCHOR_DECODED_PARAMS_SHA256,
        "current_forward_decoded_geometry_identity_sha256": _canonical_sha({
            name: decoded_replay[name] for name in ANCHOR_DECODE_IDENTITY_FIELDS
        }),
        "current_forward_decode_differences_from_persisted": (
            forward_differences
        ),
        "constraint_G_sha256": ANCHOR_CONSTRAINT_G_SHA256,
        "coordinate_sha256": _canonical_sha(coordinate),
        "coordinate_authority": (
            "persisted_decoded_params_pure_inverse_reconstruction_"
            "then_current_repair_and_forward_decode"
        ),
        "original_terminal_chromosome_authority": False,
        "cross_platform_optimizer_replay_authority": False,
        "optimizer_deterministic_replay_claimed": False,
        "selection_role": (
            "resonance_Llt_density_size_pass_core3_thermal_near_anchor"
        ),
    }
    return coordinate, evidence, payloads


def _decoded_topology(coordinate: list[float], code_root: Path) -> tuple[int, int]:
    _load_sobol_schema(code_root)
    from module.input_parameter_260706 import (  # noqa: PLC0415
        decode_unit_sample,
        unit_to_dims,
    )

    decoded = decode_unit_sample(
        unit_to_dims(coordinate), allow_space_shrink=False, space_min=40.0,
        fixed_cw1_mm=5.0,
    )
    return int(decoded["N2_main"]), int(decoded["N2_side"])


def build(
    *, size_warm: Path, thermal_warm: Path, anchor_record: Path,
    code_root: Path, output: Path,
) -> dict:
    size_warm = size_warm.resolve(strict=True)
    thermal_warm = thermal_warm.resolve(strict=True)
    anchor_record = anchor_record.resolve(strict=True)
    code_root = code_root.resolve(strict=True)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output must be absent or empty")
    source_contracts = _validate_source_contracts(
        size_warm, thermal_warm, code_root,
    )
    size_values = _load_warm(size_warm)
    thermal_values = _load_warm(thermal_warm)
    anchor_coordinate, anchor_evidence, anchor_payloads = (
        _authenticated_anchor(anchor_record, code_root)
    )
    size_indices = _diverse_indices(size_values, SOURCE_COUNT - 2)
    thermal_indices = _diverse_indices(thermal_values)
    topology_29_coordinate = list(anchor_coordinate)
    n2_side_index = next(
        index for index, (name, _lower, _upper) in enumerate(
            _load_sobol_schema(code_root)[0]
        ) if name == "u_N2_side"
    )
    topology_29_coordinate[n2_side_index] = 21.0 / 40.0
    if _decoded_topology(topology_29_coordinate, code_root) != (29, 21):
        raise RuntimeError("derived N2 topology did not decode to 29+21")
    anchor_evidence["derived_topology_29"] = {
        "derivation": "task59961_coordinate_with_only_u_N2_side_for_21_of_50",
        "expected_N2_main": 29,
        "expected_N2_side": 21,
        "coordinate_sha256": _canonical_sha(topology_29_coordinate),
        "physical_replay_required": True,
    }

    output.mkdir(parents=True, exist_ok=True)
    captured = output / "captured_sources"
    for name, warm in (
        ("size_rx_core", size_warm),
        ("thermal_crossover", thermal_warm),
    ):
        shutil.copytree(warm.parent, captured / name)
    anchor_capture = output / anchor_evidence["captured_root"]
    anchor_capture.mkdir(parents=True, exist_ok=False)
    for filename, payload in anchor_payloads.items():
        (anchor_capture / filename).write_bytes(payload)

    rows = []
    provenance = []
    for rank, thermal_index in enumerate(thermal_indices):
        even_source = (
            "authenticated_anchor" if rank == 0
            else (
                "authenticated_anchor_topology_29_derived"
                if rank == 1 else "size_rx_core"
            )
        )
        even_index = 0 if rank < 2 else size_indices[rank - 2]
        for source_name, source_index in (
            (even_source, even_index),
            ("thermal_crossover", thermal_index),
        ):
            row = (
                list(anchor_coordinate)
                if source_name == "authenticated_anchor"
                else (
                    list(topology_29_coordinate)
                    if source_name
                    == "authenticated_anchor_topology_29_derived"
                    else [
                        float(value)
                        for value in (
                            size_values if source_name == "size_rx_core"
                            else thermal_values
                        )[source_index]
                    ]
                )
            )
            rows.append(row)
            provenance.append({
                "warm_index": len(rows) - 1,
                "pair_rank": rank,
                "source": source_name,
                "source_index": int(source_index),
                "coordinate_sha256": _canonical_sha(row),
            })
    merged = np.asarray(rows, dtype=float)
    if merged.shape != WARM_SHAPE or len({_canonical_sha(row) for row in rows}) < 32:
        raise RuntimeError("merged warm pool is malformed or degenerate")
    warm_output = output / WARM_FILENAME
    temporary_warm = output / f"{WARM_FILENAME}.tmp.{uuid.uuid4().hex}"
    with temporary_warm.open("wb") as stream:
        np.save(stream, merged, allow_pickle=False)
    os.replace(temporary_warm, warm_output)

    source_records = {}
    for name in SOURCE_NAMES:
        root = captured / name
        source_records[name] = {
            **source_contracts[name],
            "captured_root": root.relative_to(output).as_posix(),
            "captured_inventory": _tree_inventory(root),
            "captured_inventory_sha256": _canonical_sha(_tree_inventory(root)),
        }
    contract = {
        "schema_version": SCHEMA,
        "selection_contract": "deterministic_maxmin32_each_interleaved_v1",
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "source_count_each": SOURCE_COUNT,
        "source_records": source_records,
        "authenticated_anchor": anchor_evidence,
        "selected_provenance": provenance,
        "selected_provenance_sha256": _canonical_sha(provenance),
        "warm_start": {
            "filename": WARM_FILENAME,
            "shape": list(WARM_SHAPE),
            "sha256": _sha(warm_output),
        },
        "post_fixed_N1_repair_preflight_required": True,
        "warm_rows_are_coordinate_donors_only": True,
        "source_surrogate_or_thermal_pass_inheritance_allowed": False,
        "paired_rows_do_not_claim_parent_performance_transfer": True,
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    contract["sha256"] = _canonical_sha(contract)
    _atomic_json(output / CONTRACT_FILENAME, contract)
    validate_contract(warm_output, code_root)
    return contract


def _contained(root: Path, relative: object) -> Path:
    value = Path(str(relative or ""))
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise RuntimeError("captured source path is not contained")
    path = (root / value).resolve(strict=True)
    if path != root and root not in path.parents:
        raise RuntimeError("captured source escaped handoff root")
    return path


def validate_contract(
    warm_start: Path, code_root: Path | None,
) -> tuple[Path, str]:
    if code_root is None:
        raise RuntimeError("deep crossover warm handoff requires NSGA code")
    code_root = code_root.resolve(strict=True)
    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(CONTRACT_FILENAME).resolve(strict=True)
    root = contract_path.parent.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    unsigned = dict(contract)
    recorded_sha = unsigned.pop("sha256", None)
    warm = contract.get("warm_start") or {}
    provenance = contract.get("selected_provenance")
    source_records = contract.get("source_records")
    anchor_evidence = contract.get("authenticated_anchor")
    if (
        contract.get("schema_version") != SCHEMA
        or recorded_sha != _canonical_sha(unsigned)
        or contract.get("selection_contract")
        != "deterministic_maxmin32_each_interleaved_v1"
        or contract.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or contract.get("source_count_each") != SOURCE_COUNT
        or set(source_records or {}) != set(SOURCE_NAMES)
        or not isinstance(anchor_evidence, dict)
        or not isinstance(provenance, list)
        or len(provenance) != WARM_SHAPE[0]
        or _canonical_sha(provenance)
        != contract.get("selected_provenance_sha256")
        or warm.get("filename") != warm_start.name
        or warm.get("shape") != list(WARM_SHAPE)
        or warm.get("sha256") != _sha(warm_start)
        or contract.get("post_fixed_N1_repair_preflight_required") is not True
        or contract.get("warm_rows_are_coordinate_donors_only") is not True
        or contract.get(
            "source_surrogate_or_thermal_pass_inheritance_allowed"
        ) is not False
        or contract.get(
            "paired_rows_do_not_claim_parent_performance_transfer"
        ) is not True
        or contract.get("physical_hard_spec_mutation") is not False
        or contract.get("objective_mutation") is not False
        or contract.get("scheduler_write_performed") is not False
        or contract.get("fea_submission_performed") is not False
        or contract.get("aedt_used") is not False
        or contract.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("deep crossover warm handoff contract mismatch")

    captured_warms = {}
    for name in SOURCE_NAMES:
        record = source_records[name]
        captured_root = _contained(root, record.get("captured_root"))
        inventory = _tree_inventory(captured_root)
        if (
            inventory != record.get("captured_inventory")
            or _canonical_sha(inventory)
            != record.get("captured_inventory_sha256")
        ):
            raise RuntimeError("captured warm source inventory mismatch")
        captured_warm = (captured_root / WARM_FILENAME).resolve(strict=True)
        if _sha(captured_warm) != record.get("warm_sha256"):
            raise RuntimeError("captured source warm SHA mismatch")
        captured_warms[name] = captured_warm
    verified = _validate_source_contracts(
        captured_warms["size_rx_core"],
        captured_warms["thermal_crossover"],
        code_root,
    )
    if any(
        verified[name]["contract_filename"]
        != source_records[name].get("contract_filename")
        or verified[name]["contract_sha256"]
        != source_records[name].get("contract_sha256")
        for name in SOURCE_NAMES
    ):
        raise RuntimeError("captured source contract identity mismatch")

    anchor_root = _contained(root, anchor_evidence.get("captured_root"))
    expected_anchor_files = anchor_evidence.get("files")
    if set(expected_anchor_files or {}) != {
        "record.json", "result.json", "seed_status.json",
    }:
        raise RuntimeError("authenticated anchor file set mismatch")
    for filename, expected_sha in expected_anchor_files.items():
        if _sha(anchor_root / filename) != expected_sha:
            raise RuntimeError("authenticated anchor captured bytes drifted")
    anchor_coordinate, verified_anchor, _payloads = _authenticated_anchor(
        anchor_root / "record.json", code_root,
    )
    recorded_derived = anchor_evidence.get("derived_topology_29")
    recorded_base_anchor = dict(anchor_evidence)
    recorded_base_anchor.pop("derived_topology_29", None)
    if verified_anchor != recorded_base_anchor:
        raise RuntimeError("authenticated anchor evidence drifted")
    if (
        recorded_base_anchor.get("coordinate_authority")
        != (
            "persisted_decoded_params_pure_inverse_reconstruction_"
            "then_current_repair_and_forward_decode"
        )
        or recorded_base_anchor.get("coordinate_sha256")
        != ANCHOR_INVERSE_COORDINATE_SHA256
        or recorded_base_anchor.get(
            "original_terminal_chromosome_authority"
        ) is not False
        or recorded_base_anchor.get(
            "cross_platform_optimizer_replay_authority"
        ) is not False
        or recorded_base_anchor.get(
            "optimizer_deterministic_replay_claimed"
        ) is not False
    ):
        raise RuntimeError("authenticated anchor authority wording drifted")
    topology_29_coordinate = list(anchor_coordinate)
    dimensions = _load_sobol_schema(code_root)[0]
    n2_side_index = next(
        index for index, (name, _lower, _upper) in enumerate(dimensions)
        if name == "u_N2_side"
    )
    topology_29_coordinate[n2_side_index] = 21.0 / 40.0
    if _decoded_topology(topology_29_coordinate, code_root) != (29, 21):
        raise RuntimeError("derived N2 topology replay drifted")
    if recorded_derived != {
        "derivation": "task59961_coordinate_with_only_u_N2_side_for_21_of_50",
        "expected_N2_main": 29,
        "expected_N2_side": 21,
        "coordinate_sha256": _canonical_sha(topology_29_coordinate),
        "physical_replay_required": True,
    }:
        raise RuntimeError("derived N2 topology anchor evidence drifted")

    source_values = {
        name: _load_warm(path) for name, path in captured_warms.items()
    }
    reconstructed = []
    source_counts = {name: 0 for name in SOURCE_NAMES}
    for expected_index, item in enumerate(provenance):
        source = item.get("source")
        index = item.get("source_index")
        expected_source = (
            "authenticated_anchor" if expected_index == 0
            else (
                "authenticated_anchor_topology_29_derived"
                if expected_index == 2 else SOURCE_NAMES[expected_index % 2]
            )
        )
        if (
            item.get("warm_index") != expected_index
            or item.get("pair_rank") != expected_index // 2
            or source != expected_source
            or isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < WARM_SHAPE[0]
        ):
            raise RuntimeError("deep crossover selection provenance mismatch")
        row = (
            list(anchor_coordinate)
            if source == "authenticated_anchor"
            else (
                list(topology_29_coordinate)
                if source == "authenticated_anchor_topology_29_derived"
                else [float(value) for value in source_values[source][index]]
            )
        )
        if _canonical_sha(row) != item.get("coordinate_sha256"):
            raise RuntimeError("deep crossover coordinate provenance mismatch")
        if source in source_counts:
            source_counts[source] += 1
        reconstructed.append(row)
    if source_counts != {
        "size_rx_core": SOURCE_COUNT - 2,
        "thermal_crossover": SOURCE_COUNT,
    }:
        raise RuntimeError("deep crossover source quota mismatch")
    values = _load_warm(warm_start)
    if not np.allclose(values, np.asarray(reconstructed), rtol=0.0, atol=0.0):
        raise RuntimeError("deep crossover warm bytes disagree with provenance")
    return contract_path, _sha(contract_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-warm", type=Path, default=DEFAULT_SIZE_SOURCE)
    parser.add_argument(
        "--thermal-warm", type=Path, default=DEFAULT_THERMAL_SOURCE,
    )
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE)
    parser.add_argument(
        "--anchor-record", type=Path, default=DEFAULT_ANCHOR_RECORD,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build(
        size_warm=args.size_warm, thermal_warm=args.thermal_warm,
        anchor_record=args.anchor_record, code_root=args.code_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
