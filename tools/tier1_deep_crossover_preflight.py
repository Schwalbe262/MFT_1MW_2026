"""Run and seal four parallel local, no-Slurm deep-crossover preflights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

try:
    from tier1_deep_crossover_contract import (
        DEEP_CROSSOVER_ISLANDS,
        RESONANCE_SCALE_HZ,
        TERMINATION_STRATEGY,
        canonical_sha,
        island_profile,
        topology_evolution_contract,
        validate_anchor_current_replay,
    )
    from tier1_fixed_n1_5_deep_crossover_warm_handoff import (
        ANCHOR_CONSTRAINT_G_SHA256,
        ANCHOR_DECODED_PARAMS_SHA256,
        ANCHOR_INVERSE_COORDINATE_SHA256,
        _authenticated_anchor,
        validate_contract as validate_n1_5_warm,
    )
    from tier1_fixed_n1_6_anchor_islands_warm_handoff import (
        validate_handoff_contract as validate_n1_6_warm,
    )
    from tier1_n1_6_anchor_island_contract import optimizer_allowance_contract
    from tier1_resonance_feedback import (
        CONSTRAINT_VERSION,
        EXPECTED_NSGA_REVISION,
        HARD_SPEC,
        optimizer_termination_contract,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_deep_crossover_contract import (
        DEEP_CROSSOVER_ISLANDS,
        RESONANCE_SCALE_HZ,
        TERMINATION_STRATEGY,
        canonical_sha,
        island_profile,
        topology_evolution_contract,
        validate_anchor_current_replay,
    )
    from tools.tier1_fixed_n1_5_deep_crossover_warm_handoff import (
        ANCHOR_CONSTRAINT_G_SHA256,
        ANCHOR_DECODED_PARAMS_SHA256,
        ANCHOR_INVERSE_COORDINATE_SHA256,
        _authenticated_anchor,
        validate_contract as validate_n1_5_warm,
    )
    from tools.tier1_fixed_n1_6_anchor_islands_warm_handoff import (
        validate_handoff_contract as validate_n1_6_warm,
    )
    from tools.tier1_n1_6_anchor_island_contract import (
        optimizer_allowance_contract,
    )
    from tools.tier1_resonance_feedback import (
        CONSTRAINT_VERSION,
        EXPECTED_NSGA_REVISION,
        HARD_SPEC,
        optimizer_termination_contract,
    )


SCHEMA = "mft-tier1-deep-crossover-preflight-v1"
RESULT_SCHEMA = "mft-tier1-corrected-search-seed-v1"
ANCHOR_REPLAY_SCHEMA = "mft-tier1-n1-5-anchor-current-physical-replay-v2"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _snapshot_valid(snapshot: object, physical_G: object) -> bool:
    return bool(
        isinstance(snapshot, dict)
        and canonical_sha(snapshot.get("coordinate_unit"))
        == snapshot.get("coordinate_unit_sha256")
        and canonical_sha(snapshot.get("decoded_params"))
        == snapshot.get("decoded_params_sha256")
        and canonical_sha(snapshot.get("physical_constraint_G"))
        == snapshot.get("physical_constraint_G_sha256")
        and snapshot.get("physical_constraint_G") == physical_G
    )


def _anchor_physical_replay(
    *, model_dir: Path, code_root: Path, anchor_record: Path,
    inference_threads: int,
) -> dict:
    import pandas as pd  # noqa: PLC0415

    from tools import tier1_resonance_feedback as sealed  # noqa: PLC0415

    models, report, _manifest = sealed._load_search_models(
        model_dir, code_root,
    )
    for model in models.values():
        configure = getattr(model, "configure_inference_threads", None)
        if callable(configure):
            try:
                configure(inference_threads)
            except ValueError:
                for family, fitted in model.bundle["models"]:
                    if str(family).lower() != "catboost":
                        fitted.n_jobs = inference_threads
    from optimization.nsga2_problem import MFTProblem  # noqa: PLC0415
    from predictor import DensityGate  # noqa: PLC0415

    strict = pd.read_parquet(model_dir / "strict_target_snapshot.parquet")
    problem = MFTProblem(
        models, spec=HARD_SPEC,
        density_gate=DensityGate(strict, report["features"]),
    )
    physical_evaluate, _normalization = (
        sealed.install_optimizer_constraint_normalization(
            problem, HARD_SPEC,
            resonance_scale_hz=RESONANCE_SCALE_HZ,
            llt_scale_uh=0.30,
            all_thermal_scale_c=2.0,
        )
    )
    coordinate, anchor_evidence, payloads = _authenticated_anchor(
        anchor_record, code_root,
    )
    evaluated = {}
    physical_evaluate(np.asarray([coordinate], dtype=float), evaluated)
    matrix = np.asarray(evaluated["G"], dtype=float)
    if matrix.shape != (1, len(problem.constraint_names)):
        raise RuntimeError("anchor physical replay constraint shape mismatch")
    physical_g = {
        name: float(matrix[0, index])
        for index, name in enumerate(problem.constraint_names)
    }
    frame = evaluated["frame"].iloc[0]
    decoded = {}
    for key, value in frame.to_dict().items():
        if not isinstance(value, (str, int, float, np.integer, np.floating)):
            continue
        value = value.item() if hasattr(value, "item") else value
        if isinstance(value, float) and not math.isfinite(value):
            continue
        decoded[key] = value
    source_result = json.loads(payloads["result.json"].decode("utf-8"))
    source_candidate = source_result["next_target_fea_batch_plan"][
        "candidates"
    ][0]
    match = validate_anchor_current_replay(
        current_decoded=decoded, replayed_physical_g=physical_g,
        source_candidate=source_candidate,
        expected_decoded_sha256=ANCHOR_DECODED_PARAMS_SHA256,
        expected_source_physical_g_sha256=ANCHOR_CONSTRAINT_G_SHA256,
    )
    return {
        "schema_version": ANCHOR_REPLAY_SCHEMA,
        "anchor_evidence": anchor_evidence,
        **match,
        "model_manifest_sha256": _sha(model_dir / "manifest.json"),
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "physical_evaluator_mutation": False,
        "optimizer_normalization_not_used_for_replay": True,
        "inverse_coordinate_authority": (
            "persisted_source_decoded_params_not_terminal_X"
        ),
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "verified": True,
    }


def _n1_5_topology_evidence(warm_start: Path, code_root: Path) -> dict:
    from tools.tier1_terminal_followup import _load_sobol_schema  # noqa: PLC0415

    dimensions, n1_min, n1_max, _defaults = _load_sobol_schema(code_root)
    from module.input_parameter_260706 import (  # noqa: PLC0415
        decode_unit_sample,
        unit_to_dims,
    )

    values = np.asarray(np.load(warm_start, allow_pickle=False), dtype=float)
    names = [name for name, _lower, _upper in dimensions]
    n1_index = names.index("u_N1")
    values[:, n1_index] = (
        5 - n1_min + 0.5
    ) / (n1_max - n1_min + 0.9999)
    topologies = []
    for coordinate in values:
        decoded = decode_unit_sample(
            unit_to_dims(coordinate), allow_space_shrink=False,
            space_min=40.0, fixed_cw1_mm=5.0,
        )
        topologies.append((int(decoded["N2_main"]), int(decoded["N2_side"])))
    required = (28, 29, 30, 31, 32, 33)
    counts = {
        str(main): sum(item[0] == main for item in topologies)
        for main in sorted({item[0] for item in topologies})
    }
    indices = {
        str(main): [
            index for index, item in enumerate(topologies) if item[0] == main
        ]
        for main in required
    }
    if any(not indices[str(main)] for main in required):
        raise RuntimeError("N1=5 warm pool lacks required N2 topology stratum")
    evidence = {
        "schema_version": "mft-tier1-n1-5-warm-topology-strata-v1",
        "warm_start_sha256": _sha(warm_start),
        "fixed_primary_turns": 5,
        "coordinate_count": len(values),
        "required_N2_main_values": list(required),
        "N2_main_counts": counts,
        "required_N2_main_warm_indices": indices,
        "all_required_topologies_present": True,
        "pinned_decoder_replay": True,
    }
    evidence["sha256"] = canonical_sha(evidence)
    return evidence


def _summary(result: dict, island, max_generations: int) -> dict:
    names = result.get("constraint_names")
    normalization = result.get("optimizer_constraint_normalization") or {}
    scales = normalization.get("scales") or {}
    thermal_names = [
        name for name in names or []
        if isinstance(name, str) and name.startswith("temperature_robust_limit:")
    ]
    expected_allowance = None
    if island.optimizer_resonance_allowance_hz is not None:
        expected_allowance = optimizer_allowance_contract(
            names,
            resonance_allowance_hz=island.optimizer_resonance_allowance_hz,
            llt_allowance_uh=island.optimizer_llt_allowance_uh,
        )
    physical_fields = (
        "constraint_minimum_G",
        "terminal_population_best_constraint_G",
        "optimizer_terminal_best_physical_constraint_G",
    )
    expected_topology_contract = topology_evolution_contract(
        island.fixed_primary_turns
    )
    topology_contract = result.get("optimizer_topology_evolution_contract")
    topology_audit = result.get("optimizer_topology_evolution_audit")
    topology_counts = (
        (topology_audit or {}).get("terminal_topology_counts") or {}
    )
    minimum_topology_count = expected_topology_contract["survival"][
        "minimum_survivors_per_turn_split_sub_island"
    ]
    if (
        not isinstance(names, list) or len(names) != len(set(names))
        or result.get("schema_version") != RESULT_SCHEMA
        or result.get("constraint_version") != CONSTRAINT_VERSION
        or result.get("hard_spec") != HARD_SPEC
        or result.get("hard_spec_sha256") != canonical_sha(HARD_SPEC)
        or result.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or result.get("fixed_primary_turns") != island.fixed_primary_turns
        or result.get("terminal_population_fixed_primary_turns_verified")
        is not True
        or result.get("terminal_population_primary_turn_values")
        != [island.fixed_primary_turns]
        or result.get("max_generations") != max_generations
        or result.get("completed_generations") != max_generations + 1
        or result.get("evaluated_generations") != max_generations
        or result.get("optimizer_termination_strategy") != TERMINATION_STRATEGY
        or result.get("optimizer_termination_contract")
        != optimizer_termination_contract(TERMINATION_STRATEGY, max_generations)
        or topology_contract != expected_topology_contract
        or canonical_sha({
            key: value for key, value in (topology_contract or {}).items()
            if key != "sha256"
        }) != (topology_contract or {}).get("sha256")
        or not isinstance(topology_audit, dict)
        or canonical_sha({
            key: value for key, value in topology_audit.items()
            if key != "sha256"
        }) != topology_audit.get("sha256")
        or topology_audit.get("paired_parent_pairs_emitted", 0) < 1
        or topology_audit.get("migration_events", 0) < 1
        or topology_audit.get("migrants_created", 0) < 6
        or topology_audit.get("survival_calls", 0) < 2
        or topology_audit.get("all_required_topologies_preserved") is not True
        or topology_audit.get("warm_donor_prediction_inheritance") is not False
        or set(topology_counts) != {
            str(value) for value in expected_topology_contract[
                "turn_split_sub_islands_N2_main"
            ]
        }
        or any(
            int(value) < minimum_topology_count
            for value in topology_counts.values()
        )
        or result.get("fixed_generation_gate_satisfied") is not True
        or normalization.get("constraint_order") != names
        or normalization.get("authoritative_terminal_G") != "physical_unscaled"
        or scales.get("half_magnetizing_resonance_minimum")
        != RESONANCE_SCALE_HZ
        or scales.get("Llt_robust_band") != island.optimizer_llt_scale_uh
        or scales.get("Llt_ensemble_disagreement")
        != 2.0 * island.optimizer_llt_scale_uh
        or any(
            scales.get(name) != island.optimizer_all_thermal_scale_c
            for name in thermal_names
        )
        or result.get("optimizer_resonance_allowance_Hz")
        != island.optimizer_resonance_allowance_hz
        or result.get("optimizer_Llt_allowance_uH")
        != island.optimizer_llt_allowance_uh
        or result.get("optimizer_constraint_allowance_contract")
        != expected_allowance
        or result.get("acquisition_ranking_contract") is not None
        or any(
            not isinstance(result.get(field), dict)
            or set(result[field]) != set(names)
            or any(
                isinstance(value, bool) or not math.isfinite(float(value))
                for value in result[field].values()
            )
            for field in physical_fields
        )
        or not _snapshot_valid(
            result.get("terminal_population_best_design"),
            result.get("terminal_population_best_constraint_G"),
        )
        or not _snapshot_valid(
            result.get("optimizer_terminal_best_design"),
            result.get("optimizer_terminal_best_physical_constraint_G"),
        )
        or any(
            result.get(field) is not False
            for field in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed", "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError(f"{island.island_id} preflight result mismatch")
    initialization = result.get("initialization_audit") or {}
    warm_filter = initialization.get("warm_filter") or {}
    if int(warm_filter.get("decoded_unique_count", -1)) < 32:
        raise RuntimeError(f"{island.island_id} warm population degenerated")
    return {
        "island_id": island.island_id,
        "variant": island.variant,
        "island_profile": island_profile(island),
        "seed": int(result["seed"]),
        "population": int(result["population"]),
        "requested_evolution_generations": int(result["max_generations"]),
        "observed_algorithm_n_gen_counter": int(result["completed_generations"]),
        "evaluated_generations": int(result["evaluated_generations"]),
        "post_repair_decoded_unique_count": int(
            warm_filter.get("decoded_unique_count", -1)
        ),
        "fixed_primary_turns_verified": True,
        "physical_terminal_replay_verified": True,
        "turn_split_topologies_preserved": topology_counts,
        "paired_parent_pairs_emitted": int(
            topology_audit["paired_parent_pairs_emitted"]
        ),
        "migration_events": int(topology_audit["migration_events"]),
        "last_optimizer_epsilon": float(
            topology_audit["last_optimizer_epsilon"]
        ),
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }


def validate_preflight(path: Path) -> dict:
    path = path.resolve(strict=True)
    root = path.parent.resolve(strict=True)
    contract = _read(path)
    records = contract.get("results")
    warm_sources = contract.get("warm_sources")
    anchor_replay_record = contract.get("anchor_physical_replay")
    topology_evidence = contract.get("n1_5_warm_topology_evidence")
    if (
        contract.get("schema_version") != SCHEMA
        or contract.get("constraint_version") != CONSTRAINT_VERSION
        or contract.get("hard_spec") != HARD_SPEC
        or contract.get("hard_spec_sha256") != canonical_sha(HARD_SPEC)
        or contract.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or contract.get("optimizer_termination_strategy")
        != TERMINATION_STRATEGY
        or not isinstance(contract.get("max_generations"), int)
        or contract["max_generations"] < 1
        or contract.get("parallel_local_processes") is not True
        or contract.get("island_count") != len(DEEP_CROSSOVER_ISLANDS)
        or set(warm_sources or {}) != {"n1_5", "n1_6"}
        or not isinstance(anchor_replay_record, dict)
        or not isinstance(topology_evidence, dict)
        or topology_evidence.get("schema_version")
        != "mft-tier1-n1-5-warm-topology-strata-v1"
        or canonical_sha({
            key: value for key, value in topology_evidence.items()
            if key != "sha256"
        }) != topology_evidence.get("sha256")
        or topology_evidence.get("warm_start_sha256")
        != (warm_sources or {}).get("n1_5", {}).get("sha256")
        or topology_evidence.get("required_N2_main_values")
        != [28, 29, 30, 31, 32, 33]
        or topology_evidence.get("all_required_topologies_present") is not True
        or any(
            not (topology_evidence.get("required_N2_main_warm_indices") or {}).get(
                str(value)
            )
            for value in (28, 29, 30, 31, 32, 33)
        )
        or not isinstance(records, list)
        or len(records) != len(DEEP_CROSSOVER_ISLANDS)
        or any(
            contract.get(field) is not False
            for field in (
                "scheduler_write_performed", "slurm_submission_performed",
                "fea_submission_performed", "aedt_used",
            )
        )
    ):
        raise RuntimeError("deep crossover preflight top-level mismatch")
    anchor_relative = Path(str(anchor_replay_record.get("path") or ""))
    anchor_path = (root / anchor_relative).resolve(strict=True)
    if (
        anchor_relative.is_absolute() or not anchor_relative.parts
        or ".." in anchor_relative.parts or root not in anchor_path.parents
        or _sha(anchor_path) != anchor_replay_record.get("sha256")
        or anchor_path.stat().st_size != anchor_replay_record.get("size_bytes")
    ):
        raise RuntimeError("deep crossover anchor replay bytes drifted")
    anchor_replay = _read(anchor_path)
    source_physical_g = anchor_replay.get("source_physical_constraint_G")
    replayed_physical_g = anchor_replay.get("replayed_physical_constraint_G")
    replay_names_match = (
        isinstance(source_physical_g, dict)
        and isinstance(replayed_physical_g, dict)
        and set(source_physical_g) == set(replayed_physical_g)
    )
    replay_maximum_delta = (
        max(
            abs(
                float(replayed_physical_g[name])
                - float(source_physical_g[name])
            )
            for name in source_physical_g
        )
        if replay_names_match else math.inf
    )
    replay_classification_match = replay_names_match and all(
        (float(replayed_physical_g[name]) <= 1e-9)
        == (float(source_physical_g[name]) <= 1e-9)
        for name in source_physical_g
    )
    if (
        anchor_replay.get("schema_version") != ANCHOR_REPLAY_SCHEMA
        or anchor_replay.get("source_decoded_params_sha256")
        != ANCHOR_DECODED_PARAMS_SHA256
        or canonical_sha(anchor_replay.get("source_decoded_params"))
        != ANCHOR_DECODED_PARAMS_SHA256
        or canonical_sha(anchor_replay.get(
            "current_replayed_decoded_params"
        )) != anchor_replay.get(
            "current_replayed_decoded_params_sha256"
        )
        or anchor_replay.get("source_physical_constraint_G_sha256")
        != ANCHOR_CONSTRAINT_G_SHA256
        or canonical_sha(anchor_replay.get("source_physical_constraint_G"))
        != ANCHOR_CONSTRAINT_G_SHA256
        or canonical_sha(anchor_replay.get("replayed_physical_constraint_G"))
        != anchor_replay.get("replayed_physical_constraint_G_sha256")
        or not math.isfinite(float(anchor_replay.get(
            "maximum_observed_physical_constraint_G_absolute_drift", math.inf,
        )))
        or not math.isclose(
            float(anchor_replay.get(
                "maximum_observed_physical_constraint_G_absolute_drift",
                math.inf,
            )), replay_maximum_delta, rel_tol=0.0, abs_tol=0.0,
        )
        or not replay_classification_match
        or anchor_replay.get(
            "physical_constraint_pass_classification_identical"
        ) is not True
        or anchor_replay.get(
            "source_vs_current_physical_G_numerical_equality_claimed"
        ) is not False
        or anchor_replay.get(
            "optimizer_deterministic_replay_claimed"
        ) is not False
        or anchor_replay.get(
            "original_terminal_chromosome_authority"
        ) is not False
        or anchor_replay.get("inverse_coordinate_authority")
        != "persisted_source_decoded_params_not_terminal_X"
        or (anchor_replay.get("anchor_evidence") or {}).get(
            "coordinate_sha256"
        ) != ANCHOR_INVERSE_COORDINATE_SHA256
        or anchor_replay.get("verified") is not True
        or any(
            anchor_replay.get(field) is not False
            for field in (
                "physical_evaluator_mutation", "scheduler_write_performed",
                "fea_submission_performed", "aedt_used",
            )
        )
        or anchor_replay.get(
            "optimizer_normalization_not_used_for_replay"
        ) is not True
    ):
        raise RuntimeError("deep crossover anchor physical replay mismatch")
    expected = {item.variant: item for item in DEEP_CROSSOVER_ISLANDS}
    summaries = []
    seen = set()
    for record in records:
        variant = record.get("variant")
        if variant in seen or variant not in expected:
            raise RuntimeError("deep crossover preflight identity mismatch")
        seen.add(variant)
        relative = Path(str(record.get("path") or ""))
        result_path = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute() or not relative.parts or ".." in relative.parts
            or root not in result_path.parents
            or _sha(result_path) != record.get("sha256")
            or result_path.stat().st_size != record.get("size_bytes")
        ):
            raise RuntimeError("deep crossover preflight bytes drifted")
        summary = _summary(
            _read(result_path), expected[variant], contract["max_generations"],
        )
        if summary != record.get("verified_summary"):
            raise RuntimeError("deep crossover preflight summary drifted")
        summaries.append(summary)
    if seen != set(expected) or canonical_sha(summaries) != contract.get(
        "verified_summaries_sha256"
    ):
        raise RuntimeError("deep crossover preflight result set mismatch")
    return contract


def seal_existing_preflight(
    *, feedback: Path, model_dir: Path, code_root: Path,
    n1_5_warm: Path, n1_6_warm: Path, output: Path,
    max_generations: int = 2, inference_threads: int = 8,
) -> dict:
    """Authenticate existing local results without rerunning the optimizers."""
    feedback = feedback.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    code_root = code_root.resolve(strict=True)
    n1_5_warm = n1_5_warm.resolve(strict=True)
    n1_6_warm = n1_6_warm.resolve(strict=True)
    output = output.resolve(strict=True)
    validate_n1_5_warm(n1_5_warm, code_root)
    validate_n1_6_warm(n1_6_warm, code_root)
    anchor_replay = _anchor_physical_replay(
        model_dir=model_dir, code_root=code_root,
        anchor_record=n1_5_warm.parent / "authenticated_anchor" / "record.json",
        inference_threads=inference_threads,
    )
    anchor_replay_path = output / "anchor_physical_replay.json"
    _atomic_json(anchor_replay_path, anchor_replay)
    topology_evidence = _n1_5_topology_evidence(n1_5_warm, code_root)
    warm_paths = {5: n1_5_warm, 6: n1_6_warm}
    warm_shas = {turns: _sha(path) for turns, path in warm_paths.items()}
    records = []
    summaries = []
    for island in DEEP_CROSSOVER_ISLANDS:
        result_path = output / "results" / island.island_id / "result.json"
        result_path = result_path.resolve(strict=True)
        summary = _summary(_read(result_path), island, max_generations)
        summaries.append(summary)
        records.append({
            "variant": island.variant,
            "path": result_path.relative_to(output).as_posix(),
            "sha256": _sha(result_path),
            "size_bytes": result_path.stat().st_size,
            "verified_summary": summary,
        })
    contract = {
        "schema_version": SCHEMA,
        "created_at": _now(),
        "constraint_version": CONSTRAINT_VERSION,
        "hard_spec": HARD_SPEC,
        "hard_spec_sha256": canonical_sha(HARD_SPEC),
        "nsga_code_revision": EXPECTED_NSGA_REVISION,
        "feedback_script_sha256": _sha(feedback),
        "model_manifest_sha256": _sha(model_dir / "manifest.json"),
        "warm_sources": {
            "n1_5": {"path": str(n1_5_warm), "sha256": warm_shas[5]},
            "n1_6": {"path": str(n1_6_warm), "sha256": warm_shas[6]},
        },
        "anchor_physical_replay": {
            "path": anchor_replay_path.relative_to(output).as_posix(),
            "sha256": _sha(anchor_replay_path),
            "size_bytes": anchor_replay_path.stat().st_size,
        },
        "n1_5_warm_topology_evidence": topology_evidence,
        "optimizer_termination_strategy": TERMINATION_STRATEGY,
        "max_generations": int(max_generations),
        "island_count": len(DEEP_CROSSOVER_ISLANDS),
        "parallel_local_processes": True,
        "results": records,
        "verified_summaries_sha256": canonical_sha(summaries),
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    path = output / "preflight_contract.json"
    _atomic_json(path, contract)
    return validate_preflight(path)


def run_preflight(
    *, python: Path, feedback: Path, model_dir: Path, code_root: Path,
    n1_5_warm: Path, n1_6_warm: Path, output: Path,
    population: int = 64, max_generations: int = 2,
    inference_threads: int = 8,
) -> dict:
    python = python.resolve(strict=True)
    feedback = feedback.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    code_root = code_root.resolve(strict=True)
    n1_5_warm = n1_5_warm.resolve(strict=True)
    n1_6_warm = n1_6_warm.resolve(strict=True)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("preflight output must be absent or empty")
    validate_n1_5_warm(n1_5_warm, code_root)
    validate_n1_6_warm(n1_6_warm, code_root)
    output.mkdir(parents=True, exist_ok=True)
    warm_paths = {5: n1_5_warm, 6: n1_6_warm}
    warm_shas = {turns: _sha(path) for turns, path in warm_paths.items()}
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{count}"] = code_root.as_posix()
    environment["GIT_CONFIG_COUNT"] = str(count + 1)
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(int(inference_threads))
    processes = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for island in DEEP_CROSSOVER_ISLANDS:
        warm = warm_paths[island.fixed_primary_turns]
        target = output / "results" / island.island_id
        target.mkdir(parents=True, exist_ok=False)
        command = [
            str(python), str(feedback), "search-seed",
            "--model-dir", str(model_dir),
            "--nsga-code-root", str(code_root),
            "--output", str(target),
            "--seed", str(island.seed_start),
            "--population", str(int(population)),
            "--max-generations", str(int(max_generations)),
            "--optimizer-termination-strategy", TERMINATION_STRATEGY,
            "--warm-start", str(warm),
            "--warm-start-sha256", warm_shas[island.fixed_primary_turns],
            "--inference-threads", str(int(inference_threads)),
            "--optimizer-resonance-scale-hz", str(RESONANCE_SCALE_HZ),
            "--optimizer-llt-scale-uh", str(island.optimizer_llt_scale_uh),
            "--optimizer-all-thermal-scale-c",
            str(island.optimizer_all_thermal_scale_c),
            "--fixed-primary-turns", str(island.fixed_primary_turns),
        ]
        if island.optimizer_resonance_allowance_hz is not None:
            command.extend([
                "--optimizer-resonance-allowance-hz",
                str(island.optimizer_resonance_allowance_hz),
                "--optimizer-llt-allowance-uh",
                str(island.optimizer_llt_allowance_uh),
            ])
        stdout = (target / "stdout.log").open("wb")
        stderr = (target / "stderr.log").open("wb")
        process = subprocess.Popen(
            command, cwd=code_root, stdout=stdout, stderr=stderr,
            stdin=subprocess.DEVNULL, creationflags=creationflags,
            env=environment,
        )
        processes.append((island, target, command, process, stdout, stderr))
    failures = []
    for island, target, command, process, stdout, stderr in processes:
        returncode = process.wait()
        stdout.close()
        stderr.close()
        if returncode != 0:
            failures.append({
                "island_id": island.island_id, "returncode": returncode,
                "command": command, "stderr": str(target / "stderr.log"),
            })
    if failures:
        raise RuntimeError(f"deep crossover local preflight failed: {failures}")
    return seal_existing_preflight(
        feedback=feedback, model_dir=model_dir, code_root=code_root,
        n1_5_warm=n1_5_warm, n1_6_warm=n1_6_warm, output=output,
        max_generations=max_generations,
        inference_threads=inference_threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--feedback", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--code-root", type=Path, required=True)
    run.add_argument("--n1-5-warm", type=Path, required=True)
    run.add_argument("--n1-6-warm", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--population", type=int, default=64)
    run.add_argument("--max-generations", type=int, default=2)
    run.add_argument("--inference-threads", type=int, default=8)
    seal = commands.add_parser("seal-existing")
    seal.add_argument("--feedback", type=Path, required=True)
    seal.add_argument("--model-dir", type=Path, required=True)
    seal.add_argument("--code-root", type=Path, required=True)
    seal.add_argument("--n1-5-warm", type=Path, required=True)
    seal.add_argument("--n1-6-warm", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--max-generations", type=int, default=2)
    seal.add_argument("--inference-threads", type=int, default=8)
    validate = commands.add_parser("validate")
    validate.add_argument("--contract", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        value = run_preflight(
            python=args.python, feedback=args.feedback,
            model_dir=args.model_dir, code_root=args.code_root,
            n1_5_warm=args.n1_5_warm, n1_6_warm=args.n1_6_warm,
            output=args.output, population=args.population,
            max_generations=args.max_generations,
            inference_threads=args.inference_threads,
        )
    elif args.command == "seal-existing":
        value = seal_existing_preflight(
            feedback=args.feedback, model_dir=args.model_dir,
            code_root=args.code_root, n1_5_warm=args.n1_5_warm,
            n1_6_warm=args.n1_6_warm, output=args.output,
            max_generations=args.max_generations,
            inference_threads=args.inference_threads,
        )
    else:
        value = validate_preflight(args.contract)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
