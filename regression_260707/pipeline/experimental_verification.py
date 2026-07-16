"""Sealed non-production NSGA-II -> three-candidate full FEA feedback lane.

This module intentionally does not reuse the production 33 -> 3 promotion
policy.  It authenticates an immutable experimental optimization generation,
spans its Pareto front with exactly three designs, and records enough
provenance for the ordinary scheduler collector to feed the FEA truth back
into later training snapshots.  Results remain experimental regardless of FEA
outcome because the source surrogate failed the production quality gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .artifacts import GenerationStore, sha256_file
from .scheduler_verification import run_scheduler_verification
from .verification_adapter import (
    SCHEMA_VERSION,
    _atomic_json,
    _load_completed_results,
    _standard_selection,
)


EXPECTED_KIND = "experimental_optimization"
EXPECTED_COUNT = 3


def _exact_git_sha(value) -> bool:
    text = str(value or "").lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _load_experimental_generation(path: str | os.PathLike[str]):
    resolved = Path(path).resolve()
    if len(resolved.parents) < 2:
        raise RuntimeError("experimental verification input is not a generation")
    generation = GenerationStore(resolved.parents[1]).load(resolved)
    if generation.kind != EXPECTED_KIND:
        raise RuntimeError(
            f"experimental verification input kind mismatch: "
            f"{generation.kind}!={EXPECTED_KIND}"
        )
    return generation


def _optimization_contract(generation):
    manifest_path = generation.path / "optimization_manifest.json"
    front_path = generation.path / "pareto_front.csv"
    x_path = generation.path / "pareto_X.npy"
    for path in (manifest_path, front_path, x_path):
        if not path.is_file():
            raise RuntimeError(f"experimental optimization artifact is missing: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blockers = manifest.get("quality_blockers")
    quality_sha = str(manifest.get("quality_status_sha256") or "").lower()
    training_solver = str(manifest.get("training_solver_revision") or "").lower()
    training_library = str(manifest.get("training_library_revision") or "").lower()
    fea_solver = str(manifest.get("fea_solver_revision") or "").lower()
    fea_library = str(manifest.get("fea_library_revision") or "").lower()
    if (
        manifest.get("experimental_active_learning") is not True
        or manifest.get("production_eligible") is not False
        or manifest.get("quality_gate_passed") is not False
        or not isinstance(blockers, dict)
        or not blockers
        or int(manifest.get("strict_full_rows") or 0) < 2000
        or manifest.get("experimental_minimum_strict_full_rows") != 2000
        or not all(_exact_git_sha(value) for value in (
            training_solver, training_library, fea_solver, fea_library
        ))
        or str(manifest.get("solver_revision") or "").lower()
        != training_solver
        or str(manifest.get("library_revision") or "").lower()
        != training_library
        or len(quality_sha) != 64
        or any(char not in "0123456789abcdef" for char in quality_sha)
        or manifest.get("pareto_front_sha256") != sha256_file(front_path)
        or manifest.get("pareto_X_sha256") != sha256_file(x_path)
    ):
        raise RuntimeError("experimental optimization provenance is invalid")
    metadata = generation.manifest.get("metadata")
    if isinstance(metadata, dict) and (
        metadata.get("production_eligible") not in (None, False)
        or metadata.get("quality_gate_passed") not in (None, False)
    ):
        raise RuntimeError("experimental generation metadata permits production use")
    return manifest, blockers, quality_sha


def run(input_generation, output_dir, adapter_config):
    generation = _load_experimental_generation(input_generation)
    optimization, blockers, quality_sha = _optimization_contract(generation)
    candidates = _standard_selection(generation, EXPECTED_COUNT)
    if len(candidates) != EXPECTED_COUNT:
        raise RuntimeError("experimental Pareto selection is not exactly three")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "selection.json"
    result_path = output / "verification_results.json"
    status_path = output / "verification_status.json"
    truth_path = output / "truth_ingest_manifest.json"
    request = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fine",
        "expected_count": EXPECTED_COUNT,
        "selection_policy": "experimental_pareto_span_3_v1",
        "experimental_active_learning": True,
        "production_eligible": False,
        "quality_gate_passed": False,
        "quality_blockers": blockers,
        "quality_status_sha256": quality_sha,
        "source_dataset_sha256": optimization.get("dataset_sha256"),
        "source_training_run_id": optimization.get("training_run_id"),
        "source_optimization_manifest_sha256": sha256_file(
            generation.path / "optimization_manifest.json"
        ),
        "input_generation_id": generation.generation_id,
        # The legacy request names are execution pins.  Model provenance is
        # carried separately and must never select the downstream checkout.
        "solver_revision": optimization["fea_solver_revision"],
        "library_revision": optimization["fea_library_revision"],
        "training_solver_revision": optimization["training_solver_revision"],
        "training_library_revision": optimization["training_library_revision"],
        "fea_solver_revision": optimization["fea_solver_revision"],
        "fea_library_revision": optimization["fea_library_revision"],
        "input_manifest_sha256": sha256_file(generation.path / "manifest.json"),
        "candidates": candidates,
    }
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous != request:
            raise RuntimeError("experimental FEA selection changed across retry")
    else:
        _atomic_json(request, request_path)

    run_scheduler_verification(request_path, result_path, adapter_config)
    _, results = _load_completed_results(
        result_path, "fine", candidates
    )
    result_by_id = {item["candidate_id"]: item for item in results}
    feedback = []
    for candidate in candidates:
        item = result_by_id[candidate["candidate_id"]]
        feedback.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_index": candidate["source_index"],
                "scheduler_task_id": item.get("task_id"),
                "attempt": item.get("attempt"),
                "valid": item.get("valid"),
                "actual_volume_L": item.get("actual_volume_L"),
                "actual_total_loss_W": item.get("actual_total_loss_W"),
                "predicted_volume_L": candidate.get("predicted_volume_L"),
                "predicted_total_loss_W": candidate.get(
                    "predicted_total_loss_W"
                ),
            }
        )
    truth = {
        "schema_version": SCHEMA_VERSION,
        "lane": "experimental_active_learning_full_fea",
        "production_eligible": False,
        "quality_gate_passed": False,
        "input_generation_id": generation.generation_id,
        "source_dataset_sha256": optimization.get("dataset_sha256"),
        "source_training_run_id": optimization.get("training_run_id"),
        "quality_status_sha256": quality_sha,
        "quality_blockers": blockers,
        "solver_revision": optimization["fea_solver_revision"],
        "library_revision": optimization["fea_library_revision"],
        "training_solver_revision": optimization["training_solver_revision"],
        "training_library_revision": optimization["training_library_revision"],
        "fea_solver_revision": optimization["fea_solver_revision"],
        "fea_library_revision": optimization["fea_library_revision"],
        "selection_sha256": sha256_file(request_path),
        "verification_results_sha256": sha256_file(result_path),
        "feedback_contract": {
            "source": "scheduler_result_json",
            "collector": "periodic_mft_scheduler_collection",
            "dedupe_identity": "scheduler_task_id",
            "next_training_use": "strict_contract_after_collection",
            "production_promotion": "forbidden_until_independent_quality_gate_passes",
        },
        "results": feedback,
    }
    _atomic_json(truth, truth_path)
    status = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fine",
        "lane": "experimental_active_learning_full_fea",
        "completed": True,
        "production_eligible": False,
        "requested_count": EXPECTED_COUNT,
        "terminal_count": len(results),
        "valid_count": sum(bool(item["valid"]) for item in results),
        "request_sha256": sha256_file(request_path),
        "results_sha256": sha256_file(result_path),
        "truth_ingest_manifest_sha256": sha256_file(truth_path),
    }
    if not all(
        isinstance(item.get("scheduler_task_id"), int)
        and item["scheduler_task_id"] > 0
        for item in feedback
    ):
        raise RuntimeError("experimental FEA truth has no scheduler identities")
    if any(
        value is not None
        and isinstance(value, float)
        and not math.isfinite(value)
        for item in feedback
        for value in (item.get("actual_volume_L"), item.get("actual_total_loss_W"))
    ):
        raise RuntimeError("experimental FEA truth contains non-finite metrics")
    _atomic_json(status, status_path)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "verification_status_sha256": sha256_file(status_path),
            "truth_ingest_manifest_sha256": sha256_file(truth_path),
        },
        output / "COMPLETED",
    )
    return status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-config-json", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.adapter_config_json)
    if not isinstance(config, dict):
        parser.error("adapter config JSON must be an object")
    print(
        json.dumps(
            run(args.input_generation, args.output_dir, config),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
