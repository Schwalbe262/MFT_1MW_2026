"""Fail-closed adapter for the reviewed standard/fine FEA hand-off.

The external command owns scheduler submission and reconciliation, but it may
only consume the exact request written here and must return one terminal record
for every requested candidate.  No marker is written until identity and count
validation succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .artifacts import GenerationStore
from .policy import FINE_VERIFICATION_COUNT, STANDARD_VERIFICATION_COUNT


SCHEMA_VERSION = 1


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=1,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _load_generation(path: str | os.PathLike[str], expected_kind: str):
    resolved = Path(path).resolve()
    if len(resolved.parents) < 2:
        raise RuntimeError("verification input is not a generation directory")
    generation = GenerationStore(resolved.parents[1]).load(resolved)
    if generation.kind != expected_kind:
        raise RuntimeError(
            f"verification input kind mismatch: {generation.kind}!={expected_kind}"
        )
    return generation


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _standard_selection(generation, expected_count: int) -> list[dict]:
    front = pd.read_csv(generation.path / "pareto_front.csv")
    vectors = np.load(generation.path / "pareto_X.npy", allow_pickle=False)
    if len(front) != len(vectors):
        raise RuntimeError("Pareto row/vector count mismatch")
    if len(front) < expected_count:
        raise RuntimeError(
            f"standard verification requires {expected_count} Pareto candidates; "
            f"found {len(front)}"
        )
    for column in ("volume_L", "total_loss_W"):
        if column not in front:
            raise RuntimeError(f"Pareto front is missing {column}")
        values = pd.to_numeric(front[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Pareto front has non-finite {column}")

    # The reviewed policy spans the entire non-dominated front, including both
    # extrema, without depending on mutable model state.
    ordered = np.lexsort(
        (
            np.arange(len(front)),
            front["total_loss_W"].to_numpy(float),
            front["volume_L"].to_numpy(float),
        )
    )
    positions = np.rint(
        np.linspace(0, len(ordered) - 1, expected_count)
    ).astype(int)
    selected = ordered[positions]
    if len(set(int(value) for value in selected)) != expected_count:
        raise RuntimeError("reviewed Pareto span did not produce an exact count")

    candidates = []
    for source_index in selected:
        source_index = int(source_index)
        row = {
            str(key): _json_value(value)
            for key, value in front.iloc[source_index].to_dict().items()
        }
        candidates.append(
            {
                "candidate_id": (
                    f"{generation.generation_id}:{source_index:08d}"
                ),
                "source_index": source_index,
                "x_unit": [float(value) for value in vectors[source_index]],
                "predicted_volume_L": float(front.iloc[source_index]["volume_L"]),
                "predicted_total_loss_W": float(
                    front.iloc[source_index]["total_loss_W"]
                ),
                "parameters": row,
            }
        )
    return candidates


def _load_completed_results(path: Path, stage: str, candidates: list[dict]):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported verification result schema")
    if document.get("stage") != stage:
        raise RuntimeError("verification result stage mismatch")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != len(candidates):
        raise RuntimeError("verification result count mismatch")
    expected = [item["candidate_id"] for item in candidates]
    actual = [
        item.get("candidate_id") if isinstance(item, dict) else None
        for item in results
    ]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RuntimeError("verification result candidate identity mismatch")
    by_id = {item["candidate_id"]: item for item in results}
    ordered = []
    for candidate_id in expected:
        item = by_id[candidate_id]
        if item.get("completed") is not True:
            raise RuntimeError(f"candidate is not terminal: {candidate_id}")
        if not isinstance(item.get("valid"), bool):
            raise RuntimeError(f"candidate validity is ambiguous: {candidate_id}")
        if item["valid"]:
            volume = item.get("actual_volume_L")
            if (
                not isinstance(volume, (int, float))
                or not math.isfinite(float(volume))
                or float(volume) <= 0
            ):
                raise RuntimeError(
                    f"valid candidate has no physical volume: {candidate_id}"
                )
        ordered.append(item)
    return document, ordered


def _fine_selection(generation, expected_count: int) -> list[dict]:
    request = json.loads(
        (generation.path / "selection.json").read_text(encoding="utf-8")
    )
    if (
        request.get("stage") != "standard"
        or request.get("expected_count") != STANDARD_VERIFICATION_COUNT
    ):
        raise RuntimeError("standard selection contract is unavailable")
    candidates = request.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != STANDARD_VERIFICATION_COUNT:
        raise RuntimeError("standard selection count is invalid")
    _, results = _load_completed_results(
        generation.path / "verification_results.json", "standard", candidates
    )
    by_id = {item["candidate_id"]: item for item in candidates}
    eligible = [item for item in results if item["valid"]]
    eligible.sort(
        key=lambda item: (
            float(item["actual_volume_L"]),
            float(item.get("actual_total_loss_W") or math.inf),
            item["candidate_id"],
        )
    )
    if len(eligible) < expected_count:
        raise RuntimeError(
            f"fine verification requires {expected_count} valid standard designs; "
            f"found {len(eligible)}"
        )
    output = []
    for result in eligible[:expected_count]:
        candidate = dict(by_id[result["candidate_id"]])
        candidate["standard_evidence"] = {
            key: result.get(key)
            for key in (
                "candidate_id", "task_id", "attempt",
                "actual_volume_L", "actual_total_loss_W",
            )
        }
        output.append(candidate)
    return output


def _exact_git_sha(value) -> bool:
    text = str(value or "").lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _source_revisions(stage, generation):
    if stage == "standard":
        source = json.loads(
            (generation.path / "optimization_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        source = json.loads(
            (generation.path / "selection.json").read_text(encoding="utf-8")
        )
    revisions = {
        key: str(source.get(key) or "").lower()
        for key in (
            "training_solver_revision",
            "training_library_revision",
            "fea_solver_revision",
            "fea_library_revision",
        )
    }
    if not all(_exact_git_sha(value) for value in revisions.values()):
        raise RuntimeError(
            "verification source has no separate exact training/FEA revisions"
        )
    # Old names remain model provenance in optimization manifests, but are
    # execution pins in selection requests.  Both meanings are authenticated
    # explicitly instead of silently copying one revision into the other.
    if stage == "standard":
        if (
            str(source.get("solver_revision") or "").lower()
            != revisions["training_solver_revision"]
            or str(source.get("library_revision") or "").lower()
            != revisions["training_library_revision"]
        ):
            raise RuntimeError("optimization training provenance is inconsistent")
    elif (
        str(source.get("solver_revision") or "").lower()
        != revisions["fea_solver_revision"]
        or str(source.get("library_revision") or "").lower()
        != revisions["fea_library_revision"]
    ):
        raise RuntimeError("verification FEA execution provenance is inconsistent")
    return revisions


def run(stage, input_generation, output_dir, expected_count, adapter_config):
    expected = (
        STANDARD_VERIFICATION_COUNT if stage == "standard"
        else FINE_VERIFICATION_COUNT
    )
    if int(expected_count) != expected:
        raise RuntimeError(
            f"{stage} verification count is sealed at {expected}"
        )
    generation = _load_generation(
        input_generation,
        "optimization" if stage == "standard" else "verification_standard",
    )
    candidates = (
        _standard_selection(generation, expected)
        if stage == "standard"
        else _fine_selection(generation, expected)
    )
    if len(candidates) != expected:
        raise RuntimeError("verification selection count mismatch")
    revisions = _source_revisions(stage, generation)
    solver_revision = revisions["fea_solver_revision"]
    library_revision = revisions["fea_library_revision"]

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "selection.json"
    result_path = output / "verification_results.json"
    status_path = output / "verification_status.json"
    for stale in (result_path, status_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    request = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "expected_count": expected,
        "selection_policy": (
            "pareto_span_v1" if stage == "standard"
            else "smallest_valid_actual_volume_v1"
        ),
        "input_generation_id": generation.generation_id,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        **revisions,
        "input_manifest_sha256": hashlib.sha256(
            (generation.path / "manifest.json").read_bytes()
        ).hexdigest(),
        "candidates": candidates,
    }
    _atomic_json(request, request_path)
    from .scheduler_verification import run_scheduler_verification

    run_scheduler_verification(request_path, result_path, adapter_config)
    _, results = _load_completed_results(result_path, stage, candidates)
    status = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "completed": True,
        "requested_count": expected,
        "terminal_count": len(results),
        "valid_count": sum(bool(item["valid"]) for item in results),
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    _atomic_json(status, status_path)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("standard", "fine"), required=True)
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--adapter-config-json", required=True)
    args = parser.parse_args()
    adapter_config = json.loads(args.adapter_config_json)
    if not isinstance(adapter_config, dict):
        parser.error("adapter config JSON must be an object")
    print(
        json.dumps(
            run(
                args.stage,
                args.input_generation,
                args.output_dir,
                args.expected_count,
                adapter_config,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
