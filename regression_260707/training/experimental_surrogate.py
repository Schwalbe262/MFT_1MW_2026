"""Fail-closed publication for the FEA-disabled experimental surrogate lane."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from filelock import FileLock
import pandas as pd


MODEL_SOURCE_SCHEMA = "mft-nsga-model-source-v1"
POINTER_SCHEMA = "mft-experimental-surrogate-pointer-v1"
EXPERIMENTAL_MINIMUM_ROWS = 2000
PRODUCTION_MINIMUM_ROWS = 3000
HEX = frozenset("0123456789abcdef")


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=1, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    fd, staged = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
        if sha256_file(target) != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"atomic JSON fingerprint mismatch: {target}")
    finally:
        try:
            os.remove(staged)
        except FileNotFoundError:
            pass


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _exact_revision(value, label):
    normalized = str(value or "").lower()
    if len(normalized) != 40 or any(char not in HEX for char in normalized):
        raise RuntimeError(f"{label} must be an exact 40-character revision")
    return normalized


def _within(path, root):
    path = os.path.abspath(os.fspath(path))
    root = os.path.abspath(os.fspath(root))
    return os.path.commonpath([path, root]) == root


def validate_source_descriptor(path):
    """Authenticate one descriptor exactly as the continuous NSGA consumer does."""
    source_path = os.path.abspath(os.fspath(path))
    value = _read_json(source_path)
    if value.get("schema_version") != MODEL_SOURCE_SCHEMA:
        raise RuntimeError("unsupported experimental source schema")
    expected = {
        "lane": "experimental",
        "production_eligible": False,
        "fea_submission_approved": False,
    }
    mismatches = [key for key, item in expected.items() if value.get(key) != item]
    if mismatches:
        raise RuntimeError(
            "experimental isolation contract mismatch: " + ",".join(mismatches)
        )
    registry = os.path.abspath(value["registry"])
    generation = os.path.abspath(value["generation"])
    if not _within(generation, os.path.join(registry, "generations")):
        raise RuntimeError("experimental generation escapes registry")
    dataset = os.path.abspath(value["dataset"])
    quality_path = os.path.abspath(value["quality_status"])
    from train_models import load_generation

    record = load_generation(registry, generation, require_accepted=False)
    quality = _read_json(quality_path)
    dataset_sha256 = sha256_file(dataset)
    checks = {
        "dataset_sha256": dataset_sha256,
        "generation_report_sha256": record["generation_report_sha256"],
    }
    for key, expected_value in checks.items():
        if quality.get(key) != expected_value:
            raise RuntimeError(f"experimental quality {key} mismatch")
    if record["report"].get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("experimental report dataset mismatch")
    if quality.get("passed") is not False:
        raise RuntimeError("experimental quality evidence is not explicitly failed")
    blockers = quality.get("failed_targets")
    if not isinstance(blockers, dict) or not blockers:
        raise RuntimeError("experimental quality blockers are unavailable")
    return {
        "path": source_path,
        "source": value,
        "source_sha256": sha256_file(source_path),
        "registry": registry,
        "generation": generation,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "quality_status": quality_path,
        "quality_status_sha256": sha256_file(quality_path),
        "quality": quality,
        "record": record,
    }


def _compact_reason(reason):
    value = str(reason)
    if ":" in value:
        prefix, suffix = value.split(":", 1)
        if prefix in {
            "metric_above_maximum", "metric_below_minimum", "nonfinite_metric"
        }:
            return suffix
    return value


def build_experimental_quality(
    registry, generation, dataset, thresholds_path, profile_path,
    solver_revision, library_revision,
):
    """Seal a production-gate failure for >=2k active-learning/NSGA use."""
    from model_quality_gate import evaluate_generation
    from quality_contract import annotate_validity
    from train_models import load_generation

    solver_revision = _exact_revision(solver_revision, "solver revision")
    library_revision = _exact_revision(library_revision, "library revision")
    with open(thresholds_path, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    record = load_generation(registry, generation, require_accepted=False)
    evaluated = evaluate_generation(registry, generation, dataset, thresholds)
    if evaluated.get("passed") is not False:
        raise RuntimeError(
            "experimental publisher refuses a production-passing generation"
        )
    audited = annotate_validity(
        pd.read_parquet(dataset), profile_path,
        expected_solver_revision=solver_revision,
        expected_library_revision=library_revision,
    )
    strict_full_rows = int(audited["_strict_valid_full"].sum())
    if strict_full_rows < EXPERIMENTAL_MINIMUM_ROWS:
        raise RuntimeError(
            f"experimental surrogate requires >={EXPERIMENTAL_MINIMUM_ROWS} rows"
        )
    if strict_full_rows != int(record["report"].get("strict_full_rows") or -1):
        raise RuntimeError("experimental strict row identity mismatch")
    failed_targets = {}
    for target, status in (evaluated.get("targets") or {}).items():
        reasons = status.get("reasons") or []
        if status.get("blocking") and reasons:
            failed_targets[target] = [_compact_reason(item) for item in reasons]
    target_prefixes = tuple(f"{target}:" for target in (evaluated.get("targets") or {}))
    global_reasons = [
        reason for reason in evaluated.get("reasons", [])
        if not str(reason).startswith(target_prefixes)
    ]
    if global_reasons:
        failed_targets["__global__"] = [str(item) for item in global_reasons]
    if not failed_targets:
        raise RuntimeError("experimental quality gate produced no blockers")
    return {
        "schema_version": 1,
        "evaluated_at": _now(),
        "lane": "provisional_2000_surrogate",
        "eligibility": "FEA-NOT-APPROVED",
        "passed": False,
        "activation_performed": False,
        "nsga2_enqueued": False,
        "verification_enqueued": False,
        "strict_full_rows": strict_full_rows,
        "strict_em_rows": int(audited["_strict_valid_em"].sum()),
        "raw_rows": int(len(audited)),
        "production_minimum_strict_full_rows": int(
            thresholds.get("minimum_strict_full_rows", PRODUCTION_MINIMUM_ROWS)
        ),
        "provisional_minimum_strict_full_rows": EXPERIMENTAL_MINIMUM_ROWS,
        "solver_revision_pin": solver_revision,
        "library_revision_pin": library_revision,
        "dataset_sha256": sha256_file(dataset),
        "generation": record["generation_relative"],
        "generation_report_sha256": record["generation_report_sha256"],
        "profile_sha256": record["report"].get("profile_sha256"),
        "provisional_thresholds_sha256": sha256_file(thresholds_path),
        "failed_targets": failed_targets,
        "raw_gate_reasons": evaluated.get("reasons", []),
        "terminal_reason": "provisional_quality_gate_failed",
    }


def _descriptor(
    registry, record, dataset, quality_path, solver_revision, library_revision,
    comparison, published_at,
):
    return {
        "schema_version": MODEL_SOURCE_SCHEMA,
        "pointer_schema": POINTER_SCHEMA,
        "lane": "experimental",
        "label": "EXPERIMENTAL / FEA-NOT-APPROVED",
        "eligibility": "FEA-NOT-APPROVED",
        "production_eligible": False,
        "fea_submission_approved": False,
        "registry": os.path.abspath(registry),
        "generation": record["generation"],
        "generation_report_sha256": record["generation_report_sha256"],
        "training_run_id": record["report"]["training_run_id"],
        "strict_full_rows": int(record["report"]["strict_full_rows"]),
        "dataset": os.path.abspath(dataset),
        "dataset_sha256": sha256_file(dataset),
        "quality_status": os.path.abspath(quality_path),
        "quality_status_sha256": sha256_file(quality_path),
        "fea_solver_revision": _exact_revision(solver_revision, "FEA solver"),
        "fea_library_revision": _exact_revision(library_revision, "FEA library"),
        "incumbent_comparison": comparison,
        "published_at": published_at,
        "consumer_contract": (
            "read and authenticate once at child-search launch; never hot-swap "
            "an already-running NSGA seed batch"
        ),
    }


def initialize_pointer(source_path, pointer_path, consumer_path=None):
    """Migrate an already authenticated experimental source into the pointer."""
    incumbent = validate_source_descriptor(source_path)
    source = dict(incumbent["source"])
    source.update({
        "pointer_schema": POINTER_SCHEMA,
        "eligibility": "FEA-NOT-APPROVED",
        "generation_report_sha256": incumbent["record"][
            "generation_report_sha256"
        ],
        "dataset_sha256": incumbent["dataset_sha256"],
        "quality_status_sha256": incumbent["quality_status_sha256"],
        "training_run_id": incumbent["record"]["report"]["training_run_id"],
        "strict_full_rows": int(
            incumbent["record"]["report"]["strict_full_rows"]
        ),
        "published_at": _now(),
        "consumer_contract": (
            "read and authenticate once at child-search launch; never hot-swap "
            "an already-running NSGA seed batch"
        ),
    })
    atomic_json(pointer_path, source)
    if consumer_path:
        atomic_json(consumer_path, source)
        if sha256_file(pointer_path) != sha256_file(consumer_path):
            raise RuntimeError("experimental pointer/consumer descriptor diverged")
    return source


def publish_if_better(
    *, registry, generation, dataset, quality, evidence_root,
    pointer_path, consumer_path, incumbent_source,
    solver_revision, library_revision,
):
    """Compare, then atomically expose a candidate to only the next NSGA run."""
    from train_models import compare_candidate_to_incumbent, load_generation

    registry = os.path.abspath(registry)
    candidate = load_generation(registry, generation, require_accepted=False)
    incumbent = validate_source_descriptor(incumbent_source)
    comparison = compare_candidate_to_incumbent(
        candidate["report"], incumbent["record"]["report"]
    )
    run_id = candidate["report"]["training_run_id"]
    evidence_dir = Path(evidence_root).resolve() / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    quality_path = evidence_dir / "quality_status.json"
    atomic_json(quality_path, quality)
    result = {
        "schema_version": 1,
        "candidate_training_run_id": run_id,
        "candidate_generation": candidate["generation"],
        "candidate_generation_report_sha256": candidate[
            "generation_report_sha256"
        ],
        "candidate_dataset_sha256": sha256_file(dataset),
        "quality_status": str(quality_path),
        "quality_status_sha256": sha256_file(quality_path),
        "incumbent_source_sha256": incumbent["source_sha256"],
        "comparison": comparison,
        "promoted": False,
        "evaluated_at": _now(),
    }
    if not comparison["passed"]:
        atomic_json(evidence_dir / "promotion_result.json", result)
        return result

    lock_path = os.path.abspath(pointer_path) + ".lock"
    with FileLock(lock_path, timeout=5):
        current = validate_source_descriptor(incumbent_source)
        if current["source_sha256"] != incumbent["source_sha256"]:
            comparison = compare_candidate_to_incumbent(
                candidate["report"], current["record"]["report"]
            )
            result["comparison_after_pointer_change"] = comparison
            if not comparison["passed"]:
                atomic_json(evidence_dir / "promotion_result.json", result)
                return result
        published_at = _now()
        descriptor = _descriptor(
            registry, candidate, dataset, quality_path,
            solver_revision, library_revision, comparison, published_at,
        )
        # Each destination is atomic.  The audit pointer commits first, then
        # the consumer descriptor; both are byte-identical and verified before
        # success is reported.  Existing child searches remain pinned.
        atomic_json(pointer_path, descriptor)
        atomic_json(consumer_path, descriptor)
        pointer_sha = sha256_file(pointer_path)
        if pointer_sha != sha256_file(consumer_path):
            raise RuntimeError("experimental pointer/consumer descriptor diverged")
        validate_source_descriptor(consumer_path)
        result.update({
            "promoted": True,
            "published_at": published_at,
            "pointer": os.path.abspath(pointer_path),
            "pointer_sha256": pointer_sha,
            "consumer_source": os.path.abspath(consumer_path),
            "consumer_source_sha256": pointer_sha,
        })
        atomic_json(evidence_dir / "promotion_result.json", result)
        return result
