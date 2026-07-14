"""Fail-closed surrogate accuracy and uncertainty gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.path.join(HERE, "registry")
DEFAULT_THRESHOLDS = os.path.join(HERE, "model_quality_thresholds.json")
REGISTRY_SCHEMA_VERSION = 2
QUALITY_GATE_FILENAME = "quality_gate.json"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, default=str)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _load_generation(registry, generation=None):
    registry = os.path.abspath(registry)
    pointer = None
    if generation is None:
        pointer_path = os.path.join(registry, "current.json")
        with open(pointer_path, encoding="utf-8") as handle:
            pointer = json.load(handle)
        relative = pointer.get("generation")
    else:
        relative = os.fspath(generation)
    if not isinstance(relative, str) or not relative.strip():
        raise RuntimeError("registry generation is missing")
    generation = os.path.abspath(
        relative if os.path.isabs(relative) else os.path.join(registry, relative)
    )
    generations_root = os.path.abspath(os.path.join(registry, "generations"))
    if (
        generation == generations_root
        or os.path.commonpath([generation, generations_root]) != generations_root
    ):
        raise RuntimeError("registry generation escapes generations root")
    with open(
        os.path.join(generation, "train_report.json"), encoding="utf-8"
    ) as handle:
        report = json.load(handle)
    relative = os.path.relpath(generation, registry).replace("\\", "/")
    return pointer, generation, relative, report


def evaluate_registry(registry, dataset, thresholds, generation=None):
    reasons = []
    advisories = []
    target_status = {}
    try:
        pointer, generation, generation_relative, report = _load_generation(
            registry, generation
        )
    except Exception as exc:
        return {
            "passed": False,
            "reasons": [f"registry_unavailable:{exc}"],
            "advisories": [],
            "targets": {},
        }

    run_id = report.get("training_run_id")
    report_path = os.path.join(generation, "train_report.json")
    report_sha256 = _sha256(report_path)
    thresholds_sha256 = hashlib.sha256(
        json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if pointer is not None:
        if pointer.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            reasons.append("ungated_pointer_schema")
        if pointer.get("training_run_id") != run_id:
            reasons.append("pointer_training_run_mismatch")
        if pointer.get("generation_report_sha256") != report_sha256:
            reasons.append("pointer_generation_report_mismatch")
        if pointer.get("profile_sha256") != report.get("profile_sha256"):
            reasons.append("pointer_profile_fingerprint_mismatch")
        if pointer.get("thresholds_sha256") != thresholds_sha256:
            reasons.append("pointer_threshold_fingerprint_mismatch")
        if pointer.get("strict_full_rows") != report.get("strict_full_rows"):
            reasons.append("pointer_strict_row_count_mismatch")
    dataset_sha = _sha256(dataset)
    if report.get("dataset_sha256") != dataset_sha:
        reasons.append("dataset_fingerprint_mismatch")
    if pointer is not None and pointer.get("dataset_sha256") != dataset_sha:
        reasons.append("pointer_dataset_fingerprint_mismatch")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        reasons.append("artifact_manifest_missing")
    else:
        for relative_path, expected_sha256 in artifacts.items():
            artifact = os.path.abspath(os.path.join(generation, relative_path))
            if os.path.commonpath([artifact, generation]) != generation:
                reasons.append(f"artifact_path_escape:{relative_path}")
            elif not os.path.isfile(artifact):
                reasons.append(f"artifact_missing:{relative_path}")
            elif _sha256(artifact) != expected_sha256:
                reasons.append(f"artifact_fingerprint_mismatch:{relative_path}")
    strict_rows = report.get("strict_full_rows")
    if not _finite(strict_rows) or int(strict_rows) < int(
        thresholds["minimum_strict_full_rows"]
    ):
        reasons.append("insufficient_strict_full_rows")

    features = report.get("features")
    if not isinstance(features, list) or not features:
        reasons.append("feature_schema_missing")
    else:
        forbidden_prefixes = (
            "result_", "thermal_", "conv_", "mesh_", "git_", "fail_",
            "P_", "B_", "T_", "Tprobe_", "Ltx", "Lrx", "Llt", "Llr",
            "Lmt", "Lmr", "task_",
        )
        forbidden_names = {"sample_weight", "source", "saved_at", "project_name"}
        leaked = [
            feature for feature in features
            if feature in forbidden_names
            or any(feature.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        if leaked:
            reasons.append(f"postsolve_feature_leakage:{','.join(leaked)}")

    minimum_coverage = float(thresholds["minimum_interval_coverage"])
    for target, limits in thresholds["targets"].items():
        target_reasons = []
        blocking_value = limits.get("blocking", True)
        blocking = isinstance(blocking_value, bool) and blocking_value
        if not isinstance(blocking_value, bool):
            target_reasons.append("invalid_threshold_config:blocking")
            blocking = True
        meta_path = os.path.join(generation, target, "meta.json")
        model_path = os.path.join(generation, target, "models.pkl")
        if not os.path.isfile(meta_path) or not os.path.isfile(model_path):
            target_reasons.append("missing_model_artifact")
            meta = {}
        else:
            try:
                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
            except Exception as exc:
                meta = {}
                target_reasons.append(f"invalid_meta:{exc}")
        if meta.get("training_run_id") != run_id:
            target_reasons.append("mixed_registry_generation")
        if meta.get("dataset_sha256") != dataset_sha:
            target_reasons.append("stale_dataset_fingerprint")
        if meta.get("features") != features:
            target_reasons.append("feature_schema_mismatch")
        metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
        coverage = metrics.get("interval_coverage")
        if not _finite(coverage) or float(coverage) < minimum_coverage:
            target_reasons.append("interval_coverage_below_minimum")
        for key, limit in limits.items():
            if key == "blocking":
                continue
            metric = key.removeprefix("min_").removeprefix("max_")
            value = metrics.get(metric)
            if not _finite(value):
                target_reasons.append(f"nonfinite_metric:{metric}")
            elif key.startswith("min_") and float(value) < float(limit):
                target_reasons.append(f"metric_below_minimum:{metric}")
            elif key.startswith("max_") and float(value) > float(limit):
                target_reasons.append(f"metric_above_maximum:{metric}")
        target_status[target] = {
            "passed": not target_reasons,
            "blocking": blocking,
            "reasons": target_reasons,
            "metrics": metrics,
        }
        qualified_reasons = [
            f"{target}:{reason}" for reason in target_reasons
        ]
        if blocking:
            reasons.extend(qualified_reasons)
        else:
            advisories.extend(qualified_reasons)

    if pointer is not None:
        gate_path = os.path.join(generation, QUALITY_GATE_FILENAME)
        try:
            with open(gate_path, encoding="utf-8") as handle:
                accepted = json.load(handle)
            gate_sha256 = _sha256(gate_path)
            if pointer.get("quality_gate_sha256") != gate_sha256:
                reasons.append("pointer_quality_gate_mismatch")
            if accepted.get("passed") is not True:
                reasons.append("active_generation_not_accepted")
            if accepted.get("training_run_id") != run_id:
                reasons.append("accepted_training_run_mismatch")
            if accepted.get("dataset_sha256") != dataset_sha:
                reasons.append("accepted_dataset_fingerprint_mismatch")
            if accepted.get("profile_sha256") != report.get("profile_sha256"):
                reasons.append("accepted_profile_fingerprint_mismatch")
            if accepted.get("thresholds_sha256") != thresholds_sha256:
                reasons.append("accepted_threshold_fingerprint_mismatch")
            if accepted.get("generation") != generation_relative:
                reasons.append("accepted_generation_path_mismatch")
            if accepted.get("generation_report_sha256") != report_sha256:
                reasons.append("accepted_generation_report_mismatch")
        except Exception as exc:
            reasons.append(f"acceptance_evidence_unavailable:{exc}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "advisories": advisories,
        "training_run_id": run_id,
        "dataset_sha256": dataset_sha,
        "profile_sha256": report.get("profile_sha256"),
        "thresholds_sha256": thresholds_sha256,
        "generation": generation_relative,
        "generation_report_sha256": report_sha256,
        "strict_full_rows": strict_rows,
        "targets": target_status,
        "manufacturing_tolerance_policy": (
            "excluded; exact-as-FEA geometry is assumed"
        ),
    }


def evaluate_generation(registry, generation, dataset, thresholds):
    """Evaluate an inactive candidate without consulting or changing current.json."""
    return evaluate_registry(
        registry, dataset, thresholds, generation=generation
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--status", default=None)
    parser.add_argument("--generation", default=None)
    args = parser.parse_args()

    with open(args.thresholds, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    result = evaluate_registry(
        args.registry, args.dataset, thresholds, generation=args.generation
    )
    result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    result["thresholds_path"] = os.path.abspath(args.thresholds)
    result["quality_thresholds_sha256"] = _sha256(args.thresholds)
    status_path = args.status or os.path.join(args.registry, "quality_status.json")
    _atomic_json(result, status_path)
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
