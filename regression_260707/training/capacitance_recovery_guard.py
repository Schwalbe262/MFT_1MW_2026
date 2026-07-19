"""Fail-closed provenance checks for recovered capacitance model artifacts.

Training remains allowed to build an inactive candidate without this contract.
Quality acceptance and registry promotion, however, must independently prove
that every target artifact belongs to the same recovered dataset/profile
identity.  This keeps legacy quantized candidates available for forensics while
making them ineligible for production activation.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from collections.abc import Mapping


CAPACITANCE_RECOVERY_CONTRACT = "mft-capacitance-lc-inverse-v1"
CAPACITANCE_RECOVERY_MAX_ABS_DELTA_F = 5.1e-11


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _positive_integer(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric != parsed or parsed <= 0:
        return None
    return parsed


def _bounded_delta(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(parsed)
        or parsed < 0.0
        or parsed > CAPACITANCE_RECOVERY_MAX_ABS_DELTA_F
    ):
        return None
    return parsed


def _recovery_identity(value, location):
    """Return a normalized identity and deterministic validation reasons."""
    reasons = []
    if not isinstance(value, Mapping):
        return None, [f"{location}:evidence_missing"]

    contract = value.get("contract")
    if contract != CAPACITANCE_RECOVERY_CONTRACT:
        reasons.append(f"{location}:contract_mismatch")

    recovered_row_count = _positive_integer(value.get("recovered_row_count"))
    if recovered_row_count is None:
        reasons.append(f"{location}:recovered_row_count_not_positive")

    max_delta = _bounded_delta(value.get("max_observed_abs_delta_F"))
    if max_delta is None:
        reasons.append(f"{location}:max_observed_abs_delta_F_out_of_bounds")

    if reasons:
        return None, reasons
    return {
        "contract": contract,
        "recovered_row_count": recovered_row_count,
        "max_observed_abs_delta_F": max_delta,
    }, []


def _artifact_path(generation, target, filename):
    generation = os.path.abspath(generation)
    artifact = os.path.abspath(os.path.join(generation, target, filename))
    try:
        if os.path.commonpath([artifact, generation]) != generation:
            return None
    except ValueError:
        return None
    return artifact


def validate_generation_capacitance_recovery(
    generation,
    report,
    *,
    dataset_sha256=None,
    profile_sha256=None,
):
    """Validate recovery provenance across report, meta, and model bundles.

    ``dataset_sha256`` and ``profile_sha256`` are independent caller-side
    identities when supplied.  Omitting them only derives the expected values
    from the report; both report identities still have to be non-empty.
    """
    reasons = []
    target_status = {}
    if not isinstance(report, Mapping):
        return {
            "schema_version": 1,
            "passed": False,
            "reasons": ["train_report_unavailable"],
            "targets": {},
        }

    report_dataset = report.get("dataset_sha256")
    report_profile = report.get("profile_sha256")
    if not _nonempty_string(report_dataset):
        reasons.append("train_report:dataset_identity_missing")
    if not _nonempty_string(report_profile):
        reasons.append("train_report:profile_identity_missing")
    if dataset_sha256 is not None and report_dataset != dataset_sha256:
        reasons.append("train_report:dataset_identity_mismatch")
    if profile_sha256 is not None and report_profile != profile_sha256:
        reasons.append("train_report:profile_identity_mismatch")

    report_recovery, evidence_reasons = _recovery_identity(
        report.get("capacitance_recovery"), "train_report"
    )
    reasons.extend(evidence_reasons)

    targets = report.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not _nonempty_string(target) for target in targets)
        or len(set(targets)) != len(targets)
    ):
        reasons.append("train_report:target_manifest_invalid")
        targets = []
    declared_targets = set(targets)

    target_reports = report.get("report")
    if not isinstance(target_reports, Mapping):
        reasons.append("train_report:target_metrics_manifest_missing")
    elif targets and set(target_reports) != set(targets):
        reasons.append("train_report:target_metrics_manifest_mismatch")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        reasons.append("train_report:artifact_manifest_missing")
        artifacts = {}

    artifact_targets = set()
    for relative in artifacts:
        normalized = os.fspath(relative).replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) == 2 and parts[1] in {"meta.json", "models.pkl"}:
            artifact_targets.add(parts[0])
    if artifact_targets != declared_targets:
        reasons.append("train_report:artifact_target_manifest_mismatch")

    actual_targets = set()
    try:
        for name in os.listdir(generation):
            path = os.path.join(generation, name)
            if os.path.isdir(path) and (
                os.path.isfile(os.path.join(path, "meta.json"))
                or os.path.isfile(os.path.join(path, "models.pkl"))
            ):
                actual_targets.add(name)
    except OSError:
        reasons.append("generation:target_inventory_unavailable")
    if actual_targets != declared_targets:
        reasons.append("generation:target_inventory_mismatch")

    run_id = report.get("training_run_id")
    targets_to_validate = sorted(
        declared_targets | artifact_targets | actual_targets
    )
    for target in targets_to_validate:
        target_reasons = []
        expected_artifacts = (
            f"{target}/meta.json",
            f"{target}/models.pkl",
        )
        for relative in expected_artifacts:
            if not _nonempty_string(artifacts.get(relative)):
                target_reasons.append(f"artifact_manifest_missing:{relative}")

        meta_path = _artifact_path(generation, target, "meta.json")
        model_path = _artifact_path(generation, target, "models.pkl")
        if meta_path is None or model_path is None:
            target_reasons.append("artifact_path_escape")
            meta = None
            bundle = None
        else:
            try:
                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
                if not isinstance(meta, Mapping):
                    raise TypeError("metadata is not an object")
            except Exception as exc:
                meta = None
                target_reasons.append(
                    f"meta_unavailable:{type(exc).__name__}"
                )
            try:
                with open(model_path, "rb") as handle:
                    bundle = pickle.load(handle)
                if not isinstance(bundle, Mapping):
                    raise TypeError("model bundle is not a mapping")
            except Exception as exc:
                bundle = None
                target_reasons.append(
                    f"model_bundle_unavailable:{type(exc).__name__}"
                )

        for artifact_name, payload in (("meta", meta), ("model_bundle", bundle)):
            if payload is None:
                continue
            if payload.get("training_run_id") != run_id:
                target_reasons.append(f"{artifact_name}:training_run_mismatch")
            if payload.get("dataset_sha256") != report_dataset:
                target_reasons.append(f"{artifact_name}:dataset_identity_mismatch")
            if payload.get("profile_sha256") != report_profile:
                target_reasons.append(f"{artifact_name}:profile_identity_mismatch")
            identity, identity_reasons = _recovery_identity(
                payload.get("capacitance_recovery"), artifact_name
            )
            target_reasons.extend(identity_reasons)
            if (
                identity is not None
                and report_recovery is not None
                and identity != report_recovery
            ):
                target_reasons.append(
                    f"{artifact_name}:recovery_identity_mismatch"
                )

        target_status[target] = {
            "passed": not target_reasons,
            "reasons": target_reasons,
        }
        reasons.extend(f"{target}:{reason}" for reason in target_reasons)

    identity = report_recovery or {
        "contract": None,
        "recovered_row_count": None,
        "max_observed_abs_delta_F": None,
    }
    return {
        "schema_version": 1,
        "passed": not reasons,
        "reasons": reasons,
        **identity,
        "dataset_sha256": report_dataset,
        "profile_sha256": report_profile,
        "targets": target_status,
    }
