"""Fail-closed, single-use claim ledger for the final v2 outer holdout.

Claiming is deliberately separate from model evaluation.  A claim reserves the
sealed holdout exactly once; an interrupted evaluator must not delete or reuse
the claim.  Its immutable evaluation result belongs in a separate artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


RECEIPT_SCHEMA = "mft-production-blocker-hpo-result-v2"
LEDGER_SCHEMA = "mft-outer-acceptance-single-use-claim-v1"


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_final_stage_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get(
        "schema_version"
    ) != RECEIPT_SCHEMA:
        raise RuntimeError("outer acceptance receipt schema mismatch")
    fingerprint = receipt.get("sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
    if not isinstance(fingerprint, str) or fingerprint != _canonical_sha256(unsigned):
        raise RuntimeError("outer acceptance receipt fingerprint mismatch")
    metadata = receipt.get("metadata")
    policy = metadata.get("outer_acceptance_policy") if isinstance(
        metadata, dict
    ) else None
    if not isinstance(policy, dict):
        raise RuntimeError("outer acceptance policy is missing")
    selected = metadata.get("selected_cumulative_trials_per_job")
    configured_final = metadata.get("configured_final_trials_per_job")
    if (
        selected != configured_final
        or selected != policy.get("eligible_after_cumulative_trials")
        or policy.get("single_use_required") is not True
        or policy.get("eligible") is not True
    ):
        raise RuntimeError("outer acceptance is not eligible before the final stage")
    for key in (
        "production_eligible",
        "production_model_eligible",
        "fea_submission_approved",
        "promotion_approved",
    ):
        if metadata.get(key) is not False:
            raise RuntimeError(f"outer acceptance pre-gate flag mismatch: {key}")
    return receipt


def claim_outer_acceptance_once(receipt_path, ledger_path, consumer_id):
    """Atomically reserve one final-stage receipt for exactly one evaluation."""

    if not isinstance(consumer_id, str) or not consumer_id.strip():
        raise ValueError("outer acceptance consumer_id must be nonempty")
    receipt_path = Path(receipt_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    if not receipt_path.is_file():
        raise RuntimeError("outer acceptance receipt is unavailable")
    with receipt_path.open(encoding="utf-8") as handle:
        receipt = validate_final_stage_receipt(json.load(handle))

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": LEDGER_SCHEMA,
        "status": "claimed_no_result",
        "policy": "never delete, overwrite, or retry this outer holdout claim",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "consumer_id": consumer_id.strip(),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": _file_sha256(receipt_path),
        "receipt_canonical_sha256": receipt["sha256"],
        "params_sha256": receipt["params_sha256"],
        "outer_acceptance_holdouts_sha256": receipt["metadata"][
            "outer_acceptance_holdouts_sha256"
        ],
        "selected_cumulative_trials_per_job": receipt["metadata"][
            "selected_cumulative_trials_per_job"
        ],
    }
    sealed = {**document, "sha256": _canonical_sha256(document)}
    encoded = json.dumps(
        sealed, sort_keys=True, indent=1, ensure_ascii=False
    ).encode("utf-8")
    try:
        descriptor = os.open(
            ledger_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "outer acceptance was already claimed; preserve the existing ledger"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial claim is intentionally retained and blocks retry.  Removing
        # it would make a crash indistinguishable from an unused holdout.
        raise
    return sealed
