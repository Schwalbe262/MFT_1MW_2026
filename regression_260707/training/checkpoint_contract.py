"""Content-addressed identity for checkpoint-training contracts.

The solver and library revisions identify the rows that may enter training,
while these four inputs identify how those rows are validated and which model
targets are trained.  Launchers use the short key for directory names; the
orchestrator persists and verifies the full digest in its fail-closed state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
DEFAULT_PROFILE = REGRESSION_ROOT / "verify" / "profiles" / "standard.json"
DEFAULT_THRESHOLDS = HERE / "model_quality_thresholds.json"
DEFAULT_QUALITY_CONTRACT = REGRESSION_ROOT / "quality_contract.py"
DEFAULT_MODEL_TARGETS = REGRESSION_ROOT / "model_targets.py"
CONTRACT_SCHEMA_VERSION = 1
DEFAULT_KEY_LENGTH = 16


def _canonical_json_sha256(path: str | Path) -> str:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_contract_identity(
    profile: str | Path = DEFAULT_PROFILE,
    thresholds: str | Path = DEFAULT_THRESHOLDS,
    quality_contract: str | Path = DEFAULT_QUALITY_CONTRACT,
    model_targets: str | Path = DEFAULT_MODEL_TARGETS,
    *,
    key_length: int = DEFAULT_KEY_LENGTH,
) -> dict:
    """Return stable content hashes plus a compact directory-safe key."""
    if not 12 <= int(key_length) <= 64:
        raise ValueError("checkpoint contract key length must be between 12 and 64")
    components = {
        "profile_sha256": _canonical_json_sha256(profile),
        "thresholds_sha256": _canonical_json_sha256(thresholds),
        "quality_contract_sha256": _file_sha256(quality_contract),
        "model_targets_sha256": _file_sha256(model_targets),
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        **components,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        **payload,
        "checkpoint_contract_sha256": digest,
        "checkpoint_contract_key": digest[: int(key_length)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--quality-contract", default=str(DEFAULT_QUALITY_CONTRACT))
    parser.add_argument("--model-targets", default=str(DEFAULT_MODEL_TARGETS))
    parser.add_argument("--key-length", type=int, default=DEFAULT_KEY_LENGTH)
    parser.add_argument(
        "--json", action="store_true",
        help="print the full identity instead of only the directory key",
    )
    args = parser.parse_args()
    identity = checkpoint_contract_identity(
        args.profile,
        args.thresholds,
        args.quality_contract,
        args.model_targets,
        key_length=args.key_length,
    )
    if args.json:
        print(json.dumps(identity, sort_keys=True))
    else:
        print(identity["checkpoint_contract_key"])


if __name__ == "__main__":
    main()
