"""Seal the exact H390 winner's electrical and thermal verdicts.

This combines two independent read-only authorities:

* the high-accuracy Tx/Rx graded-capacitance pair; and
* the terminal exact-gap full-physics 1/8 FEA collection.

The original and latest relaxed thermal limits are intentionally evaluated
separately so a relaxed pass can never be reported as an original-limit pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


OUTPUT_SCHEMA = "mft-goal-h390-final-winner-authority-v1"
FEA_SCHEMA = "mft-goal-core-rescue-fea-collection-v1"
PAIR_SCHEMA = "mft-goal-h390-final-txrx-cap-collection-v1"

STRICT_LIMITS = {
    "Tx_C": 100.0,
    "Rx_C": 100.0,
    "core_C": 120.0,
}
RELAXED_LIMITS = {
    "Tx_C": 110.0,
    "Rx_C": 130.0,
    "core_C": 130.0,
}


class AuthorityError(RuntimeError):
    """Raised when an authority source is incomplete or inconsistent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sealed(path: Path, schema: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuthorityError(f"JSON object required: {path}")
    if value.get("schema_version") != schema:
        raise AuthorityError(f"schema mismatch: {path}")
    expected = value.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AuthorityError(f"payload seal missing: {path}")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    if _sha(payload) != expected:
        raise AuthorityError(f"payload seal mismatch: {path}")
    return value


def _finite(record: Mapping[str, Any], key: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AuthorityError(f"finite {key} is required") from exc
    if not math.isfinite(value):
        raise AuthorityError(f"finite {key} is required")
    return value


def _thermal_tier(
    temperatures: Mapping[str, float], limits: Mapping[str, float]
) -> dict[str, Any]:
    margins = {
        name: float(limits[name]) - float(temperatures[name])
        for name in ("Tx_C", "Rx_C", "core_C")
    }
    component_pass = {
        name: margin >= 0.0 for name, margin in margins.items()
    }
    return {
        "limits": dict(limits),
        "temperatures": dict(temperatures),
        "margins_C": margins,
        "component_pass": component_pass,
        "pass": all(component_pass.values()),
        "minimum_margin_C": min(margins.values()),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_authority(
    *,
    fea_collection_path: Path,
    electrical_pair_path: Path,
    params_path: Path,
) -> dict[str, Any]:
    fea_path = fea_collection_path.resolve(strict=True)
    pair_path = electrical_pair_path.resolve(strict=True)
    final_params_path = params_path.resolve(strict=True)
    fea = _load_sealed(fea_path, FEA_SCHEMA)
    pair = _load_sealed(pair_path, PAIR_SCHEMA)

    if fea.get("complete") is not True:
        raise AuthorityError("exact full-physics FEA collection is incomplete")
    records = fea.get("ranked_records")
    if not isinstance(records, list) or len(records) != 1:
        raise AuthorityError("exactly one exact winner FEA record is required")
    record = records[0]
    if not isinstance(record, dict):
        raise AuthorityError("winner FEA record must be an object")
    if record.get("mode") != "matrix_turngraded_cap_loss_thermal":
        raise AuthorityError("winner record is not full-physics")
    if record.get("exact_gap_task") is not True:
        raise AuthorityError("winner record is not the exact interpolated gap")

    candidate_sha = str(record.get("candidate_sha256") or "")
    if len(candidate_sha) != 64:
        raise AuthorityError("winner candidate SHA-256 is missing")
    if pair.get("candidate_physics_sha256") != candidate_sha:
        raise AuthorityError("electrical/thermal candidate identity mismatch")
    if pair.get("all_contracts_passed") is not True:
        raise AuthorityError("electrical pair contracts did not all pass")

    temperatures = {
        "Tx_C": _finite(record, "T_max_Tx_C"),
        "Rx_C": _finite(record, "T_max_Rx_C"),
        "core_C": _finite(record, "T_max_core_C"),
    }
    strict = _thermal_tier(temperatures, STRICT_LIMITS)
    relaxed = _thermal_tier(temperatures, RELAXED_LIMITS)

    common_checks = {
        "candidate_identity_match": True,
        "full_physics_contract_valid": record.get("contract_valid") is True,
        "geometry_pass": record.get("geometry_pass") is True,
        "exact_Lm_pass": record.get("exact_Lm_pass") is True,
        "electrical_pair_contracts_pass": True,
        "electrical_pair_resonance_pass_15kHz": (
            pair.get("pair_resonance_pass_15kHz") is True
        ),
        "electrical_pair_minimum_resonance_consistent": (
            _finite(pair, "minimum_resonance_Hz") >= 15000.0
        ),
        "collector_relaxed_thermal_flag_consistent": (
            record.get("thermal_pass") is relaxed["pass"]
        ),
    }
    common_pass = all(common_checks.values())

    package: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_physics_sha256": candidate_sha,
        "task_id": int(record["task_id"]),
        "source_authorities": {
            "exact_full_physics_fea": {
                "path": str(fea_path),
                "file_sha256": _file_sha(fea_path),
                "payload_sha256": fea["payload_sha256"],
                "result_json_sha256": record.get("result_json_sha256"),
                "stdout_sha256": record.get("stdout_sha256"),
            },
            "high_accuracy_txrx_cap_pair": {
                "path": str(pair_path),
                "file_sha256": _file_sha(pair_path),
                "payload_sha256": pair["payload_sha256"],
                "physical_identity_sha256": pair.get(
                    "physical_identity_sha256"
                ),
            },
            "final_params": {
                "path": str(final_params_path),
                "file_sha256": _file_sha(final_params_path),
            },
        },
        "design": {
            "W_mm": _finite(record, "W_mm"),
            "L_mm": _finite(record, "L_mm"),
            "H_mm": _finite(record, "H_mm"),
            "N1": int(record["N1"]),
            "N2": int(record["N2"]),
            "N2_main": int(record["N2_main"]),
            "N2_side": int(record["N2_side"]),
            "gap_mm": _finite(record, "gap_mm"),
            "Lm_primary_full_mH": _finite(
                record, "Lm_primary_full_mH"
            ),
            "Lm_primary_relative_error": _finite(
                record, "Lm_primary_relative_error"
            ),
        },
        "electrical_high_accuracy": {
            "minimum_resonance_Hz": _finite(
                pair, "minimum_resonance_Hz"
            ),
            "limiting_winding": (
                "Rx"
                if float(pair["rx"]["resonance_Hz"])
                <= float(pair["tx"]["resonance_Hz"])
                else "Tx"
            ),
            "Tx": {
                "capacitance_F": _finite(
                    pair["tx"], "terminal_capacitance_F"
                ),
                "self_inductance_H": _finite(
                    pair["tx"], "self_inductance_H"
                ),
                "resonance_Hz": _finite(pair["tx"], "resonance_Hz"),
                "task_id": int(pair["tx"]["task_id"]),
            },
            "Rx": {
                "capacitance_F": _finite(
                    pair["rx"], "terminal_capacitance_F"
                ),
                "self_inductance_H": _finite(
                    pair["rx"], "self_inductance_H"
                ),
                "resonance_Hz": _finite(pair["rx"], "resonance_Hz"),
                "task_id": int(pair["rx"]["task_id"]),
            },
        },
        "losses_W": {
            "Tx": _finite(record, "Tx_loss_W"),
            "Rx": _finite(record, "Rx_loss_W"),
            "core": _finite(record, "core_loss_W"),
        },
        "common_checks": common_checks,
        "common_pass": common_pass,
        "thermal_verdicts": {
            "original_strict_Tx100_Rx100_core120": strict,
            "latest_relaxed_Tx110_Rx130_core130": relaxed,
        },
        "final_pass_original_strict": common_pass and strict["pass"],
        "final_pass_latest_relaxed": common_pass and relaxed["pass"],
        "scheduler_mutation_performed": False,
    }
    package["payload_sha256"] = _sha(package)
    return package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fea-collection", type=Path, required=True)
    parser.add_argument("--electrical-pair", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package = build_authority(
        fea_collection_path=args.fea_collection,
        electrical_pair_path=args.electrical_pair,
        params_path=args.params,
    )
    output = args.output.resolve()
    _atomic_json(output, package)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
