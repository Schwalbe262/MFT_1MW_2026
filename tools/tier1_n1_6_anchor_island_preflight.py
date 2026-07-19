"""Run and seal six local, no-Slurm fixed-N1=6 anchor-island preflights."""

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
from typing import Any
import uuid

try:
    from tier1_fixed_n1_6_anchor_islands_warm_handoff import (
        validate_handoff_contract,
    )
    from tier1_n1_6_anchor_island_contract import (
        ALL_THERMAL_SCALE_C,
        ANCHOR_ISLANDS,
        LLT_SCALE_UH,
        RESONANCE_SCALE_HZ,
        canonical_sha,
        island_profile,
        optimizer_allowance_contract,
    )
    from tier1_resonance_feedback import (
        CONSTRAINT_VERSION,
        EXPECTED_NSGA_REVISION,
        HARD_SPEC,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_fixed_n1_6_anchor_islands_warm_handoff import (
        validate_handoff_contract,
    )
    from tools.tier1_n1_6_anchor_island_contract import (
        ALL_THERMAL_SCALE_C,
        ANCHOR_ISLANDS,
        LLT_SCALE_UH,
        RESONANCE_SCALE_HZ,
        canonical_sha,
        island_profile,
        optimizer_allowance_contract,
    )
    from tools.tier1_resonance_feedback import (
        CONSTRAINT_VERSION,
        EXPECTED_NSGA_REVISION,
        HARD_SPEC,
    )


SCHEMA = "mft-tier1-fixed-n1-6-anchor-islands-preflight-v1"
RESULT_SCHEMA = "mft-tier1-corrected-search-seed-v1"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _result_summary(result: dict[str, Any], island) -> dict[str, Any]:
    names = result.get("constraint_names")
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise RuntimeError(f"{island.island_id} constraint schema is invalid")
    normalization = result.get("optimizer_constraint_normalization") or {}
    expected_allowance = optimizer_allowance_contract(
        names,
        resonance_allowance_hz=island.resonance_allowance_hz,
        llt_allowance_uh=island.llt_allowance_uh,
    )
    scales = normalization.get("scales") or {}
    thermal_names = [
        name for name in names
        if name.startswith("temperature_robust_limit:")
    ]
    physical_fields = (
        "constraint_minimum_G",
        "terminal_population_best_constraint_G",
        "optimizer_terminal_best_physical_constraint_G",
    )
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("constraint_version") != CONSTRAINT_VERSION
        or result.get("hard_spec") != HARD_SPEC
        or result.get("hard_spec_sha256") != canonical_sha(HARD_SPEC)
        or result.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or result.get("fixed_primary_turns") != 6
        or result.get("terminal_population_fixed_primary_turns_verified")
        is not True
        or result.get("terminal_population_primary_turn_values") != [6]
        or normalization.get("constraint_order") != names
        or normalization.get("authoritative_terminal_G")
        != "physical_unscaled"
        or scales.get("half_magnetizing_resonance_minimum")
        != RESONANCE_SCALE_HZ
        or scales.get("Llt_robust_band") != LLT_SCALE_UH
        or scales.get("Llt_ensemble_disagreement") != 2.0 * LLT_SCALE_UH
        or any(scales.get(name) != ALL_THERMAL_SCALE_C for name in thermal_names)
        or result.get("optimizer_resonance_allowance_Hz")
        != island.resonance_allowance_hz
        or result.get("optimizer_Llt_allowance_uH")
        != island.llt_allowance_uh
        or result.get("optimizer_constraint_allowance_contract")
        != expected_allowance
        or result.get("acquisition_ranking_contract") is not None
        or any(
            not isinstance(result.get(field), dict)
            or set(result[field]) != set(names)
            or any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in result[field].values()
            )
            for field in physical_fields
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
        raise RuntimeError(f"{island.island_id} preflight result seal mismatch")
    return {
        "island_id": island.island_id,
        "variant": island.variant,
        "island_profile": island_profile(island),
        "seed": int(result["seed"]),
        "population": int(result["population"]),
        "max_generations": int(result["max_generations"]),
        "completed_generations": int(result["completed_generations"]),
        "fixed_primary_turns_verified": True,
        "optimizer_allowance_contract_sha256": expected_allowance["sha256"],
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "soft_axis_pressure_enabled": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }


def validate_preflight(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    root = path.parent.resolve(strict=True)
    contract = _read_json(path)
    result_records = contract.get("results")
    if (
        contract.get("schema_version") != SCHEMA
        or contract.get("constraint_version") != CONSTRAINT_VERSION
        or contract.get("hard_spec") != HARD_SPEC
        or contract.get("hard_spec_sha256") != canonical_sha(HARD_SPEC)
        or contract.get("nsga_code_revision") != EXPECTED_NSGA_REVISION
        or contract.get("parallel_local_processes") is not True
        or contract.get("island_count") != len(ANCHOR_ISLANDS)
        or not isinstance(result_records, list)
        or len(result_records) != len(ANCHOR_ISLANDS)
        or any(
            contract.get(field) is not False
            for field in (
                "scheduler_write_performed", "slurm_submission_performed",
                "fea_submission_performed", "aedt_used",
            )
        )
    ):
        raise RuntimeError("anchor-island preflight top-level seal mismatch")
    expected_by_variant = {item.variant: item for item in ANCHOR_ISLANDS}
    seen = set()
    summaries = []
    for record in result_records:
        variant = record.get("variant")
        if variant in seen or variant not in expected_by_variant:
            raise RuntimeError("anchor-island preflight result identity mismatch")
        seen.add(variant)
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError("anchor-island preflight result path is unsafe")
        result_path = (root / relative).resolve(strict=True)
        if root not in result_path.parents:
            raise RuntimeError("anchor-island preflight result escaped root")
        if (
            _sha256(result_path) != record.get("sha256")
            or result_path.stat().st_size != record.get("size_bytes")
        ):
            raise RuntimeError("anchor-island preflight result bytes drifted")
        summary = _result_summary(
            _read_json(result_path), expected_by_variant[variant],
        )
        if summary != record.get("verified_summary"):
            raise RuntimeError("anchor-island preflight summary drifted")
        summaries.append(summary)
    if set(seen) != set(expected_by_variant):
        raise RuntimeError("anchor-island preflight island set is incomplete")
    if canonical_sha(summaries) != contract.get("verified_summaries_sha256"):
        raise RuntimeError("anchor-island preflight summary set SHA mismatch")
    return contract


def run_preflight(
    *, python: Path, feedback: Path, model_dir: Path, code_root: Path,
    warm_start: Path, output: Path, population: int = 64,
    max_generations: int = 1, inference_threads: int = 8,
) -> dict[str, Any]:
    python = python.resolve(strict=True)
    feedback = feedback.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    code_root = code_root.resolve(strict=True)
    warm_start = warm_start.resolve(strict=True)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("preflight output must be absent or empty")
    validate_handoff_contract(warm_start, code_root)
    output.mkdir(parents=True, exist_ok=True)
    warm_sha = _sha256(warm_start)
    processes = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child_environment = os.environ.copy()
    git_config_count = int(child_environment.get("GIT_CONFIG_COUNT", "0"))
    child_environment[f"GIT_CONFIG_KEY_{git_config_count}"] = "safe.directory"
    child_environment[f"GIT_CONFIG_VALUE_{git_config_count}"] = (
        code_root.as_posix()
    )
    child_environment["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        child_environment[variable] = str(int(inference_threads))
    for island in ANCHOR_ISLANDS:
        island_output = output / "results" / island.island_id
        island_output.mkdir(parents=True, exist_ok=False)
        command = [
            str(python), str(feedback), "search-seed",
            "--model-dir", str(model_dir),
            "--nsga-code-root", str(code_root),
            "--output", str(island_output),
            "--seed", str(island.seed_start - 1),
            "--population", str(int(population)),
            "--max-generations", str(int(max_generations)),
            "--warm-start", str(warm_start),
            "--warm-start-sha256", warm_sha,
            "--inference-threads", str(int(inference_threads)),
            "--optimizer-resonance-scale-hz", str(RESONANCE_SCALE_HZ),
            "--optimizer-llt-scale-uh", str(LLT_SCALE_UH),
            "--optimizer-all-thermal-scale-c", str(ALL_THERMAL_SCALE_C),
            "--optimizer-resonance-allowance-hz",
            str(island.resonance_allowance_hz),
            "--optimizer-llt-allowance-uh", str(island.llt_allowance_uh),
            "--fixed-primary-turns", "6",
        ]
        stdout = (island_output / "stdout.log").open("wb")
        stderr = (island_output / "stderr.log").open("wb")
        process = subprocess.Popen(
            command, cwd=code_root, stdout=stdout, stderr=stderr,
            stdin=subprocess.DEVNULL, creationflags=creationflags,
            env=child_environment,
        )
        processes.append((island, island_output, command, process, stdout, stderr))
    failures = []
    for island, island_output, command, process, stdout, stderr in processes:
        returncode = process.wait()
        stdout.close()
        stderr.close()
        if returncode != 0:
            failures.append({
                "island_id": island.island_id,
                "returncode": returncode,
                "command": command,
                "stderr": str(island_output / "stderr.log"),
            })
    if failures:
        raise RuntimeError(f"anchor-island local preflight failed: {failures}")
    records = []
    summaries = []
    for island, island_output, _command, _process, _stdout, _stderr in processes:
        result_path = island_output / "result.json"
        summary = _result_summary(_read_json(result_path), island)
        summaries.append(summary)
        records.append({
            "variant": island.variant,
            "path": result_path.relative_to(output).as_posix(),
            "sha256": _sha256(result_path),
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
        "feedback_script_sha256": _sha256(feedback),
        "model_manifest_sha256": _sha256(model_dir / "manifest.json"),
        "warm_start_sha256": warm_sha,
        "island_count": len(ANCHOR_ISLANDS),
        "parallel_local_processes": True,
        "results": records,
        "verified_summaries_sha256": canonical_sha(summaries),
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    contract_path = output / "preflight_contract.json"
    _atomic_json(contract_path, contract)
    return validate_preflight(contract_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--feedback", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--code-root", type=Path, required=True)
    run.add_argument("--warm-start", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--population", type=int, default=64)
    run.add_argument("--max-generations", type=int, default=1)
    run.add_argument("--inference-threads", type=int, default=8)
    validate = commands.add_parser("validate")
    validate.add_argument("--contract", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        value = run_preflight(
            python=args.python, feedback=args.feedback,
            model_dir=args.model_dir, code_root=args.code_root,
            warm_start=args.warm_start, output=args.output,
            population=args.population, max_generations=args.max_generations,
            inference_threads=args.inference_threads,
        )
    else:
        value = validate_preflight(args.contract)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
