"""Render authenticated 1/4/8-lane Phase-B remote-smoke tasks.

This command is deliberately dry: it reads an immutable launch plan, bundle
manifest, and READY seal and emits Scheduler task envelopes to stdout.  It has
no HTTP client and cannot submit, cancel, or restart work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_slurm_bundle import (
        READY_SCHEMA,
        validate_required_runtime_code,
    )
    from tier1_final1000_multiseed_contract import canonical_sha256, json_bytes
    from tier1_final1000_multiseed_phase_b_contract import (
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
        build_concurrent_batch_task,
        validate_batch_task,
    )
    from tier1_final1000_slurm_controller import _refill_task, _task_templates
    from tier1_final1000_slurm_launch import validate_launch_plan
    from tier1_final1000_stage_profiles import BY_ID, STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_slurm_bundle import (
        READY_SCHEMA,
        validate_required_runtime_code,
    )
    from tools.tier1_final1000_multiseed_contract import canonical_sha256, json_bytes
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
        build_concurrent_batch_task,
        validate_batch_task,
    )
    from tools.tier1_final1000_slurm_controller import _refill_task, _task_templates
    from tools.tier1_final1000_slurm_launch import validate_launch_plan
    from tools.tier1_final1000_stage_profiles import BY_ID, STAGES


SMOKE_RELEASE_SCHEMA = "mft-tier1-final1000-phase-b-remote-smoke-release-v1"
SMOKE_SHAPES = (1, 4, 8)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a SHA-256")
    return value


def _validate_ready(
    ready: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> dict[str, Any]:
    files = manifest.get("files")
    runtime = manifest.get("runtime")
    if (
        not isinstance(files, dict)
        or not isinstance(runtime, dict)
        or ready.get("schema_version") != READY_SCHEMA
        or ready.get("bundle_id") != manifest.get("bundle_id")
        or ready.get("bundle_manifest_sha256") != manifest_file_sha256
        or ready.get("contract_sha256") != manifest.get("contract_sha256")
        or ready.get("file_count") != len(files)
        or ready.get("byte_count")
        != sum(int(record["size"]) for record in files.values())
        or ready.get("every_file_sha256_verified") is not True
        or ready.get("all_file_sha256_verified") is not True
        or ready.get("runtime_verified") is not True
        or ready.get("runtime_packages") != runtime.get("critical_packages")
        or ready.get("code_inventory_sha256") != manifest.get("code_inventory_sha256")
        or ready.get("relocation_contract_sha256")
        != (manifest.get("relocation") or {}).get("contract_sha256")
        or ready.get("remote_git_checkout_performed") is not False
        or ready.get("artifacts_read_only") is not True
        or ready.get("runs_mode") != "1777"
        or not isinstance(ready.get("published_at"), str)
        or not ready["published_at"]
    ):
        raise RuntimeError("Phase-B smoke READY seal mismatch")
    return copy.deepcopy(dict(ready))


def _next_seed(plan: Mapping[str, Any], stage_id: str) -> int:
    seeds = [
        int(task["payload_json"]["seed"])
        for wave in ("canaries", "ramp")
        for task in plan["task_waves"][wave]
        if task["payload_json"]["final_goal_stage_id"] == stage_id
    ]
    if not seeds:
        raise RuntimeError("smoke stage has no authenticated task template")
    return max(seeds) + 1


def build_smoke_release(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ready: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
    stage_id: str | None = None,
    seed_start: int | None = None,
) -> dict[str, Any]:
    """Build dry Scheduler envelopes for one-, four-, and eight-lane smoke."""

    source = validate_launch_plan(plan)
    manifest_sha = _sha256(manifest_file_sha256, "bundle manifest hash")
    runtime_attestation = validate_required_runtime_code(
        manifest,
        REMOTE_CODE_FILES,
        required_code_sha256={
            str(INFERENCE_SAFETY["required_helper_path"]): str(
                INFERENCE_SAFETY["required_helper_sha256"]
            )
        },
    )
    ready_seal = _validate_ready(ready, manifest, manifest_sha)
    selected_stage = stage_id or STAGES[0].stage_id
    stage = BY_ID.get(selected_stage)
    if stage is None:
        raise RuntimeError("unknown Phase-B smoke stage")
    templates = _task_templates(source)
    template = templates[selected_stage]
    if (
        template["payload_json"].get("bundle_id") != manifest.get("bundle_id")
        or template["payload_json"].get("bundle_manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("smoke launch plan and immutable bundle disagree")
    first_seed = (
        _next_seed(source, selected_stage) if seed_start is None else seed_start
    )
    if isinstance(first_seed, bool) or not isinstance(first_seed, int):
        raise RuntimeError("smoke seed start must be an integer")
    logical_count = sum(SMOKE_SHAPES)
    if not (
        stage.seed_start <= first_seed
        and first_seed + logical_count <= stage.seed_window_end_exclusive
    ):
        raise RuntimeError("smoke seeds escape the authenticated stage window")

    tasks: list[dict[str, Any]] = []
    cursor = first_seed
    for shape in SMOKE_SHAPES:
        children = [
            _refill_task(
                template,
                stage_id=selected_stage,
                seed=cursor + offset,
                wave="refill",
            )
            for offset in range(shape)
        ]
        task = validate_batch_task(build_concurrent_batch_task(children))
        tasks.append(task)
        cursor += shape

    logical_children = [
        child for task in tasks for child in task["payload_json"]["children"]
    ]
    seeds = [int(child["seed"]) for child in logical_children]
    dedupes = [str(child["logical_dedupe_key"]) for child in logical_children]
    if len(seeds) != len(set(seeds)) or len(dedupes) != len(set(dedupes)):
        raise RuntimeError("Phase-B smoke release duplicates logical work")
    unsigned = {
        "schema_version": SMOKE_RELEASE_SCHEMA,
        "source_launch_plan_sha256": source["launch_plan_sha256"],
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": manifest_sha,
        "required_runtime_code_sha256": runtime_attestation["sha256"],
        "ready_sha256": canonical_sha256(ready_seal),
        "stage_id": selected_stage,
        "seed_start": first_seed,
        "seed_end": cursor - 1,
        "lane_shapes": list(SMOKE_SHAPES),
        "logical_seed_count": logical_count,
        "tasks": tasks,
        "task_inventory_sha256": canonical_sha256({"tasks": tasks}),
        "diagnostic_execution_order": [1, 4, 8],
        "production_gate_order": [4, 8],
        "four_lane_is_mandatory_first_production_canary": True,
        "gates": {
            "failure_count": 0,
            "semlock_or_enospc_count": 0,
            "affinity_escape_count": 0,
            "sibling_survival_required": True,
            "maximum_child_peak_rss_bytes": 22 * 1024**3,
            "maximum_dispatch_fill_seconds": 35.0,
            "maximum_p50_wall_regression_fraction": 0.10,
            "maximum_p95_wall_regression_fraction": 0.20,
            "maximum_cpu_hours_per_seed_regression_fraction": 0.20,
            "minimum_8_vs_4_throughput_ratio": 1.70,
            "minimum_overall_throughput_improvement_fraction": 0.10,
        },
        "explicit_mutation_authority_required": True,
        "smoke_ready": True,
        "production_eligible": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "cancellation_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "release_sha256": canonical_sha256(unsigned)}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--stage-id")
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args(argv)
    manifest_bytes = args.bundle_manifest.read_bytes()
    result = build_smoke_release(
        _read_object(args.launch_plan, "launch plan"),
        _read_object(args.bundle_manifest, "bundle manifest"),
        _read_object(args.ready, "READY seal"),
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        stage_id=args.stage_id,
        seed_start=args.seed_start,
    )
    sys.stdout.buffer.write(json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
