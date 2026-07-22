"""Preflight and render the open-ended surrogate-only final1000 Slurm wave.

The renderer consumes four independently published, content-addressed
current7 bundles (one exact dynamic hard spec per final1000 stage).  It
authenticates every local source and publication/READY receipt before it can
render scheduler envelopes.  The CLI intentionally has no submit/apply/watch
surface: its only write is the requested local JSON launch plan.

The topology successor keeps 500 logical seed tasks active
(300/150/40/10 by stage) until an explicit stop is requested by a separate
authorized controller.  No task
uses AEDT or FEA; ``aedt_backend=standalone`` is retained solely because it is
a required scheduler API envelope field and does not reserve an AEDT session.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
        sha256_file,
    )
    from tier1_corrected_current7_slurm_publish import (
        load_bundle_plan,
        validate_publication_receipt,
    )
    from tier1_final1000_stage_profiles import (
        BY_ID,
        FIXED_GENERATIONS,
        FIXED_PRIMARY_TURNS,
        INFERENCE_THREADS,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        FinalGoalStage,
        hard_spec,
        stage_inventory,
        stage_profile,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
        sha256_file,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        load_bundle_plan,
        validate_publication_receipt,
    )
    from tools.tier1_final1000_stage_profiles import (
        BY_ID,
        FIXED_GENERATIONS,
        FIXED_PRIMARY_TURNS,
        INFERENCE_THREADS,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        FinalGoalStage,
        hard_spec,
        stage_inventory,
        stage_profile,
    )


BINDINGS_SCHEMA = "mft-tier1-final1000-stage-bundle-bindings-v1"
LAUNCH_SCHEMA = "mft-tier1-final1000-slurm-launch-plan-v1"
RESULT_PREFLIGHT_SCHEMA = "mft-tier1-final1000-result-preflight-v1"
TOPOLOGY_CANARY_LAUNCH_SCHEMA = (
    "mft-tier1-final1000-topology-pre-cutover-canary-plan-v1"
)
CURRENT7_RESULT_SCHEMA = "mft-tier1-current7-search-seed-v1"
CURRENT7_REPLAY_SCHEMA = "mft-tier1-current7-terminal-replay-v1"

DEFAULT_CPUS = 4
DEFAULT_MEMORY_MB = 28 * 1024
# ``max_workers_per_node`` is an allocation-level cap for ``standard`` tasks
# in the deployed scheduler (the name is historical).  CPU and memory fit are
# still hard gates.  Thirty-two preserves the already-proven 4c/28GiB Current7
# production identity (2,606/2,606 completed) without reducing a 64-core
# allocation to the eight-worker cap used by the predecessor release.
DEFAULT_MAX_WORKERS_PER_NODE = 32
# This active final-goal search should be admitted ahead of the older
# Current7 priority-0 refills while staying inside the user's normal 0--9
# simulation priority band.  It does not preempt already-running work.
DEFAULT_PRIORITY = 1
DEFAULT_TIMEOUT_SECONDS = 86_400
DEFAULT_PEAK_RSS_GATE_BYTES = 22 * 1024**3

# Preserve the 2026-07-22 running release as an authenticated predecessor
# policy.  The evidence-driven topology successor below shifts additional
# capacity to the entry N36/N37 transition after 11,555 terminal completions
# still yielded zero physically feasible rows.
PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS = {
    "entry-1200-t125": 200,
    "bridge-1150-t115": 160,
    "close-1075-t107p5": 90,
    "final-1000-t100": 50,
}
SUCCESSOR_ACTIVE_QUOTAS = {
    "entry-1200-t125": 300,
    "bridge-1150-t115": 150,
    "close-1075-t107p5": 40,
    "final-1000-t100": 10,
}
if (
    set(PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS) != set(SUCCESSOR_ACTIVE_QUOTAS)
    or sum(PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS.values()) != TOTAL_ACTIVE_QUOTA
    or sum(SUCCESSOR_ACTIVE_QUOTAS.values()) != TOTAL_ACTIVE_QUOTA
):  # pragma: no cover
    raise RuntimeError("final1000 successor active quotas must sum to 500")

REQUIRED_SCHEDULER_FIELDS = frozenset(
    {
        "name",
        "remote_cwd",
        "command",
        "payload_json",
        "required_capability",
        "env_profile",
        "cpus",
        "memory_mb",
        "scheduling_profile",
        "aedt_backend",
        "gpus",
        "priority",
        "timeout_seconds",
        "dedupe_key",
        "max_workers_per_node",
    }
)
REQUIRED_REMOTE_CODE = frozenset(
    {
        "artifacts/code/tools/tier1_corrected_generation_preflight.py",
        "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py",
        "artifacts/code/tools/tier1_deep_crossover_contract.py",
        "artifacts/code/tools/tier1_final1000_topology_niche_contract.py",
        "artifacts/code/tools/tier1_final1000_stage_profiles.py",
    }
)
PAYLOAD_SHA_PATTERN = re.compile(
    r"(--payload-sha256\s+)([0-9a-f]{64})(?=\s*$)", re.MULTILINE
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sealed_mapping(value: Any) -> bool:
    if not isinstance(value, dict) or not _is_sha256(value.get("sha256")):
        return False
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return value["sha256"] == canonical_sha256(unsigned)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _optimizer_contract() -> tuple[Any, Any]:
    """Load the dynamic evaluator contract lazily to avoid circular imports."""

    try:
        from tier1_corrected_generation_preflight import (
            stage_constraint_names,
            validate_stage_spec,
        )
    except ImportError:  # pragma: no cover - repository import path
        from tools.tier1_corrected_generation_preflight import (
            stage_constraint_names,
            validate_stage_spec,
        )
    return validate_stage_spec, stage_constraint_names


def optimizer_stage_contract(
    stage: FinalGoalStage,
) -> tuple[dict[str, Any], list[str]]:
    validate_stage_spec, stage_constraint_names = _optimizer_contract()
    expected = hard_spec(stage)
    normalized = validate_stage_spec(expected)
    if normalized != expected:
        raise RuntimeError(
            f"{stage.stage_id} profile is not canonical under optimizer validation"
        )
    constraints = list(stage_constraint_names(normalized))
    if (
        "half_magnetizing_resonance_minimum" not in constraints
        or "half_magnetizing_resonance_maximum" not in constraints
        or len(constraints) != len(set(constraints))
    ):
        raise RuntimeError("optimizer omitted the sealed resonance-band constraints")
    return normalized, constraints


def logical_lanes() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one canary per stage plus the remaining production ramp lanes."""

    canaries: list[dict[str, Any]] = []
    ramp: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for stage in STAGES:
        quota = SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
        if stage.seed_start + quota > stage.seed_window_end_exclusive:
            raise RuntimeError(f"{stage.stage_id} active quota escapes its seed window")
        for offset in range(quota):
            seed = stage.seed_start + offset
            if seed in occupied:
                raise RuntimeError("final1000 logical seed allocation overlaps")
            occupied.add(seed)
            lane = {
                "stage_id": stage.stage_id,
                "variant": stage.variant,
                "seed": seed,
                "fixed_primary_turns": FIXED_PRIMARY_TURNS,
                "wave": "canary" if offset == 0 else "ramp",
            }
            (canaries if offset == 0 else ramp).append(lane)
    if (
        len(canaries) != len(STAGES)
        or len(ramp) != TOTAL_ACTIVE_QUOTA - len(STAGES)
        or len(occupied) != TOTAL_ACTIVE_QUOTA
    ):
        raise RuntimeError("final1000 launch shape is not 4 + 496")
    return canaries, ramp


def _resolve_from(root: Path, raw: Any, label: str) -> Path:
    value = str(raw or "")
    if not value:
        raise RuntimeError(f"missing {label}")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"unavailable {label}: {path}") from exc


def _validate_local_inventory(
    manifest: Mapping[str, Any], source_map: Mapping[str, Path]
) -> None:
    files = manifest.get("files") or {}
    if set(files) != set(source_map):
        raise RuntimeError("bundle source map differs from sealed file inventory")
    for relative, record in files.items():
        source = source_map[relative]
        if (
            not source.is_file()
            or source.stat().st_size != int(record.get("size", -1))
            or sha256_file(source) != record.get("sha256")
        ):
            raise RuntimeError(f"bundle source changed after plan: {relative}")


def _selected_base_island(
    manifest: Mapping[str, Any], stage: FinalGoalStage
) -> tuple[str, dict[str, Any]]:
    matches = []
    refill_windows = (manifest.get("open_ended_refill") or {}).get("seed_windows") or {}
    for island_id, island in (manifest.get("islands") or {}).items():
        profile = (island or {}).get("current7_profile") or {}
        window = refill_windows.get(island_id) or {}
        if (
            profile.get("fixed_primary_turns") == FIXED_PRIMARY_TURNS
            and int(window.get("start", -1)) <= stage.seed_start
            and stage.seed_window_end_exclusive <= int(window.get("end_exclusive", -1))
        ):
            matches.append((str(island_id), dict(profile)))
    if len(matches) != 1:
        raise RuntimeError(
            f"{stage.stage_id} seed window must map to exactly one N1=6 island"
        )
    island_id, profile = matches[0]
    if (
        float(profile.get("optimizer_all_current7_thermal_scale_C", -1.0))
        != stage.optimizer_all_thermal_scale_C
        or profile.get("terminal_physical_replay_required") is not True
        or profile.get("offspring_physics_repair_required") is not True
    ):
        raise RuntimeError(f"{stage.stage_id} base N1=6 island pressure drifted")
    return island_id, profile


def validate_stage_binding(
    *, bindings_root: Path, record: Mapping[str, Any], stage: FinalGoalStage
) -> dict[str, Any]:
    if set(record) != {"stage_id", "offload_plan", "publication_receipt"}:
        raise RuntimeError(f"{stage.stage_id} bundle binding fields drifted")
    if record.get("stage_id") != stage.stage_id:
        raise RuntimeError("bundle binding stage identity mismatch")
    plan_path = _resolve_from(bindings_root, record.get("offload_plan"), "offload plan")
    publication_path = _resolve_from(
        bindings_root,
        record.get("publication_receipt"),
        "publication receipt",
    )
    plan, manifest, source_map = load_bundle_plan(plan_path)
    _validate_local_inventory(manifest, source_map)
    publication = validate_publication_receipt(
        _read_json(publication_path), plan=plan, manifest=manifest
    )

    expected_spec, expected_constraints = optimizer_stage_contract(stage)
    identity = (manifest.get("adapter_receipt") or {}).get("identity") or {}
    if (
        manifest.get("hard_spec") != expected_spec
        or identity.get("hard_spec") != expected_spec
        or manifest.get("hard_spec_sha256") != canonical_sha256(expected_spec)
        or identity.get("hard_spec_sha256") != canonical_sha256(expected_spec)
        or manifest.get("constraint_names") != expected_constraints
        or identity.get("constraint_names") != expected_constraints
    ):
        raise RuntimeError(f"{stage.stage_id} published hard-spec identity mismatch")
    execution = manifest.get("search_execution") or {}
    if (
        execution.get("entrypoint")
        != "artifacts/code/tools/tier1_corrected_generation_preflight.py"
        or execution.get("runner")
        != "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py"
        or execution.get("optimizer_processes_per_task") != 1
        or execution.get("model_mapping_instances_per_process") != 1
        or not REQUIRED_REMOTE_CODE.issubset(manifest.get("code_inventory") or {})
    ):
        raise RuntimeError(f"{stage.stage_id} published runtime contract mismatch")
    scheduler_contract = manifest.get("scheduler_api_contract") or {}
    if (
        scheduler_contract.get("method") != "POST"
        or scheduler_contract.get("endpoint_path") != "/api/tasks"
        or set(scheduler_contract.get("required_fields") or [])
        != REQUIRED_SCHEDULER_FIELDS
        or scheduler_contract.get("requested_allocation_id_allowed") is not False
    ):
        raise RuntimeError("unsupported scheduler envelope contract")
    island_id, island_profile = _selected_base_island(manifest, stage)
    return {
        "stage_id": stage.stage_id,
        "plan_path": plan_path,
        "publication_path": publication_path,
        "plan": plan,
        "manifest": manifest,
        "publication": publication,
        "base_island_id": island_id,
        "base_island_profile_sha256": island_profile["sha256"],
        "hard_spec": expected_spec,
        "constraint_names": expected_constraints,
    }


def load_stage_bindings(path: Path) -> dict[str, dict[str, Any]]:
    path = path.resolve(strict=True)
    value = _read_json(path)
    records = value.get("stages")
    if (
        value.get("schema_version") != BINDINGS_SCHEMA
        or not isinstance(records, list)
        or len(records) != len(STAGES)
    ):
        raise RuntimeError("final1000 stage bundle binding inventory mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("stage bundle binding must be an object")
        stage_id = str(record.get("stage_id") or "")
        if stage_id in by_id or stage_id not in BY_ID:
            raise RuntimeError("stage bundle binding identity is unknown or repeated")
        by_id[stage_id] = record
    if set(by_id) != set(BY_ID):
        raise RuntimeError("stage bundle binding set is incomplete")
    return {
        stage.stage_id: validate_stage_binding(
            bindings_root=path.parent, record=by_id[stage.stage_id], stage=stage
        )
        for stage in STAGES
    }


def _replace_payload_sha(command: str, payload_sha256: str) -> str:
    replaced, count = PAYLOAD_SHA_PATTERN.subn(
        lambda match: match.group(1) + payload_sha256, command
    )
    if count != 1:
        raise RuntimeError("current7 task command has no unique payload SHA seal")
    return replaced


def build_stage_task(
    base_task: Mapping[str, Any],
    *,
    stage: FinalGoalStage,
    wave: str,
) -> dict[str, Any]:
    """Add the exact stage spec and 4c/28GiB scheduler resource seal."""

    if set(base_task) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError("base current7 task envelope fields drifted")
    payload = dict(base_task.get("payload_json") or {})
    lane = dict(payload.get("lane") or {})
    expected_spec, expected_constraints = optimizer_stage_contract(stage)
    seed = int(payload.get("seed", -1))
    if (
        lane.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or lane.get("seed") != seed
        or not stage.seed_start <= seed < stage.seed_window_end_exclusive
        or payload.get("max_generations") != FIXED_GENERATIONS
        or payload.get("inference_threads") != INFERENCE_THREADS
        or payload.get("optimizer_processes") != 1
    ):
        raise RuntimeError(f"{stage.stage_id} base task execution contract mismatch")
    lane.update(
        {
            "stage_id": stage.stage_id,
            "stage_variant": stage.variant,
            "wave": wave,
        }
    )
    payload.update(
        {
            "lane": lane,
            "hard_spec": expected_spec,
            "hard_spec_sha256": canonical_sha256(expected_spec),
            "stage_spec_sha256": canonical_sha256(expected_spec),
            "constraint_names": expected_constraints,
            "final_goal_stage_id": stage.stage_id,
            "final_goal_stage_profile_sha256": stage_profile(stage)["sha256"],
            "maximum_peak_rss_bytes": DEFAULT_PEAK_RSS_GATE_BYTES,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        }
    )
    payload_sha = canonical_sha256(payload)
    command = _replace_payload_sha(str(base_task.get("command") or ""), payload_sha)
    if any(token in command.lower() for token in ("ansysedt", "pyaedt.desktop")):
        raise RuntimeError("surrogate-only task command unexpectedly references AEDT")
    resources = {
        "cpus": DEFAULT_CPUS,
        "memory_mb": DEFAULT_MEMORY_MB,
        "scheduling_profile": "standard",
        # Required API discriminator only; payload and command use no AEDT.
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": DEFAULT_PRIORITY,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
    }
    dedupe = canonical_sha256(
        {
            "goal": "final1000",
            "stage_profile_sha256": stage_profile(stage)["sha256"],
            "bundle_id": payload.get("bundle_id"),
            "payload": payload,
            "resources": resources,
        }
    )
    task = {
        **dict(base_task),
        "name": f"{stage.task_name_stem}-{wave}-{seed}",
        "command": command,
        "payload_json": payload,
        **resources,
        "dedupe_key": f"mft-tier1-final1000:{dedupe}",
    }
    validate_task(task, expected_stage=stage, expected_wave=wave)
    return task


def validate_task(
    task: Mapping[str, Any],
    *,
    expected_stage: FinalGoalStage | None = None,
    expected_wave: str | None = None,
) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS or "requested_allocation_id" in task:
        raise RuntimeError("final1000 scheduler envelope fields drifted")
    payload = task.get("payload_json") or {}
    stage_id = str(payload.get("final_goal_stage_id") or "")
    stage = BY_ID.get(stage_id)
    if stage is None or (expected_stage is not None and stage != expected_stage):
        raise RuntimeError("final1000 task stage identity mismatch")
    spec, constraints = optimizer_stage_contract(stage)
    lane = payload.get("lane") or {}
    wave = str(lane.get("wave") or "")
    seed = int(payload.get("seed", -1))
    payload_sha = canonical_sha256(payload)
    command = str(task.get("command") or "")
    if (
        (expected_wave is not None and wave != expected_wave)
        or wave not in {"canary", "ramp", "refill"}
        or lane.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or lane.get("seed") != seed
        or not stage.seed_start <= seed < stage.seed_window_end_exclusive
        or payload.get("hard_spec") != spec
        or payload.get("hard_spec_sha256") != canonical_sha256(spec)
        or payload.get("stage_spec_sha256") != canonical_sha256(spec)
        or payload.get("constraint_names") != constraints
        or payload.get("final_goal_stage_profile_sha256")
        != stage_profile(stage)["sha256"]
        or payload.get("max_generations") != FIXED_GENERATIONS
        or payload.get("inference_threads") != INFERENCE_THREADS
        or payload.get("maximum_peak_rss_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or any(
            payload.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
        or task.get("cpus") != DEFAULT_CPUS
        or task.get("memory_mb") != DEFAULT_MEMORY_MB
        or task.get("scheduling_profile") != "standard"
        or task.get("aedt_backend") != "standalone"
        or task.get("gpus") != 0
        or task.get("priority") != DEFAULT_PRIORITY
        or task.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or task.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or task.get("required_capability") != "conda:pyaedt2026v1"
        or task.get("env_profile") != "pyaedt2026v1"
        or not str(task.get("remote_cwd") or "").startswith("/")
        or not str(task.get("dedupe_key") or "").startswith("mft-tier1-final1000:")
        or not str(task.get("name") or "").startswith("mft-t1fg-")
        or f"--payload-sha256 {payload_sha}" not in command
        or any(token in command.lower() for token in ("ansysedt", "pyaedt.desktop"))
    ):
        raise RuntimeError("final1000 surrogate-only task seal mismatch")
    return dict(task)


def build_launch_plan(bindings_path: Path) -> dict[str, Any]:
    bindings = load_stage_bindings(bindings_path)
    canary_lanes, ramp_lanes = logical_lanes()
    return _build_launch_plan_from_lanes(
        bindings,
        canary_lanes=canary_lanes,
        ramp_lanes=ramp_lanes,
    )


def _build_launch_plan_from_lanes(
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    canary_lanes: Sequence[Mapping[str, Any]],
    ramp_lanes: Sequence[Mapping[str, Any]],
    topology_science_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the controller's exact 4+496 shape from sealed logical lanes."""

    task_waves: dict[str, list[dict[str, Any]]] = {"canaries": [], "ramp": []}
    for wave_name, lanes in (
        ("canaries", canary_lanes),
        ("ramp", ramp_lanes),
    ):
        wave = "canary" if wave_name == "canaries" else "ramp"
        for lane in lanes:
            stage = BY_ID[lane["stage_id"]]
            binding = bindings[stage.stage_id]
            base = build_current7_task_payload(
                binding["plan"],
                binding["manifest"],
                seed=int(lane["seed"]),
                priority=DEFAULT_PRIORITY,
            )
            logical_child = lane.get("topology_logical_child")
            if logical_child is not None:
                payload = dict(base["payload_json"])
                payload.update(
                    {
                        "topology_successor_science_plan_sha256": (
                            topology_science_identity["science_plan_sha256"]
                        ),
                        "topology_successor_logical_payload_sha256": (
                            logical_child["payload_sha256"]
                        ),
                        "topology_successor_logical_dedupe_key": (
                            logical_child["dedupe_key"]
                        ),
                    }
                )
                base = {**base, "payload_json": payload}
            task = build_stage_task(base, stage=stage, wave=wave)
            if logical_child is not None:
                payload = task["payload_json"]
                if (
                    payload.get("lane", {}).get("island_id")
                    != logical_child["optimizer_island_id"]
                    or payload.get("topology_niche_contract_sha256")
                    != topology_science_identity["topology_niche_contract_sha256"]
                    or payload.get("population") != logical_child["population"]
                    or payload.get("max_generations")
                    != logical_child["max_generations"]
                ):
                    raise RuntimeError(
                        "topology logical child/bundle island binding mismatch"
                    )
            task_waves[wave_name].append(task)
    summaries = {
        stage.stage_id: {
            "bundle_id": bindings[stage.stage_id]["plan"]["bundle_id"],
            "bundle_manifest_sha256": bindings[stage.stage_id]["plan"][
                "bundle_manifest_sha256"
            ],
            "remote_bundle": bindings[stage.stage_id]["plan"]["remote_bundle"],
            "publication_receipt_sha256": bindings[stage.stage_id]["publication"][
                "receipt_sha256"
            ],
            "ready": bindings[stage.stage_id]["publication"]["ready"],
            "ready_sha256": bindings[stage.stage_id]["publication"]["ready_sha256"],
            "base_island_id": bindings[stage.stage_id]["base_island_id"],
            "stage_spec_sha256": stage_profile(stage)["stage_spec_sha256"],
        }
        for stage in STAGES
    }
    unsigned = {
        "schema_version": LAUNCH_SCHEMA,
        "created_at": _now(),
        "stage_inventory": stage_inventory(),
        "stage_bindings": summaries,
        "resources": {
            "cpus_per_task": DEFAULT_CPUS,
            "memory_mb_per_task": DEFAULT_MEMORY_MB,
            "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
            "priority": DEFAULT_PRIORITY,
            "scheduling_profile": "standard",
            "gpus": 0,
        },
        "fast_ramp": {
            "canary_count": len(task_waves["canaries"]),
            "ramp_count": len(task_waves["ramp"]),
            "release_trigger": (
                "all_four_remote_model_load_rss_repair_and_stage_spec_gates_pass"
            ),
            "wait_for_optimizer_completion": False,
            "submit_remaining_in_one_wave": True,
        },
        "open_ended_refill": {
            "policy": "maintain_500_active_until_explicit_stop",
            "logical_active_target": TOTAL_ACTIVE_QUOTA,
            "stage_active_quotas": {
                stage.stage_id: SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
                for stage in STAGES
            },
            "refill_each_observed_terminal_gap": True,
            "wait_for_full_wave": False,
            "seed_reuse_allowed": False,
            "stop_requires_explicit_flag": True,
        },
        "task_waves": task_waves,
        "surrogate_only": True,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    if topology_science_identity is not None:
        identity = dict(topology_science_identity)
        unsigned_identity = {
            key: item for key, item in identity.items() if key != "sha256"
        }
        if identity.get("sha256") != canonical_sha256(unsigned_identity):
            raise RuntimeError("topology successor science identity seal mismatch")
        unsigned["topology_successor_science_identity"] = identity
    plan = {**unsigned, "launch_plan_sha256": canonical_sha256(unsigned)}
    return validate_launch_plan(plan)


def build_topology_successor_launch_plan(
    bindings_path: Path,
    science_plan_path: Path,
) -> dict[str, Any]:
    """Bind topology-balanced exact500 science to the proven single-seed path.

    The separate 16-seed science canary remains a pre-cutover gate.  Once it
    passes, this launch plan exposes exactly one controller canary template per
    stage and 496 ramp tasks, so rolling migration can inherit the live 500 and
    fill only natural terminal gaps without changing controller semantics.
    """

    try:
        from tier1_final1000_topology_successor_plan import validate_plan
    except ImportError:  # pragma: no cover - repository import path
        from tools.tier1_final1000_topology_successor_plan import validate_plan

    bindings = load_stage_bindings(bindings_path)
    raw = _read_json(science_plan_path.resolve(strict=True))
    science = validate_plan(raw.get("plan") if "plan" in raw else raw)
    production = science["production"]["logical_children"]
    canary_lanes: list[dict[str, Any]] = []
    ramp_lanes: list[dict[str, Any]] = []
    for stage in STAGES:
        children = [
            child for child in production if child["stage_id"] == stage.stage_id
        ]
        expected = SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
        if len(children) != expected:
            raise RuntimeError("topology production quota differs from controller")
        for index, child in enumerate(children):
            lane = {
                "stage_id": stage.stage_id,
                "variant": stage.variant,
                "seed": int(child["seed"]),
                "fixed_primary_turns": FIXED_PRIMARY_TURNS,
                "wave": "canary" if index == 0 else "ramp",
                "topology_logical_child": child,
            }
            (canary_lanes if index == 0 else ramp_lanes).append(lane)
    child_identities = [
        {
            "stage_id": child["stage_id"],
            "seed": child["seed"],
            "payload_sha256": child["payload_sha256"],
            "dedupe_key": child["dedupe_key"],
        }
        for child in production
    ]
    identity = {
        "schema_version": "mft-tier1-final1000-topology-rollout-binding-v1",
        "science_plan_sha256": science["sha256"],
        "topology_niche_contract_sha256": science[
            "topology_niche_contract_sha256"
        ],
        "pre_cutover_canary_logical_seed_count": science["canary"][
            "logical_seed_count"
        ],
        "controller_canary_template_count": len(canary_lanes),
        "controller_ramp_count": len(ramp_lanes),
        "production_logical_child_count": len(production),
        "production_logical_child_identity_sha256": canonical_sha256(
            child_identities
        ),
        "single_seed_adapter_task_path_unchanged": True,
        "parent_lane_packing_is_science_independent": True,
        "scheduler_submission_performed": False,
    }
    identity["sha256"] = canonical_sha256(identity)
    return _build_launch_plan_from_lanes(
        bindings,
        canary_lanes=canary_lanes,
        ramp_lanes=ramp_lanes,
        topology_science_identity=identity,
    )


def build_topology_pre_cutover_canary_plan(
    bindings_path: Path,
    science_plan_path: Path,
) -> dict[str, Any]:
    """Render the sealed 16-seed terminal gate without scheduler mutation."""

    try:
        from tier1_final1000_topology_successor_plan import validate_plan
    except ImportError:  # pragma: no cover - repository import path
        from tools.tier1_final1000_topology_successor_plan import validate_plan

    bindings = load_stage_bindings(bindings_path)
    raw = _read_json(science_plan_path.resolve(strict=True))
    science = validate_plan(raw.get("plan") if "plan" in raw else raw)
    tasks = []
    for child in science["canary"]["logical_children"]:
        stage = BY_ID[child["stage_id"]]
        binding = bindings[stage.stage_id]
        base = build_current7_task_payload(
            binding["plan"],
            binding["manifest"],
            seed=int(child["seed"]),
            priority=DEFAULT_PRIORITY,
        )
        base_payload = {
            **base["payload_json"],
            "topology_successor_science_plan_sha256": science["sha256"],
            "topology_successor_logical_payload_sha256": child[
                "payload_sha256"
            ],
            "topology_successor_logical_dedupe_key": child["dedupe_key"],
            "topology_pre_cutover_canary": True,
        }
        task = build_stage_task(
            {**base, "payload_json": base_payload},
            stage=stage,
            wave="canary",
        )
        payload = task["payload_json"]
        if (
            payload.get("lane", {}).get("island_id")
            != child["optimizer_island_id"]
            or payload.get("topology_niche_contract_sha256")
            != science["topology_niche_contract_sha256"]
            or payload.get("population") != child["population"]
            or payload.get("max_generations") != child["max_generations"]
        ):
            raise RuntimeError("topology canary/bundle island binding mismatch")
        tasks.append(task)
    unsigned = {
        "schema_version": TOPOLOGY_CANARY_LAUNCH_SCHEMA,
        "created_at": _now(),
        "science_plan_sha256": science["sha256"],
        "topology_niche_contract_sha256": science[
            "topology_niche_contract_sha256"
        ],
        "logical_seed_count": len(tasks),
        "stage_counts": dict(science["canary"]["stage_counts"]),
        "resources": {
            "cpus_per_task": DEFAULT_CPUS,
            "memory_mb_per_task": DEFAULT_MEMORY_MB,
            "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
            "priority": DEFAULT_PRIORITY,
            "scheduling_profile": "standard",
            "gpus": 0,
        },
        "tasks": tasks,
        "terminal_gate": dict(science["canary"]["gate"]),
        "same_single_seed_adapter_task_path_as_production": True,
        "production_launch_allowed_before_all_16_terminal": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    value = {**unsigned, "sha256": canonical_sha256(unsigned)}
    return validate_topology_pre_cutover_canary_plan(value)


def validate_topology_pre_cutover_canary_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    tasks = value.get("tasks")
    if (
        value.get("schema_version") != TOPOLOGY_CANARY_LAUNCH_SCHEMA
        or value.get("sha256") != canonical_sha256(unsigned)
        or not _is_sha256(value.get("science_plan_sha256"))
        or not _is_sha256(value.get("topology_niche_contract_sha256"))
        or not isinstance(tasks, list)
        or value.get("logical_seed_count") != 16
        or len(tasks) != 16
        or value.get("stage_counts")
        != {
            "entry-1200-t125": 6,
            "bridge-1150-t115": 6,
            "close-1075-t107p5": 2,
            "final-1000-t100": 2,
        }
        or value.get("same_single_seed_adapter_task_path_as_production") is not True
        or value.get("production_launch_allowed_before_all_16_terminal")
        is not False
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "submission_performed",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("topology pre-cutover canary top-level seal mismatch")
    for task in tasks:
        validate_task(task, expected_wave="canary")
        payload = task["payload_json"]
        if (
            payload.get("topology_pre_cutover_canary") is not True
            or payload.get("topology_successor_science_plan_sha256")
            != value["science_plan_sha256"]
            or payload.get("topology_niche_contract_sha256")
            != value["topology_niche_contract_sha256"]
            or not _is_sha256(
                payload.get("topology_successor_logical_payload_sha256")
            )
            or not str(
                payload.get("topology_successor_logical_dedupe_key") or ""
            ).startswith("mft-final1000-topology-canary-")
        ):
            raise RuntimeError("topology pre-cutover canary task mismatch")
    if (
        len({task["dedupe_key"] for task in tasks}) != 16
        or len(
            {
                task["payload_json"]["topology_successor_logical_dedupe_key"]
                for task in tasks
            }
        )
        != 16
        or {
            stage.stage_id: sum(
                task["payload_json"]["final_goal_stage_id"] == stage.stage_id
                for task in tasks
            )
            for stage in STAGES
        }
        != value["stage_counts"]
    ):
        raise RuntimeError("topology pre-cutover canary identity/quota mismatch")
    return dict(value)


def validate_launch_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "launch_plan_sha256"}
    waves = value.get("task_waves") or {}
    canaries = waves.get("canaries") if isinstance(waves, dict) else None
    ramp = waves.get("ramp") if isinstance(waves, dict) else None
    bindings = value.get("stage_bindings")
    if (
        value.get("schema_version") != LAUNCH_SCHEMA
        or value.get("stage_inventory") != stage_inventory()
        or value.get("launch_plan_sha256") != canonical_sha256(unsigned)
        or not isinstance(canaries, list)
        or not isinstance(ramp, list)
        or not isinstance(bindings, dict)
        or set(bindings) != set(BY_ID)
        or len(canaries) != 4
        or len(ramp) != TOTAL_ACTIVE_QUOTA - 4
        or value.get("surrogate_only") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "submission_performed",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
        or (value.get("open_ended_refill") or {}).get("logical_active_target")
        != TOTAL_ACTIVE_QUOTA
        or (value.get("open_ended_refill") or {}).get("stage_active_quotas")
        != SUCCESSOR_ACTIVE_QUOTAS
    ):
        raise RuntimeError("final1000 launch-plan top-level seal mismatch")
    for stage in STAGES:
        binding = bindings[stage.stage_id]
        ready = binding.get("ready") if isinstance(binding, dict) else None
        if (
            not isinstance(ready, dict)
            or binding.get("ready_sha256") != canonical_sha256(ready)
            or binding.get("stage_spec_sha256")
            != stage_profile(stage)["stage_spec_sha256"]
            or not str(binding.get("remote_bundle") or "").startswith("/")
            or not isinstance(binding.get("bundle_id"), str)
            or not binding["bundle_id"]
            or not isinstance(binding.get("bundle_manifest_sha256"), str)
            or len(binding["bundle_manifest_sha256"]) != 64
            or not isinstance(binding.get("publication_receipt_sha256"), str)
            or len(binding["publication_receipt_sha256"]) != 64
            or not isinstance(binding.get("base_island_id"), str)
            or not binding["base_island_id"]
        ):
            raise RuntimeError(f"{stage.stage_id} launch binding/READY seal mismatch")
    tasks = [*canaries, *ramp]
    for task in canaries:
        validate_task(task, expected_wave="canary")
    for task in ramp:
        validate_task(task, expected_wave="ramp")
    dedupe = [task["dedupe_key"] for task in tasks]
    identities = [
        (
            task["payload_json"]["bundle_id"],
            int(task["payload_json"]["seed"]),
        )
        for task in tasks
    ]
    if len(set(dedupe)) != TOTAL_ACTIVE_QUOTA or len(set(identities)) != len(tasks):
        raise RuntimeError("final1000 launch plan contains duplicate work")
    stage_counts = {
        stage.stage_id: sum(
            task["payload_json"]["final_goal_stage_id"] == stage.stage_id
            for task in tasks
        )
        for stage in STAGES
    }
    if stage_counts != SUCCESSOR_ACTIVE_QUOTAS:
        raise RuntimeError("final1000 launch stage quota drifted")
    canary_stages = {task["payload_json"]["final_goal_stage_id"] for task in canaries}
    if canary_stages != set(BY_ID):
        raise RuntimeError("final1000 canary set is incomplete")
    topology_identity = value.get("topology_successor_science_identity")
    topology_fields = (
        "topology_successor_science_plan_sha256",
        "topology_successor_logical_payload_sha256",
        "topology_successor_logical_dedupe_key",
    )
    if topology_identity is None:
        if any(
            any(field in task["payload_json"] for field in topology_fields)
            for task in tasks
        ):
            raise RuntimeError("unsealed topology successor task metadata")
    else:
        if not _sealed_mapping(topology_identity):
            raise RuntimeError("topology successor science identity is unsealed")
        logical_identities = []
        for stage in STAGES:
            stage_tasks = [
                task
                for task in tasks
                if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
            ]
            for task in stage_tasks:
                payload = task["payload_json"]
                if (
                    payload.get("topology_successor_science_plan_sha256")
                    != topology_identity.get("science_plan_sha256")
                    or payload.get("topology_niche_contract_sha256")
                    != topology_identity.get("topology_niche_contract_sha256")
                    or not _is_sha256(
                        payload.get("topology_successor_logical_payload_sha256")
                    )
                    or not str(
                        payload.get("topology_successor_logical_dedupe_key") or ""
                    ).startswith("mft-final1000-topology-production-")
                ):
                    raise RuntimeError("topology successor task identity mismatch")
                logical_identities.append(
                    {
                        "stage_id": stage.stage_id,
                        "seed": payload["seed"],
                        "payload_sha256": payload[
                            "topology_successor_logical_payload_sha256"
                        ],
                        "dedupe_key": payload[
                            "topology_successor_logical_dedupe_key"
                        ],
                    }
                )
        if (
            topology_identity.get("schema_version")
            != "mft-tier1-final1000-topology-rollout-binding-v1"
            or not _is_sha256(topology_identity.get("science_plan_sha256"))
            or not _is_sha256(
                topology_identity.get("topology_niche_contract_sha256")
            )
            or topology_identity.get("pre_cutover_canary_logical_seed_count") != 16
            or topology_identity.get("controller_canary_template_count") != 4
            or topology_identity.get("controller_ramp_count") != 496
            or topology_identity.get("production_logical_child_count") != 500
            or topology_identity.get(
                "production_logical_child_identity_sha256"
            )
            != canonical_sha256(logical_identities)
            or topology_identity.get("single_seed_adapter_task_path_unchanged")
            is not True
            or topology_identity.get("parent_lane_packing_is_science_independent")
            is not True
            or topology_identity.get("scheduler_submission_performed") is not False
            or len(
                {
                    identity["dedupe_key"] for identity in logical_identities
                }
            )
            != TOTAL_ACTIVE_QUOTA
        ):
            raise RuntimeError("topology successor rollout binding mismatch")
    return dict(value)


def _compact_terminal_replay_seal_matches(
    result: Mapping[str, Any], constraints: Sequence[str]
) -> bool:
    """Authenticate the compact result emitted by the deployed preflight.

    The current7 preflight persists the full terminal physical matrices as
    hash-addressed artifacts.  It intentionally does not copy the older
    three convenience summary dictionaries into ``result.json``.  Accept
    that compact schema only when its result payload, replay audit,
    inventory, shapes, and per-constraint infeasibility projection all form
    one internally sealed physical replay.
    """

    if result.get("schema_version") != CURRENT7_RESULT_SCHEMA:
        return False
    payload_sha = result.get("payload_sha256")
    unsigned_result = {
        key: value for key, value in result.items() if key != "payload_sha256"
    }
    population = result.get("population")
    terminal_count = result.get("terminal_population_count")
    if (
        not _is_sha256(payload_sha)
        or payload_sha != canonical_sha256(unsigned_result)
        or isinstance(population, bool)
        or not isinstance(population, int)
        or population <= 0
        or terminal_count != population
    ):
        return False

    replay = result.get("terminal_physical_replay_evidence")
    repair_audit = result.get("optimizer_repair_audit")
    replay_repair = replay.get("repair") if isinstance(replay, dict) else None
    if (
        not _sealed_mapping(replay)
        or replay.get("schema_version") != CURRENT7_REPLAY_SCHEMA
        or replay.get("coordinate_count") != population
        or replay.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or replay.get("optimizer_objectives_match") is not True
        or replay.get("optimizer_physical_G_match") is not True
        or replay.get("all_winding_budget_identities_passed") is not True
        or any(
            not _is_sha256(replay.get(field))
            for field in (
                "objective_sha256",
                "optimizer_G_sha256",
                "physical_unscaled_G_sha256",
            )
        )
        or not _sealed_mapping(replay_repair)
        or replay_repair.get("fixed_point_idempotent") is not True
        or replay_repair.get("fixed_primary_coordinate_verified") is not True
        or replay_repair.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or replay_repair.get("input_count") != population
        or replay_repair.get("output_count") != population
        or not _sealed_mapping(repair_audit)
        or repair_audit.get("authoritative_terminal_G") != "physical_unscaled_replay"
        or repair_audit.get("terminal_physical_replay") != replay
        or repair_audit.get("stages")
        != {
            "initial_population": True,
            "warm_start": True,
            "every_offspring": True,
            "terminal_physical_replay": True,
        }
    ):
        return False

    inventory = result.get("artifact_inventory")
    if not isinstance(inventory, dict) or result.get(
        "artifact_inventory_sha256"
    ) != canonical_sha256(inventory):
        return False
    expected_arrays = {
        "terminal_X": ("terminal_X.npy", [population, 25]),
        "terminal_F": ("terminal_F.npy", [population, 2]),
        "terminal_G_optimizer": (
            "terminal_G_optimizer.npy",
            [population, len(constraints)],
        ),
        "terminal_G_physical": (
            "terminal_G_physical.npy",
            [population, len(constraints)],
        ),
    }
    for name, (path, shape) in expected_arrays.items():
        record = inventory.get(name)
        if (
            not isinstance(record, dict)
            or record.get("path") != path
            or record.get("shape") != shape
            or record.get("dtype") != "float64"
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
        ):
            return False

    infeasibility = result.get("infeasibility_report")
    rows = infeasibility.get("constraints") if isinstance(infeasibility, dict) else None
    least_physical = (
        infeasibility.get("least_physical_constraint_G")
        if isinstance(infeasibility, dict)
        else None
    )
    physical_feasible_count = result.get("physical_feasible_count")
    if (
        not isinstance(infeasibility, dict)
        or infeasibility.get("schema_version")
        != "mft-tier1-current7-infeasibility-report-v1"
        or infeasibility.get("authoritative_constraints")
        != "terminal_unscaled_physical_replay"
        or infeasibility.get("population_size") != population
        or isinstance(physical_feasible_count, bool)
        or not isinstance(physical_feasible_count, int)
        or not 0 <= physical_feasible_count <= population
        or infeasibility.get("physical_feasible_count") != physical_feasible_count
        or not isinstance(rows, list)
        or len(rows) != len(constraints)
        or not isinstance(least_physical, dict)
        or set(least_physical) != set(constraints)
        or any(not _finite_number(value) for value in least_physical.values())
        or any(
            not isinstance(row, dict)
            or row.get("index") != index
            or row.get("name") != name
            or row.get("finite_count") != population
            or isinstance(row.get("passing_count"), bool)
            or not isinstance(row.get("passing_count"), int)
            or not 0 <= row["passing_count"] <= population
            or any(
                not _finite_number(row.get(field))
                for field in (
                    "minimum_physical_G",
                    "median_physical_G",
                    "minimum_positive_violation",
                )
            )
            for index, (name, row) in enumerate(zip(constraints, rows))
        )
        or any(
            infeasibility.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_performed",
                "automatic_promotion_allowed",
            )
        )
    ):
        return False
    return True


def _physical_replay_seal_matches(
    result: Mapping[str, Any], constraints: Sequence[str]
) -> bool:
    physical_fields = (
        "constraint_minimum_G",
        "terminal_population_best_constraint_G",
        "optimizer_terminal_best_physical_constraint_G",
    )
    present = [field in result for field in physical_fields]
    if all(present):
        return all(
            isinstance(result.get(field), dict)
            and set(result[field]) == set(constraints)
            and all(_finite_number(value) for value in result[field].values())
            for field in physical_fields
        )
    if any(present):
        return False
    return _compact_terminal_replay_seal_matches(result, constraints)


def validate_stage_result(
    result: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    task_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] = validate_task,
) -> dict[str, Any]:
    """Validate terminal physics after authenticating its scheduler envelope.

    ``task_validator`` exists for rolling migrations whose immutable legacy
    envelopes use the prior 8-core resource policy.  The default remains the
    current launch policy, and callers must supply an equally strict validator
    rather than bypassing task authentication.
    """

    validated_task = dict(task_validator(task))
    payload = validated_task["payload_json"]
    stage = BY_ID[payload["final_goal_stage_id"]]
    spec, constraints = optimizer_stage_contract(stage)
    if (
        result.get("hard_spec") != spec
        or result.get("stage_spec_sha256") != canonical_sha256(spec)
        or result.get("constraint_names") != constraints
        or result.get("seed") != payload["seed"]
        or result.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or result.get("terminal_population_primary_turn_values")
        != [FIXED_PRIMARY_TURNS]
        or result.get("terminal_population_fixed_primary_turns_verified") is not True
        or result.get("terminal_physical_replay_attested") is not True
        or not _physical_replay_seal_matches(result, constraints)
        or any(
            result.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("final1000 terminal result stage/replay seal mismatch")
    value = {
        "schema_version": RESULT_PREFLIGHT_SCHEMA,
        "stage_id": stage.stage_id,
        "seed": payload["seed"],
        "stage_profile_sha256": stage_profile(stage)["sha256"],
        "stage_spec_sha256": canonical_sha256(spec),
        "constraint_names": constraints,
        "terminal_physical_replay_attested": True,
        "surrogate_only": True,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profiles")
    render = commands.add_parser("render")
    render.add_argument("--bindings", type=Path, required=True)
    render.add_argument(
        "--topology-science-plan",
        type=Path,
        help=(
            "optional sealed topology successor plan; renders its high-offset "
            "exact500 logical seeds through the existing single-seed controller"
        ),
    )
    render.add_argument("--output", type=Path, required=True)
    topology_canary = commands.add_parser("render-topology-canary")
    topology_canary.add_argument("--bindings", type=Path, required=True)
    topology_canary.add_argument("--topology-science-plan", type=Path, required=True)
    topology_canary.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    result = commands.add_parser("validate-result")
    result.add_argument("--result", type=Path, required=True)
    result.add_argument("--task", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "profiles":
        value = stage_inventory()
    elif args.command == "render":
        value = (
            build_launch_plan(args.bindings)
            if args.topology_science_plan is None
            else build_topology_successor_launch_plan(
                args.bindings, args.topology_science_plan
            )
        )
        _atomic_json(args.output, value)
    elif args.command == "render-topology-canary":
        value = build_topology_pre_cutover_canary_plan(
            args.bindings, args.topology_science_plan
        )
        _atomic_json(args.output, value)
    elif args.command == "validate":
        value = validate_launch_plan(_read_json(args.plan.resolve(strict=True)))
    else:
        value = validate_stage_result(
            _read_json(args.result.resolve(strict=True)),
            _read_json(args.task.resolve(strict=True)),
        )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
