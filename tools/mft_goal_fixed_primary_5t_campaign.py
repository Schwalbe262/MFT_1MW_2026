"""Launch the bounded 5 mm / 1.6 mm primary-manufacturing NSGA-II repair.

This is an isolated scientific sidecar.  It reuses the already authenticated
G0 surrogate deployment, embeds this exact runner in each Scheduler payload,
and never imports or modifies Scheduler application code.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Iterable, Mapping
import urllib.request
import zlib


SCHEMA = "mft-goal-fixed-primary-5t-nsga-task-v1"
RESULT_SCHEMA = "mft-goal-fixed-primary-5t-nsga-result-v1"
CAMPAIGN_SCHEMA = "mft-goal-fixed-primary-5t-nsga-submission-v1"
CAMPAIGN_ID = "mft-goal-fixed-primary-5t-gap1-1p6-axis-v5"
HEDGE_CAMPAIGN_ID = "mft-goal-fixed-primary-5t-gap1-variable-20260727-v1"
PRIMARY_CONDUCTOR_MM = 5.0
PRIMARY_GAP_MM = 1.6
POPULATION = 320
GENERATIONS = 80
SEED_START = 2_707_275_600
SEED_COUNT = 16
HEDGE_SEED_START = 2_707_275_400
HEDGE_SEED_COUNT = 8
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 7_200
MAX_WORKERS_PER_NODE = 8
SCHEDULER_URL = "http://127.0.0.1:8002"
REMOTE_BUNDLE = (
    "/gpfs/tmp_cpu2/mft_goal_20260726/"
    "mft-goal-763dbb46ae74e1722969d2c2"
)
LOCAL_GOAL_BUNDLE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\campaign_rolling512_a0977dd"
)
GLOBAL_TERMINAL_CSV = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\aggregate_rolling512_d4e4d60\global_terminal_candidates.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_nsga_v5"
)
BASE_TASKS = {
    5: "artifacts/campaign/tasks/seed-2607262000-n1-5.json",
    6: "artifacts/campaign/tasks/seed-2607262001-n1-6.json",
    7: "artifacts/campaign/tasks/seed-2607262002-n1-7.json",
    8: "artifacts/campaign/tasks/seed-2607262003-n1-8.json",
}
BASE_RELOCATIONS = {
    5: "artifacts/campaign/relocations/seed-2607262000-n1-5.json",
    6: "artifacts/campaign/relocations/seed-2607262001-n1-6.json",
    7: "artifacts/campaign/relocations/seed-2607262002-n1-7.json",
    8: "artifacts/campaign/relocations/seed-2607262003-n1-8.json",
}
HARD_SPEC = {
    "schema_version": "mft-goal-fixed-primary-manufacturing-v1",
    "primary_conductor_thickness_mm": PRIMARY_CONDUCTOR_MM,
    "primary_interturn_gap_mm": PRIMARY_GAP_MM,
    "primary_controls_are_hard_fixed": True,
    "size_limits_mm": {"W": 1000.0, "L": 1200.0, "H": 750.0},
    "axis_contract": {
        "W": "drawing_x_original_973mm_side-secondary-direction",
        "L": "drawing_y_perpendicular-direction",
        "rotation_or_axis_swap_allowed": False,
    },
    "self_resonance_min_Hz": 15_000.0,
    "winding_temperature_max_C": 100.0,
    "core_temperature_max_C": 120.0,
    "fan_velocity_m_s": 1.5,
    "cooling_and_TIM_mutation_allowed": False,
    "downstream_FEA": {
        "topology": "symmetric",
        "winding_geometry": "unrounded",
        "rounded_FEA_allowed": False,
        "rounded_use": "drawing_and_final-shape-visualization-only",
    },
}
HEDGE_HARD_SPEC = {
    **{
        key: copy.deepcopy(value)
        for key, value in HARD_SPEC.items()
        if key
        not in {
            "primary_interturn_gap_mm",
            "primary_controls_are_hard_fixed",
        }
    },
    "schema_version": "mft-goal-fixed-primary-5t-gap-variable-v1",
    "primary_conductor_is_hard_fixed": True,
    "primary_interturn_gap_search_mm": {
        "minimum": 0.3,
        "maximum": 5.0,
    },
    "hedge_only": True,
    "strict_1p6_lane_has_selection_priority": True,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["payload_sha256"] = _sha(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _sha(unsigned):
        raise RuntimeError(f"{schema} seal mismatch")
    return dict(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _safe_remote_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe remote relative path: {value}")
    return path.as_posix()


def _api_json(
    url: str, *, method: str = "GET", payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        data=None if payload is None else _canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Scheduler returned a non-object response")
    return value


def _source_bytes() -> bytes:
    return Path(__file__).resolve(strict=True).read_bytes()


def _warm_coordinates_by_turns(
    source: Path, *, maximum_rows: int = 384
) -> dict[int, bytes]:
    result: dict[int, bytes] = {}
    for turns in sorted(BASE_TASKS):
        with source.resolve(strict=True).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            selected = heapq.nsmallest(
                maximum_rows,
                (
                    row
                    for row in reader
                    if row["source_island_id"] == f"n1-{turns}"
                ),
                key=lambda row: (
                    float(row["normalized_constraint_violation"]),
                    float(row["objective_volume_L"]),
                    float(row["objective_total_loss_W"]),
                ),
            )
        coordinates = [
            [float(item) for item in json.loads(row["coordinate_unit_json"])]
            for row in selected
        ]
        if (
            len(coordinates) < POPULATION // 2
            or any(len(row) != 25 for row in coordinates)
            or any(
                not 0.0 <= item <= 1.0
                for row in coordinates
                for item in row
            )
        ):
            raise RuntimeError(f"N1={turns} warm coordinate pool is incomplete")
        result[turns] = zlib.compress(
            json.dumps(
                coordinates,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            level=9,
        )
    return result


def _task_payload(
    *,
    seed: int,
    turns: int,
    source_bytes: bytes,
    warm_bytes: bytes,
    campaign_id: str,
    lane: str,
    hard_spec: Mapping[str, Any],
) -> dict[str, Any]:
    hard_spec_sha = _sha(hard_spec)
    runner_sha = hashlib.sha256(source_bytes).hexdigest()
    warm_sha = hashlib.sha256(warm_bytes).hexdigest()
    base_task_local = (
        LOCAL_GOAL_BUNDLE
        / "tasks"
        / Path(BASE_TASKS[turns]).name
    )
    base_task = _read_json(base_task_local)
    physics = {
        "base_task_payload_sha256": base_task["payload_sha256"],
        "base_code_revision": base_task["source_identity"]["code_revision"],
        "dataset_sha256": base_task["source_identity"]["dataset_sha256"],
        "evaluation_model_sha256": base_task["source_identity"][
            "evaluation_model_sha256"
        ],
        "hard_spec_sha256": hard_spec_sha,
        "runner_source_sha256": runner_sha,
        "population": POPULATION,
        "generations": GENERATIONS,
    }
    return _seal(
        {
            "schema_version": SCHEMA,
            "campaign_id": campaign_id,
            "lane": lane,
            "created_at": _now(),
            "seed": int(seed),
            "fixed_primary_turns": int(turns),
            "population": POPULATION,
            "generations": GENERATIONS,
            "base_task_relative_path": BASE_TASKS[turns],
            "base_relocation_relative_path": BASE_RELOCATIONS[turns],
            "base_task_payload_sha256": base_task["payload_sha256"],
            "base_source_identity": base_task["source_identity"],
            "hard_spec": copy.deepcopy(dict(hard_spec)),
            "hard_spec_sha256": hard_spec_sha,
            "physics_sha256": _sha(physics),
            "runner_source_sha256": runner_sha,
            "runner_source_b64": base64.b64encode(source_bytes).decode("ascii"),
            "warm_start_sha256": warm_sha,
            "warm_start_b64": base64.b64encode(warm_bytes).decode("ascii"),
            "global_nds_artifact_required": True,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )


def _scheduler_payload(task: Mapping[str, Any], *, priority: int) -> dict[str, Any]:
    seed = int(task["seed"])
    turns = int(task["fixed_primary_turns"])
    command = "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:'
            '?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) '
            'payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path" >&2; exit 66 ;; esac',
            'runner_path="$(dirname "$payload_path")/fixed_primary_5t.py"',
            "python -c 'import base64,json,pathlib,sys;"
            "p=json.load(open(sys.argv[1],encoding=\"utf-8\"));"
            "pathlib.Path(sys.argv[2]).write_bytes("
            "base64.b64decode(p[\"runner_source_b64\"]))' "
            '"$payload_path" "$runner_path"',
            'exec python "$runner_path" worker '
            '--payload "$payload_path" '
            '--output "$PWD/runs/fixed-primary-5t-task-'
            '${SLURM_SCHED_TASK_ID:?missing task id}"',
        ]
    )
    lane = str(task["lane"])
    dedupe = _sha(
        {
            "campaign_id": task["campaign_id"],
            "physics_sha256": task["physics_sha256"],
            "task_payload_sha256": task["payload_sha256"],
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
        }
    )
    return {
        "name": (
            f"mft-5t-{'g1p6' if lane == 'strict' else 'gapvar'}-"
            f"s{seed}-n1-{turns}"
        ),
        "remote_cwd": REMOTE_BUNDLE,
        "command": command,
        "payload_json": dict(task),
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": int(priority),
        "timeout_seconds": TIMEOUT_SECONDS,
        "dedupe_key": f"mft-goal-fixed-primary-5t-{lane}:{dedupe}",
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
    }


def submit(
    *,
    output: Path,
    scheduler_url: str,
    priority: int,
    apply: bool,
    lane: str,
    seed_offset: int = 0,
    seed_count_override: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"campaign output already exists: {output}")
    output.mkdir(parents=True)
    source = _source_bytes()
    warm = _warm_coordinates_by_turns(GLOBAL_TERMINAL_CSV)
    if lane == "strict":
        campaign_id = CAMPAIGN_ID
        hard_spec = HARD_SPEC
        seed_start = SEED_START
        seed_count = SEED_COUNT
    elif lane == "gap-variable":
        campaign_id = HEDGE_CAMPAIGN_ID
        hard_spec = HEDGE_HARD_SPEC
        seed_start = HEDGE_SEED_START
        seed_count = HEDGE_SEED_COUNT
    else:
        raise RuntimeError(f"unsupported lane: {lane}")
    if seed_offset < 0 or seed_offset >= seed_count:
        raise RuntimeError("seed offset is outside the bounded campaign")
    requested_count = (
        seed_count - seed_offset
        if seed_count_override is None
        else int(seed_count_override)
    )
    if requested_count < 1 or seed_offset + requested_count > seed_count:
        raise RuntimeError("requested seed slice is outside the campaign")
    tasks = []
    for index in range(seed_offset, seed_offset + requested_count):
        turns = 5 + index % 4
        task = _task_payload(
            seed=seed_start + index,
            turns=turns,
            source_bytes=source,
            warm_bytes=warm[turns],
            campaign_id=campaign_id,
            lane=lane,
            hard_spec=hard_spec,
        )
        _atomic_json(output / "tasks" / f"seed-{task['seed']}.json", task)
        tasks.append(task)
    scheduler_payloads = [
        _scheduler_payload(task, priority=priority) for task in tasks
    ]
    submissions = []
    if apply:
        for payload in scheduler_payloads:
            response = _api_json(
                scheduler_url.rstrip("/") + "/api/tasks",
                method="POST",
                payload=payload,
            )
            submissions.append(
                {
                    "task_id": int(response["task_id"]),
                    "deduped": bool(response.get("deduped")),
                    "name": payload["name"],
                    "seed": int(payload["payload_json"]["seed"]),
                    "fixed_primary_turns": int(
                        payload["payload_json"]["fixed_primary_turns"]
                    ),
                    "task_payload_sha256": payload["payload_json"][
                        "payload_sha256"
                    ],
                    "physics_sha256": payload["payload_json"][
                        "physics_sha256"
                    ],
                    "submitted_at": _now(),
                }
            )
    manifest = _seal(
        {
            "schema_version": CAMPAIGN_SCHEMA,
            "campaign_id": campaign_id,
            "lane": lane,
            "created_at": _now(),
            "apply": bool(apply),
            "scheduler_url": scheduler_url.rstrip("/"),
            "hard_spec": copy.deepcopy(hard_spec),
            "hard_spec_sha256": _sha(hard_spec),
            "runner_source_sha256": hashlib.sha256(source).hexdigest(),
            "population": POPULATION,
            "generations": GENERATIONS,
            "seed_count": requested_count,
            "seed_start": seed_start + seed_offset,
            "campaign_total_seed_count": seed_count,
            "campaign_seed_offset": seed_offset,
            "turn_strata": [5, 6, 7, 8],
            "scheduler_resources_per_task": {
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "timeout_seconds": TIMEOUT_SECONDS,
                "max_workers_per_node": MAX_WORKERS_PER_NODE,
            },
            "scheduler_project_modified": False,
            "scheduler_project_code_included": False,
            "existing_surrogate_reused": True,
            "warm_source": {
                "path": str(GLOBAL_TERMINAL_CSV),
                "sha256": _sha_file(GLOBAL_TERMINAL_CSV),
                "projected_to_fixed_primary_controls_on_worker": True,
            },
            "task_payload_sha256": [
                task["payload_sha256"] for task in tasks
            ],
            "submissions": submissions,
            "global_nds_ready_artifact": (
                "terminal_physical_candidates.csv per successful task"
            ),
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    _atomic_json(output / "submission_manifest.json", manifest)
    return manifest


def _write_warm(task: Mapping[str, Any], output: Path) -> tuple[Path, str]:
    import numpy as np

    payload = base64.b64decode(task["warm_start_b64"], validate=True)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != task["warm_start_sha256"]:
        raise RuntimeError("warm-start payload SHA-256 mismatch")
    try:
        coordinates = np.asarray(
            json.loads(zlib.decompress(payload).decode("utf-8")),
            dtype=float,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("warm-start coordinate payload is invalid") from exc
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] < POPULATION // 2
        or coordinates.shape[1] != 25
        or not np.isfinite(coordinates).all()
        or np.any(coordinates < 0.0)
        or np.any(coordinates > 1.0)
    ):
        raise RuntimeError("warm-start coordinate array shape is invalid")
    path = output / "warm_start.npy"
    np.save(path, coordinates, allow_pickle=False)
    return path, _sha_file(path)


def worker(*, payload_path: Path, output: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    task = _validate_seal(_read_json(payload_path.resolve(strict=True)), SCHEMA)
    lane = str(task.get("lane") or "")
    expected_campaign = (
        CAMPAIGN_ID if lane == "strict" else HEDGE_CAMPAIGN_ID
    )
    expected_hard_spec = (
        HARD_SPEC if lane == "strict" else HEDGE_HARD_SPEC
    )
    if (
        lane not in {"strict", "gap-variable"}
        or task["campaign_id"] != expected_campaign
        or task["hard_spec"] != expected_hard_spec
        or task["hard_spec_sha256"] != _sha(expected_hard_spec)
        or task["runner_source_sha256"]
        != hashlib.sha256(_source_bytes()).hexdigest()
        or int(task["population"]) != POPULATION
        or int(task["generations"]) != GENERATIONS
    ):
        raise RuntimeError("fixed-primary worker contract mismatch")
    if output.exists():
        raise RuntimeError(f"worker output already exists: {output}")
    output.mkdir(parents=True)

    repository = Path.cwd() / "artifacts" / "code"
    if not repository.is_dir():
        raise RuntimeError("staged 1MW_MFT code root is unavailable")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from module.mft_goal_20260726_contract import cw1_unit_coordinate
    from tools import mft_goal_20260726_launch as goal
    from tools import tier1_corrected_generation_preflight as preflight

    base_task_path = Path.cwd() / _safe_remote_relative(
        task["base_task_relative_path"]
    )
    relocation_path = Path.cwd() / _safe_remote_relative(
        task["base_relocation_relative_path"]
    )
    base_task = goal.validate_task_payload(_read_json(base_task_path))
    if (
        base_task["payload_sha256"] != task["base_task_payload_sha256"]
        or base_task["source_identity"] != task["base_source_identity"]
        or int(base_task["fixed_primary_turns"])
        != int(task["fixed_primary_turns"])
    ):
        raise RuntimeError("authenticated base task identity mismatch")
    source, code_manifest = goal.resolve_worker_source(
        base_task, relocation_path
    )
    runner = preflight.build_authenticated_runner(
        generation=Path(source["generation"]),
        candidate_path=Path(source["candidate"]),
        quality_path=Path(source["quality_status"]),
        code_root=Path(source["code_root"]),
        expected_code_revision=source["expected_code_revision"],
        fixed_primary_turns=int(task["fixed_primary_turns"]),
        stage_spec=base_task["stage_spec"],
        inference_threads=goal.INFERENCE_THREADS,
        dataset_path_override=Path(source["dataset"]),
        profile_path_override=Path(source["profile"]),
        expected_documentary_generation_path=str(
            base_task["source"]["generation"]
        ),
        code_manifest_path=code_manifest,
        expected_code_manifest_payload_sha256=base_task[
            "source_identity"
        ]["code_manifest_payload_sha256"],
        expected_code_inventory_sha256=base_task["source_identity"][
            "code_inventory_sha256"
        ],
    )
    identity = base_task["source_identity"]
    if (
        runner.authenticated.evidence["train_report"]["sha256"]
        != identity["train_report_sha256"]
        or runner.authenticated.evidence["candidate"]["sha256"]
        != identity["candidate_sha256"]
        or runner.authenticated.evidence["dataset"]["sha256"]
        != identity["dataset_sha256"]
        or runner.code_identity["revision"] != identity["code_revision"]
    ):
        raise RuntimeError("authenticated surrogate source changed")

    problem = runner.problem
    axis_stage_spec = copy.deepcopy(problem.stage_spec)
    axis_stage_spec["size_limits_mm"] = copy.deepcopy(
        task["hard_spec"]["size_limits_mm"]
    )
    problem.spec = copy.deepcopy(problem.spec)
    problem.spec["size_limits_mm"] = copy.deepcopy(
        task["hard_spec"]["size_limits_mm"]
    )
    problem.stage_spec = axis_stage_spec
    problem.stage_spec_sha256 = _sha(axis_stage_spec)
    axis_hard_contract = copy.deepcopy(problem.hard_constraint_contract)
    axis_hard_contract["stage"] = (
        "mft-goal-fixed-primary-5t-axis-specific-v1"
    )
    axis_hard_contract["stage_spec_sha256"] = (
        problem.stage_spec_sha256
    )
    axis_hard_contract["size_limits_mm"] = copy.deepcopy(
        task["hard_spec"]["size_limits_mm"]
    )
    axis_hard_contract["axis_contract"] = copy.deepcopy(
        task["hard_spec"]["axis_contract"]
    )
    problem.hard_constraint_contract = axis_hard_contract
    problem.hard_constraint_contract_sha256 = _sha(axis_hard_contract)
    cw_index = int(problem.cw1_coordinate_index)
    gap_index = problem.sobol_dimension_names.index("gap1")
    cw_unit = float(cw1_unit_coordinate(PRIMARY_CONDUCTOR_MM))
    gap_unit = float(
        problem._unit_from_physical("gap1", PRIMARY_GAP_MM)
    )
    problem.xl[cw_index] = cw_unit
    problem.xu[cw_index] = cw_unit
    if lane == "strict":
        problem.xl[gap_index] = gap_unit
        problem.xu[gap_index] = gap_unit

    physical_evaluate, scaling = preflight.install_optimizer_scaling(
        problem,
        resonance_scale_hz=150.0,
        llt_scale_uh=0.55,
        all_thermal_scale_c=10.0,
    )
    runner.physical_evaluate = physical_evaluate
    runner.optimizer_scaling = scaling
    repair = runner.install_offspring_repair_operator(
        topology_contract=preflight.goal_topology_contract(
            int(task["fixed_primary_turns"])
        )
    )
    warm_path, warm_sha = _write_warm(task, output)
    warm_values = np.load(warm_path, allow_pickle=False)
    projected, projection_audit = runner.repair_coordinates(
        warm_values, stage="fixed_primary_warm_projection"
    )
    projected_path = output / "projected_warm_start.npy"
    np.save(projected_path, projected, allow_pickle=False)
    projected_sha = _sha_file(projected_path)
    smoke = runner.evaluate_coordinates(projected[:8], physical=True)
    decoded = smoke["frame"]
    if (
        not np.isclose(
            decoded["cw1"].to_numpy(dtype=float),
            PRIMARY_CONDUCTOR_MM,
            rtol=0.0,
            atol=1e-12,
        ).all()
        or (
            lane == "strict"
            and not np.isclose(
                decoded["gap1"].to_numpy(dtype=float),
                PRIMARY_GAP_MM,
                rtol=0.0,
                atol=1e-12,
            ).all()
        )
    ):
        raise RuntimeError("fixed primary controls escaped projected smoke")
    expected_limits = (
        float(task["hard_spec"]["size_limits_mm"]["W"]),
        float(task["hard_spec"]["size_limits_mm"]["L"]),
        float(task["hard_spec"]["size_limits_mm"]["H"]),
    )
    if expected_limits != (1000.0, 1200.0, 750.0):
        raise RuntimeError("axis-specific size contract drifted")
    for row_index in range(len(decoded)):
        _volume, observed_dimensions = (
            runner.modules.geometry_metrics.bounding_box_lit(
                decoded.iloc[row_index]
            )
        )
        for constraint_name, observed, limit in zip(
            (
                "exterior_width_limit",
                "exterior_length_limit",
                "exterior_height_limit",
            ),
            observed_dimensions,
            expected_limits,
        ):
            constraint_index = problem.constraint_index[constraint_name]
            if not np.isclose(
                float(smoke["G"][row_index, constraint_index]),
                float(observed) - limit,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    "axis-specific physical size constraint mapping escaped"
                )

    def pre_optimization(value: Mapping[str, Any]) -> None:
        _atomic_json(
            output / "pre_optimization.json",
            _seal(
                {
                    "schema_version": (
                        "mft-goal-fixed-primary-5t-pre-optimization-v1"
                    ),
                    "task_payload_sha256": task["payload_sha256"],
                    "hard_spec_sha256": task["hard_spec_sha256"],
                    "physics_sha256": task["physics_sha256"],
                    "effective_stage_spec_sha256": (
                        problem.stage_spec_sha256
                    ),
                    "effective_hard_constraint_contract_sha256": (
                        problem.hard_constraint_contract_sha256
                    ),
                    "axis_contract": copy.deepcopy(
                        task["hard_spec"]["axis_contract"]
                    ),
                    "bounds": {
                        "cw1": {
                            "coordinate_index": cw_index,
                            "unit": cw_unit,
                            "physical_mm": PRIMARY_CONDUCTOR_MM,
                        },
                        "gap1": (
                            {
                                "coordinate_index": gap_index,
                                "unit": gap_unit,
                                "physical_mm": PRIMARY_GAP_MM,
                                "hard_fixed": True,
                            }
                            if lane == "strict"
                            else {
                                "coordinate_index": gap_index,
                                "hard_fixed": False,
                                "search_mm": [0.3, 5.0],
                            }
                        ),
                    },
                    "warm_source_sha256": warm_sha,
                    "projected_warm_sha256": projected_sha,
                    "warm_projection_audit": projection_audit,
                    "optimizer_repair_installation": repair,
                    "base_optimizer_evidence": copy.deepcopy(dict(value)),
                }
            ),
        )

    result = runner.run_one(
        seed=int(task["seed"]),
        population=POPULATION,
        max_generations=GENERATIONS,
        warm_start_path=projected_path,
        warm_start_sha256=projected_sha,
        optimizer_termination_strategy=(
            preflight.FIXED_GENERATION_TERMINATION_STRATEGY
        ),
        pre_optimization_callback=pre_optimization,
    )
    artifacts = preflight.persist_search_outputs(
        runner,
        result,
        output,
        source_identity={
            "seed": int(task["seed"]),
            "task_id": str(
                os.environ.get("SLURM_SCHED_TASK_ID")
                or f"local-{task['seed']}"
            ),
            "bundle_id": task["payload_sha256"],
            "island_id": f"fixed5t-n1-{task['fixed_primary_turns']}",
            "dataset_sha256": identity["dataset_sha256"],
            "model_artifacts_sha256": identity["evaluation_model_sha256"],
            "model_generation_sha256": identity["train_report_sha256"],
            "evaluation_spec_sha256": problem.stage_spec_sha256,
            "temperature_contract_sha256": (
                problem.temperature_contract_sha256
            ),
            "hard_constraint_contract_sha256": (
                problem.hard_constraint_contract_sha256
            ),
        },
    )
    terminal_path = output / "terminal_physical_candidates.csv"
    terminal = pd.read_csv(
        terminal_path,
        usecols=["decoded_physical_params_json"],
    )
    escaped = []
    for index, raw in enumerate(terminal["decoded_physical_params_json"]):
        params = json.loads(raw)
        if (
            float(params["cw1"]) != PRIMARY_CONDUCTOR_MM
            or (
                lane == "strict"
                and float(params["gap1"]) != PRIMARY_GAP_MM
            )
        ):
            escaped.append(index)
    if escaped:
        raise RuntimeError("terminal population escaped fixed primary controls")

    artifact_paths = {
        "terminal_physical_candidates.csv": terminal_path,
        "terminal_physical_candidates.manifest.json": (
            output / "terminal_physical_candidates.manifest.json"
        ),
        "terminal_X.npy": output / "terminal_X.npy",
        "terminal_F.npy": output / "terminal_F.npy",
        "terminal_G_physical.npy": output / "terminal_G_physical.npy",
    }
    inventory = {
        name: {
            "sha256": _sha_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    value = _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "campaign_id": task["campaign_id"],
            "lane": lane,
            "completed_at": _now(),
            "task_payload_sha256": task["payload_sha256"],
            "physics_sha256": task["physics_sha256"],
            "hard_spec": copy.deepcopy(task["hard_spec"]),
            "hard_spec_sha256": task["hard_spec_sha256"],
            "seed": int(task["seed"]),
            "fixed_primary_turns": int(task["fixed_primary_turns"]),
            "population": POPULATION,
            "evaluated_generations": int(
                result.tier1_evaluated_generations
            ),
            "completed_generations": int(
                result.tier1_completed_generations
            ),
            "terminal_population_count": artifacts[
                "terminal_population_count"
            ],
            "physical_feasible_count": artifacts[
                "physical_feasible_count"
            ],
            "feasible_pareto_count": artifacts["feasible_pareto_count"],
            "terminal_primary_controls_attested": True,
            "terminal_gap1_hard_fixed": lane == "strict",
            "terminal_primary_control_escape_count": 0,
            "global_nds_ready": True,
            "artifact_inventory": inventory,
            "base_source_identity": copy.deepcopy(identity),
            "scheduler_project_modified": False,
            "aedt_used": False,
            "fea_submission_performed": False,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        }
    )
    _atomic_json(output / "result.json", value)
    print("RESULT_JSON=" + json.dumps(value, separators=(",", ":")))
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("submit")
    launch.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    launch.add_argument("--scheduler-url", default=SCHEDULER_URL)
    launch.add_argument("--priority", type=int, default=40)
    launch.add_argument(
        "--lane",
        choices=("strict", "gap-variable"),
        default="strict",
    )
    launch.add_argument("--seed-offset", type=int, default=0)
    launch.add_argument("--seed-count", type=int)
    launch.add_argument("--apply", action="store_true")
    execute = commands.add_parser("worker")
    execute.add_argument("--payload", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "submit":
        value = submit(
            output=args.output.resolve(),
            scheduler_url=args.scheduler_url,
            priority=args.priority,
            apply=args.apply,
            lane=args.lane,
            seed_offset=args.seed_offset,
            seed_count_override=args.seed_count,
        )
    else:
        value = worker(
            payload_path=args.payload,
            output=args.output.resolve(),
        )
    if args.command == "submit":
        print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
