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
CAMPAIGN_ID = "mft-goal-fixed-primary-5t-gap1-1p6-axis-v6"
HEDGE_CAMPAIGN_ID = "mft-goal-fixed-primary-5t-gap1-variable-20260727-v1"
TARGETED_CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-splittemp-v2"
)
PRIMARY_CONDUCTOR_MM = 5.0
PRIMARY_GAP_MM = 1.6
POPULATION = 320
GENERATIONS = 80
SEED_START = 2_707_275_700
SEED_COUNT = 16
HEDGE_SEED_START = 2_707_275_400
HEDGE_SEED_COUNT = 8
TARGETED_SEED_START = 2_707_277_000
TARGETED_SEED_COUNT = 64
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
TARGETED_DIAGNOSTIC_CSV = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_w1200_l1000_fixed_lm2mh_rescore"
    r"\all_terminal_rescored_5120.csv"
)
OFFICIAL5_SOURCE_CSV = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\slurm_offload\mft-goal-763dbb46ae74e1722969d2c2"
    r"\harvest\task-95913\terminal_physical_candidates.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_nsga_v6"
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
FIXED_LM2MH_RESONANCE_CONTRACT = {
    "schema_version": "mft-goal-fixed-lm2mh-resonance-contract-v1",
    "Lm_primary_referred_H": 0.002,
    "Llt_phys_source_unit": "uH",
    "C_prediction_basis": "full-transformer-restored-F",
    "formulas": {
        "Llt_H": "Llt_phys_uH*1e-6",
        "Ltx_H": "0.002+Llt_H",
        "ratio": "N2/N1",
        "Lrx_H": "Ltx_H*ratio**2",
        "fTx_Hz": "1/(2*pi*sqrt(Ltx_H*C_tx_tx_F))",
        "fRx_Hz": "1/(2*pi*sqrt(Lrx_H*C_rx_rx_F))",
        "fInter_Hz": (
            "1/(2*pi*sqrt(Llt_H*C_tx_rx_F)); diagnostic-only"
        ),
        "hard_gate": "min(fTx_Hz,fRx_Hz)>=15000",
    },
    "geometry_predicted_Lm_used": False,
    "explicit_air_gap_tuning_assumed": True,
    "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": False,
    "classification": "screening-only",
}
TARGETED_HARD_SPEC = {
    **{
        key: copy.deepcopy(value)
        for key, value in HARD_SPEC.items()
        if key != "winding_temperature_max_C"
    },
    "schema_version": "mft-goal-fixed-primary-5t-axis-w1200-l1000-lm2mh-v1",
    "size_limits_mm": {"W": 1200.0, "L": 1000.0, "H": 750.0},
    "axis_contract": {
        "W": "drawing_x_original_973mm_direction",
        "L": "drawing_y_perpendicular_direction",
        "rotation_or_axis_swap_allowed": False,
    },
    "magnetizing_inductance_H": 0.002,
    "magnetizing_inductance_basis": "full-physical-primary-referred",
    "magnetizing_inductance_tuning": "explicit-air-gap",
    "temperature_family_limits_C": {
        "primary_winding": 100.0,
        "secondary_winding": 120.0,
        "core": 120.0,
    },
    "legacy_scalar_winding_temperature_limit_allowed": False,
    "resonance_contract": copy.deepcopy(FIXED_LM2MH_RESONANCE_CONTRACT),
    "search_strategy": {
        "diagnostic_recenter_expand": True,
        "official5_projected_anchor_required": True,
        "fresh_random_fraction": 0.5,
        "prior_objectives_or_constraints_inherited": False,
    },
}
SECONDARY_TEMPERATURE_CONSTRAINTS = (
    "temperature_robust_limit:T_max_Rx_main",
    "temperature_robust_limit:T_max_Rx_side",
    "temperature_robust_limit:Tprobe_Rx_main_leeward_max",
    "temperature_robust_limit:Tprobe_Rx_side_leeward_max",
)
SECONDARY_TEMPERATURE_ALLOWANCE_C = 20.0


def _apply_targeted_split_temperature_constraints(
    physical_g: Any,
    *,
    constraint_index: Mapping[str, int],
    valid_indices: Any,
) -> Any:
    """Apply the authorized 120 C secondary limit inside NSGA evaluation.

    The authenticated base problem expresses every winding constraint against
    the historical 100 C scalar limit.  Primary constraints therefore remain
    unchanged, while the four secondary constraints receive the exact 20 C
    allowance before optimizer scaling and non-dominated selection.
    """

    for name in SECONDARY_TEMPERATURE_CONSTRAINTS:
        if name not in constraint_index:
            raise RuntimeError(
                f"split-temperature constraint is missing from NSGA: {name}"
            )
        physical_g[valid_indices, int(constraint_index[name])] -= (
            SECONDARY_TEMPERATURE_ALLOWANCE_C
        )
    return physical_g


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


def _targeted_diagnostic_coordinates_by_turns(
    source: Path,
) -> dict[int, dict[str, Any]]:
    """Build diverse coordinate-only anchors for the fixed-Lm targeted lane."""

    rows_by_turns: dict[int, list[dict[str, Any]]] = {
        turns: [] for turns in BASE_TASKS
    }
    with source.resolve(strict=True).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            turns = int(row["N1_fixed_lm2mh"])
            if (
                turns in rows_by_turns
                and str(row["decoder_valid"]).lower() == "true"
            ):
                rows_by_turns[turns].append(row)

    official_coordinate = None
    official_geometry = None
    with OFFICIAL5_SOURCE_CSV.resolve(strict=True).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if row["physical_geometry_sha256"].startswith("909d249ebe45"):
                official_geometry = row["physical_geometry_sha256"]
                official_coordinate = [
                    float(value)
                    for value in json.loads(row["coordinate_unit_json"])
                ]
                break
    if official_coordinate is None or len(official_coordinate) != 25:
        raise RuntimeError("official#5 projected diagnostic anchor is missing")
    # Hard-project the old official#5 manufacturing escape before any worker
    # repair: f1_split -> cw1=5 mm, gap1 -> 1.6 mm.
    official_coordinate[19] = (5.0 - 1.0) / (10.0 - 1.0)
    official_coordinate[20] = (1.6 - 0.3) / (5.0 - 0.3)

    result: dict[int, dict[str, Any]] = {}
    criteria = (
        lambda row: float(
            row["normalized_constraint_violation_l2_fixed_lm2mh"]
        ),
        lambda row: float(
            json.loads(row["normalized_G_fixed_lm2mh_json"])[
                "Llt_robust_band"
            ]
        ),
        lambda row: float(
            json.loads(row["normalized_G_fixed_lm2mh_json"])[
                "Llt_ensemble_disagreement"
            ]
        ),
        lambda row: float(
            json.loads(row["normalized_G_fixed_lm2mh_json"])[
                "temperature_robust_limit:T_max_Rx_main"
            ]
        ),
        lambda row: float(
            json.loads(row["normalized_G_fixed_lm2mh_json"])[
                "temperature_robust_limit:T_max_Tx"
            ]
        ),
        lambda row: -float(row["resonance_min_Hz_fixed_lm2mh"]),
    )
    for turns, candidates in rows_by_turns.items():
        ranked = [
            sorted(
                candidates,
                key=lambda row, criterion=criterion: (
                    criterion(row),
                    float(row["objective_volume_L"]),
                    row["physical_geometry_sha256"],
                ),
            )
            for criterion in criteria
        ]
        selected: list[list[float]] = []
        selected_geometry: list[str] = []
        seen = set()
        if turns == 6:
            selected.append(list(official_coordinate))
            selected_geometry.append(str(official_geometry))
            seen.add(str(official_geometry))
        cursor = 0
        while len(selected) < 80:
            added = False
            for ranking in ranked:
                if cursor >= len(ranking):
                    continue
                row = ranking[cursor]
                identity = row["physical_geometry_sha256"]
                if identity in seen:
                    continue
                coordinate = [
                    float(value)
                    for value in json.loads(row["coordinate_unit_json"])
                ]
                if len(coordinate) != 25:
                    raise RuntimeError("targeted diagnostic coordinate width drifted")
                selected.append(coordinate)
                selected_geometry.append(identity)
                seen.add(identity)
                added = True
                if len(selected) == 80:
                    break
            cursor += 1
            if not added and cursor >= max(len(ranking) for ranking in ranked):
                break
        if len(selected) != 80:
            raise RuntimeError(f"N1={turns} targeted diagnostic pool underfilled")
        packed = zlib.compress(
            json.dumps(
                selected, separators=(",", ":"), allow_nan=False
            ).encode("utf-8"),
            level=9,
        )
        result[turns] = {
            "compressed": packed,
            "compressed_sha256": hashlib.sha256(packed).hexdigest(),
            "coordinate_sha256": _sha(selected),
            "geometry_sha256": selected_geometry,
            "official5_projected_included": turns == 6,
            "source_csv": str(source),
            "source_csv_sha256": _sha_file(source),
            "selection": (
                "round-robin least overall/Llt/disagreement/Rx/Tx/"
                "fixed-Lm resonance with geometry dedupe"
            ),
        }
    return result


def _task_payload(
    *,
    seed: int,
    turns: int,
    source_bytes: bytes,
    campaign_id: str,
    lane: str,
    hard_spec: Mapping[str, Any],
    initialization: str = "cold-random-axis-contract-v1",
    targeted_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hard_spec_sha = _sha(hard_spec)
    runner_sha = hashlib.sha256(source_bytes).hexdigest()
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
        "initialization": initialization,
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
            "initialization": initialization,
            "global_nds_artifact_required": True,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
            **(
                {}
                if targeted_diagnostic is None
                else {
                    "targeted_diagnostic_b64": base64.b64encode(
                        targeted_diagnostic["compressed"]
                    ).decode("ascii"),
                    "targeted_diagnostic_compressed_sha256": (
                        targeted_diagnostic["compressed_sha256"]
                    ),
                    "targeted_diagnostic_coordinate_sha256": (
                        targeted_diagnostic["coordinate_sha256"]
                    ),
                    "targeted_diagnostic_geometry_sha256": list(
                        targeted_diagnostic["geometry_sha256"]
                    ),
                    "targeted_diagnostic_source_csv": (
                        targeted_diagnostic["source_csv"]
                    ),
                    "targeted_diagnostic_source_csv_sha256": (
                        targeted_diagnostic["source_csv_sha256"]
                    ),
                    "official5_projected_included": (
                        targeted_diagnostic[
                            "official5_projected_included"
                        ]
                    ),
                    "targeted_diagnostic_selection": (
                        targeted_diagnostic["selection"]
                    ),
                }
            ),
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
            '--output "$PWD/runs/'
            + (
                "fixed-lm2mh-targeted-task-"
                if task["lane"] == "fixed-lm-targeted"
                else "fixed-primary-5t-task-"
            )
            +
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
            f"mft-5t-"
            f"{'g1p6' if lane == 'strict' else ('gapvar' if lane == 'gap-variable' else 'lm2')}-"
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
        targeted_diagnostics = None
        initialization = "cold-random-axis-contract-v1"
    elif lane == "fixed-lm-targeted":
        campaign_id = TARGETED_CAMPAIGN_ID
        hard_spec = TARGETED_HARD_SPEC
        seed_start = TARGETED_SEED_START
        seed_count = TARGETED_SEED_COUNT
        targeted_diagnostics = _targeted_diagnostic_coordinates_by_turns(
            TARGETED_DIAGNOSTIC_CSV
        )
        initialization = "diagnostic-anchor-jitter-plus-fresh-random-v1"
    else:
        raise RuntimeError(f"unsupported lane: {lane}")
    if lane == "strict":
        targeted_diagnostics = None
        initialization = "cold-random-axis-contract-v1"
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
            campaign_id=campaign_id,
            lane=lane,
            hard_spec=hard_spec,
            initialization=initialization,
            targeted_diagnostic=(
                None
                if targeted_diagnostics is None
                else targeted_diagnostics[turns]
            ),
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
            "initialization": {
                "mode": initialization,
                "reason": (
                    "old-axis warm pool has no hard-feasible W<=1000 design"
                    if lane != "fixed-lm-targeted"
                    else (
                        "80 diverse re-evaluated coordinate anchors plus "
                        "80 local jitters plus 160 full-range fresh random"
                    )
                ),
                "old_objectives_or_constraints_inherited": False,
                "diagnostic_coordinates_re_evaluated": (
                    lane == "fixed-lm-targeted"
                ),
                "official5_projected_anchor_in_N1_6": (
                    lane == "fixed-lm-targeted"
                ),
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
    expected_campaign = {
        "strict": CAMPAIGN_ID,
        "gap-variable": HEDGE_CAMPAIGN_ID,
        "fixed-lm-targeted": TARGETED_CAMPAIGN_ID,
    }.get(lane)
    expected_hard_spec = {
        "strict": HARD_SPEC,
        "gap-variable": HEDGE_HARD_SPEC,
        "fixed-lm-targeted": TARGETED_HARD_SPEC,
    }.get(lane)
    expected_initialization = (
        "diagnostic-anchor-jitter-plus-fresh-random-v1"
        if lane == "fixed-lm-targeted"
        else "cold-random-axis-contract-v1"
    )
    if (
        lane not in {"strict", "gap-variable", "fixed-lm-targeted"}
        or task["campaign_id"] != expected_campaign
        or task["hard_spec"] != expected_hard_spec
        or task["hard_spec_sha256"] != _sha(expected_hard_spec)
        or task["runner_source_sha256"]
        != hashlib.sha256(_source_bytes()).hexdigest()
        or int(task["population"]) != POPULATION
        or int(task["generations"]) != GENERATIONS
        or task.get("initialization") != expected_initialization
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
        (
            "mft-goal-fixed-primary-5t-axis-w1200-l1000-lm2mh-v1"
            if lane == "fixed-lm-targeted"
            else "mft-goal-fixed-primary-5t-axis-specific-v1"
        )
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
    if lane == "fixed-lm-targeted":
        axis_stage_spec["magnetizing_inductance_H"] = 0.002
        axis_stage_spec["resonance_contract"] = copy.deepcopy(
            FIXED_LM2MH_RESONANCE_CONTRACT
        )
        problem.stage_spec = axis_stage_spec
        problem.stage_spec_sha256 = _sha(axis_stage_spec)
        axis_hard_contract["stage_spec_sha256"] = (
            problem.stage_spec_sha256
        )
        axis_hard_contract["fixed_lm2mh_resonance_contract"] = (
            copy.deepcopy(FIXED_LM2MH_RESONANCE_CONTRACT)
        )
        axis_hard_contract["fixed_lm2mh_resonance_contract_sha256"] = (
            _sha(FIXED_LM2MH_RESONANCE_CONTRACT)
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
    if lane in {"strict", "fixed-lm-targeted"}:
        problem.xl[gap_index] = gap_unit
        problem.xu[gap_index] = gap_unit

    if lane == "fixed-lm-targeted":
        original_physical_evaluate = problem._evaluate
        resonance_index = problem.constraint_index[
            "half_magnetizing_resonance_minimum"
        ]

        def fixed_lm2mh_physical_evaluate(
            values: Any,
            out: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            original_physical_evaluate(values, out, *args, **kwargs)
            physical_g = np.asarray(out["G"], dtype=float)
            valid = np.asarray(out["decoder_valid"], dtype=bool)
            frame = out["frame"]
            indices = np.flatnonzero(valid)
            if len(indices):
                sub = frame.iloc[indices]
                means = {}
                for target in ("Llt_phys", "C_tx_tx_F", "C_rx_rx_F"):
                    mean, _half_width = runner.models[
                        target
                    ].predict_mu_sigma(sub, conformal=True)
                    means[target] = np.asarray(mean, dtype=float).reshape(-1)
                n1 = sub["N1"].to_numpy(dtype=float)
                n2 = sub["N2"].to_numpy(dtype=float)
                llt_h = means["Llt_phys"] * 1e-6
                ltx_h = 0.002 + llt_h
                lrx_h = ltx_h * np.square(n2 / n1)
                positive = (
                    (llt_h > 0.0)
                    & (ltx_h > 0.0)
                    & (lrx_h > 0.0)
                    & (means["C_tx_tx_F"] > 0.0)
                    & (means["C_rx_rx_F"] > 0.0)
                    & (n1 > 0.0)
                    & (n2 > 0.0)
                )
                frequency = np.full(len(indices), 0.0, dtype=float)
                frequency[positive] = np.minimum(
                    1.0
                    / (
                        2.0
                        * np.pi
                        * np.sqrt(
                            ltx_h[positive]
                            * means["C_tx_tx_F"][positive]
                        )
                    ),
                    1.0
                    / (
                        2.0
                        * np.pi
                        * np.sqrt(
                            lrx_h[positive]
                            * means["C_rx_rx_F"][positive]
                        )
                    ),
                )
                physical_g[indices, resonance_index] = (
                    15_000.0 - frequency
                )
                physical_g = _apply_targeted_split_temperature_constraints(
                    physical_g,
                    constraint_index=problem.constraint_index,
                    valid_indices=indices,
                )
            out["G"] = physical_g

        problem._evaluate = fixed_lm2mh_physical_evaluate

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
    raw_smoke = np.random.default_rng(
        int(task["seed"]) ^ 0x5A17
    ).random((8, int(problem.n_var)))
    projected, projection_audit = runner.repair_coordinates(
        raw_smoke, stage="fixed_primary_cold_smoke_projection"
    )
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
            lane in {"strict", "fixed-lm-targeted"}
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
    required_limits = (
        (1200.0, 1000.0, 750.0)
        if lane == "fixed-lm-targeted"
        else (1000.0, 1200.0, 750.0)
    )
    if expected_limits != required_limits:
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

    targeted_initial = None
    targeted_initialization_audit = None
    if lane == "fixed-lm-targeted":
        packed = base64.b64decode(
            task["targeted_diagnostic_b64"], validate=True
        )
        if (
            hashlib.sha256(packed).hexdigest()
            != task["targeted_diagnostic_compressed_sha256"]
        ):
            raise RuntimeError("targeted diagnostic payload SHA mismatch")
        diagnostic = np.asarray(
            json.loads(zlib.decompress(packed).decode("utf-8")),
            dtype=float,
        )
        if (
            diagnostic.shape != (80, int(problem.n_var))
            or not np.isfinite(diagnostic).all()
            or np.any(diagnostic < 0.0)
            or np.any(diagnostic > 1.0)
            or _sha(diagnostic.tolist())
            != task["targeted_diagnostic_coordinate_sha256"]
        ):
            raise RuntimeError("targeted diagnostic coordinate contract mismatch")
        anchors, anchor_repair = runner.repair_coordinates(
            diagnostic, stage="fixed_lm2mh_diagnostic_anchor_projection"
        )
        anchor_decode = runner.evaluate_coordinates(
            anchors[:1], physical=True
        )
        anchor_row = anchor_decode["frame"].iloc[0]
        if (
            not np.isclose(
                float(anchor_row["cw1"]), 5.0, rtol=0.0, atol=1e-12
            )
            or not np.isclose(
                float(anchor_row["gap1"]), 1.6, rtol=0.0, atol=1e-12
            )
        ):
            raise RuntimeError("targeted projected anchor escaped 5T/1.6")
        if (
            int(task["fixed_primary_turns"]) == 6
            and task.get("official5_projected_included") is not True
        ):
            raise RuntimeError("N1=6 targeted lane omitted official#5 anchor")

        target_rng = np.random.default_rng(int(task["seed"]) ^ 0x2A11)
        jitter_scale = np.full(int(problem.n_var), 0.04, dtype=float)
        jitter_scale[3:8] = 0.10
        jitter_scale[10:19] = 0.12
        jitter_scale[21] = 0.18
        jitter_scale[22:25] = 0.10
        jittered_raw = anchors + target_rng.normal(
            0.0, jitter_scale, size=anchors.shape
        )
        jittered_raw = np.minimum(
            np.maximum(jittered_raw, problem.xl), problem.xu
        )
        jittered, jitter_repair = runner.repair_coordinates(
            jittered_raw, stage="fixed_lm2mh_expanded_local_jitter"
        )
        fresh_raw = problem.xl + target_rng.random(
            (160, int(problem.n_var))
        ) * (problem.xu - problem.xl)
        fresh, fresh_repair = runner.repair_coordinates(
            fresh_raw, stage="fixed_lm2mh_full_range_fresh_random"
        )
        targeted_initial = np.vstack([anchors, jittered, fresh])
        targeted_initial, combined_repair = runner.repair_coordinates(
            targeted_initial,
            stage="fixed_lm2mh_combined_initial_population",
        )
        if targeted_initial.shape != (POPULATION, int(problem.n_var)):
            raise RuntimeError("targeted initial population shape mismatch")
        targeted_initialization_audit = {
            "schema_version": (
                "mft-goal-fixed-lm2mh-targeted-initialization-v1"
            ),
            "population": POPULATION,
            "exact_reprojected_diagnostic_anchor_count": 80,
            "expanded_local_jitter_count": 80,
            "full_range_fresh_random_count": 160,
            "diagnostic_coordinates_re_evaluated": True,
            "prior_objectives_or_constraints_inherited": False,
            "official5_projected_anchor_included": bool(
                task.get("official5_projected_included")
            ),
            "official5_projected_anchor_first_row": (
                int(task["fixed_primary_turns"]) == 6
            ),
            "projected_anchor_smoke": {
                "cw1_mm": float(anchor_row["cw1"]),
                "gap1_mm": float(anchor_row["gap1"]),
                "N1": int(anchor_row["N1"]),
                "physical_G": {
                    name: float(anchor_decode["G"][0, position])
                    for position, name in enumerate(
                        problem.constraint_names
                    )
                },
            },
            "jitter_scale_by_coordinate": jitter_scale.tolist(),
            "anchor_repair": anchor_repair,
            "jitter_repair": jitter_repair,
            "fresh_repair": fresh_repair,
            "combined_repair": combined_repair,
            "coordinate_sha256": _sha(targeted_initial.tolist()),
        }
        targeted_initialization_audit["payload_sha256"] = _sha(
            targeted_initialization_audit
        )

    def pre_optimization(value: Mapping[str, Any]) -> None:
        initialization = (
            (value.get("initial_population") or {}).get(
                "current_run_nsga2"
            )
            or {}
        )
        if (
            value.get("authenticated_warm_start") is not None
            or initialization.get("population") != POPULATION
            or initialization.get("authenticated_warm_injected_count") != 0
            or initialization.get("fresh_random_count") != POPULATION
        ):
            raise RuntimeError(
                "axis-v6 optimizer did not preserve cold random initialization"
            )
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
                            if lane in {"strict", "fixed-lm-targeted"}
                            else {
                                "coordinate_index": gap_index,
                                "hard_fixed": False,
                                "search_mm": [0.3, 5.0],
                            }
                        ),
                    },
                    "initialization": expected_initialization,
                    "cold_smoke_projection_audit": projection_audit,
                    "optimizer_repair_installation": repair,
                    "base_optimizer_evidence": copy.deepcopy(dict(value)),
                }
            ),
        )

    if lane == "fixed-lm-targeted":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        if targeted_initial is None or targeted_initialization_audit is None:
            raise RuntimeError("targeted initial population was not prepared")
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
                    "fixed_lm2mh_resonance_contract_sha256": _sha(
                        FIXED_LM2MH_RESONANCE_CONTRACT
                    ),
                    "axis_contract": copy.deepcopy(
                        task["hard_spec"]["axis_contract"]
                    ),
                    "initialization": targeted_initialization_audit,
                    "optimizer_repair_installation": repair,
                    "screening_only": True,
                }
            ),
        )
        algorithm = NSGA2(
            pop_size=POPULATION,
            sampling=targeted_initial,
            repair=runner.prepared_repair_operator,
            eliminate_duplicates=True,
        )
        result = minimize(
            problem,
            algorithm,
            ("n_gen", GENERATIONS),
            seed=int(task["seed"]),
            verbose=False,
            save_history=False,
        )
        terminal = getattr(result, "pop", None)
        if terminal is None:
            terminal = getattr(
                getattr(result, "algorithm", None), "pop", None
            )
        if terminal is None or not callable(getattr(terminal, "get", None)):
            raise RuntimeError("targeted optimizer returned no terminal population")
        terminal_x = np.asarray(terminal.get("X"), dtype=float)
        terminal_f = np.asarray(terminal.get("F"), dtype=float)
        terminal_g = np.asarray(terminal.get("G"), dtype=float)
        problem._tier1_optimizer_generation = GENERATIONS
        replay, replay_audit = runner.terminal_physical_replay(
            terminal_x,
            expected_f=terminal_f,
            expected_g=terminal_g,
            terminal_population=terminal,
        )
        if (
            replay_audit.get("terminal_population_canonicalized") is not True
            or int(result.algorithm.n_gen) != GENERATIONS + 1
        ):
            raise RuntimeError("targeted terminal replay/counter audit failed")
        result.tier1_terminal_physical_replay = replay
        result.tier1_evaluated_generations = GENERATIONS
        result.tier1_completed_generations = int(result.algorithm.n_gen)
    else:
        result = runner.run_one(
            seed=int(task["seed"]),
            population=POPULATION,
            max_generations=GENERATIONS,
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
                lane in {"strict", "fixed-lm-targeted"}
                and float(params["gap1"]) != PRIMARY_GAP_MM
            )
        ):
            escaped.append(index)
    if escaped:
        raise RuntimeError("terminal population escaped fixed primary controls")

    fixed_lm_terminal_path = None
    fixed_lm_manifest_path = None
    fixed_lm_feasible_count = None
    if lane == "fixed-lm-targeted":
        fixed_lm_terminal = pd.read_csv(terminal_path)
        decoded_frame = pd.DataFrame(
            [
                json.loads(raw)
                for raw in fixed_lm_terminal[
                    "decoded_physical_params_json"
                ]
            ]
        )
        predictions = {}
        half_widths = {}
        for target in (
            "Llt_phys",
            "C_tx_tx_F",
            "C_rx_rx_F",
            "C_tx_rx_F",
        ):
            mean, half_width = runner.models[target].predict_mu_sigma(
                decoded_frame, conformal=True
            )
            predictions[target] = np.asarray(mean, dtype=float).reshape(-1)
            half_widths[target] = np.asarray(
                half_width, dtype=float
            ).reshape(-1)
        llt_h = predictions["Llt_phys"] * 1e-6
        ltx_h = 0.002 + llt_h
        ratio = (
            decoded_frame["N2"].to_numpy(dtype=float)
            / decoded_frame["N1"].to_numpy(dtype=float)
        )
        lrx_h = ltx_h * np.square(ratio)
        f_tx = 1.0 / (
            2.0
            * np.pi
            * np.sqrt(ltx_h * predictions["C_tx_tx_F"])
        )
        f_rx = 1.0 / (
            2.0
            * np.pi
            * np.sqrt(lrx_h * predictions["C_rx_rx_F"])
        )
        f_inter = 1.0 / (
            2.0
            * np.pi
            * np.sqrt(llt_h * predictions["C_tx_rx_F"])
        )
        f_min = np.minimum(f_tx, f_rx)
        physical_g_fixed = []
        new_feasible = []
        for row_index, raw_g in enumerate(
            fixed_lm_terminal["physical_G_json"]
        ):
            physical_g = json.loads(raw_g)
            observed = float(
                physical_g["half_magnetizing_resonance_minimum"]
            )
            expected = 15_000.0 - float(f_min[row_index])
            if not np.isclose(
                observed, expected, rtol=0.0, atol=1e-7
            ):
                raise RuntimeError(
                    "terminal fixed-Lm resonance G replay mismatch"
                )
            physical_g.pop("half_magnetizing_resonance_minimum")
            physical_g[
                "fixed_Lm2mH_self_resonance_minimum"
            ] = expected
            new_feasible.append(
                str(
                    fixed_lm_terminal.iloc[row_index][
                        "decoder_valid"
                    ]
                ).lower()
                == "true"
                and str(
                    fixed_lm_terminal.iloc[row_index][
                        "surrogate_physical_valid"
                    ]
                ).lower()
                == "true"
                and all(float(value) <= 0.0 for value in physical_g.values())
            )
            physical_g_fixed.append(
                json.dumps(
                    physical_g,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        contract_sha = _sha(FIXED_LM2MH_RESONANCE_CONTRACT)
        fixed_lm_terminal[
            "fixed_lm2mh_resonance_contract_sha256"
        ] = contract_sha
        fixed_lm_terminal["pred_Llt_phys_uH_fixed_lm2mh"] = (
            predictions["Llt_phys"]
        )
        fixed_lm_terminal["pred_C_tx_tx_F_fixed_lm2mh"] = (
            predictions["C_tx_tx_F"]
        )
        fixed_lm_terminal["pred_C_rx_rx_F_fixed_lm2mh"] = (
            predictions["C_rx_rx_F"]
        )
        fixed_lm_terminal["pred_C_tx_rx_F_fixed_lm2mh"] = (
            predictions["C_tx_rx_F"]
        )
        fixed_lm_terminal["fTx_Hz_fixed_lm2mh"] = f_tx
        fixed_lm_terminal["fRx_Hz_fixed_lm2mh"] = f_rx
        fixed_lm_terminal[
            "fInter_Hz_fixed_lm2mh_diagnostic"
        ] = f_inter
        fixed_lm_terminal["resonance_min_Hz_fixed_lm2mh"] = f_min
        fixed_lm_terminal["physical_G_fixed_lm2mh_json"] = (
            physical_g_fixed
        )
        fixed_lm_terminal["screening_feasible_fixed_lm2mh"] = (
            new_feasible
        )
        fixed_lm_terminal["screening_only_fixed_lm2mh"] = True
        fixed_lm_terminal["production_eligible_fixed_lm2mh"] = False
        fixed_lm_terminal_path = (
            output / "terminal_fixed_lm2mh_candidates.csv"
        )
        fixed_lm_terminal.to_csv(fixed_lm_terminal_path, index=False)
        fixed_lm_feasible_count = int(sum(new_feasible))
        fixed_lm_manifest_path = (
            output / "terminal_fixed_lm2mh_candidates.manifest.json"
        )
        _atomic_json(
            fixed_lm_manifest_path,
            _seal(
                {
                    "schema_version": (
                        "mft-goal-fixed-lm2mh-terminal-candidates-v1"
                    ),
                    "task_payload_sha256": task["payload_sha256"],
                    "row_count": int(len(fixed_lm_terminal)),
                    "screening_feasible_count": fixed_lm_feasible_count,
                    "resonance_contract": copy.deepcopy(
                        FIXED_LM2MH_RESONANCE_CONTRACT
                    ),
                    "resonance_contract_sha256": contract_sha,
                    "csv_sha256": _sha_file(fixed_lm_terminal_path),
                    "thermal_screening_only": True,
                    "production_eligible": False,
                }
            ),
        )

    artifact_paths = {
        "terminal_physical_candidates.csv": terminal_path,
        "terminal_physical_candidates.manifest.json": (
            output / "terminal_physical_candidates.manifest.json"
        ),
        "terminal_X.npy": output / "terminal_X.npy",
        "terminal_F.npy": output / "terminal_F.npy",
        "terminal_G_physical.npy": output / "terminal_G_physical.npy",
        **(
            {}
            if fixed_lm_terminal_path is None
            else {
                "terminal_fixed_lm2mh_candidates.csv": (
                    fixed_lm_terminal_path
                ),
                "terminal_fixed_lm2mh_candidates.manifest.json": (
                    fixed_lm_manifest_path
                ),
            }
        ),
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
            "terminal_gap1_hard_fixed": lane in {
                "strict",
                "fixed-lm-targeted",
            },
            "terminal_primary_control_escape_count": 0,
            "global_nds_ready": True,
            "fixed_lm2mh_resonance_contract": (
                copy.deepcopy(FIXED_LM2MH_RESONANCE_CONTRACT)
                if lane == "fixed-lm-targeted"
                else None
            ),
            "fixed_lm2mh_resonance_contract_sha256": (
                _sha(FIXED_LM2MH_RESONANCE_CONTRACT)
                if lane == "fixed-lm-targeted"
                else None
            ),
            "fixed_lm2mh_screening_feasible_count": (
                fixed_lm_feasible_count
            ),
            "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": (
                False if lane == "fixed-lm-targeted" else None
            ),
            "screening_only": lane == "fixed-lm-targeted",
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
        choices=("strict", "gap-variable", "fixed-lm-targeted"),
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
