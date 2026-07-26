"""Tune the rank-1 MFT center-leg air gap to 2.000 mH with symmetric FEA.

The driver is intentionally separate from the pre-gap 16-candidate FEA lane.
It prepares an eight-point parallel matrix-only bracket, authenticates native
L11/L22/M/k/Lm readback and physical gap geometry, then advances with guarded
secant/bisection refinement.  A successful terminal manifest emits exact tuned
parameter payloads for the later symmetric loss/thermal, full, and rounded
model stages.

Scheduler mutation is possible only through ``submit --apply`` or
``drive --apply``.  Preparation, collection, and dry-run commands are local or
GET-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    get_drawing_default_params,
    validation_check,
)
from regression_260707.verify import scheduler_client  # noqa: E402


CAMPAIGN_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-campaign-v1"
ROUND_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-round-v1"
SUBMISSION_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-submission-v1"
OBSERVATION_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-observations-v1"
FINAL_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-final-v1"
DRIVE_STATUS_SCHEMA = "mft-goal-lm2mh-physical-gap-tuning-drive-status-v1"
PROFILE_SCHEMA = "mft-goal-lm2mh-matrix-only-profile-v1"
CAMPAIGN_ID = "mft-goal-rank1-lm2mh-physical-center-gap-v1"

DEFAULT_AUTHORITY_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_targeted_w1200_l1000_v1_global_nds"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_physical_gap_tuning_v1"
)
SCHEDULER_URL = "http://127.0.0.1:8002"

TARGET_LM_H = 0.002
LM_ABS_TOLERANCE_H = 0.00002  # 0.020 mH, 1.0% of the target.
LM_REL_TOLERANCE = LM_ABS_TOLERANCE_H / TARGET_LM_H
MAX_CENTER_GAP_MM = 10.0
MAX_REFINEMENT_ITERATIONS = 12
MIN_GAP_INTERVAL_MM = 0.002
INITIAL_GAPS_MM = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0)

CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 7_200
MAX_WORKERS_PER_NODE = 1
PRIORITY = 99
PROJECT_ACTIVE_TASK_CAP = 500
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class GapTuningError(RuntimeError):
    """The physical-gap tuning contract or evidence is invalid."""


def _builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _builtin(value),
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise GapTuningError("payload is already sealed")
    result["payload_sha256"] = _sha(result)
    return result


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GapTuningError(f"{schema} payload must be an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _sha(unsigned):
        raise GapTuningError(f"{schema} seal mismatch")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GapTuningError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GapTuningError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise GapTuningError(f"output already exists: {destination}")
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    _builtin(value),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return destination


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise GapTuningError(f"regular file required: {resolved}")
    display = (
        str(resolved.relative_to(relative_to.resolve()))
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": display.replace("\\", "/"),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha_file(resolved),
    }


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GapTuningError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise GapTuningError(f"{label} must be finite")
    return number


def _truth(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise GapTuningError(f"{label} is not boolean")


def _profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA,
        "stage": "symmetric_matrix_only_physical_gap_tuning",
        "comment": (
            "Eighth-symmetry Matrix-only physical center-gap tuning; no "
            "capacitance, loss, thermal, or retained AEDT artifact"
        ),
        "reviewed_solver_path": "run_simulation_260706.py --fixed --headless",
        "cli_flags": "--headless",
        "param_overrides": {
            "full_model": 0,
            "round_corner": 0,
            "matrix_on": 1,
            "cap_on": 0,
            "loss_on": 0,
            "thermal_on": 0,
            "matrix_skin_mesh": 0,
            "matrix_percent_error": 0.5,
            "matrix_max_passes": 30,
            "matrix_min_converged": 1,
            "keep_project": 0,
            "fan_velocity": 1.5,
            "fan_config": "dual",
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
        },
        "fixed_boundary_contract": {
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "TIM_mutated": False,
        },
        "mem_mb": MEMORY_MB,
        "cpus": CPUS,
        "timeout_seconds": TIMEOUT_SECONDS,
    }


def _core_environment(solver_revision: str) -> dict[str, str]:
    contract = "mft-standalone-core-optin-v1"
    auth = _sha(
        {
            "backend": "standalone",
            "contract_version": contract,
            "requested_num_cores": CPUS,
            "required_slurm_cpus_per_task": CPUS,
            "solver_revision": solver_revision,
        }
    )
    return {
        "MFT_STANDALONE_CORE_CONTRACT": contract,
        "MFT_STANDALONE_CORE_COUNT": str(CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": auth,
    }


def _authority_rank1(authority_root: Path) -> tuple[dict[str, Any], Path]:
    root = authority_root.resolve(strict=True)
    status_path = root / "collector_status.json"
    pareto_path = root / "global_pareto_manifest.json"
    candidate_path = root / "fea_acquisition_candidates.csv"
    status = _read_json(status_path)
    pareto = _read_json(pareto_path)
    for value, label in ((status, "collector"), (pareto, "pareto")):
        unsigned = dict(value)
        observed = unsigned.pop("payload_sha256", None)
        if not HEX64.fullmatch(str(observed or "")) or observed != _sha(unsigned):
            raise GapTuningError(f"{label} authority seal mismatch")
    if (
        status.get("global_nds_final") is not True
        or int(status.get("expected_seed_count") or 0) != 16
        or int(status.get("raw_terminal_row_count") or 0) != 5120
        or int(pareto.get("fea_acquisition_candidate_count") or 0) < 1
    ):
        raise GapTuningError("final targeted authority is incomplete")

    selected: list[dict[str, Any]] = []
    with candidate_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise GapTuningError("FEA acquisition CSV header is absent")
        for row_number, row in enumerate(reader, start=2):
            try:
                rank = int(float(row.get("acquisition_rank") or "nan"))
            except (TypeError, ValueError, OverflowError):
                continue
            if rank == 1:
                item = dict(row)
                item["_row_number"] = row_number
                selected.append(item)
    if len(selected) != 1:
        raise GapTuningError(
            f"exactly one acquisition-rank 1 row required, got {len(selected)}"
        )
    row = selected[0]
    geometry_sha = str(row.get("physical_geometry_sha256") or "")
    if not HEX64.fullmatch(geometry_sha):
        raise GapTuningError("rank-1 geometry SHA is invalid")
    try:
        decoded = json.loads(str(row["decoded_physical_params_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise GapTuningError("rank-1 decoded params are invalid") from exc
    if not isinstance(decoded, dict):
        raise GapTuningError("rank-1 decoded params must be an object")

    defaults = get_drawing_default_params()
    params = {
        key: _builtin(decoded[key] if key in decoded else defaults[key])
        for key in ALL_INPUT_KEYS
    }
    params.update(_profile()["param_overrides"])
    params["core_center_gap_mm"] = 0.0
    ok, validated = validation_check(
        create_input_parameter(params), strict=True
    )
    if not ok:
        raise GapTuningError("rank-1 parameters failed strict validation")
    for key, expected in (
        ("cw1", 5.0),
        ("gap1", 1.6),
        ("full_model", 0),
        ("round_corner", 0),
        ("fan_velocity", 1.5),
        ("core_plate_pad_t", 2.0),
        ("wcp_pad_t", 2.0),
    ):
        actual = _finite(validated[key].iloc[0], f"rank-1 {key}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise GapTuningError(f"rank-1 fixed contract drifted: {key}")
    return {
        "row_number": int(row["_row_number"]),
        "physical_geometry_sha256": geometry_sha,
        "raw_row_sha256": _sha(
            {key: value for key, value in row.items() if key != "_row_number"}
        ),
        "acquisition_rank": 1,
        "selection_role": str(row.get("acquisition_selection_role") or ""),
        "source_scheduler_task_id": int(
            _finite(row.get("scheduler_task_id"), "source scheduler task")
        ),
        "source_seed": int(_finite(row.get("source_seed"), "source seed")),
        "dimensions_mm": {
            "W_drawing_x": _finite(
                row.get("exterior_W_drawing_x_mm"), "rank-1 W"
            ),
            "L_perpendicular_y": _finite(
                row.get("exterior_L_perpendicular_y_mm"), "rank-1 L"
            ),
            "H": _finite(row.get("exterior_H_mm"), "rank-1 H"),
        },
        "base_params": params,
        "base_params_sha256": _sha(params),
        "authority": {
            "collector_status": _file_record(status_path),
            "collector_status_payload_sha256": status["payload_sha256"],
            "pareto_manifest": _file_record(pareto_path),
            "pareto_manifest_payload_sha256": pareto["payload_sha256"],
            "fea_acquisition_candidates": _file_record(candidate_path),
        },
    }, candidate_path


def _gap_token(gap_mm: float) -> str:
    return f"{int(round(float(gap_mm) * 1_000_000)):08d}"


def _lane(
    *,
    root: Path,
    candidate: Mapping[str, Any],
    profile: Mapping[str, Any],
    round_index: int,
    lane_index: int,
    gap_mm: float,
    solver_revision: str,
    library_revision: str,
) -> dict[str, Any]:
    gap = _finite(gap_mm, "center gap")
    h1 = _finite(candidate["base_params"]["h1"], "h1")
    if gap < 0.0 or gap > MAX_CENTER_GAP_MM or gap >= h1:
        raise GapTuningError(f"center gap is outside the tuning domain: {gap}")
    params = dict(candidate["base_params"])
    params.update(profile["param_overrides"])
    params["core_center_gap_mm"] = gap
    ok, _validated = validation_check(
        create_input_parameter(params), strict=True
    )
    if not ok:
        raise GapTuningError(f"gap {gap} mm parameters failed validation")
    stem = candidate["physical_geometry_sha256"][:10]
    token = _gap_token(gap)
    name = f"mft-lm2-gap-r{round_index:02d}-g{token}-{stem}"
    workdir = f"mft_lm2_gap_r{round_index:02d}_g{token}_{stem}"
    params_path = _atomic_json(
        root / "params" / f"gap-{token}.json", params
    )
    identity = scheduler_client.verification_submission_identity(
        name,
        params,
        dict(profile),
        solver_revision,
        library_revision,
    )
    return {
        "lane_index": lane_index,
        "core_center_gap_mm": gap,
        "params": _file_record(params_path, relative_to=root),
        "params_sha256": _sha(params),
        "scheduler": {
            "project": scheduler_client.MFT_PROJECT,
            "name": name,
            "workdir": workdir,
            "dedupe_key": identity["dedupe_key"],
            "parameter_digest": identity["parameter_digest"],
            "effective_params_sha256": _sha(identity["merged"]),
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
            "priority": PRIORITY,
            "aedt_backend": "standalone",
            "environment": _core_environment(solver_revision),
        },
    }


def _write_round(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    round_index: int,
    gaps_mm: Sequence[float],
    method: str,
    prior_observations: Sequence[Mapping[str, Any]],
) -> Path:
    root = campaign_root / f"round-{round_index:02d}"
    if root.exists():
        raise GapTuningError(f"round directory already exists: {root}")
    root.mkdir(parents=True)
    profile = _read_json(campaign_root / campaign["profile"]["path"])
    candidate = copy.deepcopy(campaign["candidate"])
    candidate["base_params"] = _read_json(
        campaign_root / campaign["base_params"]["path"]
    )
    lanes = [
        _lane(
            root=root,
            candidate=candidate,
            profile=profile,
            round_index=round_index,
            lane_index=index,
            gap_mm=gap,
            solver_revision=campaign["solver_revision"],
            library_revision=campaign["library_revision"],
        )
        for index, gap in enumerate(gaps_mm, start=1)
    ]
    value = _seal(
        {
            "schema_version": ROUND_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": _now(),
            "campaign_payload_sha256": campaign["payload_sha256"],
            "round_index": round_index,
            "method": method,
            "target_Lm_primary_referred_H": TARGET_LM_H,
            "absolute_tolerance_H": LM_ABS_TOLERANCE_H,
            "relative_tolerance": LM_REL_TOLERANCE,
            "maximum_center_gap_mm": MAX_CENTER_GAP_MM,
            "maximum_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
            "prior_observation_payload_sha256": [
                item["payload_sha256"] for item in prior_observations
            ],
            "lanes": lanes,
            "parallel_execution_requested": len(lanes) > 1,
            "matrix_only": True,
            "symmetric_eighth_model": True,
            "full_physical_inductance_scale": 2.0,
            "scheduler_submission_performed": False,
        }
    )
    return _atomic_json(root / "round_plan.json", value)


def prepare(
    *,
    authority_root: Path,
    output: Path,
    solver_revision: str,
    library_revision: str,
) -> Path:
    solver = str(solver_revision).lower()
    library = str(library_revision).lower()
    if not HEX40.fullmatch(solver) or not HEX40.fullmatch(library):
        raise GapTuningError("full 40-character solver/library revisions required")
    candidate, _candidate_path = _authority_rank1(authority_root)
    destination = output.resolve()
    if destination.exists():
        raise GapTuningError(f"campaign output already exists: {destination}")
    destination.mkdir(parents=True)
    profile_path = _atomic_json(destination / "matrix_only_profile.json", _profile())
    base_params_path = _atomic_json(
        destination / "rank1_base_params.json", candidate.pop("base_params")
    )
    campaign = _seal(
        {
            "schema_version": CAMPAIGN_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": _now(),
            "solver_revision": solver,
            "library_revision": library,
            "candidate": candidate,
            "base_params": _file_record(
                base_params_path, relative_to=destination
            ),
            "profile": _file_record(profile_path, relative_to=destination),
            "profile_sha256": _sha(_profile()),
            "tuning_contract": {
                "target_Lm_primary_referred_H": TARGET_LM_H,
                "absolute_tolerance_H": LM_ABS_TOLERANCE_H,
                "relative_tolerance": LM_REL_TOLERANCE,
                "initial_parallel_gaps_mm": list(INITIAL_GAPS_MM),
                "maximum_center_gap_mm": MAX_CENTER_GAP_MM,
                "minimum_gap_interval_mm": MIN_GAP_INTERVAL_MM,
                "maximum_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
                "algorithm": (
                    "parallel bracket sweep then guarded secant/bisection"
                ),
                "native_eighth_to_full_inductance_scale": 2.0,
            },
            "downstream_required_variants": [
                "symmetric_matrix_confirmation",
                "symmetric_loss_thermal",
                "full_unrounded",
                "full_rounded",
            ],
            "pre_gap_16_candidate_lane_modified": False,
            "scheduler_repository_modified": False,
        }
    )
    campaign_path = _atomic_json(
        destination / "campaign_manifest.json", campaign
    )
    _write_round(
        campaign_root=destination,
        campaign=campaign,
        round_index=0,
        gaps_mm=INITIAL_GAPS_MM,
        method="initial_parallel_bracket",
        prior_observations=[],
    )
    return campaign_path


def _load_campaign(path: Path) -> tuple[dict[str, Any], Path]:
    campaign_path = path.resolve(strict=True)
    root = campaign_path.parent
    campaign = _validate_seal(_read_json(campaign_path), CAMPAIGN_SCHEMA)
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("pre_gap_16_candidate_lane_modified") is not False
        or campaign.get("scheduler_repository_modified") is not False
    ):
        raise GapTuningError("campaign contract drifted")
    profile_path = (root / campaign["profile"]["path"]).resolve(strict=True)
    base_params_path = (
        root / campaign["base_params"]["path"]
    ).resolve(strict=True)
    if (
        _file_record(profile_path, relative_to=root) != campaign["profile"]
        or _file_record(base_params_path, relative_to=root)
        != campaign["base_params"]
        or _sha(_read_json(profile_path)) != campaign["profile_sha256"]
    ):
        raise GapTuningError("campaign input bytes drifted")
    return campaign, root


def _load_round(
    round_path: Path, campaign: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], Path]:
    path = round_path.resolve(strict=True)
    round_root = path.parent
    value = _validate_seal(_read_json(path), ROUND_SCHEMA)
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("campaign_payload_sha256")
        != campaign["payload_sha256"]
        or value.get("matrix_only") is not True
        or value.get("symmetric_eighth_model") is not True
    ):
        raise GapTuningError("round contract drifted")
    profile = _read_json(root / campaign["profile"]["path"])
    identities: set[str] = set()
    for lane in value["lanes"]:
        params_path = (round_root / lane["params"]["path"]).resolve(strict=True)
        params = _read_json(params_path)
        scheduler = lane["scheduler"]
        identity = scheduler_client.verification_submission_identity(
            scheduler["name"],
            params,
            profile,
            campaign["solver_revision"],
            campaign["library_revision"],
        )
        if (
            _file_record(params_path, relative_to=round_root) != lane["params"]
            or _sha(params) != lane["params_sha256"]
            or identity["dedupe_key"] != scheduler["dedupe_key"]
            or identity["parameter_digest"] != scheduler["parameter_digest"]
            or _sha(identity["merged"]) != scheduler["effective_params_sha256"]
            or scheduler["name"] in identities
            or scheduler["dedupe_key"] in identities
        ):
            raise GapTuningError("round lane identity drifted")
        identities.update({scheduler["name"], scheduler["dedupe_key"]})
    try:
        round_root.relative_to(root)
    except ValueError as exc:
        raise GapTuningError("round path escapes campaign root") from exc
    return value, round_root


def submit(
    *,
    campaign_path: Path,
    round_path: Path,
    output: Path,
    scheduler_url: str,
    apply: bool,
    submitter: Callable[..., Any] = scheduler_client.submit_verification,
) -> Path:
    campaign, root = _load_campaign(campaign_path)
    round_plan, round_root = _load_round(round_path, campaign, root)
    if output.exists():
        raise GapTuningError(f"submission output already exists: {output}")
    if not apply:
        return _atomic_json(
            output,
            _seal(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "created_at_utc": _now(),
                    "campaign_payload_sha256": campaign["payload_sha256"],
                    "round_payload_sha256": round_plan["payload_sha256"],
                    "round_index": round_plan["round_index"],
                    "scheduler_url": scheduler_url.rstrip("/"),
                    "requested_lane_count": len(round_plan["lanes"]),
                    "submitted_lane_count": 0,
                    "submissions": [],
                    "complete": False,
                    "dry_run": True,
                    "scheduler_POST_performed": False,
                }
            ),
        )
    profile = _read_json(root / campaign["profile"]["path"])
    submissions = []
    for lane in round_plan["lanes"]:
        params = _read_json(round_root / lane["params"]["path"])
        scheduler = lane["scheduler"]
        evidence = submitter(
            scheduler["name"],
            scheduler["workdir"],
            params,
            profile,
            mem_mb=scheduler["memory_mb"],
            cpus=scheduler["cpus"],
            solver_revision=campaign["solver_revision"],
            library_revision=campaign["library_revision"],
            priority=scheduler["priority"],
            max_workers_per_node=scheduler["max_workers_per_node"],
            aedt_backend="standalone",
            submission_env=scheduler["environment"],
            required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
            return_submission_evidence=True,
            max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
            scheduler_url=scheduler_url,
        )
        if not isinstance(evidence, dict):
            raise GapTuningError("Scheduler submission evidence is absent")
        readback = (
            evidence.get("api_pre_submission_readback")
            or evidence.get("api_post_submission_response")
            or evidence
        )
        if isinstance(readback, dict) and isinstance(readback.get("task"), dict):
            readback = readback["task"]
        if not isinstance(readback, dict):
            raise GapTuningError("Scheduler submission readback is absent")
        task_id = int(
            evidence.get(
                "task_id",
                evidence.get(
                    "id", readback.get("id", readback.get("task_id", 0))
                ),
            )
        )
        if (
            task_id <= 0
            or readback.get("name") != scheduler["name"]
            or readback.get("dedupe_key") != scheduler["dedupe_key"]
            or str(readback.get("project") or "")
            != scheduler_client.MFT_PROJECT
            or int(readback.get("cpus") or 0) != scheduler["cpus"]
            or int(readback.get("memory_mb") or 0) != scheduler["memory_mb"]
        ):
            raise GapTuningError("Scheduler submission readback drifted")
        submissions.append(
            {
                "lane_index": lane["lane_index"],
                "core_center_gap_mm": lane["core_center_gap_mm"],
                "task_id": task_id,
                "name": scheduler["name"],
                "dedupe_key": scheduler["dedupe_key"],
                "readback": readback,
                "submission_evidence": evidence,
            }
        )
    value = _seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "created_at_utc": _now(),
            "campaign_payload_sha256": campaign["payload_sha256"],
            "round_payload_sha256": round_plan["payload_sha256"],
            "round_index": round_plan["round_index"],
            "scheduler_url": scheduler_url.rstrip("/"),
            "requested_lane_count": len(round_plan["lanes"]),
            "submitted_lane_count": len(submissions),
            "submissions": submissions,
            "complete": len(submissions) == len(round_plan["lanes"]),
            "dry_run": False,
            "parallel_execution_requested": len(submissions) > 1,
            "scheduler_POST_performed": True,
        }
    )
    return _atomic_json(output, value)


def _api_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise GapTuningError("Scheduler returned a non-object")
    return value


def _result_json(task: Mapping[str, Any]) -> dict[str, Any] | None:
    result = task.get("result_json")
    if isinstance(result, dict):
        return dict(result)
    output = str(task.get("stdout") or task.get("output") or "")
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON=") or line.startswith("RESULT_JSON "):
            encoded = line.split("=", 1)[1] if "=" in line[:12] else line.split(
                " ", 1
            )[1]
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _matrix_observation(
    result: Mapping[str, Any], expected_gap_mm: float
) -> dict[str, Any]:
    gap = _finite(expected_gap_mm, "expected center gap")
    reasons = []

    def exact(name: str, expected: Any) -> None:
        if result.get(name) != expected:
            reasons.append(f"identity_mismatch:{name}")

    exact("full_model", 0)
    exact("round_corner", 0)
    exact("matrix_on", 1)
    exact("cap_on", 0)
    exact("loss_on", 0)
    exact("thermal_on", 0)
    observed_gap = _finite(
        result.get("core_center_gap_mm"), "result core_center_gap_mm"
    )
    if not math.isclose(observed_gap, gap, rel_tol=0.0, abs_tol=1e-9):
        reasons.append("core_center_gap_input_mismatch")
    if int(_finite(
        result.get("core_center_gap_geometry_attested"),
        "gap geometry attestation",
    )) != 1:
        reasons.append("core_center_gap_geometry_not_attested")
    gap_readback = _finite(
        result.get("core_center_gap_readback_mm"), "gap readback"
    )
    if not math.isclose(gap_readback, gap, rel_tol=0.0, abs_tol=1e-6):
        reasons.append("core_center_gap_readback_mismatch")
    if gap > 0.0:
        if (
            result.get("core_center_gap_topology")
            != "center_leg_bottom_top_physical_air_interval"
        ):
            reasons.append("physical_gap_topology_mismatch")
        if int(_finite(
            result.get("core_center_gap_symmetry_geometry_attested"),
            "symmetric gap attestation",
        )) != 1:
            reasons.append("symmetric_gap_geometry_not_attested")

    native_l11_uH = _finite(result.get("Ltx"), "native L11")
    native_l22_uH = _finite(result.get("Lrx"), "native L22")
    native_m_uH = _finite(result.get("M"), "native mutual inductance")
    coupling = _finite(result.get("k"), "native coupling coefficient")
    native_lm_uH = _finite(result.get("Lmt"), "native Lm")
    native_llt_uH = _finite(result.get("Llt"), "native leakage")
    if min(native_l11_uH, native_l22_uH, native_m_uH, native_lm_uH) <= 0:
        reasons.append("nonpositive_matrix_readback")
    if not 0.0 < coupling <= 1.0:
        reasons.append("invalid_coupling_readback")
    formula_lm = native_l11_uH * coupling * coupling
    if not math.isclose(
        native_lm_uH,
        formula_lm,
        rel_tol=1e-8,
        abs_tol=1e-6,
    ):
        reasons.append("Lm_formula_readback_mismatch")
    formula_k = native_m_uH / math.sqrt(native_l11_uH * native_l22_uH)
    if not math.isclose(
        coupling, formula_k, rel_tol=1e-8, abs_tol=1e-9
    ):
        reasons.append("coupling_formula_readback_mismatch")

    tolerance = _finite(
        result.get("matrix_percent_error"), "matrix percent error"
    )
    convergence = {
        "passes": _finite(result.get("conv_passes_matrix"), "matrix passes"),
        "consecutive": _finite(
            result.get("conv_consecutive_matrix"), "matrix consecutive"
        ),
        "energy_error_pct": _finite(
            result.get("conv_error_pct_matrix"), "matrix energy error"
        ),
        "delta_energy_pct": _finite(
            result.get("conv_delta_pct_matrix"), "matrix delta energy"
        ),
        "tolerance_pct": tolerance,
    }
    if (
        convergence["passes"] < 1
        or convergence["consecutive"]
        < _finite(result.get("matrix_min_converged"), "matrix minimum")
        or convergence["energy_error_pct"] > tolerance
        or convergence["delta_energy_pct"] > tolerance
    ):
        reasons.append("matrix_not_converged")

    scale = 2.0
    physical_lm_h = scale * native_lm_uH * 1e-6
    return {
        "contract_valid": not reasons,
        "reasons": reasons,
        "core_center_gap_mm": gap,
        "core_center_gap_readback_mm": gap_readback,
        "geometry_attestation": {
            "topology": result.get("core_center_gap_topology"),
            "geometry_attested": int(
                _finite(
                    result.get("core_center_gap_geometry_attested"),
                    "geometry attested",
                )
            ),
            "symmetry_geometry_attested": int(
                _finite(
                    result.get(
                        "core_center_gap_symmetry_geometry_attested",
                        1 if gap == 0.0 else math.nan,
                    ),
                    "symmetry geometry attested",
                )
            ),
            "removed_volume_readback_mm3": _finite(
                result.get("core_center_gap_removed_volume_readback_mm3"),
                "removed volume readback",
            ),
            "removed_volume_relative_error": _finite(
                result.get("core_center_gap_removed_volume_rel_error"),
                "removed volume relative error",
            ),
        },
        "native_eighth_matrix_readback": {
            "L11_uH": native_l11_uH,
            "L22_uH": native_l22_uH,
            "M_uH": native_m_uH,
            "k": coupling,
            "Lm_primary_uH": native_lm_uH,
            "Lleak_primary_uH": native_llt_uH,
        },
        "full_physical_matrix_readback": {
            "restoration_scale": scale,
            "L11_uH": scale * native_l11_uH,
            "L22_uH": scale * native_l22_uH,
            "M_uH": scale * native_m_uH,
            "k": coupling,
            "Lm_primary_referred_H": physical_lm_h,
            "Lm_primary_referred_mH": physical_lm_h * 1e3,
            "Lleak_primary_uH": scale * native_llt_uH,
        },
        "convergence": convergence,
        "target_error_H": physical_lm_h - TARGET_LM_H,
        "target_abs_error_H": abs(physical_lm_h - TARGET_LM_H),
        "within_tolerance": abs(physical_lm_h - TARGET_LM_H)
        <= LM_ABS_TOLERANCE_H,
        "result_sha256": _sha(dict(result)),
    }


def collect(
    *,
    campaign_path: Path,
    round_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str | None = None,
    getter: Callable[[str], dict[str, Any]] = _api_json,
    replace: bool = False,
) -> dict[str, Any]:
    campaign, root = _load_campaign(campaign_path)
    round_plan, _round_root = _load_round(round_path, campaign, root)
    submission = _validate_seal(
        _read_json(submission_path.resolve(strict=True)), SUBMISSION_SCHEMA
    )
    if (
        submission.get("complete") is not True
        or submission.get("dry_run") is not False
        or submission.get("round_payload_sha256")
        != round_plan["payload_sha256"]
    ):
        raise GapTuningError("complete applied submission receipt required")
    origin = (scheduler_url or submission["scheduler_url"]).rstrip("/")
    receipt_by_lane = {
        int(item["lane_index"]): item for item in submission["submissions"]
    }
    rows = []
    terminal = 0
    valid = 0
    for lane in round_plan["lanes"]:
        receipt = receipt_by_lane[lane["lane_index"]]
        task_id = int(receipt["task_id"])
        query = urllib.parse.urlencode(
            {"include_output": "true", "output_limit": "200000"}
        )
        task = getter(f"{origin}/api/tasks/{task_id}?{query}")
        if (
            int(task.get("id", task.get("task_id", -1))) != task_id
            or task.get("name") != lane["scheduler"]["name"]
            or task.get("dedupe_key") != lane["scheduler"]["dedupe_key"]
            or str(task.get("project") or "")
            != scheduler_client.MFT_PROJECT
        ):
            raise GapTuningError(f"Scheduler task {task_id} identity drifted")
        status = str(task.get("status") or "unknown").lower()
        item: dict[str, Any] = {
            "lane_index": lane["lane_index"],
            "core_center_gap_mm": lane["core_center_gap_mm"],
            "task_id": task_id,
            "status": status,
            "node": task.get("actual_node_name"),
            "result_available": False,
            "contract_valid": False,
        }
        if status in TERMINAL_STATUSES:
            terminal += 1
        if status == "completed" and int(task.get("exit_code") or 0) == 0:
            result = _result_json(task)
            if result is not None:
                observation = _matrix_observation(
                    result, lane["core_center_gap_mm"]
                )
                item.update(observation)
                item["result_available"] = True
                if observation["contract_valid"]:
                    valid += 1
        rows.append(item)
    value = _seal(
        {
            "schema_version": OBSERVATION_SCHEMA,
            "created_at_utc": _now(),
            "campaign_payload_sha256": campaign["payload_sha256"],
            "round_payload_sha256": round_plan["payload_sha256"],
            "submission_payload_sha256": submission["payload_sha256"],
            "round_index": round_plan["round_index"],
            "scheduler_url": origin,
            "lane_count": len(rows),
            "terminal_count": terminal,
            "valid_observation_count": valid,
            "all_terminal": terminal == len(rows),
            "all_valid": valid == len(rows),
            "rows": rows,
            "scheduler_mutation_performed": False,
        }
    )
    _atomic_json(output, value, replace=replace)
    return value


def _valid_observations(
    observation_payloads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_gap: dict[float, dict[str, Any]] = {}
    for payload in observation_payloads:
        for row in payload["rows"]:
            if not row.get("contract_valid"):
                continue
            gap = _finite(row["core_center_gap_mm"], "observation gap")
            prior = by_gap.get(gap)
            if prior is not None and prior["result_sha256"] != row["result_sha256"]:
                raise GapTuningError(f"conflicting observations at gap {gap}")
            by_gap[gap] = dict(row)
    return [by_gap[key] for key in sorted(by_gap)]


def _next_gap_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    refinement_count: int,
) -> dict[str, Any]:
    if not rows:
        return {"status": "need_observation", "next_gaps_mm": []}
    ordered = sorted(rows, key=lambda row: float(row["core_center_gap_mm"]))
    best = min(ordered, key=lambda row: float(row["target_abs_error_H"]))
    if float(best["target_abs_error_H"]) <= LM_ABS_TOLERANCE_H:
        return {
            "status": "tuned",
            "selected": dict(best),
            "next_gaps_mm": [],
        }
    if refinement_count >= MAX_REFINEMENT_ITERATIONS:
        return {
            "status": "iteration_limit",
            "selected": dict(best),
            "next_gaps_mm": [],
        }

    above = [
        row for row in ordered
        if float(
            row["full_physical_matrix_readback"]["Lm_primary_referred_H"]
        ) >= TARGET_LM_H
    ]
    below = [
        row for row in ordered
        if float(
            row["full_physical_matrix_readback"]["Lm_primary_referred_H"]
        ) <= TARGET_LM_H
    ]
    brackets = [
        (left, right)
        for left in above
        for right in below
        if float(left["core_center_gap_mm"])
        < float(right["core_center_gap_mm"])
    ]
    if brackets:
        low, high = min(
            brackets,
            key=lambda pair: (
                float(pair[1]["core_center_gap_mm"])
                - float(pair[0]["core_center_gap_mm"])
            ),
        )
        gap_low = float(low["core_center_gap_mm"])
        gap_high = float(high["core_center_gap_mm"])
        lm_low = float(
            low["full_physical_matrix_readback"]["Lm_primary_referred_H"]
        )
        lm_high = float(
            high["full_physical_matrix_readback"]["Lm_primary_referred_H"]
        )
        interval = gap_high - gap_low
        if interval <= MIN_GAP_INTERVAL_MM:
            return {
                "status": "gap_interval_limit",
                "selected": dict(best),
                "bracket": [dict(low), dict(high)],
                "next_gaps_mm": [],
            }
        midpoint = (gap_low + gap_high) / 2.0
        if lm_low > lm_high:
            secant = gap_low + (
                (lm_low - TARGET_LM_H) / (lm_low - lm_high)
            ) * interval
        else:
            secant = midpoint
        guard = 0.20 * interval
        next_gap = min(
            max(secant, gap_low + guard), gap_high - guard
        )
        if any(
            math.isclose(
                next_gap,
                float(row["core_center_gap_mm"]),
                rel_tol=0.0,
                abs_tol=MIN_GAP_INTERVAL_MM / 4.0,
            )
            for row in ordered
        ):
            next_gap = midpoint
        return {
            "status": "refine_bracket",
            "algorithm": "guarded_secant_with_bisection_fallback",
            "bracket": [dict(low), dict(high)],
            "next_gaps_mm": [round(next_gap, 6)],
        }

    minimum_gap = float(ordered[0]["core_center_gap_mm"])
    maximum_gap = float(ordered[-1]["core_center_gap_mm"])
    minimum_lm = float(
        ordered[-1]["full_physical_matrix_readback"][
            "Lm_primary_referred_H"
        ]
    )
    maximum_lm = float(
        ordered[0]["full_physical_matrix_readback"][
            "Lm_primary_referred_H"
        ]
    )
    if minimum_gap <= 1e-12 and maximum_lm < TARGET_LM_H:
        return {
            "status": "infeasible_gap_only_cannot_raise_Lm",
            "selected": dict(best),
            "next_gaps_mm": [],
        }
    if minimum_lm > TARGET_LM_H and maximum_gap < MAX_CENTER_GAP_MM:
        next_gap = min(
            MAX_CENTER_GAP_MM,
            max(maximum_gap * 1.75, maximum_gap + 0.5),
        )
        return {
            "status": "expand_upper_bracket",
            "next_gaps_mm": [round(next_gap, 6)],
        }
    return {
        "status": "no_bracket_within_maximum_gap",
        "selected": dict(best),
        "next_gaps_mm": [],
    }


def _observation_files(root: Path) -> list[Path]:
    return sorted(root.glob("round-*/observations.json"))


def _load_observations(
    root: Path, campaign: Mapping[str, Any]
) -> list[dict[str, Any]]:
    values = []
    for path in _observation_files(root):
        value = _validate_seal(_read_json(path), OBSERVATION_SCHEMA)
        if value.get("campaign_payload_sha256") != campaign["payload_sha256"]:
            raise GapTuningError(f"foreign observation file: {path}")
        values.append(value)
    return values


def _variant_params(
    base: Mapping[str, Any], gap_mm: float
) -> dict[str, dict[str, Any]]:
    common = dict(base)
    common["core_center_gap_mm"] = gap_mm
    variants = {
        "symmetric_matrix_confirmation": dict(
            common,
            full_model=0,
            round_corner=0,
            matrix_on=1,
            cap_on=0,
            loss_on=0,
            thermal_on=0,
            keep_project=1,
        ),
        "symmetric_loss_thermal": dict(
            common,
            full_model=0,
            round_corner=0,
            matrix_on=1,
            cap_on=1,
            loss_on=1,
            thermal_on=1,
            loss_sym_on=1,
            thermal_symmetry="eighth",
            keep_project=1,
        ),
        "full_unrounded": dict(
            common,
            full_model=1,
            round_corner=0,
            matrix_on=1,
            cap_on=1,
            loss_on=1,
            thermal_on=0,
            loss_sym_on=0,
            keep_project=1,
        ),
        "full_rounded": dict(
            common,
            full_model=1,
            round_corner=1,
            matrix_on=1,
            cap_on=1,
            loss_on=1,
            thermal_on=0,
            loss_sym_on=0,
            keep_project=1,
        ),
    }
    for name, params in variants.items():
        ok, _validated = validation_check(
            create_input_parameter(params), strict=True
        )
        if not ok:
            raise GapTuningError(f"downstream variant is invalid: {name}")
    return variants


def finalize_or_refine(
    *,
    campaign_path: Path,
) -> dict[str, Any]:
    campaign, root = _load_campaign(campaign_path)
    payloads = _load_observations(root, campaign)
    if not payloads or payloads[-1].get("all_terminal") is not True:
        return {
            "status": "awaiting_terminal_observations",
            "observation_round_count": len(payloads),
        }
    invalid = [
        row
        for payload in payloads
        for row in payload["rows"]
        if not row.get("contract_valid")
    ]
    if invalid:
        raise GapTuningError(
            f"{len(invalid)} terminal matrix observations are invalid"
        )
    rows = _valid_observations(payloads)
    refinement_count = max(
        0, max(int(payload["round_index"]) for payload in payloads)
    )
    decision = _next_gap_decision(
        rows, refinement_count=refinement_count
    )
    if decision["status"] == "tuned":
        final_path = root / "tuned_gap_manifest.json"
        if final_path.exists():
            return _validate_seal(_read_json(final_path), FINAL_SCHEMA)
        selected = decision["selected"]
        gap = float(selected["core_center_gap_mm"])
        base = _read_json(root / campaign["base_params"]["path"])
        variants = _variant_params(base, gap)
        variant_records = {}
        for name, params in variants.items():
            path = _atomic_json(
                root / "downstream_params" / f"{name}.json", params
            )
            variant_records[name] = _file_record(path, relative_to=root)
        final = _seal(
            {
                "schema_version": FINAL_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": _now(),
                "campaign_payload_sha256": campaign["payload_sha256"],
                "source_candidate": copy.deepcopy(campaign["candidate"]),
                "algorithm": (
                    "parallel bracket sweep then guarded secant/bisection"
                ),
                "target_Lm_primary_referred_H": TARGET_LM_H,
                "absolute_tolerance_H": LM_ABS_TOLERANCE_H,
                "relative_tolerance": LM_REL_TOLERANCE,
                "selected_observation": selected,
                "tuned_core_center_gap_mm": gap,
                "tuned_Lm_primary_referred_H": selected[
                    "full_physical_matrix_readback"
                ]["Lm_primary_referred_H"],
                "tuned_Lm_primary_referred_mH": selected[
                    "full_physical_matrix_readback"
                ]["Lm_primary_referred_mH"],
                "target_abs_error_H": selected["target_abs_error_H"],
                "physical_gap_geometry_attested": True,
                "symmetric_matrix_convergence_attested": True,
                "native_L11_L22_M_k_Lm_readback_attested": True,
                "observation_files": [
                    _file_record(path, relative_to=root)
                    for path in _observation_files(root)
                ],
                "downstream_params": variant_records,
                "fan_velocity_m_s": 1.5,
                "TIM_mutated": False,
                "pre_gap_16_candidate_lane_modified": False,
                "production_eligible": False,
                "remaining_gates": [
                    "symmetric loss and thermal FEA",
                    "resonance recomputation with measured leakage/capacitance",
                    "full unrounded model artifact",
                    "full rounded model artifact",
                ],
            }
        )
        _atomic_json(final_path, final)
        return final

    next_gaps = decision.get("next_gaps_mm") or []
    if next_gaps:
        next_round = max(int(item["round_index"]) for item in payloads) + 1
        next_path = root / f"round-{next_round:02d}" / "round_plan.json"
        if not next_path.exists():
            _write_round(
                campaign_root=root,
                campaign=campaign,
                round_index=next_round,
                gaps_mm=next_gaps,
                method=str(decision["status"]),
                prior_observations=payloads,
            )
        return {
            "status": decision["status"],
            "next_round": next_round,
            "next_round_plan": str(next_path),
            "next_gaps_mm": next_gaps,
        }
    return decision


def _latest_round(root: Path) -> Path:
    paths = sorted(root.glob("round-*/round_plan.json"))
    if not paths:
        raise GapTuningError("campaign has no round plan")
    return paths[-1]


def drive(
    *,
    campaign_path: Path,
    scheduler_url: str,
    apply: bool,
    poll_seconds: float,
    once: bool,
) -> dict[str, Any]:
    if poll_seconds < 10.0 or poll_seconds > 60.0:
        raise GapTuningError("poll_seconds must be within 10..60")
    campaign, root = _load_campaign(campaign_path)
    while True:
        final_path = root / "tuned_gap_manifest.json"
        if final_path.exists():
            return _validate_seal(_read_json(final_path), FINAL_SCHEMA)
        round_path = _latest_round(root)
        round_plan = _validate_seal(_read_json(round_path), ROUND_SCHEMA)
        round_root = round_path.parent
        submission_path = round_root / "submission.json"
        observation_path = round_root / "observations.json"
        if not submission_path.exists():
            if not apply:
                return {
                    "status": "ready_to_submit",
                    "round_index": round_plan["round_index"],
                    "round_plan": str(round_path),
                    "lane_count": len(round_plan["lanes"]),
                    "scheduler_POST_performed": False,
                }
            submit(
                campaign_path=campaign_path,
                round_path=round_path,
                output=submission_path,
                scheduler_url=scheduler_url,
                apply=True,
            )
        observations = collect(
            campaign_path=campaign_path,
            round_path=round_path,
            submission_path=submission_path,
            output=observation_path,
            scheduler_url=scheduler_url,
            replace=observation_path.exists(),
        )
        if observations["all_terminal"]:
            outcome = finalize_or_refine(campaign_path=campaign_path)
            if outcome.get("schema_version") == FINAL_SCHEMA:
                return outcome
            if outcome.get("next_round") is not None:
                if once:
                    return outcome
                continue
            return outcome
        status = {
            "schema_version": DRIVE_STATUS_SCHEMA,
            "observed_at_utc": _now(),
            "campaign_payload_sha256": campaign["payload_sha256"],
            "round_index": round_plan["round_index"],
            "terminal_count": observations["terminal_count"],
            "lane_count": observations["lane_count"],
            "status": "running",
        }
        _atomic_json(
            root / "drive_status.json", status, replace=True
        )
        if once:
            return status
        time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument(
        "--authority-root", type=Path, default=DEFAULT_AUTHORITY_ROOT
    )
    prepare_cmd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_cmd.add_argument("--solver-revision", required=True)
    prepare_cmd.add_argument("--library-revision", required=True)

    submit_cmd = commands.add_parser("submit")
    submit_cmd.add_argument("--campaign", type=Path, required=True)
    submit_cmd.add_argument("--round", dest="round_path", type=Path, required=True)
    submit_cmd.add_argument("--output", type=Path, required=True)
    submit_cmd.add_argument("--scheduler-url", default=SCHEDULER_URL)
    submit_cmd.add_argument("--apply", action="store_true")

    collect_cmd = commands.add_parser("collect")
    collect_cmd.add_argument("--campaign", type=Path, required=True)
    collect_cmd.add_argument("--round", dest="round_path", type=Path, required=True)
    collect_cmd.add_argument("--submission", type=Path, required=True)
    collect_cmd.add_argument("--output", type=Path, required=True)
    collect_cmd.add_argument("--scheduler-url", default=None)
    collect_cmd.add_argument("--replace", action="store_true")

    finalize_cmd = commands.add_parser("finalize-or-refine")
    finalize_cmd.add_argument("--campaign", type=Path, required=True)

    drive_cmd = commands.add_parser("drive")
    drive_cmd.add_argument("--campaign", type=Path, required=True)
    drive_cmd.add_argument("--scheduler-url", default=SCHEDULER_URL)
    drive_cmd.add_argument("--apply", action="store_true")
    drive_cmd.add_argument("--poll-seconds", type=float, default=20.0)
    drive_cmd.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result: Any = prepare(
            authority_root=args.authority_root,
            output=args.output,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
        )
    elif args.command == "submit":
        result = submit(
            campaign_path=args.campaign,
            round_path=args.round_path,
            output=args.output,
            scheduler_url=args.scheduler_url,
            apply=args.apply,
        )
    elif args.command == "collect":
        result = collect(
            campaign_path=args.campaign,
            round_path=args.round_path,
            submission_path=args.submission,
            output=args.output,
            scheduler_url=args.scheduler_url,
            replace=args.replace,
        )
    elif args.command == "finalize-or-refine":
        result = finalize_or_refine(campaign_path=args.campaign)
    elif args.command == "drive":
        result = drive(
            campaign_path=args.campaign,
            scheduler_url=args.scheduler_url,
            apply=args.apply,
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(
        json.dumps(
            _builtin(result), indent=2, sort_keys=True, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
