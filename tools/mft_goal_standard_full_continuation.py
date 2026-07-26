"""One-shot Standard-result to same-candidate Full-FEA continuation.

The post-deadline Standard collectors are GET-only and intentionally cannot
submit work.  This module is the narrow mutation boundary after those
collectors:

* wait until every configured Standard lane is terminal;
* reauthenticate the immutable post-success snapshot and source collection;
* choose only measured hard-feasible rank-0 truth, using the established
  ``(volume, loss, candidate)`` front order;
* recheck dimensions, resonance, primary/secondary/core temperatures, and
  the fixed cooling identity from the retained solver result;
* require a fresh 16-core license snapshot and a different, FEA-empty strict
  node lane; and
* consume one immutable attempt ledger before the only possible Scheduler
  POST.

Pending or infeasible Standard results perform zero Scheduler mutations.  A
Standard pass permits a Full *submission* only; it never claims a Full
scientific pass or completed promotion.  The Scheduler repository is read via
its API and is never imported, edited, or combined with this project.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
from typing import Any, Callable, ContextManager, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    TEMPERATURE_FAMILY_LIMITS_C,
)
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_full_fastlane_watcher as fastlane  # noqa: E402
from tools import mft_goal_postdeadline_standard_postsuccess as postsuccess  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


CAMPAIGN_ID = "mft-goal-20260726"
SCHEDULER_URL = "http://127.0.0.1:8002"
PROJECT = scheduler_client.MFT_PROJECT
CPUS = 16
MEMORY_MB = 98_304
MAX_WORKERS_PER_NODE = 1
SOLVER_TIMEOUT_SECONDS = 79_200
SCHEDULER_TIMEOUT_SECONDS = 86_400
KILL_GRACE_SECONDS = 300
PRIORITY = 100
MIN_INTERVAL_SECONDS = 10
MAX_INTERVAL_SECONDS = 300
DEFAULT_INTERVAL_SECONDS = 30
AUTHORIZATION_TOKEN = "measured-standard-pass-to-full-v1"

PLAN_SCHEMA = "mft-goal-standard-full-continuation-plan-v2"
ATTEMPT_SCHEMA = "mft-goal-standard-full-continuation-attempt-v1"
RECEIPT_SCHEMA = "mft-goal-standard-full-continuation-receipt-v1"
STATE_SCHEMA = "mft-goal-standard-full-continuation-state-v1"

SAFETY_FLAGS = {
    "original_deadline_missed": True,
    "postdeadline": True,
    "canonical": False,
    "production_eligible": False,
    "scientific_pass_claimed": False,
    "full_result_available": False,
    "full_actual_constraints_passed": False,
    "promotion_completed": False,
}

JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]
PayloadBuilder = Callable[
    [
        dict[str, Any],
        dict[str, Any],
        str,
        str,
        str,
        str,
        "StrictLane",
        Path,
    ],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]
PostOnce = Callable[[str, Mapping[str, Any]], tuple[int, Any]]
LockFactory = Callable[[], ContextManager[Any]]


class ContinuationError(RuntimeError):
    """A measured-truth, placement, or one-shot mutation contract drifted."""


@dataclass(frozen=True, order=True)
class StrictLane:
    """One exact Scheduler account/node lane allowed for Full FEA."""

    account_name: str
    node_name: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    if "payload_sha256" in output:
        raise ContinuationError("payload is already sealed")
    output["payload_sha256"] = _payload_sha256(output)
    return output


def _validate_seal(value: Any, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} is not an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _payload_sha256(
        unsigned
    ):
        raise ContinuationError(f"{label} seal mismatch")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} is not an object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    try:
        return production._file_record(path.resolve(strict=True))  # noqa: SLF001
    except Exception as exc:
        raise ContinuationError(f"file record failed: {path}") from exc


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContinuationError(f"immutable output already exists: {target}") from exc
    return target


def _task_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [copy.deepcopy(item) for item in value if isinstance(item, dict)]
    if isinstance(value, Mapping):
        for key in ("items", "tasks", "allocations"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [
                    copy.deepcopy(item) for item in rows if isinstance(item, dict)
                ]
    raise ContinuationError("Scheduler inventory is malformed")


def get_json(
    path: str,
    query: Sequence[tuple[str, Any]] | None = None,
    *,
    scheduler_url: str = SCHEDULER_URL,
) -> Any:
    url = f"{scheduler_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urlencode(list(query), doseq=True)}"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"Scheduler GET failed: {url}") from exc


def _post_json_once(
    url: str, payload: Mapping[str, Any]
) -> tuple[int, Any]:
    request = Request(
        url,
        data=_canonical_bytes(payload),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
            return response.status, json.loads(body.decode("utf-8"))
    except HTTPError as exc:
        body = exc.read()
        try:
            decoded: Any = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            decoded = body.decode("utf-8", errors="replace")
        return exc.code, decoded


def _parse_lane(value: str) -> StrictLane:
    account, separator, node = value.partition("=")
    if (
        not separator
        or not account.strip()
        or not node.strip()
        or any(character.isspace() for character in account + node)
    ):
        raise argparse.ArgumentTypeError(
            "strict lane must be ACCOUNT_NAME=NODE_NAME"
        )
    return StrictLane(account.strip(), node.strip())


def _postsuccess_authority(
    state_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state = _validate_seal(
        _read_json(state_path, "post-success state"),
        postsuccess.STATE_SCHEMA,
        "post-success state",
    )
    if (
        any(state.get(key) is not expected for key, expected in postsuccess.SAFETY_FLAGS.items())
        or state.get("scientific_pass_claimed") is not False
        or state.get("production_claimed") is not False
        or state.get("production_pareto_emitted") is not False
        or state.get("scheduler_mutation_performed") is not False
    ):
        raise ContinuationError("post-success state safety contract drifted")
    pending = state.get("pending_count")
    count = state.get("collection_count")
    if (
        isinstance(pending, bool)
        or not isinstance(pending, int)
        or pending < 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise ContinuationError("post-success state counts are invalid")
    latest = state.get("latest_snapshot")
    if count == 0:
        if latest is not None:
            raise ContinuationError("empty post-success state has a snapshot")
        return state, None
    if not isinstance(latest, Mapping):
        raise ContinuationError("post-success snapshot reference is absent")
    record = latest.get("snapshot_manifest")
    if not isinstance(record, Mapping):
        raise ContinuationError("post-success snapshot record is absent")
    snapshot_path = Path(str(record.get("path") or ""))
    if _file_record(snapshot_path) != dict(record):
        raise ContinuationError("post-success snapshot bytes drifted")
    snapshot = postsuccess._load_snapshot(  # noqa: SLF001
        snapshot_path,
        expected_snapshot_id=str(latest.get("snapshot_id") or ""),
    )
    if (
        snapshot.get("authenticated_observation_count") != count
        or snapshot.get("measured_actual_nds_performed") is not True
        or snapshot.get("scientific_pass_claimed") is not False
        or snapshot.get("production_claimed") is not False
    ):
        raise ContinuationError("post-success snapshot authority drifted")
    return state, snapshot


def _best_measured_row(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = snapshot.get("ranked_rows")
    if not isinstance(rows, list):
        raise ContinuationError("post-success ranked rows are absent")
    eligible = [
        copy.deepcopy(dict(row))
        for row in rows
        if isinstance(row, Mapping)
        and row.get("measured_hard_constraints_passed") is True
        and row.get("hard_feasible_non_dominated_rank") == 0
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            float(row["actual_volume_L"]),
            float(row["actual_total_loss_W"]),
            str(row["candidate_physics_sha256"]),
        )
    )
    return eligible[0]


def _source_collection(
    state: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    task_id = int(row["task_id"])
    lane_matches = [
        lane
        for lane in state.get("lanes", [])
        if isinstance(lane, Mapping) and lane.get("task_id") == task_id
    ]
    if len(lane_matches) != 1:
        raise ContinuationError("selected Standard lane is absent or ambiguous")
    root = Path(str(lane_matches[0].get("collection_root") or ""))
    receipt_path = (root / "collection_receipt.json").resolve(strict=True)
    view = postsuccess.authenticate_collection(receipt_path)
    measured = postsuccess._measured_classification(view)  # noqa: SLF001
    expected = {
        "candidate_physics_sha256": row["candidate_physics_sha256"],
        "actual_volume_L": row["actual_volume_L"],
        "actual_total_loss_W": row["actual_total_loss_W"],
        "actual_resonance_Hz": row["actual_resonance_Hz"],
        "actual_primary_winding_max_C": row[
            "actual_primary_winding_max_C"
        ],
        "actual_secondary_winding_max_C": row[
            "actual_secondary_winding_max_C"
        ],
        "actual_winding_max_C": row["actual_winding_max_C"],
        "actual_core_max_C": row["actual_core_max_C"],
        "measured_hard_constraints_passed": True,
    }
    observed = {
        "candidate_physics_sha256": measured["candidate_physics_sha256"],
        "actual_volume_L": measured["actual_volume_L"],
        "actual_total_loss_W": measured["actual_total_loss_W"],
        "actual_resonance_Hz": measured["actual_resonance_Hz"],
        "actual_primary_winding_max_C": measured[
            "actual_primary_winding_max_C"
        ],
        "actual_secondary_winding_max_C": measured[
            "actual_secondary_winding_max_C"
        ],
        "actual_winding_max_C": measured["actual_winding_max_C"],
        "actual_core_max_C": measured["actual_core_max_C"],
        "measured_hard_constraints_passed": measured[
            "measured_hard_constraints_passed"
        ],
    }
    if expected != observed or measured["task_id"] != task_id:
        raise ContinuationError("selected measured row failed source reauthentication")
    evidence = measured.get("hard_constraint_evidence")
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != postsuccess.HARD_CONSTRAINT_EVIDENCE_KEYS
        or not postsuccess._temperature_family_evidence_valid(  # noqa: SLF001
            measured
        )
        or not all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in evidence.values()
        )
        or measured["actual_dimensions_mm"]["W"] > GOAL_SIZE_LIMITS_MM["W"]
        or measured["actual_dimensions_mm"]["L"] > GOAL_SIZE_LIMITS_MM["L"]
        or measured["actual_dimensions_mm"]["H"] > GOAL_SIZE_LIMITS_MM["H"]
        or measured["actual_resonance_Hz"]
        < GOAL_STAGE_SPEC["resonance_min_Hz"]
        or measured["actual_primary_winding_max_C"]
        > TEMPERATURE_FAMILY_LIMITS_C["primary_winding"]
        or measured["actual_secondary_winding_max_C"]
        > TEMPERATURE_FAMILY_LIMITS_C["secondary_winding"]
        or measured["actual_core_max_C"]
        > TEMPERATURE_FAMILY_LIMITS_C["core"]
        or measured["actual_winding_max_C"]
        != max(
            measured["actual_primary_winding_max_C"],
            measured["actual_secondary_winding_max_C"],
        )
    ):
        raise ContinuationError("selected Standard actual constraints are not all passed")
    fixed = measured.get("fixed_identity_attestation")
    expected_fixed = {
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_on": 1,
        "wcp_on": 1,
    }
    fixed_expected = fixed.get("expected") if isinstance(fixed, Mapping) else None
    fixed_observed = fixed.get("observed") if isinstance(fixed, Mapping) else None
    if (
        not isinstance(fixed, Mapping)
        or fixed.get("attested") is not True
        or fixed.get("mismatches") != []
        or not isinstance(fixed_expected, Mapping)
        or not isinstance(fixed_observed, Mapping)
        or any(
            fixed_expected.get(key) != value
            or fixed_observed.get(key) != value
            for key, value in expected_fixed.items()
        )
    ):
        raise ContinuationError("selected Standard fixed cooling identity drifted")
    raw_receipt = _read_json(receipt_path, "selected collection receipt")
    terminal_record = raw_receipt.get("source_files", {}).get(
        "scheduler_terminal_task.json"
    )
    terminal_path = postsuccess._resolve_record(  # noqa: SLF001
        terminal_record,
        root=receipt_path.parent,
        label="selected Scheduler terminal task",
    )
    terminal = _read_json(terminal_path, "selected Scheduler terminal task")
    source_node = str(
        terminal.get("actual_node_name")
        or terminal.get("allocation_node_name")
        or terminal.get("node_name")
        or ""
    )
    if (
        not source_node
        or terminal.get("strict_node_placement") is not True
        or terminal.get("placement_contract_satisfied") is not True
        or int(terminal.get("task_id") or terminal.get("id") or 0) != task_id
    ):
        raise ContinuationError("selected Standard strict-node lineage drifted")
    return receipt_path, view, measured, source_node


def _capacity_query(lane: StrictLane) -> list[tuple[str, Any]]:
    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", lane.account_name),
        ("node_name", lane.node_name),
    ]


def _lane_preflight(
    *,
    lane: StrictLane,
    source_node: str,
    task_name: str,
    dedupe_key: str,
    source_task_id: int,
    reader: JsonReader,
) -> dict[str, Any]:
    if lane.node_name == source_node:
        raise ContinuationError("Full lane equals the selected Standard node")
    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{source_task_id}", None)
    capacity = reader("/api/task-capacity", _capacity_query(lane))
    allocation_rows = _task_rows(reader("/api/allocations", None))
    active = _task_rows(
        reader(
            "/api/tasks",
            [
                ("status", "queued"),
                ("status", "attaching"),
                ("status", "running"),
                ("limit", 10000),
            ],
        )
    )
    collisions = _task_rows(
        reader(
            "/api/tasks",
            [
                ("limit", 10000),
                ("project", PROJECT),
                ("name_prefix", task_name),
            ],
        )
    )
    if (
        not isinstance(health, Mapping)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise ContinuationError("Scheduler health gate failed")
    if (
        not isinstance(source_task, Mapping)
        or int(source_task.get("task_id") or source_task.get("id") or 0)
        != source_task_id
        or source_task.get("status") != "completed"
        or source_task.get("state") != "succeeded"
        or source_task.get("exit_code") != 0
        or source_task.get("actual_node_name") != source_node
    ):
        raise ContinuationError("live Standard source task drifted")
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("queue_state") != "ready"
        or capacity.get("memory_pressure_state") != "ok"
        or int(capacity.get("ready_fit_slots") or 0) < 1
        or int(capacity.get("standalone_aedt_available") or 0) < 1
    ):
        raise ContinuationError("strict Full capacity is not ready")
    matches = [
        item
        for item in capacity.get("allocations", [])
        if isinstance(item, Mapping)
        and item.get("account_name") == lane.account_name
        and item.get("node_name") == lane.node_name
        and item.get("state") == "active"
        and int(item.get("fit_slots") or 0) >= 1
        and int(item.get("free_cpus") or 0) >= CPUS
        and int(item.get("free_memory_mb") or 0) >= MEMORY_MB
        and item.get("memory_pressure_state") == "ok"
    ]
    if not matches:
        raise ContinuationError("exact strict Full allocation is absent")
    capacity_allocation = max(
        matches,
        key=lambda item: (
            int(item.get("fit_slots") or 0),
            int(item.get("allocation_id") or 0),
        ),
    )
    allocation_id = int(capacity_allocation.get("allocation_id") or 0)
    exact = [
        item
        for item in allocation_rows
        if int(item.get("id") or 0) == allocation_id
    ]
    if len(exact) != 1:
        raise ContinuationError("strict Full allocation identity is ambiguous")
    allocation = exact[0]
    if (
        allocation.get("account_name") != lane.account_name
        or allocation.get("node_name") != lane.node_name
        or allocation.get("state") != "active"
        or int(allocation.get("free_cpus") or 0) < CPUS
        or int(allocation.get("free_memory_mb") or 0) < MEMORY_MB
        or int(allocation.get("node_fea_requested_cpus") or 0) != 0
    ):
        raise ContinuationError("strict Full allocation is not FEA-empty")
    active_fea = [
        item
        for item in active
        if (
            int(item.get("assigned_allocation") or item.get("allocation_id") or 0)
            == allocation_id
            or item.get("node_name") == lane.node_name
        )
        and (
            item.get("aedt_backend") == "standalone"
            or item.get("scheduling_profile") == "fea_bursty"
        )
    ]
    if active_fea:
        raise ContinuationError("strict Full target node has active FEA work")
    exact_collisions = [
        item
        for item in collisions
        if item.get("name") == task_name or item.get("dedupe_key") == dedupe_key
    ]
    if exact_collisions:
        raise ContinuationError("Full continuation task name/dedupe already exists")
    return {
        "schema_version": "mft-goal-standard-full-live-preflight-v1",
        "observed_at_utc": _now(),
        "source_standard_task": copy.deepcopy(dict(source_task)),
        "source_standard_node": source_node,
        "target_lane": {
            "account_name": lane.account_name,
            "node_name": lane.node_name,
        },
        "capacity_query": _capacity_query(lane),
        "capacity": copy.deepcopy(dict(capacity)),
        "selected_allocation": allocation,
        "selected_allocation_id": allocation_id,
        "active_fea_on_target_count": 0,
        "collision_count": 0,
        "separate_strict_node_lane": True,
    }


def _capture_full_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
    task_name: str,
    workdir: str,
    solver_revision: str,
    library_revision: str,
    lane: StrictLane,
    license_snapshot: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment, core_evidence = production._submission_environment(  # noqa: SLF001
        stage="full",
        solver_revision=solver_revision,
        license_snapshot_path=license_snapshot,
    )
    captured: dict[str, Any] = {}

    class Response:
        status_code = 201

        @staticmethod
        def json() -> dict[str, int]:
            return {"task_id": 99999}

        @staticmethod
        def raise_for_status() -> None:
            return None

    def fake_post(
        _url: str,
        json: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Response:
        captured["payload"] = copy.deepcopy(json)
        return Response()

    requests_module = sys.modules.get("requests")
    if requests_module is None:
        requests_module = types.ModuleType("requests")
        sys.modules["requests"] = requests_module
    old_post = getattr(requests_module, "post", None)
    old_get = getattr(requests_module, "get", None)
    old_reconcile = scheduler_client.reconcile_task_record
    old_snapshot = scheduler_client.live_project_submission_snapshot
    depth_was_set = hasattr(scheduler_client._CAMPAIGN_LOCK_STATE, "depth")
    old_depth = getattr(scheduler_client._CAMPAIGN_LOCK_STATE, "depth", 0)
    requests_module.post = fake_post
    requests_module.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ContinuationError("unexpected GET while deriving Full payload")
    )
    scheduler_client.reconcile_task_record = lambda *_args, **_kwargs: None
    scheduler_client.live_project_submission_snapshot = (
        lambda *_args, **_kwargs: {"project_submission_slots": 1}
    )
    scheduler_client._CAMPAIGN_LOCK_STATE.depth = 1
    try:
        scheduler_client.submit_verification(
            task_name,
            workdir,
            params,
            profile,
            mem_mb=MEMORY_MB,
            cpus=CPUS,
            solver_revision=solver_revision,
            library_revision=library_revision,
            priority=PRIORITY,
            account_name=lane.account_name,
            node_name=lane.node_name,
            max_workers_per_node=MAX_WORKERS_PER_NODE,
            aedt_backend="standalone",
            submission_env=environment,
            required_hard_cap=500,
            max_project_active_tasks=500,
            scheduler_url=SCHEDULER_URL,
            node_name_policy="strict",
            submission_env_after_workdir=True,
            required_workdir_prefix="/enroot/",
        )
        retained = scheduler_client.retained_aedt_identity(
            task_name,
            params,
            profile,
            solver_revision,
            library_revision,
        )
    finally:
        if depth_was_set:
            scheduler_client._CAMPAIGN_LOCK_STATE.depth = old_depth
        else:
            del scheduler_client._CAMPAIGN_LOCK_STATE.depth
        scheduler_client.reconcile_task_record = old_reconcile
        scheduler_client.live_project_submission_snapshot = old_snapshot
        if old_post is None:
            delattr(requests_module, "post")
        else:
            requests_module.post = old_post
        if old_get is None:
            delattr(requests_module, "get")
        else:
            requests_module.get = old_get
    payload = captured.get("payload")
    if not isinstance(payload, dict) or not isinstance(retained, dict):
        raise ContinuationError("Full Scheduler payload derivation failed")
    command = str(payload.get("command") or "")
    random_delay = "sleep $((RANDOM % 300)); "
    solver = (
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--full --params cand.json; simulation_rc=$?;"
    )
    bounded = (
        f"timeout --signal=TERM --kill-after={KILL_GRACE_SECONDS}s "
        f"{SOLVER_TIMEOUT_SECONDS}s {solver}"
    )
    if command.count(random_delay) != 1 or command.count(solver) != 1:
        raise ContinuationError("reviewed Full command template drifted")
    payload["command"] = command.replace(random_delay, "", 1).replace(
        solver, bounded, 1
    )
    payload["timeout_seconds"] = SCHEDULER_TIMEOUT_SECONDS
    expected = {
        "name": task_name,
        "project": PROJECT,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "account_name": lane.account_name,
        "node_name": lane.node_name,
        "node_name_policy": "strict",
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if (
        drift
        or retained.get("stage") != "full"
        or retained.get("artifact_path", "").endswith("/full_model.aedt") is False
        or "MFT_STANDALONE_CORE_RUNTIME_LICENSE_REFRESH" not in payload["command"]
        or "case \"$MFT_WORKDIR\" in /enroot/*)" not in payload["command"]
    ):
        raise ContinuationError(f"derived Full payload drifted: {drift}")
    return payload, retained, core_evidence


def _candidate_authority(
    state_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    Path | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    str | None,
]:
    state, snapshot = _postsuccess_authority(state_path)
    if state["pending_count"] != 0:
        return state, snapshot, None, None, None, None, None
    if snapshot is None:
        return state, None, None, None, None, None, None
    row = _best_measured_row(snapshot)
    if row is None:
        return state, snapshot, None, None, None, None, None
    receipt, view, measured, source_node = _source_collection(state, row)
    return state, snapshot, row, receipt, view, measured, source_node


def _task_identity(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    stem = str(row["candidate_physics_sha256"])[:12]
    task_id = int(row["task_id"])
    return (
        f"mft-goal-postdeadline-full-from-standard-t{task_id}-{stem}-v1",
        f"mft_goal_postdeadline_full_from_standard_t{task_id}_{stem}_v1",
    )


def _plan_path(output_root: Path) -> Path:
    return output_root.resolve() / "prepared" / "full_continuation_plan.json"


def _attempt_path(output_root: Path) -> Path:
    return output_root.resolve() / "scheduler_post_attempt.json"


def _receipt_path(output_root: Path) -> Path:
    return output_root.resolve() / "submission_receipt.json"


def prepare(
    *,
    state_path: Path,
    output_root: Path,
    strict_lanes: Sequence[StrictLane],
    license_snapshot_directory: Path,
    reader: JsonReader = get_json,
    payload_builder: PayloadBuilder = _capture_full_payload,
) -> tuple[str, Path | None, dict[str, Any]]:
    if not strict_lanes or len(set(strict_lanes)) != len(strict_lanes):
        raise ContinuationError("strict Full lanes are absent or duplicated")
    state, snapshot, row, collection_path, view, measured, source_node = (
        _candidate_authority(state_path)
    )
    if state["pending_count"] != 0:
        return "pending_standard_collections", None, {
            "pending_count": state["pending_count"],
            "collection_count": state["collection_count"],
        }
    if row is None:
        return "terminal_no_measured_pass", None, {
            "pending_count": 0,
            "collection_count": state["collection_count"],
        }
    assert snapshot is not None
    assert collection_path is not None
    assert view is not None
    assert measured is not None
    assert source_node is not None
    params = {
        key: view["params"][key] for key in sorted(ALL_INPUT_KEYS)
    }
    if set(params) != set(ALL_INPUT_KEYS):
        raise ContinuationError("selected Standard candidate params are incomplete")
    profile, profile_source = promotion._full_profile()  # noqa: SLF001
    solver = production._require_revision(  # noqa: SLF001
        view["plan"]["solver_revision"], "solver_revision"
    )
    library = production._require_revision(  # noqa: SLF001
        view["plan"]["library_revision"], "library_revision"
    )
    license_path = fastlane._latest_license_snapshot(  # noqa: SLF001
        license_snapshot_directory
    )
    task_name, workdir = _task_identity(row)
    errors = []
    selected: tuple[
        StrictLane,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ] | None = None
    for lane in strict_lanes:
        if lane.node_name == source_node:
            errors.append(f"{lane.account_name}={lane.node_name}:same_source_node")
            continue
        try:
            payload, retained, core = payload_builder(
                params,
                profile,
                task_name,
                workdir,
                solver,
                library,
                lane,
                license_path,
            )
            preflight = _lane_preflight(
                lane=lane,
                source_node=source_node,
                task_name=task_name,
                dedupe_key=str(payload["dedupe_key"]),
                source_task_id=int(row["task_id"]),
                reader=reader,
            )
        except ContinuationError as exc:
            errors.append(f"{lane.account_name}={lane.node_name}:{exc}")
            continue
        selected = lane, payload, retained, core, preflight
        break
    if selected is None:
        return "measured_pass_waiting_for_strict_full_lane", None, {
            "candidate_physics_sha256": row["candidate_physics_sha256"],
            "standard_task_id": row["task_id"],
            "source_standard_node": source_node,
            "lane_errors": errors,
        }
    lane, payload, retained, core, preflight = selected
    target = output_root.resolve() / "prepared"
    plan_path = target / "full_continuation_plan.json"
    if plan_path.exists():
        plan = load_plan(plan_path)
        if (
            plan["candidate_physics_sha256"]
            != row["candidate_physics_sha256"]
            or plan["standard_task_id"] != row["task_id"]
            or plan["scheduler_payload"] != payload
        ):
            raise ContinuationError("existing continuation plan identity drifted")
        return "prepared_full_continuation", plan_path, plan
    if target.exists():
        raise ContinuationError("partial continuation prepare directory exists")
    staging = target.with_name(
        f".{target.name}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        params_path = _write_exclusive(staging / "full_params.json", params)
        profile_path = _write_exclusive(staging / "full_profile.json", profile)
        plan = _sealed(
            {
                "schema_version": PLAN_SCHEMA,
                "created_at_utc": _now(),
                **SAFETY_FLAGS,
                "campaign_id": CAMPAIGN_ID,
                "postsuccess_snapshot": _file_record(
                    Path(snapshot["measured_actual_observations_csv"]["path"]).parent
                    / "snapshot_manifest.json"
                ),
                "postsuccess_snapshot_payload_sha256": snapshot["payload_sha256"],
                "all_expected_standard_lanes_terminal": True,
                "selected_front_contract": (
                    "measured_hard_feasible_rank0_then_volume_loss_candidate"
                ),
                "candidate_physics_sha256": row["candidate_physics_sha256"],
                "standard_task_id": row["task_id"],
                "standard_collection": _file_record(collection_path),
                "standard_collection_payload_sha256": view["collection"][
                    "source_collection_receipt_payload_sha256"
                ],
                "standard_source_node": source_node,
                "measured_standard_evidence": measured,
                "measured_standard_hard_constraints_passed": True,
                "temperature_family_gate_contract": (
                    postsuccess._temperature_family_gate_contract()  # noqa: SLF001
                ),
                "temperature_family_actuals_C": {
                    "primary_winding": measured[
                        "actual_primary_winding_max_C"
                    ],
                    "secondary_winding": measured[
                        "actual_secondary_winding_max_C"
                    ],
                    "core": measured["actual_core_max_C"],
                },
                "aggregate_winding_temperature_audit_alias": {
                    "field": "actual_winding_max_C",
                    "actual_C": measured["actual_winding_max_C"],
                    "audit_only": True,
                    "hard_gate": False,
                },
                "fixed_cooling_unchanged": True,
                "full_params": {
                    "path": "full_params.json",
                    "sha256": _file_record(params_path)["sha256"],
                    "size_bytes": params_path.stat().st_size,
                },
                "full_profile": {
                    "path": "full_profile.json",
                    "sha256": _file_record(profile_path)["sha256"],
                    "size_bytes": profile_path.stat().st_size,
                    "reviewed_source": profile_source,
                },
                "solver_revision": solver,
                "library_revision": library,
                "task_name": task_name,
                "workdir": workdir,
                "target_lane": {
                    "account_name": lane.account_name,
                    "node_name": lane.node_name,
                    "node_name_policy": "strict",
                    "same_node_as_task_id": 0,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                },
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "scheduler_timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
                    "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                },
                "license_snapshot": _file_record(license_path),
                "core_policy": core,
                "prepare_live_preflight": preflight,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": _payload_sha256(payload),
                "retained_full_aedt_bundle": retained,
                "retained_full_aedt_bundle_sha256": _payload_sha256(retained),
                "attempt_ledger_path": str(_attempt_path(output_root)),
                "submission_receipt_path": str(_receipt_path(output_root)),
                "authorization_token": AUTHORIZATION_TOKEN,
                "maximum_scheduler_posts": 1,
                "scheduler_post_attempts_consumed": 0,
                "scheduler_mutation_performed": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_repository_modified": False,
            }
        )
        _write_exclusive(staging / plan_path.name, plan)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return "prepared_full_continuation", plan_path, plan


def _contained(root: Path, record: Any, label: str) -> Path:
    if (
        not isinstance(record, Mapping)
        or set(record) - {"path", "sha256", "size_bytes"}
        or set(record) != {"path", "sha256", "size_bytes"}
    ):
        raise ContinuationError(f"{label} record is malformed")
    candidate = (root / str(record["path"])).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContinuationError(f"{label} escapes prepare directory") from exc
    if _file_record(candidate) != {
        "path": str(candidate),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }:
        raise ContinuationError(f"{label} bytes drifted")
    return candidate


def load_plan(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    plan = _validate_seal(
        _read_json(resolved, "continuation plan"), PLAN_SCHEMA, "continuation plan"
    )
    measured = plan.get("measured_standard_evidence")
    temperature_actuals = plan.get("temperature_family_actuals_C")
    aggregate_alias = plan.get(
        "aggregate_winding_temperature_audit_alias"
    )
    if (
        any(plan.get(key) is not expected for key, expected in SAFETY_FLAGS.items())
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("all_expected_standard_lanes_terminal") is not True
        or plan.get("measured_standard_hard_constraints_passed") is not True
        or plan.get("temperature_family_gate_contract")
        != postsuccess._temperature_family_gate_contract()  # noqa: SLF001
        or not isinstance(measured, Mapping)
        or not postsuccess._temperature_family_evidence_valid(  # noqa: SLF001
            measured
        )
        or temperature_actuals
        != {
            "primary_winding": measured.get(
                "actual_primary_winding_max_C"
            ),
            "secondary_winding": measured.get(
                "actual_secondary_winding_max_C"
            ),
            "core": measured.get("actual_core_max_C"),
        }
        or aggregate_alias
        != {
            "field": "actual_winding_max_C",
            "actual_C": measured.get("actual_winding_max_C"),
            "audit_only": True,
            "hard_gate": False,
        }
        or plan.get("fixed_cooling_unchanged") is not True
        or plan.get("maximum_scheduler_posts") != 1
        or plan.get("scheduler_post_attempts_consumed") != 0
        or plan.get("scheduler_mutation_performed") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("authorization_token") != AUTHORIZATION_TOKEN
        or plan.get("attempt_ledger_path") != str(_attempt_path(resolved.parents[1]))
        or plan.get("submission_receipt_path") != str(_receipt_path(resolved.parents[1]))
    ):
        raise ContinuationError("continuation plan safety contract drifted")
    _contained(resolved.parent, plan.get("full_params"), "Full params")
    profile_record = plan.get("full_profile")
    if not isinstance(profile_record, Mapping):
        raise ContinuationError("Full profile record is absent")
    _contained(
        resolved.parent,
        {key: profile_record[key] for key in ("path", "sha256", "size_bytes")},
        "Full profile",
    )
    payload = plan.get("scheduler_payload")
    retained = plan.get("retained_full_aedt_bundle")
    if (
        not isinstance(payload, Mapping)
        or plan.get("scheduler_payload_sha256") != _payload_sha256(payload)
        or not isinstance(retained, Mapping)
        or plan.get("retained_full_aedt_bundle_sha256")
        != _payload_sha256(retained)
        or payload.get("name") != plan.get("task_name")
        or payload.get("project") != PROJECT
        or payload.get("node_name") != plan.get("target_lane", {}).get("node_name")
        or payload.get("account_name")
        != plan.get("target_lane", {}).get("account_name")
        or payload.get("node_name_policy") != "strict"
        or payload.get("cpus") != CPUS
        or payload.get("memory_mb") != MEMORY_MB
        or payload.get("timeout_seconds") != SCHEDULER_TIMEOUT_SECONDS
    ):
        raise ContinuationError("continuation Scheduler payload drifted")
    return plan


def submit_prepared(
    *,
    plan_path: Path,
    state_path: Path,
    license_snapshot_directory: Path,
    authorization: str,
    reader: JsonReader = get_json,
    post_once: PostOnce = _post_json_once,
    lock_factory: LockFactory = scheduler_client.campaign_mutation_lock,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    output_root = plan_path.resolve().parents[1]
    attempt_path = _attempt_path(output_root)
    receipt_path = _receipt_path(output_root)
    if authorization != AUTHORIZATION_TOKEN:
        raise ContinuationError("Full continuation POST authorization is invalid")
    if receipt_path.exists():
        return _validate_seal(
            _read_json(receipt_path, "submission receipt"),
            RECEIPT_SCHEMA,
            "submission receipt",
        )
    if attempt_path.exists():
        raise ContinuationError(
            "Full continuation attempt already consumed; re-POST is forbidden"
        )
    with lock_factory():
        if receipt_path.exists():
            return _validate_seal(
                _read_json(receipt_path, "submission receipt"),
                RECEIPT_SCHEMA,
                "submission receipt",
            )
        if attempt_path.exists():
            raise ContinuationError(
                "Full continuation attempt already consumed inside lock"
            )
        state, snapshot, row, collection_path, _view, measured, source_node = (
            _candidate_authority(state_path)
        )
        if (
            state["pending_count"] != 0
            or snapshot is None
            or row is None
            or collection_path is None
            or measured is None
            or source_node is None
            or row["candidate_physics_sha256"]
            != plan["candidate_physics_sha256"]
            or row["task_id"] != plan["standard_task_id"]
            or _file_record(collection_path) != plan["standard_collection"]
            or measured != plan["measured_standard_evidence"]
        ):
            raise ContinuationError(
                "measured Standard authority changed before Full POST"
            )
        latest_license = fastlane._latest_license_snapshot(  # noqa: SLF001
            license_snapshot_directory
        )
        production._validate_license_snapshot(latest_license)  # noqa: SLF001
        lane = StrictLane(
            str(plan["target_lane"]["account_name"]),
            str(plan["target_lane"]["node_name"]),
        )
        preflight = _lane_preflight(
            lane=lane,
            source_node=source_node,
            task_name=plan["task_name"],
            dedupe_key=str(plan["scheduler_payload"]["dedupe_key"]),
            source_task_id=int(plan["standard_task_id"]),
            reader=reader,
        )
        attempt = _sealed(
            {
                "schema_version": ATTEMPT_SCHEMA,
                "created_at_utc": _now(),
                **SAFETY_FLAGS,
                "plan": _file_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "candidate_physics_sha256": plan[
                    "candidate_physics_sha256"
                ],
                "standard_task_id": plan["standard_task_id"],
                "task_name": plan["task_name"],
                "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
                "locked_pre_submit_live_preflight": preflight,
                "fresh_license_snapshot": _file_record(latest_license),
                "post_attempt_consumed": True,
                "maximum_scheduler_posts": 1,
                "scheduler_post_outcome_known": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_repository_modified": False,
            }
        )
        _write_exclusive(attempt_path, attempt)
        status, response = post_once(
            f"{SCHEDULER_URL}/api/tasks", plan["scheduler_payload"]
        )
        if status not in {200, 201} or not isinstance(response, Mapping):
            raise ContinuationError(
                f"Scheduler POST outcome is not successful: HTTP {status}"
            )
        task_id = response.get("task_id", response.get("id"))
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise ContinuationError("Scheduler POST returned no durable task ID")
        readback = reader(f"/api/tasks/{task_id}", None)
        if (
            not isinstance(readback, Mapping)
            or int(readback.get("task_id") or readback.get("id") or 0)
            != task_id
            or readback.get("name") != plan["task_name"]
            or readback.get("dedupe_key")
            != plan["scheduler_payload"]["dedupe_key"]
            or readback.get("project") != PROJECT
            or readback.get("requested_node_name") != lane.node_name
            or readback.get("requested_node_name_policy") != "strict"
            or readback.get("cpus") != CPUS
            or readback.get("memory_mb") != MEMORY_MB
        ):
            raise ContinuationError("submitted Full task readback drifted")
        receipt = _sealed(
            {
                "schema_version": RECEIPT_SCHEMA,
                "created_at_utc": _now(),
                **SAFETY_FLAGS,
                "status": "full_submitted_pending_actual_result",
                "plan": _file_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "attempt_ledger": _file_record(attempt_path),
                "attempt_ledger_payload_sha256": attempt["payload_sha256"],
                "candidate_physics_sha256": plan[
                    "candidate_physics_sha256"
                ],
                "standard_task_id": plan["standard_task_id"],
                "full_task_id": task_id,
                "task_name": plan["task_name"],
                "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
                "target_lane": copy.deepcopy(plan["target_lane"]),
                "task_readback": copy.deepcopy(dict(readback)),
                "task_readback_sha256": _payload_sha256(readback),
                "scheduler_post_attempts_consumed": 1,
                "scheduler_mutation_performed": status == 201,
                "scheduler_project_mutation_performed": False,
                "scheduler_repository_modified": False,
            }
        )
        _write_exclusive(receipt_path, receipt)
        return receipt


def cycle(
    *,
    state_path: Path,
    output_root: Path,
    strict_lanes: Sequence[StrictLane],
    license_snapshot_directory: Path,
    authorization: str | None = None,
    reader: JsonReader = get_json,
    payload_builder: PayloadBuilder = _capture_full_payload,
    post_once: PostOnce = _post_json_once,
    lock_factory: LockFactory = scheduler_client.campaign_mutation_lock,
) -> dict[str, Any]:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = _receipt_path(root)
    if receipt_path.exists():
        receipt = _validate_seal(
            _read_json(receipt_path, "submission receipt"),
            RECEIPT_SCHEMA,
            "submission receipt",
        )
        status = "full_submitted_pending_actual_result"
        plan_path: Path | None = Path(receipt["plan"]["path"])
    elif _attempt_path(root).exists():
        status = "submission_outcome_uncertain_no_repost"
        plan_path = _plan_path(root) if _plan_path(root).exists() else None
        receipt = None
    else:
        status, plan_path, detail = prepare(
            state_path=state_path,
            output_root=root,
            strict_lanes=strict_lanes,
            license_snapshot_directory=license_snapshot_directory,
            reader=reader,
            payload_builder=payload_builder,
        )
        receipt = None
        if (
            status == "prepared_full_continuation"
            and authorization == AUTHORIZATION_TOKEN
            and plan_path is not None
        ):
            receipt = submit_prepared(
                plan_path=plan_path,
                state_path=state_path,
                license_snapshot_directory=license_snapshot_directory,
                authorization=authorization,
                reader=reader,
                post_once=post_once,
                lock_factory=lock_factory,
            )
            status = "full_submitted_pending_actual_result"
        elif status == "prepared_full_continuation":
            status = "prepared_waiting_for_post_authorization"
    state = _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "observed_at_utc": _now(),
            **SAFETY_FLAGS,
            "status": status,
            "source_postsuccess_state": str(state_path.resolve()),
            "plan": _file_record(plan_path) if plan_path is not None else None,
            "attempt_ledger": (
                _file_record(_attempt_path(root))
                if _attempt_path(root).exists()
                else None
            ),
            "submission_receipt": (
                _file_record(receipt_path) if receipt_path.exists() else None
            ),
            "full_task_id": (
                receipt["full_task_id"] if receipt is not None else None
            ),
            "maximum_scheduler_posts": 1,
            "scheduler_post_attempts_consumed": int(
                _attempt_path(root).exists()
            ),
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
        }
    )
    _write_atomic(root / "state.json", state)
    return state


def watch(
    *,
    interval_seconds: int,
    **kwargs: Any,
) -> dict[str, Any]:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS
    ):
        raise ContinuationError("watch interval is invalid")
    while True:
        try:
            state = cycle(**kwargs)
        except Exception as exc:
            root = Path(kwargs["output_root"]).resolve()
            state = _sealed(
                {
                    "schema_version": STATE_SCHEMA,
                    "observed_at_utc": _now(),
                    **SAFETY_FLAGS,
                    "status": "fail_closed_retrying_get_gates",
                    "source_postsuccess_state": str(
                        Path(kwargs["state_path"]).resolve()
                    ),
                    "plan": None,
                    "attempt_ledger": (
                        _file_record(_attempt_path(root))
                        if _attempt_path(root).exists()
                        else None
                    ),
                    "submission_receipt": None,
                    "full_task_id": None,
                    "maximum_scheduler_posts": 1,
                    "scheduler_post_attempts_consumed": int(
                        _attempt_path(root).exists()
                    ),
                    "scheduler_project_mutation_performed": False,
                    "scheduler_repository_modified": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            _write_atomic(root / "state.json", state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "full_task_id": state["full_task_id"],
                    "post_attempts": state[
                        "scheduler_post_attempts_consumed"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state["status"] in {
            "full_submitted_pending_actual_result",
            "terminal_no_measured_pass",
            "submission_outcome_uncertain_no_repost",
        }:
            return state
        time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous, one-shot same-candidate Full continuation after "
            "authenticated post-deadline Standard results."
        )
    )
    parser.add_argument("--postsuccess-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--strict-lane",
        type=_parse_lane,
        action="append",
        required=True,
        help="repeat ACCOUNT_NAME=NODE_NAME in preferred order",
    )
    parser.add_argument(
        "--license-snapshot-directory", type=Path, required=True
    )
    parser.add_argument("--authorize-post")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "state_path": args.postsuccess_state,
        "output_root": args.output_root,
        "strict_lanes": args.strict_lane,
        "license_snapshot_directory": args.license_snapshot_directory,
        "authorization": args.authorize_post,
    }
    state = (
        watch(**kwargs, interval_seconds=args.interval)
        if args.watch
        else cycle(**kwargs)
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "state": str(args.output_root.resolve() / "state.json"),
                "full_task_id": state["full_task_id"],
                "scientific_pass_claimed": False,
                "promotion_completed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
