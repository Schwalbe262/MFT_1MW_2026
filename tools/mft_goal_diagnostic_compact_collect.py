"""GET-only collector and global NDS publisher for the compact exact100 scout.

This module is deliberately isolated from every production collector.  It
accepts only the sealed diagnostic compact offload plan and its exact100
submission receipt, observes Scheduler state with GET, and reads immutable
worker artifacts with SFTP.  It has no Scheduler mutation method.

An early aggregate is published once at least ten terminal-success seeds have
authenticated.  The exact100 aggregate is published only after all one hundred
seed populations authenticate.  Both products remain surrogate screening
evidence: the raw ``C_rx_rx_F`` UCB gate and every resulting front are
provisional and can never confer production or final-design authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import posixpath
import tempfile
import time
from typing import Any, Iterable, Mapping, Protocol
import urllib.request

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402
from tools import mft_goal_global_pareto as global_pareto  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


COLLECTION_RECORD_SCHEMA = (
    "mft-goal-diagnostic-compact-terminal-collection-v1"
)
COLLECTOR_STATUS_SCHEMA = "mft-goal-diagnostic-compact-collector-status-v1"
SCREENING_AUTHORITY_SCHEMA = (
    "mft-goal-diagnostic-compact-global-nds-authority-v1"
)
EARLY_SUCCESS_MINIMUM = 10
FINAL_SUCCESS_COUNT = offload.EXACT_TASK_COUNT
TERMINAL_POPULATION_COUNT = scout.POPULATION
RAW_CRX_PHYSICAL_COLUMN = f"physical_G:{scout.RAW_CRX_CONSTRAINT_NAME}"
RAW_CRX_NORMALIZED_COLUMN = f"normalized_G:{scout.RAW_CRX_CONSTRAINT_NAME}"
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_TERMINAL_TABLE_BYTES = 256 * 1024 * 1024
DEFAULT_SCHEDULER_URL = offload.DEFAULT_SCHEDULER_URL
DEFAULT_ACCOUNTS = offload.DEFAULT_ACCOUNTS
DEFAULT_SCHEDULER_SOURCE = offload.DEFAULT_SCHEDULER_SOURCE
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})
OBJECTIVE_COLUMNS = ("objective_volume_L", "objective_total_loss_W")
IDENTITY_COLUMNS = (
    "dataset_sha256",
    "evaluation_model_sha256",
    "constraint_spec_sha256",
    "cooling_contract_sha256",
    "operating_point_sha256",
)
FAIL_CLOSED_FLAGS = {
    "screening_only": True,
    "production_eligible": False,
    "final_design_claim_allowed": False,
    "automatic_promotion_allowed": False,
}


class SchedulerReader(Protocol):
    get_count: int

    def get_task(self, task_id: int) -> dict[str, Any]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("collector value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_sealed(
    value: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("collector seal must be an object")
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} seal mismatch")
    return result


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable snapshot, or authenticate the existing snapshot."""

    if path.exists():
        if _read_json(path) != dict(value):
            raise RuntimeError(f"immutable snapshot changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            if _read_json(path) != dict(value):
                raise RuntimeError(f"immutable snapshot raced: {path}")
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _canonical_bool(series: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RuntimeError(f"{label} is not canonical boolean evidence")
    return normalized.eq("true").to_numpy(dtype=bool)


def _one_string(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        raise RuntimeError(f"terminal table misses identity column {column}")
    values = {str(value).strip().lower() for value in frame[column]}
    if len(values) != 1 or not next(iter(values)):
        raise RuntimeError(f"terminal table {column} identity is mixed")
    return next(iter(values))


class ReadOnlySchedulerClient:
    """Scheduler client exposing only authenticated GET."""

    def __init__(self, base_url: str = DEFAULT_SCHEDULER_URL, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.get_count = 0

    def get_task(self, task_id: int) -> dict[str, Any]:
        self.get_count += 1
        request = urllib.request.Request(
            self.base_url + f"/api/tasks/{int(task_id)}",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Scheduler task {int(task_id)} returned invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Scheduler task {int(task_id)} is not an object")
        return value


def authenticate_context(
    *,
    plan_path: Path,
    receipt_path: Path,
    scheduler_url: str,
) -> dict[str, Any]:
    """Bind the collector to the exact local plan, receipt, and task ledger."""

    plan, _deployment, tasks, authentication = offload.authenticate_plan(
        plan_path
    )
    if (
        plan.get("task_count") != FINAL_SUCCESS_COUNT
        or tuple(offload._authorized_plan_seeds(plan)) != offload.EXACT_SEEDS
    ):
        raise RuntimeError("collector plan is not the compact exact100 authority")
    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            offload.scheduler_payload(
                plan=plan,
                task=task,
                priority=offload.SCHEDULER_PRIORITY,
            )
            for task in tasks
        )
    }
    receipt_raw = _read_json(receipt_path)
    receipt = offload._validate_receipt(
        receipt_raw,
        plan=plan,
        expected=expected,
        authentication=authentication,
        ready=receipt_raw.get("remote_ready") or {},
        scheduler_url=scheduler_url,
    )
    task_by_seed = {int(task["seed"]): task for task in tasks}
    entries = []
    for row in sorted(receipt["tasks"], key=lambda item: int(item["seed"])):
        seed = int(row["seed"])
        task = task_by_seed.get(seed)
        expected_payload = expected.get(str(row["dedupe_key"]))
        if (
            task is None
            or expected_payload is None
            or int(expected_payload["payload_json"]["seed"]) != seed
        ):
            raise RuntimeError("receipt/task exact100 mapping changed")
        entries.append(
            {
                "seed": seed,
                "task_id": int(row["task_id"]),
                "dedupe_key": str(row["dedupe_key"]),
                "task": task,
                "scheduler_payload": expected_payload,
            }
        )
    if (
        len(entries) != FINAL_SUCCESS_COUNT
        or len({entry["task_id"] for entry in entries})
        != FINAL_SUCCESS_COUNT
    ):
        raise RuntimeError("collector receipt is not a unique exact100 mapping")
    return {
        "plan": plan,
        "authentication": authentication,
        "receipt": receipt,
        "entries": entries,
        "scheduler_url": scheduler_url.rstrip("/"),
        "plan_path": str(plan_path.resolve(strict=True)),
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "plan_file_sha256": _sha256_file(plan_path.resolve(strict=True)),
        "receipt_file_sha256": _sha256_file(receipt_path.resolve(strict=True)),
    }


def observe_tasks(
    context: Mapping[str, Any],
    scheduler: SchedulerReader,
) -> list[dict[str, Any]]:
    rows = []
    for entry in context["entries"]:
        try:
            detail = scheduler.get_task(int(entry["task_id"]))
            task_id, status = offload._task_authentication(
                detail,
                entry["scheduler_payload"],
                label="diagnostic collector GET",
            )
            raw_exit = detail.get("exit_code")
            if isinstance(raw_exit, bool):
                raise RuntimeError("Scheduler exit_code has invalid type")
            exit_code = None if raw_exit is None else int(raw_exit)
            account_name = str(detail.get("account_name") or "")
            success = status == "completed" and exit_code == 0
            if success and not account_name:
                raise RuntimeError("terminal-success task has no owning account")
            rows.append(
                {
                    **entry,
                    "task_id": task_id,
                    "scheduler_status": status,
                    "exit_code": exit_code,
                    "account_name": account_name or None,
                    "actual_node_name": (
                        str(
                            detail.get("actual_node_name")
                            or detail.get("allocation_node_name")
                            or ""
                        )
                        or None
                    ),
                    "finished_at": detail.get("finished_at"),
                    "terminal": status in TERMINAL_STATES,
                    "terminal_success": success,
                    "observation_error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **entry,
                    "scheduler_status": "query_error",
                    "exit_code": None,
                    "account_name": None,
                    "actual_node_name": None,
                    "finished_at": None,
                    "terminal": False,
                    "terminal_success": False,
                    "observation_error": f"{type(exc).__name__}:{exc}",
                }
            )
    return rows


def _artifact_record(
    inventory: Mapping[str, Any], role: str, expected_name: str
) -> dict[str, Any]:
    raw = inventory.get(role)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"result inventory misses {role}")
    path = str(raw.get("path") or "")
    digest = str(raw.get("sha256") or "").lower()
    size = raw.get("size_bytes")
    if (
        path != expected_name
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError(f"result inventory {role} identity is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _validate_result(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result = scout.goal_launch._validate_seal(
        _read_json(path),
        schema=scout.RESULT_SCHEMA,
    )
    task = entry["task"]
    activation = task["activation"]
    profile = activation["manufacturing_search_profile"]
    if (
        result.get("campaign_id") != scout.CAMPAIGN_ID
        or result.get("task_payload_sha256") != task["payload_sha256"]
        or int(result.get("seed", -1)) != int(entry["seed"])
        or result.get("fixed_primary_turns") != scout.FIXED_PRIMARY_TURNS
        or result.get("population") != TERMINAL_POPULATION_COUNT
        or result.get("generations") != scout.GENERATIONS
        or result.get("terminal_population_count")
        != TERMINAL_POPULATION_COUNT
        or result.get("manufacturing_search_profile_payload_sha256")
        != profile["payload_sha256"]
        or result.get("geometry_constraint_profile_sha256")
        != profile["geometry_constraint_profile_sha256"]
        or result.get("raw_same_metric_C_rx_rx_F_UCB_gate_active") is not True
        or result.get("fixed_lm2mh_resonance_contract_sha256")
        != activation["fixed_lm2mh_resonance_contract_sha256"]
        or result.get("screening_only") is not True
        or result.get("production_eligible") is not False
        or result.get("final_design_claim_allowed") is not False
        or result.get("fresh512_activation_evidence") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("fea_submission_performed") is not False
        or result.get("symmetric_FEA_validation_still_required") is not True
        or result.get("scheduler_write_performed") is not False
        or result.get("scheduler_submission_performed") is not False
    ):
        raise RuntimeError("diagnostic terminal result identity mismatch")
    inventory = result.get("artifact_inventory")
    if (
        not isinstance(inventory, Mapping)
        or result.get("artifact_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise RuntimeError("diagnostic result artifact inventory seal mismatch")
    selected = {
        "terminal_physical_candidates": _artifact_record(
            inventory,
            "terminal_physical_candidates",
            "terminal_physical_candidates.csv",
        ),
        "terminal_physical_candidates_manifest": _artifact_record(
            inventory,
            "terminal_physical_candidates_manifest",
            "terminal_physical_candidates.manifest.json",
        ),
    }
    return result, selected


def _validate_terminal_table(
    *,
    csv_path: Path,
    manifest_path: Path,
    entry: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = scout.goal_launch._validate_seal(
        _read_json(manifest_path),
        schema=preflight.GOAL_TERMINAL_TABLE_SCHEMA,
    )
    task = entry["task"]
    activation = task["activation"]
    source = manifest.get("source_identity") or {}
    csv_record = manifest.get("csv") or {}
    if (
        manifest.get("row_count") != TERMINAL_POPULATION_COUNT
        or manifest.get("terminal_population_index_min") != 0
        or manifest.get("terminal_population_index_max")
        != TERMINAL_POPULATION_COUNT - 1
        or manifest.get("one_row_per_terminal_individual") is not True
        or manifest.get("physical_deduplication_key")
        != "physical_geometry_sha256"
        or manifest.get("global_pareto_provenance_ready") is not True
        or csv_record.get("path") != csv_path.name
        or csv_record.get("sha256") != _sha256_file(csv_path)
        or int(csv_record.get("size_bytes", -1)) != csv_path.stat().st_size
        or csv_record.get("sha256")
        != selected["terminal_physical_candidates"]["sha256"]
        or int(csv_record.get("size_bytes", -1))
        != selected["terminal_physical_candidates"]["size_bytes"]
        or int(source.get("seed", -1)) != int(entry["seed"])
        or str(source.get("task_id") or "") != str(entry["task_id"])
        or source.get("bundle_id") != task["payload_sha256"]
        or source.get("island_id") != "diagnostic-n1-6-compact"
        or source.get("dataset_sha256")
        != activation["source_identity"]["dataset_sha256"]
        or source.get("model_artifacts_sha256")
        != activation["source_identity"]["evaluation_model_sha256"]
        or source.get("temperature_contract_sha256")
        != task["temperature_contract_sha256"]
        or source.get("hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
        or manifest.get("temperature_contract_sha256")
        != task["temperature_contract_sha256"]
        or manifest.get("hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
        or manifest.get("fixed_lm2mh_resonance_contract_sha256")
        != activation["fixed_lm2mh_resonance_contract_sha256"]
    ):
        raise RuntimeError("diagnostic terminal table manifest mismatch")

    frame = pd.read_csv(csv_path)
    if (
        len(frame) != TERMINAL_POPULATION_COUNT
        or list(frame.columns) != list(manifest.get("columns") or [])
    ):
        raise RuntimeError("diagnostic terminal table shape/columns mismatch")
    indices = pd.to_numeric(
        frame["terminal_population_index"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.array_equal(
        indices, np.arange(TERMINAL_POPULATION_COUNT, dtype=float)
    ):
        raise RuntimeError("terminal population index is not exactly 0..319")
    if not _canonical_bool(frame["decoder_valid"], "decoder_valid").all():
        raise RuntimeError("terminal population contains decoder-invalid rows")
    for name, expected in (
        ("source_seed", str(entry["seed"])),
        ("source_task_id", str(entry["task_id"])),
        ("source_bundle_id", task["payload_sha256"]),
        ("source_island_id", "diagnostic-n1-6-compact"),
    ):
        observed = {str(value) for value in frame[name]}
        if observed != {expected}:
            raise RuntimeError(f"terminal table {name} identity mismatch")
    identities = {name: _one_string(frame, name) for name in IDENTITY_COLUMNS}
    if (
        identities["dataset_sha256"]
        != activation["source_identity"]["dataset_sha256"]
        or identities["evaluation_model_sha256"]
        != activation["source_identity"]["evaluation_model_sha256"]
        or _one_string(frame, "evaluation_temperature_contract_sha256")
        != task["temperature_contract_sha256"]
        or _one_string(frame, "evaluation_hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
    ):
        raise RuntimeError("terminal table scientific identity mismatch")
    physical = sorted(
        column for column in frame if column.startswith("physical_G:")
    )
    normalized = sorted(
        column for column in frame if column.startswith("normalized_G:")
    )
    if (
        not physical
        or [name.removeprefix("physical_G:") for name in physical]
        != [name.removeprefix("normalized_G:") for name in normalized]
        or RAW_CRX_PHYSICAL_COLUMN not in physical
        or RAW_CRX_NORMALIZED_COLUMN not in normalized
    ):
        raise RuntimeError("terminal table raw C_rx_rx_F constraint authority missing")
    for column in (*OBJECTIVE_COLUMNS, *physical, *normalized):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"terminal table {column} is non-finite")
    return {
        "manifest_payload_sha256": manifest["payload_sha256"],
        "identities": identities,
        "physical_constraint_columns": physical,
        "normalized_constraint_columns": normalized,
    }


def _validate_existing_collection(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    record = _validate_sealed(
        _read_json(path), schema=COLLECTION_RECORD_SCHEMA
    )
    task_dir = path.parent
    artifacts = record.get("artifacts") or {}
    for name in (
        "result.json",
        "terminal_physical_candidates.csv",
        "terminal_physical_candidates.manifest.json",
    ):
        artifact = artifacts.get(name) or {}
        local = task_dir / name
        if (
            not local.is_file()
            or artifact.get("sha256") != _sha256_file(local)
            or int(artifact.get("size_bytes", -1)) != local.stat().st_size
        ):
            raise RuntimeError("existing diagnostic collection bytes changed")
    if (
        record.get("task_id") != int(entry["task_id"])
        or record.get("seed") != int(entry["seed"])
        or record.get("task_payload_sha256")
        != entry["task"]["payload_sha256"]
        or any(record.get(name) != expected for name, expected in FAIL_CLOSED_FLAGS.items())
    ):
        raise RuntimeError("existing diagnostic collection identity changed")
    return record


def collect_terminal_success(
    *,
    context: Mapping[str, Any],
    observed: Mapping[str, Any],
    connection: transport._PersistentAccountConnection,
    output_root: Path,
    retries: int = 3,
) -> dict[str, Any]:
    """Read and authenticate one terminal-success task without Scheduler writes."""

    if (
        observed.get("terminal_success") is not True
        or observed.get("scheduler_status") != "completed"
        or observed.get("exit_code") != 0
    ):
        raise RuntimeError("only terminal-success tasks may be collected")
    task_id = int(observed["task_id"])
    task_dir = output_root / f"task-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = task_dir / "collection_record.json"
    if receipt_path.is_file():
        return _validate_existing_collection(receipt_path, entry=observed)

    remote_root = posixpath.join(
        str(context["plan"]["remote_bundle"]).rstrip("/"),
        "runs",
        f"task-{task_id}",
    )
    result_path = task_dir / "result.json"
    result_download = transport._download_verified_sftp(
        connection,
        posixpath.join(remote_root, "result.json"),
        result_path,
        retries=retries,
        max_bytes=MAX_RESULT_BYTES,
    )
    result, selected = _validate_result(result_path, entry=observed)
    downloads: dict[str, Mapping[str, Any]] = {"result.json": result_download}
    limits = {
        "terminal_physical_candidates.csv": MAX_TERMINAL_TABLE_BYTES,
        "terminal_physical_candidates.manifest.json": MAX_MANIFEST_BYTES,
    }
    for role, record in selected.items():
        name = record["path"]
        downloads[name] = transport._download_verified_sftp(
            connection,
            posixpath.join(remote_root, name),
            task_dir / name,
            retries=retries,
            expected_identity={
                "bytes": record["size_bytes"],
                "sha256": record["sha256"],
            },
            max_bytes=limits[name],
        )
    table_evidence = _validate_terminal_table(
        csv_path=task_dir / "terminal_physical_candidates.csv",
        manifest_path=(
            task_dir / "terminal_physical_candidates.manifest.json"
        ),
        entry=observed,
        selected=selected,
    )
    artifacts = {
        name: {
            "path": name,
            "sha256": _sha256_file(task_dir / name),
            "size_bytes": (task_dir / name).stat().st_size,
            "transport": downloads[name]["transport"],
        }
        for name in (
            "result.json",
            "terminal_physical_candidates.csv",
            "terminal_physical_candidates.manifest.json",
        )
    }
    record = _sealed(
        {
            "schema_version": COLLECTION_RECORD_SCHEMA,
            "campaign_id": scout.CAMPAIGN_ID,
            "bundle_id": context["plan"]["bundle_id"],
            "task_id": task_id,
            "seed": int(observed["seed"]),
            "task_payload_sha256": observed["task"]["payload_sha256"],
            "scheduler_status": "completed",
            "exit_code": 0,
            "account_name": observed["account_name"],
            "actual_node_name": observed["actual_node_name"],
            "result_payload_sha256": result["payload_sha256"],
            "terminal_manifest_payload_sha256": table_evidence[
                "manifest_payload_sha256"
            ],
            "terminal_population_count": TERMINAL_POPULATION_COUNT,
            "identities": table_evidence["identities"],
            "objective_columns": list(OBJECTIVE_COLUMNS),
            "physical_constraint_columns": table_evidence[
                "physical_constraint_columns"
            ],
            "normalized_constraint_columns": table_evidence[
                "normalized_constraint_columns"
            ],
            "artifacts": artifacts,
            "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
            "raw_same_metric_C_rx_rx_F_front_classification": (
                "provisional_surrogate_screening_only"
            ),
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_access_mode": "GET_status_plus_read_only_SFTP",
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "collected_at": _now(),
        }
    )
    _atomic_json(receipt_path, record)
    return record


def terminal_population_manifest(
    *,
    records: Iterable[Mapping[str, Any]],
    output_root: Path,
    snapshot_class: str,
) -> dict[str, Any]:
    ordered = sorted(
        (copy.deepcopy(dict(record)) for record in records),
        key=lambda record: int(record["seed"]),
    )
    if len(ordered) < EARLY_SUCCESS_MINIMUM:
        raise RuntimeError("global NDS requires at least ten terminal successes")
    if snapshot_class not in {
        "provisional_first_threshold",
        "final_integrated_exact100",
    }:
        raise RuntimeError("unknown diagnostic NDS snapshot class")
    if (
        snapshot_class == "final_integrated_exact100"
        and (
            len(ordered) != FINAL_SUCCESS_COUNT
            or [int(record["seed"]) for record in ordered]
            != list(offload.EXACT_SEEDS)
        )
    ):
        raise RuntimeError("final integrated NDS requires exact100 seed coverage")
    first = ordered[0]
    for record in ordered:
        if (
            record.get("identities") != first["identities"]
            or record.get("objective_columns") != list(OBJECTIVE_COLUMNS)
            or record.get("physical_constraint_columns")
            != first["physical_constraint_columns"]
            or record.get("normalized_constraint_columns")
            != first["normalized_constraint_columns"]
            or any(
                record.get(name) != expected
                for name, expected in FAIL_CLOSED_FLAGS.items()
            )
            or record.get("raw_same_metric_C_rx_rx_F_UCB_gate_active")
            is not True
        ):
            raise RuntimeError("diagnostic terminal collections mix authority")
    population_records = []
    for record in ordered:
        task_id = int(record["task_id"])
        table = output_root / f"task-{task_id}" / (
            "terminal_physical_candidates.csv"
        )
        if (
            not table.is_file()
            or _sha256_file(table)
            != record["artifacts"]["terminal_physical_candidates.csv"][
                "sha256"
            ]
        ):
            raise RuntimeError("diagnostic terminal table collection changed")
        relative = table.resolve().relative_to(output_root.resolve()).as_posix()
        population_records.append(
            {
                "seed": int(record["seed"]),
                "task_id": task_id,
                "bundle_id": record["task_payload_sha256"],
                "terminal_authenticated": True,
                "terminal_population_count": TERMINAL_POPULATION_COUNT,
                "table_path": relative,
                "table_sha256": _sha256_file(table),
                "source_model_sha256": record["identities"][
                    "evaluation_model_sha256"
                ],
                "identities": copy.deepcopy(record["identities"]),
                "collection_record_payload_sha256": record[
                    "payload_sha256"
                ],
            }
        )
    value = {
        "schema_version": global_pareto.MANIFEST_SCHEMA,
        "snapshot_class": snapshot_class,
        "authenticated_seed_count": len(ordered),
        "expected_final_seed_count": FINAL_SUCCESS_COUNT,
        "seeds": [int(record["seed"]) for record in ordered],
        "identities": copy.deepcopy(first["identities"]),
        "objective_columns": list(OBJECTIVE_COLUMNS),
        "physical_constraint_columns": copy.deepcopy(
            first["physical_constraint_columns"]
        ),
        "normalized_constraint_columns": copy.deepcopy(
            first["normalized_constraint_columns"]
        ),
        "records": population_records,
        "global_sort_scope": (
            "all_authenticated_terminal_population_rows_then_physical_"
            "geometry_deduplication_then_exact_global_non_dominated_sorting"
        ),
        "seed_local_front_union_used": False,
        "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
        "raw_same_metric_C_rx_rx_F_front_classification": (
            "provisional_surrogate_screening_only"
        ),
        **FAIL_CLOSED_FLAGS,
        "symmetric_FEA_validation_still_required": True,
        "integrated_seed_scope_complete": (
            snapshot_class == "final_integrated_exact100"
        ),
    }
    value["payload_sha256"] = canonical_sha256(value)
    return value


def publish_screening_nds(
    *,
    records: Iterable[Mapping[str, Any]],
    output_root: Path,
    snapshot_class: str,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest = terminal_population_manifest(
        records=records,
        output_root=output_root,
        snapshot_class=snapshot_class,
    )
    if snapshot_class == "provisional_first_threshold":
        manifest_path = output_root / "provisional_terminal_population_manifest.json"
        nds_dir = output_root / "nds" / "provisional-first-threshold"
    else:
        manifest_path = output_root / "final_exact100_terminal_population_manifest.json"
        nds_dir = output_root / "nds" / "final-integrated-exact100"
    _immutable_json(manifest_path, manifest)
    if nds_dir.exists():
        summary = _read_json(nds_dir / "summary.json")
    else:
        summary = global_pareto.build_global_pareto(
            manifest_path=manifest_path,
            output_dir=nds_dir,
        )
    if int(summary.get("authenticated_seed_count", -1)) != int(
        manifest["authenticated_seed_count"]
    ):
        raise RuntimeError("global NDS summary seed coverage mismatch")
    authority = _sealed(
        {
            "schema_version": SCREENING_AUTHORITY_SCHEMA,
            "snapshot_class": snapshot_class,
            "terminal_population_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "payload_sha256": manifest["payload_sha256"],
            },
            "global_nds_summary": {
                "path": str(nds_dir / "summary.json"),
                "sha256": _sha256_file(nds_dir / "summary.json"),
                "content_sha256": summary["content_sha256"],
            },
            "authenticated_seed_count": manifest[
                "authenticated_seed_count"
            ],
            "terminal_row_count": summary["terminal_row_count"],
            "unique_physical_candidate_count": summary[
                "unique_physical_candidate_count"
            ],
            "hard_feasible_count": summary["hard_feasible_count"],
            "screening_feasible_front0_count": summary[
                "feasible_front0_count"
            ],
            "screening_objective_front0_count": summary[
                "objective_front0_count"
            ],
            "integrated_seed_scope_complete": manifest[
                "integrated_seed_scope_complete"
            ],
            "front_files_are_provisional_screening_only": True,
            "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
            "raw_same_metric_C_rx_rx_F_front_classification": (
                "provisional_surrogate_screening_only"
            ),
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "published_at": _now(),
        }
    )
    authority_path = nds_dir / "screening_authority.json"
    if authority_path.exists():
        prior = _validate_sealed(
            _read_json(authority_path),
            schema=SCREENING_AUTHORITY_SCHEMA,
        )
        stable_keys = (
            "snapshot_class",
            "terminal_population_manifest",
            "global_nds_summary",
            "authenticated_seed_count",
            "terminal_row_count",
            "unique_physical_candidate_count",
            "hard_feasible_count",
            "screening_feasible_front0_count",
            "screening_objective_front0_count",
            "integrated_seed_scope_complete",
        )
        if any(prior[key] != authority[key] for key in stable_keys):
            raise RuntimeError("existing diagnostic NDS authority changed")
        authority = prior
    else:
        _atomic_json(authority_path, authority)
    return {
        "manifest": str(manifest_path),
        "nds_output": str(nds_dir),
        "screening_authority": str(authority_path),
        "authenticated_seed_count": manifest["authenticated_seed_count"],
        "integrated_seed_scope_complete": manifest[
            "integrated_seed_scope_complete"
        ],
        "front_classification": "provisional_surrogate_screening_only",
    }


def _existing_collection_records(
    output_root: Path,
    entries_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not output_root.exists():
        return records
    for path in output_root.glob("task-*/collection_record.json"):
        try:
            task_id = int(path.parent.name.removeprefix("task-"))
        except ValueError:
            continue
        entry = entries_by_id.get(task_id)
        if entry is None:
            raise RuntimeError("foreign task collection exists in output root")
        records[task_id] = _validate_existing_collection(path, entry=entry)
    return records


def collect_once(
    *,
    context: Mapping[str, Any],
    output_root: Path,
    scheduler: SchedulerReader,
    accounts_path: Path,
    scheduler_source: Path,
    retries: int = 3,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    observed = observe_tasks(context, scheduler)
    entries_by_id = {
        int(entry["task_id"]): entry for entry in context["entries"]
    }
    collections = _existing_collection_records(output_root, entries_by_id)
    connections: dict[str, transport._PersistentAccountConnection] = {}
    collection_errors: dict[int, str] = {}
    try:
        for row in observed:
            task_id = int(row["task_id"])
            if row.get("terminal_success") is not True or task_id in collections:
                continue
            account_name = str(row["account_name"])
            connection = connections.get(account_name)
            if connection is None:
                account, ssh_session = transport._account(
                    accounts_path.resolve(strict=True),
                    scheduler_source.resolve(strict=True),
                    account_name,
                )
                connection = transport._PersistentAccountConnection(
                    account, ssh_session
                )
                connections[account_name] = connection
            try:
                collections[task_id] = collect_terminal_success(
                    context=context,
                    observed=row,
                    connection=connection,
                    output_root=output_root,
                    retries=retries,
                )
            except Exception as exc:
                collection_errors[task_id] = f"{type(exc).__name__}:{exc}"
    finally:
        for connection in connections.values():
            connection.close()

    ordered_records = sorted(
        collections.values(), key=lambda record: int(record["seed"])
    )
    provisional = None
    provisional_authority = (
        output_root
        / "nds"
        / "provisional-first-threshold"
        / "screening_authority.json"
    )
    if provisional_authority.is_file():
        authority = _validate_sealed(
            _read_json(provisional_authority),
            schema=SCREENING_AUTHORITY_SCHEMA,
        )
        provisional = {
            "manifest": str(
                output_root / "provisional_terminal_population_manifest.json"
            ),
            "nds_output": str(provisional_authority.parent),
            "screening_authority": str(provisional_authority),
            "authenticated_seed_count": authority[
                "authenticated_seed_count"
            ],
            "integrated_seed_scope_complete": False,
            "front_classification": "provisional_surrogate_screening_only",
        }
    elif len(ordered_records) >= EARLY_SUCCESS_MINIMUM:
        provisional = publish_screening_nds(
            records=ordered_records,
            output_root=output_root,
            snapshot_class="provisional_first_threshold",
        )

    final = None
    if len(ordered_records) == FINAL_SUCCESS_COUNT:
        final = publish_screening_nds(
            records=ordered_records,
            output_root=output_root,
            snapshot_class="final_integrated_exact100",
        )
    statuses = Counter(str(row["scheduler_status"]) for row in observed)
    terminal_count = sum(bool(row["terminal"]) for row in observed)
    terminal_success_count = sum(
        bool(row["terminal_success"]) for row in observed
    )
    status = _sealed(
        {
            "schema_version": COLLECTOR_STATUS_SCHEMA,
            "campaign_id": scout.CAMPAIGN_ID,
            "bundle_id": context["plan"]["bundle_id"],
            "plan_path": context["plan_path"],
            "plan_file_sha256": context["plan_file_sha256"],
            "receipt_path": context["receipt_path"],
            "receipt_file_sha256": context["receipt_file_sha256"],
            "expected_task_count": FINAL_SUCCESS_COUNT,
            "scheduler_status_counts": dict(sorted(statuses.items())),
            "scheduler_terminal_count": terminal_count,
            "scheduler_terminal_success_count": terminal_success_count,
            "authenticated_collection_count": len(ordered_records),
            "authenticated_seeds": [
                int(record["seed"]) for record in ordered_records
            ],
            "collection_errors": {
                str(task_id): message
                for task_id, message in sorted(collection_errors.items())
            },
            "provisional_global_nds": provisional,
            "final_integrated_global_nds": final,
            "early_threshold_reached": (
                len(ordered_records) >= EARLY_SUCCESS_MINIMUM
            ),
            "final_exact100_reached": (
                len(ordered_records) == FINAL_SUCCESS_COUNT
            ),
            "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
            "raw_same_metric_C_rx_rx_F_front_classification": (
                "provisional_surrogate_screening_only"
            ),
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_access_mode": "GET_status_plus_read_only_SFTP",
            "scheduler_get_count": int(scheduler.get_count),
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "updated_at": _now(),
        }
    )
    _atomic_json(output_root / "collector_status.json", status)
    return status


def run(
    *,
    plan_path: Path,
    receipt_path: Path,
    output_root: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    retries: int = 3,
    max_polls: int = 1,
    poll_seconds: float = 30.0,
    require_early: bool = False,
    require_final: bool = False,
    scheduler: SchedulerReader | None = None,
) -> tuple[dict[str, Any], int]:
    if not 1 <= retries <= 10:
        raise ValueError("retries must be within 1..10")
    if not 1 <= max_polls <= 10_000:
        raise ValueError("max_polls must be within 1..10000")
    if not 0.0 <= poll_seconds <= 300.0:
        raise ValueError("poll_seconds must be within 0..300")
    context = authenticate_context(
        plan_path=plan_path,
        receipt_path=receipt_path,
        scheduler_url=scheduler_url,
    )
    client = scheduler or ReadOnlySchedulerClient(scheduler_url)
    status: dict[str, Any] = {}
    for poll in range(1, max_polls + 1):
        status = collect_once(
            context=context,
            output_root=output_root,
            scheduler=client,
            accounts_path=accounts_path,
            scheduler_source=scheduler_source,
            retries=retries,
        )
        if status["final_exact100_reached"]:
            break
        if (
            require_early
            and not require_final
            and status["early_threshold_reached"]
        ):
            break
        if poll < max_polls:
            time.sleep(poll_seconds)
    exit_code = 0
    if require_final and not status["final_exact100_reached"]:
        exit_code = 2
    elif require_early and not status["early_threshold_reached"]:
        exit_code = 2
    return status, exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--require-early", action="store_true")
    parser.add_argument("--require-final", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status, exit_code = run(
        plan_path=args.plan,
        receipt_path=args.receipt,
        output_root=args.output,
        scheduler_url=args.scheduler_url,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        retries=args.retries,
        max_polls=args.max_polls,
        poll_seconds=args.poll_seconds,
        require_early=args.require_early,
        require_final=args.require_final,
    )
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
