"""Reviewed-only adopted refill controller for the sealed SHA688 replacement fleet.

Default execution is read-only. Scheduler mutation additionally requires
``--execute`` and the exact corrected manifest SHA supplied through
``--reviewed-manifest-sha``. The controller authenticates the initial 250 tasks
on every cycle and delegates any refill exclusively through feeder's formal
adopted authorization and step APIs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import feeder
import pinned_pilot
import rapid_campaign
import scheduler_client
from training.checkpoint_contract import (
    checkpoint_status_revision_identity_matches,
)


SOLVER = "688c6f9ae8b1368d2b4424e42fc8973b3c580d24"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SEED = 260710
PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
INITIAL_COUNT = 250
INITIAL_FIRST_ID = 27471
INITIAL_LAST_ID = 27720
INITIAL_FIRST_SERIAL = 17112
INITIAL_LAST_SERIAL = 17361
INITIAL_CURSOR_START = 939
INITIAL_CURSOR_END = 1843
INITIAL_LAST_RAW_INDEX = 1842
TARGET_ACTIVE = 300
TARGET_STRICT_ROWS = 3_000
MAX_SAMPLES = 12_000
CPUS = 4
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
FLEET_MIN_TERMINAL = 20
FLEET_MIN_VALID_RATE = 0.90
EVIDENCE_MODE = "preloaded250_v1"
MANIFEST_SHA256 = "10b9524fd2b21368fb29b63eac3c9ab2bb5efe5b99dd5e89bbd05cf8eb9c2c57"
MANIFEST_PATH = HERE / "pilot_manifests" / (
    "replacement-s688c6f9-le6b9b9d-seed260710-cursor939.json"
)
SUBMISSION_JOURNAL_PATH = MANIFEST_PATH.with_name(
    "replacement-s688c6f9-le6b9b9d-seed260710-cursor939.journal.json"
)
STATE_PATH = HERE / "adopted_refill_688c6f9_state.json"
FEEDER_STATE_PATH = HERE / "adopted_refill_688c6f9_feeder_state.json"
CYCLE_ROOT = HERE / "pilot_manifests" / "adopted-refill-s688c6f9-le6b9b9d"
STRICT_STATUS_PATH = REGRESSION_ROOT / "training" / "strict_data_status.json"
ATOMIC_ATTEMPTS = 20
ATOMIC_RETRY_SECONDS = 0.25
STRICT_STALL_SECONDS = 90 * 60
STATE_GENERATIONS_TO_KEEP = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_json_bytes(path: Path) -> bytes | None:
    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        return data
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def _write_bytes_verified(
        path: Path, data: bytes, *, attempts: int = ATOMIC_ATTEMPTS) -> None:
    """Bounded same-path write used for a replace staging file.

    The caller owns the controller state lock.  RaiDrive can report WinError 5
    after a write has actually reached the remote filesystem, so every error is
    reconciled by reading the exact bytes back before retrying.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with path.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            last_error = exc
        try:
            if path.read_bytes() == data:
                return
        except OSError as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(ATOMIC_RETRY_SECONDS)
    raise RuntimeError(f"verified JSON write failed for {path}: {last_error}")


def _atomic_json(
        path: Path, payload: dict, *, attempts: int | None = None) -> None:
    if attempts is None:
        attempts = ATOMIC_ATTEMPTS
    if attempts < 1:
        raise ValueError("atomic JSON attempts must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")
    _write_bytes_verified(staged, encoded, attempts=attempts)
    last_error = None
    committed = False
    try:
        for attempt in range(1, attempts + 1):
            try:
                os.replace(staged, path)
            except OSError as exc:
                last_error = exc
            if _valid_json_bytes(path) == encoded:
                committed = True
                return
            if attempt < attempts:
                time.sleep(ATOMIC_RETRY_SECONDS)
        raise RuntimeError(
            f"atomic canonical replace failed for {path}; staged bytes remain "
            f"available for recovery: {last_error}"
        ) from last_error
    finally:
        if committed and staged.exists():
            try:
                staged.unlink()
            except OSError:
                pass


def _state_json_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")


def _state_revision(payload: dict, label: str) -> int:
    revision = payload.get("state_revision")
    if type(revision) is not int or revision < 0:
        raise RuntimeError(f"{label} state revision is invalid")
    return revision


def _generation_path(path: Path, payload: dict) -> Path:
    revision = _state_revision(payload, path.name)
    return path.with_name(
        f"{path.name}.gen-{revision:020d}-{_sha(payload)}.json"
    )


def _read_json_dict(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"unreadable JSON state candidate {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON state candidate is not an object: {path}")
    return payload


def _write_immutable_generation(path: Path, payload: dict) -> Path:
    """Create and verify one checksummed state generation without overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generation = _generation_path(path, payload)
    encoded = _state_json_bytes(payload)
    last_error = None
    for attempt in range(1, ATOMIC_ATTEMPTS + 1):
        try:
            with generation.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc

        # A mounted drive can report an error after committing the create.
        # Reconcile by parsing and hashing the immutable path on every attempt.
        if generation.exists():
            try:
                observed = _read_json_dict(generation)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"immutable state generation is corrupt: {generation}"
                ) from exc
            if observed != payload or _sha(observed) != _sha(payload):
                raise RuntimeError(
                    f"immutable state generation collision: {generation}"
                )
            return generation
        if attempt < ATOMIC_ATTEMPTS:
            time.sleep(ATOMIC_RETRY_SECONDS)
    raise RuntimeError(
        f"immutable state generation create failed for {generation}: {last_error}"
    ) from last_error


def _generation_records(path: Path, validator) -> list[dict]:
    pattern = re.compile(
        rf"^{re.escape(path.name)}\.gen-(\d{{20}})-([0-9a-f]{{64}})\.json$"
    )
    records = []
    for generation in sorted(path.parent.glob(f"{path.name}.gen-*.json")):
        match = pattern.fullmatch(generation.name)
        if match is None:
            raise RuntimeError(f"malformed immutable state generation: {generation}")
        payload = validator(_read_json_dict(generation))
        revision = _state_revision(payload, generation.name)
        digest = _sha(payload)
        if revision != int(match.group(1)) or digest != match.group(2):
            raise RuntimeError(
                f"immutable state generation seal mismatch: {generation}"
            )
        records.append({
            "path": generation,
            "payload": payload,
            "revision": revision,
            "digest": digest,
            "source": "generation",
        })
    return records


def _recovery_artifact_paths(path: Path) -> list[Path]:
    candidates = [
        path.with_name(f"{path.name}.bak"),
        path.with_name(f"{path.name}.tmp"),
        *sorted(path.parent.glob(f"{path.name}.*.tmp")),
    ]
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen and candidate.exists():
            seen.add(key)
            unique.append(candidate)
    return unique


def _assert_no_revision_conflicts(records: list[dict], label: str) -> None:
    by_revision = {}
    for record in records:
        by_revision.setdefault(record["revision"], {}).setdefault(
            record["digest"], []).append(str(record["path"]))
    conflicts = {
        revision: digests for revision, digests in by_revision.items()
        if len(digests) > 1
    }
    if conflicts:
        raise RuntimeError(
            f"conflicting {label} state payloads at the same revision: {conflicts}"
        )


def _canonical_record(path: Path, validator) -> tuple[dict | None, Exception | None]:
    if not path.exists():
        return None, None
    try:
        payload = validator(_read_json_dict(path))
        return ({
            "path": path,
            "payload": payload,
            "revision": _state_revision(payload, path.name),
            "digest": _sha(payload),
            "source": "canonical",
        }, None)
    except Exception as exc:
        return None, exc


def _best_effort_canonical(path: Path, payload: dict) -> bool:
    current = None
    try:
        if path.exists():
            current = _read_json_dict(path)
    except Exception:
        # A corrupt canonical is replaceable because the immutable generation
        # has already committed.  Continue to the bounded repair below.
        current = None
    if current == payload:
        return True
    try:
        # The immutable generation has already committed before this helper is
        # called.  Some mounted filesystems allow create but permanently reject
        # replacing an existing canonical path.  Retrying that convenience
        # view for five seconds on every state transition only stalls the live
        # refill; the next transition naturally gives it another chance.
        _atomic_json(path, payload, attempts=1)
        return _read_json_dict(path) == payload
    except Exception as exc:
        print(
            f"warning: canonical state repair deferred for {path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False


def _prune_generations(path: Path, validator) -> None:
    try:
        records = _generation_records(path, validator)
    except RuntimeError:
        # Never delete anything when history itself does not validate.
        return
    revisions = sorted({record["revision"] for record in records}, reverse=True)
    keep = set(revisions[:max(2, STATE_GENERATIONS_TO_KEEP)])
    for record in records:
        if record["revision"] in keep:
            continue
        try:
            record["path"].unlink()
        except OSError:
            pass


def _authoritative_state(path: Path, validator, *, repair: bool) -> dict | None:
    """Select the newest sealed/canonical state, failing closed on ambiguity."""
    generations = _generation_records(path, validator)
    canonical, canonical_error = _canonical_record(path, validator)
    authoritative = list(generations)
    if canonical is not None:
        authoritative.append(canonical)
    _assert_no_revision_conflicts(authoritative, path.name)

    if authoritative:
        highest_revision = max(record["revision"] for record in authoritative)
        highest = [
            record for record in authoritative
            if record["revision"] == highest_revision
        ]
        selected = copy.deepcopy(highest[0]["payload"])
        sealed = any(
            record["source"] == "generation"
            and record["revision"] == highest_revision
            and record["digest"] == _sha(selected)
            for record in generations
        )
        if repair and not sealed:
            _write_immutable_generation(path, selected)
        if repair and (canonical is None or canonical["payload"] != selected):
            _best_effort_canonical(path, selected)
        return selected

    # Recovery artifacts are considered only when neither a valid canonical
    # file nor an immutable generation exists.  They can rescue legacy staged
    # writes, but can never supersede sealed history.
    artifacts = _recovery_artifact_paths(path)
    history_exists = bool(path.exists() or artifacts or canonical_error)
    recovered = []
    artifact_errors = []
    for artifact in artifacts:
        try:
            payload = validator(_read_json_dict(artifact))
            recovered.append({
                "path": artifact,
                "payload": payload,
                "revision": _state_revision(payload, artifact.name),
                "digest": _sha(payload),
                "source": "recovery",
            })
        except Exception as exc:
            artifact_errors.append(f"{artifact}: {type(exc).__name__}: {exc}")
    if recovered:
        _assert_no_revision_conflicts(recovered, f"{path.name} recovery")
        highest_revision = max(record["revision"] for record in recovered)
        selected_records = [
            record for record in recovered
            if record["revision"] == highest_revision
        ]
        selected = copy.deepcopy(selected_records[0]["payload"])
        if repair:
            _write_immutable_generation(path, selected)
            promoted = copy.deepcopy(selected)
            promoted["state_revision"] = highest_revision + 1
            if "updated_at" in promoted:
                promoted["updated_at"] = _now()
            validator(promoted)
            _write_immutable_generation(path, promoted)
            _best_effort_canonical(path, promoted)
            _prune_generations(path, validator)
            return promoted
        return selected
    if history_exists:
        detail = artifact_errors or [
            f"canonical: {type(canonical_error).__name__}: {canonical_error}"
        ]
        raise RuntimeError(
            f"state history exists but no valid recovery candidate for {path}: {detail}"
        )
    return None


def _initialize_durable_state(path: Path, payload: dict, validator) -> dict:
    initial = copy.deepcopy(validator(payload))
    if _state_revision(initial, path.name) != 0:
        raise RuntimeError(f"fresh state must start at revision zero: {path}")
    _write_immutable_generation(path, initial)
    promoted = copy.deepcopy(initial)
    promoted["state_revision"] = 1
    if "updated_at" in promoted:
        promoted["updated_at"] = _now()
    validator(promoted)
    _write_immutable_generation(path, promoted)
    _best_effort_canonical(path, promoted)
    return promoted


def _load_durable_state(path: Path, validator, factory, *, create: bool) -> dict:
    selected = _authoritative_state(path, validator, repair=create)
    if selected is not None:
        return selected
    fresh = factory()
    if not create:
        return fresh
    return _initialize_durable_state(path, fresh, validator)


def _save_durable_state(
        path: Path, state: dict, validator, *, transition_validator=None) -> None:
    validator(state)
    memory_revision = _state_revision(state, path.name)
    disk = _authoritative_state(path, validator, repair=False)
    if disk is None:
        raise RuntimeError(f"refusing to save over missing durable state history: {path}")
    disk_revision = _state_revision(disk, path.name)
    if disk_revision != memory_revision:
        raise RuntimeError(
            f"stale state save refused for {path}: memory revision "
            f"{memory_revision}, durable revision {disk_revision}"
        )
    if transition_validator is not None:
        transition_validator(disk, state)

    committed = copy.deepcopy(state)
    committed["state_revision"] = memory_revision + 1
    if "updated_at" in committed:
        committed["updated_at"] = _now()
    validator(committed)
    _write_immutable_generation(path, committed)

    # Once the immutable generation verifies, it is the commit point.  Keep
    # the caller's object aligned even if the canonical view cannot be replaced.
    state.clear()
    state.update(copy.deepcopy(committed))
    _best_effort_canonical(path, committed)
    _prune_generations(path, validator)


def _migrate_legacy_state(path: Path, validator) -> dict:
    """One-shot, lock-protected migration from a pre-generation canonical."""
    if _generation_records(path, validator):
        raise RuntimeError(f"immutable state generations already exist: {path}")
    payload = _read_json_dict(path)
    if "state_revision" in payload:
        raise RuntimeError(f"state is already revisioned: {path}")
    payload["state_revision"] = 0
    validator(payload)
    return _initialize_durable_state(path, payload, validator)


def _profile() -> dict:
    payload = json.loads(Path(feeder.PROFILE_PATH).read_text(encoding="utf-8"))
    if payload.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT:
        raise RuntimeError("standard profile contract drifted")
    payload["timeout_seconds"] = TIMEOUT_SECONDS
    return payload


def _validate_candidate_contract(params: dict, label: str) -> None:
    try:
        primary_turns = int(params["N1_main"]) + int(params["N1_side"])
        cold_plates = tuple(float(params[key]) for key in ("wcp_t", "core_plate_t"))
        pads = tuple(float(params[key]) for key in ("wcp_pad_t", "core_plate_pad_t"))
        cw1 = float(params["cw1"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} candidate contract is unreadable") from exc
    if cw1 > 10.0 or primary_turns > 8:
        raise RuntimeError(
            f"{label} candidate primary cap mismatch: cw1={cw1}, turns={primary_turns}")
    if not all(10.0 <= value <= 30.0 for value in cold_plates):
        raise RuntimeError(f"{label} candidate cold-plate range mismatch")
    if pads != (2.0, 2.0):
        raise RuntimeError(f"{label} candidate pad thickness mismatch")


def _load_manifest() -> dict:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    seal = unsigned.pop("manifest_sha256", None)
    if seal != MANIFEST_SHA256 or _sha(unsigned) != MANIFEST_SHA256:
        raise RuntimeError("corrected replacement manifest seal mismatch")
    expected = {
        "schema_version": 2,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "candidate_cursor_start": INITIAL_CURSOR_START,
        "candidate_cursor_end": INITIAL_CURSOR_END,
        "task_count": INITIAL_COUNT,
        "first_serial": INITIAL_FIRST_SERIAL,
        "last_serial": INITIAL_LAST_SERIAL,
        "task_prefix": PREFIX,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"replacement manifest identity mismatch: {mismatches}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != INITIAL_COUNT:
        raise RuntimeError("replacement manifest does not contain exact 250 tasks")

    profile = _profile()
    cursor = INITIAL_CURSOR_START
    names = set()
    dedupes = set()
    for offset, record in enumerate(tasks):
        cursor, raw_index, params = pinned_pilot.next_valid_candidate(cursor, seed=SEED)
        params = json.loads(_canonical(params))
        serial = INITIAL_FIRST_SERIAL + offset
        name = f"{PREFIX}{serial:05d}"
        identity = scheduler_client.verification_submission_identity(
            name, params, profile, SOLVER, LIBRARY
        )
        checks = {
            "index": record.get("index") == offset,
            "serial": record.get("serial") == serial,
            "name": record.get("name") == name,
            "raw": record.get("candidate_raw_index") == raw_index,
            "cursor": record.get("candidate_cursor_after") == cursor,
            "params_sha": record.get("params_sha256") == pinned_pilot.candidate_digest(params),
            "params": record.get("params") == params,
            "effective": record.get("effective_params") == identity["merged"],
            "digest": record.get("parameter_digest") == identity["parameter_digest"],
            "dedupe": record.get("dedupe_key") == identity["dedupe_key"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"replacement manifest task {offset} mismatch: {checks}")
        _validate_candidate_contract(params, f"replacement[{offset}]")
        names.add(name)
        dedupes.add(identity["dedupe_key"])
    if cursor != INITIAL_CURSOR_END or len(names) != INITIAL_COUNT or len(dedupes) != INITIAL_COUNT:
        raise RuntimeError("replacement cursor or uniqueness seal mismatch")
    if tasks[-1]["candidate_raw_index"] != INITIAL_LAST_RAW_INDEX:
        raise RuntimeError("replacement final raw index mismatch")
    return payload


def _load_submission_journal(manifest: dict) -> dict:
    journal = json.loads(SUBMISSION_JOURNAL_PATH.read_text(encoding="utf-8"))
    if (journal.get("completed") is not True
            or journal.get("manifest_sha256") != MANIFEST_SHA256
            or (journal.get("audit") or {}).get("task_count") != INITIAL_COUNT):
        raise RuntimeError("replacement submission journal is incomplete or mismatched")
    submissions = journal.get("submissions")
    if not isinstance(submissions, dict) or len(submissions) != INITIAL_COUNT:
        raise RuntimeError("replacement journal lacks exact 250 submission IDs")
    for record in manifest["tasks"]:
        item = submissions.get(str(record["index"]))
        expected_id = INITIAL_FIRST_ID + int(record["index"])
        if (not isinstance(item, dict) or item.get("task_id") != expected_id
                or item.get("name") != record["name"]
                or item.get("dedupe_key") != record["dedupe_key"]):
            raise RuntimeError(f"replacement submission ledger mismatch: {record['index']}")
    return journal


def _authenticate_scheduler_cohort(manifest: dict, journal: dict) -> dict:
    inventory = feeder.campaign_inventory()
    selected = [task for task in inventory if str(task.get("name") or "").startswith(PREFIX)]
    by_name = {str(task.get("name")): task for task in selected}
    if len(by_name) != len(selected):
        raise RuntimeError("scheduler SHA688 generation contains duplicate task names")
    initial_names = {record["name"] for record in manifest["tasks"]}
    initial = [task for task in selected if str(task.get("name")) in initial_names]
    if len(initial) != INITIAL_COUNT:
        raise RuntimeError(f"scheduler initial replacement cohort is not exact 250: {len(initial)}")
    statuses = {}
    for record in manifest["tasks"]:
        task = by_name.get(record["name"])
        expected_id = INITIAL_FIRST_ID + int(record["index"])
        checks = {
            "id": int(task.get("id") or 0) == expected_id if task else False,
            "journal": (
                journal["submissions"][str(record["index"])]["task_id"] == expected_id
            ),
            "project": task.get("project") == scheduler_client.MFT_PROJECT if task else False,
            "dedupe": task.get("dedupe_key") == record["dedupe_key"] if task else False,
            "cpus": int(task.get("cpus") or 0) == CPUS if task else False,
            "memory": int(task.get("memory_mb") or 0) == MEMORY_MB if task else False,
            "gpus": int(task.get("gpus") or 0) == 0 if task else False,
            "timeout": int(task.get("timeout_seconds") or 0) == TIMEOUT_SECONDS if task else False,
            "capability": task.get("required_capability") == "conda:pyaedt2026v1" if task else False,
            "env": task.get("env_profile") == "pyaedt2026v1" if task else False,
            "scheduling": task.get("scheduling_profile") == "fea_bursty" if task else False,
            "remote_cwd": task.get("remote_cwd") == scheduler_client.GPFS_RUNS_REMOTE_CWD if task else False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"scheduler replacement task mismatch {expected_id}: {checks}")
        status = str(task.get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "inventory": inventory,
        "initial_tasks": initial,
        "generation_task_count": len(selected),
        "statuses": statuses,
    }


def authenticate_adoption() -> dict:
    manifest = _load_manifest()
    journal = _load_submission_journal(manifest)
    scheduler = _authenticate_scheduler_cohort(manifest, journal)
    return {
        "manifest": manifest,
        "submission_journal": journal,
        **scheduler,
        "adoption_sha256": MANIFEST_SHA256,
    }


def _local3_passed() -> bool:
    root = pinned_pilot.campaign_manifest_dir()
    path = root / f"{pinned_pilot.local_gate_tag(SOLVER, LIBRARY)}.json"
    if not path.is_file():
        return False
    pinned_pilot.validate_local_gate(SOLVER, LIBRARY, manifest_dir=root)
    return True


def _strict_rows() -> int:
    payload = json.loads(STRICT_STATUS_PATH.read_text(encoding="utf-8"))
    if not checkpoint_status_revision_identity_matches(
        payload, SOLVER, LIBRARY
    ):
        raise RuntimeError("strict status is not pinned to SHA688/library e6")
    stamp = datetime.fromisoformat(str(payload["time"]).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    if age > 20 * 60:
        raise RuntimeError(f"strict status is stale by {age:.0f}s")
    rows = int(payload.get("strict_full_rows") or 0)
    if rows < 0:
        raise RuntimeError("strict row count is negative")
    return rows


def _new_controller_state() -> dict:
    return {
        "schema_version": 1,
        "state_revision": 0,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "adoption_sha256": MANIFEST_SHA256,
        "initial_count": INITIAL_COUNT,
        "initial_first_id": INITIAL_FIRST_ID,
        "initial_last_id": INITIAL_LAST_ID,
        "initial_cursor_end": INITIAL_CURSOR_END,
        "initial_serial_end": INITIAL_LAST_SERIAL,
        "target_active": TARGET_ACTIVE,
        "target_strict_rows": TARGET_STRICT_ROWS,
        "task_outcomes": {},
        "cycle_serial": 0,
        "paused": False,
        "pause_reasons": [],
        "promoted_at": None,
        "last_strict_rows": 0,
        "last_strict_growth_at": None,
        "last_action": None,
        "last_evidence": None,
        "updated_at": None,
    }


def _validate_controller_state(state: dict) -> dict:
    if not isinstance(state, dict):
        raise RuntimeError("adopted controller state is not an object")
    expected = _new_controller_state()
    immutable = (
        "schema_version", "solver_revision", "library_revision", "seed",
        "adoption_sha256", "initial_count", "initial_first_id", "initial_last_id",
        "initial_cursor_end", "initial_serial_end", "target_active", "target_strict_rows",
    )
    mismatches = {key: (state.get(key), expected[key]) for key in immutable if state.get(key) != expected[key]}
    if mismatches:
        raise RuntimeError(f"adopted controller state identity mismatch: {mismatches}")
    _state_revision(state, "adopted controller")
    if not isinstance(state.get("task_outcomes"), dict):
        raise RuntimeError("adopted controller outcome cache is invalid")
    if type(state.get("cycle_serial")) is not int or state["cycle_serial"] < 0:
        raise RuntimeError("adopted controller cycle serial is invalid")
    if type(state.get("paused")) is not bool or not isinstance(
            state.get("pause_reasons"), list):
        raise RuntimeError("adopted controller pause state is invalid")
    if (type(state.get("last_strict_rows")) is not int
            or state["last_strict_rows"] < 0):
        raise RuntimeError("adopted controller strict progress is invalid")
    for key in ("promoted_at", "last_strict_growth_at"):
        value = state.get(key)
        if value is not None and rapid_campaign._parse_time(value) is None:
            raise RuntimeError(f"adopted controller {key} is invalid")
    return state


def _load_controller_state(create: bool) -> dict:
    return _load_durable_state(
        STATE_PATH,
        _validate_controller_state,
        _new_controller_state,
        create=create,
    )


def _save_controller_state(state: dict) -> None:
    _save_durable_state(STATE_PATH, state, _validate_controller_state)


def _initial_feeder_state(adoption: dict) -> dict:
    generation = f"{SOLVER}:{LIBRARY}:seed{SEED}"
    ids = list(range(INITIAL_FIRST_ID, INITIAL_LAST_ID + 1))
    return {
        "state_revision": 0,
        "serial": INITIAL_LAST_SERIAL,
        "submitted_samples": INITIAL_COUNT,
        "outstanding": ids,
        "candidate_generation": generation,
        "candidate_cursor": INITIAL_CURSOR_END,
        "candidate_cursors": {generation: INITIAL_CURSOR_END},
        "candidate_raw_index": INITIAL_LAST_RAW_INDEX,
        "task_ids_by_generation": {generation: ids},
        "task_expected_rows": {str(task_id): 1 for task_id in ids},
        "adoption_sha256": adoption["adoption_sha256"],
        "adoption_manifest": str(MANIFEST_PATH.resolve()),
    }


def _validate_feeder_state(state: dict) -> dict:
    if not isinstance(state, dict):
        raise RuntimeError("dedicated feeder state is not an object")
    _state_revision(state, "dedicated feeder")
    generation = f"{SOLVER}:{LIBRARY}:seed{SEED}"
    if (int(state.get("serial") or -1) < INITIAL_LAST_SERIAL
            or int(state.get("candidate_cursor") or -1) < INITIAL_CURSOR_END
            or int((state.get("candidate_cursors") or {}).get(generation, -1)) < INITIAL_CURSOR_END
            or state.get("candidate_generation") != generation
            or state.get("adoption_sha256") != MANIFEST_SHA256):
        raise RuntimeError("dedicated feeder state would replay the old cursor/serial")
    return state


def _validate_feeder_transition(previous: dict, proposed: dict) -> None:
    monotonic_fields = (
        "serial", "submitted_samples", "candidate_cursor", "candidate_raw_index",
    )
    regressed = {}
    for key in monotonic_fields:
        try:
            before = int(previous[key])
            after = int(proposed[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"dedicated feeder monotonic field is invalid: {key}"
            ) from exc
        if after < before:
            regressed[key] = (before, after)
    previous_cursors = previous.get("candidate_cursors") or {}
    proposed_cursors = proposed.get("candidate_cursors") or {}
    for generation, before_value in previous_cursors.items():
        try:
            before = int(before_value)
            after = int(proposed_cursors[generation])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"dedicated feeder generation cursor is invalid: {generation}"
            ) from exc
        if after < before:
            regressed[f"candidate_cursors[{generation}]"] = (before, after)
    if regressed:
        raise RuntimeError(
            f"dedicated feeder state transition would replay progress: {regressed}"
        )


def _save_feeder_state(state: dict) -> None:
    _save_durable_state(
        FEEDER_STATE_PATH,
        state,
        _validate_feeder_state,
        transition_validator=_validate_feeder_transition,
    )


def _load_feeder_state(adoption: dict, create: bool) -> dict:
    return _load_durable_state(
        FEEDER_STATE_PATH,
        _validate_feeder_state,
        lambda: _initial_feeder_state(adoption),
        create=create,
    )


@contextmanager
def _dedicated_feeder_io(
        adoption: dict, cycle_path: Path, cycle: dict, formal_journal: dict):
    original_state = feeder.STATE
    original_load = feeder.load_state
    original_save = feeder.save_state
    original_next_candidate = feeder.next_valid_candidate

    def load_state():
        return copy.deepcopy(_load_feeder_state(adoption, create=True))

    def save_state(state):
        _validate_feeder_state(state)
        _save_feeder_state(state)
        # feeder commits the durable state immediately before it flips the
        # in-memory journal event to ``ledger_committed``.  Reflect the durable
        # task IDs in the cycle snapshot so an abrupt process exit cannot leave
        # a misleading uncommitted journal beside an already advanced cursor.
        snapshot = copy.deepcopy(formal_journal)
        committed_ids = {
            int(task_id) for task_id in state.get("outstanding", [])
            if isinstance(task_id, int) and not isinstance(task_id, bool)
        }
        for event in snapshot.get("events", []):
            task_id = event.get("task_id") if isinstance(event, dict) else None
            if task_id in committed_ids:
                event["accepted_or_reconciled"] = True
                event["ledger_committed"] = True
        snapshot["submitted_count"] = sum(
            bool(event.get("ledger_committed"))
            for event in snapshot.get("events", []) if isinstance(event, dict)
        )
        cycle["formal_journal"] = snapshot
        cycle["feeder_state_serial"] = int(state["serial"])
        cycle["feeder_state_cursor"] = int(state["candidate_cursor"])
        cycle["updated_at"] = _now()
        _atomic_json(cycle_path, cycle)

    def next_valid_candidate(cursor=0, seed=SEED, max_attempts=1000):
        next_cursor, raw_index, params = original_next_candidate(
            cursor, seed=seed, max_attempts=max_attempts)
        _validate_candidate_contract(params, f"refill_raw[{raw_index}]")
        return next_cursor, raw_index, params

    feeder.STATE = str(FEEDER_STATE_PATH)
    feeder.load_state = load_state
    feeder.save_state = save_state
    feeder.next_valid_candidate = next_valid_candidate
    try:
        yield
    finally:
        feeder.STATE = original_state
        feeder.load_state = original_load
        feeder.save_state = original_save
        feeder.next_valid_candidate = original_next_candidate


def _strict_progress_preview(state: dict, strict_rows: int, now: datetime) -> dict:
    previous = int(state.get("last_strict_rows") or 0)
    reasons = []
    if strict_rows < previous:
        reasons.append(f"strict_row_count_regressed:{previous}->{strict_rows}")
    growth_at = rapid_campaign._parse_time(state.get("last_strict_growth_at"))
    if strict_rows > previous:
        growth_at = now
    promoted_at = rapid_campaign._parse_time(state.get("promoted_at"))
    if promoted_at is not None:
        anchor = max(
            item for item in (promoted_at, growth_at) if item is not None)
        if (now - anchor).total_seconds() >= STRICT_STALL_SECONDS:
            reasons.append("strict_dataset_growth_stalled_90m")
    return {
        "previous_rows": previous,
        "observed_rows": strict_rows,
        "growth_at": growth_at.isoformat() if growth_at is not None else None,
        "reasons": reasons,
    }


def _evidence(adoption: dict, state: dict) -> dict:
    production = rapid_campaign.inspect_production_tasks(
        adoption["inventory"], SOLVER, LIBRARY,
        cached_outcomes=state.get("task_outcomes"),
    )
    outcomes = production["outcomes"]
    terminal = len(outcomes)
    valid = sum(item["state"] == "valid" for item in outcomes)
    valid_rate = valid / terminal if terminal else None
    strict_rows = _strict_rows()
    local3 = _local3_passed()
    now = datetime.now(timezone.utc)
    strict_progress = _strict_progress_preview(state, strict_rows, now)
    health_reasons = rapid_campaign._production_gate_reasons(production)
    pause_reasons = sorted(set([
        *state.get("pause_reasons", []),
        *health_reasons,
        *strict_progress["reasons"],
    ]))
    paused = bool(state.get("paused") or pause_reasons)
    if strict_rows >= TARGET_STRICT_ROWS:
        action = "target_reached_drain"
    elif paused:
        action = "manual_intervention"
    elif not local3:
        action = "wait_local3"
    elif terminal < FLEET_MIN_TERMINAL:
        action = "wait_fleet20"
    elif valid_rate is None or valid_rate < FLEET_MIN_VALID_RATE:
        action = "wait_fleet90"
    else:
        action = "refill_300"
    return {
        "time": now.isoformat(timespec="seconds"),
        "action": action,
        "paused": paused,
        "pause_reasons": pause_reasons,
        "local3_passed": local3,
        "production_active": int(production["active"]),
        "production_terminal": terminal,
        "production_valid": valid,
        "production_invalid": terminal - valid,
        "production_valid_rate": valid_rate,
        "strict_full_rows": strict_rows,
        "strict_progress": strict_progress,
        "target_strict_rows": TARGET_STRICT_ROWS,
        "task_outcomes": production["cache"],
        "initial_statuses": adoption["statuses"],
    }


def _decision(evidence: dict) -> dict:
    return {
        "paused": bool(evidence["paused"]),
        "target_active": TARGET_ACTIVE,
        "action": evidence["action"],
        "production": {
            "terminal": evidence["production_terminal"],
            "valid": evidence["production_valid"],
            "valid_rate": evidence["production_valid_rate"],
        },
    }


def _cycle_path(serial: int) -> Path:
    return CYCLE_ROOT / f"cycle-{serial:06d}.json"


def _result_from_evidence(evidence: dict, execute: bool) -> dict:
    return {
        **{key: value for key, value in evidence.items() if key != "task_outcomes"},
        "mode": "execute" if execute else "read_only",
        "adoption_sha256": MANIFEST_SHA256,
        "manifest": str(MANIFEST_PATH.resolve()),
        "submission_journal": str(SUBMISSION_JOURNAL_PATH.resolve()),
        "dedicated_feeder_state": str(FEEDER_STATE_PATH.resolve()),
        "mutation": None,
    }


def _apply_evidence(state: dict, evidence: dict) -> None:
    state["task_outcomes"] = evidence["task_outcomes"]
    observed = int(evidence["strict_full_rows"])
    if observed >= int(state.get("last_strict_rows") or 0):
        state["last_strict_rows"] = observed
        state["last_strict_growth_at"] = evidence["strict_progress"]["growth_at"]
    state["paused"] = bool(evidence["paused"])
    state["pause_reasons"] = list(evidence["pause_reasons"])
    state["last_action"] = evidence["action"]
    state["last_evidence"] = {
        key: value for key, value in evidence.items() if key != "task_outcomes"
    }


def _recover_incomplete_cycles(state: dict) -> list[str]:
    """Mark prior interrupted journals; feeder state remains the replay seal."""
    recovered = []
    if not CYCLE_ROOT.is_dir():
        return recovered
    for path in sorted(CYCLE_ROOT.glob("cycle-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"unreadable adopted refill cycle: {path}: {exc}") from exc
        serial = payload.get("cycle_serial")
        if (type(serial) is not int or serial < 1
                or serial > int(state["cycle_serial"])
                or payload.get("adoption_sha256") != MANIFEST_SHA256):
            raise RuntimeError(f"adopted refill cycle identity mismatch: {path}")
        if payload.get("status") not in ("authorized_pending", "authorized_running"):
            continue
        payload["status"] = "interrupted_recoverable"
        payload["recovered_at"] = _now()
        payload["updated_at"] = _now()
        payload["recovery_feeder_state"] = str(FEEDER_STATE_PATH.resolve())
        _atomic_json(path, payload)
        recovered.append(str(path.resolve()))
    return recovered


def run_once(execute: bool = False, reviewed_manifest_sha: str | None = None) -> dict:
    if execute and reviewed_manifest_sha != MANIFEST_SHA256:
        raise RuntimeError("execute requires root-reviewed corrected manifest SHA")
    feeder._require_deployed_revisions(SOLVER, LIBRARY)
    adoption = authenticate_adoption()

    # Read-only inspection intentionally does not acquire the persistent state
    # lock or create controller/feeder state files.
    if not execute:
        state = _load_controller_state(create=False)
        _load_feeder_state(adoption, create=False)
        return _result_from_evidence(_evidence(adoption, state), execute=False)

    with FileLock(str(STATE_PATH) + ".lock", timeout=30):
        state = _load_controller_state(create=True)
        _load_feeder_state(adoption, create=True)
        recovered_cycles = _recover_incomplete_cycles(state)
        evidence = _evidence(adoption, state)
        result = _result_from_evidence(evidence, execute=True)
        result["recovered_cycles"] = recovered_cycles
        _apply_evidence(state, evidence)
        if evidence["action"] != "refill_300":
            _save_controller_state(state)
            return result

        cycle = None
        cycle_path = None
        formal_journal = None
        try:
            with scheduler_client.campaign_mutation_lock():
                # Re-authenticate and recompute promotion evidence in the same
                # mutation epoch that issues the one-cycle authorization.  A
                # newly terminal invalid result can therefore revoke the gate
                # before any task is submitted.
                adoption = authenticate_adoption()
                locked_evidence = _evidence(adoption, state)
                _apply_evidence(state, locked_evidence)
                result = _result_from_evidence(locked_evidence, execute=True)
                result["recovered_cycles"] = recovered_cycles
                if locked_evidence["action"] != "refill_300":
                    _save_controller_state(state)
                    return result

                state["cycle_serial"] += 1
                # Persist the serial before creating/using the journal.  A hard
                # exit may leave a harmless gap, but can never reuse a cycle or
                # an older feeder cursor on restart.
                _save_controller_state(state)
                cycle_path = _cycle_path(state["cycle_serial"])
                if cycle_path.exists():
                    raise RuntimeError(
                        f"adopted refill cycle already exists: {cycle_path}")
                cycle = {
                    "schema_version": 1,
                    "cycle_serial": state["cycle_serial"],
                    "created_at": _now(),
                    "updated_at": _now(),
                    "status": "authorized_pending",
                    "adoption_sha256": MANIFEST_SHA256,
                    "evidence": state["last_evidence"],
                    "formal_journal": {"events": []},
                    "result": None,
                    "error": None,
                }
                _atomic_json(cycle_path, cycle)
                authorization = feeder._authorize_adopted_refill(
                    _decision(locked_evidence),
                    max_samples=MAX_SAMPLES,
                    solver_revision=SOLVER,
                    library_revision=LIBRARY,
                    candidate_seed=SEED,
                    local_passed=locked_evidence["local3_passed"],
                    adoption_sha256=MANIFEST_SHA256,
                    initial_count=INITIAL_COUNT,
                    cpus=CPUS,
                    memory_mb=MEMORY_MB,
                    timeout_seconds=TIMEOUT_SECONDS,
                    evidence_mode=EVIDENCE_MODE,
                    strict_rows=locked_evidence["strict_full_rows"],
                    target_strict_rows=TARGET_STRICT_ROWS,
                )
                if state.get("promoted_at") is None:
                    promoted_at = _now()
                    state["promoted_at"] = promoted_at
                    if state.get("last_strict_growth_at") is None:
                        state["last_strict_growth_at"] = promoted_at
                    _save_controller_state(state)
                cycle["status"] = "authorized_running"
                _atomic_json(cycle_path, cycle)
                formal_journal = cycle["formal_journal"]
                with _dedicated_feeder_io(
                        adoption, cycle_path, cycle, formal_journal):
                    step_result = feeder._step_from_adopted_controller(
                        MAX_SAMPLES,
                        authorization=authorization,
                        target=TARGET_ACTIVE,
                        buffer=0,
                        solver_revision=SOLVER,
                        library_revision=LIBRARY,
                        candidate_seed=SEED,
                        adoption_sha256=MANIFEST_SHA256,
                        initial_count=INITIAL_COUNT,
                        cpus=CPUS,
                        memory_mb=MEMORY_MB,
                        timeout_seconds=TIMEOUT_SECONDS,
                        evidence_mode=EVIDENCE_MODE,
                        strict_rows=locked_evidence["strict_full_rows"],
                        target_strict_rows=TARGET_STRICT_ROWS,
                        journal=formal_journal,
                    )
            cycle["status"] = "completed"
            cycle["result"] = bool(step_result)
            cycle["formal_journal"] = copy.deepcopy(formal_journal)
            _atomic_json(cycle_path, cycle)
            result["mutation"] = {
                "cycle": state["cycle_serial"],
                "cycle_journal": str(cycle_path.resolve()),
                "formal_result": bool(step_result),
                "submitted_count": int(cycle["formal_journal"].get("submitted_count") or 0),
            }
        except BaseException as exc:
            if cycle is not None and cycle_path is not None:
                cycle["status"] = "failed_closed"
                cycle["error"] = f"{type(exc).__name__}: {exc}"
                if formal_journal is not None:
                    cycle["formal_journal"] = copy.deepcopy(formal_journal)
                _atomic_json(cycle_path, cycle)
            _save_controller_state(state)
            raise
        _save_controller_state(state)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewed-manifest-sha")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args(argv)
    if args.loop is not None and args.loop < 60:
        parser.error("--loop must be at least 60 seconds")
    if args.loop is not None and not args.execute:
        parser.error("--loop requires --execute")
    if args.execute and args.reviewed_manifest_sha != MANIFEST_SHA256:
        parser.error("--execute requires the exact --reviewed-manifest-sha")
    while True:
        try:
            print(json.dumps(
                run_once(args.execute, args.reviewed_manifest_sha),
                ensure_ascii=False, sort_keys=True,
            ), flush=True)
        except Exception as exc:
            print(json.dumps({
                "time": _now(),
                "mode": "execute" if args.execute else "read_only",
                "action": "error_no_mutation_or_failed_closed",
                "error": f"{type(exc).__name__}: {exc}",
            }, ensure_ascii=False, sort_keys=True), flush=True)
            if args.loop is None:
                return 2
        if args.loop is None:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
