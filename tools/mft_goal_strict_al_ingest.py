"""Build an isolated strict-training dataset from authenticated Standard FEA.

This is a deliberately narrow active-learning boundary for the 2026-07-26
MFT goal.  It accepts only immutable collection artifacts that are
reauthenticated by their owning contract module.  Generic JSON results, CSV
files, scheduler task IDs, and unsealed rows are not accepted.

The canonical source parquet is never changed.  A successful build creates a
new immutable directory containing a derived parquet and a sealed provenance
manifest.  This tool never submits, cancels, or otherwise mutates Scheduler
state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ROOT = REPOSITORY_ROOT / "regression_260707"
TRAINING_ROOT = REGRESSION_ROOT / "training"
for import_root in (REPOSITORY_ROOT, REGRESSION_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    FIXED_GEOMETRY_INPUT_KEYS,
    TURN_GRADED_ELECTROSTATIC_INPUT_KEYS,
    get_drawing_default_params,
)
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_G0_MODEL_TARGETS,
    GOAL_N1_MAX_TURNS,
    GOAL_N1_MIN_TURNS,
    GOAL_TEMPERATURE_TARGETS,
    attest_fixed_identity,
    canonical_sha256,
)
from module import thermal_truth_contract as thermal_truth  # noqa: E402
from regression_260707.quality_contract import (  # noqa: E402
    annotate_validity,
    validate_record,
)
from regression_260707.training.checkpoint_train import (  # noqa: E402
    filter_valid_training_rows,
    to_physical,
)
from tools import mft_goal_fea_handoff as production  # noqa: E402


MANIFEST_SCHEMA = "mft-goal-strict-al-dataset-v1"
PRODUCTION_COLLECTION_SCHEMA = production.COLLECTION_SCHEMA
DIAGNOSTIC_COLLECTION_SCHEMA = (
    "mft-goal-diagnostic-standard-collection-v1"
)
DIAGNOSTIC_AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-diagnostic-standard-authenticated-collection-v1"
)
POSTDEADLINE_COLLECTION_SCHEMA = (
    "mft-goal-postdeadline-standard-collection-v1"
)
POSTDEADLINE_AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-postdeadline-standard-authenticated-collection-v1"
)
CORRECTED_REPLACEMENT_COLLECTION_SCHEMA = (
    "mft-goal-corrected-replacement-strict-collection-v1"
)
CORRECTED_REPLACEMENT_AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-corrected-replacement-strict-authenticated-collection-v1"
)
CORRECTED_REPLACEMENT_MANIFEST_SCHEMA = (
    "mft-goal-corrected-replacement-strict-manifest-v1"
)
CORRECTED_REPLACEMENT_AUTHENTICATED_MANIFEST_SCHEMA = (
    "mft-goal-corrected-replacement-strict-authenticated-manifest-v1"
)
GOAL_PROFILE_PATH = (
    REGRESSION_ROOT / "verify" / "profiles" / "goal_standard.json"
)
DEFAULT_MINIMUM_USEFUL_ROWS = 8
DEFAULT_MINIMUM_UNIQUE_GEOMETRIES = 8
DEFAULT_MINIMUM_SOURCE_TASKS = 4
RECOMMENDED_NEW_ROWS = 12
MAX_DIAGNOSTIC_SELECTION_ROWS = 12
TARGETED_PRIMARY_TURNS = 6
BASE_INPUT_NORMALIZATION_SCHEMA = (
    "mft-goal-strict-al-base-input-normalization-v1"
)
APPEND_ONLY_BASE_INPUT_KEYS = (
    *FIXED_GEOMETRY_INPUT_KEYS,
    *TURN_GRADED_ELECTROSTATIC_INPUT_KEYS,
)
NEXT_CAMPAIGN_SEED_START = 2_607_263_000
NEXT_CAMPAIGN_SEED_COUNT = 512
NEXT_CAMPAIGN_SEED_END = (
    NEXT_CAMPAIGN_SEED_START + NEXT_CAMPAIGN_SEED_COUNT - 1
)
REQUIRED_THERMAL_MESH_POLICY = (
    "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
)
REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION = "thermal-mesh-plan-v8"
KST = timezone(timedelta(hours=9))
STANDARD_MODE = {
    "full_model": 0,
    "matrix_on": 1,
    "loss_on": 1,
    "thermal_on": 1,
    "loss_sym_on": 1,
    "thermal_symmetry": "eighth",
    "n_explicit_turns": 0,
    "matrix_skin_mesh": 0,
    "keep_project": 1,
}
DIAGNOSTIC_FLAGS = {
    "diagnostic_only": True,
    "standard_only": True,
    "production_eligible": False,
    "automatic_promotion": False,
    "full_submission_allowed": False,
    "production_package_allowed": False,
}


class StrictALIngestError(RuntimeError):
    """Raised when a prospective truth row cannot enter the strict dataset."""


@dataclass(frozen=True)
class AuthenticatedTruth:
    """One collection after its owning module has authenticated all links."""

    adapter_kind: str
    collection_path: Path
    collection_file_sha256: str
    collection: Mapping[str, Any]
    plan: Mapping[str, Any]
    result: Mapping[str, Any]
    solver_revision: str
    library_revision: str
    source_task_payload_sha256: str
    source_seed: int
    source_fixed_primary_turns: int


@dataclass(frozen=True)
class PreparedIngest:
    """Validated in-memory output and its audit facts."""

    frame: pd.DataFrame
    base_record: Mapping[str, Any]
    collection_records: tuple[Mapping[str, Any], ...]
    base_target_rows: Mapping[str, int]
    output_target_rows: Mapping[str, int]
    physics_data_revision: str
    retraining_admission: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StrictALIngestError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_file() or resolved.is_symlink() or path.is_symlink():
        raise StrictALIngestError(f"{label} is not a regular file: {path}")
    return resolved


def _file_record(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "artifact")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "JSON artifact")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrictALIngestError(
            f"JSON artifact cannot be decoded: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise StrictALIngestError(f"JSON artifact is not an object: {resolved}")
    return value


def _resolve_file_record(
    record: Any, *, owner_path: Path, label: str
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise StrictALIngestError(f"{label} file record is malformed")
    raw_path = Path(str(record["path"]))
    if raw_path.is_absolute():
        target = _regular_file(raw_path, label)
    else:
        root = owner_path.resolve(strict=True).parent
        target = _regular_file(root / raw_path, label)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise StrictALIngestError(
                f"{label} relative path escapes its collection directory"
            ) from exc
    try:
        size = int(record["size_bytes"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictALIngestError(f"{label} size is invalid") from exc
    if (
        size != target.stat().st_size
        or str(record["sha256"]).lower() != _sha256_file(target)
    ):
        raise StrictALIngestError(f"{label} bytes drifted")
    return target


def _collection_plan_path(
    collection: Mapping[str, Any], collection_path: Path
) -> Path:
    return _resolve_file_record(
        collection.get("plan"),
        owner_path=collection_path,
        label="collection plan",
    )


def _authenticate_production_collection(path: Path) -> AuthenticatedTruth:
    raw = _read_json(path)
    if raw.get("schema_version") != PRODUCTION_COLLECTION_SCHEMA:
        raise StrictALIngestError("production collection schema mismatch")
    try:
        sealed = production._validate_seal(
            raw, PRODUCTION_COLLECTION_SCHEMA
        )
        plan_path = _collection_plan_path(sealed, path)
        plan, params, selected = production._load_plan(plan_path)
        collection = production._load_collection(
            path,
            plan=plan,
            params=params,
            selected=selected,
            expected_stage="standard",
        )
    except (OSError, ValueError, production.HandoffContractError) as exc:
        raise StrictALIngestError(
            f"production collection authentication failed: {path}"
        ) from exc
    if (
        collection.get("stage") != "standard"
        or collection.get("scheduler_get_only_collection") is not True
        or collection.get("scheduler_mutation_performed") is not False
        or collection.get("production_eligible") is not False
    ):
        raise StrictALIngestError(
            "production Standard collection safety flags drifted"
        )
    return _authenticated_truth(
        "production",
        path,
        collection=collection,
        plan=plan,
        selected=selected,
    )


def _authenticate_diagnostic_collection(path: Path) -> AuthenticatedTruth:
    raw = _read_json(path)
    if raw.get("schema_version") != DIAGNOSTIC_COLLECTION_SCHEMA:
        raise StrictALIngestError("diagnostic collection schema mismatch")
    try:
        diagnostic = importlib.import_module(
            "tools.mft_goal_diagnostic_standard_probe"
        )
    except ImportError as exc:
        raise StrictALIngestError(
            "diagnostic collection adapter is unavailable in this code revision"
        ) from exc
    if getattr(diagnostic, "COLLECTION_SCHEMA", None) != (
        DIAGNOSTIC_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError("diagnostic adapter schema drifted")
    authenticator = getattr(diagnostic, "authenticate_collection", None)
    if not callable(authenticator):
        raise StrictALIngestError(
            "diagnostic public collection authenticator is unavailable"
        )
    try:
        view = authenticator(path)
    except (OSError, TypeError, ValueError) as exc:
        raise StrictALIngestError(
            f"diagnostic collection authentication failed: {path}"
        ) from exc
    except Exception as exc:
        contract_error = getattr(diagnostic, "HandoffContractError", None)
        if contract_error is not None and isinstance(exc, contract_error):
            raise StrictALIngestError(
                f"diagnostic collection authentication failed: {path}"
            ) from exc
        raise
    if not isinstance(view, Mapping) or set(view) != {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    } or view.get("schema_version") != (
        DIAGNOSTIC_AUTHENTICATED_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError(
            "diagnostic authenticated collection view drifted"
        )
    collection = view.get("collection")
    plan = view.get("plan")
    selected = view.get("selected")
    if not all(
        isinstance(value, Mapping)
        for value in (
            collection,
            plan,
            view.get("params"),
            selected,
            view.get("submission"),
        )
    ):
        raise StrictALIngestError(
            "diagnostic authenticated collection objects are incomplete"
        )
    if any(collection.get(key) is not expected for key, expected in (
        DIAGNOSTIC_FLAGS.items()
    )):
        raise StrictALIngestError(
            "diagnostic collection no-promotion flags drifted"
        )
    if (
        collection.get("scheduler_status") != "completed"
        or collection.get("scheduler_mutation_performed") is not False
        or not isinstance(
            collection.get("selected_candidate_identity"), Mapping
        )
        or not isinstance(collection.get("truth_evidence"), Mapping)
    ):
        raise StrictALIngestError(
            "diagnostic collection Scheduler evidence drifted"
        )
    selected_identity = collection["selected_candidate_identity"]
    if (
        set(selected_identity)
        != {
            "candidate_physics_sha256",
            "source_task_payload_sha256",
            "source_result_sha256",
            "seed",
            "fixed_primary_turns",
        }
        or selected_identity.get("candidate_physics_sha256")
        != collection.get("candidate_physics_sha256")
        or selected_identity.get("source_task_payload_sha256")
        != (selected.get("task_identity") or {}).get("payload_sha256")
    ):
        raise StrictALIngestError(
            "diagnostic selected candidate identity drifted"
        )
    return _authenticated_truth(
        "diagnostic",
        path,
        collection=collection,
        plan=plan,
        selected=selected,
    )


def _authenticate_postdeadline_collection(path: Path) -> AuthenticatedTruth:
    """Adapt the custom post-deadline collector without accepting raw JSON."""

    raw = _read_json(path)
    if raw.get("schema_version") != POSTDEADLINE_COLLECTION_SCHEMA:
        raise StrictALIngestError(
            "post-deadline Standard collection schema mismatch"
        )
    try:
        adapter = importlib.import_module(
            "tools.mft_goal_postdeadline_standard_postsuccess"
        )
    except ImportError as exc:
        raise StrictALIngestError(
            "post-deadline Standard collection adapter is unavailable"
        ) from exc
    if getattr(adapter, "COLLECTION_SCHEMA", None) != (
        POSTDEADLINE_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError(
            "post-deadline Standard adapter schema drifted"
        )
    authenticator = getattr(adapter, "authenticate_collection", None)
    if not callable(authenticator):
        raise StrictALIngestError(
            "post-deadline Standard public authenticator is unavailable"
        )
    try:
        view = authenticator(path)
    except Exception as exc:
        raise StrictALIngestError(
            "post-deadline Standard collection authentication failed: "
            f"{path}"
        ) from exc
    required_view_fields = {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    }
    if (
        not isinstance(view, Mapping)
        or set(view) != required_view_fields
        or view.get("schema_version")
        != POSTDEADLINE_AUTHENTICATED_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError(
            "post-deadline authenticated collection view drifted"
        )
    collection = view.get("collection")
    plan = view.get("plan")
    selected = view.get("selected")
    if not all(
        isinstance(value, Mapping)
        for value in (
            collection,
            plan,
            view.get("params"),
            selected,
            view.get("submission"),
        )
    ):
        raise StrictALIngestError(
            "post-deadline authenticated collection objects are incomplete"
        )
    if (
        collection.get("schema_version")
        != POSTDEADLINE_AUTHENTICATED_COLLECTION_SCHEMA
        or collection.get("diagnostic_only") is not True
        or collection.get("search_only") is not True
        or collection.get("canonical") is not False
        or collection.get("production_eligible") is not False
        or collection.get("original_deadline_missed") is not True
        or collection.get("scheduler_get_only_collection") is not True
        or collection.get("scheduler_mutation_performed") is not False
        or collection.get("scientific_pass_claimed") is not False
        or collection.get("production_claimed") is not False
    ):
        raise StrictALIngestError(
            "post-deadline collection safety boundary drifted"
        )
    return _authenticated_truth(
        "postdeadline_standard",
        path,
        collection=collection,
        plan=plan,
        selected=selected,
    )


def _corrected_replacement_adapter() -> Any:
    try:
        adapter = importlib.import_module(
            "tools.mft_goal_corrected_replacement_strict_collect"
        )
    except ImportError as exc:
        raise StrictALIngestError(
            "corrected replacement collection adapter is unavailable"
        ) from exc
    expected = {
        "COLLECTION_SCHEMA": CORRECTED_REPLACEMENT_COLLECTION_SCHEMA,
        "AUTHENTICATED_COLLECTION_SCHEMA": (
            CORRECTED_REPLACEMENT_AUTHENTICATED_COLLECTION_SCHEMA
        ),
        "MANIFEST_SCHEMA": CORRECTED_REPLACEMENT_MANIFEST_SCHEMA,
        "AUTHENTICATED_MANIFEST_SCHEMA": (
            CORRECTED_REPLACEMENT_AUTHENTICATED_MANIFEST_SCHEMA
        ),
    }
    if any(
        getattr(adapter, name, None) != value
        for name, value in expected.items()
    ):
        raise StrictALIngestError(
            "corrected replacement adapter schema drifted"
        )
    return adapter


def _authenticate_corrected_replacement_collection(
    path: Path,
) -> AuthenticatedTruth:
    raw = _read_json(path)
    if (
        raw.get("schema_version")
        != CORRECTED_REPLACEMENT_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError(
            "corrected replacement collection schema mismatch"
        )
    adapter = _corrected_replacement_adapter()
    authenticator = getattr(adapter, "authenticate_collection", None)
    if not callable(authenticator):
        raise StrictALIngestError(
            "corrected replacement public authenticator is unavailable"
        )
    try:
        view = authenticator(path)
    except Exception as exc:
        contract_error = getattr(
            adapter, "CorrectedReplacementCollectionError", None
        )
        if contract_error is not None and isinstance(exc, contract_error):
            raise StrictALIngestError(
                "corrected replacement collection authentication failed: "
                f"{path}"
            ) from exc
        raise
    required_view_fields = {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    }
    if (
        not isinstance(view, Mapping)
        or set(view) != required_view_fields
        or view.get("schema_version")
        != CORRECTED_REPLACEMENT_AUTHENTICATED_COLLECTION_SCHEMA
    ):
        raise StrictALIngestError(
            "corrected replacement authenticated view drifted"
        )
    collection = view.get("collection")
    plan = view.get("plan")
    selected = view.get("selected")
    if not all(
        isinstance(value, Mapping)
        for value in (
            collection,
            plan,
            view.get("params"),
            selected,
            view.get("submission"),
        )
    ):
        raise StrictALIngestError(
            "corrected replacement authenticated objects are incomplete"
        )
    required_safety = {
        "scheduler_status": "completed",
        "scheduler_get_only_collection": True,
        "scheduler_mutation_performed": False,
        "scientific_valid": True,
        "unique_geometry_admitted": True,
        "truth_use_only": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "legacy_collection_reused": False,
        "legacy_collection_rows_reused": 0,
    }
    if any(
        collection.get(key) != expected
        for key, expected in required_safety.items()
    ):
        raise StrictALIngestError(
            "corrected replacement collection safety boundary drifted"
        )
    return _authenticated_truth(
        "corrected_replacement",
        path,
        collection=collection,
        plan=plan,
        selected=selected,
    )


def _authenticated_truth(
    adapter_kind: str,
    path: Path,
    *,
    collection: Mapping[str, Any],
    plan: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> AuthenticatedTruth:
    result = collection.get("result")
    if not isinstance(result, Mapping):
        raise StrictALIngestError("authenticated collection result is absent")
    solver = str(plan.get("solver_revision") or "").strip().lower()
    library = str(plan.get("library_revision") or "").strip().lower()
    for revision, label in (
        (solver, "solver revision"),
        (library, "library revision"),
    ):
        if (
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise StrictALIngestError(f"authenticated {label} is invalid")
    task_identity = selected.get("task_identity")
    if not isinstance(task_identity, Mapping):
        raise StrictALIngestError(
            "authenticated source task identity is absent"
        )
    source_task = _require_sha256(
        task_identity.get("payload_sha256"),
        "authenticated source task payload SHA",
    )
    source_seed = task_identity.get("seed")
    source_turns = task_identity.get("fixed_primary_turns")
    if (
        isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or source_seed < 0
        or isinstance(source_turns, bool)
        or not isinstance(source_turns, int)
        or not GOAL_N1_MIN_TURNS <= source_turns <= GOAL_N1_MAX_TURNS
    ):
        raise StrictALIngestError(
            "authenticated source task seed/N1 identity is invalid"
        )
    return AuthenticatedTruth(
        adapter_kind=adapter_kind,
        collection_path=path.resolve(strict=True),
        collection_file_sha256=_sha256_file(path.resolve(strict=True)),
        collection=collection,
        plan=plan,
        result=result,
        solver_revision=solver,
        library_revision=library,
        source_task_payload_sha256=source_task,
        source_seed=source_seed,
        source_fixed_primary_turns=source_turns,
    )


def authenticate_collection(path: Path) -> AuthenticatedTruth:
    """Dispatch only exact, named collection schemas to their owning loader."""
    resolved = _regular_file(path, "collection")
    schema = _read_json(resolved).get("schema_version")
    if schema == PRODUCTION_COLLECTION_SCHEMA:
        return _authenticate_production_collection(resolved)
    if schema == DIAGNOSTIC_COLLECTION_SCHEMA:
        return _authenticate_diagnostic_collection(resolved)
    if schema == POSTDEADLINE_COLLECTION_SCHEMA:
        return _authenticate_postdeadline_collection(resolved)
    if schema == CORRECTED_REPLACEMENT_COLLECTION_SCHEMA:
        return _authenticate_corrected_replacement_collection(resolved)
    raise StrictALIngestError(
        "unsupported collection schema; raw JSON/CSV ingestion is forbidden: "
        f"{schema!r}"
    )


def collections_from_corrected_manifests(
    manifest_paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Expand only authenticated corrected-replacement aggregate manifests."""

    if not manifest_paths:
        return ()
    if len(set(map(str, manifest_paths))) != len(manifest_paths):
        raise StrictALIngestError(
            "corrected replacement manifest paths are duplicated"
        )
    adapter = _corrected_replacement_adapter()
    authenticator = getattr(adapter, "authenticate_manifest", None)
    if not callable(authenticator):
        raise StrictALIngestError(
            "corrected replacement manifest authenticator is unavailable"
        )
    collections: list[Path] = []
    for manifest_path in manifest_paths:
        resolved = _regular_file(
            manifest_path, "corrected replacement manifest"
        )
        raw = _read_json(resolved)
        if (
            raw.get("schema_version")
            != CORRECTED_REPLACEMENT_MANIFEST_SCHEMA
        ):
            raise StrictALIngestError(
                "corrected replacement manifest schema mismatch"
            )
        try:
            view = authenticator(resolved)
        except Exception as exc:
            contract_error = getattr(
                adapter, "CorrectedReplacementCollectionError", None
            )
            if contract_error is not None and isinstance(
                exc, contract_error
            ):
                raise StrictALIngestError(
                    "corrected replacement manifest authentication failed: "
                    f"{resolved}"
                ) from exc
            raise
        if (
            not isinstance(view, Mapping)
            or set(view)
            != {
                "schema_version",
                "manifest",
                "collection_paths",
                "retraining_trigger",
            }
            or view.get("schema_version")
            != CORRECTED_REPLACEMENT_AUTHENTICATED_MANIFEST_SCHEMA
            or not isinstance(view.get("manifest"), Mapping)
            or not isinstance(view.get("retraining_trigger"), Mapping)
            or not isinstance(view.get("collection_paths"), tuple)
        ):
            raise StrictALIngestError(
                "corrected replacement authenticated manifest view drifted"
            )
        collections.extend(view["collection_paths"])
    if len(collections) != len(set(collections)):
        raise StrictALIngestError(
            "corrected replacement manifests duplicate a collection"
        )
    return tuple(collections)


def _same_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return actual == expected
    if isinstance(actual, bool):
        return False
    try:
        number = float(actual)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number == float(expected)


def _task_token(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value or "").strip()
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return ""
    return str(int(number))


def _nonempty_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        raise StrictALIngestError(f"{label} is absent")
    return text


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise StrictALIngestError(f"{label} is invalid")
    return digest


def _logical_authority_task_id(
    plan: Mapping[str, Any],
    *,
    execution_task_id: int,
) -> int:
    """Return the sealed cohort slot represented by one execution."""

    pressure = plan.get("retry_of_operational_pressure")
    timeout = plan.get("retry_of_timeout")
    if pressure is not None and timeout is not None:
        raise StrictALIngestError(
            "authenticated retry ancestry semantics are mixed"
        )
    value = execution_task_id
    if pressure is not None:
        if not isinstance(pressure, Mapping):
            raise StrictALIngestError(
                "operational-pressure retry ancestry is malformed"
            )
        value = pressure.get("logical_authority_task_id")
    elif timeout is not None:
        if not isinstance(timeout, Mapping):
            raise StrictALIngestError(
                "timeout retry ancestry is malformed"
            )
        value = timeout.get("retry_of_task_id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StrictALIngestError(
            "logical authority task ID is invalid"
        )
    return value


def _validate_truth_row(
    truth: AuthenticatedTruth,
    *,
    profile: Mapping[str, Any],
    physics_data_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = dict(truth.result)
    if any(
        not _same_exact(result.get(key), expected)
        for key, expected in STANDARD_MODE.items()
    ):
        raise StrictALIngestError(
            "authenticated result is not the retained eighth-symmetry "
            "Standard mode"
        )
    if (
        result.get("thermal_mesh_policy") != REQUIRED_THERMAL_MESH_POLICY
        or result.get("thermal_mesh_plan_contract_version")
        != REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
    ):
        raise StrictALIngestError(
            "authenticated result is not exact B7/thermal-mesh-plan-v8 truth"
        )
    observed_temperatures = {}
    for target in GOAL_TEMPERATURE_TARGETS:
        try:
            value = float(result.get(target))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            observed_temperatures[target] = value
    scientific = thermal_truth.thermal_scientific_truth_contract(
        result,
        observed_temperatures_C=observed_temperatures,
    )
    if scientific["valid"] is not True:
        raise StrictALIngestError(
            "authenticated result fails thermal scientific truth contract: "
            + ";".join(scientific["reasons"])
        )
    try:
        fixed_evidence = attest_fixed_identity(
            result, require_thermal_pad_metadata=True
        )
    except Exception as exc:
        raise StrictALIngestError(
            "authenticated result changed the fixed operating/cooling identity"
        ) from exc
    if (
        str(result.get("git_hash") or "").strip().lower()
        != truth.solver_revision
        or str(result.get("pyaedt_library_git_hash") or "").strip().lower()
        != truth.library_revision
    ):
        raise StrictALIngestError(
            "result solver/library revisions differ from the sealed plan"
        )
    if str(result.get("physics_data_revision") or "").strip() != (
        physics_data_revision
    ):
        raise StrictALIngestError(
            "result physics_data_revision differs from the base cohort"
        )
    missing_inputs = sorted(
        key for key in ALL_INPUT_KEYS if key not in result
    )
    if missing_inputs:
        raise StrictALIngestError(
            "authenticated result lacks exact solver inputs: "
            + ",".join(missing_inputs)
        )
    validation = validate_record(
        result,
        profile,
        expected_solver_revision=truth.solver_revision,
        expected_library_revision=truth.library_revision,
    )
    if not validation.full_valid:
        raise StrictALIngestError(
            "authenticated result fails strict EM/thermal validity: "
            + ";".join(validation.reasons)
        )
    task_id = _task_token(truth.collection.get("task_id"))
    if not task_id:
        raise StrictALIngestError("collection task_id is invalid")
    result_task = result.get("task_id")
    if result_task is not None and _task_token(result_task) != task_id:
        raise StrictALIngestError(
            "result task_id differs from the authenticated collection"
        )
    result["task_id"] = int(task_id)
    project_name = _nonempty_text(result.get("project_name"), "project_name")
    saved_at = _nonempty_text(result.get("saved_at"), "saved_at")

    one = annotate_validity(
        pd.DataFrame([result]),
        profile,
        expected_solver_revision=truth.solver_revision,
        expected_library_revision=truth.library_revision,
    )
    physical = to_physical(one)
    target_readiness: dict[str, int] = {}
    for target in GOAL_G0_MODEL_TARGETS:
        eligible = filter_valid_training_rows(physical, target, profile)
        target_readiness[target] = len(eligible)
        if len(eligible) != 1:
            raise StrictALIngestError(
                f"authenticated result is not trainable for target {target}"
            )
        if eligible.attrs.get("physics_data_revision_cohort") != (
            physics_data_revision
        ):
            raise StrictALIngestError(
                f"target {target} physics cohort drifted"
            )
    n1 = result.get("N1")
    if isinstance(n1, bool):
        raise StrictALIngestError("result N1 is invalid")
    try:
        n1_number = float(n1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictALIngestError("result N1 is invalid") from exc
    if (
        not n1_number.is_integer()
        or not GOAL_N1_MIN_TURNS <= int(n1_number) <= GOAL_N1_MAX_TURNS
    ):
        raise StrictALIngestError("result N1 is outside goal strata 5..8")
    if int(n1_number) != truth.source_fixed_primary_turns:
        raise StrictALIngestError(
            "result N1 differs from the authenticated source task stratum"
        )
    if int(n1_number) != TARGETED_PRIMARY_TURNS:
        raise StrictALIngestError(
            f"active-learning truth is not targeted N1={TARGETED_PRIMARY_TURNS}"
        )

    facts = {
        "adapter_kind": truth.adapter_kind,
        "collection": _file_record(truth.collection_path),
        "collection_schema": truth.collection.get("schema_version"),
        "collection_payload_sha256": _require_sha256(
            truth.collection.get("payload_sha256"),
            "collection payload SHA",
        ),
        "plan_payload_sha256": _require_sha256(
            truth.plan.get("payload_sha256"),
            "plan payload SHA",
        ),
        "candidate_physics_sha256": _require_sha256(
            truth.collection.get("candidate_physics_sha256"),
            "candidate physics SHA",
        ),
        "source_task_payload_sha256": truth.source_task_payload_sha256,
        "source_seed": truth.source_seed,
        "source_fixed_primary_turns": (
            truth.source_fixed_primary_turns
        ),
        "task_id": int(task_id),
        "logical_authority_task_id": _logical_authority_task_id(
            truth.plan,
            execution_task_id=int(task_id),
        ),
        "result_sha256": _require_sha256(
            truth.collection.get("result_sha256"),
            "result SHA",
        ),
        "solver_revision": truth.solver_revision,
        "library_revision": truth.library_revision,
        "project_name": project_name,
        "saved_at": saved_at,
        "N1": int(n1_number),
        "thermal_mesh_policy": REQUIRED_THERMAL_MESH_POLICY,
        "thermal_mesh_plan_contract_version": (
            REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
        ),
        "strict_full_valid": True,
        "all_25_targets_trainable": True,
        "target_readiness": target_readiness,
        "fixed_identity_attestation_sha256": fixed_evidence["sha256"],
        "goal_physical_spec_passed": truth.collection.get(
            "goal_physical_spec_passed"
        ),
        "truth_use_only": True,
        "promotion_authority_granted": False,
    }
    return result, facts


def _profile_content() -> dict[str, Any]:
    profile = _read_json(GOAL_PROFILE_PATH)
    if profile.get("stage") != "standard":
        raise StrictALIngestError("goal Standard profile stage drifted")
    return profile


def _target_counts(
    frame: pd.DataFrame, profile: Mapping[str, Any]
) -> dict[str, int]:
    physical = to_physical(frame)
    return {
        target: len(filter_valid_training_rows(physical, target, profile))
        for target in GOAL_G0_MODEL_TARGETS
    }


def _normalize_append_only_base_inputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Default only solver controls that were append-only after base capture.

    The pinned 6,151-row base predates the center-gap and turn-graded
    electrostatic controls.  Both contracts explicitly define missing
    pre-extension values: zero gap and turn-graded mode off.  Add those
    values in memory so the immutable source bytes stay unchanged.  Missing
    Sobol/design inputs and every other solver input remain fatal.
    """

    missing = tuple(key for key in ALL_INPUT_KEYS if key not in frame.columns)
    unsupported = sorted(set(missing) - set(APPEND_ONLY_BASE_INPUT_KEYS))
    if unsupported:
        raise StrictALIngestError(
            "base dataset lacks solver inputs: " + ",".join(unsupported)
        )
    defaults = get_drawing_default_params()
    appended = {
        key: defaults[key]
        for key in APPEND_ONLY_BASE_INPUT_KEYS
        if key in missing
    }
    normalized = frame.copy() if appended else frame
    for key, value in appended.items():
        normalized[key] = value
    remaining = sorted(
        key for key in ALL_INPUT_KEYS if key not in normalized.columns
    )
    if remaining:
        raise StrictALIngestError(
            "base dataset lacks solver inputs after append-only "
            "normalization: "
            + ",".join(remaining)
        )
    return normalized, {
        "schema_version": BASE_INPUT_NORMALIZATION_SCHEMA,
        "performed": bool(appended),
        "policy": (
            "in_memory_defaults_for_append_only_fixed_run_controls_only"
        ),
        "append_only_allowed_keys": list(APPEND_ONLY_BASE_INPUT_KEYS),
        "appended_defaults": appended,
        "appended_keys": list(appended),
        "source_dataset_mutated": False,
        "source_bytes_rewritten": False,
    }


def _concat_without_implicit_all_na_dtype(
    base: pd.DataFrame, incoming: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate while making pandas' all-NA dtype choice explicit."""
    columns = [
        *base.columns,
        *(column for column in incoming.columns if column not in base.columns),
    ]
    base_all_na = [
        column for column in base if base[column].isna().all()
    ]
    incoming_all_na = [
        column for column in incoming if incoming[column].isna().all()
    ]
    combined = pd.concat(
        [
            base.drop(columns=base_all_na),
            incoming.drop(columns=incoming_all_na),
        ],
        ignore_index=True,
        sort=False,
    )
    for column in columns:
        if column in combined:
            continue
        template = (
            base[column] if column in base else incoming[column]
        )
        combined[column] = template.iloc[:0].reindex(combined.index)
    return combined.reindex(columns=columns)


def _base_audit(
    base_dataset: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], str, dict[str, int]]:
    base_path = _regular_file(base_dataset, "base dataset")
    observed_sha = _sha256_file(base_path)
    if observed_sha != expected_sha256.lower():
        raise StrictALIngestError(
            "base dataset SHA-256 differs from the explicitly pinned source"
        )
    try:
        frame = pd.read_parquet(base_path)
    except Exception as exc:
        raise StrictALIngestError("base parquet cannot be read") from exc
    if len(frame) != expected_rows:
        raise StrictALIngestError(
            f"base row count drifted: expected {expected_rows}, got {len(frame)}"
        )
    source_column_count = len(frame.columns)
    frame, input_normalization = _normalize_append_only_base_inputs(frame)
    audited = annotate_validity(frame, profile)
    if not audited["_strict_valid_full"].fillna(False).all():
        invalid = int((~audited["_strict_valid_full"].fillna(False)).sum())
        raise StrictALIngestError(
            f"base dataset contains {invalid} non-strict rows"
        )
    if "physics_data_revision" not in audited:
        raise StrictALIngestError("base physics_data_revision is absent")
    revisions = tuple(sorted({
        str(value).strip()
        for value in audited["physics_data_revision"].dropna().tolist()
        if str(value).strip()
    }))
    if len(revisions) != 1:
        raise StrictALIngestError(
            f"base physics_data_revision cohort is not unique: {revisions}"
        )
    if "task_id" not in audited:
        raise StrictALIngestError("base task_id column is absent")
    task_tokens = audited["task_id"].map(_task_token)
    if (task_tokens == "").any() or task_tokens.duplicated().any():
        raise StrictALIngestError("base task_id identity is incomplete or duplicated")
    for column in ("project_name", "saved_at"):
        if column not in audited:
            raise StrictALIngestError(f"base {column} column is absent")
    pairs = audited[["project_name", "saved_at"]].fillna("").astype(str)
    if (
        pairs.eq("").any(axis=None)
        or pairs.duplicated().any()
    ):
        raise StrictALIngestError(
            "base project_name/saved_at identity is incomplete or duplicated"
        )
    target_rows = _target_counts(audited, profile)
    if any(count <= 0 for count in target_rows.values()):
        raise StrictALIngestError("base does not support all 25 target models")
    record = _file_record(base_path)
    record["row_count"] = len(audited)
    record["column_count"] = source_column_count
    record["normalized_column_count"] = len(frame.columns)
    record["audited_column_count"] = len(audited.columns)
    record["input_schema_normalization"] = input_normalization
    record["strict_full_row_count"] = int(
        audited["_strict_valid_full"].sum()
    )
    return audited, record, revisions[0], target_rows


def _admission(
    collection_facts: Sequence[Mapping[str, Any]],
    *,
    minimum_useful_rows: int,
    minimum_source_tasks: int,
) -> dict[str, Any]:
    if minimum_useful_rows < DEFAULT_MINIMUM_USEFUL_ROWS:
        raise StrictALIngestError(
            f"minimum_useful_rows cannot be below "
            f"{DEFAULT_MINIMUM_USEFUL_ROWS}"
        )
    if minimum_source_tasks < DEFAULT_MINIMUM_SOURCE_TASKS:
        raise StrictALIngestError(
            f"minimum_source_tasks cannot be below "
            f"{DEFAULT_MINIMUM_SOURCE_TASKS}"
        )
    source_tasks = {
        str(fact["source_task_payload_sha256"])
        for fact in collection_facts
    }
    targeted_strata = sorted({
        int(fact["N1"]) for fact in collection_facts
    })
    unique_geometries = {
        str(fact["candidate_physics_sha256"])
        for fact in collection_facts
    }
    reasons = []
    if len(collection_facts) < minimum_useful_rows:
        reasons.append(
            f"strict_new_rows<{minimum_useful_rows}"
        )
    if len(unique_geometries) < DEFAULT_MINIMUM_UNIQUE_GEOMETRIES:
        reasons.append(
            "unique_complete_geometries"
            f"<{DEFAULT_MINIMUM_UNIQUE_GEOMETRIES}"
        )
    if len(source_tasks) < minimum_source_tasks:
        reasons.append(
            f"unique_source_tasks<{minimum_source_tasks}"
        )
    if targeted_strata != [TARGETED_PRIMARY_TURNS]:
        reasons.append(
            f"targeted_strata!=[{TARGETED_PRIMARY_TURNS}]"
        )
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "strict_new_rows": len(collection_facts),
        "minimum_useful_rows": minimum_useful_rows,
        "unique_complete_geometries": len(unique_geometries),
        "minimum_unique_complete_geometries": (
            DEFAULT_MINIMUM_UNIQUE_GEOMETRIES
        ),
        "unique_source_tasks": len(source_tasks),
        "minimum_source_tasks": minimum_source_tasks,
        "targeted_strata": targeted_strata,
        "global_N1_coverage_claimed": False,
        "recommended_new_rows": RECOMMENDED_NEW_ROWS,
        "diagnostic_selection_ceiling": MAX_DIAGNOSTIC_SELECTION_ROWS,
        "all_25_targets_required_per_new_row": True,
        "quality_threshold_relaxation_allowed": False,
        "new_generation_required": True,
        "new_NSGA2_campaign_all_N1_strata_required": True,
        "old_campaign_result_mixing_allowed": False,
    }


def _validate_new_collection_identities(
    facts: Sequence[Mapping[str, Any]],
) -> None:
    identity_fields = {
        "collection_payload_sha256": [
            str(fact["collection_payload_sha256"]) for fact in facts
        ],
        "candidate_physics_sha256": [
            str(fact["candidate_physics_sha256"]) for fact in facts
        ],
        "task_id": [str(fact["task_id"]) for fact in facts],
        "logical_authority_task_id": [
            str(fact["logical_authority_task_id"]) for fact in facts
        ],
        "project_name/saved_at": [
            f"{fact['project_name']}\0{fact['saved_at']}"
            for fact in facts
        ],
    }
    for label, values in identity_fields.items():
        if len(values) != len(set(values)):
            raise StrictALIngestError(
                f"new collection {label} identity is duplicated"
            )


def prepare_ingest(
    *,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
    collection_paths: Sequence[Path],
    minimum_useful_rows: int = DEFAULT_MINIMUM_USEFUL_ROWS,
    minimum_source_tasks: int = DEFAULT_MINIMUM_SOURCE_TASKS,
) -> PreparedIngest:
    """Authenticate sources and return a new in-memory strict dataset."""
    if not collection_paths:
        raise StrictALIngestError("at least one collection is required")
    if len(set(map(str, collection_paths))) != len(collection_paths):
        raise StrictALIngestError("collection paths are duplicated")
    profile = _profile_content()
    base, base_record, revision, base_counts = _base_audit(
        base_dataset,
        expected_sha256=expected_base_sha256,
        expected_rows=expected_base_rows,
        profile=profile,
    )
    truths = [authenticate_collection(path) for path in collection_paths]
    rows_and_facts = [
        _validate_truth_row(
            truth,
            profile=profile,
            physics_data_revision=revision,
        )
        for truth in truths
    ]
    rows = [item[0] for item in rows_and_facts]
    facts = [item[1] for item in rows_and_facts]

    _validate_new_collection_identities(facts)

    base_tasks = set(base["task_id"].map(_task_token))
    base_pairs = set(
        base["project_name"].astype(str)
        + "\0"
        + base["saved_at"].astype(str)
    )
    base_logical_authorities = set()
    if "goal_al_logical_authority_task_id" in base:
        base_logical_authorities = {
            _task_token(value)
            for value in base["goal_al_logical_authority_task_id"]
            if _task_token(value)
        }
    base_candidate_physics = set()
    if "goal_al_candidate_physics_sha256" in base:
        base_candidate_physics = {
            str(value).strip().lower()
            for value in base["goal_al_candidate_physics_sha256"]
            if str(value).strip().lower()
            not in {"", "nan", "none", "<na>"}
        }
    if any(str(fact["task_id"]) in base_tasks for fact in facts):
        raise StrictALIngestError("new task_id already exists in the base")
    if any(
        f"{fact['project_name']}\0{fact['saved_at']}" in base_pairs
        for fact in facts
    ):
        raise StrictALIngestError(
            "new project_name/saved_at already exists in the base"
        )
    if any(
        str(fact["logical_authority_task_id"])
        in base_logical_authorities
        for fact in facts
    ):
        raise StrictALIngestError(
            "new logical authority task ID already exists in the base"
        )
    if any(
        str(fact["candidate_physics_sha256"])
        in base_candidate_physics
        for fact in facts
    ):
        raise StrictALIngestError(
            "new candidate physics SHA already exists in the base"
        )

    aligned_rows = []
    for row, fact in zip(rows, facts):
        aligned = {
            column: row.get(column, pd.NA)
            for column in base.columns
            if not column.startswith("_strict_")
        }
        aligned["task_id"] = fact["task_id"]
        aligned.update({
            "goal_al_collection_schema": fact["collection_schema"],
            "goal_al_collection_payload_sha256": fact[
                "collection_payload_sha256"
            ],
            "goal_al_collection_file_sha256": fact["collection"]["sha256"],
            "goal_al_plan_payload_sha256": fact["plan_payload_sha256"],
            "goal_al_result_sha256": fact["result_sha256"],
            "goal_al_candidate_physics_sha256": fact[
                "candidate_physics_sha256"
            ],
            "goal_al_logical_authority_task_id": fact[
                "logical_authority_task_id"
            ],
            "goal_al_source_task_payload_sha256": fact[
                "source_task_payload_sha256"
            ],
            "goal_al_adapter_kind": fact["adapter_kind"],
            "goal_al_authenticated_standard_truth": 1,
        })
        aligned_rows.append(aligned)

    base_copy = base.drop(
        columns=[column for column in base if column.startswith("_strict_")],
        errors="ignore",
    ).copy()
    combined = _concat_without_implicit_all_na_dtype(
        base_copy,
        pd.DataFrame(aligned_rows),
    )
    combined = annotate_validity(combined, profile)
    if not combined["_strict_valid_full"].fillna(False).all():
        raise StrictALIngestError(
            "schema-aligned output lost strict validity"
        )
    combined = to_physical(combined)
    output_counts = _target_counts(combined, profile)
    expected_added = len(rows)
    for target in GOAL_G0_MODEL_TARGETS:
        if output_counts[target] != base_counts[target] + expected_added:
            raise StrictALIngestError(
                f"output target {target} did not gain every new truth row"
            )
    revisions = set(
        combined["physics_data_revision"].dropna().astype(str).str.strip()
    )
    if revisions != {revision}:
        raise StrictALIngestError(
            "output physics_data_revision cohort is mixed"
        )
    admission = _admission(
        facts,
        minimum_useful_rows=minimum_useful_rows,
        minimum_source_tasks=minimum_source_tasks,
    )
    return PreparedIngest(
        frame=combined,
        base_record=base_record,
        collection_records=tuple(facts),
        base_target_rows=base_counts,
        output_target_rows=output_counts,
        physics_data_revision=revision,
        retraining_admission=admission,
    )


def _repository_identity() -> dict[str, Any]:
    environment = os.environ.copy()
    config_count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment[f"GIT_CONFIG_KEY_{config_count}"] = "safe.directory"
    environment[f"GIT_CONFIG_VALUE_{config_count}"] = (
        REPOSITORY_ROOT.resolve(strict=True).as_posix()
    )
    environment["GIT_CONFIG_COUNT"] = str(config_count + 1)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StrictALIngestError(
            "tool repository identity cannot be authenticated"
        ) from exc
    if dirty:
        raise StrictALIngestError(
            "tool repository must be clean before an immutable dataset build"
        )
    return {
        "revision": revision,
        "dirty": False,
        "tool_source": _file_record(Path(__file__)),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def build_immutable_bundle(
    prepared: PreparedIngest,
    *,
    output_dir: Path,
    base_dataset: Path,
    repository_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish a new parquet+manifest directory."""
    output = output_dir.resolve()
    if output.exists():
        raise StrictALIngestError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    base_path = _regular_file(base_dataset, "base dataset")
    before_sha = _sha256_file(base_path)
    if before_sha != prepared.base_record["sha256"]:
        raise StrictALIngestError("base dataset changed after validation")
    repository = dict(repository_identity or _repository_identity())
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    ))
    try:
        dataset_path = staging / "strict_al.parquet"
        prepared.frame.to_parquet(dataset_path, index=False)
        dataset_record = {
            "path": dataset_path.name,
            "sha256": _sha256_file(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "row_count": len(prepared.frame),
            "column_count": len(prepared.frame.columns),
        }
        unsigned = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_kst": datetime.now(KST).isoformat(
                timespec="seconds"
            ),
            "repository": repository,
            "goal_profile": _file_record(GOAL_PROFILE_PATH),
            "goal_targets": list(GOAL_G0_MODEL_TARGETS),
            "base_dataset": dict(prepared.base_record),
            "authenticated_standard_collections": list(
                prepared.collection_records
            ),
            "authenticated_thermal_mesh_truth": {
                "thermal_mesh_policy": REQUIRED_THERMAL_MESH_POLICY,
                "thermal_mesh_plan_contract_version": (
                    REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
                ),
                "authenticated_row_count": len(
                    prepared.collection_records
                ),
                "every_authenticated_row_exact_B7_v8": all(
                    record.get("thermal_mesh_policy")
                    == REQUIRED_THERMAL_MESH_POLICY
                    and record.get("thermal_mesh_plan_contract_version")
                    == REQUIRED_THERMAL_MESH_PLAN_CONTRACT_VERSION
                    for record in prepared.collection_records
                ),
            },
            "output_dataset": dataset_record,
            "base_target_eligible_rows": dict(prepared.base_target_rows),
            "output_target_eligible_rows": dict(prepared.output_target_rows),
            "physics_data_revision_cohort": (
                prepared.physics_data_revision
            ),
            "retraining_admission": dict(
                prepared.retraining_admission
            ),
            "canonical_source_mutated": False,
            "scheduler_submission_performed": False,
            "scheduler_mutation_performed": False,
            "physics_override_performed": False,
            "quality_threshold_relaxation_performed": False,
            "new_model_generation_required": True,
            "new_seed_campaign_required": True,
            "old_generation_result_mixing_allowed": False,
            "next_campaign_contract": {
                "seed_start": NEXT_CAMPAIGN_SEED_START,
                "seed_end_inclusive": NEXT_CAMPAIGN_SEED_END,
                "seed_count": NEXT_CAMPAIGN_SEED_COUNT,
                "all_four_N1_strata_required": True,
                "single_dataset_sha256_required": dataset_record["sha256"],
                "single_model_generation_required": True,
                "old_generation_result_mixing_allowed": False,
            },
            "production_promotion_authority_granted": False,
        }
        manifest = dict(unsigned)
        manifest["payload_sha256"] = canonical_sha256(unsigned)
        _write_manifest(staging / "manifest.json", manifest)
        if _sha256_file(base_path) != before_sha:
            raise StrictALIngestError(
                "base dataset changed while writing isolated output"
            )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def _summary(prepared: PreparedIngest) -> dict[str, Any]:
    return {
        "schema_version": "mft-goal-strict-al-inspection-v1",
        "base_dataset": dict(prepared.base_record),
        "new_rows": len(prepared.collection_records),
        "output_rows": len(prepared.frame),
        "physics_data_revision_cohort": prepared.physics_data_revision,
        "all_25_targets_gain_each_new_row": all(
            prepared.output_target_rows[target]
            == prepared.base_target_rows[target]
            + len(prepared.collection_records)
            for target in GOAL_G0_MODEL_TARGETS
        ),
        "retraining_admission": dict(prepared.retraining_admission),
        "scheduler_mutation_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reauthenticate sealed Standard FEA collections and build an "
            "isolated strict active-learning dataset."
        )
    )
    parser.add_argument("command", choices=("inspect", "build"))
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-base-rows", type=int, required=True)
    parser.add_argument(
        "--collection",
        type=Path,
        action="append",
        help=(
            "Repeat for each sealed production/diagnostic collection. "
            "Cannot be mixed with --collection-manifest."
        ),
    )
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        action="append",
        help=(
            "Repeat for each sealed corrected-replacement strict manifest. "
            "The manifest's authenticated collections are consumed directly."
        ),
    )
    parser.add_argument(
        "--minimum-useful-rows",
        type=int,
        default=DEFAULT_MINIMUM_USEFUL_ROWS,
    )
    parser.add_argument(
        "--minimum-source-tasks",
        type=int,
        default=DEFAULT_MINIMUM_SOURCE_TASKS,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--require-retraining-ready",
        action="store_true",
        help="Fail before writing unless the 8-row/4-source-task gate passes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.expected_base_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_base_sha256.lower()
    ):
        raise StrictALIngestError("expected base SHA-256 is invalid")
    if args.expected_base_rows <= 0:
        raise StrictALIngestError("expected base row count must be positive")
    if args.collection and args.collection_manifest:
        raise StrictALIngestError(
            "direct collections cannot be mixed with corrected replacement "
            "manifests"
        )
    if args.collection_manifest:
        collection_paths = collections_from_corrected_manifests(
            args.collection_manifest
        )
    else:
        collection_paths = tuple(args.collection or ())
    if not collection_paths:
        raise StrictALIngestError(
            "at least one authenticated collection or corrected replacement "
            "manifest collection is required"
        )
    prepared = prepare_ingest(
        base_dataset=args.base_dataset,
        expected_base_sha256=args.expected_base_sha256.lower(),
        expected_base_rows=args.expected_base_rows,
        collection_paths=collection_paths,
        minimum_useful_rows=args.minimum_useful_rows,
        minimum_source_tasks=args.minimum_source_tasks,
    )
    summary = _summary(prepared)
    if args.require_retraining_ready and not (
        prepared.retraining_admission["allowed"]
    ):
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise StrictALIngestError(
            "strict rows are valid but the retraining admission gate did not pass"
        )
    if args.command == "inspect":
        if args.output_dir is not None:
            raise StrictALIngestError(
                "--output-dir is forbidden for read-only inspect"
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        raise StrictALIngestError("--output-dir is required for build")
    manifest = build_immutable_bundle(
        prepared,
        output_dir=args.output_dir,
        base_dataset=args.base_dataset,
    )
    output = dict(summary)
    output["manifest"] = str(manifest)
    output["manifest_sha256"] = _sha256_file(manifest)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StrictALIngestError as exc:
        print(f"[strict-al-ingest] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
