"""Sealed post-deadline Standard probe for official candidate number six.

``prepare`` is local-write plus Scheduler GET only.  It authenticates the
frozen 512-seed aggregate, the source island result, candidate physics, the
reviewed solver command, and one empty strict ``dw16/n113`` execution lane.

``submit`` is deliberately separate and requires an explicit authorization
token.  It reauthenticates every input and repeats the live GET gates before
performing at most one Scheduler POST.  Neither command edits the independent
Slurm Scheduler repository or its project configuration.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
CAMPAIGN_ID = "mft-goal-20260726"
SCHEDULER_URL = "http://127.0.0.1:8002"
PROJECT = "MFT_1MW_2026v1"
ACCOUNT_NAME = "dw16"
NODE_NAME = "n113"

CANDIDATE_SHA256 = (
    "2772aed82a8c2c5d7e861b7b864e24d11d3ccfb601dfb1b0535b5f1e1ceb9a9e"
)
SOURCE_SEED = 2607262505
SOURCE_TASK_ID = 96185
SOURCE_TASK_NAME = "mft-goal-nsga-s2607262505-n1-6"
SOURCE_TASK_DEDUPE_KEY = (
    "mft-goal-20260726-nsga:"
    "264eb24fbb8e283eee75da263af89812bd543a611ec8dfd7e92cdc5b0df54a44"
)
SOURCE_BUNDLE_ID = (
    "7e6276c58b44f5f2065ca80a16093617d121dc26cbf268406dacaa49b895ae38"
)
SOURCE_RESULT_SHA256 = (
    "7cb9aab49a7cfde376f4aa7e10345d0c587271b82e9a0f34f72b9813206dd2b7"
)

AGGREGATE_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\aggregate_rolling512_d4e4d60"
)
STANDARD_CANDIDATES = AGGREGATE_ROOT / "standard_candidates.csv"
AGGREGATE_MANIFEST = AGGREGATE_ROOT / "aggregate_manifest.json"
STANDARD_CANDIDATES_SHA256 = (
    "f99ee4b7fba62a83e5b958d24c31260a5ba48c6fa77e6d25f2f189896df3ff6d"
)
AGGREGATE_MANIFEST_SHA256 = (
    "0ea4feb43f9eb37c8b057e7f06720f0c23590b463111443f99d8ffa6d08c4a63"
)
AGGREGATE_PAYLOAD_SHA256 = (
    "246417be7fb19629c0bb576999ebe711396f9774cebf83996be9dc83538bd96b"
)

PROFILE_PATH = (
    REPOSITORY
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_timeout12h_retry.json"
)
PROFILE_SHA256 = (
    "9d8763c80b782da87d185f734484294c16e22a2b30d88a17a6740bd849d9e844"
)
INPUT_PARAMETER_PATH = "module/input_parameter_260706.py"
SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
CORE_AUTH_SHA256 = (
    "e6685d09a0c4b11cc9d80a62aff12c9e3f0298519b0101c357d409df4319c299"
)

CPUS = 8
MEMORY_MB = 98_304
SOLVER_SECONDS = 43_200
KILL_GRACE_SECONDS = 300
RETENTION_SECONDS = 1_800
SCHEDULER_SECONDS = SOLVER_SECONDS + KILL_GRACE_SECONDS + RETENTION_SECONDS
MAX_WORKERS_PER_NODE = 1
PRIORITY = 100

TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official6-"
    "s96185-2772aed82a8c-n113"
)
WORKDIR = (
    "mft_goal_diag_standard_postdeadline_official6_"
    "s96185_2772aed82a8c_n113"
)
POST_AUTHORIZATION = "official6-dw16-n113"
ATTEMPT_LEDGER_NAME = "scheduler_post_attempt.json"
SUBMISSION_DIRECTORY_NAME = "submission"
ORIGINAL_DEADLINE_UTC = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)

PLAN_SCHEMA = "mft-goal-official-standard-postdeadline-plan-v1"
CANDIDATE_SCHEMA = (
    "mft-goal-official-standard-postdeadline-selected-candidate-v1"
)
PREPARE_SCHEMA = "mft-goal-official-standard-postdeadline-prepare-v1"
INTENT_SCHEMA = "mft-goal-official-standard-postdeadline-submit-intent-v1"
SUBMISSION_SCHEMA = "mft-goal-official-standard-postdeadline-submission-v1"
FINAL_SEAL_SCHEMA = "mft-goal-official-standard-postdeadline-final-seal-v1"

SAFETY_FLAGS = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "noncanonical": True,
    "production_eligible": False,
    "automatic_promotion": False,
    "scientific_pass_claimed": False,
    "original_deadline_missed": True,
}
FIXED_BOUNDARY = {
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
    "fan_velocity_m_s": 1.5,
    "fan_config": "dual",
    "core_plate_on": 1,
    "wcp_on": 1,
}
BOUNDARY_PROJECTION = {
    "k_ins": 0.2,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
    "fan_velocity": 1.5,
    "fan_config": "dual",
    "core_plate_on": 1,
    "wcp_on": 1,
}

JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]
PayloadBuilder = Callable[
    [dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]


class PostdeadlineContractError(RuntimeError):
    """Raised whenever immutable or live submission authority drifts."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise PostdeadlineContractError("payload is already sealed")
    result["payload_sha256"] = payload_sha256(result)
    return result


def validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PostdeadlineContractError(f"{schema} payload is not an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != payload_sha256(
        unsigned
    ):
        raise PostdeadlineContractError(f"{schema} payload seal mismatch")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    if not target.is_file() or target.is_symlink():
        raise PostdeadlineContractError(f"not a regular file: {target}")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def relative_record(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    return {
        "path": target.name,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostdeadlineContractError(
            f"JSON artifact is unavailable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PostdeadlineContractError(f"JSON object required: {path}")
    return value


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable output already exists: {target}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one durable file without an overwrite race."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o444,
        )
    except FileExistsError as exc:
        raise PostdeadlineContractError(
            f"exclusive immutable output already exists: {target}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(target)
        except FileNotFoundError:
            pass
        raise
    return target


class PortableFileLock:
    """Small FileLock-compatible OS lock used when the package is unavailable."""

    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        **_kwargs: Any,
    ) -> None:
        self.lock_file = str(lock_file)
        self.timeout = float(timeout)
        self._stream: Any = None

    def acquire(self, timeout: float | None = None, **_kwargs: Any) -> PortableFileLock:
        limit = self.timeout if timeout is None else float(timeout)
        started = time.monotonic()
        path = Path(self.lock_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    stream.write(b"\0")
                    stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                self._stream = stream
                return self
            except OSError:
                if limit == 0 or (
                    limit > 0 and time.monotonic() - started >= limit
                ):
                    stream.close()
                    raise TimeoutError(
                        f"timed out acquiring lock: {self.lock_file}"
                    )
                time.sleep(0.05)

    def release(self, **_kwargs: Any) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> PortableFileLock:
        return self.acquire()

    def __exit__(self, *_args: Any) -> None:
        self.release()


def _contained_artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise PostdeadlineContractError(f"{label} record is malformed")
    candidate = (root / str(record["path"])).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise PostdeadlineContractError(f"{label} escapes plan root") from exc
    if file_record(candidate) != {
        "path": str(candidate),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }:
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return candidate


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.resolve(strict=True).open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PostdeadlineContractError(f"CSV is unavailable: {path}") from exc


def _one_candidate(
    rows: Sequence[dict[str, str]], *, label: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("candidate_physics_sha") == CANDIDATE_SHA256
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            f"{label} must contain candidate exactly once"
        )
    return matches[0]


def _git_show(path: str, revision: str = SOLVER_REVISION) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY.as_posix()}",
            "-C",
            str(REPOSITORY),
            "show",
            f"{revision}:{path}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode or not result.stdout:
        raise PostdeadlineContractError(
            f"pinned solver source is unavailable: {revision}:{path}"
        )
    return result.stdout


def _literal_sequence(node: ast.AST, values: Mapping[str, list[str]]) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise PostdeadlineContractError("input key contract is not a sequence")
    output: list[str] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            output.append(element.value)
        elif (
            isinstance(element, ast.Starred)
            and isinstance(element.value, ast.Name)
            and element.value.id in values
        ):
            output.extend(values[element.value.id])
        else:
            raise PostdeadlineContractError(
                "input key contract contains unsupported syntax"
            )
    return output


def all_input_keys(
    source_reader: Callable[[str, str], bytes] = _git_show,
) -> tuple[list[str], dict[str, Any]]:
    source = source_reader(INPUT_PARAMETER_PATH, SOLVER_REVISION)
    try:
        tree = ast.parse(source.decode("utf-8"), filename=INPUT_PARAMETER_PATH)
    except (UnicodeError, SyntaxError) as exc:
        raise PostdeadlineContractError(
            "pinned input parameter contract cannot be parsed"
        ) from exc
    values: dict[str, list[str]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        names = [
            target.id
            for target in statement.targets
            if isinstance(target, ast.Name)
        ]
        if not names:
            continue
        try:
            value = _literal_sequence(statement.value, values)
        except PostdeadlineContractError:
            continue
        for name in names:
            values[name] = value
    keys = values.get("ALL_INPUT_KEYS")
    if not keys or len(keys) != len(set(keys)):
        raise PostdeadlineContractError("ALL_INPUT_KEYS contract is absent")
    return keys, {
        "path": INPUT_PARAMETER_PATH,
        "revision": SOLVER_REVISION,
        "sha256": hashlib.sha256(source).hexdigest(),
        "size_bytes": len(source),
    }


def authenticate_official_candidate(
    *,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    source_reader: Callable[[str, str], bytes] = _git_show,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(standard_candidates_path) != STANDARD_CANDIDATES_SHA256:
        raise PostdeadlineContractError("official Standard12 CSV bytes drifted")
    if sha256_file(aggregate_manifest_path) != AGGREGATE_MANIFEST_SHA256:
        raise PostdeadlineContractError("aggregate manifest bytes drifted")
    manifest = validate_seal(
        read_json(aggregate_manifest_path),
        "mft-goal-20260726-global-pareto-v1",
    )
    standard_record = manifest.get("artifacts", {}).get(
        "standard_candidates"
    )
    if (
        manifest.get("payload_sha256") != AGGREGATE_PAYLOAD_SHA256
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("seed_count") != 512
        or manifest.get("standard_candidate_count") != 12
        or manifest.get("production_eligible") is not False
        or manifest.get("automatic_promotion_allowed") is not False
        or manifest.get("search_only_proposal") is not True
        or not isinstance(standard_record, dict)
        or standard_record.get("path") != "standard_candidates.csv"
        or standard_record.get("row_count") != 12
        or standard_record.get("sha256") != STANDARD_CANDIDATES_SHA256
    ):
        raise PostdeadlineContractError(
            "official aggregate Standard12 authority drifted"
        )
    official_rows = _csv_rows(standard_candidates_path)
    if len(official_rows) != 12:
        raise PostdeadlineContractError("official Standard12 row count drifted")
    row = _one_candidate(official_rows, label="official Standard12")
    expected_row = {
        "source_seed": str(SOURCE_SEED),
        "source_task_id": str(SOURCE_TASK_ID),
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "standard_selection_order": "6",
        "standard_selection_roles": "objective_space_maximin",
        "standard_selection_basis": "near_feasible_fallback",
        "physical_feasible": "False",
        "hard_feasible": "False",
        "feasible_rank": "-1",
        "global_non_dominated_rank": "-1",
    }
    drift = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in expected_row.items()
        if row.get(key) != expected
    }
    if drift:
        raise PostdeadlineContractError(
            f"official candidate lineage/rank drifted: {drift}"
        )
    source_result_path = Path(str(row.get("source_result_path") or ""))
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise PostdeadlineContractError("source task result bytes drifted")
    source_result = read_json(source_result_path)
    unsigned_result = dict(source_result)
    source_result_payload = unsigned_result.pop("payload_sha256", None)
    if source_result_payload != payload_sha256(unsigned_result):
        raise PostdeadlineContractError("source task result seal drifted")
    terminal_manifest = source_result.get(
        "terminal_physical_candidates_manifest"
    )
    source_identity = (
        terminal_manifest.get("source_identity")
        if isinstance(terminal_manifest, dict)
        else None
    )
    if (
        source_result.get("campaign_id") != CAMPAIGN_ID
        or source_result.get("seed") != SOURCE_SEED
        or source_result.get("task_payload_sha256") != SOURCE_BUNDLE_ID
        or source_result.get("search_only_proposal") is not True
        or source_result.get("production_eligible") is not False
        or source_result.get("automatic_promotion_allowed") is not False
        or not isinstance(source_identity, dict)
        or source_identity.get("seed") != SOURCE_SEED
        or str(source_identity.get("task_id")) != str(SOURCE_TASK_ID)
        or source_identity.get("bundle_id") != SOURCE_BUNDLE_ID
    ):
        raise PostdeadlineContractError("source task result identity drifted")
    source_csv_record = source_result.get("artifact_inventory", {}).get(
        "terminal_physical_candidates"
    )
    if not isinstance(source_csv_record, dict):
        raise PostdeadlineContractError("source terminal CSV record is absent")
    source_csv_path = source_result_path.parent / str(
        source_csv_record.get("path") or ""
    )
    if (
        not source_csv_path.is_file()
        or sha256_file(source_csv_path) != source_csv_record.get("sha256")
    ):
        raise PostdeadlineContractError("source terminal CSV bytes drifted")
    source_row = _one_candidate(
        _csv_rows(source_csv_path), label="source task terminal CSV"
    )
    exact_columns = {
        "terminal_population_index",
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
        "objective_volume_L",
        "objective_total_loss_W",
        "physical_constraint_feasible",
        "physical_feasible",
        "source_seed",
        "source_task_id",
        "source_bundle_id",
        "source_island_id",
        "dataset_sha256",
        "evaluation_model_sha256",
        "constraint_spec_sha256",
        "cooling_contract_sha256",
        "operating_point_sha256",
        "evaluation_model_artifacts_sha256",
        "evaluation_model_generation_sha256",
        "evaluation_spec_sha256",
        "evaluation_temperature_contract_sha256",
        "evaluation_hard_constraint_contract_sha256",
    }
    if any(row.get(key) != source_row.get(key) for key in exact_columns):
        raise PostdeadlineContractError(
            "official row differs from authenticated source terminal row"
        )
    parsed_columns: dict[str, Any] = {}
    for column in (
        "decoded_physical_params_json",
        "physical_G_json",
        "normalized_G_json",
        "coordinate_unit_json",
    ):
        try:
            official_value = json.loads(str(row.get(column)))
            source_value = json.loads(str(source_row.get(column)))
        except json.JSONDecodeError as exc:
            raise PostdeadlineContractError(
                f"candidate {column} is malformed"
            ) from exc
        if official_value != source_value:
            raise PostdeadlineContractError(
                f"official candidate {column} differs from source"
            )
        parsed_columns[column] = official_value
    decoded = parsed_columns["decoded_physical_params_json"]
    projection = {
        key: decoded.get(key) for key in sorted(BOUNDARY_PROJECTION)
    }
    if projection != {
        key: BOUNDARY_PROJECTION[key] for key in sorted(BOUNDARY_PROJECTION)
    }:
        raise PostdeadlineContractError("candidate fixed cooling physics drifted")
    hard_cooling = source_result.get("hard_spec", {}).get(
        "fixed_cooling_identity"
    )
    if hard_cooling != {
        "core_k_thermal": 2.0,
        "core_plate_on": 1,
        "core_plate_pad_t": 2.0,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "thermal_pad_conductivity_W_mK": 0.2,
        "wcp_on": 1,
        "wcp_pad_t": 2.0,
    }:
        raise PostdeadlineContractError(
            "source hard-spec cooling identity drifted"
        )
    keys, key_contract = all_input_keys(source_reader)
    missing = [key for key in keys if key not in decoded]
    if missing:
        raise PostdeadlineContractError(
            f"candidate lacks solver input keys: {missing}"
        )
    params = {key: decoded[key] for key in keys}
    authentication = {
        "official_standard_candidates": file_record(standard_candidates_path),
        "aggregate_manifest": file_record(aggregate_manifest_path),
        "aggregate_manifest_payload_sha256": manifest["payload_sha256"],
        "aggregate_seed_count": 512,
        "aggregate_standard_candidate_count": 12,
        "candidate_physics_sha256": CANDIDATE_SHA256,
        "candidate_standard_selection_order": 6,
        "candidate_standard_selection_role": "objective_space_maximin",
        "source_seed": SOURCE_SEED,
        "source_task_id": SOURCE_TASK_ID,
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "source_result": file_record(source_result_path),
        "source_result_payload_sha256": source_result_payload,
        "source_terminal_candidates": file_record(source_csv_path),
        "source_terminal_population_index": int(
            row["terminal_population_index"]
        ),
        "input_parameter_contract": key_contract,
        "candidate_boundary_projection": projection,
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "candidate_reauthenticated": True,
        "source_reauthenticated": True,
        "fixed_physics_unchanged": True,
    }
    selected = {
        "schema_version": CANDIDATE_SCHEMA,
        **SAFETY_FLAGS,
        "campaign_id": CAMPAIGN_ID,
        "authentication": authentication,
        "official_row": row,
        "source_row": source_row,
        "decoded_physical_params": decoded,
        "fea_params_sha256": payload_sha256(params),
    }
    return sealed(selected), params


def reviewed_profile() -> dict[str, Any]:
    if sha256_file(PROFILE_PATH) != PROFILE_SHA256:
        raise PostdeadlineContractError("reviewed 12-hour profile bytes drifted")
    profile = read_json(PROFILE_PATH)
    if (
        profile.get("schema_version")
        != "mft-goal-diagnostic-standard-timeout12h-retry-profile-v1"
        or profile.get("stage") != "standard"
        or profile.get("cpus") != CPUS
        or profile.get("timeout_seconds") != SOLVER_SECONDS
        or profile.get("fixed_boundary_contract") != FIXED_BOUNDARY
        or profile.get("param_overrides", {}).get("full_model") != 0
        or profile.get("param_overrides", {}).get("thermal_symmetry")
        != "eighth"
        or profile.get("artifact_retention", {}).get("artifact_filename")
        != "symmetric.aedt"
        or profile.get("artifact_retention", {}).get("results_directory")
        != "symmetric.aedtresults"
        or profile.get("artifact_retention", {}).get(
            "prune_protection_required"
        )
        is not True
    ):
        raise PostdeadlineContractError("reviewed Standard profile drifted")
    return profile


def _capture_scheduler_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        url: str,
        json: dict[str, Any] | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> Response:
        captured.update(
            {"url": url, "payload": copy.deepcopy(json), "timeout": timeout}
        )
        return Response()

    requests_module = sys.modules.get("requests")
    if requests_module is None:
        requests_module = types.ModuleType("requests")
        sys.modules["requests"] = requests_module
    filelock_module = sys.modules.get("filelock")
    if filelock_module is None:
        filelock_module = types.ModuleType("filelock")
        filelock_module.FileLock = PortableFileLock
        sys.modules["filelock"] = filelock_module
    scheduler = importlib.import_module(
        "regression_260707.verify.scheduler_client"
    )
    old_post = getattr(requests_module, "post", None)
    old_get = getattr(requests_module, "get", None)
    old_reconcile = scheduler.reconcile_task_record
    old_snapshot = scheduler.live_project_submission_snapshot
    depth_was_set = hasattr(scheduler._CAMPAIGN_LOCK_STATE, "depth")
    old_depth = getattr(scheduler._CAMPAIGN_LOCK_STATE, "depth", 0)
    requests_module.post = fake_post
    requests_module.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        PostdeadlineContractError(
            "unexpected GET while deriving Scheduler payload"
        )
    )
    scheduler.reconcile_task_record = lambda *_args, **_kwargs: None
    scheduler.live_project_submission_snapshot = (
        lambda *_args, **_kwargs: {"project_submission_slots": 1}
    )
    scheduler._CAMPAIGN_LOCK_STATE.depth = 1
    environment = {
        "MFT_STANDALONE_CORE_CONTRACT": "mft-standalone-core-optin-v1",
        "MFT_STANDALONE_CORE_COUNT": str(CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": CORE_AUTH_SHA256,
    }
    try:
        scheduler.submit_verification(
            TASK_NAME,
            WORKDIR,
            params,
            profile,
            mem_mb=MEMORY_MB,
            cpus=CPUS,
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            priority=PRIORITY,
            account_name=ACCOUNT_NAME,
            node_name=NODE_NAME,
            max_workers_per_node=MAX_WORKERS_PER_NODE,
            aedt_backend="standalone",
            submission_env=environment,
            required_hard_cap=500,
            max_project_active_tasks=500,
            scheduler_url=SCHEDULER_URL,
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        retained = scheduler.retained_aedt_identity(
            TASK_NAME,
            params,
            profile,
            SOLVER_REVISION,
            LIBRARY_REVISION,
        )
    finally:
        if depth_was_set:
            scheduler._CAMPAIGN_LOCK_STATE.depth = old_depth
        else:
            del scheduler._CAMPAIGN_LOCK_STATE.depth
        scheduler.reconcile_task_record = old_reconcile
        scheduler.live_project_submission_snapshot = old_snapshot
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
        raise PostdeadlineContractError(
            "Scheduler client did not derive retained submission payload"
        )
    command = str(payload.get("command") or "")
    random_delay = "sleep $((RANDOM % 300)); "
    solver = (
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json; simulation_rc=$?;"
    )
    bounded_solver = (
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json; simulation_rc=$?;"
    )
    if command.count(random_delay) != 1 or command.count(solver) != 1:
        raise PostdeadlineContractError(
            "reviewed Scheduler command template drifted"
        )
    payload["command"] = command.replace(random_delay, "", 1).replace(
        solver, bounded_solver, 1
    )
    payload["timeout_seconds"] = SCHEDULER_SECONDS
    return payload, environment, retained


def validate_scheduler_payload(
    payload: Mapping[str, Any], retained: Mapping[str, Any]
) -> None:
    expected = {
        "name": TASK_NAME,
        "project": PROJECT,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": 0,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "priority": PRIORITY,
        "timeout_seconds": SCHEDULER_SECONDS,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    command = str(payload.get("command") or "")
    if (
        drift
        or "same_node_as_task_id" in payload
        or payload.get("dedupe_key") != retained.get("dedupe_key")
        or command.count(
            "timeout --signal=TERM --kill-after=300s 43200s "
            "python run_simulation_260706.py"
        )
        != 1
        or "sleep $((RANDOM % 300))" in command
        or retained.get("artifact_path", "").endswith("/symmetric.aedt")
        is not True
        or retained.get("results_path", "").endswith(
            "/symmetric.aedtresults"
        )
        is not True
        or retained.get("transport", {}).get("chunk_directory", "").endswith(
            "/symmetric.aedt.chunks"
        )
        is not True
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
    ):
        raise PostdeadlineContractError(
            f"derived Scheduler payload/retention drifted: {drift}"
        )


def scheduler_campaign_lock() -> Any:
    scheduler = importlib.import_module(
        "regression_260707.verify.scheduler_client"
    )
    return scheduler.campaign_mutation_lock()


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
        raise PostdeadlineContractError(f"Scheduler GET failed: {url}") from exc


def capacity_query() -> list[tuple[str, Any]]:
    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", ACCOUNT_NAME),
        ("node_name", NODE_NAME),
    ]


def _task_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("tasks"), list):
        return [
            item for item in value["tasks"] if isinstance(item, dict)
        ]
    raise PostdeadlineContractError("Scheduler task inventory is malformed")


def live_preflight(
    *,
    reader: JsonReader = get_json,
    expected_dedupe_key: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError("preflight time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if now <= ORIGINAL_DEADLINE_UTC:
        raise PostdeadlineContractError(
            "post-deadline probe cannot precede original deadline"
        )
    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{SOURCE_TASK_ID}", None)
    capacity = reader("/api/task-capacity", capacity_query())
    allocations = reader("/api/allocations", None)
    active = reader(
        "/api/tasks",
        [
            ("status", "queued"),
            ("status", "attaching"),
            ("status", "running"),
            ("limit", 10000),
        ],
    )
    collision_inventory = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    if (
        not isinstance(health, dict)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")
    expected_source = {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "status": "completed",
        "exit_code": 0,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
    }
    source_drift = {
        key: {"expected": value, "actual": source_task.get(key)}
        for key, value in expected_source.items()
        if not isinstance(source_task, dict)
        or source_task.get(key) != value
    }
    if source_drift:
        raise PostdeadlineContractError(
            f"source Scheduler task identity drifted: {source_drift}"
        )
    if (
        not isinstance(capacity, dict)
        or capacity.get("queue_state") != "ready"
        or int(capacity.get("ready_fit_slots") or 0) < 1
        or capacity.get("memory_pressure_state") != "ok"
        or int(capacity.get("standalone_aedt_available") or 0) < 1
    ):
        raise PostdeadlineContractError(
            "exact dw16/n113 task-capacity gate is not ready"
        )
    capacity_allocations = [
        row
        for row in capacity.get("allocations", [])
        if isinstance(row, dict)
        and row.get("account_name") == ACCOUNT_NAME
        and row.get("node_name") == NODE_NAME
        and row.get("state") == "active"
        and int(row.get("fit_slots") or 0) >= 1
        and int(row.get("free_cpus") or 0) >= CPUS
        and int(row.get("free_memory_mb") or 0) >= MEMORY_MB
    ]
    if not capacity_allocations:
        raise PostdeadlineContractError(
            "exact dw16/n113 allocation capacity is absent"
        )
    capacity_allocation = max(
        capacity_allocations,
        key=lambda row: (
            int(row.get("fit_slots") or 0),
            int(row.get("allocation_id") or 0),
        ),
    )
    allocation_id = int(capacity_allocation.get("allocation_id") or 0)
    allocation_rows = (
        allocations
        if isinstance(allocations, list)
        else allocations.get("allocations", [])
        if isinstance(allocations, dict)
        else []
    )
    exact_allocations = [
        row
        for row in allocation_rows
        if isinstance(row, dict) and int(row.get("id") or 0) == allocation_id
    ]
    if len(exact_allocations) != 1:
        raise PostdeadlineContractError(
            "target allocation identity is absent or ambiguous"
        )
    allocation = exact_allocations[0]
    if (
        allocation.get("account_name") != ACCOUNT_NAME
        or allocation.get("node_name") != NODE_NAME
        or allocation.get("state") != "active"
        or not str(allocation.get("slurm_job_id") or "").isdigit()
        or int(allocation.get("free_cpus") or 0) < CPUS
        or int(allocation.get("free_memory_mb") or 0) < MEMORY_MB
        or int(allocation.get("node_fea_requested_cpus") or 0) != 0
    ):
        raise PostdeadlineContractError(
            "target allocation resource/identity gate failed"
        )
    active_rows = _task_rows(active)
    active_fea = [
        row
        for row in active_rows
        if (
            int(row.get("assigned_allocation") or row.get("allocation_id") or 0)
            == allocation_id
            or row.get("node_name") == NODE_NAME
        )
        and (
            row.get("aedt_backend") == "standalone"
            or row.get("scheduling_profile") == "fea_bursty"
        )
    ]
    if active_fea:
        raise PostdeadlineContractError(
            "target n113 lane is no longer FEA-empty"
        )
    collisions = [
        row
        for row in _task_rows(collision_inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collisions:
        raise PostdeadlineContractError(
            "fresh official #6 task name or dedupe already exists"
        )
    return {
        "schema_version": "mft-goal-official-standard-live-preflight-v1",
        "observed_at_utc": now.isoformat(),
        "scheduler_health": health,
        "source_task_identity": source_task,
        "capacity_query": capacity_query(),
        "capacity": capacity,
        "selected_allocation": allocation,
        "selected_allocation_id": allocation_id,
        "selected_slurm_job_id": str(allocation["slurm_job_id"]),
        "active_fea_on_target_count": 0,
        "collision_count": 0,
        "exact_account_node_ready": True,
        "fixed_physics_unchanged": True,
    }


def prepare(
    *,
    output: Path,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    reader: JsonReader = get_json,
    payload_builder: PayloadBuilder = _capture_scheduler_payload,
    source_reader: Callable[[str, str], bytes] = _git_show,
    observed_at: datetime | None = None,
) -> Path:
    target = output.resolve()
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable prepare output already exists: {target}"
        )
    selected, params = authenticate_official_candidate(
        standard_candidates_path=standard_candidates_path,
        aggregate_manifest_path=aggregate_manifest_path,
        source_reader=source_reader,
    )
    profile = reviewed_profile()
    payload, environment, retained = payload_builder(params, profile)
    validate_scheduler_payload(payload, retained)
    preflight = live_preflight(
        reader=reader,
        expected_dedupe_key=str(payload["dedupe_key"]),
        observed_at=observed_at,
    )
    now = (
        observed_at.astimezone(timezone.utc)
        if observed_at is not None
        else datetime.now(timezone.utc)
    )
    attempt_nonce = payload_sha256(
        {
            "schema_version": (
                "mft-goal-official-standard-postdeadline-attempt-nonce-v1"
            ),
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": payload["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "scheduler_payload_sha256": payload_sha256(payload),
        }
    )
    attempt_path = target / ATTEMPT_LEDGER_NAME
    submission_path = target / SUBMISSION_DIRECTORY_NAME
    staging = target.with_name(
        f".{target.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_path = write_immutable_json(
            staging / "selected_candidate.json", selected
        )
        params_path = write_immutable_json(staging / "fea_params.json", params)
        profile_path = write_immutable_json(
            staging / "execution_profile.json", profile
        )
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "campaign_id": CAMPAIGN_ID,
                "standard_only": True,
                "symmetric_model": True,
                "full_model": False,
                "thermal_symmetry": "eighth",
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "source_seed": SOURCE_SEED,
                "source_task_id": SOURCE_TASK_ID,
                "official_standard_selection_order": 6,
                "selected_candidate": relative_record(selected_path),
                "selected_candidate_payload_sha256": selected[
                    "payload_sha256"
                ],
                "fea_params": relative_record(params_path),
                "fea_params_sha256": payload_sha256(params),
                "execution_profile": relative_record(profile_path),
                "execution_profile_canonical_sha256": payload_sha256(
                    profile
                ),
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "task_name": TASK_NAME,
                "workdir": WORKDIR,
                "dedupe_key": payload["dedupe_key"],
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "solver_seconds": SOLVER_SECONDS,
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                    "retention_seconds": RETENTION_SECONDS,
                    "scheduler_timeout_seconds": SCHEDULER_SECONDS,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                },
                "placement": {
                    "account_name": ACCOUNT_NAME,
                    "node_name": NODE_NAME,
                    "node_name_policy": "strict",
                    "same_node_as_task_id": 0,
                    "dependency_task_id": 0,
                    "allocation_id_at_prepare": preflight[
                        "selected_allocation_id"
                    ],
                    "slurm_job_id_at_prepare": preflight[
                        "selected_slurm_job_id"
                    ],
                },
                "scheduler_url": SCHEDULER_URL,
                "scheduler_project": PROJECT,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "fixed_physics_unchanged": True,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(environment),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "fresh_retention_identity": True,
                "fresh_chunk_identity": True,
                "preflight": preflight,
                "submit_authorization_token": POST_AUTHORIZATION,
                "single_attempt_contract": {
                    "schema_version": (
                        "mft-goal-official-standard-postdeadline-"
                        "single-attempt-contract-v1"
                    ),
                    "attempt_ledger_path": str(attempt_path),
                    "submission_output_path": str(submission_path),
                    "attempt_nonce": attempt_nonce,
                    "post_call_budget": 1,
                    "ledger_consumed_before_network": True,
                    "output_override_allowed": False,
                },
            }
        )
        plan_path = write_immutable_json(
            staging / "diagnostic_postdeadline_plan.json", plan
        )
        final_plan_path = target / plan_path.name
        submit_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "submit",
            "--plan",
            str(final_plan_path),
            "--output",
            str(submission_path),
            "--authorize-post",
            POST_AUTHORIZATION,
        ]
        prepare_receipt = sealed(
            {
                "schema_version": PREPARE_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "plan": relative_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "task_name": TASK_NAME,
                "dedupe_key": payload["dedupe_key"],
                "account_name": ACCOUNT_NAME,
                "node_name": NODE_NAME,
                "selected_allocation_id": preflight[
                    "selected_allocation_id"
                ],
                "selected_slurm_job_id": preflight[
                    "selected_slurm_job_id"
                ],
                "submit_command_argv": submit_command,
                "scheduler_get_preflight_performed": True,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "attempt_ledger_path": str(attempt_path),
                "attempt_nonce": attempt_nonce,
                "ready_for_explicit_submit": True,
            }
        )
        write_immutable_json(staging / "prepare_receipt.json", prepare_receipt)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "diagnostic_postdeadline_plan.json"


def load_plan(
    plan_path: Path,
    *,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    payload_builder: PayloadBuilder = _capture_scheduler_payload,
    source_reader: Callable[[str, str], bytes] = _git_show,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = plan_path.resolve(strict=True)
    plan = validate_seal(read_json(resolved), PLAN_SCHEMA)
    single_attempt = plan.get("single_attempt_contract")
    expected_attempt_path = resolved.parent / ATTEMPT_LEDGER_NAME
    expected_submission_path = resolved.parent / SUBMISSION_DIRECTORY_NAME
    expected_attempt_nonce = payload_sha256(
        {
            "schema_version": (
                "mft-goal-official-standard-postdeadline-attempt-nonce-v1"
            ),
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan.get("dedupe_key"),
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "scheduler_payload_sha256": plan.get(
                "scheduler_payload_sha256"
            ),
        }
    )
    if (
        any(plan.get(key) is not value for key, value in SAFETY_FLAGS.items())
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("candidate_physics_sha256") != CANDIDATE_SHA256
        or plan.get("source_seed") != SOURCE_SEED
        or plan.get("source_task_id") != SOURCE_TASK_ID
        or plan.get("task_name") != TASK_NAME
        or plan.get("workdir") != WORKDIR
        or plan.get("scheduler_url") != SCHEDULER_URL
        or plan.get("scheduler_project") != PROJECT
        or plan.get("placement", {}).get("account_name") != ACCOUNT_NAME
        or plan.get("placement", {}).get("node_name") != NODE_NAME
        or plan.get("placement", {}).get("node_name_policy") != "strict"
        or plan.get("resources", {}).get("cpus") != CPUS
        or plan.get("resources", {}).get("memory_mb") != MEMORY_MB
        or plan.get("resources", {}).get("scheduler_timeout_seconds")
        != SCHEDULER_SECONDS
        or plan.get("resources", {}).get("max_workers_per_node")
        != MAX_WORKERS_PER_NODE
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or not isinstance(single_attempt, dict)
        or single_attempt.get("schema_version")
        != (
            "mft-goal-official-standard-postdeadline-"
            "single-attempt-contract-v1"
        )
        or Path(
            str(single_attempt.get("attempt_ledger_path") or "")
        ).resolve()
        != expected_attempt_path
        or Path(
            str(single_attempt.get("submission_output_path") or "")
        ).resolve()
        != expected_submission_path
        or single_attempt.get("attempt_nonce") != expected_attempt_nonce
        or single_attempt.get("post_call_budget") != 1
        or single_attempt.get("ledger_consumed_before_network") is not True
        or single_attempt.get("output_override_allowed") is not False
    ):
        raise PostdeadlineContractError("sealed plan contract drifted")
    root = resolved.parent
    selected_path = _contained_artifact(
        root, plan.get("selected_candidate"), "selected candidate"
    )
    params_path = _contained_artifact(
        root, plan.get("fea_params"), "FEA params"
    )
    profile_path = _contained_artifact(
        root, plan.get("execution_profile"), "execution profile"
    )
    selected = validate_seal(read_json(selected_path), CANDIDATE_SCHEMA)
    params = read_json(params_path)
    profile = read_json(profile_path)
    fresh_selected, fresh_params = authenticate_official_candidate(
        standard_candidates_path=standard_candidates_path,
        aggregate_manifest_path=aggregate_manifest_path,
        source_reader=source_reader,
    )
    reviewed = reviewed_profile()
    fresh_payload, environment, retained = payload_builder(
        fresh_params, reviewed
    )
    validate_scheduler_payload(fresh_payload, retained)
    if (
        selected != fresh_selected
        or params != fresh_params
        or profile != reviewed
        or plan.get("fea_params_sha256") != payload_sha256(params)
        or plan.get("execution_profile_canonical_sha256")
        != payload_sha256(profile)
        or plan.get("scheduler_payload") != fresh_payload
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(fresh_payload)
        or plan.get("dedupe_key") != fresh_payload["dedupe_key"]
        or plan.get("submission_environment") != environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(environment)
        or plan.get("retained_aedt_bundle") != retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(retained)
    ):
        raise PostdeadlineContractError(
            "sealed plan no longer matches reauthenticated inputs"
        )
    return plan, params, profile


def _post_json_once(
    url: str, payload: Mapping[str, Any]
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    request = Request(
        url,
        data=canonical_bytes(payload),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
            return response.status, raw if isinstance(raw, dict) else None, None
    except HTTPError as exc:
        return (
            exc.code,
            None,
            exc.read().decode("utf-8", errors="replace"),
        )
    except OSError as exc:
        return None, None, str(exc)


def submit(
    *,
    plan_path: Path,
    output: Path,
    authorize_post: str,
    reader: JsonReader = get_json,
    poster: Callable[
        [str, Mapping[str, Any]],
        tuple[int | None, dict[str, Any] | None, str | None],
    ] = _post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    plan_loader: Callable[
        [Path],
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = load_plan,
) -> Path:
    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit official6 dw16/n113 POST authorization is absent"
        )
    plan, _params, _profile = plan_loader(plan_path)
    payload = plan["scheduler_payload"]
    single_attempt = plan["single_attempt_contract"]
    target = output.resolve()
    expected_target = Path(
        single_attempt["submission_output_path"]
    ).resolve()
    attempt_path = Path(
        single_attempt["attempt_ledger_path"]
    ).resolve()
    if target != expected_target:
        raise PostdeadlineContractError(
            "submission output differs from the sealed single-attempt path"
        )
    if attempt_path.exists():
        raise PostdeadlineContractError(
            "sealed Scheduler POST attempt was already consumed"
        )
    initial_preflight = live_preflight(
        reader=reader,
        expected_dedupe_key=plan["dedupe_key"],
        observed_at=observed_at,
    )
    target.mkdir(parents=True, exist_ok=False)
    intent = sealed(
        {
            "schema_version": INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "scheduler_payload_sha256": plan[
                "scheduler_payload_sha256"
            ],
            "initial_live_preflight": initial_preflight,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "authorization": POST_AUTHORIZATION,
            "attempt_ledger_path": str(attempt_path),
            "attempt_nonce": single_attempt["attempt_nonce"],
            "scheduler_post_calls_before": 0,
            "post_authorized": True,
        }
    )
    intent_path = write_immutable_json(target / "submission_intent.json", intent)
    with lock_factory():
        locked_preflight = live_preflight(
            reader=reader,
            expected_dedupe_key=plan["dedupe_key"],
            observed_at=observed_at,
        )
        attempt = sealed(
            {
                "schema_version": (
                    "mft-goal-official-standard-postdeadline-"
                    "scheduler-post-attempt-v1"
                ),
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan": file_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "intent": file_record(intent_path),
                "intent_payload_sha256": intent["payload_sha256"],
                "attempt_nonce": single_attempt["attempt_nonce"],
                "submission_output_path": str(target),
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "campaign_mutation_lock_acquired": True,
                "locked_pre_submit_live_preflight": locked_preflight,
                "post_call_budget": 1,
                "post_call_consumed_before_network": True,
                "scheduler_post_calls_before": 0,
                "scheduler_post_calls_authorized": 1,
            }
        )
        attempt_path = write_exclusive_json(attempt_path, attempt)
        status, response, error = poster(
            f"{SCHEDULER_URL}/api/tasks", payload
        )
    task_id = (
        response.get("task_id") or response.get("id")
        if isinstance(response, dict)
        else None
    )
    if task_id is None:
        inventory = reader(
            "/api/tasks",
            [
                ("limit", 10000),
                ("project", PROJECT),
                ("name_prefix", TASK_NAME),
            ],
        )
        exact = [
            row
            for row in _task_rows(inventory)
            if row.get("name") == TASK_NAME
            and row.get("dedupe_key") == plan["dedupe_key"]
        ]
        if len(exact) == 1:
            task_id = exact[0].get("task_id") or exact[0].get("id")
        else:
            failure = sealed(
                {
                    "schema_version": (
                        "mft-goal-official-standard-postdeadline-"
                        "post-failure-v1"
                    ),
                    **SAFETY_FLAGS,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "plan": file_record(plan_path),
                    "intent": file_record(intent_path),
                    "attempt_ledger": file_record(attempt_path),
                    "attempt_ledger_payload_sha256": attempt[
                        "payload_sha256"
                    ],
                    "campaign_mutation_lock_acquired": True,
                    "locked_pre_submit_live_preflight": locked_preflight,
                    "http_status": status,
                    "http_error": error,
                    "exact_reconciliation_count": len(exact),
                    "scheduler_post_calls": 1,
                }
            )
            write_immutable_json(target / "post_failure.json", failure)
            raise PostdeadlineContractError(
                "Scheduler POST failed without exact reconciliation"
            )
    task_id = int(task_id)
    readback = reader(f"/api/tasks/{task_id}", None)
    expected_readback = {
        "name": TASK_NAME,
        "dedupe_key": plan["dedupe_key"],
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": value, "actual": readback.get(key)}
        for key, value in expected_readback.items()
        if not isinstance(readback, dict) or readback.get(key) != value
    }
    if (
        drift
        or readback.get("status") not in {"queued", "attaching", "running"}
        or (
            readback.get("account_name")
            and readback.get("account_name") != ACCOUNT_NAME
        )
        or (
            readback.get("node_name")
            and readback.get("node_name") != NODE_NAME
        )
    ):
        raise PostdeadlineContractError(
            f"durable Scheduler readback drifted: {drift}"
        )
    receipt = sealed(
        {
            "schema_version": SUBMISSION_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "intent": file_record(intent_path),
            "intent_payload_sha256": intent["payload_sha256"],
            "attempt_ledger": file_record(attempt_path),
            "attempt_ledger_payload_sha256": attempt[
                "payload_sha256"
            ],
            "initial_live_preflight": initial_preflight,
            "locked_pre_submit_live_preflight": locked_preflight,
            "campaign_mutation_lock_acquired": True,
            "pre_submit_get_repeated_inside_lock": True,
            "scheduler_payload_sha256": plan[
                "scheduler_payload_sha256"
            ],
            "scheduler_post_calls": 1,
            "scheduler_submission_performed": status == 201,
            "scheduler_post_http_status": status,
            "scheduler_post_response": response,
            "scheduler_post_error": error,
            "task_id": task_id,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "task_readback": readback,
            "task_readback_sha256": payload_sha256(readback),
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
        }
    )
    receipt_path = write_immutable_json(
        target / "submission_receipt.json", receipt
    )
    final_seal = sealed(
        {
            "schema_version": FINAL_SEAL_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "intent": file_record(intent_path),
            "intent_payload_sha256": intent["payload_sha256"],
            "attempt_ledger": file_record(attempt_path),
            "attempt_ledger_payload_sha256": attempt[
                "payload_sha256"
            ],
            "receipt": file_record(receipt_path),
            "receipt_payload_sha256": receipt["payload_sha256"],
            "task_id": task_id,
            "scheduler_post_calls": 1,
            "immutable_evidence_complete": True,
        }
    )
    return write_immutable_json(target / "final_seal.json", final_seal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sealed official Standard12 candidate #6 post-deadline probe"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="authenticate, GET-preflight, and seal without POST"
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    inspect_parser = commands.add_parser(
        "inspect", help="reauthenticate a plan and repeat GET preflight"
    )
    inspect_parser.add_argument("--plan", type=Path, required=True)
    submit_parser = commands.add_parser(
        "submit", help="reauthenticate and perform at most one Scheduler POST"
    )
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--output", type=Path, required=True)
    submit_parser.add_argument("--authorize-post", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        plan_path = prepare(output=args.output)
        plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
        receipt = read_json(plan_path.parent / "prepare_receipt.json")
        result = {
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "allocation_id": plan["placement"][
                "allocation_id_at_prepare"
            ],
            "slurm_job_id": plan["placement"]["slurm_job_id_at_prepare"],
            "scheduler_post_calls": 0,
            "submit_command_argv": receipt["submit_command_argv"],
        }
    elif args.command == "inspect":
        plan, _params, _profile = load_plan(args.plan)
        preflight = live_preflight(
            expected_dedupe_key=plan["dedupe_key"]
        )
        result = {
            "plan": str(args.plan.resolve()),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "allocation_id": preflight["selected_allocation_id"],
            "slurm_job_id": preflight["selected_slurm_job_id"],
            "scheduler_post_calls": 0,
            "ready_for_explicit_submit": True,
        }
    else:
        seal_path = submit(
            plan_path=args.plan,
            output=args.output,
            authorize_post=args.authorize_post,
        )
        final_seal = validate_seal(
            read_json(seal_path), FINAL_SEAL_SCHEMA
        )
        result = {
            "final_seal": str(seal_path),
            "final_seal_sha256": sha256_file(seal_path),
            "task_id": final_seal["task_id"],
            "scheduler_post_calls": final_seal[
                "scheduler_post_calls"
            ],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
