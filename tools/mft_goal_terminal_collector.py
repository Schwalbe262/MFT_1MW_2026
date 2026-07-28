#!/usr/bin/env python3
"""GET-only terminal watcher and diagnostic artifact collector.

Two already-submitted tasks are supported:

* ``full96326``: the sealed post-deadline diagnostic Full retry; and
* ``thermal96324``: the sealed corrected-thermal continuation.

The watcher has no POST/cancel/priority surface.  Every poll revalidates the
exact Scheduler task identity, strict node, task.sh, extracted command, local
plan/submission evidence, and fixed-physics boundary before looking at state.
Failed/timed-out tasks produce ledger evidence only.  Successful tasks trigger
GET-only collection, but never a scientific PASS or production claim.

Large binary files are collected only through an authenticated transport
contract.  Full AEDT chunks satisfy that rule and are retained plus
reassembled.  Missing ``.aedtresults`` or an unchunked thermal AEDT yields the
explicit terminal state ``success_pending_collection``.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib import error, parse, request


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_INTERVAL_SECONDS = 60
MAX_REMOTE_FILE_BYTES = 1_048_576
MAX_TASK_OUTPUT_BYTES = 16 * 1024 * 1024
FULL_CANDIDATE_SHA256 = (
    "62f2b846d05bbc5880165c6086fe9897e04d0588b0b8767fceb35102e4be9a44"
)
FULL_CANDIDATE_SIZE = 1532
THERMAL_FIXED_PHYSICS = {
    "fan_velocity_m_per_s": 1.5,
    "thermal_pad_thickness_mm": 2.0,
    "tim_conductivity_w_per_mk": 0.2,
}
CLASSIFICATION = {
    "diagnostic_only": True,
    "canonical": False,
    "production_truth_eligible": False,
    "scientific_pass_claimed": False,
    "production_claimed": False,
}
TERMINAL_FAILURE_STATUSES = {
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "timeout",
}


class CollectorError(RuntimeError):
    """Fail-closed local, Scheduler, or retained-artifact contract error."""


class TransientGetError(CollectorError):
    """A GET failed without proving contract drift."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return data + (b"\n" if newline else b"")


def _now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectorError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CollectorError(f"sealed JSON root is not an object: {path}")
    return value


def _verify_file(path: Path, expected_sha256: str) -> bytes:
    try:
        data = path.resolve(strict=True).read_bytes()
    except OSError as exc:
        raise CollectorError(f"sealed local file is unavailable: {path}") from exc
    observed = _sha256(data)
    if observed != expected_sha256:
        raise CollectorError(
            f"sealed local file drifted: {path}: {observed} != {expected_sha256}"
        )
    return data


def _verify_tracked_file(
    repo_root: Path,
    path: Path,
    expected_sha256: str,
) -> bytes:
    """Read one exact sealed blob despite a clean CRLF checkout projection."""

    try:
        return _verify_file(path, expected_sha256)
    except CollectorError as checkout_error:
        try:
            root = repo_root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise checkout_error from error
        if path.is_symlink():
            raise checkout_error
        git_prefix = [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
        ]
        clean = subprocess.run(
            [*git_prefix, "diff", "--quiet", "HEAD", "--", relative],
            check=False,
            capture_output=True,
        )
        if clean.returncode != 0:
            raise checkout_error
        blob = subprocess.run(
            [*git_prefix, "show", f"HEAD:{relative}"],
            check=False,
            capture_output=True,
        )
        if blob.returncode != 0 or _sha256(blob.stdout) != expected_sha256:
            raise checkout_error
        return blob.stdout


def _verify_embedded_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str = "payload_sha256",
) -> None:
    claimed = value.get(digest_field)
    if not isinstance(claimed, str):
        raise CollectorError(f"embedded digest is absent: {digest_field}")
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    observed = set()
    for ensure_ascii in (False, True):
        canonical = json.dumps(
            unsigned,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        observed.add(_sha256(canonical))
        observed.add(_sha256(canonical + b"\n"))
    if claimed not in observed:
        raise CollectorError(f"embedded digest drifted: {digest_field}")


def _safe_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
    ):
        raise CollectorError(f"unsafe remote relative path: {path!r}")
    return pure.as_posix()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CollectorError(f"exclusive watcher file already exists: {path}") from exc


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical_bytes(value) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


class GetOnlyClient:
    """Scheduler client whose implementation can issue GET requests only."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.get_count = 0

    def get_bytes(self, endpoint: str) -> bytes:
        self.get_count += 1
        req = request.Request(
            self.base_url + endpoint,
            method="GET",
            headers={
                "Accept": "*/*",
                "User-Agent": "mft-terminal-collector-get-only/1",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                if int(response.status) != 200:
                    raise TransientGetError(
                        f"GET {endpoint} returned HTTP {response.status}"
                    )
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TransientGetError(
                f"GET {endpoint} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise TransientGetError(f"GET {endpoint} failed: {exc.reason}") from exc

    def get_json(self, endpoint: str) -> Any:
        try:
            return json.loads(self.get_bytes(endpoint))
        except json.JSONDecodeError as exc:
            raise TransientGetError(f"GET {endpoint} returned invalid JSON") from exc

    def remote_file(
        self,
        task_id: int,
        path: str,
        *,
        base: str = "remote_cwd",
        max_bytes: int = MAX_REMOTE_FILE_BYTES,
    ) -> bytes:
        query = parse.urlencode(
            {
                "base": base,
                "path": _safe_relative(path),
                "max_bytes": max_bytes,
            }
        )
        return self.get_bytes(f"/api/tasks/{task_id}/remote-file?{query}")

    def remote_files(
        self,
        task_id: int,
        glob: str,
        *,
        base: str = "remote_cwd",
    ) -> list[str]:
        query = parse.urlencode({"base": base, "glob": _safe_relative(glob)})
        value = self.get_json(f"/api/tasks/{task_id}/remote-files?{query}")
        if not isinstance(value, dict) or not isinstance(value.get("files"), list):
            raise TransientGetError("remote-files response shape drifted")
        files = [str(item) for item in value["files"]]
        for path in files:
            _safe_relative(path)
        return sorted(files)

    def task_output(
        self,
        task_id: int,
        stream: str,
        *,
        max_bytes: int = MAX_REMOTE_FILE_BYTES,
    ) -> bytes:
        if stream not in {"stdout", "stderr"}:
            raise CollectorError(f"unknown task output stream: {stream}")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 0 < max_bytes <= MAX_TASK_OUTPUT_BYTES
        ):
            raise CollectorError("task output byte bound is invalid")
        query = parse.urlencode({"max_bytes": max_bytes})
        return self.get_bytes(f"/api/tasks/{task_id}/{stream}?{query}")


@dataclass(frozen=True)
class Contract:
    key: str
    kind: str
    task_id: int
    name: str
    dedupe_key: str
    account: str
    node: str
    cpus: int
    memory_mb: int
    timeout_seconds: int
    task_sh_size: int
    task_sh_sha256: str
    command_sentinel: bytes
    command_size: int
    command_sha256: str
    command: bytes
    local_evidence: dict[str, dict[str, Any]]
    retained_root: str
    retained_destination: str
    plan: dict[str, Any]
    submission: dict[str, Any]


def _full_contract(repo_root: Path) -> Contract:
    paths = {
        "plan": (
            repo_root / "artifacts/mft_goal_postdeadline_full_retry_plan_v1.json",
            "bd91a3549e75308782b4ad7a9160b1c4faca8e145d5d101d32bfae1f990a835d",
        ),
        "attempt": (
            repo_root
            / (
                "artifacts/mft_goal_postdeadline_full_retry_plan_v1.json."
                "post-attempt.json"
            ),
            "8e407f04039ed9309d9a2056f501325e2986f452543d8c9c29efbaa91e2ad708",
        ),
        "submission": (
            repo_root
            / (
                "artifacts/mft_goal_postdeadline_full_retry_plan_v1.json."
                "submission-receipt.json"
            ),
            "486bbaba51f55000c076ebbbfa3a629fed1843ee923c1b855119ea2167f0f095",
        ),
    }
    local: dict[str, dict[str, Any]] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for label, (path, digest) in paths.items():
        data = _verify_tracked_file(repo_root, path, digest)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise CollectorError(f"Full {label} root is not an object")
        local[label] = {
            "path": str(path.resolve()),
            "size_bytes": len(data),
            "sha256": digest,
        }
        parsed[label] = value
    plan = parsed["plan"]
    attempt = parsed["attempt"]
    submission = parsed["submission"]
    unsigned_plan = dict(plan)
    seal = unsigned_plan.pop("seal", None)
    if (
        not isinstance(seal, dict)
        or seal.get("sha256")
        != "3b39768fffea3e0c4527f588efd8702f347e9e0510d1196b813d97894d9d82df"
        or _sha256(_canonical_bytes(unsigned_plan)) != seal["sha256"]
    ):
        raise CollectorError("Full plan seal drifted")
    name = "mft-goal-postdeadline-diagnostic-full-retry-t96307-v1"
    dedupe = (
        "mft-al:mft-goal-postdeadline-diagnostic-full-retry-t96307-v1:"
        "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
        "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
        "62f2b846d05bbc58:postdeadline-v1"
    )
    if (
        plan.get("classification")
        != {
            "canonical": False,
            "canonical_dataset_mutation_allowed": False,
            "diagnostic_only": True,
            "original_deadline_missed": True,
            "post_deadline_retry": True,
            "production": False,
            "production_truth_eligible": False,
            "repairs_original_deadline": False,
        }
        or plan["payload"].get("name") != name
        or plan["payload"].get("dedupe_key") != dedupe
        or plan["payload"].get("timeout_seconds") != 86_400
        or plan["command_reuse"]["physics_candidate"].get("sha256")
        != FULL_CANDIDATE_SHA256
        or plan["command_reuse"]["physics_candidate"].get("byte_exact_unchanged")
        is not True
    ):
        raise CollectorError("Full plan identity/classification drifted")
    command = str(plan["payload"]["command"]).encode("utf-8")
    if (
        len(command) != 10_302
        or _sha256(command)
        != "979da42b7df65a1804892a783ed9756383863ea258cdda2cd1db20dfae792652"
    ):
        raise CollectorError("Full plan command drifted")
    if (
        attempt.get("task_name") != name
        or attempt.get("dedupe_key") != dedupe
        or attempt.get("post_attempt_consumed") is not True
        or attempt.get("maximum_post_attempts") != 1
        or submission.get("task_id") != 96326
        or submission.get("task_name") != name
        or submission.get("dedupe_key") != dedupe
        or submission.get("submitted") is not True
        or submission.get("post_attempts_consumed") != 1
        or submission.get("production") is not False
        or submission.get("canonical") is not False
    ):
        raise CollectorError("Full attempt/submission receipt drifted")
    receipt_unsigned = dict(submission)
    claimed_receipt = receipt_unsigned.pop("receipt_sha256", None)
    if (
        claimed_receipt
        != "ce5bf3fa7242dc6f871b6aeaa54c7576fa50744793d8035027a6e7eabb274eaa"
        or _sha256(_canonical_bytes(receipt_unsigned)) != claimed_receipt
    ):
        raise CollectorError("Full submission receipt payload seal drifted")
    retained = str(plan["payload"]["payload_json"]["retained_artifact_path"])
    return Contract(
        key="full96326",
        kind="full",
        task_id=96326,
        name=name,
        dedupe_key=dedupe,
        account="dhj02",
        node="n116",
        cpus=16,
        memory_mb=98_304,
        timeout_seconds=86_400,
        task_sh_size=14_105,
        task_sh_sha256=(
            "b646e60c31e57548da7aa59624d47ff18a240a2dea97544b5e824be101b0e723"
        ),
        command_sentinel=(
            b"source /etc/profile.d/lmod.sh 2>/dev/null || true; "
            b"module load ansys-electronics/v252"
        ),
        command_size=len(command),
        command_sha256=_sha256(command),
        command=command,
        local_evidence=local,
        retained_root=PurePosixPath(retained).parent.as_posix(),
        retained_destination=retained,
        plan=plan,
        submission=submission,
    )


def _thermal_contract(_repo_root: Path) -> Contract:
    runtime_root = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_goal_20260726\postdeadline_symmetric_r6"
    )
    paths = {
        "plan": (
            runtime_root / "postdeadline_symmetric_r6_plan.json",
            "efe8ad5cd5991e28a774e23dd3c17a4889f9578d675ebd0e2a860d44b63b4d79",
        ),
        "submission": (
            runtime_root / "postdeadline_symmetric_r6_submission.json",
            "81db11dd38900e2fc480b4615fce80908447e26cfb611a109c37a76097badf53",
        ),
    }
    local: dict[str, dict[str, Any]] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for label, (path, digest) in paths.items():
        data = _verify_file(path, digest)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise CollectorError(f"thermal {label} root is not an object")
        local[label] = {
            "path": str(path.resolve()),
            "size_bytes": len(data),
            "sha256": digest,
        }
        parsed[label] = value
    plan = parsed["plan"]
    submission = parsed["submission"]
    name = "mft-goal-corrected-thermal-l96230-b7c30cb70b95-postdeadline-r6-n111"
    dedupe = (
        "mft-al:mft-goal-corrected-thermal-l96230-b7c30cb70b95-"
        "postdeadline-r6-n111:"
        "a0208331949f70c21cde948e853a331d7f7f9824:"
        "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
        "086c88ea0cf2a41a"
    )
    if (
        plan.get("plan_payload_sha256")
        != "8e3e4de4ea23f993d1787199af5bc48f4895dbdbe4b46b983254ad030b651352"
        or plan.get("canonical") is not False
        or plan.get("diagnostic_only") is not True
        or plan.get("production_truth_eligible") is not False
        or plan["task_identity"].get("name") != name
        or plan["task_identity"].get("dedupe_key") != dedupe
        or plan["submission_profile"].get("timeout_seconds") != 45_000
        or plan["contract"].get("fixed_physics") != THERMAL_FIXED_PHYSICS
        or plan["checkpoint_manifest"].get("physics_boundary") != THERMAL_FIXED_PHYSICS
        or plan["executor"].get("revision")
        != "a0208331949f70c21cde948e853a331d7f7f9824"
        or plan["executor"].get("required_ancestor")
        != "a927ef7ba7d4b5577e47a43377922dacd77b1993"
    ):
        raise CollectorError("thermal plan identity/physics/revision drifted")
    command = str(plan["canonical_command"]).encode("utf-8")
    if (
        len(command) != 8_782
        or _sha256(command)
        != "41dfe4f93c47a7dedba9e86ed5fb4cbb7481845d19e2a5e881d504e29ef63a7b"
    ):
        raise CollectorError("thermal canonical command drifted")
    readback = submission.get("task_readback")
    if (
        submission.get("payload_sha256")
        != "c05740e6737b50a2fbeb80b1518b3d12a090ef1514346dcbcd33988409c2b2eb"
        or submission.get("command_sha256") != _sha256(command)
        or submission.get("scheduler_post_calls") != 1
        or submission.get("scheduler_submission_performed") is not True
        or submission.get("diagnostic_only") is not True
        or submission.get("canonical") is not False
        or submission.get("production_truth_eligible") is not False
        or not isinstance(readback, dict)
        or readback.get("task_id") != 96324
        or readback.get("name") != name
        or readback.get("dedupe_key") != dedupe
        or readback.get("node_name") != "n111"
        or readback.get("timeout_seconds") != 45_000
    ):
        raise CollectorError("thermal submission receipt drifted")
    absolute_retained = str(plan["retention"]["retained_root"])
    marker = "/slurm_scheduler/"
    if marker not in absolute_retained:
        raise CollectorError("thermal retained root is outside Scheduler workspace")
    retained_root = absolute_retained.split(marker, 1)[1]
    destination = (
        f"{retained_root}/"
        f"b7c-{plan['execution_contract']['checkpoint_manifest_sha256'][:12]}-"
        f"{plan['execution_contract']['executor_revision'][:12]}"
    )
    return Contract(
        key="thermal96324",
        kind="thermal",
        task_id=96324,
        name=name,
        dedupe_key=dedupe,
        account="r1jae262",
        node="n111",
        cpus=8,
        memory_mb=294_912,
        timeout_seconds=45_000,
        task_sh_size=10_115,
        task_sh_sha256=(
            "193ec5ed5e6a8eb983a950c29a04f158d3045e5e2fdb76aa6d1bfc28b02accf6"
        ),
        command_sentinel=(
            b"set -euo pipefail\nscratch='/enroot/mft-corrected-4e978e65dd2b8f5d'"
        ),
        command_size=len(command),
        command_sha256=_sha256(command),
        command=command,
        local_evidence=local,
        retained_root=retained_root,
        retained_destination=destination,
        plan=plan,
        submission=submission,
    )


def load_contract(target: str, repo_root: Path) -> Contract:
    if target == "full96326":
        return _full_contract(repo_root.resolve(strict=True))
    if target == "thermal96324":
        return _thermal_contract(repo_root.resolve(strict=True))
    raise CollectorError(f"unsupported target: {target}")


def _extract_command(task_sh: bytes, contract: Contract) -> bytes:
    if (
        len(task_sh) != contract.task_sh_size
        or _sha256(task_sh) != contract.task_sh_sha256
    ):
        raise CollectorError(
            f"{contract.key} task.sh drifted: "
            f"size={len(task_sh)} sha256={_sha256(task_sh)}"
        )
    if task_sh.count(contract.command_sentinel) != 1:
        raise CollectorError(f"{contract.key} command sentinel drifted")
    command = task_sh[task_sh.index(contract.command_sentinel) :]
    if not command.endswith(b"\n"):
        raise CollectorError(f"{contract.key} wrapper newline drifted")
    command = command[:-1]
    if (
        len(command) != contract.command_size
        or _sha256(command) != contract.command_sha256
        or command != contract.command
    ):
        raise CollectorError(f"{contract.key} extracted command drifted")
    return command


def _validate_full_candidate(command: bytes) -> dict[str, Any]:
    matches = list(
        re.finditer(
            rb"printf '%s' '(\{\"I1_rated\".*?\})' > cand\.json",
            command,
        )
    )
    if len(matches) != 1:
        raise CollectorError("Full candidate multiplicity drifted")
    candidate = matches[0].group(1)
    if (
        len(candidate) != FULL_CANDIDATE_SIZE
        or _sha256(candidate) != FULL_CANDIDATE_SHA256
    ):
        raise CollectorError("Full fixed-physics candidate drifted")
    value = json.loads(candidate)
    required = {
        "full_model": 1,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "fan_velocity": 1.5,
        "conductor_temp_C": 80.0,
        "air_temp": 50.0,
        "plate_temp": 50.0,
        "physics_data_revision": "mft1mw-1k101-native-lamination-kf0p85-v3",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise CollectorError(f"Full fixed physics drifted for {key}")
    return value


def validate_live_task(
    task: Mapping[str, Any],
    task_sh: bytes,
    contract: Contract,
) -> bytes:
    expected = {
        "task_id": contract.task_id,
        "id": contract.task_id,
        "name": contract.name,
        "dedupe_key": contract.dedupe_key,
        "account_name": contract.account,
        "requested_account_name": contract.account,
        "requested_node_name": contract.node,
        "node_name": contract.node,
        "node_name_policy": "strict",
        "strict_node_placement": True,
        "cpus": contract.cpus,
        "memory_mb": contract.memory_mb,
        "gpus": 0,
        "timeout_seconds": contract.timeout_seconds,
        "project": "MFT_1MW_2026v1",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise CollectorError(
                f"{contract.key} task field drifted for {key}: "
                f"{task.get(key)!r} != {value!r}"
            )
    for key in ("actual_node_name", "allocation_node_name"):
        observed = str(task.get(key) or "")
        if observed and observed != contract.node:
            raise CollectorError(
                f"{contract.key} actual placement drifted for {key}: {observed}"
            )
    command = _extract_command(task_sh, contract)
    if contract.kind == "full":
        _validate_full_candidate(command)
    elif (
        contract.plan["contract"].get("fixed_physics") != THERMAL_FIXED_PHYSICS
        or contract.plan["checkpoint_manifest"].get("physics_boundary")
        != THERMAL_FIXED_PHYSICS
    ):
        raise CollectorError("thermal fixed-physics boundary drifted")
    return command


def _task_state(task: Mapping[str, Any]) -> str:
    status = str(task.get("status") or "").lower()
    state = str(task.get("state") or "").lower()
    exit_code = task.get("exit_code")
    if status == "completed" and exit_code == 0:
        return "success"
    if status in TERMINAL_FAILURE_STATUSES or state in TERMINAL_FAILURE_STATUSES:
        return "failure"
    if exit_code is not None and status not in {"queued", "running", "pending"}:
        return "success" if int(exit_code) == 0 else "failure"
    return "watching"


def _parse_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CollectorError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CollectorError(f"{label} root is not an object")
    return value


def _save_remote(
    client: GetOnlyClient,
    contract: Contract,
    remote_path: str,
    local_path: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = MAX_REMOTE_FILE_BYTES,
) -> bytes:
    data = client.remote_file(
        contract.task_id,
        remote_path,
        max_bytes=max_bytes,
    )
    if expected_sha256 is not None and _sha256(data) != expected_sha256:
        raise CollectorError(f"remote file digest drifted: {remote_path}")
    _atomic_write(local_path, data)
    return data


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CollectorError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CollectorError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise CollectorError(f"{label} must be finite")
    return number


def _exact_integer(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    if not number.is_integer():
        raise CollectorError(f"{label} must be an integer")
    return int(number)


def _full_result_from_stdout(
    stdout: bytes,
    *,
    contract: Contract,
    expected_slurm_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _validate_full_candidate(contract.command)
    library_marker: str | None = None
    result: dict[str, Any] | None = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("MFT_LIBRARY_GIT_HASH "):
            observed = line[len("MFT_LIBRARY_GIT_HASH ") :].strip().lower()
            if re.fullmatch(r"[0-9a-f]{40}", observed):
                library_marker = observed
        if not line.startswith("RESULT_JSON "):
            continue
        try:
            parsed = json.loads(line[len("RESULT_JSON ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    if result is None:
        raise TransientGetError("Full terminal stdout has no RESULT_JSON")
    solver_revision = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
    library_revision = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
    parameters_match = True
    for key, expected in candidate.items():
        if key not in result:
            parameters_match = False
            break
        observed = result[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                matched = math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError, OverflowError):
                matched = False
        else:
            matched = str(observed) == str(expected)
        if not matched:
            parameters_match = False
            break
    if (
        library_marker != library_revision
        or str(result.get("git_hash") or "").lower() != solver_revision
        or str(result.get("pyaedt_library_git_hash") or "").lower()
        != library_revision
        or not parameters_match
        or _exact_integer(result.get("full_model"), "Full model mode") != 1
        or str(result.get("thermal_symmetry") or "").lower() != "full"
    ):
        raise CollectorError("Full RESULT_JSON identity drifted")
    expected_job = str(expected_slurm_job_id or "")
    if re.fullmatch(r"[1-9][0-9]*", expected_job) is None:
        raise CollectorError("Full terminal Slurm job identity is invalid")
    required_core = {
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": "mft-standalone-core-16-optin-v1",
        "solver_core_backend": "standalone",
        "solver_core_license_contract": "mft-aedt-hpc-license-snapshot-v1",
    }
    if (
        any(result.get(key) != value for key, value in required_core.items())
        or _exact_integer(result.get("solver_core_opt_in"), "core opt-in") != 1
        or _exact_integer(
            result.get("solver_num_cores_requested"),
            "requested core count",
        )
        != 16
        or _exact_integer(
            result.get("solver_num_cores_effective"),
            "effective core count",
        )
        != 16
        or _exact_integer(
            result.get("solver_num_tasks_effective"),
            "effective task count",
        )
        != 1
        or _exact_integer(
            result.get("solver_core_affinity_count_readback"),
            "core affinity count",
        )
        < 16
        or str(result.get("solver_core_slurm_cpus_per_task_readback") or "")
        != "16"
        or str(result.get("solver_core_scheduler_task_id_readback") or "")
        != str(contract.task_id)
        or str(result.get("solver_core_slurm_job_id_readback") or "")
        != expected_job
        or _exact_integer(
            result.get("solver_matrix_hpc_num_cores_readback"),
            "matrix core readback",
        )
        != 16
        or _exact_integer(
            result.get("solver_matrix_hpc_num_engines_readback"),
            "matrix engine readback",
        )
        != 1
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(result.get("solver_matrix_hpc_acf_sha256") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(result.get("solver_core_license_snapshot_sha256") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(result.get("solver_core_auth_sha256") or ""),
        )
        is None
    ):
        raise CollectorError("Full RESULT_JSON core identity drifted")
    scientific_flags = (
        "result_valid_em",
        "result_valid_thermal",
        "thermal_solved",
        "thermal_extraction_complete",
        "thermal_convergence_available",
        "thermal_converged",
    )
    if (
        any(
            _exact_integer(result.get(key), f"Full result {key}") != 1
            for key in scientific_flags
        )
        or _exact_integer(
            result.get("thermal_required_missing_count"),
            "Full thermal missing count",
        )
        != 0
        or not _finite_number(
            result.get("f_res_min_tx_rx_only_Hz"),
            "Full actual resonance",
        )
        > 0.0
    ):
        raise CollectorError("Full RESULT_JSON scientific structure drifted")
    identity = {
        "candidate_physics_sha256": FULL_CANDIDATE_SHA256,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "scheduler_task_id": contract.task_id,
        "slurm_job_id": expected_job,
        "full_model": 1,
        "thermal_symmetry": "full",
        "result_payload_sha256": _sha256(_canonical_bytes(result)),
    }
    return result, identity


def _collect_full(
    client: GetOnlyClient,
    contract: Contract,
    output_root: Path,
    *,
    expected_slurm_job_id: str,
) -> dict[str, Any]:
    root = contract.retained_root
    inventory = client.remote_files(contract.task_id, f"{root}/**")
    collection = output_root / "collection"
    _atomic_json(collection / "remote-inventory.json", inventory)
    expected_paths = {
        f"{root}/full.aedt",
        f"{root}/full.aedt.receipt.json",
        f"{root}/.slurm-scheduler-preserve.json",
    }
    missing = sorted(expected_paths - set(inventory))
    chunk_paths = sorted(
        path
        for path in inventory
        if path.startswith(f"{root}/full.aedt.chunks/")
        and re.fullmatch(r".*/[0-9]{8}\.b64", path)
    )
    results_paths = [
        path
        for path in inventory
        if "/full.aedtresults/" in path or path.endswith("/full.aedtresults")
    ]
    pending: list[str] = []
    if missing:
        pending.append("missing retained Full paths: " + ", ".join(missing))
    if not results_paths:
        pending.append(
            "full.aedtresults is absent from the submitted retention contract"
        )
    if missing:
        return {
            "state": "success_pending_collection",
            "pending_reasons": pending,
            "remote_inventory": inventory,
            "collected_files": [],
            **CLASSIFICATION,
        }

    receipt_path = f"{root}/full.aedt.receipt.json"
    receipt_data = _save_remote(
        client,
        contract,
        receipt_path,
        collection / "full.aedt.receipt.json",
    )
    receipt = _parse_json_bytes(receipt_data, "Full retention receipt")
    marker_data = _save_remote(
        client,
        contract,
        f"{root}/.slurm-scheduler-preserve.json",
        collection / ".slurm-scheduler-preserve.json",
    )
    marker = _parse_json_bytes(marker_data, "Full prune marker")
    if (
        receipt.get("schema_version") != "mft-goal-fea-remote-artifact-receipt-v1"
        or receipt.get("dedupe_key") != contract.dedupe_key
        or receipt.get("artifact_path") != f"{contract.retained_root}/full.aedt"
        or receipt.get("stage") != "full"
        or receipt.get("solver_revision") != "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
        or receipt.get("library_revision") != "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
        or receipt.get("parameter_digest") != "62f2b846d05bbc58"
        or marker.get("schema") != "slurm-scheduler-prune-protection-v1"
        or marker.get("preserve") is not True
        or contract.dedupe_key not in str(marker.get("reason") or "")
    ):
        raise CollectorError("Full retained receipt/marker identity drifted")
    expected_count = int(receipt.get("transport_chunk_count") or -1)
    expected_size = int(receipt.get("artifact_size_bytes") or -1)
    expected_artifact_sha = str(receipt.get("artifact_sha256") or "")
    if (
        expected_count <= 0
        or expected_count != len(chunk_paths)
        or expected_size <= 0
        or len(expected_artifact_sha) != 64
    ):
        pending.append("Full chunk inventory is incomplete")
        return {
            "state": "success_pending_collection",
            "pending_reasons": pending,
            "remote_inventory": inventory,
            "receipt": receipt,
            "collected_files": [
                "full.aedt.receipt.json",
                ".slurm-scheduler-preserve.json",
            ],
            **CLASSIFICATION,
        }
    expected_names = [
        f"{root}/full.aedt.chunks/{index:08d}.b64" for index in range(expected_count)
    ]
    if chunk_paths != expected_names:
        raise CollectorError("Full chunks are not contiguous and canonical")
    encoded_estimate = ((expected_size + 2) // 3) * 4
    required_free = expected_size + encoded_estimate + 1024**3
    free = shutil.disk_usage(collection).free
    if free < required_free:
        pending.append(f"local collection free space {free} is below {required_free}")
        return {
            "state": "success_pending_collection",
            "pending_reasons": pending,
            "remote_inventory": inventory,
            "receipt": receipt,
            "collected_files": [
                "full.aedt.receipt.json",
                ".slurm-scheduler-preserve.json",
            ],
            **CLASSIFICATION,
        }

    chunks_root = collection / "full.aedt.chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)
    partial = collection / f".full.aedt.{os.getpid()}.partial"
    digest = hashlib.sha256()
    size = 0
    with partial.open("xb") as assembled:
        for index, remote_path in enumerate(chunk_paths):
            encoded = client.remote_file(
                contract.task_id,
                remote_path,
                max_bytes=MAX_REMOTE_FILE_BYTES,
            )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise CollectorError(
                    f"Full chunk is invalid base64: {remote_path}"
                ) from exc
            if index + 1 < expected_count and len(raw) != 768_000:
                raise CollectorError(
                    f"Full non-final chunk size drifted: {remote_path}"
                )
            _atomic_write(chunks_root / PurePosixPath(remote_path).name, encoded)
            assembled.write(raw)
            digest.update(raw)
            size += len(raw)
        assembled.flush()
        os.fsync(assembled.fileno())
    if size != expected_size or digest.hexdigest() != expected_artifact_sha:
        raise CollectorError("reassembled Full AEDT authentication failed")
    final_aedt = collection / "full.aedt"
    os.replace(partial, final_aedt)
    _atomic_json(
        collection / "full-aedt-authentication.json",
        {
            "artifact_size_bytes": size,
            "artifact_sha256": digest.hexdigest(),
            "transport_chunk_count": expected_count,
            "transport_chunks_retained": True,
            **CLASSIFICATION,
        },
    )
    try:
        stdout = client.task_output(
            contract.task_id,
            "stdout",
            max_bytes=MAX_TASK_OUTPUT_BYTES,
        )
        stderr = client.task_output(
            contract.task_id,
            "stderr",
            max_bytes=MAX_TASK_OUTPUT_BYTES,
        )
        result, result_identity = _full_result_from_stdout(
            stdout,
            contract=contract,
            expected_slurm_job_id=expected_slurm_job_id,
        )
    except TransientGetError as exc:
        pending.append(str(exc))
        result = None
        result_identity = None
    else:
        _atomic_write(collection / "stdout.log", stdout)
        _atomic_write(collection / "stderr.log", stderr)
        _atomic_json(collection / "full_result.json", result)
    result_record = None
    if result is not None and result_identity is not None:
        result_path = collection / "full_result.json"
        result_record = {
            "path": str(result_path),
            "size_bytes": result_path.stat().st_size,
            "sha256": _sha256(result_path.read_bytes()),
            "identity": result_identity,
        }
    return {
        "state": (
            "success_collected_diagnostic"
            if not pending
            else "success_pending_collection"
        ),
        "pending_reasons": pending,
        "remote_inventory": inventory,
        "receipt": receipt,
        "reassembled_full_aedt": {
            "path": str(final_aedt),
            "size_bytes": size,
            "sha256": digest.hexdigest(),
        },
        "retained_chunk_count": expected_count,
        "aedtresults_remote_paths": results_paths,
        "full_result_truth": result_record,
        **CLASSIFICATION,
    }


def _last_prefixed_json(data: bytes, prefix: str) -> dict[str, Any] | None:
    result = None
    for line in data.decode("utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result = value
    return result


def _collect_thermal(
    client: GetOnlyClient,
    contract: Contract,
    output_root: Path,
) -> dict[str, Any]:
    destination = contract.retained_destination
    inventory = client.remote_files(contract.task_id, f"{destination}/**")
    collection = output_root / "collection"
    _atomic_json(collection / "remote-inventory.json", inventory)
    manifest_remote = f"{destination}/manifest.json"
    if manifest_remote not in inventory:
        return {
            "state": "success_pending_collection",
            "pending_reasons": ["thermal retention manifest is not remotely visible"],
            "remote_inventory": inventory,
            "collected_files": [],
            **CLASSIFICATION,
        }
    stdout = client.task_output(contract.task_id, "stdout")
    stderr = client.task_output(contract.task_id, "stderr")
    _atomic_write(collection / "stdout.log", stdout)
    _atomic_write(collection / "stderr.log", stderr)
    marker = _last_prefixed_json(stdout, "CORRECTED_THERMAL_JSON ")
    if marker is None:
        raise CollectorError("thermal success marker is absent from stdout")
    _verify_embedded_digest(marker)
    if (
        marker.get("diagnostic_only") is not True
        or marker.get("canonical") is not False
        or marker.get("production_truth_eligible") is not False
        or marker.get("task_name") != contract.name
        or marker.get("dedupe_key") != contract.dedupe_key
        or marker.get("execution_exit_code") != 0
    ):
        raise CollectorError("thermal success marker identity drifted")
    _atomic_json(collection / "CORRECTED_THERMAL_JSON.json", marker)

    manifest_data = _save_remote(
        client,
        contract,
        manifest_remote,
        collection / "manifest.json",
        expected_sha256=str(marker.get("retention_manifest_file_sha256") or ""),
    )
    manifest = _parse_json_bytes(manifest_data, "thermal retained manifest")
    _verify_embedded_digest(manifest)
    if (
        manifest.get("schema") != "mft-corrected-thermal-minimum-retained-package-v1"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("candidate_sha256")
        != contract.plan["task_identity"]["candidate_sha256"]
        or manifest.get("checkpoint_manifest_sha256")
        != contract.plan["execution_contract"]["checkpoint_manifest_sha256"]
        or manifest.get("executor_solver_revision")
        != "a0208331949f70c21cde948e853a331d7f7f9824"
    ):
        raise CollectorError("thermal retained manifest contract drifted")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 5:
        raise CollectorError("thermal retained manifest inventory drifted")
    rows: dict[str, dict[str, Any]] = {}
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise CollectorError("thermal retained file record drifted")
        relative = _safe_relative(str(raw["path"]))
        rows[relative] = raw
    required_names = {
        "symmetric.aedt",
        "corrected_result.json",
        "execution_receipt.json",
    }
    if (
        not required_names.issubset(rows)
        or len([name for name in rows if name.startswith("convergence/")]) != 1
        or len([name for name in rows if name.startswith("profile/")]) != 1
    ):
        raise CollectorError("thermal retained member names drifted")
    controls = {
        "manifest.json",
        ".slurm-scheduler-preserve.json",
        "retention_receipt.json",
    }
    expected_remote = {f"{destination}/{relative}" for relative in set(rows) | controls}
    missing = sorted(expected_remote - set(inventory))
    pending: list[str] = []
    if missing:
        pending.append("missing thermal retained paths: " + ", ".join(missing))
    collected: list[dict[str, Any]] = []
    for relative, row in sorted(rows.items()):
        size = int(row["size_bytes"])
        remote_path = f"{destination}/{relative}"
        if remote_path not in inventory:
            continue
        if size > MAX_REMOTE_FILE_BYTES:
            pending.append(
                f"unchunked retained artifact exceeds GET window: {relative} ({size})"
            )
            continue
        data = _save_remote(
            client,
            contract,
            remote_path,
            collection / "artifacts" / Path(*PurePosixPath(relative).parts),
            expected_sha256=str(row["sha256"]),
            max_bytes=max(size, 1),
        )
        if len(data) != size:
            raise CollectorError(f"thermal retained size drifted: {relative}")
        collected.append(
            {"path": relative, "size_bytes": size, "sha256": row["sha256"]}
        )
    for relative in sorted(controls - {"manifest.json"}):
        remote_path = f"{destination}/{relative}"
        if remote_path not in inventory:
            continue
        data = _save_remote(
            client,
            contract,
            remote_path,
            collection / relative,
        )
        value = _parse_json_bytes(data, f"thermal {relative}")
        if value.get("diagnostic_only") not in {None, True}:
            raise CollectorError(f"thermal {relative} classification drifted")
    if not any(item["path"] == "symmetric.aedt" for item in collected):
        if not any(
            reason.startswith("unchunked retained artifact") for reason in pending
        ):
            pending.append("thermal symmetric.aedt was not byte-collected")
    return {
        "state": (
            "success_collected_diagnostic"
            if not pending
            else "success_pending_collection"
        ),
        "pending_reasons": pending,
        "remote_inventory": inventory,
        "retention_marker": marker,
        "manifest": manifest,
        "collected_files": collected,
        **CLASSIFICATION,
    }


def _task_snapshot(task: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "name",
        "status",
        "state",
        "exit_code",
        "failure_message",
        "created_at",
        "started_at",
        "finished_at",
        "allocation_id",
        "slurm_job_id",
        "account_name",
        "node_name",
        "actual_node_name",
        "cpus",
        "memory_mb",
        "timeout_seconds",
        "dedupe_key",
    )
    return {key: task.get(key) for key in keys}


def poll_once(
    contract: Contract,
    client: GetOnlyClient,
    output_root: Path,
    *,
    poll_number: int,
) -> dict[str, Any]:
    task_value = client.get_json(f"/api/tasks/{contract.task_id}")
    if not isinstance(task_value, dict):
        raise TransientGetError("task status response is not an object")
    task_sh = client.remote_file(
        contract.task_id,
        "task.sh",
        base="remote_dir",
        max_bytes=MAX_REMOTE_FILE_BYTES,
    )
    command = validate_live_task(task_value, task_sh, contract)
    observed = _task_state(task_value)
    event: dict[str, Any] = {
        "schema": "mft-terminal-collector-event-v1",
        "observed_at_utc": _now(),
        "poll_number": poll_number,
        "target": contract.key,
        "task": _task_snapshot(task_value),
        "contract": {
            "task_sh_size_bytes": len(task_sh),
            "task_sh_sha256": _sha256(task_sh),
            "command_size_bytes": len(command),
            "command_sha256": _sha256(command),
            "local_evidence": contract.local_evidence,
            "fixed_physics_verified": True,
            "task_identity_verified": True,
            "strict_node_verified": True,
        },
        "scheduler_state": observed,
        **CLASSIFICATION,
    }
    if observed == "failure":
        event.update(
            {
                "state": "failure_ledger_only",
                "artifact_collection_attempted": False,
                "failure_or_timeout": True,
            }
        )
    elif observed == "success":
        collection = (
            _collect_full(
                client,
                contract,
                output_root,
                expected_slurm_job_id=str(task_value.get("slurm_job_id") or ""),
            )
            if contract.kind == "full"
            else _collect_thermal(client, contract, output_root)
        )
        event.update(collection)
        event["artifact_collection_attempted"] = True
        event["failure_or_timeout"] = False
    else:
        event.update(
            {
                "state": "watching",
                "artifact_collection_attempted": False,
                "failure_or_timeout": False,
            }
        )
    event["event_sha256"] = _sha256(_canonical_bytes(event))
    return event


def _record_event(output_root: Path, event: Mapping[str, Any]) -> None:
    if not (output_root / "first-poll.json").exists():
        _exclusive_json(output_root / "first-poll.json", event)
    _atomic_json(output_root / "latest.json", event)
    _append_event(output_root / "events.jsonl", event)
    print(json.dumps(event, sort_keys=True), flush=True)


def run_watcher(
    contract: Contract,
    client: GetOnlyClient,
    output_root: Path,
    *,
    watch: bool,
    interval_seconds: int,
) -> int:
    if interval_seconds <= 0:
        raise CollectorError("watch interval must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    _exclusive_json(
        output_root / "watcher.json",
        {
            "schema": "mft-terminal-collector-watcher-v1",
            "started_at_utc": _now(),
            "pid": os.getpid(),
            "target": contract.key,
            "task_id": contract.task_id,
            "interval_seconds": interval_seconds,
            "get_only": True,
            "post_surface_present": False,
            **CLASSIFICATION,
        },
    )
    poll_number = 0
    while True:
        poll_number += 1
        try:
            event = poll_once(
                contract,
                client,
                output_root,
                poll_number=poll_number,
            )
        except TransientGetError as exc:
            event = {
                "schema": "mft-terminal-collector-event-v1",
                "observed_at_utc": _now(),
                "poll_number": poll_number,
                "target": contract.key,
                "state": "get_error_retry_pending",
                "error": str(exc),
                **CLASSIFICATION,
            }
            event["event_sha256"] = _sha256(_canonical_bytes(event))
            _record_event(output_root, event)
            if not watch:
                return 3
            time.sleep(interval_seconds)
            continue
        except CollectorError as exc:
            event = {
                "schema": "mft-terminal-collector-event-v1",
                "observed_at_utc": _now(),
                "poll_number": poll_number,
                "target": contract.key,
                "state": "contract_violation",
                "error": str(exc),
                **CLASSIFICATION,
            }
            event["event_sha256"] = _sha256(_canonical_bytes(event))
            _record_event(output_root, event)
            return 2
        _record_event(output_root, event)
        if event["state"] != "watching" or not watch:
            return 0
        time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("full96326", "thermal96324"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        contract = load_contract(args.target, repo_root)
        client = GetOnlyClient(args.scheduler_url)
        return run_watcher(
            contract,
            client,
            args.output_root.resolve(),
            watch=bool(args.watch),
            interval_seconds=args.interval_seconds,
        )
    except CollectorError as exc:
        print(f"COLLECTOR_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
