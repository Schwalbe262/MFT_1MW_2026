#!/usr/bin/env python3
"""Create and audit a fail-closed GPFS checkpoint for thermal continuation.

This utility has no Scheduler API client and cannot submit or cancel work.  It
copies only the sealed Icepak premesh allowlist from a still-running AEDT tree.
The live private solution, lock, EM-result, geometry-cache, and pyaedt trees are
not traversed.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "mft-corrected-thermal-static-checkpoint-v3"
ALLOWED_DESTINATION_ROOT = Path(
    "/gpfs/home1/r1jae262/slurm_scheduler/mft_goal_20260726/immutable_sources_v1"
)
COPY_CHUNK_BYTES = 8 * 1024 * 1024
MANIFEST_BUDGET_BYTES = 1024 * 1024
INODE_BUDGET_BYTES = 4096
PHYSICAL_FREE_RESERVE_BYTES = 50 * 1024**3
MAX_QUOTA_EVIDENCE_AGE_SECONDS = 120
AT_FDCWD = -100
RENAME_NOREPLACE = 1
BINARY_FLAG = getattr(os, "O_BINARY", 0)
SHA256_HEX_LENGTH = 64
TRANSITION_FILE_MODES = frozenset((0o600, 0o400))
TRANSITION_DIRECTORY_MODES = frozenset((0o700, 0o500))

SOURCE_PROVENANCE = {
    "allocation_id": 14492,
    "candidate_sha256": (
        "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
    ),
    "execution_task_id": 96304,
    "library_revision": "e6b9b9d20a832ff5c3f7ca97218737a0b8650781",
    "logical_task_id": 96230,
    "node": "n114",
    "slurm_job_id": 824575,
    "solver_revision": "a1e4f70cefa1af04673c73a6131bf490c0cc14b5",
    "source_plan_identity_sha256": (
        "08a6e426d261803660e933ad85985e17af37fea8e2b4a29a9ee2b5eaaf4483c3"
    ),
}
PUBLIC_RESULT_SUFFIXES = (
    "ManagedFiles_Design7.asol",
    "icepak_thermal.asol",
    "icepak_thermal.results/DV274_S271_V0.profile",
    "icepak_thermal.results/DV274_S271_V275.profile",
)
MESH_FAMILIES = (
    ("DV274_Meshes", "_V213.sd"),
    ("DV274_S271_Meshes", "_V0.sd"),
)
MESH_INDICES = (0, 9, 10, 11, 12)
MESH_FILES = ("grid_mapping", "grid_output")


class CheckpointError(RuntimeError):
    """A fail-closed checkpoint contract violation."""


@dataclass(frozen=True)
class StaticFile:
    path: str
    size: int
    allocated: int
    mtime_ns: int
    mode: str
    inode: int
    device: int
    nlink: int

    def metadata_identity(self) -> tuple[Any, ...]:
        return (
            self.path,
            self.size,
            self.allocated,
            self.mtime_ns,
            self.mode,
        )


@dataclass(frozen=True)
class Writer:
    pid: int
    fd: int
    command: str
    path: str
    included: bool
    flags_octal: str


@dataclass(frozen=True)
class Snapshot:
    root: str
    observed_epoch: float
    files: tuple[StaticFile, ...]
    directories: tuple[str, ...]
    writers: tuple[Writer, ...]
    metadata_sha256: str
    logical_bytes: int
    allocated_bytes: int

    @property
    def included_writers(self) -> tuple[Writer, ...]:
        return tuple(row for row in self.writers if row.included)

    @property
    def required_budget_bytes(self) -> int:
        inodes = len(self.files) + len(self.directories) + 1
        return (
            max(self.logical_bytes, self.allocated_bytes)
            + inodes * INODE_BUDGET_BYTES
            + MANIFEST_BUDGET_BYTES
        )

    @property
    def required_inodes(self) -> int:
        return len(self.files) + len(self.directories) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "directory_count": len(self.directories),
            "included_count": len(self.files),
            "included_writer_count": len(self.included_writers),
            "logical_bytes": self.logical_bytes,
            "metadata_sha256": self.metadata_sha256,
            "observed_epoch": self.observed_epoch,
            "required_budget_bytes": self.required_budget_bytes,
            "required_inodes": self.required_inodes,
            "root": self.root,
            "writer_count": len(self.writers),
        }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointError(f"{label} is not a lowercase SHA-256")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _allocated_bytes(file_stat: os.stat_result) -> int:
    blocks = getattr(file_stat, "st_blocks", None)
    if blocks is not None:
        return int(blocks) * 512
    return ((int(file_stat.st_size) + 511) // 512) * 512


def _metadata_digest(files: Iterable[StaticFile]) -> str:
    payload = "".join(
        f"{path}\0{size}\0{allocated}\0{mtime_ns}\0{mode}\n"
        for path, size, allocated, mtime_ns, mode in (
            row.metadata_identity() for row in files
        )
    ).encode()
    return _sha256_bytes(payload)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _lstat_real_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise CheckpointError(f"{label} cannot be lstat'ed: {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise CheckpointError(f"{label} is not a real directory: {path}")
    return value


def _assert_real_directory_chain(root: Path, relative_parent: Path) -> None:
    current = root
    _lstat_real_directory(current, "source root")
    for part in relative_parent.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise CheckpointError("allowlisted path attempts parent traversal")
        current = current / part
        _lstat_real_directory(current, "allowlisted source ancestor")


def _allowed_relative_paths(source: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    aedt_entries: list[str] = []
    try:
        root_entries = tuple(os.scandir(source))
    except OSError as exc:
        raise CheckpointError(f"cannot scan source root: {exc}") from exc
    for entry in root_entries:
        if not entry.name.endswith(".aedt"):
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise CheckpointError(f"cannot lstat AEDT entry: {entry.path}") from exc
        if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
            raise CheckpointError(
                f"AEDT entry is not a real regular file: {entry.path}"
            )
        aedt_entries.append(entry.name)
    if len(aedt_entries) != 1:
        raise CheckpointError(
            f"expected exactly one direct-child AEDT, found {aedt_entries!r}"
        )
    aedt_name = aedt_entries[0]
    project_stem = aedt_name[: -len(".aedt")]
    results_root = f"{project_stem}.aedtresults"
    thermal_results = f"{results_root}/icepak_thermal.results"

    allowed = [aedt_name]
    allowed.extend(f"{results_root}/{suffix}" for suffix in PUBLIC_RESULT_SUFFIXES)
    mesh_directories: list[str] = []
    for family, suffix in MESH_FAMILIES:
        for index in MESH_INDICES:
            directory = f"{thermal_results}/{family}{index}{suffix}"
            mesh_directories.append(directory)
            allowed.extend(f"{directory}/{name}" for name in MESH_FILES)
    return tuple(sorted(allowed)), tuple(sorted(mesh_directories))


def _required_directories(paths: Iterable[str]) -> tuple[str, ...]:
    directories = {"."}
    for relative in paths:
        parent = Path(relative).parent
        while os.fspath(parent) not in ("", "."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories))


def _command_name(pid: int, proc_root: Path) -> str:
    try:
        return (
            (proc_root / str(pid) / "comm")
            .read_text(encoding="utf-8", errors="replace")
            .strip()
        )
    except OSError:
        return "unknown"


def _transient_proc_error(exc: OSError) -> bool:
    return exc.errno in (errno.ENOENT, errno.ESRCH)


def _open_writers(
    root: Path,
    included: set[str],
    *,
    proc_root: Path = Path("/proc"),
    slurm_job_id: int = int(SOURCE_PROVENANCE["slurm_job_id"]),
) -> tuple[Writer, ...]:
    if not proc_root.is_dir():
        raise CheckpointError(f"proc filesystem is unavailable: {proc_root}")
    current_uid = os.getuid()
    rows: list[Writer] = []
    root_text = str(root)
    root_prefix = root_text + os.sep
    try:
        process_entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise CheckpointError(f"cannot enumerate {proc_root}: {exc}") from exc
    for process in process_entries:
        if not process.name.isdigit():
            continue
        try:
            process_stat = process.stat()
        except OSError as exc:
            if _transient_proc_error(exc):
                continue
            raise CheckpointError(f"cannot stat process {process}: {exc}") from exc
        if process_stat.st_uid != current_uid:
            continue
        try:
            cgroup = (process / "cgroup").read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            if _transient_proc_error(exc):
                continue
            raise CheckpointError(
                f"cannot read process cgroup {process}: {exc}"
            ) from exc
        if f"/job_{slurm_job_id}/" not in cgroup:
            continue
        fd_root = process / "fd"
        try:
            descriptors = tuple(fd_root.iterdir())
        except OSError as exc:
            if _transient_proc_error(exc):
                continue
            raise CheckpointError(f"cannot enumerate {fd_root}: {exc}") from exc
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                if target.endswith(" (deleted)"):
                    target = target[:-10]
                resolved = os.path.realpath(target)
                fdinfo = (process / "fdinfo" / descriptor.name).read_text(
                    encoding="ascii", errors="replace"
                )
                flags_line = next(
                    row for row in fdinfo.splitlines() if row.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
            except OSError as exc:
                if _transient_proc_error(exc):
                    continue
                raise CheckpointError(
                    f"cannot inspect process fd {descriptor}: {exc}"
                ) from exc
            except (StopIteration, ValueError) as exc:
                raise CheckpointError(
                    f"cannot parse process fd flags: {descriptor}"
                ) from exc
            if (flags & os.O_ACCMODE) == os.O_RDONLY:
                continue
            if resolved not in included and not resolved.startswith(root_prefix):
                continue
            rows.append(
                Writer(
                    command=_command_name(int(process.name), proc_root),
                    fd=int(descriptor.name),
                    flags_octal=oct(flags),
                    included=resolved in included,
                    path=resolved,
                    pid=int(process.name),
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.pid, row.fd, row.path)))


def snapshot_static_source(source: Path) -> Snapshot:
    source = source.resolve(strict=True)
    _lstat_real_directory(source, "source")
    relative_paths, mesh_directories = _allowed_relative_paths(source)
    for relative in mesh_directories:
        mesh = source / relative
        _assert_real_directory_chain(source, Path(relative))
        try:
            entries = tuple(os.scandir(mesh))
        except OSError as exc:
            raise CheckpointError(f"cannot scan mesh directory: {mesh}") from exc
        names = {entry.name for entry in entries}
        if names != set(MESH_FILES):
            raise CheckpointError(
                f"mesh directory has non-contract children: {mesh}: {sorted(names)!r}"
            )

    files: list[StaticFile] = []
    included_absolute: set[str] = set()
    for relative in relative_paths:
        path = source / relative
        _assert_real_directory_chain(source, Path(relative).parent)
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise CheckpointError(f"allowlisted file is unavailable: {path}") from exc
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise CheckpointError(f"allowlisted file is not real/regular: {path}")
        if file_stat.st_nlink != 1:
            raise CheckpointError(
                f"allowlisted file has st_nlink={file_stat.st_nlink}: {path}"
            )
        files.append(
            StaticFile(
                allocated=_allocated_bytes(file_stat),
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
                mode=oct(stat.S_IMODE(file_stat.st_mode)),
                mtime_ns=file_stat.st_mtime_ns,
                nlink=file_stat.st_nlink,
                path=relative,
                size=file_stat.st_size,
            )
        )
        included_absolute.add(str(path.resolve(strict=True)))
    files.sort(key=lambda row: row.path)
    writers = _open_writers(source, included_absolute)
    return Snapshot(
        allocated_bytes=sum(row.allocated for row in files),
        directories=_required_directories(relative_paths),
        files=tuple(files),
        logical_bytes=sum(row.size for row in files),
        metadata_sha256=_metadata_digest(files),
        observed_epoch=time.time(),
        root=str(source),
        writers=writers,
    )


def _assert_expected_snapshot(
    snapshot: Snapshot,
    *,
    metadata_sha256: str,
    count: int,
    directory_count: int,
    logical_bytes: int,
    allocated_bytes: int,
) -> None:
    actual = {
        "allocated_bytes": snapshot.allocated_bytes,
        "count": len(snapshot.files),
        "directory_count": len(snapshot.directories),
        "logical_bytes": snapshot.logical_bytes,
        "metadata_sha256": snapshot.metadata_sha256,
    }
    expected = {
        "allocated_bytes": allocated_bytes,
        "count": count,
        "directory_count": directory_count,
        "logical_bytes": logical_bytes,
        "metadata_sha256": metadata_sha256,
    }
    if actual != expected:
        raise CheckpointError(
            f"source differs from sealed admission: expected={expected}, actual={actual}"
        )
    if snapshot.included_writers:
        raise CheckpointError(
            "allowlisted files are open for write: "
            + json.dumps(
                [asdict(row) for row in snapshot.included_writers],
                sort_keys=True,
            )
        )


def _decode_login_quota_evidence(
    encoded: str,
    *,
    expected_canonical_sha256: str,
    maximum_age_seconds: int,
    now: float | None = None,
) -> dict[str, int | float | str]:
    if maximum_age_seconds <= 0 or maximum_age_seconds > MAX_QUOTA_EVIDENCE_AGE_SECONDS:
        raise CheckpointError("quota evidence max age must be in [1, 120] seconds")
    try:
        raw = base64.b64decode(encoded, validate=True)
        evidence = json.loads(raw, parse_constant=_reject_json_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise CheckpointError("login quota evidence is not valid base64 JSON") from exc
    if not isinstance(evidence, dict):
        raise CheckpointError("login quota evidence is not a JSON object")
    claimed = evidence.pop("canonical_sha256", None)
    canonical = _sha256_bytes(_canonical_json(evidence))
    if claimed != canonical or expected_canonical_sha256 != canonical:
        raise CheckpointError("login quota evidence canonical SHA-256 mismatch")
    required = {
        "filesystem",
        "quota_type",
        "uid",
        "usage_bytes",
        "soft_limit_bytes",
        "hard_limit_bytes",
        "in_doubt_bytes",
        "files_used",
        "files_soft_limit",
        "files_hard_limit",
        "files_in_doubt",
        "observed_at_epoch",
        "source",
    }
    if set(evidence) != required:
        raise CheckpointError(
            "login quota evidence fields mismatch: "
            f"missing={sorted(required - set(evidence))}, "
            f"extra={sorted(set(evidence) - required)}"
        )
    if evidence["filesystem"] != "gpfs" or evidence["quota_type"] != "USR":
        raise CheckpointError("login quota evidence filesystem/type mismatch")
    if evidence["source"] != "gate2:mmlsquota-Y":
        raise CheckpointError("login quota evidence authority mismatch")
    integer_fields = {
        "uid",
        "usage_bytes",
        "soft_limit_bytes",
        "hard_limit_bytes",
        "in_doubt_bytes",
        "files_used",
        "files_soft_limit",
        "files_hard_limit",
        "files_in_doubt",
    }
    for name in integer_fields:
        value = evidence[name]
        if type(value) is not int or value < 0:
            raise CheckpointError(
                f"quota evidence field must be a nonnegative exact integer: {name}"
            )
    if int(evidence["uid"]) != os.getuid():
        raise CheckpointError("login quota evidence UID mismatch")
    observed_value = evidence["observed_at_epoch"]
    if (
        isinstance(observed_value, bool)
        or not isinstance(observed_value, (int, float))
        or not math.isfinite(float(observed_value))
        or float(observed_value) < 0
    ):
        raise CheckpointError(
            "quota evidence observed_at_epoch must be finite and nonnegative"
        )
    observed = float(observed_value)
    current = time.time() if now is None else now
    age = current - observed
    if age < -5 or age > maximum_age_seconds:
        raise CheckpointError(
            f"login quota evidence is stale/future: age_seconds={age:.3f}"
        )
    evidence["canonical_sha256"] = canonical
    evidence["age_seconds_at_validation"] = age
    return evidence


def _filesystem_capacity(
    path: Path,
    *,
    required_free_bytes: int,
    expected_device: int | None = None,
    expected_fsid: int | None = None,
) -> dict[str, int | bool]:
    directory = _lstat_real_directory(path, "destination filesystem anchor")
    values = os.statvfs(path)
    free_bytes = int(values.f_bavail) * int(values.f_frsize)
    readonly_flag = int(getattr(os, "ST_RDONLY", 1))
    readonly = bool(int(values.f_flag) & readonly_flag)
    fsid = int(getattr(values, "f_fsid", 0))
    if readonly:
        raise CheckpointError(f"destination filesystem is read-only: {path}")
    if free_bytes < required_free_bytes:
        raise CheckpointError(
            f"destination physical free below contract: "
            f"{free_bytes} < {required_free_bytes}"
        )
    if expected_device is not None and directory.st_dev != expected_device:
        raise CheckpointError("destination filesystem device changed")
    if expected_fsid is not None and fsid != expected_fsid:
        raise CheckpointError("destination filesystem fsid changed")
    return {
        "anchor_device": int(directory.st_dev),
        "bavail": int(values.f_bavail),
        "block_size": int(values.f_frsize),
        "free_bytes": free_bytes,
        "fsid": fsid,
        "readonly": readonly,
        "required_free_bytes": required_free_bytes,
    }


def _assert_quota(
    quota: dict[str, int | str],
    *,
    budget_bytes: int,
    budget_inodes: int,
    minimum_soft_headroom_bytes: int,
    minimum_hard_headroom_bytes: int,
    minimum_soft_inode_headroom: int,
    minimum_hard_inode_headroom: int,
) -> dict[str, int]:
    usage = int(quota["usage_bytes"])
    in_doubt = int(quota["in_doubt_bytes"])
    files_used = int(quota["files_used"])
    files_in_doubt = int(quota["files_in_doubt"])
    shadow = {
        "hard_headroom_after_bytes": (
            int(quota["hard_limit_bytes"]) - usage - in_doubt - budget_bytes
        ),
        "hard_inode_headroom_after": (
            int(quota["files_hard_limit"]) - files_used - files_in_doubt - budget_inodes
        ),
        "soft_headroom_after_bytes": (
            int(quota["soft_limit_bytes"]) - usage - in_doubt - budget_bytes
        ),
        "soft_inode_headroom_after": (
            int(quota["files_soft_limit"]) - files_used - files_in_doubt - budget_inodes
        ),
    }
    limits = {
        "hard_headroom_after_bytes": minimum_hard_headroom_bytes,
        "hard_inode_headroom_after": minimum_hard_inode_headroom,
        "soft_headroom_after_bytes": minimum_soft_headroom_bytes,
        "soft_inode_headroom_after": minimum_soft_inode_headroom,
    }
    failing = {
        name: (shadow[name], minimum)
        for name, minimum in limits.items()
        if shadow[name] < minimum
    }
    if failing:
        raise CheckpointError(f"quota shadow below contract: {failing}")
    return shadow


def _assert_ancestor_chain(path: Path) -> None:
    absolute = path.absolute()
    chain = tuple(reversed(absolute.parents)) + (absolute,)
    for current in chain:
        if not _lexists(current):
            raise CheckpointError(f"destination ancestor does not exist: {current}")
        _lstat_real_directory(current, "destination ancestor")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.realpath(left))
    right_text = os.path.normcase(os.path.realpath(right))
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common in (left_text, right_text)


def _validate_destination(
    source: Path,
    destination: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[Path, Path]:
    if allowed_root is None:
        allowed_root = ALLOWED_DESTINATION_ROOT
    raw = os.fspath(destination)
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw:
        raise CheckpointError("destination must be an absolute normalized path")
    allowed_raw = os.fspath(allowed_root)
    if os.path.dirname(raw) != allowed_raw or Path(raw).parent != allowed_root:
        raise CheckpointError(
            f"destination must be a direct child of literal root {allowed_root}"
        )
    if not Path(raw).name or Path(raw).name.startswith("."):
        raise CheckpointError("destination child name is invalid")
    _assert_ancestor_chain(allowed_root.parent)
    if _lexists(allowed_root):
        _lstat_real_directory(allowed_root, "allowed destination root")
        if allowed_root.resolve(strict=True) != allowed_root.absolute():
            raise CheckpointError("allowed destination root resolves through a symlink")
    else:
        resolved_parent = allowed_root.parent.resolve(strict=True)
        if resolved_parent != allowed_root.parent.absolute():
            raise CheckpointError("allowed-root parent resolves through a symlink")
    resolved_source = source.resolve(strict=True)
    resolved_destination = allowed_root.resolve(strict=False) / Path(raw).name
    if _paths_overlap(resolved_source, resolved_destination):
        raise CheckpointError("source and destination trees overlap")
    return resolved_destination, allowed_root


def _acl_lines(path: Path) -> tuple[str, ...]:
    process = subprocess.run(
        ["getfacl", "-cp", os.fspath(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise CheckpointError(f"getfacl failed for {path}: {process.stderr[-1000:]}")
    return tuple(
        row.strip()
        for row in process.stdout.splitlines()
        if row.strip() and not row.startswith("#")
    )


def _assert_private_directory(path: Path) -> None:
    value = _lstat_real_directory(path, "private destination root")
    if value.st_uid != os.getuid() or value.st_gid != os.getgid():
        raise CheckpointError(f"destination ownership mismatch: {path}")
    if stat.S_IMODE(value.st_mode) != 0o700:
        raise CheckpointError(f"destination root mode is not 0700: {path}")
    if _acl_lines(path) != ("user::rwx", "group::---", "other::---"):
        raise CheckpointError(f"destination root ACL is not minimal: {path}")
    if path.resolve(strict=True) != path.absolute():
        raise CheckpointError(f"destination root is symlink-resolved: {path}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise FileExistsError(errno.EEXIST, "destination exists", str(destination))
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CheckpointError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def _assert_lstat_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CheckpointError(f"cannot prove {label} absent: {path}: {exc}") from exc
    raise FileExistsError(errno.EEXIST, f"{label} exists", str(path))


def _mountinfo_unescape(value: str) -> str:
    return (
        value.replace(r"\040", " ")
        .replace(r"\011", "\t")
        .replace(r"\012", "\n")
        .replace(r"\134", "\\")
    )


def _gpfs_mount_identity(
    path: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, str]:
    target = os.path.realpath(path)
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckpointError("cannot authenticate GPFS via mountinfo") from exc
    matches: list[tuple[int, str, str, str]] = []
    for line in lines:
        try:
            left, right = line.split(" - ", 1)
            left_fields = left.split()
            right_fields = right.split()
            mount_point = _mountinfo_unescape(left_fields[4])
            filesystem_type = right_fields[0]
            mount_source = right_fields[1]
            if os.path.commonpath((target, mount_point)) != mount_point:
                continue
        except (IndexError, ValueError):
            continue
        matches.append(
            (len(os.path.normpath(mount_point)), mount_point, filesystem_type, mount_source)
        )
    if not matches:
        raise CheckpointError(f"no mountinfo entry covers guarded rename: {path}")
    _, mount_point, filesystem_type, mount_source = max(matches)
    if filesystem_type != "gpfs":
        raise CheckpointError(
            f"guarded plain rename fallback requires gpfs, got {filesystem_type}"
        )
    return {
        "filesystem_type": filesystem_type,
        "mount_point": mount_point,
        "mount_source": mount_source,
    }


def _claim_target_identity(path: Path) -> dict[str, Any]:
    if not _lexists(path):
        return {"exists": False}
    value = path.lstat()
    result: dict[str, Any] = {
        "device": int(value.st_dev),
        "exists": True,
        "inode": int(value.st_ino),
        "mode": oct(stat.S_IMODE(value.st_mode)),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
    }
    manifest = path / ".checkpoint_manifest.json"
    try:
        result["manifest_sha256"] = _full_sha256(manifest)
    except (CheckpointError, OSError):
        result["manifest_sha256"] = None
    return result


def _finalize_rename_claim(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_RDWR | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise CheckpointError(f"rename claim identity mismatch: {path}")
        encoded = _canonical_json(payload)
        os.ftruncate(descriptor, 0)
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or stat.S_IMODE(after.st_mode) != 0o400
        ):
            raise CheckpointError("rename claim final metadata mismatch")
    finally:
        os.close(descriptor)
    path_value = path.lstat()
    if (
        path_value.st_dev != before.st_dev
        or path_value.st_ino != before.st_ino
        or stat.S_IMODE(path_value.st_mode) != 0o400
    ):
        raise CheckpointError("rename claim path identity mismatch")


def _guarded_rename_noreplace(
    source: Path,
    destination: Path,
    *,
    operation: str,
    tool_payload_sha256: str,
) -> dict[str, Any]:
    """Rename without replacement, with a narrow GPFS EINVAL fallback.

    The plain-rename fallback is safe only under this tool's cooperative
    same-UID threat model: the shared parent is owner-only 0700 and an O_EXCL
    owner-only claim serializes all cooperating publishers.
    """
    _assert_sha256(tool_payload_sha256, "rename tool payload SHA-256")
    if source.parent != destination.parent:
        raise CheckpointError("guarded rename source/destination parents differ")
    parent = source.parent
    try:
        _rename_noreplace(source, destination)
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
    else:
        _fsync_directory(parent)
        return {
            "claim": None,
            "method": "renameat2-rename-noreplace",
            "operation": operation,
            "threat_model": "kernel-enforced destination nonreplacement",
        }

    _assert_private_directory(parent)
    mount_identity = _gpfs_mount_identity(parent)
    source_value = _lstat_real_directory(source, "guarded rename source")
    if source_value.st_uid != os.getuid() or source_value.st_gid != os.getgid():
        raise CheckpointError("guarded rename source ownership mismatch")
    if not operation or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in operation
    ):
        raise CheckpointError("guarded rename operation name is invalid")
    claim = parent / (
        f".{destination.name}.rename.claim.{operation}."
        f"{tool_payload_sha256[:12]}"
    )
    started_epoch = time.time()
    claim_base = {
        "destination": str(destination),
        "method": "gpfs-einval-guarded-plain-rename",
        "mount_identity": mount_identity,
        "operation": operation,
        "pid": os.getpid(),
        "source": str(source),
        "started_epoch": started_epoch,
        "threat_model": (
            "cooperative same-UID publishers under owner-only 0700 parent "
            "and O_EXCL claim; final lstat absence checked immediately "
            "before plain rename"
        ),
        "tool_payload_sha256": tool_payload_sha256,
    }
    _write_exclusive(
        claim,
        _canonical_json({**claim_base, "status": "claimed-not-finalized"}),
    )
    renamed = False
    try:
        _fsync_directory(parent)
        _assert_private_directory(parent)
        current_source = _lstat_real_directory(source, "guarded rename source")
        if (
            current_source.st_dev != source_value.st_dev
            or current_source.st_ino != source_value.st_ino
            or current_source.st_uid != os.getuid()
            or current_source.st_gid != os.getgid()
        ):
            raise CheckpointError("guarded rename source changed under claim")
        # This is deliberately the final operation before plain rename.  A
        # collision is rejected; plain rename is allowed only because the
        # parent and O_EXCL claim exclude cooperating same-UID publishers.
        _assert_lstat_absent(destination, "guarded rename destination")
        os.rename(source, destination)
        renamed = True
        _fsync_directory(parent)
        published = _lstat_real_directory(destination, "guarded rename destination")
        if (
            published.st_dev != source_value.st_dev
            or published.st_ino != source_value.st_ino
            or published.st_uid != os.getuid()
            or published.st_gid != os.getgid()
            or _lexists(source)
        ):
            raise CheckpointError("guarded plain rename postcondition mismatch")
        completed_epoch = time.time()
        final_identity = _claim_target_identity(destination)
        _finalize_rename_claim(
            claim,
            {
                **claim_base,
                "completed_epoch": completed_epoch,
                "destination_identity": final_identity,
                "source_exists_after": _lexists(source),
                "status": "complete",
            },
        )
        return {
            "claim": str(claim),
            "claim_sha256": _full_sha256(claim),
            "destination_identity": final_identity,
            "method": "gpfs-einval-guarded-plain-rename",
            "mount_identity": mount_identity,
            "operation": operation,
            "threat_model": claim_base["threat_model"],
        }
    except BaseException as original:
        try:
            _finalize_rename_claim(
                claim,
                {
                    **claim_base,
                    "completed_epoch": time.time(),
                    "destination_identity": _claim_target_identity(destination),
                    "error": f"{type(original).__name__}: {original}",
                    "source_identity": _claim_target_identity(source),
                    "status": "failed-no-delete",
                },
            )
        except BaseException as claim_error:
            raise CheckpointError(
                "guarded rename failed and claim finalization failed; "
                f"rename={original}; claim={claim_error}"
            ) from original
        raise
    finally:
        if renamed and _lexists(source):
            raise CheckpointError("guarded rename left both source and destination")


def _stat_matches_source(value: os.stat_result, expected: StaticFile) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and not stat.S_ISLNK(value.st_mode)
        and value.st_nlink == 1
        and value.st_size == expected.size
        and _allocated_bytes(value) == expected.allocated
        and value.st_mtime_ns == expected.mtime_ns
        and oct(stat.S_IMODE(value.st_mode)) == expected.mode
        and value.st_ino == expected.inode
        and value.st_dev == expected.device
    )


def _hash_open_fd(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, COPY_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _copy_stable_file(
    source_root: Path,
    destination_root: Path,
    expected: StaticFile,
) -> str:
    source = source_root / expected.path
    destination = destination_root / expected.path
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_fd = os.open(
        source,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not _stat_matches_source(os.fstat(source_fd), expected):
            raise CheckpointError(f"opened source identity mismatch: {expected.path}")
        pre_hash = _hash_open_fd(source_fd)
        if not _stat_matches_source(os.fstat(source_fd), expected):
            raise CheckpointError(f"source changed during prehash: {expected.path}")
        os.lseek(source_fd, 0, os.SEEK_SET)
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | BINARY_FLAG
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        during_digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(source_fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                during_digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(destination_fd, view) :]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        if not _stat_matches_source(os.fstat(source_fd), expected):
            raise CheckpointError(f"source changed during copy: {expected.path}")
        post_hash = _hash_open_fd(source_fd)
        if not _stat_matches_source(os.fstat(source_fd), expected):
            raise CheckpointError(f"source changed during posthash: {expected.path}")
    finally:
        os.close(source_fd)
    hashes = (pre_hash, during_digest.hexdigest(), post_hash)
    if len(set(hashes)) != 1:
        raise CheckpointError(
            f"source pre/copy/post content mismatch for {expected.path}"
        )
    copied = destination.lstat()
    if (
        not stat.S_ISREG(copied.st_mode)
        or stat.S_ISLNK(copied.st_mode)
        or copied.st_nlink != 1
        or copied.st_size != expected.size
    ):
        raise CheckpointError(f"copied destination identity mismatch: {expected.path}")
    try:
        os.utime(
            destination,
            ns=(expected.mtime_ns, expected.mtime_ns),
            follow_symlinks=False,
        )
    except NotImplementedError:
        if os.name != "nt":
            raise
        os.utime(destination, ns=(expected.mtime_ns, expected.mtime_ns))
    return pre_hash


def _full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise CheckpointError(f"hash target is not a single-link file: {path}")
        while True:
            chunk = os.read(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _hash_static_source_file(source_root: Path, expected: StaticFile) -> str:
    path = source_root / expected.path
    descriptor = os.open(
        path,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not _stat_matches_source(os.fstat(descriptor), expected):
            raise CheckpointError(
                f"source identity differs before resume hash: {expected.path}"
            )
        digest = _hash_open_fd(descriptor)
        if not _stat_matches_source(os.fstat(descriptor), expected):
            raise CheckpointError(
                f"source identity differs after resume hash: {expected.path}"
            )
    finally:
        os.close(descriptor)
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _chmod_nofollow(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode, follow_symlinks=False)
    except NotImplementedError:
        if os.name != "nt":
            raise
        os.chmod(path, mode)


def _protect_path_fd(path: Path, mode: int, *, directory: bool) -> None:
    """Protect and fsync one already-admitted object through its open fd."""
    if os.name == "nt":
        # Local Windows tests have neither POSIX directory fds nor durable
        # read-only fsync semantics.  GPFS production never enters this branch.
        _chmod_nofollow(path, mode)
        if directory:
            _fsync_directory(path)
        else:
            _fsync_file(path)
        return
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if os.name != "nt" and nofollow is None:
        raise CheckpointError("O_NOFOLLOW is unavailable for fd protection")
    if directory and os.name != "nt" and directory_flag is None:
        raise CheckpointError("O_DIRECTORY is unavailable for fd protection")
    flags = os.O_RDONLY | BINARY_FLAG | (nofollow or 0)
    if directory:
        flags |= directory_flag or 0
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"cannot open checkpoint object safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_kind(before.st_mode):
            raise CheckpointError(f"checkpoint object type mismatch: {path}")
        if not directory and before.st_nlink != 1:
            raise CheckpointError(f"checkpoint file link count mismatch: {path}")
        if before.st_uid != os.getuid() or before.st_gid != os.getgid():
            raise CheckpointError(f"checkpoint object ownership mismatch: {path}")
        path_before = path.lstat()
        if (
            path_before.st_dev != before.st_dev
            or path_before.st_ino != before.st_ino
            or stat.S_ISLNK(path_before.st_mode)
        ):
            raise CheckpointError(f"checkpoint object changed while opening: {path}")
        if directory:
            if stat.S_IMODE(before.st_mode) not in TRANSITION_DIRECTORY_MODES:
                raise CheckpointError(
                    f"directory is outside transitional modes: {path}"
                )
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or stat.S_IMODE(after.st_mode) != mode
            ):
                raise CheckpointError(f"fd protection verification failed: {path}")
        else:
            if mode != 0o400:
                raise CheckpointError("regular-file final mode must be 0400")
            if stat.S_IMODE(before.st_mode) not in TRANSITION_FILE_MODES:
                raise CheckpointError(f"file is outside transitional modes: {path}")
            # fsync on an O_RDONLY regular-file fd may return EBADF.  Elevate
            # only this already-open inode to owner read/write, then reopen the
            # same path O_RDWR|O_NOFOLLOW and prove the inode identity again.
            if stat.S_IMODE(before.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
            writable = os.open(
                path,
                os.O_RDWR | BINARY_FLAG | (nofollow or 0),
            )
            try:
                writable_value = os.fstat(writable)
                if (
                    not stat.S_ISREG(writable_value.st_mode)
                    or writable_value.st_nlink != 1
                    or writable_value.st_uid != os.getuid()
                    or writable_value.st_gid != os.getgid()
                    or writable_value.st_dev != before.st_dev
                    or writable_value.st_ino != before.st_ino
                    or stat.S_IMODE(writable_value.st_mode) != 0o600
                ):
                    raise CheckpointError(
                        f"writable reopen identity mismatch: {path}"
                    )
                os.fsync(writable)
                os.fchmod(writable, 0o400)
                os.fsync(writable)
                after = os.fstat(writable)
                if (
                    after.st_dev != before.st_dev
                    or after.st_ino != before.st_ino
                    or stat.S_IMODE(after.st_mode) != 0o400
                ):
                    raise CheckpointError(
                        f"fd protection verification failed: {path}"
                    )
            except BaseException:
                try:
                    current = os.fstat(writable)
                    if (
                        current.st_dev == before.st_dev
                        and current.st_ino == before.st_ino
                    ):
                        os.fchmod(writable, 0o400)
                        os.fsync(writable)
                except OSError:
                    pass
                raise
            finally:
                os.close(writable)
    finally:
        os.close(descriptor)
    path_after = path.lstat()
    if (
        path_after.st_dev != before.st_dev
        or path_after.st_ino != before.st_ino
        or stat.S_IMODE(path_after.st_mode) != mode
    ):
        raise CheckpointError(f"path changed after fd protection: {path}")


def _walk_real_tree(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    directories: list[Path] = []
    files: list[Path] = []
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        _lstat_real_directory(current, "checkpoint directory")
        directories.append(current)
        for name in tuple(dir_names):
            child = current / name
            value = child.lstat()
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                raise CheckpointError(f"non-real child directory: {child}")
        for name in file_names:
            child = current / name
            value = child.lstat()
            if (
                stat.S_ISLNK(value.st_mode)
                or not stat.S_ISREG(value.st_mode)
                or value.st_nlink != 1
            ):
                raise CheckpointError(f"non-contract child file: {child}")
            files.append(child)
    return tuple(sorted(directories)), tuple(sorted(files))


def _fsync_and_protect_tree(root: Path) -> None:
    directories, files = _walk_real_tree(root)
    for path in files:
        _protect_path_fd(path, 0o400, directory=False)
    for path in reversed(directories):
        _protect_path_fd(path, 0o500, directory=True)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | BINARY_FLAG
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_protected_tree(
    root: Path,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    directories, files = _walk_real_tree(root)
    expected_files = set(expected_hashes) | {".checkpoint_manifest.json"}
    actual_files = {path.relative_to(root).as_posix() for path in files}
    if actual_files != expected_files:
        raise CheckpointError(
            "checkpoint children mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_directories = set(_required_directories(expected_hashes))
    actual_directories = {
        "." if path == root else path.relative_to(root).as_posix()
        for path in directories
    }
    if actual_directories != expected_directories:
        raise CheckpointError("checkpoint directory set mismatch")
    uid = os.getuid()
    gid = os.getgid()
    for path in directories:
        value = path.lstat()
        if (
            value.st_uid != uid
            or value.st_gid != gid
            or stat.S_IMODE(value.st_mode) != 0o500
        ):
            raise CheckpointError(f"checkpoint directory metadata mismatch: {path}")
    verified_hashes: dict[str, str] = {}
    for path in files:
        value = path.lstat()
        if (
            value.st_uid != uid
            or value.st_gid != gid
            or stat.S_IMODE(value.st_mode) != 0o400
            or value.st_nlink != 1
        ):
            raise CheckpointError(f"checkpoint file metadata mismatch: {path}")
        relative = path.relative_to(root).as_posix()
        verified_hashes[relative] = _full_sha256(path)
        if relative != ".checkpoint_manifest.json":
            if verified_hashes[relative] != expected_hashes[relative]:
                raise CheckpointError(f"checkpoint content mismatch: {relative}")
    return {
        "directory_count": len(directories),
        "file_count": len(files),
        "manifest_sha256": verified_hashes[".checkpoint_manifest.json"],
    }


def _quarantine_partial(partial: Path, destination: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tool_payload_sha256 = _full_sha256(Path(__file__).resolve())
    for counter in range(100_000):
        quarantine = destination.with_name(
            f"{destination.name}.incomplete.{stamp}.{os.getpid()}.{counter:05d}"
        )
        try:
            _guarded_rename_noreplace(
                partial,
                quarantine,
                operation="quarantine-partial-no-delete",
                tool_payload_sha256=tool_payload_sha256,
            )
            return quarantine
        except FileExistsError:
            continue
    raise CheckpointError(
        f"cannot allocate deterministic quarantine name for partial {partial}"
    )


def quarantine_published_checkpoint(
    checkpoint: Path,
    *,
    expected_manifest_sha256: str,
    reason: str,
) -> dict[str, Any]:
    raw = os.fspath(checkpoint)
    allowed_root = ALLOWED_DESTINATION_ROOT
    if (
        not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
        or checkpoint.parent != allowed_root
        or ".incomplete." in checkpoint.name
    ):
        raise CheckpointError("published checkpoint is not an allowed final child")
    _assert_private_directory(allowed_root)
    value = _lstat_real_directory(checkpoint, "published checkpoint")
    if (
        value.st_uid != os.getuid()
        or value.st_gid != os.getgid()
        or stat.S_IMODE(value.st_mode) != 0o500
    ):
        raise CheckpointError("published checkpoint ownership/mode mismatch")
    manifest = checkpoint / ".checkpoint_manifest.json"
    actual_manifest_sha256 = _full_sha256(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise CheckpointError("published checkpoint manifest SHA-256 mismatch")
    quarantine = _quarantine_partial(checkpoint, checkpoint)
    return {
        "expected_manifest_sha256": expected_manifest_sha256,
        "quarantine": str(quarantine),
        "reason": reason,
        "source_final": str(checkpoint),
        "status": "quarantined-no-delete",
    }


def _read_owned_regular(
    path: Path,
    *,
    allowed_modes: frozenset[int],
    maximum_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        path,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
        ):
            raise CheckpointError(f"owned regular-file contract mismatch: {path}")
        if os.name != "nt" and stat.S_IMODE(before.st_mode) not in allowed_modes:
            raise CheckpointError(f"transitional file mode mismatch: {path}")
        path_value = path.lstat()
        if (
            stat.S_ISLNK(path_value.st_mode)
            or path_value.st_dev != before.st_dev
            or path_value.st_ino != before.st_ino
        ):
            raise CheckpointError(f"file identity changed while opening: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise CheckpointError(f"file exceeds read budget: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, min(COPY_CHUNK_BYTES, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise CheckpointError(f"file changed while reading: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks), before


def _hash_owned_resume_file(
    path: Path,
    *,
    expected_size: int,
    allowed_modes: frozenset[int] = TRANSITION_FILE_MODES,
) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY | BINARY_FLAG | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or before.st_size != expected_size
        ):
            raise CheckpointError(f"incoming file identity/size mismatch: {path}")
        if os.name != "nt" and stat.S_IMODE(before.st_mode) not in allowed_modes:
            raise CheckpointError(f"incoming file mode mismatch: {path}")
        path_value = path.lstat()
        if (
            stat.S_ISLNK(path_value.st_mode)
            or path_value.st_dev != before.st_dev
            or path_value.st_ino != before.st_ino
        ):
            raise CheckpointError(f"incoming file changed while opening: {path}")
        digest = _hash_open_fd(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_nlink != 1
        ):
            raise CheckpointError(f"incoming file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest


def _assert_manifest_snapshot(
    summary: Any,
    *,
    source: Path,
    snapshot: Snapshot,
    label: str,
) -> None:
    if not isinstance(summary, dict):
        raise CheckpointError(f"manifest {label} snapshot is not an object")
    expected = {
        "allocated_bytes": snapshot.allocated_bytes,
        "directory_count": len(snapshot.directories),
        "included_count": len(snapshot.files),
        "included_writer_count": 0,
        "logical_bytes": snapshot.logical_bytes,
        "metadata_sha256": snapshot.metadata_sha256,
        "required_budget_bytes": snapshot.required_budget_bytes,
        "required_inodes": snapshot.required_inodes,
        "root": str(source),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise CheckpointError(f"manifest {label} snapshot mismatch: {key}")
    if set(summary) != set(expected) | {"observed_epoch", "writer_count"}:
        raise CheckpointError(f"manifest {label} snapshot fields mismatch")
    observed = summary["observed_epoch"]
    writer_count = summary["writer_count"]
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
        or type(writer_count) is not int
        or writer_count < 0
    ):
        raise CheckpointError(f"manifest {label} snapshot values invalid")


def _load_and_validate_resume_manifest(
    incoming: Path,
    *,
    source: Path,
    destination: Path,
    snapshot: Snapshot,
    expected_manifest_sha256: str,
    expected_manifest_payload_sha256: str,
    expected_checkpoint_tool_payload_sha256: str,
    expected_aedt_sha256: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, int]]:
    expected_manifest_sha256 = _assert_sha256(
        expected_manifest_sha256, "expected manifest SHA-256"
    )
    expected_manifest_payload_sha256 = _assert_sha256(
        expected_manifest_payload_sha256,
        "expected manifest payload SHA-256",
    )
    expected_checkpoint_tool_payload_sha256 = _assert_sha256(
        expected_checkpoint_tool_payload_sha256,
        "expected checkpoint tool payload SHA-256",
    )
    expected_aedt_sha256 = _assert_sha256(
        expected_aedt_sha256, "expected AEDT SHA-256"
    )
    manifest_path = incoming / ".checkpoint_manifest.json"
    raw, _ = _read_owned_regular(
        manifest_path,
        allowed_modes=TRANSITION_FILE_MODES,
        maximum_bytes=MANIFEST_BUDGET_BYTES,
    )
    if _sha256_bytes(raw) != expected_manifest_sha256:
        raise CheckpointError("incoming manifest SHA-256 mismatch")
    try:
        manifest = json.loads(raw, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointError("incoming manifest is not valid strict JSON") from exc
    if not isinstance(manifest, dict):
        raise CheckpointError("incoming manifest is not an object")
    if raw != _canonical_json(manifest):
        raise CheckpointError("incoming manifest is not canonical JSON")
    payload_manifest = dict(manifest)
    claimed_payload = payload_manifest.pop("manifest_payload_sha256", None)
    actual_payload = _sha256_bytes(_canonical_json(payload_manifest))
    if (
        claimed_payload != expected_manifest_payload_sha256
        or actual_payload != expected_manifest_payload_sha256
    ):
        raise CheckpointError("incoming manifest payload SHA-256 mismatch")
    required_keys = {
        "canonical",
        "checkpoint_creator",
        "completed_epoch",
        "composite_reauthentication_required",
        "destination",
        "diagnostic_only",
        "files",
        "filesystem_after_copy",
        "filesystem_before",
        "manifest_payload_sha256",
        "physics_boundary",
        "post_login_quota_reauthentication_required",
        "quota_admission",
        "quota_before_login_evidence",
        "schema_version",
        "source",
        "source_provenance",
        "source_snapshot_after",
        "source_snapshot_before",
        "started_epoch",
        "traversal_policy",
    }
    if set(manifest) != required_keys:
        raise CheckpointError("incoming manifest top-level fields mismatch")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError("incoming manifest schema mismatch")
    if manifest["source"] != str(source) or manifest["destination"] != str(destination):
        raise CheckpointError("incoming manifest source/destination mismatch")
    if (
        manifest["canonical"] is not False
        or manifest["diagnostic_only"] is not True
        or manifest["composite_reauthentication_required"] is not True
        or manifest["post_login_quota_reauthentication_required"] is not True
    ):
        raise CheckpointError("incoming manifest checkpoint policy mismatch")
    if manifest["checkpoint_creator"] != {
        "tool_payload_sha256": expected_checkpoint_tool_payload_sha256
    }:
        raise CheckpointError("incoming manifest checkpoint tool SHA-256 mismatch")
    expected_provenance = {
        **SOURCE_PROVENANCE,
        "source_project_sha256": expected_aedt_sha256,
        "source_static_metadata_sha256": snapshot.metadata_sha256,
    }
    if manifest["source_provenance"] != expected_provenance:
        raise CheckpointError("incoming manifest source provenance mismatch")
    if manifest["physics_boundary"] != {
        "fan_velocity_m_per_s": 1.5,
        "thermal_pad_thickness_mm": 2.0,
        "tim_conductivity_w_per_mk": 0.2,
    }:
        raise CheckpointError("incoming manifest physics boundary mismatch")
    if manifest["traversal_policy"] != {
        "allowlist_only": True,
        "excluded_directories_pruned": True,
        "mesh_directory_children": list(MESH_FILES),
    }:
        raise CheckpointError("incoming manifest traversal policy mismatch")
    _assert_manifest_snapshot(
        manifest["source_snapshot_before"],
        source=source,
        snapshot=snapshot,
        label="before",
    )
    _assert_manifest_snapshot(
        manifest["source_snapshot_after"],
        source=source,
        snapshot=snapshot,
        label="after",
    )

    files = manifest["files"]
    if not isinstance(files, list) or len(files) != len(snapshot.files):
        raise CheckpointError("incoming manifest file count mismatch")
    expected_rows = {row.path: row for row in snapshot.files}
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    row_keys = set(StaticFile.__dataclass_fields__) | {"sha256"}
    for item in files:
        if not isinstance(item, dict) or set(item) != row_keys:
            raise CheckpointError("incoming manifest file row fields mismatch")
        relative = item.get("path")
        if not isinstance(relative, str) or relative not in expected_rows:
            raise CheckpointError("incoming manifest file path mismatch")
        if relative in hashes:
            raise CheckpointError("incoming manifest duplicate file path")
        expected_row = expected_rows[relative]
        metadata = {key: item[key] for key in StaticFile.__dataclass_fields__}
        if metadata != asdict(expected_row):
            raise CheckpointError(f"incoming manifest file metadata mismatch: {relative}")
        hashes[relative] = _assert_sha256(
            item["sha256"], f"incoming manifest file hash {relative}"
        )
        sizes[relative] = expected_row.size
    if set(hashes) != set(expected_rows):
        raise CheckpointError("incoming manifest file set mismatch")
    return manifest, hashes, sizes


def _verify_resume_tree_before_protect(
    incoming: Path,
    *,
    hashes: dict[str, str],
    sizes: dict[str, int],
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    directories, files = _walk_real_tree(incoming)
    expected_files = set(hashes) | {".checkpoint_manifest.json"}
    actual_files = {path.relative_to(incoming).as_posix() for path in files}
    if actual_files != expected_files:
        raise CheckpointError(
            "incoming children mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_directories = set(_required_directories(hashes))
    actual_directories = {
        "." if path == incoming else path.relative_to(incoming).as_posix()
        for path in directories
    }
    if actual_directories != expected_directories:
        raise CheckpointError("incoming directory set mismatch")
    for directory in directories:
        value = directory.lstat()
        if value.st_uid != os.getuid() or value.st_gid != os.getgid():
            raise CheckpointError(f"incoming directory ownership mismatch: {directory}")
        if (
            os.name != "nt"
            and stat.S_IMODE(value.st_mode) not in TRANSITION_DIRECTORY_MODES
        ):
            raise CheckpointError(f"incoming directory mode mismatch: {directory}")
    verified: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(incoming).as_posix()
        size = path.lstat().st_size if relative == ".checkpoint_manifest.json" else sizes[relative]
        verified[relative] = _hash_owned_resume_file(path, expected_size=size)
        expected = (
            expected_manifest_sha256
            if relative == ".checkpoint_manifest.json"
            else hashes[relative]
        )
        if verified[relative] != expected:
            raise CheckpointError(f"incoming content hash mismatch: {relative}")
    return {
        "directory_count": len(directories),
        "file_count": len(files),
        "manifest_sha256": verified[".checkpoint_manifest.json"],
    }


def resume_protect_publish(
    *,
    source: Path,
    destination: Path,
    expected_metadata_sha256: str,
    expected_count: int,
    expected_directory_count: int,
    expected_logical_bytes: int,
    expected_allocated_bytes: int,
    expected_aedt_sha256: str,
    expected_manifest_sha256: str,
    expected_manifest_payload_sha256: str,
    expected_checkpoint_tool_payload_sha256: str,
    resume_tool_payload_sha256: str,
    login_quota_evidence_base64: str,
    login_quota_evidence_sha256: str,
    login_quota_evidence_max_age_seconds: int,
    minimum_soft_headroom_bytes: int,
    minimum_hard_headroom_bytes: int,
    minimum_soft_inode_headroom: int,
    minimum_hard_inode_headroom: int,
    read_only_preflight: bool = False,
) -> dict[str, Any]:
    """Resume only protection/publication of one already-copied exact incoming."""
    if type(read_only_preflight) is not bool:
        raise CheckpointError("read_only_preflight must be an exact boolean")
    started = time.time()
    source = source.resolve(strict=True)
    destination, allowed_root = _validate_destination(source, destination)
    if not _lexists(allowed_root):
        raise CheckpointError("allowed destination root is unavailable")
    _assert_private_directory(allowed_root)
    incoming = destination.with_name(destination.name + ".incoming")
    _assert_lstat_absent(destination, "final destination")
    incoming_stat = _lstat_real_directory(incoming, "resume incoming")
    if (
        incoming_stat.st_uid != os.getuid()
        or incoming_stat.st_gid != os.getgid()
        or (
            os.name != "nt"
            and stat.S_IMODE(incoming_stat.st_mode) not in TRANSITION_DIRECTORY_MODES
        )
    ):
        raise CheckpointError("resume incoming ownership/mode mismatch")
    resume_tool_payload_sha256 = _assert_sha256(
        resume_tool_payload_sha256, "resume tool payload SHA-256"
    )
    if _full_sha256(Path(__file__).resolve()) != resume_tool_payload_sha256:
        raise CheckpointError("resume tool payload SHA-256 self-check mismatch")

    before = snapshot_static_source(source)
    _assert_expected_snapshot(
        before,
        metadata_sha256=expected_metadata_sha256,
        count=expected_count,
        directory_count=expected_directory_count,
        logical_bytes=expected_logical_bytes,
        allocated_bytes=expected_allocated_bytes,
    )
    manifest, hashes, sizes = _load_and_validate_resume_manifest(
        incoming,
        source=source,
        destination=destination,
        snapshot=before,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_payload_sha256=expected_manifest_payload_sha256,
        expected_checkpoint_tool_payload_sha256=(
            expected_checkpoint_tool_payload_sha256
        ),
        expected_aedt_sha256=expected_aedt_sha256,
    )
    incoming_verification = _verify_resume_tree_before_protect(
        incoming,
        hashes=hashes,
        sizes=sizes,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    source_hashes = {
        row.path: _hash_static_source_file(source, row) for row in before.files
    }
    if source_hashes != hashes:
        raise CheckpointError("fresh source hashes differ from incoming manifest")
    aedt = next(row for row in before.files if row.path.endswith(".aedt"))
    if source_hashes[aedt.path] != expected_aedt_sha256:
        raise CheckpointError("fresh source AEDT SHA-256 differs from admission")
    after_hash = snapshot_static_source(source)
    _assert_expected_snapshot(
        after_hash,
        metadata_sha256=expected_metadata_sha256,
        count=expected_count,
        directory_count=expected_directory_count,
        logical_bytes=expected_logical_bytes,
        allocated_bytes=expected_allocated_bytes,
    )
    _assert_lstat_absent(destination, "final destination after hashing")

    # Decode this as late as possible: evidence older than 120 seconds fails
    # before any chmod or rename mutation.
    quota_before_publish = _decode_login_quota_evidence(
        login_quota_evidence_base64,
        expected_canonical_sha256=login_quota_evidence_sha256,
        maximum_age_seconds=login_quota_evidence_max_age_seconds,
    )
    quota_admission = _assert_quota(
        quota_before_publish,
        budget_bytes=MANIFEST_BUDGET_BYTES,
        budget_inodes=1,
        minimum_soft_headroom_bytes=minimum_soft_headroom_bytes,
        minimum_hard_headroom_bytes=minimum_hard_headroom_bytes,
        minimum_soft_inode_headroom=minimum_soft_inode_headroom,
        minimum_hard_inode_headroom=minimum_hard_inode_headroom,
    )
    historical_filesystem = manifest["filesystem_after_copy"]
    if not isinstance(historical_filesystem, dict):
        raise CheckpointError("manifest filesystem evidence is invalid")
    filesystem_before_publish = _filesystem_capacity(
        allowed_root,
        required_free_bytes=PHYSICAL_FREE_RESERVE_BYTES,
        expected_device=int(historical_filesystem["anchor_device"]),
        expected_fsid=int(historical_filesystem["fsid"]),
    )
    if read_only_preflight:
        return {
            "destination": str(destination),
            "elapsed_seconds": time.time() - started,
            "filesystem_before_publish": filesystem_before_publish,
            "incoming_preprotect_verification": incoming_verification,
            "manifest_payload_sha256": expected_manifest_payload_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "mutation_count": 0,
            "quota_admission": quota_admission,
            "resume_tool_payload_sha256": resume_tool_payload_sha256,
            "schema_version": SCHEMA_VERSION,
            "source_snapshot_after_hash": after_hash.summary(),
            "status": "read-only-preflight-complete",
        }

    published = False
    rename_receipt: dict[str, Any] | None = None
    try:
        _fsync_and_protect_tree(incoming)
        protected = _verify_protected_tree(incoming, hashes)
        if protected["manifest_sha256"] != expected_manifest_sha256:
            raise CheckpointError("protected manifest SHA-256 mismatch")
        _assert_lstat_absent(destination, "final destination before publish")
        rename_receipt = _guarded_rename_noreplace(
            incoming,
            destination,
            operation="resume-protect-publish",
            tool_payload_sha256=resume_tool_payload_sha256,
        )
        published = True
        final = _verify_protected_tree(destination, hashes)
        if final["manifest_sha256"] != expected_manifest_sha256:
            raise CheckpointError("published manifest SHA-256 mismatch")
        source_final = snapshot_static_source(source)
        _assert_expected_snapshot(
            source_final,
            metadata_sha256=expected_metadata_sha256,
            count=expected_count,
            directory_count=expected_directory_count,
            logical_bytes=expected_logical_bytes,
            allocated_bytes=expected_allocated_bytes,
        )
        return {
            "audit_command": (
                "python3 tools/mft_goal_corrected_thermal_continuation.py "
                f"audit --checkpoint {destination}"
            ),
            "destination": str(destination),
            "elapsed_seconds": time.time() - started,
            "filesystem_before_publish": filesystem_before_publish,
            "incoming_preprotect_verification": incoming_verification,
            "manifest": str(destination / ".checkpoint_manifest.json"),
            "manifest_payload_sha256": expected_manifest_payload_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "post_login_quota_reauthentication_required": True,
            "post_protect_verification": protected,
            "post_publish_verification": final,
            "quota_admission": quota_admission,
            "rename": rename_receipt,
            "resume_tool_payload_sha256": resume_tool_payload_sha256,
            "schema_version": SCHEMA_VERSION,
            "source_final_snapshot": source_final.summary(),
            "status": "complete-awaiting-controller-login-quota-reauthentication",
        }
    except BaseException as original:
        recovery: dict[str, Any] = {
            "incoming_preserved": _lexists(incoming),
            "original_error": f"{type(original).__name__}: {original}",
            "published": published,
            "status": "failed-no-delete",
        }
        if published and _lexists(destination):
            try:
                quarantine = _quarantine_partial(destination, destination)
                recovery["quarantine"] = str(quarantine)
                recovery["published"] = False
            except BaseException as quarantine_error:
                recovery["quarantine_error"] = (
                    f"{type(quarantine_error).__name__}: {quarantine_error}"
                )
        print(json.dumps(recovery, sort_keys=True), file=sys.stderr)
        if recovery.get("quarantine_error"):
            raise CheckpointError(
                "resume failed after publish and guarded quarantine failed; "
                f"original={original}; quarantine={recovery['quarantine_error']}"
            ) from original
        raise


def stage_static_source(
    *,
    source: Path,
    destination: Path,
    expected_metadata_sha256: str,
    expected_count: int,
    expected_directory_count: int,
    expected_logical_bytes: int,
    expected_allocated_bytes: int,
    expected_aedt_sha256: str,
    checkpoint_tool_payload_sha256: str,
    login_quota_evidence_base64: str,
    login_quota_evidence_sha256: str,
    login_quota_evidence_max_age_seconds: int,
    minimum_soft_headroom_bytes: int,
    minimum_hard_headroom_bytes: int,
    minimum_soft_inode_headroom: int,
    minimum_hard_inode_headroom: int,
) -> dict[str, Any]:
    started = time.time()
    source = source.resolve(strict=True)
    destination, allowed_root = _validate_destination(source, destination)
    incoming = destination.with_name(destination.name + ".incoming")
    if _lexists(destination) or _lexists(incoming):
        raise CheckpointError("destination or incoming collision")

    before = snapshot_static_source(source)
    _assert_expected_snapshot(
        before,
        metadata_sha256=expected_metadata_sha256,
        count=expected_count,
        directory_count=expected_directory_count,
        logical_bytes=expected_logical_bytes,
        allocated_bytes=expected_allocated_bytes,
    )
    aedt = next(row for row in before.files if row.path.endswith(".aedt"))
    if _full_sha256(source / aedt.path) != expected_aedt_sha256:
        raise CheckpointError("source AEDT SHA-256 differs from admission")
    quota_before = _decode_login_quota_evidence(
        login_quota_evidence_base64,
        expected_canonical_sha256=login_quota_evidence_sha256,
        maximum_age_seconds=login_quota_evidence_max_age_seconds,
    )
    quota_admission = _assert_quota(
        quota_before,
        budget_bytes=before.required_budget_bytes,
        budget_inodes=before.required_inodes,
        minimum_soft_headroom_bytes=minimum_soft_headroom_bytes,
        minimum_hard_headroom_bytes=minimum_hard_headroom_bytes,
        minimum_soft_inode_headroom=minimum_soft_inode_headroom,
        minimum_hard_inode_headroom=minimum_hard_inode_headroom,
    )
    filesystem_before = _filesystem_capacity(
        allowed_root.parent,
        required_free_bytes=(
            before.required_budget_bytes + PHYSICAL_FREE_RESERVE_BYTES
        ),
    )

    if not _lexists(allowed_root):
        os.mkdir(allowed_root, mode=0o700)
        _fsync_directory(allowed_root.parent)
    _assert_private_directory(allowed_root)
    if _lexists(destination) or _lexists(incoming):
        raise CheckpointError("destination collision after root admission")
    os.mkdir(incoming, mode=0o700)
    source_hashes: dict[str, str] = {}
    published = False
    try:
        for row in before.files:
            source_hashes[row.path] = _copy_stable_file(source, incoming, row)
        after = snapshot_static_source(source)
        _assert_expected_snapshot(
            after,
            metadata_sha256=expected_metadata_sha256,
            count=expected_count,
            directory_count=expected_directory_count,
            logical_bytes=expected_logical_bytes,
            allocated_bytes=expected_allocated_bytes,
        )
        destination_hashes = {
            row.path: _full_sha256(incoming / row.path) for row in before.files
        }
        if source_hashes != destination_hashes:
            raise CheckpointError("independent destination hashes differ")
        filesystem_after_copy = _filesystem_capacity(
            allowed_root,
            required_free_bytes=PHYSICAL_FREE_RESERVE_BYTES,
            expected_device=int(filesystem_before["anchor_device"]),
            expected_fsid=int(filesystem_before["fsid"]),
        )
        source_provenance = {
            **SOURCE_PROVENANCE,
            "source_project_sha256": expected_aedt_sha256,
            "source_static_metadata_sha256": expected_metadata_sha256,
        }
        manifest = {
            "canonical": False,
            "checkpoint_creator": {
                "tool_payload_sha256": checkpoint_tool_payload_sha256,
            },
            "completed_epoch": time.time(),
            "composite_reauthentication_required": True,
            "destination": str(destination),
            "diagnostic_only": True,
            "files": [
                {**asdict(row), "sha256": source_hashes[row.path]}
                for row in before.files
            ],
            "physics_boundary": {
                "fan_velocity_m_per_s": 1.5,
                "thermal_pad_thickness_mm": 2.0,
                "tim_conductivity_w_per_mk": 0.2,
            },
            "filesystem_after_copy": filesystem_after_copy,
            "filesystem_before": filesystem_before,
            "post_login_quota_reauthentication_required": True,
            "quota_admission": quota_admission,
            "quota_before_login_evidence": quota_before,
            "schema_version": SCHEMA_VERSION,
            "source": str(source),
            "source_provenance": source_provenance,
            "source_snapshot_after": after.summary(),
            "source_snapshot_before": before.summary(),
            "started_epoch": started,
            "traversal_policy": {
                "allowlist_only": True,
                "excluded_directories_pruned": True,
                "mesh_directory_children": list(MESH_FILES),
            },
        }
        payload = _canonical_json(manifest)
        manifest["manifest_payload_sha256"] = _sha256_bytes(payload)
        _write_exclusive(
            incoming / ".checkpoint_manifest.json",
            _canonical_json(manifest),
        )
        _fsync_and_protect_tree(incoming)
        verified = _verify_protected_tree(incoming, source_hashes)
        _fsync_directory(allowed_root)
        _rename_noreplace(incoming, destination)
        published = True
        _fsync_directory(allowed_root)
        final = _verify_protected_tree(destination, source_hashes)
        return {
            "audit_command": (
                "python3 tools/mft_goal_corrected_thermal_continuation.py "
                f"audit --checkpoint {destination}"
            ),
            "destination": str(destination),
            "elapsed_seconds": time.time() - started,
            "manifest": str(destination / ".checkpoint_manifest.json"),
            "metadata_sha256": before.metadata_sha256,
            "post_protect_verification": verified,
            "post_publish_verification": final,
            "post_login_quota_reauthentication_required": True,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
        }
    except BaseException as original:
        partial = destination if published and _lexists(destination) else incoming
        if _lexists(partial):
            try:
                quarantine = _quarantine_partial(partial, destination)
            except BaseException as quarantine_error:
                raise CheckpointError(
                    "checkpoint failed and quarantine also failed; "
                    f"partial={partial}; original={original}; "
                    f"quarantine={quarantine_error}"
                ) from quarantine_error
            recovery = {
                "audit_command": (
                    "python3 tools/mft_goal_corrected_thermal_continuation.py "
                    f"audit-incomplete --checkpoint {quarantine}"
                ),
                "original_error": f"{type(original).__name__}: {original}",
                "quarantine": str(quarantine),
                "recoverability": (
                    "source is untouched; inspect quarantine read-only, then rerun "
                    "stage with the same final destination"
                ),
            }
            print(json.dumps(recovery, sort_keys=True), file=sys.stderr)
        raise


def audit_checkpoint(checkpoint: Path, *, incomplete: bool = False) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    manifest_path = checkpoint / ".checkpoint_manifest.json"
    if not manifest_path.is_file():
        if not incomplete:
            raise CheckpointError("checkpoint manifest is unavailable")
        directories, files = _walk_real_tree(checkpoint)
        return {
            "checkpoint": str(checkpoint),
            "directory_count": len(directories),
            "file_count": len(files),
            "logical_bytes": sum(path.stat().st_size for path in files),
            "status": "incomplete-no-manifest",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_payload_sha256 = manifest.pop("manifest_payload_sha256", None)
    if claimed_payload_sha256 != _sha256_bytes(_canonical_json(manifest)):
        raise CheckpointError("manifest payload SHA-256 mismatch")
    expected_hashes = {
        str(row["path"]): str(row["sha256"]) for row in manifest["files"]
    }
    verification = _verify_protected_tree(checkpoint, expected_hashes)
    return {
        "checkpoint": str(checkpoint),
        "manifest_source_provenance": manifest["source_provenance"],
        "status": "verified",
        "verification": verification,
    }


def _snapshot_command(args: argparse.Namespace) -> int:
    value = snapshot_static_source(args.source)
    print(
        json.dumps(
            {
                **value.summary(),
                "directories": list(value.directories),
                "files": [asdict(row) for row in value.files],
                "writers": [asdict(row) for row in value.writers],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if value.included_writers else 0


def _stage_command(args: argparse.Namespace) -> int:
    value = stage_static_source(
        source=args.source,
        destination=args.destination,
        expected_metadata_sha256=args.expected_metadata_sha256,
        expected_count=args.expected_count,
        expected_directory_count=args.expected_directory_count,
        expected_logical_bytes=args.expected_logical_bytes,
        expected_allocated_bytes=args.expected_allocated_bytes,
        expected_aedt_sha256=args.expected_aedt_sha256,
        checkpoint_tool_payload_sha256=args.checkpoint_tool_payload_sha256,
        login_quota_evidence_base64=args.login_quota_evidence_base64,
        login_quota_evidence_sha256=args.login_quota_evidence_sha256,
        login_quota_evidence_max_age_seconds=(
            args.login_quota_evidence_max_age_seconds
        ),
        minimum_soft_headroom_bytes=args.minimum_soft_headroom_bytes,
        minimum_hard_headroom_bytes=args.minimum_hard_headroom_bytes,
        minimum_soft_inode_headroom=args.minimum_soft_inode_headroom,
        minimum_hard_inode_headroom=args.minimum_hard_inode_headroom,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _resume_protect_publish_command(args: argparse.Namespace) -> int:
    value = resume_protect_publish(
        source=args.source,
        destination=args.destination,
        expected_metadata_sha256=args.expected_metadata_sha256,
        expected_count=args.expected_count,
        expected_directory_count=args.expected_directory_count,
        expected_logical_bytes=args.expected_logical_bytes,
        expected_allocated_bytes=args.expected_allocated_bytes,
        expected_aedt_sha256=args.expected_aedt_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_payload_sha256=args.expected_manifest_payload_sha256,
        expected_checkpoint_tool_payload_sha256=(
            args.expected_checkpoint_tool_payload_sha256
        ),
        resume_tool_payload_sha256=args.resume_tool_payload_sha256,
        login_quota_evidence_base64=args.login_quota_evidence_base64,
        login_quota_evidence_sha256=args.login_quota_evidence_sha256,
        login_quota_evidence_max_age_seconds=(
            args.login_quota_evidence_max_age_seconds
        ),
        minimum_soft_headroom_bytes=args.minimum_soft_headroom_bytes,
        minimum_hard_headroom_bytes=args.minimum_hard_headroom_bytes,
        minimum_soft_inode_headroom=args.minimum_soft_inode_headroom,
        minimum_hard_inode_headroom=args.minimum_hard_inode_headroom,
        read_only_preflight=args.read_only_preflight,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _quarantine_published_command(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            quarantine_published_checkpoint(
                args.checkpoint,
                expected_manifest_sha256=args.expected_manifest_sha256,
                reason=args.reason,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _audit_command(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            audit_checkpoint(args.checkpoint, incomplete=args.incomplete),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--source", required=True, type=Path)
    snapshot.set_defaults(function=_snapshot_command)

    stage = commands.add_parser("stage")
    stage.add_argument("--source", required=True, type=Path)
    stage.add_argument("--destination", required=True, type=Path)
    stage.add_argument("--expected-metadata-sha256", required=True)
    stage.add_argument("--expected-count", required=True, type=int)
    stage.add_argument("--expected-directory-count", required=True, type=int)
    stage.add_argument("--expected-logical-bytes", required=True, type=int)
    stage.add_argument("--expected-allocated-bytes", required=True, type=int)
    stage.add_argument("--expected-aedt-sha256", required=True)
    stage.add_argument("--checkpoint-tool-payload-sha256", required=True)
    stage.add_argument("--login-quota-evidence-base64", required=True)
    stage.add_argument("--login-quota-evidence-sha256", required=True)
    stage.add_argument(
        "--login-quota-evidence-max-age-seconds",
        default=MAX_QUOTA_EVIDENCE_AGE_SECONDS,
        type=int,
    )
    stage.add_argument(
        "--minimum-soft-headroom-bytes",
        default=50 * 1024**3,
        type=int,
    )
    stage.add_argument(
        "--minimum-hard-headroom-bytes",
        default=60 * 1024**3,
        type=int,
    )
    stage.add_argument(
        "--minimum-soft-inode-headroom",
        default=1_000_000,
        type=int,
    )
    stage.add_argument(
        "--minimum-hard-inode-headroom",
        default=1_000_000,
        type=int,
    )
    stage.set_defaults(function=_stage_command)

    resume = commands.add_parser("resume-protect-publish")
    resume.add_argument("--source", required=True, type=Path)
    resume.add_argument("--destination", required=True, type=Path)
    resume.add_argument("--expected-metadata-sha256", required=True)
    resume.add_argument("--expected-count", required=True, type=int)
    resume.add_argument("--expected-directory-count", required=True, type=int)
    resume.add_argument("--expected-logical-bytes", required=True, type=int)
    resume.add_argument("--expected-allocated-bytes", required=True, type=int)
    resume.add_argument("--expected-aedt-sha256", required=True)
    resume.add_argument("--expected-manifest-sha256", required=True)
    resume.add_argument("--expected-manifest-payload-sha256", required=True)
    resume.add_argument(
        "--expected-checkpoint-tool-payload-sha256",
        required=True,
    )
    resume.add_argument("--resume-tool-payload-sha256", required=True)
    resume.add_argument("--login-quota-evidence-base64", required=True)
    resume.add_argument("--login-quota-evidence-sha256", required=True)
    resume.add_argument(
        "--login-quota-evidence-max-age-seconds",
        default=MAX_QUOTA_EVIDENCE_AGE_SECONDS,
        type=int,
    )
    resume.add_argument(
        "--minimum-soft-headroom-bytes",
        default=50 * 1024**3,
        type=int,
    )
    resume.add_argument(
        "--minimum-hard-headroom-bytes",
        default=60 * 1024**3,
        type=int,
    )
    resume.add_argument(
        "--minimum-soft-inode-headroom",
        default=1_000_000,
        type=int,
    )
    resume.add_argument(
        "--minimum-hard-inode-headroom",
        default=1_000_000,
        type=int,
    )
    resume.add_argument("--read-only-preflight", action="store_true")
    resume.set_defaults(function=_resume_protect_publish_command)

    quarantine = commands.add_parser("quarantine-published")
    quarantine.add_argument("--checkpoint", required=True, type=Path)
    quarantine.add_argument("--expected-manifest-sha256", required=True)
    quarantine.add_argument("--reason", required=True)
    quarantine.set_defaults(function=_quarantine_published_command)

    for name, incomplete in (("audit", False), ("audit-incomplete", True)):
        audit = commands.add_parser(name)
        audit.add_argument("--checkpoint", required=True, type=Path)
        audit.set_defaults(function=_audit_command, incomplete=incomplete)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except CheckpointError as exc:
        print(f"CHECKPOINT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
