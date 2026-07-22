"""Fail-closed cgroup v1/v2 memory discovery for the resource2 canary.

The public helpers are deliberately dependency-free so their exact source can
also be embedded in the one-shot remote wrapper.  The parser accepts Linux
procfs text plus a cgroup sysfs root, which makes every hierarchy and path
decision independently testable without a Scheduler task.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path, PurePosixPath
import textwrap


CGROUP_DIAGNOSTIC_SCHEMA = "mft-tier1-cgroup-memory-diagnostics-v2"
CGROUP_SNAPSHOT_SCHEMA = "mft-tier1-cgroup-memory-snapshot-v2"
CGROUP_DIAGNOSTIC_MAX_BYTES = 256 * 1024
CGROUP_V1_UNLIMITED_MIN_BYTES = 1 << 60


def _canonical_absolute_posix(value, label):
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or "\x00" in value
    ):
        raise RuntimeError(label + " is not one canonical absolute POSIX path")
    parsed = PurePosixPath(value)
    if ".." in parsed.parts or "." in parsed.parts:
        raise RuntimeError(label + " contains a traversal component")
    canonical = "/" if parsed == PurePosixPath("/") else "/" + "/".join(parsed.parts[1:])
    if value != canonical:
        raise RuntimeError(label + " is not canonical")
    return parsed


def _mountinfo_unescape(value):
    result = value
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        result = result.replace(encoded, decoded)
    if "\\" in result or "\x00" in result or "\n" in result or "\r" in result:
        raise RuntimeError("unsupported or unsafe mountinfo path escape")
    return result


def _parse_cgroup_memberships(text):
    if not isinstance(text, str) or not text or "\x00" in text:
        raise RuntimeError("/proc/self/cgroup is empty or malformed")
    result = []
    seen_hierarchy_ids = set()
    seen_v2 = 0
    seen_memory = 0
    for line in text.splitlines():
        if not line:
            raise RuntimeError("/proc/self/cgroup contains an empty record")
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isascii() or not fields[0].isdigit():
            raise RuntimeError("/proc/self/cgroup contains a malformed record")
        hierarchy_id = int(fields[0])
        if hierarchy_id in seen_hierarchy_ids:
            raise RuntimeError("/proc/self/cgroup repeats a hierarchy id")
        seen_hierarchy_ids.add(hierarchy_id)
        controllers_raw = fields[1]
        controllers = [] if not controllers_raw else controllers_raw.split(",")
        if any(
            not controller
            or not controller.isascii()
            or any(not (character.isalnum() or character in "_-=.") for character in controller)
            for controller in controllers
        ):
            raise RuntimeError("/proc/self/cgroup controller list is malformed")
        if len(controllers) != len(set(controllers)):
            raise RuntimeError("/proc/self/cgroup repeats a controller")
        membership = _canonical_absolute_posix(fields[2], "cgroup membership")
        version = None
        if not controllers:
            if hierarchy_id != 0:
                raise RuntimeError("empty controller membership is not cgroup v2")
            seen_v2 += 1
            version = "v2"
        elif "memory" in controllers:
            if hierarchy_id == 0:
                raise RuntimeError("cgroup v1 memory hierarchy id must be nonzero")
            seen_memory += 1
            version = "v1"
        if version is not None:
            result.append(
                {
                    "version": version,
                    "hierarchy_id": hierarchy_id,
                    "controllers": sorted(controllers),
                    "membership_path": membership.as_posix(),
                }
            )
    if seen_v2 > 1 or seen_memory > 1:
        raise RuntimeError("cgroup memory membership is ambiguous")
    if not result:
        raise RuntimeError("no cgroup v1 memory or v2 membership is available")
    return result


def _parse_cgroup_mounts(text):
    if not isinstance(text, str) or not text or "\x00" in text:
        raise RuntimeError("/proc/self/mountinfo is empty or malformed")
    result = []
    for line in text.splitlines():
        fields = line.split()
        separators = [index for index, field in enumerate(fields) if field == "-"]
        if len(separators) != 1:
            raise RuntimeError("mountinfo record has an ambiguous separator")
        separator = separators[0]
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeError("mountinfo record is truncated")
        filesystem = fields[separator + 1]
        if filesystem not in {"cgroup", "cgroup2"}:
            continue
        mount_root_raw = _mountinfo_unescape(fields[3])
        mount_point_raw = _mountinfo_unescape(fields[4])
        mount_root = _canonical_absolute_posix(mount_root_raw, "cgroup mount root")
        mount_point = _canonical_absolute_posix(mount_point_raw, "cgroup mount point")
        sysfs = PurePosixPath("/sys/fs/cgroup")
        if mount_point != sysfs and sysfs not in mount_point.parents:
            raise RuntimeError("cgroup mount point escaped /sys/fs/cgroup")
        option_tokens = set()
        for field in [*fields[5:separator], *fields[separator + 2 :]]:
            option_tokens.update(token for token in field.split(",") if token)
        if filesystem == "cgroup" and "memory" not in option_tokens:
            continue
        result.append(
            {
                "version": "v2" if filesystem == "cgroup2" else "v1",
                "mount_root": mount_root.as_posix(),
                "mount_point": mount_point.as_posix(),
            }
        )
    if not result:
        raise RuntimeError("no cgroup memory-capable mount is available")
    return result


def _relative_membership(membership_path, mount_root):
    membership = PurePosixPath(membership_path)
    root = PurePosixPath(mount_root)
    if membership == root:
        return PurePosixPath(".")
    if root != PurePosixPath("/") and root not in membership.parents:
        return None
    if root == PurePosixPath("/"):
        return PurePosixPath(*membership.parts[1:])
    return membership.relative_to(root)


def _safe_cgroup_file(directory, filename, mount_path):
    candidate = directory / filename
    if not candidate.exists():
        if candidate.is_symlink():
            raise RuntimeError("broken cgroup accounting symlink")
        return None
    if candidate.is_symlink():
        raise RuntimeError("cgroup accounting file must not be a symlink")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != directory or (resolved != mount_path and mount_path not in resolved.parents):
        raise RuntimeError("cgroup accounting file escaped its mount")
    if not resolved.is_file():
        raise RuntimeError("cgroup accounting path is not a file")
    return resolved


def _read_cgroup_integer(path, allow_unbounded, version):
    with path.open("rb") as stream:
        raw_bytes = stream.read(129)
    if len(raw_bytes) > 128:
        raise RuntimeError("cgroup accounting value exceeds 128 bytes")
    try:
        raw = raw_bytes.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("cgroup accounting value is not ASCII") from exc
    if not raw or any(character.isspace() for character in raw):
        raise RuntimeError("cgroup accounting value is empty or multiline")
    if raw == "max":
        if not allow_unbounded:
            raise RuntimeError("only a cgroup limit may be unbounded")
        return None, raw
    if not raw.isascii() or not raw.isdigit():
        raise RuntimeError("cgroup accounting value is not an unsigned integer")
    value = int(raw)
    if allow_unbounded and version == "v1" and value >= (1 << 60):
        return None, raw
    return value, raw


def _candidate_snapshot(candidate, mount, sysfs_root):
    relative = _relative_membership(candidate["membership_path"], mount["mount_root"])
    if relative is None:
        return None
    sysfs_root = Path(sysfs_root).resolve(strict=True)
    logical_mount = PurePosixPath(mount["mount_point"])
    logical_sysfs = PurePosixPath("/sys/fs/cgroup")
    mount_relative = (
        PurePosixPath(".")
        if logical_mount == logical_sysfs
        else logical_mount.relative_to(logical_sysfs)
    )
    mount_path = (sysfs_root / mount_relative.as_posix()).resolve(strict=True)
    if mount_path != sysfs_root and sysfs_root not in mount_path.parents:
        raise RuntimeError("resolved cgroup mount escaped the sysfs root")
    leaf = (mount_path / relative.as_posix()).resolve(strict=True)
    if leaf != mount_path and mount_path not in leaf.parents:
        raise RuntimeError("resolved cgroup membership escaped its mount")
    if not leaf.is_dir():
        return None
    filenames = (
        ("memory.max", "memory.current", "memory.peak")
        if candidate["version"] == "v2"
        else (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
        )
    )
    records = []
    current = leaf
    depth = 0
    while True:
        files = [_safe_cgroup_file(current, name, mount_path) for name in filenames]
        if any(item is not None for item in files):
            if not all(item is not None for item in files):
                raise RuntimeError("cgroup memory accounting files are partial")
            limit, raw_limit = _read_cgroup_integer(files[0], True, candidate["version"])
            usage, _ = _read_cgroup_integer(files[1], False, candidate["version"])
            peak, _ = _read_cgroup_integer(files[2], False, candidate["version"])
            if peak < usage:
                raise RuntimeError("cgroup peak memory is below current usage")
            records.append(
                {
                    "depth_from_leaf": depth,
                    "relative_path": (
                        "."
                        if current == sysfs_root
                        else current.relative_to(sysfs_root).as_posix()
                    ),
                    "memory_max_raw": raw_limit,
                    "memory_limit_bytes": limit,
                    "memory_current_bytes": usage,
                    "memory_peak_bytes": peak,
                }
            )
        if current == mount_path:
            break
        current = current.parent
        depth += 1
    if not records:
        return None
    finite = [record for record in records if record["memory_limit_bytes"] is not None]
    if not finite:
        return None
    return {
        "schema_version": "mft-tier1-cgroup-memory-snapshot-v2",
        "cgroup_version": candidate["version"],
        "hierarchy_id": candidate["hierarchy_id"],
        "membership_controllers": list(candidate["controllers"]),
        "membership_path": candidate["membership_path"],
        "mount_root": mount["mount_root"],
        "mount_relative_path": mount_relative.as_posix(),
        "leaf_relative_path": (
            "." if leaf == sysfs_root else leaf.relative_to(sysfs_root).as_posix()
        ),
        "nearest_accounting_depth": records[0]["depth_from_leaf"],
        "nearest_accounting_limit_unbounded": records[0]["memory_limit_bytes"] is None,
        "limit_filename": filenames[0],
        "current_filename": filenames[1],
        "peak_filename": filenames[2],
        "ancestors": records,
        "selected_finite_ancestor": dict(finite[0]),
    }


def cgroup_snapshot_from_text(cgroup_text, mountinfo_text, sysfs_root):
    memberships = _parse_cgroup_memberships(cgroup_text)
    mounts = _parse_cgroup_mounts(mountinfo_text)
    candidates = []
    seen = set()
    for membership in memberships:
        for mount in mounts:
            if membership["version"] != mount["version"]:
                continue
            identity = (
                membership["version"],
                membership["membership_path"],
                mount["mount_root"],
                mount["mount_point"],
            )
            if identity in seen:
                raise RuntimeError("cgroup memory mount mapping is duplicated")
            seen.add(identity)
            snapshot = _candidate_snapshot(membership, mount, sysfs_root)
            if snapshot is not None:
                candidates.append(snapshot)
    if len(candidates) != 1:
        raise RuntimeError("unique finite cgroup memory hierarchy is unavailable")
    return candidates[0]


def _bounded_diagnostic_record(path, maximum_bytes):
    record = {
        "path": str(path),
        "maximum_bytes": maximum_bytes,
        "bytes_captured": 0,
        "truncated": False,
        "utf8_valid": False,
        "sha256": None,
        "text": None,
        "error": None,
    }
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        record["truncated"] = len(raw) > maximum_bytes
        bounded = raw[:maximum_bytes]
        record["bytes_captured"] = len(bounded)
        record["sha256"] = hashlib.sha256(bounded).hexdigest()
        try:
            record["text"] = bounded.decode("utf-8")
            record["utf8_valid"] = True
        except UnicodeDecodeError:
            record["text"] = bounded.decode("utf-8", errors="replace")
    except Exception as exc:
        record["error"] = type(exc).__name__ + ":" + str(exc)
    return record


def capture_cgroup_diagnostics(
    cgroup_path=Path("/proc/self/cgroup"),
    mountinfo_path=Path("/proc/self/mountinfo"),
    sysfs_root=Path("/sys/fs/cgroup"),
    maximum_bytes=CGROUP_DIAGNOSTIC_MAX_BYTES,
):
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise RuntimeError("cgroup diagnostic byte bound is invalid")
    return {
        "schema_version": "mft-tier1-cgroup-memory-diagnostics-v2",
        "maximum_bytes_per_file": maximum_bytes,
        "sysfs_root": str(sysfs_root).replace("\\", "/"),
        "proc_self_cgroup": _bounded_diagnostic_record(cgroup_path, maximum_bytes),
        "proc_self_mountinfo": _bounded_diagnostic_record(mountinfo_path, maximum_bytes),
    }


def cgroup_snapshot_from_diagnostics(diagnostics, sysfs_root=Path("/sys/fs/cgroup")):
    if not isinstance(diagnostics, dict):
        raise RuntimeError("cgroup diagnostics are absent")
    cgroup = diagnostics.get("proc_self_cgroup")
    mountinfo = diagnostics.get("proc_self_mountinfo")
    if not isinstance(cgroup, dict) or not isinstance(mountinfo, dict):
        raise RuntimeError("cgroup diagnostic records are absent")
    for record in (cgroup, mountinfo):
        if (
            record.get("error") is not None
            or record.get("truncated") is not False
            or record.get("utf8_valid") is not True
            or not isinstance(record.get("text"), str)
        ):
            raise RuntimeError("cgroup diagnostic input is incomplete or truncated")
    return cgroup_snapshot_from_text(cgroup["text"], mountinfo["text"], sysfs_root)


def validate_cgroup_diagnostics(value):
    required = {
        "schema_version",
        "maximum_bytes_per_file",
        "sysfs_root",
        "proc_self_cgroup",
        "proc_self_mountinfo",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != CGROUP_DIAGNOSTIC_SCHEMA
        or value.get("maximum_bytes_per_file") != CGROUP_DIAGNOSTIC_MAX_BYTES
        or value.get("sysfs_root") != "/sys/fs/cgroup"
    ):
        raise RuntimeError("cgroup diagnostic envelope drifted")
    record_fields = {
        "path",
        "maximum_bytes",
        "bytes_captured",
        "truncated",
        "utf8_valid",
        "sha256",
        "text",
        "error",
    }
    expected_paths = {
        "proc_self_cgroup": "/proc/self/cgroup",
        "proc_self_mountinfo": "/proc/self/mountinfo",
    }
    result = dict(value)
    for label, expected_path in expected_paths.items():
        record = value.get(label)
        if not isinstance(record, dict) or set(record) != record_fields:
            raise RuntimeError("cgroup diagnostic record drifted")
        text = record.get("text")
        size = record.get("bytes_captured")
        digest = record.get("sha256")
        if (
            record.get("path") != expected_path
            or record.get("maximum_bytes") != CGROUP_DIAGNOSTIC_MAX_BYTES
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= CGROUP_DIAGNOSTIC_MAX_BYTES
            or record.get("truncated") is not False
            or record.get("utf8_valid") is not True
            or record.get("error") is not None
            or not isinstance(text, str)
            or len(text.encode("utf-8")) != size
            or not isinstance(digest, str)
            or digest != hashlib.sha256(text.encode("utf-8")).hexdigest()
        ):
            raise RuntimeError("cgroup diagnostic record is incomplete or unsealed")
        result[label] = dict(record)
    return result


def _validated_relative_path(value, label):
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise RuntimeError(label + " is not one safe relative path")
    parsed = PurePosixPath(value)
    if ".." in parsed.parts or (value != "." and value != parsed.as_posix()):
        raise RuntimeError(label + " is not canonical")
    return parsed


def validate_cgroup_snapshot(value):
    required = {
        "schema_version",
        "cgroup_version",
        "hierarchy_id",
        "membership_controllers",
        "membership_path",
        "mount_root",
        "mount_relative_path",
        "leaf_relative_path",
        "nearest_accounting_depth",
        "nearest_accounting_limit_unbounded",
        "limit_filename",
        "current_filename",
        "peak_filename",
        "ancestors",
        "selected_finite_ancestor",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != CGROUP_SNAPSHOT_SCHEMA
    ):
        raise RuntimeError("cgroup memory snapshot fields drifted")
    version = value.get("cgroup_version")
    hierarchy_id = value.get("hierarchy_id")
    controllers = value.get("membership_controllers")
    expected_files = {
        "v1": (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
        ),
        "v2": ("memory.max", "memory.current", "memory.peak"),
    }
    if (
        version not in expected_files
        or isinstance(hierarchy_id, bool)
        or not isinstance(hierarchy_id, int)
        or hierarchy_id < 0
        or not isinstance(controllers, list)
        or controllers != sorted(set(controllers))
        or (version == "v2" and (hierarchy_id != 0 or controllers))
        or (version == "v1" and (hierarchy_id == 0 or "memory" not in controllers))
        or tuple(
            value.get(field)
            for field in ("limit_filename", "current_filename", "peak_filename")
        )
        != expected_files[version]
    ):
        raise RuntimeError("cgroup memory hierarchy identity drifted")
    membership = _canonical_absolute_posix(value.get("membership_path"), "membership path")
    mount_root = _canonical_absolute_posix(value.get("mount_root"), "mount root")
    membership_relative = _relative_membership(membership.as_posix(), mount_root.as_posix())
    if membership_relative is None:
        raise RuntimeError("cgroup membership is outside the mounted hierarchy")
    mount_relative = _validated_relative_path(value.get("mount_relative_path"), "mount path")
    leaf_relative = _validated_relative_path(value.get("leaf_relative_path"), "leaf path")
    expected_leaf = (
        mount_relative
        if membership_relative == PurePosixPath(".")
        else mount_relative / membership_relative
    )
    if expected_leaf != leaf_relative:
        raise RuntimeError("cgroup leaf does not match membership/mount roots")
    ancestors = value.get("ancestors")
    selected = value.get("selected_finite_ancestor")
    if not isinstance(ancestors, list) or not ancestors or not isinstance(selected, dict):
        raise RuntimeError("cgroup memory ancestry is absent")
    record_fields = {
        "depth_from_leaf",
        "relative_path",
        "memory_max_raw",
        "memory_limit_bytes",
        "memory_current_bytes",
        "memory_peak_bytes",
    }
    normalized = []
    prior_depth = -1
    prior_path = None
    for raw in ancestors:
        if not isinstance(raw, dict) or set(raw) != record_fields:
            raise RuntimeError("cgroup memory ancestor record drifted")
        depth = raw.get("depth_from_leaf")
        path = _validated_relative_path(raw.get("relative_path"), "ancestor path")
        limit_value = raw.get("memory_limit_bytes")
        current_value = raw.get("memory_current_bytes")
        peak_value = raw.get("memory_peak_bytes")
        raw_limit = raw.get("memory_max_raw")
        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth <= prior_depth
            or (leaf_relative != path and path not in leaf_relative.parents)
            or (
                prior_path is not None
                and (path not in prior_path.parents or depth - prior_depth != len(prior_path.parts) - len(path.parts))
            )
            or (
                limit_value is not None
                and (
                    isinstance(limit_value, bool)
                    or not isinstance(limit_value, int)
                    or limit_value < 0
                )
            )
            or isinstance(current_value, bool)
            or not isinstance(current_value, int)
            or current_value < 0
            or isinstance(peak_value, bool)
            or not isinstance(peak_value, int)
            or peak_value < current_value
            or not isinstance(raw_limit, str)
        ):
            raise RuntimeError("cgroup memory ancestor value drifted")
        if version == "v2":
            expected_limit = None if raw_limit == "max" else int(raw_limit) if raw_limit.isdigit() else object()
        else:
            expected_limit = (
                None
                if raw_limit.isdigit() and int(raw_limit) >= CGROUP_V1_UNLIMITED_MIN_BYTES
                else int(raw_limit) if raw_limit.isdigit() else object()
            )
        if expected_limit != limit_value:
            raise RuntimeError("cgroup memory limit normalization drifted")
        prior_depth = depth
        prior_path = path
        normalized.append(dict(raw))
    nearest_depth = value.get("nearest_accounting_depth")
    finite = [record for record in normalized if record["memory_limit_bytes"] is not None]
    if (
        isinstance(nearest_depth, bool)
        or not isinstance(nearest_depth, int)
        or nearest_depth != normalized[0]["depth_from_leaf"]
        or not isinstance(value.get("nearest_accounting_limit_unbounded"), bool)
        or value.get("nearest_accounting_limit_unbounded")
        is not (normalized[0]["memory_limit_bytes"] is None)
        or not finite
        or selected != finite[0]
    ):
        raise RuntimeError("nearest finite cgroup memory ancestor drifted")
    result = dict(value)
    result["ancestors"] = normalized
    result["selected_finite_ancestor"] = dict(selected)
    return result


_EMBEDDED_FUNCTIONS = (
    _canonical_absolute_posix,
    _mountinfo_unescape,
    _parse_cgroup_memberships,
    _parse_cgroup_mounts,
    _relative_membership,
    _safe_cgroup_file,
    _read_cgroup_integer,
    _candidate_snapshot,
    cgroup_snapshot_from_text,
    _bounded_diagnostic_record,
    capture_cgroup_diagnostics,
    cgroup_snapshot_from_diagnostics,
)


def embedded_runtime_source() -> str:
    """Return the exact tested helper implementation for the remote wrapper."""

    constants = (
        f'CGROUP_DIAGNOSTIC_SCHEMA = "{CGROUP_DIAGNOSTIC_SCHEMA}"\n'
        f'CGROUP_SNAPSHOT_SCHEMA = "{CGROUP_SNAPSHOT_SCHEMA}"\n'
        f"CGROUP_DIAGNOSTIC_MAX_BYTES = {CGROUP_DIAGNOSTIC_MAX_BYTES}\n"
        f"CGROUP_V1_UNLIMITED_MIN_BYTES = {CGROUP_V1_UNLIMITED_MIN_BYTES}\n"
    )
    functions = "\n\n".join(
        textwrap.dedent(inspect.getsource(function)).rstrip()
        for function in _EMBEDDED_FUNCTIONS
    )
    return constants + "\n" + functions + "\n"
