"""Capture fail-closed AEDT 16-core license evidence on the compute node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA = "mft-aedt-license-headroom-snapshot-v1"
SERVER = "1055@172.16.10.81"
CORE_CONTRACT = "mft-standalone-core-16-optin-v1"
LICENSE_CONTRACT = "mft-aedt-hpc-license-snapshot-v1"
REQUIRED_HEADROOM = {
    "anshpc": 16,
    "elec_solve_maxwell": 1,
    "electronics_desktop": 1,
    "electronics3d_gui": 1,
}
LMUTIL_CANDIDATES = (
    "/opt/ohpc/pub/Electronics/v252/Linux64/licensingclient/linx64/lmutil",
    "/opt/ohpc/pub/Electronics/v242/Linux64/licensingclient/linx64/lmutil",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def parse_lmstat(text: str) -> dict[str, dict[str, int]]:
    observed: dict[str, dict[str, int]] = {}
    pattern = re.compile(
        r"(?m)^Users of ([^:\s]+):.*?Total of (\d+) licenses? issued;"
        r"\s*Total of (\d+) licenses? in use"
    )
    for feature, total, used in pattern.findall(text):
        if feature not in REQUIRED_HEADROOM or feature in observed:
            continue
        observed[feature] = {"total": int(total), "used": int(used)}
    if set(observed) != set(REQUIRED_HEADROOM):
        missing = sorted(set(REQUIRED_HEADROOM) - set(observed))
        raise RuntimeError(f"lmstat lacks required AEDT features: {missing}")
    for feature, minimum in REQUIRED_HEADROOM.items():
        record = observed[feature]
        if (
            record["total"] < 0
            or record["used"] < 0
            or record["used"] > record["total"]
            or record["total"] - record["used"] < minimum
        ):
            raise RuntimeError(
                f"runtime license headroom failed for {feature}: "
                f"required={minimum}, total={record['total']}, "
                f"used={record['used']}"
            )
    return observed


def contract_auth_sha256(
    solver_revision: str, snapshot_sha256: str
) -> str:
    revision = str(solver_revision or "").strip().lower()
    digest = str(snapshot_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("solver revision must be exact 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("snapshot SHA-256 must be exact 64-hex")
    payload = {
        "backend": "standalone",
        "contract_version": CORE_CONTRACT,
        "requested_num_cores": 16,
        "required_slurm_cpus_per_task": 16,
        "solver_revision": revision,
        "license_contract": LICENSE_CONTRACT,
        "license_snapshot_sha256": digest,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _lmutil_path(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    root = os.environ.get("ANSYSEM_ROOT252", "").strip()
    if root:
        candidates.append(
            Path(root) / "licensingclient" / "linx64" / "lmutil"
        )
    candidates.extend(Path(value) for value in LMUTIL_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError("AEDT lmutil executable is unavailable")


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def capture(
    *,
    output_dir: Path,
    solver_revision: str,
    lmutil: str | None = None,
    server: str = SERVER,
    now: datetime | None = None,
) -> dict[str, str]:
    if server != SERVER:
        raise RuntimeError("license server identity drifted")
    executable = _lmutil_path(lmutil)
    completed = subprocess.run(
        [str(executable), "lmstat", "-c", server, "-a"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"lmstat failed with exit code {completed.returncode}"
        )
    features = parse_lmstat(completed.stdout)
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    snapshot = {
        "schema": SCHEMA,
        "server_up": True,
        "server": SERVER,
        "checked_at": checked_at.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "features": features,
    }
    snapshot_json = canonical_json(snapshot)
    snapshot_sha = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
    auth_sha = contract_auth_sha256(solver_revision, snapshot_sha)
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    _write_exclusive(destination / "snapshot.json", snapshot_json.encode("utf-8"))
    _write_exclusive(
        destination / "snapshot.sha256", snapshot_sha.encode("ascii")
    )
    _write_exclusive(destination / "auth.sha256", auth_sha.encode("ascii"))
    return {
        "snapshot_path": str(destination / "snapshot.json"),
        "snapshot_sha256": snapshot_sha,
        "auth_sha256": auth_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--lmutil")
    parser.add_argument("--server", default=SERVER)
    args = parser.parse_args(argv)
    result = capture(
        output_dir=args.output_dir,
        solver_revision=args.solver_revision,
        lmutil=args.lmutil,
        server=args.server,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
