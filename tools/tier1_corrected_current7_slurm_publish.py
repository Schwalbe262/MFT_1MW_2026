"""Publish one immutable current7 bundle to GPFS, with READY written last.

The public entry point is deliberately dry-run by default.  Network writes are
only possible when ``apply=True`` and a publication transport is supplied (or
the CLI is invoked with ``--apply``).  The deterministic incoming directory is
also the restart journal: every completed file is authenticated and skipped on
the next invocation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterator, Mapping, Protocol

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        BUNDLE_SCHEMA,
        PLAN_SCHEMA,
        READY_SCHEMA,
        read_json,
        sha256_file,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        BUNDLE_SCHEMA,
        PLAN_SCHEMA,
        READY_SCHEMA,
        read_json,
        sha256_file,
    )


PUBLICATION_RECEIPT_SCHEMA = "mft-tier1-current7-publication-receipt-v1"
PUBLICATION_JOURNAL_SCHEMA = "mft-tier1-current7-publication-journal-v1"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class PublicationTransport(Protocol):
    """Minimal SFTP/SSH surface used by the restartable publisher."""

    write_count: int

    def exists(self, path: str) -> bool: ...

    def is_dir(self, path: str) -> bool: ...

    def read_bytes(self, path: str) -> bytes: ...

    def mkdir(self, path: str, mode: int = 0o755) -> None: ...

    def upload_file(self, local: Path, remote: str) -> None: ...

    def write_bytes(self, path: str, value: bytes) -> None: ...

    def replace(self, source: str, destination: str) -> None: ...

    def file_record(self, path: str) -> Mapping[str, Any] | None: ...

    def verify_runtime(
        self,
        root: str,
        *,
        requirements_relative: str,
        expected_packages: Mapping[str, str],
    ) -> Mapping[str, str]: ...

    def seal_permissions(self, root: str) -> None: ...

    def promote_directory(self, incoming: str, destination: str) -> None: ...


def load_bundle_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Load and authenticate the local plan, manifest, and relocation map."""

    plan_path = plan_path.resolve(strict=True)
    plan = read_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("unsupported current7 Slurm plan schema")
    manifest_path = Path(str(plan.get("bundle_manifest") or "")).resolve(strict=True)
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("bundle_id") != plan.get("bundle_id")
        or manifest.get("contract_sha256") != plan.get("contract_sha256")
        or sha256_file(manifest_path) != plan.get("bundle_manifest_sha256")
    ):
        raise RuntimeError("current7 plan/manifest identity mismatch")
    unsigned_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"bundle_id", "contract_sha256"}
    }
    if canonical_sha256(unsigned_manifest) != manifest.get("contract_sha256"):
        raise RuntimeError("current7 bundle contract SHA mismatch")
    remote_root = str(plan.get("remote_root") or "").rstrip("/")
    expected_remote = f"{remote_root}/{plan['bundle_id']}"
    if not remote_root.startswith("/") or plan.get("remote_bundle") != expected_remote:
        raise RuntimeError("unsafe or non-canonical current7 remote bundle path")
    publication = plan.get("publication_contract") or {}
    required_publication_flags = (
        "stage_below_unique_incoming_directory",
        "verify_every_file_sha256_before_ready",
        "verify_exact_runtime_packages_before_ready",
        "write_ready_after_all_verification",
        "atomic_rename_incoming_to_content_addressed_bundle",
        "make_artifacts_read_only_before_publication",
    )
    if any(publication.get(flag) is not True for flag in required_publication_flags):
        raise RuntimeError("plan lacks the sealed immutable publication contract")
    source_map_path = Path(str(plan.get("local_sources") or "")).resolve(strict=True)
    raw_sources = read_json(source_map_path)
    files = manifest.get("files") or {}
    if set(raw_sources) != set(files):
        raise RuntimeError("local source map does not match bundle file inventory")
    source_map = {
        str(relative): Path(str(source)).resolve(strict=True)
        for relative, source in raw_sources.items()
    }
    return plan, manifest, source_map


def _validate_local_inventory(
    manifest: Mapping[str, Any], source_map: Mapping[str, Path]
) -> None:
    for relative, expected in sorted(manifest["files"].items()):
        source = source_map[relative]
        if (
            not source.is_file()
            or source.stat().st_size != int(expected["size"])
            or sha256_file(source) != expected["sha256"]
        ):
            raise RuntimeError(f"local bundle source changed after plan: {relative}")


def _parse_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid remote JSON: {label}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"remote JSON object required: {label}")
    return payload


def _ready_matches(
    value: Mapping[str, Any], plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    return (
        value.get("schema_version") == READY_SCHEMA
        and value.get("bundle_id") == plan.get("bundle_id")
        and value.get("bundle_manifest_sha256") == plan.get("bundle_manifest_sha256")
        and value.get("contract_sha256") == manifest.get("contract_sha256")
        and value.get("file_count") == len(manifest["files"])
        and value.get("byte_count")
        == sum(int(item["size"]) for item in manifest["files"].values())
        and value.get("runtime_verified") is True
        and value.get("all_file_sha256_verified") is True
        and value.get("every_file_sha256_verified") is True
        and value.get("code_inventory_sha256") == manifest.get("code_inventory_sha256")
        and value.get("relocation_contract_sha256")
        == manifest["relocation"]["contract_sha256"]
        and value.get("remote_git_checkout_performed") is False
        and value.get("artifacts_read_only") is True
    )


def _publication_receipt(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ready: Mapping[str, Any] | None,
    already_ready: bool,
    apply: bool,
    incoming: str,
    uploaded_files: int,
    resumed_files: int,
) -> dict[str, Any]:
    receipt = {
        "schema_version": PUBLICATION_RECEIPT_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "contract_sha256": manifest["contract_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "remote_bundle": plan["remote_bundle"],
        "incoming_bundle": incoming,
        "apply": bool(apply),
        "already_ready": bool(already_ready),
        "publication_complete": ready is not None,
        "uploaded_files": int(uploaded_files),
        "resumed_files": int(resumed_files),
        "remote_write_performed": bool(apply and not already_ready),
        "ready": dict(ready) if ready is not None else None,
        "ready_sha256": canonical_sha256(ready) if ready is not None else None,
    }
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


def validate_publication_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    ready = value.get("ready")
    if (
        value.get("schema_version") != PUBLICATION_RECEIPT_SCHEMA
        or value.get("bundle_id") != plan.get("bundle_id")
        or value.get("contract_sha256") != manifest.get("contract_sha256")
        or value.get("bundle_manifest_sha256") != plan.get("bundle_manifest_sha256")
        or value.get("remote_bundle") != plan.get("remote_bundle")
        or value.get("publication_complete") is not True
        or not isinstance(ready, dict)
        or not _ready_matches(ready, plan, manifest)
        or value.get("ready_sha256") != canonical_sha256(ready)
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("publication receipt/READY identity mismatch")
    return dict(value)


def publish_bundle(
    plan_path: Path,
    *,
    apply: bool = False,
    transport: PublicationTransport | None = None,
) -> dict[str, Any]:
    """Publish a bundle, or render the exact non-mutating publication plan."""

    plan, manifest, source_map = load_bundle_plan(plan_path)
    _validate_local_inventory(manifest, source_map)
    incoming = (
        f"{plan['remote_root']}/.incoming-{plan['bundle_id']}-"
        f"{plan['bundle_manifest_sha256'][:12]}"
    )
    if not apply:
        return _publication_receipt(
            plan=plan,
            manifest=manifest,
            ready=None,
            already_ready=False,
            apply=False,
            incoming=incoming,
            uploaded_files=0,
            resumed_files=0,
        )
    if transport is None:
        raise RuntimeError("--apply requires an explicit publication transport")

    final = plan["remote_bundle"]
    ready_path = f"{final}/READY.json"
    if transport.exists(ready_path):
        ready = _parse_json_bytes(transport.read_bytes(ready_path), ready_path)
        if not _ready_matches(ready, plan, manifest):
            raise RuntimeError("content-addressed destination has a mismatched READY")
        return _publication_receipt(
            plan=plan,
            manifest=manifest,
            ready=ready,
            already_ready=True,
            apply=True,
            incoming=incoming,
            uploaded_files=0,
            resumed_files=len(manifest["files"]),
        )
    if transport.exists(final):
        raise RuntimeError("content-addressed destination exists without exact READY")

    transport.mkdir(incoming, 0o755)
    transport.mkdir(f"{incoming}/runs", 0o1777)
    journal_path = f"{incoming}/.publication-journal.json"
    journal = {
        "schema_version": PUBLICATION_JOURNAL_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
    }
    if transport.exists(journal_path):
        if (
            _parse_json_bytes(transport.read_bytes(journal_path), journal_path)
            != journal
        ):
            raise RuntimeError("incoming publication journal identity mismatch")
    else:
        journal_part = journal_path + ".part"
        transport.write_bytes(journal_part, _json_bytes(journal))
        transport.replace(journal_part, journal_path)

    uploaded = 0
    resumed = 0
    for relative, expected in sorted(manifest["files"].items()):
        source = source_map[relative]
        remote = f"{incoming}/{relative}"
        record = transport.file_record(remote)
        if record is not None and (
            int(record.get("size", -1)) == int(expected["size"])
            and record.get("sha256") == expected["sha256"]
        ):
            resumed += 1
            continue
        parent = remote.rsplit("/", 1)[0]
        transport.mkdir(parent, 0o755)
        part = remote + f".part.{plan['bundle_manifest_sha256'][:12]}"
        transport.upload_file(source, part)
        part_record = transport.file_record(part)
        if part_record is None or (
            int(part_record.get("size", -1)) != int(expected["size"])
            or part_record.get("sha256") != expected["sha256"]
        ):
            raise RuntimeError(f"uploaded file authentication failed: {relative}")
        transport.replace(part, remote)
        final_record = transport.file_record(remote)
        if final_record != part_record:
            raise RuntimeError(f"uploaded file changed during promotion: {relative}")
        uploaded += 1

    manifest_local = Path(plan["bundle_manifest"])
    manifest_remote = f"{incoming}/bundle_manifest.json"
    manifest_part = manifest_remote + ".part"
    transport.upload_file(manifest_local, manifest_part)
    manifest_record = transport.file_record(manifest_part)
    if manifest_record is None or (
        int(manifest_record.get("size", -1)) != manifest_local.stat().st_size
        or manifest_record.get("sha256") != plan["bundle_manifest_sha256"]
    ):
        raise RuntimeError("remote bundle manifest authentication failed")
    transport.replace(manifest_part, manifest_remote)

    actual_packages = dict(
        transport.verify_runtime(
            incoming,
            requirements_relative=manifest["runtime"]["requirements_lock"],
            expected_packages=manifest["runtime"]["critical_packages"],
        )
    )
    if actual_packages != manifest["runtime"]["critical_packages"]:
        raise RuntimeError("remote critical runtime package mismatch")
    for relative, expected in sorted(manifest["files"].items()):
        record = transport.file_record(f"{incoming}/{relative}")
        if record is None or (
            int(record.get("size", -1)) != int(expected["size"])
            or record.get("sha256") != expected["sha256"]
        ):
            raise RuntimeError(f"pre-READY bundle verification failed: {relative}")
    transport.seal_permissions(incoming)

    ready = {
        "schema_version": READY_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "contract_sha256": manifest["contract_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "file_count": len(manifest["files"]),
        "byte_count": sum(int(item["size"]) for item in manifest["files"].values()),
        "runtime_verified": True,
        "runtime_packages": actual_packages,
        "all_file_sha256_verified": True,
        "every_file_sha256_verified": True,
        "code_inventory_sha256": manifest["code_inventory_sha256"],
        "relocation_contract_sha256": manifest["relocation"]["contract_sha256"],
        "remote_git_checkout_performed": False,
        "artifacts_read_only": True,
        "runs_mode": "1777",
        "published_at": _now(),
    }
    # This is intentionally the final remote file write.  Only an atomic
    # same-filesystem directory rename follows it.
    transport.write_bytes(f"{incoming}/READY.json", _json_bytes(ready))
    transport.promote_directory(incoming, final)
    live_ready = _parse_json_bytes(transport.read_bytes(ready_path), ready_path)
    if live_ready != ready or not _ready_matches(live_ready, plan, manifest):
        raise RuntimeError("published READY changed during atomic promotion")
    return _publication_receipt(
        plan=plan,
        manifest=manifest,
        ready=ready,
        already_ready=False,
        apply=True,
        incoming=incoming,
        uploaded_files=uploaded,
        resumed_files=resumed,
    )


class SSHPublicationTransport:
    """Production transport backed by the scheduler's configured SSH session."""

    def __init__(self, session: Any, env_setup: str):
        self.session = session
        self.env_setup = env_setup
        self.write_count = 0

    def _run(self, command: str, timeout: int = 300) -> str:
        result = self.session.run(command, timeout=timeout)
        if result.exit_code != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or command
            )
        return result.stdout

    def exists(self, path: str) -> bool:
        return (
            self.session.run(f"test -e {shlex.quote(path)}", timeout=60).exit_code == 0
        )

    def is_dir(self, path: str) -> bool:
        return (
            self.session.run(f"test -d {shlex.quote(path)}", timeout=60).exit_code == 0
        )

    def read_bytes(self, path: str) -> bytes:
        return self.session.read_text_file(path).encode("utf-8")

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        self._run(
            f"mkdir -p {shlex.quote(path)} && chmod {mode:o} {shlex.quote(path)}",
            60,
        )
        self.write_count += 1

    def upload_file(self, local: Path, remote: str) -> None:
        self.session.upload_file(str(local), remote)
        self.write_count += 1

    def write_bytes(self, path: str, value: bytes) -> None:
        self.session.write_text_file(path, value.decode("utf-8"))
        self.write_count += 1

    def replace(self, source: str, destination: str) -> None:
        self._run(f"mv -f -- {shlex.quote(source)} {shlex.quote(destination)}", 60)
        self.write_count += 1

    def file_record(self, path: str) -> Mapping[str, Any] | None:
        command = (
            f"test -f {shlex.quote(path)} && "
            f"stat -c '%s' {shlex.quote(path)} && sha256sum {shlex.quote(path)}"
        )
        result = self.session.run(command, timeout=1800)
        if result.exit_code != 0:
            return None
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            raise RuntimeError(f"invalid remote file record: {path}")
        return {"size": int(lines[0]), "sha256": lines[1].split()[0]}

    def verify_runtime(
        self,
        root: str,
        *,
        requirements_relative: str,
        expected_packages: Mapping[str, str],
    ) -> Mapping[str, str]:
        site = f"{root}/artifacts/python-site"
        requirements = f"{root}/{requirements_relative}"
        script = (
            "import importlib.metadata,json\n"
            f"names={json.dumps(sorted(expected_packages))}\n"
            "print(json.dumps({n:importlib.metadata.version(n) for n in names},sort_keys=True))\n"
        )
        expected = shlex.quote(json.dumps(dict(expected_packages), sort_keys=True))
        command = "\n".join(
            [
                "set -euo pipefail",
                self.env_setup,
                # A crash after permission sealing but before READY leaves a
                # complete read-only incoming tree.  Authenticate that runtime
                # first so restart needs no chmod or reinstall in the common
                # case; only a missing/mismatched site is reopened and repaired.
                f"if test -d {shlex.quote(site)}; then",
                f"  export PYTHONPATH={shlex.quote(site)}",
                "  actual=$(python - <<'PY'\n" + script.rstrip() + "\nPY\n)",
                f"  test \"$actual\" = {expected} || actual=''",
                "else actual=''; fi",
                'if test -z "$actual"; then',
                f"  if test -e {shlex.quote(site)}; then chmod -R u+w {shlex.quote(site)}; fi",
                f"  mkdir -p {shlex.quote(site)}",
                "  python -m pip install --disable-pip-version-check --no-input "
                f"--target {shlex.quote(site)} -r {shlex.quote(requirements)}",
                f"  export PYTHONPATH={shlex.quote(site)}",
                "fi",
                f"export PYTHONPATH={shlex.quote(site)}",
                "python - <<'PY'",
                script.rstrip(),
                "PY",
            ]
        )
        output = self._run("bash -lc " + shlex.quote(command), 3600)
        self.write_count += 1
        return json.loads(output.strip().splitlines()[-1])

    def seal_permissions(self, root: str) -> None:
        command = "\n".join(
            [
                "set -euo pipefail",
                f"chmod -R a-w {shlex.quote(root + '/artifacts')}",
                f"find {shlex.quote(root + '/artifacts')} -type d -exec chmod a+rx {{}} +",
                f"chmod 1777 {shlex.quote(root + '/runs')}",
            ]
        )
        self._run("bash -lc " + shlex.quote(command), 300)
        self.write_count += 1

    def promote_directory(self, incoming: str, destination: str) -> None:
        self._run(
            f"test ! -e {shlex.quote(destination)} && "
            f"mv -T -- {shlex.quote(incoming)} {shlex.quote(destination)}",
            60,
        )
        self.write_count += 1


@contextmanager
def scheduler_publication_transport(
    *, accounts_path: Path, scheduler_source: Path, account_name: str
) -> Iterator[SSHPublicationTransport]:
    source = scheduler_source.resolve(strict=True)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from slurm_scheduler.config import load_accounts
    from slurm_scheduler.slurm import SSHSession

    accounts = {account.name: account for account in load_accounts(accounts_path)}
    if account_name not in accounts:
        raise RuntimeError(f"unknown scheduler account: {account_name}")
    account = accounts[account_name]
    env_setup = (account.env_profiles or {}).get("pyaedt2026v1", "")
    with SSHSession(account, default_timeout=300) as session:
        yield SSHPublicationTransport(session, env_setup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--account", default="harry261")
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    if args.apply:
        with scheduler_publication_transport(
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            account_name=args.account,
        ) as transport:
            receipt = publish_bundle(args.plan, apply=True, transport=transport)
    else:
        receipt = publish_bundle(args.plan, apply=False)
    if args.receipt_out is not None:
        if not args.apply:
            raise RuntimeError(
                "--receipt-out requires --apply; dry-run writes zero files"
            )
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt_out.with_name(args.receipt_out.name + ".part")
        temporary.write_bytes(_json_bytes(receipt))
        temporary.replace(args.receipt_out)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
