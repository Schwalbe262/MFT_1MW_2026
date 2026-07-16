"""Read-only SSH audit for q22 eligible account package checkouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paramiko
import yaml


def audit(config_path: Path, accounts: list[str], expected: str) -> list[dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("accounts config has no accounts list")
    configured = {
        str(row.get("name") or ""): row for row in rows if isinstance(row, dict)
    }
    command_prefix = (
        "set -eu; root=\"$HOME/slurm_scheduler/aedt_pool_pkg\"; echo stage=root; "
        f"test \"$(git -C \"$root\" rev-parse HEAD)\" = \"{expected}\"; "
        "echo stage=head; git -C \"$root\" diff --quiet HEAD --; "
        # The host wrapper writes one untracked diagnostic log at the package
        # root. It is not importable source. Everything else, including any
        # untracked Python file, remains fail-closed.
        "status=$(git -C \"$root\" status --porcelain --untracked-files=all "
        "| grep -Ev '^\\?\\? batch\\.log$' || true); "
        "test -z \"$status\" || { printf 'status=%s\\n' \"$status\"; exit 1; }; "
        "echo stage=clean; "
    )
    command_suffix = (
        "; PYTHONPATH=\"$root\" python -c \"from slurm_scheduler.aedt_attach_client "
        "import AedtProjectLease; assert hasattr(AedtProjectLease, "
        "'wait_for_native_pipeline_barrier')\"; echo stage=import; "
        f"printf '%s\\n' {expected}"
    )
    result = []
    for name in accounts:
        row = configured.get(name)
        if not row:
            raise RuntimeError(f"unknown eligible account: {name}")
        if "conda:pyaedt2026v1" not in (row.get("capabilities") or []):
            raise RuntimeError(f"eligible account lacks pyaedt2026v1: {name}")
        profiles = row.get("env_profiles") or {}
        env_setup = str(profiles.get("pyaedt2026v1") or "").strip()
        if not env_setup:
            raise RuntimeError(f"eligible account has no pyaedt2026v1 setup: {name}")
        command = command_prefix + env_setup + command_suffix
        key = Path(str(row.get("private_key_path") or ""))
        if not key.is_file():
            raise RuntimeError(f"SSH key is missing for {name}: {key}")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=str(row.get("host") or ""),
                port=int(row.get("port") or 22),
                username=str(row.get("username") or ""),
                key_filename=str(key),
                look_for_keys=False,
                allow_agent=False,
                timeout=15,
                banner_timeout=15,
                auth_timeout=15,
            )
            _stdin, stdout, stderr = client.exec_command(command, timeout=45)
            output = stdout.read().decode("utf-8", errors="replace").strip()
            error = stderr.read().decode("utf-8", errors="replace").strip()
            return_code = stdout.channel.recv_exit_status()
        finally:
            client.close()
        if return_code or output.splitlines()[-1:] != [expected]:
            raise RuntimeError(
                f"remote package audit failed for {name} rc={return_code}: "
                f"{error or output or '<no output>'}"
            )
        result.append({"account": name, "package": expected})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-config", required=True, type=Path)
    parser.add_argument("--account", action="append", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.accounts_config, args.account, args.expected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
