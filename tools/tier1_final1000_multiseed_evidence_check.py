"""Verify multi-seed evidence hashes against committed Git blob bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import read_json
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import read_json


DEFAULT_EVIDENCE = Path("docs/tier1_final1000_multiseed_deploy_evidence_20260722.json")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or str(path) != value
    ):
        raise RuntimeError(f"unsafe evidence path: {value}")
    return str(path)


def _git(
    code_root: Path, arguments: Sequence[str], *, text: bool
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={code_root}", *arguments],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode:
        error = completed.stderr if text else completed.stderr.decode(errors="replace")
        raise RuntimeError(error.strip() or "git evidence authentication failed")
    return completed


def _blob(code_root: Path, revision: str, relative: str) -> bytes:
    return bytes(
        _git(
            code_root, ["cat-file", "blob", f"{revision}:{relative}"], text=False
        ).stdout
    )


def validate_evidence(
    evidence_path: Path,
    *,
    code_root: Path,
    require_head_match: bool = True,
) -> dict[str, Any]:
    """Validate hashes using Git's canonical committed bytes, never worktree EOLs."""

    root = code_root.resolve(strict=True)
    evidence = read_json(evidence_path.resolve(strict=True))
    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    if evidence.get("evidence_sha256") != canonical_sha256(unsigned):
        raise RuntimeError("multi-seed evidence seal mismatch")
    revision = str(evidence.get("verified_code_revision") or "")
    hashes = evidence.get("code_sha256")
    if _REVISION.fullmatch(revision) is None or not isinstance(hashes, Mapping):
        raise RuntimeError(
            "evidence has no exact verified code revision/hash inventory"
        )
    _git(root, ["cat-file", "-e", f"{revision}^{{commit}}"], text=True)
    _git(root, ["merge-base", "--is-ancestor", revision, "HEAD"], text=True)
    head = _git(root, ["rev-parse", "HEAD"], text=True).stdout.strip()
    checked: dict[str, str] = {}
    for raw_relative, raw_expected in sorted(hashes.items()):
        relative = _relative(str(raw_relative))
        expected = str(raw_expected)
        if _SHA256.fullmatch(expected) is None:
            raise RuntimeError(f"invalid evidence SHA256: {relative}")
        committed = _blob(root, revision, relative)
        actual = hashlib.sha256(committed).hexdigest()
        if actual != expected:
            raise RuntimeError(f"committed/archive byte SHA mismatch: {relative}")
        if require_head_match and _blob(root, head, relative) != committed:
            raise RuntimeError(f"evidenced code changed after verification: {relative}")
        checked[relative] = actual
    if not checked:
        raise RuntimeError("evidence code hash inventory is empty")
    return {
        "verified_code_revision": revision,
        "head_revision": head,
        "hash_policy": "sha256(git-cat-file-blob-revision:path)",
        "head_tree_match_required": bool(require_head_match),
        "checked_file_count": len(checked),
        "checked": checked,
        "passed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-head-drift", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_evidence(
        args.evidence,
        code_root=args.code_root,
        require_head_match=not args.allow_head_drift,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
