from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import tier1_final1000_multiseed_evidence_check as evidence_check
from tools.tier1_corrected_current7_receipt import canonical_sha256


def _git(root: Path, *arguments: str, text: bool = True):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "evidence@example.invalid")
    _git(root, "config", "user.name", "Evidence Test")
    source = root / "worker.py"
    source.write_bytes(b"VALUE = 1\n")
    _git(root, "add", "worker.py")
    _git(root, "commit", "-m", "fixture")
    revision = _git(root, "rev-parse", "HEAD").strip()
    unsigned = {
        "schema_version": "test-multiseed-evidence-v1",
        "verified_code_revision": revision,
        "code_sha256": {
            "worker.py": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        },
    }
    evidence = {**unsigned, "evidence_sha256": canonical_sha256(unsigned)}
    path = root / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return root, path, revision


def test_evidence_self_check_hashes_committed_bytes_not_worktree_eol(tmp_path: Path):
    root, evidence, revision = _fixture(tmp_path)
    (root / "worker.py").write_bytes(b"VALUE = 1\r\n")

    result = evidence_check.validate_evidence(evidence, code_root=root)

    assert result["passed"] is True
    assert result["verified_code_revision"] == revision
    assert result["checked_file_count"] == 1
    assert result["hash_policy"] == "sha256(git-cat-file-blob-revision:path)"


def test_evidence_self_check_rejects_hash_and_evidence_seal_tamper(tmp_path: Path):
    root, path, _ = _fixture(tmp_path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["code_sha256"]["worker.py"] = "f" * 64
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence seal mismatch"):
        evidence_check.validate_evidence(path, code_root=root)

    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    evidence["evidence_sha256"] = canonical_sha256(unsigned)
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed/archive byte SHA mismatch"):
        evidence_check.validate_evidence(path, code_root=root)
