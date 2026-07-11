"""Fail closed unless pinned solver/library commits are remotely fetchable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def advertised_heads(repo_root):
    repo_root = Path(repo_root).resolve()
    output = subprocess.check_output(
        ["git", "ls-remote", "--heads", "origin"],
        cwd=repo_root, text=True, stderr=subprocess.STDOUT,
    )
    heads = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and FULL_SHA.fullmatch(parts[0].lower()) \
                and parts[1].startswith("refs/heads/"):
            heads[parts[1]] = parts[0].lower()
    if not heads:
        raise RuntimeError(f"origin advertises no branch heads: {repo_root}")
    return heads


def require_advertised_revision(repo_root, revision, label):
    revision = str(revision or "").strip().lower()
    if not FULL_SHA.fullmatch(revision):
        raise RuntimeError(f"{label} revision must be a full SHA")
    heads = advertised_heads(repo_root)
    refs = sorted(ref for ref, commit in heads.items() if commit == revision)
    if not refs:
        raise RuntimeError(
            f"{label} revision {revision} is not an advertised origin branch head"
        )
    return {"revision": revision, "refs": refs, "repo_root": str(Path(repo_root).resolve())}


def validate_deployment(
    solver_root, solver_revision, library_root, library_revision,
):
    return {
        "solver": require_advertised_revision(
            solver_root, solver_revision, "solver"
        ),
        "library": require_advertised_revision(
            library_root, library_revision, "library"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-root", required=True)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-root", required=True)
    parser.add_argument("--library-revision", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(validate_deployment(
        args.solver_root, args.solver_revision,
        args.library_root, args.library_revision,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
