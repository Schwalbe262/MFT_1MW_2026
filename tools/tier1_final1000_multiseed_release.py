"""Prepare, but never publish, the Phase-A multi-seed release.

This module intentionally has no remote transport and no Scheduler client.  It
derives a new local candidate from each authenticated 30811ea parent, overlays
the clean checkout, computes the complete repository-local Python import
closure, and asks the existing delta planner to seal a successor plan.  READY
is parent evidence only: it is never copied into, or generated below, the
candidate output directory.
"""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        atomic_json,
        read_json,
        sha256_file,
    )
    from tier1_corrected_current7_slurm_publish import load_bundle_plan
    from tier1_final1000_delta_publish import ParentIdentity, create_delta_plan
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        atomic_json,
        read_json,
        sha256_file,
    )
    from tools.tier1_corrected_current7_slurm_publish import load_bundle_plan
    from tools.tier1_final1000_delta_publish import ParentIdentity, create_delta_plan


RELEASE_PREPARATION_SCHEMA = "mft-tier1-final1000-multiseed-release-preparation-v1"
MULTISTAGE_PREPARATION_SCHEMA = (
    "mft-tier1-final1000-multiseed-four-stage-release-preparation-v1"
)
SCIENCE_BASE_REVISION = "579c651b621198953f5a45ddf898a454f599e612"
ROOT_MODULES = (
    "tools/tier1_corrected_generation_preflight.py",
    "tools/tier1_corrected_current7_slurm_seed_runner.py",
    "tools/tier1_final1000_multiseed_contract.py",
    "tools/tier1_final1000_multiseed_lane_runner.py",
    "tools/tier1_final1000_multiseed_consumer.py",
)
MANDATORY_CLOSURE = frozenset(
    {
        *ROOT_MODULES,
        "tools/tier1_final1000_slurm_launch.py",
        "tools/tier1_final1000_stage_profiles.py",
        "tools/tier1_corrected_current7_slurm_publish.py",
    }
)


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or str(path) != value
    ):
        raise RuntimeError(f"unsafe repository-relative path: {value}")
    return str(path)


def _git(code_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={code_root}", *arguments],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git authentication failed")
    return completed.stdout.strip()


def authenticate_checkout(code_root: Path) -> dict[str, Any]:
    root = code_root.resolve(strict=True)
    revision = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("release preparation requires a clean tracked checkout")
    if _git(root, "merge-base", "--is-ancestor", SCIENCE_BASE_REVISION, revision) != "":
        # merge-base --is-ancestor prints nothing on success.
        raise AssertionError("unreachable")
    return {
        "revision": revision,
        "clean": True,
        "science_base_revision": SCIENCE_BASE_REVISION,
    }


def _module_candidates(
    code_root: Path, source: str, module: str, level: int
) -> Iterable[Path]:
    source_path = PurePosixPath(source)
    module_parts = [part for part in module.split(".") if part]
    if level:
        package = list(source_path.with_suffix("").parts[:-1])
        keep = len(package) - level + 1
        if keep < 0:
            return ()
        roots = [package[:keep] + module_parts]
    else:
        roots = [module_parts]
        if module_parts and module_parts[0] not in {"tools", "regression_260707"}:
            roots.extend(
                [["tools", *module_parts], ["regression_260707", *module_parts]]
            )
    result: list[Path] = []
    for parts in roots:
        if not parts:
            continue
        result.extend(
            [
                code_root.joinpath(*parts).with_suffix(".py"),
                code_root.joinpath(*parts, "__init__.py"),
            ]
        )
    return result


def _resolve_module(
    code_root: Path, source: str, module: str, level: int
) -> str | None:
    for candidate in _module_candidates(code_root, source, module, level):
        if candidate.is_file():
            return candidate.resolve().relative_to(code_root).as_posix()
    return None


def repository_import_closure(
    code_root: Path, roots: Sequence[str] = ROOT_MODULES
) -> tuple[str, ...]:
    """Return the deterministic transitive closure of repository Python imports."""

    code_root = code_root.resolve(strict=True)
    pending = [_safe_relative(item) for item in roots]
    seen: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        source = (code_root / relative).resolve(strict=True)
        try:
            source.relative_to(code_root)
        except ValueError as exc:
            raise RuntimeError("import root escaped the checkout") from exc
        if source.suffix != ".py":
            raise RuntimeError("import closure accepts Python sources only")
        seen.add(relative)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            requested: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                requested.extend((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                requested.append((base, node.level))
                # ``from package import local_module`` needs the submodule even
                # when package/__init__.py itself is absent or empty.
                requested.extend(
                    (
                        f"{base}.{alias.name}" if base else alias.name,
                        node.level,
                    )
                    for alias in node.names
                    if alias.name != "*"
                )
            for module, level in requested:
                dependency = _resolve_module(code_root, relative, module, level)
                if dependency is not None and dependency not in seen:
                    pending.append(dependency)
    missing = sorted(MANDATORY_CLOSURE.difference(seen))
    if missing:
        raise RuntimeError(
            f"multi-seed transitive import closure is incomplete: {missing}"
        )
    return tuple(sorted(seen))


def _record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def _parent_identity(
    plan_path: Path, publication_path: Path
) -> tuple[ParentIdentity, dict[str, Any], dict[str, Any], dict[str, Path]]:
    plan, manifest, sources = load_bundle_plan(plan_path)
    identity = ParentIdentity(
        bundle_id=str(plan["bundle_id"]),
        contract_sha256=str(plan["contract_sha256"]),
        manifest_sha256=str(plan["bundle_manifest_sha256"]),
        plan_sha256=sha256_file(plan_path),
        source_map_sha256=sha256_file(Path(str(plan["local_sources"]))),
        publication_file_sha256=sha256_file(publication_path),
        remote_bundle=str(plan["remote_bundle"]),
    )
    return identity, plan, manifest, sources


def _write_candidate(
    *,
    parent_plan_path: Path,
    parent_publication_path: Path,
    code_root: Path,
    output_root: Path,
    checkout: Mapping[str, Any],
    closure: Sequence[str],
    created_at: str,
) -> tuple[Path, ParentIdentity, dict[str, Any]]:
    identity, parent_plan, parent_manifest, parent_sources = _parent_identity(
        parent_plan_path, parent_publication_path
    )
    sources = {key: Path(value) for key, value in parent_sources.items()}
    files = copy.deepcopy(parent_manifest["files"])

    # Overlay every already-bundled repository source, not merely the roots.
    # This makes .source-revision truthful while the computed closure is the
    # independently enforced minimum needed by an isolated worker.
    overlay: set[str] = set(closure)
    for bundle_relative in parent_manifest.get("code_inventory") or {}:
        prefix = "artifacts/code/"
        if not str(bundle_relative).startswith(prefix):
            continue
        repository_relative = str(bundle_relative)[len(prefix) :]
        candidate = code_root / repository_relative
        if candidate.is_file():
            overlay.add(repository_relative)
    for repository_relative in sorted(overlay):
        source = (code_root / repository_relative).resolve(strict=True)
        bundle_relative = f"artifacts/code/{repository_relative}"
        sources[bundle_relative] = source
        files[bundle_relative] = _record(source)

    candidate_root = output_root / "candidate"
    candidate_root.mkdir(parents=True, exist_ok=True)
    revision_path = candidate_root / ".source-revision"
    revision_path.write_text(str(checkout["revision"]) + "\n", encoding="ascii")
    sources["artifacts/code/.source-revision"] = revision_path
    files["artifacts/code/.source-revision"] = _record(revision_path)

    stable = {
        key: copy.deepcopy(value)
        for key, value in parent_manifest.items()
        if key not in {"bundle_id", "contract_sha256", "delta_publication"}
    }
    stable.update(
        {
            "bundle_code_revision": checkout["revision"],
            "bundle_code_clean_at_plan": True,
            "science_base_revision": checkout["science_base_revision"],
            "release_overlay": {
                "schema_version": RELEASE_PREPARATION_SCHEMA,
                "transitive_import_roots": list(ROOT_MODULES),
                "transitive_import_closure": list(closure),
                "transitive_import_closure_sha256": canonical_sha256(list(closure)),
                "isolated_remote_import_smoke_required": True,
                "parent_ready_reuse_allowed": False,
            },
            "files": files,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )
    code_inventory = {
        key: files[key] for key in sorted(files) if key.startswith("artifacts/code/")
    }
    stable["code_inventory"] = code_inventory
    stable["code_inventory_sha256"] = canonical_sha256(code_inventory)
    contract_sha = canonical_sha256(stable)
    bundle_id = f"current7-{contract_sha[:20]}"
    manifest = {**stable, "bundle_id": bundle_id, "contract_sha256": contract_sha}
    manifest_path = candidate_root / "bundle_manifest.json"
    source_map_path = candidate_root / "local_sources.json"
    plan_path = candidate_root / "offload_plan.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        source_map_path, {key: str(path) for key, path in sorted(sources.items())}
    )

    plan = {
        key: copy.deepcopy(value)
        for key, value in parent_plan.items()
        if key not in {"delta_publication", "bridge_clone"}
    }
    plan.update(
        {
            "created_at": created_at,
            "bundle_id": bundle_id,
            "contract_sha256": contract_sha,
            "bundle_manifest": str(manifest_path),
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "local_sources": str(source_map_path),
            "remote_bundle": f"{str(plan['remote_root']).rstrip('/')}/{bundle_id}",
            "stage_performed": False,
            "submission_performed": False,
        }
    )
    atomic_json(plan_path, plan)
    if any(path.name == "READY" for path in candidate_root.rglob("*")):
        raise RuntimeError("candidate preparation attempted to reuse READY")
    load_bundle_plan(plan_path)
    return plan_path, identity, manifest


def isolated_import_smoke(
    plan_path: Path, closure: Sequence[str], *, python: str = sys.executable
) -> dict[str, Any]:
    """Import every closure member from a directory containing only bundle code."""

    _, manifest, sources = load_bundle_plan(plan_path)
    with tempfile.TemporaryDirectory(prefix="mft-multiseed-import-") as temporary:
        code = Path(temporary) / "artifacts" / "code"
        for relative in closure:
            bundle_relative = f"artifacts/code/{relative}"
            if bundle_relative not in manifest["code_inventory"]:
                raise RuntimeError(f"closure source is absent from bundle: {relative}")
            destination = code / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[bundle_relative], destination)
        imports = [relative[:-3].replace("/", ".") for relative in closure]
        script = (
            "import importlib,sys;"
            f"sys.path[:0]=[{str(code)!r},{str(code / 'tools')!r}];"
            f"mods={imports!r};"
            "[importlib.import_module(m) for m in mods]"
        )
        completed = subprocess.run(
            [python, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "isolated remote import smoke failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
    return {
        "module_count": len(closure),
        "closure_sha256": canonical_sha256(list(closure)),
        "isolated": True,
        "passed": True,
    }


def prepare_stage(
    *,
    parent_plan_path: Path,
    parent_publication_path: Path,
    code_root: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    checkout = authenticate_checkout(code_root)
    closure = repository_import_closure(code_root)
    candidate_path, identity, candidate_manifest = _write_candidate(
        parent_plan_path=parent_plan_path.resolve(strict=True),
        parent_publication_path=parent_publication_path.resolve(strict=True),
        code_root=code_root.resolve(strict=True),
        output_root=output_root.resolve(),
        checkout=checkout,
        closure=closure,
        created_at=created_at,
    )
    smoke = isolated_import_smoke(candidate_path, closure)
    delta_plan, delta_manifest, delta_receipt = create_delta_plan(
        candidate_path,
        parent_plan_path,
        parent_publication_path,
        output_root / "delta",
        identity=identity,
    )
    if any(path.name == "READY" for path in output_root.rglob("*")):
        raise RuntimeError("release preparation output contains READY")
    unsigned = {
        "schema_version": RELEASE_PREPARATION_SCHEMA,
        "checkout": dict(checkout),
        "parent_identity": asdict(identity),
        "candidate_plan": str(candidate_path),
        "candidate_bundle_id": candidate_manifest["bundle_id"],
        "candidate_contract_sha256": candidate_manifest["contract_sha256"],
        "delta_plan": str(
            output_root / "delta" / delta_plan["bundle_id"] / "offload_plan.json"
        ),
        "delta_bundle_id": delta_manifest["bundle_id"],
        "delta_contract_sha256": delta_manifest["contract_sha256"],
        "delta_plan_receipt_sha256": delta_receipt["receipt_sha256"],
        "transitive_import_closure": list(closure),
        "isolated_import_smoke": smoke,
        "ready_reused": False,
        "remote_write_performed": False,
        "scheduler_submission_performed": False,
    }
    receipt = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    atomic_json(output_root / "release_preparation_receipt.json", receipt)
    return receipt


def prepare_release(
    *,
    release_root: Path,
    code_root: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    """Prepare all four successors pinned by the sealed 30811ea remote gate."""

    release_root = release_root.resolve(strict=True)
    gate_path = release_root / "evidence" / "remote_release_gate.json"
    gate = read_json(gate_path.resolve(strict=True))
    gate_unsigned = {
        key: value for key, value in gate.items() if key != "evidence_sha256"
    }
    stages = gate.get("stages")
    if (
        gate.get("source_revision") != "30811ea159c6379647291e67f1961d74bcb23658"
        or gate.get("status")
        != "passed_remote_publish_no_write_replay_and_two_pass_full_sha"
        or gate.get("evidence_sha256") != canonical_sha256(gate_unsigned)
        or not isinstance(stages, list)
        or len(stages) != 4
        or len({stage.get("stage_id") for stage in stages if isinstance(stage, dict)})
        != 4
    ):
        raise RuntimeError("30811ea four-stage release gate is not authenticated")
    receipts: dict[str, Any] = {}
    for row in sorted(stages, key=lambda item: str(item["stage_id"])):
        stage_id = str(row["stage_id"])
        plan_path = Path(str(row["plan_path"])).resolve(strict=True)
        publication = row.get("publication") or {}
        pass2 = publication.get("pass2") or {}
        publication_path = Path(str(pass2.get("receipt_path") or "")).resolve(
            strict=True
        )
        for path in (plan_path, publication_path):
            try:
                path.relative_to(release_root)
            except ValueError as exc:
                raise RuntimeError(
                    "30811ea parent evidence escaped release root"
                ) from exc
        if (
            sha256_file(plan_path) != row.get("plan_file_sha256")
            or sha256_file(publication_path) != pass2.get("receipt_file_sha256")
            or pass2.get("already_ready") is not True
            or pass2.get("remote_write_performed") is not False
        ):
            raise RuntimeError("30811ea stage parent identity drifted")
        receipts[stage_id] = prepare_stage(
            parent_plan_path=plan_path,
            parent_publication_path=publication_path,
            code_root=code_root,
            output_root=output_root / stage_id,
            created_at=created_at,
        )
    unsigned = {
        "schema_version": MULTISTAGE_PREPARATION_SCHEMA,
        "parent_release_gate_path": str(gate_path),
        "parent_release_gate_file_sha256": sha256_file(gate_path),
        "parent_release_gate_evidence_sha256": gate["evidence_sha256"],
        "stages": receipts,
        "stage_count": 4,
        "ready_reused": False,
        "remote_write_performed": False,
        "scheduler_submission_performed": False,
    }
    receipt = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    atomic_json(output_root / "release_preparation_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--parent-publication", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.release_root is not None:
        if args.parent_plan is not None or args.parent_publication is not None:
            raise RuntimeError("choose either --release-root or one explicit parent")
        receipt = prepare_release(
            release_root=args.release_root,
            code_root=args.code_root,
            output_root=args.output_root,
            created_at=args.created_at,
        )
    else:
        if args.parent_plan is None or args.parent_publication is None:
            raise RuntimeError("explicit preparation requires both parent paths")
        receipt = prepare_stage(
            parent_plan_path=args.parent_plan,
            parent_publication_path=args.parent_publication,
            code_root=args.code_root,
            output_root=args.output_root,
            created_at=args.created_at,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
