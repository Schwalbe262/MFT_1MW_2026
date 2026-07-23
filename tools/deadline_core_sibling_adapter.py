"""Fail-closed adapter for the reviewed deadline Full core-policy siblings.

The native solver result is never rewritten.  This tool authenticates the
reviewed 8/16-core wrapper revisions, the immutable runtime collector evidence,
and the existing ``deadline_fea_gate`` physics/quality contracts before it
wraps that native result in the canonical Full-result schema.

It has no Scheduler client and cannot submit, cancel, promote, or mutate data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any


PHYSICS_SOLVER_REVISION = "8a8d90f68e8728669282f586f24304c7cc807029"
CORE8_SOLVER_REVISION = "8323c46e85e9ac7dbb153bc46024223ea2ee99ea"
CORE16_SOLVER_REVISION = "146142be579e2f3e45f12961214018a4e28445c5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SUBMISSION_CODE_REVISION = "c5518fc2a4de8fa2261462bfcf4b04483aaa3b63"

DEADLINE_GATE_SOURCE_SHA256 = (
    "62a85927e2d45faeb22f98ed392ef3646035017d6253312cd71baaa828074acf"
)
RESULT_SCHEMA = "mft-deadline-design-fea-result-v1"
JOIN_SCHEMA = "mft-deadline-speculative-full-join-v1"
REVISION_ATTESTATION_SCHEMA = (
    "mft-deadline-core-sibling-revision-attestation-v1"
)
EXECUTION_ATTESTATION_SCHEMA = (
    "mft-deadline-full-runtime-execution-attestation-v1"
)
ADAPTER_RECEIPT_SCHEMA = "mft-deadline-core-wrapper-submission-adapter-v1"
START_EVIDENCE_SCHEMA = "mft-deadline-full-runtime-start-evidence-v1"
TERMINAL_CAPTURE_SCHEMA = "mft-deadline-full-runtime-capture-v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# These are reviewed, content-addressed Git facts.  Any additional file, parent,
# tree, or byte-level patch fails closed before a runtime result is considered.
REVIEWED_REVISIONS = {
    PHYSICS_SOLVER_REVISION: {
        "parent": "47191a5146e087c10bdcdfaab72043fce5d65b73",
        "tree": "c34a7e3e0ec9583d879386898e54d8feaee045fa",
        "blobs": {
            "run_simulation_260706.py":
                "550e1a026330b3f98781ae1a0c5396ab61f0401c",
        },
    },
    CORE8_SOLVER_REVISION: {
        "parent": PHYSICS_SOLVER_REVISION,
        "tree": "a065cae974c33b3e22c1e4dfda127a8ee50c1ab7",
        "blobs": {
            "run_simulation_260706.py":
                "3b3c2e7505bbae148edace1daa81e1e1c52ba6b6",
            "tests/test_standalone_core_optin.py":
                "ca7cac1b4506ca823e34a6d376f7050cb42f5557",
        },
    },
    CORE16_SOLVER_REVISION: {
        "parent": CORE8_SOLVER_REVISION,
        "tree": "a6df011bc2e429990c77b94fda77f25c584326a3",
        "blobs": {
            "run_simulation_260706.py":
                "ad5fa8adb834d71558e15c01a1230bc55808303e",
            "tests/test_standalone_core_optin.py":
                "3f086a7d0e4033c01d42ddd666aa5e45ac5f8611",
        },
    },
}

REVIEWED_EDGES = (
    {
        "parent": PHYSICS_SOLVER_REVISION,
        "child": CORE8_SOLVER_REVISION,
        "changed_paths": {
            "run_simulation_260706.py": "M",
            "tests/test_standalone_core_optin.py": "A",
        },
        "patch_sha256":
            "c4c8d68e709fb0d7d1894eb791f9ec7f24fdcae37de9482a927d2ad6afbb2a80",
        "source_patch_sha256":
            "2d8fb11b36d4cd07b14800e070ffc244a79d55684320c42dfe26d832cb2379c4",
        "test_patch_sha256":
            "0a4a4232d8d284e446da2f75c752e5013f76baea7fe5c6a8e46d2d65ed538485",
        "numstat": {
            "run_simulation_260706.py": [297, 8],
            "tests/test_standalone_core_optin.py": [246, 0],
        },
    },
    {
        "parent": CORE8_SOLVER_REVISION,
        "child": CORE16_SOLVER_REVISION,
        "changed_paths": {
            "run_simulation_260706.py": "M",
            "tests/test_standalone_core_optin.py": "M",
        },
        "patch_sha256":
            "02313b9b404f4360f25f9d23b1e825265d912defe3bf455c4c0f8cc92fe92759",
        "source_patch_sha256":
            "0c603b34ca3145d6e0e9b1b51445070c94f518787b68ad7276471f17fd047e3d",
        "test_patch_sha256":
            "82ab9e2b9772f1a15aa0c860d9800cb1fa15dca1ad7ef390a06776a89c8b4b5e",
        "numstat": {
            "run_simulation_260706.py": [206, 14],
            "tests/test_standalone_core_optin.py": [166, 0],
        },
    },
)

# This intentionally authorizes only the already-running offline Full siblings.
# In particular, task 95016 can join only Standard task 95014 or 95015.
CANDIDATES = {
    95016: {
        "task_name": "mft-d418full8c-canary-f-868642944b2b3247",
        "plan_sha256":
            "91acf9616360f81af2032c79a59cf30de4fc934574f8cd71864b9ca798b3f273",
        "source_receipt_sha256":
            "71f8bde8bafe3c66c66043a39bdec2afa2214d81d540f73713a8099f2bb3e6eb",
        "identity":
            "f66f0d95affe97e738f2d2a82cc8634dbfb3c10e8fce5ebe12f413780ab69968",
        "digest":
            "868642944b2b3247f9690bd52ca7891f9cb36db590ed2f8c04eabbcf5fc7fe7f",
        "runtime_revision": CORE8_SOLVER_REVISION,
        "cores": 8,
        "core_contract_version": "mft-standalone-core-optin-v1",
        "core_auth":
            "9f8a84190adfefa3d4776892f46b6b6ec59b246935096ece0e3863a27301acc7",
        "license_snapshot_sha256": None,
        "standard_task_ids": frozenset({95014, 95015}),
    },
    95019: {
        "task_name": "mft-b428tim2k3fan6specfull8c-f-41de9e22c0841cd3",
        "plan_sha256":
            "0b183bb3489b3596f9b1b43d0ff921fbd20494253d76396357ff3a7e5fb82a7a",
        "source_receipt_sha256":
            "230b93288ca05cba60d7bc437031190319609745eaf030e54fc12b5e0277b917",
        "identity":
            "a9e1dc42f43a63e4c5a03eb85bd7f32f2d2ea225d5676573d652b602c10f304b",
        "digest":
            "41de9e22c0841cd3773020a0f6ccd96ee15e05ea10058cd537bd88d2f0e306ca",
        "runtime_revision": CORE8_SOLVER_REVISION,
        "cores": 8,
        "core_contract_version": "mft-standalone-core-optin-v1",
        "core_auth":
            "9f8a84190adfefa3d4776892f46b6b6ec59b246935096ece0e3863a27301acc7",
        "license_snapshot_sha256": None,
        "standard_task_ids": frozenset({95035}),
    },
    95030: {
        "task_name": "mft-b428t1f5p5full16c-f-d07236e9d0f60381",
        "plan_sha256":
            "e8a16c1a8ed5a3f067e01028fb2cf4e8dea356a0a06908f59da5ebbcfbb96c4d",
        "source_receipt_sha256":
            "9704d78fff02793d327a23142fbf9545c9c3d8e270e52d88560c8c423131ce0f",
        "identity":
            "9df44ec75896c09fafe4b6712d7ded31cc6cc2b98d6b11e33614372ebd3868e0",
        "digest":
            "d07236e9d0f603810554bfecacd5debf88f3c83223e870236ab6076c9e780b87",
        "runtime_revision": CORE16_SOLVER_REVISION,
        "cores": 16,
        "core_contract_version": "mft-standalone-core-16-optin-v1",
        "core_auth":
            "a0fd4f74f215705f5f4208ab5a3e6cf9aec626dc295e8ed58dd490d0c3f0fbe3",
        "license_snapshot_sha256":
            "c3ca75613837264ddccd4f1aaf980f2fce7d2b9849c71ca2b9f57d4ade1162b9",
        "standard_task_ids": frozenset({95022}),
    },
    95032: {
        "task_name": "mft-a422t1f5p5full16c-f-711812e66bd3c82b",
        "plan_sha256":
            "3a0f72383589a82bda02857a052bc570677cee5672f85747839d2df4f5095033",
        "source_receipt_sha256":
            "883a6271a7277be1456f8811ce04aa38b9f961215256ed25aa1ba73a71773146",
        "identity":
            "c691d766d4ba9bb4a54d050c65166fef513c8cad58ce96a6f30442cf8358cbd1",
        "digest":
            "711812e66bd3c82bc1abb5249d7220ada03392f5dac32024fae7549d45ac0697",
        "runtime_revision": CORE16_SOLVER_REVISION,
        "cores": 16,
        "core_contract_version": "mft-standalone-core-16-optin-v1",
        "core_auth":
            "9f43cd81cf79fc4bc786eea4a11937facfae3f4d0723d161d903b13c20f1b32f",
        "license_snapshot_sha256":
            "04d600475854ff002f976a1c8b06a816781149fd452de7b51e3bb40cadbdf812",
        "standard_task_ids": frozenset({95031}),
    },
    95033: {
        "task_name": "mft-b428t2f6p5full16c-f-697dd4ac3e0b086b",
        "plan_sha256":
            "752d2f1d87e2c93d3108f62205fa7347b6a268f8cea7161a4f2aa9a8a83d3ecf",
        "source_receipt_sha256":
            "9bebbdb6cd02f47eba107e0df1983530fee106dc3836ea93dcb708b77bcffab4",
        "identity":
            "816ce57169fe58f0dcea4958cc31d80e54526bca22fddf563585efedde42bf5f",
        "digest":
            "697dd4ac3e0b086ba8fe308833dab72a4eb6d782a5037b3041b766d76d05530d",
        "runtime_revision": CORE16_SOLVER_REVISION,
        "cores": 16,
        "core_contract_version": "mft-standalone-core-16-optin-v1",
        "core_auth":
            "f1781e043a78fb9cdbb39347c4d19ba0b547d955b46c9022ec03ddab9d127c49",
        "license_snapshot_sha256":
            "dcc373513b50efd4981932af8dacca6e5bec0087388c9f6f42f01954d106d269",
        "standard_task_ids": frozenset({95037}),
    },
    95039: {
        "task_name": "mft-d414t2f5full16cs-f-dce5021c699da8ee",
        "plan_sha256":
            "7f192796acdc8fc64524c388929957f64f3b0d1ba1981b19b038d1bcf3af0066",
        "source_receipt_sha256":
            "9393c819d2612f43ac37b20a0f1bee60eae78656ff955f0cc5cf53a33d64d293",
        "identity":
            "9d29a2a3afe8798a11df66e6cccdc7015c8590577be4e0b76b085678dc8c2aca",
        "digest":
            "dce5021c699da8ee8e5337befbd188ba04d6e66199fffa8819db728731773b3c",
        "runtime_revision": CORE16_SOLVER_REVISION,
        "cores": 16,
        "core_contract_version": "mft-standalone-core-16-optin-v1",
        "core_auth":
            "2ac69edd94ba4947083a41ea6a6052b7b8bd20452a20ca24a181a45d8d6b4933",
        "license_snapshot_sha256":
            "64ec5f53b744b13cd8f7348039e10d2b806404ec8866d1bba4599c5897e362f2",
        "standard_task_ids": frozenset({95034}),
    },
    95040: {
        "task_name": "mft-b428t2f7full16cs-f-e2d5eaff695cd6dc",
        "plan_sha256":
            "aa6c836b5f8ed18a7833b2c7954ed1a261ba1f12d9792521f075fe73d9894d20",
        "source_receipt_sha256":
            "43924664c38920e8c351d8885579c5daefcb00ac2e9b7af94e0f85fd26287319",
        "identity":
            "6d04f1e38dccca3a4e60d296ddc153569aa362de494fb656b1568acd87332013",
        "digest":
            "e2d5eaff695cd6dc1d6e3e61b632f0e2e7fb78ab87b4b1a6626dd5a4279fbe5d",
        "runtime_revision": CORE16_SOLVER_REVISION,
        "cores": 16,
        "core_contract_version": "mft-standalone-core-16-optin-v1",
        "core_auth":
            "b37084d42ff934247dc4f96399e7f12dd75d7543b1b86906e459acbf259c5586",
        "license_snapshot_sha256":
            "faa26bd0efa5a638bebd82992cf5dcf5f56fed363a13ac27c1254339605b38bc",
        "standard_task_ids": frozenset({95038}),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: os.PathLike[str] | str) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(body)


def _payload_sha256(value: dict) -> str:
    body = dict(value)
    body.pop("payload_sha256", None)
    return _canonical_sha256(body)


def _exact_sha(value: Any, label: str) -> str:
    text = str(value).lower()
    if not _HEX64.fullmatch(text):
        raise RuntimeError(f"{label} must be an exact lowercase SHA-256")
    return text


def _read_json(path: os.PathLike[str] | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _read_hashed_json(
    path: os.PathLike[str] | str,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict]:
    resolved = Path(path).resolve(strict=True)
    expected = _exact_sha(expected_sha256, f"{label} SHA-256")
    if _sha256(resolved) != expected:
        raise RuntimeError(f"{label} SHA-256 changed")
    return resolved, _read_json(resolved)


def _file_reference(path: os.PathLike[str] | str) -> dict:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_reference(
    reference: Any,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(reference, dict):
        raise RuntimeError(f"{label} reference is missing")
    if set(reference) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"{label} reference fields changed")
    path = Path(reference["path"]).resolve(strict=True)
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise RuntimeError(f"{label} reference path changed")
    if (
        _sha256(path) != _exact_sha(reference["sha256"], f"{label} reference")
        or path.stat().st_size != int(reference["size_bytes"])
    ):
        raise RuntimeError(f"{label} reference content changed")
    return path


def _atomic_write_once(path: Path, payload: dict) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to replace immutable output: {output}")
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
        newline="\n",
    ) as handle:
        handle.write(body)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def _git(repo: Path, arguments: list[str], *, text: bool = True):
    command = [
        "git",
        "-c",
        f"safe.directory={repo}",
        "-C",
        str(repo),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=text,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        stderr = (
            completed.stderr.strip()
            if text else completed.stderr.decode("utf-8", errors="replace")
        )
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}: {stderr}")
    return completed.stdout


def _git_is_ancestor(repo: Path, parent: str, child: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            parent,
            child,
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def _git_patch(repo: Path, parent: str, child: str, paths=None) -> bytes:
    arguments = [
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        parent,
        child,
    ]
    if paths:
        arguments.extend(["--", *paths])
    return _git(repo, arguments, text=False)


def _git_changed_paths(repo: Path, parent: str, child: str) -> dict[str, str]:
    value = _git(repo, ["diff", "--name-status", parent, child])
    changed = {}
    for line in value.splitlines():
        status, path = line.split("\t", 1)
        changed[path.replace("\\", "/")] = status
    return changed


def _git_numstat(repo: Path, parent: str, child: str) -> dict[str, list[int]]:
    value = _git(repo, ["diff", "--numstat", parent, child])
    observed = {}
    for line in value.splitlines():
        added, removed, path = line.split("\t", 2)
        observed[path.replace("\\", "/")] = [int(added), int(removed)]
    return observed


def _git_blobs(repo: Path, revision: str, paths: list[str]) -> dict[str, str]:
    value = _git(repo, ["ls-tree", revision, "--", *paths])
    observed = {}
    for line in value.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, object_id = metadata.split()
        if mode != "100644" or kind != "blob":
            raise RuntimeError(f"reviewed path is not a regular blob: {path}")
        observed[path.replace("\\", "/")] = object_id
    return observed


def build_revision_attestation(repo_root: Path) -> dict:
    repo = repo_root.resolve(strict=True)
    revisions = {}
    for revision, expected in REVIEWED_REVISIONS.items():
        resolved = _git(repo, ["rev-parse", f"{revision}^{{commit}}"]).strip()
        parent = _git(repo, ["show", "-s", "--format=%P", revision]).strip()
        tree = _git(repo, ["show", "-s", "--format=%T", revision]).strip()
        blobs = _git_blobs(repo, revision, list(expected["blobs"]))
        if (
            resolved != revision
            or parent != expected["parent"]
            or tree != expected["tree"]
            or blobs != expected["blobs"]
        ):
            raise RuntimeError(f"reviewed revision drifted: {revision}")
        revisions[revision] = {
            "parent": parent,
            "tree": tree,
            "reviewed_blobs": blobs,
        }

    edges = []
    for expected in REVIEWED_EDGES:
        parent = expected["parent"]
        child = expected["child"]
        changed = _git_changed_paths(repo, parent, child)
        numstat = _git_numstat(repo, parent, child)
        complete_patch = _sha256_bytes(_git_patch(repo, parent, child))
        source_patch = _sha256_bytes(_git_patch(
            repo, parent, child, ["run_simulation_260706.py"]
        ))
        test_patch = _sha256_bytes(_git_patch(
            repo, parent, child, ["tests/test_standalone_core_optin.py"]
        ))
        if (
            not _git_is_ancestor(repo, PHYSICS_SOLVER_REVISION, child)
            or REVIEWED_REVISIONS[child]["parent"] != parent
            or changed != expected["changed_paths"]
            or numstat != expected["numstat"]
            or complete_patch != expected["patch_sha256"]
            or source_patch != expected["source_patch_sha256"]
            or test_patch != expected["test_patch_sha256"]
        ):
            raise RuntimeError(f"reviewed core-policy edge drifted: {child}")
        edges.append({
            "parent": parent,
            "child": child,
            "direct_child": True,
            "physics_base_is_ancestor": True,
            "changed_paths": changed,
            "numstat": numstat,
            "patch_sha256": complete_patch,
            "source_patch_sha256": source_patch,
            "test_patch_sha256": test_patch,
            "changed_path_scope": "solver_entry_core_policy_and_tests_only",
            "physical_model_or_result_extraction_change": False,
        })

    payload = {
        "schema_version": REVISION_ATTESTATION_SCHEMA,
        "created_at": _now(),
        "repository": str(repo),
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "runtime_solver_revisions": [
            CORE8_SOLVER_REVISION,
            CORE16_SOLVER_REVISION,
        ],
        "revisions": revisions,
        "edges": edges,
        "revision_chain_valid": True,
        "physical_model_or_result_extraction_change": False,
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutated": False,
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
        "final_design_approved": False,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _validate_revision_attestation(
    path: Path,
    expected_sha256: str,
    repo_root: Path | None = None,
) -> dict:
    _, evidence = _read_hashed_json(
        path, expected_sha256, "revision attestation"
    )
    if (
        evidence.get("schema_version") != REVISION_ATTESTATION_SCHEMA
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
        or evidence.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or evidence.get("runtime_solver_revisions")
        != [CORE8_SOLVER_REVISION, CORE16_SOLVER_REVISION]
        or evidence.get("revision_chain_valid") is not True
        or evidence.get("physical_model_or_result_extraction_change")
        is not False
        or any(
            evidence.get(name) is not False
            for name in (
                "automatic_promotion_allowed",
                "canonical_dataset_mutated",
                "scheduler_configuration_mutated",
                "task_cancellation_performed",
                "final_design_approved",
            )
        )
    ):
        raise RuntimeError("revision attestation contract drifted")
    exact_revisions = {
        revision: {
            "parent": contract["parent"],
            "tree": contract["tree"],
            "reviewed_blobs": contract["blobs"],
        }
        for revision, contract in REVIEWED_REVISIONS.items()
    }
    exact_edges = [{
        "parent": contract["parent"],
        "child": contract["child"],
        "direct_child": True,
        "physics_base_is_ancestor": True,
        "changed_paths": contract["changed_paths"],
        "numstat": contract["numstat"],
        "patch_sha256": contract["patch_sha256"],
        "source_patch_sha256": contract["source_patch_sha256"],
        "test_patch_sha256": contract["test_patch_sha256"],
        "changed_path_scope": "solver_entry_core_policy_and_tests_only",
        "physical_model_or_result_extraction_change": False,
    } for contract in REVIEWED_EDGES]
    if (
        evidence.get("revisions") != exact_revisions
        or evidence.get("edges") != exact_edges
    ):
        raise RuntimeError("revision ancestry/diff attestation drifted")
    expected_static = {
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "runtime_solver_revisions": [
            CORE8_SOLVER_REVISION, CORE16_SOLVER_REVISION
        ],
        "revisions": evidence.get("revisions"),
        "edges": evidence.get("edges"),
    }
    if repo_root is not None:
        current = build_revision_attestation(repo_root)
        current_static = {
            name: current[name]
            for name in expected_static
        }
        if expected_static != current_static:
            raise RuntimeError("revision attestation no longer matches Git")
    return evidence


def _load_deadline_gate(code_root: Path) -> ModuleType:
    source = code_root.resolve(strict=True) / "tools" / "deadline_fea_gate.py"
    if _sha256(source) != DEADLINE_GATE_SOURCE_SHA256:
        raise RuntimeError("deadline_fea_gate source is not the reviewed build")
    spec = importlib.util.spec_from_file_location(
        "_deadline_fea_gate_reviewed", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reviewed deadline_fea_gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.RESULT_SCHEMA != RESULT_SCHEMA
        or module.SPECULATIVE_JOIN_SCHEMA != JOIN_SCHEMA
        or module.TIMK3_SOLVER_REVISION != PHYSICS_SOLVER_REVISION
        or module.LIBRARY_REVISION != LIBRARY_REVISION
        or module.SUBMISSION_CODE_REVISION != SUBMISSION_CODE_REVISION
    ):
        raise RuntimeError("deadline_fea_gate constants drifted")
    return module


def _candidate(task_id: int) -> dict:
    try:
        return CANDIDATES[int(task_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Full task is not in the exact sibling allowlist") from exc


def _identity_matches(value: dict, spec: dict) -> bool:
    return (
        value.get("candidate_identity_sha256") == spec["identity"]
        and value.get("candidate_digest") == spec["digest"]
    )


def _authority_false(value: dict) -> bool:
    return all(
        value.get(name) is False
        for name in (
            "automatic_promotion_allowed",
            "canonical_dataset_mutated",
            "scheduler_configuration_mutated",
            "task_cancellation_performed",
            "final_design_approved",
        )
    )


def _validate_adapter_receipt(
    receipt: dict,
    receipt_path: Path,
    plan: dict,
    plan_path: Path,
) -> tuple[int, dict]:
    task_id = int(receipt.get("task_id", -1))
    spec = _candidate(task_id)
    source_path = _validate_reference(
        receipt.get("source_submission_receipt"),
        "source submission receipt",
    )
    runtime = receipt.get("runtime_core_contract")
    resources = receipt.get("submission_resources")
    if (
        receipt.get("schema_version") != RESULT_SCHEMA.replace(
            "result", "submission"
        )
        or receipt.get("adapter_schema_version") != ADAPTER_RECEIPT_SCHEMA
        or receipt.get("fidelity") != "full"
        or receipt.get("plan_kind") != plan.get("plan_kind")
        or receipt.get("diagnostic_near_truth") is not True
        or receipt.get("speculative_full_diagnostic") is not True
        or receipt.get("standard_prerequisite_deferred") is not True
        or receipt.get("standard_result_prerequisite") is not None
        or receipt.get("task_name") != spec["task_name"]
        or not _identity_matches(receipt, spec)
        or receipt.get("solver_revision") != spec["runtime_revision"]
        or receipt.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or receipt.get("library_revision") != LIBRARY_REVISION
        or receipt.get("solver_variant") != "deadline-tim-k3"
        or receipt.get("plan", {}).get("sha256") != spec["plan_sha256"]
        or receipt.get("source_submission_receipt", {}).get("sha256")
        != spec["source_receipt_sha256"]
        or not isinstance(resources, dict)
        or resources.get("backend") != "standalone"
        or resources.get("scheduling_profile") != "standard"
        or int(resources.get("cpus", -1)) != spec["cores"]
        or not isinstance(runtime, dict)
        or runtime.get("version") != spec["core_contract_version"]
        or runtime.get("auth_sha256") != spec["core_auth"]
        or int(runtime.get("requested_num_cores", -1)) != spec["cores"]
        or int(runtime.get("required_slurm_cpus_per_task", -1))
        != spec["cores"]
        or runtime.get("runtime_contract_readback_pending") is not True
        or runtime.get("matrix_acf_readback_pending") is not True
        or not _authority_false(receipt)
    ):
        raise RuntimeError("canonical adapter receipt contract drifted")
    if (
        spec["cores"] == 16
        and (
            runtime.get("license_contract")
            != "mft-aedt-hpc-license-snapshot-v1"
            or runtime.get("license_snapshot_sha256")
            != spec["license_snapshot_sha256"]
        )
    ):
        raise RuntimeError("16-core license receipt contract drifted")
    _validate_reference(receipt.get("plan"), "plan", plan_path)
    if _sha256(source_path) != spec["source_receipt_sha256"]:
        raise RuntimeError("source submission receipt is not allowlisted")
    if (
        plan["candidate"]["candidate_identity_sha256"] != spec["identity"]
        or plan["candidate"]["candidate_digest"] != spec["digest"]
        or plan["solver_contract"]["solver_revision"]
        != PHYSICS_SOLVER_REVISION
        or plan["solver_contract"]["library_revision"] != LIBRARY_REVISION
    ):
        raise RuntimeError("plan is not the exact physics-base candidate")
    # receipt_path is passed explicitly so callers cannot validate one object
    # while referencing another file.
    if receipt_path != receipt_path.resolve(strict=True):
        raise RuntimeError("adapter receipt path is not canonical")
    return task_id, spec


def _validate_core_contract(contract: Any, task_id: int, spec: dict) -> None:
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "mft-solver-core-policy-v1"
        or contract.get("contract_version") != spec["core_contract_version"]
        or contract.get("backend") != "standalone"
        or contract.get("opt_in") is not True
        or int(contract.get("requested_num_cores", -1)) != spec["cores"]
        or int(contract.get("effective_num_cores", -1)) != spec["cores"]
        or int(contract.get("affinity_count_readback", -1)) < spec["cores"]
        or int(contract.get("slurm_cpus_per_task_readback", -1))
        != spec["cores"]
        or int(contract.get("scheduler_task_id_readback", -1)) != task_id
        or contract.get("solver_revision") != spec["runtime_revision"]
        or int(contract.get("solver_dirty", -1)) != 0
        or contract.get("auth_sha256") != spec["core_auth"]
    ):
        raise RuntimeError("runtime core contract drifted")
    if spec["cores"] == 16 and (
        contract.get("license_contract")
        != "mft-aedt-hpc-license-snapshot-v1"
        or contract.get("license_snapshot_sha256")
        != spec["license_snapshot_sha256"]
    ):
        raise RuntimeError("runtime 16-core license readback drifted")


def _validate_dispatches(value: Any, spec: dict) -> None:
    if not isinstance(value, list) or not value:
        raise RuntimeError("runtime dispatch evidence is missing")
    matrix = 0
    for record in value:
        if (
            not isinstance(record, dict)
            or record.get("schema") != "mft-solver-core-dispatch-v1"
            or record.get("backend") != "standalone"
            or int(record.get("cores_argument", -1)) != spec["cores"]
        ):
            raise RuntimeError("runtime dispatch contract drifted")
        if (
            record.get("stage") == "matrix"
            and record.get("dispatch") == "pyaedt_setup_analyze"
        ):
            matrix += 1
    if matrix != 1:
        raise RuntimeError("exactly one matrix solve dispatch is required")


def _validate_start_evidence(
    evidence: dict,
    evidence_path: Path,
    plan_path: Path,
    receipt_path: Path,
    task_id: int,
    spec: dict,
) -> None:
    if (
        evidence.get("schema_version") != START_EVIDENCE_SCHEMA
        or int(evidence.get("task_id", -1)) != task_id
        or evidence.get("task_name") != spec["task_name"]
        or evidence.get("task_status") not in {"running", "completed"}
        or not _identity_matches(evidence, spec)
        or evidence.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or evidence.get("runtime_solver_revision") != spec["runtime_revision"]
        or evidence.get("library_revision") != LIBRARY_REVISION
        or evidence.get("runtime_contract_valid") is not True
        or evidence.get("physics_change") != "none"
        or not _authority_false(evidence)
    ):
        raise RuntimeError("runtime start evidence contract drifted")
    _validate_reference(evidence.get("plan"), "start plan", plan_path)
    _validate_reference(
        evidence.get("canonical_adapter_receipt"),
        "start adapter receipt",
        receipt_path,
    )
    _validate_reference(
        evidence.get("source_submission_receipt"),
        "start source receipt",
    )
    _validate_core_contract(evidence.get("core_contract"), task_id, spec)
    _validate_dispatches(evidence.get("dispatch_records"), spec)
    scheduler = evidence.get("scheduler")
    if (
        not isinstance(scheduler, dict)
        or int(scheduler.get("cpus", -1)) != spec["cores"]
        or int(scheduler.get("memory_mb", -1)) != 98304
        or not scheduler.get("actual_node_name")
        or not scheduler.get("slurm_job_id")
    ):
        raise RuntimeError("runtime start Scheduler readback drifted")
    if evidence_path != evidence_path.resolve(strict=True):
        raise RuntimeError("start evidence path is not canonical")


def _validate_terminal_capture(
    capture: dict,
    capture_path: Path,
    plan_path: Path,
    receipt_path: Path,
    start_path: Path,
    task_id: int,
    spec: dict,
) -> dict:
    if (
        capture.get("schema_version") != TERMINAL_CAPTURE_SCHEMA
        or int(capture.get("task_id", -1)) != task_id
        or capture.get("task_name") != spec["task_name"]
        or capture.get("task_status") != "completed"
        or capture.get("exit_code") not in (0, "0", None)
        or str(capture.get("failure_message") or "")
        or not _identity_matches(capture, spec)
        or capture.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or capture.get("runtime_solver_revision") != spec["runtime_revision"]
        or capture.get("library_revision") != LIBRARY_REVISION
        or capture.get("core_contract_valid") is not True
        or int(capture.get("result_count", -1)) != 1
        or capture.get("result_runtime_identity_valid") is not True
        or capture.get("collection_errors") != []
        or capture.get("collection_valid") is not True
        or capture.get("actual_hard_spec_pass") is not False
        or capture.get("actual_hard_spec_evaluation_pending") is not True
        or not _authority_false(capture)
    ):
        raise RuntimeError("terminal runtime capture is not clean and complete")
    _validate_reference(capture.get("plan"), "terminal plan", plan_path)
    _validate_reference(
        capture.get("canonical_adapter_receipt"),
        "terminal adapter receipt",
        receipt_path,
    )
    _validate_reference(capture.get("start_evidence"), "terminal start", start_path)
    _validate_reference(
        capture.get("source_submission_receipt"),
        "terminal source receipt",
    )
    _validate_core_contract(capture.get("core_contract"), task_id, spec)
    _validate_dispatches(capture.get("dispatch_records"), spec)
    result = capture.get("result")
    if (
        not isinstance(result, dict)
        or result.get("git_hash") != spec["runtime_revision"]
        or int(result.get("git_dirty", -1)) != 0
        or result.get("pyaedt_library_git_hash") != LIBRARY_REVISION
        or int(result.get("pyaedt_library_git_dirty", -1)) != 0
        or result.get("solver_core_contract_version")
        != spec["core_contract_version"]
        or int(result.get("solver_num_cores_effective", -1))
        != spec["cores"]
        or str(result.get("solver_core_scheduler_task_id_readback"))
        != str(task_id)
    ):
        raise RuntimeError("native result runtime identity drifted")
    stdout = capture.get("stdout")
    stderr = capture.get("stderr")
    for label, stream in (("stdout", stdout), ("stderr", stderr)):
        if (
            not isinstance(stream, dict)
            or _sha256_bytes(
                str(stream.get("value", "")).encode("utf-8")
            ) != stream.get("sha256")
            or len(str(stream.get("value", "")).encode("utf-8"))
            != int(stream.get("size_bytes", -1))
        ):
            raise RuntimeError(f"terminal {label} content digest drifted")
    if capture_path != capture_path.resolve(strict=True):
        raise RuntimeError("terminal capture path is not canonical")
    return result


def _validate_start_bundle(args, require_terminal: bool):
    gate = _load_deadline_gate(Path(args.deadline_gate_code_root))
    plan_path, _ = _read_hashed_json(
        args.plan, args.plan_sha256, "plan"
    )
    plan = gate._validate_plan(plan_path, args.plan_sha256)
    receipt_path, receipt = _read_hashed_json(
        args.adapter_receipt,
        args.adapter_receipt_sha256,
        "canonical adapter receipt",
    )
    task_id, spec = _validate_adapter_receipt(
        receipt, receipt_path, plan, plan_path
    )
    start_path, start = _read_hashed_json(
        args.start_evidence,
        args.start_evidence_sha256,
        "runtime start evidence",
    )
    _validate_start_evidence(
        start, start_path, plan_path, receipt_path, task_id, spec
    )
    revision_path, _ = _read_hashed_json(
        args.revision_attestation,
        args.revision_attestation_sha256,
        "revision attestation",
    )
    revision = _validate_revision_attestation(
        revision_path,
        args.revision_attestation_sha256,
        Path(args.solver_git_root),
    )
    result = None
    capture_path = None
    capture = None
    if require_terminal:
        capture_path, capture = _read_hashed_json(
            args.terminal_capture,
            args.terminal_capture_sha256,
            "runtime terminal capture",
        )
        result = _validate_terminal_capture(
            capture,
            capture_path,
            plan_path,
            receipt_path,
            start_path,
            task_id,
            spec,
        )
    return {
        "gate": gate,
        "plan": plan,
        "plan_path": plan_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "start": start,
        "start_path": start_path,
        "revision": revision,
        "revision_path": revision_path,
        "capture": capture,
        "capture_path": capture_path,
        "result": result,
        "task_id": task_id,
        "spec": spec,
    }


def command_attest_revisions(args: argparse.Namespace) -> int:
    payload = build_revision_attestation(Path(args.repo_root))
    output = Path(args.output)
    _atomic_write_once(output, payload)
    print(json.dumps({
        "status": "reviewed_revision_chain_attested",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "revision_chain_valid": True,
        "physical_model_or_result_extraction_change": False,
        "mutation_performed": False,
    }, indent=2))
    return 0


def command_preflight_start(args: argparse.Namespace) -> int:
    bundle = _validate_start_bundle(args, require_terminal=False)
    print(json.dumps({
        "status": "runtime_start_evidence_authenticated",
        "task_id": bundle["task_id"],
        "runtime_solver_revision": bundle["spec"]["runtime_revision"],
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "effective_cores": bundle["spec"]["cores"],
        "revision_chain_valid": True,
        "runtime_contract_valid": True,
        "mutation_performed": False,
    }, indent=2))
    return 0


def _make_execution_attestation(bundle: dict) -> dict:
    spec = bundle["spec"]
    task_id = bundle["task_id"]
    capture = bundle["capture"]
    payload = {
        "schema_version": EXECUTION_ATTESTATION_SCHEMA,
        "created_at": _now(),
        "task_id": task_id,
        "task_name": spec["task_name"],
        "candidate_identity_sha256": spec["identity"],
        "candidate_digest": spec["digest"],
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "runtime_solver_revision": spec["runtime_revision"],
        "library_revision": LIBRARY_REVISION,
        "effective_cores": spec["cores"],
        "core_contract_version": spec["core_contract_version"],
        "revision_attestation": _file_reference(bundle["revision_path"]),
        "canonical_adapter_receipt": _file_reference(bundle["receipt_path"]),
        "runtime_start_evidence": _file_reference(bundle["start_path"]),
        "runtime_terminal_capture": _file_reference(bundle["capture_path"]),
        "native_result_sha256": _canonical_sha256(bundle["result"]),
        "revision_chain_valid": True,
        "physical_model_or_result_extraction_change": False,
        "core_contract_valid": capture["core_contract_valid"] is True,
        "result_runtime_identity_valid":
            capture["result_runtime_identity_valid"] is True,
        "collection_valid": capture["collection_valid"] is True,
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutated": False,
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
        "final_design_approved": False,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _validate_execution_attestation(
    path: Path,
    expected_sha256: str,
    expected_task_id: int | None = None,
    *,
    plan: dict | None = None,
    plan_path: Path | None = None,
) -> dict:
    _, evidence = _read_hashed_json(
        path, expected_sha256, "runtime execution attestation"
    )
    task_id = int(evidence.get("task_id", -1))
    spec = _candidate(task_id)
    if (
        evidence.get("schema_version") != EXECUTION_ATTESTATION_SCHEMA
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
        or (expected_task_id is not None and task_id != expected_task_id)
        or evidence.get("task_name") != spec["task_name"]
        or not _identity_matches(evidence, spec)
        or evidence.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or evidence.get("runtime_solver_revision") != spec["runtime_revision"]
        or evidence.get("library_revision") != LIBRARY_REVISION
        or int(evidence.get("effective_cores", -1)) != spec["cores"]
        or evidence.get("core_contract_version")
        != spec["core_contract_version"]
        or evidence.get("native_result_sha256") is None
        or any(
            evidence.get(name) is not True
            for name in (
                "revision_chain_valid",
                "core_contract_valid",
                "result_runtime_identity_valid",
                "collection_valid",
            )
        )
        or evidence.get("physical_model_or_result_extraction_change")
        is not False
        or not _authority_false(evidence)
    ):
        raise RuntimeError("runtime execution attestation contract drifted")
    revision_path = _validate_reference(
        evidence.get("revision_attestation"), "execution revision"
    )
    _validate_revision_attestation(
        revision_path,
        evidence["revision_attestation"]["sha256"],
    )
    receipt_path = _validate_reference(
        evidence.get("canonical_adapter_receipt"), "execution adapter receipt"
    )
    start_path = _validate_reference(
        evidence.get("runtime_start_evidence"), "execution start evidence"
    )
    capture_path = _validate_reference(
        evidence.get("runtime_terminal_capture"), "execution terminal capture"
    )
    capture = _read_json(capture_path)
    if (
        int(capture.get("task_id", -1)) != task_id
        or _canonical_sha256(capture.get("result"))
        != evidence["native_result_sha256"]
    ):
        raise RuntimeError("execution attestation native result link drifted")
    if plan is not None or plan_path is not None:
        if plan is None or plan_path is None:
            raise RuntimeError("execution plan and path must be supplied together")
        receipt = _read_json(receipt_path)
        validated_task_id, validated_spec = _validate_adapter_receipt(
            receipt, receipt_path, plan, plan_path
        )
        if validated_task_id != task_id or validated_spec != spec:
            raise RuntimeError("execution adapter task identity drifted")
        start = _read_json(start_path)
        _validate_start_evidence(
            start,
            start_path,
            plan_path,
            receipt_path,
            task_id,
            spec,
        )
        native = _validate_terminal_capture(
            capture,
            capture_path,
            plan_path,
            receipt_path,
            start_path,
            task_id,
            spec,
        )
        if _canonical_sha256(native) != evidence["native_result_sha256"]:
            raise RuntimeError("execution native result content drifted")
    return evidence


def command_canonicalize_full(args: argparse.Namespace) -> int:
    bundle = _validate_start_bundle(args, require_terminal=True)
    gate = bundle["gate"]
    plan = bundle["plan"]
    result = bundle["result"]
    spec = bundle["spec"]
    task_id = bundle["task_id"]

    keys, validate_record, bounding_box_lit, scheduler_client = (
        gate._load_submission_contract(Path(args.submission_code_root))
    )
    profile_ref = plan["solver_contract"]["fine_profile"]
    profile = gate._read_json(profile_ref["path"])
    gate._profile_contract(profile, "full")
    effective = scheduler_client.effective_verification_params(
        plan["candidate"]["decoded_params"], profile
    )
    identity_matches = scheduler_client.result_matches_params(
        result, effective, required_keys=frozenset(keys)
    )
    validity = (
        validate_record(
            result,
            profile=profile,
            expected_solver_revision=spec["runtime_revision"],
            expected_library_revision=LIBRARY_REVISION,
        )
        if identity_matches else None
    )
    solver_variant_valid = gate._solver_result_variant_contract(plan, result)
    contract_valid = bool(
        identity_matches
        and validity is not None
        and validity.full_valid
        and solver_variant_valid
    )
    if not contract_valid:
        raise RuntimeError(
            "native Full result failed canonical quality/identity contract"
        )
    actual_gate = gate._actual_hard_gate(
        result,
        "full",
        plan["target"],
        plan["source"]["temperature_targets"],
        bounding_box_lit,
    )
    if actual_gate.get("pass") is not True:
        raise RuntimeError(
            "native Full result failed actual hard gate: "
            + ",".join(actual_gate.get("reasons") or [])
        )

    execution = _make_execution_attestation(bundle)
    execution_output = Path(args.execution_attestation_output)
    _atomic_write_once(execution_output, execution)
    execution_ref = _file_reference(execution_output)

    evidence = {
        "schema_version": RESULT_SCHEMA,
        "canonical_adapter_schema_version":
            "mft-deadline-core-sibling-result-adapter-v1",
        "plan_kind": plan["plan_kind"],
        "diagnostic_near_truth": True,
        "standard_only_calibration": False,
        "candidate_surrogate_hard_feasible":
            plan["candidate_surrogate_hard_feasible"],
        "automatic_promotion_allowed": False,
        "created_at": _now(),
        "fidelity": "full",
        "plan": _file_reference(bundle["plan_path"]),
        "submission_receipt": _file_reference(bundle["receipt_path"]),
        "candidate_identity_sha256": spec["identity"],
        "candidate_digest": spec["digest"],
        "task_id": task_id,
        "task_name": spec["task_name"],
        "task_status": "completed",
        "result_state": "valid",
        "solver_revision": PHYSICS_SOLVER_REVISION,
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "runtime_solver_revision": spec["runtime_revision"],
        "library_revision": LIBRARY_REVISION,
        "runtime_execution_evidence": execution_ref,
        "result_contract_valid": True,
        "solver_variant_contract_valid": True,
        "candidate_identity_matches": True,
        "actual_hard_spec_pass": True,
        "actual_hard_spec_gate": actual_gate,
        "truth_validation_tier": "q0_actual_full_fea",
        "q1_surrogate_uncertainty_preserved": True,
        "physical_constraint_relaxation_applied": False,
        "resonance_contract_kind": plan["_resonance_contract_kind"],
        "thermal_material_contract_kind":
            plan.get("_tim_solver_contract_kind", "baseline"),
        "thermal_calibration_only": False,
        "fan_velocity_profile_identity":
            plan.get("_fan_velocity_profile_identity"),
        "fan_velocity_profile_mismatch_quarantine": False,
        "scheduler_diagnostics": {
            "source": "immutable_runtime_terminal_capture",
            "runtime_terminal_capture":
                _file_reference(bundle["capture_path"]),
            "read_only": True,
        },
        "next_stage": "awaiting_standard_join",
        "canonical_dataset_mutated": False,
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
        "final_design_approved": False,
        "speculative_full_diagnostic": True,
        "standard_prerequisite_deferred": True,
        "result": result,
    }
    evidence["payload_sha256"] = _payload_sha256(evidence)
    output = Path(args.output)
    _atomic_write_once(output, evidence)
    print(json.dumps({
        "status": "canonical_full_result_captured",
        "task_id": task_id,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "runtime_execution_evidence": execution_ref,
        "actual_hard_spec_pass": True,
        "automatic_promotion_allowed": False,
        "final_design_approved": False,
    }, indent=2))
    return 0


def _validate_canonical_full(
    path: Path,
    expected_sha256: str,
    plan: dict,
) -> tuple[dict, dict]:
    _, evidence = _read_hashed_json(path, expected_sha256, "canonical Full")
    task_id = int(evidence.get("task_id", -1))
    spec = _candidate(task_id)
    if (
        evidence.get("schema_version") != RESULT_SCHEMA
        or evidence.get("canonical_adapter_schema_version")
        != "mft-deadline-core-sibling-result-adapter-v1"
        or evidence.get("payload_sha256") != _payload_sha256(evidence)
        or evidence.get("fidelity") != "full"
        or evidence.get("plan_kind") != plan.get("plan_kind")
        or evidence.get("diagnostic_near_truth") is not True
        or evidence.get("speculative_full_diagnostic") is not True
        or evidence.get("standard_prerequisite_deferred") is not True
        or not _identity_matches(evidence, spec)
        or evidence.get("task_name") != spec["task_name"]
        or evidence.get("task_status") != "completed"
        or evidence.get("result_state") != "valid"
        or evidence.get("solver_revision") != PHYSICS_SOLVER_REVISION
        or evidence.get("physics_solver_revision")
        != PHYSICS_SOLVER_REVISION
        or evidence.get("runtime_solver_revision")
        != spec["runtime_revision"]
        or evidence.get("library_revision") != LIBRARY_REVISION
        or any(
            evidence.get(name) is not True
            for name in (
                "result_contract_valid",
                "solver_variant_contract_valid",
                "candidate_identity_matches",
                "actual_hard_spec_pass",
            )
        )
        or evidence.get("actual_hard_spec_gate", {}).get("pass") is not True
        or not _authority_false(evidence)
    ):
        raise RuntimeError("canonical Full evidence is not an actual PASS")
    canonical_plan_path = _validate_reference(
        evidence.get("plan"), "canonical Full plan"
    )
    if evidence["plan"]["sha256"] != spec["plan_sha256"]:
        raise RuntimeError("canonical Full plan is not exactly allowlisted")
    canonical_receipt_path = _validate_reference(
        evidence.get("submission_receipt"),
        "canonical Full adapter receipt",
    )
    result = evidence.get("result")
    if (
        not isinstance(result, dict)
        or result.get("git_hash") != spec["runtime_revision"]
        or _canonical_sha256(result)
        != _read_json(_validate_reference(
            evidence.get("runtime_execution_evidence"),
            "Full runtime execution",
        ))["native_result_sha256"]
    ):
        raise RuntimeError("canonical Full native result link drifted")
    execution_path = _validate_reference(
        evidence["runtime_execution_evidence"],
        "Full runtime execution",
    )
    execution = _validate_execution_attestation(
        execution_path,
        evidence["runtime_execution_evidence"]["sha256"],
        task_id,
        plan=plan,
        plan_path=canonical_plan_path,
    )
    if (
        execution["canonical_adapter_receipt"]
        != _file_reference(canonical_receipt_path)
    ):
        raise RuntimeError("canonical Full adapter/execution link drifted")
    if (
        plan["candidate"]["candidate_identity_sha256"] != spec["identity"]
        or plan["candidate"]["candidate_digest"] != spec["digest"]
    ):
        raise RuntimeError("canonical Full candidate differs from plan")
    return evidence, execution


def command_join(args: argparse.Namespace) -> int:
    gate = _load_deadline_gate(Path(args.deadline_gate_code_root))
    plan_path, _ = _read_hashed_json(args.plan, args.plan_sha256, "plan")
    plan = gate._validate_plan(plan_path, args.plan_sha256)
    standard_path = Path(args.standard_result).resolve(strict=True)
    standard = gate._validate_result_evidence(
        standard_path, args.standard_result_sha256, plan
    )
    full_path = Path(args.full_result).resolve(strict=True)
    full, execution = _validate_canonical_full(
        full_path, args.full_result_sha256, plan
    )
    full_task_id = int(full["task_id"])
    standard_task_id = int(standard.get("task_id", -1))
    spec = _candidate(full_task_id)
    if (
        standard_task_id not in spec["standard_task_ids"]
        or standard_task_id == full_task_id
        or standard.get("task_status") != "completed"
        or standard.get("result_state") != "valid"
        or standard.get("actual_hard_spec_gate", {}).get("pass") is not True
        or standard.get("scheduler_configuration_mutated") is not False
        or standard.get("task_cancellation_performed") is not False
    ):
        raise RuntimeError("Standard result is not the exact allowed PASS sibling")

    attestation = {
        "schema_version": JOIN_SCHEMA,
        "canonical_adapter_schema_version":
            "mft-deadline-core-sibling-speculative-join-v1",
        "created_at": _now(),
        "plan": _file_reference(plan_path),
        "standard_result": _file_reference(standard_path),
        "speculative_full_result": _file_reference(full_path),
        "runtime_execution_evidence":
            full["runtime_execution_evidence"],
        "candidate_identity_sha256": spec["identity"],
        "candidate_digest": spec["digest"],
        "solver_revision": PHYSICS_SOLVER_REVISION,
        "physics_solver_revision": PHYSICS_SOLVER_REVISION,
        "runtime_solver_revision": spec["runtime_revision"],
        "solver_variant": plan["solver_contract"]["solver_variant"],
        "thermal_pad_conductivity_W_mK":
            plan["solver_contract"]["thermal_pad_conductivity_W_mK"],
        "thermal_pad_material_policy":
            plan["solver_contract"]["thermal_pad_material_policy"],
        "thermal_pad_native_readback_contract_version":
            plan["solver_contract"][
                "thermal_pad_native_readback_contract_version"
            ],
        "thermal_pad_native_readback_required": True,
        "standard_task_id": standard_task_id,
        "speculative_full_task_id": full_task_id,
        "allowed_standard_task_ids": sorted(spec["standard_task_ids"]),
        "exact_standard_task_pair_allowlist_pass": True,
        "standard_actual_hard_spec_pass": True,
        "full_actual_hard_spec_pass": True,
        "runtime_execution_attested":
            execution["collection_valid"] is True,
        "hash_authenticated_join": True,
        "sendable_design_evidence_ready": True,
        "truth_validation_tier": "q0_actual_standard_and_full_fea",
        "q1_surrogate_uncertainty_preserved": True,
        "physical_constraint_relaxation_applied": False,
        "automatic_promotion_allowed": False,
        "final_design_approved": False,
        "canonical_dataset_mutated": False,
        "scheduler_configuration_mutated": False,
        "task_cancellation_performed": False,
    }
    attestation["payload_sha256"] = _payload_sha256(attestation)
    output = Path(args.output)
    _atomic_write_once(output, attestation)
    print(json.dumps({
        "status": "core_sibling_speculative_full_join_attested",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "standard_task_id": standard_task_id,
        "speculative_full_task_id": full_task_id,
        "sendable_design_evidence_ready": True,
        "final_design_approved": False,
    }, indent=2))
    return 0


def _add_static_bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--adapter-receipt", required=True)
    parser.add_argument("--adapter-receipt-sha256", required=True)
    parser.add_argument("--start-evidence", required=True)
    parser.add_argument("--start-evidence-sha256", required=True)
    parser.add_argument("--revision-attestation", required=True)
    parser.add_argument("--revision-attestation-sha256", required=True)
    parser.add_argument("--solver-git-root", required=True)
    parser.add_argument("--deadline-gate-code-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    attest = sub.add_parser("attest-revisions")
    attest.add_argument("--repo-root", required=True)
    attest.add_argument("--output", required=True)
    attest.set_defaults(function=command_attest_revisions)

    preflight = sub.add_parser("preflight-start")
    _add_static_bundle_args(preflight)
    preflight.set_defaults(function=command_preflight_start)

    canonical = sub.add_parser("canonicalize-full")
    _add_static_bundle_args(canonical)
    canonical.add_argument("--terminal-capture", required=True)
    canonical.add_argument("--terminal-capture-sha256", required=True)
    canonical.add_argument("--submission-code-root", required=True)
    canonical.add_argument("--execution-attestation-output", required=True)
    canonical.add_argument("--output", required=True)
    canonical.set_defaults(function=command_canonicalize_full)

    join = sub.add_parser("join")
    join.add_argument("--plan", required=True)
    join.add_argument("--plan-sha256", required=True)
    join.add_argument("--standard-result", required=True)
    join.add_argument("--standard-result-sha256", required=True)
    join.add_argument("--full-result", required=True)
    join.add_argument("--full-result-sha256", required=True)
    join.add_argument("--deadline-gate-code-root", required=True)
    join.add_argument("--output", required=True)
    join.set_defaults(function=command_join)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except Exception as exc:
        print(json.dumps({
            "status": "blocked_fail_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "mutation_performed": False,
        }, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
