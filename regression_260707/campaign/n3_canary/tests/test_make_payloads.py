from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
KIT = HERE.parent
FIXTURE = HERE / "fixture_state.json"


def test_make_payloads_builds_offline_n3_templates(tmp_path: Path) -> None:
    output = tmp_path / "payloads"
    result = subprocess.run(
        [
            sys.executable,
            str(KIT / "make_payloads.py"),
            "--allocation-id",
            "8678",
            "--account",
            "fixture-account",
            "--state",
            str(FIXTURE),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["source_standalone_serials"] == [345, 344, 341]

    host = json.loads((output / "host_payload.json").read_text(encoding="utf-8"))
    client_paths = sorted(output.glob("client_payload_*.json"))
    clients = [json.loads(path.read_text(encoding="utf-8")) for path in client_paths]

    assert len(clients) == 3
    assert host["name"] == "mft-aedt-n3canary-260714-host"
    assert host["project"] == "_aedt_pool_hosts"
    assert host["cpus"] == 1
    assert host["memory_mb"] == 4096
    assert host["timeout_seconds"] == 0
    assert host["requested_allocation_id"] == 8678
    assert host["payload_json"]["aedt_canary_expected_projects"] == 3
    assert "--expected-projects 3" in host["command"]

    payloads = [host, *clients]
    assert all(not payload["name"].startswith("mft-camp") for payload in payloads)
    assert [payload["name"] for payload in clients] == [
        "mft-aedt-n3canary-260714-client-1",
        "mft-aedt-n3canary-260714-client-2",
        "mft-aedt-n3canary-260714-client-3",
    ]
    assert len({payload["dedupe_key"] for payload in payloads}) == 4

    expected_parameter_digests = [
        "ba8027b6c4ef9cd7",
        "23bc302afb09415f",
        "28cdbc52f41eb537",
    ]

    for payload, parameter_digest in zip(clients, expected_parameter_digests):
        serialized = json.dumps(payload, sort_keys=True)
        assert "{HOST_TASK_ID}" in serialized
        assert "{SCHEDULER_URL}" in serialized
        assert "{HOST_CLONE_ROOT}" in serialized
        assert payload["same_node_as_task_id"] == "{HOST_TASK_ID}"
        assert payload["entrypoint"] == "aedt_node_canary_client"
        assert payload["payload_json"]["aedt_canary_expected_projects"] == 3
        assert f":{parameter_digest}:scope-" in payload["dedupe_key"]
