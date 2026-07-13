"""Fail-closed Icepak-only replay for the preserved simulation470 artifact.

The default mode is read-only.  ``--execute`` creates a unique clone containing
only the AEDT project file, clears only the clone's internal solution references,
edits only the clone's Tx mesh operation bytes (and, for a split policy, the
MeshSetup ``NextUniqueID`` value), and solves only the cloned Icepak design.

The preserved simulation470 result is an invalid/old-Rx source artifact used
only to reconstruct the geometry.  It is never an accuracy baseline.  Hybrid
results must be compared with a latest production-valid, same-geometry all-L4
result before any production use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "simulation" / "simulation470" / "simulation470.aedt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "simulation" / "thermal_replays"
DEFAULT_LIBRARY_ROOT = Path(
    os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", r"Y:\git\pyaedt_library_mft_clean")
)
THERMAL_DESIGN = "icepak_thermal"
THERMAL_SETUP = "ThermalSetup"
EXPECTED_TX_OBJECTS = tuple(f"Tx_main_{index}_0" for index in range(8))
EXPECTED_TX_OBJECT_IDS = (1445, 1497, 1549, 1601, 1653, 1705, 1757, 1809)
EXPECTED_TX_ZONES = tuple(name.lower() + "_solid" for name in EXPECTED_TX_OBJECTS)
EXPECTED_RX_MAIN_OBJECTS = ("Rx_main_block_xn", "Rx_main_block_yp")
EXPECTED_RX_SIDE_OBJECTS = (
    "Rx_side_block_xn", "Rx_side_block_xp", "Rx_side_block_yp",
)
EXPECTED_CORE_OBJECTS = ("core_2", "core_3")
EXPECTED_PROBE_OBJECTS = (
    "Tprobe_Tx_leeward", "Tprobe_Tx_side",
    "Tprobe_Rx_main_leeward", "Tprobe_Rx_main_side",
    "Tprobe_Rx_side_leeward", "Tprobe_Rx_side_side",
    "Tprobe_core_center_leg", "Tprobe_core_side_leg",
    "Tprobe_core_top_yoke",
)
EXPECTED_THERMAL_TARGET_OBJECTS = (
    EXPECTED_TX_OBJECTS + EXPECTED_RX_MAIN_OBJECTS + EXPECTED_RX_SIDE_OBJECTS
    + EXPECTED_CORE_OBJECTS + EXPECTED_PROBE_OBJECTS
)
SOURCE_MISSING_ZONE = "tx_main_3_0_solid"
TX_MESH_LEVEL = 4
TX_COARSE_MESH_LEVEL = 3
MESH_POLICY_UNIFORM = "uniform"
MESH_POLICY_HYBRID = "hybrid"
DEFAULT_HYBRID_EDGE_FINE_TURNS = 3
EXPECTED_TX_OPERATION_ID = 2
EXPECTED_SOURCE_MESH_OPERATION_IDS = (1, 2, 3)
EXPECTED_SOURCE_NEXT_UNIQUE_ID = 4
HYBRID_OPERATION_IDS = (2, 4, 5)
HYBRID_NEXT_UNIQUE_ID = 6
HYBRID_OPERATION_NAMES = {
    "left_fine": "tx_mesh_level_edge_left_L4",
    "interior_coarse": "tx_mesh_level_interior_L3",
    "right_fine": "tx_mesh_level_edge_right_L4",
    "all_fine": "tx_mesh_level_all_L4",
}
ACCURACY_BASELINE_WARNING = (
    "simulation470 source result is invalid and uses an older Rx mesh; do not "
    "use its result as an accuracy baseline. Compare against the latest "
    "production-valid same-geometry all-L4 result."
)
EXPECTED_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
EXPECTED_SOURCE_SHA256 = "06D02D8864E3F3C7A1C347247EAC5B5AE87490C8B8A75A427D1142D967707F87"
EXPECTED_SOURCE_CASE_RELATIVE = Path(
    "simulation470.aedtresults/icepak_thermal.results/"
    "DV241_S227_V0.restart/current.nc_cas"
)
EXPECTED_SOURCE_CASE_SHA256 = "C4A0E40B83BD3152508E1D8B83645F1464E6864A02193FB34BBF91ECA198400B"
EXPECTED_SOURCE_MONITOR_RELATIVE = Path(
    "simulation470.aedtresults/icepak_thermal.results/DV241_S227_MON0_V0.sd"
)
EXPECTED_SOURCE_MONITOR_SHA256 = "1661F6EE620E85424D1E46E7051FBF3775B6E0B1A896538CBAA8DF34516CC1A8"
_TX_OBJECT_RE = re.compile(r"^Tx_main_(\d+)_0$")
_TX_ZONE_RE = re.compile(r"tx_main_(\d+)_0_solid", re.IGNORECASE)
_CASE_NAME_RE = re.compile(r"^DV(?P<dv>\d+)_S(?P<s>\d+)_V(?P<v>\d+)\.restart$")
_MONITOR_NAME_RE = re.compile(
    r"^DV(?P<dv>\d+)_S(?P<s>\d+)_MON(?P<mon>\d+)_V(?P<v>\d+)\.sd$"
)


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _file_identity(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_identity_with_hash(path: Path) -> dict | None:
    identity = _file_identity(path)
    if identity is None:
        return None
    return {**identity, "sha256": _sha256(path)}


def _source_artifacts(source: Path) -> tuple[Path, Path]:
    return (
        source.parent / EXPECTED_SOURCE_CASE_RELATIVE,
        source.parent / EXPECTED_SOURCE_MONITOR_RELATIVE,
    )


def _results_manifest(project_file: Path) -> list[dict]:
    """Return a stable file-only manifest for the complete AEDT results tree."""
    root = project_file.with_suffix(".aedtresults")
    if not root.is_dir():
        return []
    manifest = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix().lower(),
    ):
        stat = path.stat()
        manifest.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return manifest


def _single_paired_result_artifacts(project_file: Path) -> tuple[Path, Path, dict]:
    """Require exactly one new thermal case and monitor with the same DV/S/V."""
    results = project_file.with_suffix(".aedtresults")
    cases = list(results.glob(f"{THERMAL_DESIGN}.results/**/current.nc_cas")) \
        if results.is_dir() else []
    monitors = [
        path
        for path in results.glob(f"{THERMAL_DESIGN}.results/*_S*_MON*_V*.sd")
        if "_SOL" not in path.name
    ] if results.is_dir() else []
    if len(cases) != 1 or len(monitors) != 1:
        raise RuntimeError(
            "fresh clone must contain exactly one case and one monitor, got "
            f"cases={len(cases)} monitors={len(monitors)}"
        )
    case_path = cases[0]
    monitor_path = monitors[0]
    case_match = _CASE_NAME_RE.fullmatch(case_path.parent.name)
    monitor_match = _MONITOR_NAME_RE.fullmatch(monitor_path.name)
    if case_match is None or monitor_match is None:
        raise RuntimeError(
            f"unrecognized fresh case/monitor naming: {case_path}, {monitor_path}"
        )
    case_identity = tuple(case_match.group(key) for key in ("dv", "s", "v"))
    monitor_identity = tuple(monitor_match.group(key) for key in ("dv", "s", "v"))
    if case_identity != monitor_identity:
        raise RuntimeError(
            "fresh case/monitor DV/S/V pairing mismatch: "
            f"case={case_identity} monitor={monitor_identity}"
        )
    return case_path, monitor_path, {
        "dv": case_identity[0], "s": case_identity[1], "v": case_identity[2],
        "monitor": monitor_match.group("mon"),
    }


def _duration_seconds(value: str) -> int:
    match = re.fullmatch(r"(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)", str(value))
    if match is None:
        raise RuntimeError(f"invalid profile elapsed time: {value!r}")
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )


def _profile_artifact(project_file: Path, identity: dict) -> Path:
    results = project_file.with_suffix(".aedtresults") / f"{THERMAL_DESIGN}.results"
    expected = results / (
        f"DV{identity['dv']}_S{identity['s']}_V{identity['v']}.profile"
    )
    profiles = list(results.glob("DV*_S*_V*.profile")) if results.is_dir() else []
    if profiles != [expected]:
        raise RuntimeError(
            f"fresh clone profile identity mismatch: profiles={profiles} expected={expected}"
        )
    return expected


def _parse_icepak_profile(profile_path: Path) -> dict:
    text = profile_path.read_text(encoding="utf-8", errors="replace")
    solution_elapsed_values = re.findall(
        r"Name='Solution Process'.*?\$begin 'TotalInfo'.*?"
        r"I\(1, 'Elapsed Time', '([^']+)'\)",
        text,
        flags=re.DOTALL,
    )
    meshing_elapsed_values = re.findall(
        r"Name='Meshing Process'.*?\$begin 'TotalInfo'.*?"
        r"I\(1, 'Elapsed Time', '([^']+)'\)",
        text,
        flags=re.DOTALL,
    )
    if not solution_elapsed_values or len(meshing_elapsed_values) != 1:
        raise RuntimeError(f"profile elapsed-time records are incomplete: {profile_path}")
    items = {}
    for name in ("Global", "Populate Solver Input", "Solver Initialization", "Solve"):
        matches = re.findall(
            r"ProfileItem\('" + re.escape(name)
            + r"',\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),",
            text,
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"profile must contain exactly one {name!r} item, got {len(matches)}"
            )
        cpu_seconds, _cpu_fraction, real_seconds, _real_fraction, memory = (
            int(value) for value in matches[0]
        )
        items[name] = {
            "cpu_seconds": cpu_seconds,
            "real_seconds": real_seconds,
            "memory_raw": memory,
        }
    counts = {}
    for label in ("Nodes", "Faces", "Cells"):
        values = re.findall(
            r"\\?'Total " + label + r"\\?',\s*(\d+)", text
        )
        if len(values) != 1:
            raise RuntimeError(
                f"profile must contain exactly one Total {label}, got {len(values)}"
            )
        counts[label.lower()] = int(values[0])
    statuses = re.findall(
        r"\\?'Status\\?',\s*\\?'([^'\\]+)\\?'", text
    )
    if not statuses or any(status != "Normal Completion" for status in statuses):
        raise RuntimeError(f"profile status is not Normal Completion: {statuses}")
    solution_elapsed_seconds = [
        _duration_seconds(value) for value in solution_elapsed_values
    ]
    return {
        "path": str(profile_path),
        "sha256": _sha256(profile_path),
        "total_elapsed_seconds": sum(solution_elapsed_seconds),
        "solution_process_elapsed_seconds": solution_elapsed_seconds,
        "meshing_elapsed_seconds": _duration_seconds(meshing_elapsed_values[0]),
        "nodes": counts["nodes"],
        "faces": counts["faces"],
        "cells": counts["cells"],
        "mesh": items["Global"],
        "populate_solver_input": items["Populate Solver Input"],
        "solver_initialization": items["Solver Initialization"],
        "solve": items["Solve"],
        "status": "Normal Completion",
        "normal_completion_count": len(statuses),
    }


def _source_snapshot(source: Path) -> dict:
    case_path, monitor_path = _source_artifacts(source)
    return {
        "project": _file_identity_with_hash(source),
        "lock": _file_identity(Path(str(source) + ".lock")),
        "case": _file_identity_with_hash(case_path),
        "monitor": _file_identity_with_hash(monitor_path),
        "results_manifest": _results_manifest(source),
    }


def _icepak_section(project_file: Path) -> str:
    text = project_file.read_text(encoding="utf-8", errors="replace")
    start_token = "$begin 'IcepakModel'"
    end_token = "$end 'IcepakModel'"
    start = text.find(start_token)
    end = text.find(end_token, start + len(start_token))
    if start < 0 or end < 0:
        raise RuntimeError("source project has no complete IcepakModel section")
    section = text[start : end + len(end_token)]
    if f"Name='{THERMAL_DESIGN}'" not in section:
        raise RuntimeError(f"source Icepak design is not {THERMAL_DESIGN!r}")
    return section


def _named_section_bytes(data: bytes, name: bytes, start: int = 0) -> tuple[int, int]:
    begin_marker = b"$begin '" + name + b"'"
    end_marker = b"$end '" + name + b"'"
    begin = data.find(begin_marker, start)
    if begin < 0 or data.find(begin_marker, begin + len(begin_marker)) >= 0:
        raise RuntimeError(
            f"project must contain exactly one {name.decode('ascii', 'replace')} section"
        )
    end_start = data.find(end_marker, begin + len(begin_marker))
    if end_start < 0:
        raise RuntimeError(
            f"project has no complete {name.decode('ascii', 'replace')} section"
        )
    return begin, end_start + len(end_marker)


def _mesh_setup_byte_contract(data: bytes) -> dict:
    icepak_start, icepak_end = _named_section_bytes(data, b"IcepakModel")
    relative_start, relative_end = _named_section_bytes(
        data[icepak_start:icepak_end], b"MeshSetup"
    )
    setup_start = icepak_start + relative_start
    setup_end = icepak_start + relative_end
    setup = data[setup_start:setup_end]
    matches = list(re.finditer(rb"(?m)^\s*NextUniqueID=(\d+)\s*$", setup))
    if len(matches) != 1:
        raise RuntimeError(
            f"MeshSetup must have exactly one NextUniqueID, got {len(matches)}"
        )
    value_start = setup_start + matches[0].start(1)
    value_end = setup_start + matches[0].end(1)
    return {
        "setup_span": (setup_start, setup_end),
        "next_unique_id": int(matches[0].group(1)),
        "next_unique_id_span": (value_start, value_end),
    }


def _tx_mesh_block_spans(data: bytes) -> list[dict]:
    setup_contract = _mesh_setup_byte_contract(data)
    setup_start, setup_end = setup_contract["setup_span"]
    setup = data[setup_start:setup_end]
    begins = list(re.finditer(rb"\$begin '(tx_mesh_level[^']*)'", setup))
    spans = []
    for match in begins:
        name = match.group(1)
        absolute_start = setup_start + match.start()
        end_marker = b"$end '" + name + b"'"
        absolute_end_start = data.find(
            end_marker, setup_start + match.end(), setup_end
        )
        if absolute_end_start < 0:
            raise RuntimeError(
                f"incomplete Tx mesh operation {name.decode('ascii', 'replace')!r}"
            )
        absolute_end = absolute_end_start + len(end_marker)
        spans.append({
            "name": name.decode("ascii"),
            "span": (absolute_start, absolute_end),
            "block": data[absolute_start:absolute_end],
        })
    return spans


def _mesh_setup_operation_contract(project_file: Path) -> dict:
    data = project_file.read_bytes()
    setup_contract = _mesh_setup_byte_contract(data)
    setup_start, setup_end = setup_contract["setup_span"]
    setup = data[setup_start:setup_end].decode("utf-8", errors="replace")
    operations = []
    for match in re.finditer(r"\$begin '([^']+)'", setup):
        name = match.group(1)
        end_marker = f"$end '{name}'"
        end = setup.find(end_marker, match.end())
        if end < 0:
            raise RuntimeError(f"incomplete MeshSetup block {name!r}")
        body = setup[match.end():end]
        header = body.split("$begin", 1)[0]
        if not re.search(r"(?m)^\s*DType='OpT'\s*$", header):
            continue
        ids = re.findall(r"(?m)^\s*ID=(\d+)\s*$", header)
        if len(ids) != 1:
            raise RuntimeError(
                f"MeshSetup operation {name!r} must have exactly one ID"
            )
        operations.append({"name": name, "id": int(ids[0])})
    ids = [operation["id"] for operation in operations]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"MeshSetup operation ID overlap: {ids}")
    return {
        "next_unique_id": setup_contract["next_unique_id"],
        "operations": operations,
        "operation_ids": ids,
    }


def _tx_mesh_contracts(project_file: Path) -> list[dict]:
    section = _icepak_section(project_file)
    contracts = []
    for match in re.finditer(
        r"\$begin '(?P<name>tx_mesh_level[^']*)'(?P<body>.*?)"
        r"\$end '(?P=name)'",
        section,
        flags=re.DOTALL,
    ):
        operation_name = match.group("name")
        body = match.group("body")
        object_rows = re.findall(r"(?m)^\s*Objects\(([^)]*)\)\s*$", body)
        if len(object_rows) != 1:
            raise RuntimeError(
                f"Tx mesh operation {operation_name!r} must have exactly one "
                f"Objects row, got {len(object_rows)}"
            )
        try:
            object_ids = tuple(
                int(value.strip())
                for value in object_rows[0].split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Tx mesh operation {operation_name!r} has non-numeric object IDs"
            ) from exc
        operation_ids = re.findall(r"(?m)^\s*ID=(\d+)\s*$", body)
        if len(operation_ids) != 1:
            raise RuntimeError(
                f"Tx mesh operation {operation_name!r} must have exactly one ID"
            )
        levels = {
            int(value)
            for value in re.findall(
                r"(?m)^\s*(?:Level|MaxLevel|MinLevel)='(\d+)'$", body
            )
        }
        separate = re.findall(
            r"'Mesh Object\(s\) Separately Enabled'=(true|false)", body
        )
        contracts.append({
            "mesh_operation": operation_name,
            "operation_id": int(operation_ids[0]),
            "mesh_object_ids": list(object_ids),
            "mesh_levels": sorted(levels),
            "separate": separate[-1] if separate else None,
        })
    return contracts


def _project_tx_objects(project_file: Path) -> tuple[str, ...]:
    section = _icepak_section(project_file)
    return tuple(sorted(
        set(re.findall(r"Name='(Tx_main_\d+_0)'", section)),
        key=lambda name: int(_TX_OBJECT_RE.fullmatch(name).group(1)),
    ))


def _heat_source_contract(project_file: Path) -> dict:
    section = _icepak_section(project_file)
    entries = []
    seen_blocks = set()
    for power_match in re.finditer(
        r"(?m)^\s*'Total Power'='(?P<power>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?)W'\s*$",
        section,
    ):
        block_start = section.rfind("$begin '", 0, power_match.start())
        if block_start < 0:
            raise RuntimeError("heat-source Total Power has no containing block")
        name_start = block_start + len("$begin '")
        name_end = section.find("'", name_start)
        if name_end < 0:
            raise RuntimeError("heat-source boundary name is malformed")
        name = section[name_start:name_end]
        end_marker = f"$end '{name}'"
        block_end = section.find(end_marker, power_match.end())
        if block_end < 0:
            raise RuntimeError(f"heat-source boundary {name!r} is incomplete")
        if block_start in seen_blocks:
            raise RuntimeError(f"heat-source boundary {name!r} has duplicate power rows")
        seen_blocks.add(block_start)
        body = section[name_end + 1:block_end]
        if not re.search(r"(?m)^\s*BoundType='Block'\s*$", body) \
                or not re.search(
                    r"(?m)^\s*'Use Total Power'=true\s*$", body
                ):
            continue
        object_rows = re.findall(r"(?m)^\s*Objects\(([^)]*)\)\s*$", body)
        if len(object_rows) != 1:
            raise RuntimeError(
                f"heat-source boundary {name!r} is malformed"
            )
        try:
            object_ids = [
                int(value.strip()) for value in object_rows[0].split(",")
                if value.strip()
            ]
            watts = float(power_match.group("power"))
        except ValueError as exc:
            raise RuntimeError(
                f"heat-source boundary {name!r} is non-numeric"
            ) from exc
        if not object_ids or not math.isfinite(watts) or watts < 0:
            raise RuntimeError(
                f"heat-source boundary {name!r} is invalid"
            )
        entries.append({
            "boundary": name,
            "object_ids": object_ids,
            "total_power_w": watts,
        })
    if not entries:
        raise RuntimeError("Icepak model has no finite total-power heat sources")
    assigned_object_ids = [
        object_id for entry in entries for object_id in entry["object_ids"]
    ]
    if len(assigned_object_ids) != len(set(assigned_object_ids)):
        raise RuntimeError(
            f"heat-source object assignment overlap: {assigned_object_ids}"
        )
    canonical = json.dumps(
        entries, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return {
        "entries": entries,
        "entry_count": len(entries),
        "assigned_object_count": len(assigned_object_ids),
        "total_assigned_power_w": sum(
            entry["total_power_w"] for entry in entries
        ),
        "contract_sha256": hashlib.sha256(canonical).hexdigest().upper(),
        "finite_nonnegative": True,
        "assignment_no_overlap": True,
    }


def _thermal_boundary_temperatures(project_file: Path) -> dict:
    section = _icepak_section(project_file)
    ambient = re.findall(
        r"(?m)^\s*AmbientTemperature='([+-]?(?:\d+(?:\.\d*)?|\.\d+))cel'\s*$",
        section,
    )
    boundary_name = "cold_plates_fixed_T"
    begin = section.find(f"$begin '{boundary_name}'")
    end = section.find(f"$end '{boundary_name}'", begin + 1)
    if len(ambient) != 1 or begin < 0 or end < 0:
        raise RuntimeError("sealed thermal air/cold-plate temperature contract is missing")
    boundary = section[begin:end]
    plate = re.findall(
        r"(?m)^\s*Temperature='([+-]?(?:\d+(?:\.\d*)?|\.\d+))cel'\s*$",
        boundary,
    )
    if len(plate) != 1:
        raise RuntimeError(
            f"cold-plate fixed-temperature contract is malformed: {plate}"
        )
    values = {
        "air_temperature_c": float(ambient[0]),
        "cold_plate_temperature_c": float(plate[0]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"thermal boundary temperatures are non-finite: {values}")
    return values


def _source_icepak_contract(project_file: Path) -> dict:
    names = _project_tx_objects(project_file)
    mesh_contracts = _tx_mesh_contracts(project_file)
    if len(mesh_contracts) != 1:
        raise RuntimeError(
            "source must have exactly one tx_mesh_level operation, got "
            f"{len(mesh_contracts)}"
        )
    mesh = mesh_contracts[0]
    return {
        "design": THERMAL_DESIGN,
        "tx_objects": list(names),
        **mesh,
    }


def _tx_zones_from_case(case_path: Path) -> tuple[str, ...]:
    with case_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("(cfd-post-mesh-info"):
                continue
            names = {
                match.group(0).lower() for match in _TX_ZONE_RE.finditer(line)
            }
            return tuple(sorted(names, key=lambda name: int(_TX_ZONE_RE.fullmatch(name).group(1))))
    raise RuntimeError(f"no cfd-post-mesh-info record in {case_path}")


def _git_identity(root: Path) -> dict:
    if not (root / ".git").exists():
        raise RuntimeError(f"library root is not a git worktree: {root}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError(f"invalid library git revision: {head!r}")
    return {"root": str(root.resolve()), "revision": head, "dirty": dirty}


def _assert_source_lock_absent(source: Path) -> None:
    lock = Path(str(source) + ".lock")
    if lock.exists():
        raise RuntimeError(f"protected source lock exists; replay is prohibited: {lock}")


def _assert_source_unchanged(source: Path, expected: dict, stage: str) -> None:
    _assert_source_lock_absent(source)
    actual = _source_snapshot(source)
    if actual != expected:
        raise RuntimeError(f"protected source identity changed {stage}")


def _validate_source(source: Path) -> dict:
    source = source.resolve()
    if source != DEFAULT_SOURCE.resolve():
        raise RuntimeError(
            f"replay source is sealed to {DEFAULT_SOURCE.resolve()}, got {source}"
        )
    if not source.is_file() or source.suffix.lower() != ".aedt":
        raise RuntimeError(f"source AEDT project is missing: {source}")
    _assert_source_lock_absent(source)

    source_snapshot = _source_snapshot(source)
    if source_snapshot["project"] is None or (
        source_snapshot["project"].get("sha256") != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "sealed source project hash mismatch: "
            f"{source_snapshot['project'] and source_snapshot['project'].get('sha256')}"
        )
    if source_snapshot["case"] is None or (
        source_snapshot["case"].get("sha256") != EXPECTED_SOURCE_CASE_SHA256
    ):
        raise RuntimeError(
            "sealed source case path/hash mismatch: "
            f"path={source.parent / EXPECTED_SOURCE_CASE_RELATIVE} "
            f"hash={source_snapshot['case'] and source_snapshot['case'].get('sha256')}"
        )
    if source_snapshot["monitor"] is None or (
        source_snapshot["monitor"].get("sha256") != EXPECTED_SOURCE_MONITOR_SHA256
    ):
        raise RuntimeError(
            "sealed source monitor path/hash mismatch: "
            f"path={source.parent / EXPECTED_SOURCE_MONITOR_RELATIVE} "
            f"hash={source_snapshot['monitor'] and source_snapshot['monitor'].get('sha256')}"
        )

    contract = _source_icepak_contract(source)
    mesh_setup_contract = _mesh_setup_operation_contract(source)
    heat_source_contract = _heat_source_contract(source)
    boundary_temperatures = _thermal_boundary_temperatures(source)
    if tuple(contract["tx_objects"]) != EXPECTED_TX_OBJECTS:
        raise RuntimeError(
            f"source Tx object set mismatch: {contract['tx_objects']}"
        )
    if contract["separate"] != "false":
        raise RuntimeError("source Tx mesh operation is not shared-region meshing")
    if tuple(contract["mesh_object_ids"]) != EXPECTED_TX_OBJECT_IDS:
        raise RuntimeError(
            f"source Tx mesh object ID set mismatch: {contract['mesh_object_ids']}"
        )
    if contract["mesh_levels"] != [2]:
        raise RuntimeError(
            f"source Tx mesh level is not the sealed level-2 baseline: {contract['mesh_levels']}"
        )
    if contract["operation_id"] != EXPECTED_TX_OPERATION_ID:
        raise RuntimeError(
            "source Tx mesh operation ID mismatch: "
            f"{contract['operation_id']} != {EXPECTED_TX_OPERATION_ID}"
        )
    if tuple(mesh_setup_contract["operation_ids"]) != \
            EXPECTED_SOURCE_MESH_OPERATION_IDS:
        raise RuntimeError(
            "source MeshSetup operation ID/order mismatch: "
            f"{mesh_setup_contract['operation_ids']}"
        )
    if mesh_setup_contract["next_unique_id"] != EXPECTED_SOURCE_NEXT_UNIQUE_ID:
        raise RuntimeError(
            "source MeshSetup NextUniqueID mismatch: "
            f"{mesh_setup_contract['next_unique_id']}"
        )

    case_path, monitor_path = _source_artifacts(source)
    zones = _tx_zones_from_case(case_path)
    expected_baseline = tuple(
        name for name in EXPECTED_TX_ZONES if name != SOURCE_MISSING_ZONE
    )
    if zones != expected_baseline:
        raise RuntimeError(
            f"source no longer has the sealed simulation470 missing-zone signature: {zones}"
        )
    return {
        "source": str(source),
        "source_contract": contract,
        "source_mesh_setup_contract": mesh_setup_contract,
        "source_heat_source_contract": heat_source_contract,
        "source_boundary_temperatures": boundary_temperatures,
        "source_case": str(case_path),
        "source_monitor": str(monitor_path),
        "source_tx_zones": list(zones),
        "source_missing_tx_zones": [SOURCE_MISSING_ZONE],
        "source_result_eligible_as_accuracy_baseline": False,
        "accuracy_baseline_warning": ACCURACY_BASELINE_WARNING,
        "source_snapshot": source_snapshot,
    }


def _normalize_mesh_policy(policy: str, edge_fine_turns: int) -> tuple[str, int]:
    policy = str(policy).strip().lower()
    if policy not in {MESH_POLICY_UNIFORM, MESH_POLICY_HYBRID}:
        raise RuntimeError(
            f"mesh policy must be uniform or hybrid, got {policy!r}"
        )
    try:
        edge_fine_turns = int(edge_fine_turns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("hybrid edge-fine turns must be an integer") from exc
    if edge_fine_turns < 1:
        raise RuntimeError("hybrid edge-fine turns must be at least 1")
    return policy, edge_fine_turns


def _hybrid_tx_partition(
    tx_objects=EXPECTED_TX_OBJECTS,
    tx_object_ids=EXPECTED_TX_OBJECT_IDS,
    edge_fine_turns: int = DEFAULT_HYBRID_EDGE_FINE_TURNS,
) -> dict:
    tx_objects = tuple(str(value) for value in tx_objects)
    try:
        tx_object_ids = tuple(int(value) for value in tx_object_ids)
        edge_fine_turns = int(edge_fine_turns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("hybrid Tx IDs and k must be integers") from exc
    if not tx_objects or len(tx_objects) != len(tx_object_ids):
        raise RuntimeError(
            "hybrid Tx object names/IDs must be non-empty and have equal length"
        )
    if len(set(tx_objects)) != len(tx_objects):
        raise RuntimeError(f"hybrid Tx object-name overlap: {tx_objects}")
    if len(set(tx_object_ids)) != len(tx_object_ids):
        raise RuntimeError(f"hybrid Tx object-ID overlap: {tx_object_ids}")
    if edge_fine_turns < 1:
        raise RuntimeError("hybrid edge-fine turns must be at least 1")

    count = len(tx_objects)
    all_fine = count <= 2 * edge_fine_turns
    if all_fine:
        group_specs = [(
            "all_fine", range(count), TX_MESH_LEVEL,
            EXPECTED_TX_OPERATION_ID,
        )]
        next_unique_id = EXPECTED_SOURCE_NEXT_UNIQUE_ID
    else:
        group_specs = [
            (
                "left_fine", range(0, edge_fine_turns), TX_MESH_LEVEL,
                HYBRID_OPERATION_IDS[0],
            ),
            (
                "interior_coarse",
                range(edge_fine_turns, count - edge_fine_turns),
                TX_COARSE_MESH_LEVEL,
                HYBRID_OPERATION_IDS[1],
            ),
            (
                "right_fine", range(count - edge_fine_turns, count),
                TX_MESH_LEVEL, HYBRID_OPERATION_IDS[2],
            ),
        ]
        next_unique_id = HYBRID_NEXT_UNIQUE_ID

    operations = []
    assigned_indices = []
    for role, index_range, level, operation_id in group_specs:
        indices = tuple(index_range)
        if not indices:
            raise RuntimeError(f"hybrid Tx operation {role!r} is empty")
        assigned_indices.extend(indices)
        operations.append({
            "role": role,
            "name": HYBRID_OPERATION_NAMES[role],
            "operation_id": int(operation_id),
            "level": int(level),
            "indices": list(indices),
            "objects": [tx_objects[index] for index in indices],
            "object_ids": [tx_object_ids[index] for index in indices],
        })
    if sorted(assigned_indices) != list(range(count)) \
            or len(assigned_indices) != len(set(assigned_indices)):
        raise RuntimeError(
            f"hybrid Tx partition is incomplete or overlapping: {assigned_indices}"
        )
    operation_ids = [item["operation_id"] for item in operations]
    if len(operation_ids) != len(set(operation_ids)):
        raise RuntimeError(f"hybrid mesh operation ID overlap: {operation_ids}")

    turn_policy = []
    for index, (name, object_id) in enumerate(zip(tx_objects, tx_object_ids)):
        operation = next(
            item for item in operations if index in item["indices"]
        )
        turn_policy.append({
            "i": index,
            "edge_distance": min(index, count - 1 - index),
            "object": name,
            "object_id": object_id,
            "role": operation["role"],
            "level": operation["level"],
        })
    return {
        "policy": MESH_POLICY_HYBRID,
        "N": count,
        "edge_fine_turns": edge_fine_turns,
        "fine_level": TX_MESH_LEVEL,
        "coarse_level": TX_COARSE_MESH_LEVEL,
        "all_fine_due_to_small_N": all_fine,
        "next_unique_id": next_unique_id,
        "operations": operations,
        "turn_policy": turn_policy,
    }


def build_dry_run(
    source: Path,
    output_root: Path,
    library_root: Path,
    mesh_policy: str = MESH_POLICY_UNIFORM,
    edge_fine_turns: int = DEFAULT_HYBRID_EDGE_FINE_TURNS,
) -> dict:
    mesh_policy, edge_fine_turns = _normalize_mesh_policy(
        mesh_policy, edge_fine_turns,
    )
    source_info = _validate_source(source)
    library_root = library_root.resolve()
    library_src = library_root / "src"
    if not library_src.is_dir():
        raise RuntimeError(f"pyaedt library src is missing: {library_src}")
    output_root = output_root.resolve()
    simulation_root = (REPO_ROOT / "simulation").resolve()
    if output_root != simulation_root and simulation_root not in output_root.parents:
        raise RuntimeError(
            f"replay output must remain under the workspace simulation directory: {output_root}"
        )
    if output_root == source.parent.resolve() or source.parent.resolve() in output_root.parents:
        raise RuntimeError("replay output root may not be the protected source directory")
    library = _git_identity(library_root)
    if library["revision"] != EXPECTED_LIBRARY_REVISION or library["dirty"]:
        raise RuntimeError(
            "replay library identity mismatch: "
            f"revision={library['revision']} dirty={library['dirty']}"
        )
    report = {
        "mode": "dry-run",
        "time": _iso_now(),
        **source_info,
        "library": library,
        "output_root": str(output_root),
        "execute_required_for_mutation": True,
        "mesh_policy": mesh_policy,
        "planned_mesh_level": TX_MESH_LEVEL,
        "planned_separate_object_meshing": False,
        "planned_solve_calls": 1,
        "source_result_eligible_as_accuracy_baseline": False,
        "accuracy_baseline_warning": ACCURACY_BASELINE_WARNING,
    }
    if mesh_policy == MESH_POLICY_HYBRID:
        report["hybrid_tx_mesh_plan"] = _hybrid_tx_partition(
            edge_fine_turns=edge_fine_turns,
        )
    return report


def _capture_process_tree(root_pid: int, captured: dict[int, float]) -> None:
    import psutil

    if not isinstance(root_pid, int) or root_pid <= 0 or root_pid == os.getpid():
        raise RuntimeError(f"invalid owned AEDT PID: {root_pid!r}")
    root = psutil.Process(root_pid)
    for process in [root, *root.children(recursive=True)]:
        try:
            captured[process.pid] = process.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _descendant_process_identities(root_pid: int) -> dict[int, dict]:
    """Snapshot only descendants of this replay Python, never global AEDT."""
    import psutil

    identities = {}
    try:
        descendants = psutil.Process(root_pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return identities
    for process in descendants:
        try:
            cmdline = process.cmdline()
            identities[int(process.pid)] = {
                "pid": int(process.pid),
                "ppid": int(process.ppid()),
                "create_time": float(process.create_time()),
                "name": str(process.name() or ""),
                "commandline": " ".join(map(str, cmdline or [])),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError, OSError):
            continue
    return identities


def _same_process_identity(left: dict | None, right: dict | None) -> bool:
    if not left or not right:
        return False
    return int(left["pid"]) == int(right["pid"]) and abs(
        float(left["create_time"]) - float(right["create_time"])
    ) <= 0.05


def _is_headless_grpc_aedt(record: dict) -> bool:
    commandline = str(record.get("commandline") or "").lower()
    return (
        str(record.get("name") or "").lower() == "ansysedt.exe"
        and "-grpcsrv" in commandline
        and re.search(r"(?:^|\s)-ng(?:\s|$)", commandline) is not None
    )


def _new_owned_headless_tree(
    before: dict[int, dict], after: dict[int, dict], root_pid: int,
) -> dict[int, dict]:
    """Select a new headless AEDT plus its new worker ancestry/descendants.

    Both snapshots are scoped to descendants of ``root_pid``.  A candidate is
    rejected unless every ancestor up to the replay Python was also created
    after the pre-constructor snapshot.  This intentionally cannot select a
    concurrently running GUI or any other pre-existing process.
    """
    new = {
        pid: record for pid, record in after.items()
        if not _same_process_identity(before.get(pid), record)
    }
    headless = {pid for pid, record in new.items() if _is_headless_grpc_aedt(record)}
    if not headless:
        return {}

    selected = set(headless)
    for pid in tuple(headless):
        cursor = int(new[pid]["ppid"])
        visited = set()
        while cursor != root_pid:
            if cursor in visited or cursor not in new:
                raise RuntimeError(
                    f"new headless AEDT {pid} does not have exclusively new replay ancestry"
                )
            visited.add(cursor)
            selected.add(cursor)
            cursor = int(new[cursor]["ppid"])

    changed = True
    while changed:
        changed = False
        for pid, record in new.items():
            if pid not in selected and int(record["ppid"]) in selected:
                selected.add(pid)
                changed = True
    return {pid: new[pid] for pid in selected}


def _capture_identity_records(records: dict[int, dict], captured: dict[int, float]) -> None:
    for pid, record in records.items():
        captured[int(pid)] = float(record["create_time"])


def _terminate_captured_processes(captured: dict[int, float], wait_seconds: int = 15) -> None:
    import psutil

    live = []
    for pid, create_time in reversed(list(captured.items())):
        if pid == os.getpid():
            continue
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - create_time) > 0.05:
                continue
            process.terminate()
            live.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, survivors = psutil.wait_procs(live, timeout=wait_seconds)
    for process in survivors:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if survivors:
        _, survivors = psutil.wait_procs(survivors, timeout=wait_seconds)
    remaining = []
    for pid, create_time in captured.items():
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - create_time) <= 0.05 and process.is_running():
                remaining.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if remaining:
        raise RuntimeError(f"replay cleanup left owned PIDs alive: {remaining}")


def _assert_fresh(path: Path, started_ns: int, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"fresh {label} artifact is missing: {path}")
    if path.stat().st_mtime_ns < started_ns:
        raise RuntimeError(
            f"stale {label} artifact predates replay start: {path}"
        )


def _modeler_object_by_name(ipk, name: str):
    modeler = ipk.modeler
    getter = getattr(modeler, "get_object_from_name", None)
    if callable(getter):
        obj = getter(name)
        if obj is not None:
            return obj
    try:
        obj = modeler[name]
        if obj is not None:
            return obj
    except Exception:
        pass
    for obj in getattr(modeler, "objects", {}).values():
        if str(getattr(obj, "name", "")) == name:
            return obj
    raise RuntimeError(f"live thermal target object is unavailable: {name}")


def _field_summary_object_group(
    ipk, solution: str, object_names,
) -> dict[str, dict[str, float]]:
    object_names = tuple(object_names)
    if not object_names:
        raise RuntimeError("thermal target Field Summary group is empty")
    post = ipk.post
    if callable(post) and not hasattr(post, "create_field_summary"):
        post = post()
    summary = post.create_field_summary()
    for name in object_names:
        obj = _modeler_object_by_name(ipk, name)
        summary.add_calculation(
            "Object", "Volume" if getattr(obj, "is3d", True) else "Surface",
            name, "Temperature",
        )
    frame = summary.get_field_summary_data(setup=solution, pandas_output=True)
    if frame is None or isinstance(frame, bool) or not hasattr(frame, "columns") or not len(frame):
        raise RuntimeError("thermal target Field Summary returned no data")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    name_column = columns.get(
        "geometry name", columns.get("entity name", list(frame.columns)[2])
    )
    max_column = columns.get("max")
    mean_column = columns.get("mean")
    if max_column is None or mean_column is None:
        raise RuntimeError(
            f"thermal target Field Summary columns are incomplete: {list(frame.columns)}"
        )
    values = {}
    for name in object_names:
        rows = frame[frame[name_column].astype(str) == name]
        if len(rows) != 1:
            raise RuntimeError(
                f"thermal target Field Summary row count for {name}: {len(rows)}"
            )
        maximum = float(rows.iloc[0][max_column])
        mean = float(rows.iloc[0][mean_column])
        if not (math.isfinite(maximum) and math.isfinite(mean)):
            raise RuntimeError(f"thermal target Field Summary is non-finite for {name}")
        values[name] = {"max": maximum, "mean": mean}
    return values


def _field_summary_thermal_targets(
    ipk, solution: str,
) -> dict[str, dict[str, float]]:
    groups = (
        EXPECTED_TX_OBJECTS,
        EXPECTED_RX_MAIN_OBJECTS + EXPECTED_RX_SIDE_OBJECTS
        + EXPECTED_CORE_OBJECTS,
        EXPECTED_PROBE_OBJECTS,
    )
    values = {}
    for object_names in groups:
        last_error = None
        for attempt in range(1, 4):
            try:
                group_values = _field_summary_object_group(
                    ipk, solution, object_names,
                )
                values.update(group_values)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2)
        if last_error is not None:
            raise RuntimeError(
                "thermal target Field Summary group failed after 3 attempts: "
                f"objects={list(object_names)} error={last_error}"
            ) from last_error
    if set(values) != set(EXPECTED_THERMAL_TARGET_OBJECTS):
        raise RuntimeError(
            f"thermal target Field Summary object set mismatch: {sorted(values)}"
        )
    return values


def _field_calculator_object_group(
    ipk, solution: str, object_names,
) -> dict[str, dict[str, float]]:
    """Extract object temperatures through the scalar field calculator.

    AEDT 2025.2 can terminate ``ExportFieldsSummary`` abnormally when a
    completed Icepak project is reopened headlessly even though its solution
    fields remain readable.  The calculator uses those same saved fields but
    evaluates one object/statistic at a time, so it is an independent recovery
    path rather than a second solve.
    """
    object_names = tuple(object_names)
    if not object_names:
        raise RuntimeError("thermal target field-calculator group is empty")
    post = ipk.post
    if callable(post) and not hasattr(post, "get_scalar_field_value"):
        post = post()
    values = {}
    for name in object_names:
        obj = _modeler_object_by_name(ipk, name)
        object_type = "volume" if getattr(obj, "is3d", True) else "surface"
        stats = {}
        for label, scalar_function in (("max", "Maximum"), ("mean", "Mean")):
            value = post.get_scalar_field_value(
                "Temp",
                scalar_function=scalar_function,
                solution=solution,
                object_name=name,
                object_type=object_type,
            )
            if isinstance(value, bool) or value is None:
                raise RuntimeError(
                    "thermal target field calculator returned no value for "
                    f"{name} {scalar_function}: {value!r}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(
                    f"thermal target field calculator is non-finite for {name} "
                    f"{scalar_function}: {value!r}"
                )
            stats[label] = value
        values[name] = stats
    return values


def _field_calculator_thermal_targets(
    ipk, solution: str,
) -> dict[str, dict[str, float]]:
    groups = (
        EXPECTED_TX_OBJECTS,
        EXPECTED_RX_MAIN_OBJECTS + EXPECTED_RX_SIDE_OBJECTS
        + EXPECTED_CORE_OBJECTS,
        EXPECTED_PROBE_OBJECTS,
    )
    values = {}
    for object_names in groups:
        values.update(_field_calculator_object_group(ipk, solution, object_names))
    if set(values) != set(EXPECTED_THERMAL_TARGET_OBJECTS):
        raise RuntimeError(
            f"thermal target field-calculator object set mismatch: {sorted(values)}"
        )
    return values


def _extract_thermal_targets(ipk, solution: str) -> tuple[dict, dict]:
    """Read temperatures without ever remeshing or solving.

    Field Summary remains the fast primary path.  A saved-solution calculator
    fallback is used only for the AEDT reopen failure described above, and the
    chosen path plus the primary error are sealed into the replay manifest.
    """
    try:
        return _field_summary_thermal_targets(ipk, solution), {
            "method": "field_summary",
            "solution": solution,
            "solve_calls": 0,
        }
    except Exception as field_summary_error:
        values = _field_calculator_thermal_targets(ipk, solution)
        return values, {
            "method": "scalar_field_calculator_fallback",
            "quantity": "Temp",
            "statistics": ["Maximum", "Mean"],
            "solution": solution,
            "solve_calls": 0,
            "field_summary_error": (
                f"{type(field_summary_error).__name__}: {field_summary_error}"
            ),
        }


def _thermal_target_summary(
    values: dict[str, dict[str, float]],
) -> dict:
    def group_max(names):
        return max(values[name]["max"] for name in names)

    probes = {
        name: dict(values[name]) for name in EXPECTED_PROBE_OBJECTS
    }
    core_probe_names = (
        "Tprobe_core_center_leg",
        "Tprobe_core_side_leg",
        "Tprobe_core_top_yoke",
    )
    hottest_core_probe = max(
        core_probe_names, key=lambda name: values[name]["max"]
    )
    turns_by_max = sorted(
        EXPECTED_TX_OBJECTS,
        key=lambda name: values[name]["max"],
        reverse=True,
    )
    return {
        "tx_temperatures": {
            name: dict(values[name]) for name in EXPECTED_TX_OBJECTS
        },
        "other_object_temperatures": {
            name: dict(values[name])
            for name in EXPECTED_RX_MAIN_OBJECTS
            + EXPECTED_RX_SIDE_OBJECTS + EXPECTED_CORE_OBJECTS
        },
        "probe_temperatures": probes,
        "T_max_Tx": group_max(EXPECTED_TX_OBJECTS),
        "T_max_Rx_main": group_max(EXPECTED_RX_MAIN_OBJECTS),
        "T_max_Rx_side": group_max(EXPECTED_RX_SIDE_OBJECTS),
        "T_max_core": group_max(EXPECTED_CORE_OBJECTS),
        "Tprobe_core_center_max": values[hottest_core_probe]["max"],
        "Tprobe_core_center_mean": values[hottest_core_probe]["mean"],
        "hottest_core_probe": hottest_core_probe,
        "tx_hotspot_turn": turns_by_max[0],
        "tx_hotspot_rank": turns_by_max,
    }


def _attach_runtime(library_root: Path):
    library_src = str((library_root / "src").resolve())
    if library_src not in sys.path:
        sys.path.insert(0, library_src)
    os.environ["MFT_PYAEDT_LIBRARY_ROOT"] = str(library_root.resolve())
    os.environ.setdefault("FLEXLM_TIMEOUT", "3000000")
    from ansys.aedt.core import settings
    from pyaedt_module.core import pyDesktop

    settings.skip_license_check = True
    settings.wait_for_license = False
    return pyDesktop


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _prove_clone_containment(native_project, ipk, clone_file: Path, source: Path) -> dict:
    """Prove native AEDT and PyAEDT paths point only into the unique clone."""
    clone_file = clone_file.resolve()
    clone_dir = clone_file.parent
    source = source.resolve()
    native_directory = Path(str(native_project.GetPath())).resolve()
    native_file = (native_directory / f"{native_project.GetName()}.aedt").resolve()
    solver_project = Path(str(ipk.project_file)).resolve()
    solver_results = Path(str(ipk.results_directory)).resolve()
    expected_results = clone_file.with_suffix(".aedtresults").resolve()
    checks = {
        "native_project": native_file,
        "solver_project": solver_project,
        "solver_results": solver_results,
    }
    for label, path in checks.items():
        if not _is_within(path, clone_dir):
            raise RuntimeError(f"{label} escaped unique clone directory: {path}")
        if _is_within(path, source.parent):
            raise RuntimeError(f"{label} points into protected source directory: {path}")
    if native_file != clone_file or solver_project != clone_file:
        raise RuntimeError(
            "loaded project path mismatch: "
            f"native={native_file} solver={solver_project} clone={clone_file}"
        )
    if solver_results != expected_results:
        raise RuntimeError(
            f"loaded results path mismatch: {solver_results} != {expected_results}"
        )
    return {label: str(path) for label, path in checks.items()}


def _live_tx_objects(ipk) -> tuple[str, ...]:
    names = [
        str(name) for name in ipk.modeler.object_names
        if _TX_OBJECT_RE.fullmatch(str(name))
    ]
    return tuple(sorted(set(names), key=lambda name: int(_TX_OBJECT_RE.fullmatch(name).group(1))))


def _tx_mesh_operations(ipk) -> list:
    return [
        operation for operation in list(ipk.mesh.meshoperations)
        if str(getattr(operation, "name", "")).startswith("tx_mesh_level")
    ]


def _tx_mesh_operation(ipk):
    operations = _tx_mesh_operations(ipk)
    if len(operations) != 1:
        raise RuntimeError(
            f"cloned design must have exactly one Tx mesh operation, got {len(operations)}"
        )
    return operations[0]


def _assignment_names(ipk, values) -> tuple[str, ...]:
    if isinstance(values, (str, int)):
        values = [values]
    names = []
    for value in list(values or []):
        text = str(value)
        if _TX_OBJECT_RE.fullmatch(text):
            names.append(text)
            continue
        try:
            object_id = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"unrecognized Tx mesh assignment: {value!r}") from exc
        obj = None
        try:
            obj = ipk.modeler.objects.get(object_id)
        except (AttributeError, TypeError):
            pass
        if obj is not None:
            name = getattr(obj, "name", None)
        else:
            name = None
        if not name:
            editor = getattr(ipk, "oeditor", None) or getattr(ipk.modeler, "oeditor", None)
            if editor is not None:
                name = editor.GetObjectNameByID(object_id)
        if not name or not _TX_OBJECT_RE.fullmatch(str(name)):
            raise RuntimeError(f"Tx mesh object ID {object_id} did not resolve to a Tx object")
        names.append(str(name))
    return tuple(sorted(names, key=lambda name: int(_TX_OBJECT_RE.fullmatch(name).group(1))))


def _verify_operation_objects(
    ipk, operation, expected_objects=EXPECTED_TX_OBJECTS,
) -> tuple[str, ...]:
    expected_objects = tuple(expected_objects)
    prop_names = _assignment_names(ipk, operation.props.get("Objects", []))
    # Icepak 2025.2 exposes EditMeshOperation but does not support
    # GetMeshOpAssignment over gRPC.  Calling that dynamic proxy can block for
    # minutes instead of raising.  This replay is always a new gRPC desktop, so
    # native assignment readback is deliberately not attempted.  The live
    # wrapper props are bracketed by exact pre-load and post-save AEDT-file
    # object-ID/order contracts, providing independent durable attestation.
    native_names = None
    if prop_names != expected_objects or (
            native_names is not None and native_names != expected_objects):
        raise RuntimeError(
            "Tx mesh exact Objects set mismatch: "
            f"props={prop_names} native={native_names}"
        )
    return prop_names


def _edit_tx_mesh_operation_in_place(ipk) -> dict:
    tx_objects = _live_tx_objects(ipk)
    if tx_objects != EXPECTED_TX_OBJECTS:
        raise RuntimeError(f"live cloned Tx object set mismatch: {tx_objects}")
    operation = _tx_mesh_operation(ipk)
    operation_name = str(operation.name)
    _verify_operation_objects(ipk, operation)
    operation.auto_update = False
    for key in ("Level", "MaxLevel", "MinLevel"):
        operation.props[key] = str(TX_MESH_LEVEL)
    operation.props["Mesh Object(s) Separately Enabled"] = False
    if operation.update() is not True:
        raise RuntimeError("in-place Tx mesh operation EditMeshOperation failed")
    _verify_operation_objects(ipk, operation)
    for key in ("Level", "MaxLevel", "MinLevel"):
        if str(operation.props.get(key)) != str(TX_MESH_LEVEL):
            raise RuntimeError(f"in-place Tx mesh {key} readback mismatch: {operation.props}")
    if operation.props.get("Mesh Object(s) Separately Enabled") is not False:
        raise RuntimeError("in-place Tx mesh operation is not shared-region meshing")
    return {
        "edited_operation": operation_name,
        "objects": list(tx_objects),
        "level": TX_MESH_LEVEL,
        "separate": False,
    }


def _assert_durable_tx_mesh(project_file: Path, operation_name: str) -> dict:
    contract = _source_icepak_contract(project_file)
    if contract["mesh_operation"] != operation_name:
        raise RuntimeError(
            "durable Tx mesh operation identity changed: "
            f"{contract['mesh_operation']} != {operation_name}"
        )
    if tuple(contract["mesh_object_ids"]) != EXPECTED_TX_OBJECT_IDS:
        raise RuntimeError(
            f"durable Tx mesh Objects changed: {contract['mesh_object_ids']}"
        )
    if contract["mesh_levels"] != [TX_MESH_LEVEL] or contract["separate"] != "false":
        raise RuntimeError(f"durable Tx mesh policy mismatch: {contract}")
    if contract["operation_id"] != EXPECTED_TX_OPERATION_ID:
        raise RuntimeError(f"durable Tx mesh operation ID mismatch: {contract}")
    mesh_setup = _mesh_setup_operation_contract(project_file)
    if tuple(mesh_setup["operation_ids"]) != EXPECTED_SOURCE_MESH_OPERATION_IDS \
            or mesh_setup["next_unique_id"] != EXPECTED_SOURCE_NEXT_UNIQUE_ID:
        raise RuntimeError(f"durable uniform MeshSetup policy mismatch: {mesh_setup}")
    return contract


def _edit_cloned_tx_mesh_file(project_file: Path, level: int) -> dict:
    """Edit only the sealed clone's one Tx level block, preserving all other bytes."""
    level = int(level)
    if level < 1 or level > 5:
        raise RuntimeError(f"Tx replay mesh level must be in [1, 5], got {level}")
    before = project_file.read_bytes()
    marker = b"$begin 'tx_mesh_level"
    if before.count(marker) != 1:
        raise RuntimeError("cloned project does not contain exactly one Tx mesh block")
    start = before.index(marker)
    name_start = start + len(b"$begin '")
    name_end = before.index(b"'", name_start)
    operation_name = before[name_start:name_end].decode("ascii")
    end_marker = b"$end '" + operation_name.encode("ascii") + b"'"
    end = before.index(end_marker, name_end) + len(end_marker)
    block = before[start:end]
    for key in (b"MaxLevel", b"MinLevel"):
        old = key + b"='2'"
        if block.count(old) != 1:
            raise RuntimeError(
                f"sealed clone Tx block does not have exactly one {old!r}"
            )
        block = block.replace(old, key + b"='" + str(level).encode("ascii") + b"'")
    after = before[:start] + block + before[end:]
    if after == before:
        raise RuntimeError("cloned Tx mesh byte edit made no change")
    project_file.write_bytes(after)
    contract = _source_icepak_contract(project_file)
    if contract["mesh_levels"] != [level] \
            or tuple(contract["mesh_object_ids"]) != EXPECTED_TX_OBJECT_IDS \
            or contract["separate"] != "false":
        raise RuntimeError(f"cloned Tx mesh byte-edit contract mismatch: {contract}")
    return contract


def _replace_one_bytes(
    data: bytes, pattern: bytes, replacement: bytes, label: str,
) -> bytes:
    matches = list(re.finditer(pattern, data, flags=re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"Tx mesh template must have exactly one {label}")
    match = matches[0]
    return data[:match.start()] + match.expand(replacement) + data[match.end():]


def _render_hybrid_tx_block(
    template: bytes, source_name: str, operation: dict,
) -> bytes:
    source_name_bytes = source_name.encode("ascii")
    target_name_bytes = str(operation["name"]).encode("ascii")
    rendered = _replace_one_bytes(
        template,
        rb"\$begin '" + re.escape(source_name_bytes) + rb"'",
        b"$begin '" + target_name_bytes + b"'",
        "begin name",
    )
    rendered = _replace_one_bytes(
        rendered,
        rb"\$end '" + re.escape(source_name_bytes) + rb"'",
        b"$end '" + target_name_bytes + b"'",
        "end name",
    )
    rendered = _replace_one_bytes(
        rendered,
        rb"(?m)^(?P<indent>[ \t]*)ID=2(?P<ending>\r?)$",
        b"\\g<indent>ID=" + str(operation["operation_id"]).encode("ascii")
        + b"\\g<ending>",
        "ID=2 row",
    )
    object_row = b", ".join(
        str(value).encode("ascii") for value in operation["object_ids"]
    )
    rendered = _replace_one_bytes(
        rendered,
        rb"(?m)^(?P<indent>[ \t]*)Objects\([^\r\n]*\)(?P<ending>\r?)$",
        b"\\g<indent>Objects(" + object_row + b")\\g<ending>",
        "Objects row",
    )
    for key in (b"MaxLevel", b"MinLevel"):
        rendered = _replace_one_bytes(
            rendered,
            rb"(?m)^(?P<indent>[ \t]*)" + key
            + rb"='2'(?P<ending>\r?)$",
            b"\\g<indent>" + key + b"='"
            + str(operation["level"]).encode("ascii")
            + b"'\\g<ending>",
            key.decode("ascii") + "='2' row",
        )
    if rendered.count(b"'Mesh Object(s) Separately Enabled'=false") != 1:
        raise RuntimeError("Tx mesh template is not shared-region meshing")
    if rendered.count(b"DType='OpT'") != 1:
        raise RuntimeError("Tx mesh template is not exactly one mesh operation")
    return rendered


def _assert_hybrid_byte_change_scope(before: bytes, after: bytes) -> dict:
    """Prove all changed bytes are Tx blocks or MeshSetup NextUniqueID."""
    before_setup = _mesh_setup_byte_contract(before)
    after_setup = _mesh_setup_byte_contract(after)
    before_blocks = _tx_mesh_block_spans(before)
    after_blocks = _tx_mesh_block_spans(after)
    if len(before_blocks) != 1 or not after_blocks:
        raise RuntimeError(
            "hybrid byte-scope proof requires one source and at least one target Tx block"
        )
    for left, right in zip(after_blocks, after_blocks[1:]):
        gap = after[left["span"][1]:right["span"][0]]
        if gap.strip():
            raise RuntimeError("non-whitespace content was inserted between hybrid Tx blocks")

    before_next_start, before_next_end = before_setup["next_unique_id_span"]
    after_next_start, after_next_end = after_setup["next_unique_id_span"]
    before_tx_start, before_tx_end = before_blocks[0]["span"]
    after_tx_start = after_blocks[0]["span"][0]
    after_tx_end = after_blocks[-1]["span"][1]
    comparisons = (
        (before[:before_next_start], after[:after_next_start], "prefix"),
        (
            before[before_next_end:before_tx_start],
            after[after_next_end:after_tx_start],
            "NextUniqueID-to-Tx",
        ),
        (before[before_tx_end:], after[after_tx_end:], "suffix"),
    )
    for original, edited, label in comparisons:
        if original != edited:
            raise RuntimeError(
                f"hybrid byte edit changed unrelated {label} bytes"
            )
    return {
        "allowed_regions": ["MeshSetup.NextUniqueID", "Tx mesh operation block"],
        "before_next_unique_id": before_setup["next_unique_id"],
        "after_next_unique_id": after_setup["next_unique_id"],
        "before_tx_operation_count": len(before_blocks),
        "after_tx_operation_count": len(after_blocks),
        "unrelated_bytes_invariant": True,
    }


def _assert_hybrid_file_contract(project_file: Path, plan: dict) -> dict:
    tx_objects = _project_tx_objects(project_file)
    if tx_objects != EXPECTED_TX_OBJECTS:
        raise RuntimeError(
            f"durable hybrid Tx object name/order mismatch: {tx_objects}"
        )
    tx_contracts = _tx_mesh_contracts(project_file)
    mesh_setup = _mesh_setup_operation_contract(project_file)
    expected_operations = list(plan["operations"])
    if len(tx_contracts) != len(expected_operations):
        raise RuntimeError(
            "durable hybrid Tx operation count mismatch: "
            f"{len(tx_contracts)} != {len(expected_operations)}"
        )
    assigned_ids = []
    for actual, expected in zip(tx_contracts, expected_operations):
        expected_core = {
            "mesh_operation": expected["name"],
            "operation_id": expected["operation_id"],
            "mesh_object_ids": expected["object_ids"],
            "mesh_levels": [expected["level"]],
            "separate": "false",
        }
        actual_core = {key: actual[key] for key in expected_core}
        if actual_core != expected_core:
            raise RuntimeError(
                "durable hybrid Tx operation mismatch: "
                f"actual={actual_core} expected={expected_core}"
            )
        assigned_ids.extend(actual["mesh_object_ids"])
    if assigned_ids != list(EXPECTED_TX_OBJECT_IDS) \
            or len(assigned_ids) != len(set(assigned_ids)):
        raise RuntimeError(
            f"durable hybrid Tx assignments overlap or changed order: {assigned_ids}"
        )
    expected_all_operation_ids = (
        {1, 2, 3} if plan["all_fine_due_to_small_N"]
        else {1, 2, 3, 4, 5}
    )
    if set(mesh_setup["operation_ids"]) != expected_all_operation_ids \
            or len(mesh_setup["operation_ids"]) != len(expected_all_operation_ids):
        raise RuntimeError(
            f"durable hybrid MeshSetup operation IDs mismatch: {mesh_setup}"
        )
    expected_tx_operation_ids = [
        operation["operation_id"] for operation in expected_operations
    ]
    actual_tx_operation_ids = [item["operation_id"] for item in tx_contracts]
    if actual_tx_operation_ids != expected_tx_operation_ids:
        raise RuntimeError(
            "durable hybrid Tx operation ID/order mismatch: "
            f"{actual_tx_operation_ids} != {expected_tx_operation_ids}"
        )
    if mesh_setup["next_unique_id"] != plan["next_unique_id"]:
        raise RuntimeError(
            "durable hybrid NextUniqueID mismatch: "
            f"{mesh_setup['next_unique_id']} != {plan['next_unique_id']}"
        )
    return {
        "policy": MESH_POLICY_HYBRID,
        "tx_operations": tx_contracts,
        "mesh_setup": mesh_setup,
        "roles": [operation["role"] for operation in expected_operations],
        "partition_no_overlap": True,
        "partition_preserves_order": True,
    }


def _edit_cloned_tx_mesh_hybrid_file(
    project_file: Path,
    edge_fine_turns: int = DEFAULT_HYBRID_EDGE_FINE_TURNS,
) -> dict:
    """Replace only the sealed clone Tx block with a generalized hybrid plan."""
    before = project_file.read_bytes()
    source_contract = _source_icepak_contract(project_file)
    source_mesh_setup = _mesh_setup_operation_contract(project_file)
    if tuple(source_contract["tx_objects"]) != EXPECTED_TX_OBJECTS \
            or tuple(source_contract["mesh_object_ids"]) != EXPECTED_TX_OBJECT_IDS \
            or source_contract["mesh_levels"] != [2] \
            or source_contract["separate"] != "false" \
            or source_contract["operation_id"] != EXPECTED_TX_OPERATION_ID:
        raise RuntimeError(
            f"sealed clone Tx source contract mismatch: {source_contract}"
        )
    if tuple(source_mesh_setup["operation_ids"]) != \
            EXPECTED_SOURCE_MESH_OPERATION_IDS \
            or source_mesh_setup["next_unique_id"] != \
            EXPECTED_SOURCE_NEXT_UNIQUE_ID:
        raise RuntimeError(
            f"sealed clone MeshSetup source contract mismatch: {source_mesh_setup}"
        )
    plan = _hybrid_tx_partition(edge_fine_turns=edge_fine_turns)
    source_blocks = _tx_mesh_block_spans(before)
    if len(source_blocks) != 1:
        raise RuntimeError(
            f"sealed clone must have exactly one Tx block, got {len(source_blocks)}"
        )
    source_block = source_blocks[0]
    rendered_blocks = [
        _render_hybrid_tx_block(
            source_block["block"], source_block["name"], operation,
        )
        for operation in plan["operations"]
    ]
    line_start = before.rfind(b"\n", 0, source_block["span"][0]) + 1
    indentation = before[line_start:source_block["span"][0]]
    if indentation.strip():
        raise RuntimeError("sealed Tx block indentation is malformed")
    line_ending = b"\r\n" if b"\r\n" in source_block["block"] else b"\n"
    replacement = (line_ending + indentation).join(rendered_blocks)

    next_start, next_end = _mesh_setup_byte_contract(before)[
        "next_unique_id_span"
    ]
    edits = [
        (source_block["span"][0], source_block["span"][1], replacement),
        (
            next_start, next_end,
            str(plan["next_unique_id"]).encode("ascii"),
        ),
    ]
    after = before
    for start, end, value in sorted(edits, reverse=True):
        after = after[:start] + value + after[end:]
    if after == before:
        raise RuntimeError("cloned hybrid Tx mesh byte edit made no change")
    byte_scope = _assert_hybrid_byte_change_scope(before, after)
    project_file.write_bytes(after)
    durable_contract = _assert_hybrid_file_contract(project_file, plan)
    return {
        "plan": plan,
        "byte_scope": byte_scope,
        "durable_contract": durable_contract,
    }


def _attest_live_tx_mesh(ipk) -> dict:
    operation = _tx_mesh_operation(ipk)
    objects = _verify_operation_objects(ipk, operation)
    levels = {
        int(operation.props[key]) for key in ("MaxLevel", "MinLevel")
    }
    if levels != {int(TX_MESH_LEVEL)}:
        raise RuntimeError(f"live cloned Tx mesh level mismatch: {operation.props}")
    if operation.props.get("Mesh Object(s) Separately Enabled") is not False:
        raise RuntimeError("live cloned Tx mesh operation is not shared-region meshing")
    return {
        "edited_operation": str(operation.name),
        "objects": list(objects),
        "level": int(TX_MESH_LEVEL),
        "separate": False,
    }


def _attest_live_hybrid_tx_mesh(ipk, plan: dict) -> dict:
    tx_objects = _live_tx_objects(ipk)
    if tx_objects != EXPECTED_TX_OBJECTS:
        raise RuntimeError(f"live cloned Tx object set mismatch: {tx_objects}")
    operations = _tx_mesh_operations(ipk)
    expected_operations = list(plan["operations"])
    actual_names = [str(operation.name) for operation in operations]
    expected_names = [item["name"] for item in expected_operations]
    if actual_names != expected_names:
        raise RuntimeError(
            "live hybrid Tx operation name/order mismatch: "
            f"{actual_names} != {expected_names}"
        )
    live_contracts = []
    assigned_objects = []
    for operation, expected in zip(operations, expected_operations):
        expected_objects = tuple(expected["objects"])
        objects = _verify_operation_objects(
            ipk, operation, expected_objects=expected_objects,
        )
        levels = {
            int(operation.props[key]) for key in ("MaxLevel", "MinLevel")
        }
        if levels != {int(expected["level"])}:
            raise RuntimeError(
                f"live hybrid Tx mesh level mismatch for {operation.name}: "
                f"{operation.props}"
            )
        if operation.props.get("Mesh Object(s) Separately Enabled") is not False:
            raise RuntimeError(
                f"live hybrid Tx operation is not shared-region meshing: {operation.name}"
            )
        assigned_objects.extend(objects)
        live_contracts.append({
            "role": expected["role"],
            "operation": str(operation.name),
            "objects": list(objects),
            "level": int(expected["level"]),
            "separate": False,
        })
    if tuple(assigned_objects) != EXPECTED_TX_OBJECTS \
            or len(assigned_objects) != len(set(assigned_objects)):
        raise RuntimeError(
            "live hybrid Tx assignments overlap or changed order: "
            f"{assigned_objects}"
        )
    return {
        "policy": MESH_POLICY_HYBRID,
        "operations": live_contracts,
        "partition_no_overlap": True,
        "partition_preserves_order": True,
    }


def execute_replay(
    source: Path,
    output_root: Path,
    library_root: Path,
    cores: int = 4,
    mesh_policy: str = MESH_POLICY_UNIFORM,
    edge_fine_turns: int = DEFAULT_HYBRID_EDGE_FINE_TURNS,
) -> dict:
    mesh_policy, edge_fine_turns = _normalize_mesh_policy(
        mesh_policy, edge_fine_turns,
    )
    dry_run = build_dry_run(
        source, output_root, library_root,
        mesh_policy=mesh_policy,
        edge_fine_turns=edge_fine_turns,
    )
    source = source.resolve()
    source_before = dry_run["source_snapshot"]
    output_root = output_root.resolve()
    timestamp = datetime.now().astimezone().strftime("%y%m%d_%H%M%S")
    replay_label = (
        f"tx_l{TX_MESH_LEVEL}" if mesh_policy == MESH_POLICY_UNIFORM
        else f"tx_hybrid_k{edge_fine_turns}"
    )
    clone_dir = output_root / (
        f"simulation470_{replay_label}_{timestamp}_{uuid.uuid4().hex[:8]}"
    )
    clone_file = clone_dir / f"simulation470_{replay_label}_replay.aedt"
    clone_results = clone_file.with_suffix(".aedtresults")
    if clone_dir.exists():
        raise RuntimeError(f"unique replay directory already exists: {clone_dir}")
    clone_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, clone_file)
    if clone_results.exists() or Path(str(clone_file) + ".lock").exists():
        raise RuntimeError("aedt-only clone unexpectedly contains results or a lock")
    if _sha256(clone_file) != source_before["project"]["sha256"]:
        raise RuntimeError("aedt-only clone hash does not match protected source")
    if mesh_policy == MESH_POLICY_UNIFORM:
        cloned_file_contract = _edit_cloned_tx_mesh_file(
            clone_file, TX_MESH_LEVEL,
        )
        mesh_plan = None
    else:
        cloned_file_contract = _edit_cloned_tx_mesh_hybrid_file(
            clone_file, edge_fine_turns=edge_fine_turns,
        )
        mesh_plan = cloned_file_contract["plan"]

    pyDesktop = _attach_runtime(library_root.resolve())
    desktop = None
    desktop_owned = False
    captured: dict[int, float] = {}
    result = None
    primary_error = None
    primary_traceback = None
    failure_stage = "desktop_launch"
    finalization_errors = []
    replay_pid = os.getpid()
    descendants_before = _descendant_process_identities(replay_pid)
    try:
        desktop = pyDesktop(
            version=None,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=True,
        )
        desktop_pid = int(desktop.pid)
        import psutil

        desktop_create_time = psutil.Process(desktop_pid).create_time()
        descendants_after = _descendant_process_identities(replay_pid)
        owned_tree = _new_owned_headless_tree(
            descendants_before, descendants_after, replay_pid,
        )
        desktop_record = owned_tree.get(desktop_pid)
        if desktop_record is None or abs(
                float(desktop_record["create_time"]) - desktop_create_time) > 0.05:
            raise RuntimeError(
                "new_desktop replay did not create one owned headless AEDT tree: "
                f"desktop_pid={desktop_pid} owned={sorted(owned_tree)}"
            )
        desktop_owned = True
        failure_stage = "project_load"
        _capture_identity_records(owned_tree, captured)
        project = desktop.load_project(path=str(clone_file))
        native_project = project.project
        design_handle = native_project.SetActiveDesign(THERMAL_DESIGN)
        if not design_handle or design_handle.GetName() != THERMAL_DESIGN:
            raise RuntimeError("failed to activate cloned Icepak design")
        wrapper = project.create_design(
            name=THERMAL_DESIGN,
            solver="icepak",
            solution="SteadyState TemperatureAndFlow",
        )
        ipk = wrapper.solver_instance
        if str(ipk.design_name) != THERMAL_DESIGN:
            raise RuntimeError(f"attached wrong cloned design: {ipk.design_name}")

        cleanup_ok = ipk.cleanup_solution(
            variations="All",
            entire_solution=True,
            linked_data=True,
        )
        if cleanup_ok is not True:
            raise RuntimeError("cloned thermal cleanup_solution failed")
        if any(clone_results.glob(f"{THERMAL_DESIGN}.results/**/*_MON*_V*.sd")) \
                or any(clone_results.glob(f"{THERMAL_DESIGN}.results/**/current.nc_cas")):
            raise RuntimeError("stale cloned thermal artifacts survived cleanup_solution")

        failure_stage = "live_mesh_attestation"
        if mesh_policy == MESH_POLICY_UNIFORM:
            mesh_contract = _attest_live_tx_mesh(ipk)
        else:
            mesh_contract = _attest_live_hybrid_tx_mesh(ipk, mesh_plan)
        if ipk.save_project(file_name=str(clone_file), overwrite=True) is not True:
            raise RuntimeError("failed to save cloned project before replay")
        failure_stage = "durable_mesh_attestation"
        if mesh_policy == MESH_POLICY_UNIFORM:
            durable_mesh_contract = _assert_durable_tx_mesh(
                clone_file, mesh_contract["edited_operation"],
            )
        else:
            durable_mesh_contract = _assert_hybrid_file_contract(
                clone_file, mesh_plan,
            )

        replay_started_ns = time.time_ns()
        failure_stage = "mesh_generation"
        if ipk.mesh.generate_mesh(name=THERMAL_SETUP) is not True:
            raise RuntimeError("cloned ThermalSetup mesh generation failed")
        solve_calls = 0
        solve_calls += 1
        failure_stage = "thermal_analyze"
        analyze_result = ipk.analyze(setup=THERMAL_SETUP, cores=int(cores))
        if solve_calls != 1:
            raise RuntimeError(f"replay used {solve_calls} analyze calls")
        if analyze_result is False:
            raise RuntimeError("cloned thermal analyze returned False")
        _capture_process_tree(int(desktop.pid), captured)
        if ipk.save_project(file_name=str(clone_file), overwrite=True) is not True:
            raise RuntimeError("failed to save cloned project after replay")

        failure_stage = "artifact_attestation"
        case_path, monitor_path, artifact_identity = \
            _single_paired_result_artifacts(clone_file)
        profile_path = _profile_artifact(clone_file, artifact_identity)
        _assert_fresh(case_path, replay_started_ns, "case")
        _assert_fresh(monitor_path, replay_started_ns, "monitor")
        _assert_fresh(profile_path, replay_started_ns, "profile")
        zones = _tx_zones_from_case(case_path)
        if zones != EXPECTED_TX_ZONES:
            raise RuntimeError(f"fresh cloned Tx zone set mismatch: {zones}")

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from module.thermal_260706 import _parse_thermal_residual_monitor

        convergence = _parse_thermal_residual_monitor(monitor_path)
        if convergence.get("converged") is not True:
            raise RuntimeError(f"fresh cloned thermal replay did not converge: {convergence}")
        sweeps = list(ipk.existing_analysis_sweeps)
        if len(sweeps) != 1:
            raise RuntimeError(f"cloned thermal solution sweep count mismatch: {sweeps}")
        failure_stage = "field_summary"
        thermal_values, temperature_extraction = _extract_thermal_targets(
            ipk, sweeps[0],
        )
        thermal_summary = _thermal_target_summary(thermal_values)
        profile = _parse_icepak_profile(profile_path)
        clone_heat_source_contract = _heat_source_contract(clone_file)
        source_heat_source_contract = dry_run["source_heat_source_contract"]
        if clone_heat_source_contract != source_heat_source_contract:
            raise RuntimeError(
                "clone heat-source assignment contract changed from sealed source"
            )
        clone_boundary_temperatures = _thermal_boundary_temperatures(clone_file)
        if clone_boundary_temperatures != dry_run["source_boundary_temperatures"]:
            raise RuntimeError(
                "clone thermal boundary temperatures changed from sealed source"
            )
        failure_stage = "manifest_write"
        result = {
            "mode": "execute",
            "time": _iso_now(),
            "source": str(source),
            "clone": str(clone_file),
            "clone_results": str(clone_results),
            "library": dry_run["library"],
            "source_snapshot": source_before,
            "mesh_policy": mesh_policy,
            "mesh_contract": mesh_contract,
            "cloned_file_contract": cloned_file_contract,
            "durable_mesh_contract": durable_mesh_contract,
            "mesh_generated": True,
            "solve_calls": solve_calls,
            "case": str(case_path),
            "monitor": str(monitor_path),
            "profile": profile,
            "artifact_identity": artifact_identity,
            "tx_zones": list(zones),
            "convergence": convergence,
            "temperature_extraction": temperature_extraction,
            **thermal_summary,
            "boundary_temperatures": clone_boundary_temperatures,
            "heat_balance": {
                "input_assignment_invariant": True,
                "source_and_clone_contract_sha256":
                    clone_heat_source_contract["contract_sha256"],
                "entry_count": clone_heat_source_contract["entry_count"],
                "assigned_object_count":
                    clone_heat_source_contract["assigned_object_count"],
                "total_assigned_power_w":
                    clone_heat_source_contract["total_assigned_power_w"],
                "finite_nonnegative": True,
                "assignment_no_overlap": True,
                "solver_output_energy_balance_available": False,
            },
            "source_result_eligible_as_accuracy_baseline": False,
            "accuracy_baseline_warning": ACCURACY_BASELINE_WARNING,
        }
        manifest_path = clone_dir / "replay_manifest.json"
        result["manifest"] = str(manifest_path)
        manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except BaseException as exc:  # preserve cleanup/source-integrity evidence
        primary_error = exc
        primary_traceback = traceback.format_exc()
    finally:
        if desktop is not None:
            if desktop_owned:
                try:
                    _capture_process_tree(int(desktop.pid), captured)
                except Exception as exc:
                    finalization_errors.append(f"capture={type(exc).__name__}: {exc}")
                try:
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                except Exception as exc:
                    finalization_errors.append(f"release={type(exc).__name__}: {exc}")
            else:
                try:
                    desktop.release_desktop(close_projects=False, close_on_exit=False)
                except Exception as exc:
                    finalization_errors.append(
                        f"nonowned_release={type(exc).__name__}: {exc}"
                    )
        try:
            _terminate_captured_processes(captured)
        except Exception as exc:
            finalization_errors.append(f"pid_cleanup={type(exc).__name__}: {exc}")
        try:
            source_after = _source_snapshot(source)
            if source_after != source_before:
                finalization_errors.append("protected source identity changed during replay")
        except Exception as exc:
            finalization_errors.append(f"source_check={type(exc).__name__}: {exc}")

    if primary_error is not None or finalization_errors:
        details = []
        if primary_error is not None:
            details.append(
                f"stage={failure_stage} primary={type(primary_error).__name__}: "
                f"{primary_error} traceback={primary_traceback}"
            )
        details.extend(finalization_errors)
        raise RuntimeError("thermal replay failed closed: " + "; ".join(details)) from primary_error
    if result is None:
        raise RuntimeError("thermal replay produced no result")
    return result


def postprocess_replay_clone(
    clone_file: Path,
    library_root: Path,
) -> dict:
    """Extract a completed replay clone without cleanup, meshing, or solving."""
    clone_file = clone_file.resolve()
    simulation_root = (REPO_ROOT / "simulation").resolve()
    if not clone_file.is_file() or clone_file.suffix.lower() != ".aedt":
        raise RuntimeError(f"postprocess clone is missing: {clone_file}")
    if not _is_within(clone_file, simulation_root) \
            or _is_within(clone_file, DEFAULT_SOURCE.parent.resolve()):
        raise RuntimeError(f"postprocess clone is outside an allowed replay path: {clone_file}")
    if Path(str(clone_file) + ".lock").exists():
        raise RuntimeError(f"postprocess clone lock exists: {clone_file}.lock")
    source_info = _validate_source(DEFAULT_SOURCE)
    source_before = source_info["source_snapshot"]
    library = _git_identity(library_root.resolve())
    if library["revision"] != EXPECTED_LIBRARY_REVISION or library["dirty"]:
        raise RuntimeError(f"postprocess library identity mismatch: {library}")

    tx_contracts = _tx_mesh_contracts(clone_file)
    if len(tx_contracts) == 1:
        mesh_policy = MESH_POLICY_UNIFORM
        contract = tx_contracts[0]
        if contract["mesh_levels"] != [TX_MESH_LEVEL] \
                or tuple(contract["mesh_object_ids"]) != EXPECTED_TX_OBJECT_IDS \
                or contract["separate"] != "false":
            raise RuntimeError(f"postprocess uniform mesh contract mismatch: {contract}")
    else:
        mesh_policy = MESH_POLICY_HYBRID
        assigned = [
            object_id for contract in tx_contracts
            for object_id in contract["mesh_object_ids"]
        ]
        if assigned != list(EXPECTED_TX_OBJECT_IDS) \
                or len(assigned) != len(set(assigned)) \
                or [item["mesh_levels"] for item in tx_contracts] != [[4], [3], [4]]:
            raise RuntimeError(f"postprocess hybrid mesh contract mismatch: {tx_contracts}")

    case_path, monitor_path, artifact_identity = \
        _single_paired_result_artifacts(clone_file)
    profile_path = _profile_artifact(clone_file, artifact_identity)
    zones = _tx_zones_from_case(case_path)
    if zones != EXPECTED_TX_ZONES:
        raise RuntimeError(f"postprocess Tx zone set mismatch: {zones}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from module.thermal_260706 import _parse_thermal_residual_monitor
    convergence = _parse_thermal_residual_monitor(monitor_path)
    if convergence.get("converged") is not True:
        raise RuntimeError(f"postprocess replay did not converge: {convergence}")
    profile = _parse_icepak_profile(profile_path)

    pyDesktop = _attach_runtime(library_root.resolve())
    desktop = None
    desktop_owned = False
    captured: dict[int, float] = {}
    primary_error = None
    primary_traceback = None
    thermal_summary = None
    replay_pid = os.getpid()
    descendants_before = _descendant_process_identities(replay_pid)
    try:
        desktop = pyDesktop(
            version=None, non_graphical=True, new_desktop=True, close_on_exit=True,
        )
        desktop_pid = int(desktop.pid)
        import psutil
        desktop_create_time = psutil.Process(desktop_pid).create_time()
        owned_tree = _new_owned_headless_tree(
            descendants_before,
            _descendant_process_identities(replay_pid),
            replay_pid,
        )
        desktop_record = owned_tree.get(desktop_pid)
        if desktop_record is None or abs(
                float(desktop_record["create_time"]) - desktop_create_time) > 0.05:
            raise RuntimeError(
                f"postprocess did not create an owned headless AEDT tree: {owned_tree}"
            )
        desktop_owned = True
        _capture_identity_records(owned_tree, captured)
        project = desktop.load_project(path=str(clone_file))
        native_project = project.project
        design_handle = native_project.SetActiveDesign(THERMAL_DESIGN)
        if not design_handle or design_handle.GetName() != THERMAL_DESIGN:
            raise RuntimeError("postprocess failed to activate cloned Icepak design")
        wrapper = project.create_design(
            name=THERMAL_DESIGN,
            solver="icepak",
            solution="SteadyState TemperatureAndFlow",
        )
        ipk = wrapper.solver_instance
        _prove_clone_containment(native_project, ipk, clone_file, DEFAULT_SOURCE)
        sweeps = list(ipk.existing_analysis_sweeps)
        if len(sweeps) != 1:
            raise RuntimeError(f"postprocess solution sweep count mismatch: {sweeps}")
        thermal_values, temperature_extraction = _extract_thermal_targets(
            ipk, sweeps[0],
        )
        thermal_summary = _thermal_target_summary(thermal_values)
    except BaseException as exc:
        primary_error = exc
        primary_traceback = traceback.format_exc()
    finally:
        if desktop is not None:
            if desktop_owned:
                try:
                    _capture_process_tree(int(desktop.pid), captured)
                except Exception:
                    pass
                try:
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                except Exception:
                    pass
            else:
                try:
                    desktop.release_desktop(close_projects=False, close_on_exit=False)
                except Exception:
                    pass
        _terminate_captured_processes(captured)
        _assert_source_unchanged(DEFAULT_SOURCE, source_before, "during postprocess")
    if primary_error is not None:
        raise RuntimeError(
            "thermal replay postprocess failed closed: "
            f"{type(primary_error).__name__}: {primary_error} "
            f"traceback={primary_traceback}"
        ) from primary_error
    if thermal_summary is None:
        raise RuntimeError("thermal replay postprocess produced no temperature summary")

    clone_heat_source_contract = _heat_source_contract(clone_file)
    if clone_heat_source_contract != source_info["source_heat_source_contract"]:
        raise RuntimeError("postprocess clone heat-source contract changed")
    boundary_temperatures = _thermal_boundary_temperatures(clone_file)
    if boundary_temperatures != source_info["source_boundary_temperatures"]:
        raise RuntimeError("postprocess clone boundary temperatures changed")
    result = {
        "mode": "postprocess-existing-solved-clone",
        "time": _iso_now(),
        "clone": str(clone_file),
        "clone_results": str(clone_file.with_suffix(".aedtresults")),
        "mesh_policy": mesh_policy,
        "tx_mesh_contracts": tx_contracts,
        "library": library,
        "mesh_generated": True,
        "solve_calls_in_postprocess": 0,
        "case": str(case_path),
        "monitor": str(monitor_path),
        "profile": profile,
        "artifact_identity": artifact_identity,
        "tx_zones": list(zones),
        "convergence": convergence,
        "temperature_extraction": temperature_extraction,
        **thermal_summary,
        "boundary_temperatures": boundary_temperatures,
        "heat_balance": {
            "input_assignment_invariant": True,
            "source_and_clone_contract_sha256":
                clone_heat_source_contract["contract_sha256"],
            "entry_count": clone_heat_source_contract["entry_count"],
            "assigned_object_count": clone_heat_source_contract["assigned_object_count"],
            "total_assigned_power_w": clone_heat_source_contract["total_assigned_power_w"],
            "finite_nonnegative": True,
            "assignment_no_overlap": True,
            "solver_output_energy_balance_available": False,
        },
        "source_result_eligible_as_accuracy_baseline": False,
        "accuracy_baseline_warning": ACCURACY_BASELINE_WARNING,
    }
    manifest_path = clone_file.parent / "replay_manifest.json"
    result["manifest"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument(
        "--mesh-policy",
        choices=(MESH_POLICY_UNIFORM, MESH_POLICY_HYBRID),
        default=MESH_POLICY_UNIFORM,
    )
    parser.add_argument(
        "--hybrid-edge-fine-turns",
        type=int,
        default=DEFAULT_HYBRID_EDGE_FINE_TURNS,
        help=(
            "hybrid k: edge_distance < k uses L4 and interior turns use L3; "
            "N <= 2k remains all-L4"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--postprocess-clone", type=Path,
        help="extract one completed replay clone without cleanup, meshing, or solving",
    )
    args = parser.parse_args(argv)
    try:
        if args.cores < 1 or args.cores > 4:
            raise RuntimeError("replay cores must be in [1, 4]")
        if args.execute and args.postprocess_clone is not None:
            raise RuntimeError("--execute and --postprocess-clone are mutually exclusive")
        if args.postprocess_clone is not None:
            report = postprocess_replay_clone(
                args.postprocess_clone, args.library_root,
            )
        elif args.execute:
            report = execute_replay(
                args.source, args.output_root, args.library_root,
                cores=args.cores,
                mesh_policy=args.mesh_policy,
                edge_fine_turns=args.hybrid_edge_fine_turns,
            )
        else:
            report = build_dry_run(
                args.source, args.output_root, args.library_root,
                mesh_policy=args.mesh_policy,
                edge_fine_turns=args.hybrid_edge_fine_turns,
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "mode": (
                "postprocess-existing-solved-clone"
                if args.postprocess_clone is not None
                else ("execute" if args.execute else "dry-run")
            ),
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
