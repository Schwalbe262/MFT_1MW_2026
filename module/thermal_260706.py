"""
Icepak 열해석 모듈 (설계도면260706 파이프라인 design3)

- 지오메트리: 항상 풀모델 (팬 유동 +y -> -y 가 대칭을 깨므로), 라운드 없이 직각
- 하이브리드 권선: Tx(5mm)는 턴 전부 explicit, Rx foil은 안/밖 n_explicit_turns씩 explicit,
  중간 턴들은 변마다 1개씩 4개의 직육면체 블록으로 균질화 (이방성 열전도율)
- 손실 주입: EM loss 디자인(design2) 계산기 값 사용. 대칭 EM인 경우 오브젝트별 환산:
    * 대칭 모델의 체적 적분값은 "절단 평면 수 c"에 대해 (실제값) x 2^c / 4 로 나타남
      (검증 실험: 3면 절단 오브젝트 = 실제의 1/2, 2면 절단 = 실제와 동일)
    * 따라서 실제값 = 대칭값 x 4 / 2^c, 대칭 모델에서 삭제된 미러 오브젝트는 대응값 복제
- 경계조건: 콜드플레이트/권선냉각판(Al) 고정온도, region +y면 velocity inlet, -y면 pressure opening
"""

import hashlib
import json
import math
import logging
import re
import time
from pathlib import Path

import pandas as pd

from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    compute_layer_positions,
)
from module.input_parameter_260706 import get_tx_y_gaps, set_design_variables
from module.core_material_contract import PHYSICS_DATA_REVISION
from module.thermal_probe_contract import (
    ProbeSheetCollection,
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
    aggregate_rx_side_faces,
    parse_temperature_celsius,
    serialize_probe_failures,
    validate_probe_rectangle,
)


def _native_solver(app):
    """Return the PyAEDT solver behind a pyDesign wrapper."""
    solver = getattr(app, "solver_instance", None)
    return solver if solver is not None else app


_THERMAL_DESIGN_NAME = "icepak_thermal"
_THERMAL_SETUP_NAME = "ThermalSetup"


def _power_value_w(value):
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*",
        str(value),
    )
    if not match:
        raise RuntimeError(f"cannot parse native block power: {value!r}")
    factors = {"w": 1.0, "mw": 1e-3, "kw": 1e3}
    unit = match.group(2).lower() or "w"
    if unit not in factors:
        raise RuntimeError(f"unsupported native block power unit: {unit!r}")
    power = float(match.group(1)) * factors[unit]
    if not math.isfinite(power) or power < 0:
        raise RuntimeError(f"invalid native block power: {value!r}")
    return power


def _native_block_readback(boundary, obj):
    """Read a freshly assigned Icepak block through the native OO child tree."""
    child = getattr(boundary, "_child_object", None)
    if child is None:
        raise RuntimeError(
            f"native boundary child unavailable for {getattr(boundary, 'name', boundary)!r}"
        )
    prop_names = set(str(name) for name in (child.GetPropNames() or []))
    # AEDT 2025.2 exposes the block-type property as plain "Type" (observed
    # live on the cluster: Assignment/Name/Total Power/Type/Use External
    # Conditions/Use Total Power); older builds named it "Block Type".
    block_type_prop = next(
        (name for name in ("Block Type", "Type") if name in prop_names), None
    )
    missing = sorted({"Use Total Power", "Total Power"} - prop_names)
    if block_type_prop is None:
        missing = sorted(missing + ["Block Type|Type"])
    if missing:
        raise RuntimeError(
            f"native block readback missing properties {missing}: {sorted(prop_names)}"
        )
    block_type = str(child.GetPropValue(block_type_prop))
    use_total = child.GetPropValue("Use Total Power")
    if block_type not in ("Solid", "Solid Block") or str(use_total).strip().lower() not in {
        "true", "1"
    }:
        raise RuntimeError(
            f"native block contract mismatch: type={block_type!r}, "
            f"use_total={use_total!r}"
        )
    power_w = _power_value_w(child.GetPropValue("Total Power"))

    assignment_prop = next(
        (name for name in ("Objects", "Assignment", "Parts") if name in prop_names),
        None,
    )
    if assignment_prop is None:
        raise RuntimeError("native block readback has no assignment property")
    assignment = child.GetPropValue(assignment_prop)
    values = list(assignment) if isinstance(assignment, (list, tuple)) else [assignment]
    expected_name = str(obj.name)
    names = {str(value).strip().strip('"') for value in values}
    if expected_name not in names:
        editor = getattr(obj, "_oeditor", None)
        get_name = getattr(editor, "GetObjectNameByID", None)
        if callable(get_name):
            converted = set()
            for value in values:
                try:
                    converted.add(str(get_name(int(value))))
                except (TypeError, ValueError, OverflowError):
                    converted.add(str(value).strip().strip('"'))
            names = converted
    if names != {expected_name}:
        raise RuntimeError(
            f"native block assignment mismatch: {names!r} != {{{expected_name!r}}}"
        )
    return {"object": expected_name, "power_w": power_w}


def _native_design_name(design):
    """Return an AEDT design's leaf name without project qualification."""
    value = design
    get_name = getattr(design, "GetName", None)
    if callable(get_name):
        value = get_name()
    return str(value or "").split(";")[-1].strip()


def _activate_thermal_design(app, design_name=None, native_project=None):
    """Re-acquire the active native design and refresh the raw PyAEDT handle."""
    solver = _native_solver(app)
    project = native_project
    if project is None:
        project = getattr(solver, "oproject", None)
    set_active = getattr(project, "SetActiveDesign", None)
    if not callable(set_active):
        raise RuntimeError("native Icepak project handle has no SetActiveDesign")

    expected_name = design_name or getattr(solver, "design_name", None) \
        or getattr(app, "design_name", None)
    if not expected_name:
        raise RuntimeError("thermal design name is unavailable")
    design = set_active(expected_name)
    if not design or not callable(getattr(design, "GetModule", None)):
        raise RuntimeError(f"failed to activate thermal design: {expected_name}")
    if design_name is not None:
        actual_name = _native_design_name(design)
        if actual_name != design_name:
            raise RuntimeError(
                "thermal design identity mismatch: "
                f"expected={design_name!r}, actual={actual_name or '<empty>'!r}"
            )

    solver._oproject = project
    solver._odesign = design
    design_solutions = getattr(solver, "design_solutions", None)
    if design_solutions is not None:
        design_solutions._odesign = design
    return solver, design


def _require_boundary(result, operation, expected_props=None):
    """Require a created PyAEDT boundary and, when requested, its input props."""
    if not result:
        raise RuntimeError(f"{operation} returned no boundary")
    if expected_props is None:
        return result

    props = getattr(result, "props", None)
    if not hasattr(props, "get"):
        raise RuntimeError(f"{operation} returned a boundary without readable props")
    for key, expected in expected_props.items():
        actual = props.get(key)
        if key == "Objects":
            actual_names = [actual] if isinstance(actual, str) else list(actual or [])
            expected_names = [expected] if isinstance(expected, str) else list(expected)
            if set(map(str, actual_names)) != set(map(str, expected_names)):
                raise RuntimeError(
                    f"{operation} property {key!r} mismatch: {actual_names!r} != {expected_names!r}"
                )
        elif str(actual) != str(expected):
            raise RuntimeError(f"{operation} property {key!r} mismatch: {actual!r} != {expected!r}")
    return result


def _require_thermal_geometry(
    objs,
    mode,
    n2_side,
    require_core_plates=False,
    require_core_pads=False,
    require_wcp_plates=False,
    require_wcp_pads=False,
):
    """Require every physical group that the requested thermal model should contain."""
    required_groups = {
        "core": objs["core"],
        "Tx": objs["Tx"],
        "Rx_main": objs["Rx_main_explicit"] + objs["Rx_main_blocks"],
    }
    if int(n2_side) > 0:
        required_groups["Rx_side"] = objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
        if mode == "full":
            required_groups["Rx_side2"] = objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"]
    if require_core_plates:
        required_groups["core_plates"] = objs["core_plates"]
    if require_core_pads:
        required_groups["core_pads"] = objs["core_pads"]
    if require_wcp_plates:
        required_groups["wcp_plates"] = objs["wcp_plates"]
    if require_wcp_pads:
        required_groups["wcp_pads"] = objs["wcp_pads"]
    missing_groups = [name for name, group in required_groups.items() if not group]
    if missing_groups:
        raise RuntimeError(f"thermal geometry is missing required groups: {missing_groups}")


def _assign_thermal_mesh(ipk, objs, side_block_level=5):
    """Keep thin solids represented without isolating their thermal interfaces."""
    def _assign_levels(levels, name):
        if not levels:
            return
        operation_names = ipk.mesh.assign_mesh_level(levels, name=name)
        if not isinstance(operation_names, (list, tuple)) or not operation_names:
            raise RuntimeError(f"{name} assignment returned no mesh operation")
        operations = {
            str(getattr(operation, "name", "")): operation
            for operation in getattr(ipk.mesh, "meshoperations", [])
        }
        for item in operation_names:
            operation = item if callable(getattr(item, "update", None)) else operations.get(str(item))
            update = getattr(operation, "update", None)
            if not callable(update):
                raise RuntimeError(f"{name} mesh operation is unavailable: {item}")
            # Separate-object cut-cell regions can leave a retained solid with no
            # conductive/convective path to the surrounding fluid. Keep the object
            # level control, but mesh all controlled solids in the shared region.
            operation.auto_update = False
            operation.props["Mesh Object(s) Separately Enabled"] = False
            if not update():
                raise RuntimeError(f"{name} mesh operation update failed: {item}")
            if operation.props.get("Mesh Object(s) Separately Enabled") is not False:
                raise RuntimeError(f"{name} shared-region mesh setting was not retained: {item}")

    pad_names = [o.name for o in objs.get("wcp_pads", []) + objs.get("core_pads", [])]
    _assign_levels({name: 2 for name in pad_names}, "pad_mesh_level")

    # Tx turns as thin as the sampled 1 mm lower bound can disappear from the
    # shared cut-cell mesh. Level 4 preserves those solid zones while one shared
    # operation avoids the isolated heat paths of separate-object meshing.
    tx_names = list(dict.fromkeys(obj.name for obj in objs.get("Tx", [])))
    _assign_levels({name: 4 for name in tx_names}, "tx_mesh_level")

    # Keep each physical Rx pack in its own shared refinement region. Combining
    # the distant main and side packs creates one very large cut-cell region;
    # two production cases with a single 0.300--0.435 mm side turn then lost all
    # three retained side solution zones even though the main pack survived.
    # Exact singleton turns remain protected at level 5 below. Multi-turn side
    # blocks use the fixed-run A/B level while all blocks in one pack continue
    # to share a region so their conductive interfaces are not isolated.
    side_block_level = int(side_block_level)
    if side_block_level not in (4, 5):
        raise ValueError("side_block_level must be 4 or 5")
    rx_block_specs = (
        ("Rx_main_blocks", "rx_main_block_mesh_level", 4),
        ("Rx_side_blocks", "rx_side_block_mesh_level", side_block_level),
        ("Rx_side2_blocks", "rx_side2_block_mesh_level", side_block_level),
    )
    for key, operation_name, level in rx_block_specs:
        names = list(dict.fromkeys(obj.name for obj in objs.get(key, [])))
        _assign_levels({name: level for name in names}, operation_name)

    explicit_rx = []
    singleton_specs = (
        ("Rx_main_explicit", "Rx_main_blocks", "rx_main_single_turn_mesh_level"),
        ("Rx_side_explicit", "Rx_side_blocks", "rx_side_single_turn_mesh_level"),
        ("Rx_side2_explicit", "Rx_side2_blocks", "rx_side2_single_turn_mesh_level"),
    )
    for explicit_key, block_key, operation_name in singleton_specs:
        group = list(objs.get(explicit_key, []))
        if len(group) == 1 and not objs.get(block_key, []):
            # Level 5 is Icepak's finest predefined object level. Keep this as a
            # pack-local shared region so a 0.3 mm exact copper turn remains a
            # solved solid without refining the distant main pack.
            _assign_levels({group[0].name: 5}, operation_name)
        else:
            explicit_rx.extend(group)

    # Per-object cut-cell subregions created million-cell meshes with skew above
    # 0.96 on thin foils. One object-level control keeps the foil represented
    # without introducing subregion interfaces around every retained turn.
    rx_names = list(dict.fromkeys(obj.name for obj in explicit_rx))
    _assign_levels({name: 3 for name in rx_names}, "rx_mesh_level")


_THERMAL_RESIDUAL_FIELDS = (
    "Continuity",
    "XVelocity",
    "YVelocity",
    "ZVelocity",
    "Energy",
)
_THERMAL_RESIDUAL_ROW = re.compile(
    r"^\s*(?P<iteration>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_THERMAL_RESIDUAL_VALUE = re.compile(
    r"(?P<name>Continuity|XVelocity|YVelocity|ZVelocity|Energy)"
    r"\((?P<value>[^)]*)\)"
)


def _parse_thermal_residual_monitor(path, flow_limit=1e-3, energy_limit=1e-7):
    """Parse the final complete Icepak residual row and apply its solve criteria."""
    last_record = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _THERMAL_RESIDUAL_ROW.match(line)
        if not match:
            continue
        iteration_value = float(match.group("iteration"))
        if not math.isfinite(iteration_value) or iteration_value < 0:
            continue
        tokens = list(_THERMAL_RESIDUAL_VALUE.finditer(line))
        raw_values = {
            item.group("name"): item.group("value").strip()
            for item in tokens
        }
        last_record = (iteration_value, raw_values, len(tokens))
    if last_record is None:
        raise ValueError(f"no residual rows in {path}")

    iteration_value, raw_values, token_count = last_record
    if not iteration_value.is_integer() or iteration_value <= 0:
        raise ValueError(f"final residual iteration is invalid in {path}")
    iteration = int(iteration_value)
    if token_count != len(_THERMAL_RESIDUAL_FIELDS) \
            or set(raw_values) != set(_THERMAL_RESIDUAL_FIELDS):
        raise ValueError(f"final residual row is incomplete in {path}")
    try:
        values = {name: float(raw_values[name]) for name in _THERMAL_RESIDUAL_FIELDS}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"final residual row is non-numeric in {path}") from exc
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError(f"final residual row is non-finite in {path}")
    flow_max = max(values[name] for name in _THERMAL_RESIDUAL_FIELDS[:-1])
    converged = flow_max <= float(flow_limit) and values["Energy"] <= float(energy_limit)
    return {
        "iteration": iteration,
        "values": values,
        "flow_limit": float(flow_limit),
        "energy_limit": float(energy_limit),
        "converged": converged,
    }


def _thermal_monitor_roots(sim, ipk):
    """Return de-duplicated native results roots without recursive traversal."""
    native_ipk = _native_solver(ipk)
    roots = []
    results_directory = getattr(native_ipk, "results_directory", None)
    if results_directory:
        roots.append(Path(str(results_directory)))
    project_path = getattr(sim, "project_path", None)
    project_name = str(getattr(sim, "PROJECT_NAME", "") or "").strip()
    if project_path and project_name:
        roots.append(Path(project_path) / f"{project_name}.aedtresults")

    unique = []
    seen = set()
    for root in roots:
        key = str(root.resolve(strict=False)).casefold()
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _thermal_monitor_signature(path, sample_bytes=4096):
    """Return a bounded signature that detects changes despite coarse mtimes."""
    path = Path(path)
    stat = path.stat()
    size = int(stat.st_size)
    sample_bytes = max(1, int(sample_bytes))
    with path.open("rb") as stream:
        head = stream.read(min(sample_bytes, size))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            tail = stream.read(sample_bytes)
        else:
            tail = b""
    digest = hashlib.sha256(head + b"\0" + tail).hexdigest()
    return size, int(stat.st_mtime_ns), digest


def _thermal_monitor_candidates(sim, ipk):
    design_name = str(getattr(ipk, "design_name", "") or _THERMAL_DESIGN_NAME)
    candidates = {}
    for root in _thermal_monitor_roots(sim, ipk):
        design_results = root / f"{design_name}.results"
        search_root = design_results if design_results.is_dir() else root
        if not search_root.is_dir():
            continue
        for path in search_root.glob("*_S*_MON*_V*.sd"):
            if "_SOL" in path.name:
                continue
            try:
                signature = _thermal_monitor_signature(path)
                key = str(path.resolve(strict=False)).casefold()
            except OSError:
                continue
            candidates[key] = (path, signature)
    return candidates


def _snapshot_thermal_monitors(sim, ipk):
    """Snapshot existing residual artifacts before a solve is dispatched."""
    return {
        key: signature
        for key, (_path, signature) in _thermal_monitor_candidates(sim, ipk).items()
    }


def _thermal_convergence_telemetry(
    sim, ipk, setup, attempts=3, retry_seconds=2, not_before_ns=None,
    monitor_snapshot=None,
):
    """Read a fresh native Icepak residual monitor; missing evidence fails closed."""
    defaults = {
        "thermal_convergence_available": 0,
        "thermal_converged": 0,
        "thermal_iterations": 0,
        "thermal_residual_continuity": float("nan"),
        "thermal_residual_x_velocity": float("nan"),
        "thermal_residual_y_velocity": float("nan"),
        "thermal_residual_z_velocity": float("nan"),
        "thermal_residual_energy": float("nan"),
        "thermal_residual_flow_limit": float("nan"),
        "thermal_residual_energy_limit": float("nan"),
        "thermal_convergence_reason": "monitor_missing",
        "thermal_monitor_file": "",
    }
    try:
        flow_limit = float(setup.props.get("Convergence Criteria - Flow", 1e-3))
        energy_limit = float(setup.props.get("Convergence Criteria - Energy", 1e-7))
    except (TypeError, ValueError, OverflowError):
        return {**defaults, "thermal_convergence_reason": "invalid_setup_criteria"}
    if not (math.isfinite(flow_limit) and 0 < flow_limit <= 1e-3
            and math.isfinite(energy_limit) and 0 < energy_limit <= 1e-7):
        return {**defaults, "thermal_convergence_reason": "invalid_setup_criteria"}

    last_error = None
    malformed_monitor = ""
    for attempt in range(1, attempts + 1):
        candidates = []
        for key, (path, signature) in _thermal_monitor_candidates(sim, ipk).items():
            if monitor_snapshot is not None:
                previous = monitor_snapshot.get(key)
                # A metadata-only touch is not solve evidence.  Require a new
                # monitor path or a bounded content/size change; mtime remains
                # in the signature only for deterministic candidate ordering.
                if previous is not None \
                        and (previous[0], previous[2]) == (signature[0], signature[2]):
                    continue
            elif not_before_ns is not None and signature[1] < int(not_before_ns):
                continue
            candidates.append((path, signature))
        candidates.sort(key=lambda item: (item[1][1], item[0].name), reverse=True)
        for monitor, _signature in candidates:
            try:
                parsed = _parse_thermal_residual_monitor(
                    monitor, flow_limit=flow_limit, energy_limit=energy_limit
                )
                values = parsed["values"]
                return {
                    "thermal_convergence_available": 1,
                    "thermal_converged": 1 if parsed["converged"] else 0,
                    "thermal_iterations": parsed["iteration"],
                    "thermal_residual_continuity": values["Continuity"],
                    "thermal_residual_x_velocity": values["XVelocity"],
                    "thermal_residual_y_velocity": values["YVelocity"],
                    "thermal_residual_z_velocity": values["ZVelocity"],
                    "thermal_residual_energy": values["Energy"],
                    "thermal_residual_flow_limit": parsed["flow_limit"],
                    "thermal_residual_energy_limit": parsed["energy_limit"],
                    "thermal_convergence_reason": (
                        "converged" if parsed["converged"] else "residual_threshold"
                    ),
                    "thermal_monitor_file": monitor.name,
                }
            except (OSError, ValueError) as exc:
                last_error = exc
                malformed_monitor = monitor.name
        if attempt < attempts:
            time.sleep(retry_seconds)

    if last_error is not None:
        logging.error("[thermal] residual monitor validation failed: %s", last_error)
        return {
            **defaults,
            "thermal_convergence_reason": "monitor_malformed",
            "thermal_monitor_file": malformed_monitor,
        }
    return defaults


def _thermal_bool(value):
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "enabled", "on", "1"}:
        return True
    if normalized in {"false", "no", "disabled", "off", "0"}:
        return False
    raise RuntimeError(f"unrecognized ThermalSetup Enabled value: {value!r}")


def _thermal_desktop_handle(sim, ipk):
    getter = getattr(sim, "_native_desktop_handle", None)
    if callable(getter):
        desktop = getter()
        if desktop is not None and desktop is not False:
            return desktop
    for owner in (_native_solver(ipk), ipk):
        desktop = getattr(owner, "odesktop", None)
        if desktop is not None and desktop is not False:
            return desktop
    raise RuntimeError("native AEDT Desktop handle is unavailable")


def _thermal_running_state(sim, ipk):
    # Desktop-wide running state is meaningless on a shared pooled AEDT
    # session (sibling clients solve concurrently); callers treat this
    # exception as "no evidence" rather than a false positive.
    from module.aedt_pool_adapter import pooled_backend_enabled
    if pooled_backend_enabled():
        raise RuntimeError(
            "Desktop-wide simulation state is not meaningful on a shared "
            "pooled AEDT session"
        )
    desktop = _thermal_desktop_handle(sim, ipk)
    is_running = getattr(desktop, "AreThereSimulationsRunning", None)
    if not callable(is_running):
        raise RuntimeError("native AEDT Desktop has no simulation-state query")
    value = is_running()
    if value is False or value == 0:
        return False
    if value is True or value == 1:
        return True
    normalized = str(value or "").strip().lower()
    if normalized in {"false", "no", "off", "0"}:
        return False
    if normalized in {"true", "yes", "on", "1"}:
        return True
    raise RuntimeError(f"unrecognized AEDT simulation-running state: {value!r}")


def _prepare_thermal_dispatch(
    sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
    setup_name=_THERMAL_SETUP_NAME,
):
    """Rebind and attest the one exact native Icepak setup before dispatch."""
    expected_project = str(getattr(sim, "PROJECT_NAME", "") or "").strip()
    if not expected_project:
        raise RuntimeError("thermal project identity is unavailable")
    rebind = getattr(sim, "_rebind_native_project_for_design_creation", None)
    if not callable(rebind):
        raise RuntimeError("thermal project rebind is unavailable")
    native_project = rebind()
    if native_project is None or native_project is False:
        raise RuntimeError("thermal project rebind returned no native project")
    get_project_name = getattr(native_project, "GetName", None)
    if not callable(get_project_name):
        raise RuntimeError("rebound thermal project has no identity readback")
    actual_project = str(get_project_name() or "").strip()
    if actual_project != expected_project:
        raise RuntimeError(
            "thermal project identity mismatch: "
            f"expected={expected_project!r}, actual={actual_project or '<empty>'!r}"
        )

    native_ipk, native_design = _activate_thermal_design(
        ipk, design_name=design_name, native_project=native_project
    )
    wrapper_name = str(getattr(native_ipk, "design_name", "") or "").strip()
    if wrapper_name != design_name:
        raise RuntimeError(
            "thermal wrapper design identity mismatch: "
            f"expected={design_name!r}, actual={wrapper_name or '<empty>'!r}"
        )
    get_design_type = getattr(native_design, "GetDesignType", None)
    design_type = str(get_design_type() or "") if callable(get_design_type) else ""
    if design_type and "icepak" not in design_type.lower():
        raise RuntimeError(f"thermal design is not Icepak: {design_type!r}")

    analysis = native_design.GetModule("AnalysisSetup")
    if analysis is None or analysis is False:
        raise RuntimeError("active thermal design returned no AnalysisSetup module")
    get_setups = getattr(analysis, "GetSetups", None)
    if not callable(get_setups):
        raise RuntimeError("thermal AnalysisSetup has no setup readback")
    setups = tuple(str(name) for name in (get_setups() or []))
    if setups != (setup_name,):
        raise RuntimeError(
            f"native thermal setup mismatch: expected={(setup_name,)}, actual={setups}"
        )
    # PyAEDT caches the AnalysisSetup module independently from ``_odesign``.
    # Merely rebinding the native design can therefore leave ``setup_names``
    # pointed at the prior design; in that state ``analyze(setup=...)`` returns
    # its default success value without calling oDesign.Analyze at all.  Keep the
    # cache in the same attested transaction and require the wrapper readback to
    # agree before dispatch.
    try:
        native_ipk._oanalysis = analysis
        wrapper_setups = tuple(str(name) for name in (native_ipk.setup_names or []))
    except Exception as exc:
        raise RuntimeError(
            f"thermal PyAEDT AnalysisSetup rebind failed: {type(exc).__name__}: {exc}"
        ) from exc
    if wrapper_setups != (setup_name,):
        raise RuntimeError(
            "thermal PyAEDT setup cache mismatch: "
            f"expected={(setup_name,)}, actual={wrapper_setups}"
        )
    actual_setup_name = str(getattr(setup, "name", "") or "").strip()
    if actual_setup_name != setup_name:
        raise RuntimeError(
            "thermal setup wrapper identity mismatch: "
            f"expected={setup_name!r}, actual={actual_setup_name or '<empty>'!r}"
        )
    props = getattr(setup, "props", None)
    if not hasattr(props, "get") or not _thermal_bool(props.get("Enabled")):
        raise RuntimeError("ThermalSetup is disabled or has no Enabled readback")
    enabled_source = "wrapper"
    native_enabled = None
    try:
        analysis_child = native_design.GetChildObject("Analysis")
        setup_child = analysis_child.GetChildObject(setup_name)
        get_enabled = getattr(setup_child, "GetPropValue", None)
        if callable(get_enabled):
            native_enabled = get_enabled("Enabled")
    except Exception:
        # GetSetups above is the authoritative native identity check on AEDT
        # versions that do not expose setup Enabled through the object tree.
        pass
    if native_enabled is not None:
        if not _thermal_bool(native_enabled):
            raise RuntimeError("native ThermalSetup Enabled readback is false")
        enabled_source = "native+wrapper"

    from module.aedt_pool_adapter import pooled_backend_enabled
    if not pooled_backend_enabled():
        running = _thermal_running_state(sim, ipk)
        if running is not False:
            raise RuntimeError(
                f"AEDT reports an overlapping simulation: {running!r}"
            )
    return {
        "project": actual_project,
        "design": design_name,
        "design_type": design_type,
        "setups": list(setups),
        "wrapper_setups": list(wrapper_setups),
        "enabled": True,
        "enabled_source": enabled_source,
        "native_ipk": native_ipk,
    }


def _bounded_thermal_messages(sim, ipk, limit=12, char_limit=2048):
    """Capture a bounded AEDT message tail without letting cleanup mask the solve."""
    try:
        desktop = _thermal_desktop_handle(sim, ipk)
        get_messages = getattr(desktop, "GetMessages", None)
        if not callable(get_messages):
            return []
        project_name = str(getattr(sim, "PROJECT_NAME", "") or "")
        design_name = str(getattr(ipk, "design_name", "") or _THERMAL_DESIGN_NAME)
        messages = list(get_messages(project_name, design_name, 0) or [])[-int(limit):]
    except Exception as exc:
        return [f"message-capture-error:{type(exc).__name__}:{str(exc)[:256]}"]
    bounded = []
    remaining = max(0, int(char_limit))
    for message in messages:
        value = str(message).replace("\r", " ").replace("\n", " ")[:512]
        if remaining <= 0:
            break
        value = value[:remaining]
        bounded.append(value)
        remaining -= len(value)
    return bounded


def _bounded_thermal_model_context(ipk, operation_limit=24):
    """Capture bounded mesh/object context for solve-start pilot diagnostics."""
    context = {"object_count": None, "model_bounds": [], "mesh_operations": []}
    try:
        modeler = getattr(ipk, "modeler", None)
        names = list(getattr(modeler, "object_names", []) or [])
        context["object_count"] = len(names)
        bounds = getattr(modeler, "obounding_box", None)
        if bounds is not None:
            context["model_bounds"] = [str(value)[:64] for value in list(bounds)[:6]]
    except Exception as exc:
        context["model_context_error"] = f"{type(exc).__name__}: {str(exc)[:256]}"
    try:
        mesh = getattr(ipk, "mesh", None)
        operations = list(getattr(mesh, "meshoperations", []) or [])[:int(operation_limit)]
        for operation in operations:
            props = getattr(operation, "props", None)
            get_prop = props.get if hasattr(props, "get") else lambda _key, default=None: default
            objects = get_prop("Objects", get_prop("Mesh Object(s)", []))
            if isinstance(objects, str):
                object_count = 1
            else:
                try:
                    object_count = len(list(objects or []))
                except TypeError:
                    object_count = None
            context["mesh_operations"].append({
                "name": str(getattr(operation, "name", ""))[:128],
                "level": str(get_prop("Level", ""))[:64],
                "object_count": object_count,
            })
    except Exception as exc:
        context["mesh_context_error"] = f"{type(exc).__name__}: {str(exc)[:256]}"
    return context


def _poll_thermal_dispatch_evidence(
    sim, ipk, setup, monitor_snapshot, timeout_s=30.0, poll_s=2.0,
    clock=time.monotonic, sleeper=time.sleep,
):
    """Poll monitor evidence and native running state for a bounded grace period."""
    deadline = clock() + max(0.0, float(timeout_s))
    convergence = None
    running = None
    running_error = ""
    while True:
        convergence = _thermal_convergence_telemetry(
            sim, ipk, setup, attempts=1, monitor_snapshot=monitor_snapshot
        )
        try:
            running = _thermal_running_state(sim, ipk)
            running_error = ""
        except Exception as exc:
            running = None
            running_error = f"{type(exc).__name__}: {str(exc)[:512]}"
        if convergence["thermal_convergence_reason"] in {
            "converged", "residual_threshold",
        }:
            break
        now = clock()
        if now >= deadline:
            break
        sleeper(min(max(0.0, float(poll_s)), max(0.0, deadline - now)))
    return convergence, running, running_error


def _thermal_forensic_json(attempts, convergence):
    """Serialize the bounded solve-start evidence used for retry decisions."""
    payload = {
        "schema": "thermal-dispatch-forensic-v1",
        "attempts": attempts[:2],
        "final_convergence": {
            "available": convergence.get("thermal_convergence_available", 0),
            "converged": convergence.get("thermal_converged", 0),
            "reason": str(convergence.get("thermal_convergence_reason", ""))[:128],
            "monitor_file": str(convergence.get("thermal_monitor_file", ""))[:256],
        },
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _solve_exact_thermal_setup(
    sim, ipk, setup, setup_name=_THERMAL_SETUP_NAME,
    monitor_grace_s=30.0, poll_s=2.0,
    clock=time.monotonic, sleeper=time.sleep,
):
    """Dispatch only ThermalSetup, with one evidence-gated startup retry."""
    attempts = []
    convergence = None
    previous_snapshot = None
    for requested_attempt in range(1, 3):
        preflight = _prepare_thermal_dispatch(
            sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
            setup_name=setup_name,
        )

        # Close the small gap between the first grace poll and a permitted retry.
        # A delayed first-attempt monitor is authoritative and prevents dispatch 2.
        if requested_attempt == 2 and previous_snapshot is not None:
            late = _thermal_convergence_telemetry(
                sim, ipk, setup, attempts=1, monitor_snapshot=previous_snapshot
            )
            if late["thermal_convergence_reason"] != "monitor_missing":
                convergence = late
                if attempts:
                    attempts[-1]["monitor_reason"] = late["thermal_convergence_reason"]
                    attempts[-1]["monitor_file"] = str(
                        late.get("thermal_monitor_file", "")
                    )[:256]
                    attempts[-1]["late_after_grace"] = True
                break

        monitor_snapshot = _snapshot_thermal_monitors(sim, ipk)
        previous_snapshot = monitor_snapshot
        native_ipk = preflight.pop("native_ipk")
        started = clock()
        status = "success"
        returned = None
        exception_type = ""
        exception_message = ""
        from module.aedt_pool_adapter import pooled_backend_enabled

        pooled_backend = pooled_backend_enabled()
        try:
            analyze_kwargs = {"setup": setup_name, "blocking": True}
            if not pooled_backend:
                analyze_kwargs["cores"] = sim.NUM_CORE
            # Passing ``cores=`` asks PyAEDT to rewrite and later restore the
            # Desktop-global Icepak DSO registry.  A pooled Desktop can have a
            # sibling MFT/IPMSM project, so it must reuse the host-owned,
            # read-back-verified profile instead.
            if pooled_backend:
                # Set uncertainty at the last possible moment. Geometry,
                # materials, mesh, and setup failures above are project-local
                # script errors and must not quarantine a healthy shared host.
                sim.solver_may_be_running = True
            returned = native_ipk.analyze(**analyze_kwargs)
            if pooled_backend:
                # blocking=True returned normally, so this project's solver is
                # no longer an unknown in-flight native operation. PyAEDT also
                # returns False when native Analyze raised internally, so that
                # value remains uncertain until the evidence poll proves idle.
                if returned is not False:
                    sim.solver_may_be_running = False
            if returned is False:
                status = "false"
                logging.warning(
                    "[thermal] exact ThermalSetup dispatch returned False; "
                    "polling native monitor evidence before any retry."
                )
        except Exception as exc:
            status = "exception"
            exception_type = type(exc).__name__
            exception_message = str(exc)[:512]
            logging.exception("[thermal] exact ThermalSetup dispatch failed: %s", exc)

        messages = _bounded_thermal_messages(sim, ipk) if status != "success" else []
        convergence, running, running_error = _poll_thermal_dispatch_evidence(
            sim, ipk, setup, monitor_snapshot,
            timeout_s=monitor_grace_s, poll_s=poll_s,
            clock=clock, sleeper=sleeper,
        )
        reason = convergence["thermal_convergence_reason"]
        if pooled_backend and running is False:
            # Only an exact project-scoped idle result proves that a False or
            # exceptional native dispatch did not leave work in flight. A
            # monitor can prove startup/progress, but not that the solver
            # process has stopped.
            sim.solver_may_be_running = False
        if status != "success" or convergence["thermal_convergence_reason"] != "converged":
            preflight["model_context"] = _bounded_thermal_model_context(ipk)
            if not messages:
                messages = _bounded_thermal_messages(sim, ipk)
        try:
            sim.save_project()
        except Exception:
            pass
        attempts.append({
            "attempt": len(attempts) + 1,
            "dispatch_status": status,
            "return_type": type(returned).__name__ if status != "exception" else "",
            "exception_type": exception_type,
            "exception_message": exception_message,
            "elapsed_s": round(max(0.0, clock() - started), 3),
            "native_running": running,
            "running_state_error": running_error,
            "monitor_reason": convergence["thermal_convergence_reason"],
            "monitor_file": str(convergence.get("thermal_monitor_file", ""))[:256],
            "aedt_messages": messages,
            "identity": preflight,
        })

        if reason in {"converged", "residual_threshold", "monitor_malformed"}:
            break
        retryable = status in {"false", "exception"} \
            and reason == "monitor_missing" and running is False
        if requested_attempt == 1 and retryable:
            logging.warning(
                "[thermal] exact ThermalSetup startup produced no fresh monitor; "
                "reacquiring the exact design/setup and retrying once."
            )
            continue
        break

    if convergence is None:
        raise RuntimeError("thermal convergence telemetry was not collected")
    final_attempt = attempts[-1] if attempts else {
        "dispatch_status": "not-dispatched",
        "exception_type": "",
        "exception_message": "",
    }
    forensic_json = _thermal_forensic_json(attempts, convergence)
    logging.warning("[thermal] dispatch forensic: %s", forensic_json)
    if pooled_backend and bool(getattr(sim, "solver_may_be_running", False)):
        # Never flow into normal project close/release with an uncertain native
        # Icepak operation. The outer pooled failure path reports this as a
        # solver fault and lets the session host quarantine/recycle safely.
        raise RuntimeError(
            "pooled thermal native solve state is uncertain: "
            + forensic_json[:4000]
        )
    return {
        "convergence": convergence,
        "solve_attempts": len(attempts),
        "analyze_call_ok": final_attempt["dispatch_status"] == "success",
        "analyze_return_false": any(
            item["dispatch_status"] == "false" for item in attempts
        ),
        "dispatch_status": final_attempt["dispatch_status"],
        "dispatch_exception_type": final_attempt["exception_type"],
        "dispatch_exception_message": final_attempt["exception_message"],
        "forensic_json": forensic_json,
    }


def _split_retained(ipk, objects, plane, sides):
    """Split geometry and require at least one retained input object to remain live."""
    if not objects:
        raise RuntimeError(f"cannot split an empty thermal geometry on {plane}")
    result = ipk.modeler.split(assignment=objects, plane=plane, sides=sides)
    if not result:
        raise RuntimeError(f"thermal geometry split failed on {plane} ({sides})")
    existing = set(ipk.modeler.object_names)
    alive = [obj for obj in objects if obj.name in existing]
    if not alive:
        raise RuntimeError(f"thermal geometry split on {plane} retained no live objects")
    return alive


# ---------------------------------------------------------------------------
# 손실 환산 (대칭 EM -> 풀모델 실제값)
# ---------------------------------------------------------------------------

def _sym_factor(spans_x0, spans_y0, spans_z0):
    """
    대칭 모델 적분값 -> 실제값 환산 계수.
    검증 실험: 대칭 모델의 체적 적분 = 실제값 x 4 / 2^c (c = 오브젝트를 지나는 절단평면 수)
      c=3 (중심 권선 턴, 중앙 코어) -> 대칭값이 실제의 1/2  -> x2
      c=2 (측면 권선 링, 바깥 코어/플레이트) -> 대칭값 = 실제값 -> x1
    따라서 실제값 = 대칭값 x 2^c / 4
    """
    cuts = int(spans_x0) + int(spans_y0) + int(spans_z0)
    return (2 ** cuts) / 4.0


class LossAllocator:
    """
    EM loss 디자인의 실물 기준 손실(loss_map_phys)을 열해석 오브젝트에 배분.
    - full thermal: 오브젝트당 실물값 그대로 (미러 오브젝트는 대응 키 폴백)
    - eighth thermal: 실물값 x 보유 체적 분율(1/2^c)
    """

    def __init__(self, sim, eighth=False, mode=None):
        self.sim = sim
        self.df = sim.df_plus
        physical_loss_map = getattr(sim, "loss_map_phys", None)
        contract_version = ""
        if "core_material_contract_version" in self.df.columns:
            contract_version = str(
                self.df["core_material_contract_version"].iloc[0] or ""
            ).strip()
        physics_revision = (
            str(self.df["physics_data_revision"].iloc[0] or "").strip()
            if "physics_data_revision" in self.df.columns else ""
        )
        native_contract = physics_revision == PHYSICS_DATA_REVISION
        if native_contract and not physical_loss_map:
            raise RuntimeError(
                "native-lamination physics revision requires loss_map_phys; "
                "raw EM losses cannot be injected into Icepak"
            )
        self.loss_map = physical_loss_map or getattr(sim, "loss_map", {})
        self.core_loss_contract_version = contract_version or "legacy_unspecified"
        self.core_loss_source = (
            "aedt_native_lamination_loss_attested_then_margin_adjusted"
            if native_contract else "legacy_loss_map_fallback_allowed"
        )
        self.native_core_contract = native_contract
        self.mode = mode or ("eighth" if eighth else "full")
        self.eighth = self.mode == "eighth"

    def _get(self, key):
        v = self.loss_map.get(key)
        alt = None
        if v is None:
            # 미러 오브젝트 폴백 (대칭 EM에 없는 y<0 쪽 등): 대응 오브젝트 키로 대체
            alt = key.replace("Rx_side2", "Rx_side").replace("_n", "_p")
            v = self.loss_map.get(alt)
        if v is None:
            alias = f" (alias {alt})" if alt and alt != key else ""
            raise KeyError(f"required thermal loss key missing: {key}{alias}")
        return float(v)

    def _retained_fraction(self, obj_name):
        if self.mode == "full":
            return 1.0
        from module.input_parameter_260706 import sym_cut_count
        c = sym_cut_count(obj_name, self.df)
        if self.mode == "quarter":
            c = max(c - 1, 0)  # z절단 없음 (모든 오브젝트가 z=0 스팬이므로 -1)
        return 1.0 / (2 ** c)

    def turn_loss(self, expr_key, obj_name=None, **_legacy):
        """개별 턴/오브젝트에 주입할 손실 [W] (열모델 보유 체적 기준)"""
        name = obj_name or expr_key.replace("P_turn_", "").replace("P_", "")
        return self._get(expr_key) * self._retained_fraction(name)

    def group_loss(self, expr_key, obj_name=None, **_legacy):
        name = obj_name or expr_key.replace("P_", "").replace("_group", "")
        return self._get(expr_key) * self._retained_fraction(name)


# ---------------------------------------------------------------------------
# 재질
# ---------------------------------------------------------------------------

_CORE_THERMAL_MODEL_LEGACY = "isotropic_legacy"
_CORE_THERMAL_MODEL_WOUND = "anisotropic_wound_rule_of_mixtures_v1"
_CORE_THERMAL_MATERIAL_LEGACY = "core_amorphous_thermal"
_CORE_THERMAL_MATERIAL_LEG = "core_amorphous_thermal_leg"
_CORE_THERMAL_MATERIAL_YOKE = "core_amorphous_thermal_yoke"


def _derive_wound_core_conductivity(
        lamination_factor, k_alloy, k_interlayer):
    """Return in-plane and through-stack conductivity in W/mK."""
    kf = float(lamination_factor)
    alloy = float(k_alloy)
    interlayer = float(k_interlayer)
    if not math.isfinite(kf) or not 0.0 <= kf <= 1.0:
        raise ValueError(
            f"core_lamination_factor must be finite and in [0, 1], got {kf}"
        )
    for name, value in (
            ("core_k_alloy", alloy),
            ("core_k_interlayer", interlayer)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    k_inplane = kf * alloy + (1.0 - kf) * interlayer
    k_throughstack = 1.0 / (
        kf / alloy + (1.0 - kf) / interlayer
    )
    return k_inplane, k_throughstack


def _core_thermal_conductivity_contract(df):
    """Resolve the selected core model and both directional conductivities."""
    enabled_value = float(df["core_k_anisotropic"].iloc[0])
    if (
            not math.isfinite(enabled_value)
            or not enabled_value.is_integer()
            or int(enabled_value) not in (0, 1)):
        raise ValueError(
            f"core_k_anisotropic must be 0 or 1, got {enabled_value}"
        )
    enabled = int(enabled_value)
    if enabled == 0:
        legacy = float(df["core_k_thermal"].iloc[0])
        return {
            "anisotropic": False,
            "thermal_core_conductivity_model": _CORE_THERMAL_MODEL_LEGACY,
            "thermal_core_k_inplane": legacy,
            "thermal_core_k_throughstack": legacy,
        }
    k_inplane, k_throughstack = _derive_wound_core_conductivity(
        df["core_lamination_factor"].iloc[0],
        df["core_k_alloy"].iloc[0],
        df["core_k_interlayer"].iloc[0],
    )
    return {
        "anisotropic": True,
        "thermal_core_conductivity_model": _CORE_THERMAL_MODEL_WOUND,
        "thermal_core_k_inplane": k_inplane,
        "thermal_core_k_throughstack": k_throughstack,
    }


def _core_thermal_material_for_piece(piece_name):
    """Map one segmented core name to its wound-ribbon material orientation."""
    name = str(piece_name)
    if re.fullmatch(r"core_\d+_leg_(?:left|center|right)", name):
        return _CORE_THERMAL_MATERIAL_LEG
    if re.fullmatch(r"core_\d+_yoke_(?:top|bottom)", name):
        return _CORE_THERMAL_MATERIAL_YOKE
    raise ValueError(f"unrecognized segmented core piece name: {name!r}")


def _create_thermal_materials(ipk, df):
    """코어 등가재질 + 권선 균질화 이방성 재질 2종 생성"""
    core_contract = _core_thermal_conductivity_contract(df)
    k_ins = float(df["k_ins"].iloc[0])
    cw2 = float(df["cw2"].iloc[0])
    gap2 = float(df["gap2"].iloc[0])

    ff = cw2 / (cw2 + gap2)              # foil 채움율
    k_in = ff * 385.0 + (1 - ff) * k_ins        # foil 면내 방향 (병렬)
    k_th = 1.0 / (ff / 385.0 + (1 - ff) / k_ins)  # 적층 방향 (직렬)

    mats = ipk.materials

    if core_contract["anisotropic"]:
        core_materials = (
            (
                _CORE_THERMAL_MATERIAL_LEG,
                [
                    core_contract["thermal_core_k_throughstack"],
                    core_contract["thermal_core_k_inplane"],
                    core_contract["thermal_core_k_inplane"],
                ],
            ),
            (
                _CORE_THERMAL_MATERIAL_YOKE,
                [
                    core_contract["thermal_core_k_inplane"],
                    core_contract["thermal_core_k_inplane"],
                    core_contract["thermal_core_k_throughstack"],
                ],
            ),
        )
        for material_name, conductivity in core_materials:
            if material_name not in mats.material_keys:
                m = mats.add_material(material_name)
                # A three-element Python list is PyAEDT's supported
                # AnisoProperty form; component order is global X/Y/Z.
                m.thermal_conductivity = conductivity
                m.mass_density = 7180
                m.specific_heat = 540
    elif _CORE_THERMAL_MATERIAL_LEGACY not in mats.material_keys:
        # Preserve the pre-extension scalar material path exactly when opted out.
        m = mats.add_material(_CORE_THERMAL_MATERIAL_LEGACY)
        m.thermal_conductivity = core_contract["thermal_core_k_inplane"]
        m.mass_density = 7180
        m.specific_heat = 540

    # x측 블록: 적층(through) 방향 = x
    if "winding_homog_x" not in mats.material_keys:
        m = mats.add_material("winding_homog_x")
        m.thermal_conductivity = [k_th, k_in, k_in]
        m.mass_density = 8900 * ff
        m.specific_heat = 385

    # y측 블록: 적층(through) 방향 = y
    if "winding_homog_y" not in mats.material_keys:
        m = mats.add_material("winding_homog_y")
        m.thermal_conductivity = [k_in, k_th, k_in]
        m.mass_density = 8900 * ff
        m.specific_heat = 385

    if "thermal_pad" not in mats.material_keys:
        m = mats.add_material("thermal_pad")
        m.conductivity = 0
        m.thermal_conductivity = 0.2

    return k_in, k_th


# ---------------------------------------------------------------------------
# 지오메트리
# ---------------------------------------------------------------------------

def _rx_layout(df, prefix):
    """Rx 권선 그룹의 배치 정보 (턴 중심 위치, 도체 폭 등)"""
    cw2 = float(df["cw2"].iloc[0])
    gap2 = float(df["gap2"].iloc[0])
    if prefix == "main":
        N = int(df["N2_main"].iloc[0])
        slx = float(df["sl2_main_x"].iloc[0])
        sly = float(df["sl2_main_y"].iloc[0])
    else:
        N = int(df["N2_side"].iloc[0])
        slx = float(df["sl2_side_x"].iloc[0])
        sly = float(df["sl2_side_y"].iloc[0])
    gaps = [gap2] * (N - 1)
    x_pos = compute_layer_positions(slx / 2, cw2, gaps)
    y_pos = compute_layer_positions(sly / 2, cw2, gaps)
    return N, cw2, x_pos, y_pos


def _rx_side_face_x_ranges(df, offset_x):
    """Return transformer-outward/inward x ranges for one side winding.

    ``_rx_layout`` describes a winding about its local leg centre with two
    x-directed packs at ``+/-x``.  For the negative-x side leg the positive
    local pack faces the transformer centre; for the positive-x mirror the
    negative local pack does.  Deriving the direction from ``offset_x`` keeps
    the same rule valid for symmetry and full models.
    """
    centre = float(offset_x)
    if not math.isfinite(centre) or math.isclose(centre, 0.0, abs_tol=1e-12):
        raise ValueError("Rx-side probe requires a non-zero side-leg offset")
    _n, cw, x_pos, _y_pos = _rx_layout(df, "side")
    x_in = x_pos[0] - cw / 2.0
    x_out = x_pos[-1] + cw / 2.0
    if not (math.isfinite(x_in) and math.isfinite(x_out) and 0 <= x_in < x_out):
        raise ValueError(
            f"invalid Rx-side probe radial bounds: inner={x_in}, outer={x_out}"
        )
    inward_sign = -1.0 if centre > 0 else 1.0

    def _ordered(sign):
        return tuple(sorted((centre + sign * x_in, centre + sign * x_out)))

    return {
        "outward": _ordered(-inward_sign),
        "inward": _ordered(inward_sign),
    }


def _partition_rx_turns(windings, n_explicit):
    """Return retained explicit turns and the turns replaced by thermal blocks."""
    windings = list(windings)
    # A one-turn group contains no inter-turn insulation to homogenize. Keeping
    # the exact copper turn also avoids zero-cell omission of sub-millimetre
    # blocks in the Icepak cut-cell mesh.
    if len(windings) == 1:
        return windings, []
    count = int(n_explicit)
    if count < 0 or 2 * count >= len(windings):
        return windings, []
    if count == 0:
        return [], windings
    return windings[:count] + windings[-count:], windings[count:-count]


def _build_homog_blocks(ipk, df, prefix, name, offset_x, height):
    """Replace the selected Rx pack span with four anisotropic solid blocks."""
    n_exp = int(df["n_explicit_turns"].iloc[0])
    N, cw, x_pos, y_pos = _rx_layout(df, prefix)

    if n_exp == 0:
        # Full-pack homogenization includes every foil and every inter-turn gap.
        bx_in = x_pos[0] - cw / 2
        bx_out = x_pos[-1] + cw / 2
        by_in = y_pos[0] - cw / 2
        by_out = y_pos[-1] + cw / 2
    else:
        # Middle region between the retained inner and outer explicit turns.
        bx_in = x_pos[n_exp - 1] + cw / 2
        bx_out = x_pos[N - n_exp] - cw / 2
        by_in = y_pos[n_exp - 1] + cw / 2
        by_out = y_pos[N - n_exp] - cw / 2

    z0 = -height / 2
    blocks = []

    # x측 블록 2개 (y 전체 폭, 적층방향 x)
    for sign, tag in [(1, "xp"), (-1, "xn")]:
        obj = ipk.modeler.create_box(
            origin=[f"{(bx_in if sign > 0 else -bx_out) + offset_x}mm", f"{-by_out}mm", f"{z0}mm"],
            sizes=[f"{bx_out - bx_in}mm", f"{2 * by_out}mm", f"{height}mm"],
            name=f"{name}_block_{tag}",
            material="winding_homog_x"
        )
        blocks.append(obj)

    # y측 블록 2개 (x는 안쪽 경계 사이, 적층방향 y)
    for sign, tag in [(1, "yp"), (-1, "yn")]:
        obj = ipk.modeler.create_box(
            origin=[f"{-bx_in + offset_x}mm", f"{(by_in if sign > 0 else -by_out)}mm", f"{z0}mm"],
            sizes=[f"{2 * bx_in}mm", f"{by_out - by_in}mm", f"{height}mm"],
            name=f"{name}_block_{tag}",
            material="winding_homog_y"
        )
        blocks.append(obj)

    return blocks


def _build_rx_group(ipk, df, prefix, name, offset_x, n_explicit, height):
    """Build the final Rx thermal representation without disposable geometry."""
    turn_count = int(df[f"N2_{prefix}"].iloc[0])
    n_explicit = int(n_explicit)

    # The standard campaign retains zero explicit turns. Creating every foil
    # polyline only to delete it before installing these same complete-pack
    # blocks adds no final geometry or physics. A one-turn pack remains exact
    # copper because it has no inter-turn insulation to homogenize.
    if n_explicit == 0 and turn_count > 1:
        return [], _build_homog_blocks(
            ipk, df, prefix, name, offset_x, height,
        )

    windings, _, _, _, _, _ = create_coil(
        design=ipk, name=name,
        window_height=df["nwh2"].iloc[0],
        window_length=df[f"nwl2_{prefix}"].iloc[0],
        window_layer=turn_count, N_input=1,
        width_fill_factor=df[f"wff2_{prefix}"].iloc[0],
        space_length=df[f"sl2_{prefix}_x"].iloc[0],
        space_width=df[f"sl2_{prefix}_y"].iloc[0],
        shape="rectangle", offset=[offset_x, 0, 0], color=[10, 10, 255],
        round_corner=False,
    )
    explicit, middle = _partition_rx_turns(windings, n_explicit)
    if not middle:
        return explicit, []

    names = [w.name for w in middle]
    ipk.modeler.delete(names)
    survivors = [n for n in names if n in set(ipk.modeler.object_names)]
    for survivor in survivors:
        try:
            ipk.modeler.delete(survivor)
        except Exception:
            pass
    survivors = [n for n in names if n in set(ipk.modeler.object_names)]
    if survivors:
        raise RuntimeError(
            f"middle turn deletion failed for {len(survivors)} objects "
            f"({survivors[:3]}...) - aborting thermal build"
        )
    return explicit, _build_homog_blocks(
        ipk, df, prefix, name, offset_x, height,
    )


def _build_geometry(ipk, sim, eighth=False, mode=None):
    """열해석 지오메트리 생성. mode: full / quarter(x,y 분할) / eighth(x,y,z 분할)"""
    mode = mode or ("eighth" if eighth else "full")
    df = sim.df_plus
    n_exp = int(df["n_explicit_turns"].iloc[0])
    l1 = float(df["l1"].iloc[0])
    l2 = float(df["l2"].iloc[0])
    nwh1 = float(df["nwh1"].iloc[0])
    nwh2 = float(df["nwh2"].iloc[0])

    objs = {}

    # ---- 코어 + 콜드플레이트 + 패드 ----
    n_group = int(df["n_core_group"].iloc[0])
    plate_on = int(df["core_plate_on"].iloc[0]) != 0
    pad_on = float(df["core_plate_pad_t"].iloc[0]) > 0
    core_contract = _core_thermal_conductivity_contract(df)
    if core_contract["anisotropic"]:
        core_objs, plate_objs, pad_objs = create_core(
            design=ipk, name="core",
            core_material=_CORE_THERMAL_MATERIAL_LEGACY,
            n_group=n_group, plate_material="aluminum",
            pad_material="thermal_pad", plate_on=plate_on, pad_on=pad_on,
            plate_color=[144, 190, 144], pad_color=[200, 160, 200],
            segmented_lamination=True,
            core_material_leg=_core_thermal_material_for_piece(
                "core_1_leg_center"
            ),
            core_material_yoke=_core_thermal_material_for_piece(
                "core_1_yoke_top"
            ),
        )
    else:
        core_objs, plate_objs, pad_objs = create_core(
            design=ipk, name="core",
            core_material=_CORE_THERMAL_MATERIAL_LEGACY,
            n_group=n_group, plate_material="aluminum",
            pad_material="thermal_pad", plate_on=plate_on, pad_on=pad_on,
            plate_color=[144, 190, 144], pad_color=[200, 160, 200]
        )
    objs["core"] = core_objs
    objs["core_plates"] = plate_objs
    objs["core_pads"] = pad_objs

    # ---- Tx (전 턴 explicit, 직각) ----
    tx_y_gaps, tx_slots = get_tx_y_gaps(df)
    tx_windings, _, tx_cw, _, _, _ = create_coil(
        design=ipk, name="Tx_main",
        window_height=df["nwh1"].iloc[0], window_length=df["nwl1_main"].iloc[0],
        window_layer=df["N1_main"].iloc[0], N_input=1,
        width_fill_factor=df["wff1_main"].iloc[0],
        space_length=df["sl1_main_x"].iloc[0], space_width=df["sl1_main_y"].iloc[0],
        shape="rectangle", offset=[0, 0, 0], color=[255, 10, 10],
        y_slot_gaps=tx_y_gaps, round_corner=False
    )
    objs["Tx"] = tx_windings

    # ---- 권선 냉각 플레이트 ----
    wcp_on = int(df["wcp_on"].iloc[0]) != 0
    if wcp_on and len(tx_slots) > 0:
        wcp_plates, wcp_pads = create_winding_cooling_plates(
            design=ipk, name="Tx_main_wcp",
            space_width=df["sl1_main_y"].iloc[0], coil_width=tx_cw,
            y_gaps=tx_y_gaps, slot_indices=tx_slots,
            wcp_len_x=float(df["wcp_len_x"].iloc[0]), wcp_t=float(df["wcp_t"].iloc[0]),
            pad_t=float(df["wcp_pad_t"].iloc[0]), height=nwh1,
            plate_material="aluminum", pad_material="thermal_pad",
            plate_color=[144, 190, 144], pad_color=[200, 160, 200]
        )
        objs["wcp_plates"] = wcp_plates
        objs["wcp_pads"] = wcp_pads
    else:
        objs["wcp_plates"] = []
        objs["wcp_pads"] = []

    # ---- Rx 하이브리드 (main 1조 + side 2조) ----
    # n_explicit_turns = -1 이면 전 턴 explicit (블록 없음, 균질화 가정 제거).
    # 2*n_exp >= N 인 경우도 전 턴 explicit으로 처리 (중복/퇴화 블록 방지)
    objs["Rx_main_explicit"], objs["Rx_main_blocks"] = _build_rx_group(
        ipk, df, "main", "Rx_main", 0.0, n_exp, nwh2,
    )

    objs["Rx_side_explicit"] = []
    objs["Rx_side_blocks"] = []
    objs["Rx_side2_explicit"] = []
    objs["Rx_side2_blocks"] = []
    if int(df["N2_side"].iloc[0]) > 0:
        off = l1 + l2 + l1 / 2
        objs["Rx_side_explicit"], objs["Rx_side_blocks"] = _build_rx_group(
            ipk, df, "side", "Rx_side", -off, n_exp, nwh2,
        )
        if mode == "full":
            # 대칭 모드에서는 +x 측 링이 어차피 절단 제거되므로 생성 생략 (모델링 시간 절약)
            objs["Rx_side2_explicit"], objs["Rx_side2_blocks"] = _build_rx_group(
                ipk, df, "side", "Rx_side2", +off, n_exp, nwh2,
            )

    if mode in ("eighth", "quarter"):
        # EM 대칭 모델과 동일 옥탄트 (x<=0, y>=0[, z>=0])
        all_objs = []
        for grp in objs.values():
            all_objs.extend(grp)
        if mode == "eighth":
            all_objs = _split_retained(ipk, all_objs, plane="XY", sides="PositiveOnly")
        all_objs = _split_retained(ipk, all_objs, plane="XZ", sides="PositiveOnly")
        _split_retained(ipk, all_objs, plane="YZ", sides="NegativeOnly")

    existing = set(ipk.modeler.object_names)
    for key in list(objs.keys()):
        objs[key] = [o for o in objs[key] if o.name in existing]

    _require_thermal_geometry(
        objs,
        mode,
        int(df["N2_side"].iloc[0]),
        require_core_plates=plate_on,
        require_core_pads=plate_on and pad_on,
        require_wcp_plates=wcp_on,
        require_wcp_pads=wcp_on and float(df["wcp_pad_t"].iloc[0]) > 0,
    )

    return objs


def _core_probe_y_positions(df, mode):
    """Return core-group mid-depth planes that contain core, never a plate.

    With an odd group count the middle core is centered at y=0. With an even
    count the stack centered at y=0 is a cold-plate assembly, so the adjacent
    core centers are at +/-(d + stack_t)/2. Symmetry models retain only y>=0.
    """
    n_group = int(df["n_core_group"].iloc[0])
    if n_group <= 0:
        raise ValueError(f"n_core_group must be positive, got {n_group}")
    w1 = float(df["w1"].iloc[0])
    plate_t = float(df["core_plate_t"].iloc[0])
    pad_t = float(df["core_plate_pad_t"].iloc[0])
    stack_t = plate_t + 2.0 * pad_t
    core_depth = (w1 - (n_group + 1) * stack_t) / n_group
    if not math.isfinite(core_depth) or core_depth <= 0:
        raise ValueError(
            f"invalid core depth for probes: n={n_group}, w1={w1}, "
            f"stack_t={stack_t}, depth={core_depth}"
        )
    if n_group % 2:
        return [0.0]
    offset = 0.5 * (core_depth + stack_t)
    return [offset] if mode in ("eighth", "quarter") else [-offset, offset]


def _create_probe_sheets(ipk, df, objs, eighth=False, mode=None):
    """
    회귀학습용 온도 프로브 시트 생성 (비모델 - 메시에 영향 없음).

    체적 max 온도는 메시 스파이크에 취약하므로, 대칭면 위치(x=0/y=0 평면)에
    권선 단면 크기의 시트를 만들어 그 위에서 Temp를 추출한다.
    위치가 파라미터만으로 결정되므로 모든 샘플에서 기하학적으로 동일 -> 데이터 일관성.
    팬(+y -> -y) 기준 풍하측(y-) 단면을 잡아 핫스팟 쪽을 캡처한다.
    """
    l1 = float(df["l1"].iloc[0])
    h1 = float(df["h1"].iloc[0])
    nwh1 = float(df["nwh1"].iloc[0])
    nwh2 = float(df["nwh2"].iloc[0])
    l2 = float(df["l2"].iloc[0])
    cw1 = float(df["cw1"].iloc[0])

    sheets = ProbeSheetCollection()

    def _sheet(name, orientation, origin, sizes, *, required=True):
        sheets.expect(name)
        try:
            orientation, origin, sizes = validate_probe_rectangle(
                name, orientation, origin, sizes
            )
            obj = ipk.modeler.create_rectangle(
                orientation=orientation, origin=[f"{v}mm" for v in origin],
                sizes=[f"{v}mm" for v in sizes], name=name
            )
            if isinstance(obj, bool) or obj is None:
                raise RuntimeError(f"create_rectangle returned {obj!r}")
            if str(getattr(obj, "name", "")) != name:
                raise RuntimeError(
                    f"created object name is {getattr(obj, 'name', None)!r}"
                )
            if getattr(obj, "is3d", None) is not False:
                raise RuntimeError(
                    f"created object is not a sheet (is3d={getattr(obj, 'is3d', None)!r})"
                )
            obj.model = False
            sheets.append(obj)
            return obj
        except Exception as e:
            stage = "geometry" if isinstance(e, ValueError) else "creation"
            reason = "invalid_rectangle" if stage == "geometry" else "sheet_creation_failed"
            sheets.record_failure(name, stage, reason, e)
            level = logging.error if required else logging.warning
            level("probe sheet %s failed before solve: %s", name, e)
            return None

    mode = mode or ("eighth" if eighth else "full")
    eighth = mode == "eighth"
    sym_xy = mode in ("eighth", "quarter")  # x<=0, y>=0 옥탄트 (quarter는 z 전체)

    # ---- Tx (1차): y측 단면 (x=0 평면) + x측 단면 (y=0 평면) ----
    tx_gaps, _ = get_tx_y_gaps(df)
    N1 = int(df["N1_main"].iloc[0])
    tx_x = compute_layer_positions(float(df["sl1_main_x"].iloc[0]) / 2, cw1, [float(df["gap1"].iloc[0])] * (N1 - 1))
    tx_y = compute_layer_positions(float(df["sl1_main_y"].iloc[0]) / 2, cw1, tx_gaps)
    zh = 0.48 * nwh1
    z_half_only = (mode == "eighth")  # quarter/full은 z 전체

    def _z_range(zh_):
        return (0.0, zh_) if z_half_only else (-zh_, zh_)

    # 이하 배치 로직에서 "eighth"는 x/y 옥탄트 배치를 의미하므로 quarter도 동일하게 취급
    eighth = sym_xy

    z0, z1 = _z_range(zh)
    # YZ 평면 시트: y측 런의 단면 (eighth: +y측 / full: -y 풍하측)
    # eighth 모드에서는 x=0이 region 경계면과 겹쳐 필드 평가가 실패하므로 1mm 안쪽(x<0)에 배치
    x_probe = -1.0 if eighth else 0.0
    y_start = (tx_y[0] - cw1 / 2) if eighth else -(tx_y[-1] + cw1 / 2)
    _sheet("Tprobe_Tx_leeward", "YZ", [x_probe, y_start, z0], [(tx_y[-1] - tx_y[0]) + cw1, z1 - z0])
    # XZ 평면 시트 (y=0): x- 런의 단면 (보유 옥탄트가 x<=0)
    # 주의: Y-법선("XZ"/"ZX") 사각형의 AEDT 치수 순서는 [z스팬, x스팬] (전치 버그 수정)
    _sheet(
        "Tprobe_Tx_side", "XZ", [-(tx_x[-1] + cw1 / 2), 0, z0],
        [z1 - z0, (tx_x[-1] - tx_x[0]) + cw1],
        required=int(df["N1_side"].iloc[0]) > 0,
    )

    # ---- Rx 그룹 공통 생성기 ----
    def _rx_probes(prefix, name, offset_x):
        N, cw, x_pos, y_pos = _rx_layout(df, prefix)
        zh2 = 0.48 * nwh2
        za, zb = _z_range(zh2)
        y_in, y_out = y_pos[0] - cw / 2, y_pos[-1] + cw / 2
        x_in, x_out = x_pos[0] - cw / 2, x_pos[-1] + cw / 2
        ys = y_in if eighth else -y_out
        # eighth: 링 중심(x=offset)이 0이면 region 경계와 겹침 -> 1mm 안쪽
        xs = offset_x if offset_x < 0 else (-1.0 if eighth else offset_x)
        _sheet(f"Tprobe_{name}_leeward", "YZ", [xs, ys, za], [y_out - y_in, zb - za])
        # y=0 평면: 바깥쪽(코어 중심에서 먼 쪽) 런 - 보유 옥탄트 x<=0 기준
        _sheet(f"Tprobe_{name}_side", "XZ", [offset_x - x_out, 0, za], [zb - za, x_out - x_in])

    _rx_probes("main", "Rx_main", 0.0)
    if int(df["N2_side"].iloc[0]) > 0:
        off = l1 + l2 + l1 / 2

        def _rx_side_probes(name, offset_x):
            """Probe both radial packs, classified relative to x=0."""
            _n, cw, _x_pos, y_pos = _rx_layout(df, "side")
            zh2 = 0.48 * nwh2
            za, zb = _z_range(zh2)
            y_in = y_pos[0] - cw / 2.0
            y_out = y_pos[-1] + cw / 2.0
            ys = y_in if eighth else -y_out
            xs = offset_x if offset_x < 0 else (
                -1.0 if eighth else offset_x
            )
            # Keep the airflow-leeward plane as an explicit diagnostic.  The
            # established Rx-side surrogate target is aggregated below from
            # the two transformer-relative radial faces instead.
            _sheet(
                f"Tprobe_{name}_flow_leeward", "YZ", [xs, ys, za],
                [y_out - y_in, zb - za],
            )
            ranges = _rx_side_face_x_ranges(df, offset_x)
            for relation, (x0, x1) in ranges.items():
                # ``side`` is the historical outward field name and remains
                # available byte-for-byte for the negative-x physical group.
                if relation == "outward":
                    probe_name = f"Tprobe_{name}_side"
                else:
                    # Use an explicit side-1 name so the raw left-face field
                    # cannot collide with the all-side inner aggregate.
                    physical_name = "Rx_side1" if name == "Rx_side" else name
                    probe_name = f"Tprobe_{physical_name}_inner"
                _sheet(
                    probe_name, "XZ", [x0, 0, za],
                    [zb - za, x1 - x0],
                    required=True,
                )

        _rx_side_probes("Rx_side", -off)
        if mode == "full":
            _rx_side_probes("Rx_side2", +off)

    # ---- 코어: 올바른 깊이 중심에서 center/side leg + top yoke ----
    zc = 0.48 * (h1 + 2 * l1)
    zca, zcb = _z_range(zc)
    # Probe both the central and outer side legs at a core mid-depth plane.
    # y=0 is valid only for odd core-group counts; with an even count it lies
    # inside the central cold-plate stack instead of inside core material.
    core_y_positions = _core_probe_y_positions(df, mode)
    center_x0 = -0.9 * l1
    center_x_span = 0.9 * l1 if eighth else 1.8 * l1
    side_x0 = -(2.0 * l1 + l2) + 0.05 * l1
    side_x_span = 0.9 * l1
    # The I plates contact the side and center legs, but the upper yoke across
    # the window between those two strips has no direct plate contact. Probe
    # that uncooled band explicitly instead of relying on a leg sheet to
    # happen to catch it.
    top_yoke_margin = 0.05 * l1
    top_yoke_x0 = -(l1 + l2) + top_yoke_margin
    top_yoke_x_span = l2 - 2.0 * top_yoke_margin
    top_yoke_z0 = h1 / 2.0 + 0.05 * l1
    top_yoke_z_span = 0.9 * l1
    for y_core in core_y_positions:
        if len(core_y_positions) == 1:
            suffix = ""
        else:
            suffix = "_neg" if y_core < 0 else "_pos"
        _sheet(
            f"Tprobe_core_center_leg{suffix}", "XZ",
            [center_x0, y_core, zca], [zcb - zca, center_x_span],
        )
        _sheet(
            f"Tprobe_core_side_leg{suffix}", "XZ",
            [side_x0, y_core, zca], [zcb - zca, side_x_span],
        )
        _sheet(
            f"Tprobe_core_top_yoke{suffix}", "XZ",
            [top_yoke_x0, y_core, top_yoke_z0],
            [top_yoke_z_span, top_yoke_x_span],
        )

    return sheets


# ---------------------------------------------------------------------------
# 손실 주입 / 경계조건
# ---------------------------------------------------------------------------


def _volume_weighted_powers(objects, total_power):
    """Distribute a finite non-negative group loss and preserve it exactly."""
    total = float(total_power)
    if not math.isfinite(total) or total < 0:
        raise ValueError(f"invalid group loss: {total_power}")
    objects = list(objects)
    if not objects:
        if math.isclose(total, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return []
        raise RuntimeError(f"cannot distribute {total}W without thermal blocks")
    volumes = [abs(float(obj.volume)) for obj in objects]
    if any(not math.isfinite(volume) or volume <= 0 for volume in volumes):
        raise RuntimeError(f"invalid thermal block volumes: {volumes}")
    volume_total = sum(volumes)
    powers = [total * volume / volume_total for volume in volumes]
    if not math.isclose(sum(powers), total, rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError(
            f"thermal block power distribution mismatch: {sum(powers)} != {total}"
        )
    return powers


def _assign_losses(ipk, sim, objs, eighth=False, mode=None):
    """실물 기준 손실(loss_map_phys)을 열모델 오브젝트에 주입.
    eighth 모드에서는 보유 체적 분율(1/2^c)이 LossAllocator에서 자동 적용된다."""
    df = sim.df_plus
    alloc = LossAllocator(sim, eighth=eighth, mode=mode)
    injected = {}
    rx_power_balance = []
    native_core_readbacks = []

    def _block(obj, watts, *, native_core_readback=False):
        w = max(float(watts), 0.0)
        if not math.isfinite(w):
            raise ValueError(f"non-finite thermal loss for {obj.name}: {w}")
        power = f"{w}W"
        boundary = ipk.assign_solid_block(obj.name, power)
        _require_boundary(
            boundary,
            f"solid block source for {obj.name}",
            {
                "Block Type": "Solid",
                "Objects": [obj.name],
                "Total Power": power,
            },
        )
        injected[obj.name] = w
        if native_core_readback:
            readback = _native_block_readback(boundary, obj)
            if not math.isclose(
                readback["power_w"], w, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"native Icepak block power mismatch for {obj.name}: "
                    f"{readback['power_w']} != {w}"
                )
            native_core_readbacks.append(readback)

    # Tx 턴별
    for w in objs["Tx"]:
        _block(w, alloc.turn_loss(f"P_turn_{w.name}", w.name))

    # Rx 그룹 공통: explicit 턴 + 중간 균질 블록 (그룹 총손실 - explicit, 체적 비례)
    def _rx_group(explicit_objs, blocks, group_key, name_hint):
        p_total = alloc.group_loss(group_key, name_hint)
        if len(explicit_objs) == 1 and not blocks:
            # The sole exact turn is the complete physical group. Production
            # n_explicit_turns=0 reports omit per-turn Rx expressions, so its
            # symmetry-adjusted group loss is also its exact turn loss.
            _block(explicit_objs[0], p_total)
            rx_power_balance.append({
                "group": group_key,
                "name_hint": name_hint,
                "expected_w": p_total,
                "assigned_w": p_total,
            })
            return
        p_exp_total = 0.0
        for w in explicit_objs:
            em_name = w.name.replace("Rx_side2", "Rx_side")
            p = alloc.turn_loss(f"P_turn_{em_name}", em_name)
            p_exp_total += p
            _block(w, p)
        p_mid = p_total - p_exp_total
        if p_mid < -max(1e-9, abs(p_total) * 1e-12):
            raise RuntimeError(
                f"explicit Rx loss exceeds group loss for {group_key}: "
                f"{p_exp_total} > {p_total}"
            )
        block_powers = _volume_weighted_powers(blocks, max(p_mid, 0.0))
        for block, power in zip(blocks, block_powers):
            _block(block, power)
        assigned = p_exp_total + sum(block_powers)
        if not math.isclose(assigned, p_total, rel_tol=1e-12, abs_tol=1e-9):
            raise RuntimeError(
                f"Rx thermal power balance failed for {group_key}: {assigned} != {p_total}"
            )
        rx_power_balance.append({
            "group": group_key,
            "name_hint": name_hint,
            "expected_w": p_total,
            "assigned_w": assigned,
        })

    _rx_group(objs["Rx_main_explicit"], objs["Rx_main_blocks"], "P_Rx_main_group", "Rx_main_0_0")
    if objs["Rx_side_explicit"] or objs["Rx_side_blocks"]:
        _rx_group(objs["Rx_side_explicit"], objs["Rx_side_blocks"], "P_Rx_side_group", "Rx_side_0_0")
    if objs["Rx_side2_explicit"] or objs["Rx_side2_blocks"]:
        _rx_group(objs["Rx_side2_explicit"], objs["Rx_side2_blocks"], "P_Rx_side_group", "Rx_side_0_0")

    # 코어 그룹: loss_map_phys에 있는 키 우선, 없으면(풀 열해석 + 대칭 EM 조합의 미러) 대응 그룹
    n_group = int(df["n_core_group"].iloc[0])
    core_expected_injected_w = 0.0
    segmented_core = any(
        re.fullmatch(
            r"core_\d+_(?:leg_(?:left|center|right)|yoke_(?:top|bottom))",
            str(c.name),
        )
        for c in objs["core"]
    )
    if segmented_core:
        # Maxwell reports one P_core_i total for the five leg/yoke pieces.  In
        # a symmetry model, first recover the retained gross-frame group total
        # with the canonical unsuffixed name, then distribute it by each live
        # post-split volume.  This preserves both total power and uniform source
        # density instead of injecting the group total into every piece.
        core_groups = {}
        for c in objs["core"]:
            _core_thermal_material_for_piece(c.name)
            i = int(c.name.split("_")[1])
            core_groups.setdefault(i, []).append(c)
        for i, pieces in sorted(core_groups.items()):
            key = f"P_core_{i}"
            if key not in alloc.loss_map:
                key = f"P_core_{n_group + 1 - i}"  # y-미러 그룹
            group_power = alloc.turn_loss(key, f"core_{i}")
            piece_powers = _volume_weighted_powers(pieces, group_power)
            core_expected_injected_w += group_power
            for piece, piece_power in zip(pieces, piece_powers):
                _block(
                    piece,
                    piece_power,
                    native_core_readback=alloc.native_core_contract,
                )
    else:
        # Legacy path intentionally remains byte-for-behavior equivalent.
        for c in objs["core"]:
            try:
                i = int(c.name.split("_")[1])
            except (IndexError, ValueError):
                i = 1
            key = f"P_{c.name}"
            if key not in alloc.loss_map:
                key = f"P_core_{n_group + 1 - i}"  # y-미러 그룹
            core_power = alloc.turn_loss(key, c.name)
            core_expected_injected_w += core_power
            _block(
                c, core_power, native_core_readback=alloc.native_core_contract
            )

    # 콜드플레이트/냉각판은 고정온도 경계라 열원 주입 생략
    sim.thermal_injected = injected
    sim.thermal_core_loss_contract_version = (
        alloc.core_loss_contract_version
    )
    sim.thermal_core_loss_source = alloc.core_loss_source
    sim.thermal_core_loss_correction_factor = float(
        df["core_loss_correction_factor"].iloc[0]
    ) if "core_loss_correction_factor" in df.columns else float("nan")
    if "n_explicit_turns" in df.columns:
        n_explicit = int(df["n_explicit_turns"].iloc[0])
    else:
        has_explicit_rx = any(
            objs.get(key)
            for key in ("Rx_main_explicit", "Rx_side_explicit", "Rx_side2_explicit")
        )
        n_explicit = 1 if has_explicit_rx else 0
    sim.thermal_rx_model = (
        "homogenized_blocks" if n_explicit == 0 else "hybrid_explicit"
    )
    sim.thermal_rx_power_balance = rx_power_balance
    tx_sum = sum(v for k, v in injected.items() if k.startswith("Tx_"))
    rx_sum = sum(v for k, v in injected.items() if k.startswith("Rx_"))
    core_sum = sum(v for k, v in injected.items() if k.startswith("core"))
    core_balance_abs_error_w = abs(core_sum - core_expected_injected_w)
    core_balance_rel_error = core_balance_abs_error_w / max(
        abs(core_expected_injected_w), 1e-12
    )
    if not math.isclose(
        core_sum, core_expected_injected_w, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise RuntimeError(
            "Icepak core-source power balance failed: "
            f"assigned={core_sum:.12g}W, "
            f"expected_margin_adjusted={core_expected_injected_w:.12g}W"
        )
    native_core_sum = (
        sum(item["power_w"] for item in native_core_readbacks)
        if alloc.native_core_contract else float("nan")
    )
    if alloc.native_core_contract and not math.isclose(
        native_core_sum, core_expected_injected_w,
        rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise RuntimeError(
            "native Icepak core-source sum mismatch: "
            f"native={native_core_sum:.12g}W, "
            f"expected={core_expected_injected_w:.12g}W"
        )
    full_core_expected_w = float("nan")
    df_loss_summary = getattr(sim, "df_loss_summary", None)
    if df_loss_summary is not None and "P_core_total" in df_loss_summary.columns:
        full_core_expected_w = float(df_loss_summary["P_core_total"].iloc[0])
    elif mode == "full":
        full_core_expected_w = core_expected_injected_w
    restore_factor = (
        full_core_expected_w / core_expected_injected_w
        if math.isfinite(full_core_expected_w)
        and core_expected_injected_w > 0 else float("nan")
    )
    native_restored_full_w = (
        native_core_sum * restore_factor
        if math.isfinite(native_core_sum) and math.isfinite(restore_factor)
        else float("nan")
    )
    restored_rel_error = (
        abs(native_restored_full_w - full_core_expected_w)
        / max(abs(full_core_expected_w), 1e-12)
        if math.isfinite(native_restored_full_w)
        and math.isfinite(full_core_expected_w) else float("nan")
    )
    if alloc.native_core_contract and (
        not math.isfinite(restored_rel_error) or restored_rel_error > 1e-12
    ):
        raise RuntimeError(
            "native Icepak restored full core power mismatch: "
            f"restored={native_restored_full_w!r}, "
            f"EM_margin_adjusted={full_core_expected_w!r}"
        )
    sim.thermal_core_expected_injected_w = core_expected_injected_w
    sim.thermal_core_requested_wrapper_echo_w = core_sum
    sim.thermal_core_native_readback_w = native_core_sum
    sim.thermal_core_restore_factor = restore_factor
    sim.thermal_core_native_restored_full_w = native_restored_full_w
    sim.thermal_core_full_expected_margin_adjusted_w = full_core_expected_w
    sim.thermal_core_native_restored_rel_error = restored_rel_error
    sim.thermal_core_native_readback_count = len(native_core_readbacks)
    sim.thermal_core_power_balance_abs_error_w = core_balance_abs_error_w
    sim.thermal_core_power_balance_rel_error = core_balance_rel_error
    logging.warning(f"thermal injection totals [W]: Tx={tx_sum:.2f} Rx={rx_sum:.2f} core={core_sum:.2f} "
                 f"(eighth={eighth}, n_obj={len(injected)})")
    for k, v in injected.items():
        if k.startswith("Tx_main_0"):
            logging.warning(f"  inject {k} = {v:.3f} W")
    return injected


def _assign_boundaries(ipk, sim, objs, eighth=False, mode=None):
    mode = mode or ("eighth" if eighth else "full")
    df = sim.df_plus
    plate_temp = float(df["plate_temp"].iloc[0])
    air_temp = float(df["air_temp"].iloc[0])
    fan_v = float(df["fan_velocity"].iloc[0])

    def _bc(result, name):
        return _require_boundary(result, f"thermal boundary {name}")

    # 콜드플레이트 + 권선 냉각판 (Al) 고정온도
    fixed_objs = [o.name for o in objs["core_plates"] + objs["wcp_plates"]]
    if fixed_objs:
        fixed_temperature = f"{plate_temp}cel"
        boundary = ipk.assign_source(
            assignment=fixed_objs,
            thermal_condition="Temperature",
            assignment_value=fixed_temperature,
            boundary_name="cold_plates_fixed_T",
        )
        boundary = _require_boundary(
            boundary,
            "fixed temperature source cold_plates_fixed_T",
            {
                "Objects": fixed_objs,
                "Temperature": fixed_temperature,
            },
        )
        # PyAEDT 0.22 uses ``Temperature`` as its wrapper input, but AEDT
        # 2025.2 otherwise persists the source as Total Power=0W.
        old_auto_update = getattr(boundary, "auto_update", None)
        if old_auto_update is not None:
            boundary.auto_update = False
        boundary.props["Thermal Condition"] = "Fixed Temperature"
        boundary.props["Temperature"] = fixed_temperature
        update = getattr(boundary, "update", None)
        if not callable(update) or not update():
            raise RuntimeError("fixed temperature source cold_plates_fixed_T update failed")
        if old_auto_update is not None:
            boundary.auto_update = old_auto_update
        _require_boundary(
            boundary,
            "fixed temperature source cold_plates_fixed_T after update",
            {
                "Objects": fixed_objs,
                "Thermal Condition": "Fixed Temperature",
                "Temperature": fixed_temperature,
            },
        )

    # 주변온도
    ipk.set_ambient_temp(air_temp)

    def _fresh_region(**pads):
        # Icepak 디자인은 생성 시 AEDT가 Region을 자동 삽입함 -> 삭제 후 원하는 패딩으로 재생성.
        # (이전 코드는 create_air_region이 False를 반환해 경계 전부가 조용히 누락된 채
        #  밀폐 상자로 해석되는 치명적 버그가 있었음 - 게이트1에서 발견)
        try:
            if "Region" in ipk.modeler.object_names:
                ipk.modeler.delete("Region")
        except Exception as e:
            logging.warning(f"default Region delete failed: {e}")
        region = ipk.modeler.create_air_region(is_percentage=True, **pads)
        if not region:
            raise RuntimeError("create_air_region failed (Region conflict?)")
        return region

    # 경계 실패는 조용히 넘기지 않음: BC가 틀린 열해석은 실패보다 나쁨 (캠페인 데이터 오염)
    if mode == "quarter":
        # x/y 대칭 + z 전체 + 부력 on: 부력(z대칭 가정) 분리 검증용
        region = _fresh_region(x_pos=0.0, y_pos=100.0, z_pos=100.0,
                               x_neg=100.0, y_neg=0.0, z_neg=100.0)
        _bc(ipk.assign_symmetry_wall(
            geometry=region.top_face_x.id, boundary_name="sym_x0"), "sym_x0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_y.id, boundary_name="sym_y0"), "sym_y0")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet")
        for face, nm in [(region.bottom_face_x, "outlet_xn"),
                         (region.top_face_z, "outlet_zp"), (region.bottom_face_z, "outlet_zn")]:
            _bc(ipk.assign_pressure_free_opening(
                assignment=[face.id], boundary_name=nm, temperature=f"{air_temp}cel"), nm)
        return region

    if eighth:
        # 1/8: 대칭면 3개(x=0/y=0/z=0)는 region 면을 플러시로 두고 symmetry wall 할당.
        # +y 외곽 = 팬 유입 (양측 팬의 y대칭 유동 가정), -x/+z 외곽 = 배기 opening
        region = _fresh_region(x_pos=0.0, y_pos=100.0, z_pos=100.0,
                               x_neg=100.0, y_neg=0.0, z_neg=0.0)
        _bc(ipk.assign_symmetry_wall(
            geometry=region.top_face_x.id, boundary_name="sym_x0"), "sym_x0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_y.id, boundary_name="sym_y0"), "sym_y0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_z.id, boundary_name="sym_z0"), "sym_z0")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id],
            boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]
        ), "fan_inlet")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.bottom_face_x.id],
            boundary_name="outlet_x",
            temperature=f"{air_temp}cel"
        ), "outlet_x")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.top_face_z.id],
            boundary_name="outlet_z",
            temperature=f"{air_temp}cel"
        ), "outlet_z")
        return region

    # 풀모델: region 전방향
    region = _fresh_region(x_pos=100.0, y_pos=100.0, z_pos=100.0,
                           x_neg=100.0, y_neg=100.0, z_neg=100.0)
    fan_config = str(df.get("fan_config", pd.Series(["dual"])).iloc[0])
    if fan_config == "dual":
        # 양방향 팬 (냉각 스펙: +-y 양측 유입, 배기 +-x/+-z) - 1/8 모델과 동일 물리
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet_pos",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet_pos")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.bottom_face_y.id], boundary_name="fan_inlet_neg",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet_neg")
        for face, nm in [(region.top_face_x, "outlet_xp"), (region.bottom_face_x, "outlet_xn"),
                         (region.top_face_z, "outlet_zp"), (region.bottom_face_z, "outlet_zn")]:
            _bc(ipk.assign_pressure_free_opening(
                assignment=[face.id], boundary_name=nm, temperature=f"{air_temp}cel"), nm)
    else:
        # 단방향 팬 (+y -> -y)
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.bottom_face_y.id], boundary_name="outlet",
            temperature=f"{air_temp}cel"), "outlet")

    return region


# ---------------------------------------------------------------------------
# 메인 엔트리
# ---------------------------------------------------------------------------

def run_thermal_analysis(sim):
    """
    EM loss 디자인 결과(sim.loss_map)를 이용해 Icepak 열해석 수행.
    반환: 온도 요약 1행 DataFrame (T_max_*, T_mean_*)
    """
    thermal_started = time.monotonic()
    df = sim.df_plus

    mode = str(df["thermal_symmetry"].iloc[0])
    eighth = mode == "eighth"

    # Copied Maxwell loss extraction can leave pyProject's cached gRPC proxy
    # stale even though the original Desktop session remains healthy. Rebind
    # the exact project before pyDesign asks project.name while creating Icepak.
    sim._rebind_native_project_for_design_creation()
    ipk = sim.project.create_design(name="icepak_thermal", solver="icepak",
                                    solution="SteadyState TemperatureAndFlow")
    sim.design_thermal = ipk

    set_design_variables(ipk, sim.input_df)
    core_conductivity = _core_thermal_conductivity_contract(df)
    _create_thermal_materials(ipk, df)
    objs = _build_geometry(ipk, sim, eighth=eighth, mode=mode)
    probe_sheets = _create_probe_sheets(ipk, df, objs, eighth=eighth, mode=mode)
    _assign_losses(ipk, sim, objs, eighth=eighth, mode=mode)
    rx_balance = list(getattr(sim, "thermal_rx_power_balance", []))
    if not rx_balance:
        raise RuntimeError("thermal Rx power accounting produced no groups")
    rx_expected_power = sum(float(item["expected_w"]) for item in rx_balance)
    rx_assigned_power = sum(float(item["assigned_w"]) for item in rx_balance)
    rx_balance_errors = [
        abs(float(item["assigned_w"]) - float(item["expected_w"]))
        for item in rx_balance
    ]
    rx_balance_max_abs = max(rx_balance_errors)
    rx_balance_ok = all(
        math.isclose(
            float(item["assigned_w"]),
            float(item["expected_w"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for item in rx_balance
    )
    if not rx_balance_ok:
        raise RuntimeError(f"thermal Rx power accounting mismatch: {rx_balance}")
    _assign_boundaries(ipk, sim, objs, eighth=eighth, mode=mode)

    # 서멀패드 메시 해상 강제: 패드(2mm)가 메시에 안 잡히면 도체가 고정온도 Al에
    # 수치적으로 직결되어 온도가 플레이트에 고정됨 (풀 도메인에서 실측된 함정)
    _assign_thermal_mesh(
        ipk,
        objs,
        side_block_level=int(
            df.get(
                "thermal_rx_side_block_mesh_level", pd.Series([5])
            ).iloc[0]
        ),
    )
    thermal_build_s = time.monotonic() - thermal_started

    setup_started = time.monotonic()
    setup = ipk.create_setup(name=_THERMAL_SETUP_NAME)
    if not setup:
        raise RuntimeError("create_setup returned no ThermalSetup")
    try:
        setup.props["Enabled"] = True
        setup.props["Flow Regime"] = "Turbulent"
        setup.props["Convergence Criteria - Max Iterations"] = int(df["thermal_max_iterations"].iloc[0])
        setup.props["Convergence Criteria - Flow"] = "0.001"
        setup.props["Convergence Criteria - Energy"] = "1e-07"
        setup.props["Solution Initialization - Use Model Based Flow Initialization"] = False
        setup.props["Under-relaxation - Pressure"] = "0.7"
        setup.props["Sequential Solve of Flow and Energy Equations"] = False
        if eighth:
            # 1/8 대칭의 z대칭은 부력 무시 가정에 기반 -> 중력 비활성
            setup.props["Include Gravity"] = False
        if not setup.update():
            raise RuntimeError("ThermalSetup update returned False")
    except Exception as e:
        raise RuntimeError(f"ThermalSetup configuration failed: {e}") from e
    thermal_setup_s = time.monotonic() - setup_started

    # Dispatch the one exact setup. Convergence evidence is independent of PyAEDT's
    # return value because the wrapper can report False after native work completed.
    solve_started = time.monotonic()
    solve_result = _solve_exact_thermal_setup(sim, ipk, setup)
    thermal_solve_s = time.monotonic() - solve_started
    solve_attempts = solve_result["solve_attempts"]
    analyze_call_ok = solve_result["analyze_call_ok"]
    analyze_return_false = solve_result["analyze_return_false"]
    dispatch_status = solve_result["dispatch_status"]
    dispatch_exception_type = solve_result["dispatch_exception_type"]
    dispatch_exception_message = solve_result["dispatch_exception_message"]
    dispatch_forensic_json = solve_result["forensic_json"]
    convergence = solve_result["convergence"]
    logging.warning(
        "[thermal] convergence: available=%s converged=%s iteration=%s "
        "continuity=%s energy=%s reason=%s",
        convergence["thermal_convergence_available"],
        convergence["thermal_converged"],
        convergence["thermal_iterations"],
        convergence["thermal_residual_continuity"],
        convergence["thermal_residual_energy"],
        convergence["thermal_convergence_reason"],
    )
    extraction_started = time.monotonic()

    # ---- 온도 추출 (필드 계산기 직접 평가 - 리포트 기계 미사용) ----
    # 프로브 시트 (회귀학습용 주력 데이터: 위치 고정, 보간값이라 메시 스파이크에 강함)
    # + 그룹별 체적 평균/최대
    temps = {}
    probe = []
    for s in probe_sheets:
        probe.append((s, f"{s.name}_max", "max"))
        probe.append((s, f"{s.name}_mean", "mean"))
    actual_probe_sheet_names = [sheet.name for sheet in probe_sheets]
    expected_probe_sheet_names = list(getattr(
        probe_sheets, "expected_names", actual_probe_sheet_names
    ))
    probe_failures = list(getattr(probe_sheets, "failures", []))
    vol_objs = (objs["Tx"] + objs["Rx_main_explicit"] + objs["Rx_main_blocks"]
                + objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
                + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"] + objs["core"])
    for o in vol_objs:
        probe.append((o, f"T_mean_{o.name}", "mean"))
        probe.append((o, f"T_max_{o.name}", "max"))

    expected_probe_cols = [
        f"{name}_{stat}"
        for name in expected_probe_sheet_names
        for stat in ("max", "mean")
    ]
    field_expected_cols = list(dict.fromkeys(
        [col for _, col, _ in probe] + expected_probe_cols
    ))
    core_region_sheet_names = {
        "center": [
            name for name in expected_probe_sheet_names
            if name.startswith("Tprobe_core_center_leg")
        ],
        "side": [
            name for name in expected_probe_sheet_names
            if name.startswith("Tprobe_core_side_leg")
        ],
        "top_yoke": [
            name for name in expected_probe_sheet_names
            if name.startswith("Tprobe_core_top_yoke")
        ],
    }
    rx_side_outer_sheet_names = [
        name for name in expected_probe_sheet_names
        if name in {"Tprobe_Rx_side_side", "Tprobe_Rx_side2_side"}
    ]
    rx_side_inner_sheet_names = [
        name for name in expected_probe_sheet_names
        if name in {"Tprobe_Rx_side1_inner", "Tprobe_Rx_side2_inner"}
    ]
    rx_side_face_names = {
        "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
        "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
    }
    actual_rx_side_face_count = sum(
        name in rx_side_face_names for name in actual_probe_sheet_names
    )
    aggregate_core_cols = []
    if all(core_region_sheet_names.values()):
        aggregate_core_cols = [
            "Tprobe_core_center_leg_max", "Tprobe_core_center_leg_mean",
            "Tprobe_core_side_leg_max", "Tprobe_core_side_leg_mean",
            "Tprobe_core_top_yoke_max", "Tprobe_core_top_yoke_mean",
            "Tprobe_core_center_max", "Tprobe_core_center_mean",
        ]
    aggregate_rx_side_cols = []
    if rx_side_outer_sheet_names and (
        len(rx_side_outer_sheet_names) == len(rx_side_inner_sheet_names)
    ):
        aggregate_rx_side_cols = [
            "Tprobe_Rx_side_outer_max", "Tprobe_Rx_side_outer_mean",
            "Tprobe_Rx_side_inner_max", "Tprobe_Rx_side_inner_mean",
            "Tprobe_Rx_side_leeward_max", "Tprobe_Rx_side_leeward_mean",
        ]
    expected_cols = list(dict.fromkeys(
        field_expected_cols + aggregate_core_cols + aggregate_rx_side_cols
    ))
    optional_cols = set()
    if int(df["N1_side"].iloc[0]) == 0:
        optional_cols.update({"Tprobe_Tx_side_max", "Tprobe_Tx_side_mean"})
    required_expected_cols = [col for col in expected_cols if col not in optional_cols]
    group_objects = {
        "T_max_Tx": list(objs["Tx"]),
        "T_max_Rx_main": list(objs["Rx_main_explicit"] + objs["Rx_main_blocks"]),
        "T_max_Rx_side": list(
            objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
            + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"]
        ),
        "T_max_core": list(objs["core"]),
    }
    group_bits = {
        "T_max_Tx": 1,
        "T_max_Rx_main": 2,
        "T_max_Rx_side": 4,
        "T_max_core": 8,
    }
    required_keys = ["T_max_Tx", "T_max_Rx_main", "T_max_core"]
    if int(df["N2_side"].iloc[0]) > 0:
        required_keys.append("T_max_Rx_side")
    required_group_mask = sum(group_bits[key] for key in required_keys)
    required_group_count = len(required_keys)

    if convergence["thermal_converged"] != 1:
        thermal_extraction_s = time.monotonic() - extraction_started
        summary = {
            "thermal_solved": [0],
            "thermal_extraction_complete": [0],
            "thermal_missing_count": [len(expected_cols)],
            "thermal_required_missing_count": [required_group_count],
            "thermal_required_group_mask": [required_group_mask],
            "thermal_required_group_count": [required_group_count],
            "thermal_solve_attempts": [solve_attempts],
            "thermal_analyze_call_ok": [1 if analyze_call_ok else 0],
            "thermal_analyze_return_false": [1 if analyze_return_false else 0],
            "thermal_dispatch_status": [dispatch_status],
            "thermal_dispatch_exception_type": [dispatch_exception_type],
            "thermal_dispatch_exception_message": [dispatch_exception_message],
            "thermal_dispatch_forensic_json": [dispatch_forensic_json],
            "thermal_solution_data_available": [0],
            "thermal_field_summary_attempts": [0],
            "thermal_field_summary_value_count": [0],
            "thermal_calculator_attempts": [0],
            "thermal_extraction_method": ["not_attempted"],
            "thermal_extraction_failure_reason": [
                f"solve_not_converged:{convergence['thermal_convergence_reason']}"
            ],
            "thermal_probe_failure_count": [len(probe_failures)],
            "thermal_probe_failures_json": [
                serialize_probe_failures(probe_failures)
            ],
            "thermal_build_s": [thermal_build_s],
            "thermal_setup_s": [thermal_setup_s],
            "thermal_solve_s": [thermal_solve_s],
            "thermal_extraction_s": [thermal_extraction_s],
            "thermal_rx_model": [sim.thermal_rx_model],
            "thermal_core_conductivity_model": [
                core_conductivity["thermal_core_conductivity_model"]
            ],
            "thermal_core_k_inplane": [
                core_conductivity["thermal_core_k_inplane"]
            ],
            "thermal_core_k_throughstack": [
                core_conductivity["thermal_core_k_throughstack"]
            ],
            "thermal_rx_power_balance_ok": [1 if rx_balance_ok else 0],
            "thermal_rx_power_balance_group_count": [len(rx_balance)],
            "thermal_rx_power_balance_max_abs_w": [rx_balance_max_abs],
            "thermal_rx_expected_power_w": [rx_expected_power],
            "thermal_rx_assigned_power_w": [rx_assigned_power],
            "thermal_rx_side_probe_contract_version": [
                RX_SIDE_FACE_PROBE_CONTRACT_VERSION
            ],
            "thermal_rx_side_probe_max_rule": [RX_SIDE_FACE_MAX_RULE],
            "thermal_rx_side_probe_mean_rule": [RX_SIDE_FACE_MEAN_RULE],
            "thermal_rx_side_probe_selected_face": [""],
            "thermal_rx_side_probe_face_count": [
                actual_rx_side_face_count
            ],
            "T_max_Tx": [float("nan")],
            "T_max_Rx_main": [float("nan")],
            "T_max_Rx_side": [float("nan")],
            "T_max_core": [float("nan")],
        }
        summary.update({key: [value] for key, value in convergence.items()})
        for col in expected_cols:
            summary[col] = [float("nan")]
        logging.error(
            "[thermal] solve rejected before extraction: analyze-call-ok=%s, "
            "converged=%s, reason=%s",
            analyze_call_ok,
            convergence["thermal_converged"],
            convergence["thermal_convergence_reason"],
        )
        sim.df_thermal = pd.DataFrame(summary)
        return sim.df_thermal

    native_ipk = _native_solver(ipk)
    try:
        _activate_thermal_design(ipk, design_name=_THERMAL_DESIGN_NAME)
        solution = native_ipk.existing_analysis_sweeps[0]
    except Exception:
        solution = "ThermalSetup : SteadyState"

    def _post_of(app):
        """Return the actual post processor when a wrapper exposes it as a callable."""
        po = app.post
        if callable(po) and not hasattr(po, "create_field_summary"):
            po = po()
        return po

    # ---- 1차: Field Summary 일괄 (호출 1회, GUI/리눅스 공통 신뢰 경로) ----
    field_summary_issues = {}

    def _field_summary_bulk(entries):
        fs = _post_of(native_ipk).create_field_summary()
        seen = set()
        for obj, col, op in entries:
            is3d = getattr(obj, "is3d", True)
            key = (obj.name, is3d)
            if key in seen:
                continue
            seen.add(key)
            fs.add_calculation("Object", "Volume" if is3d else "Surface",
                               obj.name, "Temperature")
        df_fs = fs.get_field_summary_data(setup=solution, pandas_output=True)
        if df_fs is None or isinstance(df_fs, bool) or not hasattr(df_fs, "columns") or not len(df_fs):
            raise RuntimeError(f"field summary returned {type(df_fs).__name__} (no data)")
        # 컬럼: Entity/Geometry/Quantity/Min/Max/Mean ... (버전에 따라 대소문자 상이)
        cols = {str(c).strip().lower(): c for c in df_fs.columns}
        name_c = cols.get("geometry name", cols.get("entity name", list(df_fs.columns)[2]))
        unit_c = cols.get("unit", cols.get("units"))
        got = {}
        for obj, col, op in entries:
            row = df_fs[df_fs[name_c].astype(str) == str(obj.name)]
            if not len(row):
                field_summary_issues[col] = "entity_not_returned"
                continue
            want = cols.get("max" if op == "max" else "mean")
            if want is None:
                field_summary_issues[col] = f"missing_{op}_column"
                continue
            try:
                unit = row.iloc[0][unit_c] if unit_c is not None else None
                got[col] = parse_temperature_celsius(row.iloc[0][want], unit)
                field_summary_issues.pop(col, None)
            except Exception as exc:
                field_summary_issues[col] = (
                    f"invalid_temperature:{type(exc).__name__}:{exc}"
                )[:512]
        return got

    def _scalar_probe_temperature(obj, op):
        """Independent saved-field fallback, deliberately limited to sheets."""
        post = _post_of(native_ipk)
        getter = getattr(post, "get_scalar_field_value", None)
        if not callable(getter):
            raise RuntimeError("post processor has no scalar field API")
        value = getter(
            "Temp",
            scalar_function="Maximum" if op == "max" else "Mean",
            solution=solution,
            object_name=obj.name,
            object_type="surface",
        )
        return parse_temperature_celsius(value)

    def _refresh_core_probe_aggregates():
        """Select the hottest depth plane per leg, then the hottest leg."""
        selected = {}
        output_prefixes = {
            "center": "Tprobe_core_center_leg",
            "side": "Tprobe_core_side_leg",
            "top_yoke": "Tprobe_core_top_yoke",
        }
        for region, names in core_region_sheet_names.items():
            if not names:
                continue
            candidates = []
            for name in names:
                max_col = f"{name}_max"
                mean_col = f"{name}_mean"
                if max_col not in temps or mean_col not in temps:
                    continue
                maximum = float(temps[max_col])
                mean = float(temps[mean_col])
                if math.isfinite(maximum) and math.isfinite(mean):
                    candidates.append((maximum, mean, name))
            if len(candidates) != len(names):
                continue
            maximum, mean, name = max(candidates, key=lambda item: item[0])
            prefix = output_prefixes[region]
            temps[f"{prefix}_max"] = maximum
            temps[f"{prefix}_mean"] = mean
            selected[region] = (maximum, mean, name)
        if len(selected) == len(core_region_sheet_names) == 3:
            maximum, mean, _name = max(
                selected.values(), key=lambda item: item[0]
            )
            # Preserve the established model/quality-contract target while
            # making it the hottest value across both physical leg locations.
            temps["Tprobe_core_center_max"] = maximum
            temps["Tprobe_core_center_mean"] = mean

    rx_side_selected_face = ""

    def _refresh_rx_side_face_aggregates():
        """Aggregate complete inner/outer face sets without averaging faces."""
        nonlocal rx_side_selected_face
        if not aggregate_rx_side_cols:
            return
        aggregate, selected = aggregate_rx_side_faces(
            temps, rx_side_outer_sheet_names, rx_side_inner_sheet_names
        )
        if aggregate:
            # Compatibility target: its revised, versioned meaning is the
            # hottest transformer-inward/outward radial face.  Pair the mean
            # from that same face; never average unlike or unequal surfaces.
            temps.update(aggregate)
            rx_side_selected_face = selected

    field_summary_attempts = 0
    for attempt in range(1, 4):
        missing_entries = [entry for entry in probe if entry[1] not in temps]
        if not missing_entries:
            break
        field_summary_attempts = attempt
        try:
            _activate_thermal_design(ipk, design_name=_THERMAL_DESIGN_NAME)
            temps.update(_field_summary_bulk(missing_entries))
            _refresh_core_probe_aggregates()
            _refresh_rx_side_face_aggregates()
        except Exception as e:
            logging.warning(f"[thermal] field summary attempt {attempt}/3 failed: {e}")
        if all(col in temps for col in required_expected_cols):
            break
        if attempt < 3:
            time.sleep(10)
    _refresh_core_probe_aggregates()
    _refresh_rx_side_face_aggregates()
    n_fs = sum(1 for col in field_expected_cols if col in temps)

    n_calc = 0
    calc_attempts = 0
    scalar_probe_issues = {}
    # AEDT can fail ExportFieldsSummary while the saved surface field remains
    # readable.  Use the replay-proven scalar API only for missing probe-sheet
    # statistics.  Never replace a missing modeled-volume maximum with a probe:
    # that case indicates a genuine zero-mesh-volume thermal failure.
    for obj, col, op in probe:
        if col in temps or col in optional_cols or getattr(obj, "is3d", True):
            continue
        calc_attempts += 1
        try:
            _activate_thermal_design(ipk, design_name=_THERMAL_DESIGN_NAME)
            temps[col] = _scalar_probe_temperature(obj, op)
            n_calc += 1
        except Exception as exc:
            scalar_probe_issues[col] = (
                f"{type(exc).__name__}:{exc}"
            )[:512]
    _refresh_core_probe_aggregates()
    _refresh_rx_side_face_aggregates()

    missing_cols = [col for col in expected_cols if col not in temps]
    required_missing_cols = [col for col in missing_cols if col not in optional_cols]
    geometry_failure_names = {
        str(failure.get("probe", "")) for failure in probe_failures
    }
    for name in expected_probe_sheet_names:
        missing_stats = [
            stat for stat in ("max", "mean")
            if f"{name}_{stat}" in required_missing_cols
        ]
        if not missing_stats or name in geometry_failure_names:
            continue
        columns = [f"{name}_{stat}" for stat in missing_stats]
        details = []
        for column in columns:
            if column in field_summary_issues:
                details.append(f"field_summary={field_summary_issues[column]}")
            if column in scalar_probe_issues:
                details.append(f"scalar={scalar_probe_issues[column]}")
        failure = {
            "probe": name,
            "stage": "extraction",
            "reason": "saved_field_fallback_exhausted",
            "columns": columns,
        }
        if details:
            failure["detail"] = "; ".join(details)[:512]
        probe_failures.append(failure)
    n_fail = len(missing_cols)
    logging.warning(
        f"[thermal] extraction: field-summary {n_fs}, calculator {n_calc}, "
        f"failed {n_fail} / total {len(probe)} "
        f"(required-column failures={len(required_missing_cols)})"
    )

    def _group_max(group_objects):
        # A partial maximum can silently understate component temperature. Require
        # every modeled maximum for the physical group before emitting its summary.
        cols = list(dict.fromkeys(f"T_max_{obj.name}" for obj in group_objects))
        vals = [temps[col] for col in cols if col in temps and math.isfinite(float(temps[col]))]
        return max(vals) if cols and len(vals) == len(cols) else float("nan")

    group_values = {
        key: _group_max(objects) for key, objects in group_objects.items()
    }
    thermal_extraction_s = time.monotonic() - extraction_started
    required_missing_count = sum(
        1 for key in required_keys
        if not group_objects[key] or not math.isfinite(float(group_values[key]))
    )
    required_complete = required_missing_count == 0 and required_group_count > 0
    solution_data_available = n_fs > 0
    solved = (
        convergence["thermal_converged"] == 1
        and solution_data_available
        and required_complete
    )
    if required_missing_count:
        extraction_failure_reason = "required_volume_temperature_missing"
    elif required_missing_cols:
        extraction_failure_reason = "required_probe_temperature_missing"
    else:
        extraction_failure_reason = ""
    extraction_method = (
        "field_summary+scalar_field_calculator"
        if calc_attempts else "field_summary"
    )

    summary = {
        "thermal_solved": [1 if solved else 0],
        "thermal_extraction_complete": [1 if not required_missing_cols else 0],
        "thermal_missing_count": [len(missing_cols)],
        "thermal_required_missing_count": [required_missing_count],
        # Bit mask: Tx=1, Rx_main=2, Rx_side=4, core=8. Rx_side is optional
        # when N2_side=0 and is then deliberately excluded from the gate.
        "thermal_required_group_mask": [required_group_mask],
        "thermal_required_group_count": [required_group_count],
        "thermal_solve_attempts": [solve_attempts],
        "thermal_analyze_call_ok": [1 if analyze_call_ok else 0],
        "thermal_analyze_return_false": [1 if analyze_return_false else 0],
        "thermal_dispatch_status": [dispatch_status],
        "thermal_dispatch_exception_type": [dispatch_exception_type],
        "thermal_dispatch_exception_message": [dispatch_exception_message],
        "thermal_dispatch_forensic_json": [dispatch_forensic_json],
        "thermal_solution_data_available": [1 if solution_data_available else 0],
        "thermal_field_summary_attempts": [field_summary_attempts],
        "thermal_field_summary_value_count": [n_fs],
        "thermal_calculator_attempts": [calc_attempts],
        "thermal_extraction_method": [extraction_method],
        "thermal_extraction_failure_reason": [extraction_failure_reason],
        "thermal_probe_failure_count": [len(probe_failures)],
        "thermal_probe_failures_json": [
            serialize_probe_failures(probe_failures)
        ],
        "thermal_build_s": [thermal_build_s],
        "thermal_setup_s": [thermal_setup_s],
        "thermal_solve_s": [thermal_solve_s],
        "thermal_extraction_s": [thermal_extraction_s],
        "thermal_rx_model": [sim.thermal_rx_model],
        "thermal_core_conductivity_model": [
            core_conductivity["thermal_core_conductivity_model"]
        ],
        "thermal_core_k_inplane": [
            core_conductivity["thermal_core_k_inplane"]
        ],
        "thermal_core_k_throughstack": [
            core_conductivity["thermal_core_k_throughstack"]
        ],
        "thermal_rx_power_balance_ok": [1 if rx_balance_ok else 0],
        "thermal_rx_power_balance_group_count": [len(rx_balance)],
        "thermal_rx_power_balance_max_abs_w": [rx_balance_max_abs],
        "thermal_rx_expected_power_w": [rx_expected_power],
        "thermal_rx_assigned_power_w": [rx_assigned_power],
        "thermal_rx_side_probe_contract_version": [
            RX_SIDE_FACE_PROBE_CONTRACT_VERSION
        ],
        "thermal_rx_side_probe_max_rule": [RX_SIDE_FACE_MAX_RULE],
        "thermal_rx_side_probe_mean_rule": [RX_SIDE_FACE_MEAN_RULE],
        "thermal_rx_side_probe_selected_face": [rx_side_selected_face],
        "thermal_rx_side_probe_face_count": [
            actual_rx_side_face_count
        ],
        "thermal_core_loss_contract_version": [
            sim.thermal_core_loss_contract_version
        ],
        "thermal_core_loss_source": [sim.thermal_core_loss_source],
        "thermal_core_loss_correction_factor": [
            sim.thermal_core_loss_correction_factor
        ],
        "thermal_core_expected_injected_w": [
            sim.thermal_core_expected_injected_w
        ],
        "thermal_core_requested_wrapper_echo_w": [
            sim.thermal_core_requested_wrapper_echo_w
        ],
        "thermal_core_native_readback_w": [
            sim.thermal_core_native_readback_w
        ],
        "thermal_core_restore_factor": [
            sim.thermal_core_restore_factor
        ],
        "thermal_core_native_restored_full_w": [
            sim.thermal_core_native_restored_full_w
        ],
        "thermal_core_full_expected_margin_adjusted_w": [
            sim.thermal_core_full_expected_margin_adjusted_w
        ],
        "thermal_core_native_restored_rel_error": [
            sim.thermal_core_native_restored_rel_error
        ],
        "thermal_core_native_readback_count": [
            sim.thermal_core_native_readback_count
        ],
        "thermal_core_power_balance_abs_error_w": [
            sim.thermal_core_power_balance_abs_error_w
        ],
        "thermal_core_power_balance_rel_error": [
            sim.thermal_core_power_balance_rel_error
        ],
        "T_max_Tx": [group_values["T_max_Tx"]],
        "T_max_Rx_main": [group_values["T_max_Rx_main"]],
        "T_max_Rx_side": [group_values["T_max_Rx_side"]],
        "T_max_core": [group_values["T_max_core"]],
    }
    summary.update({key: [value] for key, value in convergence.items()})
    if not solved:
        logging.error(
            "[thermal] validation failed: field-summary-data=%s, required-missing=%d, "
            "missing-total=%d, analyze-call-ok=%s",
            solution_data_available,
            required_missing_count,
            len(missing_cols),
            analyze_call_ok,
        )
    # 개별 값도 함께 저장
    for col in expected_cols:
        summary[col] = [temps.get(col, float("nan"))]

    sim.df_thermal = pd.DataFrame(summary)
    return sim.df_thermal
