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

import csv
import hashlib
import io
import json
import math
import logging
import os
import re
import shlex
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    compute_layer_positions,
)
from module.input_parameter_260706 import get_tx_y_gaps, set_design_variables
from module.core_material_contract import PHYSICS_DATA_REVISION, _aedt_number
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
from module.aedt_terminal_attestation import (
    advance_scoped_message_cursor,
    capture_scoped_message_cursor,
)
from module.fixed_boundary_contract import (
    FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK,
)


def _native_solver(app):
    """Return the PyAEDT solver behind a pyDesign wrapper."""
    solver = getattr(app, "solver_instance", None)
    return solver if solver is not None else app


_THERMAL_DESIGN_NAME = "icepak_thermal"
_THERMAL_SETUP_NAME = "ThermalSetup"
THERMAL_PAD_CONDUCTIVITY_W_MK = FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
THERMAL_PAD_MATERIAL_POLICY = (
    "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
    "electrically_insulating_v1"
)
THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION = (
    "thermal-pad-native-material-readback-v1"
)
RX_EXPLICIT_INSULATION_MATERIAL = "winding_insulation"
RX_EXPLICIT_INSULATION_POLICY = (
    "rx-explicit-interturn-solid-candidate-k-ins-native-attested-v1"
)
RX_EXPLICIT_INSULATION_NATIVE_READBACK_CONTRACT_VERSION = (
    "rx-explicit-insulation-native-material-readback-v1"
)
_RX_INSULATION_KEYS = (
    "Rx_main_insulation",
    "Rx_side_insulation",
    "Rx_side2_insulation",
)
THERMAL_MESH_POLICY = "b3-rxmain-l5-wcp-pad-padded-regions-v1"
THERMAL_MESH_PLAN_CONTRACT_VERSION = "thermal-mesh-plan-v4"
THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION = "thermal-mesh-preflight-v2"
THERMAL_MESH_STATS_CONTRACT_VERSION = "thermal-native-mesh-stats-v1"
THERMAL_SETUP_CONTROL_READBACK_CONTRACT_VERSION = (
    "thermal-native-setup-control-readback-v1"
)
THERMAL_MESH_STATS_FILENAME = "icepak_thermal_mesh_quality.ms"
THERMAL_MESH_STATS_MAX_BYTES = 64 * 1024 * 1024
WCP_PAD_MESH_REGION_CONTRACT_VERSION = (
    "wcp-pad-per-object-anisotropic-region-v1"
)
WCP_PAD_MESH_REGION_PADDING_TYPE = "Absolute Offset"
WCP_PAD_MESH_REGION_PADDING_MM = 2.0
_WCP_PAD_MESH_REGION_DIRECTIONS = (
    "+X", "-X", "+Y", "-Y", "+Z", "-Z",
)
THERMAL_FLUENT_PROCESS_CONTRACT_VERSION = (
    "thermal-fluent-total-processes-v2"
)
_THERMAL_UNMESHED_OBJECT = re.compile(
    r"'(?P<object>[^']+)'\s*:\s*Object\s+does\s+not\s+have\s+mesh\b",
    re.IGNORECASE,
)


def _standalone_thermal_parallel_policy(sim):
    """Map one standalone CPU allocation to total Fluent worker processes.

    Maxwell interprets ``NumCores`` as the requested core count and keeps
    ``NumEngines=1``.  Icepak/Fluent uses the same local-engine contract:
    ``NumCores`` is the total local process count while ``NumEngines`` remains
    one.  Setting both values to the allocated CPU count makes AEDT divide the
    cores across engines and can degrade a local launch to ``-t1``.  Preserve
    one engine and request the allocation through ``NumCores`` only.
    """

    try:
        total_processes = int(sim.NUM_CORE)
        maxwell_tasks = int(getattr(sim, "NUM_TASK", 1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "standalone Icepak parallel policy has invalid runner counts"
        ) from exc
    if total_processes < 1:
        raise RuntimeError(
            "standalone Icepak parallel policy requires a positive process count"
        )
    if maxwell_tasks != 1:
        raise RuntimeError(
            "standalone Icepak parallel policy must not change the one-engine "
            f"Maxwell contract: NUM_TASK={maxwell_tasks}"
        )

    core_policy = dict(getattr(sim, "solver_core_policy", {}) or {})
    strict = core_policy.get("opt_in") is True
    if strict:
        expected = {
            "backend": "standalone",
            "requested_num_cores": total_processes,
            "effective_num_cores": total_processes,
            "num_tasks": 1,
            "slurm_cpus_per_task_readback": total_processes,
        }
        mismatches = {}
        for key, value in expected.items():
            actual = core_policy.get(key)
            if key != "backend":
                try:
                    actual = int(actual)
                except (TypeError, ValueError, OverflowError):
                    pass
            if actual != value:
                mismatches[key] = {"expected": value, "actual": actual}
        try:
            affinity = int(core_policy.get("affinity_count_readback"))
        except (TypeError, ValueError, OverflowError):
            affinity = -1
        if affinity < total_processes:
            mismatches["affinity_count_readback"] = {
                "expected_minimum": total_processes,
                "actual": affinity,
            }
        if mismatches:
            raise RuntimeError(
                "standalone Icepak parallel policy does not match the "
                f"authenticated allocation: {mismatches}"
            )

    policy = {
        "schema": THERMAL_FLUENT_PROCESS_CONTRACT_VERSION,
        "backend": "standalone",
        "strict_attestation": strict,
        "allocated_cpus": total_processes,
        "maxwell_num_engines_unchanged": maxwell_tasks,
        "pyaedt_cores_argument": total_processes,
        "pyaedt_tasks_argument": maxwell_tasks,
        "pyaedt_use_auto_settings_argument": False,
        "expected_fluent_processes": total_processes,
        "expected_num_engines": maxwell_tasks,
        "icepak_num_engines_semantics": (
            "single_local_engine_with_num_cores_as_total_fluent_processes"
        ),
    }
    sim.thermal_parallel_policy = dict(policy)
    return policy


def _thermal_hpc_acf_snapshot(native_ipk):
    """Capture the exact Icepak ACF identity before PyAEDT rewrites it."""

    working_directory = str(
        getattr(native_ipk, "working_directory", "") or ""
    ).strip()
    if not working_directory:
        raise RuntimeError(
            "standalone Icepak HPC working directory is unavailable"
        )
    path = Path(working_directory).resolve() / "pyaedt_config.acf"
    snapshot = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat_result = path.stat()
        snapshot.update({
            "size_bytes": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return snapshot


def _validated_thermal_hpc_acf(native_ipk, policy, before):
    """Fail closed unless PyAEDT wrote the exact Icepak 16-process ACF."""

    expected_path = Path(str(before.get("path", ""))).resolve()
    current_path = Path(
        str(_thermal_hpc_acf_snapshot(native_ipk)["path"])
    ).resolve()
    if current_path != expected_path:
        raise RuntimeError(
            "standalone Icepak HPC ACF path changed across dispatch: "
            f"before={expected_path}, after={current_path}"
        )
    if not current_path.is_file():
        raise RuntimeError(
            f"standalone Icepak HPC ACF is missing: {current_path}"
        )
    stat_result = current_path.stat()
    if stat_result.st_size <= 0 or stat_result.st_size > 65536:
        raise RuntimeError(
            f"standalone Icepak HPC ACF has invalid size: {current_path}"
        )
    text = current_path.read_text(encoding="utf-8", errors="strict")
    total = int(policy["expected_fluent_processes"])
    engines = int(policy.get("expected_num_engines", 1))
    expected = {
        "ConfigName": "'pyaedt_config'",
        "DesignType": "'Icepak'",
        "MachineName": "'localhost'",
        "NumEngines": str(engines),
        "NumCores": str(total),
        "NumGPUs": "0",
        "UseAutoSettings": "False",
    }
    mismatches = {}
    for key, value in expected.items():
        matches = re.findall(
            rf"(?m)^\s*{re.escape(key)}\s*=\s*([^\r\n]+?)\s*$",
            text,
        )
        if matches != [value]:
            mismatches[key] = {"expected": value, "actual": matches}
    begin_count = text.count("$begin 'DSOConfig'")
    end_count = text.count("$end 'DSOConfig'")
    if begin_count != 1 or end_count != 1:
        mismatches["DSOConfig"] = {
            "expected": {"begin": 1, "end": 1},
            "actual": {"begin": begin_count, "end": end_count},
        }
    if mismatches:
        raise RuntimeError(
            "standalone Icepak HPC ACF contract mismatch: "
            f"{mismatches}"
        )

    current_identity = {
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if before.get("exists") is True and all(
        current_identity.get(key) == before.get(key)
        for key in ("size_bytes", "mtime_ns", "sha256")
    ):
        raise RuntimeError(
            "standalone Icepak HPC ACF was not freshly rewritten for dispatch"
        )
    return {
        "schema": "thermal-icepak-hpc-acf-readback-v1",
        "passed": True,
        "path": str(current_path),
        "num_cores_readback": total,
        "num_engines_readback": engines,
        "num_gpus_readback": 0,
        "use_auto_settings_readback": False,
        "acf_sha256": current_identity["sha256"],
        "fresh_rewrite_attested": True,
    }


def _fluent_process_counts(commandline):
    """Return explicit Fluent total-process declarations from one command."""

    text = str(commandline or "")
    thread_counts = [
        int(value) for value in re.findall(
            r"(?<!\S)-t\s*([1-9][0-9]*)(?=\s|$)", text
        )
    ]
    nprocs_counts = [
        int(value) for value in re.findall(
            r"(?:^|[\s,;])nprocs_string\s*=\s*"
            r"['\"]?([1-9][0-9]*)['\"]?(?=$|[\s,;])",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return {
        "thread_counts": thread_counts,
        "nprocs_counts": nprocs_counts,
    }


def _is_fluent_runtime_command(commandline):
    text = str(commandline or "").casefold()
    return any(token in text for token in (
        "fluent", "cortex", "3ddp_host", "3ddp_node", "nprocs_string=",
    ))


class _StandaloneThermalProcessAttestor:
    """Observe only new descendants and attest Fluent's actual ``-t`` value."""

    def __init__(self, expected_processes, root_pid=None, poll_s=0.1):
        self.expected_processes = int(expected_processes)
        self.root_pid = int(os.getpid() if root_pid is None else root_pid)
        self.poll_s = float(poll_s)
        self._stop = threading.Event()
        self._thread = None
        self._baseline = {}
        self._records = {}
        self._scan_count = 0
        self._successful_scan_count = 0
        self._scan_errors = []

    @staticmethod
    def _descendants(root_pid):
        import psutil

        root = psutil.Process(int(root_pid))
        records = {}
        for process in root.children(recursive=True):
            try:
                argv = [str(value) for value in (process.cmdline() or [])]
                commandline = " ".join(shlex.quote(value) for value in argv)
                records[int(process.pid)] = {
                    "pid": int(process.pid),
                    "ppid": int(process.ppid()),
                    "create_time": float(process.create_time()),
                    "name": str(process.name() or ""),
                    "argv": argv,
                    "commandline": commandline,
                }
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                OSError,
                TypeError,
                ValueError,
            ):
                continue
        return records

    @staticmethod
    def _identity(record):
        return (int(record["pid"]), round(float(record["create_time"]), 3))

    def start(self):
        self._baseline = {
            self._identity(record): record
            for record in self._descendants(self.root_pid).values()
        }
        self._thread = threading.Thread(
            target=self._run,
            name="mft-icepak-process-attestor",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self._scan_count += 1
            try:
                current = self._descendants(self.root_pid)
                self._successful_scan_count += 1
                for record in current.values():
                    identity = self._identity(record)
                    if identity in self._baseline:
                        continue
                    commandline = record["commandline"]
                    if not _is_fluent_runtime_command(commandline):
                        continue
                    counts = _fluent_process_counts(commandline)
                    key = (
                        identity,
                        commandline,
                    )
                    if key in self._records:
                        continue
                    captured = {
                        **record,
                        **counts,
                    }
                    self._records[key] = captured
                    if counts["thread_counts"] or counts["nprocs_counts"]:
                        logging.warning(
                            "[thermal] Fluent process command observed: %s",
                            json.dumps(
                                captured,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ),
                        )
            except Exception as exc:
                self._scan_errors.append(
                    f"{type(exc).__name__}: {str(exc)[:256]}"
                )
                self._scan_errors = self._scan_errors[-8:]
            self._stop.wait(self.poll_s)

    def finish(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_s * 10.0))
            if self._thread.is_alive():
                raise RuntimeError(
                    "standalone Icepak process attestor did not stop"
                )
        records = list(self._records.values())
        explicit = [
            record for record in records
            if record["thread_counts"] or record["nprocs_counts"]
        ]
        thread_values = [
            value for record in explicit
            for value in record["thread_counts"]
        ]
        nprocs_values = [
            value for record in explicit
            for value in record["nprocs_counts"]
        ]
        all_values = [*thread_values, *nprocs_values]
        expected = self.expected_processes
        mismatches = sorted(set(
            value for value in all_values if value != expected
        ))
        evidence = {
            "schema": "thermal-fluent-process-command-attestation-v1",
            "passed": bool(thread_values) and not mismatches,
            "expected_fluent_processes": expected,
            "thread_count_readbacks": thread_values,
            "nprocs_count_readbacks": nprocs_values,
            "mismatched_process_counts": mismatches,
            "oversubscription_detected": any(
                value > expected for value in all_values
            ),
            "scan_count": int(self._scan_count),
            "successful_scan_count": int(self._successful_scan_count),
            "scan_errors": list(self._scan_errors),
            "commands": explicit[:32],
        }
        if self._successful_scan_count < 1:
            raise RuntimeError(
                "standalone Icepak process attestation had no successful scan: "
                + json.dumps(evidence, sort_keys=True)[:2000]
            )
        if not thread_values:
            raise RuntimeError(
                "standalone Icepak process attestation observed no Fluent -t "
                "command: " + json.dumps(evidence, sort_keys=True)[:2000]
            )
        if mismatches:
            raise RuntimeError(
                "standalone Icepak process count mismatch: "
                + json.dumps(evidence, sort_keys=True)[:4000]
            )
        return evidence


def _raw_aedt_material_props(materials, material_name):
    """Return fresh native material data, bypassing PyAEDT's assigned cache."""
    manager = getattr(materials, "omaterial_manager", None)
    if manager is None or not callable(getattr(manager, "GetData", None)):
        raise RuntimeError("AEDT material manager cannot provide native readback")
    try:
        raw = list(manager.GetData(str(material_name)))
        from ansys.aedt.core.generic.data_handlers import _arg2dict

        parsed = {}
        _arg2dict(raw, parsed)
    except Exception as exc:
        raise RuntimeError(
            f"native AEDT material readback failed for {material_name!r}"
        ) from exc
    if len(parsed) != 1:
        raise RuntimeError(
            f"unexpected native material payload for {material_name!r}: "
            f"top-level keys={list(parsed)}"
        )
    props = next(iter(parsed.values()))
    if not isinstance(props, dict) or not props:
        raise RuntimeError(
            f"native AEDT material payload is empty for {material_name!r}"
        )
    return props


def _thermal_pad_native_readback(materials):
    """Fail closed unless AEDT itself reports the requested TIM properties."""
    props = _raw_aedt_material_props(materials, "thermal_pad")
    thermal_k = _aedt_number(
        props.get("thermal_conductivity"),
        "thermal_pad.thermal_conductivity",
    )
    electrical_sigma = _aedt_number(
        props.get("conductivity"),
        "thermal_pad.conductivity",
    )
    if not math.isclose(
        thermal_k,
        THERMAL_PAD_CONDUCTIVITY_W_MK,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "native thermal_pad thermal_conductivity mismatch: "
            f"{thermal_k!r} != {THERMAL_PAD_CONDUCTIVITY_W_MK!r}"
        )
    if not math.isclose(
        electrical_sigma,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "native thermal_pad conductivity mismatch: "
            f"{electrical_sigma!r} != 0.0"
        )
    return {
        "thermal_conductivity_W_mK": thermal_k,
        "electrical_conductivity_S_m": electrical_sigma,
    }


def _thermal_pad_result_metadata(native_readback):
    """Record both requested policy and fresh native AEDT material readback."""
    if not isinstance(native_readback, dict):
        raise RuntimeError("thermal_pad native readback evidence is missing")
    return {
        "thermal_pad_conductivity_W_mK": [
            THERMAL_PAD_CONDUCTIVITY_W_MK
        ],
        "thermal_pad_material_policy": [THERMAL_PAD_MATERIAL_POLICY],
        "thermal_pad_native_readback_contract_version": [
            THERMAL_PAD_NATIVE_READBACK_CONTRACT_VERSION
        ],
        "thermal_pad_native_readback_attested": [1],
        "thermal_pad_native_thermal_conductivity_W_mK": [
            float(native_readback["thermal_conductivity_W_mK"])
        ],
        "thermal_pad_native_electrical_conductivity_S_m": [
            float(native_readback["electrical_conductivity_S_m"])
        ],
    }


@contextmanager
def _blocking_solve_log_heartbeat(project_name, design_name, setup_name):
    """Emit progress without making any AEDT calls from the helper thread."""

    raw_interval = os.environ.get(
        "MFT_AEDT_SOLVE_HEARTBEAT_SECONDS", "60"
    ).strip()
    try:
        interval_s = float(raw_interval)
    except (TypeError, ValueError, OverflowError):
        logging.warning(
            "[thermal] invalid MFT_AEDT_SOLVE_HEARTBEAT_SECONDS=%r; using 60",
            raw_interval,
        )
        interval_s = 60.0
    if not math.isfinite(interval_s):
        interval_s = 60.0
    interval_s = min(600.0, max(5.0, interval_s))

    started = time.monotonic()
    stop = threading.Event()

    def emit_heartbeat():
        while not stop.wait(interval_s):
            logging.warning(
                "[thermal] pooled blocking solve heartbeat elapsed_s=%.1f "
                "project=%s design=%s setup=%s",
                time.monotonic() - started,
                project_name,
                design_name,
                setup_name,
            )

    logging.warning(
        "[thermal] pooled blocking solve started project=%s design=%s setup=%s "
        "heartbeat_s=%.1f",
        project_name,
        design_name,
        setup_name,
        interval_s,
    )
    worker = threading.Thread(
        target=emit_heartbeat,
        name="mft-thermal-solve-heartbeat",
        daemon=True,
    )
    worker.start()
    try:
        yield
    except BaseException:
        logging.exception(
            "[thermal] pooled blocking solve raised after %.1fs project=%s "
            "design=%s setup=%s",
            time.monotonic() - started,
            project_name,
            design_name,
            setup_name,
        )
        raise
    finally:
        stop.set()
        worker.join(timeout=1.0)
        logging.warning(
            "[thermal] pooled blocking solve ended elapsed_s=%.1f project=%s "
            "design=%s setup=%s",
            time.monotonic() - started,
            project_name,
            design_name,
            setup_name,
        )


def _rx_insulation_native_readback(materials, expected_thermal_k):
    """Fail closed unless AEDT reports the explicit Rx insulation material."""
    expected = float(expected_thermal_k)
    if not math.isfinite(expected) or expected <= 0.0:
        raise ValueError(
            "winding_insulation expected thermal conductivity must be "
            f"finite and > 0, got {expected_thermal_k!r}"
        )
    props = _raw_aedt_material_props(
        materials, RX_EXPLICIT_INSULATION_MATERIAL
    )
    thermal_k = _aedt_number(
        props.get("thermal_conductivity"),
        f"{RX_EXPLICIT_INSULATION_MATERIAL}.thermal_conductivity",
    )
    electrical_sigma = _aedt_number(
        props.get("conductivity"),
        f"{RX_EXPLICIT_INSULATION_MATERIAL}.conductivity",
    )
    if not math.isclose(
        thermal_k, expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "native winding_insulation thermal_conductivity mismatch: "
            f"{thermal_k!r} != {expected!r}"
        )
    if not math.isclose(
        electrical_sigma, 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "native winding_insulation conductivity mismatch: "
            f"{electrical_sigma!r} != 0.0"
        )
    return {
        "thermal_conductivity_W_mK": thermal_k,
        "electrical_conductivity_S_m": electrical_sigma,
    }


def _rx_insulation_result_metadata(material_readback, counts):
    """Emit topology and native-material evidence for explicit Rx insulation."""
    normalized = {
        key: int(counts.get(key, 0)) for key in _RX_INSULATION_KEYS
    }
    if any(value < 0 for value in normalized.values()):
        raise RuntimeError(
            f"invalid explicit Rx insulation counts: {normalized}"
        )
    total = sum(normalized.values())
    native = (
        material_readback.get("rx_explicit_insulation")
        if isinstance(material_readback, dict) else None
    )
    if total and not isinstance(native, dict):
        raise RuntimeError(
            "explicit Rx insulation exists without native material readback"
        )
    return {
        "thermal_rx_explicit_insulation_model": [
            "solid_interturn_candidate_k_ins_v1"
            if total else "not_required_no_adjacent_explicit_foils_v1"
        ],
        "thermal_rx_explicit_insulation_policy": [
            RX_EXPLICIT_INSULATION_POLICY
        ],
        "thermal_rx_explicit_insulation_material": [
            RX_EXPLICIT_INSULATION_MATERIAL if total else ""
        ],
        "thermal_rx_explicit_insulation_count": [total],
        "thermal_rx_explicit_insulation_counts_json": [
            json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        ],
        "thermal_rx_explicit_insulation_native_readback_contract_version": [
            RX_EXPLICIT_INSULATION_NATIVE_READBACK_CONTRACT_VERSION
        ],
        "thermal_rx_explicit_insulation_native_readback_attested": [
            1 if total else 0
        ],
        "thermal_rx_explicit_insulation_native_thermal_conductivity_W_mK": [
            float(native["thermal_conductivity_W_mK"])
            if total else float("nan")
        ],
        "thermal_rx_explicit_insulation_native_electrical_conductivity_S_m": [
            float(native["electrical_conductivity_S_m"])
            if total else float("nan")
        ],
    }


def _field_summary_data_frame(sim, field_summary, setup):
    """Return field-summary data through a host-visible export path.

    PyAEDT's ``get_field_summary_data`` creates its CSV in the caller's local
    ``/tmp``.  With a pooled Desktop the caller and AEDT can run on different
    nodes, so AEDT writes a different node-local file and the caller reads its
    still-empty placeholder.  Export pooled summaries through the lease's
    task-unique GPFS workspace instead.  The file must be writable by the
    session-host account, which can differ from the client account.
    """
    from module.aedt_pool_adapter import pooled_backend_enabled

    if not pooled_backend_enabled():
        return field_summary.get_field_summary_data(
            setup=setup, pandas_output=True
        )

    create_target = getattr(sim, "_new_aedt_export_target", None)
    read_target = getattr(sim, "_read_attested_aedt_export", None)
    remove_target = getattr(sim, "_remove_attested_aedt_export", None)
    if not all(callable(item) for item in (
            create_target, read_target, remove_target)):
        raise RuntimeError(
            "pooled field summary requires attested AEDT export helpers"
        )

    output_path = None
    provenance = None
    try:
        output_path, provenance = create_target("thermal_field_summary")
        started = time.time()
        exported = field_summary.export_csv(output_path, setup=setup)
        if exported is False:
            raise RuntimeError("pooled field-summary export returned False")
        exported_text = read_target(
            output_path,
            provenance,
            started,
            "thermal_field_summary",
        )
        with io.StringIO(exported_text, newline="") as stream:
            for _ in range(4):
                if stream.readline() == "":
                    raise RuntimeError("pooled field-summary export is empty")
            frame = pd.DataFrame(list(csv.DictReader(stream)))
        for column in ("Min", "Max", "Mean", "Stdev", "Total"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame
    finally:
        if output_path and provenance:
            remove_target(
                output_path,
                provenance,
                "thermal_field_summary",
            )


@contextmanager
def _pooled_field_summary_window(
    sim,
    ipk,
    setup,
    timeout_s=7200.0,
    poll_s=2.0,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    """Yield an idle, exact thermal design while keeping one short lock.

    ``run_thermal_analysis`` owns an outer automation transaction.  A pooled
    Desktop can remain busy for many minutes while sibling projects solve, so
    holding that transaction during the drain starves those siblings and can
    itself trigger the automation-lock timeout.  Suspend the outer depth, take
    short poll transactions, and release each busy observation immediately.

    Once idle is observed, yield *inside that same inner transaction*.  The
    field-summary and scalar fallback therefore run without an idle-check to
    export race, while no lock is held during the potentially long drain.
    Desktop-wide running state is only this export barrier; it never attests
    the current lease's solve success.
    """
    from module.aedt_pool_adapter import pooled_backend_enabled

    if not pooled_backend_enabled():
        yield ipk
        return

    native_window = getattr(sim, "aedt_native_solve_window", None)
    automation_transaction = getattr(sim, "aedt_automation_transaction", None)
    if not callable(native_window) or not callable(automation_transaction):
        raise RuntimeError(
            "pooled thermal field-summary window has no lock discipline"
        )

    started = clock()
    deadline = started + max(0.0, float(timeout_s))
    waiting_logged = False
    with native_window():
        while True:
            with automation_transaction():
                postflight = _prepare_thermal_dispatch(
                    sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
                    setup_name=_THERMAL_SETUP_NAME,
                )
                native_ipk = postflight["native_ipk"]
                desktop = _thermal_desktop_handle(sim, native_ipk)
                is_running = getattr(
                    desktop, "AreThereSimulationsRunning", None
                )
                if not callable(is_running):
                    raise RuntimeError(
                        "pooled thermal field-summary drain has no "
                        "simulation-state query"
                    )
                value = is_running()
                if value is False or value == 0:
                    running = False
                elif value is True or value == 1:
                    running = True
                else:
                    normalized = str(value or "").strip().casefold()
                    if normalized in {"false", "no", "off", "0"}:
                        running = False
                    elif normalized in {"true", "yes", "on", "1"}:
                        running = True
                    else:
                        raise RuntimeError(
                            "pooled thermal field-summary drain returned an "
                            f"unrecognized state: {value!r}"
                        )
                if not running:
                    elapsed = max(0.0, clock() - started)
                    if waiting_logged:
                        logging.warning(
                            "[thermal] pooled field-summary drain completed "
                            "in %.1fs",
                            elapsed,
                        )
                    # Deliberately yield before leaving this transaction.
                    yield native_ipk
                    return

            if not waiting_logged:
                logging.warning(
                    "[thermal] waiting for in-flight sibling solves before "
                    "Desktop-global field-summary export"
                )
                waiting_logged = True
            now = clock()
            if now >= deadline:
                raise RuntimeError(
                    "timed out waiting for pooled AEDT to become idle before "
                    f"thermal field-summary export ({float(timeout_s):.1f}s)"
                )
            sleeper(min(
                max(0.05, float(poll_s)),
                max(0.0, deadline - now),
            ))


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


def _cooling_plate_mesh_assemblies(objs):
    """Return exact physical cooling-plate assemblies for local refinement.

    A transformer-wide pad operation makes its refinement box span distant
    core and winding plates.  Group each aluminum plate with only its two
    adjacent TIM solids instead, while meshing every controlled object
    separately so none of the thin pads can disappear from a shared bounding
    region.  This mesh control does not change the model's thermal contacts.
    """

    specs = (
        {
            "kind": "core_plate",
            "plates": objs.get("core_plates", []),
            "pads": objs.get("core_pads", []),
            "plate": re.compile(
                r"^core_plate_(?P<index>\d+)_"
                r"(?P<side>side_left|center|side_right)$"
            ),
            "pad": re.compile(
                r"^core_plate_pad_(?P<index>\d+)_(?P<layer>a|b)_"
                r"(?P<side>side_left|center|side_right)$"
            ),
            "layers": ("a", "b"),
            # Core TIMs were all represented in the failed replicas.  Preserve
            # their proven level while removing the giant global pad box.
            "level": 2,
        },
        {
            "kind": "wcp",
            "plates": objs.get("wcp_plates", []),
            "pads": objs.get("wcp_pads", []),
            "plate": re.compile(
                r"^Tx_main_wcp_(?P<index>\d+)_(?P<side>[pn])$"
            ),
            "pad": re.compile(
                r"^Tx_main_wcp_pad_(?P<index>\d+)_"
                r"(?P<layer>in|out)_(?P<side>[pn])$"
            ),
            "layers": ("in", "out"),
            # The candidate's eight 1 mm winding TIMs disappeared at level 2.
            # Level 5 is bounded to four local plate assemblies.
            "level": 5,
        },
    )
    assemblies = []
    for spec in specs:
        plates = list(spec["plates"])
        pads = list(spec["pads"])
        if not pads:
            continue
        by_key = {}
        for obj in plates:
            name = str(obj.name)
            match = spec["plate"].fullmatch(name)
            if not match:
                raise RuntimeError(
                    f"unexpected {spec['kind']} plate name: {name!r}"
                )
            key = (int(match.group("index")), match.group("side"))
            entry = by_key.setdefault(
                key, {"plate": None, "pads": {}}
            )
            if entry["plate"] is not None:
                raise RuntimeError(
                    f"duplicate {spec['kind']} plate assembly: {key!r}"
                )
            entry["plate"] = obj
        for obj in pads:
            name = str(obj.name)
            match = spec["pad"].fullmatch(name)
            if not match:
                raise RuntimeError(
                    f"unexpected {spec['kind']} pad name: {name!r}"
                )
            key = (int(match.group("index")), match.group("side"))
            layer = match.group("layer")
            entry = by_key.setdefault(
                key, {"plate": None, "pads": {}}
            )
            if layer in entry["pads"]:
                raise RuntimeError(
                    f"duplicate {spec['kind']} pad {key!r}/{layer!r}"
                )
            entry["pads"][layer] = obj
        expected_layers = set(spec["layers"])
        for key in sorted(by_key, key=lambda item: (item[0], item[1])):
            entry = by_key[key]
            actual_layers = set(entry["pads"])
            if entry["plate"] is None or actual_layers != expected_layers:
                raise RuntimeError(
                    f"incomplete {spec['kind']} assembly {key!r}: "
                    f"plate={entry['plate'] is not None}, "
                    f"pad_layers={sorted(actual_layers)!r}, "
                    f"expected={sorted(expected_layers)!r}"
                )
            index, side = key
            assemblies.append({
                "kind": spec["kind"],
                "name": (
                    f"{spec['kind']}_assembly_mesh_level_{index}_{side}"
                ),
                "level": int(spec["level"]),
                "objects": [
                    entry["plate"],
                    *(entry["pads"][layer] for layer in spec["layers"]),
                ],
            })
    return assemblies


def _thermal_mesh_length_mm(value, label):
    """Normalize one AEDT mesh-envelope length to millimetres."""
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a mesh length: {value!r}")
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s*([A-Za-z\u00b5]*)\s*",
        str(value),
    )
    if not match:
        raise ValueError(f"{label} is not a mesh length: {value!r}")
    number = float(match.group(1))
    unit = match.group(2).casefold().replace("\u00b5", "u")
    scale = {
        "": 1.0,
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "um": 1e-3,
        "micrometer": 1e-3,
        "micrometers": 1e-3,
        "cm": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
    }.get(unit)
    if scale is None or not math.isfinite(number):
        raise ValueError(
            f"{label} has unsupported/nonfinite units: {value!r}"
        )
    return number * scale


def _assign_thermal_mesh(ipk, objs, side_block_level=5):
    """Install assembly-local refinements resolved per controlled solid."""
    plan = []
    assigned = {}

    def _assign_levels(levels, name, category):
        if not levels:
            return
        normalized = {
            str(object_name): int(level)
            for object_name, level in levels.items()
        }
        for object_name in normalized:
            previous = assigned.get(object_name)
            if previous is not None:
                raise RuntimeError(
                    "thermal mesh object assigned to multiple operations: "
                    f"{object_name!r} in {previous!r} and {name!r}"
                )
        operation_names = ipk.mesh.assign_mesh_level(levels, name=name)
        if not isinstance(operation_names, (list, tuple)) or not operation_names:
            raise RuntimeError(f"{name} assignment returned no mesh operation")
        operations = {
            str(getattr(operation, "name", "")): operation
            for operation in getattr(ipk.mesh, "meshoperations", [])
        }
        actual_operation_names = []
        for item in operation_names:
            operation = item if callable(getattr(item, "update", None)) else operations.get(str(item))
            update = getattr(operation, "update", None)
            if not callable(update):
                raise RuntimeError(f"{name} mesh operation is unavailable: {item}")
            actual_operation_names.append(
                str(getattr(operation, "name", "") or item)
            )
            # C3 proved that one shared assembly region can retain the OO
            # assignment while creating zero cells for a subset of its thin
            # solids. Resolve the same bounded assembly operation per object;
            # the later native message scan still rejects any zero-cell solid.
            operation.auto_update = False
            # PyAEDT 0.22 exposes AEDT's read-only command metadata in the mesh
            # operation property bag.  Sending it back through ``update`` emits
            # ``Script macro error: Command is read only`` even though the mesh
            # operation itself is accepted.  Remove only that metadata field;
            # every writable mesh property remains subject to the readback below.
            for key in tuple(operation.props):
                if str(key).strip().casefold() == "command":
                    del operation.props[key]
            operation.props["Mesh Object(s) Separately Enabled"] = True
            if not update():
                raise RuntimeError(f"{name} mesh operation update failed: {item}")
            if operation.props.get("Mesh Object(s) Separately Enabled") is not True:
                raise RuntimeError(
                    f"{name} separate-object mesh setting was not retained: "
                    f"{item}"
                )
            readback_objects = operation.props.get("Objects", [])
            if isinstance(readback_objects, str):
                readback_objects = [readback_objects]
            if set(map(str, readback_objects or [])) != set(normalized):
                raise RuntimeError(
                    f"{name} object readback mismatch: "
                    f"{list(readback_objects or [])!r} != "
                    f"{list(normalized)!r}"
                )
            readback_level = operation.props.get("Level")
            expected_levels = set(normalized.values())
            if len(expected_levels) != 1 or str(readback_level) != str(
                next(iter(expected_levels))
            ):
                raise RuntimeError(
                    f"{name} level readback mismatch: "
                    f"{readback_level!r} != {sorted(expected_levels)!r}"
                )
        assigned.update({object_name: name for object_name in normalized})
        plan.append({
            "name": str(name),
            "category": str(category),
            "operation_type": "object_level",
            "level": next(iter(set(normalized.values()))),
            "objects": sorted(normalized),
            "shared_region": False,
            "separate_objects": True,
            "actual_operation_names": actual_operation_names,
        })

    def _assign_single_object_region(obj, name, category):
        object_name = str(obj.name)
        previous = assigned.get(object_name)
        if previous is not None:
            raise RuntimeError(
                "thermal mesh object assigned to multiple controls: "
                f"{object_name!r} in {previous!r} and {name!r}"
            )
        assign_region = getattr(ipk.mesh, "assign_mesh_region", None)
        if not callable(assign_region):
            raise RuntimeError(
                "Icepak per-object mesh-region API is unavailable"
            )
        region = assign_region(
            assignment=[object_name],
            level=5,
            name=name,
        )
        if region is None or region is False:
            raise RuntimeError(
                f"{name} assignment returned no mesh region"
            )
        actual_name = str(getattr(region, "name", "") or "")
        if actual_name != str(name):
            raise RuntimeError(
                f"{name} mesh-region name mismatch: {actual_name!r}"
            )
        if getattr(region, "enable", None) is not True:
            raise RuntimeError(
                f"{name} mesh region is not enabled"
            )
        if getattr(region, "manual_settings", None) is not False:
            raise RuntimeError(
                f"{name} mesh region did not retain automatic settings"
            )
        settings = getattr(region, "settings", None)
        try:
            level_readback = settings["MeshRegionResolution"]
        except Exception as exc:
            raise RuntimeError(
                f"{name} mesh-region level readback is unavailable"
            ) from exc
        if str(level_readback) != "5":
            raise RuntimeError(
                f"{name} mesh-region level readback mismatch: "
                f"{level_readback!r} != 5"
            )
        assignment = getattr(region, "assignment", None)
        parts = getattr(assignment, "parts", None)
        if not isinstance(parts, dict):
            raise RuntimeError(
                f"{name} mesh region has no exact subregion-part readback"
            )
        part_names = sorted(map(str, parts))
        if part_names != [object_name]:
            raise RuntimeError(
                f"{name} mesh-region object readback mismatch: "
                f"{part_names!r} != {[object_name]!r}"
            )
        # PyAEDT creates a SubRegion with zero percentage padding by default.
        # Its six faces then coincide with the source pad and native Icepak
        # serializes ``OverlappingMRFaces[0:]``: the local mesh has domains but
        # no parent/global mesh interface.  Expand only the non-model mesh
        # envelope; this does not mutate the physical TIM geometry.
        expected_padding_types = [
            WCP_PAD_MESH_REGION_PADDING_TYPE
        ] * len(_WCP_PAD_MESH_REGION_DIRECTIONS)
        expected_padding_values = [
            f"{WCP_PAD_MESH_REGION_PADDING_MM:g}mm"
        ] * len(_WCP_PAD_MESH_REGION_DIRECTIONS)
        try:
            assignment.padding_types = list(expected_padding_types)
            assignment.padding_values = list(expected_padding_values)
            padding_types = [
                " ".join(str(value).split())
                for value in assignment.padding_types
            ]
            padding_values_mm = [
                _thermal_mesh_length_mm(
                    value, f"{name} {direction} padding"
                )
                for direction, value in zip(
                    _WCP_PAD_MESH_REGION_DIRECTIONS,
                    assignment.padding_values,
                )
            ]
        except Exception as exc:
            raise RuntimeError(
                f"{name} mesh-region padding update/readback failed"
            ) from exc
        if padding_types != expected_padding_types:
            raise RuntimeError(
                f"{name} mesh-region padding type mismatch: "
                f"{padding_types!r} != {expected_padding_types!r}"
            )
        if (
            len(padding_values_mm)
            != len(_WCP_PAD_MESH_REGION_DIRECTIONS)
            or any(
                not math.isclose(
                    value,
                    WCP_PAD_MESH_REGION_PADDING_MM,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for value in padding_values_mm
            )
        ):
            raise RuntimeError(
                f"{name} mesh-region padding value mismatch: "
                f"{padding_values_mm!r}"
            )
        update = getattr(region, "update", None)
        if not callable(update) or update() is not True:
            raise RuntimeError(
                f"{name} padded mesh-region update failed"
            )
        region_object_name = str(
            getattr(assignment, "name", "") or ""
        )
        if not region_object_name:
            raise RuntimeError(
                f"{name} mesh region has no native subregion identity"
            )
        assigned[object_name] = name
        plan.append({
            "name": str(name),
            "category": str(category),
            "operation_type": "mesh_region",
            "level": 5,
            "objects": [object_name],
            "shared_region": False,
            "separate_objects": None,
            "actual_operation_names": [actual_name],
            "native_region_object_name": region_object_name,
            "padding_types": expected_padding_types,
            "padding_values_mm": padding_values_mm,
        })

    assemblies = _cooling_plate_mesh_assemblies(objs)
    for assembly in assemblies:
        assembly_objects = list(assembly["objects"])
        if assembly["kind"] == "wcp":
            # B3 replaces the two pad assignments in each object-level
            # assembly with one padded MeshRegion per pad. Retain the plate
            # itself as a level-5 object operation, so every controlled solid
            # still has exactly one mesh control.
            assembly_objects = [
                obj for obj in assembly_objects
                if not str(obj.name).startswith("Tx_main_wcp_pad_")
            ]
            if len(assembly_objects) != 1:
                raise RuntimeError(
                    "B3 winding cooling-plate assembly does not contain "
                    f"exactly one plate: {assembly['name']!r}"
                )
        _assign_levels(
            {
                str(obj.name): int(assembly["level"])
                for obj in assembly_objects
            },
            assembly["name"],
            assembly["kind"],
        )

    for pad in sorted(
        objs.get("wcp_pads", []), key=lambda item: str(item.name)
    ):
        pad_name = str(pad.name)
        match = re.fullmatch(
            r"Tx_main_wcp_pad_(\d+)_(in|out)_([pn])",
            pad_name,
        )
        if not match:
            raise RuntimeError(
                f"unexpected B3 winding TIM name: {pad_name!r}"
            )
        index, layer, side = match.groups()
        _assign_single_object_region(
            pad,
            f"wcp_pad_mesh_region_{index}_{layer}_{side}",
            "wcp_pad_region",
        )

    # Tx turns as thin as the sampled 1 mm lower bound can disappear from a
    # multi-object cut-cell region. Level 4 plus per-object resolution preserves
    # those solid zones without changing the model's thermal contacts.
    tx_names = list(dict.fromkeys(obj.name for obj in objs.get("Tx", [])))
    _assign_levels(
        {name: 4 for name in tx_names},
        "tx_mesh_level",
        "tx_pack",
    )

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
        _assign_levels(
            {name: level for name in names},
            operation_name,
            key,
        )

    # C4 separate-object meshing reduced zero-cell solids from 35 to ten, but
    # both retained Rx_main turns still disappeared at level 3.  Keep all three
    # localized retained packs at level 5; their boxes remain pack-local.
    retained_pack_specs = (
        ("Rx_main_explicit", "rx_main_retained_pack_mesh_level", 5),
        ("Rx_side_explicit", "rx_side_retained_pack_mesh_level", 5),
        ("Rx_side2_explicit", "rx_side2_retained_pack_mesh_level", 5),
    )
    retained_pack_count = 0
    for explicit_key, operation_name, level in retained_pack_specs:
        group = list(objs.get(explicit_key, []))
        names = list(dict.fromkeys(
            str(obj.name) for obj in group
        ))
        if names:
            retained_pack_count += 1
        _assign_levels(
            {name: level for name in names},
            operation_name,
            explicit_key.replace("_explicit", "_retained_pack"),
        )

    # Explicit insulation keeps its previous level 4, but each pack receives a
    # local operation with per-object resolution.  It is intentionally not
    # folded into either level-3 main copper or level-5 side copper, which would
    # force one of the two physically distinct thin-solid classes to an
    # unsupported density.
    for insulation_key in _RX_INSULATION_KEYS:
        insulation = list(objs.get(insulation_key, []))
        explicit_key = insulation_key.replace("_insulation", "_explicit")
        if insulation and not objs.get(explicit_key, []):
            raise RuntimeError(
                f"{insulation_key} exists without retained copper"
            )
        names = list(dict.fromkeys(
            str(obj.name) for obj in insulation
        ))
        _assign_levels(
            {name: 4 for name in names},
            insulation_key.lower() + "_mesh_level",
            insulation_key,
        )

    required_thin = {
        str(obj.name)
        for key in (
            "core_pads",
            "wcp_pads",
            "Rx_main_explicit",
            "Rx_side_explicit",
            "Rx_side2_explicit",
            *_RX_INSULATION_KEYS,
        )
        for obj in objs.get(key, [])
    }
    required_objects_missing = sorted(required_thin - set(assigned))
    if required_objects_missing:
        raise RuntimeError(
            "thermal thin-solid mesh coverage is incomplete: "
            f"{required_objects_missing!r}"
        )
    core_assembly_count = sum(
        item["kind"] == "core_plate" for item in assemblies
    )
    wcp_assembly_count = sum(
        item["kind"] == "wcp" for item in assemblies
    )
    plan_payload = {
        "schema": THERMAL_MESH_PLAN_CONTRACT_VERSION,
        "policy": THERMAL_MESH_POLICY,
        "operations": plan,
        "operation_count": len(plan),
        "assigned_object_count": len(assigned),
        "required_thin_object_count": len(required_thin),
        "required_thin_objects": sorted(required_thin),
        "required_objects_missing": required_objects_missing,
        "core_plate_assembly_count": core_assembly_count,
        "wcp_assembly_count": wcp_assembly_count,
        "wcp_pad_mesh_region_count": sum(
            operation["category"] == "wcp_pad_region"
            for operation in plan
        ),
        "rx_retained_pack_count": retained_pack_count,
        "shared_operation_count": 0,
        "object_level_operation_count": sum(
            operation["operation_type"] == "object_level"
            for operation in plan
        ),
        "mesh_region_operation_count": sum(
            operation["operation_type"] == "mesh_region"
            for operation in plan
        ),
        "separate_object_operation_count": sum(
            operation["separate_objects"] is True
            for operation in plan
        ),
    }
    canonical_payload = {
        **plan_payload,
        "operations": [
            {
                key: value
                for key, value in operation.items()
                if key != "actual_operation_names"
            }
            for operation in plan
        ],
    }
    canonical = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":")
    )
    plan_payload["plan_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return plan_payload


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


def _thermal_mesh_artifact_candidates(sim, ipk):
    """Return complete native Icepak grid artifacts without recursive scans."""
    design_name = str(
        getattr(ipk, "design_name", "") or _THERMAL_DESIGN_NAME
    )
    candidates = {}
    for root in _thermal_monitor_roots(sim, ipk):
        design_results = root / f"{design_name}.results"
        search_root = design_results if design_results.is_dir() else root
        if not search_root.is_dir() or search_root.is_symlink():
            continue
        for mesh_dir in search_root.glob("*_Meshes*_V*.sd"):
            try:
                if not mesh_dir.is_dir() or mesh_dir.is_symlink():
                    continue
                output = mesh_dir / "grid_output"
                mapping = mesh_dir / "grid_mapping"
                if (
                    output.is_symlink()
                    or mapping.is_symlink()
                    or not output.is_file()
                    or not mapping.is_file()
                ):
                    continue
                output_signature = _thermal_monitor_signature(output)
                mapping_signature = _thermal_monitor_signature(mapping)
                if output_signature[0] <= 0 or mapping_signature[0] <= 0:
                    continue
                key = str(mesh_dir.resolve(strict=False)).casefold()
            except OSError:
                continue
            candidates[key] = {
                "path": mesh_dir,
                "grid_output": output_signature,
                "grid_mapping": mapping_signature,
                "mtime_ns": max(
                    output_signature[1], mapping_signature[1]
                ),
            }
    return candidates


def _snapshot_thermal_mesh_artifacts(sim, ipk):
    """Snapshot complete native grid artifacts before GenerateMesh."""
    return {
        key: (
            value["grid_output"][0],
            value["grid_output"][2],
            value["grid_mapping"][0],
            value["grid_mapping"][2],
        )
        for key, value in _thermal_mesh_artifact_candidates(
            sim, ipk
        ).items()
    }


def _fresh_thermal_mesh_artifacts(sim, ipk, snapshot):
    """Return only new or content-changed complete native grid artifacts."""
    fresh = []
    for key, value in _thermal_mesh_artifact_candidates(sim, ipk).items():
        signature = (
            value["grid_output"][0],
            value["grid_output"][2],
            value["grid_mapping"][0],
            value["grid_mapping"][2],
        )
        if snapshot.get(key) == signature:
            continue
        fresh.append({
            "directory": str(value["path"]),
            "name": value["path"].name,
            "grid_output_size": value["grid_output"][0],
            "grid_output_sha256_sample": value["grid_output"][2],
            "grid_mapping_size": value["grid_mapping"][0],
            "grid_mapping_sha256_sample": value["grid_mapping"][2],
            "mtime_ns": value["mtime_ns"],
        })
    fresh.sort(key=lambda item: (item["mtime_ns"], item["name"]))
    return fresh


def _parse_thermal_grid_mapping(path):
    """Read one native Icepak ``grid_mapping`` without trusting file presence.

    ``GenerateMesh`` can return ``True`` and publish large ``grid_output`` files
    even when a thin solid owns no domain or a local MeshRegion is disconnected
    from its parent.  The accompanying text ``grid_mapping`` is the native
    source of truth for both conditions.
    """

    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(
            f"native thermal grid_mapping is a symlink: {candidate}"
        )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(
            f"native thermal grid_mapping is not a file: {candidate}"
        )
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"native thermal grid_mapping is unreadable: {resolved}"
        ) from exc
    if not text.strip().startswith("$begin 'MeshRegion'"):
        raise RuntimeError(
            f"native thermal grid_mapping has no MeshRegion root: {resolved}"
        )

    objects_start = text.find("$begin 'Objects'")
    objects_end = text.find("$end 'Objects'", objects_start + 1)
    if objects_start < 0 or objects_end < 0:
        raise RuntimeError(
            f"native thermal grid_mapping has no complete Objects tree: "
            f"{resolved}"
        )
    header = text[:objects_start]
    objects_text = text[objects_start:objects_end]

    def _one(pattern, label, source=text):
        matches = list(re.finditer(pattern, source, re.MULTILINE))
        if len(matches) != 1:
            raise RuntimeError(
                "native thermal grid_mapping requires exactly one "
                f"{label}: {resolved}; count={len(matches)}"
            )
        return matches[0]

    region_name = _one(
        r"^\s*Name='(?P<value>(?:''|[^'])*)'\s*$",
        "root region name",
        header,
    ).group("value").replace("''", "'")
    parent_region = int(_one(
        r"^\s*ParentRegion=(?P<value>\d+)\s*$",
        "ParentRegion",
        header,
    ).group("value"))
    has_mesh_value = _one(
        r"^\s*HasMesh=(?P<value>true|false)\s*$",
        "HasMesh",
        header,
    ).group("value").casefold()
    object_count = int(_one(
        r"^\s*Count=(?P<value>\d+)\s*$",
        "Objects Count",
        objects_text,
    ).group("value"))

    object_blocks = list(re.finditer(
        r"\$begin 'Object(?P<index>\d+)'\s*"
        r"(?P<body>.*?)"
        r"\$end 'Object(?P=index)'",
        objects_text,
        re.DOTALL,
    ))
    if len(object_blocks) != object_count:
        raise RuntimeError(
            "native thermal grid_mapping Objects count mismatch: "
            f"{resolved}; parsed={len(object_blocks)}, "
            f"declared={object_count}"
        )
    object_domains = {}
    for block in object_blocks:
        body = block.group("body")
        name = _one(
            r"^\s*Name='(?P<value>(?:''|[^'])*)'\s*$",
            f"Object{block.group('index')} Name",
            body,
        ).group("value").replace("''", "'")
        domains = _one(
            r"^\s*Domains\[(?P<count>\d+):(?P<values>[^\]]*)\]\s*$",
            f"Object{block.group('index')} Domains",
            body,
        )
        declared = int(domains.group("count"))
        raw_values = [
            item.strip()
            for item in domains.group("values").split(",")
            if item.strip()
        ]
        if len(raw_values) != declared:
            raise RuntimeError(
                "native thermal grid_mapping domain count mismatch: "
                f"{resolved}; object={name!r}, parsed={len(raw_values)}, "
                f"declared={declared}"
            )
        if name in object_domains:
            raise RuntimeError(
                "native thermal grid_mapping has duplicate object: "
                f"{resolved}; object={name!r}"
            )
        object_domains[name] = declared

    overlaps = _one(
        r"^\s*OverlappingMRFaces\[(?P<count>\d+):"
        r"(?P<values>[^\]]*)\]\s*$",
        "OverlappingMRFaces",
    )
    overlap_count = int(overlaps.group("count"))
    overlap_values = [
        item.strip()
        for item in overlaps.group("values").split(",")
        if item.strip()
    ]
    if len(overlap_values) != overlap_count:
        raise RuntimeError(
            "native thermal grid_mapping overlap count mismatch: "
            f"{resolved}; parsed={len(overlap_values)}, "
            f"declared={overlap_count}"
        )

    return {
        "schema": "thermal-grid-mapping-readback-v1",
        "region_name": region_name,
        "parent_region": parent_region,
        "has_mesh": has_mesh_value == "true",
        "object_count": object_count,
        "objects_with_domains": sorted(
            name for name, count in object_domains.items() if count > 0
        ),
        "object_domain_value_counts": {
            name: object_domains[name] for name in sorted(object_domains)
        },
        "overlapping_mr_face_count": overlap_count,
    }


def _thermal_mesh_mapping_coverage(mesh_plan, fresh_artifacts):
    """Attest solid domains and parent coupling in fresh native mesh maps."""

    required_objects = sorted(set(map(
        str, mesh_plan.get("required_thin_objects", [])
    )))
    expected_regions = {
        str(operation["name"]): sorted(set(map(
            str, operation.get("objects", [])
        )))
        for operation in mesh_plan.get("operations", [])
        if operation.get("operation_type") == "mesh_region"
    }
    readbacks = []
    errors = []
    for artifact in fresh_artifacts:
        directory = Path(str(artifact.get("directory", "")))
        mapping = directory / "grid_mapping"
        try:
            if directory.is_symlink():
                raise RuntimeError(
                    f"native thermal mesh artifact directory is a symlink: "
                    f"{directory}"
                )
            resolved_directory = directory.resolve(strict=True)
            resolved_mapping = mapping.resolve(strict=True)
            if resolved_mapping.parent != resolved_directory:
                raise RuntimeError(
                    "native thermal grid_mapping escaped its mesh directory"
                )
            signature = _thermal_monitor_signature(resolved_mapping)
            if (
                signature[0]
                != int(artifact.get("grid_mapping_size", -1))
                or signature[2]
                != str(artifact.get("grid_mapping_sha256_sample", ""))
            ):
                raise RuntimeError(
                    "native thermal grid_mapping changed after idle barrier"
                )
            readback = _parse_thermal_grid_mapping(resolved_mapping)
            readbacks.append({
                **readback,
                "artifact_name": str(artifact.get("name", "")),
                "grid_mapping_size": signature[0],
                "grid_mapping_sha256_sample": signature[2],
            })
        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {str(exc)[:1000]}"
            )

    mapped_objects = sorted({
        name
        for readback in readbacks
        if readback["has_mesh"]
        for name in readback["objects_with_domains"]
    })
    required_missing = sorted(set(required_objects) - set(mapped_objects))
    global_regions = [
        item for item in readbacks if item["parent_region"] == 0
    ]
    region_missing = sorted(
        name for name in expected_regions
        if not any(item["region_name"] == name for item in readbacks)
    )
    region_without_mesh = sorted({
        name for name in expected_regions
        for item in readbacks
        if item["region_name"] == name and item["has_mesh"] is not True
    })
    region_uncoupled = sorted({
        name for name in expected_regions
        for item in readbacks
        if (
            item["region_name"] == name
            and (
                item["parent_region"] <= 0
                or item["overlapping_mr_face_count"] <= 0
            )
        )
    })
    region_objects_missing = sorted({
        object_name
        for region_name, object_names in expected_regions.items()
        for item in readbacks
        if item["region_name"] == region_name
        for object_name in object_names
        if object_name not in item["objects_with_domains"]
    })
    passed = (
        bool(readbacks)
        and not errors
        and bool(global_regions)
        and all(item["has_mesh"] is True for item in global_regions)
        and not required_missing
        and not region_missing
        and not region_without_mesh
        and not region_uncoupled
        and not region_objects_missing
    )
    return {
        "schema": "thermal-grid-mapping-coverage-v1",
        "passed": passed,
        "fresh_grid_mapping_count": len(readbacks),
        "parse_errors": errors,
        "global_region_count": len(global_regions),
        "required_object_count": len(required_objects),
        "mapped_required_object_count": (
            len(required_objects) - len(required_missing)
        ),
        "required_objects_missing": required_missing,
        "expected_local_region_count": len(expected_regions),
        "missing_local_regions": region_missing,
        "local_regions_without_mesh": region_without_mesh,
        "uncoupled_local_regions": region_uncoupled,
        "local_region_objects_missing": region_objects_missing,
        "readbacks": readbacks,
    }


def _unmeshed_objects_from_messages(messages):
    """Extract exact native objects that Icepak reported without mesh."""
    found = []
    unparsed = False
    for message in messages or []:
        value = str(message)
        matches = list(_THERMAL_UNMESHED_OBJECT.finditer(value))
        found.extend(match.group("object") for match in matches)
        if "object does not have mesh" in value.casefold() and not matches:
            unparsed = True
    if unparsed:
        found.append("<unparsed-native-unmeshed-object>")
    return sorted(set(found))


def _thermal_terminal_solver_messages(messages):
    """Return exact native Icepak terminal markers not covered by severity."""
    markers = (
        "failed to run solver",
        "simulation completed with execution error",
    )
    return list(dict.fromkeys(
        str(message)
        for message in messages or []
        if any(marker in str(message).casefold() for marker in markers)
    ))


def _bounded_message_values(messages, limit=128, char_limit=32768):
    """Bound stored forensic text after scanning the complete fresh suffix."""
    values = list(messages or [])[-max(0, int(limit)):]
    bounded = []
    remaining = max(0, int(char_limit))
    for message in values:
        if remaining <= 0:
            break
        value = str(message).replace("\r", " ").replace("\n", " ")
        value = value[:min(2048, remaining)]
        bounded.append(value)
        remaining -= len(value)
    return bounded


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


def _thermal_running_state(sim, ipk, desktop=None):
    # Desktop-wide running state is meaningless on a shared pooled AEDT
    # session (sibling clients solve concurrently); callers treat this
    # exception as "no evidence" rather than a false positive.
    from module.aedt_pool_adapter import pooled_backend_enabled
    if pooled_backend_enabled():
        raise RuntimeError(
            "Desktop-wide simulation state is not meaningful on a shared "
            "pooled AEDT session"
        )
    # Capture the raw Desktop proxy before Analyze. PyAEDT can invalidate or
    # clear its wrapper-side Desktop attributes while restoring the global DSO
    # configuration even though the native Icepak engine is still running.
    # A caller-supplied proxy therefore remains the preferred completion
    # barrier handle across the Analyze return boundary.
    if desktop is None:
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
    from module.aedt_pool_adapter import pooled_backend_enabled

    pooled_backend = pooled_backend_enabled()
    rebind = getattr(sim, "_rebind_native_project_for_design_creation", None)
    if not callable(rebind):
        raise RuntimeError("thermal project rebind is unavailable")
    native_project = rebind()
    if native_project is None or native_project is False:
        raise RuntimeError("thermal project rebind returned no native project")
    if pooled_backend:
        desktop = _thermal_desktop_handle(sim, ipk)
        set_active_project = getattr(desktop, "SetActiveProject", None)
        if not callable(set_active_project):
            raise RuntimeError(
                "pooled thermal Desktop cannot activate the exact project"
            )
        native_project = set_active_project(expected_project)
        if native_project is None or native_project is False:
            raise RuntimeError(
                f"SetActiveProject returned no project ({expected_project})"
            )
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
    setup_child = None
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

    setup_control_readback = _thermal_setup_control_readback(
        setup, setup_child
    )

    if not pooled_backend:
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
        "setup_control_readback": setup_control_readback,
        "native_ipk": native_ipk,
        "native_design": native_design,
    }


def _thermal_setup_control_readback(setup, native_setup_child=None):
    """Attest the unchanged Icepak controls from wrapper and native OO state."""

    props = getattr(setup, "props", None)
    if not hasattr(props, "get"):
        raise RuntimeError("ThermalSetup control wrapper readback is absent")
    try:
        max_iterations = int(
            props.get("Convergence Criteria - Max Iterations")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "ThermalSetup maximum-iteration readback is invalid"
        ) from exc
    if max_iterations <= 0:
        raise RuntimeError(
            "ThermalSetup maximum-iteration readback is not positive"
        )
    expected = {
        "Flow Regime": "Turbulent",
        "Convergence Criteria - Max Iterations": max_iterations,
        "Convergence Criteria - Flow": 0.001,
        "Convergence Criteria - Energy": 1e-7,
        "Solution Initialization - Use Model Based Flow Initialization": False,
        "Under-relaxation - Pressure": 0.7,
        "Sequential Solve of Flow and Energy Equations": False,
        "Include Gravity": False,
    }
    def normalize(name, value):
        wanted = expected[name]
        if isinstance(wanted, bool):
            return _thermal_bool(value)
        if isinstance(wanted, str):
            return str(value or "").strip()
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"ThermalSetup control {name!r} is nonnumeric: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise RuntimeError(
                f"ThermalSetup control {name!r} is nonfinite"
            )
        return int(number) if isinstance(wanted, int) else number

    wrapper = {}
    for name, wanted in expected.items():
        observed = normalize(name, props.get(name))
        if observed != wanted:
            raise RuntimeError(
                "ThermalSetup wrapper control drifted: "
                f"{name}={observed!r}, expected={wanted!r}"
            )
        wrapper[name] = observed

    native = {}
    native_missing = []
    get_value = getattr(native_setup_child, "GetPropValue", None)
    get_names = getattr(native_setup_child, "GetPropNames", None)
    native_property_names = set()
    native_property_inventory_available = False
    if callable(get_names):
        try:
            native_property_names = {
                str(name) for name in (get_names() or [])
            }
            native_property_inventory_available = True
        except Exception:
            native_property_names = set()
    for name, wanted in expected.items():
        if (
            not callable(get_value)
            or not native_property_inventory_available
            or name not in native_property_names
        ):
            native_missing.append(name)
            continue
        try:
            raw_value = get_value(name)
            if raw_value is None or (
                isinstance(raw_value, str) and not raw_value.strip()
            ):
                native_missing.append(name)
                continue
            observed = normalize(name, raw_value)
        except Exception:
            native_missing.append(name)
            continue
        if observed != wanted:
            raise RuntimeError(
                "native ThermalSetup control drifted: "
                f"{name}={observed!r}, expected={wanted!r}"
            )
        native[name] = observed
    return {
        "schema": THERMAL_SETUP_CONTROL_READBACK_CONTRACT_VERSION,
        "expected": expected,
        "wrapper": wrapper,
        "native": native,
        "native_missing": native_missing,
        "native_property_inventory_available": (
            native_property_inventory_available
        ),
        "wrapper_passed": True,
        "native_complete": not native_missing,
        "mesh_quality_checks_modified": False,
        "mesh_quality_check_disable_requested": False,
    }


def _native_mesh_assignment_names(value, editor=None):
    """Normalize AEDT OO ``Assignment``/``Parts`` readback to object names."""
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = []
    else:
        text = str(value).strip()
        if not text:
            values = []
        else:
            if text[:1] in "[(" and text[-1:] in "])":
                text = text[1:-1]
            values = re.split(r"\s*[,;]\s*", text)
    names = {
        str(item).strip().strip("'\"")
        for item in values
        if str(item).strip().strip("'\"")
    }
    get_name = getattr(editor, "GetObjectNameByID", None)
    if callable(get_name):
        converted = set()
        for name in names:
            try:
                converted.add(str(get_name(int(name))))
            except (TypeError, ValueError, OverflowError):
                converted.add(name)
        names = converted
    return sorted(names)


def _native_mesh_property_name(prop_names, *candidates):
    by_normalized = {
        re.sub(r"[^a-z0-9]", "", str(name).casefold()): str(name)
        for name in prop_names
    }
    for candidate in candidates:
        actual = by_normalized.get(
            re.sub(r"[^a-z0-9]", "", str(candidate).casefold())
        )
        if actual is not None:
            return actual
    return ""


def _native_mesh_integer_readback(get_prop_value, prop_name, operation_name):
    """Read one native mesh integer without accepting truncation or booleans."""
    try:
        raw_value = get_prop_value(prop_name)
    except Exception as exc:
        raise RuntimeError(
            "native thermal mesh integer property readback failed: "
            f"{operation_name!r} {prop_name!r}"
        ) from exc
    if isinstance(raw_value, bool):
        raise RuntimeError(
            "native thermal mesh integer property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    try:
        numeric_value = float(str(raw_value).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "native thermal mesh integer property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        ) from exc
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise RuntimeError(
            "native thermal mesh integer property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    return int(numeric_value)


def _native_mesh_region_part_names(native_editor, region_object_name):
    """Resolve one native CreateSubRegion object to its exact source parts."""

    get_child = getattr(native_editor, "GetChildObject", None)
    if not callable(get_child):
        raise RuntimeError(
            "native thermal editor has no subregion object-tree readback"
        )
    region_child = get_child(str(region_object_name))
    if region_child is None or region_child is False:
        raise RuntimeError(
            "native thermal mesh subregion object is missing: "
            f"{region_object_name!r}"
        )
    candidates = [region_child]
    get_history_names = getattr(region_child, "GetChildNames", None)
    if callable(get_history_names):
        history_names = tuple(str(name) for name in (
            get_history_names() or []
        ))
        create_names = [
            name for name in history_names
            if name.casefold().startswith("createsubregion")
        ]
        if len(create_names) != 1:
            raise RuntimeError(
                "native thermal mesh subregion lacks one CreateSubRegion "
                f"history: {region_object_name!r} -> {history_names!r}"
            )
        get_history_child = getattr(region_child, "GetChildObject", None)
        if not callable(get_history_child):
            raise RuntimeError(
                "native thermal mesh subregion has no history readback"
            )
        history_child = get_history_child(create_names[0])
        if history_child is None or history_child is False:
            raise RuntimeError(
                "native thermal mesh CreateSubRegion history disappeared: "
                f"{region_object_name!r}"
            )
        candidates.insert(0, history_child)

    for candidate in candidates:
        get_prop_names = getattr(candidate, "GetPropNames", None)
        get_prop_value = getattr(candidate, "GetPropValue", None)
        if not callable(get_prop_names) or not callable(get_prop_value):
            continue
        prop_names = tuple(str(name) for name in (
            get_prop_names() or []
        ))
        parts_prop = _native_mesh_property_name(
            prop_names, "Part Names", "Parts", "Objects"
        )
        if not parts_prop:
            continue
        raw_parts = get_prop_value(parts_prop)
        if isinstance(raw_parts, str):
            raw_parts = [
                item.strip()
                for item in raw_parts.split(",")
                if item.strip()
            ]
        return _native_mesh_assignment_names(
            raw_parts, editor=native_editor
        )
    raise RuntimeError(
        "native thermal mesh subregion has no exact part readback: "
        f"{region_object_name!r}"
    )


def _native_mesh_number_readback(
    get_prop_value, prop_name, operation_name
):
    try:
        raw_value = get_prop_value(prop_name)
        numeric_value = float(str(raw_value).strip())
    except Exception as exc:
        raise RuntimeError(
            "native thermal mesh numeric property readback failed: "
            f"{operation_name!r} {prop_name!r}"
        ) from exc
    if not math.isfinite(numeric_value):
        raise RuntimeError(
            "native thermal mesh numeric property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    return numeric_value


def _native_mesh_length_mm_readback(
    get_prop_value, prop_name, operation_name
):
    try:
        raw_value = get_prop_value(prop_name)
    except Exception as exc:
        raise RuntimeError(
            "native thermal mesh length property readback failed: "
            f"{operation_name!r} {prop_name!r}"
        ) from exc
    if isinstance(raw_value, bool):
        raise RuntimeError(
            "native thermal mesh length property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s*([A-Za-z\u00b5]*)\s*",
        str(raw_value),
    )
    if not match:
        raise RuntimeError(
            "native thermal mesh length property readback is invalid: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    value = float(match.group(1))
    unit = match.group(2).casefold().replace("\u00b5", "u")
    scale = {
        "": 1.0,
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "um": 1e-3,
        "micrometer": 1e-3,
        "micrometers": 1e-3,
        "cm": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
    }.get(unit)
    if scale is None or not math.isfinite(value):
        raise RuntimeError(
            "native thermal mesh length unit is unsupported: "
            f"{operation_name!r} {prop_name!r}={raw_value!r}"
        )
    return value * scale


def _native_thermal_mesh_region_readback(
    mesh_child, native_names, mesh_plan
):
    """Require each WCP pad's manual anisotropic MeshRegion in native OO."""
    regions = list(mesh_plan.get("mesh_regions", []))
    if len(regions) != int(mesh_plan.get("mesh_region_count", -1)):
        raise RuntimeError(
            "thermal mesh plan has inconsistent MeshRegion count"
        )
    expected_names = []
    by_name = {}
    for region in regions:
        actual_name = str(region.get("actual_region_name", ""))
        if not actual_name or actual_name in by_name:
            raise RuntimeError(
                "thermal mesh plan has invalid MeshRegion identities"
            )
        expected_names.append(actual_name)
        by_name[actual_name] = region
    missing = sorted(set(expected_names) - set(native_names))
    if missing:
        raise RuntimeError(
            "native thermal MeshRegion readback is incomplete: "
            f"{missing!r}"
        )

    required_props = {
        "assignment": ("Assignment", "Objects", "Parts"),
        "enabled": ("Enabled", "Enable"),
        "enclosing": ("Enclosing Geometries",),
        "max_x": (
            "MaxElementSizeX",
            "Max Element Size X",
            "Maximum Element Size/X",
        ),
        "max_y": (
            "MaxElementSizeY",
            "Max Element Size Y",
            "Maximum Element Size/Y",
        ),
        "max_z": (
            "MaxElementSizeZ",
            "Max Element Size Z",
            "Maximum Element Size/Z",
        ),
        "min_elements_gap": (
            "MinElementsInGap",
            "Minimum Elements in Gap",
            "Mesh Parameters/Min Elements in Gap",
        ),
        "min_elements_edge": (
            "MinElementsOnEdge",
            "Minimum Elements on Edge",
            "Mesh Parameters/Min Elements on Edge",
        ),
        "max_ratio": (
            "MaxSizeRatio",
            "Maximum Size Ratio",
            "Mesh Parameters/Max Size Ratio",
        ),
        "min_gap_x": (
            "MinGapX", "Minimum Gap X", "Minimum Gap/X"
        ),
        "min_gap_y": (
            "MinGapY", "Minimum Gap Y", "Minimum Gap/Y"
        ),
        "min_gap_z": (
            "MinGapZ", "Minimum Gap Z", "Minimum Gap/Z"
        ),
        "uniform_type": (
            "UniformMeshParametersType",
            "Uniform Mesh Parameters Type",
            "Mesh Parameters/Uniform Mesh Parameters",
        ),
    }
    readbacks = []
    assignment_union = set()
    for actual_name in expected_names:
        region = by_name[actual_name]
        child = mesh_child.GetChildObject(actual_name)
        if child is None or child is False:
            raise RuntimeError(
                "native thermal MeshRegion child disappeared: "
                f"{actual_name!r}"
            )
        get_prop_names = getattr(child, "GetPropNames", None)
        get_prop_value = getattr(child, "GetPropValue", None)
        if not callable(get_prop_names) or not callable(get_prop_value):
            raise RuntimeError(
                "native thermal MeshRegion has no OO property readback: "
                f"{actual_name!r}"
            )
        prop_names = tuple(str(name) for name in (get_prop_names() or []))
        props = {
            key: _native_mesh_property_name(prop_names, *candidates)
            for key, candidates in required_props.items()
        }
        missing_props = sorted(
            key for key, value in props.items() if not value
        )
        if missing_props:
            raise RuntimeError(
                "native thermal MeshRegion lacks required OO properties: "
                f"{actual_name!r} missing={missing_props!r}, "
                f"available={list(prop_names)!r}"
            )
        expected_subregion = str(region["actual_subregion_name"])
        actual_assignment = _native_mesh_assignment_names(
            get_prop_value(props["assignment"])
        )
        if actual_assignment != [expected_subregion]:
            raise RuntimeError(
                "native thermal MeshRegion assignment mismatch: "
                f"{actual_name!r}: {actual_assignment!r} != "
                f"{[expected_subregion]!r}"
            )
        object_names = sorted(map(str, region.get("objects", [])))
        actual_enclosing = _native_mesh_assignment_names(
            get_prop_value(props["enclosing"])
        )
        if actual_enclosing != object_names:
            raise RuntimeError(
                "native thermal MeshRegion enclosing-geometry mismatch: "
                f"{actual_name!r}: {actual_enclosing!r} != "
                f"{object_names!r}"
            )
        if _thermal_bool(get_prop_value(props["enabled"])) is not True:
            raise RuntimeError(
                "native thermal MeshRegion is not enabled: "
                f"{actual_name!r}"
            )
        expected = region["manual_settings"]
        actual_lengths = {
            "MaxElementSizeX": _native_mesh_length_mm_readback(
                get_prop_value, props["max_x"], actual_name
            ),
            "MaxElementSizeY": _native_mesh_length_mm_readback(
                get_prop_value, props["max_y"], actual_name
            ),
            "MaxElementSizeZ": _native_mesh_length_mm_readback(
                get_prop_value, props["max_z"], actual_name
            ),
            "MinGapX": _native_mesh_length_mm_readback(
                get_prop_value, props["min_gap_x"], actual_name
            ),
            "MinGapY": _native_mesh_length_mm_readback(
                get_prop_value, props["min_gap_y"], actual_name
            ),
            "MinGapZ": _native_mesh_length_mm_readback(
                get_prop_value, props["min_gap_z"], actual_name
            ),
        }
        mismatched_lengths = {
            key: (actual, float(expected[key]))
            for key, actual in actual_lengths.items()
            if not math.isclose(
                actual, float(expected[key]),
                rel_tol=1e-9, abs_tol=1e-9,
            )
        }
        if mismatched_lengths:
            raise RuntimeError(
                "native thermal MeshRegion length readback mismatch: "
                f"{actual_name!r} {mismatched_lengths!r}"
            )
        actual_gap_elements = _native_mesh_integer_readback(
            get_prop_value, props["min_elements_gap"], actual_name
        )
        actual_edge_elements = _native_mesh_integer_readback(
            get_prop_value, props["min_elements_edge"], actual_name
        )
        actual_ratio = _native_mesh_number_readback(
            get_prop_value, props["max_ratio"], actual_name
        )
        if (
            actual_gap_elements != int(expected["MinElementsInGap"])
            or actual_edge_elements != int(expected["MinElementsOnEdge"])
            or not math.isclose(
                actual_ratio, float(expected["MaxSizeRatio"]),
                rel_tol=0.0, abs_tol=1e-12,
            )
            or re.sub(
                r"[^a-z0-9]",
                "",
                str(get_prop_value(props["uniform_type"])).casefold(),
            ) != "xyzmaxsizes"
        ):
            raise RuntimeError(
                "native thermal MeshRegion manual setting mismatch: "
                f"{actual_name!r}"
            )
        if object_names != list(region["subregion_parts_readback"]):
            raise RuntimeError(
                "thermal MeshRegion SubRegion part binding changed: "
                f"{actual_name!r}"
            )
        assignment_union.update(object_names)
        readbacks.append({
            "name": actual_name,
            "assignment": actual_assignment,
            "enclosing_geometries": actual_enclosing,
            "objects": object_names,
            "manual_settings": True,
            "manual_settings_evidence": (
                "native_manual_only_properties_complete"
            ),
            "wrapper_manual_settings_readback": True,
            "wrapper_enforce_cutcell_readback": (
                expected.get("EnforceCutCellMeshing") is True
            ),
            "manual_setting_values": {
                **actual_lengths,
                "MinElementsInGap": actual_gap_elements,
                "MinElementsOnEdge": actual_edge_elements,
                "MaxSizeRatio": actual_ratio,
                "UniformMeshParametersType": "XYZ Max. Sizes",
            },
            "thin_axis": region["thin_axis"],
            "thin_axis_divisions": int(
                region["thin_axis_divisions"]
            ),
            "dimensions_mm": list(region["dimensions_mm"]),
        })
    expected_objects = set(map(
        str, mesh_plan.get("wcp_pad_mesh_region_objects", [])
    ))
    if assignment_union != expected_objects:
        raise RuntimeError(
            "native thermal MeshRegion WCP-pad coverage mismatch: "
            f"missing={sorted(expected_objects - assignment_union)!r}, "
            f"unexpected={sorted(assignment_union - expected_objects)!r}"
        )
    return {
        "contract_version": WCP_PAD_MESH_REGION_CONTRACT_VERSION,
        "expected_mesh_region_count": len(regions),
        "missing_mesh_region_names": missing,
        "assigned_object_count": len(assignment_union),
        "assigned_objects": sorted(assignment_union),
        "mesh_region_readbacks": readbacks,
    }


def _native_thermal_mesh_operation_readback(
    native_design, mesh_plan, native_editor=None
):
    """Require exact assignment, level, and object-separation OO readback.

    PyAEDT 0.22 itself reads Icepak mesh assignments through
    ``oDesign.GetChildObject("Mesh")`` and the child ``Assignment`` property.
    Use the same nonblocking object tree and fail closed rather than falling
    back to ``GetMeshOpAssignment``, which can block on large models.
    """
    get_child = getattr(native_design, "GetChildObject", None)
    if not callable(get_child):
        raise RuntimeError(
            "native thermal design has no mesh object-tree readback"
        )
    mesh_child = get_child("Mesh")
    if mesh_child is None or mesh_child is False:
        raise RuntimeError("native thermal design returned no Mesh child")
    get_names = getattr(mesh_child, "GetChildNames", None)
    if not callable(get_names):
        raise RuntimeError(
            "native thermal Mesh child has no operation-name readback"
        )
    native_names = tuple(str(name) for name in (get_names() or []))
    operations_by_actual_name = {}
    for operation in mesh_plan.get("operations", []):
        actual_names = [
            str(name)
            for name in operation.get("actual_operation_names", [])
        ]
        if len(actual_names) != 1:
            raise RuntimeError(
                "thermal mesh plan operation does not have one native identity: "
                f"{operation.get('name', '')!r} -> {actual_names!r}"
            )
        operations_by_actual_name[actual_names[0]] = operation
    expected_names = tuple(operations_by_actual_name)
    if (
        not expected_names
        or len(expected_names) != int(mesh_plan["operation_count"])
        or len(expected_names) != len(set(expected_names))
    ):
        raise RuntimeError(
            "thermal mesh plan has invalid native operation identities"
        )
    missing = sorted(set(expected_names) - set(native_names))
    if missing:
        raise RuntimeError(
            "native thermal mesh operation readback is incomplete: "
            f"{missing!r}"
        )
    operation_readbacks = []
    assignment_union = set()
    for actual_name in expected_names:
        child = mesh_child.GetChildObject(actual_name)
        if child is None or child is False:
            raise RuntimeError(
                "native thermal mesh child disappeared during readback: "
                f"{actual_name!r}"
            )
        get_prop_names = getattr(child, "GetPropNames", None)
        get_prop_value = getattr(child, "GetPropValue", None)
        if not callable(get_prop_names) or not callable(get_prop_value):
            raise RuntimeError(
                "native thermal mesh child has no property readback: "
                f"{actual_name!r}"
            )
        prop_names = tuple(str(name) for name in (get_prop_names() or []))
        expected_operation = operations_by_actual_name[actual_name]
        operation_type = str(
            expected_operation.get("operation_type", "object_level")
        )
        if operation_type == "mesh_region":
            assignment_prop = _native_mesh_property_name(
                prop_names, "Assignment", "Parts", "Objects"
            )
            resolution_prop = _native_mesh_property_name(
                prop_names,
                "MeshRegionResolution",
                "Mesh Region Resolution",
                "Mesh Resolution",
                "Level",
            )
            enabled_prop = _native_mesh_property_name(
                prop_names, "Enable", "Enabled"
            )
            enclosing_prop = _native_mesh_property_name(
                prop_names, "Enclosing Geometries"
            )
            missing_props = [
                label
                for label, value in (
                    ("Assignment", assignment_prop),
                    ("MeshRegionResolution", resolution_prop),
                    ("Enable", enabled_prop),
                    ("Enclosing Geometries", enclosing_prop),
                )
                if not value
            ]
            if missing_props:
                raise RuntimeError(
                    "native thermal mesh region lacks required OO "
                    f"properties: {actual_name!r} "
                    f"missing={missing_props!r}, "
                    f"available={list(prop_names)!r}"
                )
            expected_region_object = str(
                expected_operation.get(
                    "native_region_object_name", ""
                ) or ""
            )
            if not expected_region_object:
                raise RuntimeError(
                    "thermal mesh-region plan lacks native subregion "
                    f"identity: {actual_name!r}"
                )
            actual_region_objects = _native_mesh_assignment_names(
                get_prop_value(assignment_prop),
                editor=native_editor,
            )
            if actual_region_objects != [expected_region_object]:
                raise RuntimeError(
                    "native thermal mesh-region assignment mismatch: "
                    f"{actual_name!r}: {actual_region_objects!r} != "
                    f"{[expected_region_object]!r}"
                )
            expected_objects = sorted(map(
                str, expected_operation.get("objects", [])
            ))
            enclosing_objects = _native_mesh_assignment_names(
                get_prop_value(enclosing_prop),
                editor=native_editor,
            )
            if enclosing_objects != expected_objects:
                raise RuntimeError(
                    "native thermal mesh-region enclosing-geometry "
                    f"mismatch: {actual_name!r}: "
                    f"{enclosing_objects!r} != {expected_objects!r}"
                )
            if _thermal_bool(get_prop_value(enabled_prop)) is not True:
                raise RuntimeError(
                    "native thermal mesh region is disabled: "
                    f"{actual_name!r}"
                )
            expected_level = int(expected_operation["level"])
            actual_level = _native_mesh_integer_readback(
                get_prop_value, resolution_prop, actual_name
            )
            if actual_level != expected_level:
                raise RuntimeError(
                    "native thermal mesh-region level readback mismatch: "
                    f"{actual_name!r}: {actual_level} != {expected_level}"
                )
            actual_objects = _native_mesh_region_part_names(
                native_editor, expected_region_object
            )
            if actual_objects != expected_objects:
                raise RuntimeError(
                    "native thermal mesh-region part readback mismatch: "
                    f"{actual_name!r}: {actual_objects!r} != "
                    f"{expected_objects!r}"
                )
            assignment_union.update(actual_objects)
            operation_readbacks.append({
                "name": actual_name,
                "operation_type": "mesh_region",
                "assignment_count": len(actual_objects),
                "assignment_sha256": hashlib.sha256(
                    json.dumps(
                        actual_objects, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "level": actual_level,
                "level_schema": str(resolution_prop),
                "min_level": None,
                "max_level": None,
                "incr_level": None,
                "separate_objects": None,
                "region_object_name": expected_region_object,
                "region_object_count": len(actual_region_objects),
                "enclosing_geometries": enclosing_objects,
            })
            continue
        if operation_type != "object_level":
            raise RuntimeError(
                "thermal mesh plan has unsupported operation type: "
                f"{actual_name!r} -> {operation_type!r}"
            )
        assignment_prop = _native_mesh_property_name(
            prop_names, "Assignment", "Parts", "Objects"
        )
        level_prop = _native_mesh_property_name(prop_names, "Level")
        min_level_prop = _native_mesh_property_name(prop_names, "MinLevel")
        max_level_prop = _native_mesh_property_name(prop_names, "MaxLevel")
        incr_level_prop = _native_mesh_property_name(prop_names, "IncrLevel")
        separate_prop = _native_mesh_property_name(
            prop_names, "Mesh Object(s) Separately Enabled"
        )
        missing_props = [
            label
            for label, value in (
                ("Assignment", assignment_prop),
                ("Mesh Object(s) Separately Enabled", separate_prop),
            )
            if not value
        ]
        if missing_props:
            raise RuntimeError(
                "native thermal mesh operation lacks required OO properties: "
                f"{actual_name!r} missing={missing_props!r}, "
                f"available={list(prop_names)!r}"
            )
        ranged_level_props = (
            min_level_prop, max_level_prop, incr_level_prop
        )
        if level_prop and any(ranged_level_props):
            raise RuntimeError(
                "native thermal mesh operation has ambiguous level schema: "
                f"{actual_name!r}, available={list(prop_names)!r}"
            )
        if not level_prop and not all(ranged_level_props):
            missing_level_props = [
                label
                for label, value in (
                    ("MinLevel", min_level_prop),
                    ("MaxLevel", max_level_prop),
                    ("IncrLevel", incr_level_prop),
                )
                if not value
            ]
            raise RuntimeError(
                "native thermal mesh operation lacks one complete level schema: "
                f"{actual_name!r} missing={missing_level_props!r}, "
                f"available={list(prop_names)!r}"
            )
        expected_objects = sorted(map(
            str, expected_operation.get("objects", [])
        ))
        actual_objects = _native_mesh_assignment_names(
            get_prop_value(assignment_prop), editor=native_editor
        )
        if actual_objects != expected_objects:
            raise RuntimeError(
                "native thermal mesh assignment readback mismatch: "
                f"{actual_name!r}: {actual_objects!r} != "
                f"{expected_objects!r}"
            )
        expected_level = int(expected_operation["level"])
        if level_prop:
            level_schema = "Level"
            actual_level = _native_mesh_integer_readback(
                get_prop_value, level_prop, actual_name
            )
            actual_min_level = None
            actual_max_level = None
            actual_incr_level = None
            if actual_level != expected_level:
                raise RuntimeError(
                    "native thermal mesh level readback mismatch: "
                    f"{actual_name!r}: {actual_level} != {expected_level}"
                )
        else:
            # AEDT's fixed object-level operation is serialized with equal
            # MinLevel/MaxLevel and IncrLevel='0' (also exercised by the
            # sealed replay contract).  Its 2025.2 OO tree exposes those
            # resolved properties instead of PyAEDT's input-only ``Level``.
            level_schema = "MinLevel/MaxLevel/IncrLevel"
            actual_min_level = _native_mesh_integer_readback(
                get_prop_value, min_level_prop, actual_name
            )
            actual_max_level = _native_mesh_integer_readback(
                get_prop_value, max_level_prop, actual_name
            )
            actual_incr_level = _native_mesh_integer_readback(
                get_prop_value, incr_level_prop, actual_name
            )
            if (
                actual_min_level != expected_level
                or actual_max_level != expected_level
            ):
                raise RuntimeError(
                    "native thermal mesh level range readback mismatch: "
                    f"{actual_name!r}: MinLevel={actual_min_level}, "
                    f"MaxLevel={actual_max_level}, expected={expected_level}"
                )
            if actual_incr_level != 0:
                raise RuntimeError(
                    "native thermal mesh increment readback mismatch: "
                    f"{actual_name!r}: IncrLevel={actual_incr_level} != 0"
                )
            actual_level = expected_level
        expected_separate = expected_operation.get("separate_objects")
        if type(expected_separate) is not bool:
            raise RuntimeError(
                "thermal mesh plan operation lacks exact separation intent: "
                f"{actual_name!r}"
            )
        actual_separate = _thermal_bool(get_prop_value(separate_prop))
        if actual_separate is not expected_separate:
            raise RuntimeError(
                "native thermal mesh object-separation readback mismatch: "
                f"{actual_name!r}: {actual_separate!r} != "
                f"{expected_separate!r}"
            )
        assignment_union.update(actual_objects)
        operation_readbacks.append({
            "name": actual_name,
            "operation_type": "object_level",
            "assignment_count": len(actual_objects),
            "assignment_sha256": hashlib.sha256(
                json.dumps(
                    actual_objects, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "level": actual_level,
            "level_schema": level_schema,
            "min_level": actual_min_level,
            "max_level": actual_max_level,
            "incr_level": actual_incr_level,
            "separate_objects": actual_separate,
        })
    expected_assignment_union = {
        str(name)
        for operation in mesh_plan.get("operations", [])
        for name in operation.get("objects", [])
    }
    if assignment_union != expected_assignment_union:
        raise RuntimeError(
            "native thermal mesh total assignment coverage mismatch: "
            f"missing={sorted(expected_assignment_union - assignment_union)!r}, "
            f"unexpected={sorted(assignment_union - expected_assignment_union)!r}"
        )
    expected_required_thin = set(map(
        str, mesh_plan.get("required_thin_objects", [])
    ))
    required_thin_objects = assignment_union & expected_required_thin
    required_thin_missing = sorted(
        expected_required_thin - required_thin_objects
    )
    if required_thin_missing:
        raise RuntimeError(
            "native thermal mesh required-thin coverage is incomplete: "
            f"{required_thin_missing!r}"
        )
    payload = {
        "expected_operation_count": len(expected_names),
        "native_operation_count": len(native_names),
        "missing_operation_names": missing,
        "expected_operation_names": list(expected_names),
        "operation_readbacks": operation_readbacks,
        "assigned_object_count": len(assignment_union),
        "required_thin_object_count": len(required_thin_objects),
        "required_thin_objects_missing": required_thin_missing,
        "object_level_operation_count": sum(
            item.get("operation_type") == "object_level"
            for item in operation_readbacks
        ),
        "mesh_region_operation_count": sum(
            item.get("operation_type") == "mesh_region"
            for item in operation_readbacks
        ),
        "mesh_region_part_readback_passed": all(
            item.get("region_object_count") == 1
            for item in operation_readbacks
            if item.get("operation_type") == "mesh_region"
        ),
    }
    payload["native_operation_names_sha256"] = hashlib.sha256(
        json.dumps(
            sorted(native_names), separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _attest_standalone_mesh_desktop(sim, ipk, desktop):
    """Re-attest one cached PID/endpoint immediately before idle readback."""
    attestor = getattr(sim, "_attest_cached_native_desktop", None)
    if not callable(attestor):
        raise RuntimeError(
            "standalone thermal mesh Desktop attestation is unavailable"
        )
    attested = attestor(desktop, require_endpoint_identity=True)
    if attested is None or attested is False:
        raise RuntimeError(
            "standalone thermal mesh Desktop attestation returned no proxy"
        )
    return _thermal_running_state(sim, ipk, desktop=desktop)


def _wait_for_standalone_mesh_idle(
    sim,
    ipk,
    desktop,
    artifact_snapshot,
    *,
    require_fresh_artifact,
    timeout_s=None,
    poll_s=1.0,
    stable_artifact_s=1.0,
    failure_idle_grace_s=30.0,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    """Prove post-GenerateMesh idle without touching live AEDT automation.

    A single immediate ``False`` can precede delayed native mesh startup.
    Successful generation therefore requires a fresh grid pair, a second
    independently attested idle observation, and a stable artifact interval.
    Failed/exceptional generation has no grid contract, but still receives a
    bounded multi-poll idle grace before the uncertainty flag may be cleared.
    """
    timeout_s = _standalone_thermal_completion_timeout(timeout_s)
    poll_s = float(poll_s)
    stable_artifact_s = float(stable_artifact_s)
    failure_idle_grace_s = float(failure_idle_grace_s)
    if (
        not math.isfinite(poll_s)
        or poll_s < 0
        or not math.isfinite(stable_artifact_s)
        or stable_artifact_s < 0
        or not math.isfinite(failure_idle_grace_s)
        or failure_idle_grace_s < 0
    ):
        raise ValueError("thermal mesh idle-barrier timing is invalid")

    started = clock()
    deadline = started + timeout_s
    first_idle_at = None
    idle_observations = 0
    stable_signature = None
    stable_since = None
    stable_observations = 0
    fresh_artifacts = []
    transitions = []
    attestation_attempts = 0
    last_running = None
    last_error = ""
    while True:
        attestation_attempts += 1
        running = None
        error = ""
        try:
            running = _attest_standalone_mesh_desktop(
                sim, ipk, desktop
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:512]}"
        now = clock()
        last_running = running
        last_error = error
        transition = (running, error)
        if not transitions or transitions[-1]["state"] != transition:
            if len(transitions) < 64:
                transitions.append({
                    "elapsed_s": round(max(0.0, now - started), 3),
                    "running": running,
                    "error": error,
                    "state": transition,
                })

        if running is False and not error:
            idle_observations += 1
            if first_idle_at is None:
                first_idle_at = now
            # Filesystem evidence is inspected only after the exact cached
            # Desktop was freshly re-attested and explicitly reported idle.
            fresh_artifacts = _fresh_thermal_mesh_artifacts(
                sim, ipk, artifact_snapshot
            )
            artifact_signature = json.dumps(
                fresh_artifacts,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            if fresh_artifacts:
                if artifact_signature != stable_signature:
                    stable_signature = artifact_signature
                    stable_since = now
                    stable_observations = 1
                else:
                    stable_observations += 1
            else:
                stable_signature = None
                stable_since = None
                stable_observations = 0

            artifact_stable = (
                bool(fresh_artifacts)
                and stable_observations >= 2
                and stable_since is not None
                and now - stable_since >= stable_artifact_s
            )
            failure_idle_stable = (
                not require_fresh_artifact
                and idle_observations >= 2
                and now - first_idle_at >= failure_idle_grace_s
            )
            if (
                (require_fresh_artifact and artifact_stable)
                or failure_idle_stable
            ):
                return {
                    "schema": "thermal-mesh-idle-barrier-v1",
                    "passed": True,
                    "require_fresh_artifact": bool(
                        require_fresh_artifact
                    ),
                    "desktop_attestation_attempts": attestation_attempts,
                    "idle_observations": idle_observations,
                    "stable_artifact_observations": (
                        stable_observations
                    ),
                    "elapsed_s": round(
                        max(0.0, now - started), 3
                    ),
                    "last_running": False,
                    "last_error": "",
                    "fresh_mesh_artifacts": fresh_artifacts,
                    "running_transitions": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "state"
                        }
                        for item in transitions
                    ],
                }
        else:
            first_idle_at = None
            idle_observations = 0
            stable_signature = None
            stable_since = None
            stable_observations = 0
            fresh_artifacts = []

        if now >= deadline:
            return {
                "schema": "thermal-mesh-idle-barrier-v1",
                "passed": False,
                "require_fresh_artifact": bool(
                    require_fresh_artifact
                ),
                "desktop_attestation_attempts": attestation_attempts,
                "idle_observations": idle_observations,
                "stable_artifact_observations": stable_observations,
                "elapsed_s": round(max(0.0, now - started), 3),
                "last_running": last_running,
                "last_error": last_error,
                "fresh_mesh_artifacts": fresh_artifacts,
                "running_transitions": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "state"
                    }
                    for item in transitions
                ],
            }
        sleeper(min(
            poll_s,
            max(0.0, deadline - now),
        ))


def _mesh_quality_canary_requested(mesh_plan):
    """Return true only for the reviewed level-4 Rx-side numerical canary."""

    operations = mesh_plan.get("operations", [])
    side_operations = [
        operation
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("category")
        in {"Rx_side_blocks", "Rx_side2_blocks"}
        and operation.get("objects")
    ]
    return bool(side_operations) and all(
        operation.get("operation_type") == "object_level"
        and int(operation.get("level", -1)) == 4
        for operation in side_operations
    )


def _export_thermal_mesh_stats(native_ipk, setup_name):
    """Export and seal native mesh statistics without changing mesh checks."""

    results_directory = str(
        getattr(native_ipk, "results_directory", "") or ""
    ).strip()
    if not results_directory:
        raise RuntimeError(
            "native mesh statistics require an AEDT results directory"
        )
    root = Path(results_directory).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(
            "native mesh statistics results directory is unsafe"
        )
    target = root / THERMAL_MESH_STATS_FILENAME
    if target.exists():
        raise RuntimeError("native mesh statistics target already exists")
    exporter = getattr(native_ipk, "export_mesh_stats", None)
    if not callable(exporter):
        raise RuntimeError(
            "native Icepak ExportMeshStats API is unavailable"
        )
    returned = exporter(setup_name, output_file=str(target))
    try:
        returned_path = Path(str(returned)).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "native ExportMeshStats returned no durable file"
        ) from exc
    if returned_path != target.resolve(strict=True):
        raise RuntimeError(
            "native ExportMeshStats returned an unexpected path"
        )
    if not returned_path.is_file() or returned_path.is_symlink():
        raise RuntimeError(
            "native mesh statistics artifact is not a regular file"
        )
    size = returned_path.stat().st_size
    if not 0 < size <= THERMAL_MESH_STATS_MAX_BYTES:
        raise RuntimeError(
            f"native mesh statistics size is invalid: {size}"
        )
    raw = returned_path.read_bytes()
    if len(raw) != size:
        raise RuntimeError(
            "native mesh statistics changed during authentication"
        )
    text = raw.decode("utf-8", errors="replace")
    patterns = {
        "skewness": r"\bskew(?:ness)?\b",
        "element_volume": r"\b(?:element|cell\s+)?volume\b",
        "face_alignment": r"\bface\s+alignment\b",
    }
    occurrences = {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in patterns.items()
    }
    return {
        "schema": THERMAL_MESH_STATS_CONTRACT_VERSION,
        "export_api": "Icepak.export_mesh_stats/oDesign.ExportMeshStats",
        "setup_name": setup_name,
        "path": returned_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": size,
        "line_count": len(text.splitlines()),
        "quality_metric_occurrences": occurrences,
        "quality_metrics_reported": sorted(
            name for name, count in occurrences.items() if count > 0
        ),
        "quality_metric_availability_recorded": True,
        "mesh_quality_checks_modified": False,
        "mesh_quality_check_disable_requested": False,
        "exported": True,
    }


def _generate_and_attest_thermal_mesh(
    sim,
    ipk,
    setup,
    mesh_plan,
    *,
    idle_timeout_s=None,
    idle_poll_s=1.0,
    stable_artifact_s=1.0,
    failure_idle_grace_s=30.0,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    """Generate and attest a fresh native grid before Analyze.

    ``MeshIcepak.generate_mesh`` returns a Python boolean for AEDT's native
    zero status.  Other return types are not interpreted as success.  A true
    return is still insufficient: the exact project/design message cursor must
    advance safely, every expected native mesh operation must remain present,
    and AEDT must publish a new nonempty ``grid_output``/``grid_mapping`` pair.
    """
    required_objects_missing = list(
        mesh_plan.get("required_objects_missing", [])
    )
    operation_count = int(mesh_plan.get("operation_count", 0))
    object_level_operation_count = int(
        mesh_plan.get("object_level_operation_count", -1)
    )
    mesh_region_operation_count = int(
        mesh_plan.get("mesh_region_operation_count", -1)
    )
    static_contract_passed = (
        mesh_plan.get("schema") == THERMAL_MESH_PLAN_CONTRACT_VERSION
        and mesh_plan.get("policy") == THERMAL_MESH_POLICY
        and not required_objects_missing
        and operation_count > 0
        and int(mesh_plan.get("shared_operation_count", -1)) == 0
        and object_level_operation_count + mesh_region_operation_count
        == operation_count
        and int(mesh_plan.get("separate_object_operation_count", -1))
        == object_level_operation_count
        and mesh_region_operation_count
        == int(mesh_plan.get("wcp_pad_mesh_region_count", -1))
        and len(mesh_plan.get("required_thin_objects", []))
        == int(mesh_plan.get("required_thin_object_count", -1))
    )
    from module.aedt_pool_adapter import pooled_backend_enabled
    if pooled_backend_enabled():
        raise RuntimeError(
            "explicit native GenerateMesh preflight is standalone-only"
        )
    preflight = _prepare_thermal_dispatch(
        sim,
        ipk,
        setup,
        design_name=_THERMAL_DESIGN_NAME,
        setup_name=_THERMAL_SETUP_NAME,
    )
    native_ipk = preflight["native_ipk"]
    native_design = preflight["native_design"]

    native_operation_readback = (
        _native_thermal_mesh_operation_readback(
            native_design,
            mesh_plan,
            native_editor=getattr(
                getattr(native_ipk, "modeler", None), "oeditor", None
            ),
        )
    )
    artifact_snapshot = _snapshot_thermal_mesh_artifacts(
        sim, native_ipk
    )
    desktop = _thermal_desktop_handle(sim, native_ipk)
    cursor = capture_scoped_message_cursor(
        desktop, preflight["project"], preflight["design"]
    )
    pre_running = _attest_standalone_mesh_desktop(
        sim, native_ipk, desktop
    )
    if pre_running is not False:
        raise RuntimeError(
            "standalone thermal mesh preflight did not prove initial idle: "
            f"{pre_running!r}"
        )
    generator = getattr(
        getattr(native_ipk, "mesh", None), "generate_mesh", None
    )
    if not callable(generator):
        raise RuntimeError("native Icepak mesh generation API is unavailable")

    started = clock()
    returned = None
    generation_exception = ""
    sim.solver_may_be_running = True
    try:
        returned = generator(_THERMAL_SETUP_NAME)
    except Exception as exc:
        generation_exception = (
            f"{type(exc).__name__}: {str(exc)[:512]}"
        )

    idle_barrier = _wait_for_standalone_mesh_idle(
        sim,
        native_ipk,
        desktop,
        artifact_snapshot,
        require_fresh_artifact=(
            returned is True and not generation_exception
        ),
        timeout_s=idle_timeout_s,
        poll_s=idle_poll_s,
        stable_artifact_s=stable_artifact_s,
        failure_idle_grace_s=failure_idle_grace_s,
        clock=clock,
        sleeper=sleeper,
    )
    if idle_barrier.get("passed") is not True:
        # Do not scan messages, files, rebind a project, save, or release the
        # Desktop.  The recovery process boundary owns containment while this
        # flag remains asserted.
        forensic = json.dumps(
            {
                "schema": THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
                "passed": False,
                "generate_mesh_returned": returned is True,
                "generation_exception": generation_exception,
                "standalone_idle_barrier": idle_barrier,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        logging.error(
            "[thermal] unsafe native mesh completion: %s", forensic
        )
        raise RuntimeError(
            "standalone native thermal mesh completion remains uncertain: "
            + forensic[:8000]
        )
    sim.solver_may_be_running = False

    message_scan_complete = False
    new_messages = ()
    new_errors = ()
    fatal_messages = ()
    message_scan_error = ""
    try:
        update = advance_scoped_message_cursor(desktop, cursor)
        new_messages = tuple(update.new_messages)
        new_errors = tuple(update.new_errors)
        fatal_messages = tuple(update.fatal_messages)
        message_scan_complete = True
    except Exception as exc:
        message_scan_error = (
            f"{type(exc).__name__}: {str(exc)[:512]}"
        )
    unmeshed_objects = _unmeshed_objects_from_messages(
        (*new_messages, *new_errors)
    )

    fresh_artifacts = list(
        idle_barrier.get("fresh_mesh_artifacts", [])
    )
    artifact_readback_passed = bool(fresh_artifacts)
    mapping_coverage = _thermal_mesh_mapping_coverage(
        mesh_plan, fresh_artifacts
    )

    postflight_passed = False
    postflight_error = ""
    if returned is True:
        try:
            postflight = _prepare_thermal_dispatch(
                sim,
                native_ipk,
                setup,
                design_name=_THERMAL_DESIGN_NAME,
                setup_name=_THERMAL_SETUP_NAME,
            )
            postflight_passed = (
                postflight["project"] == preflight["project"]
                and postflight["design"] == preflight["design"]
                and postflight["setups"] == preflight["setups"]
            )
            if not postflight_passed:
                raise RuntimeError(
                    "thermal premesh postflight identity changed"
                )
        except Exception as exc:
            postflight_error = (
                f"{type(exc).__name__}: {str(exc)[:512]}"
            )

    mesh_quality_canary = _mesh_quality_canary_requested(mesh_plan)
    setup_control_readback = preflight.get(
        "setup_control_readback", {}
    )
    mesh_stats = {
        "schema": THERMAL_MESH_STATS_CONTRACT_VERSION,
        "exported": False,
        "not_requested": True,
        "mesh_quality_checks_modified": False,
        "mesh_quality_check_disable_requested": False,
    }
    if mesh_quality_canary:
        if (
            not isinstance(setup_control_readback, dict)
            or setup_control_readback.get("schema")
            != THERMAL_SETUP_CONTROL_READBACK_CONTRACT_VERSION
            or setup_control_readback.get("wrapper_passed") is not True
            or setup_control_readback.get("native_complete") is not True
            or setup_control_readback.get(
                "mesh_quality_check_disable_requested"
            )
            is not False
        ):
            raise RuntimeError(
                "mesh-quality canary lacks complete native setup-control "
                "readback"
            )
        mesh_stats = _export_thermal_mesh_stats(
            native_ipk, _THERMAL_SETUP_NAME
        )

    message_payload = {
        "messages": list(new_messages),
        "errors": list(new_errors),
    }
    message_sha256 = hashlib.sha256(
        json.dumps(
            message_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    native_errors = list(dict.fromkeys(
        [*new_errors, *fatal_messages]
    ))
    passed = (
        static_contract_passed
        and returned is True
        and not generation_exception
        and message_scan_complete
        and not native_errors
        and not unmeshed_objects
        and artifact_readback_passed
        and mapping_coverage["passed"] is True
        and postflight_passed
        and not native_operation_readback["missing_operation_names"]
        and not native_operation_readback[
            "required_thin_objects_missing"
        ]
        and idle_barrier.get("passed") is True
        and (
            not mesh_quality_canary
            or mesh_stats.get("exported") is True
        )
    )
    evidence = {
        "schema": THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
        "status": (
            "passed_standalone_native_premesh"
            if passed else "failed_standalone_native_premesh"
        ),
        "passed": passed,
        "static_contract_passed": static_contract_passed,
        "generate_mesh_returned": returned is True,
        "generate_mesh_return_type": type(returned).__name__,
        "generate_mesh_return_repr": repr(returned)[:128],
        "generation_exception": generation_exception,
        "message_scan_complete": message_scan_complete,
        "message_scan_error": message_scan_error,
        "native_message_sha256": message_sha256,
        "native_errors": native_errors,
        "required_objects_missing": required_objects_missing,
        "unmeshed_objects": unmeshed_objects,
        "native_operation_readback_passed": (
            not native_operation_readback["missing_operation_names"]
        ),
        "native_operation_readback": native_operation_readback,
        "standalone_idle_barrier_passed": True,
        "standalone_idle_barrier": idle_barrier,
        "mesh_artifact_readback_passed": artifact_readback_passed,
        "mesh_mapping_coverage_passed": mapping_coverage["passed"],
        "mesh_mapping_coverage": mapping_coverage,
        "fresh_mesh_artifact_count": len(fresh_artifacts),
        "fresh_mesh_artifacts": fresh_artifacts,
        "postflight_identity_passed": postflight_passed,
        "postflight_error": postflight_error,
        "mesh_quality_canary_requested": mesh_quality_canary,
        "setup_control_readback": setup_control_readback,
        "native_mesh_stats": mesh_stats,
        "analysis_dispatched_after_premesh": False,
        "mesh_policy": THERMAL_MESH_POLICY,
        "mesh_plan_sha256": mesh_plan["plan_sha256"],
        "mesh_operation_count": mesh_plan["operation_count"],
        "mesh_assigned_object_count": (
            mesh_plan["assigned_object_count"]
        ),
        "object_level_operation_count": (
            mesh_plan["object_level_operation_count"]
        ),
        "mesh_region_operation_count": (
            mesh_plan["mesh_region_operation_count"]
        ),
        "wcp_pad_mesh_region_count": (
            mesh_plan["wcp_pad_mesh_region_count"]
        ),
        "core_plate_assembly_count": (
            mesh_plan["core_plate_assembly_count"]
        ),
        "wcp_assembly_count": mesh_plan["wcp_assembly_count"],
        "rx_retained_pack_count": (
            mesh_plan["rx_retained_pack_count"]
        ),
        "elapsed_s": round(
            max(0.0, clock() - started), 3
        ),
    }
    forensic = json.dumps(
        evidence,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    logging.warning("[thermal] native mesh preflight: %s", forensic)
    if not passed:
        raise RuntimeError(
            "native thermal mesh preflight failed: "
            + forensic[:8000]
        )
    return evidence


def _pooled_thermal_mesh_preflight_not_applicable(mesh_plan):
    """Record why pooled production keeps its existing exact-solve protocol."""
    operation_count = int(mesh_plan.get("operation_count", 0))
    object_level_operation_count = int(
        mesh_plan.get("object_level_operation_count", -1)
    )
    mesh_region_operation_count = int(
        mesh_plan.get("mesh_region_operation_count", -1)
    )
    static_contract_passed = (
        mesh_plan.get("schema") == THERMAL_MESH_PLAN_CONTRACT_VERSION
        and mesh_plan.get("policy") == THERMAL_MESH_POLICY
        and mesh_plan.get("required_objects_missing") == []
        and operation_count > 0
        and int(mesh_plan.get("shared_operation_count", -1)) == 0
        and object_level_operation_count + mesh_region_operation_count
        == operation_count
        and int(mesh_plan.get("separate_object_operation_count", -1))
        == object_level_operation_count
        and mesh_region_operation_count
        == int(mesh_plan.get("wcp_pad_mesh_region_count", -1))
    )
    if not static_contract_passed:
        raise RuntimeError(
            "pooled thermal mesh static assignment contract failed"
        )
    return {
        "schema": THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION,
        "status": "not_applicable_pooled_exact_analyze_protocol",
        "passed": False,
        "static_contract_passed": True,
        "generate_mesh_returned": False,
        "message_scan_complete": False,
        "analysis_dispatched_after_premesh": False,
        "required_objects_missing": [],
        "unmeshed_objects": [],
        "mesh_artifact_readback_passed": False,
        "mesh_mapping_coverage_passed": False,
        "mesh_mapping_coverage": {
            "schema": "thermal-grid-mapping-coverage-v1",
            "passed": False,
            "status": "not_applicable_pooled_exact_analyze_protocol",
        },
        "native_operation_readback_passed": False,
        "postflight_identity_passed": False,
        "mesh_quality_canary_requested": (
            _mesh_quality_canary_requested(mesh_plan)
        ),
        "setup_control_readback": {},
        "native_mesh_stats": {
            "schema": THERMAL_MESH_STATS_CONTRACT_VERSION,
            "exported": False,
            "not_requested": True,
            "mesh_quality_checks_modified": False,
            "mesh_quality_check_disable_requested": False,
        },
        "mesh_policy": THERMAL_MESH_POLICY,
        "mesh_plan_sha256": mesh_plan["plan_sha256"],
        "mesh_operation_count": mesh_plan["operation_count"],
        "mesh_assigned_object_count": (
            mesh_plan["assigned_object_count"]
        ),
        "object_level_operation_count": (
            mesh_plan["object_level_operation_count"]
        ),
        "mesh_region_operation_count": (
            mesh_plan["mesh_region_operation_count"]
        ),
        "wcp_pad_mesh_region_count": (
            mesh_plan["wcp_pad_mesh_region_count"]
        ),
        "core_plate_assembly_count": (
            mesh_plan["core_plate_assembly_count"]
        ),
        "wcp_assembly_count": mesh_plan["wcp_assembly_count"],
        "rx_retained_pack_count": (
            mesh_plan["rx_retained_pack_count"]
        ),
    }


def _thermal_mesh_result_metadata(mesh_plan, preflight):
    """Serialize the exact mesh plan and native preflight into one result row."""
    pooled_not_applicable = (
        preflight.get("status")
        == "not_applicable_pooled_exact_analyze_protocol"
    )
    common_valid = (
        preflight.get("schema")
        == THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION
        and preflight.get("static_contract_passed") is True
        and preflight.get("analysis_dispatched_after_premesh") is True
        and preflight.get("required_objects_missing") == []
        and preflight.get("unmeshed_objects") == []
    )
    standalone_valid = (
        preflight.get("passed") is True
        and preflight.get("generate_mesh_returned") is True
        and preflight.get("message_scan_complete") is True
        and preflight.get("mesh_artifact_readback_passed") is True
        and preflight.get("mesh_mapping_coverage_passed") is True
        and preflight.get("native_operation_readback_passed") is True
        and preflight.get("standalone_idle_barrier_passed") is True
        and preflight.get("postflight_identity_passed") is True
    )
    if not common_valid or (
        not pooled_not_applicable and not standalone_valid
    ):
        raise RuntimeError(
            "thermal mesh result metadata requires a passed pre-solve "
            "attestation and a later Analyze dispatch"
        )
    mesh_stats = preflight.get("native_mesh_stats", {})
    setup_control_readback = preflight.get(
        "setup_control_readback", {}
    )
    return {
        "thermal_mesh_policy": [THERMAL_MESH_POLICY],
        "thermal_mesh_plan_contract_version": [
            THERMAL_MESH_PLAN_CONTRACT_VERSION
        ],
        "thermal_mesh_plan_sha256": [mesh_plan["plan_sha256"]],
        "thermal_mesh_operation_count": [
            int(mesh_plan["operation_count"])
        ],
        "thermal_mesh_assigned_object_count": [
            int(mesh_plan["assigned_object_count"])
        ],
        "thermal_mesh_required_thin_object_count": [
            int(mesh_plan["required_thin_object_count"])
        ],
        "thermal_mesh_shared_operation_count": [
            int(mesh_plan["shared_operation_count"])
        ],
        "thermal_mesh_separate_object_operation_count": [
            int(mesh_plan["separate_object_operation_count"])
        ],
        "thermal_mesh_object_level_operation_count": [
            int(mesh_plan["object_level_operation_count"])
        ],
        "thermal_mesh_region_operation_count": [
            int(mesh_plan["mesh_region_operation_count"])
        ],
        "thermal_mesh_wcp_pad_region_count": [
            int(mesh_plan["wcp_pad_mesh_region_count"])
        ],
        "thermal_mesh_core_plate_assembly_count": [
            int(mesh_plan["core_plate_assembly_count"])
        ],
        "thermal_mesh_wcp_assembly_count": [
            int(mesh_plan["wcp_assembly_count"])
        ],
        "thermal_mesh_rx_retained_pack_count": [
            int(mesh_plan["rx_retained_pack_count"])
        ],
        "thermal_mesh_preflight_contract_version": [
            THERMAL_MESH_PREFLIGHT_CONTRACT_VERSION
        ],
        "thermal_mesh_preflight_status": [
            preflight.get("status", "passed_standalone_native_premesh")
        ],
        "thermal_mesh_native_generation_passed": [
            0 if pooled_not_applicable else 1
        ],
        "thermal_mesh_unmeshed_object_count": [0],
        "thermal_mesh_unmeshed_objects_json": ["[]"],
        "thermal_mesh_quality_canary_requested": [
            1
            if preflight.get("mesh_quality_canary_requested") is True
            else 0
        ],
        "thermal_mesh_stats_exported": [
            1 if mesh_stats.get("exported") is True else 0
        ],
        "thermal_mesh_stats_contract_version": [
            str(mesh_stats.get("schema") or "")
        ],
        "thermal_mesh_stats_path": [
            str(mesh_stats.get("path") or "")
        ],
        "thermal_mesh_stats_sha256": [
            str(mesh_stats.get("sha256") or "")
        ],
        "thermal_mesh_stats_size_bytes": [
            int(mesh_stats.get("size_bytes") or 0)
        ],
        "thermal_mesh_quality_metrics_reported_json": [
            json.dumps(
                mesh_stats.get("quality_metrics_reported", []),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ],
        "thermal_mesh_quality_metric_occurrences_json": [
            json.dumps(
                mesh_stats.get("quality_metric_occurrences", {}),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ],
        "thermal_setup_control_readback_json": [
            json.dumps(
                setup_control_readback,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ],
        "thermal_mesh_preflight_json": [
            json.dumps(
                preflight,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ],
    }


def _thermal_mesh_postsolve_probe_object_names(objs):
    """Return the prior nine zero-cell solids when present in this topology."""
    return sorted({
        str(obj.name)
        for obj in (
            list(objs.get("wcp_pads", []))
            + list(objs.get("Rx_side2_explicit", []))
        )
        if (
            str(obj.name).startswith("Tx_main_wcp_pad_")
            or str(obj.name) == "Rx_side2_22_0"
        )
    })


def _thermal_mesh_postsolve_probe_status(object_names, temperatures):
    """Require finite mean and maximum volume temperature for each solid."""
    names = sorted(set(map(str, object_names)))
    missing = [
        name
        for name in names
        if not all(
            column in temperatures
            and math.isfinite(float(temperatures[column]))
            for column in (f"T_mean_{name}", f"T_max_{name}")
        )
    ]
    return {
        "object_count": len(names),
        "missing_count": len(missing),
        "missing_objects": missing,
        "complete": not missing,
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


def _bounded_thermal_model_context(ipk, operation_limit=64):
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
    clock=time.monotonic, sleeper=time.sleep, completion_barrier=False,
    idle_monitor_grace_s=30.0, desktop=None, desktop_attestor=None,
):
    """Poll fresh monitor evidence and the native standalone solve state.

    The legacy startup-evidence mode is retained for callers that do not set
    ``completion_barrier``. Completion mode is deliberately stronger: a
    native ``running=True`` state can never leave the loop, the first verified
    idle state starts a short monitor-publication grace period, and a fresh
    terminal residual monitor is authoritative once the engine is not
    explicitly running. This covers PyAEDT builds where
    ``analyze(blocking=True)`` returns before the native Icepak engine has
    finished.
    """
    if completion_barrier and not callable(desktop_attestor):
        raise ValueError(
            "standalone completion barrier requires a Desktop attestor"
        )
    started = clock()
    deadline = started + max(0.0, float(timeout_s))
    convergence = None
    running = None
    running_error = ""
    idle_since = None
    transitions = []
    previous_transition = object()
    outcome = "startup_grace_expired"
    timed_out = False
    desktop_attested = False
    desktop_attestation_attempts = 0
    desktop_attestation_error = ""
    while True:
        convergence = _thermal_convergence_telemetry(
            sim, ipk, setup, attempts=1, monitor_snapshot=monitor_snapshot
        )
        if completion_barrier:
            # Identity evidence is iteration-scoped, never sticky. In
            # particular, the exact raw proxy is re-attested immediately
            # before the running=False observation that can authorize
            # extraction and release.
            desktop_attested = False
            desktop_attestation_attempts += 1
            try:
                attested = desktop_attestor(desktop)
                if attested is None or attested is False:
                    raise RuntimeError(
                        "standalone Desktop attestation returned no proxy"
                    )
                desktop_attested = True
                desktop_attestation_error = ""
            except Exception as exc:
                desktop_attestation_error = (
                    f"{type(exc).__name__}: {str(exc)[:512]}"
                )
        try:
            running = _thermal_running_state(sim, ipk, desktop=desktop)
            running_error = ""
        except Exception as exc:
            running = None
            running_error = f"{type(exc).__name__}: {str(exc)[:512]}"
        now = clock()
        transition = (running, running_error)
        if transition != previous_transition:
            if len(transitions) < 64:
                transitions.append({
                    "elapsed_s": round(max(0.0, now - started), 3),
                    "running": running,
                    "error": running_error,
                })
            previous_transition = transition
        if convergence["thermal_convergence_reason"] in {
            "converged", "residual_threshold",
        } and (
            not completion_barrier
            or (running is False and desktop_attested)
        ):
            outcome = "terminal_fresh_monitor"
            break

        if not completion_barrier:
            if now >= deadline:
                break
            sleeper(min(max(0.0, float(poll_s)), max(0.0, deadline - now)))
            continue

        if running is True:
            # A verified running engine is an absolute no-release/no-retry
            # barrier. If it later reports idle, the monitor still receives a
            # bounded publication grace period before the result is rejected.
            idle_since = None
        elif running is False:
            if idle_since is None:
                idle_since = now
            idle_deadline = idle_since + max(
                0.0, float(idle_monitor_grace_s)
            )
            if now >= idle_deadline:
                outcome = "idle_monitor_grace_expired"
                break
        else:
            # Unknown is not proof of idle. Keep waiting for a terminal fresh
            # monitor or the overall bounded completion timeout.
            idle_since = None

        if now >= deadline:
            outcome = "completion_timeout"
            timed_out = True
            break
        sleep_until = deadline
        if running is False and idle_since is not None:
            sleep_until = min(
                sleep_until,
                idle_since + max(0.0, float(idle_monitor_grace_s)),
            )
        sleeper(min(
            max(0.0, float(poll_s)),
            max(0.0, sleep_until - now),
        ))

    if completion_barrier:
        convergence = dict(convergence)
        convergence["_thermal_completion_poll"] = {
            "schema": "thermal-standalone-completion-poll-v1",
            "timeout_s": float(timeout_s),
            "idle_monitor_grace_s": float(idle_monitor_grace_s),
            "poll_s": float(poll_s),
            "elapsed_s": round(max(0.0, clock() - started), 3),
            "outcome": outcome,
            "timed_out": timed_out,
            "last_running": running,
            "last_running_error": running_error,
            "desktop_attested": desktop_attested,
            "desktop_attestation_attempts": desktop_attestation_attempts,
            "desktop_attestation_error": desktop_attestation_error,
            "running_transitions": transitions,
        }
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


def _standalone_thermal_completion_timeout(timeout_s=None):
    """Return the bounded post-Analyze completion barrier timeout."""
    if timeout_s is None:
        raw_timeout = os.environ.get(
            "MFT_AEDT_STANDALONE_SOLVE_TIMEOUT_SECONDS", "7200"
        ).strip()
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                "MFT_AEDT_STANDALONE_SOLVE_TIMEOUT_SECONDS must be numeric"
            ) from error
        if not 30 <= timeout_s <= 86400:
            raise RuntimeError(
                "MFT_AEDT_STANDALONE_SOLVE_TIMEOUT_SECONDS must be between "
                "30 and 86400"
            )
    else:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(
                "standalone thermal completion timeout must be positive"
            )
    return timeout_s


def _pooled_thermal_poll_settings(timeout_s=None, poll_s=None):
    """Return the same bounded polling contract used by pooled EM stages."""

    if timeout_s is None:
        raw_timeout = os.environ.get(
            "MFT_AEDT_POOLED_SOLVE_TIMEOUT_SECONDS", "7200"
        ).strip()
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                "MFT_AEDT_POOLED_SOLVE_TIMEOUT_SECONDS must be numeric"
            ) from error
        if not 30 <= timeout_s <= 86400:
            raise RuntimeError(
                "MFT_AEDT_POOLED_SOLVE_TIMEOUT_SECONDS must be between 30 and 86400"
            )
    else:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("pooled thermal solve timeout must be positive")

    if poll_s is None:
        raw_poll = os.environ.get(
            "MFT_AEDT_POOLED_SOLVE_POLL_SECONDS", "2"
        ).strip()
        try:
            poll_s = float(raw_poll)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                "MFT_AEDT_POOLED_SOLVE_POLL_SECONDS must be numeric"
            ) from error
        if not 0.1 <= poll_s <= 30:
            raise RuntimeError(
                "MFT_AEDT_POOLED_SOLVE_POLL_SECONDS must be between 0.1 and 30"
            )
    else:
        poll_s = float(poll_s)
        if not math.isfinite(poll_s) or poll_s < 0:
            raise ValueError(
                "pooled thermal solve poll interval must be non-negative"
            )
    return timeout_s, poll_s


def _solve_exact_pooled_thermal_setup(
    sim, ipk, setup, setup_name=_THERMAL_SETUP_NAME,
    timeout_s=None, monitor_grace_s=30.0, poll_s=None,
    clock=time.monotonic, sleeper=time.sleep,
):
    """Blocking exact-design dispatch with fresh residual attestation.

    ``run_thermal_analysis`` already owns the session automation transaction.
    Capture the exact design while that transaction is held, then suspend it
    around ``Analyze(..., True)``.  This keeps sibling native solves parallel
    without leaving an asynchronous AEDT script macro exposed to a sibling's
    Desktop-global project activation.  The lock is restored before terminal
    messages and residual artifacts are read.
    """

    timeout_s, poll_s = _pooled_thermal_poll_settings(timeout_s, poll_s)
    preflight = _prepare_thermal_dispatch(
        sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
        setup_name=setup_name,
    )
    prepare_results = getattr(sim, "_ensure_pooled_shared_results_directory", None)
    if not callable(prepare_results):
        raise RuntimeError(
            "pooled thermal shared-results preparation is unavailable"
        )
    prepare_results(str(preflight["design"]))
    monitor_snapshot = _snapshot_thermal_monitors(sim, ipk)
    native_design = preflight["native_design"]
    desktop = _thermal_desktop_handle(sim, ipk)
    message_cursor = capture_scoped_message_cursor(
        desktop, preflight["project"], preflight["design"]
    )

    sim.solver_may_be_running = True
    started = clock()
    with _blocking_solve_log_heartbeat(
        preflight["project"], preflight["design"], setup_name
    ):
        with sim.aedt_native_solve_window():
            returned = native_design.Analyze(setup_name, True)
    if returned is not None and (type(returned) is not int or returned != 0):
        raise RuntimeError(
            "[thermal] native blocking Analyze returned invalid status: "
            f"{returned!r}"
        )

    normal_completion = False
    normal_seen_at = None
    terminal_postflight = None
    convergence = _thermal_convergence_telemetry(
        sim, ipk, setup, attempts=1, monitor_snapshot=monitor_snapshot
    )
    deadline = started + timeout_s
    terminal_monitor_reasons = {
        "converged", "residual_threshold", "monitor_malformed",
    }

    while True:
        # The blocking Analyze macro is complete.  Exact project reactivation
        # is now safe and remains protected by the caller's restored lock.
        postflight = _prepare_thermal_dispatch(
            sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
            setup_name=setup_name,
        )
        terminal_postflight = postflight
        update = advance_scoped_message_cursor(
            _thermal_desktop_handle(sim, ipk), message_cursor
        )
        message_cursor = update.cursor
        fresh_messages = tuple(dict.fromkeys((
            *update.new_messages,
            *update.new_errors,
        )))
        unmeshed_objects = _unmeshed_objects_from_messages(fresh_messages)
        terminal_messages = list(dict.fromkeys((
            *update.fatal_messages,
            *_thermal_terminal_solver_messages(fresh_messages),
        )))
        if unmeshed_objects or terminal_messages:
            evidence = " | ".join([
                *(
                    ["unmeshed=" + ",".join(unmeshed_objects)]
                    if unmeshed_objects else []
                ),
                *terminal_messages[-6:],
            ])[:2000]
            raise RuntimeError(
                "[thermal] exact AEDT design reported a terminal error: "
                f"{evidence}"
            )
        if update.normal_completion and not normal_completion:
            normal_seen_at = clock()
        normal_completion = normal_completion or update.normal_completion
        convergence = _thermal_convergence_telemetry(
            sim, postflight["native_ipk"], setup, attempts=1,
            monitor_snapshot=monitor_snapshot,
        )
        reason = convergence["thermal_convergence_reason"]
        if normal_completion and reason in terminal_monitor_reasons:
            sim.solver_may_be_running = False
            sim.save_project()
            break

        now = clock()
        if normal_completion and normal_seen_at is not None \
                and now >= normal_seen_at + max(0.0, float(monitor_grace_s)):
            # The exact Normal message proves this solver stopped, but a
            # missing fresh residual artifact makes the thermal row invalid.
            # Do not quarantine healthy siblings for an artifact failure.
            sim.solver_may_be_running = False
            break
        if now >= deadline:
            raise TimeoutError(
                "[thermal] timed out waiting for exact project/design "
                "Normal completion and a fresh residual monitor; "
                f"normal_completion={normal_completion}, "
                "reason="
                f"{convergence['thermal_convergence_reason']}"
            )
        # The blocking macro has ended, so only release the transaction while
        # waiting for message/result-file flush.  Each following postflight
        # reacquires and re-attests the exact thermal design.
        with sim.aedt_native_solve_window():
            sleeper(min(poll_s, max(0.0, deadline - now)))

    identity = {
        key: value for key, value in preflight.items()
        if key not in {"native_ipk", "native_design"}
    }
    attempts = [{
        "attempt": 1,
        "dispatch_status": "success",
        "return_type": type(returned).__name__,
        "exception_type": "",
        "exception_message": "",
        "elapsed_s": round(max(0.0, clock() - started), 3),
        "native_running": False,
        "running_state_error": "",
        "monitor_reason": convergence["thermal_convergence_reason"],
        "monitor_file": str(convergence.get("thermal_monitor_file", ""))[:256],
        "aedt_messages": [],
        "identity": identity,
        "blocking": True,
        "normal_completion": normal_completion,
    }]
    forensic_json = _thermal_forensic_json(attempts, convergence)
    logging.warning("[thermal] dispatch forensic: %s", forensic_json)
    return {
        "solve_attempts": 1,
        "analyze_call_ok": True,
        "analyze_return_false": False,
        "dispatch_status": "success",
        "dispatch_exception_type": "",
        "dispatch_exception_message": "",
        "forensic_json": forensic_json,
        "convergence": convergence,
        "postflight": terminal_postflight,
    }


def _wait_for_pooled_native_pipeline(sim):
    """Join the sealed cohort barrier without holding Desktop automation."""

    from module.aedt_pool_adapter import pooled_backend_enabled

    if not pooled_backend_enabled():
        return None
    native_window = getattr(sim, "aedt_native_solve_window", None)
    waiter = getattr(sim, "wait_for_pooled_native_pipeline", None)
    if not callable(native_window) or not callable(waiter):
        raise RuntimeError(
            "pooled native-pipeline barrier has no lock discipline"
        )
    with native_window():
        return waiter()


def _solve_exact_thermal_setup(
    sim, ipk, setup, setup_name=_THERMAL_SETUP_NAME,
    monitor_grace_s=30.0, poll_s=2.0,
    clock=time.monotonic, sleeper=time.sleep,
):
    """Dispatch only ThermalSetup, with one evidence-gated startup retry."""
    from module.aedt_pool_adapter import pooled_backend_enabled

    if pooled_backend_enabled():
        return _solve_exact_pooled_thermal_setup(
            sim, ipk, setup, setup_name=setup_name,
            monitor_grace_s=monitor_grace_s, poll_s=poll_s,
            clock=clock, sleeper=sleeper,
        )

    # Validate the fail-closed completion timeout before any native dispatch.
    # A successful PyAEDT return is not sufficient completion evidence on
    # standalone Icepak; live production has observed it return while the
    # native engine continues to consume CPU.
    completion_timeout_s = _standalone_thermal_completion_timeout()
    standalone_parallel_policy = _standalone_thermal_parallel_policy(sim)
    attempts = []
    convergence = None
    previous_snapshot = None
    for requested_attempt in range(1, 3):
        from module.aedt_pool_adapter import pooled_backend_enabled

        pooled_backend = pooled_backend_enabled()
        preflight = _prepare_thermal_dispatch(
            sim, ipk, setup, design_name=_THERMAL_DESIGN_NAME,
            setup_name=setup_name,
        )
        if pooled_backend:
            prepare_results = getattr(
                sim, "_ensure_pooled_shared_results_directory", None
            )
            if not callable(prepare_results):
                raise RuntimeError(
                    "pooled thermal shared-results preparation is unavailable"
                )
            # This runs while the caller still owns the project automation
            # transaction and before native Analyze yields it to siblings.
            prepare_results(str(preflight["design"]))

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
        native_design = preflight.pop("native_design")
        strict_parallel_attestation = bool(
            standalone_parallel_policy["strict_attestation"]
        )
        acf_before = None
        acf_evidence = {}
        acf_attestation_error = ""
        process_attestor = None
        process_evidence = {}
        process_attestation_error = ""
        if strict_parallel_attestation:
            acf_before = _thermal_hpc_acf_snapshot(native_ipk)
        # Preserve the exact raw proxy that proved this Desktop idle during
        # preflight. PyAEDT may clear its wrapper handle while restoring DSO
        # settings after Analyze, but the raw proxy remains valid for
        # AreThereSimulationsRunning.
        native_desktop = _thermal_desktop_handle(sim, native_ipk)
        attest_cached_desktop = getattr(
            sim, "_attest_cached_native_desktop", None
        )
        if not callable(attest_cached_desktop):
            raise RuntimeError(
                "standalone post-Analyze Desktop attestation is unavailable"
            )
        def desktop_attestor(desktop):
            return attest_cached_desktop(
                desktop, require_endpoint_identity=True
            )

        message_cursor = capture_scoped_message_cursor(
            native_desktop, preflight["project"], preflight["design"]
        )
        started = clock()
        status = "success"
        returned = None
        exception_type = ""
        exception_message = ""
        try:
            analyze_kwargs = {"setup": setup_name, "blocking": True}
            if not pooled_backend:
                analyze_kwargs.update({
                    "cores": standalone_parallel_policy[
                        "pyaedt_cores_argument"
                    ],
                    "tasks": standalone_parallel_policy[
                        "pyaedt_tasks_argument"
                    ],
                    "gpus": 0,
                    "use_auto_settings": standalone_parallel_policy[
                        "pyaedt_use_auto_settings_argument"
                    ],
                })
            # Passing ``cores=`` asks PyAEDT to rewrite and later restore the
            # Desktop-global Icepak DSO registry.  A pooled Desktop can have a
            # sibling MFT/IPMSM project, so it must reuse the host-owned,
            # read-back-verified profile instead.
            if pooled_backend:
                # Set uncertainty at the last possible moment. Geometry,
                # materials, mesh, and setup failures above are project-local
                # script errors and must not quarantine a healthy shared host.
                sim.solver_may_be_running = True
            else:
                # Standalone PyAEDT can return from blocking Analyze before
                # the native Icepak engine stops. Keep the same uncertainty
                # flag asserted until the completion barrier proves either a
                # terminal monitor with no verified running engine or an
                # explicit idle state.
                sim.solver_may_be_running = True
                if strict_parallel_attestation:
                    process_attestor = _StandaloneThermalProcessAttestor(
                        standalone_parallel_policy[
                            "expected_fluent_processes"
                        ]
                    ).start()
            if pooled_backend:
                # The surrounding thermal transaction protects build and
                # extraction. Yield it only around the exact project-scoped
                # native solve so sibling projects can model/solve in parallel.
                with sim.aedt_native_solve_window():
                    returned = native_design.Analyze(setup_name, True)
                if returned is not None and (
                        type(returned) is not int or returned != 0):
                    raise RuntimeError(
                        "[thermal] native Analyze returned invalid status: "
                        f"{returned!r}"
                    )
            else:
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

        messages = []
        try:
            convergence, running, running_error = (
                _poll_thermal_dispatch_evidence(
                    sim, ipk, setup, monitor_snapshot,
                    timeout_s=completion_timeout_s, poll_s=poll_s,
                    clock=clock, sleeper=sleeper,
                    completion_barrier=True,
                    idle_monitor_grace_s=monitor_grace_s,
                    desktop=native_desktop,
                    desktop_attestor=desktop_attestor,
                )
            )
        finally:
            if process_attestor is not None:
                try:
                    process_evidence = process_attestor.finish()
                except Exception as exc:
                    process_attestation_error = (
                        f"{type(exc).__name__}: {str(exc)[:4000]}"
                    )
        if strict_parallel_attestation:
            try:
                acf_evidence = _validated_thermal_hpc_acf(
                    native_ipk, standalone_parallel_policy, acf_before
                )
            except Exception as exc:
                acf_attestation_error = (
                    f"{type(exc).__name__}: {str(exc)[:4000]}"
                )
        convergence = dict(convergence)
        completion_poll = convergence.pop("_thermal_completion_poll", {})
        reason = convergence["thermal_convergence_reason"]
        parallel_attestation_error = " | ".join(
            value for value in (
                acf_attestation_error,
                process_attestation_error,
            ) if value
        )
        parallel_attestation_failed = bool(parallel_attestation_error)
        if parallel_attestation_failed:
            status = "parallel_attestation_error"
            exception_type = "NativeIcepakParallelAttestationError"
            exception_message = parallel_attestation_error[:512]
            convergence["thermal_converged"] = 0
            convergence["thermal_convergence_reason"] = (
                "parallel_process_attestation_failed"
            )
            reason = convergence["thermal_convergence_reason"]
        safe_idle = (
            completion_poll.get("desktop_attested") is True
            and running is False
        )
        attempt_record = {
            "attempt": len(attempts) + 1,
            "dispatch_status": status,
            "return_type": type(returned).__name__ if status != "exception" else "",
            "exception_type": exception_type,
            "exception_message": exception_message,
            "elapsed_s": round(max(0.0, clock() - started), 3),
            "native_running": running,
            "running_state_error": running_error,
            "completion_poll": completion_poll,
            "monitor_reason": convergence["thermal_convergence_reason"],
            "monitor_file": str(
                convergence.get("thermal_monitor_file", "")
            )[:256],
            "aedt_messages": messages,
            "identity": preflight,
            "mesh_preflight": {
                "schema": getattr(
                    sim, "thermal_mesh_preflight", {}
                ).get("schema", ""),
                "passed": getattr(
                    sim, "thermal_mesh_preflight", {}
                ).get("passed", False),
                "mesh_plan_sha256": getattr(
                    sim, "thermal_mesh_preflight", {}
                ).get("mesh_plan_sha256", ""),
                "mesh_artifact_readback_passed": getattr(
                    sim, "thermal_mesh_preflight", {}
                ).get("mesh_artifact_readback_passed", False),
                "mesh_mapping_coverage_passed": getattr(
                    sim, "thermal_mesh_preflight", {}
                ).get("mesh_mapping_coverage_passed", False),
                "analysis_dispatched_after_premesh": True,
            },
            "parallel_policy": dict(standalone_parallel_policy),
            "parallel_attestation": {
                "schema": "thermal-standalone-parallel-attestation-v1",
                "passed": (
                    strict_parallel_attestation
                    and not parallel_attestation_failed
                    and acf_evidence.get("passed") is True
                    and process_evidence.get("passed") is True
                ) if strict_parallel_attestation else None,
                "strict": strict_parallel_attestation,
                "acf": acf_evidence,
                "acf_error": acf_attestation_error,
                "process": process_evidence,
                "process_error": process_attestation_error,
            },
        }
        if not safe_idle:
            # No model/message/save/retry automation is permitted after a
            # premature Analyze return until the exact cached Desktop is
            # re-attested and explicitly reports idle.
            attempts.append(attempt_record)
            forensic_json = _thermal_forensic_json(attempts, convergence)
            logging.warning(
                "[thermal] unsafe completion forensic: %s", forensic_json
            )
            raise RuntimeError(
                "standalone thermal completion barrier did not prove exact "
                "Desktop idle: " + forensic_json[:4000]
            )
        sim.solver_may_be_running = False
        try:
            message_update = advance_scoped_message_cursor(
                native_desktop, message_cursor
            )
        except Exception as exc:
            raise RuntimeError(
                "standalone thermal post-Analyze exact message scan failed: "
                f"{type(exc).__name__}: {str(exc)[:512]}"
            ) from exc
        fresh_messages = tuple(dict.fromkeys((
            *message_update.new_messages,
            *message_update.new_errors,
        )))
        messages = _bounded_message_values(fresh_messages)
        unmeshed_objects = _unmeshed_objects_from_messages(
            fresh_messages
        )
        terminal_messages = list(dict.fromkeys((
            *message_update.fatal_messages,
            *_thermal_terminal_solver_messages(fresh_messages),
        )))
        nonretryable_native_failure = bool(
            unmeshed_objects or terminal_messages
            or parallel_attestation_failed
        )
        if nonretryable_native_failure:
            status = "terminal_error"
            exception_type = "NativeIcepakTerminalError"
            exception_message = " | ".join(
                [
                    *(
                        ["unmeshed=" + ",".join(unmeshed_objects)]
                        if unmeshed_objects else []
                    ),
                    *terminal_messages[-6:],
                ]
            )[:512]
            convergence = dict(convergence)
            convergence["thermal_converged"] = 0
            convergence["thermal_convergence_reason"] = (
                "native_unmeshed_objects"
                if unmeshed_objects else "native_terminal_error"
            )
            reason = convergence["thermal_convergence_reason"]
        if (
            status != "success"
            or convergence["thermal_convergence_reason"] != "converged"
        ):
            preflight["model_context"] = _bounded_thermal_model_context(ipk)
        try:
            sim.save_project()
        except Exception:
            pass
        attempt_record["elapsed_s"] = round(
            max(0.0, clock() - started), 3
        )
        attempt_record["aedt_messages"] = messages
        attempt_record["unmeshed_objects"] = unmeshed_objects
        attempt_record["terminal_messages"] = _bounded_message_values(
            terminal_messages, limit=12, char_limit=4096
        )
        attempt_record["message_scan_complete"] = True
        attempt_record["dispatch_status"] = status
        attempt_record["exception_type"] = exception_type
        attempt_record["exception_message"] = exception_message
        attempt_record["monitor_reason"] = (
            convergence["thermal_convergence_reason"]
        )
        attempts.append(attempt_record)

        if nonretryable_native_failure:
            logging.error(
                "[thermal] refusing solver retry after exact native "
                "terminal evidence: unmeshed=%s terminal=%s",
                unmeshed_objects,
                _bounded_message_values(
                    terminal_messages, limit=6, char_limit=2048
                ),
            )
            break
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
    sim.thermal_parallel_evidence = {
        "schema": "thermal-standalone-parallel-evidence-v1",
        "policy": dict(standalone_parallel_policy),
        "attempts": [
            dict(item.get("parallel_attestation", {}))
            for item in attempts
        ],
        "passed": bool(attempts) and all(
            item.get("parallel_attestation", {}).get("passed") is True
            for item in attempts
        ) if standalone_parallel_policy["strict_attestation"] else None,
    }
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

    # The retained explicit Rx foils still require the physical inter-turn
    # electrical insulation. Earlier Full models left these gaps as fluid,
    # while the homogenized middle pack included the same insulation in its
    # effective conductivity. That topology disconnected the end foils
    # thermally and produced non-physical 250--3000 C local temperatures.
    #
    # Do not add or attest this extra project material for the standard
    # n_explicit_turns=0 topology: preserving that path avoids extending the
    # production TIM3 contract when no explicit-insulation solid can exist.
    needs_explicit_insulation = False
    if "n_explicit_turns" in df.columns and "N2_main" in df.columns:
        n_explicit = int(df["n_explicit_turns"].iloc[0])
        turn_counts = [int(df["N2_main"].iloc[0])]
        if "N2_side" in df.columns:
            turn_counts.append(int(df["N2_side"].iloc[0]))
        needs_explicit_insulation = any(
            _explicit_rx_gap_indices(turn_count, n_explicit)
            for turn_count in turn_counts
        )

    rx_insulation_readback = None
    if needs_explicit_insulation:
        if RX_EXPLICIT_INSULATION_MATERIAL not in mats.material_keys:
            m = mats.add_material(RX_EXPLICIT_INSULATION_MATERIAL)
        else:
            m = mats[RX_EXPLICIT_INSULATION_MATERIAL]
        if m is None or m is False:
            raise RuntimeError("winding_insulation material is unavailable")
        # Reapply unconditionally because a copied Maxwell design may already
        # contain a stale project material with this name.
        m.conductivity = 0
        m.thermal_conductivity = k_ins
        m.mass_density = 1200
        m.specific_heat = 1000
        rx_insulation_readback = _rx_insulation_native_readback(
            mats, k_ins
        )

    if "thermal_pad" not in mats.material_keys:
        m = mats.add_material("thermal_pad")
    else:
        m = mats["thermal_pad"]
    if m is None or m is False:
        raise RuntimeError("thermal_pad material is unavailable")

    # Maxwell creates this project material at 0.2 W/(m*K) before the Icepak
    # design is copied. Reapply both properties unconditionally: merely
    # checking material existence silently retained the stale Maxwell value.
    m.conductivity = 0
    m.thermal_conductivity = THERMAL_PAD_CONDUCTIVITY_W_MK
    thermal_pad_readback = _thermal_pad_native_readback(mats)

    if rx_insulation_readback is not None:
        thermal_pad_readback["rx_explicit_insulation"] = (
            rx_insulation_readback
        )
    return k_in, k_th, thermal_pad_readback


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


def _explicit_rx_gap_indices(turn_count, n_explicit):
    """Return physical gaps whose two neighboring foils stay explicit."""
    count = int(n_explicit)
    turns = int(turn_count)
    if turns <= 1 or count == 0:
        return []
    if count < 0 or 2 * count >= turns:
        return list(range(turns - 1))
    indices = list(range(max(count - 1, 0)))
    indices.extend(range(turns - count, turns - 1))
    return sorted(set(indices))


def _expected_rx_insulation_counts(df, n_explicit, mode):
    """Return the exact retained insulation-ring count for each Rx pack."""
    mode = str(mode)
    main_count = len(_explicit_rx_gap_indices(
        int(df["N2_main"].iloc[0]), n_explicit
    ))
    side_turns = int(df["N2_side"].iloc[0])
    side_count = (
        len(_explicit_rx_gap_indices(side_turns, n_explicit))
        if side_turns > 0 else 0
    )
    return {
        "Rx_main_insulation": main_count,
        "Rx_side_insulation": side_count,
        "Rx_side2_insulation": side_count if mode == "full" else 0,
    }


def _build_explicit_rx_insulation(
        ipk, df, prefix, name, offset_x, n_explicit, height):
    """Fill retained-foil gaps with the candidate's solid insulation.

    The middle homogenized block already includes every internal insulation
    layer.  Only gaps between adjacent retained explicit foils are created
    here; the explicit-to-block interfaces already meet the homogenized solid.
    """
    turn_count = int(df[f"N2_{prefix}"].iloc[0])
    if turn_count <= 1:
        return []
    turn_count, cw, x_pos, y_pos = _rx_layout(df, prefix)
    gap_indices = _explicit_rx_gap_indices(turn_count, n_explicit)
    if not gap_indices:
        return []

    expected_gap = float(df["gap2"].iloc[0])
    if not math.isfinite(expected_gap) or expected_gap <= 0:
        raise RuntimeError(
            f"{name} explicit insulation requires finite gap2 > 0"
        )
    insulation = []
    for index in gap_indices:
        x = 0.5 * (x_pos[index] + x_pos[index + 1])
        y = 0.5 * (y_pos[index] + y_pos[index + 1])
        gap_x = x_pos[index + 1] - x_pos[index] - cw
        gap_y = y_pos[index + 1] - y_pos[index] - cw
        if (
            not math.isclose(gap_x, expected_gap, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(gap_y, expected_gap, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise RuntimeError(
                f"{name} explicit insulation gap mismatch at {index}: "
                f"x={gap_x!r}, y={gap_y!r}, expected={expected_gap!r}"
            )
        points = [
            [f"{x}mm + {offset_x}mm", f"{y}mm", "0mm"],
            [f"{-x}mm + {offset_x}mm", f"{y}mm", "0mm"],
            [f"{-x}mm + {offset_x}mm", f"{-y}mm", "0mm"],
            [f"{x}mm + {offset_x}mm", f"{-y}mm", "0mm"],
            [f"{x}mm + {offset_x}mm", f"{y}mm", "0mm"],
        ]
        obj = ipk.modeler.create_polyline(
            points=points,
            name=f"{name}_insulation_gap_{index}",
            material=RX_EXPLICIT_INSULATION_MATERIAL,
            xsection_orient="Auto",
            xsection_type="Rectangle",
            xsection_width=expected_gap,
            xsection_height=float(height),
            xsection_num_seg=6,
            xsection_topwidth=expected_gap,
        )
        if not obj:
            raise RuntimeError(
                f"failed to create {name} explicit insulation gap {index}"
            )
        insulation.append(obj)
    return insulation


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


def _build_rx_group(
        ipk, df, prefix, name, offset_x, n_explicit, height,
        insulation_sink=None):
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
    insulation = _build_explicit_rx_insulation(
        ipk, df, prefix, name, offset_x, n_explicit, height
    )
    if insulation_sink is not None:
        insulation_sink.extend(insulation)
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
    expected_insulation_counts = _expected_rx_insulation_counts(
        df, n_exp, mode
    )

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
    objs["Rx_main_insulation"] = []
    objs["Rx_main_explicit"], objs["Rx_main_blocks"] = _build_rx_group(
        ipk, df, "main", "Rx_main", 0.0, n_exp, nwh2,
        insulation_sink=objs["Rx_main_insulation"],
    )

    objs["Rx_side_explicit"] = []
    objs["Rx_side_blocks"] = []
    objs["Rx_side_insulation"] = []
    objs["Rx_side2_explicit"] = []
    objs["Rx_side2_blocks"] = []
    objs["Rx_side2_insulation"] = []
    if int(df["N2_side"].iloc[0]) > 0:
        off = l1 + l2 + l1 / 2
        objs["Rx_side_explicit"], objs["Rx_side_blocks"] = _build_rx_group(
            ipk, df, "side", "Rx_side", -off, n_exp, nwh2,
            insulation_sink=objs["Rx_side_insulation"],
        )
        if mode == "full":
            # 대칭 모드에서는 +x 측 링이 어차피 절단 제거되므로 생성 생략 (모델링 시간 절약)
            objs["Rx_side2_explicit"], objs["Rx_side2_blocks"] = _build_rx_group(
                ipk, df, "side", "Rx_side2", +off, n_exp, nwh2,
                insulation_sink=objs["Rx_side2_insulation"],
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

    actual_insulation_counts = {
        key: len(objs.get(key, [])) for key in _RX_INSULATION_KEYS
    }
    if actual_insulation_counts != expected_insulation_counts:
        raise RuntimeError(
            "explicit Rx insulation topology mismatch after symmetry split: "
            f"actual={actual_insulation_counts}, "
            f"expected={expected_insulation_counts}"
        )

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
    _, _, thermal_pad_readback = _create_thermal_materials(ipk, df)
    objs = _build_geometry(ipk, sim, eighth=eighth, mode=mode)
    rx_insulation_counts = {
        key: len(objs.get(key, [])) for key in _RX_INSULATION_KEYS
    }
    rx_insulation_metadata = _rx_insulation_result_metadata(
        thermal_pad_readback, rx_insulation_counts
    )
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
    thermal_mesh_plan = _assign_thermal_mesh(
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

    from module.aedt_pool_adapter import pooled_backend_enabled
    pooled_backend = pooled_backend_enabled()
    if pooled_backend:
        # A Desktop-wide idle state cannot identify one project in an attached
        # multi-project AEDT host.  Preserve the existing project-scoped
        # nonblocking Analyze protocol; standalone recovery alone opts into
        # strict native GenerateMesh attestation.
        thermal_mesh_preflight = (
            _pooled_thermal_mesh_preflight_not_applicable(
                thermal_mesh_plan
            )
        )
    else:
        # Generate the exact native grid before Analyze and require native OO
        # assignment, exact messages, stable filesystem artifacts, and the
        # cached Desktop idle barrier.  This turns deterministic zero-mesh
        # solids into one pre-solve failure instead of two 10--14 minute runs.
        thermal_mesh_preflight = _generate_and_attest_thermal_mesh(
            sim, ipk, setup, thermal_mesh_plan
        )
    # The exact evidence belongs to the same solver transaction and is copied
    # into every attempt's forensic record.  Keep the pre-dispatch value false;
    # it is promoted only after the native Analyze call is actually made.
    sim.thermal_mesh_preflight = dict(thermal_mesh_preflight)

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
    if int(solve_attempts) < 1:
        raise RuntimeError(
            "thermal Analyze was not dispatched after native mesh preflight"
        )
    thermal_mesh_preflight = dict(thermal_mesh_preflight)
    thermal_mesh_preflight["analysis_dispatched_after_premesh"] = True
    sim.thermal_mesh_preflight = dict(thermal_mesh_preflight)
    thermal_mesh_metadata = _thermal_mesh_result_metadata(
        thermal_mesh_plan, thermal_mesh_preflight
    )
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
    from module.aedt_pool_adapter import pooled_backend_enabled
    pooled_backend = pooled_backend_enabled()
    if pooled_backend:
        # ``AreThereSimulationsRunning == False`` is only a momentary
        # Desktop-idle observation.  A sibling may still be waiting for the
        # automation lock to build or launch its final thermal design.  If the
        # first completed project starts the long Desktop-global field-summary
        # export in that gap, it starves the sibling's remaining native solve.
        # Mark this exact sealed-batch member complete and wait, with every
        # outer automation-lock depth suspended, until all cohort members have
        # finished their full native pipeline.
        barrier_started = time.monotonic()
        _wait_for_pooled_native_pipeline(sim)
        sim.stage_timings["stage_time_native_pipeline_barrier_s"] = (
            time.monotonic() - barrier_started
        )
    extraction_started = time.monotonic()

    # ---- 온도 추출 (필드 계산기 직접 평가 - 리포트 기계 미사용) ----
    # 프로브 시트 (회귀학습용 주력 데이터: 위치 고정, 보간값이라 메시 스파이크에 강함)
    # + 그룹별 체적 평균/최대
    temps = {}
    # Object3d instances retain the editor proxy that was active when geometry
    # was built.  A pooled native solve deliberately yields the Desktop lock,
    # so siblings can switch or close projects before thermal extraction
    # resumes.  Preserve only immutable extraction identity and the geometry
    # dimension known at creation time; never ask a pre-solve Object3d for
    # ``is3d`` after that boundary.
    probe = []
    for s in probe_sheets:
        name = str(s.name)
        probe.append((name, False, f"{name}_max", "max"))
        probe.append((name, False, f"{name}_mean", "mean"))
    actual_probe_sheet_names = [sheet.name for sheet in probe_sheets]
    expected_probe_sheet_names = list(getattr(
        probe_sheets, "expected_names", actual_probe_sheet_names
    ))
    probe_failures = list(getattr(probe_sheets, "failures", []))
    vol_objs = (objs["Tx"] + objs["Rx_main_explicit"] + objs["Rx_main_blocks"]
                + objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
                + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"]
                + objs["core"] + objs.get("wcp_pads", []))
    mesh_postsolve_probe_objects = (
        _thermal_mesh_postsolve_probe_object_names(objs)
    )
    for o in vol_objs:
        name = str(o.name)
        probe.append((name, True, f"T_mean_{name}", "mean"))
        probe.append((name, True, f"T_max_{name}", "max"))

    expected_probe_cols = [
        f"{name}_{stat}"
        for name in expected_probe_sheet_names
        for stat in ("max", "mean")
    ]
    field_expected_cols = list(dict.fromkeys(
        [col for _, _, col, _ in probe] + expected_probe_cols
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
        "T_max_Tx": [str(obj.name) for obj in objs["Tx"]],
        "T_max_Rx_main": [
            str(obj.name)
            for obj in objs["Rx_main_explicit"] + objs["Rx_main_blocks"]
        ],
        "T_max_Rx_side": [
            str(obj.name)
            for obj in (
                objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
                + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"]
            )
        ],
        "T_max_core": [str(obj.name) for obj in objs["core"]],
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
            "thermal_mesh_postsolve_probe_object_count": [
                len(mesh_postsolve_probe_objects)
            ],
            "thermal_mesh_postsolve_probe_missing_count": [
                len(mesh_postsolve_probe_objects)
            ],
            "thermal_mesh_postsolve_probe_missing_objects_json": [
                json.dumps(mesh_postsolve_probe_objects)
            ],
            "thermal_mesh_postsolve_probe_complete": [0],
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
            **thermal_mesh_metadata,
            **_thermal_pad_result_metadata(thermal_pad_readback),
            **rx_insulation_metadata,
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

    # Analyze yielded the pooled Desktop automation lock for the full native
    # solve.  Cached project/design child proxies are therefore no longer
    # authoritative even when the Desktop itself is healthy.  Re-enumerate
    # this lease's exact project and re-attest the Icepak design/setup before
    # any post-processing call can touch those proxies.
    # The cohort barrier deliberately allowed siblings to reactivate their
    # projects/designs.  Never reuse the pre-barrier child proxy in pooled
    # mode; re-enumerate and attest this lease again under the restored lock.
    postflight = (
        None if pooled_backend
        else solve_result.get("postflight")
    )
    if not postflight:
        try:
            postflight = _prepare_thermal_dispatch(
                sim,
                ipk,
                setup,
                design_name=_THERMAL_DESIGN_NAME,
                setup_name=_THERMAL_SETUP_NAME,
            )
        except Exception as exc:
            raise RuntimeError(
                "thermal post-solve exact project/design rebind failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    native_ipk = postflight["native_ipk"]
    try:
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
        for name, is_volume, _col, _op in entries:
            key = (name, is_volume)
            if key in seen:
                continue
            seen.add(key)
            fs.add_calculation(
                "Object", "Volume" if is_volume else "Surface",
                name, "Temperature"
            )
        df_fs = _field_summary_data_frame(sim, fs, solution)
        if df_fs is None or isinstance(df_fs, bool) or not hasattr(df_fs, "columns") or not len(df_fs):
            raise RuntimeError(f"field summary returned {type(df_fs).__name__} (no data)")
        # 컬럼: Entity/Geometry/Quantity/Min/Max/Mean ... (버전에 따라 대소문자 상이)
        cols = {str(c).strip().lower(): c for c in df_fs.columns}
        name_c = cols.get("geometry name", cols.get("entity name", list(df_fs.columns)[2]))
        unit_c = cols.get("unit", cols.get("units"))
        got = {}
        for name, _is_volume, col, op in entries:
            row = df_fs[df_fs[name_c].astype(str) == name]
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

    def _scalar_probe_temperature(name, op):
        """Independent saved-field fallback, deliberately limited to sheets."""
        post = _post_of(native_ipk)
        getter = getattr(post, "get_scalar_field_value", None)
        if not callable(getter):
            raise RuntimeError("post processor has no scalar field API")
        value = getter(
            "Temp",
            scalar_function="Maximum" if op == "max" else "Mean",
            solution=solution,
            object_name=name,
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
    with _pooled_field_summary_window(
            sim, native_ipk, setup) as extraction_ipk:
        native_ipk = extraction_ipk
        for attempt in range(1, 4):
            missing_entries = [entry for entry in probe if entry[2] not in temps]
            if not missing_entries:
                break
            field_summary_attempts = attempt
            try:
                _activate_thermal_design(
                    native_ipk, design_name=_THERMAL_DESIGN_NAME
                )
                temps.update(_field_summary_bulk(missing_entries))
                _refresh_core_probe_aggregates()
                _refresh_rx_side_face_aggregates()
            except Exception as e:
                logging.warning(
                    f"[thermal] field summary attempt {attempt}/3 failed: {e}"
                )
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
        # AEDT can fail ExportFieldsSummary while the saved surface field
        # remains readable. Use the replay-proven scalar API only for missing
        # probe-sheet statistics. Never replace a missing modeled-volume
        # maximum with a probe: that indicates a zero-mesh-volume failure.
        for name, is_volume, col, op in probe:
            if col in temps or col in optional_cols or is_volume:
                continue
            calc_attempts += 1
            try:
                _activate_thermal_design(
                    native_ipk, design_name=_THERMAL_DESIGN_NAME
                )
                temps[col] = _scalar_probe_temperature(name, op)
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

    def _group_max(group_names):
        # A partial maximum can silently understate component temperature. Require
        # every modeled maximum for the physical group before emitting its summary.
        cols = list(dict.fromkeys(f"T_max_{name}" for name in group_names))
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
    mesh_postsolve_probe_status = _thermal_mesh_postsolve_probe_status(
        mesh_postsolve_probe_objects, temps
    )
    mesh_postsolve_probe_missing = (
        mesh_postsolve_probe_status["missing_objects"]
    )
    mesh_postsolve_probe_complete = (
        mesh_postsolve_probe_status["complete"]
    )
    solution_data_available = n_fs > 0
    solved = (
        convergence["thermal_converged"] == 1
        and solution_data_available
        and required_complete
        and mesh_postsolve_probe_complete
    )
    if required_missing_count:
        extraction_failure_reason = "required_volume_temperature_missing"
    elif mesh_postsolve_probe_missing:
        extraction_failure_reason = "mesh_postsolve_probe_missing"
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
        "thermal_mesh_postsolve_probe_object_count": [
            len(mesh_postsolve_probe_objects)
        ],
        "thermal_mesh_postsolve_probe_missing_count": [
            len(mesh_postsolve_probe_missing)
        ],
        "thermal_mesh_postsolve_probe_missing_objects_json": [
            json.dumps(mesh_postsolve_probe_missing)
        ],
        "thermal_mesh_postsolve_probe_complete": [
            1 if mesh_postsolve_probe_complete else 0
        ],
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
        **thermal_mesh_metadata,
        **_thermal_pad_result_metadata(thermal_pad_readback),
        **rx_insulation_metadata,
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
