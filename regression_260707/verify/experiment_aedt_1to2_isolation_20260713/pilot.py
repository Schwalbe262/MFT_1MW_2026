#!/usr/bin/env python3
"""Isolated one-Desktop/two-project AEDT fault-isolation pilot.

This script is deliberately standalone.  It creates two synthetic Maxwell 3D
projects in one AEDT 2025 R2 Desktop, launches both solves asynchronously, and
collects process, FlexLM, identity, timing, and gRPC evidence.  It never calls
the Desktop-global ``stop_simulations`` until the healthy project has finished.

An exact SIGTERM is permitted only when a unique ``3dedy`` child is mapped to
project A by a two-phase launch and one-then-two local Maxwell checkout proof.
If that proof is absent the destructive phase is skipped and activation fails
closed.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback


FEATURES = ("electronics_desktop", "electronics3d_gui", "elec_solve_maxwell")
SOLVER_COMM = {"3dedy", "maxwellcomengin", "maxwellcomengine"}


class Recorder:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.events = self.root / "events.jsonl"

    def event(self, kind: str, **data):
        payload = {
            "kind": kind,
            "epoch": time.time(),
            "elapsed_s": time.time() - self.started,
            **data,
        }
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        return payload

    def write_json(self, name: str, value):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proc_snapshot(desktop_pid: int | None = None) -> list[dict]:
    records = []
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(errors="replace")
            values = {}
            for line in status.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            real_uid = int(values.get("Uid", "-1").split()[0])
            if real_uid != uid:
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            for secret_name in (
                "ANSYS_LICENSING_CONTEXT_DATA",
                "ANSYS_LICENSING_WORKFLOW_SESSION",
                "ANSYSEM_FEATURE_CONTROL",
            ):
                cmdline = re.sub(
                    rf"{secret_name}='[^']*'",
                    f"{secret_name}='<redacted>'",
                    cmdline,
                )
            records.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(values.get("PPid", "-1")),
                    "comm": values.get("Name", ""),
                    "state": values.get("State", ""),
                    "cmdline": cmdline,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    by_pid = {item["pid"]: item for item in records}

    def descends(pid: int, ancestor: int) -> bool:
        seen = set()
        while pid in by_pid and pid not in seen:
            if pid == ancestor:
                return True
            seen.add(pid)
            pid = by_pid[pid]["ppid"]
        return False

    for item in records:
        item["desktop_descendant"] = bool(
            desktop_pid and descends(item["pid"], int(desktop_pid))
        )
    return sorted(records, key=lambda item: item["pid"])


def descendant_solver_roots(snapshot: list[dict], desktop_pid: int) -> list[dict]:
    selected = {
        item["pid"]: item
        for item in snapshot
        if item.get("desktop_descendant")
        and item["pid"] != desktop_pid
        and item["comm"].casefold() in SOLVER_COMM
    }
    return [
        item for pid, item in selected.items() if item["ppid"] not in selected
    ]


def parse_lmstat(text: str, username: str, hostname: str) -> dict:
    result = {}
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Users of ")]
    starts.append(len(lines))
    host_short = hostname.split(".")[0]
    for pos, start in enumerate(starts[:-1]):
        header = lines[start]
        match = re.match(
            r"Users of ([^:]+):.*Total of (\d+) licenses issued;\s+"
            r"Total of (\d+) licenses in use",
            header,
        )
        if not match or match.group(1) not in FEATURES:
            continue
        block = lines[start : starts[pos + 1]]
        local_lines = [
            line
            for line in block[1:]
            if re.search(rf"\b{re.escape(username)}\b", line)
            and (
                re.search(rf"\b{re.escape(hostname)}\b", line)
                or re.search(rf"\b{re.escape(host_short)}\b", line)
            )
        ]
        local_pids = []
        for line in local_lines:
            fields = line.split()
            if len(fields) >= 4 and fields[3].isdigit():
                local_pids.append(int(fields[3]))
        result[match.group(1)] = {
            "issued": int(match.group(2)),
            "global_in_use": int(match.group(3)),
            "local_checkout_count": len(local_lines),
            "local_pids": local_pids,
            "local_lines": local_lines,
        }
    for feature in FEATURES:
        result.setdefault(
            feature,
            {
                "issued": None,
                "global_in_use": None,
                "local_checkout_count": 0,
                "local_pids": [],
                "local_lines": [],
            },
        )
    return result


def capture_lmstat(rec: Recorder, label: str, lmutil: str, server: str) -> dict:
    completed = subprocess.run(
        [lmutil, "lmstat", "-c", server, "-a"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
        check=False,
    )
    lm_dir = rec.root / "lmstat"
    lm_dir.mkdir(exist_ok=True)
    raw_path = lm_dir / f"{label}.txt"
    raw_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    parsed = parse_lmstat(completed.stdout, getpass.getuser(), socket.gethostname())
    rec.event(
        "lmstat",
        label=label,
        returncode=completed.returncode,
        parsed=parsed,
        raw_path=str(raw_path),
    )
    return parsed


def grpc_probe(rec: Recorder, label: str, port: int, expected_pid: int) -> dict:
    code = r'''
import json, sys
from ansys.aedt.core import Desktop
d = Desktop(version="2025.2", non_graphical=True, new_desktop=False,
            close_on_exit=False, port=int(sys.argv[1]))
out = {
    "pid": int(d.odesktop.GetProcessID()),
    "projects": sorted(str(v) for v in d.project_list),
    "running": bool(d.are_there_simulations_running),
}
print(json.dumps(out, sort_keys=True))
d.release_desktop(close_projects=False, close_on_exit=False)
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(port)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        value = json.loads(lines[-1]) if lines else {}
        ok = completed.returncode == 0 and value.get("pid") == expected_pid
        result = {
            "ok": ok,
            "returncode": completed.returncode,
            "value": value,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "ok": False,
            "timeout": True,
            "stdout_tail": str(exc.stdout or "")[-2000:],
            "stderr_tail": str(exc.stderr or "")[-2000:],
        }
    rec.event("grpc_probe", label=label, **result)
    return result


def create_project(app, tag: str, *, cylinders: int, mesh_mm: float, passes: int):
    app.modeler.model_units = "mm"
    objects = []
    top_faces = []
    bottom_faces = []
    side = int(math.ceil(math.sqrt(cylinders)))
    for index in range(cylinders):
        row, col = divmod(index, side)
        obj = app.modeler.create_cylinder(
            orientation="X",
            origin=[0, row * 22.0, col * 22.0],
            radius=8.0,
            height=100.0,
            num_sides=24,
            name=f"{tag}_conductor_{index:02d}",
            material="copper",
        )
        if not obj:
            raise RuntimeError(f"failed to create {tag} cylinder {index}")
        objects.append(obj)
        top_faces.append(obj.top_face_x.id)
        bottom_faces.append(obj.bottom_face_x.id)
    # Maxwell requires each Current terminal boundary to contain exactly one
    # face selection.  Keep every disconnected conduction path explicit.
    for index, (top_face, bottom_face) in enumerate(zip(top_faces, bottom_faces)):
        if not app.assign_current(
            top_face,
            amplitude="1A",
            solid=True,
            name=f"{tag}_current_{index:02d}",
        ):
            raise RuntimeError(f"failed to assign {tag} current {index}")
        if not app.assign_voltage(
            bottom_face,
            amplitude="0V",
            name=f"{tag}_ground_{index:02d}",
        ):
            raise RuntimeError(f"failed to assign {tag} ground {index}")
    mesh = app.mesh.assign_length_mesh(
        [obj.name for obj in objects],
        inside_selection=True,
        maximum_length=f"{mesh_mm}mm",
        maximum_elements=2_000_000,
        name=f"{tag}_length",
    )
    if not mesh:
        raise RuntimeError(f"failed to assign {tag} mesh")
    setup = app.create_setup(
        name="Setup1",
        MaximumPasses=passes,
        MinimumPasses=passes,
        MinimumConvergedPasses=passes,
        PercentRefinement=2,
        PercentError=1e-12,
        SolveFieldOnly=False,
        SolveMatrixAtLast=True,
    )
    if not setup:
        raise RuntimeError(f"failed to create {tag} setup")
    app.save_project()
    return {
        "objects": sorted(obj.name for obj in objects),
        "setups": sorted(app.setup_names),
        "project": app.project_name,
        "design": app.design_name,
    }


def process_phase(rec: Recorder, label: str, desktop_pid: int) -> tuple[list[dict], list[dict]]:
    snapshot = proc_snapshot(desktop_pid)
    roots = descendant_solver_roots(snapshot, desktop_pid)
    rec.write_json(f"processes/{label}.json", snapshot)
    rec.event(
        "process_snapshot",
        label=label,
        desktop_pid=desktop_pid,
        descendant_count=sum(bool(item["desktop_descendant"]) for item in snapshot),
        solver_roots=roots,
    )
    return snapshot, roots


def licensed_solver_processes(
    snapshot: list[dict], license_state: dict, desktop_pid: int
) -> list[dict]:
    """Map FlexLM's local Maxwell checkout PIDs back to Desktop descendants."""
    by_pid = {item["pid"]: item for item in snapshot}
    pids = license_state.get("elec_solve_maxwell", {}).get("local_pids", [])
    return [
        by_pid[pid]
        for pid in pids
        if pid in by_pid
        and by_pid[pid].get("desktop_descendant")
        and pid != desktop_pid
    ]


def wait_for_phase(
    rec: Recorder,
    label: str,
    desktop_pid: int,
    lmutil: str,
    server: str,
    expected_solver_checkouts: int,
    timeout_s: float,
) -> dict:
    deadline = time.time() + timeout_s
    best = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        snapshot, heuristic_roots = process_phase(
            rec, f"{label}_{attempt:03d}", desktop_pid
        )
        license_state = capture_lmstat(
            rec, f"{label}_{attempt:03d}", lmutil, server
        )
        roots = licensed_solver_processes(snapshot, license_state, desktop_pid)
        local = license_state["elec_solve_maxwell"]["local_checkout_count"]
        best = {
            "attempt": attempt,
            "roots": roots,
            "heuristic_roots": heuristic_roots,
            "license": license_state,
            "local_solver_checkouts": local,
        }
        if local >= expected_solver_checkouts and len(roots) >= expected_solver_checkouts:
            rec.event("phase_ready", label=label, **best)
            return best
        time.sleep(2)
    rec.event("phase_timeout", label=label, best=best)
    return best or {}


def export_convergence(app, target: Path) -> dict:
    try:
        result = app.export_convergence("Setup1", output_file=str(target))
        exists = target.exists() and target.stat().st_size > 0
        return {
            "call_result": str(result),
            "exists": exists,
            "size": target.stat().st_size if target.exists() else 0,
        }
    except Exception as exc:
        return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}


def safe_project_identity(app) -> dict:
    try:
        return {
            "ok": True,
            "project": app.project_name,
            "design": app.design_name,
            "objects": sorted(app.modeler.object_names),
            "setups": sorted(app.setup_names),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--lmutil", required=True)
    parser.add_argument("--license-server", required=True)
    parser.add_argument("--cylinders", type=int, default=12)
    parser.add_argument("--mesh-mm", type=float, default=0.55)
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--phase-timeout", type=float, default=180)
    parser.add_argument("--completion-timeout", type=float, default=900)
    args = parser.parse_args()

    artifact = Path(args.artifact_dir).resolve()
    rec = Recorder(artifact)
    projects = artifact / "projects"
    projects.mkdir(exist_ok=True)
    verdict = {
        "schema": 1,
        "activation_allowed": False,
        "quarantine_policy": (
            "On any ambiguous mapping or shared-failure evidence: stop attaching new "
            "work, mark the Desktop quarantined, let or stop only after the healthy "
            "sibling is terminal, drain the allocation, and requeue unfinished work "
            "onto a fresh one-project Desktop."
        ),
    }
    desktop = None
    app_a = None
    app_b = None
    try:
        import ansys.aedt.core as pyaedt
        from ansys.aedt.core import Desktop, Maxwell3d

        identity = {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "user": getpass.getuser(),
            "uid": os.getuid(),
            "pid": os.getpid(),
            "python": sys.version,
            "pyaedt": pyaedt.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "artifact": str(artifact),
        }
        rec.write_json("identity.json", identity)
        rec.event("pilot_start", identity=identity, args=vars(args))
        baseline_license = capture_lmstat(
            rec, "00_before_desktop", args.lmutil, args.license_server
        )
        process_phase(rec, "00_before_desktop", 0)

        desktop = Desktop(
            version="2025.2",
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        desktop_pid = int(desktop.odesktop.GetProcessID())
        port = int(desktop.port)
        rec.event(
            "desktop_started",
            desktop_pid=desktop_pid,
            grpc_port=port,
            aedt_version=str(desktop.aedt_version_id),
        )
        desktop_license = capture_lmstat(
            rec, "01_after_desktop", args.lmutil, args.license_server
        )
        grpc_before = grpc_probe(rec, "before_projects", port, desktop_pid)

        project_a_path = projects / "pilot_A.aedt"
        project_b_path = projects / "pilot_B.aedt"
        app_a = Maxwell3d(
            project=str(project_a_path),
            design="DesignA",
            solution_type="ElectroDCConduction",
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
            port=port,
        )
        identity_a = create_project(
            app_a,
            "A",
            cylinders=args.cylinders,
            mesh_mm=args.mesh_mm,
            passes=args.passes,
        )
        app_b = Maxwell3d(
            project=str(project_b_path),
            design="DesignB",
            solution_type="ElectroDCConduction",
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
            port=port,
        )
        identity_b = create_project(
            app_b,
            "B",
            cylinders=args.cylinders,
            mesh_mm=args.mesh_mm,
            passes=args.passes,
        )
        same_desktop = (
            int(app_a.desktop_class.odesktop.GetProcessID()) == desktop_pid
            and int(app_b.desktop_class.odesktop.GetProcessID()) == desktop_pid
        )
        project_list = sorted(str(value) for value in desktop.project_list)
        two_projects = {identity_a["project"], identity_b["project"]}.issubset(
            set(project_list)
        )
        rec.event(
            "projects_created",
            same_desktop=same_desktop,
            two_projects=two_projects,
            desktop_pid=desktop_pid,
            project_list=project_list,
            identity_a=identity_a,
            identity_b=identity_b,
        )
        pre_a_pids = {item["pid"] for item in proc_snapshot(desktop_pid)}

        dispatch_a_epoch = time.time()
        result_a = app_a.analyze_setup(
            "Setup1", cores=2, tasks=1, use_auto_settings=False, blocking=False
        )
        rec.event("dispatch_A", result=bool(result_a), epoch_dispatch=dispatch_a_epoch)
        phase_a = wait_for_phase(
            rec,
            "02_A_running",
            desktop_pid,
            args.lmutil,
            args.license_server,
            expected_solver_checkouts=1,
            timeout_s=args.phase_timeout,
        )
        roots_a = [
            item
            for item in phase_a.get("roots", [])
            if item["pid"] not in pre_a_pids
        ]
        a_mapping_unique = (
            phase_a.get("local_solver_checkouts") == 1 and len(roots_a) == 1
        )
        rec.event(
            "A_mapping",
            unique=a_mapping_unique,
            roots=roots_a,
            local_solver_checkouts=phase_a.get("local_solver_checkouts"),
        )

        pre_b_pids = {item["pid"] for item in proc_snapshot(desktop_pid)}
        dispatch_b_epoch = time.time()
        result_b = app_b.analyze_setup(
            "Setup1", cores=2, tasks=1, use_auto_settings=False, blocking=False
        )
        rec.event("dispatch_B", result=bool(result_b), epoch_dispatch=dispatch_b_epoch)
        phase_b = wait_for_phase(
            rec,
            "03_AB_overlap",
            desktop_pid,
            args.lmutil,
            args.license_server,
            expected_solver_checkouts=2,
            timeout_s=args.phase_timeout,
        )
        roots_b_new = [
            item
            for item in phase_b.get("roots", [])
            if item["pid"] not in pre_b_pids
        ]
        overlap_license = phase_b.get("license", {})
        overlap_desktop_count = overlap_license.get("electronics_desktop", {}).get(
            "local_checkout_count"
        )
        overlap_solver_count = overlap_license.get("elec_solve_maxwell", {}).get(
            "local_checkout_count"
        )
        overlap_proven = (
            overlap_desktop_count == 1
            and overlap_solver_count is not None
            and overlap_solver_count >= 2
            and len(phase_b.get("roots", [])) >= 2
        )
        b_mapping_unique = len(roots_b_new) == 1
        grpc_overlap = grpc_probe(rec, "during_overlap", port, desktop_pid)
        rec.event(
            "overlap_verdict",
            proven=overlap_proven,
            desktop_checkout_count=overlap_desktop_count,
            solver_checkout_count=overlap_solver_count,
            roots=phase_b.get("roots", []),
            roots_b_new=roots_b_new,
        )

        local_failure_observed = False
        try:
            app_a.odesign.Analyze("INTENTIONALLY_MISSING_SETUP", False)
        except Exception as exc:
            local_failure_observed = True
            rec.event(
                "project_local_failure",
                observed=True,
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
        else:
            rec.event("project_local_failure", observed=False)
        grpc_after_local_failure = grpc_probe(
            rec, "after_project_local_failure", port, desktop_pid
        )

        current_snapshot = proc_snapshot(desktop_pid)
        current_license = capture_lmstat(
            rec, "03_pre_fault_mapping", args.lmutil, args.license_server
        )
        current_roots = licensed_solver_processes(
            current_snapshot, current_license, desktop_pid
        )
        root_by_pid = {item["pid"]: item for item in current_roots}
        a_live = [root_by_pid[item["pid"]] for item in roots_a if item["pid"] in root_by_pid]
        b_live = [
            root_by_pid[item["pid"]]
            for item in roots_b_new
            if item["pid"] in root_by_pid
        ]
        exact_sigterm_mapping = (
            a_mapping_unique
            and b_mapping_unique
            and overlap_proven
            and len(a_live) == 1
            and len(b_live) == 1
            and a_live[0]["pid"] != b_live[0]["pid"]
            and a_live[0]["pid"] in current_license["elec_solve_maxwell"]["local_pids"]
            and b_live[0]["pid"] in current_license["elec_solve_maxwell"]["local_pids"]
        )
        sigterm_sent = False
        target_pid = a_live[0]["pid"] if exact_sigterm_mapping else None
        if exact_sigterm_mapping:
            os.kill(target_pid, signal.SIGTERM)
            sigterm_sent = True
            rec.event(
                "exact_solver_sigterm",
                sent=True,
                target_pid=target_pid,
                mapping={
                    "A": a_live[0],
                    "B": b_live[0],
                    "A_phase_local_checkout_count": phase_a.get(
                        "local_solver_checkouts"
                    ),
                    "AB_phase_local_checkout_count": overlap_solver_count,
                },
            )
        else:
            rec.event(
                "exact_solver_sigterm",
                sent=False,
                reason="ambiguous_or_incomplete_two_phase_solver_mapping",
                roots_a=roots_a,
                roots_b_new=roots_b_new,
                current_roots=current_roots,
            )

        time.sleep(5)
        after_fault_license = capture_lmstat(
            rec, "04_after_fault", args.lmutil, args.license_server
        )
        after_fault_snapshot, after_fault_heuristic = process_phase(
            rec, "04_after_fault", desktop_pid
        )
        after_fault_roots = licensed_solver_processes(
            after_fault_snapshot, after_fault_license, desktop_pid
        )
        grpc_after_sigterm = grpc_probe(rec, "after_sigterm", port, desktop_pid)
        b_pid_survived_fault = bool(
            b_live and any(item["pid"] == b_live[0]["pid"] for item in after_fault_roots)
        )
        a_pid_left_fault = bool(
            target_pid is not None
            and all(item["pid"] != target_pid for item in after_fault_roots)
        )
        rec.event(
            "post_fault_isolation",
            b_pid_survived=b_pid_survived_fault,
            a_pid_left=a_pid_left_fault,
            roots=after_fault_roots,
            heuristic_roots=after_fault_heuristic,
            license=after_fault_license,
        )

        deadline = time.time() + args.completion_timeout
        desktop_running_cleared = False
        while time.time() < deadline:
            running = bool(desktop.are_there_simulations_running)
            _, roots = process_phase(rec, "05_completion_poll", desktop_pid)
            rec.event("completion_poll", running=running, roots=roots)
            if not running:
                desktop_running_cleared = True
                break
            time.sleep(10)
        else:
            rec.event("completion_timeout", timeout_s=args.completion_timeout)

        grpc_after_completion = grpc_probe(rec, "after_completion", port, desktop_pid)
        post_completion_license = capture_lmstat(
            rec, "05_after_completion", args.lmutil, args.license_server
        )
        post_completion_solver_pids = post_completion_license[
            "elec_solve_maxwell"
        ]["local_pids"]
        killed_solver_checkout_released = bool(
            target_pid is not None and target_pid not in post_completion_solver_pids
        )
        identity_a_after = safe_project_identity(app_a)
        identity_b_after = safe_project_identity(app_b)
        b_identity_intact = bool(
            identity_b_after.get("ok")
            and {
                key: identity_b_after[key]
                for key in ("project", "design", "objects", "setups")
            }
            == identity_b
        )
        convergence_a = export_convergence(app_a, artifact / "A_convergence.csv")
        convergence_b = export_convergence(app_b, artifact / "B_convergence.csv")
        app_b.save_project()
        b_project_valid = project_b_path.exists() and project_b_path.stat().st_size > 0
        b_project_sha = sha256(project_b_path) if b_project_valid else None
        final_license = capture_lmstat(
            rec, "06_before_desktop_close", args.lmutil, args.license_server
        )

        checks = {
            "same_desktop": same_desktop,
            "two_projects": two_projects,
            "desktop_checkout_exactly_one_during_overlap": overlap_desktop_count == 1,
            "solver_checkouts_at_least_two_during_overlap": bool(
                overlap_solver_count is not None and overlap_solver_count >= 2
            ),
            "two_solver_processes_during_overlap": len(phase_b.get("roots", [])) >= 2,
            "grpc_during_overlap": bool(grpc_overlap.get("ok")),
            "project_local_failure_observed": local_failure_observed,
            "grpc_after_local_failure": bool(grpc_after_local_failure.get("ok")),
            "exact_sigterm_mapping": exact_sigterm_mapping,
            "sigterm_sent": sigterm_sent,
            "A_solver_left_after_sigterm": a_pid_left_fault,
            "B_solver_survived_after_sigterm": b_pid_survived_fault,
            "grpc_after_sigterm": bool(grpc_after_sigterm.get("ok")),
            "B_identity_intact": b_identity_intact,
            "B_convergence_exported": bool(convergence_b.get("exists")),
            "B_project_valid": b_project_valid,
            "grpc_after_completion": bool(grpc_after_completion.get("ok")),
            "desktop_running_cleared_without_global_stop": desktop_running_cleared,
            "killed_solver_checkout_released": killed_solver_checkout_released,
        }
        verdict.update(
            {
                "activation_allowed": all(checks.values()),
                "checks": checks,
                "desktop_pid": desktop_pid,
                "grpc_port": port,
                "baseline_license": baseline_license,
                "overlap_license": overlap_license,
                "final_license_before_close": final_license,
                "post_completion_license": post_completion_license,
                "identity_A_before": identity_a,
                "identity_A_after": identity_a_after,
                "identity_B_before": identity_b,
                "identity_B_after": identity_b_after,
                "B_project_sha256": b_project_sha,
                "convergence_A": convergence_a,
                "convergence_B": convergence_b,
                "fault_target_pid": target_pid,
            }
        )
        rec.write_json("verdict.json", verdict)
        rec.event("pilot_verdict", verdict=verdict)
        return 0 if verdict["activation_allowed"] else 3
    except Exception as exc:
        verdict.update(
            {
                "activation_allowed": False,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        rec.write_json("verdict.json", verdict)
        rec.event("pilot_fatal", verdict=verdict)
        return 2
    finally:
        # No Desktop-global stop while a healthy sibling may still be solving.
        # The Slurm job cgroup is the final containment if release fails.
        try:
            if desktop is not None and not desktop.are_there_simulations_running:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
                rec.event("desktop_released", clean=True)
            elif desktop is not None:
                rec.event(
                    "desktop_released",
                    clean=False,
                    reason="simulation_still_running_no_global_stop_called",
                )
        except Exception as exc:
            rec.event(
                "desktop_release_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )


if __name__ == "__main__":
    raise SystemExit(main())
