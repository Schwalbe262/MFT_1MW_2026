"""
AL 검증용 스케줄러 클라이언트: 후보 params JSON -> 태스크 제출 -> RESULT_JSON 회수.
"""
import json
import shlex
import time

import requests

SCHEDULER = "http://127.0.0.1:8000"
BASE = ("source /etc/profile.d/lmod.sh 2>/dev/null || true; "
        "module load ansys-electronics/v252 || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64; ")


def submit_verification(name, workdir, params: dict, profile: dict, mem_mb=32768, cpus=4):
    """후보 파라미터를 인라인 JSON으로 실어 fixed 모드 검증 태스크 제출. 반환: task_id 또는 None"""
    merged = dict(params)
    merged.update(profile.get("param_overrides", {}))
    pjson = json.dumps(merged, separators=(",", ":"))
    extra = profile.get("cli_flags", "")
    lib_clone = ("([ -d pyaedt_library/src ] || { git clone -q --depth 1 "
                 "https://github.com/Schwalbe262/pyaedt_library.git pyaedt_library.tmp.$$ "
                 "&& { mv -T pyaedt_library.tmp.$$ pyaedt_library 2>/dev/null || rm -rf pyaedt_library.tmp.$$; }; }) && "
                 "[ -d pyaedt_library/src ] && ")
    cmd = (BASE + lib_clone +
           f"([ -d {workdir} ] || git clone -q --depth 1 https://github.com/Schwalbe262/MFT_1MW_2026.git {workdir}) && "
           f"cd {workdir} && git pull -q && "
           f"printf '%s' {shlex.quote(pjson)} > cand.json && "
           f"python run_simulation_260706.py --fixed {extra} --params cand.json")
    r = requests.post(f"{SCHEDULER}/tasks", data={
        "name": name, "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__",
        "command": cmd, "required_capability": "conda:pyaedt2026v1", "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty", "cpus": cpus, "memory_mb": mem_mb, "gpus": 0,
    }, allow_redirects=False, timeout=20)
    if r.status_code not in (200, 201, 303):
        return None
    # 방금 제출한 태스크 ID 확인 (이름으로 역조회)
    time.sleep(1)
    try:
        t = requests.get(f"{SCHEDULER}/api/tasks", params={"limit": 50}, timeout=15).json()
        for x in (t if isinstance(t, list) else t.get("tasks", [])):
            if x.get("name") == name:
                return x["id"]
    except Exception:
        pass
    return None


def get_status(task_id):
    try:
        return requests.get(f"{SCHEDULER}/api/tasks/{task_id}", timeout=15).json().get("status")
    except Exception:
        return None


def cancel(task_id):
    try:
        requests.post(f"{SCHEDULER}/tasks/{task_id}/cancel", timeout=10)
    except Exception:
        pass


def fetch_result_json(task_id):
    """stdout의 RESULT_JSON 라인 파싱 -> dict 또는 None"""
    try:
        out = requests.get(f"{SCHEDULER}/api/tasks/{task_id}/stdout", timeout=30).text
    except Exception:
        return None
    for line in reversed(out.splitlines()):
        if line.startswith("RESULT_JSON "):
            try:
                return json.loads(line[len("RESULT_JSON "):])
            except Exception:
                return None
    return None


def wait_all(task_ids, poll_s=120, timeout_s=6 * 3600, on_progress=None):
    """태스크 집합 완료 대기. 반환: {task_id: status}"""
    t0 = time.time()
    status = {tid: None for tid in task_ids}
    while time.time() - t0 < timeout_s:
        pending = [tid for tid, s in status.items()
                   if s not in ("completed", "failed", "cancelled")]
        if not pending:
            break
        for tid in pending:
            s = get_status(tid)
            if s:
                status[tid] = s
        if on_progress:
            on_progress(status)
        if all(s in ("completed", "failed", "cancelled") for s in status.values()):
            break
        time.sleep(poll_s)
    return status
