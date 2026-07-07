"""
게이트1 자동 판정기: 대칭 vs 풀 태스크 쌍의 결과를 대조표로 정리.

사용:
  python gate1_report.py --pairs mft-gate1-modelB-sym-v3:mft-gate1-modelB-full-v3 \
                                 mft-gate1-drawing-sym-v3:mft-gate1-drawing-full-v3
"""
import argparse
import io
import sys

import numpy as np
import pandas as pd
import requests

SCHEDULER = "http://127.0.0.1:8000"

EM_KEYS = ["P_core_total", "B_mean_core", "B_max_core", "P_Tx_main_group",
           "P_Rx_main_group", "P_Rx_side_total", "P_winding_total",
           "P_core_plate_total", "P_wcp_total", "Llt", "k"]
T_KEYS = ["Tprobe_Tx_leeward_max", "Tprobe_Tx_side_max",
          "Tprobe_Rx_main_leeward_max", "Tprobe_Rx_main_side_max",
          "Tprobe_Rx_side_leeward_max", "Tprobe_Rx_side_side_max",
          "Tprobe_core_center_max", "T_max_Tx", "T_max_Rx_main",
          "T_max_Rx_side", "T_max_core"]
TIME_KEYS = ["time_matrix", "time_loss", "time_thermal", "time"]


def fetch_row(task_name):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_wave import list_tasks
    # ID-스캔 지원 목록 조회 (구식 스케줄러의 200개 페이지 제한 대응)
    prefix = task_name.rsplit("-", 2)[0]
    cand = [x for x in list_tasks(prefix) if x.get("name") == task_name]
    if not cand:
        return None, f"task not found: {task_name}"
    task = sorted(cand, key=lambda x: x["id"])[-1]
    if task.get("status") != "completed":
        return None, f"{task_name}: status={task.get('status')}"
    out = requests.get(f"{SCHEDULER}/api/tasks/{task['id']}/stdout", timeout=30).text
    if "===RESULT_CSV===" not in out:
        return None, f"{task_name}: no RESULT_CSV"
    block = out.split("===RESULT_CSV===")[-1].split("===FAILED_CSV===")[0].strip()
    df = pd.read_csv(io.StringIO(block))
    return df.iloc[-1], None


def compare(sym_row, full_row, label):
    print(f"\n===== 게이트1 대조: {label} =====")
    print(f"{'항목':32s} {'대칭(1/8, 보정)':>16s} {'풀(참조)':>16s} {'편차':>9s}")
    worst_em = 0.0
    for k in EM_KEYS:
        s, f = sym_row.get(k), full_row.get(k)
        if pd.isna(s) or pd.isna(f):
            continue
        s, f = float(s), float(f)
        if k == "Llt":
            # 매트릭스 기준이 다를 때만 환산 (대칭 매트릭스 = 실물의 1/2)
            if int(sym_row.get("full_model", 0) or 0) == 0:
                s *= 2.0
            if int(full_row.get("full_model", 0) or 0) == 0:
                f *= 2.0
        dev = (s / f - 1) * 100 if f else np.nan
        flag = "" if abs(dev) <= 10 else "  <-- 주의"
        if k not in ("B_max_core",):
            worst_em = max(worst_em, abs(dev))
        print(f"{k:32s} {s:16.4f} {f:16.4f} {dev:+8.1f}%{flag}")

    worst_T = 0.0
    any_T = False
    for k in T_KEYS:
        s, f = sym_row.get(k), full_row.get(k)
        if pd.isna(s) or pd.isna(f):
            continue
        any_T = True
        dT = float(s) - float(f)
        worst_T = max(worst_T, abs(dT))
        flag = "" if abs(dT) <= 5 else "  <-- 주의"
        print(f"{k:32s} {float(s):16.2f} {float(f):16.2f} {dT:+8.2f}C{flag}")

    for k in TIME_KEYS:
        s, f = sym_row.get(k), full_row.get(k)
        if pd.notna(s) and pd.notna(f):
            print(f"{k:32s} {float(s):16.1f} {float(f):16.1f}   x{float(f)/max(float(s),1e-9):.1f}")

    verdict_em = worst_em <= 10
    verdict_T = (not any_T) or worst_T <= 5
    print(f"\n판정: EM {'통과' if verdict_em else '미달'} (최대 |편차| {worst_em:.1f}% / 기준 10%) | "
          f"열 {'통과' if verdict_T else '미달'} (최대 |dT| {worst_T:.1f}C / 기준 5C)"
          + ("" if any_T else " [열 데이터 없음]"))
    return verdict_em and verdict_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="sym_task:full_task ...")
    args = ap.parse_args()

    all_pass = True
    for pair in args.pairs:
        sname, fname = pair.split(":")
        srow, e1 = fetch_row(sname)
        frow, e2 = fetch_row(fname)
        if e1 or e2:
            print(f"[미완] {pair}: {e1 or ''} {e2 or ''}")
            all_pass = False
            continue
        all_pass &= compare(srow, frow, pair)
    print(f"\n== 게이트1 종합: {'통과' if all_pass else '미완/미달'} ==")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
