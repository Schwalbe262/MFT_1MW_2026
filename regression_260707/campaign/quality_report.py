"""
캠페인 품질 리포트: golden 드리프트 + 실패 분류 + 커버리지 요약.

사용: python quality_report.py [--golden-prefix mft-camp] [--dataset ../data/dataset/train.parquet]
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
import requests

SCHEDULER = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))

GOLDEN_KEYS = ["Llt", "k", "P_core_total", "P_winding_total", "B_mean_core",
               "Tprobe_Tx_leeward_max", "Tprobe_core_center_max"]
DRIFT_TOL = {"Llt": 1.0, "k": 0.2, "P_core_total": 2.0, "P_winding_total": 2.0,
             "B_mean_core": 1.0}  # %
DRIFT_TOL_T = 2.0  # C


def golden_drift():
    path = os.path.join(HERE, "..", "..", "golden_history_260706.csv")
    if not os.path.isfile(path):
        print("[golden] 이력 파일 없음 (클러스터 golden은 stdout 회수 필요)")
        return True
    df = pd.read_csv(path)
    if len(df) < 2:
        print(f"[golden] 이력 {len(df)}건 - 비교 불가")
        return True
    base, last = df.iloc[0], df.iloc[-1]
    ok = True
    print(f"[golden] {len(df)}건 (기준 {base.get('git_hash')} -> 최신 {last.get('git_hash')})")
    for k in GOLDEN_KEYS:
        if k not in df.columns or pd.isna(base.get(k)) or pd.isna(last.get(k)):
            continue
        b, l = float(base[k]), float(last[k])
        if k.startswith("Tprobe"):
            d = abs(l - b)
            bad = d > DRIFT_TOL_T
            print(f"  {k:28s} {b:10.3f} -> {l:10.3f}  d={d:.2f}C {'<-- DRIFT' if bad else ''}")
        else:
            d = abs(l / b - 1) * 100 if b else 0
            bad = d > DRIFT_TOL.get(k, 2.0)
            print(f"  {k:28s} {b:10.3f} -> {l:10.3f}  d={d:.2f}% {'<-- DRIFT' if bad else ''}")
        ok &= not bad
    return ok


def failure_taxonomy(prefix):
    import sys
    sys.path.insert(0, HERE)
    from collect_wave import list_tasks
    ts = list_tasks(prefix)
    failed = [t for t in ts if t.get("status") == "failed"]
    print(f"[failures] {prefix}: {len(failed)}/{len(ts)} failed ({100*len(failed)/max(len(ts),1):.1f}%)")
    buckets = {}
    for t in failed[:60]:
        try:
            err = requests.get(f"{SCHEDULER}/api/tasks/{t['id']}/stderr", timeout=20).text[-3000:]
        except Exception:
            continue
        key = "unknown"
        for pat, name in [("license", "license"), ("GrpcApiError", "grpc_api"),
                          ("MemoryError", "memory"), ("CANCELLED", "slurm_cancel"),
                          ("No such file", "missing_file"), ("Errno 28", "disk_full"),
                          ("validation", "validation"), ("is_solved", "solve_fail")]:
            if pat.lower() in err.lower():
                key = name
                break
        buckets[key] = buckets.get(key, 0) + 1
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v}")
    return len(failed) / max(len(ts), 1)


def coverage(dataset):
    if not os.path.isfile(dataset):
        print("[coverage] dataset 없음")
        return
    df = pd.read_parquet(dataset)
    print(f"[coverage] {len(df)} rows")
    if "Llt" in df.columns:
        llt_phys = df["Llt"] * np.where(df.get("full_model", 0).fillna(0) == 0, 2.0, 1.0)
        q = np.nanpercentile(llt_phys, [5, 25, 50, 75, 95])
        in_band = ((llt_phys >= 20) & (llt_phys <= 40)).mean() * 100
        print(f"  Llt_phys 분포 [uH]: p5={q[0]:.1f} p25={q[1]:.1f} p50={q[2]:.1f} "
              f"p75={q[3]:.1f} p95={q[4]:.1f} | 20-40uH 비율 {in_band:.1f}%")
    for c in ["Tprobe_Tx_leeward_max", "P_core_total"]:
        if c in df.columns:
            v = df[c].dropna()
            if len(v):
                print(f"  {c}: n={len(v)} p50={v.median():.1f} p95={v.quantile(0.95):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-prefix", default="mft-camp")
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data", "dataset", "train.parquet"))
    args = ap.parse_args()
    ok = golden_drift()
    rate = failure_taxonomy(args.golden_prefix)
    coverage(args.dataset)
    print(f"\n== 품질 종합: golden {'OK' if ok else 'DRIFT'} | 실패율 {rate*100:.1f}% "
          f"({'게이트2 통과' if rate < 0.05 else '기준(5%) 초과'}) ==")


if __name__ == "__main__":
    main()
