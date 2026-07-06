"""
캠페인 결과 회수·병합기.

완료된 mft-camp-* 태스크의 stdout에서 ===RESULT_CSV=== 블록을 파싱해
스키마-유니온으로 병합하고, 수렴 필터·중복 제거 후 dataset/train.parquet에 축적한다.

사용:
  python collect_wave.py --prefix mft-camp-w1          # 웨이브 1 회수
  python collect_wave.py --prefix mft-camp --all       # 전체 회수
"""
import argparse
import io
import json
import os
from datetime import datetime

import pandas as pd
import requests

SCHEDULER = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "data", "dataset")


def list_tasks(prefix):
    # 신 스케줄러(limit/name_prefix 지원) 우선
    matched = []
    try:
        t = requests.get(f"{SCHEDULER}/api/tasks",
                         params={"limit": 5000, "name_prefix": prefix}, timeout=30).json()
        tasks = t if isinstance(t, list) else t.get("tasks", [])
        matched = [x for x in tasks if str(x.get("name", "")).startswith(prefix)]
        if len(tasks) > 250:
            return matched
    except Exception:
        pass

    # 구 스케줄러(200개 페이지): ID 연속 스캔으로 누락 보완
    seen = {x["id"]: x for x in matched}
    if not seen:
        return []
    def probe(tid):
        try:
            x = requests.get(f"{SCHEDULER}/api/tasks/{tid}", timeout=10).json()
        except Exception:
            return None
        return x if str(x.get("name", "")).startswith(prefix) else False

    ids = sorted(seen)
    lo, hi = ids[0], ids[-1]
    # 경계 확장: 프리픽스 연속 구간 가정, 밖으로 miss 20회까지
    for direction in (-1, +1):
        cur = lo if direction < 0 else hi
        misses = 0
        while misses < 20 and cur > 0:
            cur += direction
            r = probe(cur)
            if r:
                seen[cur] = r
                misses = 0
            else:
                misses += 1
    # 범위 내 구멍 채우기 (페이지에 안 담긴 중간 ID)
    lo, hi = min(seen), max(seen)
    for tid in range(lo, hi + 1):
        if tid not in seen:
            r = probe(tid)
            if r:
                seen[tid] = r
    return list(seen.values())


def fetch_result_rows(task_id):
    try:
        out = requests.get(f"{SCHEDULER}/api/tasks/{task_id}/stdout", timeout=30).text
    except Exception:
        return None
    if "===RESULT_CSV===" not in out:
        return None
    block = out.split("===RESULT_CSV===")[-1].split("===FAILED_CSV===")[0].strip()
    if not block or "," not in block:
        return None
    try:
        return pd.read_csv(io.StringIO(block))
    except Exception:
        return None


def convergence_filter(df, max_err=1.5):
    keep = pd.Series(True, index=df.index)
    for col in ["conv_error_pct_matrix", "conv_error_pct_loss"]:
        if col in df.columns:
            keep &= (df[col].isna()) | (df[col] <= max_err)
    return df[keep], int((~keep).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mft-camp")
    ap.add_argument("--max-conv-err", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)

    tasks = list_tasks(args.prefix)
    done = [t for t in tasks if t.get("status") == "completed"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    print(f"tasks: {len(tasks)} (completed {len(done)}, failed {len(failed)}, "
          f"running/queued {len(tasks) - len(done) - len(failed)})")

    frames = []
    for t in done:
        df = fetch_result_rows(t["id"])
        if df is not None and len(df):
            df["task_id"] = t["id"]
            df["task_name"] = t.get("name", "")
            frames.append(df)

    if not frames:
        print("no result rows collected")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # 중복 제거 (재회수/재시도 대비)
    dedup_keys = [c for c in ["project_name", "saved_at"] if c in merged.columns]
    if dedup_keys:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
        print(f"dedup: {before} -> {len(merged)}")

    merged, n_filtered = convergence_filter(merged, args.max_conv_err)
    print(f"convergence filter (<= {args.max_conv_err}%): -{n_filtered} rows")

    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    part_path = os.path.join(DATASET_DIR, f"collected_{args.prefix.replace('/', '_')}_{stamp}.parquet")
    merged.to_parquet(part_path, index=False)

    # 마스터 병합 (스키마-유니온)
    master_path = os.path.join(DATASET_DIR, "train.parquet")
    if os.path.isfile(master_path):
        old = pd.read_parquet(master_path)
        allf = pd.concat([old, merged], ignore_index=True, sort=False)
        if dedup_keys:
            allf = allf.drop_duplicates(subset=dedup_keys, keep="last")
    else:
        allf = merged
    allf.to_parquet(master_path, index=False)

    manifest = {
        "updated": stamp, "total_rows": len(allf), "new_rows": len(merged),
        "git_hashes": sorted(allf["git_hash"].dropna().unique().tolist()) if "git_hash" in allf.columns else [],
        "prefix": args.prefix,
    }
    with open(os.path.join(DATASET_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"dataset: {len(allf)} rows total -> {master_path}")


if __name__ == "__main__":
    main()
