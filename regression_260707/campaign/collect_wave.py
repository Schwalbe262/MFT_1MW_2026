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
import tempfile
import time
from datetime import datetime

import pandas as pd
import requests
from filelock import FileLock

SCHEDULER = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "..", "data", "dataset")
LOCAL_RESULTS_CSV = os.path.join(HERE, "..", "..", "simulation_results_260706.csv")

FETCH_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.5
RETRY_MAX_SECONDS = 30.0


class FetchError(RuntimeError):
    """A scheduler response could not be fetched reliably."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _retry_delay(response, attempt):
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(RETRY_MAX_SECONDS, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2 ** attempt))


def _get_response(path, *, params=None, timeout=30, attempts=FETCH_ATTEMPTS):
    """GET with bounded retry for 429, 5xx, and connection failures."""
    url = path if path.startswith("http") else f"{SCHEDULER}{path}"
    last_error = None
    for attempt in range(attempts):
        response = None
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = FetchError(f"GET {path} failed: {exc}")
        else:
            status_code = int(response.status_code)
            if status_code != 429 and status_code < 500:
                if status_code >= 400:
                    raise FetchError(
                        f"GET {path} returned HTTP {status_code}", status_code
                    )
                return response
            last_error = FetchError(
                f"GET {path} returned HTTP {status_code}", status_code
            )
        if attempt + 1 < attempts:
            time.sleep(_retry_delay(response, attempt))
    raise last_error or FetchError(f"GET {path} failed")


def _get_json(path, *, params=None, timeout=30):
    response = _get_response(path, params=params, timeout=timeout)
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        raise FetchError(f"GET {path} returned invalid JSON: {exc}") from exc


def fetch_stdout(task_id, timeout=30):
    return _get_response(f"/api/tasks/{task_id}/stdout", timeout=timeout).text


def list_tasks(prefix):
    # 신 스케줄러(limit/name_prefix 지원) 우선
    t = _get_json("/api/tasks",
                  params={"limit": 10000, "name_prefix": prefix}, timeout=30)
    tasks = t if isinstance(t, list) else t.get("tasks", [])
    matched = [x for x in tasks if str(x.get("name", "")).startswith(prefix)]
    if len(tasks) > 250:
        return matched

    # 구 스케줄러(200개 페이지): ID 연속 스캔으로 누락 보완
    seen = {x["id"]: x for x in matched}
    if not seen:
        return []
    def probe(tid):
        try:
            x = _get_json(f"/api/tasks/{tid}", timeout=15)
        except FetchError as exc:
            return False if exc.status_code == 404 else None
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
            elif r is False:
                misses += 1
            else:
                break
    # 범위 내 구멍 채우기 (페이지에 안 담긴 중간 ID) - 이미 판정된 터미널 ID는 생략
    try:
        c = _load_cache()
        judged = set(c.get("nodata", [])) | set(c.get("harvested", []))
    except Exception:
        judged = set()
    lo, hi = min(seen), max(seen)
    for tid in range(lo, hi + 1):
        if tid not in seen and tid not in judged:
            r = probe(tid)
            if r:
                seen[tid] = r
    return list(seen.values())


# 터미널 태스크의 회수 결과 캐시: 재수집 시 재조회 생략 (회수 시간 수분 -> 초)
CACHE_PATH = os.path.join(DATASET_DIR, "collect_cache.json")


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"nodata": [], "harvested": []}


def _save_cache(c):
    os.makedirs(DATASET_DIR, exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(c, f)
    os.replace(tmp, CACHE_PATH)


def _commit_pending_cache(pending_harvested, pending_nodata):
    """Merge pending terminal IDs into the latest on-disk cache."""
    if not pending_harvested and not pending_nodata:
        return
    cache = _load_cache()
    harvested = cache.setdefault("harvested", [])
    nodata = cache.setdefault("nodata", [])
    harvested.extend(tid for tid in pending_harvested if tid not in harvested)
    nodata.extend(tid for tid in pending_nodata if tid not in nodata)
    _save_cache(cache)


def fetch_result_rows(task_id, out=None):
    if out is None:
        out = fetch_stdout(task_id)
    if "===RESULT_CSV===" not in out:
        return None
    block = out.split("===RESULT_CSV===")[-1].split("===FAILED_CSV===")[0].strip()
    if not block or "," not in block:
        return None
    try:
        return pd.read_csv(io.StringIO(block))
    except Exception:
        return None


# 프로브 전치 버그 수정 커밋 (2026-07-07). 이전 코드로 돌린 행은 _side/core_center
# 프로브가 전치된 시트에서 평가된 값이라 무효 -> NaN 처리 (T_max_*, leeward는 유효)
PROBE_FIX_HASHES_OK = None  # lazy: 수정 커밋 이후 해시 집합



def fetch_streamed_rows(task_id, out=None):
    """stdout의 RESULT_JSON 라인들 -> DataFrame (새 공유폴더 방식의 기본 회수 경로)"""
    if out is None:
        out = fetch_stdout(task_id)
    rows = []
    for l in out.splitlines():
        if l.startswith("RESULT_JSON "):
            try:
                rows.append(json.loads(l[12:]))
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else None


def sanitize_bad_probes(df):
    import subprocess
    global PROBE_FIX_HASHES_OK
    if "git_hash" not in df.columns:
        return df, 0
    if PROBE_FIX_HASHES_OK is None:
        try:
            out = subprocess.run(["git", "log", "--format=%h", "8f00000..HEAD"],
                                 capture_output=True, text=True, cwd=os.path.join(HERE, "..", ".."))
            # 수정 커밋부터 HEAD까지의 해시 (실패 시 빈 집합 -> 전부 유효 취급 안 함)
            out2 = subprocess.run(["git", "log", "--format=%h"],
                                  capture_output=True, text=True, cwd=os.path.join(HERE, "..", ".."))
            all_h = out2.stdout.split()
            # 수정 커밋: 'Fix transposed probe sheets' 메시지 기준
            log = subprocess.run(["git", "log", "--format=%h %s"],
                                 capture_output=True, text=True, cwd=os.path.join(HERE, "..", "..")).stdout
            fix_h = next((l.split()[0] for l in log.splitlines() if "transposed probe" in l), None)
            PROBE_FIX_HASHES_OK = set(all_h[:all_h.index(fix_h) + 1]) if fix_h in all_h else set()
        except Exception:
            PROBE_FIX_HASHES_OK = set()
    bad_cols = [c for c in df.columns
                if c.startswith("Tprobe_") and ("_side_" in c or "core_center" in c)]
    mask = ~df["git_hash"].isin(PROBE_FIX_HASHES_OK)
    n = int(mask.sum())
    if n and bad_cols:
        df.loc[mask, bad_cols] = float("nan")
    return df, n


def convergence_filter(df, max_err=1.5):
    keep = pd.Series(True, index=df.index)
    for col in ["conv_error_pct_matrix", "conv_error_pct_loss"]:
        if col in df.columns:
            keep &= (df[col].isna()) | (df[col] <= max_err)
    return df[keep], int((~keep).sum())


def _row_hashes(df, columns):
    normalized = df.loc[:, columns].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False)


def _fallback_columns(frame, other=None):
    """Columns used for legacy identity, excluding collector-added metadata."""
    excluded = {"task_id", "task_name"}
    if other is None:
        return [c for c in frame.columns if c not in excluded]
    return [c for c in frame.columns if c in other.columns and c not in excluded]


def _complete_key_mask(frame, dedup_keys):
    if not dedup_keys or not all(c in frame.columns for c in dedup_keys):
        return pd.Series(False, index=frame.index)
    return frame.loc[:, dedup_keys].notna().all(axis=1)


def deduplicate_rows(frame, dedup_keys):
    """Dedup complete keys by key and incomplete keys by stable row content."""
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame

    complete = _complete_key_mask(frame, dedup_keys)
    keep = pd.Series(True, index=frame.index)
    if complete.any():
        keyed_duplicates = frame.loc[complete, dedup_keys].duplicated(keep="last")
        keep.loc[complete] = ~keyed_duplicates.to_numpy()

    incomplete = ~complete
    if incomplete.any():
        fallback_columns = _fallback_columns(frame)
        if fallback_columns:
            hashes = _row_hashes(frame.loc[incomplete], fallback_columns)
            keep.loc[incomplete] = ~hashes.duplicated(keep="last").to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def select_new_unique_rows(incoming, old, dedup_keys):
    """Select incoming sample identities that are absent from the master."""
    incoming = deduplicate_rows(incoming, dedup_keys)
    if old is None or old.empty:
        return incoming.reset_index(drop=True)

    new_mask = pd.Series(True, index=incoming.index)
    can_compare_keys = bool(dedup_keys) and all(c in old.columns for c in dedup_keys)
    incoming_complete = _complete_key_mask(incoming, dedup_keys)
    if can_compare_keys and incoming_complete.any():
        old_complete = _complete_key_mask(old, dedup_keys)
        old_key_hashes = set(
            _row_hashes(old.loc[old_complete], dedup_keys).tolist())
        incoming_key_hashes = _row_hashes(
            incoming.loc[incoming_complete], dedup_keys)
        new_mask.loc[incoming_complete] = ~incoming_key_hashes.isin(
            old_key_hashes).to_numpy()

    fallback_mask = ~incoming_complete if can_compare_keys else pd.Series(
        True, index=incoming.index)
    if fallback_mask.any():
        fallback_columns = _fallback_columns(incoming, old)
        if fallback_columns:
            old_hashes = set(_row_hashes(old, fallback_columns).tolist())
            incoming_hashes = _row_hashes(
                incoming.loc[fallback_mask], fallback_columns)
            new_mask.loc[fallback_mask] = ~incoming_hashes.isin(
                old_hashes).to_numpy()
    return incoming.loc[new_mask].reset_index(drop=True)


def _stage_path(target):
    fd, path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=os.path.dirname(target))
    os.close(fd)
    return path


def _stage_parquet(frame, target):
    staged = _stage_path(target)
    try:
        frame.to_parquet(staged, index=False)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _stage_manifest(manifest, target):
    staged = _stage_path(target)
    try:
        with open(staged, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return staged
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise


def _replace_staged(staged_targets):
    """Install fully serialized files; the master is deliberately replaced last."""
    try:
        for staged, target in staged_targets:
            os.replace(staged, target)
    finally:
        for staged, _ in staged_targets:
            if os.path.exists(staged):
                os.remove(staged)


def merge_dataset(merged, dedup_keys, prefix, pending_harvested=(), pending_nodata=()):
    """Merge rows and terminal cache state under one inter-process lock."""
    master_path = os.path.join(DATASET_DIR, "train.parquet")
    manifest_path = os.path.join(DATASET_DIR, "manifest.json")
    lock_path = master_path + ".lock"
    with FileLock(lock_path):
        old = pd.read_parquet(master_path) if os.path.isfile(master_path) else None
        new_rows = select_new_unique_rows(merged, old, dedup_keys)
        new_unique_rows = len(new_rows)
        if not new_unique_rows:
            _commit_pending_cache(pending_harvested, pending_nodata)
            return new_unique_rows, 0 if old is None else len(old), master_path

        stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        part_path = os.path.join(
            DATASET_DIR, f"collected_{prefix.replace('/', '_')}_{stamp}.parquet")
        allf = new_rows if old is None else pd.concat(
            [old, new_rows], ignore_index=True, sort=False)
        manifest = {
            "updated": stamp, "total_rows": len(allf), "new_rows": new_unique_rows,
            "new_unique_rows": new_unique_rows,
            "git_hashes": sorted(allf["git_hash"].dropna().unique().tolist())
            if "git_hash" in allf.columns else [],
            "prefix": prefix,
        }

        staged = []
        try:
            staged.append((_stage_parquet(new_rows, part_path), part_path))
            staged.append((_stage_manifest(manifest, manifest_path), manifest_path))
            staged.append((_stage_parquet(allf, master_path), master_path))
            _replace_staged(staged)
        except Exception:
            for temp_path, _ in staged:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise

        _commit_pending_cache(pending_harvested, pending_nodata)
        return new_unique_rows, len(allf), master_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mft-camp")
    ap.add_argument("--max-conv-err", type=float, default=1.5)
    args = ap.parse_args(argv)

    os.makedirs(DATASET_DIR, exist_ok=True)

    tasks = list_tasks(args.prefix)
    done = [t for t in tasks if t.get("status") == "completed"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    print(f"tasks: {len(tasks)} (completed {len(done)}, failed {len(failed)}, "
          f"running/queued {len(tasks) - len(done) - len(failed)})")

    # 실패 태스크도 stdout에 결과가 있으면 회수 (pyaedt teardown 크래시가
    # 성공 샘플을 실패로 둔갑시키는 케이스 실측됨 - 데이터는 유효)
    # + 실행 중 태스크의 RESULT_JSON 라인도 스트리밍 회수 (샘플 단위 실시간성)
    import json as _json
    running = [t for t in tasks if t.get("status") in ("running", "attaching")]
    cache = _load_cache()
    skip = set(cache.get("nodata", [])) | set(cache.get("harvested", []))
    frames = []
    n_salvaged = 0
    n_streamed = 0
    n_skipped = 0
    n_fetch_errors = 0
    pending_harvested = []
    pending_nodata = []
    for t in done + failed:
        if t["id"] in skip:
            n_skipped += 1
            continue
        try:
            out = fetch_stdout(t["id"])
        except FetchError as exc:
            n_fetch_errors += 1
            print(f"fetch error task {t['id']}: {exc}")
            continue
        df = fetch_result_rows(t["id"], out=out)
        if df is None or not len(df):
            df = fetch_streamed_rows(t["id"], out=out)
        if df is not None and len(df):
            df["task_id"] = t["id"]
            df["task_name"] = t.get("name", "")
            frames.append(df)
            pending_harvested.append(t["id"])
            if t.get("status") == "failed":
                n_salvaged += 1
        else:
            # Only a successful stdout fetch proves that a task has no data.
            pending_nodata.append(t["id"])
    if n_skipped:
        print(f"cache skip: {n_skipped} terminal tasks")
    for t in running:
        try:
            out = fetch_stdout(t["id"], timeout=20)
        except FetchError as exc:
            n_fetch_errors += 1
            print(f"fetch error task {t['id']}: {exc}")
            continue
        rows = []
        for line in out.splitlines():
            if line.startswith("RESULT_JSON "):
                try:
                    rows.append(_json.loads(line[len("RESULT_JSON "):]))
                except Exception:
                    pass
        if rows:
            df = pd.DataFrame(rows)
            df["task_id"] = t["id"]
            df["task_name"] = t.get("name", "")
            frames.append(df)
            n_streamed += len(rows)
    if n_salvaged:
        print(f"salvaged from failed tasks: {n_salvaged}")
    if n_streamed:
        print(f"streamed from running tasks: {n_streamed} rows")
    if n_fetch_errors:
        print(f"fetch_errors: {n_fetch_errors} (not cached as nodata)")

    # 로컬 생산 라인: 로컬 CSV도 병합 (클러스터 대비 샘플당 15-25x 빠름)
    if os.path.isfile(LOCAL_RESULTS_CSV):
        try:
            ldf = pd.read_csv(LOCAL_RESULTS_CSV, on_bad_lines="skip")
            ldf["task_name"] = "local"
            frames.append(ldf)
            print(f"local rows: {len(ldf)}")
        except Exception as e:
            print(f"local csv read failed: {e}")

    if not frames:
        master_path = os.path.join(DATASET_DIR, "train.parquet")
        with FileLock(master_path + ".lock"):
            _commit_pending_cache(pending_harvested, pending_nodata)
        print("new_unique_rows: 0")
        print("no result rows collected")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # 중복 제거 (재회수/재시도 대비)
    # Both fields are required for the primary identity. A missing column sends
    # every row through the content-hash legacy fallback.
    dedup_keys = ["project_name", "saved_at"]
    before = len(merged)
    merged = deduplicate_rows(merged, dedup_keys)
    print(f"dedup: {before} -> {len(merged)}")

    merged, n_bad_probe = sanitize_bad_probes(merged)
    if n_bad_probe:
        print(f"probe-fix 이전 행 {n_bad_probe}개: side/core_center 프로브 컬럼 NaN 처리 (T_max/leeward는 유지)")
    merged, n_filtered = convergence_filter(merged, args.max_conv_err)
    print(f"convergence filter (<= {args.max_conv_err}%): -{n_filtered} rows")

    new_unique_rows, total_rows, master_path = merge_dataset(
        merged, dedup_keys, args.prefix,
        pending_harvested=pending_harvested, pending_nodata=pending_nodata)
    print(f"new_unique_rows: {new_unique_rows}")
    if not new_unique_rows:
        print(f"dataset unchanged: {total_rows} rows -> {master_path}")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}
    print(f"dataset: {total_rows} rows total -> {master_path}")
    return {"new_unique_rows": new_unique_rows, "fetch_errors": n_fetch_errors}


if __name__ == "__main__":
    main()
