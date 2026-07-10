"""
캠페인 결과 회수·병합기.

완료된 mft-camp-* 태스크의 stdout에서 ===RESULT_CSV=== 블록을 파싱해
스키마-유니온으로 병합하고, 수렴 필터·중복 제거 후 dataset/train.parquet에 축적한다.

사용:
  python collect_wave.py --prefix mft-camp-w1          # 웨이브 1 회수
  python collect_wave.py --prefix mft-camp --all       # 전체 회수
"""
import argparse
import glob
import io
import json
import math
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
LOCAL_RESULTS_PARTS_DIR = os.path.join(HERE, "..", "..", "results_parts_260706")
FEEDER_STATE_PATH = os.path.join(HERE, "feeder_state.json")

SOURCE_RANK_COLUMN = "_collector_source_rank"
SOURCE_RANK_TERMINAL_CSV = 10
SOURCE_RANK_JSON = 20
SOURCE_RANK_LOCAL_CSV = 30
SOURCE_RANK_LOCAL_PART = 40

FETCH_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.5
RETRY_MAX_SECONDS = 30.0
MAX_STDOUT_BYTES = 1_048_576


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
    return _get_response(
        f"/api/tasks/{task_id}/stdout",
        params={"max_bytes": MAX_STDOUT_BYTES}, timeout=timeout).text


def list_tasks(prefix):
    # 신 스케줄러(limit/name_prefix 지원) 우선
    t = _get_json("/api/tasks",
                  params={"limit": 10000, "name_prefix": prefix}, timeout=30)
    tasks = t if isinstance(t, list) else t.get("tasks", [])
    matched = [x for x in tasks if str(x.get("name", "")).startswith(prefix)]
    page_seen = {x["id"]: x for x in matched}
    seen = dict(page_seen)

    def probe(tid):
        try:
            x = _get_json(f"/api/tasks/{tid}", timeout=15)
        except FetchError as exc:
            return False if exc.status_code == 404 else None
        return x if str(x.get("name", "")).startswith(prefix) else False

    # The scheduler has no pagination and applies its limit before prefix
    # filtering. Always recover unseen feeder IDs from the durable local ledger.
    try:
        with open(FEEDER_STATE_PATH, encoding="utf-8") as stream:
            feeder_state = json.load(stream)
        ledger_ids = {int(task_id) for task_id in feeder_state.get("outstanding", [])}
    except FileNotFoundError:
        ledger_ids = set()
    except (OSError, ValueError, TypeError) as exc:
        raise FetchError(f"feeder task ledger is unreadable: {exc}") from exc
    cache = _load_cache()
    judged = set(cache.get("nodata", [])) | set(cache.get("harvested", []))
    missing_ids = sorted(ledger_ids - set(seen) - judged)[:500]
    for task_id in missing_ids:
        recovered = probe(task_id)
        if recovered:
            seen[task_id] = recovered
    if len(tasks) > 250:
        return list(seen.values())

    # 구 스케줄러(200개 페이지): ID 연속 스캔으로 누락 보완
    if not page_seen:
        return list(seen.values())

    scan_seen = dict(page_seen)
    ids = sorted(scan_seen)
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
                scan_seen[cur] = r
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
    lo, hi = min(scan_seen), max(scan_seen)
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
        return {"nodata": [], "harvested": [], "local_parts": []}


def _save_cache(c):
    os.makedirs(DATASET_DIR, exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(c, f)
    os.replace(tmp, CACHE_PATH)


def _commit_pending_cache(pending_harvested, pending_nodata, pending_local_parts=()):
    """Merge pending terminal IDs into the latest on-disk cache."""
    if not pending_harvested and not pending_nodata and not pending_local_parts:
        return
    cache = _load_cache()
    harvested = cache.setdefault("harvested", [])
    nodata = cache.setdefault("nodata", [])
    local_parts = cache.setdefault("local_parts", [])
    harvested.extend(tid for tid in pending_harvested if tid not in harvested)
    nodata.extend(tid for tid in pending_nodata if tid not in nodata)
    known_local_parts = set(local_parts)
    for part in pending_local_parts:
        if part not in known_local_parts:
            local_parts.append(part)
            known_local_parts.add(part)
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
PROBE_FIX_COMMIT = "6245ae84ba2734d2f1b6619fba3b2a8f15d20f42"



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


def _tag_source(frame, rank):
    tagged = frame.copy()
    tagged[SOURCE_RANK_COLUMN] = rank
    return tagged


def _deduplicate_ranked_rows(frame, dedup_keys):
    """Deduplicate after ordering rows by their durable source precedence."""
    if SOURCE_RANK_COLUMN in frame.columns:
        frame = frame.sort_values(SOURCE_RANK_COLUMN, kind="stable")
    return deduplicate_rows(frame, dedup_keys)


def load_local_result_frames(cache):
    """Load the current CSV and only parquet parts not committed previously."""
    frames = []
    pending_parts = []
    read_errors = 0

    if os.path.isfile(LOCAL_RESULTS_CSV):
        try:
            # The producer uses the same lock for append and schema rotation.
            # saved_at/project_name are its final columns, so requiring both also
            # rejects a row left truncated by an interrupted append.
            with FileLock(LOCAL_RESULTS_CSV + ".lock"):
                frame = pd.read_csv(LOCAL_RESULTS_CSV, on_bad_lines="skip")
            complete = _complete_key_mask(frame, ["project_name", "saved_at"])
            incomplete_count = int((~complete).sum())
            if incomplete_count:
                print(f"local csv incomplete rows skipped: {incomplete_count}")
                frame = frame.loc[complete].reset_index(drop=True)
            if len(frame):
                frames.append(_tag_source(frame, SOURCE_RANK_LOCAL_CSV))
                print(f"local csv rows: {len(frame)}")
        except Exception as exc:
            read_errors += 1
            print(f"local csv read failed: {exc}")

    cached_parts = set(cache.get("local_parts", []))
    pattern = os.path.join(LOCAL_RESULTS_PARTS_DIR, "*.parquet")
    for path in sorted(glob.glob(pattern)):
        part_id = os.path.basename(path)
        if part_id in cached_parts:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            # A part can be visible before its parquet footer is complete. Leave it
            # uncached so a later collector pass retries it.
            read_errors += 1
            print(f"local parquet read failed ({part_id}): {exc}")
            continue
        pending_parts.append(part_id)
        if len(frame):
            frames.append(_tag_source(frame, SOURCE_RANK_LOCAL_PART))

    if pending_parts:
        print(f"local parquet parts: {len(pending_parts)} new files")
    if read_errors:
        print(f"local read errors: {read_errors} (left uncached for retry)")
    return frames, pending_parts


def sanitize_bad_probes(df):
    import subprocess
    global PROBE_FIX_HASHES_OK
    if "git_hash" not in df.columns:
        return df, 0
    if PROBE_FIX_HASHES_OK is None:
        try:
            git_run = {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "cwd": os.path.join(HERE, "..", ".."),
            }
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", PROBE_FIX_COMMIT, "HEAD"],
                check=True,
                **git_run,
            )
            descendants = subprocess.run(
                ["git", "rev-list", "--ancestry-path", f"{PROBE_FIX_COMMIT}..HEAD"],
                check=True,
                **git_run,
            ).stdout.split()
            PROBE_FIX_HASHES_OK = {PROBE_FIX_COMMIT}
            PROBE_FIX_HASHES_OK.update(descendants)
        except Exception as exc:
            raise RuntimeError("probe-fix git ancestry classification failed") from exc
    bad_cols = [
        column for column in df.columns
        if column.startswith("Tprobe_")
        and (
            column.endswith("_side_max")
            or column.endswith("_side_mean")
            or column.startswith("Tprobe_core_center_")
        )
    ]
    def _is_post_fix(value):
        if not isinstance(value, str):
            return False
        revision = value.strip().lower()
        if not revision:
            return False
        return any(commit.startswith(revision) for commit in PROBE_FIX_HASHES_OK)

    mask = ~df["git_hash"].map(_is_post_fix)
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


def normalize_thermal_validity(df):
    """Demote thermal success flags without complete physical and convergence evidence."""
    if "thermal_solved" not in df.columns:
        return df, 0
    solved = pd.to_numeric(df["thermal_solved"], errors="coerce").eq(1)
    if not solved.any():
        return df, 0

    required = ("T_max_Tx", "T_max_Rx_main", "T_max_core")
    complete = pd.Series(True, index=df.index)
    for column in required:
        if column not in df.columns:
            complete &= False
        else:
            complete &= pd.to_numeric(df[column], errors="coerce").map(math.isfinite)

    if "N2_side" not in df.columns:
        complete &= False
        side_required = pd.Series(False, index=df.index)
    else:
        n2_side = pd.to_numeric(df["N2_side"], errors="coerce")
        complete &= n2_side.notna()
        side_required = n2_side.gt(0)
    if "T_max_Rx_side" not in df.columns:
        complete &= ~side_required
    else:
        side_finite = pd.to_numeric(df["T_max_Rx_side"], errors="coerce").map(math.isfinite)
        complete &= ~side_required | side_finite

    expected_mask = pd.Series(11, index=df.index).where(~side_required, 15)
    for column, predicate in (
        ("thermal_required_group_mask", lambda values: values.eq(expected_mask)),
        ("thermal_required_missing_count", lambda values: values.eq(0)),
        ("thermal_extraction_complete", lambda values: values.eq(1)),
    ):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            has_contract = values.notna()
            complete &= ~has_contract | predicate(values)

    # New solver rows must carry native residual proof. Legacy rows without the
    # field remain EM-only rather than being silently trusted as thermal data.
    for column in ("thermal_convergence_available", "thermal_converged"):
        if column not in df.columns:
            complete &= False
        else:
            complete &= pd.to_numeric(df[column], errors="coerce").eq(1)
    residual_columns = (
        "thermal_residual_continuity",
        "thermal_residual_x_velocity",
        "thermal_residual_y_velocity",
        "thermal_residual_z_velocity",
    )
    if "thermal_residual_flow_limit" not in df.columns:
        complete &= False
    else:
        flow_limit = pd.to_numeric(df["thermal_residual_flow_limit"], errors="coerce")
        complete &= flow_limit.map(math.isfinite) & flow_limit.gt(0) & flow_limit.le(1e-3)
        for column in residual_columns:
            if column not in df.columns:
                complete &= False
            else:
                residual = pd.to_numeric(df[column], errors="coerce")
                complete &= residual.map(math.isfinite) & residual.ge(0) & residual.le(flow_limit)
    if "thermal_residual_energy_limit" not in df.columns or "thermal_residual_energy" not in df.columns:
        complete &= False
    else:
        energy_limit = pd.to_numeric(df["thermal_residual_energy_limit"], errors="coerce")
        energy = pd.to_numeric(df["thermal_residual_energy"], errors="coerce")
        complete &= (
            energy_limit.map(math.isfinite) & energy_limit.gt(0) & energy_limit.le(1e-7)
            & energy.map(math.isfinite) & energy.ge(0) & energy.le(energy_limit)
        )
    if "thermal_iterations" not in df.columns:
        complete &= False
    else:
        complete &= pd.to_numeric(df["thermal_iterations"], errors="coerce").gt(0)

    invalid = solved & ~complete
    count = int(invalid.sum())
    if count:
        df = df.copy()
        df.loc[invalid, "thermal_solved"] = 0
        if "result_valid_thermal" not in df.columns:
            df["result_valid_thermal"] = float("nan")
        df.loc[invalid, "result_valid_thermal"] = 0
    return df, count


def reject_explicit_dirty_provenance(df):
    """Drop rows that explicitly report dirty or untrusted execution provenance."""
    if df is None or df.empty:
        return df, 0
    keep = pd.Series(True, index=df.index)
    for column in ("git_dirty", "pyaedt_library_git_dirty"):
        if column not in df.columns:
            continue
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        keep &= raw.isna() | numeric.eq(0)
    for column in ("matrix_solve_attempts", "loss_solve_attempts"):
        if column not in df.columns:
            continue
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        keep &= raw.isna() | numeric.eq(1)
    if "matrix_extraction_backend" in df.columns:
        raw = df["matrix_extraction_backend"]
        keep &= raw.isna() | raw.astype(str).eq("export_rl_matrix")
    rejected = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), rejected


def _row_hashes(df, columns):
    normalized = df.loc[:, columns].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False)


def _fallback_columns(frame, other=None):
    """Columns used for legacy identity, excluding collector-added metadata."""
    excluded = {"task_id", "task_name", SOURCE_RANK_COLUMN}
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
    """Select absent identities and higher-ranked replacements from the master."""
    incoming = _deduplicate_ranked_rows(incoming, dedup_keys)
    if old is None or old.empty:
        return incoming.reset_index(drop=True)

    new_mask = pd.Series(True, index=incoming.index)
    can_compare_keys = bool(dedup_keys) and all(c in old.columns for c in dedup_keys)
    incoming_complete = _complete_key_mask(incoming, dedup_keys)
    if can_compare_keys and incoming_complete.any():
        old_complete = _complete_key_mask(old, dedup_keys)
        old_key_hash_series = _row_hashes(old.loc[old_complete], dedup_keys)
        old_key_hashes = set(old_key_hash_series.tolist())
        incoming_key_hashes = _row_hashes(
            incoming.loc[incoming_complete], dedup_keys)
        accepted = ~incoming_key_hashes.isin(old_key_hashes)
        if SOURCE_RANK_COLUMN in incoming.columns:
            if SOURCE_RANK_COLUMN in old.columns:
                old_ranks = pd.to_numeric(
                    old.loc[old_complete, SOURCE_RANK_COLUMN], errors="coerce"
                ).fillna(0)
            else:
                old_ranks = pd.Series(0, index=old_key_hash_series.index, dtype=float)
            old_rank_by_hash = {}
            for key_hash, rank in zip(old_key_hash_series.tolist(), old_ranks.tolist()):
                old_rank_by_hash[key_hash] = max(old_rank_by_hash.get(key_hash, 0), rank)
            incoming_ranks = pd.to_numeric(
                incoming.loc[incoming_complete, SOURCE_RANK_COLUMN], errors="coerce"
            ).fillna(0)
            higher_rank = [
                rank > old_rank_by_hash.get(key_hash, float("inf"))
                for key_hash, rank in zip(incoming_key_hashes.tolist(), incoming_ranks.tolist())
            ]
            accepted |= pd.Series(higher_rank, index=accepted.index)
        new_mask.loc[incoming_complete] = accepted.to_numpy()

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


def _drop_replaced_key_rows(old, replacements, dedup_keys):
    """Remove old complete-key rows superseded by accepted incoming rows."""
    if old is None or old.empty:
        return old
    old_complete = _complete_key_mask(old, dedup_keys)
    replacement_complete = _complete_key_mask(replacements, dedup_keys)
    if not old_complete.any() or not replacement_complete.any():
        return old
    replacement_hashes = set(
        _row_hashes(replacements.loc[replacement_complete], dedup_keys).tolist())
    replaced = pd.Series(False, index=old.index)
    replaced.loc[old_complete] = _row_hashes(
        old.loc[old_complete], dedup_keys).isin(replacement_hashes).to_numpy()
    return old.loc[~replaced].reset_index(drop=True)


def _source_rank_rows(frame, dedup_keys):
    """Return one numeric source-rank row per complete sample identity."""
    columns = list(dedup_keys) + [SOURCE_RANK_COLUMN]
    if frame is None or frame.empty or not all(c in frame.columns for c in dedup_keys):
        return pd.DataFrame(columns=columns)
    complete = _complete_key_mask(frame, dedup_keys)
    if not complete.any():
        return pd.DataFrame(columns=columns)
    ranked = frame.loc[complete, list(dedup_keys)].copy()
    if SOURCE_RANK_COLUMN in frame.columns:
        ranked[SOURCE_RANK_COLUMN] = pd.to_numeric(
            frame.loc[complete, SOURCE_RANK_COLUMN], errors="coerce").fillna(0).to_numpy()
    else:
        ranked[SOURCE_RANK_COLUMN] = 0
    return _deduplicate_ranked_rows(ranked, dedup_keys).reset_index(drop=True)


def _attach_source_ranks(frame, sidecar, dedup_keys):
    """Attach sidecar provenance for comparison without exposing it to training."""
    if frame is None:
        return None
    ranked = frame.copy()
    if SOURCE_RANK_COLUMN in ranked.columns:
        legacy_rank = pd.to_numeric(ranked[SOURCE_RANK_COLUMN], errors="coerce").fillna(0)
        ranked = ranked.drop(columns=[SOURCE_RANK_COLUMN])
    else:
        legacy_rank = pd.Series(0, index=ranked.index, dtype=float)
    ranked["_collector_legacy_rank"] = legacy_rank.to_numpy()

    sidecar = _source_rank_rows(sidecar, dedup_keys)
    if len(sidecar) and all(c in ranked.columns for c in dedup_keys):
        sidecar = sidecar.rename(columns={SOURCE_RANK_COLUMN: "_collector_saved_rank"})
        ranked = ranked.merge(sidecar, on=list(dedup_keys), how="left", sort=False)
        saved_rank = pd.to_numeric(
            ranked.pop("_collector_saved_rank"), errors="coerce").fillna(0)
    else:
        saved_rank = pd.Series(0, index=ranked.index, dtype=float)
    legacy_rank = pd.to_numeric(
        ranked.pop("_collector_legacy_rank"), errors="coerce").fillna(0)
    ranked[SOURCE_RANK_COLUMN] = pd.concat(
        [legacy_rank, saved_rank], axis=1).max(axis=1)
    return ranked


def _matching_replay_rows(master, incoming, identities, dedup_keys):
    """Return deterministic replay rows only when their persisted payload matches."""
    if incoming is None:
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")
    replay = _deduplicate_ranked_rows(incoming.copy(), dedup_keys)
    try:
        persisted = identities.merge(
            master, on=dedup_keys, how="left", validate="one_to_one")
        replayed = identities.merge(
            replay, on=dedup_keys, how="left", validate="one_to_one",
            indicator="_replay_merge")
    except (pd.errors.MergeError, ValueError) as exc:
        raise RuntimeError("replay payload identity is ambiguous") from exc
    if not replayed.pop("_replay_merge").eq("both").all():
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")

    excluded = {"task_id", "task_name", SOURCE_RANK_COLUMN}
    payload_columns = sorted(
        (set(persisted.columns) | set(replayed.columns)) - excluded)
    persisted = persisted.reindex(columns=payload_columns)
    replayed = replayed.reindex(columns=payload_columns)
    try:
        pd.testing.assert_frame_equal(
            persisted.reset_index(drop=True), replayed.reset_index(drop=True),
            check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise RuntimeError(
            "replay payload differs from persisted master") from exc
    return replay


def _validated_source_rank_sidecar(master, sidecar, incoming, dedup_keys):
    """Validate rank coverage, repairing only rows replayed after master-first install."""
    columns = list(dedup_keys) + [SOURCE_RANK_COLUMN]
    if sidecar is None:
        validated = pd.DataFrame(columns=columns)
    else:
        missing_columns = [column for column in columns if column not in sidecar.columns]
        if missing_columns:
            raise RuntimeError(
                f"source rank sidecar schema is invalid; missing {missing_columns}")
        validated = sidecar[columns].copy()
        complete = _complete_key_mask(validated, dedup_keys)
        ranks = pd.to_numeric(validated[SOURCE_RANK_COLUMN], errors="coerce")
        if (not complete.all() or not ranks.map(math.isfinite).all()
                or (ranks < 0).any() or validated.duplicated(dedup_keys).any()):
            raise RuntimeError("source rank sidecar contains invalid or duplicate rows")
        validated[SOURCE_RANK_COLUMN] = ranks

    if master is None or master.empty:
        return validated
    if not all(key in master.columns for key in dedup_keys):
        raise RuntimeError("master dataset is missing source-rank identity columns")
    identities = master.loc[
        _complete_key_mask(master, dedup_keys), dedup_keys].drop_duplicates()
    coverage = identities.merge(
        validated[dedup_keys], on=dedup_keys, how="left", indicator=True)
    missing = coverage.loc[coverage["_merge"].ne("both"), dedup_keys]
    if not len(missing):
        return validated

    replay = _matching_replay_rows(master, incoming, missing, dedup_keys)
    replay_ranks = _source_rank_rows(replay, dedup_keys)
    repairable = missing.merge(
        replay_ranks, on=dedup_keys, how="left", indicator=True)
    if (not repairable["_merge"].eq("both").all()
            or not pd.to_numeric(
                repairable[SOURCE_RANK_COLUMN], errors="coerce").map(math.isfinite).all()):
        raise RuntimeError(
            "source rank sidecar does not cover every master dataset identity")
    repaired = repairable[columns]
    return pd.concat([validated, repaired], ignore_index=True, sort=False)


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
    """Install fully serialized files in the caller's recovery-safe order."""
    try:
        for staged, target in staged_targets:
            os.replace(staged, target)
    finally:
        for staged, _ in staged_targets:
            if os.path.exists(staged):
                os.remove(staged)


def merge_dataset(merged, dedup_keys, prefix, pending_harvested=(), pending_nodata=(),
                  pending_local_parts=()):
    """Merge rows and terminal cache state under one inter-process lock."""
    master_path = os.path.join(DATASET_DIR, "train.parquet")
    manifest_path = os.path.join(DATASET_DIR, "manifest.json")
    source_rank_path = os.path.join(DATASET_DIR, "source_ranks.parquet")
    lock_path = master_path + ".lock"
    with FileLock(lock_path):
        old = pd.read_parquet(master_path) if os.path.isfile(master_path) else None
        source_ranks = (
            pd.read_parquet(source_rank_path) if os.path.isfile(source_rank_path) else None
        )
        source_ranks = _validated_source_rank_sidecar(
            old, source_ranks, merged, dedup_keys)
        ranked_old = _attach_source_ranks(old, source_ranks, dedup_keys)
        new_rows = select_new_unique_rows(merged, ranked_old, dedup_keys)
        new_unique_rows = len(new_rows)
        if not new_unique_rows:
            _commit_pending_cache(
                pending_harvested, pending_nodata, pending_local_parts)
            return new_unique_rows, 0 if old is None else len(old), master_path

        stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        part_path = os.path.join(
            DATASET_DIR, f"collected_{prefix.replace('/', '_')}_{stamp}.parquet")
        retained_old = _drop_replaced_key_rows(ranked_old, new_rows, dedup_keys)
        ranked_allf = new_rows if retained_old is None else pd.concat(
            [retained_old, new_rows], ignore_index=True, sort=False)
        source_ranks = _source_rank_rows(ranked_allf, dedup_keys)
        persisted_new_rows = new_rows.drop(columns=[SOURCE_RANK_COLUMN], errors="ignore")
        allf = ranked_allf.drop(columns=[SOURCE_RANK_COLUMN], errors="ignore")
        manifest = {
            "updated": stamp, "total_rows": len(allf), "new_rows": new_unique_rows,
            "new_unique_rows": new_unique_rows,
            "git_hashes": sorted(allf["git_hash"].dropna().unique().tolist())
            if "git_hash" in allf.columns else [],
            "prefix": prefix,
        }

        staged = []
        try:
            staged.append((_stage_parquet(persisted_new_rows, part_path), part_path))
            staged.append((_stage_manifest(manifest, manifest_path), manifest_path))
            staged.append((_stage_parquet(allf, master_path), master_path))
            # Rank follows master. If this replacement fails, the part remains
            # uncached and the lower-rank sidecar causes a safe retry next pass.
            staged.append((_stage_parquet(source_ranks, source_rank_path), source_rank_path))
            _replace_staged(staged)
        except Exception:
            for temp_path, _ in staged:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise

        _commit_pending_cache(
            pending_harvested, pending_nodata, pending_local_parts)
        return new_unique_rows, len(allf), master_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mft-camp")
    ap.add_argument("--max-conv-err", type=float, default=1.5)
    ap.add_argument("--cancelled-fetch-limit", type=int, default=500)
    args = ap.parse_args(argv)

    os.makedirs(DATASET_DIR, exist_ok=True)

    tasks = list_tasks(args.prefix)
    done = [t for t in tasks if t.get("status") == "completed"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    cancelled = [t for t in tasks if t.get("status") == "cancelled"]
    status_counts = {}
    for task in tasks:
        status = str(task.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    active_count = sum(status_counts.get(status, 0) for status in ("queued", "attaching", "running"))
    print(
        f"tasks: {len(tasks)} (completed {len(done)}, failed {len(failed)}, "
        f"active {active_count}, cancelled {status_counts.get('cancelled', 0)})"
    )

    # 실패/취소 태스크도 stdout에 결과가 있으면 회수 (pyaedt teardown 크래시나
    # count 배치 중간 취소 전에 성공한 샘플이 남는 경우가 있음).
    # A bounded cancelled batch prevents historical backlog from monopolizing
    # the scheduler API; fresh completed/failed tasks are always handled first.
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
    never_started_cancelled = [
        t for t in cancelled
        if t["id"] not in skip
        and "started_at" in t
        and t["started_at"] is None
    ]
    never_started_ids = {t["id"] for t in never_started_cancelled}
    pending_nodata = list(never_started_ids)
    pending_local_parts = []
    terminal = done + failed + cancelled
    n_skipped = sum(t["id"] in skip for t in terminal)
    pending_primary = sorted(
        (t for t in done + failed if t["id"] not in skip),
        key=lambda task: int(task["id"]), reverse=True,
    )
    pending_cancelled = sorted(
        (t for t in cancelled
         if t["id"] not in skip and t["id"] not in never_started_ids),
        key=lambda task: int(task["id"]),
    )[:max(0, args.cancelled_fetch_limit)]
    pending_terminal = pending_primary + pending_cancelled
    for t in pending_terminal:
        try:
            out = fetch_stdout(t["id"])
        except FetchError as exc:
            n_fetch_errors += 1
            print(f"fetch error task {t['id']}: {exc}")
            continue
        # Read both sources: JSON wins duplicate identities by rank, while the
        # legacy aggregate can still recover a row whose JSON line was truncated.
        task_frames = []
        json_frame = fetch_streamed_rows(t["id"], out=out)
        if json_frame is not None and len(json_frame):
            task_frames.append((json_frame, SOURCE_RANK_JSON))
        csv_frame = fetch_result_rows(t["id"], out=out)
        if csv_frame is not None and len(csv_frame):
            task_frames.append((csv_frame, SOURCE_RANK_TERMINAL_CSV))
        if task_frames:
            for frame, source_rank in task_frames:
                frame["task_id"] = t["id"]
                frame["task_name"] = t.get("name", "")
                frames.append(_tag_source(frame, source_rank))
            pending_harvested.append(t["id"])
            if t.get("status") in ("failed", "cancelled"):
                n_salvaged += 1
        else:
            # Only a successful stdout fetch proves that a task has no data.
            pending_nodata.append(t["id"])
    if n_skipped:
        print(f"cache skip: {n_skipped} terminal tasks")
    if never_started_cancelled:
        print(
            f"never-started cancelled: {len(never_started_cancelled)} marked nodata")
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
            frames.append(_tag_source(df, SOURCE_RANK_JSON))
            n_streamed += len(rows)
    if n_salvaged:
        print(f"salvaged from failed tasks: {n_salvaged}")
    if n_streamed:
        print(f"streamed from running tasks: {n_streamed} rows")
    if n_fetch_errors:
        print(f"fetch_errors: {n_fetch_errors} (not cached as nodata)")

    # Local parquet parts are the durable schema-union source. The current CSV is
    # retained as a lower-ranked fallback when a parquet write failed.
    local_frames, pending_local_parts = load_local_result_frames(cache)
    for frame in local_frames:
        frame["task_name"] = "local"
        frames.append(frame)

    if not frames:
        master_path = os.path.join(DATASET_DIR, "train.parquet")
        with FileLock(master_path + ".lock"):
            _commit_pending_cache(
                pending_harvested, pending_nodata, pending_local_parts)
        print("new_unique_rows: 0")
        print("no result rows collected")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged, n_dirty = reject_explicit_dirty_provenance(merged)
    if n_dirty:
        print(f"provenance filter: {n_dirty} explicit dirty-source rows rejected")

    # 중복 제거 (재회수/재시도 대비)
    # Both fields are required for the primary identity. A missing column sends
    # every row through the content-hash legacy fallback.
    dedup_keys = ["project_name", "saved_at"]
    before = len(merged)
    merged = _deduplicate_ranked_rows(merged, dedup_keys)
    print(f"dedup: {before} -> {len(merged)}")

    merged, n_bad_probe = sanitize_bad_probes(merged)
    if n_bad_probe:
        print(f"probe-fix 이전 행 {n_bad_probe}개: side/core_center 프로브 컬럼 NaN 처리 (T_max/leeward는 유지)")
    merged, n_false_thermal = normalize_thermal_validity(merged)
    if n_false_thermal:
        print(
            f"thermal validity: {n_false_thermal} legacy false-success rows demoted to EM-only"
        )
    merged, n_filtered = convergence_filter(merged, args.max_conv_err)
    print(f"convergence filter (<= {args.max_conv_err}%): -{n_filtered} rows")

    new_unique_rows, total_rows, master_path = merge_dataset(
        merged, dedup_keys, args.prefix,
        pending_harvested=pending_harvested, pending_nodata=pending_nodata,
        pending_local_parts=pending_local_parts)
    print(f"new_unique_rows: {new_unique_rows}")
    if not new_unique_rows:
        print(f"dataset unchanged: {total_rows} rows -> {master_path}")
        return {"new_unique_rows": 0, "fetch_errors": n_fetch_errors}
    print(f"dataset: {total_rows} rows total -> {master_path}")
    return {"new_unique_rows": new_unique_rows, "fetch_errors": n_fetch_errors}


if __name__ == "__main__":
    main()
