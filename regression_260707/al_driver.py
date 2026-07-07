"""
능동학습 검증 루프 드라이버 (재개 가능 상태기계).

라운드: TRAIN -> OPTIMIZE -> SELECT -> SUBMIT -> WAIT -> INGEST -> CHECK -> (다음 라운드 | 종료)

- TRAIN: train_models.py (고정 하이퍼파라미터 재학습, 검증 데이터 sample_weight=3)
- OPTIMIZE: run_nsga2.py (16 재시작, warm start, 밀도 게이트)
- SELECT: select_candidates.select (K=33: 활용/경계/탐사/재검증)
- SUBMIT/WAIT: scheduler_client (fea_bursty, RESULT_JSON 회수, 실패 시 64GB 1회 재시도)
- INGEST: 검증 rows -> dataset 병합 + 예측 vs 실측 오차 기록 -> q 적응
- CHECK: 종료판정 (스펙 통과 >=3 + 배치 일치도 + HV 정체) 또는 하드캡 10라운드

사용:
  python al_driver.py            # state.json 이 있으면 이어서
  python al_driver.py --reset    # 처음부터
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "training"))
sys.path.insert(0, os.path.join(HERE, "verify"))

STATE_PATH = os.path.join(HERE, "al_rounds", "state.json")
DATASET = os.path.join(HERE, "data", "dataset", "train.parquet")
PY = sys.executable

SPEC = {
    "Llt_target_uH": 27.5, "Llt_tol_uH": 0.55,
    "T_limit_C": 100.0, "B_limit_T": 1.2,
    "agree_llt_med_pct": 0.5, "agree_llt_max_pct": 1.0,
    "agree_T_med_C": 3.0, "agree_T_max_C": 5.0,
    "agree_P_med_pct": 3.0,
    "max_rounds": 10, "K": 33,
}

T_TARGETS = ["Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
             "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max"]


def load_state():
    if os.path.isfile(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"round": 1, "stage": "TRAIN", "q_mult": 1.0, "task_map": {}, "history": []}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATE_PATH)


def run(cmd, **kw):
    print(f"[al] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def stage_train(st):
    run([PY, os.path.join("training", "train_models.py")])
    st["stage"] = "OPTIMIZE"


def stage_optimize(st):
    rnd = st["round"]
    spec_path = os.path.join(HERE, "al_rounds", f"round_{rnd:02d}_spec.json")
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    # q 적응 배율 반영
    spec = {"q_sigma": 1.28 * st.get("q_mult", 1.0)}
    json.dump(spec, open(spec_path, "w"))
    run([PY, os.path.join("optimization", "run_nsga2.py"),
         "--restarts", "16", "--round", str(rnd), "--spec", spec_path])
    st["stage"] = "SELECT"


def stage_select(st):
    rnd = st["round"]
    rdir = os.path.join(HERE, "al_rounds", f"round_{rnd:02d}")
    X = np.load(os.path.join(rdir, "pareto_X.npy"))
    F = np.load(os.path.join(rdir, "pareto_F.npy"))
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))

    sig_cols = [c for c in front.columns if c.startswith("sigma_")]
    sig_norm = front[sig_cols].div(front[sig_cols].mean() + 1e-12).sum(axis=1).to_numpy()
    G = np.zeros((len(X), 1))  # Pareto는 feasible만 오므로 마진은 예측치로 재구성
    margin = (SPEC["Llt_tol_uH"] - (front["pred_Llt_phys"] - SPEC["Llt_target_uH"]).abs()).to_numpy()
    t_margin = np.min([(SPEC["T_limit_C"] - front[f"pred_{t}"]).to_numpy()
                       for t in T_TARGETS if f"pred_{t}" in front.columns] or [np.full(len(X), 99.0)], axis=0)
    G = -np.column_stack([margin, t_margin])  # select()는 -G를 마진으로 봄

    verified_X = None
    vx_path = os.path.join(HERE, "al_rounds", "verified_X.npy")
    if os.path.isfile(vx_path):
        verified_X = np.load(vx_path)

    from select_candidates import select
    picked = select(X, F, G, sig_norm, verified_X=verified_X)
    np.save(os.path.join(rdir, "selected_idx.npy"), np.array(picked))
    print(f"[al] round {rnd}: {len(picked)} candidates selected")
    st["stage"] = "SUBMIT"


def stage_submit(st):
    rnd = st["round"]
    rdir = os.path.join(HERE, "al_rounds", f"round_{rnd:02d}")
    picked = np.load(os.path.join(rdir, "selected_idx.npy")).tolist()
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))

    from module.input_parameter_260706 import KEYS  # noqa
    import scheduler_client as sc
    profile = json.load(open(os.path.join(HERE, "verify", "profiles", "standard.json")))

    task_map = {}
    for j, i in enumerate(picked):
        row = front.iloc[i]
        params = {k: (row[k] if not isinstance(row[k], np.generic) else row[k].item())
                  for k in KEYS if k in front.columns and pd.notna(row[k])}
        name = f"mft-al-r{rnd:02d}-c{j:02d}"
        tid = sc.submit_verification(name, f"mft_al_r{rnd:02d}_c{j:02d}", params, profile,
                                     mem_mb=profile.get("mem_mb", 32768))
        task_map[str(i)] = tid
        time.sleep(0.5)
    st["task_map"] = task_map
    st["stage"] = "WAIT"


def stage_wait(st):
    import scheduler_client as sc
    tids = [t for t in st["task_map"].values() if t]
    status = sc.wait_all(tids, poll_s=180, timeout_s=6 * 3600)
    # 실패 후보는 64GB로 1회 재시도 (플랜: 메모리 부족/불안정 대응)
    rnd = st["round"]
    front = pd.read_csv(os.path.join(HERE, "al_rounds", f"round_{rnd:02d}", "pareto_front.csv"))
    from module.input_parameter_260706 import KEYS
    profile = json.load(open(os.path.join(HERE, "verify", "profiles", "standard.json"), encoding="utf-8"))
    retried = {}
    for idx_str, tid in list(st["task_map"].items()):
        if tid and status.get(tid) != "completed" and not sc.fetch_result_json(tid):
            row = front.iloc[int(idx_str)]
            params = {k: (row[k].item() if hasattr(row[k], "item") else row[k])
                      for k in KEYS if k in front.columns and pd.notna(row[k])}
            new_tid = sc.submit_verification(f"mft-al-r{rnd:02d}-retry-{idx_str}",
                                             f"mft_al_r{rnd:02d}_rt{idx_str}", params, profile,
                                             mem_mb=65536)
            retried[idx_str] = new_tid
            st["task_map"][idx_str] = new_tid
    if retried:
        print(f"[al] {len(retried)} candidates retried at 64GB")
        sc.wait_all([t for t in retried.values() if t], poll_s=180, timeout_s=5 * 3600)
    n_done = sum(1 for tid in st["task_map"].values() if tid and sc.fetch_result_json(tid))
    print(f"[al] wait done: {n_done}/{len(tids)} with data")
    if n_done < 0.7 * len(tids):
        print("[al] WARNING: <70% completion - 실패 태스크 로그 점검 필요")
    st["stage"] = "INGEST"


def stage_ingest(st):
    rnd = st["round"]
    rdir = os.path.join(HERE, "al_rounds", f"round_{rnd:02d}")
    front = pd.read_csv(os.path.join(rdir, "pareto_front.csv"))
    import scheduler_client as sc

    rows, errs = [], []
    for idx_str, tid in st["task_map"].items():
        if not tid:
            continue
        res = sc.fetch_result_json(tid)
        if not res:
            continue
        res["source"] = f"al_round_{rnd}"
        res["sample_weight"] = 3.0
        rows.append(res)
        i = int(idx_str)
        pred = front.iloc[i]
        llt_fea = 2.0 * float(res.get("Llt", np.nan))  # 대칭 -> 실물
        err = {
            "idx": i, "task_id": tid,
            "llt_pred": float(pred["pred_Llt_phys"]), "llt_fea": llt_fea,
            "dllt_pct": abs(float(pred["pred_Llt_phys"]) - llt_fea) / SPEC["Llt_target_uH"] * 100,
        }
        for t in T_TARGETS:
            if f"pred_{t}" in pred and t in res and pd.notna(res.get(t)):
                err[f"d_{t}"] = abs(float(pred[f"pred_{t}"]) - float(res[t]))
        errs.append(err)

    if rows:
        new = pd.DataFrame(rows)
        if os.path.isfile(DATASET):
            old = pd.read_parquet(DATASET)
            allf = pd.concat([old, new], ignore_index=True, sort=False)
        else:
            allf = new
        allf.to_parquet(DATASET, index=False)
        # 검증된 X 축적 (선정 중복 방지)
        X = np.load(os.path.join(rdir, "pareto_X.npy"))
        v_new = X[[e["idx"] for e in errs]]
        vx_path = os.path.join(HERE, "al_rounds", "verified_X.npy")
        v_all = np.vstack([np.load(vx_path), v_new]) if os.path.isfile(vx_path) else v_new
        np.save(vx_path, v_all)

    err_df = pd.DataFrame(errs)
    err_df.to_csv(os.path.join(rdir, "verification_errors.csv"), index=False)
    st["last_errs"] = err_df.to_dict("list") if len(err_df) else {}
    print(f"[al] ingested {len(rows)} verified rows")
    st["stage"] = "CHECK"


def stage_check(st):
    rnd = st["round"]
    errs = st.get("last_errs") or {}
    hist = {"round": rnd, "time": datetime.now().isoformat(timespec="seconds")}

    ok_specs = 0
    agree = False
    if errs and errs.get("dllt_pct"):
        d = np.array(errs["dllt_pct"], dtype=float)
        hist["dllt_med_pct"] = float(np.median(d))
        hist["dllt_max_pct"] = float(np.max(d))
        t_cols = [k for k in errs if k.startswith("d_Tprobe")]
        dT = np.array([v for k in t_cols for v in errs[k]], dtype=float) if t_cols else np.array([])
        hist["dT_med"] = float(np.median(dT)) if len(dT) else None
        hist["dT_max"] = float(np.max(dT)) if len(dT) else None

        agree = (hist["dllt_med_pct"] <= SPEC["agree_llt_med_pct"]
                 and hist["dllt_max_pct"] <= SPEC["agree_llt_max_pct"]
                 and (hist["dT_med"] is None or hist["dT_med"] <= SPEC["agree_T_med_C"])
                 and (hist["dT_max"] is None or hist["dT_max"] <= SPEC["agree_T_max_C"]))

        # 실측 스펙 통과 후보 수 (FEA 기준)
        llt_fea = np.array(errs["llt_fea"], dtype=float)
        band = np.abs(llt_fea - SPEC["Llt_target_uH"]) <= SPEC["Llt_tol_uH"]
        ok_specs = int(band.sum())
        hist["fea_llt_pass"] = ok_specs

        # 예측통과/실측탈락 발생 시 q 조임
        miss = (~band).sum()
        if miss > 0:
            st["q_mult"] = min(st.get("q_mult", 1.0) * 1.25, 3.0)
        elif hist["dllt_max_pct"] < 0.5 * SPEC["agree_llt_max_pct"]:
            st["q_mult"] = max(st.get("q_mult", 1.0) * 0.9, 1.0)
        hist["q_mult"] = st["q_mult"]

    st.setdefault("history", []).append(hist)
    print(f"[al] round {rnd} check: {hist}")

    if (agree and ok_specs >= 3) or rnd >= SPEC["max_rounds"]:
        st["stage"] = "DONE"
        print("[al] LOOP CONVERGED" if agree else "[al] hard cap reached")
    else:
        st["round"] = rnd + 1
        st["stage"] = "TRAIN"


STAGES = {"TRAIN": stage_train, "OPTIMIZE": stage_optimize, "SELECT": stage_select,
          "SUBMIT": stage_submit, "WAIT": stage_wait, "INGEST": stage_ingest,
          "CHECK": stage_check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-stages", type=int, default=200)
    args = ap.parse_args()

    if args.reset and os.path.isfile(STATE_PATH):
        os.remove(STATE_PATH)
    st = load_state()
    for _ in range(args.max_stages):
        if st["stage"] == "DONE":
            print("[al] done. history:")
            for h in st["history"]:
                print("  ", h)
            break
        print(f"\n[al] === round {st['round']} / stage {st['stage']} ===")
        STAGES[st["stage"]](st)
        save_state(st)


if __name__ == "__main__":
    main()
