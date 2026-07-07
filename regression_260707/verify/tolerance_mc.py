"""
게이트5(d): 최종 설계의 제작 공차 몬테카를로 (서러게이트 기반).

최종 후보 파라미터 주변에서 치수 공차를 섭동해 1만회 예측:
  - 간격류 ±0.5mm, 도체 폭 ±0.05mm(2차) / ±0.1mm(1차), 창/코어 치수 ±0.5mm
  - (참고) 코어 적층계수/µr는 Lm에 주로 작용 - Llt는 기하 지배라 치수만으로 1차 평가
산출: Llt ±2% 밴드 이탈 확률, 100C 초과 확률, 민감도 상위 파라미터.

사용: python tolerance_mc.py --params final_candidate.json [--n 10000]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

TOL = {  # 1 sigma 가정 [mm]
    "cc_w2c_space_x": 0.5, "cc_w2c_space_y": 0.5,
    "w2c_w1c_space_x": 0.5, "w2c_w1c_space_y": 0.5,
    "w1c_w2s_space_x": 0.5, "w1s_cs_space_x": 0.5, "cs_w1s_space_y": 0.5,
    "cw1": 0.1, "cw2": 0.05, "gap1": 0.1, "gap2": 0.05,
    "l1": 0.5, "l2": 0.5, "h1": 0.5, "w1": 0.5,
    "nwh1": 0.5, "nwh2": 0.5,
}

SPEC = {"Llt_target": 27.5, "Llt_tol": 0.55, "T_limit": 100.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from module.input_parameter_260706 import create_input_parameter, validation_check, KEYS
    from predictor import EnsemblePredictor

    base = json.load(open(args.params, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    rows = []
    n_invalid = 0
    for _ in range(args.n):
        p = dict(base)
        for k, s in TOL.items():
            if k in p:
                p[k] = float(p[k]) + rng.normal(0, s)
        try:
            ok, dfp = validation_check(create_input_parameter({k: v for k, v in p.items() if k in KEYS}))
            if not ok:
                n_invalid += 1
                continue
            rows.append(dfp.iloc[0])
        except Exception:
            n_invalid += 1
    X = pd.DataFrame(rows).reset_index(drop=True)
    print(f"유효 섭동 {len(X)}/{args.n} (기하 불성립 {n_invalid})")

    models = {}
    for t in ["Llt_phys", "Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
              "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max"]:
        try:
            models[t] = EnsemblePredictor.load(t)
        except Exception:
            pass

    mu, _ = models["Llt_phys"].predict_mu_sigma(X, conformal=False)
    out_band = np.abs(mu - SPEC["Llt_target"]) > SPEC["Llt_tol"]
    print(f"\nLlt 분포: 평균 {mu.mean():.2f} uH, std {mu.std():.3f} uH")
    print(f"±2% 밴드({SPEC['Llt_target']}±{SPEC['Llt_tol']}) 이탈 확률: {out_band.mean()*100:.2f}%")

    t_over = np.zeros(len(X), dtype=bool)
    for t in models:
        if not t.startswith("Tprobe_"):
            continue
        mt, _ = models[t].predict_mu_sigma(X, conformal=False)
        t_over |= mt > SPEC["T_limit"]
        print(f"{t}: 평균 {mt.mean():.1f}C, 100C 초과 {(mt > SPEC['T_limit']).mean()*100:.2f}%")
    print(f"어느 부품이든 100C 초과 확률: {t_over.mean()*100:.2f}%")

    # 민감도: |dLlt/d(param)| 근사 (섭동-예측 상관)
    print("\nLlt 민감도 상위 (표준화 회귀계수):")
    sens = {}
    for k in TOL:
        if k in X.columns and X[k].std() > 0:
            sens[k] = abs(np.corrcoef(X[k], mu)[0, 1]) * mu.std() / X[k].std() * TOL[k]
    for k, v in sorted(sens.items(), key=lambda x: -x[1])[:6]:
        print(f"  {k:20s} ±{TOL[k]}mm -> ±{v:.3f} uH")


if __name__ == "__main__":
    main()
