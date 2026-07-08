"""
MFT 최적설계 NSGA-2 문제 정의 (pymoo).

핵심 설계 (서러게이트 착취 방지 4중 구조 중 옵티마이저 측 3개):
  1. 불확실성 조임 제약: 예측 μ에 보정된 반폭 w(x)=q·σ(x)를 더해 제약 판정
  2. 데이터 밀도 게이트: 학습셋 k-NN 거리 초과 후보는 제약 위반 (외삽 봉쇄)
  3. 유전자 = 샘플러의 20차원 단위 파라미터화 (decode_unit_sample 공유)
     -> 학습/최적화 스키마 불일치 원천 제거, 후보가 곧 --params JSON

스펙 (config에서 주입):
  - |2 x Llt_pred(대칭) - 27.5uH| + q*sigma <= 0.55uH   (Llt_phys 타겟이면 x2 불필요)
  - T_c + q*sigma_T <= 100C (부품 4종)
  - B_max + q*sigma <= b_limit (기본 1.2T)
  - 모든 권선 간격 >= 40mm (디코드 하한 + shrink 금지로 불가침)
  - 목적: f1 = 외곽 박스 부피 [L], f2 = 총손실 [W]
"""
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from module.input_parameter_260706 import (  # noqa: E402
    KEYS, _SOBOL_DIMS, unit_to_dims, decode_unit_sample, create_input_parameter, validation_check,
)
from optimization.geometry_metrics import bounding_box_lit  # noqa: E402


DEFAULT_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,          # +-2%
    "T_limit_C": 97.0,   # 실스펙 100C - eighth 열모델의 핫스팟 과소평가 2~3C 보상 (게이트1 실측)
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.28,             # 불확실성 조임 계수 (컨포멀 보정으로 대체됨)
    "knn_quantile_gate": None,   # 학습셋 k-NN 거리 임계 (models 준비 시 설정)
}

# 온도 타겟 (프로브 시트 기준)
T_TARGETS = ["Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
             "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max"]


class MFTProblem(Problem):
    """
    models: dict[target] -> predictor with .predict_mu_sigma(X_df) -> (mu, sigma)
            필요 타겟: "Llt_phys", "P_winding_total", "P_core_total", "P_core_plate_total",
                       "B_max_core", T_TARGETS...
    density_gate: callable(X_features_df) -> ndarray (양수 = 위반량) 또는 None
    """

    def __init__(self, models, spec=None, density_gate=None, fixed_overrides=None):
        self.models = models
        self.spec = dict(DEFAULT_SPEC, **(spec or {}))
        self.density_gate = density_gate
        self.fixed_overrides = fixed_overrides or {}
        n_var = len(_SOBOL_DIMS)
        # 제약: Llt band(1) + T(4) + B(1) + shrink(1) + h_gap2(1) + density(1) + 앙상블불일치(1)
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=10,
                         xl=np.zeros(n_var), xu=np.ones(n_var))

    # ---- 배치 디코드: 단위 유전자 -> 파생 포함 특징 프레임 ----
    def decode_batch(self, X_unit):
        rows = []
        shrink = np.zeros(len(X_unit))
        valid = np.ones(len(X_unit), dtype=bool)
        for i, u in enumerate(X_unit):
            s = unit_to_dims(u)
            v = decode_unit_sample(s, allow_space_shrink=False,
                                   space_min=self.spec["insulation_min_mm"])
            shrink[i] = v.pop("_space_shrink_needed", 0.0)
            v.update(self.fixed_overrides)
            try:
                df = create_input_parameter({k: v[k] for k in KEYS if k in v})
                ok, dfp = validation_check(df, strict=False)
                valid[i] = bool(ok)
                rows.append(dfp.iloc[0])
            except Exception:
                valid[i] = False
                rows.append(pd.Series(dtype=float))
        frame = pd.DataFrame(rows).reset_index(drop=True)
        return frame, shrink, valid

    def _predict(self, target, X):
        mu, sigma = self.models[target].predict_mu_sigma(X)
        return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)

    def _evaluate(self, X, out, *args, **kwargs):
        spec = self.spec
        n = len(X)
        BIG = 1e6

        frame, shrink, valid = self.decode_batch(X)

        F = np.full((n, 2), BIG)
        G = np.full((n, 10), BIG)

        idx = np.where(valid)[0]
        if len(idx):
            sub = frame.iloc[idx]

            # 목적 1: 부피
            vols = np.array([bounding_box_lit(sub.iloc[j])[0] for j in range(len(sub))])

            # 서러게이트 예측
            mu_llt, sg_llt = self._predict("Llt_phys", sub)
            mu_pw, sg_pw = self._predict("P_winding_total", sub)
            mu_pc, sg_pc = self._predict("P_core_total", sub)
            mu_pp, sg_pp = (self._predict("P_core_plate_total", sub)
                            if "P_core_plate_total" in self.models else (np.zeros(len(sub)),) * 2)
            mu_b, sg_b = self._predict("B_max_core", sub)

            total_loss = mu_pw + mu_pc + mu_pp
            F[idx, 0] = vols
            F[idx, 1] = total_loss

            q = spec["q_sigma"]
            # g0: 누설 밴드 (불확실성 조임)
            G[idx, 0] = np.abs(mu_llt - spec["Llt_target_uH"]) + q * sg_llt - spec["Llt_tol_uH"]
            # g1..g4: 온도 4종
            for t_i, t_name in enumerate(T_TARGETS):
                if t_name in self.models:
                    mu_t, sg_t = self._predict(t_name, sub)
                    G[idx, 1 + t_i] = mu_t + q * sg_t - spec["T_limit_C"]
                else:
                    G[idx, 1 + t_i] = -1.0  # 모델 없으면 비활성 (파일럿 단계)
            # g5: B 한계
            G[idx, 5] = mu_b + q * sg_b - spec["B_limit_T"]
            # g6: 간격 비례축소 필요량 (절연 하한 불가침 위반)
            G[idx, 6] = shrink[idx]
            # g7: z방향 절연 - 2차 권선 상하단과 요크 간격 (2차-코어 절연쌍의 z성분)
            if "h_gap2" in sub.columns:
                G[idx, 7] = spec["insulation_min_mm"] - sub["h_gap2"].to_numpy(dtype=float)
            else:
                G[idx, 7] = -1.0
            # g8: 데이터 밀도 게이트 (외삽 봉쇄)
            if self.density_gate is not None:
                G[idx, 8] = self.density_gate(sub)
            else:
                G[idx, 8] = -1.0
            # g9: 앙상블 불일치 게이트 - Llt 예측기들의 원공간 폭이 밴드 전폭을 넘으면 신뢰 불가
            try:
                dis = self.models["Llt_phys"].disagreement(sub)
                G[idx, 9] = dis - 2.0 * spec["Llt_tol_uH"]
            except Exception:
                G[idx, 9] = -1.0

        out["F"] = F
        out["G"] = G
        # 인필 선정용 아카이브 데이터
        out["frame"] = frame
