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
  - V/(4 f N Ae_effective) <= b_limit (기본 1.2T)
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
from optimization.design_summary import design_analytical_b_field_t  # noqa: E402
from model_targets import (  # noqa: E402
    SURROGATE_TEMPERATURE_TARGETS,
    SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
)


DEFAULT_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,          # +-2%
    "T_limit_C": 97.0,   # 실스펙 100C - eighth 열모델의 핫스팟 과소평가 2~3C 보상 (게이트1 실측)
    "B_limit_T": 1.2,
    # The supplied 1K101 UU137 approval guarantees kf >= 0.85.  Its nominal
    # Net Area is 13.2 cm2 for a 22*70=15.4 cm2 gross section (0.857...).
    # Use the guaranteed minimum conservatively with decoded gross Ae_m2.
    "core_lamination_factor": 0.85,
    "core_lamination_factor_source": "1K101_UU137_approval_p6_minimum",
    "B_area_basis": "gross_geometry_times_lamination_factor",
    "insulation_min_mm": 40.0,
    "q_sigma": 1.28,             # 불확실성 조임 계수 (컨포멀 보정으로 대체됨)
    "knn_quantile_gate": None,   # 학습셋 k-NN 거리 임계 (models 준비 시 설정)
}

# 온도 타겟 (프로브 시트 기준)
T_TARGETS = list(SURROGATE_TEMPERATURE_TARGETS)
POST_TEMPERATURE_CONSTRAINT = 1 + len(T_TARGETS)
N_IEQ_CONSTRAINTS = POST_TEMPERATURE_CONSTRAINT + 5
CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{target}" for target in T_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
)
if len(CONSTRAINT_NAMES) != N_IEQ_CONSTRAINTS:
    raise RuntimeError("NSGA constraint name/schema length mismatch")

# NSGA keeps the append-only Sobol chromosome for warm-start compatibility,
# but these physical dimensions are fixed in both its bounds and decoder.
# Pads remain separate physical layers; they are not included in plate
# thickness and stay at the established 2 mm contract value.
NSGA_FIXED_THERMAL_STACK_MM = {
    "core_plate_t": 20.0,
    "wcp_t": 20.0,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}


class MFTProblem(Problem):
    """
    models: dict[target] -> predictor with .predict_mu_sigma(X_df) -> (mu, sigma)
            필요 타겟: "Llt_phys", "P_winding_total", 권선 구성요소 손실,
                       "P_core_total", "P_core_plate_total", "P_wcp_total",
                       T_TARGETS...
    density_gate: callable(X_features_df) -> ndarray (양수 = 위반량) 또는 None
    """

    def __init__(self, models, spec=None, density_gate=None, fixed_overrides=None):
        required = {
            "Llt_phys", "P_winding_total", "P_core_total",
            "P_core_plate_total", "P_wcp_total", *T_TARGETS,
            *SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
        }
        missing = sorted(required.difference(models))
        if missing:
            raise ValueError(f"required surrogate models are missing: {missing}")
        if density_gate is None:
            raise ValueError("a strict-full density gate is required")
        self.models = models
        self.spec = dict(DEFAULT_SPEC, **(spec or {}))
        self.density_gate = density_gate
        self.constraint_names = CONSTRAINT_NAMES
        supplied_overrides = dict(fixed_overrides or {})
        for name, expected in NSGA_FIXED_THERMAL_STACK_MM.items():
            if name in supplied_overrides and not np.isclose(
                float(supplied_overrides[name]), expected,
                rtol=0.0, atol=1e-12,
            ):
                raise ValueError(
                    f"NSGA fixes {name}={expected:g} mm; conflicting "
                    "override was supplied"
                )
        supplied_overrides.update(NSGA_FIXED_THERMAL_STACK_MM)
        self.fixed_overrides = supplied_overrides
        n_var = len(_SOBOL_DIMS)
        lower = np.zeros(n_var)
        upper = np.ones(n_var)
        for name in ("core_plate_t", "wcp_t"):
            index = next(
                i for i, (dimension, _, _) in enumerate(_SOBOL_DIMS)
                if dimension == name
            )
            _, physical_lower, physical_upper = _SOBOL_DIMS[index]
            unit_value = (
                (NSGA_FIXED_THERMAL_STACK_MM[name] - physical_lower)
                / (physical_upper - physical_lower)
            )
            lower[index] = unit_value
            upper[index] = unit_value
        # Llt(1) + all temperature targets + B/shrink/gap/density/disagreement(5).
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=N_IEQ_CONSTRAINTS,
                         xl=lower, xu=upper)

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
        G = np.full((n, self.n_ieq_constr), BIG)

        idx = np.where(valid)[0]
        if len(idx):
            sub = frame.iloc[idx]

            # 목적 1: 부피
            vols = np.array([bounding_box_lit(sub.iloc[j])[0] for j in range(len(sub))])

            # 서러게이트 예측
            mu_llt, sg_llt = self._predict("Llt_phys", sub)
            mu_pw, sg_pw = self._predict("P_winding_total", sub)
            mu_pc, sg_pc = self._predict("P_core_total", sub)
            mu_pp, sg_pp = self._predict("P_core_plate_total", sub)
            mu_wcp, sg_wcp = self._predict("P_wcp_total", sub)

            total_loss = mu_pw + mu_pc + mu_pp + mu_wcp
            F[idx, 0] = vols
            F[idx, 1] = total_loss

            q = spec["q_sigma"]
            # g0: 누설 밴드 (불확실성 조임)
            G[idx, 0] = np.abs(mu_llt - spec["Llt_target_uH"]) + q * sg_llt - spec["Llt_tol_uH"]
            # g1..: every independently trained temperature target.
            for t_i, t_name in enumerate(T_TARGETS):
                mu_t, sg_t = self._predict(t_name, sub)
                G[idx, 1 + t_i] = mu_t + q * sg_t - spec["T_limit_C"]
            post_temperature = POST_TEMPERATURE_CONSTRAINT
            # Bulk volt-second design B.  A pointwise mesh/edge B_max is a
            # diagnostic only and must not reject an otherwise valid design.
            try:
                design_b = np.asarray([
                    design_analytical_b_field_t(
                        sub.iloc[row_index],
                        core_lamination_factor=spec["core_lamination_factor"],
                        area_basis=spec["B_area_basis"],
                    )
                    for row_index in range(len(sub))
                ], dtype=float)
                G[idx, post_temperature] = design_b - spec["B_limit_T"]
            except (KeyError, TypeError, ValueError, OverflowError):
                G[idx, post_temperature] = BIG
            # g6: 간격 비례축소 필요량 (절연 하한 불가침 위반)
            G[idx, post_temperature + 1] = shrink[idx]
            # g7: z방향 절연 - 2차 권선 상하단과 요크 간격 (2차-코어 절연쌍의 z성분)
            if "h_gap2" in sub.columns:
                G[idx, post_temperature + 2] = (
                    spec["insulation_min_mm"] - sub["h_gap2"].to_numpy(dtype=float)
                )
            else:
                G[idx, post_temperature + 2] = BIG
            # g8: 데이터 밀도 게이트 (외삽 봉쇄)
            G[idx, post_temperature + 3] = self.density_gate(sub)
            # g9: 앙상블 불일치 게이트 - Llt 예측기들의 원공간 폭이 밴드 전폭을 넘으면 신뢰 불가
            try:
                dis = self.models["Llt_phys"].disagreement(sub)
                G[idx, post_temperature + 4] = dis - 2.0 * spec["Llt_tol_uH"]
            except Exception:
                G[idx, post_temperature + 4] = BIG

        out["F"] = F
        out["G"] = G
        # 인필 선정용 아카이브 데이터
        out["frame"] = frame
