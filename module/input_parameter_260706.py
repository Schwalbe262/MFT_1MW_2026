import math
import os

import numpy as np
import pandas as pd

from module.core_material_contract import (
    AREA_BASIS_GROSS_HOMOGENEOUS,
    PHYSICS_DATA_REVISION,
    build_core_material_contract_fields,
    effective_area_m2,
    geometry_volume_and_masses,
)


N1_MIN_TURNS = 5
N1_MAX_TURNS = 8
PRIMARY_CONDUCTOR_MAX_THICKNESS_MM = 10.0
COLD_PLATE_MIN_T_MM = 10.0
COLD_PLATE_MAX_T_MM = 30.0
WCP_LENGTH_MIN_PCT = 20.0
WCP_LENGTH_MAX_PCT = 80.0


# 설계도면260706.pdf 기반 파라미터 스키마.
# 기존 input_parameter.py의 ratio 방식(wff/window_ratio) 대신
# 물리 치수(도체 두께/간격/각종 space)를 직접 입력한다.
KEYS = [
    "N1_main", "N1_side", "N2_main", "N2_side",
    "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_on",
    "cw1", "gap1", "cw2", "gap2",
    "nwh1", "nwh2",
    "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y",
    "w1c_w2s_space_x",
    "w2s_w1s_space_x", "w1s_w2s_space_y",
    "w1s_cs_space_x", "cs_w1s_space_y",
    "wcp_t", "wcp_pad_t", "wcp_len_x", "wcp_on",
    "core_plate_pad_t",
    "core_depth_min", "core_depth_max",
    "round_corner", "corner_radius", "corner_segments",
    "full_model",
    "max_passes", "percent_error", "min_converged",
    "matrix_percent_error", "matrix_max_passes", "matrix_min_converged",
    "freq", "V1_rms", "I1_rated", "I2_rated", "I2_phase_deg",
    "P_target", "V2_rms",
    "core_cm", "core_x", "core_y",
    "matrix_on", "loss_on", "thermal_on",
    "plate_temp", "air_temp", "fan_velocity",
    "k_ins", "core_k_thermal", "n_explicit_turns", "rx_mesh_mode",
    "keep_project",
    "loss_sym_on", "thermal_symmetry", "matrix_skin_mesh", "fan_config", "loss_from_copy",
    "thermal_max_iterations", "conductor_temp_C",
]

# Material constants are explicit input/result columns but deliberately stay
# outside KEYS.  KEYS is the sealed Sobol/campaign identity schema; adding a
# fixed constant there would rewrite existing candidate hashes without changing
# their geometry.  The pinned solver revision plus these echoed columns binds
# the material policy unambiguously.
CORE_MATERIAL_INPUT_KEYS = (
    "core_lamination_factor",
    "core_loss_margin",
)
# Wound-core thermal material controls are accepted and echoed by fixed runs,
# but deliberately stay outside KEYS.  The thermal material policy must not
# rewrite the sealed Sobol/campaign identity for an unchanged geometry.
THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS = (
    "core_k_anisotropic",
    "core_k_alloy",
    "core_k_interlayer",
)
# Fixed-run-only controls stay outside KEYS so experiments do not rewrite the
# sealed Sobol/campaign candidate identity. They are still echoed in results.
EFFICIENCY_EXPERIMENT_INPUT_KEYS = (
    "thermal_rx_side_block_mesh_level",
)
# Optional electrostatic-stage controls are accepted and echoed by fixed runs,
# but deliberately stay outside KEYS.  KEYS is the sealed campaign identity;
# enabling an A/B diagnostic stage must not change the underlying candidate.
ELECTROSTATIC_STAGE_INPUT_KEYS = (
    "cap_on",
    "cap_max_passes",
    "cap_percent_error",
)
# Solver-contract metadata is accepted and echoed by fixed runs, but is not a
# Sobol coordinate or an AEDT numeric design variable.
PHYSICS_METADATA_INPUT_KEYS = (
    "physics_data_revision",
)
# The exact full-input contract emitted by e30c070 remains authenticated —
# it predates both the electrostatic-stage and wound-core-conductivity
# extensions (the two names alias the same 75-key contract). Candidate
# digests continue to project onto KEYS for every supported schema.
PRE_ELECTROSTATIC_INPUT_KEYS = PRE_ANISOTROPIC_CORE_K_INPUT_KEYS = [
    *KEYS,
    *CORE_MATERIAL_INPUT_KEYS,
    *EFFICIENCY_EXPERIMENT_INPUT_KEYS,
    *PHYSICS_METADATA_INPUT_KEYS,
]
ALL_INPUT_KEYS = [
    *KEYS,
    *CORE_MATERIAL_INPUT_KEYS,
    *THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS,
    *EFFICIENCY_EXPERIMENT_INPUT_KEYS,
    *ELECTROSTATIC_STAGE_INPUT_KEYS,
    *PHYSICS_METADATA_INPUT_KEYS,
]

# Candidate authentication remains fail-closed: sealed fronts use KEYS, while
# current full-input candidates use the complete contract. Candidate digests
# still project onto KEYS.
SUPPORTED_CANDIDATE_INPUT_SCHEMAS = frozenset({
    frozenset(KEYS),
    frozenset(PRE_ANISOTROPIC_CORE_K_INPUT_KEYS),
    frozenset(ALL_INPUT_KEYS),
})


def get_drawing_default_params():
    """설계도면260706.pdf 기준 치수 (fixed 모드 기본값)"""
    return {
        # 턴수 (1차 6턴 전부 중심, 2차 60턴 = 중심 18 + 측면 42)
        "N1_main": 6, "N1_side": 0, "N2_main": 18, "N2_side": 42,
        # 코어: 829 x 525 (x,z), 전체 깊이 530. 알루미늄 plate와
        # 양면 2T thermal pad는 별도 두께로 깊이 예산에 반영한다.
        "l1": 89.0, "l2": 236.5, "h1": 347.0, "w1": 530.0,
        "n_core_group": 3, "core_plate_t": 20.0, "core_plate_on": 1,
        # 도체: 1차 5mm/1.6mm, 2차 0.665mm/0.339mm
        "cw1": 5.0, "gap1": 1.6, "cw2": 0.665, "gap2": 0.339,
        # 권선 높이
        "nwh1": 284.5, "nwh2": 284.5,
        # 간격 (y방향은 콜드플레이트 포함 외곽(w1=530) 기준)
        "cc_w2c_space_x": 35.7, "cc_w2c_space_y": 30.0,
        "w2c_w1c_space_x": 35.7, "w2c_w1c_space_y": 30.0,
        # 1차-2차측면 최소 요구 간격 (지오메트리는 측면 레그 기준으로 배치되므로 검증용)
        "w1c_w2s_space_x": 33.1,
        # 1차 측면 권선용 간격 (N1_side=0이라 이 설계에서는 미사용, 수식 호환용)
        "w2s_w1s_space_x": 0.0, "w1s_w2s_space_y": 0.0,
        # 측면 레그 - 2차 측면 권선 간격
        "w1s_cs_space_x": 30.1, "cs_w1s_space_y": 75.6,
        # 권선 냉각 플레이트: 알루미늄 20T + 양면 서멀패드 각 2T.
        # x방향 폭은 도면 미기재로 파라미터화 (기본 2*l1)
        # Length percentage is referenced to the clear x-direction straight
        # span inside the innermost Tx_main turn.  The physical geometry still
        # consumes the resolved millimetre value ``wcp_len_x``.
        "wcp_t": 20.0, "wcp_pad_t": 2.0,
        "wcp_len_x": 178.0, "wcp_on": 1,
        # 코어 콜드플레이트: 알루미늄 20T + 양면 서멀패드 각 2T.
        "core_plate_pad_t": 2.0,
        # 코어 1조 깊이 허용 범위 [mm] (랜덤 모드에서 n_core_group 샘플링 제약.
        # 도면 설계(150mm/조)는 범위 밖이지만 fixed 모드에서는 경고만 하고 통과)
        "core_depth_min": 60.0, "core_depth_max": 120.0,
        # 모서리 라운드: off, 안쪽 턴 반경 10mm
        # corner_segments: 코너 등각 분할 수 (모든 턴 동일 점 개수 -> 균일한 표시,
        # 많을수록 메시/해석 부담 증가). 0이면 진짜 원호
        "round_corner": 0, "corner_radius": 10.0, "corner_segments": 4,
        # 1 이면 대칭(1/8 분할) 미적용 풀모델로 모델링/해석
        "full_model": 0,
        # 해석 수렴 설정
        "max_passes": 10, "percent_error": 1.5, "min_converged": 2,
        # Skin-free matrix still needs enough adaptive passes: an 8-pass random
        # run stopped at 13.254% energy error despite a 0.166% delta-energy value.
        "matrix_percent_error": 1.5, "matrix_max_passes": 20, "matrix_min_converged": 1,
        # 여자/정격 (도면 설계: 1차 1000Vrms / 1000Arms, 2차 10kVrms / 100Arms, 1kHz)
        "freq": 1000.0,          # [Hz]
        "V1_rms": 1000.0,        # 1차 전압원 [Vrms] (loss 디자인 Tx 여자)
        "I1_rated": 1000.0,      # 1차 정격 전류 [Arms] (matrix 디자인 여자)
        "I2_rated": 100.0,       # 2차 정격 전류 [Arms]
        "I2_phase_deg": 0.0,     # loss 디자인 Rx 전류 위상 [deg] (P_target>0이면 자동 계산값이 우선)
        # 목표 정격 전력 [W]: >0 이면 design1의 누설 Lk로 DAB 운전 위상 phi를 역산해
        # I2 위상(-phi/2)을 자동 주입. 0이면 I2_phase_deg 수동값 사용
        "P_target": 0.0,
        "V2_rms": 10000.0,       # 2차 전압 [Vrms] (위상 역산용)
        # 코어손실 계수 (2605SA1: P[W/kg]=6.5 f(kHz)^1.51 B^1.74, 밀도 7180kg/m3
        #  -> ANSYS Power Ferrite (W/m3, Hz): cm = 6.5*7180/1000^1.51 = 1.377)
        "core_cm": 1.377, "core_x": 1.51, "core_y": 1.74,
        # UU137 approval-sheet guaranteed minimum.  ``core_cm`` remains the
        # traceable POWERLITE base coefficient; validation derives the gross-
        # geometry coefficient assigned to Maxwell.
        "core_lamination_factor": 0.85,
        "core_loss_margin": 1.15,
        # Bind fixed-input payloads to the exact physics policy implemented by
        # this checkout. The pin is owned by core_material_contract.py.
        "physics_data_revision": PHYSICS_DATA_REVISION,
        # 디자인 활성화
        "matrix_on": 1,          # design1: L/k 매트릭스 (전류원)
        "loss_on": 1,            # design2: 손실 원샷 (Tx 전압원 + Rx 전류원 + 코어손실)
        "thermal_on": 0,         # design3: Icepak 열해석
        # Optional Maxwell Electrostatic capacitance/resonance screening stage.
        # Default ON for the durable v3.2 dataset: the cluster wall-time A/B
        # (task 31070) measured +53 s per design (~+0.9% of a full pipeline).
        "cap_on": 1,
        "cap_max_passes": 10,
        "cap_percent_error": 1.0,
        # 열해석 조건
        "plate_temp": 50.0,      # 콜드플레이트 고정온도 [cel]
        "air_temp": 50.0,        # 팬 흡입공기/주변 온도 [cel]
        "fan_velocity": 1.5,     # 팬 유속 [m/s], +y -> -y
        "k_ins": 0.2,            # 권선 절연 열전도율 [W/mK]
        "core_k_thermal": 2.0,   # 코어 등가 열전도율 [W/mK] (아몰퍼스, 보수적 등방값)
        # Wound 2605SA1 ribbon: width is y everywhere and the stack normal is
        # radial in the xz magnetic loop.  Icepak derives the directional
        # conductivities at runtime from these anchors and the lamination factor.
        "core_k_anisotropic": 1,
        "core_k_alloy": 9.0,       # Metglas 2605SA1 [W/mK]
        "core_k_interlayer": 0.2,  # epoxy/varnish convention [W/mK]
        # Production thermal model: represent the complete Rx pack with
        # anisotropic blocks. Thin explicit foils can disappear from Icepak's
        # cut-cell mesh even when an object mesh level is assigned.
        "n_explicit_turns": 0,
        # Rx foil 메시 전략: "skin"(기본) / "length" / "length-coarse" (벤치마크용)
        "rx_mesh_mode": "skin",
        # 완료 후 프로젝트 파일 보존 여부 (fixed 기본 보존 / 랜덤·클러스터는 0으로 확실히 삭제)
        "keep_project": 1,
        # 손실 디자인 대칭화: 1이면 대칭 1/8 + 전류 여자 (캠페인용, 시간 ~4x 단축).
        # 0이면 풀모델 + 전압원 (최종 검증용). 추출값은 양쪽 모두 _phys(실물 기준)로 보정 기록
        "loss_sym_on": 1,
        # 열해석 대칭화: "eighth" = 1/8 (양측 팬 y대칭 + 부력 무시 가정, 캠페인용) / "full" = 최종 검증용
        "thermal_symmetry": "eighth",
        # Inductance-only matrix: stranded windings, plate eddy off, no skin mesh.
        # The copied loss design restores solid windings and all skin operations.
        "matrix_skin_mesh": 0,
        # loss 디자인을 matrix 복제로 생성 (모델링 1회분 절약, MFT_TAB 패턴)
        "loss_from_copy": 1,
        # 풀 열해석 팬 구성: "dual" = +-y 양측 유입(냉각 스펙, 1/8과 동일 물리) / "single" = +y->-y
        "fan_config": "dual",
        # Icepak 최대 iteration (수렴 판정 기준 미달 시 상한) - iteration 배터리 테스트로 캠페인 값 결정
        "thermal_max_iterations": 250,
        # A/B-only Icepak control. Level 5 is the unchanged production default;
        # level 4 is the guarded multi-turn side-pack candidate.
        "thermal_rx_side_block_mesh_level": 5,
        # 권선 도체의 운전 온도 기준 [C]: EM 도전율을 이 온도의 구리로 설정
        # (20C 기준이면 실물(~80-100C) 권선손실 ~25% 과소평가 - 손실/온도 라벨 현실화)
        "conductor_temp_C": 80.0,
    }


def get_random_value(lower=None, upper=None, resolution=None):

    if isinstance(resolution, float):
        str_res = f"{resolution:.16f}".rstrip('0').rstrip('.')
        if "." in str_res:
            precision = len(str_res.split('.')[-1])
        else:
            precision = 0
    else:
        precision = 0

    possible_values = np.arange(lower, upper + resolution * 0.5, resolution)
    if (np.abs(possible_values[-1] - upper) > 1e-8) and (np.abs((possible_values - upper)).min() > 1e-8):
        possible_values = np.append(possible_values, upper)
        possible_values = np.unique(possible_values)
    chosen = np.random.choice(possible_values)
    value = round(chosen, precision)

    if resolution == 1 or resolution == 1.0:
        return int(value)
    return float(value)


def create_input_parameter(param=None):
    """
    param 이 dict  -> 도면 기본값 위에 병합 (부분 지정 가능, fixed 모드)
    param 이 list/DataFrame -> KEYS 순서/컬럼으로 해석
    param 이 None  -> 랜덤 생성 (기존 랜덤 스윕과 유사한 범위)
    """
    if param is not None:

        if isinstance(param, dict):
            unknown = set(param.keys()) - set(ALL_INPUT_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown parameter keys: {sorted(unknown)}. "
                    f"Valid keys: {ALL_INPUT_KEYS}"
                )
            values = get_drawing_default_params()
            values.update(param)
            param_df = pd.DataFrame(
                [[values[k] for k in ALL_INPUT_KEYS]], columns=ALL_INPUT_KEYS
            )

        elif isinstance(param, pd.DataFrame):
            param = param.copy()
            defaults = get_drawing_default_params()
            for key in (
                    *CORE_MATERIAL_INPUT_KEYS,
                    *THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS,
                    *EFFICIENCY_EXPERIMENT_INPUT_KEYS,
                    *ELECTROSTATIC_STAGE_INPUT_KEYS,
                    *PHYSICS_METADATA_INPUT_KEYS):
                if key not in param.columns:
                    param[key] = defaults[key]
            missing = set(KEYS) - set(param.columns)
            if missing:
                raise ValueError(f"Missing parameter columns: {sorted(missing)}")
            param_df = param[ALL_INPUT_KEYS].copy()

        else:
            if isinstance(param, (list, tuple)) and len(param) > 0:
                first = param[0]
                if not isinstance(first, (list, tuple, dict, pd.Series)):
                    param = [param]
            param_df = pd.DataFrame(param, columns=KEYS)

    else:
        param_df = _create_random_parameter_sobol()

    defaults = get_drawing_default_params()
    for key in (
            *CORE_MATERIAL_INPUT_KEYS,
            *THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS,
            *EFFICIENCY_EXPERIMENT_INPUT_KEYS,
            *ELECTROSTATIC_STAGE_INPUT_KEYS,
            *PHYSICS_METADATA_INPUT_KEYS):
        if key not in param_df.columns:
            param_df[key] = defaults[key]

    revision = param_df["physics_data_revision"]
    if not revision.map(lambda value: isinstance(value, str)).all():
        raise ValueError("physics_data_revision must be a string")

    return param_df[ALL_INPUT_KEYS]


# Sobol 시퀀스 상태 (프로세스 내 공유; 프로세스마다 다른 seed로 scramble)
_SOBOL_STATE = {"engine": None}

# Sobol로 뽑는 연속 차원 정의: (키, 하한, 상한)
_SOBOL_DIMS = [
    ("u_N1", 0, 1), ("u_N1_side", 0, 1), ("u_N2_side", 0, 1),
    ("l1", 40, 100), ("total_length", 500, 1200), ("total_height", 500, 1000),
    ("w1", 200, 800), ("u_ngroup", 0, 1),
    # Independent winding/core cold-plate thickness design variables.
    ("wcp_t", COLD_PLATE_MIN_T_MM, COLD_PLATE_MAX_T_MM),
    ("core_plate_t", COLD_PLATE_MIN_T_MM, COLD_PLATE_MAX_T_MM),
    # HV 절연 간격 4쌍: 설계 타겟 40mm 주변 커버리지 (25~70) - 16mm급 극소 간격은
    # 실현 불가 영역이라 데이터 예산 낭비 (사용자 확인 2026-07-07)
    ("cc_w2c_space_x", 25, 70), ("cc_w2c_space_y", 25, 70),
    ("w2c_w1c_space_x", 25, 70), ("w2c_w1c_space_y", 25, 70),
    ("w1c_w2s_space_x", 25, 70), ("w1s_cs_space_x", 25, 70),
    ("cs_w1s_space_y", 25, 70),
    # Tx측면-Rx측면 절연 간격 (N1_side>0일 때만 사용; 0이면 도체가 맞닿아 솔버 에러 - 파일럿에서 발견)
    ("w2s_w1s_space_x", 25, 70), ("w1s_w2s_space_y", 25, 70),
    ("f1_split", 0.25, 0.60),   # 창 예산 중 1차 권선 몫
    ("gap1", 0.3, 5.0), ("gap2", 0.3, 2.0),
    # wh2 상한 0.90: h_gap2(2차-요크 z 간격) = h1(1-wh2)/2 이 절연 타겟(40mm) 근방을 벗어나
    # 극소값이 되지 않게 (h1=500~1000에서 h_gap2 >= 25~50mm 보장)
    ("wh1", 0.8, 0.93), ("wh2", 0.5, 0.90),
    # Append-only Sobol schema extension: preserve every pre-existing unit
    # coordinate while adding the independent winding-plate length variable.
    ("wcp_len_pct", WCP_LENGTH_MIN_PCT, WCP_LENGTH_MAX_PCT),
]


def _sobol_next():
    """Sobol 저불일치 시퀀스에서 다음 점을 뽑아 dict로 반환"""
    from scipy.stats import qmc
    if _SOBOL_STATE["engine"] is None:
        seed = int.from_bytes(os.urandom(4), "little")
        _SOBOL_STATE["engine"] = qmc.Sobol(d=len(_SOBOL_DIMS), scramble=True, seed=seed)
    u = _SOBOL_STATE["engine"].random(1)[0]
    return {k: lo + (hi - lo) * float(ui) for (k, lo, hi), ui in zip(_SOBOL_DIMS, u)}


def unit_to_dims(u):
    """단위 하이퍼큐브 [0,1]^d -> _SOBOL_DIMS 물리 범위 dict (샘플러/NSGA2 공유)"""
    return {k: lo + (hi - lo) * float(ui) for (k, lo, hi), ui in zip(_SOBOL_DIMS, u)}


def _create_random_parameter_sobol():
    """Sobol 시퀀스 + 제약 내장 파라미터화 랜덤 샘플러 (decode_unit_sample 공유 디코드 사용)"""
    s = _sobol_next()
    values = decode_unit_sample(s, allow_space_shrink=True)
    return pd.DataFrame([[values[k] for k in KEYS]], columns=KEYS)


def decode_unit_sample(s, allow_space_shrink=True, space_min=None):
    """
    설계공간 디코드 (샘플러와 NSGA-2가 공유하는 단일 소스):
    _SOBOL_DIMS 값 dict(s) -> 전체 파라미터 dict.

    - 제약 내장: 도체 폭(cw1/cw2)을 "창 예산 - 간격 총합"에서 gap 개수까지 반영해 역산
      -> 창 x방향 배치가 구조적으로 항상 성립
    - allow_space_shrink=False (NSGA-2 모드): 간격 비례축소를 하지 않고
      values["_space_shrink_needed"]에 위반량을 기록 (제약 g로 처리, 절연 하한 불가침 보장)
    - space_min: 간격 하한 강제 (예: 절연 40mm) - s의 간격 값을 하한으로 클램프
    """
    defaults = get_drawing_default_params()
    s = dict(s)
    if space_min is not None:
        for k in ("cc_w2c_space_x", "w2c_w1c_space_x", "w1c_w2s_space_x", "w1s_cs_space_x",
                  "cc_w2c_space_y", "w2c_w1c_space_y", "cs_w1s_space_y",
                  "w2s_w1s_space_x", "w1s_w2s_space_y"):
            if k in s:
                s[k] = max(float(s[k]), space_min)

    N1 = N1_MIN_TURNS + int(
        s["u_N1"] * (N1_MAX_TURNS - N1_MIN_TURNS + 0.9999)
    )                                                       # 5..8
    # N1_side(1차 측면 권선)는 캠페인 설계공간에서 제외:
    # 실제 절연 간격(w2s_w1s 등)을 반영하면 Tx-Rx_side x 간격 예산이 거의 성립하지 않고
    # (파일럿 84% 대량 실패의 원인), 도면 계열도 N1_side=0. u_N1_side 차원은 호환용으로 유지.
    N1_side = 0
    N1_main = N1 - N1_side
    N2 = N1 * 10
    N2_side = round(N2 * (s["u_N2_side"] * 0.8))
    N2_main = N2 - N2_side

    l1 = round(s["l1"])
    l2 = (round(s["total_length"]) - 4 * l1) / 2
    h1 = round(s["total_height"]) - 2 * l1
    w1 = round(s["w1"])

    # 코어 분할 수: 1조 깊이 [core_depth_min, core_depth_max] 제약을 역산해 유효 범위에서 선택
    plate_t = round(float(s["core_plate_t"]), 1)
    wcp_t = round(float(s["wcp_t"]), 1)
    core_stack_t = plate_t + 2.0 * float(defaults["core_plate_pad_t"])
    d_min, d_max = defaults["core_depth_min"], defaults["core_depth_max"]
    n_min = max(1, math.ceil((w1 - core_stack_t) / (d_max + core_stack_t)))
    n_max = max(n_min, math.floor((w1 - core_stack_t) / (d_min + core_stack_t)))
    n_core_group = n_min + int(s["u_ngroup"] * (n_max - n_min + 0.9999))

    # ---- 창 x방향 예산 분배 (제약 내장) ----
    spaces = [s["cc_w2c_space_x"], s["w2c_w1c_space_x"], s["w1c_w2s_space_x"], s["w1s_cs_space_x"]]
    total_space = sum(spaces)
    space_shrink_needed = max(0.0, total_space - 0.45 * l2)
    if total_space > 0.45 * l2:
        if allow_space_shrink:
            # 샘플러 모드: 간격 비례 축소로 권선 예산 확보
            scale = 0.45 * l2 / total_space
            spaces = [sp * scale for sp in spaces]
        # NSGA-2 모드(allow_space_shrink=False): 축소하지 않음 - 절연 하한 불가침.
        # 위반량은 _space_shrink_needed로 반환되어 제약 g로 처리됨
    cc_x, w21_x, minclear_x, w1s_x = spaces

    budget = l2 - sum(spaces)                            # 권선 빌드 총예산
    nwl1 = budget * s["f1_split"]                        # 1차 몫 (main + side 합)
    nwl2_total = budget - nwl1                           # 2차 몫 (main + side 합)

    # 도체 폭 역산: 그룹별 gap 개수까지 정확히 반영해 "빌드 합 = 예산"이 구조적으로 성립
    gap1 = round(s["gap1"], 1)
    gap2 = round(s["gap2"], 3)
    n1_gaps = max(N1_main - 1, 0) + max(N1_side - 1, 0)
    n2_gaps = max(N2_main - 1, 0) + max(N2_side - 1, 0)
    cw1 = (nwl1 - n1_gaps * gap1) / N1 if N1 > 0 else 1.0
    cw2 = (nwl2_total - n2_gaps * gap2) / N2 if N2 > 0 else 0.6
    # 도체 최소 두께 보장: 부족하면 간격을 줄여 재역산
    if cw1 < 1.0 and n1_gaps > 0:
        gap1 = max(0.3, round((nwl1 - 1.0 * N1) / n1_gaps, 1))
        cw1 = (nwl1 - n1_gaps * gap1) / N1
    if cw2 < 0.3 and n2_gaps > 0:
        gap2 = max(0.1, round((nwl2_total - 0.3 * N2) / n2_gaps, 3))
        cw2 = (nwl2_total - n2_gaps * gap2) / N2

    nwh1 = round(h1 * s["wh1"], 1)
    nwh2 = round(h1 * s["wh2"], 1)

    # Innermost Tx_main clear straight span in x.  Random campaign geometry is
    # rectangular (round_corner defaults off), but retain the rounded-corner
    # definition so the conversion remains physically meaningful for callers
    # that opt into rounded turns.
    resolved_cw2 = round(cw2, 3)
    resolved_cc_x = round(cc_x, 1)
    resolved_w21_x = round(w21_x, 1)
    nwl2_main = (
        N2_main * resolved_cw2 + max(N2_main - 1, 0) * gap2
        if N2_main > 0 else 0.0
    )
    sl2_main_x = 2.0 * l1 + 2.0 * resolved_cc_x
    sl1_main_x = sl2_main_x + 2.0 * nwl2_main + 2.0 * resolved_w21_x
    round_corner = int(defaults["round_corner"]) != 0
    corner_radius = float(defaults["corner_radius"]) if round_corner else 0.0
    wcp_len_ref_x = sl1_main_x - 2.0 * corner_radius
    wcp_len_pct = round(float(s["wcp_len_pct"]), 1)
    wcp_len_x = round(wcp_len_ref_x * wcp_len_pct / 100.0, 1)

    values = dict(defaults)
    values.update({
        "N1_main": N1_main, "N1_side": N1_side, "N2_main": N2_main, "N2_side": N2_side,
        "l1": l1, "l2": l2, "h1": h1, "w1": w1,
        "n_core_group": n_core_group,
        "core_plate_t": plate_t,
        "wcp_t": wcp_t,
        "wcp_len_x": wcp_len_x,
        "cw1": round(cw1, 2), "gap1": gap1, "cw2": round(cw2, 3), "gap2": gap2,
        "nwh1": nwh1, "nwh2": nwh2,
        "cc_w2c_space_x": round(cc_x, 1),
        "cc_w2c_space_y": round(s["cc_w2c_space_y"], 1),
        "w2c_w1c_space_x": round(w21_x, 1),
        "w2c_w1c_space_y": round(s["w2c_w1c_space_y"], 1),
        "w1c_w2s_space_x": round(minclear_x * 0.8, 1),   # 최소 요구 간격은 실제 여유보다 작게
        "w2s_w1s_space_x": float(s["w2s_w1s_space_x"]) if N1_side > 0 else 0.0,
        "w1s_w2s_space_y": float(s["w1s_w2s_space_y"]) if N1_side > 0 else 0.0,
        "w1s_cs_space_x": round(w1s_x, 1),
        "cs_w1s_space_y": round(s["cs_w1s_space_y"], 1),
        # 랜덤 스윕은 기존처럼 매트릭스(L/k) 전용 - 손실/열해석은 fixed 모드에서
        "loss_on": 0,
        "thermal_on": 0,
        # 클러스터 스윕: 저장공간 확보를 위해 완료 즉시 삭제
        "keep_project": 0,
    })
    values["_space_shrink_needed"] = space_shrink_needed
    return values


def sym_cut_count(obj_name, df):
    """
    1/8 대칭 분할(x=0, y=0, z=0) 시 원형 오브젝트를 지나는 절단 평면 수 c.
    보유 체적 분율 = 1/2^c. 실물 환산: EMLoss x 2^c/4, CoreLoss x 2^c/2^core_y, B x 1/2.
    """
    name = obj_name
    if name.startswith("Tx_main_wcp"):
        return 2  # 냉각판: x,z 절단 (y는 한쪽에만 존재)
    if name.startswith(("Tx_main", "Rx_main")):
        return 3  # 중심 권선 링: x,y,z 모두
    if name.startswith(("Tx_side", "Rx_side")):
        return 2  # 측면 권선 링: y,z (x=0에 안 걸림)

    w1 = float(df["w1"].iloc[0])
    n = int(df["n_core_group"].iloc[0])
    t = float(df["core_plate_t"].iloc[0])
    pad = float(df["core_plate_pad_t"].iloc[0])
    stack_t = t + 2.0 * pad
    d = (w1 - (n + 1) * stack_t) / n

    if name.startswith("core_plate"):
        try:
            i = int(name.split("_")[2])  # core_plate_<i> (1-based)
        except (IndexError, ValueError):
            return 2
        y0 = -w1 / 2 + (i - 1) * (stack_t + d) + pad
        # Each I plate is wholly on one x side, always spans z, and may span y.
        if name.endswith(("_left", "_right")):
            return 2 if (y0 < 0 < y0 + t) else 1
        return 3 if (y0 < 0 < y0 + t) else 2
    if name.startswith("core_"):
        try:
            i = int(name.split("_")[1])  # core_<i> (1-based)
        except (IndexError, ValueError):
            return 2
        y0 = -w1 / 2 + i * stack_t + (i - 1) * d
        return 3 if (y0 < 0 < y0 + d) else 2
    return 3


def get_tx_y_gaps(df):
    """
    1차(Tx) 중심 권선의 y방향 인접 턴 간격 리스트와 냉각판 슬롯 인덱스 계산.

    도면/사용자 확인: 냉각 플레이트는 턴1-턴2 사이와 턴(N-1)-턴N 사이에만 삽입.
    슬롯 폭 = wcp_t + 2*wcp_pad_t (알루미늄과 양면 패드는 별도 두께).

    Returns:
        (y_gaps, slot_indices) - y_gaps 길이는 N1_main-1
    """
    N = int(df["N1_main"].iloc[0])
    gap1 = float(df["gap1"].iloc[0])
    wcp_on = int(df["wcp_on"].iloc[0]) != 0
    slot = (
        float(df["wcp_t"].iloc[0])
        + 2.0 * float(df["wcp_pad_t"].iloc[0])
    )

    y_gaps = [gap1] * max(N - 1, 0)
    slot_indices = []

    if wcp_on and N >= 2:
        y_gaps[0] = slot
        slot_indices.append(0)
        if N >= 3:
            y_gaps[-1] = slot
            slot_indices.append(N - 2)

    return y_gaps, slot_indices


def _cum_positions(start_half, width, gaps):
    pos = [start_half + width * 0.5]
    for g in gaps:
        pos.append(pos[-1] + width + g)
    return pos


def _add_derived_features(inp):
    """회귀학습용 파생 물리 특징량 컬럼 (전부 검증된 파생값 이후에 호출)"""
    l1 = float(inp["l1"].iloc[0]); l2 = float(inp["l2"].iloc[0])
    h1 = float(inp["h1"].iloc[0]); w1 = float(inp["w1"].iloc[0])
    cw1 = float(inp["cw1"].iloc[0]); gap1 = float(inp["gap1"].iloc[0])
    cw2 = float(inp["cw2"].iloc[0]); gap2 = float(inp["gap2"].iloc[0])
    nwh1 = float(inp["nwh1"].iloc[0]); nwh2 = float(inp["nwh2"].iloc[0])
    N1m = int(inp["N1_main"].iloc[0]); N2m = int(inp["N2_main"].iloc[0]); N2s = int(inp["N2_side"].iloc[0])
    d = float(inp["core_depth_each"].iloc[0]); n = int(inp["n_core_group"].iloc[0])

    iron_depth = n * d  # 콜드플레이트 제외 순수 철심 깊이 [mm]
    Ae_gross_m2 = (2 * l1 * 1e-3) * (iron_depth * 1e-3)    # gross center-leg pack area
    face_mm2 = (4 * l1 + 2 * l2) * (h1 + 2 * l1) - 2 * l2 * h1
    core_vol_gross_m3 = face_mm2 * iron_depth * 1e-9
    kf = float(inp["core_lamination_factor"].iloc[0])
    basis = str(inp["core_geometry_material_basis"].iloc[0])
    Ae_effective_m2 = effective_area_m2(
        Ae_gross_m2, kf, area_basis=basis
    )
    (
        core_vol_geometry_m3,
        core_vol_effective_m3,
        core_mass_gross_kg,
        core_mass_effective_kg,
    ) = geometry_volume_and_masses(
        core_vol_gross_m3,
        kf,
        density_kg_m3=float(inp["core_mass_density_kg_m3"].iloc[0]),
        area_basis=basis,
    )

    # Backward-compatible aliases retain their historical gross-geometry
    # meaning. New consumers must use the explicit gross/effective columns.
    inp["Ae_m2"] = [Ae_gross_m2]
    inp["core_vol_m3"] = [core_vol_geometry_m3]
    inp["core_mass_kg"] = [core_mass_gross_kg]
    inp["Ae_gross_m2"] = [Ae_gross_m2]
    inp["Ae_effective_m2"] = [Ae_effective_m2]
    inp["core_vol_gross_m3"] = [core_vol_geometry_m3]
    inp["core_vol_effective_m3"] = [core_vol_effective_m3]
    inp["core_mass_gross_kg"] = [core_mass_gross_kg]
    inp["core_mass_effective_kg"] = [core_mass_effective_kg]
    inp["Ae_m2_basis"] = ["legacy_alias_gross_geometry"]
    inp["core_mass_kg_basis"] = [
        "legacy_alias_gross_geometry_bare_alloy_density"
    ]

    def _cu(group_N, cw, gaps, slx, sly, height):
        if group_N <= 0:
            return 0.0, 0.0
        xs = _cum_positions(slx / 2, cw, gaps)
        ys = _cum_positions(sly / 2, cw, gaps)
        total_len_mm = sum(4 * (x + y) for x, y in zip(xs, ys))
        vol_m3 = total_len_mm * cw * height * 1e-9
        return total_len_mm / group_N, vol_m3 * 8940.0     # MLT[mm], 질량[kg]

    tx_gaps, _ = get_tx_y_gaps(inp)
    mlt_tx, m_tx = _cu(N1m, cw1, [gap1] * (N1m - 1),
                       float(inp["sl1_main_x"].iloc[0]), float(inp["sl1_main_y"].iloc[0]), nwh1)
    # Tx y방향 슬롯 반영 (y만 벌어짐 - 근사로 x/y 평균에 슬롯 포함)
    mlt_rxm, m_rxm = _cu(N2m, cw2, [gap2] * (N2m - 1),
                         float(inp["sl2_main_x"].iloc[0]), float(inp["sl2_main_y"].iloc[0]), nwh2)
    mlt_rxs, m_rxs = _cu(N2s, cw2, [gap2] * (N2s - 1),
                         float(inp["sl2_side_x"].iloc[0]), float(inp["sl2_side_y"].iloc[0]), nwh2)
    inp["MLT_Tx_mm"] = [mlt_tx]
    inp["MLT_Rx_main_mm"] = [mlt_rxm]
    inp["MLT_Rx_side_mm"] = [mlt_rxs]
    inp["cu_mass_Tx_kg"] = [m_tx]
    inp["cu_mass_Rx_main_kg"] = [m_rxm]
    inp["cu_mass_Rx_side_kg"] = [m_rxs * 2]                # 측면 링 2개 (실물 기준)
    inp["cu_mass_total_kg"] = [m_tx + m_rxm + m_rxs * 2]

    # 창 활용률/종횡비
    center_stack = (float(inp["cc_w2c_space_x"].iloc[0]) + float(inp["nwl2_main"].iloc[0])
                    + float(inp["w2c_w1c_space_x"].iloc[0]) + float(inp["nwl1_main"].iloc[0]))
    side_stack = float(inp["w1s_cs_space_x"].iloc[0]) + float(inp["nwl2_side"].iloc[0])
    inp["window_fill_x"] = [(center_stack + side_stack) / l2 if l2 > 0 else 0]
    inp["window_fill_z1"] = [nwh1 / h1 if h1 > 0 else 0]
    inp["aspect_h1_l2"] = [h1 / l2 if l2 > 0 else 0]
    inp["aspect_w1_l2"] = [w1 / l2 if l2 > 0 else 0]
    return inp


def validation_check(input_df, strict=False, return_errors=False):
    """
    파생값 계산 + 기하 검증.
    기존 validation_check와 동일한 파생 컬럼명(nwl1_main, wff1_main, sl1_main_x ...)을
    생성해 orchestrator 호출 형태를 보존한다.

    strict=True (fixed 모드): 위반 시 위반 항목을 담아 ValueError raise
    strict=False (랜덤 모드): (False, df) 반환 -> 재추첨
    """
    inp = input_df.copy()
    errors = []

    material_contract_ok = False
    try:
        material_fields = build_core_material_contract_fields(
            cm_base=inp["core_cm"].iloc[0],
            core_x=inp["core_x"].iloc[0],
            core_y=inp["core_y"].iloc[0],
            lamination_factor=inp["core_lamination_factor"].iloc[0],
            loss_margin=inp["core_loss_margin"].iloc[0],
            area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"core material contract invalid: {exc}")
    else:
        for key, value in material_fields.items():
            inp[key] = [value]
        material_contract_ok = True

    N1_main = int(inp["N1_main"].iloc[0])
    N1_side = int(inp["N1_side"].iloc[0])
    N2_main = int(inp["N2_main"].iloc[0])
    N2_side = int(inp["N2_side"].iloc[0])

    inp["N1"] = [N1_main + N1_side]
    inp["N2"] = [N2_main + N2_side]

    l1 = float(inp["l1"].iloc[0])
    l2 = float(inp["l2"].iloc[0])
    h1 = float(inp["h1"].iloc[0])
    w1 = float(inp["w1"].iloc[0])
    cw1 = float(inp["cw1"].iloc[0])
    gap1 = float(inp["gap1"].iloc[0])
    cw2 = float(inp["cw2"].iloc[0])
    gap2 = float(inp["gap2"].iloc[0])
    nwh1 = float(inp["nwh1"].iloc[0])
    nwh2 = float(inp["nwh2"].iloc[0])

    n_core_group = int(inp["n_core_group"].iloc[0])
    core_plate_t = float(inp["core_plate_t"].iloc[0])
    core_plate_pad_t = float(inp["core_plate_pad_t"].iloc[0])
    core_plate_stack_t = core_plate_t + 2.0 * core_plate_pad_t

    # 코어 1조 깊이
    core_depth_each = (
        w1 - (n_core_group + 1) * core_plate_stack_t
    ) / n_core_group
    inp["core_depth_each"] = [core_depth_each]
    if core_depth_each <= 0:
        errors.append(
            f"core depth per group <= 0 (w1={w1}, "
            f"plate_t={core_plate_t}, pad_t={core_plate_pad_t})")
    else:
        # 코어 1조 깊이 범위: 랜덤 모드에서는 재추첨 사유, fixed 모드에서는 경고만
        # (도면 설계 150mm/조가 기본 범위(60~120) 밖이므로 fixed는 막지 않는다)
        d_min = float(inp["core_depth_min"].iloc[0])
        d_max = float(inp["core_depth_max"].iloc[0])
        if not (d_min <= core_depth_each <= d_max):
            if strict:
                import logging
                logging.warning(f"core_depth_each {core_depth_each:.1f}mm outside [{d_min}, {d_max}] (fixed mode - allowed)")
            else:
                errors.append(f"core_depth_each {core_depth_each:.1f} outside [{d_min}, {d_max}]")

    def _build(n, cw, gap):
        # x방향 권선 빌드 (등간격)
        if n <= 0:
            return 0.0
        return n * cw + (n - 1) * gap

    # net winding length (x방향 빌드)
    nwl1_main = _build(N1_main, cw1, gap1)
    nwl1_side = _build(N1_side, cw1, gap1)
    nwl2_main = _build(N2_main, cw2, gap2)
    nwl2_side = _build(N2_side, cw2, gap2)
    inp["nwl1_main"] = [nwl1_main]
    inp["nwl1_side"] = [nwl1_side]
    inp["nwl2_main"] = [nwl2_main]
    inp["nwl2_side"] = [nwl2_side]

    # fill factor (create_coil이 coil_width를 복원할 수 있도록)
    inp["wff1_main"] = [N1_main * cw1 / nwl1_main if nwl1_main > 0 else 0]
    inp["wff1_side"] = [N1_side * cw1 / nwl1_side if nwl1_side > 0 else 0]
    inp["wff2_main"] = [N2_main * cw2 / nwl2_main if nwl2_main > 0 else 0]
    inp["wff2_side"] = [N2_side * cw2 / nwl2_side if nwl2_side > 0 else 0]

    # 호환용 별칭 컬럼
    inp["coil_gap_layer1"] = [gap1]
    inp["coil_gap_layer2"] = [gap2]

    # 1차 중심 권선 y방향 빌드 (냉각판 슬롯 포함)
    tx_y_gaps, tx_slots = get_tx_y_gaps(inp)
    nwb1_main_y = N1_main * cw1 + sum(tx_y_gaps) if N1_main > 0 else 0.0
    inp["nwb1_main_y"] = [nwb1_main_y]

    # 권선-코어 높이 간격
    inp["h_gap1"] = [(h1 - nwh1) / 2]
    inp["h_gap2"] = [(h1 - nwh2) / 2]

    # 각 권선의 안쪽 개구(space) 크기
    # x방향은 코어(중심 레그 2*l1) 기준, y방향은 콜드플레이트 포함 외곽(w1) 기준
    cc_w2c_x = float(inp["cc_w2c_space_x"].iloc[0])
    cc_w2c_y = float(inp["cc_w2c_space_y"].iloc[0])
    w2c_w1c_x = float(inp["w2c_w1c_space_x"].iloc[0])
    w2c_w1c_y = float(inp["w2c_w1c_space_y"].iloc[0])
    w1s_cs_x = float(inp["w1s_cs_space_x"].iloc[0])
    cs_w1s_y = float(inp["cs_w1s_space_y"].iloc[0])
    w2s_w1s_x = float(inp["w2s_w1s_space_x"].iloc[0])
    w1s_w2s_y = float(inp["w1s_w2s_space_y"].iloc[0])

    sl2_main_x = 2 * l1 + 2 * cc_w2c_x
    sl2_main_y = w1 + 2 * cc_w2c_y
    sl1_main_x = sl2_main_x + 2 * nwl2_main + 2 * w2c_w1c_x
    sl1_main_y = sl2_main_y + 2 * nwl2_main + 2 * w2c_w1c_y

    sl1_side_x = l1 + 2 * w1s_cs_x
    sl1_side_y = w1 + 2 * cs_w1s_y
    sl2_side_x = sl1_side_x + 2 * nwl1_side + 2 * w2s_w1s_x
    sl2_side_y = sl1_side_y + 2 * nwl1_side + 2 * w1s_w2s_y

    inp["sl2_main_x"] = [sl2_main_x]
    inp["sl2_main_y"] = [sl2_main_y]
    inp["sl1_main_x"] = [sl1_main_x]
    inp["sl1_main_y"] = [sl1_main_y]
    inp["sl1_side_x"] = [sl1_side_x]
    inp["sl1_side_y"] = [sl1_side_y]
    inp["sl2_side_x"] = [sl2_side_x]
    inp["sl2_side_y"] = [sl2_side_y]

    # 창(x방향) 안에 중심 권선 + 측면 권선이 들어가는지
    center_stack = cc_w2c_x + nwl2_main + w2c_w1c_x + nwl1_main
    if N1_side > 0:
        side_stack = w1s_cs_x + nwl1_side + w2s_w1s_x + nwl2_side
    else:
        side_stack = w1s_cs_x + nwl2_side
    w1c_w2s_gap_actual = l2 - center_stack - side_stack
    inp["w1c_w2s_gap_x_actual"] = [w1c_w2s_gap_actual]

    # ---- 검증 ----
    if l2 <= 0:
        errors.append(f"l2 <= 0 ({l2})")
    if h1 <= 0:
        errors.append(f"h1 <= 0 ({h1})")
    if nwh1 > h1:
        errors.append(f"nwh1 ({nwh1}) > h1 ({h1})")
    if nwh2 > h1:
        errors.append(f"nwh2 ({nwh2}) > h1 ({h1})")
    if not math.isfinite(cw1):
        errors.append(f"cw1 must be finite ({cw1})")
    elif cw1 <= 0:
        errors.append(f"cw1 <= 0 ({cw1})")
    elif cw1 > PRIMARY_CONDUCTOR_MAX_THICKNESS_MM:
        errors.append(
            f"cw1 {cw1} > {PRIMARY_CONDUCTOR_MAX_THICKNESS_MM}mm "
            "(primary conductor thickness cap)"
        )
    if not math.isfinite(cw2) or cw2 <= 0:
        errors.append(f"cw2 must be finite and > 0 ({cw2})")
    if gap1 <= 0 or gap2 <= 0:
        errors.append(f"conductor gap <= 0 (gap1={gap1}, gap2={gap2})")

    min_clearance = float(inp["w1c_w2s_space_x"].iloc[0])
    if w1c_w2s_gap_actual < min_clearance:
        errors.append(
            f"Tx-Rx_side x clearance {w1c_w2s_gap_actual:.2f} < required {min_clearance:.2f}"
        )

    round_corner = int(inp["round_corner"].iloc[0]) != 0
    corner_radius = float(inp["corner_radius"].iloc[0])
    wcp_on = int(inp["wcp_on"].iloc[0]) != 0
    wcp_len_x = float(inp["wcp_len_x"].iloc[0])

    # Resolve the human-readable percentage from the actual millimetre
    # geometry.  ``wcp_len_x`` remains authoritative for legacy/fixed inputs;
    # Sobol candidates set it from their sampled percentage before validation.
    wcp_len_ref_x = sl1_main_x - (2.0 * corner_radius if round_corner else 0.0)
    inp["wcp_len_ref_x"] = [wcp_len_ref_x]
    if wcp_len_ref_x > 0:
        wcp_len_pct = 100.0 * wcp_len_x / wcp_len_ref_x
        inp["wcp_len_pct"] = [wcp_len_pct]
    else:
        wcp_len_pct = float("nan")
        errors.append(
            f"Tx innermost straight x span <= 0 ({wcp_len_ref_x})"
        )

    if round_corner:
        if corner_radius < 1.0:
            errors.append(f"corner_radius ({corner_radius}) < 1mm")
        # 각 권선의 가장 안쪽 턴 반폭보다 반경이 작아야 함
        inner_half_extents = [
            ("Rx_main", sl2_main_x / 2, sl2_main_y / 2),
            ("Tx_main", sl1_main_x / 2, sl1_main_y / 2),
        ]
        if N2_side > 0:
            inner_half_extents.append(("Rx_side", sl2_side_x / 2, sl2_side_y / 2))
        if N1_side > 0:
            inner_half_extents.append(("Tx_side", sl1_side_x / 2, sl1_side_y / 2))
        for wname, hx, hy in inner_half_extents:
            if corner_radius >= min(hx, hy):
                errors.append(
                    f"corner_radius ({corner_radius}) >= min inner half extent of {wname} ({min(hx, hy):.2f})"
                )

    if wcp_on and N1_main >= 2:
        # 냉각판이 y측 직선 구간 안에 놓여야 함 (직선 반길이 = x반폭 - 반경, 전 턴 동일)
        straight_half = wcp_len_ref_x / 2
        if wcp_len_x / 2 >= straight_half:
            errors.append(
                f"wcp_len_x/2 ({wcp_len_x / 2}) >= Tx straight half length ({straight_half:.2f})"
            )
        # Allow only tiny sub-0.1-mm quantization drift at the endpoints.
        pct_tolerance = 0.05
        if (not math.isfinite(wcp_len_pct)
                or wcp_len_pct < WCP_LENGTH_MIN_PCT - pct_tolerance
                or wcp_len_pct > WCP_LENGTH_MAX_PCT + pct_tolerance):
            errors.append(
                f"wcp_len_pct {wcp_len_pct:.3f} outside "
                f"[{WCP_LENGTH_MIN_PCT}, {WCP_LENGTH_MAX_PCT}]% "
                f"(wcp_len_x={wcp_len_x:.3f}mm, reference={wcp_len_ref_x:.3f}mm)"
            )

    # 서멀패드 두께 검증 (알루미늄 판 두께 > 0 이어야 함)
    wcp_pad_t = float(inp["wcp_pad_t"].iloc[0])
    core_plate_pad_t = float(inp["core_plate_pad_t"].iloc[0])
    if wcp_pad_t < 0 or core_plate_pad_t < 0:
        errors.append(f"pad thickness < 0 (wcp_pad_t={wcp_pad_t}, core_plate_pad_t={core_plate_pad_t})")

    # ---- 해석/열 파라미터 검증 ----
    if float(inp["freq"].iloc[0]) <= 0:
        errors.append(f"freq <= 0 ({inp['freq'].iloc[0]})")
    if int(inp["loss_on"].iloc[0]) != 0 and float(inp["V1_rms"].iloc[0]) <= 0:
        errors.append(f"V1_rms <= 0 ({inp['V1_rms'].iloc[0]}) with loss_on=1")
    try:
        core_k_anisotropic = float(inp["core_k_anisotropic"].iloc[0])
        if (
                not math.isfinite(core_k_anisotropic)
                or not core_k_anisotropic.is_integer()
                or int(core_k_anisotropic) not in (0, 1)):
            errors.append(
                "core_k_anisotropic must be 0 or 1 "
                f"({inp['core_k_anisotropic'].iloc[0]})"
            )
    except (TypeError, ValueError, OverflowError):
        errors.append(
            "core_k_anisotropic must be 0 or 1 "
            f"({inp['core_k_anisotropic'].iloc[0]})"
        )
    for key in ("core_k_alloy", "core_k_interlayer"):
        try:
            conductivity = float(inp[key].iloc[0])
            if not math.isfinite(conductivity) or conductivity <= 0:
                errors.append(
                    f"{key} must be finite and > 0 ({inp[key].iloc[0]})"
                )
        except (TypeError, ValueError, OverflowError):
            errors.append(
                f"{key} must be finite and > 0 ({inp[key].iloc[0]})"
            )
    cap_on = int(inp["cap_on"].iloc[0]) != 0
    if cap_on and int(inp["matrix_on"].iloc[0]) == 0:
        errors.append("cap_on=1 requires matrix_on=1 (inductance results are required)")
    try:
        cap_max_passes = float(inp["cap_max_passes"].iloc[0])
        if (
                not math.isfinite(cap_max_passes)
                or not cap_max_passes.is_integer()
                or cap_max_passes < 1):
            errors.append(
                "cap_max_passes must be a positive integer "
                f"({inp['cap_max_passes'].iloc[0]})"
            )
    except (TypeError, ValueError, OverflowError):
        errors.append(
            "cap_max_passes must be a positive integer "
            f"({inp['cap_max_passes'].iloc[0]})"
        )
    try:
        cap_percent_error = float(inp["cap_percent_error"].iloc[0])
        if not math.isfinite(cap_percent_error) or cap_percent_error <= 0:
            errors.append(
                "cap_percent_error must be finite and > 0 "
                f"({inp['cap_percent_error'].iloc[0]})"
            )
    except (TypeError, ValueError, OverflowError):
        errors.append(
            "cap_percent_error must be finite and > 0 "
            f"({inp['cap_percent_error'].iloc[0]})"
        )

    n_exp = int(inp["n_explicit_turns"].iloc[0])
    if int(inp["thermal_on"].iloc[0]) != 0 and n_exp < -1:
        errors.append(f"n_explicit_turns ({n_exp}) < -1")
    if str(inp["rx_mesh_mode"].iloc[0]) not in ("skin", "length", "length-coarse"):
        errors.append(f"invalid rx_mesh_mode ({inp['rx_mesh_mode'].iloc[0]})")
    if str(inp["thermal_symmetry"].iloc[0]) not in ("eighth", "quarter", "full"):
        errors.append(f"invalid thermal_symmetry ({inp['thermal_symmetry'].iloc[0]})")
    side_block_mesh_level = int(
        inp["thermal_rx_side_block_mesh_level"].iloc[0]
    )
    if side_block_mesh_level not in (4, 5):
        errors.append(
            "thermal_rx_side_block_mesh_level must be 4 or 5 "
            f"({side_block_mesh_level})"
        )
    # Tx측면 권선이 있으면 Rx측면과의 절연 간격 필수 (0이면 도체 접촉 -> 솔버 에러)
    if N1_side > 0:
        if float(inp["w2s_w1s_space_x"].iloc[0]) < 1.0:
            errors.append(f"w2s_w1s_space_x too small for N1_side>0 ({inp['w2s_w1s_space_x'].iloc[0]})")
        if float(inp["w1s_w2s_space_y"].iloc[0]) < 1.0:
            errors.append(f"w1s_w2s_space_y too small for N1_side>0 ({inp['w1s_w2s_space_y'].iloc[0]})")

    # Active campaign design-domain contract: total primary turns are capped
    # globally, including fixed/strict inputs and any legacy N1_side geometry.
    n1_total = int(inp["N1"].iloc[0])
    if n1_total > N1_MAX_TURNS:
        errors.append(f"N1 {n1_total} > {N1_MAX_TURNS} (cap)")

    for plate_key in ("wcp_t", "core_plate_t"):
        plate_value = float(inp[plate_key].iloc[0])
        if not COLD_PLATE_MIN_T_MM <= plate_value <= COLD_PLATE_MAX_T_MM:
            errors.append(
                f"{plate_key} {plate_value} outside "
                f"[{COLD_PLATE_MIN_T_MM}, {COLD_PLATE_MAX_T_MM}] mm")

    # 랜덤 모드 커버리지 하한: HV 절연쌍 실간격 최소 20mm (설계 타겟 40mm 주변 데이터 확보,
    # 비례축소 후 극소 간격 샘플로 예산 낭비 방지). fixed 모드는 사용자 판단 존중.
    if not strict:
        hv_gap_cols = ["cc_w2c_space_x", "cc_w2c_space_y", "w2c_w1c_space_x", "w2c_w1c_space_y",
                       "w1c_w2s_gap_x_actual", "w1s_cs_space_x", "cs_w1s_space_y"]
        min_gap = min(float(inp[c].iloc[0]) for c in hv_gap_cols if c in inp.columns)
        if min_gap < 20.0:
            errors.append(f"HV insulation coverage floor: min gap {min_gap:.1f} < 20mm")

    # 파생 물리 특징량 (회귀학습 입력용) - 검증 통과 여부와 무관하게 계산
    if material_contract_ok:
        try:
            _add_derived_features(inp)
        except Exception as exc:
            errors.append(f"derived feature calculation failed: {exc}")

    result = len(errors) == 0

    if strict and not result:
        raise ValueError("Parameter validation failed: " + " / ".join(errors))

    if return_errors:
        return result, inp, errors
    return result, inp


# AEDT 디자인 변수로 설정하지 않는 키 (지오메트리 수식에 안 쓰이는 해석/열 파라미터, 문자열 등)
NON_DESIGN_VAR_KEYS = {
    "rx_mesh_mode",
    # Derived, human-readable plate-length metadata. Geometry and submission
    # identity remain authoritative in the exact millimetre value wcp_len_x.
    "wcp_len_pct", "wcp_len_ref_x",
    "freq", "V1_rms", "I1_rated", "I2_rated", "I2_phase_deg",
    "P_target", "V2_rms",
    "core_cm", "core_x", "core_y",
    "matrix_on", "loss_on", "thermal_on",
    "cap_on", "cap_max_passes", "cap_percent_error",
    "plate_temp", "air_temp", "fan_velocity",
    "k_ins", "core_k_thermal", "core_k_anisotropic", "core_k_alloy",
    "core_k_interlayer", "n_explicit_turns",
    "max_passes", "percent_error", "min_converged",
    "matrix_percent_error", "matrix_max_passes", "matrix_min_converged", "keep_project",
    "core_depth_min", "core_depth_max",
    "loss_sym_on", "thermal_symmetry", "matrix_skin_mesh", "fan_config", "loss_from_copy",
    "thermal_max_iterations", "conductor_temp_C",
    "thermal_rx_side_block_mesh_level",
    "core_lamination_factor", "core_loss_margin",
    "physics_data_revision",
}


def get_design_var_columns(input_parameter):
    """디자인 변수로 설정되는 컬럼 목록 (리포트 variation 리스트 생성용)"""
    return [c for c in input_parameter.columns if c not in NON_DESIGN_VAR_KEYS]


def set_design_variables(design, input_parameter):
    """
    주어진 파라미터를 Ansys 디자인 변수로 설정합니다.
    (NON_DESIGN_VAR_KEYS 는 지오메트리에 안 쓰이므로 제외)
    """
    no_unit_keys = {
        "N1", "N2", "N1_main", "N1_side", "N2_main", "N2_side",
        "n_core_group", "core_plate_on", "wcp_on", "round_corner", "corner_segments", "full_model",
    }

    for key in get_design_var_columns(input_parameter):
        value = input_parameter.iloc[0][key]
        unit = "" if key in no_unit_keys else "mm"
        design.set_variable(variable_name=key, value=value, unit=unit)
