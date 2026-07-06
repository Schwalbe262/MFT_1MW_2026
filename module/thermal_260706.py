"""
Icepak 열해석 모듈 (설계도면260706 파이프라인 design3)

- 지오메트리: 항상 풀모델 (팬 유동 +y -> -y 가 대칭을 깨므로), 라운드 없이 직각
- 하이브리드 권선: Tx(5mm)는 턴 전부 explicit, Rx foil은 안/밖 n_explicit_turns씩 explicit,
  중간 턴들은 변마다 1개씩 4개의 직육면체 블록으로 균질화 (이방성 열전도율)
- 손실 주입: EM loss 디자인(design2) 계산기 값 사용. 대칭 EM인 경우 오브젝트별 환산:
    * 대칭 모델의 체적 적분값은 "절단 평면 수 c"에 대해 (실제값) x 2^c / 4 로 나타남
      (검증 실험: 3면 절단 오브젝트 = 실제의 1/2, 2면 절단 = 실제와 동일)
    * 따라서 실제값 = 대칭값 x 4 / 2^c, 대칭 모델에서 삭제된 미러 오브젝트는 대응값 복제
- 경계조건: 콜드플레이트/권선냉각판(Al) 고정온도, region +y면 velocity inlet, -y면 pressure opening
"""

import math
import logging

import pandas as pd

from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    compute_layer_positions,
)
from module.input_parameter_260706 import get_tx_y_gaps, set_design_variables


# ---------------------------------------------------------------------------
# 손실 환산 (대칭 EM -> 풀모델 실제값)
# ---------------------------------------------------------------------------

def _sym_factor(spans_x0, spans_y0, spans_z0):
    """
    대칭 모델 적분값 -> 실제값 환산 계수.
    검증 실험: 대칭 모델의 체적 적분 = 실제값 x 4 / 2^c (c = 오브젝트를 지나는 절단평면 수)
      c=3 (중심 권선 턴, 중앙 코어) -> 대칭값이 실제의 1/2  -> x2
      c=2 (측면 권선 링, 바깥 코어/플레이트) -> 대칭값 = 실제값 -> x1
    따라서 실제값 = 대칭값 x 2^c / 4
    """
    cuts = int(spans_x0) + int(spans_y0) + int(spans_z0)
    return (2 ** cuts) / 4.0


class LossAllocator:
    """EM loss 디자인의 계산기 결과(loss_map)를 풀모델 열해석 오브젝트에 배분"""

    def __init__(self, sim):
        self.sim = sim
        self.df = sim.df_plus
        self.loss_map = getattr(sim, "loss_map", {})
        # loss 디자인은 항상 풀모델로 해석되므로 (전압원 여자 유효성) 기본적으로 무보정.
        # 대칭 loss 값이 들어오는 예외 케이스만 오브젝트별 환산 적용.
        self.full_em = getattr(sim, "loss_em_full", int(self.df["full_model"].iloc[0]) != 0)

    def _get(self, key):
        v = self.loss_map.get(key)
        if v is None:
            logging.warning(f"loss_map missing key: {key} (0W assumed)")
            return 0.0
        return float(v)

    def turn_loss(self, expr_key, spans_x0=True, spans_y0=True, spans_z0=True):
        """개별 턴/오브젝트 손실 [W]"""
        v = self._get(expr_key)
        if self.full_em:
            return v
        return v * _sym_factor(spans_x0, spans_y0, spans_z0)

    def group_loss(self, expr_key, spans_x0=True, spans_y0=True, spans_z0=True):
        v = self._get(expr_key)
        if self.full_em:
            return v
        return v * _sym_factor(spans_x0, spans_y0, spans_z0)


# ---------------------------------------------------------------------------
# 재질
# ---------------------------------------------------------------------------

def _create_thermal_materials(ipk, df):
    """코어 등가재질 + 권선 균질화 이방성 재질 2종 생성"""
    k_core = float(df["core_k_thermal"].iloc[0])
    k_ins = float(df["k_ins"].iloc[0])
    cw2 = float(df["cw2"].iloc[0])
    gap2 = float(df["gap2"].iloc[0])

    ff = cw2 / (cw2 + gap2)              # foil 채움율
    k_in = ff * 385.0 + (1 - ff) * k_ins        # foil 면내 방향 (병렬)
    k_th = 1.0 / (ff / 385.0 + (1 - ff) / k_ins)  # 적층 방향 (직렬)

    mats = ipk.materials

    if "core_amorphous_thermal" not in mats.material_keys:
        m = mats.add_material("core_amorphous_thermal")
        m.thermal_conductivity = k_core
        m.mass_density = 7180
        m.specific_heat = 540

    # x측 블록: 적층(through) 방향 = x
    if "winding_homog_x" not in mats.material_keys:
        m = mats.add_material("winding_homog_x")
        m.thermal_conductivity = [k_th, k_in, k_in]
        m.mass_density = 8900 * ff
        m.specific_heat = 385

    # y측 블록: 적층(through) 방향 = y
    if "winding_homog_y" not in mats.material_keys:
        m = mats.add_material("winding_homog_y")
        m.thermal_conductivity = [k_in, k_th, k_in]
        m.mass_density = 8900 * ff
        m.specific_heat = 385

    if "thermal_pad" not in mats.material_keys:
        m = mats.add_material("thermal_pad")
        m.conductivity = 0
        m.thermal_conductivity = 0.2

    return k_in, k_th


# ---------------------------------------------------------------------------
# 지오메트리
# ---------------------------------------------------------------------------

def _rx_layout(df, prefix):
    """Rx 권선 그룹의 배치 정보 (턴 중심 위치, 도체 폭 등)"""
    cw2 = float(df["cw2"].iloc[0])
    gap2 = float(df["gap2"].iloc[0])
    if prefix == "main":
        N = int(df["N2_main"].iloc[0])
        slx = float(df["sl2_main_x"].iloc[0])
        sly = float(df["sl2_main_y"].iloc[0])
    else:
        N = int(df["N2_side"].iloc[0])
        slx = float(df["sl2_side_x"].iloc[0])
        sly = float(df["sl2_side_y"].iloc[0])
    gaps = [gap2] * (N - 1)
    x_pos = compute_layer_positions(slx / 2, cw2, gaps)
    y_pos = compute_layer_positions(sly / 2, cw2, gaps)
    return N, cw2, x_pos, y_pos


def _build_homog_blocks(ipk, df, prefix, name, offset_x, height):
    """Rx 그룹 중간 턴들을 4개 직육면체 블록으로 균질화"""
    n_exp = int(df["n_explicit_turns"].iloc[0])
    N, cw, x_pos, y_pos = _rx_layout(df, prefix)

    # 중간 영역: 안쪽 kept 턴(index n_exp-1)의 바깥면 ~ 바깥 kept 턴(index N-n_exp)의 안쪽면
    bx_in = x_pos[n_exp - 1] + cw / 2
    bx_out = x_pos[N - n_exp] - cw / 2
    by_in = y_pos[n_exp - 1] + cw / 2
    by_out = y_pos[N - n_exp] - cw / 2

    z0 = -height / 2
    blocks = []

    # x측 블록 2개 (y 전체 폭, 적층방향 x)
    for sign, tag in [(1, "xp"), (-1, "xn")]:
        obj = ipk.modeler.create_box(
            origin=[f"{(bx_in if sign > 0 else -bx_out) + offset_x}mm", f"{-by_out}mm", f"{z0}mm"],
            sizes=[f"{bx_out - bx_in}mm", f"{2 * by_out}mm", f"{height}mm"],
            name=f"{name}_block_{tag}",
            material="winding_homog_x"
        )
        blocks.append(obj)

    # y측 블록 2개 (x는 안쪽 경계 사이, 적층방향 y)
    for sign, tag in [(1, "yp"), (-1, "yn")]:
        obj = ipk.modeler.create_box(
            origin=[f"{-bx_in + offset_x}mm", f"{(by_in if sign > 0 else -by_out)}mm", f"{z0}mm"],
            sizes=[f"{2 * bx_in}mm", f"{by_out - by_in}mm", f"{height}mm"],
            name=f"{name}_block_{tag}",
            material="winding_homog_y"
        )
        blocks.append(obj)

    return blocks


def _build_geometry(ipk, sim):
    """풀모델 열해석 지오메트리 생성. 오브젝트 그룹 dict 반환"""
    df = sim.df_plus
    n_exp = int(df["n_explicit_turns"].iloc[0])
    l1 = float(df["l1"].iloc[0])
    l2 = float(df["l2"].iloc[0])
    nwh1 = float(df["nwh1"].iloc[0])
    nwh2 = float(df["nwh2"].iloc[0])

    objs = {}

    # ---- 코어 + 콜드플레이트 + 패드 ----
    n_group = int(df["n_core_group"].iloc[0])
    plate_on = int(df["core_plate_on"].iloc[0]) != 0
    pad_on = float(df["core_plate_pad_t"].iloc[0]) > 0
    core_objs, plate_objs, pad_objs = create_core(
        design=ipk, name="core", core_material="core_amorphous_thermal",
        n_group=n_group, plate_material="aluminum", pad_material="thermal_pad",
        plate_on=plate_on, pad_on=pad_on,
        plate_color=[144, 190, 144], pad_color=[200, 160, 200]
    )
    objs["core"] = core_objs
    objs["core_plates"] = plate_objs
    objs["core_pads"] = pad_objs

    # ---- Tx (전 턴 explicit, 직각) ----
    tx_y_gaps, tx_slots = get_tx_y_gaps(df)
    tx_windings, _, tx_cw, _, _, _ = create_coil(
        design=ipk, name="Tx_main",
        window_height=df["nwh1"].iloc[0], window_length=df["nwl1_main"].iloc[0],
        window_layer=df["N1_main"].iloc[0], N_input=1,
        width_fill_factor=df["wff1_main"].iloc[0],
        space_length=df["sl1_main_x"].iloc[0], space_width=df["sl1_main_y"].iloc[0],
        shape="rectangle", offset=[0, 0, 0], color=[255, 10, 10],
        y_slot_gaps=tx_y_gaps, round_corner=False
    )
    objs["Tx"] = tx_windings

    # ---- 권선 냉각 플레이트 ----
    wcp_on = int(df["wcp_on"].iloc[0]) != 0
    if wcp_on and len(tx_slots) > 0:
        wcp_plates, wcp_pads = create_winding_cooling_plates(
            design=ipk, name="Tx_main_wcp",
            space_width=df["sl1_main_y"].iloc[0], coil_width=tx_cw,
            y_gaps=tx_y_gaps, slot_indices=tx_slots,
            wcp_len_x=float(df["wcp_len_x"].iloc[0]), wcp_t=float(df["wcp_t"].iloc[0]),
            pad_t=float(df["wcp_pad_t"].iloc[0]), height=nwh1,
            plate_material="aluminum", pad_material="thermal_pad",
            plate_color=[144, 190, 144], pad_color=[200, 160, 200]
        )
        objs["wcp_plates"] = wcp_plates
        objs["wcp_pads"] = wcp_pads
    else:
        objs["wcp_plates"] = []
        objs["wcp_pads"] = []

    # ---- Rx 하이브리드 (main 1조 + side 2조) ----
    def _build_rx_group(prefix, name, offset_x):
        windings, _, _, _, _, _ = create_coil(
            design=ipk, name=name,
            window_height=df["nwh2"].iloc[0],
            window_length=df[f"nwl2_{prefix}"].iloc[0],
            window_layer=df[f"N2_{prefix}"].iloc[0], N_input=1,
            width_fill_factor=df[f"wff2_{prefix}"].iloc[0],
            space_length=df[f"sl2_{prefix}_x"].iloc[0],
            space_width=df[f"sl2_{prefix}_y"].iloc[0],
            shape="rectangle", offset=[offset_x, 0, 0], color=[10, 10, 255],
            round_corner=False
        )
        N = len(windings)
        explicit = windings[:n_exp] + windings[-n_exp:]
        middle = windings[n_exp:N - n_exp]
        if middle:
            ipk.modeler.delete([w.name for w in middle])
        blocks = _build_homog_blocks(ipk, df, prefix, name, offset_x, nwh2)
        return explicit, blocks

    objs["Rx_main_explicit"], objs["Rx_main_blocks"] = _build_rx_group("main", "Rx_main", 0.0)

    objs["Rx_side_explicit"] = []
    objs["Rx_side_blocks"] = []
    objs["Rx_side2_explicit"] = []
    objs["Rx_side2_blocks"] = []
    if int(df["N2_side"].iloc[0]) > 0:
        off = l1 + l2 + l1 / 2
        objs["Rx_side_explicit"], objs["Rx_side_blocks"] = _build_rx_group("side", "Rx_side", -off)
        objs["Rx_side2_explicit"], objs["Rx_side2_blocks"] = _build_rx_group("side", "Rx_side2", +off)

    return objs


# ---------------------------------------------------------------------------
# 손실 주입 / 경계조건
# ---------------------------------------------------------------------------

def _assign_losses(ipk, sim, objs):
    df = sim.df_plus
    alloc = LossAllocator(sim)
    n_exp = int(df["n_explicit_turns"].iloc[0])
    injected = {}

    def _block(obj, watts):
        w = max(float(watts), 0.0)
        try:
            ipk.assign_solid_block(obj.name, f"{w}W")
            injected[obj.name] = w
        except Exception as e:
            logging.warning(f"assign_solid_block failed for {obj.name}: {e}")

    # Tx 턴별 (3면 절단 -> x2)
    for w in objs["Tx"]:
        _block(w, alloc.turn_loss(f"P_turn_{w.name}"))

    # Rx main explicit 턴 (3면 절단 -> x2)
    p_main_explicit = 0.0
    for w in objs["Rx_main_explicit"]:
        p = alloc.turn_loss(f"P_turn_{w.name}")
        p_main_explicit += p
        _block(w, p)

    # Rx main 중간 블록: 그룹 총손실 - explicit, 4블록에 체적 비례 배분
    p_main_total = alloc.group_loss("P_Rx_main_group")
    p_mid = max(p_main_total - p_main_explicit, 0.0)
    vols = [abs(b.volume) for b in objs["Rx_main_blocks"]]
    vtot = sum(vols) or 1.0
    for b, v in zip(objs["Rx_main_blocks"], vols):
        _block(b, p_mid * v / vtot)

    # Rx side: 측면 링은 x=0 평면에 안 잘리므로 (2면 절단) 대칭값 = 링 1개 실제값
    if objs["Rx_side_explicit"]:
        def _side_ring(explicit_objs, blocks):
            p_exp_total = 0.0
            for i, w in enumerate(explicit_objs):
                # loss_map 키는 EM 디자인의 좌측 링 이름(Rx_side_i_0) 기준
                em_name = w.name.replace("Rx_side2", "Rx_side")
                p = alloc.turn_loss(f"P_turn_{em_name}", spans_x0=False)
                p_exp_total += p
                _block(w, p)
            p_total = alloc.group_loss("P_Rx_side_group", spans_x0=False)
            p_mid_s = max(p_total - p_exp_total, 0.0)
            vols_s = [abs(b.volume) for b in blocks]
            vt = sum(vols_s) or 1.0
            for b, v in zip(blocks, vols_s):
                _block(b, p_mid_s * v / vt)

        _side_ring(objs["Rx_side_explicit"], objs["Rx_side_blocks"])
        _side_ring(objs["Rx_side2_explicit"], objs["Rx_side2_blocks"])

    # 코어 그룹: y=0에 걸친 그룹은 x2, 바깥 그룹은 x1 + 미러 복제
    n_group = int(df["n_core_group"].iloc[0])
    w1 = float(df["w1"].iloc[0])
    plate_t = float(df["core_plate_t"].iloc[0])
    d = (w1 - (n_group + 1) * plate_t) / n_group
    for i, c in enumerate(objs["core"]):
        y0 = -w1 / 2 + (i + 1) * plate_t + i * d
        y1 = y0 + d
        spans_y0 = (y0 < 0 < y1)
        if spans_y0:
            p = alloc.group_loss(f"P_{c.name}")
        else:
            # 대칭 모델에는 y>0 그룹만 존재 -> 미러 그룹은 대응 그룹 값 사용
            mirror_idx = n_group - 1 - i
            key = f"P_core_{mirror_idx + 1}" if f"P_core_{i + 1}" not in alloc.loss_map else f"P_core_{i + 1}"
            p = alloc.turn_loss(key, spans_y0=False)
        _block(c, p)

    # 콜드플레이트/냉각판은 고정온도 경계라 열원 주입 생략
    sim.thermal_injected = injected
    return injected


def _assign_boundaries(ipk, sim, objs):
    df = sim.df_plus
    plate_temp = float(df["plate_temp"].iloc[0])
    air_temp = float(df["air_temp"].iloc[0])
    fan_v = float(df["fan_velocity"].iloc[0])

    # 콜드플레이트 + 권선 냉각판 (Al) 고정온도
    fixed_objs = [o.name for o in objs["core_plates"] + objs["wcp_plates"]]
    if fixed_objs:
        try:
            ipk.assign_icepak_source(
                assignment=fixed_objs,
                thermal_condition="Fixed Temperature",
                assignment_value=f"{plate_temp}cel",
                boundary_name="cold_plates_fixed_T"
            )
        except Exception as e:
            logging.warning(f"Fixed temperature assignment failed: {e}")

    # 주변온도
    try:
        ipk.set_ambient_temp(air_temp)
    except Exception as e:
        logging.warning(f"set_ambient_temp failed: {e}")

    # region + 팬 유동 (+y -> -y)
    region = ipk.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0,
                                           x_neg=100.0, y_neg=100.0, z_neg=100.0,
                                           is_percentage=True)
    try:
        ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id],
            boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]
        )
        ipk.assign_pressure_free_opening(
            assignment=[region.bottom_face_y.id],
            boundary_name="outlet",
            temperature=f"{air_temp}cel"
        )
    except Exception as e:
        logging.warning(f"Opening assignment failed: {e}")

    return region


# ---------------------------------------------------------------------------
# 메인 엔트리
# ---------------------------------------------------------------------------

def run_thermal_analysis(sim):
    """
    EM loss 디자인 결과(sim.loss_map)를 이용해 Icepak 열해석 수행.
    반환: 온도 요약 1행 DataFrame (T_max_*, T_mean_*)
    """
    df = sim.df_plus

    ipk = sim.project.create_design(name="icepak_thermal", solver="icepak",
                                    solution="SteadyState TemperatureAndFlow")
    sim.design_thermal = ipk

    set_design_variables(ipk, sim.input_df)
    _create_thermal_materials(ipk, df)
    objs = _build_geometry(ipk, sim)
    _assign_losses(ipk, sim, objs)
    _assign_boundaries(ipk, sim, objs)

    setup = ipk.create_setup(name="ThermalSetup")
    try:
        setup.props["Flow Regime"] = "Turbulent"
        setup.props["Convergence Criteria - Max Iterations"] = 250
        setup.update()
    except Exception as e:
        logging.warning(f"Thermal setup props: {e}")

    ipk.analyze(cores=sim.NUM_CORE)
    try:
        sim.save_project()
    except Exception:
        pass

    # ---- 온도 추출 ----
    probe = []
    probe += [[w.name, f"T_max_{w.name}", "Temp_max"] for w in objs["Tx"]]
    probe += [[w.name, f"T_max_{w.name}", "Temp_max"] for w in objs["Rx_main_explicit"]]
    probe += [[b.name, f"T_max_{b.name}", "Temp_max"] for b in objs["Rx_main_blocks"]]
    probe += [[w.name, f"T_max_{w.name}", "Temp_max"] for w in objs["Rx_side_explicit"] + objs["Rx_side2_explicit"]]
    probe += [[b.name, f"T_max_{b.name}", "Temp_max"] for b in objs["Rx_side_blocks"] + objs["Rx_side2_blocks"]]
    probe += [[c.name, f"T_max_{c.name}", "Temp_max"] for c in objs["core"]]

    try:
        _, df_t = ipk.get_calculator_parameter(dir=sim.project.path, parameters=probe,
                                               report_name="thermal_report", file_name="thermal_report")
        temps = {c: float(df_t[c].iloc[0]) for c in df_t.columns if c.startswith("T_")}
    except Exception as e:
        logging.warning(f"Temperature extraction failed: {e}")
        temps = {}

    def _group_max(prefixes):
        vals = [v for k, v in temps.items() if any(p in k for p in prefixes)]
        return max(vals) if vals else float("nan")

    summary = {
        "T_max_Tx": [_group_max(["Tx_main"])],
        "T_max_Rx_main": [_group_max(["Rx_main"])],
        "T_max_Rx_side": [_group_max(["Rx_side"])],
        "T_max_core": [_group_max(["core_"])],
    }
    # 개별 값도 함께 저장
    for k, v in temps.items():
        summary[k] = [v]

    sim.df_thermal = pd.DataFrame(summary)
    return sim.df_thermal
