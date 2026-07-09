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
import os

import pandas as pd

from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    compute_layer_positions,
)
from module.input_parameter_260706 import get_tx_y_gaps, set_design_variables


def _native_solver(app):
    """Return the PyAEDT solver behind a pyDesign wrapper."""
    solver = getattr(app, "solver_instance", None)
    return solver if solver is not None else app


def _activate_thermal_design(app):
    """Re-acquire the active native design and refresh the raw PyAEDT handle."""
    solver = _native_solver(app)
    project = getattr(solver, "oproject", None)
    set_active = getattr(project, "SetActiveDesign", None)
    if not callable(set_active):
        raise RuntimeError("native Icepak project handle has no SetActiveDesign")

    design_name = getattr(solver, "design_name", None) or getattr(app, "design_name", None)
    if not design_name:
        raise RuntimeError("thermal design name is unavailable")
    design = set_active(design_name)
    if not design or not callable(getattr(design, "GetModule", None)):
        raise RuntimeError(f"failed to activate thermal design: {design_name}")

    solver._odesign = design
    design_solutions = getattr(solver, "design_solutions", None)
    if design_solutions is not None:
        design_solutions._odesign = design
    return solver, design


def _require_boundary(result, operation, expected_props=None):
    """Require a created PyAEDT boundary and, when requested, its input props."""
    if not result:
        raise RuntimeError(f"{operation} returned no boundary")
    if expected_props is None:
        return result

    props = getattr(result, "props", None)
    if not hasattr(props, "get"):
        raise RuntimeError(f"{operation} returned a boundary without readable props")
    for key, expected in expected_props.items():
        actual = props.get(key)
        if key == "Objects":
            actual_names = [actual] if isinstance(actual, str) else list(actual or [])
            expected_names = [expected] if isinstance(expected, str) else list(expected)
            if set(map(str, actual_names)) != set(map(str, expected_names)):
                raise RuntimeError(
                    f"{operation} property {key!r} mismatch: {actual_names!r} != {expected_names!r}"
                )
        elif str(actual) != str(expected):
            raise RuntimeError(f"{operation} property {key!r} mismatch: {actual!r} != {expected!r}")
    return result


def _split_retained(ipk, objects, plane, sides):
    """Split geometry and require at least one retained input object to remain live."""
    if not objects:
        raise RuntimeError(f"cannot split an empty thermal geometry on {plane}")
    result = ipk.modeler.split(assignment=objects, plane=plane, sides=sides)
    if not result:
        raise RuntimeError(f"thermal geometry split failed on {plane} ({sides})")
    existing = set(ipk.modeler.object_names)
    alive = [obj for obj in objects if obj.name in existing]
    if not alive:
        raise RuntimeError(f"thermal geometry split on {plane} retained no live objects")
    return alive


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
    """
    EM loss 디자인의 실물 기준 손실(loss_map_phys)을 열해석 오브젝트에 배분.
    - full thermal: 오브젝트당 실물값 그대로 (미러 오브젝트는 대응 키 폴백)
    - eighth thermal: 실물값 x 보유 체적 분율(1/2^c)
    """

    def __init__(self, sim, eighth=False, mode=None):
        self.sim = sim
        self.df = sim.df_plus
        self.loss_map = getattr(sim, "loss_map_phys", None) or getattr(sim, "loss_map", {})
        self.mode = mode or ("eighth" if eighth else "full")
        self.eighth = self.mode == "eighth"

    def _get(self, key):
        v = self.loss_map.get(key)
        alt = None
        if v is None:
            # 미러 오브젝트 폴백 (대칭 EM에 없는 y<0 쪽 등): 대응 오브젝트 키로 대체
            alt = key.replace("Rx_side2", "Rx_side").replace("_n", "_p")
            v = self.loss_map.get(alt)
        if v is None:
            alias = f" (alias {alt})" if alt and alt != key else ""
            raise KeyError(f"required thermal loss key missing: {key}{alias}")
        return float(v)

    def _retained_fraction(self, obj_name):
        if self.mode == "full":
            return 1.0
        from module.input_parameter_260706 import sym_cut_count
        c = sym_cut_count(obj_name, self.df)
        if self.mode == "quarter":
            c = max(c - 1, 0)  # z절단 없음 (모든 오브젝트가 z=0 스팬이므로 -1)
        return 1.0 / (2 ** c)

    def turn_loss(self, expr_key, obj_name=None, **_legacy):
        """개별 턴/오브젝트에 주입할 손실 [W] (열모델 보유 체적 기준)"""
        name = obj_name or expr_key.replace("P_turn_", "").replace("P_", "")
        return self._get(expr_key) * self._retained_fraction(name)

    def group_loss(self, expr_key, obj_name=None, **_legacy):
        name = obj_name or expr_key.replace("P_", "").replace("_group", "")
        return self._get(expr_key) * self._retained_fraction(name)


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


def _build_geometry(ipk, sim, eighth=False, mode=None):
    """열해석 지오메트리 생성. mode: full / quarter(x,y 분할) / eighth(x,y,z 분할)"""
    mode = mode or ("eighth" if eighth else "full")
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
    # n_explicit_turns = -1 이면 전 턴 explicit (블록 없음, 균질화 가정 제거).
    # 2*n_exp >= N 인 경우도 전 턴 explicit으로 처리 (중복/퇴화 블록 방지)
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
        if n_exp < 0 or 2 * n_exp >= N:
            return windings, []  # 전 턴 explicit
        explicit = windings[:n_exp] + windings[-n_exp:]
        middle = windings[n_exp:N - n_exp]
        if middle:
            # 대형 샘플에서 일괄 삭제가 gRPC로 부분 실패하는 사례 실측
            # ('Object does not have mesh' -> 솔버 중단) -> 삭제 후 검증 + 개별 재삭제
            names = [w.name for w in middle]
            ipk.modeler.delete(names)
            survivors = [n for n in names if n in set(ipk.modeler.object_names)]
            for n in survivors:
                try:
                    ipk.modeler.delete(n)
                except Exception:
                    pass
            survivors = [n for n in names if n in set(ipk.modeler.object_names)]
            if survivors:
                raise RuntimeError(f"middle turn deletion failed for {len(survivors)} objects "
                                   f"({survivors[:3]}...) - aborting thermal build")
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
        if mode == "full":
            # 대칭 모드에서는 +x 측 링이 어차피 절단 제거되므로 생성 생략 (모델링 시간 절약)
            objs["Rx_side2_explicit"], objs["Rx_side2_blocks"] = _build_rx_group("side", "Rx_side2", +off)

    if mode in ("eighth", "quarter"):
        # EM 대칭 모델과 동일 옥탄트 (x<=0, y>=0[, z>=0])
        all_objs = []
        for grp in objs.values():
            all_objs.extend(grp)
        if mode == "eighth":
            all_objs = _split_retained(ipk, all_objs, plane="XY", sides="PositiveOnly")
        all_objs = _split_retained(ipk, all_objs, plane="XZ", sides="PositiveOnly")
        _split_retained(ipk, all_objs, plane="YZ", sides="NegativeOnly")
        existing = set(ipk.modeler.object_names)
        for key in list(objs.keys()):
            objs[key] = [o for o in objs[key] if o.name in existing]

        required_groups = {
            "core": objs["core"],
            "Tx": objs["Tx"],
            "Rx_main": objs["Rx_main_explicit"] + objs["Rx_main_blocks"],
        }
        if int(df["N2_side"].iloc[0]) > 0:
            required_groups["Rx_side"] = objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
        missing_groups = [name for name, group in required_groups.items() if not group]
        if missing_groups:
            raise RuntimeError(f"thermal symmetry split removed required groups: {missing_groups}")

    return objs


def _create_probe_sheets(ipk, df, objs, eighth=False, mode=None):
    """
    회귀학습용 온도 프로브 시트 생성 (비모델 - 메시에 영향 없음).

    체적 max 온도는 메시 스파이크에 취약하므로, 대칭면 위치(x=0/y=0 평면)에
    권선 단면 크기의 시트를 만들어 그 위에서 Temp를 추출한다.
    위치가 파라미터만으로 결정되므로 모든 샘플에서 기하학적으로 동일 -> 데이터 일관성.
    팬(+y -> -y) 기준 풍하측(y-) 단면을 잡아 핫스팟 쪽을 캡처한다.
    """
    l1 = float(df["l1"].iloc[0])
    h1 = float(df["h1"].iloc[0])
    nwh1 = float(df["nwh1"].iloc[0])
    nwh2 = float(df["nwh2"].iloc[0])
    l2 = float(df["l2"].iloc[0])
    cw1 = float(df["cw1"].iloc[0])

    sheets = []

    def _sheet(name, orientation, origin, sizes):
        try:
            obj = ipk.modeler.create_rectangle(
                orientation=orientation, origin=[f"{v}mm" for v in origin],
                sizes=[f"{v}mm" for v in sizes], name=name
            )
            obj.model = False
            sheets.append(obj)
            return obj
        except Exception as e:
            logging.warning(f"probe sheet {name} failed: {e}")
            return None

    mode = mode or ("eighth" if eighth else "full")
    eighth = mode == "eighth"
    sym_xy = mode in ("eighth", "quarter")  # x<=0, y>=0 옥탄트 (quarter는 z 전체)

    # ---- Tx (1차): y측 단면 (x=0 평면) + x측 단면 (y=0 평면) ----
    tx_gaps, _ = get_tx_y_gaps(df)
    N1 = int(df["N1_main"].iloc[0])
    tx_x = compute_layer_positions(float(df["sl1_main_x"].iloc[0]) / 2, cw1, [float(df["gap1"].iloc[0])] * (N1 - 1))
    tx_y = compute_layer_positions(float(df["sl1_main_y"].iloc[0]) / 2, cw1, tx_gaps)
    zh = 0.48 * nwh1
    z_half_only = (mode == "eighth")  # quarter/full은 z 전체

    def _z_range(zh_):
        return (0.0, zh_) if z_half_only else (-zh_, zh_)

    # 이하 배치 로직에서 "eighth"는 x/y 옥탄트 배치를 의미하므로 quarter도 동일하게 취급
    eighth = sym_xy

    z0, z1 = _z_range(zh)
    # YZ 평면 시트: y측 런의 단면 (eighth: +y측 / full: -y 풍하측)
    # eighth 모드에서는 x=0이 region 경계면과 겹쳐 필드 평가가 실패하므로 1mm 안쪽(x<0)에 배치
    x_probe = -1.0 if eighth else 0.0
    y_start = (tx_y[0] - cw1 / 2) if eighth else -(tx_y[-1] + cw1 / 2)
    _sheet("Tprobe_Tx_leeward", "YZ", [x_probe, y_start, z0], [(tx_y[-1] - tx_y[0]) + cw1, z1 - z0])
    # XZ 평면 시트 (y=0): x- 런의 단면 (보유 옥탄트가 x<=0)
    # 주의: Y-법선("XZ"/"ZX") 사각형의 AEDT 치수 순서는 [z스팬, x스팬] (전치 버그 수정)
    _sheet("Tprobe_Tx_side", "XZ", [-(tx_x[-1] + cw1 / 2), 0, z0], [z1 - z0, (tx_x[-1] - tx_x[0]) + cw1])

    # ---- Rx 그룹 공통 생성기 ----
    def _rx_probes(prefix, name, offset_x):
        N, cw, x_pos, y_pos = _rx_layout(df, prefix)
        zh2 = 0.48 * nwh2
        za, zb = _z_range(zh2)
        y_in, y_out = y_pos[0] - cw / 2, y_pos[-1] + cw / 2
        x_in, x_out = x_pos[0] - cw / 2, x_pos[-1] + cw / 2
        ys = y_in if eighth else -y_out
        # eighth: 링 중심(x=offset)이 0이면 region 경계와 겹침 -> 1mm 안쪽
        xs = offset_x if offset_x < 0 else (-1.0 if eighth else offset_x)
        _sheet(f"Tprobe_{name}_leeward", "YZ", [xs, ys, za], [y_out - y_in, zb - za])
        # y=0 평면: 바깥쪽(코어 중심에서 먼 쪽) 런 - 보유 옥탄트 x<=0 기준
        _sheet(f"Tprobe_{name}_side", "XZ", [offset_x - x_out, 0, za], [zb - za, x_out - x_in])

    _rx_probes("main", "Rx_main", 0.0)
    if int(df["N2_side"].iloc[0]) > 0:
        off = l1 + l2 + l1 / 2
        _rx_probes("side", "Rx_side", -off)

    # ---- 코어: 중심 레그 y=0 단면 ----
    zc = 0.48 * (h1 + 2 * l1)
    zca, zcb = _z_range(zc)
    _sheet("Tprobe_core_center", "XZ", [-0.9 * l1, 0, zca], [zcb - zca, 0.9 * l1 if eighth else 1.8 * l1])

    return sheets


# ---------------------------------------------------------------------------
# 손실 주입 / 경계조건
# ---------------------------------------------------------------------------

def _assign_losses(ipk, sim, objs, eighth=False, mode=None):
    """실물 기준 손실(loss_map_phys)을 열모델 오브젝트에 주입.
    eighth 모드에서는 보유 체적 분율(1/2^c)이 LossAllocator에서 자동 적용된다."""
    df = sim.df_plus
    alloc = LossAllocator(sim, eighth=eighth, mode=mode)
    injected = {}

    def _block(obj, watts):
        w = max(float(watts), 0.0)
        if not math.isfinite(w):
            raise ValueError(f"non-finite thermal loss for {obj.name}: {w}")
        power = f"{w}W"
        boundary = ipk.assign_solid_block(obj.name, power)
        _require_boundary(
            boundary,
            f"solid block source for {obj.name}",
            {
                "Block Type": "Solid",
                "Objects": [obj.name],
                "Total Power": power,
            },
        )
        injected[obj.name] = w

    # Tx 턴별
    for w in objs["Tx"]:
        _block(w, alloc.turn_loss(f"P_turn_{w.name}", w.name))

    # Rx 그룹 공통: explicit 턴 + 중간 균질 블록 (그룹 총손실 - explicit, 체적 비례)
    def _rx_group(explicit_objs, blocks, group_key, name_hint):
        p_exp_total = 0.0
        for w in explicit_objs:
            em_name = w.name.replace("Rx_side2", "Rx_side")
            p = alloc.turn_loss(f"P_turn_{em_name}", em_name)
            p_exp_total += p
            _block(w, p)
        p_total = alloc.group_loss(group_key, name_hint)
        p_mid = max(p_total - p_exp_total, 0.0)
        vols = [abs(b.volume) for b in blocks]
        vt = sum(vols) or 1.0
        for b, v in zip(blocks, vols):
            _block(b, p_mid * v / vt)

    _rx_group(objs["Rx_main_explicit"], objs["Rx_main_blocks"], "P_Rx_main_group", "Rx_main_0_0")
    if objs["Rx_side_explicit"]:
        _rx_group(objs["Rx_side_explicit"], objs["Rx_side_blocks"], "P_Rx_side_group", "Rx_side_0_0")
    if objs["Rx_side2_explicit"]:
        _rx_group(objs["Rx_side2_explicit"], objs["Rx_side2_blocks"], "P_Rx_side_group", "Rx_side_0_0")

    # 코어 그룹: loss_map_phys에 있는 키 우선, 없으면(풀 열해석 + 대칭 EM 조합의 미러) 대응 그룹
    n_group = int(df["n_core_group"].iloc[0])
    for c in objs["core"]:
        try:
            i = int(c.name.split("_")[1])
        except (IndexError, ValueError):
            i = 1
        key = f"P_{c.name}"
        if key not in alloc.loss_map:
            key = f"P_core_{n_group + 1 - i}"  # y-미러 그룹
        _block(c, alloc.turn_loss(key, c.name))

    # 콜드플레이트/냉각판은 고정온도 경계라 열원 주입 생략
    sim.thermal_injected = injected
    tx_sum = sum(v for k, v in injected.items() if k.startswith("Tx_"))
    rx_sum = sum(v for k, v in injected.items() if k.startswith("Rx_"))
    core_sum = sum(v for k, v in injected.items() if k.startswith("core"))
    logging.warning(f"thermal injection totals [W]: Tx={tx_sum:.2f} Rx={rx_sum:.2f} core={core_sum:.2f} "
                 f"(eighth={eighth}, n_obj={len(injected)})")
    for k, v in injected.items():
        if k.startswith("Tx_main_0"):
            logging.warning(f"  inject {k} = {v:.3f} W")
    return injected


def _assign_boundaries(ipk, sim, objs, eighth=False, mode=None):
    mode = mode or ("eighth" if eighth else "full")
    df = sim.df_plus
    plate_temp = float(df["plate_temp"].iloc[0])
    air_temp = float(df["air_temp"].iloc[0])
    fan_v = float(df["fan_velocity"].iloc[0])

    def _bc(result, name):
        return _require_boundary(result, f"thermal boundary {name}")

    # 콜드플레이트 + 권선 냉각판 (Al) 고정온도
    fixed_objs = [o.name for o in objs["core_plates"] + objs["wcp_plates"]]
    if fixed_objs:
        fixed_temperature = f"{plate_temp}cel"
        boundary = ipk.assign_source(
            assignment=fixed_objs,
            thermal_condition="Temperature",
            assignment_value=fixed_temperature,
            boundary_name="cold_plates_fixed_T",
        )
        _require_boundary(
            boundary,
            "fixed temperature source cold_plates_fixed_T",
            {
                "Objects": fixed_objs,
                "Thermal Condition": "Temperature",
                "Temperature": fixed_temperature,
            },
        )

    # 주변온도
    ipk.set_ambient_temp(air_temp)

    def _fresh_region(**pads):
        # Icepak 디자인은 생성 시 AEDT가 Region을 자동 삽입함 -> 삭제 후 원하는 패딩으로 재생성.
        # (이전 코드는 create_air_region이 False를 반환해 경계 전부가 조용히 누락된 채
        #  밀폐 상자로 해석되는 치명적 버그가 있었음 - 게이트1에서 발견)
        try:
            if "Region" in ipk.modeler.object_names:
                ipk.modeler.delete("Region")
        except Exception as e:
            logging.warning(f"default Region delete failed: {e}")
        region = ipk.modeler.create_air_region(is_percentage=True, **pads)
        if not region:
            raise RuntimeError("create_air_region failed (Region conflict?)")
        return region

    # 경계 실패는 조용히 넘기지 않음: BC가 틀린 열해석은 실패보다 나쁨 (캠페인 데이터 오염)
    if mode == "quarter":
        # x/y 대칭 + z 전체 + 부력 on: 부력(z대칭 가정) 분리 검증용
        region = _fresh_region(x_pos=0.0, y_pos=100.0, z_pos=100.0,
                               x_neg=100.0, y_neg=0.0, z_neg=100.0)
        _bc(ipk.assign_symmetry_wall(
            geometry=region.top_face_x.id, boundary_name="sym_x0"), "sym_x0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_y.id, boundary_name="sym_y0"), "sym_y0")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet")
        for face, nm in [(region.bottom_face_x, "outlet_xn"),
                         (region.top_face_z, "outlet_zp"), (region.bottom_face_z, "outlet_zn")]:
            _bc(ipk.assign_pressure_free_opening(
                assignment=[face.id], boundary_name=nm, temperature=f"{air_temp}cel"), nm)
        return region

    if eighth:
        # 1/8: 대칭면 3개(x=0/y=0/z=0)는 region 면을 플러시로 두고 symmetry wall 할당.
        # +y 외곽 = 팬 유입 (양측 팬의 y대칭 유동 가정), -x/+z 외곽 = 배기 opening
        region = _fresh_region(x_pos=0.0, y_pos=100.0, z_pos=100.0,
                               x_neg=100.0, y_neg=0.0, z_neg=0.0)
        _bc(ipk.assign_symmetry_wall(
            geometry=region.top_face_x.id, boundary_name="sym_x0"), "sym_x0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_y.id, boundary_name="sym_y0"), "sym_y0")
        _bc(ipk.assign_symmetry_wall(
            geometry=region.bottom_face_z.id, boundary_name="sym_z0"), "sym_z0")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id],
            boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]
        ), "fan_inlet")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.bottom_face_x.id],
            boundary_name="outlet_x",
            temperature=f"{air_temp}cel"
        ), "outlet_x")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.top_face_z.id],
            boundary_name="outlet_z",
            temperature=f"{air_temp}cel"
        ), "outlet_z")
        return region

    # 풀모델: region 전방향
    region = _fresh_region(x_pos=100.0, y_pos=100.0, z_pos=100.0,
                           x_neg=100.0, y_neg=100.0, z_neg=100.0)
    fan_config = str(df.get("fan_config", pd.Series(["dual"])).iloc[0])
    if fan_config == "dual":
        # 양방향 팬 (냉각 스펙: +-y 양측 유입, 배기 +-x/+-z) - 1/8 모델과 동일 물리
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet_pos",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet_pos")
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.bottom_face_y.id], boundary_name="fan_inlet_neg",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet_neg")
        for face, nm in [(region.top_face_x, "outlet_xp"), (region.bottom_face_x, "outlet_xn"),
                         (region.top_face_z, "outlet_zp"), (region.bottom_face_z, "outlet_zn")]:
            _bc(ipk.assign_pressure_free_opening(
                assignment=[face.id], boundary_name=nm, temperature=f"{air_temp}cel"), nm)
    else:
        # 단방향 팬 (+y -> -y)
        _bc(ipk.assign_velocity_free_opening(
            assignment=[region.top_face_y.id], boundary_name="fan_inlet",
            temperature=f"{air_temp}cel",
            velocity=["0m_per_sec", f"-{fan_v}m_per_sec", "0m_per_sec"]), "fan_inlet")
        _bc(ipk.assign_pressure_free_opening(
            assignment=[region.bottom_face_y.id], boundary_name="outlet",
            temperature=f"{air_temp}cel"), "outlet")

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

    mode = str(df["thermal_symmetry"].iloc[0])
    eighth = mode == "eighth"

    ipk = sim.project.create_design(name="icepak_thermal", solver="icepak",
                                    solution="SteadyState TemperatureAndFlow")
    sim.design_thermal = ipk

    set_design_variables(ipk, sim.input_df)
    _create_thermal_materials(ipk, df)
    objs = _build_geometry(ipk, sim, eighth=eighth, mode=mode)
    probe_sheets = _create_probe_sheets(ipk, df, objs, eighth=eighth, mode=mode)
    _assign_losses(ipk, sim, objs, eighth=eighth, mode=mode)
    _assign_boundaries(ipk, sim, objs, eighth=eighth, mode=mode)

    # 서멀패드 메시 해상 강제: 패드(2mm)가 메시에 안 잡히면 도체가 고정온도 Al에
    # 수치적으로 직결되어 온도가 플레이트에 고정됨 (풀 도메인에서 실측된 함정)
    try:
        pad_objs = [o.name for o in objs.get("wcp_pads", []) + objs.get("core_pads", [])]
        if pad_objs:
            ipk.mesh.assign_mesh_level({name: 2 for name in pad_objs}, name="pad_mesh_level")
    except Exception as e:
        logging.warning(f"pad mesh op failed: {e}")

    setup = ipk.create_setup(name="ThermalSetup")
    if not setup:
        raise RuntimeError("create_setup returned no ThermalSetup")
    try:
        setup.props["Flow Regime"] = "Turbulent"
        setup.props["Convergence Criteria - Max Iterations"] = int(df["thermal_max_iterations"].iloc[0])
        if eighth:
            # 1/8 대칭의 z대칭은 부력 무시 가정에 기반 -> 중력 비활성
            setup.props["Include Gravity"] = False
        if not setup.update():
            raise RuntimeError("ThermalSetup update returned False")
    except Exception as e:
        raise RuntimeError(f"ThermalSetup configuration failed: {e}") from e

    # analyze() may return None after a successful blocking solve. Do not query
    # the setup completion property here: it uses the same unreliable report API that
    # caused completed thermal solves to be launched repeatedly. Field Summary
    # data below is the completion gate.
    import time as _time
    solve_attempts = 1
    analyze_call_ok = False
    analyze_return_false = False
    try:
        analyze_result = ipk.analyze(cores=sim.NUM_CORE)
        analyze_call_ok = True
        analyze_return_false = analyze_result is False
        if analyze_return_false:
            logging.warning(
                "[thermal] analyze returned False; validating the solve from Field Summary data."
            )
    except Exception as e:
        logging.exception(f"[thermal] analyze invocation failed: {e}")
        try:
            msgs = ipk.odesktop.GetMessages(sim.PROJECT_NAME, ipk.design_name, 0)
            for m in list(msgs)[-20:]:
                logging.warning(f"[AEDT] {m}")
        except Exception:
            pass
    try:
        sim.save_project()
    except Exception:
        pass

    # ---- 온도 추출 (필드 계산기 직접 평가 - 리포트 기계 미사용) ----
    # 프로브 시트 (회귀학습용 주력 데이터: 위치 고정, 보간값이라 메시 스파이크에 강함)
    # + 그룹별 체적 평균/최대
    native_ipk = _native_solver(ipk)
    try:
        _activate_thermal_design(ipk)
        solution = native_ipk.existing_analysis_sweeps[0]
    except Exception:
        solution = "ThermalSetup : SteadyState"

    def _fresh_fields_reporter():
        """pyaedt 0.22 핸들 무효화 대응: 네이티브로 활성 디자인 재획득 후
        FieldsReporter 모듈을 직접 얻는다 ('NoneType'.CalcStack / gRPC ClcEval 실패의 근원)"""
        _, od = _activate_thermal_design(ipk)
        return od.GetModule("FieldsReporter")

    def _post_of(app):
        """래퍼에 따라 .post가 함수인 경우(실측: 'function' object has no attribute
        'get_scalar_field_value') 호출해서 실제 PostProcessor를 얻는다"""
        po = app.post
        if callable(po) and not hasattr(po, "get_scalar_field_value"):
            po = po()
        return po

    def _eval_temp(obj, op):
        """오브젝트/시트의 Temp Maximum/Mean 스칼라 평가.
        1차: 계산기 직접 호출 (로컬/윈도우 검증됨)
        2차: pyaedt get_scalar_field_value (리눅스 gRPC에서 ClcEval 실패 사례 폴백)"""
        is3d = getattr(obj, "is3d", True)
        try:
            ofr = _fresh_fields_reporter()
            ofr.CalcStack("clear")
            ofr.EnterQty("Temp")
            if is3d:
                ofr.EnterVol(obj.name)
            else:
                ofr.EnterSurf(obj.name)
            ofr.CalcOp("Maximum" if op == "max" else "Mean")
            ofr.ClcEval(solution, [])
            value = float(ofr.GetTopEntryValue(solution, [])[0])
            if not math.isfinite(value):
                raise ValueError(f"non-finite calculator value for {obj.name}: {value}")
            return value
        except Exception:
            v = _post_of(native_ipk).get_scalar_field_value(
                "Temp", scalar_function=("Maximum" if op == "max" else "Mean"),
                solution=solution, object_name=obj.name,
                object_type=("volume" if is3d else "surface"))
            if v is None or v is False:
                raise
            value = float(v)
            if not math.isfinite(value):
                raise ValueError(f"non-finite scalar value for {obj.name}: {value}")
            return value

    temps = {}
    probe = []
    for s in probe_sheets:
        probe.append((s, f"{s.name}_max", "max"))
        probe.append((s, f"{s.name}_mean", "mean"))
    vol_objs = (objs["Tx"] + objs["Rx_main_explicit"] + objs["Rx_main_blocks"]
                + objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
                + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"] + objs["core"])
    for o in vol_objs:
        probe.append((o, f"T_mean_{o.name}", "mean"))
        probe.append((o, f"T_max_{o.name}", "max"))

    # ---- 1차: Field Summary 일괄 (호출 1회, GUI/리눅스 공통 신뢰 경로) ----
    # 계산기(ClcEval) 오브젝트당 호출은 gRPC에서 상습 실패 + 에러 폭탄 유발이라 폴백으로 강등
    # (2026-07-10 GUI 재현: 계산기/스칼라 전멸 후 field summary가 12/12 구조)
    def _field_summary_bulk(entries):
        fs = _post_of(native_ipk).create_field_summary()
        seen = set()
        for obj, col, op in entries:
            is3d = getattr(obj, "is3d", True)
            key = (obj.name, is3d)
            if key in seen:
                continue
            seen.add(key)
            fs.add_calculation("Object", "Volume" if is3d else "Surface",
                               obj.name, "Temperature")
        df_fs = fs.get_field_summary_data(setup=solution, pandas_output=True)
        if df_fs is None or isinstance(df_fs, bool) or not hasattr(df_fs, "columns") or not len(df_fs):
            raise RuntimeError(f"field summary returned {type(df_fs).__name__} (no data)")
        # 컬럼: Entity/Geometry/Quantity/Min/Max/Mean ... (버전에 따라 대소문자 상이)
        cols = {str(c).strip().lower(): c for c in df_fs.columns}
        name_c = cols.get("geometry name", cols.get("entity name", list(df_fs.columns)[2]))
        got = {}
        for obj, col, op in entries:
            row = df_fs[df_fs[name_c].astype(str) == str(obj.name)]
            if not len(row):
                continue
            want = cols.get("max" if op == "max" else "mean")
            if want is None:
                continue
            try:
                value = float(row.iloc[0][want])
                if math.isfinite(value):
                    got[col] = value
            except Exception:
                pass
        return got

    expected_cols = list(dict.fromkeys(col for _, col, _ in probe))
    field_summary_attempts = 0
    for attempt in range(1, 4):
        missing_entries = [entry for entry in probe if entry[1] not in temps]
        if not missing_entries:
            break
        field_summary_attempts = attempt
        try:
            _activate_thermal_design(ipk)
            temps.update(_field_summary_bulk(missing_entries))
        except Exception as e:
            logging.warning(f"[thermal] field summary attempt {attempt}/3 failed: {e}")
        if all(col in temps for col in expected_cols):
            break
        if attempt < 3:
            _time.sleep(10)
    n_fs = len(temps)

    # ---- 2차: Windows에서만 누락분 계산기 폴백 ----
    # Linux gRPC의 CalcStack/get_scalar_field_value는 개체별로 연속 실패하여
    # 태스크를 오래 붙잡는다. Windows에서도 필수 max 항목을 우선하고
    # 최대 16개 항목만 시도한다.
    n_calc = 0
    calc_attempts = 0
    if os.name == "nt":
        missing_by_col = {col: (obj, col, op) for obj, col, op in probe if col not in temps}
        fallback_entries = sorted(
            missing_by_col.values(),
            key=lambda entry: (
                0 if entry[1].startswith("T_max_") else (1 if entry[2] == "max" else 2),
                entry[1],
            ),
        )[:16]
        for obj, col, op in fallback_entries:
            calc_attempts += 1
            try:
                temps[col] = _eval_temp(obj, op)
                n_calc += 1
            except Exception:
                pass
    elif any(col not in temps for col in expected_cols):
        logging.warning("[thermal] Linux calculator fallback skipped; Field Summary remains authoritative.")

    missing_cols = [col for col in expected_cols if col not in temps]
    n_fail = len(missing_cols)
    logging.warning(f"[thermal] extraction: field-summary {n_fs}, calculator {n_calc}, "
                    f"failed {n_fail} / total {len(probe)}")

    def _group_max(group_objects):
        # A partial maximum can silently understate component temperature. Require
        # every modeled maximum for the physical group before emitting its summary.
        cols = list(dict.fromkeys(f"T_max_{obj.name}" for obj in group_objects))
        vals = [temps[col] for col in cols if col in temps and math.isfinite(float(temps[col]))]
        return max(vals) if cols and len(vals) == len(cols) else float("nan")

    group_objects = {
        "T_max_Tx": list(objs["Tx"]),
        "T_max_Rx_main": list(objs["Rx_main_explicit"] + objs["Rx_main_blocks"]),
        "T_max_Rx_side": list(
            objs["Rx_side_explicit"] + objs["Rx_side_blocks"]
            + objs["Rx_side2_explicit"] + objs["Rx_side2_blocks"]
        ),
        "T_max_core": list(objs["core"]),
    }
    group_values = {
        key: _group_max(objects) for key, objects in group_objects.items()
    }
    group_present = {key: bool(objects) for key, objects in group_objects.items()}
    group_bits = {
        "T_max_Tx": 1,
        "T_max_Rx_main": 2,
        "T_max_Rx_side": 4,
        "T_max_core": 8,
    }
    required_group_mask = sum(bit for key, bit in group_bits.items() if group_present[key])
    required_group_count = sum(1 for present in group_present.values() if present)
    required_missing_count = sum(
        1 for key, present in group_present.items()
        if present and not math.isfinite(float(group_values[key]))
    )
    required_complete = required_missing_count == 0 and required_group_count > 0
    solution_data_available = n_fs > 0
    solved = analyze_call_ok and solution_data_available and required_complete

    summary = {
        "thermal_solved": [1 if solved else 0],
        "thermal_extraction_complete": [1 if not missing_cols else 0],
        "thermal_missing_count": [len(missing_cols)],
        "thermal_required_missing_count": [required_missing_count],
        # Bit mask: Tx=1, Rx_main=2, Rx_side=4, core=8. Rx_side is optional
        # when N2_side=0 and is then deliberately excluded from the gate.
        "thermal_required_group_mask": [required_group_mask],
        "thermal_required_group_count": [required_group_count],
        "thermal_solve_attempts": [solve_attempts],
        "thermal_analyze_call_ok": [1 if analyze_call_ok else 0],
        "thermal_analyze_return_false": [1 if analyze_return_false else 0],
        "thermal_solution_data_available": [1 if solution_data_available else 0],
        "thermal_field_summary_attempts": [field_summary_attempts],
        "thermal_field_summary_value_count": [n_fs],
        "thermal_calculator_attempts": [calc_attempts],
        "T_max_Tx": [group_values["T_max_Tx"]],
        "T_max_Rx_main": [group_values["T_max_Rx_main"]],
        "T_max_Rx_side": [group_values["T_max_Rx_side"]],
        "T_max_core": [group_values["T_max_core"]],
    }
    if not solved:
        logging.error(
            "[thermal] validation failed: field-summary-data=%s, required-missing=%d, "
            "missing-total=%d, analyze-call-ok=%s",
            solution_data_available,
            required_missing_count,
            len(missing_cols),
            analyze_call_ok,
        )
    # 개별 값도 함께 저장
    for col in expected_cols:
        summary[col] = [temps.get(col, float("nan"))]

    sim.df_thermal = pd.DataFrame(summary)
    return sim.df_thermal
