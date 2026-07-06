import math

# 단면(터미널 시트) 생성은 기존 로직 그대로 재사용
# (단면은 YZ/ZX 평면 = 직선 구간에서 잘리므로 코너 라운드와 무간섭)
from module.modeling import create_coil_section


def create_core(design, name="core", core_material="ferrite", n_group=3,
                plate_material="aluminum", pad_material="thermal_pad",
                plate_on=True, pad_on=True, plate_color=None, pad_color=None):
    """
    설계도면260706 반영 코어 생성.

    코어를 y방향(깊이)으로 n_group개 조로 분할하고, 각 조 사이/바깥에
    콜드플레이트 조립체 (n_group+1)개를 삽입한다.
        w1 = n_group * d + (n_group+1) * core_plate_t   (d: 코어 1조 깊이)
    즉 디자인 변수 w1은 "코어 + 콜드플레이트 전체 깊이"를 의미한다.

    콜드플레이트 조립체(총 core_plate_t)는 실제 구조대로
    서멀패드(core_plate_pad_t) | 알루미늄(core_plate_t - 2*pad) | 서멀패드(pad)
    3층으로 만들어 코어 면에 밀착시킨다. core_plate_pad_t = 0 이면 단일 판.

    코어 조와 플레이트/패드 모두 동일한 E자 실루엣(창 2개 subtract)으로 만든다.
    (권선이 창을 통과하므로 통짜 판이면 권선과 간섭)

    Returns:
        (core_objs, plate_objs, pad_objs)  - plate_objs는 알루미늄 판만
    """
    d_expr = f"((w1 - {n_group + 1}*core_plate_t)/{n_group})"

    core_objs = []
    plate_objs = []
    pad_objs = []
    bodies = []

    for i in range(n_group):
        y0 = f"(-w1/2 + {i + 1}*core_plate_t + {i}*{d_expr})"
        core = design.modeler.create_box(
            origin=["-(4*l1+2*l2)/2", y0, "-(h1+2*l1)/2"],
            sizes=["4*l1+2*l2", d_expr, "h1+2*l1"],
            name=f"{name}_{i + 1}",
            material=core_material
        )
        core_objs.append(core)
        bodies.append(core)

    if plate_on:
        for i in range(n_group + 1):
            y0 = f"(-w1/2 + {i}*(core_plate_t + {d_expr}))"

            # 서멀패드 | 알루미늄 | 서멀패드 (조립체 전체 두께 = core_plate_t)
            if pad_on:
                layers = [
                    (f"({y0})", "core_plate_pad_t", pad_material, f"{name}_plate_pad_{i + 1}_a"),
                    (f"({y0} + core_plate_pad_t)", "(core_plate_t - 2*core_plate_pad_t)", plate_material, f"{name}_plate_{i + 1}"),
                    (f"({y0} + core_plate_t - core_plate_pad_t)", "core_plate_pad_t", pad_material, f"{name}_plate_pad_{i + 1}_b"),
                ]
            else:
                layers = [
                    (f"({y0})", "core_plate_t", plate_material, f"{name}_plate_{i + 1}"),
                ]
            for y_start, t_expr, mat, obj_name in layers:
                obj = design.modeler.create_box(
                    origin=["-(4*l1+2*l2)/2", y_start, "-(h1+2*l1)/2"],
                    sizes=["4*l1+2*l2", t_expr, "h1+2*l1"],
                    name=obj_name,
                    material=mat
                )
                if mat == plate_material:
                    if plate_color is not None:
                        obj.color = plate_color
                    plate_objs.append(obj)
                else:
                    if pad_color is not None:
                        obj.color = pad_color
                    pad_objs.append(obj)
                bodies.append(obj)

    # 창 2개를 코어 조 + 플레이트 전체에서 한 번에 subtract
    sub1 = design.modeler.create_box(
        origin=["-l1", "-w1/2", "-h1/2"],
        sizes=["-l2", "w1", "h1"],
        name=f"{name}_sub1",
        material=core_material
    )
    sub2 = design.modeler.create_box(
        origin=["l1", "-w1/2", "-h1/2"],
        sizes=["l2", "w1", "h1"],
        name=f"{name}_sub2",
        material=core_material
    )
    design.modeler.subtract(bodies, [sub1, sub2], keep_originals=False)

    return core_objs, plate_objs, pad_objs


def compute_layer_positions(space_half, coil_width, gaps):
    """
    안쪽 개구 반폭(space_half)에서 시작해 턴 중심선 위치 리스트를 계산.
    gaps: 인접 턴 사이 간격 리스트 (길이 = 턴수 - 1)
    """
    pos = [space_half + coil_width * 0.5]
    for g in gaps:
        pos.append(pos[-1] + coil_width + g)
    return pos


def _rounded_turn_points(x, y, r, z, offset, segments_per_corner=4):
    """
    반폭 (x, y), 코너 반경 r인 라운드 사각 턴의 폴리라인 점/세그먼트 생성.

    segments_per_corner > 0 (기본 4): 코너를 등각 분할한 직선 세그먼트로 근사.
      모든 턴이 동일한 각도 분할(동일 점 개수)을 쓰므로 화면 표시가 균일하고,
      인접 턴의 코너 다각형이 서로 평행(동심)이라 간격이 전 구간 일정하게 유지된다.
      점 개수 = 4*(segments_per_corner+1) + 1 (모든 턴 동일).

    segments_per_corner = 0: 진짜 원호(3점 Arc 세그먼트) 사용. 지오메트리는 정확하나
      AEDT 뷰포트가 오브젝트별로 표시용 근사를 다르게 해 턴마다 달라 보일 수 있다.

    기존 직각 턴과 같은 순서(+x+y 코너에서 시작해 top -> left -> bottom -> right)로 진행.
    """
    if r <= 0 or r >= min(x, y):
        raise ValueError(f"corner radius {r} must be in (0, min(x={x}, y={y}))")

    if segments_per_corner and segments_per_corner > 0:
        n = int(segments_per_corner)
        # 코너별 (중심, 시작각, 끝각) - 경로 진행 방향 순서
        corners = [
            ((-x + r, y - r), 90.0, 180.0),    # C2 (-x, +y)
            ((-x + r, -y + r), 180.0, 270.0),  # C3 (-x, -y)
            ((x - r, -y + r), 270.0, 360.0),   # C4 (+x, -y)
            ((x - r, y - r), 0.0, 90.0),       # C1 (+x, +y)
        ]
        pts_2d = []
        for (cx, cy), th0, th1 in corners:
            for j in range(n + 1):
                th = math.radians(th0 + (th1 - th0) * j / n)
                pts_2d.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        pts_2d.append(pts_2d[0])  # 닫힘

        points = [
            [f"{px}mm + {offset[0]}mm", f"{py}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"]
            for (px, py) in pts_2d
        ]
        return points, None  # 전부 Line 세그먼트

    k = r * (1.0 - math.sqrt(2.0) / 2.0)  # 아크 중간점의 코너로부터 안쪽 거리

    # (px, py) 순회: top edge -> C2 arc -> left edge -> C3 arc -> bottom -> C4 arc -> right -> C1 arc(닫힘)
    pts_2d = [
        (x - r, y),            # S  : top edge 시작 (C1 아크 끝점)
        (-x + r, y),           # A  : top edge 끝 (C2 아크 시작)
        (-x + k, y - k),       # M2 : C2 아크 중간점
        (-x, y - r),           # B  : C2 아크 끝
        (-x, -y + r),          # C  : left edge 끝 (C3 아크 시작)
        (-x + k, -y + k),      # M3 : C3 아크 중간점
        (-x + r, -y),          # D  : C3 아크 끝
        (x - r, -y),           # E  : bottom edge 끝 (C4 아크 시작)
        (x - k, -y + k),       # M4 : C4 아크 중간점
        (x, -y + r),           # F  : C4 아크 끝
        (x, y - r),            # G  : right edge 끝 (C1 아크 시작)
        (x - k, y - k),        # M1 : C1 아크 중간점
        (x - r, y),            # S  : 닫힘
    ]

    points = [
        [f"{px}mm + {offset[0]}mm", f"{py}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"]
        for (px, py) in pts_2d
    ]
    segments = ["Line", "Arc", "Line", "Arc", "Line", "Arc", "Line", "Arc"]
    return points, segments


def create_coil(design, name="coil", window_height=50, window_length=50, window_layer=2,
                N_input=1, width_fill_factor=0.6, space_length=10, space_width=10,
                shape="circle", offset=[0, 0, 0], color=None,
                y_slot_gaps=None, round_corner=False, corner_radius=None,
                corner_segments=4):
    """
    기존 module.modeling.create_coil 확장판 (시그니처/반환값 호환).

    추가 기능:
      - y_slot_gaps: y방향 인접 턴 간격 리스트(길이 window_layer-1).
        None이면 기존처럼 x와 동일 등간격. 1차 권선 냉각 플레이트 슬롯처럼
        y방향만 벌어지는 배치에 사용.
      - round_corner / corner_radius: 턴 모서리 라운드 처리 on/off.
        반경은 가장 안쪽 턴 기준이며 바깥 턴은 x방향 턴 피치만큼 증가
        (r_i = corner_radius + (x_i - x_0), 동심 유지).
      - corner_segments: 코너 등각 분할 수 (기본 4, 모든 턴 동일 점 개수).
        많을수록 매끈하지만 메시/해석 부담 증가. 0이면 진짜 원호(Arc 세그먼트) 사용.
    """
    N = int(N_input)
    shape = shape.lower()

    if shape == "circle":
        coil_width = window_length * width_fill_factor / window_layer
        coil_height = coil_width

        if window_layer > 1:
            coil_gap_x = (window_length - (coil_width * window_layer)) / (window_layer - 1)
        else:
            coil_gap_x = 0

        if N > 1:
            coil_gap_z = (window_height - (coil_height * N)) / (N - 1)
        else:
            coil_gap_z = 0

    elif shape == "rectangle":
        coil_width = window_length * width_fill_factor / window_layer
        coil_height = window_height / N

        if window_layer > 1:
            coil_gap_x = (window_length - (coil_width * window_layer)) / (window_layer - 1)
        else:
            coil_gap_x = 0

        if N > 1:
            coil_gap_z = (window_height - (coil_height * N)) / (N - 1)
        else:
            coil_gap_z = 0

    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # x방향은 기존과 동일한 등간격 배치
    x_gaps = [coil_gap_x] * (window_layer - 1)

    # y방향은 슬롯 간격 리스트가 주어지면 독립 배치 (1차 권선 냉각판 슬롯)
    if y_slot_gaps is None:
        y_gaps = x_gaps
    else:
        if len(y_slot_gaps) != window_layer - 1:
            raise ValueError(
                f"y_slot_gaps length must be window_layer-1 ({window_layer - 1}), got {len(y_slot_gaps)}"
            )
        y_gaps = list(y_slot_gaps)

    x_pos = compute_layer_positions(space_length / 2, coil_width, x_gaps)
    y_pos = compute_layer_positions(space_width / 2, coil_width, y_gaps)

    z_pos = []
    for i in range(N):
        z_pos.append(window_height / 2 - coil_height * (i + 0.5) - coil_gap_z * i)

    windings = []

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        for j, z in enumerate(z_pos):

            if round_corner:
                if corner_radius is None:
                    raise ValueError("corner_radius must be given when round_corner=True")
                r_i = corner_radius + (x - x_pos[0])
                points, segments = _rounded_turn_points(x, y, r_i, z, offset,
                                                        segments_per_corner=corner_segments)

                polyline_kwargs = dict(
                    points=points,
                    name=f"{name}_{i}_{j}",
                    material="copper",
                    xsection_orient="Auto",
                    xsection_type=shape,
                    xsection_width=coil_width,
                    xsection_height=coil_height,
                    xsection_num_seg=6,
                    xsection_topwidth=coil_width
                )
                if segments is not None:
                    polyline_kwargs["segment_type"] = segments
                winding = design.modeler.create_polyline(**polyline_kwargs)
            else:
                points = []
                points.append([f"{x}mm + {offset[0]}mm", f"{y}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
                points.append([f"{-x}mm + {offset[0]}mm", f"{y}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
                points.append([f"{-x}mm + {offset[0]}mm", f"{-y}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
                points.append([f"{x}mm + {offset[0]}mm", f"{-y}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
                points.append([f"{x}mm + {offset[0]}mm", f"{y}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])

                winding = design.modeler.create_polyline(
                    points=points,
                    name=f"{name}_{i}_{j}",
                    material="copper",
                    xsection_orient="Auto",
                    xsection_type=shape,
                    xsection_width=coil_width,
                    xsection_height=coil_height,
                    xsection_num_seg=6,
                    xsection_topwidth=coil_width
                )

            if color is not None:
                winding.color = color
            windings.append(winding)

    return windings, N, coil_width, coil_height, coil_gap_x, coil_gap_z


def create_winding_cooling_plates(design, name, space_width, coil_width, y_gaps, slot_indices,
                                  wcp_len_x, wcp_t, pad_t, height,
                                  plate_material="aluminum", pad_material="thermal_pad",
                                  plate_color=None, pad_color=None, offset=[0, 0, 0]):
    """
    1차 권선 y방향 측면 슬롯에 들어가는 권선 냉각 플레이트 조립체 생성.
    슬롯은 y+ / y- 양측 대칭이므로 슬롯당 1조, 총 len(slot_indices)*2 조.

    실제 구조 반영: 슬롯(y방향 간격 = wcp_t)을
        서멀패드(pad_t) | 알루미늄(wcp_t - 2*pad_t) | 서멀패드(pad_t)
    로 꽉 채워 양쪽 도체 면에 밀착시킨다. pad_t = 0 이면 알루미늄 단일 판.

    Parameters:
        space_width  : 권선 안쪽 개구 y방향 전체 폭 (sl1_main_y)
        coil_width   : 1차 도체 두께 (cw1)
        y_gaps       : y방향 인접 턴 간격 리스트 (create_coil에 넘긴 것과 동일)
        slot_indices : 냉각판이 들어가는 gap 인덱스 리스트 (예: [0, N-2])
        wcp_len_x    : 플레이트 x방향 폭
        wcp_t        : 조립체 전체 두께 (y방향, 슬롯 간격과 동일)
        pad_t        : 서멀패드 두께 (편측)
        height       : 플레이트 z방향 높이 (권선 높이와 동일, 중심 대칭)

    Returns:
        (plate_objs, pad_objs)  - plate_objs는 알루미늄 판만
    """
    y_pos = compute_layer_positions(space_width / 2, coil_width, y_gaps)

    plates = []
    pads = []
    for k, gi in enumerate(slot_indices):
        # 슬롯 시작 = 턴 gi 바깥 면 (도체 면에 밀착)
        y_in = y_pos[gi] + coil_width / 2

        if pad_t > 0:
            layers = [
                (y_in, pad_t, pad_material, f"{name}_pad_{k + 1}_in"),
                (y_in + pad_t, wcp_t - 2 * pad_t, plate_material, f"{name}_{k + 1}"),
                (y_in + wcp_t - pad_t, pad_t, pad_material, f"{name}_pad_{k + 1}_out"),
            ]
        else:
            layers = [
                (y_in, wcp_t, plate_material, f"{name}_{k + 1}"),
            ]

        for sign, tag in [(1, "p"), (-1, "n")]:
            for y_start, t, mat, base_name in layers:
                # y- 측은 대칭 위치 (박스 origin은 항상 작은 y 값)
                y_origin = y_start if sign > 0 else -(y_start + t)
                obj = design.modeler.create_box(
                    origin=[
                        f"{-wcp_len_x / 2}mm + {offset[0]}mm",
                        f"{y_origin}mm + {offset[1]}mm",
                        f"{-height / 2}mm + {offset[2]}mm"
                    ],
                    sizes=[f"{wcp_len_x}mm", f"{t}mm", f"{height}mm"],
                    name=f"{base_name}_{tag}",
                    material=mat
                )
                if mat == plate_material:
                    if plate_color is not None:
                        obj.color = plate_color
                    plates.append(obj)
                else:
                    if pad_color is not None:
                        obj.color = pad_color
                    pads.append(obj)

    return plates, pads
