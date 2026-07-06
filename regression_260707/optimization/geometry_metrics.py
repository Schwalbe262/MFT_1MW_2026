"""
설계 후보의 해석적 기하 지표 (NSGA-2 목적함수용).

부피 정의: 전체 외곽 박스 X x Y x Z [리터] (기존 260515 calculate_volume과 동일 개념,
260706 스키마로 이식). 파생 컬럼(sl*, nwl*, nwb*)은 validation_check 결과(df_plus) 기준.
"""


def bounding_box_lit(row):
    """단일 설계(dict 또는 Series, df_plus 파생 포함)의 외곽 박스 부피 [L]과 치수 [mm]"""
    l1 = float(row["l1"]); l2 = float(row["l2"]); h1 = float(row["h1"])
    w1 = float(row["w1"])
    N2s = int(row["N2_side"])

    core_x = 4 * l1 + 2 * l2
    # 중심 권선 외곽 (1차가 최외곽)
    center_x = float(row["sl1_main_x"]) + 2 * float(row["nwl1_main"])
    center_y = float(row["sl1_main_y"]) + 2 * float(row["nwb1_main_y"])
    x_candidates = [core_x, center_x]
    y_candidates = [w1, center_y]

    if N2s > 0:
        off = l1 + l2 + l1 / 2
        side_x_out = off + float(row["sl2_side_x"]) / 2 + float(row["nwl2_side"])
        side_y_out = float(row["sl2_side_y"]) / 2 + float(row["nwl2_side"])
        x_candidates.append(2 * side_x_out)
        y_candidates.append(2 * side_y_out)

    X = max(x_candidates)
    Y = max(y_candidates)
    Z = h1 + 2 * l1  # 코어 높이가 항상 최대 (권선 높이 nwh < h1)

    vol_lit = X * Y * Z * 1e-6  # mm^3 -> L
    return vol_lit, (X, Y, Z)
