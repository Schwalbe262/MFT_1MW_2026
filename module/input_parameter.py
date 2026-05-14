import numpy as np
import pandas as pd
import math


def get_random_value(lower=None, upper=None, resolution=None):

    if isinstance(resolution, float):
        # 소수점 precision 계산
        str_res = f"{resolution:.16f}".rstrip('0').rstrip('.')
        if "." in str_res:
            precision = len(str_res.split('.')[-1])
        else:
            precision = 0
    else:
        precision = 0

    # upper도 포함되게끔 endpoint 옵션 사용
    possible_values = np.arange(lower, upper + resolution * 0.5, resolution)  # endpoint 가 없어서, + resolution * 0.5로 포함되게 처리
    if (np.abs(possible_values[-1] - upper) > 1e-8) and (np.abs((possible_values - upper)).min() > 1e-8):
        # upper가 반드시 들어가게끔
        possible_values = np.append(possible_values, upper)
        possible_values = np.unique(possible_values)
    chosen = np.random.choice(possible_values)
    value = round(chosen, precision)

    # Int/float 형변환
    if resolution == 1 or resolution == 1.0:
        return int(value)
    return float(value)


def create_input_parameter(param_list=None):

    keys = [
        "N1","N2","N1_main","N1_side","N2_main","N2_side",
        "w1","l1","l2","h1",
        "cc_w2c_space_x","w2c_w1c_space_x","w1c_w2s_space_x","w2s_w1s_space_x","w1s_cs_space_x",
        "cc_w2c_space_y","w2c_w1c_space_y","cs_w1s_space_y","w1s_w2s_space_y",
        "window_ratio","wh1","wh2","wff1","wff2"
    ]

    # param_list not None case (검증 케이스)
    if param_list is not None:

        # pandas로 형변환
        if isinstance(param_list, pd.DataFrame):
            param_df = param_list
        else:
            if isinstance(param_list, (list, tuple)) and len(param_list) > 0:
                first = param_list[0]
                if not isinstance(first, (list, tuple, dict, pd.Series)):
                    param_list = [param_list]
            param_df = pd.DataFrame(param_list, columns=keys)

        # 컬럼 개수 확인
        if param_df.shape[1] != len(keys):
            raise ValueError(f"Input list must have exactly {len(keys)} elements, but got {param_df.shape[1]}.")

    # param_list None case (랜덤 시뮬레이션 케이스)
    else:
        
        N1 = get_random_value(lower=5, upper=10, resolution=1)
        N2 = N1 * 10
        N1_side = round(N1 * get_random_value(lower=0, upper=0.5, resolution=0.01))
        N1_main = N1 - N1_side
        N2_side = round(N2 * get_random_value(lower=0, upper=0.8, resolution=0.01))
        N2_main = N2 - N2_side

        # N1 = get_random_value(lower=2, upper=2, resolution=1)
        # N2 = N1 * 5
        # N2_side = round(N2 * get_random_value(lower=0, upper=0.6, resolution=0.01))
        # N2_main = N2 - N2_side
        
        w1 = get_random_value(lower=200, upper=800, resolution=1)
        l1 = get_random_value(lower=40, upper=100, resolution=1)
        total_length = get_random_value(lower=500, upper=1200, resolution=1)
        l2 = (total_length - 4*l1) / 2
        total_height = get_random_value(lower=500, upper=1000, resolution=1)
        h1 = (total_height - 2*l1)

        # 메인 코어
        cc_w2c_space_x = get_random_value(lower=10, upper=50, resolution=0.1)
        w2c_w1c_space_x = get_random_value(lower=10, upper=50, resolution=0.1)
        w1c_w2s_space_x = get_random_value(lower=10, upper=100, resolution=0.1)
        w2s_w1s_space_x = get_random_value(lower=10, upper=50, resolution=0.1)
        w1s_cs_space_x = get_random_value(lower=10, upper=50, resolution=0.1)

        cc_w2c_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        w2c_w1c_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        cs_w1s_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        w1s_w2s_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        
        # 1차 측과 2차 측의 비율
        window_ratio = get_random_value(lower=0.3, upper=0.7, resolution=0.01)

        # 높이 비율율
        wh1 = get_random_value(lower=0.8, upper=0.95, resolution=0.01)
        wh2 = get_random_value(lower=0.5, upper=0.95, resolution=0.01)

        wff1 = get_random_value(lower=0.4, upper=0.8, resolution=0.01)
        wff2 = get_random_value(lower=0.4, upper=0.75, resolution=0.01)

        param_values = [
            N1,
            N2,
            N1_main,
            N1_side,
            N2_main,
            N2_side,
            w1,
            l1,
            l2,
            h1,
            cc_w2c_space_x,
            w2c_w1c_space_x,
            w1c_w2s_space_x,
            w2s_w1s_space_x,
            w1s_cs_space_x,
            cc_w2c_space_y,
            w2c_w1c_space_y,
            cs_w1s_space_y,
            w1s_w2s_space_y,
            window_ratio,
            wh1, # 높이 비율율
            wh2,
            wff1,
            wff2
        ]
        param_df = pd.DataFrame([param_values], columns=keys)

    return param_df





def set_design_variables(design, input_parameter):
    """
    주어진 파라미터 딕셔너리를 사용하여 Ansys 디자인 변수를 설정합니다.
    """
    units = {
        "w1" : "mm", "l1" : "mm", "l2" : "mm", "h1" : "mm", 
        "cc_w2c_space_x" : "mm", "w2c_w1c_space_x" : "mm", "w1c_w2s_space_x" : "mm", "w2s_w1s_space_x" : "mm", "w1s_cs_space_x" : "mm",
        "cc_w2c_space_y" : "mm", "w2c_w1c_space_y" : "mm", "cs_w1s_space_y" : "mm", "w1s_w2s_space_y" : "mm",
        "w1c_space_z" : "mm", "w2c_space_z" : "mm"
    }

    for key, value in input_parameter.items():
        # Ansys 디자인에 변수를 설정합니다.
        value = input_parameter.iloc[0][key]
        unit = units.get(key, "")
        design.set_variable(variable_name=key, value=value, unit=unit)



def validation_check(input_df) :

    result = True

    inp = input_df.copy()

    # space 간격을 제외한 순수 net winding length
    nwl_x = inp["l2"].iloc[0] - inp["cc_w2c_space_x"].iloc[0] - inp["w2c_w1c_space_x"].iloc[0] - inp["w1c_w2s_space_x"].iloc[0] - inp["w2s_w1s_space_x"].iloc[0] - inp["w1s_cs_space_x"].iloc[0]
    inp["nwl_x"] = [nwl_x]

    # number of turns

    N1_main = inp["N1_main"].iloc[0]
    N1_side = inp["N1_side"].iloc[0]
    N2_main = inp["N2_main"].iloc[0]
    N2_side = inp["N2_side"].iloc[0]
    N1 = N1_main + N1_side
    N2 = N2_main + N2_side

    # net window length
    nwl1 = nwl_x * inp["window_ratio"].iloc[0]
    nwl2 = nwl_x - nwl1

    N1_main_turns = N1_main
    N1_side_turns = N1_side
    N2_main_turns = N2_main
    N2_side_turns = N2_side
    N1_main_gaps = N1_main - 1 if N1_main > 0 else 0
    N1_side_gaps = N1_side - 1 if N1_side > 0 else 0
    N2_main_gaps = N2_main - 1 if N2_main > 0 else 0
    N2_side_gaps = N2_side - 1 if N2_side > 0 else 0

    # winding thickness
    cw1 = (nwl1 * inp["wff1"].iloc[0]) / (N1_main_turns + N1_side_turns)
    cw2 = (nwl2 * inp["wff2"].iloc[0]) / (N2_main_turns + N2_side_turns)
    inp["cw1"] = [cw1]
    inp["cw2"] = [cw2]

    # winding gap (권선 간 간격)
    coil_gap_layer1 = (nwl1 * (1-inp["wff1"].iloc[0])) / (N1_main_gaps + N1_side_gaps)
    coil_gap_layer2 = (nwl2 * (1-inp["wff2"].iloc[0])) / (N2_main_gaps + N2_side_gaps)
    inp["coil_gap_layer1"] = [coil_gap_layer1]
    inp["coil_gap_layer2"] = [coil_gap_layer2]

    nwl1_main = cw1 * N1_main_turns + coil_gap_layer1 * N1_main_gaps
    nwl1_side = cw1 * N1_side_turns + coil_gap_layer1 * N1_side_gaps
    nwl2_main = cw2 * N2_main_turns + coil_gap_layer2 * N2_main_gaps
    nwl2_side = cw2 * N2_side_turns + coil_gap_layer2 * N2_side_gaps

    inp["nwl1_main"] = [nwl1_main]
    inp["nwl1_side"] = [nwl1_side]
    inp["nwl2_main"] = [nwl2_main]
    inp["nwl2_side"] = [nwl2_side]

    # net window height
    nwh1 = inp["h1"].iloc[0] * inp["wh1"].iloc[0]
    nwh2 = inp["h1"].iloc[0] * inp["wh2"].iloc[0]
    inp["nwh1"] = [nwh1]
    inp["nwh2"] = [nwh2]

    # 권선과 코어 간 높이 방향 간격격
    h_gap1 = (inp["h1"].iloc[0] - nwh1) / 2
    h_gap2 = (inp["h1"].iloc[0] - nwh2) / 2
    inp["h_gap1"] = [h_gap1]
    inp["h_gap2"] = [h_gap2]

    # wff 계산
    wff1_main = cw1 * N1_main_turns / (cw1 * N1_main_turns + coil_gap_layer1 * N1_main_gaps)
    wff1_side = 0 if N1_side_turns == 0 else cw1 * N1_side_turns / (cw1 * N1_side_turns + coil_gap_layer1 * N1_side_gaps)
    wff2_main = cw2 * N2_main_turns / (cw2 * N2_main_turns + coil_gap_layer2 * N2_main_gaps)
    wff2_side = 0 if N2_side_turns == 0 else cw2 * N2_side_turns / (cw2 * N2_side_turns + coil_gap_layer2 * N2_side_gaps)
    inp["wff1_main"] = [wff1_main]
    inp["wff1_side"] = [wff1_side]
    inp["wff2_main"] = [wff2_main]
    inp["wff2_side"] = [wff2_side]


    # 각 권선의 전체 length 및 width 계산
    sl2_main_x = 2*inp["l1"].iloc[0] + 2*inp["cc_w2c_space_x"].iloc[0]
    sl2_main_y = inp["w1"].iloc[0] + 2*inp["cc_w2c_space_y"].iloc[0]
    sl1_main_x = sl2_main_x + 2*inp["nwl2_main"].iloc[0] + 2*inp["w2c_w1c_space_x"].iloc[0]
    sl1_main_y = sl2_main_y + 2*inp["nwl2_main"].iloc[0] + 2*inp["w2c_w1c_space_y"].iloc[0]

    sl1_side_x = inp["l1"].iloc[0] + 2*inp["w1s_cs_space_x"].iloc[0]
    sl1_side_y = inp["w1"].iloc[0] + 2*inp["cs_w1s_space_y"].iloc[0]
    sl2_side_x = sl1_side_x + 2*inp["nwl1_side"].iloc[0] + 2*inp["w2s_w1s_space_x"].iloc[0]
    sl2_side_y = sl1_side_y + 2*inp["nwl1_side"].iloc[0] + 2*inp["w1s_w2s_space_y"].iloc[0]

    inp["sl2_main_x"] = [sl2_main_x]
    inp["sl2_main_y"] = [sl2_main_y]
    inp["sl1_main_x"] = [sl1_main_x]
    inp["sl1_main_y"] = [sl1_main_y]
    inp["sl1_side_x"] = [sl1_side_x]
    inp["sl1_side_y"] = [sl1_side_y]
    inp["sl2_side_x"] = [sl2_side_x]
    inp["sl2_side_y"] = [sl2_side_y]


    if nwl1 < 0 :
        result = False
    if nwl2 < 0 :
        result = False
    if nwh1 < 0 :
        result = False
    if nwh2 < 0 :
        result = False
    if cw1 < 1.0 or cw1 > 10 :
        result = False
    if cw2 < 0.6 :
        result = False
    if coil_gap_layer1 < 0.3 :
        result = False
    if coil_gap_layer2 < 0.3 :
        result = False


       

    return result, inp








def validation_check_old(input_df) :

    result = True

    input_df_copy = input_df.copy()

    if input_df_copy["N2_side"].iloc[0] != 0:
        nwl_t = input_df_copy["l2"].iloc[0] - input_df_copy["w1c_space_y"].iloc[0] - input_df_copy["w1w2_space_y"].iloc[0] - input_df_copy["w2c_space_y"].iloc[0] - input_df_copy["w2w2_space_y"].iloc[0]
    else:
        nwl_t = input_df_copy["l2"].iloc[0] - input_df_copy["w1c_space_y"].iloc[0] - input_df_copy["w1w2_space_y"].iloc[0] - input_df_copy["w2c_space_y"].iloc[0]
    input_df_copy["nwl_t"] = [nwl_t]

    N2_main = input_df_copy["N2_main"].iloc[0]
    N2_side = input_df_copy["N2_side"].iloc[0]

    # net window length
    nwl1 = nwl_t * input_df_copy["window_ratio"].iloc[0]
    nwl2 = nwl_t - nwl1
    nwl2_main = nwl2 * (N2_main) / (N2_main + N2_side)
    nwl2_side = nwl2 * (N2_side) / (N2_main + N2_side)
    input_df_copy["nwl1"] = [nwl1]
    input_df_copy["nwl2"] = [nwl2]
    input_df_copy["nwl2_main"] = [nwl2_main]
    input_df_copy["nwl2_side"] = [nwl2_side]
    
    # net window height
    nwh1 = (input_df_copy["h1"].iloc[0] - 2*input_df_copy["w1c_space_z"].iloc[0]) * input_df_copy["wh1"].iloc[0]
    nwh2 = (input_df_copy["h1"].iloc[0] - 2*input_df_copy["w2c_space_z"].iloc[0]) * input_df_copy["wh2"].iloc[0]
    input_df_copy["nwh1"] = [nwh1]
    input_df_copy["nwh2"] = [nwh2]

    # 권선 두께께
    coil_width1 = (nwl1 * input_df_copy["wff1"].iloc[0]) / input_df_copy["N1"].iloc[0]
    coil_width2 = (nwl2 * input_df_copy["wff2"].iloc[0]) / input_df_copy["N2"].iloc[0]
    input_df_copy["coil_width1"] = [coil_width1]
    input_df_copy["coil_width2"] = [coil_width2]

    
    # 권선 간 간격
    coil_gap_layer1 = (nwl1 - (coil_width1 * input_df_copy["N1"].iloc[0])) / (input_df_copy["N1"].iloc[0] - 1)
    coil_gap_layer2 = (nwl2 - (coil_width2 * input_df_copy["N2"].iloc[0])) / (input_df_copy["N2"].iloc[0] - 1)
    input_df_copy["coil_gap_layer1"] = [coil_gap_layer1]
    input_df_copy["coil_gap_layer2"] = [coil_gap_layer2]

    # 1차 측의 높이 방향 권선 간 간격
    coil_gap_height = (nwh1 - (coil_width1 * input_df_copy["N1"].iloc[0])) / (input_df_copy["N1"].iloc[0] - 1)
    input_df_copy["coil_gap_height"] = [coil_gap_height]
    # 1차 측의 fill factor
    fill_factor = coil_width1*input_df_copy["N1"].iloc[0] / nwh1
    input_df_copy["fill_factor"] = [fill_factor]


    # window의 내부 gap 공간 크기 (space length / space width)

    # 1차 측이 안쪽에 있는 케이스 주석 처리리
    # sl1 = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w1c_space_y"].iloc[0]
    # sw1 = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w1c_space_x"].iloc[0]
    # input_df_copy["sl1"] = [sl1]
    # input_df_copy["sw1"] = [sw1]

    # sl2_main = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w1c_space_y"].iloc[0] + 2*input_df_copy["w1w2_space_y"].iloc[0] + 2*input_df_copy["nwl1"].iloc[0]
    # sw2_main = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w1c_space_x"].iloc[0] + 2*input_df_copy["w1w2_space_x"].iloc[0] + 2*input_df_copy["nwl1"].iloc[0]
    # input_df_copy["sl2_main"] = [sl2_main]
    # input_df_copy["sw2_main"] = [sw2_main]

    sl2_main = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w1c_space_y"].iloc[0]
    sw2_main = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w1c_space_x"].iloc[0]
    input_df_copy["sl2_main"] = [sl2_main]
    input_df_copy["sw2_main"] = [sw2_main]

    sl1 = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w1c_space_y"].iloc[0] + 2*input_df_copy["w1w2_space_y"].iloc[0] + 2*input_df_copy["nwl2_main"].iloc[0]
    sw1 = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w1c_space_x"].iloc[0] + 2*input_df_copy["w1w2_space_x"].iloc[0] + 2*input_df_copy["nwl2_main"].iloc[0]
    input_df_copy["sl1"] = [sl1]
    input_df_copy["sw1"] = [sw1]
    

    sl2_side = input_df_copy["l1"].iloc[0] + 2*input_df_copy["w2c_space_y"].iloc[0]
    sw2_side = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w2c_space_x"].iloc[0]
    input_df_copy["sl2_side"] = [sl2_side]
    input_df_copy["sw2_side"] = [sw2_side]




    if nwl_t < 0 :
        result = False
    if nwl1 < 0 :
        result = False
    if nwl2 < 0 :
        result = False
    if nwh1 < 0 :
        result = False
    if nwh2 < 0 :
        result = False
    if coil_width1 < 0.6 :
        result = False
    if coil_width2 < 0.6 :
        result = False
    if coil_gap_layer1 < 0.3 :
        result = False
    if coil_gap_layer2 < 0.3 :
        result = False
    if coil_gap_height < 3 :
        result = False

       

    return result, input_df_copy


