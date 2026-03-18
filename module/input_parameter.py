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

    keys = ["N1", "N2", "N2_main", "N2_side", "w1", "l1", "l2", "h1", "w1c_space_x", "w1w2_space_x", "w2c_space_x",
            "w1c_space_y", "w1w2_space_y", "w2w2_space_y", "w2c_space_y", "w1c_space_z", "w2c_space_z",
            "window_ratio", "wh1", "wh2", "wff1", "wff2"]

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
        N2_side = round(N2 * get_random_value(lower=0, upper=0.6, resolution=0.01))
        N2_main = N2 - N2_side
        
        w1 = get_random_value(lower=300, upper=800, resolution=1)
        l1 = get_random_value(lower=50, upper=100, resolution=1)
        total_length = get_random_value(lower=500, upper=1200, resolution=1)
        l2 = (total_length - 4*l1) / 2
        total_height = get_random_value(lower=500, upper=1000, resolution=1)
        h1 = (total_height - 2*l1)

        w1c_space_x = get_random_value(lower=10, upper=50, resolution=0.1)
        w1w2_space_x = get_random_value(lower=10, upper=50, resolution=0.1)
        w2c_space_x = get_random_value(lower=10, upper=50, resolution=0.1)

        w1c_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        w1w2_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        w2w2_space_y = get_random_value(lower=10, upper=50, resolution=0.1)
        w2c_space_y = get_random_value(lower=10, upper=50, resolution=0.1)

        w1c_space_z = get_random_value(lower=10, upper=50, resolution=0.1)
        w2c_space_z = get_random_value(lower=10, upper=50, resolution=0.1)

        window_ratio = get_random_value(lower=0.3, upper=0.7, resolution=0.01)

        wh1 = get_random_value(lower=0.5, upper=1.0, resolution=0.01)
        wh2 = get_random_value(lower=0.8, upper=1.0, resolution=0.01)

        wff1 = get_random_value(lower=1.0, upper=1.0, resolution=0.01)
        wff2 = get_random_value(lower=0.4, upper=0.75, resolution=0.01)

        param_values = [
            N1,
            N2,
            N2_main,
            N2_side,
            w1,
            l1,
            l2,
            h1,
            w1c_space_x,
            w1w2_space_x,
            w2c_space_x,
            w1c_space_y,
            w1w2_space_y,
            w2w2_space_y,
            w2c_space_y,
            w1c_space_z,
            w2c_space_z,
            window_ratio,
            wh1,
            wh2,
            wff1,
            wff2,
        ]
        param_df = pd.DataFrame([param_values], columns=keys)

    return param_df





def set_design_variables(design, input_parameter):
    """
    주어진 파라미터 딕셔너리를 사용하여 Ansys 디자인 변수를 설정합니다.
    """
    units = {
        "w1" : "mm", "l1" : "mm", "l2" : "mm", "h1" : "mm", 
        "w1c_space_x" : "mm", "w1w2_space_x" : "mm", "w2c_space_x" : "mm",
        "w1c_space_y" : "mm", "w1w2_space_y" : "mm", "w2w2_space_y" : "mm", "w2c_space_y" : "mm",
        "w1c_space_z" : "mm", "w2c_space_z" : "mm",
        "window_ratio" : "mm",
    }

    for key, value in input_parameter.items():
        # Ansys 디자인에 변수를 설정합니다.
        value = input_parameter.iloc[0][key]
        unit = units.get(key, "")
        design.set_variable(variable_name=key, value=value, unit=unit)



def validation_check(input_df) :

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

    coil_width1 = nwl1 * input_df_copy["wff1"].iloc[0]
    coil_width2 = (nwl2 * input_df_copy["wff2"].iloc[0]) / input_df_copy["N2"].iloc[0]
    input_df_copy["coil_width1"] = [coil_width1]
    input_df_copy["coil_width2"] = [coil_width2]

    
    # 2차 측의 권선 간 간격
    coil_gap_layer = (nwl2 - (coil_width2 * input_df_copy["N2"].iloc[0])) / (input_df_copy["N2"].iloc[0] - 1)
    input_df_copy["coil_gap_layer"] = [coil_gap_layer]
    # 1차 측의 높이 방향 권선 간 간격
    coil_gap_height = (nwh1 - (coil_width1 * input_df_copy["N1"].iloc[0])) / (input_df_copy["N1"].iloc[0] - 1)
    input_df_copy["coil_gap_height"] = [coil_gap_height]
    # 1차 측의 fill factor
    fill_factor = coil_width1*input_df_copy["N1"].iloc[0] / nwh1
    input_df_copy["fill_factor"] = [fill_factor]


    # window의 내부 gap 공간 크기 (space length / space width)
    sl1 = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w1c_space_y"].iloc[0]
    sw1 = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w1c_space_x"].iloc[0]
    input_df_copy["sl1"] = [sl1]
    input_df_copy["sw1"] = [sw1]

    sl2_main = 2*input_df_copy["l1"].iloc[0] + 2*input_df_copy["w2c_space_y"].iloc[0] + 2*input_df_copy["w1w2_space_y"].iloc[0] + 2*input_df_copy["nwl1"].iloc[0]
    sw2_main = input_df_copy["w1"].iloc[0] + 2*input_df_copy["w2c_space_x"].iloc[0] + 2*input_df_copy["w1w2_space_x"].iloc[0] + 2*input_df_copy["nwl1"].iloc[0]
    input_df_copy["sl2_main"] = [sl2_main]
    input_df_copy["sw2_main"] = [sw2_main]

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
    if coil_width1 < 20 :
        result = False
    if coil_width2 < 0.5 :
        result = False
    if coil_gap_layer < 0.3 :
        result = False
    if coil_gap_height < 3 :
        result = False

       

    return result, input_df_copy


