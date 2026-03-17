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

    keys = ["N1", "N2", "N2_side", "w1", "l1", "l2", "h1", "w1c_space_x", "w1w2_space_x", "w2c_space_x",
            "w1c_space_y", "w1w2_space_y", "w2w2_space_y", "w2c_space_y", "w1c_space_z", "w2c_space_z",
            "window_ratio", "wh1", "wh2", "wff1", "wff2", "Nff1", "Nff2"]

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
        N2 = N1
        N2_side = round(N2 * get_random_value(lower=0, upper=0.6, resolution=0.01))
        
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
        wh2 = get_random_value(lower=0.5, upper=1.0, resolution=0.01)

        wff1 = get_random_value(lower=1.0, upper=1.0, resolution=0.01)
        wff2 = get_random_value(lower=0.5, upper=1.0, resolution=0.01)

        Nff1 = get_random_value(lower=0.5, upper=1.0, resolution=0.01)
        Nff2 = get_random_value(lower=1.0, upper=1.0, resolution=0.01)

        param_values = [
            N1,
            N2,
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
            Nff1,
            Nff2
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


