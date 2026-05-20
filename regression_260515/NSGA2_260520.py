# load model (txt + pkl 紐⑤몢 ???

import os
import pickle
import joblib
import lightgbm as lgb

print("start")

model_names = [
    "MFT_1MW_260518_Lmt_LightGBM_260518",
    "MFT_1MW_260518_Lmt_random_forest_260518",
    "MFT_1MW_260518_Lmt_extra_trees_260518",
    "MFT_1MW_260518_Lmt_gradient_boosting_260518",

    "MFT_1MW_260518_Llt_LightGBM_260518",
    "MFT_1MW_260518_Llt_random_forest_260518",
    "MFT_1MW_260518_Llt_extra_trees_260518",
    "MFT_1MW_260518_Llt_gradient_boosting_260518",

    "MFT_1MW_260518_Tx_loss_LightGBM_260518",
    "MFT_1MW_260518_Tx_loss_random_forest_260518",
    "MFT_1MW_260518_Tx_loss_extra_trees_260518",
    "MFT_1MW_260518_Tx_loss_gradient_boosting_260518",

    "MFT_1MW_260518_Rx_loss_LightGBM_260518",
    "MFT_1MW_260518_Rx_loss_random_forest_260518",
    "MFT_1MW_260518_Rx_loss_extra_trees_260518",
    "MFT_1MW_260518_Rx_loss_gradient_boosting_260518",

    "MFT_1MW_260518_P_main_winding_inner_LightGBM_260518",
    "MFT_1MW_260518_P_main_winding_inner_random_forest_260518",
    "MFT_1MW_260518_P_main_winding_inner_extra_trees_260518",
    "MFT_1MW_260518_P_main_winding_inner_gradient_boosting_260518",

    "MFT_1MW_260518_P_main_winding_outer_LightGBM_260518",
    "MFT_1MW_260518_P_main_winding_outer_random_forest_260518",
    "MFT_1MW_260518_P_main_winding_outer_extra_trees_260518",
    "MFT_1MW_260518_P_main_winding_outer_gradient_boosting_260518",

    "MFT_1MW_260518_P_side_winding_inner_LightGBM_260518",
    "MFT_1MW_260518_P_side_winding_inner_random_forest_260518",
    "MFT_1MW_260518_P_side_winding_inner_extra_trees_260518",
    "MFT_1MW_260518_P_side_winding_inner_gradient_boosting_260518",

    "MFT_1MW_260518_P_side_winding_outer_LightGBM_260518",
    "MFT_1MW_260518_P_side_winding_outer_random_forest_260518",
    "MFT_1MW_260518_P_side_winding_outer_extra_trees_260518",
    "MFT_1MW_260518_P_side_winding_outer_gradient_boosting_260518",
]

model_labels = [
    "Lmt_LightGBM",
    "Lmt_random_forest",    
    "Lmt_extra_trees",
    "Lmt_gradient_boosting",

    "Llt_LightGBM",
    "Llt_random_forest",
    "Llt_extra_trees",
    "Llt_gradient_boosting",

    "Tx_loss_LightGBM",
    "Tx_loss_random_forest",
    "Tx_loss_extra_trees",
    "Tx_loss_gradient_boosting",

    "Rx_loss_LightGBM",
    "Rx_loss_random_forest",
    "Rx_loss_extra_trees",
    "Rx_loss_gradient_boosting",

    "P_main_winding_inner_LightGBM",
    "P_main_winding_inner_random_forest",
    "P_main_winding_inner_extra_trees",
    "P_main_winding_inner_gradient_boosting",

    "P_main_winding_outer_LightGBM",
    "P_main_winding_outer_random_forest",
    "P_main_winding_outer_extra_trees",
    "P_main_winding_outer_gradient_boosting",

    "P_side_winding_inner_LightGBM",
    "P_side_winding_inner_random_forest",
    "P_side_winding_inner_extra_trees",
    "P_side_winding_inner_gradient_boosting",

    "P_side_winding_outer_LightGBM",
    "P_side_winding_outer_random_forest",
    "P_side_winding_outer_extra_trees",
    "P_side_winding_outer_gradient_boosting",
]

def load_model_flexible(model_dir):
    candidates = [
        ("lightgbm_txt", os.path.join(model_dir, "model.txt")),
        ("joblib_pkl",   os.path.join(model_dir, "model.pkl")),
        ("pickle_file",  os.path.join(model_dir, "model.pickle")),
    ]

    errors = []
    for kind, path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if kind == "lightgbm_txt":
                model = lgb.Booster(model_file=path)
            elif kind == "joblib_pkl":
                model = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    model = pickle.load(f)
            return model, path, kind
        except Exception as e:
            errors.append(f"{kind} ?ㅽ뙣 ({path}): {e}")

    raise FileNotFoundError("濡쒕뱶 媛?ν븳 model.txt/model.pkl/model.pickle ?놁쓬\n" + "\n".join(errors))

models = {}
model_info = {}

current_dir = os.path.abspath(os.getcwd())
for name, label in zip(model_names, model_labels):
    model_dir = os.path.join(current_dir, "best_model", name, "test_R2")

    try:
        model, loaded_path, loaded_type = load_model_flexible(model_dir)
        models[label] = model
        model_info[label] = {"path": loaded_path, "type": loaded_type}
        print(f"Loaded: {label} <- {loaded_path} ({loaded_type})")
    except Exception as e:
        print(f"[?ㅽ뙣] {name}")
        print(e)



import numpy as np
import pandas as pd

def plus_input_processing(input_df):
    input_cols = [
        "N1", "N2", "N1_main", "N1_side", "N2_main", "N2_side",
        "w1", "l1", "l2", "h1",
        "cc_w2c_space_x", "w2c_w1c_space_x", "w1c_w2s_space_x", "w2s_w1s_space_x", "w1s_cs_space_x",
        "cc_w2c_space_y", "w2c_w1c_space_y", "cs_w1s_space_y", "w1s_w2s_space_y",
        "window_ratio", "wh1", "wh2", "wff1", "wff2"
    ]

    # -------- normalize --------
    if isinstance(input_df, np.ndarray):
        arr = np.asarray(input_df)
        if arr.ndim == 1:
            inp = pd.DataFrame([arr], columns=input_cols)
        elif arr.ndim == 2:
            if arr.shape[1] != len(input_cols):
                raise ValueError(f"Expected {len(input_cols)} columns, got {arr.shape[1]}")
            inp = pd.DataFrame(arr, columns=input_cols)
        else:
            raise ValueError(f"numpy input must be 1D or 2D, got ndim={arr.ndim}")
    elif isinstance(input_df, pd.Series):
        inp = input_df.to_frame().T
    elif isinstance(input_df, dict):
        inp = pd.DataFrame([input_df])
    elif isinstance(input_df, pd.DataFrame):
        inp = input_df.copy()
    elif isinstance(input_df, (list, tuple)):
        if len(input_df) == len(input_cols) and (len(input_df) == 0 or not isinstance(input_df[0], (list, tuple, dict, np.ndarray, pd.Series))):
            inp = pd.DataFrame([input_df], columns=input_cols)
        else:
            inp = pd.DataFrame(input_df, columns=input_cols)
    else:
        raise TypeError("Unsupported input type for validation_check")

    missing = [c for c in input_cols if c not in inp.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    inp = inp[input_cols].copy().reset_index(drop=True)

    def safe_div(a, b, state):
        if pd.isna(b) or b == 0:
            state[0] = False
            return np.nan
        return a / b

    def calc_one_row(row_df):
        result = True
        st = [True]
        r = row_df.copy()

        # base
        nwl_x = r["l2"].iloc[0] - r["cc_w2c_space_x"].iloc[0] - r["w2c_w1c_space_x"].iloc[0] - r["w1c_w2s_space_x"].iloc[0] - r["w2s_w1s_space_x"].iloc[0] - r["w1s_cs_space_x"].iloc[0]
        r["nwl_x"] = [nwl_x]

        N1_main = r["N1_main"].iloc[0]
        N1_side = r["N1_side"].iloc[0]
        N2_main = r["N2_main"].iloc[0]
        N2_side = r["N2_side"].iloc[0]

        nwl1 = nwl_x * r["window_ratio"].iloc[0]
        nwl2 = nwl_x - nwl1

        N1_main_turns = N1_main
        N1_side_turns = N1_side
        N2_main_turns = N2_main
        N2_side_turns = N2_side
        N1_main_gaps = N1_main - 1 if N1_main > 0 else 0
        N1_side_gaps = N1_side - 1 if N1_side > 0 else 0
        N2_main_gaps = N2_main - 1 if N2_main > 0 else 0
        N2_side_gaps = N2_side - 1 if N2_side > 0 else 0

        # thickness / gaps
        cw1 = safe_div(nwl1 * r["wff1"].iloc[0], (N1_main_turns + N1_side_turns), st)
        cw2 = safe_div(nwl2 * r["wff2"].iloc[0], (N2_main_turns + N2_side_turns), st)
        r["cw1"] = [cw1]
        r["cw2"] = [cw2]

        coil_gap_layer1 = safe_div(nwl1 * (1 - r["wff1"].iloc[0]), (N1_main_gaps + N1_side_gaps), st)
        coil_gap_layer2 = safe_div(nwl2 * (1 - r["wff2"].iloc[0]), (N2_main_gaps + N2_side_gaps), st)
        r["coil_gap_layer1"] = [coil_gap_layer1]
        r["coil_gap_layer2"] = [coil_gap_layer2]

        # requested columns
        nwl1_main = cw1 * N1_main_turns + coil_gap_layer1 * N1_main_gaps
        nwl1_side = cw1 * N1_side_turns + coil_gap_layer1 * N1_side_gaps
        nwl2_main = cw2 * N2_main_turns + coil_gap_layer2 * N2_main_gaps
        nwl2_side = cw2 * N2_side_turns + coil_gap_layer2 * N2_side_gaps
        r["nwl1_main"] = [nwl1_main]
        r["nwl1_side"] = [nwl1_side]
        r["nwl2_main"] = [nwl2_main]
        r["nwl2_side"] = [nwl2_side]

        nwh1 = r["h1"].iloc[0] * r["wh1"].iloc[0]
        nwh2 = r["h1"].iloc[0] * r["wh2"].iloc[0]
        r["nwh1"] = [nwh1]
        r["nwh2"] = [nwh2]

        h_gap1 = (r["h1"].iloc[0] - nwh1) / 2
        h_gap2 = (r["h1"].iloc[0] - nwh2) / 2
        r["h_gap1"] = [h_gap1]
        r["h_gap2"] = [h_gap2]

        den_wff1_main = cw1 * N1_main_turns + coil_gap_layer1 * N1_main_gaps
        den_wff1_side = cw1 * N1_side_turns + coil_gap_layer1 * N1_side_gaps
        den_wff2_main = cw2 * N2_main_turns + coil_gap_layer2 * N2_main_gaps
        den_wff2_side = cw2 * N2_side_turns + coil_gap_layer2 * N2_side_gaps

        wff1_main = safe_div(cw1 * N1_main_turns, den_wff1_main, st)
        wff1_side = 0 if N1_side_turns == 0 else safe_div(cw1 * N1_side_turns, den_wff1_side, st)
        wff2_main = safe_div(cw2 * N2_main_turns, den_wff2_main, st)
        wff2_side = 0 if N2_side_turns == 0 else safe_div(cw2 * N2_side_turns, den_wff2_side, st)

        r["wff1_main"] = [wff1_main]
        r["wff1_side"] = [wff1_side]
        r["wff2_main"] = [wff2_main]
        r["wff2_side"] = [wff2_side]

        sl2_main_x = 2 * r["l1"].iloc[0] + 2 * r["cc_w2c_space_x"].iloc[0]
        sl2_main_y = r["w1"].iloc[0] + 2 * r["cc_w2c_space_y"].iloc[0]
        sl1_main_x = sl2_main_x + 2 * r["nwl2_main"].iloc[0] + 2 * r["w2c_w1c_space_x"].iloc[0]
        sl1_main_y = sl2_main_y + 2 * r["nwl2_main"].iloc[0] + 2 * r["w2c_w1c_space_y"].iloc[0]
        sl1_side_x = r["l1"].iloc[0] + 2 * r["w1s_cs_space_x"].iloc[0]
        sl1_side_y = r["w1"].iloc[0] + 2 * r["cs_w1s_space_y"].iloc[0]
        sl2_side_x = sl1_side_x + 2 * r["nwl1_side"].iloc[0] + 2 * r["w2s_w1s_space_x"].iloc[0]
        sl2_side_y = sl1_side_y + 2 * r["nwl1_side"].iloc[0] + 2 * r["w1s_w2s_space_y"].iloc[0]

        r["sl2_main_x"] = [sl2_main_x]
        r["sl2_main_y"] = [sl2_main_y]
        r["sl1_main_x"] = [sl1_main_x]
        r["sl1_main_y"] = [sl1_main_y]
        r["sl1_side_x"] = [sl1_side_x]
        r["sl1_side_y"] = [sl1_side_y]
        r["sl2_side_x"] = [sl2_side_x]
        r["sl2_side_y"] = [sl2_side_y]

        # checks
        if (pd.isna(nwl1) or nwl1 < 0): result = False
        if (pd.isna(nwl2) or nwl2 < 0): result = False
        if (pd.isna(nwh1) or nwh1 < 0): result = False
        if (pd.isna(nwh2) or nwh2 < 0): result = False
        if (pd.isna(cw1) or cw1 < 1.0 or cw1 > 10): result = False
        if (pd.isna(cw2) or cw2 < 0.6): result = False
        if (pd.isna(coil_gap_layer1) or coil_gap_layer1 < 0.3): result = False
        if (pd.isna(coil_gap_layer2) or coil_gap_layer2 < 0.3): result = False
        if not st[0]:
            result = False

        return result, r

    results = []
    out_rows = []
    for i in range(len(inp)):
        ok, one = calc_one_row(inp.iloc[[i]])
        results.append(ok)
        out_rows.append(one)

    out_df = pd.concat(out_rows, axis=0).reset_index(drop=True)
    # out_df["is_valid"] = results

    # 1???낅젰?대㈃ 湲곗〈 媛먭컖 ?좎?
    if len(out_df) == 1:
        return results[0], out_df
    return results, out_df

# roop

import numpy as np
import pandas as pd
import random
import math
import os
import time

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class SpinFileLock:
    """
    Cross-process lock via O_EXCL lock-file creation.
    Works on multi-node runs when shared filesystem is used.
    """
    def __init__(self, lock_path, timeout=600.0, poll_interval=0.2, stale_seconds=7200.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stale_seconds = stale_seconds
        self._acquired = False

    def acquire(self):
        start = time.time()
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} time={time.time()}\n")
                self._acquired = True
                return
            except FileExistsError:
                # Clean up stale lock file (e.g., crashed process)
                try:
                    mtime = os.path.getmtime(self.lock_path)
                    if (time.time() - mtime) > self.stale_seconds:
                        os.remove(self.lock_path)
                        continue
                except FileNotFoundError:
                    pass

                if (time.time() - start) > self.timeout:
                    raise TimeoutError(f"Timeout while waiting lock: {self.lock_path}")
                time.sleep(self.poll_interval)

    def release(self):
        if self._acquired:
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def read_int_file(path, default=0):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if raw == "":
            return default
        return int(raw)
    except Exception:
        return default


def atomic_write_text(path, text):
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_csv(df, path):
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ?꾩뿉??IndentationError媛 ???쇱씤? ?ш린???덉쑝硫??덈릺怨? ?⑥닔 ?댁뿉 ?덉뼱????

inp = np.array([[1,2,3,4,5], [2,3,4,5,6]])

# input_range (21 inputs)

input_setup = []
input_setup.append([5, 10, 1]) # N1
input_setup.append([0, 0.5, 0.01]) # N1_side_ratio
input_setup.append([0, 0.8, 0.01]) # N2_side_ratio
input_setup.append([200, 800, 1]) # w1
input_setup.append([40, 100, 1]) # l1
input_setup.append([500, 1200, 1]) # total_length
input_setup.append([500, 1000, 1]) # total_height

input_setup.append([10, 50, 0.1]) # cc_w2c_space_x (肄붿뼱~2李?以묒떖 x諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # w2c_w1c_space_x (2李?1李?以묒떖 x諛⑺뼢 媛꾧꺽)
input_setup.append([10, 100, 0.1]) # w1c_w2s_space_x (1李?以묒떖~2李??ъ씠??x諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # w2s_w1s_space_x (2李??ъ씠??1李??ъ씠??x諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # w1s_cs_space_x (1李??ъ씠??肄붿뼱 痢≫뙋 x諛⑺뼢 媛꾧꺽)

input_setup.append([10, 50, 0.1]) # cc_w2c_space_y (肄붿뼱~2李?以묒떖 y諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # w2c_w1c_space_y (2李?1李?以묒떖 y諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # cs_w1s_space_y (肄붿뼱 痢≫뙋~1李??ъ씠??y諛⑺뼢 媛꾧꺽)
input_setup.append([10, 50, 0.1]) # w1s_w2s_space_y (1李??ъ씠??2李??ъ씠??y諛⑺뼢 媛꾧꺽)

input_setup.append([0.3, 0.7, 0.01]) # window_ratio

input_setup.append([0.8, 0.95, 0.01]) # wh1
input_setup.append([0.5, 0.95, 0.01]) # wh2

input_setup.append([0.4, 0.8, 0.01]) # wff1
input_setup.append([0.4, 0.75, 0.01]) # wff2

input_setup = np.array(input_setup)

min_value = input_setup[:, 0] / input_setup[:, 2]
max_value = input_setup[:, 1] / input_setup[:, 2]


def input_processing(input_vector):
    # input_vector: (N, 21) or (21,)
    input_vector = np.array(input_vector)
    step = input_setup[:, 2]  # shape: (21,)

    # input_vector媛 1D硫?(1, 21)濡?reshape
    if input_vector.ndim == 1:
        input_vector = input_vector.reshape(1, -1)

    scaled_input = input_vector * step  # broadcasting, shape: (batch, 21)

    N1 = scaled_input[:, 0]                     # shape: (batch,)
    N2 = N1 * 10                                # shape: (batch,)
    N1_side = np.round(N1 * scaled_input[:, 1])
    N1_main = N1 - N1_side
    N2_side = np.round(N2 * scaled_input[:, 2])
    N2_main = N2 - N2_side

    w1 = scaled_input[:, 3]
    l1 = scaled_input[:, 4]
    total_length = scaled_input[:, 5]
    l2 = (total_length - 4*l1) / 2
    total_height = scaled_input[:, 6]
    h1 = (total_height - 2*l1)

    # ?ㅼ쓬 3媛쒖쓽 蹂?섎뒗 諛섎뱶???⑥닔 ?대??먯꽌 ?좎뼵?섏뼱???섍퀬, ?⑥닔 諛붽묑(湲濡쒕쾶)?먯꽌???ㅼ뿬?곌린媛 ?놁뼱???⑸땲??
    cc_w2c_space_x = scaled_input[:, 7]
    w2c_w1c_space_x = scaled_input[:, 8]
    w1c_w2s_space_x = scaled_input[:, 9]
    w2s_w1s_space_x = scaled_input[:, 10]
    w1s_cs_space_x = scaled_input[:, 11]

    cc_w2c_space_y = scaled_input[:, 12]
    w2c_w1c_space_y = scaled_input[:, 13]
    cs_w1s_space_y = scaled_input[:, 14]
    w1s_w2s_space_y = scaled_input[:, 15]
    
    # 1李?痢↔낵 2李?痢≪쓽 鍮꾩쑉
    window_ratio = scaled_input[:, 16]

    # ?믪씠 鍮꾩쑉??
    wh1 = scaled_input[:, 17]
    wh2 = scaled_input[:, 18]

    wff1 = scaled_input[:, 19]
    wff2 = scaled_input[:, 20]

    # 寃곌낵 stacking for inspection: 紐⑤뱺 二쇱슂 ?뚮씪誘명꽣 ?ы븿
    result = np.stack([
        N1,            # 0
        N2,            # 1
        N1_main,       # 2
        N1_side,       # 3
        N2_main,       # 4
        N2_side,       # 5
        w1,            # 6
        l1,            # 7
        l2,            # 8
        h1,            # 9
        cc_w2c_space_x,    # 10
        w2c_w1c_space_x,   # 11
        w1c_w2s_space_x,   # 12
        w2s_w1s_space_x,   # 13
        w1s_cs_space_x,    # 14
        cc_w2c_space_y,    # 15
        w2c_w1c_space_y,   # 16
        cs_w1s_space_y,    # 17
        w1s_w2s_space_y,   # 18
        window_ratio,      # 19
        wh1,               # 20
        wh2,               # 21
        wff1,              # 22
        wff2               # 23
    ], axis=1)
    return result



# ======================================
# ?? ??? ?? (Vectorized)
# ======================================



def calculate_core_loss(plus_inp) : 

    # 2) geometric/core terms (vectorized)
    V1 = 1e3
    freq = 1e3
    Ts = 1.0 / freq
    m = 1.0

    N1 = plus_inp["N1"].to_numpy(dtype=float)
    l1 = plus_inp["l1"].to_numpy(dtype=float) * 1e-3
    l2 = plus_inp["l2"].to_numpy(dtype=float) * 1e-3
    w1 = plus_inp["w1"].to_numpy(dtype=float) * 1e-3
    h1 = plus_inp["h1"].to_numpy(dtype=float) * 1e-3

    V_core = 2.0 * (w1 * (2.0*l1 + l2) * (2.0*l1 + h1) - w1 * (l2 * h1))
    A_core = 2.0 * w1 * l1 * 0.7

    denom = 4.0 * N1 * A_core

    # B = m*V1*Ts/4/N1/A

    with np.errstate(divide="ignore", invalid="ignore"):
        B_field = np.divide(m * V1 * Ts, denom)

    cm, xx, yy = 1.38, 1.51, 1.74  # 1K101 (W/m^3)
    core_loss = cm * (freq ** xx) * np.power(B_field, yy) * V_core  # freq[Hz], B[T]


    return V_core, A_core, B_field, core_loss



def calculate_volume(plus_inp) :

    N1 = plus_inp["N1"].to_numpy(dtype=float)
    l1 = plus_inp["l1"].to_numpy(dtype=float)
    l2 = plus_inp["l2"].to_numpy(dtype=float)
    w1 = plus_inp["w1"].to_numpy(dtype=float)
    h1 = plus_inp["h1"].to_numpy(dtype=float)

    N1_main = plus_inp["N1_main"].to_numpy(dtype=float)
    N2_main = plus_inp["N2_main"].to_numpy(dtype=float)
    N1_side = plus_inp["N1_side"].to_numpy(dtype=float)
    N2_side = plus_inp["N2_side"].to_numpy(dtype=float)

    nwl1_main = plus_inp["nwl1_main"].to_numpy(dtype=float)
    nwl1_side = plus_inp["nwl1_side"].to_numpy(dtype=float)
    nwl2_main = plus_inp["nwl2_main"].to_numpy(dtype=float)
    nwl2_side = plus_inp["nwl2_side"].to_numpy(dtype=float) 

    cc_w2c_space_x = plus_inp["cc_w2c_space_x"].to_numpy(dtype=float)
    w2c_w1c_space_x = plus_inp["w2c_w1c_space_x"].to_numpy(dtype=float)
    w1c_w2s_space_x = plus_inp["w1c_w2s_space_x"].to_numpy(dtype=float)
    w2s_w1s_space_x = plus_inp["w2s_w1s_space_x"].to_numpy(dtype=float)
    w1s_cs_space_x = plus_inp["w1s_cs_space_x"].to_numpy(dtype=float)
    cc_w2c_space_y = plus_inp["cc_w2c_space_y"].to_numpy(dtype=float)   
    w2c_w1c_space_y = plus_inp["w2c_w1c_space_y"].to_numpy(dtype=float)
    cs_w1s_space_y = plus_inp["cs_w1s_space_y"].to_numpy(dtype=float)
    w1s_w2s_space_y = plus_inp["w1s_w2s_space_y"].to_numpy(dtype=float)


    X_size = (4*l1 + 2*l2) + 2*((w2s_w1s_space_x + w1s_cs_space_x) + (nwl1_side + nwl2_side))
    Y_size_main = (w1) + 2*((cc_w2c_space_y + w2c_w1c_space_y) + (nwl1_main + nwl2_main))
    Y_size_side = (w1) + 2*((cs_w1s_space_y + w1s_w2s_space_y) + (nwl1_side + nwl2_side))
    Y_size = np.maximum(Y_size_main, Y_size_side)
    Z_size = 2*l1 + h1

    volume = X_size * Y_size * Z_size * 1e-6 # unit : liter

    return X_size, Y_size, Z_size, volume


    




from pymoo.core.problem import Problem

class TransformerProblem(Problem):
    def __init__(self):
        xl = np.array(min_value.tolist(), dtype=int)
        xu = np.array(max_value.tolist(), dtype=int)
        super().__init__(n_var=len(min_value),
                         n_obj=2,
                         n_ieq_constr=26,
                         xl=xl,
                         xu=xu,
                         vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        # X shape: (n_pop, n_var)
        X = np.atleast_2d(X)

        # 1) input expansion (batch)
        inp = input_processing(X)
        validation, plus_inp = plus_input_processing(inp)

        n = len(plus_inp)
        val = np.asarray(validation, dtype=float)
        if val.ndim == 0:
            val = np.full(n, float(val), dtype=float)


        V_core, A_core, B_field, core_loss = calculate_core_loss(plus_inp)
        X_size, Y_size, Z_size, volume = calculate_volume(plus_inp)

  
        # ?ъ씠??1李?沅뚯꽑 媛꾧꺽 (x)
        valid_mask = plus_inp.notna().all(axis=1).to_numpy()

        Tx_loss_lgbm = np.full(n, 1e6, dtype=float)
        Tx_loss_et = np.full(n, 1e6, dtype=float)
        Tx_loss_gb = np.full(n, 1e6, dtype=float)
        Tx_loss_rf = np.full(n, 1e6, dtype=float)

        Rx_loss_lgbm = np.full(n, 1e6, dtype=float)
        Rx_loss_et = np.full(n, 1e6, dtype=float)
        Rx_loss_gb = np.full(n, 1e6, dtype=float)
        Rx_loss_rf = np.full(n, 1e6, dtype=float)

        Tx_loss_main_inner_lgbm = np.full(n, 1e6, dtype=float)
        Tx_loss_main_inner_et = np.full(n, 1e6, dtype=float)
        Tx_loss_main_inner_gb = np.full(n, 1e6, dtype=float)
        Tx_loss_main_inner_rf = np.full(n, 1e6, dtype=float)

        Tx_loss_main_outer_lgbm = np.full(n, 1e6, dtype=float)
        Tx_loss_main_outer_et = np.full(n, 1e6, dtype=float)
        Tx_loss_main_outer_gb = np.full(n, 1e6, dtype=float)
        Tx_loss_main_outer_rf = np.full(n, 1e6, dtype=float)

        Tx_loss_side_inner_lgbm = np.full(n, 1e6, dtype=float)
        Tx_loss_side_inner_et = np.full(n, 1e6, dtype=float)
        Tx_loss_side_inner_gb = np.full(n, 1e6, dtype=float)
        Tx_loss_side_inner_rf = np.full(n, 1e6, dtype=float)

        Tx_loss_side_outer_lgbm = np.full(n, 1e6, dtype=float)
        Tx_loss_side_outer_et = np.full(n, 1e6, dtype=float)
        Tx_loss_side_outer_gb = np.full(n, 1e6, dtype=float)
        Tx_loss_side_outer_rf = np.full(n, 1e6, dtype=float)


        Llt1 = np.zeros(n, dtype=float)
        Llt2 = np.zeros(n, dtype=float)
        Llt3 = np.zeros(n, dtype=float)
        Llt4 = np.zeros(n, dtype=float)

        if valid_mask.any():
            valid_df = plus_inp.loc[valid_mask]
            valid_np = valid_df.to_numpy()

            Tx_loss_lgbm[valid_mask] = models["Tx_loss_LightGBM"].predict(valid_np)
            Tx_loss_et[valid_mask] = models["Tx_loss_extra_trees"].predict(valid_df)
            Tx_loss_gb[valid_mask] = models["Tx_loss_gradient_boosting"].predict(valid_df)
            Tx_loss_rf[valid_mask] = models["Tx_loss_random_forest"].predict(valid_df)

            Rx_loss_lgbm[valid_mask] = models["Rx_loss_LightGBM"].predict(valid_np)
            Rx_loss_et[valid_mask] = models["Rx_loss_extra_trees"].predict(valid_df)
            Rx_loss_gb[valid_mask] = models["Rx_loss_gradient_boosting"].predict(valid_df)
            Rx_loss_rf[valid_mask] = models["Rx_loss_random_forest"].predict(valid_df)

            Tx_loss_main_inner_lgbm[valid_mask] = models["P_main_winding_inner_LightGBM"].predict(valid_np)
            Tx_loss_main_inner_et[valid_mask] = models["P_main_winding_inner_extra_trees"].predict(valid_df)
            Tx_loss_main_inner_gb[valid_mask] = models["P_main_winding_inner_gradient_boosting"].predict(valid_df)
            Tx_loss_main_inner_rf[valid_mask] = models["P_main_winding_inner_random_forest"].predict(valid_df)

            Tx_loss_main_outer_lgbm[valid_mask] = models["P_main_winding_outer_LightGBM"].predict(valid_np)
            Tx_loss_main_outer_et[valid_mask] = models["P_main_winding_outer_extra_trees"].predict(valid_df)
            Tx_loss_main_outer_gb[valid_mask] = models["P_main_winding_outer_gradient_boosting"].predict(valid_df)
            Tx_loss_main_outer_rf[valid_mask] = models["P_main_winding_outer_random_forest"].predict(valid_df)

            Tx_loss_side_inner_lgbm[valid_mask] = models["P_side_winding_inner_LightGBM"].predict(valid_np)
            Tx_loss_side_inner_et[valid_mask] = models["P_side_winding_inner_extra_trees"].predict(valid_df)
            Tx_loss_side_inner_gb[valid_mask] = models["P_side_winding_inner_gradient_boosting"].predict(valid_df)
            Tx_loss_side_inner_rf[valid_mask] = models["P_side_winding_inner_random_forest"].predict(valid_df)

            Tx_loss_side_outer_lgbm[valid_mask] = models["P_side_winding_outer_LightGBM"].predict(valid_np)   
            Tx_loss_side_outer_et[valid_mask] = models["P_side_winding_outer_extra_trees"].predict(valid_df)
            Tx_loss_side_outer_gb[valid_mask] = models["P_side_winding_outer_gradient_boosting"].predict(valid_df)
            Tx_loss_side_outer_rf[valid_mask] = models["P_side_winding_outer_random_forest"].predict(valid_df)


            Llt1[valid_mask] = models["Llt_LightGBM"].predict(valid_np)
            Llt2[valid_mask] = models["Llt_extra_trees"].predict(valid_df)
            Llt3[valid_mask] = models["Llt_gradient_boosting"].predict(valid_df)
            Llt4[valid_mask] = models["Llt_random_forest"].predict(valid_df)

        Tx_loss = np.mean(np.column_stack([Tx_loss_lgbm, Tx_loss_et, Tx_loss_gb, Tx_loss_rf]), axis=1)
        Rx_loss = np.mean(np.column_stack([Rx_loss_lgbm, Rx_loss_et, Rx_loss_gb, Rx_loss_rf]), axis=1)
        Tx_loss_main_inner = np.mean(np.column_stack([Tx_loss_main_inner_lgbm, Tx_loss_main_inner_et, Tx_loss_main_inner_gb, Tx_loss_main_inner_rf]), axis=1)
        Tx_loss_main_outer = np.mean(np.column_stack([Tx_loss_main_outer_lgbm, Tx_loss_main_outer_et, Tx_loss_main_outer_gb, Tx_loss_main_outer_rf]), axis=1)
        Tx_loss_side_inner = np.mean(np.column_stack([Tx_loss_side_inner_lgbm, Tx_loss_side_inner_et, Tx_loss_side_inner_gb, Tx_loss_side_inner_rf]), axis=1)
        Tx_loss_side_outer = np.mean(np.column_stack([Tx_loss_side_outer_lgbm, Tx_loss_side_outer_et, Tx_loss_side_outer_gb, Tx_loss_side_outer_rf]), axis=1)

        total_loss = Tx_loss + Rx_loss + core_loss
        eff = 1e+6 / (1e+6 + total_loss) * 100

        # 4) objectives
        f1 = np.nan_to_num(volume, nan=1e12, posinf=1e12, neginf=1e12)
        # f2 is maximized via -f2 in objective, so invalid values must be small (not huge).
        f2 = np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0)

        # 5) constraints (g <= 0)
        
        target_Llt = 27.0
        Llt_error = 0.02
        g1 = target_Llt*(1.0 - Llt_error) - Llt1
        g2 = Llt1 - target_Llt*(1.0 + Llt_error)
        g3 = target_Llt*(1.0 - Llt_error) - Llt2
        g4 = Llt2 - target_Llt*(1.0 + Llt_error)
        g5 = target_Llt*(1.0 - Llt_error) - Llt3
        g6 = Llt3 - target_Llt*(1.0 + Llt_error)
        g7 = target_Llt*(1.0 - Llt_error) - Llt4
        g8 = Llt4 - target_Llt*(1.0 + Llt_error)

        g9 = 1.0 - val # geometry validatiy check
        g10 = np.where(np.isfinite(B_field), B_field - 0.75, 1e6) # B field check




        cw1 = plus_inp["cw1"].to_numpy(dtype=float)
        g11 = np.where(np.isfinite(cw1), 5.0 - cw1, 1e6)  # cw1 >= 3
        g12 = np.where(np.isfinite(cw1), cw1 - 6.0, 1e6)  # cw1 <= 6


        # Tx loss component upper bounds (<= 200)
        g13 = np.where(np.isfinite(Tx_loss_main_inner), Tx_loss_main_inner - 200.0, 1e6)
        g14 = np.where(np.isfinite(Tx_loss_main_outer), Tx_loss_main_outer - 200.0, 1e6)
        g15 = np.where(np.isfinite(Tx_loss_side_inner), Tx_loss_side_inner - 200.0, 1e6)
        g16 = np.where(np.isfinite(Tx_loss_side_outer), Tx_loss_side_outer - 200.0, 1e6)


        # ======================
        # ?덉뿰
        # ======================
        insulation_distance = 30.0



        # 硫붿씤 1李?沅뚯꽑 媛꾧꺽 (x)
        cc_w2c_space_x = plus_inp["cc_w2c_space_x"].to_numpy(dtype=float)
        g17 = np.where(np.isfinite(cc_w2c_space_x), insulation_distance - cc_w2c_space_x, 1e6)

        # 硫붿씤 1李?沅뚯꽑 媛꾧꺽 (y)
        cc_w2c_space_y = plus_inp["cc_w2c_space_y"].to_numpy(dtype=float)
        g18 = np.where(np.isfinite(cc_w2c_space_y), insulation_distance - cc_w2c_space_y, 1e6)

        # 硫붿씤 2李?沅뚯꽑 媛꾧꺽 (x)
        w2c_w1c_space_x = plus_inp["w2c_w1c_space_x"].to_numpy(dtype=float)
        g19 = np.where(np.isfinite(w2c_w1c_space_x), insulation_distance - w2c_w1c_space_x, 1e6)

        # 硫붿씤 2李?沅뚯꽑 媛꾧꺽 (y)
        w2c_w1c_space_y = plus_inp["w2c_w1c_space_y"].to_numpy(dtype=float)
        g20 = np.where(np.isfinite(w2c_w1c_space_y), insulation_distance - w2c_w1c_space_y, 1e6)

        




        # ?ъ씠??1李?沅뚯꽑 媛꾧꺽 (x) (N1_side = 0?쇰븣??怨좊젮 X)
        N1_side = plus_inp["N1_side"].to_numpy(dtype=float)
        gap_w1s_x = plus_inp["w1s_cs_space_x"].to_numpy(dtype=float)
        active_side = N1_side >= 1.0
        g21 = np.where(active_side,
                       np.where(np.isfinite(gap_w1s_x), insulation_distance - gap_w1s_x, 1e6),
                       0.0)  # if N1_side == 0, unconstrained

        # ?ъ씠??1李?沅뚯꽑 媛꾧꺽 (y) (N1_side = 0?쇰븣??怨좊젮 X)
        N1_side = plus_inp["N1_side"].to_numpy(dtype=float)
        gap_w1s_x = plus_inp["cs_w1s_space_y"].to_numpy(dtype=float)
        active_side = N1_side >= 1.0
        g22 = np.where(active_side,
                       np.where(np.isfinite(gap_w1s_x), insulation_distance - gap_w1s_x, 1e6),
                       0.0)  # if N1_side == 0, unconstrained

        # ?ъ씠??2李?沅뚯꽑 媛꾧꺽 (x)
        N1_side = plus_inp["N1_side"].to_numpy(dtype=float)
        w1s_cs_space_x = plus_inp["w1s_cs_space_x"].to_numpy(dtype=float)
        w2s_w1s_space_x = plus_inp["w2s_w1s_space_x"].to_numpy(dtype=float)
        gap_w2s_x = np.where(np.isclose(N1_side, 0.0),
                            w1s_cs_space_x + w2s_w1s_space_x,
                            w2s_w1s_space_x)
        g23 = np.where(np.isfinite(gap_w2s_x), insulation_distance - gap_w2s_x, 1e6)  # gap_w2s_x >= 20
        
        # ?ъ씠??2李?沅뚯꽑 媛꾧꺽 (y)
        N1_side = plus_inp["N1_side"].to_numpy(dtype=float)
        w1s_cs_space_y = plus_inp["cs_w1s_space_y"].to_numpy(dtype=float)
        w2s_w1s_space_y = plus_inp["w1s_w2s_space_y"].to_numpy(dtype=float)
        gap_w2s_y = np.where(np.isclose(N1_side, 0.0),
                            w1s_cs_space_y + w2s_w1s_space_y,
                            w2s_w1s_space_y)
        g24 = np.where(np.isfinite(gap_w2s_y), insulation_distance - gap_w2s_y, 1e6)  # gap_w2s_y >= 20

        # height-direction insulation gaps
        h_gap1 = plus_inp["h_gap1"].to_numpy(dtype=float)
        h_gap2 = plus_inp["h_gap2"].to_numpy(dtype=float)
        g25 = np.where(np.isfinite(h_gap1), insulation_distance - h_gap1, 1e6)
        g26 = np.where(np.isfinite(h_gap2), insulation_distance - h_gap2, 1e6)


        


        G = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26]).astype(float)
        F = np.column_stack([f1, -f2]).astype(float)

        out["F"] = F
        out["G"] = G



def build_df_from_result(res):

    if res.X is None or len(res.X) == 0:
        return pd.DataFrame()

    pareto_X = res.X.astype(int)
    rows = []

    def _scalar(v):
        return float(np.asarray(v).reshape(-1)[0])

    for x in pareto_X:
        inp = input_processing(x)
        validation, plus_inp = plus_input_processing(inp)

        # Skip invalid rows that cannot be evaluated by sklearn GB models.
        if not plus_inp.notna().all(axis=1).iloc[0]:
            continue

        V_core, A_core, B_field, core_loss = calculate_core_loss(plus_inp)
        X_size, Y_size, Z_size, volume = calculate_volume(plus_inp)

        Lmt_lgbm = models["Lmt_LightGBM"].predict(plus_inp)
        Lmt_et = models["Lmt_extra_trees"].predict(plus_inp)
        Lmt_gb = models["Lmt_gradient_boosting"].predict(plus_inp)
        Lmt_rf = models["Lmt_random_forest"].predict(plus_inp)
        Lmt_avg = np.mean(np.column_stack([Lmt_lgbm, Lmt_et, Lmt_gb, Lmt_rf]), axis=1)

        Llt_lgbm = models["Llt_LightGBM"].predict(plus_inp)
        Llt_et = models["Llt_extra_trees"].predict(plus_inp)
        Llt_gb = models["Llt_gradient_boosting"].predict(plus_inp)
        Llt_rf = models["Llt_random_forest"].predict(plus_inp)
        Llt_avg = np.mean(np.column_stack([Llt_lgbm, Llt_et, Llt_gb, Llt_rf]), axis=1)

        Tx_loss_lgbm = models["Tx_loss_LightGBM"].predict(plus_inp)
        Tx_loss_et = models["Tx_loss_extra_trees"].predict(plus_inp)
        Tx_loss_gb = models["Tx_loss_gradient_boosting"].predict(plus_inp)
        Tx_loss_rf = models["Tx_loss_random_forest"].predict(plus_inp)
        Tx_loss_avg = np.mean(np.column_stack([Tx_loss_lgbm, Tx_loss_et, Tx_loss_gb, Tx_loss_rf]), axis=1)

        Rx_loss_lgbm = models["Rx_loss_LightGBM"].predict(plus_inp)
        Rx_loss_et = models["Rx_loss_extra_trees"].predict(plus_inp)
        Rx_loss_gb = models["Rx_loss_gradient_boosting"].predict(plus_inp)
        Rx_loss_rf = models["Rx_loss_random_forest"].predict(plus_inp)
        Rx_loss_avg = np.mean(np.column_stack([Rx_loss_lgbm, Rx_loss_et, Rx_loss_gb, Rx_loss_rf]), axis=1)

        P_main_winding_inner_lgbm = models["P_main_winding_inner_LightGBM"].predict(plus_inp)
        P_main_winding_inner_et = models["P_main_winding_inner_extra_trees"].predict(plus_inp)
        P_main_winding_inner_gb = models["P_main_winding_inner_gradient_boosting"].predict(plus_inp)
        P_main_winding_inner_rf = models["P_main_winding_inner_random_forest"].predict(plus_inp)
        P_main_winding_inner_avg = np.mean(np.column_stack([P_main_winding_inner_lgbm, P_main_winding_inner_et, P_main_winding_inner_gb, P_main_winding_inner_rf]), axis=1)

        P_main_winding_outer_lgbm = models["P_main_winding_outer_LightGBM"].predict(plus_inp)
        P_main_winding_outer_et = models["P_main_winding_outer_extra_trees"].predict(plus_inp)
        P_main_winding_outer_gb = models["P_main_winding_outer_gradient_boosting"].predict(plus_inp)
        P_main_winding_outer_rf = models["P_main_winding_outer_random_forest"].predict(plus_inp)
        P_main_winding_outer_avg = np.mean(np.column_stack([P_main_winding_outer_lgbm, P_main_winding_outer_et, P_main_winding_outer_gb, P_main_winding_outer_rf]), axis=1)

        P_side_winding_inner_lgbm = models["P_side_winding_inner_LightGBM"].predict(plus_inp)
        P_side_winding_inner_et = models["P_side_winding_inner_extra_trees"].predict(plus_inp)
        P_side_winding_inner_gb = models["P_side_winding_inner_gradient_boosting"].predict(plus_inp)
        P_side_winding_inner_rf = models["P_side_winding_inner_random_forest"].predict(plus_inp)
        P_side_winding_inner_avg = np.mean(np.column_stack([P_side_winding_inner_lgbm, P_side_winding_inner_et, P_side_winding_inner_gb, P_side_winding_inner_rf]), axis=1)

        P_side_winding_outer_lgbm = models["P_side_winding_outer_LightGBM"].predict(plus_inp)
        P_side_winding_outer_et = models["P_side_winding_outer_extra_trees"].predict(plus_inp)
        P_side_winding_outer_gb = models["P_side_winding_outer_gradient_boosting"].predict(plus_inp)
        P_side_winding_outer_rf = models["P_side_winding_outer_random_forest"].predict(plus_inp)
        P_side_winding_outer_avg = np.mean(np.column_stack([P_side_winding_outer_lgbm, P_side_winding_outer_et, P_side_winding_outer_gb, P_side_winding_outer_rf]), axis=1)

        total_loss = Tx_loss_avg + Rx_loss_avg + core_loss

        row = plus_inp.iloc[0].to_dict()
        row.update({
            "validation": bool(np.asarray(validation).reshape(-1)[0]),
            "X_size": _scalar(X_size),
            "Y_size": _scalar(Y_size),
            "Z_size": _scalar(Z_size),
            "volume": _scalar(volume),
            "B_field": _scalar(B_field),
            "core_loss": _scalar(core_loss),
            "total_loss": _scalar(total_loss),
            "eff": _scalar(1e+6 / (1e+6 + total_loss) * 100),
            "Lmt_avg": _scalar(Lmt_avg),
            "Llt_avg": _scalar(Llt_avg),
            "Tx_loss_avg": _scalar(Tx_loss_avg),
            "Rx_loss_avg": _scalar(Rx_loss_avg),
            "P_main_winding_inner_avg": _scalar(P_main_winding_inner_avg),
            "P_main_winding_outer_avg": _scalar(P_main_winding_outer_avg),
            "P_side_winding_inner_avg": _scalar(P_side_winding_inner_avg),
            "P_side_winding_outer_avg": _scalar(P_side_winding_outer_avg),
            "Lmt_lgbm": _scalar(Lmt_lgbm),
            "Lmt_et": _scalar(Lmt_et),
            "Lmt_gb": _scalar(Lmt_gb),
            "Lmt_rf": _scalar(Lmt_rf),
            "Llt_lgbm": _scalar(Llt_lgbm),
            "Llt_et": _scalar(Llt_et),
            "Llt_gb": _scalar(Llt_gb),
            "Llt_rf": _scalar(Llt_rf),
            "Tx_loss_lgbm": _scalar(Tx_loss_lgbm),
            "Tx_loss_et": _scalar(Tx_loss_et),
            "Tx_loss_gb": _scalar(Tx_loss_gb),
            "Tx_loss_rf": _scalar(Tx_loss_rf),
            "Rx_loss_lgbm": _scalar(Rx_loss_lgbm),
            "Rx_loss_et": _scalar(Rx_loss_et),
            "Rx_loss_gb": _scalar(Rx_loss_gb),
            "Rx_loss_rf": _scalar(Rx_loss_rf),
            "P_main_winding_inner_lgbm": _scalar(P_main_winding_inner_lgbm),
            "P_main_winding_inner_et": _scalar(P_main_winding_inner_et),
            "P_main_winding_inner_gb": _scalar(P_main_winding_inner_gb),
            "P_main_winding_inner_rf": _scalar(P_main_winding_inner_rf),
            "P_main_winding_outer_lgbm": _scalar(P_main_winding_outer_lgbm),
            "P_main_winding_outer_et": _scalar(P_main_winding_outer_et),
            "P_main_winding_outer_gb": _scalar(P_main_winding_outer_gb),
            "P_main_winding_outer_rf": _scalar(P_main_winding_outer_rf),
            "P_side_winding_inner_lgbm": _scalar(P_side_winding_inner_lgbm),
            "P_side_winding_inner_et": _scalar(P_side_winding_inner_et),
            "P_side_winding_inner_gb": _scalar(P_side_winding_inner_gb),
            "P_side_winding_inner_rf": _scalar(P_side_winding_inner_rf),
            "P_side_winding_outer_lgbm": _scalar(P_side_winding_outer_lgbm),
            "P_side_winding_outer_et": _scalar(P_side_winding_outer_et),
            "P_side_winding_outer_gb": _scalar(P_side_winding_outer_gb),
            "P_side_winding_outer_rf": _scalar(P_side_winding_outer_rf),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ======================================
# 硫붿씤 ?ㅽ뻾遺
# ======================================
MU, NGEN = 100, 1000
CXPB, MUTPB = 0.7, 0.3
NUM_ITRS = 1000

def run_nsga2(seed):
    np.random.seed(seed)
    random.seed(seed)

    algorithm = NSGA2(
        pop_size=MU,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=CXPB, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True
    )

    return minimize(
        problem=TransformerProblem(),
        algorithm=algorithm,
        termination=('n_gen', NGEN),
        seed=seed,
        verbose=True
    )


# Save all NSGA-II artifacts under NSGA2_result/
result_dir = os.path.join(os.getcwd(), "NSGA2_result")
os.makedirs(result_dir, exist_ok=True)

pareto_file = os.path.join(result_dir, "pareto_front.csv")
loop_counter_file = os.path.join(result_dir, "loop_counter.txt")
lock_file = os.path.join(result_dir, "pareto_front.lock")
for itr in range(NUM_ITRS):
    print(f"Running NSGA-II {itr+1} / {NUM_ITRS}")
    seed = np.random.randint(0, 1000000)
    res = run_nsga2(seed)

    try:
        df = build_df_from_result(res)
        if df is not None and not df.empty:
            with SpinFileLock(lock_file, timeout=1800.0, poll_interval=0.2):
                # Always re-read shared files inside lock for multi-node safety.
                if os.path.exists(pareto_file):
                    previous_pareto = pd.read_csv(pareto_file)
                else:
                    previous_pareto = pd.DataFrame()

                loop_counter = read_int_file(loop_counter_file, default=0)

                # Merge this iteration's candidates with latest global Pareto.
                df_current = pd.concat([df, previous_pareto], ignore_index=True)
                if df_current.empty:
                    continue

                F = df_current[["volume", "eff"]].to_numpy()
                F[:, 1] = -F[:, 1]  # maximize eff -> minimize -eff

                nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                df_pareto = df_current.iloc[nds].copy()
                df_pareto["eff"] = -F[nds, 1]
                df_pareto = df_pareto.sort_values(by="eff", ascending=False).reset_index(drop=True)

                # Atomic writes while lock is held.
                atomic_write_csv(df_pareto, pareto_file)

                loop_counter += 1
                atomic_write_text(loop_counter_file, str(loop_counter))

                backup_file = os.path.join(result_dir, f"pareto_front_backup_{loop_counter}.csv")
                atomic_write_csv(df_pareto, backup_file)
                print(f"Backup created: {backup_file}")

            print(f"Iteration {itr+1} Pareto front saved (Total loops: {loop_counter})")
        else:
            print(f"Iteration {itr+1} skipped: empty result")
    except TimeoutError as e:
        print(f"Iteration {itr+1} lock timeout: {str(e)}")
        continue
    except AttributeError as e:
        print(f"Iteration {itr+1} failed: {str(e)}")
        continue
    except Exception as e:
        print(f"Unexpected error in iteration {itr+1}: {str(e)}")
        continue

if os.path.exists(pareto_file):
    df_pareto = pd.read_csv(pareto_file)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("Final Pareto front:")
    print(df_pareto)
else:
    print("No valid results found")



