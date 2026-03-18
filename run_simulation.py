import sys
import traceback
import logging
import portalocker
import os
import re

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 경로 설정 - 플랫폼에 따라 다르게 처리
if os.name == 'nt':  # Windows
    sys.path.insert(0, r"Y:/git/pyaedt_library/src/")
else:  # Linux/Unix
    # Linux 서버 경로들 시도
    possible_paths = [
        # r"/gpfs/home1/r1jae262/jupyter/git/pyaedt_library/src/",
        r"../pyaedt_library/src/",
        os.path.abspath(os.path.join(BASE_DIR, "../git/pyaedt_library/src/")),
        "/home1/r1jae262/jupyter/git/pyaedt_library/src/",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            break

import pyaedt_module
from pyaedt_module.core import pyDesktop
import os
import time
from datetime import datetime

import math
import copy

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


import platform
import csv
from filelock import FileLock



from module.input_parameter import create_input_parameter, set_design_variables, validation_check
from module.modeling import create_core, create_coil, create_coil_section




class Simulation() :

    def __init__(self, desktop=None) :

        self.NUM_CORE = 4
        self.NUM_TASK = 1
        self.desktop = desktop


    def create_simulation_name(self):

        file_path = "./simulation_num.txt"

        # 파일이 존재하지 않으면 생성
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("1")

        # 읽기/쓰기 모드로 파일 열기
        with open(file_path, "r+", encoding="utf-8") as file:
            # 파일 잠금: LOCK_EX는 배타적 잠금,  블로킹 모드로 실행 (Windows/Linux 호환)
            portalocker.lock(file, portalocker.LOCK_EX)

            # 파일에서 값 읽기
            content = int(file.read().strip())
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"
            content += 1

            # 파일 포인터를 처음으로 되돌리고, 파일 내용 초기화 후 새 값 쓰기
            file.seek(0)
            file.truncate()
            file.write(str(content))




def run(simulation=None):



    sim = simulation

    sim.create_simulation_name()

    # simulation 디렉토리 생성 (존재하지 않으면)
    simulation_dir = "./simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir, exist_ok=True)
    
    # 절대 경로로 변환
    project_path = os.path.abspath(os.path.join(simulation_dir, sim.PROJECT_NAME))
    
    # desktop이 None이거나 유효하지 않은지 확인
    if sim.desktop is None:
        raise RuntimeError("Desktop instance is None. Cannot create project.")
    
    try:
        project1 = sim.desktop.create_project(path=project_path, name=sim.PROJECT_NAME)
    except Exception as e:
        error_msg = f"Failed to create project '{sim.PROJECT_NAME}' at path '{project_path}': {e}\n"
        print(error_msg, file=sys.stderr)
        sys.stderr.flush()
        raise
    
    
    design1 = project1.create_design(name="mawell_design", solver="maxwell3d", solution="AC Magnetic")

    # input_parameter = sim1.create_input_parameter()
    # sim1.set_variable(input_parameter)

    sim.project = project1
    sim.design1 = design1



itr = 0
GUI = False


print("ANSYSEM_ROOT252 =", os.environ.get("ANSYSEM_ROOT252"))
print("ANSYSLMD_LICENSE_FILE =", os.environ.get("ANSYSLMD_LICENSE_FILE"))

# with pyDesktop(version=None, non_graphical=GUI, close_on_exit=False, new_desktop=True) as desktop:

desktop = pyDesktop(version="2025.2", non_graphical=GUI, close_on_exit=False, new_desktop=True)

sim = Simulation(desktop=desktop)
# run(simulation=sim)
sim.create_simulation_name()

# simulation 디렉토리 생성 (존재하지 않으면)
simulation_dir = "./simulation"
if not os.path.exists(simulation_dir):
    os.makedirs(simulation_dir, exist_ok=True)

# 절대 경로로 변환
project_path = os.path.abspath(os.path.join(simulation_dir, sim.PROJECT_NAME))

# desktop이 None이거나 유효하지 않은지 확인
if sim.desktop is None:
    raise RuntimeError("Desktop instance is None. Cannot create project.")

try:
    project1 = sim.desktop.create_project(path=project_path, name=sim.PROJECT_NAME)
except Exception as e:
    error_msg = f"Failed to create project '{sim.PROJECT_NAME}' at path '{project_path}': {e}\n"
    print(error_msg, file=sys.stderr)
    sys.stderr.flush()
    raise


design1 = project1.create_design(name="mawell_design", solver="maxwell3d", solution="AC Magnetic")

# input_parameter = sim1.create_input_parameter()
# sim1.set_variable(input_parameter)

sim.project = project1
sim.design1 = design1


while True :
    sim.input_df = create_input_parameter()
    result, df_plus = validation_check(sim.input_df)
    if result :
        break

set_design_variables(sim.design1, sim.input_df)

create_core(design=sim.design1, name="core", core_material="ferrite")




# Tx winding 생성
Tx_windings, Tx_N, Tx_coil_width, Tx_coil_height, Tx_coil_gap_x, Tx_coil_gap_z = create_coil(
    design = sim.design1,
    name = "Tx",
    window_height = df_plus["nwh1"].iloc[0],
    window_length = df_plus["nwl1"].iloc[0],
    window_layer = 1,
    N_input = df_plus["N1"].iloc[0],
    width_fill_factor = df_plus["wff1"].iloc[0],
    space_length = df_plus["sl1"].iloc[0],
    space_width = df_plus["sw1"].iloc[0],
    shape = "circle",
    offset = [0,0,0],
    color = [255,10,10]
)

l1 = df_plus["l1"].iloc[0]
l2 = df_plus["l2"].iloc[0]


Rx_windings1, Rx_N1, Rx_coil_width1, Rx_coil_height1, Rx_coil_gap_x1, Rx_coil_gap_z1 = create_coil(
    design = sim.design1,
    name = "Rx_center",
    window_height = df_plus["nwh2"].iloc[0],
    window_length = df_plus["nwl2_main"].iloc[0],
    window_layer = df_plus["N2_main"].iloc[0],
    N_input = 1,
    width_fill_factor = df_plus["wff2"].iloc[0],
    space_length = df_plus["sl2_main"].iloc[0],
    space_width = df_plus["sw2_main"].iloc[0],
    shape = "rectangle",
    offset = [0,0,0],
    color = [10, 10, 255]
)   

Rx_windings2, Rx_N2, Rx_coil_width2, Rx_coil_height2, Rx_coil_gap_x2, Rx_coil_gap_z2 = create_coil(
    design = sim.design1,
    name = "Rx_side1",
    window_height = df_plus["nwh2"].iloc[0],
    window_length = df_plus["nwl2_side"].iloc[0],
    window_layer = df_plus["N2_side"].iloc[0],
    N_input = 1,
    width_fill_factor = df_plus["wff2"].iloc[0],
    space_length = df_plus["sl2_side"].iloc[0],
    space_width = df_plus["sw2_side"].iloc[0],
    shape = "rectangle",
    offset=[(-l1-l2-l1/2),0,0],
    color = [10, 10, 255]
)   

Rx_windings3, Rx_N3, Rx_coil_width3, Rx_coil_height3, Rx_coil_gap_x3, Rx_coil_gap_z3 = create_coil(
    design = sim.design1,
    name = "Rx_side2",
    window_height = df_plus["nwh2"].iloc[0],
    window_length = df_plus["nwl2_side"].iloc[0],
    window_layer = df_plus["N2_side"].iloc[0],
    N_input = 1,
    width_fill_factor = df_plus["wff2"].iloc[0],
    space_length = df_plus["sl2_side"].iloc[0],
    space_width = df_plus["sw2_side"].iloc[0],
    shape = "rectangle",
    offset = [(l1+l2+l1/2),0,0],
    color = [10, 10, 255]
)   



Tx_neg_sheets, Tx_pos_sheets = create_coil_section(design=sim.design1, winding_obj=Tx_windings, sheet_prefix = None, plane = "ZX", rename_faces = False)

Rx_neg_sheets_center, Rx_pos_sheets_center = create_coil_section(design=sim.design1, winding_obj=Rx_windings1, sheet_prefix = None, plane = "ZX", rename_faces = False)
Rx_neg_sheets_side1, Rx_pos_sheets_side1 = create_coil_section(design=sim.design1, winding_obj=Rx_windings2, sheet_prefix = None, plane = "ZX", rename_faces = False)
Rx_neg_sheets_side2, Rx_pos_sheets_side2 = create_coil_section(design=sim.design1, winding_obj=Rx_windings3, sheet_prefix = None, plane = "ZX", rename_faces = False)


tx_winding = sim.design1.assign_winding(
    assignment=[], 
    winding_type="Current", 
    is_solid=True, 
    current=f"{1000*math.sqrt(2)}A",
    name="Tx_winding"
)
rx_winding1 = sim.design1.assign_winding(
    assignment=[], 
    winding_type="Current", 
    is_solid=True, 
    current=f"{100*math.sqrt(2)}A",
    name="Rx_winding"
)


Tx_coil = []
Rx_coil = []

import re

for idx, sheet in enumerate(Tx_neg_sheets, start=1):
    coil = sim.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_coil{idx}")
    Tx_coil.append(coil)

for idx, sheet in enumerate(Rx_neg_sheets_center, start=1):
    coil = sim.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_center_coil{idx}")
    Rx_coil.append(coil)

for idx, sheet in enumerate(Rx_neg_sheets_side1 + Rx_neg_sheets_side2, start=1):
    coil = sim.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil{idx}")
    Rx_coil.append(coil)





sim.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in Tx_coil])
sim.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in Rx_coil])

sim.design1.assign_matrix(matrix_name="Matrix", assignment=["Tx_winding", "Rx_winding"])


air_region = sim.design1.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=100.0, z_neg=100.0, is_percentage=True)
sim.design1.assign_radiation(assignment=[air_region.name], radiation="Radiation")


sim.design1.setup = sim.design1.create_setup(name = "Setup1")
sim.design1.setup.properties["Max. Number of Passes"] = 6 # 10
sim.design1.setup.properties["Min. Number of Passes"] = 1
sim.design1.setup.properties["Min. Converged Passes"] = 1
sim.design1.setup.properties["Percent Error"] = 2.5 # 2.5
sim.design1.setup.properties["Frequency Setup"] = f"1kHz"



import time
start_time = time.time()
sim.design1.setup.analyze(cores=16)
end_time = time.time()
print(f"Analysis execution time: {end_time - start_time:.2f} seconds")



params = [
    ["Matrix.L(Tx_winding,Tx_winding)", f"Ltx", "uH"],
    ["Matrix.L(Rx_winding,Rx_winding)", f"Lrx", "uH"],
    ["Matrix.L(Tx_winding,Rx_winding)", f"M", "uH"],
    ["abs(Matrix.CplCoef(Tx_winding,Rx_winding))", f"k", ""],
    ["Matrix.L(Tx_winding,Tx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Lmt", "uH"],
    ["Matrix.L(Rx_winding,Rx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Lmr", "uH"],
    ["Matrix.L(Tx_winding,Tx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Llt", "uH"],
    ["Matrix.L(Rx_winding,Rx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Llr", "uH"],
]

dir = sim.project.path
mod = "write"
import_report = None
report_name = "magnetic_report"
file_name = "magnetic_report"

report, df = sim.design1.get_magnetic_parameter(dir=dir, parameters=params, mod=mod, import_report=import_report, report_name=report_name, file_name=file_name)



def save_results_to_csv(results_df, filename="simulation_results.csv"):
        """Saves the DataFrame to a CSV file in a process-safe way."""
        lock_path = filename + ".lock"
        with FileLock(lock_path):
            file_exists = os.path.isfile(filename)
            results_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        logging.info(f"Results saved to {filename}")



save_results_to_csv(df, "simulation_results.csv")