import sys
import traceback
import logging
import portalocker
import os

# 경로 설정 - 플랫폼에 따라 다르게 처리
if os.name == 'nt':  # Windows
    sys.path.insert(0, r"Y:/git/pyaedt_library/src/")
else:  # Linux/Unix
    # Linux 서버 경로들 시도
    possible_paths = [
        # r"/gpfs/home1/r1jae262/jupyter/git/pyaedt_library/src/",
        r"../pyaedt_library/src/",
        os.path.join(os.path.dirname(__file__), "../git/pyaedt_library/src/"),
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


import platform
import csv



from module.input_parameter import create_input_parameter, create_input_parameter_for_test, calculate_coil_parameter, calculate_coil_offset, set_design_variables
from module.modeling import (
    create_core_model, create_all_windings, create_cold_plate, create_air,
    assign_meshing, assign_excitations, create_face, create_mold
)
from module.report import (
    get_input_parameter, get_maxwell_magnetic_parameter,
    get_maxwell_calculator_parameter, get_convergence_report, get_icepak_calculator_parameter
)



class Simulation() :

    def __init__(self, desktop=None) :

        self.NUM_CORE = 4
        self.NUM_TASK = 1

        # Desktop은 바깥에서 생성(with 포함)해서 주입하는 것을 권장
        # (루프 중 프로젝트 close/delete가 Desktop 핸들을 무효화시키는 문제를 줄이기 위함)
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

    def create_input_parameter(self, param_list=None) :
        if self.test == True :
            input_parameter = create_input_parameter_for_test(self.maxwell_design, param_list)
        else :
            input_parameter = create_input_parameter(self.maxwell_design, param_list)
        logging.info(f"input_parameter : {input_parameter}")
        logging.info("input_parameter : " + ",".join(str(float(v)) for v in input_parameter.values()))
        return input_parameter

    def set_variable(self, input_parameter) :
        # 1. Simulation 클래스 인스턴스에 속성으로 값을 설정합니다 (예: self.N1 = 10).
        for key, value in input_parameter.items():
            setattr(self.maxwell_design, key, value)
        
        self.input_df = pd.DataFrame([input_parameter])
        
        # 2. input_parameter.py의 함수를 호출하여 Ansys 디자인에 변수를 설정합니다.
        set_design_variables(self.maxwell_design, input_parameter)

    def set_maxwell_analysis(self) :
        self.maxwell_design.setup = self.maxwell_design.create_setup(name = "Setup1")
        self.maxwell_design.setup.properties["Max. Number of Passes"] = 12 # 10
        self.maxwell_design.setup.properties["Min. Number of Passes"] = 1
        self.maxwell_design.setup.properties["Min. Converged Passes"] = 3
        self.maxwell_design.setup.properties["Percent Error"] = 2.5 # 2.5
        self.maxwell_design.setup.properties["Frequency Setup"] = f"{self.maxwell_design.frequency}kHz"

    def create_geometry(self, design):

        core_params = {
            "w1": "300mm", # 변수가 아니라 값을 넣어도 됨
            "l1_leg": "50mm",
            "l1_top": "50mm",
            "l2": "200mm",
            "h1": "300mm",
            "mat": "ferrite"
        }

        core = design.modeler.create_coretype_core(name="Core", **core_params)



def run(simulation=None):
    """시뮬레이션을 실행합니다."""
    sim1 = simulation

    sim1.create_simulation_name()

    # simulation 디렉토리 생성 (존재하지 않으면)
    simulation_dir = "./simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir, exist_ok=True)
    
    # 절대 경로로 변환
    project_path = os.path.abspath(os.path.join(simulation_dir, sim1.PROJECT_NAME))
    
    # desktop이 None이거나 유효하지 않은지 확인
    if sim1.desktop is None:
        raise RuntimeError("Desktop instance is None. Cannot create project.")
    
    try:
        project1 = sim1.desktop.create_project(path=project_path, name=sim1.PROJECT_NAME)
    except Exception as e:
        error_msg = f"Failed to create project '{sim1.PROJECT_NAME}' at path '{project_path}': {e}\n"
        print(error_msg, file=sys.stderr)
        sys.stderr.flush()
        raise
    
    design1 = project1.create_design(name="mawell_design", solver="maxwell3d", solution="AC Magnetic")

    # input_parameter = sim1.create_input_parameter()
    # sim1.set_variable(input_parameter)

    sim1.project = project1
    sim1.design1 = design1



itr = 0
GUI = False
desktop = pyDesktop(version=None, non_graphical=GUI, close_on_exit=False, new_desktop=True)
sim = Simulation(desktop=desktop)
run(simulation=sim)

"""
itr = 0
GUI = False
with pyDesktop(version=None, non_graphical=GUI, close_on_exit=False, new_desktop=True) as desktop:
    print(f"loop {itr} : desktop init done (pid={getattr(desktop, 'pid', None)})", flush=True)
    print(f"loop {itr} : simulation start!!", flush=True)
    sim = Simulation(desktop=desktop)
    run(simulation=sim)
"""




def create_coil(name, window_height, window_length, window_layer, window_fill_factor, N_fill_factor, space_length, space_width):



    coil_width = window_length * window_fill_factor / window_layer
    if window_layer > 1 :
        coil_gap_x = (window_length - (coil_width * window_layer)) / (window_layer - 1)
    else :
        coil_gap_x = 0
    N = int(window_height / coil_width * N_fill_factor)
    coil_gap_z = (window_height - (coil_width * N)) / (N - 1)



    x_pos = []
    y_pos = []
    z_pos = []

    for i in range(window_layer) :
        x_pos.append(space_length/2 + coil_width*(i+0.5) + coil_gap_x*i)
        y_pos.append(space_width/2 + coil_width*(i+0.5) + coil_gap_x*i)

    for i in range(N) :
        z_pos.append(window_height/2 - coil_width*(i+0.5) - coil_gap_z*i)

    windings = []

    i = 0
    j = 0

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        for j, z in enumerate(z_pos):
                
            points = []
            points.append([f"{x}mm", f"{y}mm", f"{z}mm" ])
            points.append([f"-{x}mm", f"{y}mm", f"{z}mm"])
            points.append([f"-{x}mm", f"-{y}mm", f"{z}mm"])
            points.append([f"{x}mm", f"-{y}mm", f"{z}mm"])
            points.append([f"{x}mm", f"{y}mm", f"{z}mm" ])


            winding = sim.design1.modeler.create_polyline(
                points=points, name=f"{name}_{i}_{j}", xsection_orient="Auto",
                xsection_type="circle", xsection_width=coil_width, xsection_height=coil_width, xsection_num_seg=6, xsection_topwidth=coil_width)
            windings.append(winding)

    print(f"N = {N}")
    print(f"coil_width = {coil_width}")
    print(f"coil_gap_x = {coil_gap_x}")
    print(f"coil_gap_z = {coil_gap_z}")

    return windings, N, coil_width, coil_gap_x, coil_gap_z



def create_core(name, l1, l2, h1, w1) :

    main = sim.design1.modeler.create_box(
        origin = [f"-{(4*l1+2*l2)/2}mm", f"-{(w1)/2}mm", f"-{(h1+2*l1)/2}mm"],
        sizes = [f"{4*l1+2*l2}mm", f"{w1}mm", f"{h1+2*l1}mm"],
        name = f"{name}",
        material = "ferrite"
    )

    sub1 = sim.design1.modeler.create_box(
        origin = [f"-{l1}mm", f"-{(w1)/2}mm", f"-{(h1)/2}mm"],
        sizes = [f"-{l2}mm", f"{w1}mm", f"{h1}mm"],
        name = f"{name}_sub1",
        material = "ferrite"
    )

    sub2 = sim.design1.modeler.create_box(
        origin = [f"{l1}mm", f"-{(w1)/2}mm", f"-{(h1)/2}mm"],
        sizes = [f"{l2}mm", f"{w1}mm", f"{h1}mm"],          
        name = f"{name}_sub2",
        material = "ferrite"
    )

    
    sim.design1.modeler.subtract(
        [main],
        [sub1, sub2],
        keep_originals=False
    )

    return main









l1 = 75
l2 = 250
h1 = 650
w1 = 500

coil_coil_space = 20
coil_core_space = 30


core = create_core(name="core", l1=l1, l2=l2, h1=h1, w1=w1)




net_window_length = l2 - 2*coil_coil_space - coil_core_space

ratio = 0.6

Tx_window_length = net_window_length * ratio
Rx_window_length = net_window_length - Tx_window_length




window_height = h1 - 2*coil_coil_space
window_length = (l2 - 2*coil_coil_space - coil_core_space) / 2
window_length = Tx_window_length
window_layer = 3
window_fill_factor = 0.8
N_fill_factor = 0.95
space_length = 2*l1 + 2*coil_coil_space
space_width = w1 + 2*coil_coil_space

Tx_windings, Tx_N, Tx_coil_width, Tx_coil_gap_x, Tx_coil_gap_z = create_coil("Tx", window_height, window_length, window_layer, window_fill_factor, N_fill_factor, space_length, space_width)

for winding in Tx_windings :

    winding.transparency = 0
    winding.color = [255, 10, 10]
    winding.material_name = "copper"


window_height = h1 - 2*coil_coil_space
window_length = (l2 - 2*coil_coil_space - coil_core_space) / 2
window_length = Rx_window_length
window_layer = 1
window_fill_factor = 0.8
N_fill_factor = 0.6
space_length = 2*l1 + 2*coil_coil_space + 2*(coil_coil_space + Tx_window_length)
space_width = w1 + 2*coil_coil_space + 2*(coil_coil_space + Tx_window_length)

Rx_windings, Rx_N, Rx_coil_width, Rx_coil_gap_x, Rx_coil_gap_z = create_coil("Rx", window_height, window_length, window_layer, window_fill_factor, N_fill_factor, space_length, space_width)

for winding in Rx_windings :

    winding.transparency = 0
    winding.color = [10, 10, 255]
    winding.material_name = "copper"


def section_winding_get_two_sheets(sim, winding_obj, sheet_prefix=None, plane="ZX"):
    """
    plane="ZX"  -> global y=0 평면(=XZ plane)
    returns: (sheets, ok)
      - sheets: 새로 생성된 sheet object 리스트
      - ok: section 성공 여부(bool)
    """
    modeler = sim.design1.modeler

    if sheet_prefix is None:
        sheet_prefix = f"{winding_obj.name}_sec"

    # section 전 sheet 목록 저장
    before_sheet_names = set(modeler.sheet_names)

    # PyAEDT section 실행 (returns bool)
    ok = modeler.section(winding_obj, plane)  # plane choices: "XY","YZ","ZX"  :contentReference[oaicite:1]{index=1}
    if not ok:
        return [], False

    # section 후 새로 생긴 sheet 이름들
    after_sheet_names = set(modeler.sheet_names)
    new_sheet_names = list(after_sheet_names - before_sheet_names)

    # sheet object로 변환
    sheets = [modeler.get_object_from_name(n) for n in new_sheet_names]

    # face seperation
    sheets = sim.design1.modeler.separate_bodies(assignment=sheets)



    # (옵션) 딱 2개만 기대한다면, winding bbox 근처만 남기고 싶을 수 있음
    # 여기선 일단 다 반환
    return sheets, True


Tx_sheets_pos = []
Tx_sheets_neg = []
Rx_sheets_pos = []
Rx_sheets_neg = []




for winding in Tx_windings :
    sheets, ok = section_winding_get_two_sheets(sim, winding)
    if ok == True :
        for sheet in sheets :
            for face in sheet.faces :
                if face.center[0] > 0 :
                    Tx_sheets_pos.append(sheet)
                elif face.center[0] < 0 :
                    Tx_sheets_neg.append(sheet)

for winding in Rx_windings :
    sheets, ok = section_winding_get_two_sheets(sim, winding)
    if ok == True :
        for sheet in sheets :
            for face in sheet.faces :
                if face.center[0] > 0 :
                    Rx_sheets_pos.append(sheet)
                elif face.center[0] < 0 :
                    Rx_sheets_neg.append(sheet)



tx_winding = sim.design1.assign_winding(
    assignment=[], 
    winding_type="Current", 
    is_solid=True, 
    current=f"{100*math.sqrt(2)}A",
    name="Tx_winding"
)
rx_winding1 = sim.design1.assign_winding(
    assignment=[], 
    winding_type="Current", 
    is_solid=True, 
    current=f"{1000*math.sqrt(2)}A",
    name="Rx_winding"
)


import re

Tx_coil_pos = []
Tx_coil_neg = []
Rx_coil_pos = []
Rx_coil_neg = []



for sheet in Tx_sheets_neg :

    match = re.match(r'^(Tx_\d+_\d+).*$', sheet.name)
    if match:
        new_name = f"{match.group(1)}_face_n"
        sheet.name = new_name
    coil = sim.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"{sheet.name}_coil")
    Tx_coil_neg.append(coil)


for sheet in Rx_sheets_neg :

    match = re.match(r'^(Rx_\d+_\d+).*$', sheet.name)
    if match:
        new_name = f"{match.group(1)}_face_n"
        sheet.name = new_name
    coil = sim.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"{sheet.name}_coil")
    Rx_coil_neg.append(coil)


sim.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in Tx_coil_neg])
sim.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in Rx_coil_neg])

sim.design1.assign_matrix(matrix_name="Matrix", assignment=["Tx_winding", "Rx_winding"])



freq = 1e+3

mu0 = 4 * math.pi * 1e-7
mu_copper = mu0 
sigma_copper = 58000000
omega = 2 * math.pi * freq
skin_depth = math.sqrt(2 / (omega * mu_copper * sigma_copper)) * 1e3 # in mm

length_mesh = sim.design1.mesh.assign_length_mesh(
    assignment=[core],
    inside_selection=False,
    maximum_length="100mm",
    name="core_mesh"
)



Tx_skin_depth_mesh = sim.design1.mesh.assign_skin_depth(
    assignment=Tx_windings,
    skin_depth=f'{skin_depth}mm',
    triangulation_max_length='50mm',
    layers_number="2",
    name="Tx_winding_skin_depth"
)

Rx_skin_depth_mesh = sim.design1.mesh.assign_skin_depth(
    assignment=Rx_windings,
    skin_depth=f'{skin_depth}mm',
    triangulation_max_length='50mm',
    layers_number="2",
    name="Rx_winding_skin_depth"
)


air_region = sim.design1.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=100.0, z_neg=100.0, is_percentage=True)
sim.design1.assign_radiation(assignment=[air_region.name], radiation="Radiation")


sim.design1.setup = sim.design1.create_setup(name = "Setup1")
sim.design1.setup.properties["Max. Number of Passes"] = 12 # 10
sim.design1.setup.properties["Min. Number of Passes"] = 1
sim.design1.setup.properties["Min. Converged Passes"] = 3
sim.design1.setup.properties["Percent Error"] = 2.5 # 2.5
sim.design1.setup.properties["Frequency Setup"] = f"1kHz"


sim.design1.setup.analyze(cores=16)