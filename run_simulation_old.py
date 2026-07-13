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
with pyDesktop(version=None, non_graphical=GUI, close_on_exit=False, new_desktop=True) as desktop:

    sim = Simulation(desktop=desktop)
    run(simulation=sim)
