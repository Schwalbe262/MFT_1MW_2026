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



from module.input_parameter import create_input_parameter, set_design_variables, validation_check
from module.modeling import create_core, create_coil, create_coil_section


from ansys.aedt.core import settings

settings.skip_license_check = True
settings.wait_for_license = False



if os.name == 'nt':  # Windows
    GUI = False
else:  # Linux/Unix
    GUI = True

from filelock import FileLock
import shutil



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

    def create_project(self) :

        # simulation 디렉토리 생성 (존재하지 않으면)
        simulation_dir = "./simulation"
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir, exist_ok=True)
        
        # 절대 경로로 변환
        project_path = os.path.abspath(os.path.join(simulation_dir, sim.PROJECT_NAME))
        
        # desktop이 None이거나 유효하지 않은지 확인
        if self.desktop is None:
            raise RuntimeError("Desktop instance is None. Cannot create project.")
        
        try:
            self.project1 = self.desktop.create_project(path=project_path, name=self.PROJECT_NAME)
        except Exception as e:
            error_msg = f"Failed to create project '{self.PROJECT_NAME}' at path '{project_path}': {e}\n"
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            raise

    def create_design(self) :
        self.design1 = self.project1.create_design(name="maxwell_design", solver="maxwell3d", solution="AC Magnetic")

        # skip mesh setting
        oDesign = self.design1.odesign
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=", False,
                "Perform Minimal validation:=", False,
                "EnabledObjects:="	, [],
                "PerfectConductorThreshold:=", 1E+30,
                "InsulatorThreshold:="	, 1,
                "SolveFraction:="	, False,
                "Multiplier:="		, "1",
                "SkipMeshChecks:="	, True
            ], 
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:="	, "Strict",
                "IgnoreUnclassifiedObjects:=", False,
                "SkipIntersectionChecks:=", False
            ])

    def create_core(self) :
        self.design1.set_power_ferrite(cm=1.38, x=1.51, y=1.74) # 1K101 parameter [W/m^3]
        self.power_ferrite_mat = self.design1.materials["power_ferrite"]
        self.power_ferrite_mat.permeability = "3000"
        self.design1.main_core = create_core(design=self.design1, name="core", core_material="power_ferrite")

    def create_coil(self) :

        self.design1.Tx_winding, self.design1.Rx_winding = self.design1.get_excitation(excitation_name=["Tx_winding", "Rx_winding"])

        self.Tx_windings, self.Tx_N, self.Tx_coil_width, self.Tx_coil_height, self.Tx_coil_gap_x, self.Tx_coil_gap_z = create_coil(
            design = self.design1,
            name = "Tx",
            window_height = self.df_plus["nwh1"].iloc[0],
            window_length = self.df_plus["nwl1"].iloc[0],
            window_layer = 1,
            N_input = self.df_plus["N1"].iloc[0],
            width_fill_factor = self.df_plus["wff1"].iloc[0],
            space_length = self.df_plus["sl1"].iloc[0],
            space_width = self.df_plus["sw1"].iloc[0],
            shape = "circle",
            offset = [0,0,0],
            color = [255,10,10]
        )

        l1 = self.df_plus["l1"].iloc[0]
        l2 = self.df_plus["l2"].iloc[0]

        self.Rx_windings1, self.Rx_N1, self.Rx_coil_width1, self.Rx_coil_height1, self.Rx_coil_gap_x1, self.Rx_coil_gap_z1 = create_coil(
            design = self.design1,
            name = "Rx_center",
            window_height = self.df_plus["nwh2"].iloc[0],
            window_length = self.df_plus["nwl2_main"].iloc[0],
            window_layer = self.df_plus["N2_main"].iloc[0],
            N_input = 1,
            width_fill_factor = self.df_plus["wff2"].iloc[0],
            space_length = self.df_plus["sl2_main"].iloc[0],
            space_width = self.df_plus["sw2_main"].iloc[0],
            shape = "rectangle",
            offset = [0,0,0],
            color = [10, 10, 255]
        )

        if self.df_plus["N2_side"].iloc[0] > 0 :

            self.Rx_windings2, self.Rx_N2, self.Rx_coil_width2, self.Rx_coil_height2, self.Rx_coil_gap_x2, self.Rx_coil_gap_z2 = create_coil(
                design = self.design1,
                name = "Rx_side1",
                window_height = self.df_plus["nwh2"].iloc[0],
                window_length = self.df_plus["nwl2_side"].iloc[0],
                window_layer = self.df_plus["N2_side"].iloc[0],
                N_input = 1,
                width_fill_factor = self.df_plus["wff2"].iloc[0],
                space_length = self.df_plus["sl2_side"].iloc[0],
                space_width = self.df_plus["sw2_side"].iloc[0],
                shape = "rectangle",
                offset = [(-l1-l2-l1/2),0,0],
                color = [10, 10, 255]
            )       
            self.Rx_windings3, self.Rx_N3, self.Rx_coil_width3, self.Rx_coil_height3, self.Rx_coil_gap_x3, self.Rx_coil_gap_z3 = create_coil(
                design = self.design1,
                name = "Rx_side2",
                window_height = self.df_plus["nwh2"].iloc[0],
                window_length = self.df_plus["nwl2_side"].iloc[0],
                window_layer = self.df_plus["N2_side"].iloc[0], 
                N_input = 1,
                width_fill_factor = self.df_plus["wff2"].iloc[0],
                space_length = self.df_plus["sl2_side"].iloc[0],
                space_width = self.df_plus["sw2_side"].iloc[0],
                shape = "rectangle",
                offset = [(l1+l2+l1/2),0,0],
                color = [10, 10, 255]
            )

    def create_coil_section(self) :

        self.Tx_neg_sheets, self.Tx_pos_sheets = create_coil_section(design=self.design1, winding_obj=self.Tx_windings, sheet_prefix = None, plane = "ZX", rename_faces = False)
        self.Rx_neg_sheets_center, self.Rx_pos_sheets_center = create_coil_section(design=self.design1, winding_obj=self.Rx_windings1, sheet_prefix = None, plane = "ZX", rename_faces = False)
        if self.df_plus["N2_side"].iloc[0] != 0 :
            self.Rx_neg_sheets_side1, self.Rx_pos_sheets_side1 = create_coil_section(design=self.design1, winding_obj=self.Rx_windings2, sheet_prefix = None, plane = "ZX", rename_faces = False)
            self.Rx_neg_sheets_side2, self.Rx_pos_sheets_side2 = create_coil_section(design=self.design1, winding_obj=self.Rx_windings3, sheet_prefix = None, plane = "ZX", rename_faces = False)
    
    def assign_winding(self) :

        self.tx_winding = self.design1.assign_winding(
            assignment=[], 
            winding_type="Current", 
            is_solid=True, 
            current=f"{1000*math.sqrt(2)}A",
            name="Tx_winding"
        )

        self.rx_winding1 = self.design1.assign_winding(
            assignment=[], 
            winding_type="Current", 
            is_solid=True, 
            current=f"{100*math.sqrt(2)}A",
            name="Rx_winding1"
        )

    def assign_coil(self) :

        self.Tx_coil = []
        self.Rx_coil = []

        for idx, sheet in enumerate(self.Tx_neg_sheets, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_coil{idx}")
            self.Tx_coil.append(coil)

        for idx, sheet in enumerate(self.Rx_neg_sheets_center, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_center_coil{idx}")
            self.Rx_coil.append(coil)

        if self.df_plus["N2_side"].iloc[0] != 0 :
            for idx, sheet in enumerate(self.Rx_neg_sheets_side1 + self.Rx_neg_sheets_side2, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil{idx}")
                self.Rx_coil.append(coil)

    def assign_skin_depth(self) :

        freq = 1e+3

        mu0 = 4 * math.pi * 1e-7
        mu_copper = mu0 
        sigma_copper = 58000000
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu_copper * sigma_copper)) * 1e3 # in mm

        self.Tx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment=self.Tx_windings,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="2",
            name="Tx_winding_skin_depth"
        )

    def assign_radiation(self) :

        self.air_region = self.design1.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=100.0, z_neg=100.0, is_percentage=True)
        self.design1.assign_radiation(assignment=[self.air_region.name], radiation="Radiation")

    def create_setup(self) :

        self.design1.setup = self.design1.create_setup(name = "Setup1")
        self.design1.setup.properties["Max. Number of Passes"] = 8 # 10
        self.design1.setup.properties["Min. Number of Passes"] = 1
        self.design1.setup.properties["Min. Converged Passes"] = 2
        self.design1.setup.properties["Percent Error"] = 2.5 # 2.5
        self.design1.setup.properties["Frequency Setup"] = f"1kHz"

    def get_magnetic_parameter(self) :
        params = [
            ["Matrix.L(Tx_winding,Tx_winding)", f"Ltx", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)", f"Lrx", "uH"],
            ["Matrix.L(Tx_winding,Rx_winding)", f"M", "uH"],
            ["abs(Matrix.CplCoef(Tx_winding,Rx_winding))", f"k", ""],
            ["Matrix.L(Tx_winding,Tx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Lmt", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Lmr", "uH"],
            ["Matrix.L(Tx_winding,Tx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Llt", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", f"Llr", "uH"],
            ["PerWindingSolidLoss(Tx_winding)", f"Tx_loss", "W"],
            ["PerWindingSolidLoss(Rx_winding)", f"Rx_loss", "W"],
        ]

        dir = self.project.path
        mod = "write"
        import_report = None
        report_name = "magnetic_report"
        file_name = "magnetic_report"

        self.report1, self.df1 = self.design1.get_magnetic_parameter(dir=dir, parameters=params, mod=mod, import_report=import_report, report_name=report_name, file_name=file_name)

        return self.df1

    def save_results_to_csv(self, results_df, filename="simulation_results.csv"):
        """Saves the DataFrame to a CSV file in a process-safe way."""
        lock_path = filename + ".lock"
        with FileLock(lock_path):
            file_exists = os.path.isfile(filename)
            results_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        logging.info(f"Results saved to {filename}")


    def close_project(self):
        self.design1.cleanup_solution()
        self.design1.close_project()
        self.desktop.release_desktop(close_projects=True, close_on_exit=True)

    def delete_project_folder(self):
        time.sleep(10)
        try:
            project_folder = os.path.join(os.getcwd(), "simulation", self.PROJECT_NAME)
            if os.path.isdir(project_folder):
                shutil.rmtree(project_folder)
                logging.info(f"Successfully deleted project folder: {project_folder}")
        except Exception as e:
            logging.error(f"Error deleting project folder {project_folder}: {e}")

  

def run_one_loop():


    desktop = pyDesktop(version=None, non_graphical=GUI, close_on_exit=False, new_desktop=True)

    sim = Simulation(desktop=desktop)

    sim.create_simulation_name()

    sim.create_project()

    sim.create_design()

    # create input
    while True :
        sim.input_df = create_input_parameter()
        result, sim.df_plus = validation_check(sim.input_df)
        if result :
            break

    set_design_variables(sim.design1, sim.input_df)

    sim.create_core()
    sim.create_coil()
    sim.create_coil_section()
    sim.assign_winding()
    sim.assign_coil()
    sim.assign_skin_depth()
    sim.assign_radiation()
    sim.create_setup()

    sim.design1.setup.analyze(cores=4)

    
    result = pd.concat([sim.df_plus, sim.df1], axis=1)

    sim.save_results_to_csv(result)
    sim.close_project()
    sim.delete_project_folder()



def main() :

    while True :
        try:
            run_one_loop()
        except Exception as e:
            logging.error(f"Error running simulation: {e}")
            continue
        finally:
            time.sleep(10)

if __name__ == "__main__":
    main()
