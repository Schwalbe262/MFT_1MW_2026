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
        "/home1/dhj02/NEC/git/pyaedt_library/src/",
        "/home1/dw16/NEC/git/pyaedt_library/src/",
        "/home1/harry261/NEC/git/pyaedt_library/src/",
        "/home1/hmlee31/NEC/git/pyaedt_library/src/",
        "/home1/jji0930/NEC/git/pyaedt_library/src/",
        "/home1/wjddn5916/NEC/git/pyaedt_library/src/"
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
        simulation_dir = "./simulation"
        os.makedirs(simulation_dir, exist_ok=True)

        # 파일 생성 및 잠금
        with open(file_path, "a+", encoding="utf-8") as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.seek(0)
            raw = file.read().strip()

            # simulation 넘버 결정
            if raw.isdigit():
                current_num = int(raw)
            else:
                # 파일에 값이 없거나 손상, simulation 폴더 기준으로 찾기
                current_num = 1
                try:
                    existing_nums = []
                    for name in os.listdir(simulation_dir):
                        m = re.match(r"^simulation(\d+)$", name)
                        if m:
                            existing_nums.append(int(m.group(1)))
                    if existing_nums:
                        current_num = max(existing_nums) + 1
                except Exception:
                    pass

            self.num = current_num
            self.PROJECT_NAME = f"simulation{current_num}"
            next_num = current_num + 1

            # 파일에 다음 넘버 저장
            file.seek(0)
            file.truncate()
            file.write(str(next_num))
            file.flush()

    def create_project(self) :

        # simulation 디렉토리 생성 (존재하지 않으면)
        simulation_dir = "./simulation"
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir, exist_ok=True)
        
        # 절대 경로로 변환
        project_path = os.path.abspath(os.path.join(simulation_dir, self.PROJECT_NAME))
        
        # desktop이 None이거나 유효하지 않은지 확인
        if self.desktop is None:
            raise RuntimeError("Desktop instance is None. Cannot create project.")
        
        try:
            self.project = self.desktop.create_project(path=project_path, name=self.PROJECT_NAME)
        except Exception as e:
            error_msg = f"Failed to create project '{self.PROJECT_NAME}' at path '{project_path}': {e}\n"
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            raise

    def create_design(self) :
        self.design1 = self.project.create_design(name="maxwell_design", solver="maxwell3d", solution="AC Magnetic")

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

        l1 = self.df_plus["l1"].iloc[0]
        l2 = self.df_plus["l2"].iloc[0]

        self.design1.Tx_windings_main, self.N_Tx_main, self.Tx_coil_width_main, self.Tx_coil_height_main, self.Tx_coil_gap_x_main, self.Tx_coil_gap_z_main = create_coil(
            design = self.design1,
            name = "Tx_main",
            window_height = self.df_plus["nwh1"].iloc[0],
            window_length = self.df_plus["nwl1_main"].iloc[0],
            window_layer = self.df_plus["N1_main"].iloc[0],
            N_input = 1,
            width_fill_factor = self.df_plus["wff1_main"].iloc[0],
            space_length = self.df_plus["sl1_main_x"].iloc[0],
            space_width = self.df_plus["sl1_main_y"].iloc[0],
            shape = "rectangle",
            offset = [0,0,0],
            color = [255, 10, 10]
        )   

        self.design1.Rx_windings_main, self.N_Rx_main, self.Rx_coil_width_main, self.Rx_coil_height_main, self.Rx_coil_gap_x_main, self.Rx_coil_gap_z_main = create_coil(
            design = self.design1,
            name = "Rx_main",
            window_height = self.df_plus["nwh2"].iloc[0],
            window_length = self.df_plus["nwl2_main"].iloc[0],
            window_layer = self.df_plus["N2_main"].iloc[0],
            N_input = 1,
            width_fill_factor = self.df_plus["wff2_main"].iloc[0],
            space_length = self.df_plus["sl2_main_x"].iloc[0],
            space_width = self.df_plus["sl2_main_y"].iloc[0],
            shape = "rectangle",
            offset = [0,0,0],
            color = [10, 10, 255]
        )   

        if self.df_plus["N1_side"].iloc[0] != 0 :
            self.design1.Tx_windings_side, self.N_Tx_side, self.Tx_coil_width_side, self.Tx_coil_height_side, self.Tx_coil_gap_x_side, self.Tx_coil_gap_z_side = create_coil(
                design = self.design1,
                name = "Tx_side",
            window_height = self.df_plus["nwh1"].iloc[0],
            window_length = self.df_plus["nwl1_side"].iloc[0],
            window_layer = self.df_plus["N1_side"].iloc[0],
            N_input = 1,
            width_fill_factor = self.df_plus["wff1_side"].iloc[0],
            space_length = self.df_plus["sl1_side_x"].iloc[0],
            space_width = self.df_plus["sl1_side_y"].iloc[0],
            shape = "rectangle",
            offset = [(-l1-l2-l1/2),0,0],
            color = [255, 10, 10]
        )   

        if self.df_plus["N2_side"].iloc[0] != 0 :
            self.design1.Rx_windings_side, self.N_Rx_side, self.Rx_coil_width_side, self.Rx_coil_height_side, self.Rx_coil_gap_x_side, self.Rx_coil_gap_z_side = create_coil(
                design = self.design1,
            name = "Rx_side",
            window_height = self.df_plus["nwh2"].iloc[0],
            window_length = self.df_plus["nwl2_side"].iloc[0],
            window_layer = self.df_plus["N2_side"].iloc[0],
            N_input = 1,
            width_fill_factor = self.df_plus["wff2_side"].iloc[0],
            space_length = self.df_plus["sl2_side_x"].iloc[0],
            space_width = self.df_plus["sl2_side_y"].iloc[0],
            shape = "rectangle",
            offset = [(-l1-l2-l1/2),0,0],
            color = [10, 10, 255]
        )   

        if self.df_plus["N1_side"].iloc[0] == 0:
            self.design1.Tx_windings_side = []
            self.N_Tx_side = 0
            self.Tx_coil_width_side = 0
            self.Tx_coil_height_side = 0
            self.Tx_coil_gap_x_side = 0
            self.Tx_coil_gap_z_side = 0

        if self.df_plus["N2_side"].iloc[0] == 0:
            self.design1.Rx_windings_side = []
            self.N_Rx_side = 0
            self.Rx_coil_width_side = 0
            self.Rx_coil_height_side = 0
            self.Rx_coil_gap_x_side = 0
            self.Rx_coil_gap_z_side = 0

        self.Tx_windings = self.design1.Tx_windings_main + self.design1.Tx_windings_side
        self.Rx_windings = self.design1.Rx_windings_main + self.design1.Rx_windings_side
        self.design1.Tx_windings = self.Tx_windings
        self.design1.Rx_windings = self.Rx_windings



    def split_geometry(self) :

        geometrys = [self.design1.main_core] + self.design1.Tx_windings_main + self.design1.Rx_windings_main + self.design1.Tx_windings_side + self.design1.Rx_windings_side

        print(geometrys)
   
        self.design1.modeler.split(assignment=geometrys, plane="XY", sides="PositiveOnly")
        self.design1.modeler.split(assignment=geometrys, plane="XZ", sides="PositiveOnly")
        self.design1.modeler.split(assignment=geometrys, plane="YZ", sides="NegativeOnly")



    def create_coil_section(self) :

        self.Tx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix = None, plane = "YZ", rename_faces = False, mod="single")
        self.Tx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix = None, plane = "ZX", rename_faces = False, mod="single")

        self.Rx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix = None, plane = "ZX", rename_faces = False, mod="single")
        self.Rx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix = None, plane = "YZ", rename_faces = False, mod="single")
        
        if self.df_plus["N1_side"].iloc[0] != 0 :
            self.Tx_side_sheets_in, self.Tx_side_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_side, sheet_prefix = None, plane = "ZX", rename_faces = False, mod="both")
        if self.df_plus["N2_side"].iloc[0] != 0 :
            self.Rx_side_sheets_out, self.Rx_side_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_side, sheet_prefix = None, plane = "ZX", rename_faces = False, mod="both")
        
    
    def assign_winding(self) :

        self.tx_winding = self.design1.assign_winding(
            assignment=[], 
            winding_type="Current", 
            is_solid=True, 
            current=f"{1000*math.sqrt(2)}A",
            name="Tx_winding"
        )

        self.rx_winding = self.design1.assign_winding(
            assignment=[], 
            winding_type="Current", 
            is_solid=True, 
            current=f"{100*math.sqrt(2)}A",
            name="Rx_winding"
        )

    def assign_coil(self) :

        self.Tx_coil = []
        self.Rx_coil = []

        for idx, sheet in enumerate(self.Tx_main_sheets_in, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_center_coil_in_{idx}")
            self.Tx_coil.append(coil)
        for idx, sheet in enumerate(self.Tx_main_sheets_out, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_center_coil_out_{idx}")
            self.Tx_coil.append(coil)

        for idx, sheet in enumerate(self.Rx_main_sheets_in, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_center_coil_in_{idx}")
            self.Rx_coil.append(coil)
        for idx, sheet in enumerate(self.Rx_main_sheets_out, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_center_coil_out_{idx}")
            self.Rx_coil.append(coil)

        if self.df_plus["N1_side"].iloc[0] != 0 :
            for idx, sheet in enumerate(self.Tx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_side_coil_in_{idx}")
                self.Tx_coil.append(coil)
            for idx, sheet in enumerate(self.Tx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_side_coil_out_{idx}")
                self.Tx_coil.append(coil)

        if self.df_plus["N2_side"].iloc[0] != 0 :
            for idx, sheet in enumerate(self.Rx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil_in_{idx}")
                self.Rx_coil.append(coil)
            for idx, sheet in enumerate(self.Rx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_side_coil_out_{idx}")
                self.Rx_coil.append(coil)

        self.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in self.Tx_coil])
        self.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in self.Rx_coil])

        self.design1.assign_matrix(matrix_name="Matrix", assignment=["Tx_winding", "Rx_winding"])

    def assign_skin_depth(self) :

        freq = 1e+3

        mu0 = 4 * math.pi * 1e-7
        mu_copper = mu0 
        sigma_copper = 58000000
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu_copper * sigma_copper)) * 1e3 # in mm

        self.Tx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment=self.design1.Tx_windings,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="2",
            name="Tx_winding_skin_depth"
        )

        self.Rx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment= self.design1.Rx_windings,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="1",
            name="Rx_winding_skin_depth"
        )

    def assign_boundary(self) :

        self.air_region = self.design1.modeler.create_air_region(x_pos=0.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=0.0, z_neg=0.0, is_percentage=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_z, symmetry_name="Symmetry1", is_odd=False)
        self.design1.assign_symmetry(assignment=self.air_region.top_face_x, symmetry_name="Symmetry2", is_odd=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_y, symmetry_name="Symmetry3", is_odd=True)
        self.design1.assign_radiation(assignment=[self.air_region.top_face_z, self.air_region.bottom_face_x, self.air_region.top_face_y], radiation="Radiation")

    def create_setup(self) :

        self.design1.setup = self.design1.create_setup(name = "Setup1")
        self.design1.setup.properties["Max. Number of Passes"] = 8 # 10
        self.design1.setup.properties["Min. Number of Passes"] = 1
        self.design1.setup.properties["Min. Converged Passes"] = 1
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


    def save_calculation(self) :

        def _get_calculator_loss(self, obj, loss, name) :
            assignment = obj if isinstance(obj, str) else obj.name
            oModule = self.ofieldsreporter
            oModule.CalcStack("clear")
            oModule.EnterQty(loss)
            oModule.EnterVol(assignment)
            oModule.CalcOp("Integrate")
            name = f"P_{name}"
            oModule.AddNamedExpression(name, "Fields")
            return name

        _get_calculator_loss(self.design1, self.design1.Tx_windings_main[-1].name, "EMLoss", "main_winding_inner")
        _get_calculator_loss(self.design1, self.design1.Tx_windings_main[0].name, "EMLoss", "main_winding_outer")
        if self.df_plus["N1_side"].iloc[0] == 1 :
            _get_calculator_loss(self.design1, self.design1.Tx_windings_side[-1].name, "EMLoss", "side_winding_inner")
        elif self.df_plus["N1_side"].iloc[0] > 1 :
            _get_calculator_loss(self.design1, self.design1.Tx_windings_side[0].name, "EMLoss", "side_winding_inner")
            _get_calculator_loss(self.design1, self.design1.Tx_windings_side[-1].name, "EMLoss", "side_winding_outer")

        Y_components = ["P_main_winding_inner", "P_main_winding_outer"]
        if self.df_plus["N1_side"].iloc[0] > 0 :
            Y_components.append("P_side_winding_inner")
            Y_components.append("P_side_winding_outer")


        oDesign = self.design1
        oModule = oDesign.GetModule("ReportSetup")
        oModule.CreateReport("calculator_report", "Fields", "Data Table", "Setup1 : LastAdaptive", [], 
            [
                "Freq:="		, ["All"],
                "Phase:="		, ["0deg"],
                "N1:="			, ["Nominal"],
                "N2:="			, ["Nominal"],
                "N1_main:="		, ["Nominal"],
                "N1_side:="		, ["Nominal"],
                "N2_main:="		, ["Nominal"],
                "N2_side:="		, ["Nominal"],
                "w1:="			, ["Nominal"],
                "l1:="			, ["Nominal"],
                "l2:="			, ["Nominal"],
                "h1:="			, ["Nominal"],
                "cc_w2c_space_x:="	, ["Nominal"],
                "w2c_w1c_space_x:="	, ["Nominal"],
                "w1c_w2s_space_x:="	, ["Nominal"],
                "w2s_w1s_space_x:="	, ["Nominal"],
                "w1s_cs_space_x:="	, ["Nominal"],
                "cc_w2c_space_y:="	, ["Nominal"],
                "w2c_w1c_space_y:="	, ["Nominal"],
                "cs_w1s_space_y:="	, ["Nominal"],
                "w1s_w2s_space_y:="	, ["Nominal"],
                "window_ratio:="	, ["Nominal"],
                "wh1:="			, ["Nominal"],
                "wh2:="			, ["Nominal"],
                "wff1:="		, ["Nominal"],
                "wff2:="		, ["Nominal"]
            ], 
            [
                "X Component:="		, "Freq",
                "Y Component:="		, Y_components
            ])

        dir = self.project.path
        file_name = "calculator_report"
        export_path = os.path.join(dir, f"{file_name}.csv")
        oModule.ExportToFile("calculator_report", export_path, False)
        df_original = pd.read_csv(export_path)

        if self.df_plus["N1_side"].iloc[0] > 0 :
            df = df_original.iloc[:, -4:]  # 마지막 4개 컬럼만 선택
            df.columns = ["P_main_winding_inner", "P_main_winding_outer", "P_side_winding_inner", "P_side_winding_outer"]  # 컬럼 이름 변경
            self.df_calculator = df
        elif self.df_plus["N1_side"].iloc[0] == 0 :
            df = df_original.iloc[:, -2:]  # 마지막 2개 컬럼만 선택
            df.columns = ["P_main_winding_inner", "P_main_winding_outer"]  # 컬럼 이름 변경
            # P_side_winding_inner, P_side_winding_outer 컬럼을 0으로 추가
            df["P_side_winding_inner"] = 0
            df["P_side_winding_outer"] = 0
            self.df_calculator = df[["P_main_winding_inner", "P_main_winding_outer", "P_side_winding_inner", "P_side_winding_outer"]]
     


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

  


def run_one_loop(param):
    sim = None
    desktop = None
    try:
        # Use existing Desktop when possible and ensure it closes at context exit.
        with pyDesktop(version=None, non_graphical=GUI, close_on_exit=True, new_desktop=True) as desktop:
            
            param = None
            sim = Simulation(desktop=desktop)

            sim.create_simulation_name()
            sim.create_project()
            sim.create_design()

            while True:
                sim.input_df = create_input_parameter(param)
                result, sim.df_plus = validation_check(sim.input_df)
                if result:
                    break

            set_design_variables(sim.design1, sim.input_df)

            sim.create_core()
            sim.create_coil()
            sim.split_geometry()
            sim.create_coil_section()
            sim.assign_winding()
            sim.assign_coil()
            sim.assign_skin_depth()
            sim.assign_boundary()
            sim.create_setup()

            start_time = time.time()
            sim.design1.setup.analyze(cores=4)
            elapsed_time = time.time() - start_time
            simulation_time = pd.DataFrame({"time": [elapsed_time]})

            sim.get_magnetic_parameter()
            sim.save_calculation()
            result = pd.concat([sim.df_plus, sim.df1, sim.df_calculator, simulation_time], axis=1)

            try:
                sim.save_results_to_csv(result)
            except Exception as e:
                logging.exception(f"Error saving results to CSV: {e}")

            try:
                sim.close_project()
            except Exception as e:
                logging.exception(f"Error closing project: {e}")

            try:
                pass
                sim.delete_project_folder()
            except Exception as e:
                logging.exception(f"Error deleting project folder: {e}")
    except Exception as e:
        logging.exception(f"run_one_loop failed: {e}")
        if param is not None :
            pass
        if sim is not None:
            try:
                sim.close_project()
                time.sleep(1)
            except Exception:
                pass
            try:
                pass
                sim.delete_project_folder()
            except Exception:
                pass
    finally:
        # Safety net: force Desktop release even when internal close fails.
        if param is not None :
            pass
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
                time.sleep(1)
            except Exception:
                pass




def main() :

    while True :

        try:
            run_one_loop(param=None)
        except Exception as e:
            logging.exception(f"Error running simulation: {e}")
            continue
        
        finally:
            time.sleep(10)

if __name__ == "__main__":
    main()

