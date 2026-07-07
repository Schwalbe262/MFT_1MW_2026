"""
설계도면260706.pdf 반영 MFT 시뮬레이션 스크립트 (run_simulation_260514.py 기반)

도면 대비 추가/변경 사항:
  1. 코어 y방향 3분할 + 콜드플레이트(20T, 알루미늄) 4장
  2. 1차 권선 냉각 플레이트(20T): 턴1-2 사이, 턴(N-1)-N 사이 (y측면만, 양측 대칭)
  3. 권선 모서리 라운드 처리 on/off (반경은 안쪽 턴 기준 파라미터)
  4. 파라미터 직접 입력 모드 (--fixed / --params) + 기존 랜덤 스윕 모드

실행 예:
  python run_simulation_260706.py --fixed                  # 도면 치수 1회 (라운드 off)
  python run_simulation_260706.py --fixed --round          # 라운드 on
  python run_simulation_260706.py --fixed --params my.json # 일부 값 변경
  python run_simulation_260706.py --fixed --model-only     # 모델링만 하고 해석 생략
  python run_simulation_260706.py --fixed --full           # 대칭 미적용 풀모델
  python run_simulation_260706.py                          # 랜덤 스윕 (무한루프)
"""

import sys
import traceback
import logging
import portalocker
import os
import re
import json
import argparse

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 경로 설정 - 플랫폼에 따라 다르게 처리
if os.name == 'nt':  # Windows
    sys.path.insert(0, r"Y:/git/pyaedt_library/src/")
else:  # Linux/Unix
    possible_paths = [
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

from module.input_parameter_260706 import (
    create_input_parameter,
    set_design_variables,
    validation_check,
    get_tx_y_gaps,
    get_drawing_default_params,
    get_design_var_columns,
    sym_cut_count,
)
from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    create_coil_section,
)

from ansys.aedt.core import settings

settings.skip_license_check = True
settings.wait_for_license = False

if os.name == 'nt':  # Windows
    GUI = False
else:  # Linux/Unix
    GUI = True

from filelock import FileLock
import shutil


PLATE_COLOR = [144, 190, 144]
PAD_COLOR = [200, 160, 200]


def _git_hash():
    """데이터 이력 추적용: 이 코드 버전의 git 커밋 해시"""
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=BASE_DIR, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


GIT_HASH = _git_hash()


class Simulation():

    def __init__(self, desktop=None):

        self.NUM_CORE = 4
        self.NUM_TASK = 1
        self.desktop = desktop
        self.full_model = False

    def create_simulation_name(self):

        # slurm_scheduler dynamic_packed_srun 모드: SIMULATION_ID 환경변수 기반 이름
        # (공유 파일시스템에서 카운터 파일 락 경합 없이 고유 이름 보장)
        sim_id = os.environ.get("SIMULATION_ID")
        if sim_id:
            job_id = os.environ.get("SLURM_JOB_ID", "job")
            self.num = sim_id
            self.PROJECT_NAME = f"simulation_{job_id}_{sim_id}"
            os.makedirs("./simulation", exist_ok=True)
            return

        file_path = "./simulation_num.txt"
        simulation_dir = "./simulation"
        os.makedirs(simulation_dir, exist_ok=True)

        with open(file_path, "a+", encoding="utf-8") as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.seek(0)
            raw = file.read().strip()

            if raw.isdigit():
                current_num = int(raw)
            else:
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

            file.seek(0)
            file.truncate()
            file.write(str(next_num))
            file.flush()

    def create_project(self):

        simulation_dir = "./simulation"
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir, exist_ok=True)

        project_path = os.path.abspath(os.path.join(simulation_dir, self.PROJECT_NAME))

        if self.desktop is None:
            raise RuntimeError("Desktop instance is None. Cannot create project.")

        try:
            self.project = self.desktop.create_project(path=project_path, name=self.PROJECT_NAME)
        except Exception as e:
            error_msg = f"Failed to create project '{self.PROJECT_NAME}' at path '{project_path}': {e}\n"
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            raise

    def create_design(self, name="maxwell_design"):
        self.design1 = self.project.create_design(name=name, solver="maxwell3d", solution="AC Magnetic")

        # skip mesh setting
        oDesign = self.design1.odesign
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=", False,
                "Perform Minimal validation:=", False,
                "EnabledObjects:=", [],
                "PerfectConductorThreshold:=", 1E+30,
                "InsulatorThreshold:=", 1,
                "SolveFraction:=", False,
                "Multiplier:=", "1",
                "SkipMeshChecks:=", True
            ],
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:=", "Strict",
                "IgnoreUnclassifiedObjects:=", False,
                "SkipIntersectionChecks:=", False
            ])

    def create_thermal_pad_material(self):
        # 서멀패드(실리콘 패드): 비도전성 (AC Magnetic 해석에서는 절연체로 동작)
        if "thermal_pad" not in self.design1.materials.material_keys:
            mat = self.design1.materials.add_material("thermal_pad")
            mat.conductivity = 0
            mat.permittivity = 4
            mat.permeability = 1
            mat.thermal_conductivity = 0.2  # W/(m*K)

    def create_core(self):
        # 2605SA1/1K101 코어손실 계수 [W/m^3, Hz 기준] (데이터시트 kHz 계수에서 변환됨)
        # 재질은 프로젝트 스코프이므로 두 번째 디자인에서는 재사용
        if "power_ferrite" not in self.design1.materials.material_keys:
            self.design1.set_power_ferrite(
                cm=float(self.df_plus["core_cm"].iloc[0]),
                x=float(self.df_plus["core_x"].iloc[0]),
                y=float(self.df_plus["core_y"].iloc[0])
            )
        self.power_ferrite_mat = self.design1.materials["power_ferrite"]
        self.power_ferrite_mat.permeability = "3000"

        self.create_thermal_pad_material()

        n_group = int(self.df_plus["n_core_group"].iloc[0])
        plate_on = int(self.df_plus["core_plate_on"].iloc[0]) != 0
        pad_on = float(self.df_plus["core_plate_pad_t"].iloc[0]) > 0

        core_objs, plate_objs, pad_objs = create_core(
            design=self.design1,
            name="core",
            core_material="power_ferrite",
            n_group=n_group,
            plate_material="aluminum",
            pad_material="thermal_pad",
            plate_on=plate_on,
            pad_on=pad_on,
            plate_color=PLATE_COLOR,
            pad_color=PAD_COLOR
        )
        self.design1.core_objs = core_objs
        self.design1.core_plates = plate_objs
        self.design1.core_pads = pad_objs

    def _op_temp_conductor_material(self):
        """운전 온도 기준 도전율의 구리 재질 생성 (기본 80C).
        20C 구리(5.8e7 S/m) 기준이면 실물(~80-100C) 권선손실을 ~25% 과소평가한다.
        sigma(T) = sigma20 / (1 + 0.00393*(T-20))"""
        T = float(self.df_plus["conductor_temp_C"].iloc[0])
        name = f"copper_{int(round(T))}C"
        mats = self.design1.materials
        if name not in mats.material_keys:
            m = mats.add_material(name)
            m.conductivity = 5.8e7 / (1.0 + 0.00393 * (T - 20.0))
            m.permeability = 0.999991
        return name

    def create_coil(self):

        l1 = self.df_plus["l1"].iloc[0]
        l2 = self.df_plus["l2"].iloc[0]

        conductor_mat = self._op_temp_conductor_material()

        round_corner = int(self.df_plus["round_corner"].iloc[0]) != 0
        corner_radius = float(self.df_plus["corner_radius"].iloc[0]) if round_corner else None
        corner_segments = int(self.df_plus["corner_segments"].iloc[0])

        # 1차 중심 권선: y방향은 냉각판 슬롯 간격으로 벌어짐
        tx_y_gaps, tx_slot_indices = get_tx_y_gaps(self.df_plus)

        self.design1.Tx_windings_main, self.N_Tx_main, self.Tx_coil_width_main, self.Tx_coil_height_main, self.Tx_coil_gap_x_main, self.Tx_coil_gap_z_main = create_coil(
            design=self.design1,
            name="Tx_main",
            window_height=self.df_plus["nwh1"].iloc[0],
            window_length=self.df_plus["nwl1_main"].iloc[0],
            window_layer=self.df_plus["N1_main"].iloc[0],
            N_input=1,
            width_fill_factor=self.df_plus["wff1_main"].iloc[0],
            space_length=self.df_plus["sl1_main_x"].iloc[0],
            space_width=self.df_plus["sl1_main_y"].iloc[0],
            shape="rectangle",
            offset=[0, 0, 0],
            color=[255, 10, 10],
            y_slot_gaps=tx_y_gaps,
            round_corner=round_corner,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            material=conductor_mat
        )

        self.design1.Rx_windings_main, self.N_Rx_main, self.Rx_coil_width_main, self.Rx_coil_height_main, self.Rx_coil_gap_x_main, self.Rx_coil_gap_z_main = create_coil(
            design=self.design1,
            name="Rx_main",
            window_height=self.df_plus["nwh2"].iloc[0],
            window_length=self.df_plus["nwl2_main"].iloc[0],
            window_layer=self.df_plus["N2_main"].iloc[0],
            N_input=1,
            width_fill_factor=self.df_plus["wff2_main"].iloc[0],
            space_length=self.df_plus["sl2_main_x"].iloc[0],
            space_width=self.df_plus["sl2_main_y"].iloc[0],
            shape="rectangle",
            offset=[0, 0, 0],
            color=[10, 10, 255],
            round_corner=round_corner,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            material=conductor_mat
        )

        if self.df_plus["N1_side"].iloc[0] != 0:
            self.design1.Tx_windings_side, self.N_Tx_side, self.Tx_coil_width_side, self.Tx_coil_height_side, self.Tx_coil_gap_x_side, self.Tx_coil_gap_z_side = create_coil(
                design=self.design1,
                name="Tx_side",
                window_height=self.df_plus["nwh1"].iloc[0],
                window_length=self.df_plus["nwl1_side"].iloc[0],
                window_layer=self.df_plus["N1_side"].iloc[0],
                N_input=1,
                width_fill_factor=self.df_plus["wff1_side"].iloc[0],
                space_length=self.df_plus["sl1_side_x"].iloc[0],
                space_width=self.df_plus["sl1_side_y"].iloc[0],
                shape="rectangle",
                offset=[(-l1 - l2 - l1 / 2), 0, 0],
                color=[255, 10, 10],
                round_corner=round_corner,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                material=conductor_mat
            )

        if self.df_plus["N2_side"].iloc[0] != 0:
            self.design1.Rx_windings_side, self.N_Rx_side, self.Rx_coil_width_side, self.Rx_coil_height_side, self.Rx_coil_gap_x_side, self.Rx_coil_gap_z_side = create_coil(
                design=self.design1,
                name="Rx_side",
                window_height=self.df_plus["nwh2"].iloc[0],
                window_length=self.df_plus["nwl2_side"].iloc[0],
                window_layer=self.df_plus["N2_side"].iloc[0],
                N_input=1,
                width_fill_factor=self.df_plus["wff2_side"].iloc[0],
                space_length=self.df_plus["sl2_side_x"].iloc[0],
                space_width=self.df_plus["sl2_side_y"].iloc[0],
                shape="rectangle",
                offset=[(-l1 - l2 - l1 / 2), 0, 0],
                color=[10, 10, 255],
                round_corner=round_corner,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                material=conductor_mat
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

        # 풀모델: 대칭이 없으므로 반대쪽(+x) 측면 레그의 측면 권선도 실제로 생성
        self.design1.Tx_windings_side2 = []
        self.design1.Rx_windings_side2 = []
        if self.full_model:
            if self.df_plus["N1_side"].iloc[0] != 0:
                self.design1.Tx_windings_side2, _, _, _, _, _ = create_coil(
                    design=self.design1,
                    name="Tx_side2",
                    window_height=self.df_plus["nwh1"].iloc[0],
                    window_length=self.df_plus["nwl1_side"].iloc[0],
                    window_layer=self.df_plus["N1_side"].iloc[0],
                    N_input=1,
                    width_fill_factor=self.df_plus["wff1_side"].iloc[0],
                    space_length=self.df_plus["sl1_side_x"].iloc[0],
                    space_width=self.df_plus["sl1_side_y"].iloc[0],
                    shape="rectangle",
                    offset=[(l1 + l2 + l1 / 2), 0, 0],
                    color=[255, 10, 10],
                    round_corner=round_corner,
                    corner_radius=corner_radius,
                    corner_segments=corner_segments,
                    material=conductor_mat
                )
            if self.df_plus["N2_side"].iloc[0] != 0:
                self.design1.Rx_windings_side2, _, _, _, _, _ = create_coil(
                    design=self.design1,
                    name="Rx_side2",
                    window_height=self.df_plus["nwh2"].iloc[0],
                    window_length=self.df_plus["nwl2_side"].iloc[0],
                    window_layer=self.df_plus["N2_side"].iloc[0],
                    N_input=1,
                    width_fill_factor=self.df_plus["wff2_side"].iloc[0],
                    space_length=self.df_plus["sl2_side_x"].iloc[0],
                    space_width=self.df_plus["sl2_side_y"].iloc[0],
                    shape="rectangle",
                    offset=[(l1 + l2 + l1 / 2), 0, 0],
                    color=[10, 10, 255],
                    round_corner=round_corner,
                    corner_radius=corner_radius,
                    corner_segments=corner_segments,
                    material=conductor_mat
                )

        # 1차 권선 냉각 플레이트 (y측면 슬롯, 양측 대칭, 서멀패드|알루미늄|서멀패드)
        wcp_on = int(self.df_plus["wcp_on"].iloc[0]) != 0
        if wcp_on and len(tx_slot_indices) > 0:
            self.design1.wcp_plates, self.design1.wcp_pads = create_winding_cooling_plates(
                design=self.design1,
                name="Tx_main_wcp",
                space_width=self.df_plus["sl1_main_y"].iloc[0],
                coil_width=self.Tx_coil_width_main,
                y_gaps=tx_y_gaps,
                slot_indices=tx_slot_indices,
                wcp_len_x=float(self.df_plus["wcp_len_x"].iloc[0]),
                wcp_t=float(self.df_plus["wcp_t"].iloc[0]),
                pad_t=float(self.df_plus["wcp_pad_t"].iloc[0]),
                height=float(self.df_plus["nwh1"].iloc[0]),
                plate_material="aluminum",
                pad_material="thermal_pad",
                plate_color=PLATE_COLOR,
                pad_color=PAD_COLOR,
                offset=[0, 0, 0]
            )
        else:
            self.design1.wcp_plates = []
            self.design1.wcp_pads = []

        self.Tx_windings = self.design1.Tx_windings_main + self.design1.Tx_windings_side + self.design1.Tx_windings_side2
        self.Rx_windings = self.design1.Rx_windings_main + self.design1.Rx_windings_side + self.design1.Rx_windings_side2
        self.design1.Tx_windings = self.Tx_windings
        self.design1.Rx_windings = self.Rx_windings

    def split_geometry(self):

        # 풀모델: 대칭 분할 없이 전체 지오메트리 유지
        if self.full_model:
            return

        geometrys = (self.design1.core_objs + self.design1.core_plates + self.design1.core_pads
                     + self.design1.wcp_plates + self.design1.wcp_pads
                     + self.design1.Tx_windings_main + self.design1.Rx_windings_main
                     + self.design1.Tx_windings_side + self.design1.Rx_windings_side)

        print(geometrys)

        self.design1.modeler.split(assignment=geometrys, plane="XY", sides="PositiveOnly")
        self.design1.modeler.split(assignment=geometrys, plane="XZ", sides="PositiveOnly")
        self.design1.modeler.split(assignment=geometrys, plane="YZ", sides="NegativeOnly")

        # 대칭 분할로 완전히 잘려나간 오브젝트(y<0 쪽 콜드플레이트/냉각판 등)를 리스트에서 제거
        # (이후 eddy 설정/손실 계산이 존재하지 않는 오브젝트를 참조하지 않도록)
        existing = set(self.design1.modeler.object_names)
        self.design1.core_objs = [o for o in self.design1.core_objs if o.name in existing]
        self.design1.core_plates = [o for o in self.design1.core_plates if o.name in existing]
        self.design1.core_pads = [o for o in self.design1.core_pads if o.name in existing]
        self.design1.wcp_plates = [o for o in self.design1.wcp_plates if o.name in existing]
        self.design1.wcp_pads = [o for o in self.design1.wcp_pads if o.name in existing]

    def create_coil_section(self):

        if self.full_model:
            self._create_coil_section_full()
            return

        self.Tx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix=None, plane="YZ", rename_faces=False, mod="single")
        self.Tx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix=None, plane="ZX", rename_faces=False, mod="single")

        self.Rx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix=None, plane="ZX", rename_faces=False, mod="single")
        self.Rx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix=None, plane="YZ", rename_faces=False, mod="single")

        if self.df_plus["N1_side"].iloc[0] != 0:
            self.Tx_side_sheets_in, self.Tx_side_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_side, sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")
        if self.df_plus["N2_side"].iloc[0] != 0:
            self.Rx_side_sheets_out, self.Rx_side_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_side, sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")

    def _create_coil_section_full(self):
        """
        풀모델용 단면 생성: 닫힌 링 도체는 자르지 않고 ZX 평면 단면 시트를
        턴당 1개만 남겨 터미널로 사용한다. (ZX 단면은 링당 2개 생기므로
        한쪽을 삭제. 남기는 다리/극성은 하프모델의 전류 방향 관례와 일치시킴)
        """
        def _pick(winding_objs, keep):
            x_neg, x_pos = create_coil_section(design=self.design1, winding_obj=winding_objs,
                                               sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")
            kept, drop = (x_neg, x_pos) if keep == "neg" else (x_pos, x_neg)
            if drop:
                self.design1.modeler.delete(drop)
            return kept

        # 중심 권선: x- 다리 시트 사용 (Tx는 Negative, Rx는 Positive 극성 -> 상호 반대 방향)
        self.Tx_main_sheets_full = _pick(self.design1.Tx_windings_main, keep="neg")
        self.Rx_main_sheets_full = _pick(self.design1.Rx_windings_main, keep="neg")

        # 측면 권선 (-x 레그): 하프모델과 동일한 다리 선택
        self.Tx_side_sheets_full = []
        self.Rx_side_sheets_full = []
        self.Tx_side2_sheets_full = []
        self.Rx_side2_sheets_full = []
        if self.df_plus["N1_side"].iloc[0] != 0:
            self.Tx_side_sheets_full = _pick(self.design1.Tx_windings_side, keep="neg")   # 바깥 다리
            self.Tx_side2_sheets_full = _pick(self.design1.Tx_windings_side2, keep="pos")  # 미러: 바깥 다리
        if self.df_plus["N2_side"].iloc[0] != 0:
            self.Rx_side_sheets_full = _pick(self.design1.Rx_windings_side, keep="pos")   # 안쪽 다리
            self.Rx_side2_sheets_full = _pick(self.design1.Rx_windings_side2, keep="neg")  # 미러: 안쪽 다리

    def _assign_coil_full(self):
        """풀모델: 턴당 터미널 1개. 미러(+x) 측 권선은 반사 대칭으로 순환 방향이
        반전되므로 극성을 반대로 지정한다."""
        self.Tx_coil = []
        self.Rx_coil = []

        for idx, sheet in enumerate(self.Tx_main_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_center_coil_{idx}"))
        for idx, sheet in enumerate(self.Rx_main_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_center_coil_{idx}"))

        for idx, sheet in enumerate(self.Tx_side_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_side_coil_{idx}"))
        for idx, sheet in enumerate(self.Tx_side2_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_side2_coil_{idx}"))

        for idx, sheet in enumerate(self.Rx_side_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil_{idx}"))
        for idx, sheet in enumerate(self.Rx_side2_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_side2_coil_{idx}"))

        self.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in self.Tx_coil])
        self.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in self.Rx_coil])

    def assign_winding(self, mode="matrix"):
        """
        mode="matrix": Tx/Rx 모두 정격 전류원 (L/k 매트릭스용, 기존 방식)
        mode="loss"  : Tx 전압원(V1) + Rx 정격 전류원 -> 코어 자속이 전압으로 결정되어
                       권선손실+코어손실을 한 번에 해석 (전류 강제로 인한 비물리적 자속 방지)
        """
        I1 = float(self.df_plus["I1_rated"].iloc[0])
        I2 = float(self.df_plus["I2_rated"].iloc[0])

        if mode == "loss":
            # 손실 원샷 여자 (풀모델): Tx 전압원(V1) + Rx 정격 전류원.
            # Maxwell이 1차 전류(부하분 + 자화분)를 스스로 풀어 코어 자속이 실제 운전 수준이 됨.
            # 검증: 무부하 케이스에서 코어손실/턴손실이 자화전류(Im=V1/wLm) 주입 방식과 1% 이내 일치.
            # 주의: 전압 권선의 InputCurrent 리포트는 0으로 표시될 수 있으나 (표시 아티팩트)
            #       실제 해는 유효함 (InducedVoltage ~= V1, 손실/자속 정상).
            V1 = float(self.df_plus["V1_rms"].iloc[0])
            # P_target 자동 위상이 계산되어 있으면 우선 사용
            phase2 = getattr(self, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(self.df_plus["I2_phase_deg"].iloc[0])

            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Voltage",
                is_solid=True,
                voltage=f"{V1 * math.sqrt(2)}V",
                resistance=0,
                inductance=0,
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I2 * math.sqrt(2)}A",
                phase=f"{phase2}deg",
                name="Rx_winding"
            )
        elif mode == "loss_sym":
            # 손실 원샷 여자 (대칭 1/8, 캠페인용): 전압원이 대칭 터미널 구조에서 무효이므로
            # Tx 전류 = 부하분(N2/N1 x I2, Rx와 역상) + 자화분(Im = sqrt(2)V1/(w Lm_true), -90deg)
            # 페이저 합을 직접 주입. 선형 해석이므로 올바른 복소 전류 = 실제 운전 자속/전류 재현.
            # (Lm은 design1 매트릭스에서 자동 취득 - run_one_loop에서 self.loss_I1_* 설정)
            phase2 = getattr(self, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(self.df_plus["I2_phase_deg"].iloc[0])

            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{self.loss_I1_peak}A",
                phase=f"{self.loss_I1_phase_deg}deg",
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I2 * math.sqrt(2)}A",
                phase=f"{phase2}deg",
                name="Rx_winding"
            )
        else:
            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I1 * math.sqrt(2)}A",
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I2 * math.sqrt(2)}A",
                name="Rx_winding"
            )

    def assign_coil(self):

        if self.full_model:
            self._assign_coil_full()
            return

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

        if self.df_plus["N1_side"].iloc[0] != 0:
            for idx, sheet in enumerate(self.Tx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_side_coil_in_{idx}")
                self.Tx_coil.append(coil)
            for idx, sheet in enumerate(self.Tx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_side_coil_out_{idx}")
                self.Tx_coil.append(coil)

        if self.df_plus["N2_side"].iloc[0] != 0:
            for idx, sheet in enumerate(self.Rx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil_in_{idx}")
                self.Rx_coil.append(coil)
            for idx, sheet in enumerate(self.Rx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_side_coil_out_{idx}")
                self.Rx_coil.append(coil)

        self.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in self.Tx_coil])
        self.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in self.Rx_coil])

    def assign_matrix(self):
        self.design1.assign_matrix(matrix_name="Matrix", assignment=["Tx_winding", "Rx_winding"])

    def assign_core_loss(self):
        """loss 디자인: 코어 그룹에 코어손실 계산 활성화 (Power Ferrite 계수는 create_core에서 설정)"""
        try:
            self.design1.set_core_losses(
                assignment=[c.name for c in self.design1.core_objs],
                core_loss_on_field=False
            )
        except Exception as e:
            logging.warning(f"Failed to enable core losses: {e}")

    def assign_skin_depth(self):

        freq = float(self.df_plus["freq"].iloc[0])

        mu0 = 4 * math.pi * 1e-7
        mu_copper = mu0
        sigma_copper = 58000000
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu_copper * sigma_copper)) * 1e3  # in mm

        self.Tx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment=self.design1.Tx_windings,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="2",
            name="Tx_winding_skin_depth"
        )

        rx_mode = str(self.df_plus["rx_mesh_mode"].iloc[0])
        cw2 = float(self.df_plus["cw2"].iloc[0])

        if rx_mode == "length":
            # 실험적: foil 두께 방향 2요소 수준의 length 기반 메시 (벤치마크용)
            self.Rx_length_mesh = self.design1.mesh.assign_length_mesh(
                assignment=self.design1.Rx_windings,
                maximum_length=f"{max(cw2 / 2, 0.1)}mm",
                maximum_elements=None,
                name="Rx_winding_length_mesh"
            )
        elif rx_mode == "length-coarse":
            # 실험적: foil 두께 1요소 (최대 가속 후보, 벤치마크용)
            self.Rx_length_mesh = self.design1.mesh.assign_length_mesh(
                assignment=self.design1.Rx_windings,
                maximum_length=f"{cw2}mm",
                maximum_elements=None,
                name="Rx_winding_length_mesh"
            )
        else:
            # 기본: 기존 skin-depth op (proximity effect 반영 검증된 설정)
            self.Rx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
                assignment=self.design1.Rx_windings,
                skin_depth=f'{skin_depth}mm',
                triangulation_max_length='50mm',
                layers_number="1",
                name="Rx_winding_skin_depth"
            )

    def assign_plate_settings(self):
        """콜드플레이트/권선 냉각판 (알루미늄) 와전류 설정 + 메시"""

        plates = self.design1.core_plates + self.design1.wcp_plates
        if not plates:
            return

        plate_names = [p.name for p in plates]

        try:
            self.design1.eddy_effects_on(
                assignment=plate_names,
                enable_eddy_effects=True,
                enable_displacement_current=False
            )
        except Exception as e:
            logging.warning(f"Failed to set eddy effects on plates: {e}")

        freq = float(self.df_plus["freq"].iloc[0])
        mu0 = 4 * math.pi * 1e-7
        sigma_al = 3.8e+7
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu0 * sigma_al)) * 1e3  # in mm (~2.6mm @1kHz)

        try:
            self.plate_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
                assignment=plates,
                skin_depth=f'{skin_depth}mm',
                triangulation_max_length='50mm',
                layers_number="1",
                name="plate_skin_depth"
            )
        except Exception as e:
            logging.warning(f"Failed to assign skin depth mesh on plates: {e}")

    def assign_boundary(self):

        if self.full_model:
            # 풀모델: 대칭 경계 없이 전방향 air region + 전면 radiation
            self.air_region = self.design1.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=100.0, z_neg=100.0, is_percentage=True)
            self.design1.assign_radiation(
                assignment=[
                    self.air_region.top_face_x, self.air_region.bottom_face_x,
                    self.air_region.top_face_y, self.air_region.bottom_face_y,
                    self.air_region.top_face_z, self.air_region.bottom_face_z
                ],
                radiation="Radiation"
            )
            return

        self.air_region = self.design1.modeler.create_air_region(x_pos=0.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=0.0, z_neg=0.0, is_percentage=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_z, symmetry_name="Symmetry1", is_odd=False)
        self.design1.assign_symmetry(assignment=self.air_region.top_face_x, symmetry_name="Symmetry2", is_odd=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_y, symmetry_name="Symmetry3", is_odd=True)
        self.design1.assign_radiation(assignment=[self.air_region.top_face_z, self.air_region.bottom_face_x, self.air_region.top_face_y], radiation="Radiation")

    def create_setup(self):

        self.design1.setup = self.design1.create_setup(name="Setup1")
        self.design1.setup.properties["Max. Number of Passes"] = int(self.df_plus["max_passes"].iloc[0])
        self.design1.setup.properties["Min. Number of Passes"] = 1
        self.design1.setup.properties["Min. Converged Passes"] = 2
        self.design1.setup.properties["Percent Error"] = float(self.df_plus["percent_error"].iloc[0])
        self.design1.setup.properties["Frequency Setup"] = f"{float(self.df_plus['freq'].iloc[0])}Hz"

    def get_magnetic_parameter(self):
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

    def _report_variations(self):
        """디자인 변수 목록에서 리포트 variation 리스트를 동적 생성"""
        variations = [
            "Freq:=", ["All"],
            "Phase:=", ["0deg"],
        ]
        for col in get_design_var_columns(self.input_df):
            variations += [f"{col}:=", ["Nominal"]]
        return variations

    def _export_field_report(self, report_name, Y_components):
        oDesign = self.design1
        oModule = oDesign.GetModule("ReportSetup")
        oModule.CreateReport(
            report_name, "Fields", "Data Table", "Setup1 : LastAdaptive", [],
            self._report_variations(),
            [
                "X Component:=", "Freq",
                "Y Component:=", Y_components
            ])

        dir = self.project.path
        export_path = os.path.join(dir, f"{report_name}.csv")
        oModule.ExportToFile(report_name, export_path, False)
        return pd.read_csv(export_path)

    def save_calculation(self):

        def _get_calculator_loss(self, obj, loss, name):
            assignment = obj if isinstance(obj, str) else obj.name
            oModule = self.ofieldsreporter
            oModule.CalcStack("clear")
            oModule.EnterQty(loss)
            oModule.EnterVol(assignment)
            oModule.CalcOp("Integrate")
            name = f"P_{name}"
            oModule.AddNamedExpression(name, "Fields")
            return name

        # ---- 1차 권선 손실 ----
        _get_calculator_loss(self.design1, self.design1.Tx_windings_main[0].name, "EMLoss", "Tx_main_winding_inner")
        _get_calculator_loss(self.design1, self.design1.Tx_windings_main[-1].name, "EMLoss", "Tx_main_winding_outer")
        if self.df_plus["N1_side"].iloc[0] > 0:
            _get_calculator_loss(self.design1, self.design1.Tx_windings_side[0].name, "EMLoss", "Tx_side_winding_inner")
            _get_calculator_loss(self.design1, self.design1.Tx_windings_side[-1].name, "EMLoss", "Tx_side_winding_outer")

        # ---- 2차 권선 손실 ----
        _get_calculator_loss(self.design1, self.design1.Rx_windings_main[0].name, "EMLoss", "Rx_main_winding_inner")
        _get_calculator_loss(self.design1, self.design1.Rx_windings_main[-1].name, "EMLoss", "Rx_main_winding_outer")
        if self.df_plus["N2_side"].iloc[0] > 0:
            _get_calculator_loss(self.design1, self.design1.Rx_windings_side[0].name, "EMLoss", "Rx_side_winding_inner")
            _get_calculator_loss(self.design1, self.design1.Rx_windings_side[-1].name, "EMLoss", "Rx_side_winding_outer")

        # ---- 플레이트 손실 (콜드플레이트 / 권선 냉각판) ----
        core_plate_exprs = []
        for p in self.design1.core_plates:
            core_plate_exprs.append(_get_calculator_loss(self.design1, p.name, "EMLoss", p.name))
        wcp_exprs = []
        for p in self.design1.wcp_plates:
            wcp_exprs.append(_get_calculator_loss(self.design1, p.name, "EMLoss", p.name))

        # ---- report1: Tx 권선 손실 ----
        Y_components = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer"]
        if self.df_plus["N1_side"].iloc[0] > 0:
            Y_components.append("P_Tx_side_winding_inner")
            Y_components.append("P_Tx_side_winding_outer")

        df_original1 = self._export_field_report("calculator_report1", Y_components)

        if self.df_plus["N1_side"].iloc[0] > 0:
            df = df_original1.iloc[:, -4:]
            df.columns = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer", "P_Tx_side_winding_inner", "P_Tx_side_winding_outer"]
            self.df_calculator1 = df
        else:
            df = df_original1.iloc[:, -2:]
            df.columns = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer"]
            df["P_Tx_side_winding_inner"] = 0
            df["P_Tx_side_winding_outer"] = 0
            self.df_calculator1 = df[["P_Tx_main_winding_inner", "P_Tx_main_winding_outer", "P_Tx_side_winding_inner", "P_Tx_side_winding_outer"]]

        # ---- report2: Rx 권선 손실 ----
        Y_components = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer"]
        if self.df_plus["N2_side"].iloc[0] > 0:
            Y_components.append("P_Rx_side_winding_inner")
            Y_components.append("P_Rx_side_winding_outer")

        df_original2 = self._export_field_report("calculator_report2", Y_components)

        if self.df_plus["N2_side"].iloc[0] > 0:
            df = df_original2.iloc[:, -4:]
            df.columns = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer", "P_Rx_side_winding_inner", "P_Rx_side_winding_outer"]
            self.df_calculator2 = df
        else:
            df = df_original2.iloc[:, -2:]
            df.columns = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer"]
            df["P_Rx_side_winding_inner"] = 0
            df["P_Rx_side_winding_outer"] = 0
            self.df_calculator2 = df[["P_Rx_main_winding_inner", "P_Rx_main_winding_outer", "P_Rx_side_winding_inner", "P_Rx_side_winding_outer"]]

        # ---- report3: 플레이트 손실 ----
        plate_exprs = core_plate_exprs + wcp_exprs
        if plate_exprs:
            df_original3 = self._export_field_report("calculator_report3", plate_exprs)
            df3 = df_original3.iloc[:, -len(plate_exprs):]
            df3.columns = plate_exprs
            P_core_plate = df3[core_plate_exprs].sum(axis=1) if core_plate_exprs else 0
            P_winding_plate = df3[wcp_exprs].sum(axis=1) if wcp_exprs else 0
            self.df_calculator3 = pd.DataFrame({
                "P_core_plate": P_core_plate if core_plate_exprs else [0],
                "P_winding_plate": P_winding_plate if wcp_exprs else [0],
            })
        else:
            self.df_calculator3 = pd.DataFrame({"P_core_plate": [0], "P_winding_plate": [0]})

    def _sym_cut_count(self, obj_name):
        """대칭 1/8 분할 절단면 수 (공용 로직 위임)"""
        return sym_cut_count(obj_name, self.df_plus)

    def _mirror_mult(self, obj_name):
        """대칭 loss 디자인에서 삭제된 미러 오브젝트 몫을 총계에 반영하는 배수.
        (y=0에 걸치지 않는 코어/플레이트/냉각판은 y<0 쪽 미러가 삭제되어 있으므로 x2)
        풀모델이면 항상 1 (모든 오브젝트가 실존)."""
        if not getattr(self, "loss_is_sym", False):
            return 1.0
        name = obj_name
        if name.startswith("Tx_main_wcp"):
            return 2.0  # _p만 잔존 (_n 미러 삭제)
        if name.startswith("core_plate") or (name.startswith("core_") and not name.startswith("core_plate")):
            return 1.0 if self._sym_cut_count(name) == 3 else 2.0  # y=0 스팬이면 미러 없음
        return 1.0

    def _phys_factor(self, expr_name, is_core_loss):
        """대칭 loss 디자인의 적분값 -> 실물값 환산 계수 (풀모델이면 1)"""
        if not getattr(self, "loss_is_sym", False):
            return 1.0
        # 표현식 이름에서 오브젝트 이름 추출: P_core_3 / P_turn_Rx_main_0_0 / P_Tx_main_group ...
        name = expr_name
        for prefix in ("P_turn_", "P_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = name.replace("_group", "")
        c = self._sym_cut_count(name)
        if is_core_loss:
            core_y = float(self.df_plus["core_y"].iloc[0])
            return (2 ** c) / (2 ** core_y)
        return (2 ** c) / 4.0

    def _calc_field_expr(self, obj_name, quantity, op, expr_name):
        """계산기: quantity를 오브젝트 볼륨에 대해 op(Integrate/Mean/Maximum) 후 named expression 등록.
        quantity="B_peak"는 위상 무관한 자속밀도 페이저 크기 (Mag_B는 Phase=0 순간값이라 부적합)."""
        oModule = self.design1.ofieldsreporter
        oModule.CalcStack("clear")
        if quantity == "B_peak":
            oModule.EnterQty("B")
            oModule.CalcOp("CmplxMag")
            oModule.CalcOp("Mag")
        else:
            oModule.EnterQty(quantity)
        oModule.EnterVol(obj_name)
        oModule.CalcOp(op)
        try:
            oModule.AddNamedExpression(expr_name, "Fields")
        except Exception:
            # 동일 이름 표현식이 이미 존재 (save_calculation에서 생성된 경우) -> 재사용
            logging.info(f"named expression {expr_name} already exists - reusing")
        return expr_name

    def _calc_group_loss(self, objs, expr_name, quantity="EMLoss"):
        """여러 오브젝트의 손실 적분 합을 하나의 named expression으로 등록"""
        oModule = self.design1.ofieldsreporter
        oModule.CalcStack("clear")
        for i, o in enumerate(objs):
            name = o if isinstance(o, str) else o.name
            oModule.EnterQty(quantity)
            oModule.EnterVol(name)
            oModule.CalcOp("Integrate")
            if i > 0:
                oModule.CalcOp("+")
        oModule.AddNamedExpression(expr_name, "Fields")
        return expr_name

    def save_loss_reports(self):
        """
        loss 디자인 전용 추출:
          - 코어 그룹별 CoreLoss 적분 (P_core_i, P_core_total)
          - 코어 그룹별 B 평균/최대 (B_mean_core, B_max_core) - 자속밀도 sanity check
          - Tx 해석 전류 I1 (전압원이므로 해석 결과, 정격+자화 성분 검증용)
          - 권선 그룹 총손실 + explicit 턴별 손실 (열해석 배분용) -> self.loss_map
        """
        n_exp = int(self.df_plus["n_explicit_turns"].iloc[0])

        # ---- 코어손실 + B ----
        core_exprs = []
        b_mean_exprs = []
        b_max_exprs = []
        for c in self.design1.core_objs:
            core_exprs.append(self._calc_field_expr(c.name, "CoreLoss", "Integrate", f"P_{c.name}"))
            b_mean_exprs.append(self._calc_field_expr(c.name, "B_peak", "Mean", f"B_mean_{c.name}"))
            b_max_exprs.append(self._calc_field_expr(c.name, "B_peak", "Maximum", f"B_max_{c.name}"))

        # ---- 권선 그룹 총손실 + explicit 턴 손실 (열해석용) ----
        group_exprs = []
        turn_exprs = []
        plate_exprs = []
        group_exprs.append(self._calc_group_loss(self.design1.Tx_windings_main, "P_Tx_main_group"))
        group_exprs.append(self._calc_group_loss(self.design1.Rx_windings_main, "P_Rx_main_group"))
        if self.design1.Rx_windings_side:
            group_exprs.append(self._calc_group_loss(self.design1.Rx_windings_side, "P_Rx_side_group"))
        # 플레이트류 개별 손실: save_calculation이 이미 P_<name> 표현식을 만들었으므로 재사용
        for p in self.design1.core_plates + self.design1.wcp_plates:
            plate_exprs.append(f"P_{p.name}")
        # Tx는 전 턴 explicit (열모델에서 foil 그대로) -> 턴별 손실
        for w in self.design1.Tx_windings_main:
            turn_exprs.append(self._calc_field_expr(w.name, "EMLoss", "Integrate", f"P_turn_{w.name}"))
        # Rx explicit 턴 (안쪽 n개 / 바깥 n개)
        for grp in [self.design1.Rx_windings_main, self.design1.Rx_windings_side]:
            if not grp:
                continue
            explicit = list(grp[:n_exp]) + list(grp[-n_exp:])
            for w in explicit:
                turn_exprs.append(self._calc_field_expr(w.name, "EMLoss", "Integrate", f"P_turn_{w.name}"))

        all_exprs = core_exprs + b_mean_exprs + b_max_exprs + group_exprs + turn_exprs + plate_exprs
        df_loss = self._export_field_report("calculator_report_loss", all_exprs)
        vals = df_loss.iloc[0, -len(all_exprs):]
        vals.index = all_exprs
        self.loss_map = {k: float(v) for k, v in vals.items()}

        # 실물 기준(_phys) 환산: 대칭 loss 디자인이면 오브젝트별 절단면 수로 보정, 풀모델이면 x1
        b_factor = 0.5 if getattr(self, "loss_is_sym", False) else 1.0
        self.loss_map_phys = {}
        for e in core_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * self._phys_factor(e, is_core_loss=True)
        for e in group_exprs + turn_exprs + plate_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * self._phys_factor(e, is_core_loss=False)
        for e in b_mean_exprs + b_max_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * b_factor

        def _obj_of(expr):
            n = expr
            for pref in ("P_turn_", "P_"):
                if n.startswith(pref):
                    return n[len(pref):].replace("_group", "")
            return n

        # 총계 (대칭 모델의 삭제된 미러 몫 포함 - 실물 전체 기준)
        core_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e)) for e in core_exprs)
        cplate_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                           for e in plate_exprs if "core_plate" in e)
        wcp_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                        for e in plate_exprs if "wcp" in e)
        p_tx = self.loss_map_phys.get("P_Tx_main_group", 0.0)
        p_rxm = self.loss_map_phys.get("P_Rx_main_group", 0.0)
        p_rxs_one = self.loss_map_phys.get("P_Rx_side_group", 0.0)
        winding_total = p_tx + p_rxm + 2 * p_rxs_one  # 측면 링 2개 (좌우 대칭)

        b_mean = sum(self.loss_map_phys[e] for e in b_mean_exprs) / max(len(b_mean_exprs), 1)
        b_max = max((self.loss_map_phys[e] for e in b_max_exprs), default=0)

        # CSV에는 실물 기준 값을 기본으로 기록 (raw 대칭 적분값은 _raw 접미사)
        summary = {
            "P_core_total": [core_total],
            "P_core_plate_total": [cplate_total],
            "P_wcp_total": [wcp_total],
            "P_winding_total": [winding_total],
            "P_Rx_side_total": [2 * p_rxs_one],
            "B_mean_core": [b_mean], "B_max_core": [b_max],
        }
        for e in core_exprs + group_exprs + turn_exprs + plate_exprs:
            summary[e] = [self.loss_map_phys[e]]
            if getattr(self, "loss_is_sym", False):
                summary[f"{e}_raw"] = [self.loss_map[e]]

        # ---- Tx 해석 전류 (전압원 여자 검증용) ----
        I1_mag = float("nan")
        I1_phase = float("nan")
        try:
            oModule = self.design1.GetModule("ReportSetup")
            oModule.CreateReport(
                "winding_current_report", "AC Magnetic", "Data Table", "Setup1 : LastAdaptive", [],
                self._report_variations(),
                [
                    "X Component:=", "Freq",
                    "Y Component:=", ["mag(InputCurrent(Tx_winding))", "ang_deg(InputCurrent(Tx_winding))"]
                ])
            export_path = os.path.join(self.project.path, "winding_current_report.csv")
            oModule.ExportToFile("winding_current_report", export_path, False)
            df_i = pd.read_csv(export_path)
            # AEDT export 헤더의 [단위]를 존중해 A로 정규화 (mA로 나오는 경우 실측 확인됨)
            I1_mag = float(df_i.iloc[0, -2]) * _unit_scale(df_i.columns[-2], kind="current")
            I1_phase = float(df_i.iloc[0, -1])
        except Exception as e:
            logging.warning(f"Failed to extract Tx winding current: {e}")

        summary["I1_mag_peak"] = [I1_mag]
        summary["I1_phase_deg"] = [I1_phase]
        summary["phi_deg"] = [getattr(self, "phi_deg", float("nan"))]
        phase_used = getattr(self, "I2_phase_auto", None)
        summary["I2_phase_used_deg"] = [phase_used if phase_used is not None
                                        else float(self.df_plus["I2_phase_deg"].iloc[0])]

        self.df_loss_summary = pd.DataFrame(summary)
        return self.df_loss_summary

    def get_convergence_info(self, label):
        """수렴 메타데이터 추출: pass 수, 최종 에너지오차/델타에너지 [%], 메시 사면체 수.
        회귀 데이터 필터링용 (수렴 덜 된 샘플 식별)."""
        cols = {f"conv_passes_{label}": float("nan"), f"conv_error_pct_{label}": float("nan"),
                f"conv_delta_pct_{label}": float("nan"), f"mesh_tets_{label}": float("nan")}
        try:
            path = os.path.join(self.project.path, f"convergence_{label}.txt")
            try:
                variation = self.design1.available_variations.nominal_w_values
                if isinstance(variation, (list, tuple)):
                    variation = " ".join(str(v) for v in variation)
            except Exception:
                variation = ""
            self.design1.odesign.ExportConvergence("Setup1", variation, path)
            rows = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = [p.strip() for p in line.replace("|", " ").split()]
                    if parts and parts[0].isdigit():
                        rows.append(parts)
            if rows:
                last = rows[-1]
                cols[f"conv_passes_{label}"] = float(last[0])
                # 형식: pass, tetrahedra, total energy, energy error %, delta energy %
                if len(last) >= 2:
                    cols[f"mesh_tets_{label}"] = float(last[1].replace(",", ""))
                if len(last) >= 4:
                    cols[f"conv_error_pct_{label}"] = float(last[3])
                if len(last) >= 5:
                    cols[f"conv_delta_pct_{label}"] = float(last[4])
        except Exception as e:
            logging.warning(f"convergence info extraction failed ({label}): {e}")
        return pd.DataFrame({k: [v] for k, v in cols.items()})

    def save_results_to_csv(self, results_df, filename="simulation_results_260706.csv"):
        """Saves the DataFrame to a CSV file in a process-safe way.
        기존 파일과 컬럼 구성이 다르면(스키마 변경) 기존 파일을 백업하고 새로 시작한다.
        추가로 스키마 진화에 안전한 per-run parquet 파트를 남긴다 (병렬 안전, 락 불필요)."""
        results_df = results_df.copy()
        results_df["git_hash"] = GIT_HASH
        results_df["project_name"] = getattr(self, "PROJECT_NAME", "")
        results_df["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lock_path = filename + ".lock"
        with FileLock(lock_path):
            file_exists = os.path.isfile(filename)
            if file_exists:
                with open(filename, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                if header != list(results_df.columns):
                    backup = filename.replace(".csv", f"_old_{datetime.now().strftime('%y%m%d_%H%M%S')}.csv")
                    shutil.move(filename, backup)
                    logging.warning(f"CSV schema changed; old results moved to {backup}")
                    file_exists = False
            results_df.to_csv(filename, mode='a', header=not file_exists, index=False)

        # parquet 파트: 파일명이 유니크해 병렬 인스턴스 간 충돌 없음.
        # 여러 파트는 pd.concat(map(pd.read_parquet, glob(...)))으로 스키마가 달라도 병합 가능
        try:
            parts_dir = "results_parts_260706"
            os.makedirs(parts_dir, exist_ok=True)
            part = os.path.join(parts_dir,
                                f"part_{datetime.now().strftime('%y%m%d_%H%M%S')}_{os.getpid()}_{self.PROJECT_NAME}.parquet")
            results_df.to_parquet(part, index=False)
        except Exception as e:
            logging.warning(f"parquet part skipped (pyarrow 미설치?): {e}")

        logging.info(f"Results saved to {filename}")

    def save_project(self):
        try:
            self.design1.save_project()
        except Exception:
            try:
                self.project.oproject.Save()
            except Exception as e:
                logging.warning(f"Failed to save project: {e}")

    def close_project(self):
        # keep_project=1 이면 솔루션 데이터를 보존한 채 닫는다
        # (cleanup_solution은 저장 프로젝트의 Results를 지워버림 - 삭제 예정일 때만 수행)
        try:
            keep = int(self.df_plus["keep_project"].iloc[0]) != 0
        except Exception:
            keep = False
        if not keep:
            try:
                self.design1.cleanup_solution()
            except Exception:
                pass
        else:
            try:
                self.save_project()
            except Exception:
                pass
        self.design1.close_project()
        self.desktop.release_desktop(close_projects=True, close_on_exit=True)

    def delete_project_folder(self, max_attempts=6, wait_s=10):
        """
        완료된 시뮬레이션 파일 삭제 (슈퍼컴퓨터 저장공간 확보용 - 반드시 지워져야 함).
        AEDT가 파일 핸들을 늦게 놓는 경우가 있어 재시도하며, .lock 등 부산물도 제거한다.
        """
        project_folder = os.path.join(os.getcwd(), "simulation", self.PROJECT_NAME)

        for attempt in range(1, max_attempts + 1):
            time.sleep(wait_s)
            try:
                if os.path.isdir(project_folder):
                    shutil.rmtree(project_folder)
                # 폴더 밖에 생기는 부산물 (.lock, .auto 등)
                sim_dir = os.path.join(os.getcwd(), "simulation")
                for name in os.listdir(sim_dir):
                    if name.startswith(self.PROJECT_NAME + ".") and (
                            name.endswith(".lock") or name.endswith(".auto")
                            or name.endswith(".lock.txt") or name.endswith(".asol.lock")):
                        try:
                            os.remove(os.path.join(sim_dir, name))
                        except OSError:
                            pass
                if not os.path.isdir(project_folder):
                    logging.info(f"Successfully deleted project folder: {project_folder}")
                    return True
            except Exception as e:
                logging.warning(f"Delete attempt {attempt}/{max_attempts} failed for {project_folder}: {e}")

        logging.error(f"FAILED to delete project folder after {max_attempts} attempts: {project_folder}")
        return False


def log_failed_sample(input_df, reason, filename="failed_samples_260706.csv"):
    """실패/기각 샘플 기록 (설계공간 경계 분석용). 파라미터 + 사유."""
    try:
        row = input_df.copy()
        row["fail_reason"] = [str(reason)[:500]]
        row["fail_time"] = [datetime.now().strftime("%y%m%d_%H%M%S")]
        lock_path = filename + ".lock"
        with FileLock(lock_path):
            file_exists = os.path.isfile(filename)
            row.to_csv(filename, mode="a", header=not file_exists, index=False)
    except Exception as e:
        logging.warning(f"failed-sample logging failed: {e}")


def run_one_loop(param=None, model_only=False, hold=False, golden=False, overrides=None):
    """
    param 이 None  -> 랜덤 파라미터 1회 (검증 실패 시 재추첨), 완료 후 프로젝트 삭제
    param 이 dict 등 -> 해당 값으로 1회 (fixed 모드), 프로젝트 폴더 보존
    model_only=True -> 모델링/셋업까지만 하고 해석은 생략 (지오메트리 확인용)
    """
    fixed_mode = param is not None
    sim = None
    desktop = None
    held = [False]  # hold 성공 시 finally에서 desktop을 닫지 않기 위한 플래그
    try:
        # pyDesktop을 context manager로 쓰면 release_desktop 이후 __exit__에서
        # close_on_exit 속성 오류가 발생하므로 직접 생성하고 finally에서 해제한다.
        desktop = pyDesktop(version=None, non_graphical=GUI, close_on_exit=True, new_desktop=True)

        sim = Simulation(desktop=desktop)

        sim.create_simulation_name()
        sim.create_project()

        if fixed_mode:
            sim.input_df = create_input_parameter(param)
            # 위반 시 이유를 담아 ValueError raise
            _, sim.df_plus = validation_check(sim.input_df, strict=True)
        else:
            while True:
                sim.input_df = create_input_parameter(None)
                # CLI 오버라이드 (랜덤 모드에서도 --thermal/--loss 등 플래그 적용)
                if overrides:
                    for k, v in overrides.items():
                        sim.input_df[k] = v
                result, sim.df_plus, errors = validation_check(sim.input_df, return_errors=True)
                if result:
                    break
                # 기각 샘플 기록 (설계공간 경계 데이터)
                log_failed_sample(sim.input_df, "validation: " + " / ".join(errors))

        sim.full_model = int(sim.df_plus["full_model"].iloc[0]) != 0
        matrix_on = int(sim.df_plus["matrix_on"].iloc[0]) != 0
        loss_on = int(sim.df_plus["loss_on"].iloc[0]) != 0
        thermal_on = int(sim.df_plus["thermal_on"].iloc[0]) != 0

        def _build_em_design(design_name, mode):
            """EM 디자인 1개 생성: 지오메트리 + 여자 + 메시 + 경계 + 셋업"""
            sim.create_design(name=design_name)
            set_design_variables(sim.design1, sim.input_df)
            sim.create_core()
            sim.create_coil()
            sim.split_geometry()
            sim.create_coil_section()
            sim.assign_winding(mode=mode)
            sim.assign_coil()
            if mode == "matrix":
                sim.assign_matrix()
            else:
                sim.assign_core_loss()
            # matrix 디자인은 인덕턴스(에너지 적분)가 목적이라 skin 메시를 뺄 수 있는 옵션
            # (matrix_skin_mesh=0). 단 Llt가 스펙 라벨(+-2% 밴드)이므로 A/B 검증 통과 후에만 캠페인 적용.
            if mode == "matrix" and int(sim.df_plus["matrix_skin_mesh"].iloc[0]) == 0:
                logging.info("matrix design: skin-depth mesh ops skipped (matrix_skin_mesh=0)")
            else:
                sim.assign_skin_depth()
            sim.assign_plate_settings()
            sim.assign_boundary()
            sim.create_setup()

        def _analyze_current_design(label, max_attempts=2):
            # 간헐적으로 solve가 결과 없이 '성공'(고정 ~3분 후)으로 끝나는 케이스가 있어
            # is_solved 확인 후 1회 재시도한다. 재시도도 실패하면 AEDT 메시지를 로그에 남기고 실패 처리.
            elapsed = 0.0
            for attempt in range(1, max_attempts + 1):
                t0 = time.time()
                sim.design1.setup.analyze(cores=sim.NUM_CORE)
                elapsed = time.time() - t0
                try:
                    solved = sim.design1.setup.is_solved
                except Exception:
                    solved = True  # 확인 불가 시 기존 동작 유지
                if solved:
                    break
                logging.warning(f"[{label}] analyze attempt {attempt} finished without solution data.")
                try:
                    msgs = sim.design1.odesktop.GetMessages(sim.PROJECT_NAME, sim.design1.design_name, 0)
                    for m in list(msgs)[-10:]:
                        logging.warning(f"[AEDT] {m}")
                except Exception:
                    pass
            else:
                raise RuntimeError(f"[{label}] Setup1 finished without solution data after {max_attempts} attempts.")
            # 리포트 생성 전에 저장해 솔루션 상태를 프로젝트 파일에 반영
            sim.save_project()
            return elapsed

        result_parts = [sim.df_plus]
        total_time = 0.0

        # ---- design1: L/k 매트릭스 (전류원, 기존 방식) ----
        if matrix_on:
            _build_em_design("maxwell_matrix", "matrix")
            sim.design_matrix = sim.design1
            if not model_only:
                t_matrix = _analyze_current_design("matrix")
                total_time += t_matrix
                sim.get_magnetic_parameter()
                result_parts.append(sim.df1)
                result_parts.append(sim.get_convergence_info("matrix"))
                result_parts.append(pd.DataFrame({"time_matrix": [t_matrix]}))

        # ---- design2: 손실 원샷 ----
        # loss_sym_on=1 (캠페인 기본): 대칭 1/8 + 전류 여자 (Tx = 부하+자화 페이저 합)
        #   -> 추출 시 오브젝트별 상수 보정으로 실물(_phys) 기록. 시간 ~4x 단축.
        # loss_sym_on=0 (최종 검증): 풀모델 + Tx 전압원 (검증된 물리 기준 경로)
        if loss_on:
            loss_sym = int(sim.df_plus["loss_sym_on"].iloc[0]) != 0 and not sim.full_model

            # P_target > 0 이면 design1의 누설(Lk = Llt_true)로 DAB 운전 위상을 역산해
            # I2 위상(-phi/2)을 자동 주입: phi = asin(P w Lk / (V1 V2'))
            P_t = float(sim.df_plus["P_target"].iloc[0])
            if P_t > 0 and not model_only:
                if not matrix_on:
                    raise RuntimeError("P_target>0 requires matrix_on=1 (Lk needed for phase calculation).")
                freq = float(sim.df_plus["freq"].iloc[0])
                V1 = float(sim.df_plus["V1_rms"].iloc[0])
                V2p = float(sim.df_plus["V2_rms"].iloc[0]) * int(sim.df_plus["N1"].iloc[0]) / int(sim.df_plus["N2"].iloc[0])
                Llt_true = float(sim.df1["Llt"].iloc[0]) * 1e-6 * (1.0 if sim.full_model else 2.0)
                omega = 2 * math.pi * freq
                arg = P_t * omega * Llt_true / (V1 * V2p) if V1 * V2p > 0 else 2.0
                if arg >= 1.0:
                    logging.warning(f"P_target unreachable with Lk={Llt_true*1e6:.1f}uH (sin(phi)={arg:.2f}>1) - phi=90deg capped")
                    phi_deg = 90.0
                else:
                    phi_deg = math.degrees(math.asin(arg))
                sim.I2_phase_auto = -phi_deg / 2.0
                sim.phi_deg = phi_deg
                logging.info(f"auto phase: Lk={Llt_true*1e6:.2f}uH, phi={phi_deg:.2f}deg -> I2 phase {sim.I2_phase_auto:.2f}deg")

            if loss_sym and not model_only:
                if not matrix_on:
                    raise RuntimeError("loss_sym_on=1 requires matrix_on=1 (Lm needed for magnetizing current).")
                # Tx 합성 전류: I1 = I_load∠phase2 + Im∠-90 (복소 합)
                freq = float(sim.df_plus["freq"].iloc[0])
                V1 = float(sim.df_plus["V1_rms"].iloc[0])
                I2 = float(sim.df_plus["I2_rated"].iloc[0])
                N1 = int(sim.df_plus["N1"].iloc[0])
                N2 = int(sim.df_plus["N2"].iloc[0])
                phase2 = getattr(sim, "I2_phase_auto", None)
                if phase2 is None:
                    phase2 = float(sim.df_plus["I2_phase_deg"].iloc[0])
                Lm_true = float(sim.df1["Lmt"].iloc[0]) * 1e-6 * 2.0  # 대칭 매트릭스 L은 실물의 1/2
                omega = 2 * math.pi * freq
                Im_peak = math.sqrt(2) * V1 / (omega * Lm_true) if Lm_true > 0 else 0.0
                I_load_peak = math.sqrt(2) * I2 * N2 / N1
                z = I_load_peak * complex(math.cos(math.radians(phase2)), math.sin(math.radians(phase2))) \
                    + Im_peak * complex(0, -1)
                sim.loss_I1_peak = abs(z)
                sim.loss_I1_phase_deg = math.degrees(math.atan2(z.imag, z.real))
                logging.info(f"loss_sym excitation: I_load={I_load_peak:.2f}A + Im={Im_peak:.2f}A "
                             f"-> I1={sim.loss_I1_peak:.2f}A ang {sim.loss_I1_phase_deg:.2f}deg")
            elif loss_sym and model_only:
                sim.loss_I1_peak = math.sqrt(2) * float(sim.df_plus["I1_rated"].iloc[0])
                sim.loss_I1_phase_deg = 0.0

            prev_full = sim.full_model
            if loss_sym:
                sim.loss_em_full = False
                sim.loss_is_sym = True
                _build_em_design("maxwell_loss", "loss_sym")
            else:
                sim.full_model = True
                sim.loss_em_full = True
                sim.loss_is_sym = False
                _build_em_design("maxwell_loss", "loss")
            sim.design_loss = sim.design1
            if not model_only:
                t_loss = _analyze_current_design("loss")
                total_time += t_loss
                sim.save_calculation()      # 권선/플레이트 EMLoss (기존 리포트)
                sim.save_loss_reports()     # 코어손실 + B + I1 + 그룹/턴 손실
                result_parts += [sim.df_calculator1, sim.df_calculator2, sim.df_calculator3,
                                 sim.df_loss_summary, sim.get_convergence_info("loss"),
                                 pd.DataFrame({"time_loss": [t_loss]})]
            sim.full_model = prev_full

        # ---- design3: Icepak 열해석 (풀 지오메트리, EM 손실 주입) ----
        if thermal_on and loss_on and not model_only:
            from module.thermal_260706 import run_thermal_analysis
            t0 = time.time()
            df_thermal = run_thermal_analysis(sim)
            t_thermal = time.time() - t0
            total_time += t_thermal
            result_parts += [df_thermal, pd.DataFrame({"time_thermal": [t_thermal]})]

        if model_only:
            print(sim.df_plus)
            sim.save_project()
            logging.info(f"Model-only mode: project '{sim.PROJECT_NAME}' saved, skipping analysis.")
            try:
                sim.close_project()
            except Exception as e:
                logging.exception(f"Error closing project: {e}")
            return

        simulation_time = pd.DataFrame({"time": [total_time]})
        result = pd.concat(result_parts + [simulation_time], axis=1)

        try:
            sim.save_results_to_csv(result)
            if golden:
                # golden case: 동일 기준 케이스를 주기적으로 재해석해 결과 표류(드리프트) 감지
                sim.save_results_to_csv(result, filename="golden_history_260706.csv")
        except Exception as e:
            logging.exception(f"Error saving results to CSV: {e}")

        if fixed_mode or hold:
            print(result)
            sim.save_project()
        # 스케줄러 stdout 회수용: 결과 1행을 JSON 한 줄로 즉시 스트리밍
        # (랜덤 모드도 포함 - 태스크 완주를 기다리지 않고 샘플 단위로 데이터 회수 가능)
        try:
            print("RESULT_JSON " + result.iloc[0].to_json(), flush=True)
        except Exception as e:
            logging.warning(f"RESULT_JSON print failed: {e}")

        if hold:
            # 결과 확인용: AEDT와 프로젝트를 연 채로 종료 (사용자가 직접 닫을 때까지 유지)
            held[0] = True
            logging.info(f"HOLD mode: project '{sim.PROJECT_NAME}' left open in AEDT for inspection.")
            print(f"\n=== HOLD: AEDT에 '{sim.PROJECT_NAME}' 프로젝트가 열린 채 유지됩니다. 확인 후 직접 닫으세요. ===")
            return True

        try:
            sim.close_project()
        except Exception as e:
            logging.exception(f"Error closing project: {e}")

        # 완료된 시뮬레이션 파일 삭제 (fixed 모드는 keep_project=1 기본값으로 보존,
        # 랜덤/클러스터 스윕은 저장공간 확보를 위해 확실히 삭제)
        keep_project = int(sim.df_plus["keep_project"].iloc[0]) != 0
        if not keep_project:
            try:
                sim.delete_project_folder()
            except Exception as e:
                logging.exception(f"Error deleting project folder: {e}")

        return True
    except Exception as e:
        logging.exception(f"run_one_loop failed: {e}")
        if sim is not None and getattr(sim, "input_df", None) is not None:
            log_failed_sample(sim.input_df, f"runtime: {e}")
        if fixed_mode:
            # fixed 모드에서는 실패를 조용히 넘기지 않는다
            raise
        if sim is not None:
            try:
                sim.close_project()
                time.sleep(1)
            except Exception:
                pass
            try:
                sim.delete_project_folder()
            except Exception:
                pass
        return False
    finally:
        if desktop is not None:
            try:
                if held[0]:
                    # HOLD: 프로젝트/AEDT는 열어둔 채 python의 gRPC 세션만 해제
                    # (이걸 안 하면 python 프로세스가 AEDT를 붙잡은 채 종료되지 않음)
                    desktop.release_desktop(close_projects=False, close_on_exit=False)
                else:
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                time.sleep(1)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="MFT simulation (design 260706)")
    parser.add_argument("--fixed", action="store_true",
                        help="도면 기본값으로 1회 실행 (랜덤 루프 대신)")
    parser.add_argument("--params", type=str, default=None,
                        help="기본값 위에 덮어쓸 파라미터 JSON 파일 경로 (지정 시 fixed 모드)")
    parser.add_argument("--round", dest="round_corner", action="store_true", default=None,
                        help="권선 모서리 라운드 처리 on")
    parser.add_argument("--no-round", dest="round_corner", action="store_false",
                        help="권선 모서리 라운드 처리 off")
    parser.add_argument("--model-only", action="store_true",
                        help="모델링/셋업까지만 수행하고 해석은 생략 (지오메트리 확인용)")
    parser.add_argument("--full", action="store_true",
                        help="대칭(1/8 분할) 미적용 풀모델로 모델링/해석")
    parser.add_argument("--headless", action="store_true",
                        help="AEDT 창 없이 실행 (해석 중 GUI 조작으로 인한 블로킹 방지)")
    parser.add_argument("--count", type=int, default=None,
                        help="랜덤 모드에서 N회 성공 후 종료 (slurm_scheduler fea_bursty/packed 태스크용; 미지정 시 무한루프)")
    parser.add_argument("--no-matrix", dest="matrix_on", action="store_false", default=None,
                        help="design1(L/k 매트릭스) 생략")
    parser.add_argument("--no-loss", dest="loss_on", action="store_false", default=None,
                        help="design2(손실 원샷) 생략")
    parser.add_argument("--thermal", dest="thermal_on", action="store_true", default=None,
                        help="design3(Icepak 열해석)까지 수행")
    parser.add_argument("--hold", action="store_true",
                        help="해석 완료 후 AEDT/프로젝트를 닫지 않고 유지 (결과 직접 확인용, 1회 실행)")
    parser.add_argument("--golden", action="store_true",
                        help="golden case(고정 기준 케이스) 1회 해석 후 golden_history CSV에 기록 (드리프트 감지용)")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="파라미터 오버라이드 (반복 가능, fixed/random 공용). 예: --set P_target=1e6 --set percent_error=1.0")
    return parser.parse_args()


_UNIT_SCALES = {
    # 기준 단위로의 배율: current -> A, inductance -> uH, power -> W, flux -> T, voltage -> V
    "current": {"fA": 1e-15, "pA": 1e-12, "nA": 1e-9, "uA": 1e-6, "mA": 1e-3, "A": 1.0, "kA": 1e3},
    "inductance": {"pH": 1e-6, "nH": 1e-3, "uH": 1.0, "mH": 1e3, "H": 1e6},
    "power": {"nW": 1e-9, "uW": 1e-6, "mW": 1e-3, "W": 1.0, "kW": 1e3, "MW": 1e6},
    "flux": {"uT": 1e-6, "mT": 1e-3, "T": 1.0, "tesla": 1.0},
    "voltage": {"uV": 1e-6, "mV": 1e-3, "V": 1.0, "kV": 1e3},
}


def _unit_scale(column_name, kind):
    """AEDT export 컬럼 헤더의 '[unit]' 접미사를 파싱해 기준 단위 배율 반환.
    단위 표기가 없으면 1.0 (이미 기준 단위로 가정). 미지 단위는 경고 후 1.0."""
    m = re.search(r"\[([^\]]*)\]", str(column_name))
    if not m:
        return 1.0
    unit = m.group(1).strip()
    if not unit:
        return 1.0
    table = _UNIT_SCALES.get(kind, {})
    if unit in table:
        return table[unit]
    logging.warning(f"unknown unit '{unit}' in column '{column_name}' (kind={kind}) - no scaling applied")
    return 1.0


def _parse_set_overrides(pairs):
    """--set KEY=VALUE 목록을 타입 변환된 dict로"""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--set 형식 오류 (KEY=VALUE 필요): {p}")
        k, v = p.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    global GUI

    args = parse_args()

    if args.headless:
        GUI = True  # non_graphical=True

    if args.golden:
        # 고정 기준 케이스: 배치마다 같이 돌려 결과 표류를 시계열로 감시
        golden_path = os.path.join(BASE_DIR, "verification_params", "golden_case.json")
        with open(golden_path, "r", encoding="utf-8") as f:
            param = json.load(f)
        run_one_loop(param=param, golden=True)
        return

    fixed_mode = args.fixed or (args.params is not None)

    # model-only는 지오메트리 확인용이므로 항상 1회 실행
    if args.model_only and not fixed_mode:
        run_one_loop(param=None, model_only=True)
        return

    if fixed_mode:
        param = {}
        if args.params is not None:
            with open(args.params, "r", encoding="utf-8") as f:
                param.update(json.load(f))
        if args.round_corner is not None:
            param["round_corner"] = 1 if args.round_corner else 0
        if args.full:
            param["full_model"] = 1
        if args.matrix_on is not None:
            param["matrix_on"] = 1 if args.matrix_on else 0
        if args.loss_on is not None:
            param["loss_on"] = 1 if args.loss_on else 0
        if args.thermal_on is not None:
            param["thermal_on"] = 1 if args.thermal_on else 0
        param.update(_parse_set_overrides(args.set_overrides))

        run_one_loop(param=param, model_only=args.model_only, hold=args.hold)
        return

    # 랜덤 스윕: --count N 이면 N회 성공 후 종료 (slurm_scheduler 태스크의 완료 감지용),
    # 미지정 시 기존처럼 무한루프.
    # CLI 플래그는 랜덤 모드에도 적용 (--thermal 은 손실 해석이 선행돼야 하므로 loss도 자동 활성화)
    overrides = {}
    if args.matrix_on is not None:
        overrides["matrix_on"] = 1 if args.matrix_on else 0
    if args.loss_on is not None:
        overrides["loss_on"] = 1 if args.loss_on else 0
    if args.thermal_on:
        overrides["thermal_on"] = 1
        overrides.setdefault("loss_on", 1)
        overrides.setdefault("matrix_on", 1)
    if args.round_corner is not None:
        overrides["round_corner"] = 1 if args.round_corner else 0
    if args.full:
        overrides["full_model"] = 1
    if args.hold:
        overrides["keep_project"] = 1
    overrides.update(_parse_set_overrides(args.set_overrides))

    successes = 0
    attempts = 0
    max_attempts = args.count * 3 if args.count else None

    while True:

        try:
            ok = run_one_loop(param=None, model_only=args.model_only, hold=args.hold,
                              overrides=overrides or None)
            if ok:
                successes += 1
                if args.hold:
                    # 확인용 1회 실행 - AEDT를 연 채로 종료
                    return
        except Exception as e:
            logging.exception(f"Error running simulation: {e}")

        finally:
            time.sleep(10)

        attempts += 1
        if args.count is not None:
            if successes >= args.count:
                logging.info(f"Completed {successes}/{args.count} simulations.")
                # os._exit: pyaedt atexit 핸들러의 간헐적 teardown 크래시가
                # 성공한 런을 실패(exit 1)로 둔갑시키는 것 방지 (파일은 이미 flush됨)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            if attempts >= max_attempts:
                logging.error(f"Reached max attempts ({attempts}) with only {successes}/{args.count} successes.")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0 if successes > 0 else 1)


if __name__ == "__main__":
    main()
