import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from module.core_material_contract import (
    AREA_BASIS_EXPLICIT_NET,
    AREA_BASIS_GROSS_HOMOGENEOUS,
    CORE_LAMINATION_FACTOR_AB_CANDIDATES,
    CORE_LAMINATION_FACTOR_DATASHEET_MIN,
    CORE_LAMINATION_FACTOR_USER_CONSERVATIVE_CANDIDATE,
    CORE_MATERIAL_CONTRACT_VERSION,
    LEG_STACKING_DIRECTION,
    PHYSICS_DATA_REVISION,
    YOKE_STACKING_DIRECTION,
    build_core_material_contract_fields,
    effective_area_m2,
    effective_steinmetz_cm,
    expected_core_loss_from_bavg_moment_w,
    expected_specific_core_loss_w_kg,
    geometry_volume_and_masses,
    lamination_factor_policy_source,
    material_flux_density_t,
    native_lamination_material_specs,
    sinusoidal_b_peak_material_t,
    square_wave_b_material_t,
    validate_native_lamination_readback,
)
from module.input_parameter_260706 import (
    CORE_MATERIAL_INPUT_KEYS,
    KEYS,
    create_input_parameter,
    get_drawing_default_params,
    validation_check,
)
from module.modeling_260706 import create_core as create_core_geometry
from module.thermal_260706 import LossAllocator, _assign_losses
from run_simulation_260706 import (
    Simulation,
    _b_power_volume_integral_si,
    _core_group_index,
    _native_b_power_restore_factor,
    _native_core_report_plan,
    _sheet_area_model_units,
)


K = 0.85
Y = 1.74
MARGIN = 1.15
CM = 1.377


class SheetAreaReadbackTests(unittest.TestCase):
    def test_reads_object3d_sheet_area_from_its_single_face(self):
        sheet = SimpleNamespace(
            name="core_flux_section_1",
            faces=[SimpleNamespace(area=1540.0)],
        )

        self.assertEqual(_sheet_area_model_units(sheet), 1540.0)

    def test_rejects_missing_multiple_or_invalid_faces(self):
        cases = (
            (SimpleNamespace(name="none", faces=[]), "exactly one face"),
            (
                SimpleNamespace(
                    name="multiple",
                    faces=[SimpleNamespace(area=1.0), SimpleNamespace(area=2.0)],
                ),
                "exactly one face",
            ),
            (
                SimpleNamespace(
                    name="invalid", faces=[SimpleNamespace(area=float("nan"))]
                ),
                "invalid sheet area",
            ),
        )
        for sheet, message in cases:
            with self.subTest(sheet=sheet.name), self.assertRaisesRegex(
                    RuntimeError, message):
                _sheet_area_model_units(sheet)


def native_props(direction, kf=K):
    choice = lambda value: {
        "property_type": "ChoiceProperty", "Choice": value
    }
    return {
        "CoordinateSystemType": "Cartesian",
        "stacking_type": choice("Lamination"),
        "stacking_factor": str(kf),
        "stacking_direction": choice(direction),
        "core_loss_type": choice("Power Ferrite"),
        "core_loss_cm": f"{CM}A_per_meter",
        "core_loss_x": "1.51tesla",
        "core_loss_y": str(Y),
        "core_loss_kdc": "0",
        "core_loss_equiv_cut_depth": "0meter",
        "permeability": "3000",
        "conductivity": "0",
    }


class CoreMaterialArithmeticTests(unittest.TestCase):
    def test_native_contract_keeps_base_cm_and_separate_margin(self):
        fields = build_core_material_contract_fields(
            cm_base=CM, core_x=1.51, core_y=Y,
            lamination_factor=K, loss_margin=MARGIN,
        )
        diagnostic = CM * MARGIN * K ** (1.0 - Y)
        self.assertEqual(fields["core_cm_assigned"], CM)
        self.assertAlmostEqual(
            fields["core_cm_equivalent_gross_unassigned"], diagnostic
        )
        self.assertEqual(fields["core_loss_correction_factor"], MARGIN)
        self.assertEqual(fields["core_stacking_direction_leg"], "V(1)")
        self.assertEqual(fields["core_stacking_direction_yoke"], "V(3)")
        self.assertIn("blocked_pending_solved", fields["core_native_model_approval_status"])
        self.assertEqual(
            fields["core_material_contract_version"],
            CORE_MATERIAL_CONTRACT_VERSION,
        )
        self.assertEqual(fields["physics_data_revision"], PHYSICS_DATA_REVISION)

    def test_native_material_specs_are_directional_not_one_global_axis(self):
        specs = native_lamination_material_specs(K)
        self.assertEqual(specs["leg"], {
            "stacking_type": "Lamination",
            "stacking_factor": K,
            "stacking_direction": LEG_STACKING_DIRECTION,
        })
        self.assertEqual(
            specs["yoke"]["stacking_direction"], YOKE_STACKING_DIRECTION
        )

    def test_three_arm_factor_policy_keeps_datasheet_and_user_margin_distinct(self):
        self.assertEqual(CORE_LAMINATION_FACTOR_AB_CANDIDATES, (1.0, 0.85, 0.70))
        self.assertEqual(CORE_LAMINATION_FACTOR_DATASHEET_MIN, 0.85)
        self.assertEqual(CORE_LAMINATION_FACTOR_USER_CONSERVATIVE_CANDIDATE, 0.70)
        self.assertIn("guaranteed_minimum", lamination_factor_policy_source(0.85))
        self.assertIn("not_a_datasheet", lamination_factor_policy_source(0.70))
        fields = build_core_material_contract_fields(
            cm_base=CM, core_x=1.51, core_y=Y,
            lamination_factor=0.70, loss_margin=MARGIN,
        )
        self.assertEqual(fields["core_lamination_factor"], 0.70)
        self.assertIn("not_a_datasheet", fields["core_lamination_factor_source"])
        self.assertIn("kf0p70", fields["core_native_model_approval_status"])

    def test_strict_native_readback_accepts_exact_and_rejects_drift(self):
        attested = validate_native_lamination_readback(
            native_props(LEG_STACKING_DIRECTION),
            lamination_factor=K,
            stacking_direction=LEG_STACKING_DIRECTION,
            cm_base=CM,
            core_x=1.51,
            core_y=Y,
        )
        self.assertEqual(attested["stacking_factor"], K)
        self.assertEqual(attested["core_loss_equiv_cut_depth"], 0.0)
        mutations = (
            ("stacking_factor", "0.82"),
            ("conductivity", "1300000"),
            ("core_loss_cm", "1.785A_per_meter"),
            ("core_loss_equiv_cut_depth", "0.0001meter"),
        )
        for key, value in mutations:
            props = native_props(LEG_STACKING_DIRECTION)
            props[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_native_lamination_readback(
                    props,
                    lamination_factor=K,
                    stacking_direction=LEG_STACKING_DIRECTION,
                    cm_base=CM,
                    core_x=1.51,
                    core_y=Y,
                )

    def test_flux_and_loss_reference_formulae(self):
        self.assertAlmostEqual(material_flux_density_t(0.85, K), 1.0)
        self.assertAlmostEqual(
            square_wave_b_material_t(1000, 1000, 10, 0.01), 2.5
        )
        self.assertAlmostEqual(
            sinusoidal_b_peak_material_t(1000, 1000, 10, 0.01),
            math.sqrt(2) / (0.2 * math.pi),
        )
        volume_gross = 0.01
        b_material = 1.0
        b_average = K * b_material
        moment = b_average ** Y * volume_gross
        expected = expected_core_loss_from_bavg_moment_w(
            moment,
            cm_base=CM,
            frequency_hz=1000,
            core_x=1.51,
            core_y=Y,
            lamination_factor=K,
            loss_margin=MARGIN,
        )
        direct = (
            CM * 1000 ** 1.51 * b_material ** Y
            * volume_gross * K * MARGIN
        )
        self.assertAlmostEqual(expected, direct, places=12)
        self.assertGreater(expected_specific_core_loss_w_kg(1000, 1.0), 0)

    def test_b_power_integral_units_are_fail_closed(self):
        self.assertAlmostEqual(
            _b_power_volume_integral_si(2.0, "T^1.74*mm^3", 1.74), 2e-9
        )
        self.assertEqual(
            _b_power_volume_integral_si(2.0, "tesla^1.74*m^3", 1.74), 2.0
        )
        with self.assertRaises(RuntimeError):
            _b_power_volume_integral_si(2.0, "T", 1.74)

    def test_explicit_net_basis_does_not_apply_stacking_twice(self):
        self.assertEqual(
            effective_area_m2(0.0132, K, area_basis=AREA_BASIS_EXPLICIT_NET),
            0.0132,
        )
        self.assertEqual(
            material_flux_density_t(
                1.1, K, area_basis=AREA_BASIS_EXPLICIT_NET
            ),
            1.1,
        )
        self.assertAlmostEqual(
            effective_steinmetz_cm(
                CM, Y, K, MARGIN, area_basis=AREA_BASIS_EXPLICIT_NET
            ),
            CM * MARGIN,
        )
        geometry, effective, gross_mass, effective_mass = (
            geometry_volume_and_masses(
                0.01, K, area_basis=AREA_BASIS_EXPLICIT_NET
            )
        )
        self.assertEqual(geometry, effective)
        self.assertEqual(gross_mass, effective_mass)

    def test_invalid_contract_values_fail_closed(self):
        for key, value, message in (
            ("lamination_factor", 0.0, "0 < kf <= 1"),
            ("lamination_factor", 1.01, "0 < kf <= 1"),
            ("lamination_factor", float("nan"), "must be finite"),
            ("loss_margin", 0.99, ">= 1"),
            ("loss_margin", float("inf"), "must be finite"),
        ):
            kwargs = {
                "cm_base": CM, "core_x": 1.51, "core_y": Y,
                "lamination_factor": K, "loss_margin": MARGIN,
                "area_basis": AREA_BASIS_GROSS_HOMOGENEOUS,
            }
            kwargs[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    build_core_material_contract_fields(**kwargs)


class CoreMaterialInputContractTests(unittest.TestCase):
    def test_campaign_identity_keys_remain_free_of_fixed_material_constants(self):
        self.assertNotIn("core_lamination_factor", KEYS)
        self.assertNotIn("core_loss_margin", KEYS)
        self.assertEqual(
            CORE_MATERIAL_INPUT_KEYS,
            ("core_lamination_factor", "core_loss_margin"),
        )

    def test_validation_emits_gross_and_effective_geometry_provenance(self):
        ok, frame = validation_check(
            create_input_parameter(get_drawing_default_params()), strict=True
        )
        self.assertTrue(ok)
        self.assertEqual(float(frame["core_lamination_factor"].iloc[0]), K)
        self.assertEqual(float(frame["core_loss_margin"].iloc[0]), MARGIN)
        self.assertAlmostEqual(
            float(frame["Ae_effective_m2"].iloc[0]),
            K * float(frame["Ae_gross_m2"].iloc[0]),
        )
        self.assertAlmostEqual(
            float(frame["core_mass_effective_kg"].iloc[0]),
            K * float(frame["core_mass_gross_kg"].iloc[0]),
        )
        self.assertEqual(
            frame["core_geometry_material_basis"].iloc[0],
            AREA_BASIS_GROSS_HOMOGENEOUS,
        )


class _GeometryObj:
    def __init__(self, name, material, origin=None, sizes=None):
        self.name = name
        self.material_name = material
        self.origin = origin
        self.sizes = sizes
        self.color = None


class _GeometryModeler:
    def __init__(self):
        self.boxes = []
        self.subtract_calls = []

    def create_box(self, origin, sizes, name, material):
        obj = _GeometryObj(name, material, origin, sizes)
        self.boxes.append(obj)
        return obj

    def subtract(self, *args, **kwargs):
        self.subtract_calls.append((args, kwargs))


class CoreGeometrySegmentationTests(unittest.TestCase):
    def test_five_piece_groups_use_leg_v1_and_yoke_v3_materials(self):
        modeler = _GeometryModeler()
        design = SimpleNamespace(modeler=modeler)
        cores, plates, pads = create_core_geometry(
            design,
            n_group=2,
            plate_on=False,
            pad_on=False,
            segmented_lamination=True,
            core_material_leg="leg_v1",
            core_material_yoke="yoke_v3",
        )
        self.assertEqual(len(cores), 10)
        self.assertEqual(plates, [])
        self.assertEqual(pads, [])
        self.assertEqual(modeler.subtract_calls, [])
        for group in (1, 2):
            names = {obj.name for obj in cores if _core_group_index(obj.name) == group}
            self.assertEqual(names, {
                f"core_{group}_leg_left", f"core_{group}_leg_center",
                f"core_{group}_leg_right", f"core_{group}_yoke_bottom",
                f"core_{group}_yoke_top",
            })
        self.assertTrue(all(
            obj.material_name == "leg_v1" for obj in cores if "_leg_" in obj.name
        ))
        self.assertTrue(all(
            obj.material_name == "yoke_v3" for obj in cores if "_yoke_" in obj.name
        ))

    def test_segmented_area_identity_matches_legacy_window_subtraction(self):
        l1, l2, h1 = 22.0, 180.0, 300.0
        legacy = (4*l1 + 2*l2) * (h1 + 2*l1) - 2*l2*h1
        segmented = 4*l1*h1 + 2*(4*l1 + 2*l2)*l1
        self.assertEqual(legacy, segmented)


class NativeCoreReportPlanTests(unittest.TestCase):
    @staticmethod
    def _group(group_index):
        return [
            SimpleNamespace(name=f"core_{group_index}_{region}")
            for region in (
                "leg_left", "leg_center", "leg_right",
                "yoke_bottom", "yoke_top",
            )
        ]

    def test_same_cut_groups_become_one_complete_b_power_batch(self):
        groups = {2: self._group(2), 1: list(reversed(self._group(1)))}
        plan = _native_core_report_plan(groups, lambda _name: 2)

        self.assertEqual(len(plan["groups"]), 2)
        self.assertEqual([cut for cut, _pieces in plan["batches"]], [2])
        self.assertEqual(len(plan["batches"][0][1]), 10)
        self.assertEqual(len(plan["object_names"]), 10)
        self.assertEqual(len(set(plan["object_names"])), 10)
        self.assertEqual(len(plan["membership_sha256"]), 64)

    def test_mixed_cut_groups_create_deterministic_batches(self):
        groups = {1: self._group(1), 2: self._group(2)}
        plan = _native_core_report_plan(
            groups,
            lambda name: 3 if _core_group_index(name) == 2 else 2,
        )

        self.assertEqual(
            [(cut, len(pieces)) for cut, pieces in plan["batches"]],
            [(2, 5), (3, 5)],
        )

    def test_missing_unexpected_or_duplicate_region_fails_closed(self):
        missing = self._group(1)[:-1]
        unexpected = self._group(1) + [SimpleNamespace(name="core_1_extra")]
        duplicate = self._group(1) + [self._group(1)[0]]
        for pieces, message, kwargs in (
            (missing, "coverage mismatch", {"require_complete_groups": True}),
            (unexpected, "coverage mismatch", {}),
            (duplicate, "duplicate native core report object", {}),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                    RuntimeError, message):
                _native_core_report_plan(
                    {1: pieces}, lambda _name: 2, **kwargs
                )

    def test_symmetry_reduced_group_covers_every_retained_piece(self):
        retained = [
            piece for piece in self._group(2)
            if not piece.name.endswith(("leg_right", "yoke_bottom"))
        ]
        plan = _native_core_report_plan({2: retained}, lambda _name: 2)

        self.assertEqual(len(plan["object_names"]), 3)
        self.assertEqual(
            set(plan["object_names"]), {piece.name for piece in retained}
        )
        self.assertEqual(len(plan["batches"][0][1]), 3)

    def test_group_with_mixed_cut_count_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "multiple symmetry-cut"):
            _native_core_report_plan(
                {1: self._group(1)},
                lambda name: 3 if name.endswith("yoke_top") else 2,
            )

    def test_native_batch_restoration_equals_legacy_group_math(self):
        for cut_count in (2, 3):
            with self.subTest(cut_count=cut_count):
                legacy_physical_factor = (
                    (2.0 ** cut_count) / (2.0 ** Y)
                    * (1.0 if cut_count == 3 else 2.0)
                )
                self.assertAlmostEqual(
                    _native_b_power_restore_factor(cut_count, Y, True),
                    legacy_physical_factor,
                )
        self.assertEqual(_native_b_power_restore_factor(2, Y, False), 1.0)


class _FakeMaterial:
    def __init__(self, name):
        self.name = name
        self.core_loss_calls = []
        self.permeability = None
        self.conductivity = None
        self.stacking_type = None
        self.stacking_factor = None
        self.stacking_direction = None

    def set_power_ferrite_coreloss(self, **kwargs):
        self.core_loss_calls.append(dict(kwargs))
        return True


class _FakeMaterials:
    def __init__(self):
        self._items = {}

    @property
    def material_keys(self):
        return list(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def duplicate_material(self, source, name):
        material = _FakeMaterial(name)
        self._items[name] = material
        return material


class _FakeMaxwellDesign:
    def __init__(self):
        self.materials = _FakeMaterials()
        self.set_calls = []

    def set_power_ferrite(self, **kwargs):
        self.set_calls.append(dict(kwargs))
        material = _FakeMaterial("power_ferrite")
        self.materials._items["power_ferrite"] = material
        return material


class CoreMaterialSolverIntegrationTests(unittest.TestCase):
    def setUp(self):
        _, self.frame = validation_check(
            create_input_parameter(get_drawing_default_params()), strict=True
        )

    def test_native_material_assignment_keeps_base_cm_and_zero_cut_depth(self):
        simulation = Simulation.__new__(Simulation)
        simulation.df_plus = self.frame
        simulation.design1 = _FakeMaxwellDesign()

        def readback(_materials, name):
            direction = (
                LEG_STACKING_DIRECTION if "leg" in name
                else YOKE_STACKING_DIRECTION
            )
            return native_props(direction)

        with patch("run_simulation_260706._raw_aedt_material_props", side_effect=readback):
            leg, leg_att = simulation._configure_1k101_native_material(
                "power_ferrite_1k101_leg_v1", LEG_STACKING_DIRECTION
            )
            yoke, yoke_att = simulation._configure_1k101_native_material(
                "power_ferrite_1k101_yoke_v3", YOKE_STACKING_DIRECTION
            )

        self.assertEqual(simulation.design1.set_calls[0]["cm"], CM)
        self.assertEqual(leg.core_loss_calls[-1]["cm"], CM)
        self.assertEqual(leg.core_loss_calls[-1]["cut_depth"], 0.0)
        self.assertEqual(yoke.core_loss_calls[-1]["cut_depth"], 0.0)
        self.assertEqual(leg.stacking_direction, "V(1)")
        self.assertEqual(yoke.stacking_direction, "V(3)")
        self.assertEqual(leg_att["core_loss_cm"], CM)
        self.assertEqual(yoke_att["conductivity"], 0.0)

    def test_flux_calculator_uses_explicit_plus_z_and_scalar_magnitude(self):
        operations = []

        class Reporter:
            def EnterQty(self, value): operations.append(("qty", value))
            def CalcOp(self, value): operations.append(("op", value))
            def EnterSurf(self, value): operations.append(("surf", value))

        simulation = Simulation.__new__(Simulation)

        def add(_name, builder, **_kwargs):
            builder(Reporter())
            return _name

        simulation._add_field_expression = add
        result = simulation._calc_core_flux_integral(
            [SimpleNamespace(name="section_1")], "Phi_test"
        )
        self.assertEqual(result, "Phi_test")
        self.assertEqual(operations, [
            ("qty", "B"), ("op", "ScalarZ"), ("surf", "section_1"),
            ("op", "SurfaceValue"), ("op", "Integrate"),
            ("op", "CmplxMag"),
        ])

    def test_b_peak_uses_complxpeak_not_complex_vector_norm(self):
        operations = []

        class Reporter:
            def EnterQty(self, value): operations.append(("qty", value))
            def CalcOp(self, value): operations.append(("op", value))
            def EnterVol(self, value): operations.append(("vol", value))

        simulation = Simulation.__new__(Simulation)
        simulation._add_field_expression = lambda name, builder, **kwargs: (
            builder(Reporter()) or name
        )
        simulation._calc_field_expr("core_1", "B_peak", "Mean", "B_test")
        self.assertIn(("op", "ComplxPeak"), operations)
        self.assertNotIn(("op", "CmplxMag"), operations)

    def test_thermal_injects_margin_adjusted_loss_and_native_readback_balances(self):
        corrected_core_w = 115.0
        sim = SimpleNamespace(
            df_plus=self.frame,
            loss_map_phys={"P_Rx_main_group": 10.0, "P_core_1": corrected_core_w},
            loss_map={"P_core_1": 100.0},
            df_loss_summary=SimpleNamespace(),
        )
        # Use a real one-row frame for full-total restoration.
        import pandas as pd
        sim.df_loss_summary = pd.DataFrame({"P_core_total": [corrected_core_w]})

        class Child:
            def __init__(self, name, power):
                self.values = {
                    "Block Type": "Solid", "Use Total Power": True,
                    "Total Power": power, "Objects": [name],
                }
            def GetPropNames(self): return list(self.values)
            def GetPropValue(self, key): return self.values[key]

        class Boundary:
            def __init__(self, name, power):
                self.props = {
                    "Block Type": "Solid", "Objects": [name],
                    "Total Power": power,
                }
                self._child_object = Child(name, power)

        class Obj:
            def __init__(self, name): self.name = name

        class Ipk:
            def assign_solid_block(self, name, power):
                return Boundary(name, power)

        objs = {
            "Tx": [],
            "Rx_main_explicit": [Obj("Rx_main_0_0")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [], "Rx_side_blocks": [],
            "Rx_side2_explicit": [], "Rx_side2_blocks": [],
            "core": [Obj("core_1")],
        }
        injected = _assign_losses(Ipk(), sim, objs, mode="full")
        self.assertEqual(injected["core_1"], corrected_core_w)
        self.assertEqual(sim.thermal_core_native_readback_w, corrected_core_w)
        self.assertEqual(sim.thermal_core_native_restored_full_w, corrected_core_w)
        self.assertEqual(sim.thermal_core_native_restored_rel_error, 0.0)
        self.assertEqual(
            sim.thermal_core_loss_source,
            "aedt_native_lamination_loss_attested_then_margin_adjusted",
        )

    def test_thermal_contract_rejects_raw_loss_fallback(self):
        sim = SimpleNamespace(df_plus=self.frame, loss_map={"P_core_1": 100.0})
        with self.assertRaisesRegex(RuntimeError, "requires loss_map_phys"):
            LossAllocator(sim, mode="full")


class CoreMaterialArtifactTests(unittest.TestCase):
    def test_rerun_manifest_is_prepared_but_cannot_submit_during_incident(self):
        path = (
            Path(__file__).resolve().parent / "verify"
            / "1k101_native_ab_rerun_441fb7f.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["status"],
            "prepared_not_submitted_due_raidrive_incident",
        )
        self.assertFalse(payload["execute"])
        self.assertEqual(
            payload["solver_revision"],
            "441fb7fff4be7894474886217d30c7bc0178a580",
        )
        self.assertEqual(
            [case["lamination_factor"] for case in payload["candidates"]],
            [1.0, 0.85, 0.7],
        )
        self.assertEqual(
            payload["submission_contract"]["priority"], 10
        )
        self.assertEqual(
            payload["submission_contract"]["required_project_cap"], 300
        )
        self.assertEqual(
            payload["prior_failure"]["task_ids"], [29964, 29965, 29966]
        )

    def test_ab_artifact_is_explicitly_blocked_until_solved_evidence(self):
        path = Path(__file__).resolve().parent / "verify" / "1k101_native_ab_gate.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["physics_data_revision"], PHYSICS_DATA_REVISION)
        self.assertEqual(payload["status"], "blocked_pending_actual_solved_ab")
        self.assertFalse(payload["production_approved"])
        self.assertEqual(
            [case["stacking_factor"] for case in payload["ab_cases"]],
            [1.0, 0.85, 0.70],
        )
        self.assertIn("corner_interface_flux", payload["required_solved_metrics"])
        self.assertIn("native_core_loss_vs_Bavg_power_integral", payload["required_solved_metrics"])

    def test_new_revision_is_fail_closed_in_quality_and_scheduler_ingest(self):
        root = Path(__file__).resolve().parent
        quality = (root / "quality_contract.py").read_text(encoding="utf-8")
        scheduler = (root / "verify" / "scheduler_client.py").read_text(
            encoding="utf-8"
        )
        for source in (quality, scheduler):
            self.assertIn("core_native_material_readback_attested", source)
            self.assertIn("core_loss_native_attested", source)
            self.assertIn("flux_linkage_attested", source)
            self.assertIn("approved_by_isolated_solved_kf_ab", source)
            self.assertIn("PHYSICS_DATA_REVISION", source)

    def test_measured_reconciliation_keeps_test_core_area_basis_separate(self):
        path = Path(__file__).resolve().parent / "verify" / "1k101_measured_loss_reconciliation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["test_core"]["stacking_factor_from_area"], 24.96/28.05)
        self.assertNotEqual(
            payload["test_core"]["stacking_factor_from_area"],
            payload["production_contract"]["lamination_factor"],
        )
        self.assertEqual(
            payload["production_contract"]["loss_margin_interpretation"],
            "engineering_margin_not_measurement_fit",
        )
        errors_1khz = [
            row["pred_minus_measured_pct"] for row in payload["points"]
            if row["frequency_hz"] == 1000.0
        ]
        self.assertLess(min(errors_1khz), -29.0)
        self.assertGreater(max(errors_1khz), 22.0)


if __name__ == "__main__":
    unittest.main()
