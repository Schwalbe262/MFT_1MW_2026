import math
import unittest

from module.electrostatic_cap import (
    CAPACITANCE_EXPORT_SCHEMA_VERSION,
    CAPACITANCE_PAYLOAD_FIELDS,
    CAPACITANCE_PAYLOAD_SCHEMA_VERSION,
    CAPACITANCE_TIMING_PAYLOAD_FIELDS,
    build_capacitance_payload,
    build_capacitance_timing_payload,
    capacitance_unit_scale,
    lc_resonance_hz,
    parse_maxwell_capacitance_export,
)


# Native Maxwell export layout reproduced from the PyAEDT/Maxwell example.
OFFICIAL_MAXWELL_SAMPLE = """\
DesignVariation : extent_x_pos='1000mm' extent_x_size='-1000mm' extent_y_pos='1000mm' extent_y_size='-1000mm' Signal1_lower_elevation='0mm'
Solution : Setup1 : LastAdaptive
Parameter : Matrix1
Capacitance Unit: pF

Capacitance
        V0      V1      V2
    V0  0.12226 -0.059571       -0.062688
    V1  -0.059571      0.081662        -0.022091
    V2  -0.062688      -0.022091       0.084779

Capacitive Coupling Coefficient
        V0      V1      V2
    V0  1       -0.59619       -0.61574
    V1  -0.59619        1       -0.2655
    V2  -0.61574        -0.2655        1
"""


def _two_net_export(unit="pF", c11="10", c12="-2", c22="4"):
    return f"""\
DesignVariation :
Solution : Setup1 : LastAdaptive
Parameter : CapMatrix
Capacitance Unit: {unit}

Capacitance
        CapTx CapRx
CapTx   {c11} {c12}
CapRx   {c12} {c22}

Capacitive Coupling Coefficient
        CapTx CapRx
CapTx   1 -0.316227766
CapRx   -0.316227766 1
"""


class MaxwellCapacitanceParserTest(unittest.TestCase):
    def test_parses_official_native_export_and_ignores_coupling_table(self):
        parsed = parse_maxwell_capacitance_export(OFFICIAL_MAXWELL_SAMPLE)

        self.assertEqual(
            parsed["schema_version"], CAPACITANCE_EXPORT_SCHEMA_VERSION
        )
        self.assertEqual(parsed["unit"], "pF")
        self.assertEqual(parsed["names"], ("V0", "V1", "V2"))
        self.assertEqual(parsed["matrix_raw"][0], (0.12226, -0.059571, -0.062688))
        self.assertAlmostEqual(parsed["matrix_raw"][2][1], -0.022091)
        self.assertAlmostEqual(parsed["matrix_f"][1][1], 0.081662e-12)

    def test_capacitance_unit_conversion_covers_aedt_si_units(self):
        cases = {
            "fF": 1e-15,
            "pF": 1e-12,
            "nF": 1e-9,
            "uF": 1e-6,
            "µF": 1e-6,
            "μF": 1e-6,
            "mF": 1e-3,
            "F": 1.0,
            "farad": 1.0,
            "[pF]": 1e-12,
        }
        for unit, expected in cases.items():
            with self.subTest(unit=unit):
                self.assertEqual(capacitance_unit_scale(unit), expected)

        parsed = parse_maxwell_capacitance_export(
            _two_net_export(unit="nF", c11="1e-3", c12="-2.5e-4", c22="4e-3")
        )
        self.assertAlmostEqual(parsed["matrix_f"][0][0], 1e-12)
        self.assertAlmostEqual(parsed["matrix_f"][0][1], -2.5e-13)

    def test_rejects_malformed_and_asymmetric_matrices(self):
        malformed = {
            "missing unit": """Capacitance\n A B\n A 1 -1\n B -1 1\n""",
            "missing table": "Capacitance Unit: pF\nParameter: Matrix\n",
            "missing row": """Capacitance Unit: pF\nCapacitance\n A B\n A 1 -1\n""",
            "short row": """Capacitance Unit: pF\nCapacitance\n A B\n A 1\n B -1 1\n""",
            "duplicate columns": """Capacitance Unit: pF\nCapacitance\n A A\n A 1 -1\n A -1 1\n""",
            "non-numeric": """Capacitance Unit: pF\nCapacitance\n A B\n A 1 bad\n B bad 1\n""",
            "asymmetric": """Capacitance Unit: pF\nCapacitance\n A B\n A 1 -0.1\n B -0.2 1\n""",
        }
        for name, text in malformed.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    parse_maxwell_capacitance_export(text)

    def test_rejects_invalid_text_and_units(self):
        with self.assertRaises(TypeError):
            parse_maxwell_capacitance_export(b"Capacitance Unit: pF")
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_maxwell_capacitance_export(" \n")
        for unit in (None, True, "", "kF", "picoFarads"):
            with self.subTest(unit=unit):
                with self.assertRaises(ValueError):
                    capacitance_unit_scale(unit)


class ResonanceMathTest(unittest.TestCase):
    def test_lc_resonance_scalar_math(self):
        expected = 1.0 / (2.0 * math.pi * math.sqrt(1e-3 * 1e-6))
        self.assertAlmostEqual(lc_resonance_hz(1e-3, 1e-6), expected)

    def test_lc_resonance_rejects_nonpositive_nonfinite_and_bad_values(self):
        bad_values = (0, -1, float("nan"), float("inf"), None, True, "bad")
        for value in bad_values:
            with self.subTest(inductance=value):
                with self.assertRaises(ValueError):
                    lc_resonance_hz(value, 1e-12)
            with self.subTest(capacitance=value):
                with self.assertRaises(ValueError):
                    lc_resonance_hz(1e-6, value)
        with self.assertRaisesRegex(ValueError, "representable"):
            lc_resonance_hz(1e308, 1e308)
        with self.assertRaisesRegex(ValueError, "representable"):
            lc_resonance_hz(1e-300, 1e-300)

    def test_capacitance_timing_payload_schema_and_validation(self):
        payload = build_capacitance_timing_payload(2.0, 0.25, 3.0)
        self.assertEqual(set(payload), CAPACITANCE_TIMING_PAYLOAD_FIELDS)
        self.assertEqual(payload["time_cap"], 2.0)
        self.assertEqual(payload["cap_solve_time_s"], 2.0)
        self.assertEqual(payload["cap_extraction_time_s"], 0.25)
        self.assertEqual(payload["cap_stage_added_time_s"], 3.0)
        for values in (
            (-1.0, 0.0, 1.0),
            (1.0, float("nan"), 2.0),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    build_capacitance_timing_payload(*values)


class CapacitancePayloadTest(unittest.TestCase):
    def test_eighth_payload_restores_c_by_eight_and_l_by_two(self):
        parsed = parse_maxwell_capacitance_export(_two_net_export())
        payload = build_capacitance_payload(
            parsed, ltx_uH=100.0, lrx_uH=400.0, llt_uH=10.0,
            full_model=False,
        )

        self.assertEqual(payload["cap_schema_version"], CAPACITANCE_PAYLOAD_SCHEMA_VERSION)
        self.assertEqual(set(payload), CAPACITANCE_PAYLOAD_FIELDS)
        self.assertEqual(payload["cap_model_basis"], "eighth")
        self.assertEqual(payload["cap_raw_capacitance_basis"], "retained_eighth_geometry")
        self.assertEqual(payload["cap_output_basis"], "full_physical")
        self.assertEqual(payload["cap_capacitance_restoration_factor"], 8.0)
        self.assertEqual(payload["cap_inductance_restoration_factor"], 2.0)
        self.assertEqual(payload["cap_inductance_source"], "matrix_stage:Ltx,Lrx,Llt")
        self.assertEqual(payload["cap_inductance_source_unit"], "uH")
        self.assertEqual(payload["cap_matrix_order"], '["CapTx","CapRx"]')
        self.assertEqual(
            payload["cap_diagonal_interpretation"],
            "grounded_other_signal_maxwell_coefficient",
        )
        self.assertEqual(payload["cap_region_remote_padding_percent"], 200.0)

        self.assertAlmostEqual(payload["C_tx_tx_raw_F"], 10e-12)
        self.assertAlmostEqual(payload["C_rx_rx_raw_F"], 4e-12)
        self.assertAlmostEqual(payload["C_tx_rx_signed_raw_F"], -2e-12)
        self.assertAlmostEqual(payload["C_tx_rx_raw_F"], 2e-12)
        self.assertAlmostEqual(payload["C_tx_tx_F"], 80e-12)
        self.assertAlmostEqual(payload["C_rx_rx_F"], 32e-12)
        self.assertAlmostEqual(payload["C_tx_rx_signed_F"], -16e-12)
        self.assertAlmostEqual(payload["C_tx_rx_F"], 16e-12)

        self.assertAlmostEqual(payload["cap_L_tx_self_H"], 200e-6)
        self.assertAlmostEqual(payload["cap_L_rx_self_H"], 800e-6)
        self.assertAlmostEqual(payload["cap_L_leakage_H"], 20e-6)
        self.assertAlmostEqual(
            payload["f_res_tx_self_Hz"], lc_resonance_hz(200e-6, 80e-12)
        )
        self.assertAlmostEqual(
            payload["f_res_rx_self_Hz"], lc_resonance_hz(800e-6, 32e-12)
        )
        self.assertAlmostEqual(
            payload["f_res_interwinding_Hz"], lc_resonance_hz(20e-6, 16e-12)
        )

    def test_full_payload_uses_unit_factors_and_configurable_net_names(self):
        text = _two_net_export().replace("CapTx", "Primary").replace("CapRx", "Secondary")
        parsed = parse_maxwell_capacitance_export(text)
        payload = build_capacitance_payload(
            parsed,
            100.0,
            400.0,
            10.0,
            full_model=1,
            tx_name="Primary",
            rx_name="Secondary",
        )

        self.assertEqual(payload["cap_model_basis"], "full")
        self.assertEqual(payload["cap_raw_capacitance_basis"], "full_geometry")
        self.assertEqual(payload["cap_raw_inductance_basis"], "full_model_matrix")
        self.assertEqual(payload["cap_capacitance_restoration_factor"], 1.0)
        self.assertEqual(payload["cap_inductance_restoration_factor"], 1.0)
        self.assertEqual(payload["cap_region_remote_padding_percent"], 100.0)
        self.assertEqual(payload["C_tx_tx_F"], payload["C_tx_tx_raw_F"])
        self.assertEqual(payload["C_tx_rx_F"], payload["C_tx_rx_raw_F"])
        self.assertAlmostEqual(payload["cap_L_tx_self_H"], 100e-6)

    def test_payload_rejects_invalid_matrix_inductance_mode_and_names(self):
        parsed = parse_maxwell_capacitance_export(_two_net_export())
        for bad_l in (0, -1, float("nan"), float("inf"), None, True):
            with self.subTest(inductance=bad_l):
                with self.assertRaises(ValueError):
                    build_capacitance_payload(parsed, bad_l, 400.0, 10.0)
        for mode in (2, -1, float("nan"), "0", None):
            with self.subTest(full_model=mode):
                with self.assertRaises(ValueError):
                    build_capacitance_payload(
                        parsed, 100.0, 400.0, 10.0, full_model=mode
                    )
        with self.assertRaisesRegex(ValueError, "must contain"):
            build_capacitance_payload(
                parsed, 100.0, 400.0, 10.0, tx_name="Missing"
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_capacitance_payload(
                parsed, 100.0, 400.0, 10.0,
                tx_name="CapTx", rx_name="CapTx",
            )

        zero_coupling = parse_maxwell_capacitance_export(
            _two_net_export(c12="0")
        )
        with self.assertRaisesRegex(ValueError, "must be negative"):
            build_capacitance_payload(zero_coupling, 100.0, 400.0, 10.0)

        positive_coupling = parse_maxwell_capacitance_export(
            _two_net_export(c12="2")
        )
        with self.assertRaisesRegex(ValueError, "must be negative"):
            build_capacitance_payload(
                positive_coupling, 100.0, 400.0, 10.0
            )

        negative_ground_partial = parse_maxwell_capacitance_export(
            _two_net_export(c11="1", c12="-2", c22="10")
        )
        with self.assertRaisesRegex(ValueError, "negative ground partial"):
            build_capacitance_payload(
                negative_ground_partial, 100.0, 400.0, 10.0
            )

        nonphysical = parse_maxwell_capacitance_export(
            _two_net_export(c11="1", c12="-3", c22="4")
        )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            build_capacitance_payload(nonphysical, 100.0, 400.0, 10.0)

        corrupt = dict(parsed)
        corrupt["matrix_f"] = ((10e-12, -2e-12), (-2e-12, 99e-12))
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_capacitance_payload(corrupt, 100.0, 400.0, 10.0)


if __name__ == "__main__":
    unittest.main()
