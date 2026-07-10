import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import collect_wave  # noqa: E402
import train_io  # noqa: E402


def raw_rows():
    return pd.DataFrame([
        {
            "project_name": "sym",
            "saved_at": "2026-07-10 01:00:00",
            "full_model": 0,
            "matrix_skin_mesh": 0,
            "N1_main": 7,
            "N2_main": 65,
            "N2_side": 5,
            "Ltx": 20.0,
            "Lrx": 2000.0,
            "M": 190.0,
            "Lmt": 18.0,
            "Lmr": 1800.0,
            "Llt": 13.75,
            "Llr": 1375.0,
            "k": 0.95,
            "P_winding_total": 4000.0,
            "P_core_total": 2000.0,
            "B_max_core": 1.1,
            "T_max_Rx_main": 91.0,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
            "git_hash": "a" * 40,
            "pyaedt_library_git_hash": "b" * 40,
            "P_turn_Rx_main_0_0": 12.0,
            "P_turn_Rx_main_0_0_raw": 12.0,
            "P_core_3": 100.0,
            "P_core_3_raw": 25.0,
            "T_mean_Rx_main_0_0": 85.0,
            "T_max_Rx_main_0_0": 92.0,
            "thermal_iterations": 151,
            "thermal_residual_energy": 4e-9,
        },
        {
            "project_name": "full",
            "saved_at": "2026-07-10 02:00:00",
            "full_model": 1,
            "matrix_skin_mesh": 1,
            "N1_main": 7,
            "N2_main": 65,
            "N2_side": 5,
            "Ltx": 40.0,
            "Lrx": 4000.0,
            "M": 380.0,
            "Lmt": 36.0,
            "Lmr": 3600.0,
            "Llt": 27.5,
            "Llr": 2750.0,
            "k": 0.95,
            "P_winding_total": 4100.0,
            "P_core_total": 2100.0,
            "B_max_core": 1.05,
            "T_max_Rx_main": 90.0,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
            "git_hash": "c" * 40,
            "pyaedt_library_git_hash": "d" * 40,
        },
    ])


class TrainIoBuilderTests(unittest.TestCase):
    def test_builder_is_fixed_order_physical_and_does_not_mutate_raw(self):
        raw = raw_rows()
        before = raw.copy(deep=True)

        view = train_io.build_train_io(raw)

        self.assertEqual(tuple(view.columns), train_io.TRAIN_IO_COLUMNS)
        self.assertEqual(view["train_io_schema_version"].tolist(), [1, 1])
        self.assertEqual(
            view["inductance_source_basis"].tolist(),
            ["eighth_symmetry", "full_model"],
        )
        self.assertEqual(view["inductance_to_physical_factor"].tolist(), [2.0, 1.0])
        self.assertEqual(view["Llt_phys"].tolist(), [27.5, 27.5])
        self.assertEqual(view["M_phys"].tolist(), [380.0, 380.0])
        self.assertEqual(view["matrix_skin_mesh"].tolist(), [0, 1])
        self.assertEqual(view["P_winding_total"].tolist(), [4000.0, 4100.0])
        self.assertEqual(view["T_max_Rx_main"].tolist(), [91.0, 90.0])
        pd.testing.assert_frame_equal(raw, before)

    def test_builder_excludes_dynamic_raw_and_solver_telemetry_columns(self):
        view = train_io.build_train_io(raw_rows())

        excluded = {
            "Llt",
            "P_turn_Rx_main_0_0",
            "P_turn_Rx_main_0_0_raw",
            "P_core_3",
            "P_core_3_raw",
            "T_mean_Rx_main_0_0",
            "T_max_Rx_main_0_0",
            "thermal_iterations",
            "thermal_residual_energy",
        }
        self.assertTrue(excluded.isdisjoint(view.columns))
        self.assertFalse(any(column.endswith("_raw") for column in view.columns))
        self.assertTrue(pd.isna(view.loc[0, "Ae_m2"]))

    def test_empty_builder_still_has_the_complete_contract(self):
        view = train_io.build_train_io(pd.DataFrame())

        self.assertEqual(view.shape, (0, len(train_io.TRAIN_IO_COLUMNS)))
        self.assertEqual(tuple(view.columns), train_io.TRAIN_IO_COLUMNS)


class TrainIoCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tempdir.name)
        self.cache_path = self.dataset_dir / "collect_cache.json"
        self.patches = [
            mock.patch.object(collect_wave, "DATASET_DIR", str(self.dataset_dir)),
            mock.patch.object(collect_wave, "CACHE_PATH", str(self.cache_path)),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def ranked_rows(self):
        frame = raw_rows()
        frame[collect_wave.SOURCE_RANK_COLUMN] = collect_wave.SOURCE_RANK_JSON
        return frame

    def test_merge_preserves_raw_and_writes_matching_atomic_views(self):
        new_rows, total, _ = collect_wave.merge_dataset(
            self.ranked_rows(), ["project_name", "saved_at"], "mft-camp"
        )

        self.assertEqual((new_rows, total), (2, 2))
        raw = pd.read_parquet(self.dataset_dir / "train.parquet")
        parquet_view = pd.read_parquet(self.dataset_dir / "train_io.parquet")
        csv_view = pd.read_csv(self.dataset_dir / "train_io.csv")
        self.assertIn("P_turn_Rx_main_0_0", raw.columns)
        self.assertNotIn("P_turn_Rx_main_0_0", parquet_view.columns)
        self.assertEqual(tuple(parquet_view.columns), train_io.TRAIN_IO_COLUMNS)
        self.assertEqual(tuple(csv_view.columns), train_io.TRAIN_IO_COLUMNS)
        self.assertEqual(parquet_view["Llt_phys"].tolist(), [27.5, 27.5])
        self.assertEqual(csv_view["Llt_phys"].tolist(), [27.5, 27.5])

    def test_duplicate_merge_recreates_missing_io_without_rewriting_raw(self):
        ranked = self.ranked_rows()
        collect_wave.merge_dataset(
            ranked, ["project_name", "saved_at"], "mft-camp"
        )
        master = self.dataset_dir / "train.parquet"
        before = master.read_bytes()
        (self.dataset_dir / "train_io.csv").unlink()

        new_rows, total, _ = collect_wave.merge_dataset(
            ranked, ["project_name", "saved_at"], "mft-camp"
        )

        self.assertEqual((new_rows, total), (0, 2))
        self.assertEqual(master.read_bytes(), before)
        self.assertTrue((self.dataset_dir / "train_io.csv").is_file())

    def test_serialization_failure_preserves_both_existing_views(self):
        parquet_path = self.dataset_dir / "train_io.parquet"
        csv_path = self.dataset_dir / "train_io.csv"
        pd.DataFrame({"sentinel": [1]}).to_parquet(parquet_path, index=False)
        csv_path.write_text("sentinel\n1\n", encoding="utf-8")
        parquet_before = parquet_path.read_bytes()
        csv_before = csv_path.read_bytes()

        with mock.patch.object(
            collect_wave, "_stage_csv", side_effect=RuntimeError("csv full")
        ):
            with self.assertRaisesRegex(RuntimeError, "csv full"):
                collect_wave._replace_train_io_views(raw_rows())

        self.assertEqual(parquet_path.read_bytes(), parquet_before)
        self.assertEqual(csv_path.read_bytes(), csv_before)
        self.assertEqual(list(self.dataset_dir.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
