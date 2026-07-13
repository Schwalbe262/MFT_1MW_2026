import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from regression_260707.verify import replay_thermal_mesh as replay


def _sealed_mesh_fixture() -> bytes:
    objects = "\n".join(
        f"Name='{name}'" for name in replay.EXPECTED_TX_OBJECTS
    )
    ids = ", ".join(map(str, replay.EXPECTED_TX_OBJECT_IDS))
    return (
        "$begin 'IcepakModel'\n"
        "Name='icepak_thermal'\n"
        f"{objects}\n"
        "$begin 'MeshSetup'\n"
        "NextUniqueID=4\n"
        "$begin 'pad_mesh_level_TEST_L_2'\n"
        "DType='OpT'\nID=1\nObjects(11)\nMaxLevel='2'\nMinLevel='2'\n"
        "'Mesh Object(s) Separately Enabled'=false\n"
        "$end 'pad_mesh_level_TEST_L_2'\n"
        "$begin 'tx_mesh_level_TEST_L_2'\n"
        "DType='OpT'\nID=2\nEnable=true\n"
        f"Objects({ids})\n"
        "MaxLevel='2'\nMinLevel='2'\nIncrLevel='0'\n"
        "'Mesh Object(s) Separately Enabled'=false\n"
        "$end 'tx_mesh_level_TEST_L_2'\n"
        "$begin 'rx_block_mesh_level_TEST_L_4'\n"
        "DType='OpT'\nID=3\nObjects(22)\nMaxLevel='4'\nMinLevel='4'\n"
        "'Mesh Object(s) Separately Enabled'=false\n"
        "$end 'rx_block_mesh_level_TEST_L_4'\n"
        "$end 'MeshSetup'\n"
        "$begin 'AnalysisSetup'\nSentinel='unchanged'\n$end 'AnalysisSetup'\n"
        "$end 'IcepakModel'\n"
    ).encode("ascii")


class ThermalMeshReplayTests(unittest.TestCase):
    def test_profile_parser_reports_cells_and_stage_cpu_real_times(self):
        text = (
            "$begin 'Profile'\n"
            "Name='Solution Process'\n"
            "$begin 'TotalInfo'\n"
            "I(1, 'Elapsed Time', '00:08:11')\n"
            "$end 'TotalInfo'\n"
            "Name='Meshing Process'\n"
            "$begin 'TotalInfo'\n"
            "I(1, 'Elapsed Time', '00:01:20')\n"
            "$end 'TotalInfo'\n"
            "ProfileItem('Global', 62, 0, 61, 0, 1216544, 'x')\n"
            "ProfileItem('Populate Solver Input', 17, 0, 34, 0, 1518152, 'x')\n"
            "ProfileItem('Solver Initialization', 31, 0, 31, 0, 5789943, 'x')\n"
            "ProfileItem('Solve', 306, 0, 306, 0, 7409720, 'x')\n"
            "ProfileFootnote('I(3, 2, \\'Total Nodes\\', 1382530, false, "
            "2, \\'Total Faces\\', 511358, false, 2, \\'Total Cells\\', "
            "1116703, false)', 0)\n"
            "ProfileFootnote('I(2, 1, \\'Stop Time\\', \\'now\\', "
            "1, \\'Status\\', \\'Normal Completion\\')', 0)\n"
            "$end 'Profile'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "test.profile"
            profile.write_text(text, encoding="ascii")
            parsed = replay._parse_icepak_profile(profile)

        self.assertEqual(parsed["total_elapsed_seconds"], 491)
        self.assertEqual(parsed["meshing_elapsed_seconds"], 80)
        self.assertEqual(parsed["cells"], 1116703)
        self.assertEqual(parsed["nodes"], 1382530)
        self.assertEqual(parsed["mesh"]["cpu_seconds"], 62)
        self.assertEqual(parsed["mesh"]["real_seconds"], 61)
        self.assertEqual(parsed["solve"]["real_seconds"], 306)

    def test_thermal_target_summary_preserves_turn_hotspot_and_other_targets(self):
        values = {
            name: {"mean": 100.0 + index, "max": 110.0 + index}
            for index, name in enumerate(replay.EXPECTED_THERMAL_TARGET_OBJECTS)
        }
        values["Tx_main_1_0"] = {"mean": 250.0, "max": 300.0}
        summary = replay._thermal_target_summary(values)

        self.assertEqual(summary["tx_hotspot_turn"], "Tx_main_1_0")
        self.assertEqual(summary["tx_hotspot_rank"][0], "Tx_main_1_0")
        self.assertEqual(summary["T_max_Tx"], 300.0)
        self.assertTrue(summary["T_max_Rx_main"] > 0)
        self.assertTrue(summary["T_max_Rx_side"] > 0)
        self.assertTrue(summary["T_max_core"] > 0)

    def test_field_calculator_group_extracts_max_and_mean_from_saved_solution(self):
        calls = []

        class Post:
            @staticmethod
            def get_scalar_field_value(
                quantity, scalar_function, solution, object_name, object_type,
            ):
                calls.append((
                    quantity, scalar_function, solution, object_name, object_type,
                ))
                return 301.5 if scalar_function == "Maximum" else 250.25

        class Modeler:
            @staticmethod
            def get_object_from_name(name):
                return SimpleNamespace(name=name, is3d=not name.startswith("probe"))

        ipk = SimpleNamespace(post=Post(), modeler=Modeler())
        values = replay._field_calculator_object_group(
            ipk, "ThermalSetup : SteadyState", ("solid", "probe_sheet"),
        )

        self.assertEqual(
            values,
            {
                "solid": {"max": 301.5, "mean": 250.25},
                "probe_sheet": {"max": 301.5, "mean": 250.25},
            },
        )
        self.assertEqual([item[0] for item in calls], ["Temp"] * 4)
        self.assertEqual([item[4] for item in calls], [
            "volume", "volume", "surface", "surface",
        ])

    def test_thermal_target_extraction_records_calculator_fallback(self):
        values = {
            name: {"max": 200.0, "mean": 150.0}
            for name in replay.EXPECTED_THERMAL_TARGET_OBJECTS
        }
        with patch.object(
            replay, "_field_summary_thermal_targets",
            side_effect=RuntimeError("ExportFieldsSummary abnormal termination"),
        ), patch.object(
            replay, "_field_calculator_thermal_targets", return_value=values,
        ):
            extracted, provenance = replay._extract_thermal_targets(
                object(), "ThermalSetup : SteadyState",
            )

        self.assertIs(extracted, values)
        self.assertEqual(
            provenance["method"], "scalar_field_calculator_fallback"
        )
        self.assertEqual(provenance["solve_calls"], 0)
        self.assertIn("abnormal termination", provenance["field_summary_error"])

    def test_tx_zone_parser_requires_cfd_mesh_record(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "current.nc_cas"
            case.write_text(
                "header\n(cfd-post-mesh-info ((0 0 ("
                + " ".join(replay.EXPECTED_TX_ZONES)
                + ")))\n",
                encoding="utf-8",
            )
            self.assertEqual(
                replay._tx_zones_from_case(case), replay.EXPECTED_TX_ZONES
            )
            case.write_text("no mesh record\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cfd-post-mesh-info"):
                replay._tx_zones_from_case(case)

    def test_source_contract_reads_exact_tx_objects_and_shared_level(self):
        objects = "\n".join(
            f"Name='{name}'" for name in replay.EXPECTED_TX_OBJECTS
        )
        text = (
            "$begin 'IcepakModel'\n"
            "Name='icepak_thermal'\n"
            f"{objects}\n"
            "$begin 'tx_mesh_level_TEST_L_2'\n"
            "ID=2\n"
            f"Objects({','.join(map(str, replay.EXPECTED_TX_OBJECT_IDS))})\n"
            "MaxLevel='2'\nMinLevel='2'\n"
            "'Mesh Object(s) Separately Enabled'=false\n"
            "$end 'tx_mesh_level_TEST_L_2'\n"
            "$end 'IcepakModel'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "test.aedt"
            project.write_text(text, encoding="utf-8")
            contract = replay._source_icepak_contract(project)
        self.assertEqual(tuple(contract["tx_objects"]), replay.EXPECTED_TX_OBJECTS)
        self.assertEqual(contract["mesh_levels"], [2])
        self.assertEqual(contract["separate"], "false")

    def test_freshness_gate_rejects_old_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "monitor.sd"
            artifact.write_text("monitor", encoding="utf-8")
            started_ns = artifact.stat().st_mtime_ns + 1
            with self.assertRaisesRegex(RuntimeError, "stale monitor"):
                replay._assert_fresh(artifact, started_ns, "monitor")
            replay._assert_fresh(artifact, artifact.stat().st_mtime_ns, "monitor")

    def test_source_snapshot_detects_project_change(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "test.aedt"
            project.write_text("first", encoding="utf-8")
            before = replay._source_snapshot(project)
            project.write_text("second", encoding="utf-8")
            after = replay._source_snapshot(project)
        self.assertNotEqual(before, after)
        self.assertNotEqual(
            before["project"]["sha256"], after["project"]["sha256"]
        )

    def test_owned_headless_tree_excludes_preexisting_and_sibling_processes(self):
        root_pid = 100
        before = {
            200: {
                "pid": 200, "ppid": root_pid, "create_time": 1.0,
                "name": "AnsysEDT.exe",
                "commandline": "AnsysEDT.exe -ng -grpcsrv 5000",
            },
        }
        after = {
            **before,
            300: {
                "pid": 300, "ppid": root_pid, "create_time": 2.0,
                "name": "launcher.exe", "commandline": "launcher.exe",
            },
            301: {
                "pid": 301, "ppid": 300, "create_time": 2.1,
                "name": "AnsysEDT.exe",
                "commandline": "AnsysEDT.exe -ng -grpcsrv 5001",
            },
            302: {
                "pid": 302, "ppid": 301, "create_time": 2.2,
                "name": "worker.exe", "commandline": "worker.exe",
            },
            400: {
                "pid": 400, "ppid": root_pid, "create_time": 3.0,
                "name": "unrelated.exe", "commandline": "unrelated.exe",
            },
        }

        owned = replay._new_owned_headless_tree(before, after, root_pid)

        self.assertEqual(set(owned), {300, 301, 302})

    def test_operation_object_check_uses_props_when_native_readback_is_unavailable(self):
        class UnavailableNativeMesh:
            @staticmethod
            def GetMeshOpAssignment(_name):
                raise RuntimeError("unsupported gRPC method")

        ipk = SimpleNamespace(
            mesh=SimpleNamespace(omeshmodule=UnavailableNativeMesh()),
        )
        operation = SimpleNamespace(
            name="tx_mesh_level_TEST_L_2",
            props={"Objects": list(replay.EXPECTED_TX_OBJECTS)},
        )

        self.assertEqual(
            replay._verify_operation_objects(ipk, operation),
            replay.EXPECTED_TX_OBJECTS,
        )

    def test_uniform_live_attestation_returns_complete_contract(self):
        operation = SimpleNamespace(
            name="tx_mesh_level_TEST_L_4",
            props={
                "Objects": list(replay.EXPECTED_TX_OBJECTS),
                "MaxLevel": "4",
                "MinLevel": "4",
                "Mesh Object(s) Separately Enabled": False,
            },
        )
        ipk = SimpleNamespace(
            mesh=SimpleNamespace(meshoperations=[operation]),
        )

        contract = replay._attest_live_tx_mesh(ipk)

        self.assertEqual(contract["edited_operation"], operation.name)
        self.assertEqual(contract["objects"], list(replay.EXPECTED_TX_OBJECTS))
        self.assertEqual(contract["level"], 4)
        self.assertFalse(contract["separate"])

    def test_cloned_file_edit_changes_only_tx_levels_and_preserves_assignments(self):
        objects = ", ".join(map(str, replay.EXPECTED_TX_OBJECT_IDS))
        text = (
            "$begin 'IcepakModel'\n"
            "Name='icepak_thermal'\n"
            + "\n".join(f"Name='{name}'" for name in replay.EXPECTED_TX_OBJECTS)
            + "\n$begin 'tx_mesh_level_TEST_L_2'\n"
            "ID=2\n"
            f"Objects({objects})\n"
            "MaxLevel='2'\nMinLevel='2'\n"
            "'Mesh Object(s) Separately Enabled'=false\n"
            "$end 'tx_mesh_level_TEST_L_2'\n"
            "$begin 'pad_mesh_level_TEST_L_2'\n"
            "MaxLevel='2'\nMinLevel='2'\n"
            "$end 'pad_mesh_level_TEST_L_2'\n"
            "$end 'IcepakModel'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(text.encode("ascii"))
            contract = replay._edit_cloned_tx_mesh_file(project, 3)
            edited = project.read_text(encoding="ascii")

        self.assertEqual(contract["mesh_levels"], [3])
        self.assertEqual(tuple(contract["mesh_object_ids"]), replay.EXPECTED_TX_OBJECT_IDS)
        self.assertIn("MaxLevel='3'\nMinLevel='3'", edited)
        self.assertIn(
            "$begin 'pad_mesh_level_TEST_L_2'\nMaxLevel='2'\nMinLevel='2'",
            edited,
        )

    def test_hybrid_partition_k3_is_edge_l4_interior_l3(self):
        plan = replay._hybrid_tx_partition(edge_fine_turns=3)

        self.assertEqual(
            [item["indices"] for item in plan["operations"]],
            [[0, 1, 2], [3, 4], [5, 6, 7]],
        )
        self.assertEqual(
            [item["level"] for item in plan["operations"]], [4, 3, 4]
        )
        self.assertEqual(
            [item["operation_id"] for item in plan["operations"]], [2, 4, 5]
        )
        self.assertEqual(plan["next_unique_id"], 6)
        self.assertEqual(
            [item["edge_distance"] for item in plan["turn_policy"]],
            [0, 1, 2, 3, 3, 2, 1, 0],
        )

    def test_hybrid_partition_k2_is_edge_l4_interior_l3(self):
        plan = replay._hybrid_tx_partition(edge_fine_turns=2)

        self.assertEqual(
            [item["indices"] for item in plan["operations"]],
            [[0, 1], [2, 3, 4, 5], [6, 7]],
        )
        self.assertEqual(
            [item["level"] for item in plan["operations"]], [4, 3, 4]
        )

    def test_hybrid_partition_small_n_is_one_overlap_free_all_l4_operation(self):
        plan = replay._hybrid_tx_partition(
            tx_objects=("turn0", "turn1", "turn2", "turn3"),
            tx_object_ids=(10, 11, 12, 13),
            edge_fine_turns=2,
        )

        self.assertTrue(plan["all_fine_due_to_small_N"])
        self.assertEqual(len(plan["operations"]), 1)
        self.assertEqual(plan["operations"][0]["indices"], [0, 1, 2, 3])
        self.assertEqual(plan["operations"][0]["level"], 4)
        self.assertEqual(plan["operations"][0]["operation_id"], 2)
        self.assertEqual(plan["next_unique_id"], 4)

    def test_hybrid_partition_rejects_object_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "object-ID overlap"):
            replay._hybrid_tx_partition(
                tx_objects=("turn0", "turn1"),
                tx_object_ids=(10, 10),
                edge_fine_turns=1,
            )

    def test_hybrid_clone_edit_k3_has_exact_ids_order_and_byte_scope(self):
        before = _sealed_mesh_fixture()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(before)
            result = replay._edit_cloned_tx_mesh_hybrid_file(
                project, edge_fine_turns=3,
            )
            after = project.read_bytes()

        contracts = result["durable_contract"]["tx_operations"]
        self.assertEqual(
            [item["operation_id"] for item in contracts], [2, 4, 5]
        )
        self.assertEqual(
            [item["mesh_object_ids"] for item in contracts],
            [
                list(replay.EXPECTED_TX_OBJECT_IDS[:3]),
                list(replay.EXPECTED_TX_OBJECT_IDS[3:5]),
                list(replay.EXPECTED_TX_OBJECT_IDS[5:]),
            ],
        )
        self.assertEqual(
            result["durable_contract"]["mesh_setup"]["next_unique_id"], 6
        )
        self.assertTrue(result["byte_scope"]["unrelated_bytes_invariant"])
        self.assertIn(b"Sentinel='unchanged'", after)

    def test_hybrid_clone_edit_k2_has_expected_split(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(_sealed_mesh_fixture())
            result = replay._edit_cloned_tx_mesh_hybrid_file(
                project, edge_fine_turns=2,
            )

        contracts = result["durable_contract"]["tx_operations"]
        self.assertEqual(
            [item["mesh_object_ids"] for item in contracts],
            [
                list(replay.EXPECTED_TX_OBJECT_IDS[:2]),
                list(replay.EXPECTED_TX_OBJECT_IDS[2:6]),
                list(replay.EXPECTED_TX_OBJECT_IDS[6:]),
            ],
        )

    def test_hybrid_clone_edit_n_le_2k_keeps_one_id2_and_next_id4(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(_sealed_mesh_fixture())
            result = replay._edit_cloned_tx_mesh_hybrid_file(
                project, edge_fine_turns=4,
            )

        contracts = result["durable_contract"]["tx_operations"]
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["operation_id"], 2)
        self.assertEqual(contracts[0]["mesh_levels"], [4])
        self.assertEqual(
            contracts[0]["mesh_object_ids"], list(replay.EXPECTED_TX_OBJECT_IDS)
        )
        self.assertEqual(
            result["durable_contract"]["mesh_setup"]["next_unique_id"], 4
        )

    def test_hybrid_contract_rejects_operation_id_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(_sealed_mesh_fixture())
            result = replay._edit_cloned_tx_mesh_hybrid_file(
                project, edge_fine_turns=3,
            )
            edited = project.read_bytes().replace(b"ID=4", b"ID=2", 1)
            project.write_bytes(edited)
            with self.assertRaisesRegex(RuntimeError, "operation ID overlap"):
                replay._assert_hybrid_file_contract(project, result["plan"])

    def test_hybrid_clone_edit_rejects_malformed_tx_block(self):
        malformed = _sealed_mesh_fixture().replace(
            b"Objects(1445,", b"Objectz(1445,", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(malformed)
            with self.assertRaisesRegex(RuntimeError, "Objects row"):
                replay._edit_cloned_tx_mesh_hybrid_file(
                    project, edge_fine_turns=3,
                )

    def test_hybrid_byte_scope_rejects_changed_unrelated_bytes(self):
        before = _sealed_mesh_fixture()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "clone.aedt"
            project.write_bytes(before)
            replay._edit_cloned_tx_mesh_hybrid_file(
                project, edge_fine_turns=3,
            )
            after = project.read_bytes().replace(
                b"pad_mesh_level_TEST_L_2", b"pad_mesh_level_TEST_X_2", 1
            )

        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            replay._assert_hybrid_byte_change_scope(before, after)


if __name__ == "__main__":
    unittest.main()
