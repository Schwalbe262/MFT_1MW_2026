from tools import mft_goal_fixed_lm2mh_targeted_collect as collector


def _entry(spec, seed, task_id):
    return {
        "seed": seed,
        "task_id": task_id,
        "_collector_campaign_id": spec["campaign_id"],
        "_collector_replacement": spec["replacement"],
    }


def test_combined_contract_has_exact_528_seed_168960_row_coverage():
    assert collector.EXPECTED_SEED_COUNT == 528
    assert collector.EXPECTED_RAW_ROWS == 168_960
    assert len(collector.LEGACY_SEEDS) == 16
    assert len(collector.SPLITTEMP_V2_SEEDS) == 256
    assert len(collector.ROLLING_V3_SEEDS) == 256
    assert len(collector.FRESH_SEEDS) == 512


def test_explicit_retry_entries_replace_only_their_failed_seed_tasks():
    source_entries = []
    for spec in collector.SOURCE_SPECS:
        source_entries.extend(
            _entry(spec, seed, task_id)
            for seed, task_id in zip(spec["seeds"], spec["task_ids"])
        )

    selected, superseded = collector._select_authoritative_entries(
        source_entries
    )
    collector._validate_combined_coverage(selected, superseded)

    by_seed = {
        int(entry["seed"]): int(entry["task_id"])
        for entry in selected
        if entry["_collector_campaign_id"]
        == collector.SPLITTEMP_CAMPAIGN_ID
    }
    assert by_seed[2_707_277_022] == 96_741
    assert by_seed[2_707_277_061] == 96_742
    assert superseded == [
        {
            "seed_family": "fresh",
            "seed": 2_707_277_022,
            "superseded_task_id": 96_507,
            "selected_task_id": 96_741,
            "selected_is_replacement": True,
        },
        {
            "seed_family": "fresh",
            "seed": 2_707_277_061,
            "superseded_task_id": 96_546,
            "selected_task_id": 96_742,
            "selected_is_replacement": True,
        },
    ]


def test_highest_successful_retry_wins_after_newer_retry_fails():
    base_spec = next(
        spec
        for spec in collector.SOURCE_SPECS
        if spec["label"] == "splittemp-v2-seeds0000-0063"
    )
    seed = 2_707_277_022
    base = _entry(base_spec, seed, 96_507)
    retry1 = {
        **base,
        "task_id": 96_741,
        "_collector_replacement": True,
    }
    retry2 = {
        **base,
        "task_id": 97_100,
        "_collector_replacement": True,
    }
    selected, _superseded = collector._select_authoritative_entries(
        [base, retry1, retry2],
        {
            96_507: {"status": "failed"},
            96_741: {"status": "completed"},
            97_100: {"status": "failed"},
        },
    )

    assert [entry["task_id"] for entry in selected] == [96_741]


def test_multi_seed_retry_manifest_is_an_exact_replacement_source(
    tmp_path, monkeypatch
):
    directory = (
        tmp_path / "fixed_lm2mh_splittemp_v3_retry_seeds0237_0238"
    )
    directory.mkdir()
    path = directory / "submission_manifest.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(collector, "RETRY_SEARCH_ROOTS", (tmp_path,))
    seeds = (2_707_277_237, 2_707_277_238)
    task_ids = (97_030, 97_031)
    manifest = {
        "campaign_id": collector.ROLLING_CAMPAIGN_ID,
        "hard_spec_sha256": collector.SPLITTEMP_HARD_SPEC_SHA256,
        "runner_source_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "campaign_total_seed_count": 512,
        "submissions": [
            {"seed": seed, "task_id": task_id}
            for seed, task_id in zip(seeds, task_ids)
        ],
    }

    spec = collector._source_spec(path, manifest)

    assert spec["replacement"] is True
    assert spec["seeds"] == seeds
    assert spec["task_ids"] == task_ids


def test_collector_parser_can_keep_watching_for_late_retry_manifests():
    args = collector._parser().parse_args(
        ["--watch", "--continue-after-failure"]
    )

    assert args.watch is True
    assert args.continue_after_failure is True


def test_splittemp_rows_are_not_relaxed_a_second_time():
    physical = {
        "temperature_robust_limit:T_max_Tx": 5.0,
        "temperature_robust_limit:T_max_Rx_main": -5.0,
        "temperature_robust_limit:T_max_Rx_side": -45.0,
        "temperature_robust_limit:Tprobe_Tx_leeward_max": 4.0,
        "temperature_robust_limit:Tprobe_Rx_main_leeward_max": -6.0,
        "temperature_robust_limit:Tprobe_Rx_side_leeward_max": -44.0,
        "temperature_robust_limit:T_max_core": -2.0,
    }
    normalized = {name: value / 10.0 for name, value in physical.items()}

    native_physical, native_normalized = (
        collector._aggregate_temperature_constraints(
            physical,
            normalized,
            "splittemp_native_100C_120C",
        )
    )
    legacy_physical, _legacy_normalized = (
        collector._aggregate_temperature_constraints(
            physical,
            normalized,
            "legacy_100C_reclassify_to_splittemp",
        )
    )

    assert native_physical == physical
    assert native_normalized == normalized
    assert native_physical is not physical
    assert (
        native_physical["temperature_robust_limit:T_max_Rx_main"]
        == -5.0
    )
    assert (
        legacy_physical["temperature_robust_limit:T_max_Rx_main"]
        == -25.0
    )


def test_global_nds_compares_rows_across_both_campaigns():
    rows = [
        {
            "source_campaign_id": collector.LEGACY_CAMPAIGN_ID,
            "objective_volume_L": 800.0,
            "objective_total_loss_W": 12_000.0,
            "physical_geometry_sha256": "a" * 64,
        },
        {
            "source_campaign_id": collector.SPLITTEMP_CAMPAIGN_ID,
            "objective_volume_L": 790.0,
            "objective_total_loss_W": 11_500.0,
            "physical_geometry_sha256": "b" * 64,
        },
        {
            "source_campaign_id": collector.SPLITTEMP_CAMPAIGN_ID,
            "objective_volume_L": 780.0,
            "objective_total_loss_W": 12_500.0,
            "physical_geometry_sha256": "c" * 64,
        },
    ]

    front = collector._non_dominated(
        rows, ("objective_volume_L", "objective_total_loss_W")
    )

    assert {
        row["physical_geometry_sha256"] for row in front
    } == {"b" * 64, "c" * 64}
