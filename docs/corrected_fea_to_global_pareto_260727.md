# Corrected FEA to global Pareto handoff

Status: prepared, terminal corrected FEA truth not yet available.

This handoff keeps three scientific authorities separate:

1. The existing surrogate search is a screening authority.
2. Corrected eighth-symmetry, non-rounded FEA is measured acquisition truth.
3. Final gapped symmetric FEA and actual turn-graded capacitance are final
   design truth.

No result may move between those authorities merely because its Scheduler task
completed successfully.

## Current screening result

The independently audited aggregate is:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\fixed_lm2mh_old16_plus_splittemp512_global_nds_v3`

- logical seeds: 528 (`16` legacy plus `512` fresh);
- terminal rows pooled before sorting: `168,960`;
- geometry-deduplicated rows: `2,849`;
- hard-feasible rows: `0`;
- production Pareto rows: `0`;
- minimum-violation diagnostic front: `73`;
- FEA acquisition set: `12`.

The zero count is a surrogate-screening result, not proof of physical
infeasibility.

## Corrected acquisition inventory

The only corrected replacement authority is:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\clean_library_thermal24_replay_v1`

Its sealed replay plan and submission receipt bind clean-library tasks
`97116..97139` to 24 unique physical geometries. Tasks `97042..97065` are
source-lineage identities only: their results and the old
`collection_live_detailed_v1.json` are forbidden training inputs. Task `97041`
is the shared-interface canary and must not be added as a 25th training row.
The cancelled tasks `96743`, `97014..97029`, and `97033..97040` are diagnostic
failure evidence only and must never enter a dataset, truth front, or
model-quality calculation.

Each admitted corrected result must satisfy all of the following:

- Scheduler state `completed` and `exit_code=0`;
- exact corrected task name, dedupe key, project, 8 CPU, 65,536 MiB,
  standalone backend, and 43,200-second timeout readback;
- exact source geometry and effective parameter hashes from the replacement
  plan;
- fixed cooling identity: dual fan at 1.5 m/s, both pads 2 mm, and
  `k_ins=0.2` / TIM conductivity 0.2 W/(m*K);
- eighth symmetry, non-rounded winding, full Matrix/cap/loss/thermal result;
- all six terminal scientific fields present:
  `thermal_rx_block_interface_contract_version`,
  `thermal_rx_main_interface_coverage_passed`,
  `thermal_rx_main_unpaired_interfaces`,
  `thermal_temperature_limiter_triggered`,
  `thermal_temperature_limiter_max_K`, and
  `thermal_result_scientific_valid`;
- interface contract version
  `thermal-rx-block-interface-coverage-v1`, coverage `true`, no unpaired
  interfaces, limiter not triggered, limiter maximum below 4,990 K, and
  scientific-valid `true`;
- no 5,000 K limiter or unpaired-interface marker in the authenticated task
  logs;
- every input and all 25 surrogate targets finite and trainable under the
  unchanged goal Standard quality profile.

Thermally infeasible but scientifically valid rows are useful training data.
Scientifically invalid rows are not.

Collect an immutable GET-only snapshot with:

```powershell
python tools/mft_goal_corrected_replacement_strict_collect.py `
  --receipt C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\clean_library_thermal24_replay_v1\submission_receipt.json `
  --replay-plan C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\clean_library_thermal24_replay_v1\replay_plan.json `
  --output-dir <new-immutable-output-directory>
```

The resulting `manifest.json` contains only scientific-valid unique
collections. Supply that aggregate seal directly to strict ingest with
`--collection-manifest <path-to-manifest.json>`. Do not combine this argument
with direct `--collection` inputs.

## Retraining trigger

Do not retrain below 8 unique, fully authenticated N1=6 rows from at least four
source tasks. Twelve rows are recommended. At the trigger:

1. Build a new isolated derived dataset with
   `tools.mft_goal_strict_al_ingest`; never mutate the canonical 6,151-row
   parquet.
2. Confirm every new row increases eligibility for each of the 25 targets and
   that the strict admission report is `allowed=true`.
3. Train one new 25-target generation with
   `tools.mft_goal_al_slurm_train`; retain the unchanged quality thresholds.
4. Whether the quality gate passes or returns search-only, bind the exact new
   dataset SHA, model inventory SHA, train-report SHA, and quality-status SHA
   into the next search bundle.

The canonical base has no terminal Rx-interface scientific fields. It is
therefore historical training evidence, not evidence for the newly discovered
interface contract. Corrected rows supply the local acquisition correction;
final feasibility still requires corrected FEA.

## Fresh NSGA-II and global NDS

The retrained search must use only the reserved disjoint seeds
`2607263000..2607263511`, round-robin N1=5/6/7/8, population 320, and the
single new model generation. Release all independent Slurm waves after four
stratum canaries pass.

After all 512 terminal populations authenticate, run
`tools.mft_goal_20260726_launch aggregate` on that bundle alone. The aggregate
must pool all `512 * 320 = 163,840` terminal rows before geometry
deduplication and non-dominated sorting. Do not union seed-local fronts and do
not mix the existing generation's rows.

The production Pareto front is the rank-0 set on volume and total loss after
all authoritative constraints pass:

- `W <= 1200 mm`, `L <= 1000 mm`, `H <= 750 mm`, without axis swap;
- physical Lm=2 mH with N2/N1=10 and minimum Tx/Rx resonance >=15 kHz;
- primary winding <=100 C, secondary winding <=120 C, core <=120 C;
- unchanged fixed cooling identity.

If the fresh surrogate Pareto is still empty, publish an empty production
front plus a separately labelled minimum-violation diagnostic front and
continue bounded FEA acquisition. Never rename a diagnostic front as Pareto
feasibility.
