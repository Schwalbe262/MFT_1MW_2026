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

## Exact trigger-to-training commands

Use a clean detached scientific checkout because the immutable dataset
manifest and the training deployment must bind the same Git revision. Do not
remove unrelated files from the integration worktree to make it look clean.
The prepared clean checkout is:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\al_launch_clean`

The following path performs no Scheduler POST until the authenticated
corrected manifest reports `allowed=true`. A snapshot with fewer than eight
valid geometries remains immutable evidence and must not be reused as if it
had passed.

```powershell
$Py = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$Main = 'C:\w\mft-goal-20260726'
$CodeRoot = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\al_launch_clean'
$Runtime = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726'
$Base = 'C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet'
$BaseSha = '0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3'
$Receipt = "$Runtime\clean_library_thermal24_replay_v1\submission_receipt.json"
$Replay = "$Runtime\clean_library_thermal24_replay_v1\replay_plan.json"
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$ALRoot = "$Runtime\al-clean24-$Stamp"

$CodeRevision = (git -C $Main rev-parse HEAD).Trim()
git -C $CodeRoot fetch origin
if ($LASTEXITCODE -ne 0) { throw 'clean AL checkout fetch failed' }
git -C $CodeRoot checkout --detach $CodeRevision
if ($LASTEXITCODE -ne 0) { throw 'clean AL checkout update failed' }
$Dirty = git -C $CodeRoot status --porcelain --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $Dirty) {
  throw 'clean AL checkout is not exact and clean'
}

Push-Location $CodeRoot
try {
  & $Py tools/mft_goal_corrected_replacement_strict_collect.py `
    --receipt $Receipt `
    --replay-plan $Replay `
    --scheduler-url http://127.0.0.1:8002 `
    --output-dir "$ALRoot\strict-truth"
  if ($LASTEXITCODE -ne 0) { throw 'corrected truth collection failed' }

  $TruthManifest = "$ALRoot\strict-truth\manifest.json"
  $Truth = Get-Content -Raw -LiteralPath $TruthManifest | ConvertFrom-Json
  if (-not $Truth.retraining_trigger.allowed) {
    throw "strict trigger closed: $($Truth.retraining_trigger.reasons -join ',')"
  }

  & $Py -m tools.mft_goal_strict_al_ingest inspect `
    --base-dataset $Base `
    --expected-base-sha256 $BaseSha `
    --expected-base-rows 6151 `
    --collection-manifest $TruthManifest `
    --minimum-useful-rows 8 `
    --minimum-source-tasks 4 `
    --require-retraining-ready
  if ($LASTEXITCODE -ne 0) { throw 'strict AL inspection failed' }

  & $Py -m tools.mft_goal_strict_al_ingest build `
    --base-dataset $Base `
    --expected-base-sha256 $BaseSha `
    --expected-base-rows 6151 `
    --collection-manifest $TruthManifest `
    --minimum-useful-rows 8 `
    --minimum-source-tasks 4 `
    --require-retraining-ready `
    --output-dir "$ALRoot\dataset"
  if ($LASTEXITCODE -ne 0) { throw 'strict AL dataset build failed' }

  & $Py -m tools.mft_goal_al_slurm_train plan `
    --dataset-manifest "$ALRoot\dataset\manifest.json" `
    --code-root $CodeRoot `
    --expected-code-revision $CodeRevision `
    --local-root "$ALRoot\training-offload" `
    --remote-root /gpfs/tmp_cpu2/mft_goal_20260726/al_training
  if ($LASTEXITCODE -ne 0) { throw '25-target training plan failed' }

  $Plans = @(
    Get-ChildItem -LiteralPath "$ALRoot\training-offload" `
      -Recurse -Filter offload_plan.json
  )
  if ($Plans.Count -ne 1) { throw 'exactly one training plan required' }
  $TrainPlan = $Plans[0].FullName

  & $Py -m tools.mft_goal_al_slurm_train stage --plan $TrainPlan
  if ($LASTEXITCODE -ne 0) { throw 'training stage dry-run failed' }
  & $Py -m tools.mft_goal_al_slurm_train stage `
    --plan $TrainPlan --apply
  if ($LASTEXITCODE -ne 0) { throw 'training stage apply failed' }

  & $Py -m tools.mft_goal_al_slurm_train submit --plan $TrainPlan
  if ($LASTEXITCODE -ne 0) { throw 'training submit dry-run failed' }
  & $Py -m tools.mft_goal_al_slurm_train submit `
    --plan $TrainPlan --priority 100 --apply
  if ($LASTEXITCODE -ne 0) { throw 'training submit failed' }
} finally {
  Pop-Location
}
```

The pinned historical base predates eight append-only fixed-run controls:
`core_center_gap_mm` and the seven turn-graded electrostatic controls. Strict
ingest records a sealed in-memory normalization to their historical defaults
(`0 mm` gap and turn-graded mode `off`) without rewriting the canonical
parquet. Any missing Sobol/design input still fails closed.

After the single training task completes, continue with sections 3 through 5
of `MFT_GOAL_AL_SLURM_EXECUTION.md`: authenticate the full generation, prepare
the exact `2607263000..2607263511` bundle, submit four N1=5/6/7/8 canaries,
release all 16 remaining waves, harvest all 512 populations, and run global
non-dominated sorting. Quality-gate exit 2 is retained only as an explicitly
labelled search-only generation; thresholds are never relaxed.

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

The same aggregate publishes
`compactness_acquisition_candidates.csv` for the smaller-design question.
Its fixed domain is:

- no axis swap;
- `W <= 1170 mm` **or** `L <= 975 mm`;
- volume strictly below the current 97048 reference,
  `830.95994977 L` (`1181.040 x 992.360 x 709 mm`);
- the original `W/L/H`, temperature, resonance, cooling, operating-point,
  and all other hard constraints still pass;
- only hard-feasible compact front-0 rows are selected, up to 12;
- no near-feasible or scientifically invalid thermal fallback.

The compact FEA-acquisition trigger opens only when all reserved 512 seeds
authenticate, the new model quality gate passes, and at least one row satisfies
that complete compact hard-feasible domain. Even then the artifact is a
selection receipt, not production proof: symmetric non-rounded FEA and final
turn-graded capacitance remain required. The aggregate itself never submits a
Scheduler task.

For reference, the old-generation 2,849-row screening table contains 1,454
rows in the `W <= 1170 or L <= 975` geometric slice, but zero are hard
feasible under that generation. They are audit evidence only and are not
reused as thermal truth or compact FEA submissions.

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
