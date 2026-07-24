# MFT goal active-learning contingency

This contingency may be used only for authenticated, retained Standard FEA
truth collected before the 2026-07-26 18:00 KST deadline. It does not change
the 1.5 m/s dual-fan condition, the 2 mm pads, TIM/insulation conductivity
0.2 W/mK, operating point, solver convergence contract, or model quality
thresholds.

## Admission decision

The immutable base is:

- path:
  `C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet`
- SHA-256:
  `0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3`
- 6,151 strict-full rows and one
  `mft1mw-1k101-native-lamination-kf0p85-v3` physics-data cohort

Do not retrain until at least eight new complete, unique physical geometries
from at least four authenticated source NSGA tasks pass ingestion. Twelve new
rows are recommended if the FEA/license schedule permits. A targeted N1=6-only
batch is allowed: the current authenticated search has no non-Llt-feasible
rows in N1=5, 7, or 8, while the base already contains all four strata.
Accordingly, the new-data receipt must state `targeted_strata=[6]` and
`global_N1_coverage_claimed=false`. The subsequent NSGA campaign, unlike the
targeted FEA batch, must again cover N1=5, 6, 7, and 8.

Eight is a deadline contingency threshold, not a claim that model quality must
improve. The unchanged post-training quality gate remains the only model
quality decision.

Every new row must be trainable for all 25 targets:

```text
Llt_phys k
C_tx_tx_F C_rx_rx_F C_tx_rx_F
P_winding_total P_Tx_main_group P_Rx_main_group P_Rx_side_total
P_core_total P_core_plate_total P_wcp_total
B_max_core B_mean_core
Tprobe_Tx_leeward_max Tprobe_Rx_main_leeward_max
Tprobe_Rx_side_leeward_max
Tprobe_core_center_max Tprobe_core_center_leg_max
Tprobe_core_side_leg_max Tprobe_core_top_yoke_max
T_max_Tx T_max_Rx_main T_max_Rx_side T_max_core
```

One or more syntactically complete rows are not sufficient. The owning
production or diagnostic collection loader must reauthenticate the plan,
submission, source task, candidate geometry, raw result, retained AEDT
receipt/results manifest, and all payload/file SHAs. Raw JSON, CSV, or a task
ID is not an ingestion source.

## Build the isolated dataset

Run this only from a clean commit containing
`tools/mft_goal_strict_al_ingest.py` and the finalized diagnostic collection
adapter. `$collections` must contain only completed immutable collection JSON
files.

```powershell
$python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$codeRoot = '<clean-1MW_MFT-checkout>'
$base = 'C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet'
$alRoot = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\al-g1-<UTC-stamp>'
$collections = Get-ChildItem -LiteralPath '<diagnostic-collections>' -Filter '*.json'
$collectionArgs = @()
foreach ($collection in $collections) {
  $collectionArgs += '--collection'
  $collectionArgs += $collection.FullName
}

& $python "$codeRoot\tools\mft_goal_strict_al_ingest.py" build `
  --base-dataset $base `
  --expected-base-sha256 0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3 `
  --expected-base-rows 6151 `
  @collectionArgs `
  --minimum-useful-rows 8 `
  --minimum-source-tasks 4 `
  --require-retraining-ready `
  --output-dir "$alRoot\dataset"
if ($LASTEXITCODE -ne 0) { throw 'strict AL ingestion failed' }
```

The command creates a new `strict_al.parquet` and sealed `manifest.json`.
It recomputes strict EM and thermal validity, requires exact solver/library
provenance, checks the single physics cohort, verifies all 25 target filters,
and compares the base SHA before and after writing. It never writes the base
parquet or contacts Scheduler.

## Train exactly one new 25-target generation

Use a new registry and candidate path. Never add artifacts to
`g0b_25target_6151` or point a new campaign at its result ledger.

```powershell
$manifestPath = "$alRoot\dataset\manifest.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if (-not $manifest.retraining_admission.allowed) {
  throw 'AL manifest is not admitted for retraining'
}
$dataset = Join-Path (Split-Path $manifestPath) $manifest.output_dataset.path
$datasetGeneration = 'goal-al-' + $manifest.payload_sha256.Substring(0,16)
$trainRoot = "$alRoot\training"
New-Item -ItemType Directory -Path $trainRoot -ErrorAction Stop | Out-Null

$targets = @(
  'Llt_phys','k','C_tx_tx_F','C_rx_rx_F','C_tx_rx_F',
  'P_winding_total','P_Tx_main_group','P_Rx_main_group','P_Rx_side_total',
  'P_core_total','P_core_plate_total','P_wcp_total',
  'B_max_core','B_mean_core',
  'Tprobe_Tx_leeward_max','Tprobe_Rx_main_leeward_max',
  'Tprobe_Rx_side_leeward_max','Tprobe_core_center_max',
  'Tprobe_core_center_leg_max','Tprobe_core_side_leg_max',
  'Tprobe_core_top_yoke_max','T_max_Tx','T_max_Rx_main',
  'T_max_Rx_side','T_max_core'
)

& $python "$codeRoot\regression_260707\training\train_models.py" `
  --dataset $dataset `
  --source-dataset-path $dataset `
  --source-dataset-generation $datasetGeneration `
  --targets $targets `
  --registry "$trainRoot\registry" `
  --profile "$codeRoot\regression_260707\verify\profiles\goal_standard.json" `
  --model-threads 2 `
  --target-workers 4 `
  --max-model-thread-budget 8 `
  --result-json "$trainRoot\candidate.json"
if ($LASTEXITCODE -ne 0) { throw '25-target training failed' }

$candidate = Get-Content -Raw -LiteralPath "$trainRoot\candidate.json" |
  ConvertFrom-Json
$generation = $candidate.generation_path
```

The reference 6,151-row generation ran from 21:06:33 to 21:40:03 KST
(33 minutes 30 seconds) with four target workers and two model threads.
Reserve 45 minutes for the same eight-core budget plus the small row increase.
If training is placed on Slurm, request at least 8 CPUs and enough memory for
the four concurrent target families; do not alter the scheduler project code
or copy its functions into the 1MW MFT project.

## Run the unchanged quality gate

```powershell
& $python "$codeRoot\regression_260707\training\model_quality_gate.py" `
  --registry "$trainRoot\registry" `
  --generation $generation `
  --dataset $dataset `
  --thresholds "$codeRoot\regression_260707\training\model_quality_thresholds.json" `
  --status "$trainRoot\quality_status.json"
$qualityExit = $LASTEXITCODE
if (($qualityExit -ne 0) -and ($qualityExit -ne 2)) {
  throw 'model quality evaluation did not complete'
}
if (-not (Test-Path -LiteralPath "$trainRoot\quality_status.json")) {
  throw 'model quality status is absent'
}
```

Exit 2 means the unchanged quality thresholds failed. The generation may
still seed a sealed `search_only_proposal=true` campaign, but it is not
production-eligible and must not be automatically promoted.

## Create a new sealed NSGA-II campaign

The clean code revision, new dataset SHA, new train-report SHA, candidate SHA,
quality-status SHA, model inventory SHA, and profile SHA are rebound into
every task. Use a seed interval disjoint from the existing
2607262000..2607262511 campaign; 2607263000..2607263511 is reserved below.

```powershell
$dirty = git -C $codeRoot status --porcelain --untracked-files=all
if ($dirty) { throw 'code root is dirty' }
$codeRevision = (git -C $codeRoot rev-parse HEAD).Trim()
$campaignRoot = "$alRoot\campaign-rolling512"

& $python "$codeRoot\tools\mft_goal_20260726_launch.py" prepare `
  --generation $generation `
  --candidate "$trainRoot\candidate.json" `
  --quality-status "$trainRoot\quality_status.json" `
  --code-root $codeRoot `
  --expected-code-revision $codeRevision `
  --mode rolling `
  --seed-start 2607263000 `
  --seed-count 512 `
  --wave-size 32 `
  --output $campaignRoot
if ($LASTEXITCODE -ne 0) { throw 'new campaign sealing failed' }
```

The resulting scheduler manifest specifies 512 independent tasks, 8 CPUs and
64 GiB per task, round-robin N1=5/6/7/8, population 320, and 300 evaluated
generations. The separate Scheduler project should submit as much of this
sealed ledger concurrently as its live account and node limits permit. It
must not reuse the old campaign's task ledger, task IDs, result directory, or
aggregate output.

Aggregate only the 512 results authenticated against this exact new bundle:

```powershell
& $python "$codeRoot\tools\mft_goal_20260726_launch.py" aggregate `
  --results-root "$alRoot\runs" `
  --bundle-manifest "$campaignRoot\bundle_manifest.json" `
  --minimum-seeds 512 `
  --output "$alRoot\global-pareto"
```

This recomputes global non-dominated sorting over all authenticated new seed
terminal populations. Old-generation results cannot be mixed because their
dataset, model generation, bundle task ledger, and seed result identities do
not match the new bundle.

## Deadline allocation

- Standard FEA: submit the admitted diagnostic probes in parallel immediately;
  each retained Standard task is 8 CPUs, 32 GiB, and a 4-hour timeout.
- Dataset authentication/build: reserve 5 minutes after at least 8 complete
  rows (12 recommended) have arrived.
- New 25-target training: reserve 45 minutes; the measured baseline is 33:30.
- Quality gate and sealed campaign preparation: reserve 15 minutes.
- NSGA-II: release the new four-stratum canaries, then use all available
  Scheduler slots for the remaining sealed tasks.
- Preserve enough time before 2026-07-26 18:00 KST to authenticate all seed
  results, recompute global NDS, and perform retained Standard/Full FEA for the
  final deliverables. Do not spend the final window on a sub-eight-row retrain.
