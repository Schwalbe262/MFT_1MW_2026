# MFT goal AL Slurm execution

This is the fail-closed execution path for the active-learning contingency in
`MFT_GOAL_ACTIVE_LEARNING_CONTINGENCY_20260726.md`. It is inactive until at
least eight authenticated diagnostic Standard collections pass strict
ingestion. It does not change fan velocity, TIM/pads, the operating point,
quality thresholds, or any Scheduler project code.

## Fixed identities and gates

- Base dataset:
  `C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet`
- Base SHA-256:
  `0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3`
- Base rows: `6151`
- New rows: at least 8; 12 preferred
- Unique new geometries: at least 8
- Authenticated source NSGA tasks: at least 4
- Targeted diagnostic stratum: exactly `N1=6`
- Training targets: exactly 25, with canonical target-list SHA-256
  `6b78ef04ab8cf2792bd3d7f8c6381ab772c8b161143506bec6f2b4de58df29d8`
- Training: exactly one Slurm task, 8 CPUs, four target workers, two model
  threads per worker, total model-thread budget 8
- Quality thresholds: unchanged
- Next search seeds: exactly `2607263000..2607263511`
- Search: 512 tasks, N1 round-robin 5/6/7/8, population 320, 300 evaluated
  generations

The clean code revision used to build the strict AL dataset must be the same
revision used to plan and execute training. Build the dataset only after the
AL Slurm wrapper and relocation support are present in the clean integration
checkout.

## 1. Authenticate and build the isolated dataset

Run only after at least eight collection files exist. The allow-list below
prevents unrelated JSON files from entering the command.

```powershell
$Py = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$CodeRoot = 'C:\w\mft-goal-20260726'
$Diag = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\diagnostic_standard_retry12_sa1e4f70cefa1_q0800a8d204da_rde0b963f42d2_260725'
$Base = 'C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet'
$Stems = @(
  '068b0607b43f','11add38e114d','11e0d8daed35',
  '628c9fcfec1e','692c1a03e5fd','895990953c5a',
  'a4ae16f8a0c3','ab33ef0ba3ef','b6a83bfc7212',
  'bc50d459bcb4','c3195a79f5b2','cb3d96a527ac'
)

$Dirty = git -C $CodeRoot status --porcelain --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $Dirty) {
  throw 'AL code root must be an exact clean checkout'
}
$CodeRevision = (git -C $CodeRoot rev-parse HEAD).Trim()
if ($CodeRevision.Length -ne 40) { throw 'invalid code revision' }

$Collections = @(
  foreach ($Stem in $Stems) {
    $Path = Join-Path $Diag "collections\$Stem.json"
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
      (Resolve-Path -LiteralPath $Path).Path
    }
  }
)
if ($Collections.Count -lt 8) {
  throw "strict AL requires at least eight collections; got $($Collections.Count)"
}
$CollectionArgs = @()
foreach ($Path in $Collections) {
  $CollectionArgs += '--collection'
  $CollectionArgs += $Path
}

$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$ALRoot = "C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\al-g1-$Stamp"

& $Py -m tools.mft_goal_strict_al_ingest inspect `
  --base-dataset $Base `
  --expected-base-sha256 0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3 `
  --expected-base-rows 6151 `
  @CollectionArgs `
  --minimum-useful-rows 8 `
  --minimum-source-tasks 4 `
  --require-retraining-ready
if ($LASTEXITCODE -ne 0) { throw 'strict AL inspection failed' }

& $Py -m tools.mft_goal_strict_al_ingest build `
  --base-dataset $Base `
  --expected-base-sha256 0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3 `
  --expected-base-rows 6151 `
  @CollectionArgs `
  --minimum-useful-rows 8 `
  --minimum-source-tasks 4 `
  --require-retraining-ready `
  --output-dir "$ALRoot\dataset"
if ($LASTEXITCODE -ne 0) { throw 'strict AL build failed' }
```

`build` creates a new `strict_al.parquet` and sealed `manifest.json`; it never
writes the base dataset or contacts Scheduler.

## 2. Plan, stage, and submit one 8-CPU training task

`plan` is local-only. `stage` and `submit` are dry-runs unless `--apply` is
present. The payload binds the admitted dataset manifest, dataset SHA,
25-target inventory, clean code revision, profile, quality thresholds,
resource contract, and next seed interval.

```powershell
$DatasetManifest = "$ALRoot\dataset\manifest.json"
$TrainPlanRoot = "$ALRoot\training-offload"

& $Py -m tools.mft_goal_al_slurm_train plan `
  --dataset-manifest $DatasetManifest `
  --code-root $CodeRoot `
  --expected-code-revision $CodeRevision `
  --local-root $TrainPlanRoot `
  --remote-root /gpfs/tmp_cpu2/mft_goal_20260726/al_training
if ($LASTEXITCODE -ne 0) { throw 'AL training plan failed' }

$Plans = @(Get-ChildItem -LiteralPath $TrainPlanRoot -Recurse -Filter offload_plan.json)
if ($Plans.Count -ne 1) { throw 'exactly one AL training plan is required' }
$TrainPlan = $Plans[0].FullName

# Read-only dry-run, then the explicit immutable publication.
& $Py -m tools.mft_goal_al_slurm_train stage --plan $TrainPlan
if ($LASTEXITCODE -ne 0) { throw 'AL training stage dry-run failed' }
& $Py -m tools.mft_goal_al_slurm_train stage --plan $TrainPlan --apply
if ($LASTEXITCODE -ne 0) { throw 'AL training stage failed' }

# Read-only dry-run, then exactly one Scheduler POST.
& $Py -m tools.mft_goal_al_slurm_train submit --plan $TrainPlan
if ($LASTEXITCODE -ne 0) { throw 'AL training submit dry-run failed' }
& $Py -m tools.mft_goal_al_slurm_train submit `
  --plan $TrainPlan `
  --priority 100 `
  --apply
if ($LASTEXITCODE -ne 0) { throw 'AL training submit failed' }

$Submission = Join-Path (Split-Path $TrainPlan) 'submission.json'
```

The Scheduler request is `standard`, `cpus=8`, `memory_mb=65536`,
`gpus=0`, `timeout_seconds=7200`, `max_workers_per_node=1`,
`required_capability=conda:pyaedt2026v1`. The worker additionally requires
both `SLURM_CPUS_PER_TASK=8` and `SLURM_SCHEDULER_TASK_CPUS=8`.

The task runs `train_models.py` once. It then runs the unchanged quality gate.
Quality exit `0` means passed; quality exit `2` is a completed, authenticated
search-only generation. Any other exit fails the task.

## 3. Collect and authenticate the generation

Check task state through the read-only Scheduler endpoint. Do not call
`collect` until it is `completed` with `exit_code=0`.

```powershell
$SubmissionJson = Get-Content -Raw -LiteralPath $Submission | ConvertFrom-Json
$TaskId = [int]$SubmissionJson.task_id
$Task = Invoke-RestMethod -Uri "http://127.0.0.1:8002/api/tasks/$TaskId"
$Task | Select-Object id,status,exit_code,account_name,cpus,memory_mb
if ($Task.status -ne 'completed' -or [int]$Task.exit_code -ne 0) {
  throw 'AL training task is not ready for collection'
}

$TrainCollection = "$ALRoot\training-collection"
& $Py -m tools.mft_goal_al_slurm_train collect `
  --plan $TrainPlan `
  --submission $Submission `
  --scheduler-url http://127.0.0.1:8002 `
  --output $TrainCollection
if ($LASTEXITCODE -ne 0) { throw 'AL training collection failed' }

& $Py -m tools.mft_goal_al_slurm_train validate-collection `
  --collection "$TrainCollection\collection_manifest.json"
if ($LASTEXITCODE -ne 0) { throw 'AL training collection validation failed' }
```

Collection is Scheduler-read-only and uses the existing verified transport:
remote SHA before download, remote SHA after download, local SHA, and byte
count must all agree. Candidate, quality status, train report, and all 50 model
artifacts are collected. An interrupted transfer can resume only when its
bundle, task payload, task ID, and remote output identity are unchanged.

## 4. Prepare and launch the disjoint 512-seed search

The remotely trained candidate retains its original GPFS generation path as
documentary provenance. The three relocation arguments below preserve that
string while reauthenticating the local collected generation, dataset, and
profile by exact content identity.

```powershell
$TrainResult = Get-Content -Raw -LiteralPath "$TrainCollection\result.json" |
  ConvertFrom-Json
$Generation = Join-Path $TrainCollection $TrainResult.generation_relative
$Candidate = "$TrainCollection\candidate.json"
$Quality = "$TrainCollection\quality_status.json"
$Dataset = "$TrainCollection\inputs\strict_al.parquet"
$Profile = "$TrainCollection\inputs\goal_standard.json"
$DocumentaryGeneration = [string]$TrainResult.documentary_generation_path
$Campaign = "$ALRoot\campaign-rolling512"

& $Py -m tools.mft_goal_20260726_launch prepare `
  --generation $Generation `
  --candidate $Candidate `
  --quality-status $Quality `
  --code-root $CodeRoot `
  --expected-code-revision $CodeRevision `
  --runtime-dataset $Dataset `
  --runtime-profile $Profile `
  --expected-documentary-generation-path $DocumentaryGeneration `
  --mode rolling `
  --seed-start 2607263000 `
  --seed-count 512 `
  --wave-size 32 `
  --output $Campaign
if ($LASTEXITCODE -ne 0) { throw 'AL 512-seed campaign sealing failed' }

$SearchPlanRoot = "$ALRoot\search-offload"
& $Py -m tools.mft_goal_20260726_slurm plan `
  --goal-bundle-root $Campaign `
  --generation $Generation `
  --candidate $Candidate `
  --quality-status $Quality `
  --dataset $Dataset `
  --profile $Profile `
  --local-root $SearchPlanRoot `
  --remote-root /gpfs/tmp_cpu2/mft_goal_20260726/al_search
if ($LASTEXITCODE -ne 0) { throw 'AL search offload plan failed' }

$SearchPlans = @(Get-ChildItem -LiteralPath $SearchPlanRoot -Recurse -Filter offload_plan.json)
if ($SearchPlans.Count -ne 1) { throw 'exactly one AL search plan is required' }
$SearchPlan = $SearchPlans[0].FullName

& $Py -m tools.mft_goal_20260726_slurm stage --plan $SearchPlan
if ($LASTEXITCODE -ne 0) { throw 'AL search stage dry-run failed' }
& $Py -m tools.mft_goal_20260726_slurm stage --plan $SearchPlan --apply
if ($LASTEXITCODE -ne 0) { throw 'AL search stage failed' }

& $Py -m tools.mft_goal_20260726_slurm submit `
  --plan $SearchPlan `
  --phase canary
if ($LASTEXITCODE -ne 0) { throw 'canary submit dry-run failed' }
& $Py -m tools.mft_goal_20260726_slurm submit `
  --plan $SearchPlan `
  --phase canary `
  --apply
if ($LASTEXITCODE -ne 0) { throw 'canary submit failed' }
```

The four canaries are N1=5,6,7,8. Release all remaining sealed waves only
after all four complete with exit code zero. This loop submits every remaining
task quickly so Scheduler can use all available allocation/node capacity.

```powershell
$SearchPlanJson = Get-Content -Raw -LiteralPath $SearchPlan | ConvertFrom-Json
$SearchLedger = Join-Path $SearchPlanJson.local_plan_dir 'submissions.json'
$CanaryLedger = Get-Content -Raw -LiteralPath $SearchLedger | ConvertFrom-Json
foreach ($Entry in $CanaryLedger.submissions) {
  $State = Invoke-RestMethod -Uri "http://127.0.0.1:8002/api/tasks/$($Entry.task_id)"
  if ($State.status -ne 'completed' -or [int]$State.exit_code -ne 0) {
    throw "canary $($Entry.task_id) is not a successful terminal task"
  }
}

$SchedulerManifest = Get-Content -Raw -LiteralPath "$Campaign\scheduler_manifest.json" |
  ConvertFrom-Json
$WaveIds = @(
  $SchedulerManifest.rolling_waves.PSObject.Properties.Name |
    ForEach-Object { [int]$_ } |
    Sort-Object
)
foreach ($Wave in $WaveIds) {
  & $Py -m tools.mft_goal_20260726_slurm submit `
    --plan $SearchPlan `
    --phase wave `
    --wave $Wave `
    --apply
  if ($LASTEXITCODE -ne 0) { throw "wave $Wave submission failed" }
}
```

## 5. Harvest and recompute the new global Pareto front

Use one bounded read-only harvest poll at a time. Final aggregation requires
all 512 authenticated terminal populations from this new bundle.

```powershell
& $Py -m tools.mft_goal_20260726_slurm harvest `
  --plan $SearchPlan `
  --scheduler-url http://127.0.0.1:8002 `
  --max-workers 4 `
  --max-polls 1 `
  --require-ready
if ($LASTEXITCODE -notin @(0,2)) { throw 'AL search harvest failed' }
if ($LASTEXITCODE -eq 2) { throw 'AL search is not yet complete' }

$SearchPlanJson = Get-Content -Raw -LiteralPath $SearchPlan | ConvertFrom-Json
$Results = Join-Path $SearchPlanJson.local_plan_dir 'harvest'
$Pareto = "$ALRoot\global-pareto"
& $Py -m tools.mft_goal_20260726_launch aggregate `
  --results-root $Results `
  --bundle-manifest "$Campaign\bundle_manifest.json" `
  --minimum-seeds 512 `
  --output $Pareto
if ($LASTEXITCODE -ne 0) { throw 'AL global non-dominated sorting failed' }
```

Do not mix the old `2607262000..2607262511` task ledger, run directories,
results, or aggregate with this generation. A failed quality gate remains
search-only and grants no automatic production or FEA promotion authority.

## Time reserve

- Strict ingestion/build: 5–10 minutes
- Immutable training stage: 10–30 minutes
- 25-target training and quality gate: 40–60 minutes
- Model collection/authentication (about 20 GiB): 15–45 minutes
- Search preparation/staging: 20–45 minutes
- Four canaries: about 35–45 minutes
- Remaining 508 tasks at prior observed capacity: about 2–2.5 hours
- Harvest and global NDS: 20–40 minutes

Reserve additional time for Standard/Full FEA and the final AEDT package.
