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
prevents unrelated JSON files from entering the command. Eleven timeout
retries replace, rather than augment, their failed original logical slots.

```powershell
$Py = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$CodeRoot = 'C:\w\mft-goal-20260726'
$Base = 'C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet'
$CollectionAllowList = @(
  [pscustomobject]@{
    Root = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\diagnostic_standard_retry12_sa1e4f70cefa1_q0800a8d204da_rde0b963f42d2_260725'
    Stems = @(
      '068b0607b43f','11add38e114d','11e0d8daed35',
      '628c9fcfec1e','692c1a03e5fd','895990953c5a',
      'a4ae16f8a0c3','ab33ef0ba3ef','b6a83bfc7212',
      'bc50d459bcb4','c3195a79f5b2','cb3d96a527ac'
    )
  }
  [pscustomobject]@{
    Root = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\diagnostic_secondary_n1_6_excluded11_260725'
    Stems = @(
      '05580bda40b2','08750eb352cf','2347a292ad75',
      '2a1bb6f2be79','436565e3f360','7a6ccac265d3',
      '7a8c0bd079b1','7ce2bf976d48','90598193e992',
      'b4075d84aeee','b7c30cb70b95','efffb6518d4e'
    )
  }
  [pscustomobject]@{
    Root = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\diagnostic_expansion24_n1_6_excluded22_260725'
    Stems = @(
      '0db6751640ed','1a37a5bdf570','2fbcce18032d',
      '30801ea43777','394982e87267','48c66215f6b1',
      '52c0532d449b','5c81f854d1f6','5da1e899c646',
      '65063da2abd2','85c1d3f685ee','a2aee0d6368f',
      'a5faf61b52ec','aaa0a68f092f','b7140eaa3093',
      'c6e25efd5cb3','c9ce907e92e2','e08397ffe3ba',
      'e25be28b3941','e9a5f5d82946','edd23c7754e9',
      'f87bf164246c','f9da67b59152','fba86bc49ff0'
    )
  }
)
$TimeoutRetryRoot = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\standard_timeout_retries_r1_260725'
$TimeoutRetryAllowList = @(
  [pscustomobject]@{
    Stem = '11e0d8daed35'; OriginalTaskId = 96218
    RetryTaskId = 96256; Directory = 't96218_11e0d8daed35'
  }
  [pscustomobject]@{
    Stem = '068b0607b43f'; OriginalTaskId = 96219
    RetryTaskId = 96257; Directory = 't96219_068b0607b43f'
  }
  [pscustomobject]@{
    Stem = '436565e3f360'; OriginalTaskId = 96224
    RetryTaskId = 96258; Directory = 't96224_436565e3f360'
  }
  [pscustomobject]@{
    Stem = '08750eb352cf'; OriginalTaskId = 96221
    RetryTaskId = 96259; Directory = 't96221_08750eb352cf'
  }
  [pscustomobject]@{
    Stem = '2a1bb6f2be79'; OriginalTaskId = 96223
    RetryTaskId = 96260; Directory = 't96223_2a1bb6f2be79'
  }
  [pscustomobject]@{
    Stem = '2347a292ad75'; OriginalTaskId = 96222
    RetryTaskId = 96261; Directory = 't96222_2347a292ad75'
  }
  [pscustomobject]@{
    Stem = '05580bda40b2'; OriginalTaskId = 96220
    RetryTaskId = 96262; Directory = 't96220_05580bda40b2'
  }
  [pscustomobject]@{
    Stem = '7a8c0bd079b1'; OriginalTaskId = 96226
    RetryTaskId = 96263; Directory = 't96226_7a8c0bd079b1'
  }
  [pscustomobject]@{
    Stem = '7a6ccac265d3'; OriginalTaskId = 96225
    RetryTaskId = 96264; Directory = 't96225_7a6ccac265d3'
  }
  [pscustomobject]@{
    Stem = 'b7c30cb70b95'; OriginalTaskId = 96230
    RetryTaskId = 96265; Directory = 't96230_b7c30cb70b95'
  }
  [pscustomobject]@{
    Stem = 'b4075d84aeee'; OriginalTaskId = 96229
    RetryTaskId = 96266; Directory = 't96229_b4075d84aeee'
  }
)

$Dirty = git -C $CodeRoot status --porcelain --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $Dirty) {
  throw 'AL code root must be an exact clean checkout'
}
$CodeRevision = (git -C $CodeRoot rev-parse HEAD).Trim()
if ($CodeRevision.Length -ne 40) { throw 'invalid code revision' }

$RetryByStem = @{}
foreach ($Spec in $TimeoutRetryAllowList) {
  if ($RetryByStem.ContainsKey($Spec.Stem)) {
    throw "duplicate timeout-retry logical slot: $($Spec.Stem)"
  }
  $RetryByStem.Add($Spec.Stem, $Spec)
}
$OriginalSubmissionByStem = @{}
$Collections = @(
  foreach ($Group in $CollectionAllowList) {
    foreach ($Stem in $Group.Stems) {
      $Submission = Join-Path $Group.Root "submissions\$Stem.json"
      if (-not (Test-Path -LiteralPath $Submission -PathType Leaf)) {
        throw "allow-listed Standard submission is absent: $Submission"
      }
      $Receipt = Get-Content -Raw -LiteralPath $Submission | ConvertFrom-Json
      if (
        $Receipt.schema_version -ne 'mft-goal-diagnostic-standard-submission-v1' -or
        $Receipt.stage -ne 'standard' -or
        $Receipt.scheduler_submission_performed -ne $true -or
        [int]$Receipt.task_id -le 0 -or
        -not ([string]$Receipt.candidate_physics_sha256).StartsWith($Stem)
      ) {
        throw "allow-listed Standard submission identity drifted: $Submission"
      }
      if ($OriginalSubmissionByStem.ContainsKey($Stem)) {
        throw "duplicate original Standard logical slot: $Stem"
      }
      $OriginalSubmissionByStem.Add($Stem, $Submission)
      if ($RetryByStem.ContainsKey($Stem)) {
        $Spec = $RetryByStem[$Stem]
        if ([int]$Receipt.task_id -ne [int]$Spec.OriginalTaskId) {
          throw "timeout-retry original task identity drifted: $Submission"
        }
      } else {
        $Path = Join-Path $Group.Root "collections\$Stem.json"
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
          (Resolve-Path -LiteralPath $Path).Path
        }
      }
    }
  }
  foreach ($Spec in $TimeoutRetryAllowList) {
    $Root = Join-Path $TimeoutRetryRoot $Spec.Directory
    $Submission = Join-Path $Root 'submission.json'
    $Plan = Join-Path $Root 'plan\diagnostic_timeout_retry_plan.json'
    $OriginalSubmission = $OriginalSubmissionByStem[$Spec.Stem]
    if (
      -not $OriginalSubmission -or
      -not (Test-Path -LiteralPath $Submission -PathType Leaf) -or
      -not (Test-Path -LiteralPath $Plan -PathType Leaf)
    ) {
      throw "allow-listed timeout-retry ancestry is absent: $Root"
    }
    $Receipt = Get-Content -Raw -LiteralPath $Submission | ConvertFrom-Json
    $Retry = $Receipt.retry_of_timeout
    if (
      $Receipt.schema_version -ne 'mft-goal-diagnostic-standard-submission-v1' -or
      $Receipt.stage -ne 'standard' -or
      $Receipt.scheduler_submission_performed -ne $true -or
      [int]$Receipt.task_id -ne [int]$Spec.RetryTaskId -or
      -not ([string]$Receipt.candidate_physics_sha256).StartsWith($Spec.Stem) -or
      [string]$Receipt.plan.path -ne $Plan -or
      $Retry.schema_version -ne 'mft-goal-diagnostic-standard-timeout-retry-evidence-v1' -or
      [int]$Retry.retry_of_task_id -ne [int]$Spec.OriginalTaskId -or
      [string]$Retry.original_submission.path -ne $OriginalSubmission -or
      [int]$Retry.original_task_execution.task_id -ne [int]$Spec.OriginalTaskId -or
      [string]$Retry.original_task_execution.status -ne 'failed' -or
      [int]$Retry.original_task_execution.exit_code -ne 124
    ) {
      throw "allow-listed timeout-retry identity drifted: $Submission"
    }
    $Path = Join-Path $Root 'collection.json'
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
      (Resolve-Path -LiteralPath $Path).Path
    }
  }
)
if ($Collections.Count -ne @($Collections | Sort-Object -Unique).Count) {
  throw 'strict AL logical-slot selection produced duplicate collection paths'
}
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
writes the base dataset or contacts Scheduler. A superseded failed original
collection path is never passed to ingestion. For a retry, the diagnostic
collection authenticator follows the sealed plan back to the exact original
plan, submission, and failed/124 Scheduler evidence, and verifies that the
effective physics is unchanged. Strict ingestion also rejects duplicate
candidate geometries, so an original and its retry cannot be counted twice.

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
