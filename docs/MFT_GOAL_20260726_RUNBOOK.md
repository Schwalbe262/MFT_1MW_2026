# MFT 2026-07-26 goal campaign runbook

이 경로는 기존 `current7` stage/release 및 기존 Slurm bundle identity를
재사용하지 않는다. `1MW_MFT` 저장소는 모델·물리 계약과 실행 payload만
만들고, 별도 Scheduler 프로젝트는 생성된 `scheduler_manifest.json`과 task
payload를 읽어 제출만 담당한다.

## 고정 실행 계약

- population 320, fixed 300 evaluated generations, fixed-n-gen termination
- Pymoo의 종료 후 `algorithm.n_gen` 카운터는 301이다. 결과 계약은
  `evaluated_generations=300`, `completed_generations=301`을 함께 요구하며,
  둘 중 하나라도 다르면 거부한다.
- N1 strata 5, 6, 7, 8
- `cw1` 1.00–10.00 mm, 0.01 mm grid
- 25-target G0 generation, 24 models consumed by NSGA
- body maxima와 probe maxima를 모두 제약:
  winding 100 °C, core 120 °C
- W/L/H = 1200/1000/750 mm, self resonance >= 15 kHz
- 1.5 m/s 및 고정 TIM/cooling/operating identity 유지
- seed 결과마다 terminal 320개 전부를
  `terminal_physical_candidates.csv`로 보존

## 1. G0와 단일 seed bundle 준비

모든 경로는 절대 경로를 권장한다. 출력은 clean Git checkout 바깥에 둔다.

```powershell
python tools/mft_goal_20260726_launch.py prepare `
  --generation <registry/generations/G0> `
  --candidate <candidate.json> `
  --quality-status <quality_status.json> `
  --code-root <clean-1MW_MFT-checkout> `
  --expected-code-revision <40-char-commit> `
  --mode single `
  --seed-start 2607260001 `
  --fixed-primary-turns 5 `
  --output <outside-repo>/mft-goal-single
```

이 단계는 G0 report/candidate/quality/artifact inventory, body+probe 11개
temperature target의 exact quality thresholds, clean code revision을 인증하고 N1=5,6,7,8 각각에서 실제
decoder와 24개 모델을 실행한다. 통과 후에만 bundle/task/Scheduler manifest가
생성된다.

생성된 task를 로컬에서 그대로 실행하는 명령:

```powershell
python tools/mft_goal_20260726_launch.py execute-seed `
  --payload <outside-repo>/mft-goal-single/tasks/seed-2607260001-n1-5.json `
  --output <outside-repo>/runs/seed-2607260001
```

품질 gate가 실패한 G0도 threshold를 낮추지 않은 채 탐색 자료 생성에는 쓸 수
있다. 이 경우 모든 manifest/result가 `search_only_proposal=true`,
`production_eligible=false`, `automatic_promotion_allowed=false`로 봉인된다.
Standard/Full FEA 통과 없이 승격할 수 없다.

## 2. 32-seed canary/rolling bundle

```powershell
python tools/mft_goal_20260726_launch.py prepare `
  --generation <registry/generations/G0> `
  --candidate <candidate.json> `
  --quality-status <quality_status.json> `
  --code-root <clean-1MW_MFT-checkout> `
  --expected-code-revision <40-char-commit> `
  --mode rolling32 `
  --seed-start 2607261000 `
  --output <outside-repo>/mft-goal-rolling32
```

첫 4개 seed는 N1=5,6,7,8 각 한 개씩 canary이다. 성공한 뒤 나머지 28개를
4-task rolling wave로 해제한다. 자원이 허용되면 Scheduler는 wave 내부 task를
동시에 실행하고 task당 CPU 8개, optimizer process 1개를 사용한다.

32-seed canary 이후 512-seed 이상으로 확장할 때:

```powershell
python tools/mft_goal_20260726_launch.py prepare `
  --generation <registry/generations/G0> `
  --candidate <candidate.json> `
  --quality-status <quality_status.json> `
  --code-root <clean-1MW_MFT-checkout> `
  --expected-code-revision <40-char-commit> `
  --mode rolling `
  --seed-start 2607262000 `
  --seed-count 512 `
  --wave-size 32 `
  --output <outside-repo>/mft-goal-rolling512
```

remote worker의 경로가 prepare host와 다르면 task별 relocation JSON을 만든다.
형식은 `mft-goal-20260726-worker-relocation-v1`이며
`task_payload_sha256`, 정확히 여섯 runtime role
`generation/candidate/quality_status/code_root/dataset/profile`,
`code_manifest_path`, canonical `payload_sha256` seal을 포함한다.
또한 `source_absolute_paths_are_documentary_only=true`,
`remote_git_checkout_required=false`를 명시한다. prepare는 relocation 파일을
미리 만들지 않는다. GPFS 절대 경로를 아는 staging/Scheduler 쪽이 task별로
생성하고 seal해야 한다. 실행 시:

```powershell
python tools/mft_goal_20260726_launch.py execute-seed `
  --payload <task.json> `
  --relocation <worker-relocation.json> `
  --output <worker-run-directory>
```

prepare 결과의 code byte는 `<bundle>/artifacts/code/<repo-relative>`에 있고
revision marker는 `<bundle>/artifacts/code/.source-revision`, manifest는
`<bundle>/code_manifest.json`이다. 원본 checkout은 수정하지 않는다. worker는
relocated path를 권위로 믿지 않고 train report, candidate, quality, dataset,
profile, complete model inventory와 code inventory/revision marker를 task와
다시 대조한다. staged code root에는 `.git`이 없어도 되며, relocation을 쓰지
않는 로컬 경로는 기존 clean Git checkout 인증을 유지한다.

마감은 2026-07-26 18:00 KST (`2026-07-26T09:00:00Z`)로 manifest에
고정돼 있다. CLI 자체는 Scheduler API write/submission을 하지 않는다.

## 3. 결과 수집과 전역 Pareto

각 성공 task는 다음을 함께 내야 한다.

- `result.json`
- `terminal_physical_candidates.csv`
- `terminal_physical_candidates.manifest.json`
- seed-local Pareto/least-violation arrays와 candidate records

전역 수집기는 32개 terminal table(총 10,240 rows)을 합친 뒤
`physical_geometry_sha256`로 물리 중복을 제거하고, 공통
`dataset_sha256`, `evaluation_model_sha256`, `constraint_spec_sha256`,
`cooling_contract_sha256`, `operating_point_sha256`가 모두 동일한지 확인한
후 `decoder_valid`, 모든 physical G `<= 0`, `surrogate_physical_valid`를
동시에 만족해 canonical `physical_feasible=true`인 행에 대해서만
non-dominated sorting을 다시 한다. 음수 loss 등 비물리 surrogate 출력은
physical G를 통과해도 Pareto 입력에서 격리된다.
seed-local Pareto끼리만 합쳐서는 최종 Pareto로 인정하지 않는다.

```powershell
python tools/mft_goal_20260726_launch.py aggregate `
  --results-root <outside-repo>/runs `
  --bundle-manifest <outside-repo>/mft-goal-rolling32/bundle_manifest.json `
  --minimum-seeds 32 `
  --output <outside-repo>/global-pareto
```

512-seed 본 실행 결과에는 `--minimum-seeds 512`를 사용한다. 출력은
`global_terminal_candidates.csv`, `global_pareto_front.csv`,
`aggregate_manifest.json`이며 입력 result와 공통 모델/데이터/물리 계약 SHA가
모두 봉인된다. 수집기는 bundle의 원본 task ledger를 읽고 모든 task에 정확히
한 result가 있는지, seed/N1/population 320/fixed 300 evaluated generations,
Pymoo 종료 카운터 301과 네 N1 strata가 일치하는지 확인한다. 임의로
self-seal한 result, 누락/중복 result, result 디렉터리 밖 artifact path는
거부한다.

전역 Pareto에서 선정한 최종 후보는 동일한 fixed operating/cooling identity로
full model과 symmetric model을 재실행하고 두 `.aedt` 파일 및 FEA 결과 identity를
함께 봉인한다.
