# MFT 1MW 전용 모니터

`slurm_scheduler`와 분리된 MFT 전용 WEB UI다. MFT 저장소의 데이터셋, 학습 리포트,
AL/NSGA-II 산출물, 검증 결과를 직접 읽는다. 스케줄러 상태는 경량
`GET /api/tasks/summary`와 MFT 프로젝트 조회만 사용하며 수만 건의 작업 목록은 읽지
않는다. 유일한 쓰기 기능은 MFT 프로젝트의 병렬 유지 목표를 바꾸는 cap-only PATCH다.
repo/setup/entrypoint는 수정하지 않으며 IPMSM 프로젝트도 건드리지 않는다.

## 실행

저장소 루트의 PowerShell에서 다음 한 줄을 실행한다.

```powershell
.\start_monitor.ps1
```

최초 실행에서 전용 `.venv`와 최소 패키지를 설치한다. 서버는 외부에 노출되지 않는
`http://127.0.0.1:8010`에 바인딩된다. 다른 포트를 쓰려면
`.\start_monitor.ps1 -Port 8011`처럼 실행한다.

병렬 목표 변경 API는 추가로 loopback client, same-origin, JSON 전용 custom header를
검사한다. reverse proxy로 외부에 노출해 쓰는 운영 방식은 지원하지 않는다.

직접 실행할 수도 있다.

```powershell
py -3.11 -m venv regression_260707\monitoring\.venv
regression_260707\monitoring\.venv\Scripts\python -m pip install -r regression_260707\monitoring\requirements.txt
regression_260707\monitoring\.venv\Scripts\python -m uvicorn regression_260707.monitoring.app:app --host 127.0.0.1 --port 8010
```

## 읽는 산출물

- 데이터: `data/dataset/manifest.json`, `train.parquet`, `train_io.csv`, `collect_cache.json`
  - 활성 코호트·정전용량·열전도 모델·격리 집계는 손실 없는 `train.parquet`를 사용하며,
    Parquet가 없거나 읽는 중이면 기존 CSV 화면으로 안전하게 폴백한다.
  - 활성 코호트는 `module.core_material_contract.PHYSICS_DATA_REVISION`과 일치하는 행 중
    `saved_at`이 가장 최신인 행의 `(git_hash, physics_data_revision)` 조합이다.
- 모델: `training/registry/current.json`이 가리키는 승인 generation의
  `train_report.json`, 각 모델의 `meta.json`, `training/learning_curve.csv`
- 최적화: 최신 `al_rounds/round_*/pareto_front.csv`, 선택적으로 `al_rounds/state.json`
- standard FEA: `state.json`의 `task_records`와 최신 `verification_errors.csv`
- fine FEA: 아래 경로 중 먼저 발견되는 JSON
  - `verify/results/final_verification.json`
  - `verify/final_verification.json`
  - `monitoring/runtime/final_verification.json`

fine FEA JSON에는 `result` 객체와 선택적으로 `candidate_id`, `task_id`, `profile`,
`status` 또는 `passed`를 넣는다. `result`에는 기존 시뮬레이션 `RESULT_JSON` 필드를
그대로 사용할 수 있다. 없는 산출물은 실패가 아니라 실행 전 상태로 표시된다.

현재 화면의 압축본과 추세는 실행 중 `monitoring/runtime/monitor_snapshot.json` 및
`monitor_history.jsonl`에 원자적으로 기록된다. 이 디렉터리는 Git에서 제외된다.

## 설정 환경변수

- `MFT_MONITOR_ROOT`: 기본값은 `regression_260707` 디렉터리
- `MFT_SCHEDULER_URL`: 기본값 `http://127.0.0.1:8000`
- `MFT_MONITOR_TASK_PREFIX`: 조회할 작업 이름 접두사, 기본값 `mft`
- `MFT_SCHEDULER_PROJECT`: 병렬 목표를 제어할 프로젝트, 기본값 `MFT_1MW_2026v1`
- `MFT_SCHEDULER_TIMEOUT`: 스케줄러 GET 제한시간(초), 기본값 `2`
- `MFT_SCHEDULER_OPTIONAL_TIMEOUT`: AEDT pool/license 선택 조회 제한시간(초), 기본값 `1`
- `MFT_MONITOR_DISABLE_HISTORY=1`: runtime snapshot/history 기록 중지

## 테스트

```powershell
regression_260707\monitoring\.venv\Scripts\python -m pip install -r regression_260707\monitoring\requirements-dev.txt
regression_260707\monitoring\.venv\Scripts\python -m pytest regression_260707\monitoring\tests -q
```
