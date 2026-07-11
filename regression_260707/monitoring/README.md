# MFT 1MW 전용 모니터

`slurm_scheduler`와 완전히 분리된 읽기 전용 WEB UI다. MFT 저장소의 데이터셋,
학습 리포트, AL/NSGA-II 산출물, 검증 결과를 직접 읽고 스케줄러에는 작업 수 확인을
위한 경량 `GET /api/tasks/summary` 요청만 보낸다. 수만 건의 작업 목록은 읽지 않는다.

## 실행

저장소 루트의 PowerShell에서 다음 한 줄을 실행한다.

```powershell
.\start_monitor.ps1
```

최초 실행에서 전용 `.venv`와 최소 패키지를 설치한다. 서버는 외부에 노출되지 않는
`http://127.0.0.1:8010`에 바인딩된다. 다른 포트를 쓰려면
`.\start_monitor.ps1 -Port 8011`처럼 실행한다.

직접 실행할 수도 있다.

```powershell
py -3.11 -m venv regression_260707\monitoring\.venv
regression_260707\monitoring\.venv\Scripts\python -m pip install -r regression_260707\monitoring\requirements.txt
regression_260707\monitoring\.venv\Scripts\python -m uvicorn regression_260707.monitoring.app:app --host 127.0.0.1 --port 8010
```

## 읽는 산출물

- 데이터: `data/dataset/manifest.json`, `train_io.csv`, `collect_cache.json`
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
- `MFT_SCHEDULER_TIMEOUT`: 스케줄러 GET 제한시간(초), 기본값 `2`
- `MFT_MONITOR_DISABLE_HISTORY=1`: runtime snapshot/history 기록 중지

## 테스트

```powershell
regression_260707\monitoring\.venv\Scripts\python -m pip install -r regression_260707\monitoring\requirements-dev.txt
regression_260707\monitoring\.venv\Scripts\python -m pytest regression_260707\monitoring\tests -q
```
