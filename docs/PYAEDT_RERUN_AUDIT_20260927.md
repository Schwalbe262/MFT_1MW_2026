# PyAEDT 재실행 점검 및 작업트리 정리 — 2026-09-27

## 프로젝트와 이번 범위

이 저장소는 1 MW 중주파 변압기(MFT)의 Maxwell 3D 전자기 해석, Icepak 열해석, 결과 수집 및 최적화 캠페인 코드다. `HANDOVER.md`의 운전점과 검증 게이트는 설계 목표를 설명한다. 이번 작업은 새 변압기 설계나 최종 설계 승인 작업이 아니라, 기존 PyAEDT 시뮬레이션을 다시 실행할 수 있는지 확인하고 코드의 실패 판정과 기록을 보강하는 작업이다.

## 재실행 코드 점검

- AEDT에서 생성한 코어·권선·판의 실제 객체와 체적, 대칭 분할을 다시 읽어 형상 생성 실패를 즉시 중단한다.
- DAB 목표 전력이 계산 가능한 범위를 넘으면 해석을 성공으로 기록하지 않는다.
- Icepak native `grid_mapping`에서 메쉬를 배정한 고체와 국소 메쉬 영역의 결합을 확인한다. 같은 실행에서 생성한 `DV...`와 `DV..._S...` 메쉬 스냅샷은 별도 묶음으로 판정한다.
- 결과에 솔버·`pyaedt_library` Git revision, PyAEDT 및 AEDT 버전을 기록한다. 지정한 예상 버전과 다르면 시작 전에 중단한다.
- 손실 여기 모델은 `single_frequency_sinusoidal_proxy`로 표시한다. 이 결과를 실제 DAB 스위칭 파형의 전체 고조파 손실로 해석하면 안 된다.

## 검증 환경과 결과

- Python 3.11.14 (`pyaedt2026v1`), PyAEDT 0.22.0, 설치 AEDT 2025.2.
- `pyaedt_library`는 깨끗한 별도 checkout `e6b9b9d20a832ff5c3f7ca97218737a0b8650781`을 사용했다.
- `regression_260707/test_simulation_stability.py`와 `tests/test_thermal_stability.py`: 279 passed.
- `regression_260707/monitoring/tests`: 330 passed. 전체 `tests` 실행은 이 호스트의 외부 Scheduler 배포 파일을 요구하는 테스트에서 멈췄다(별도 환경 의존성).
- `verification_params/thermal_smoke.json`의 headless model-only 시험은 통과했다.
- 동일 조건의 Maxwell 3D + Icepak canary: Maxwell matrix·capacitance·loss 해석과 Icepak native mesh 생성 및 사전 검증을 통과했다. Icepak Fluent 유동 계산은 82회 반복에서 continuity residual `4.33e+02`, 0 K 온도 제한과 pressure outlet 역류가 나타나 시험을 중단했다. 따라서 **열해석 성공이나 수렴은 검증되지 않았다.** 조건은 `P_target=0`, `fan_velocity=1.5`, `wcp_pad_t=2`, `core_plate_pad_t=2`; 이 시험은 파이프라인 검증용이며 1 MW 운전점의 설계 검증이 아니다. 원시 로그는 아카이브의 `canary_thermal_smoke_retry.log`, `canary_thermal_smoke_retry_fluent.trn`, `canary_thermal_smoke_retry_uns_out.log`에 있다.

## 백업과 정리 범위

아카이브: `D:\MFT_1MW_2026_archive\20260927T112938Z`. Git bundle 두 개를 검증했고, 초기 547개 작업트리 HEAD와 이후 코드 에이전트의 커밋을 포함한다. 변경·미추적 파일과 비루트 ignored 보존 파일 22,044개, 54,346,343,933바이트를 복사하고 파일별 SHA-256을 확인했다. 작업 중 바뀐 임시 파일 9개(106,226바이트)는 복사되지 않았으므로 해당 `MFT_solver_pooled_260714` 작업트리는 삭제 대상에서 제외한다. 루트의 고유 ignored 산출물 약 300 GB는 원래 위치에 둔다.

정리할 폴더는 백업과 현재 상태가 일치하고 실행 중 사용되지 않는 `C:\w` 및 `C:\Users\peets\codex_worktrees`의 개발용 Git 작업트리다. `slurm_scheduler_runtime`의 배포 폴더와 현재 루트는 운영·고유 데이터 보존을 위해 유지한다.

개발용 작업트리 292개를 제거해 등록 작업트리를 547개에서 255개로 줄였다. `C:\w\mft-goal-20260726`은 내부 reparse point가 있어 자동 제거에서 제외했다.

## 해석상의 남은 제한

코어의 일정 상대투자율 3000은 재료 데이터시트 B-H 곡선으로 검증된 비선형 모델이 아니다. 정현파 단일 주파수 근사는 실제 DAB 스위칭 파형의 고조파 손실을 포함하지 않는다. 따라서 이 canary나 단위 테스트만으로 온도·손실의 물리적 정확도 또는 최종 설계 적합성을 승인할 수 없다. 다음 검증에는 대표 운전점의 실제 파형·재료 데이터와 메쉬 수렴 비교가 필요하다.

Icepak 유동 수렴 문제도 남아 있다. 초기조건·경계조건·유동 솔버 설정과 메쉬 품질을 원시 Fluent 로그에서 조사하고, 수렴한 별도의 대표 조건을 확보해야 한다. 이번 실패를 성공 샘플로 등록하거나 온도 판정에 사용하면 안 된다.
