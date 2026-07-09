# MFT 1MW 2026 캠페인 — LLM 인수인계 문서

(최종 갱신: 2026-07-10 새벽. 이 문서만 읽으면 작업을 이어받을 수 있도록 작성됨)

## 1. 미션

**1MW MFT(중주파 변압기) 최종 설계 도출** — 사용자 지시: "목표 성능을 만족하는
설계가 나올 때까지 자율적으로 계속" (Stop hook 활성). 하드웨어 제작비 1억+이므로
검증 게이트 5단계를 통과한 설계만 최종 보고.

**확정 스펙:**
| 항목 | 값 |
|---|---|
| 누설 Llt | **27.5 µH ± 2% (실물 기준)** — 대칭모델 값 ×2 = 실물. 도면 실측은 55.7µH(스펙 2배)라 최적화 필수 |
| 온도 | 전 부품 ≤ 100°C (캠페인 서러게이트 제약은 97°C — eighth 열모델 −2~3°C 편향 보상) |
| 목적함수 | 부피(외곽박스) + 총손실 최소화 (NSGA-2 Pareto) |
| 절연 | 모든 권선 간격 ≥ 40mm (HV 4쌍 + z방향 h_gap2) |
| B | 코어 ≤ 1.2 T |
| 운전점 | 1kHz / 1000V / 10kV / 100A / P_target 1MW (DAB 위상 자동 역산) |
| 기타 | N1 ≤ 10턴, N1_side=0 고정, 구리 80°C 도전율, radiation은 최종검증만 |

**로드맵**: 데이터 1만+ 수집 → 서러게이트 학습 → NSGA-2 → 능동학습(AL) 루프
→ fine 최종검증. 게이트: ①물리검증(통과) ②파일럿 ③서러게이트 품질 ④AL 수렴 ⑤최종 fine+공차MC.

## 2. 현재 상태 (인수 시점)

- 데이터: `regression_260707/data/dataset/train.parquet` ~150행 (온도 포함 ~15)
  — 하루 종일 인프라 결함 수리로 실질 수집은 이제 시작
- 클러스터: 신세대 함대 ~300 활성 (모든 수정 반영판, 커밋 2df1562+), target 200
- 로컬: 생산 2기 (Y:\git\MFT_1MW_2026 + _local2, 각각 --count 999)
- 학습 리허설: 103행에서 Llt R²=0.87 (train_models), NSGA-2/AL 기계 전체 리허설 완료
- 대기 중 판정: 클러스터 신세대 첫 온도샘플(CLUSTER_TEMPS_LIVE), pe/mc/skin 벤치 40개 표

## 3. 시스템 구조

### 저장소
- `Y:\git\MFT_1MW_2026` (github Schwalbe262/MFT_1MW_2026) — 시뮬레이션 코드. **푸시하면
  클러스터 태스크가 시작 시 auto-pull로 반영** (실행 중 태스크는 반영 안 됨 — count 소진 후)
- `Y:\git\pyaedt_library` — pyaedt 래퍼 (example/MFT_TAB에 검증된 패턴 다수)
- `Y:\git\slurm_scheduler` — 스케줄러 (수정 권한 있음, CHANGELOG_claude.md에 기록;
  라이브는 ~/NEC/slurm_scheduler라 반영은 스케줄러팀에 요청)

### 핵심 파일
- `run_simulation_260706.py` — 오케스트레이터. 흐름: 랜덤샘플(Sobol)→검증→
  **경량 matrix**(skin無, pe2.0/8pass/mc1)→DAB 위상 역산→**loss를 matrix 복제로 생성**
  (CopyDesign+Paste, 상속解 삭제 필수!)→정밀 loss(skin, pe1.5)→eighth thermal→
  RESULT_JSON 스트리밍 + per-run parquet 파트
- `module/input_parameter_260706.py` — KEYS/기본값/Sobol 샘플러/검증. 새 --set 키는
  반드시 KEYS에 등록해야 함
- `module/thermal_260706.py` — Icepak. 온도 추출은 **field summary 일괄이 1차**
  (계산기 ClcEval은 gRPC에서 상습 실패 → 조용한 폴백으로 강등)
- `regression_260707/campaign/` — feeder.py(장부 기반, target+buffer 하드캡),
  submit_wave.py(**project 방식 제출**), collect_wave.py(스트리밍+캐시 회수),
  sweep_stale.py(디스크 청소), relaunch.sh
- `regression_260707/training|optimization|verify/` — 학습/NSGA/AL (README, BIAS_PLAYBOOK 참조)

### 스케줄러 (127.0.0.1:8000)
- **Projects 기능 사용**: 제출 = `POST /api/tasks` body
  `{"project":"MFT_1MW_2026v1","entrypoint":"run_simulation_260706.py","arguments":"...","cpus":4,"memory_mb":32768,"scheduling_profile":"fea_bursty"}`
  → task_id 반환. 배포 코드는 각 계정 `~/slurm_scheduler/projects/MFT_1MW_2026v1/`
- 취소: `POST /tasks/{id}/cancel` (303=성공, **취소가 AEDT 자식까지 회수** — 스케줄러팀이 수리함)
- 회수: stdout의 `RESULT_JSON {...}` 라인 (샘플 단위 스트리밍) — collect_wave가 처리
- 계정 5개: r1jae262, harry261, dw16, jji0930, dhj02. 라이선스 550석/기능,
  `GET /api/licenses`로 사용량 확인 가능

## 4. 피눈물 교훈 (재발 금지 규칙)

1. **배포 게이트**: 스모크 통과 ≠ 안전. **로컬 랜덤 3연속 + (가능하면) 사용자 GUI 1회
   + 클러스터 파일럿 10** 통과 후에만 함대 적용. (복제-loss가 스모크만 믿고 배포됐다가
   함대 전멸시킨 전례)
2. **개입은 무손실로**: target 변경/큐 재활용은 OK, 전체 드레인은 치명 결함일 때만.
   잦은 리셋이 in-flight를 폐기해 하루 유입을 0으로 만든 전례
3. **동시수 = 코어 예산**: free_cpus/4 × 0.85. max_workers_per_node는 동적 패킹이
   무시함(하한처럼 동작). 코어 100% 예약 시 AEDT 오버헤드가 낄 곳이 없어 전체 저속
4. **리눅스 gRPC 불신 목록**: ClcEval/GetTopEntryValue/CalculatorWrite(추출),
   리포트 ExportToFile(CSV 미생성), get_scalar_field_value — 전부 재시도+대체경로 필수.
   field summary와 get_solution_data가 신뢰 경로
5. **TaskStop은 셸만 죽임** — 파이썬 자식은 powershell로 직접 확인·사살
   (`Get-CimInstance ... -match 'feeder'`). 유령 피더가 하루 종일 몰래 재충전한 전례
6. **디자인 복제 시**: 객체 핸들 재매핑(find_object) + **상속 해 삭제(DeleteFullVariation)**
   필수. MFT_TAB 레퍼런스 참조
7. **NUM_CORE는 sched_getaffinity로** — SLURM_CPUS_PER_TASK는 packed 잡에서 잡 전체 값
8. 시간 통계는 생존 편향 주의 — 처리량 지표는 train.parquet 행수/시간만 신뢰
9. 대칭 환산 관례: 손실 실물 = 대칭적분 ×2^c/4, Llt 실물 = 대칭 ×2, B 실물 = ÷2.
   `sym_cut_count()` 참조. 도면 검증 완료값: gate1_* 파일들

## 5. 상비 운영 (백그라운드)

- 피더: `cd regression_260707/campaign && python feeder.py --loop 600 --max-samples 12000 --target 200`
- 회수+체크포인트: 시간당 collect_wave + 500/1k/2k/4k/8k행 자동 학습 (auto_collect 루프)
- 디스크: 4시간 크론(사용자 설정) + sweep_stale.py + 프로젝트 cleanup_globs
- 감시: Monitor로 유입 가드(시간당, 3연속 정체 시 경보) — 정체 시 로그 정밀 판독부터

## 6. 다음 할 일 (우선순위)

1. **클러스터 신세대 첫 온도샘플 확인** → 통과 시 순수 누적 모드 (목표 1만행)
2. pe/mc/skin 벤치 40개 표 완성 → 최적 (pe, min_converged, skin) 확정 +
   **skin-off Llt 편향(+0.7~1.3% 예상) 정량화 → NSGA Llt 타겟 보정치 반영**
3. 500행 → checkpoint_train 자동 → 게이트3 (타겟별 R², conformal 커버리지)
4. 클러스터 thermal 솔브 실패(과거 2/3) 원인 — 새 계측(메시지 20줄 덤프)으로 파악
5. AL 루프 (verify/al_driver.py, K=33 구성) → 게이트4 → fine 최종검증+공차MC → final_report.md
   (뼈대: regression_260707/final_report_skeleton.md)
6. 모델링 추가 다이어트 (corner_segments 축소 등) — 여유 있을 때

## 7. 자주 쓰는 명령

```bash
# 데이터 확인
python -c "import pandas as pd; df=pd.read_parquet(r'Y:/git/MFT_1MW_2026/regression_260707/data/dataset/train.parquet'); print(len(df))"
# 수동 회수
cd regression_260707/campaign && python collect_wave.py --prefix mft
# 로컬 GUI 1회 (랜덤, 클러스터와 동일 설정)
python run_simulation_260706.py --thermal --hold --set percent_error=1.5 --set max_passes=10 --set P_target=1e6
# 학습 체크포인트
cd regression_260707 && python training/checkpoint_train.py
# NSGA 리허설
python optimization/run_nsga2.py --spec al_rounds/rehearsal_spec.json --round N
```

환경: conda `pyaedt2026v1` (파이썬은 `~/anaconda3/envs/pyaedt2026v1/python.exe`),
PYTHONIOENCODING=utf-8 권장. 메모리 파일: `C:\Users\peets\.claude\projects\Y--git-MFT-1MW-2026\memory\`
(mft-campaign-directive, mft-infra-lessons 필독).
