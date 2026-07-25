# Old-G0 추가 NSGA-II 병렬 용량 감사

판정 시각은 2026-07-25 23:25:40 KST이고 마감까지 18.57시간이 남았다.
결론은 **추가 old-G0 NSGA-II 제출 금지**이다. 이 판정은 Slurm 자원을
사용하지 말자는 일반 원칙이 아니라, 지금의 유휴/향후 자원을 성공 가능성이
낮은 search-only 반복 대신 Standard FEA truth 회수와 단일 Full fast-lane에
보존하자는 캠페인 한정 판정이다.

## 과학적 근거

- 이미 완료된 G0 rolling512는 512 seed, terminal 163,840행,
  물리 중복 제거 후 133,563개 형상을 포함한다.
- hard-feasible 형상과 production Pareto는 모두 0개이다. 제약을 무시한
  audit-only objective front만 22개이다.
- Standard 제출용으로 고른 최인접 12개는 모두 N1=6이며, 12/12가
  `Llt_robust_band`와 `Llt_ensemble_disagreement`를 동시에 위반한다.
  4/12는 `T_max_Tx` robust limit도 위반한다.
- G0 quality gate 자체가 25개 target 중 15개 target에서 실패했다.
  따라서 이 세대는 `search_only_proposal=true`,
  `production_eligible=false`, `automatic_promotion_allowed=false`이다.
  같은 고정 세대에서 seed만 늘리는 것은 이 체계적 모델/robustness blocker를
  바꾸지 않는다.
- 0/512 seed 관측을 독립·정상성 Bernoulli로 과도하게 낙관해 해석해도 seed당
  hard-feasible 발견 확률의 단측 95% 상한은 0.5834%이다. 추가 32 seed가
  하나라도 발견할 확률의 같은 상한은 17.08%일 뿐이며, 실제 seed 결과의
  상관성과 공통 모델 오차 때문에 이 계산은 제출 근거가 아니라 낙관적
  상한이다.

## 운영 근거

- 기존 512 task는 각 8 CPU/64 GiB였고 총 2,634.98 requested CPU-hours를
  사용했다. task runtime은 중앙값 2,064초, p90 3,569초였으며 캠페인 최초
  생성부터 마지막 완료까지 9,428초가 걸렸다.
- 같은 계약의 32-seed wave는 동시에 열릴 경우 256 CPU와 2 TiB의 요청
  envelope이고, 과거 평균 기준 약 164.69 CPU-hours이다.
- 판정 시점 Scheduler에는 4개의 Standard FEA가 32 CPU/128 GiB를 요청해
  실행 중이었다. 4개 active allocation 중 3개는 `queued FEA CPU demand`,
  나머지 1개는 `high CPU utilization` drain reason이었다.
- live `/api/task-capacity`는 old-G0의 8 CPU/64 GiB standard task와 Full의
  16 CPU/96 GiB FEA task 모두 `ready_fit_slots=0`이었다. 따라서 추가 NSGA는
  현재 유휴 pool에 즉시 붙는 작업이 아니라 queue와 demand-pool 개설을
  유발하는 작업이다.
- Full fast-lane은 16 CPU/96 GiB/12h, priority 100의 단일 critical task이다.
  추가 search-only wave는 다음 usable pool을 선점할 수 있으므로 마감
  critical path와 충돌한다.
- 과거 512 task의 로컬 retained harvest는 2.5744 GiB, task당 평균
  5.399 MB였다. 추가 32 seed의 단순 retained projection은 약 0.161 GiB로
  작지만, hypothetical 새 bundle의 remote transient GPFS 상한은 봉인돼
  있지 않다. 이 점은 독립적인 제출 허가가 될 수 없다.

## 격리와 재검토 조건

- 기존 old-G0 seed `2607262000..2607262511`은 그대로 보존한다.
- active-learning 전용 `2607263000..2607263511`은 손대지 않는다.
- 추가 old-G0 seed range는 할당하지 않았고, 새 bundle/aggregate/POST도
  만들지 않았다. old-G0와 새 generation 결과를 섞지 않는다.
- 재검토는 다음 중 하나가 아니라 **모두** 충족될 때만 가능하다:
  authenticated truth 8행/4 source 이상으로 새 generation이 admission 및
  품질 gate를 통과할 것, Standard/Full critical path를 침범하지 않는 별도
  ready capacity가 있을 것, 새 generation 전용 bundle과 GPFS transient
  envelope가 봉인될 것.

기계 판독 가능한 증거와 canonical payload seal은
`docs/evidence/mft_goal_old_g0_parallel_capacity_no_submit_20260725.json`에
있다. 이 감사 과정의 Scheduler method는 GET뿐이며 POST/PATCH/DELETE/CANCEL은
각각 0회이다.
