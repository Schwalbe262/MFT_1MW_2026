# AEDT attach pool incident — 2026-07-15

## 관측된 장애

라이브 풀은 `max_aedt_sessions=250`, `projects_per_aedt=3`,
`target_project_concurrency=750` 상태였지만 active session은 0이었고, 약 290개의
lease가 queued 상태에 머물렀다. session history에는 465개의 실패가 남아 있었다.

세션 host task의 terminal log에는 다음 제어면 오류가 반복되었다.

- `Remote end closed connection without response`
- `HTTP Error 502: Bad Gateway`
- heartbeat timeout
- project release-complete ACK 실패

즉 AEDT가 먼저 자발적으로 종료된 경우만 있었던 것이 아니다. relay/web 요청이 끊기면
기존 host가 이를 terminal host 오류로 취급해 Desktop을 닫았고, scheduler는 heartbeat가
사라진 session을 quarantine/recycle했다. 이 연쇄 작동 때문에 살아 있던 project도 attach
대상을 잃었다.

추가로 기존 session-host task는 공유 Desktop에서 여러 4-core project를 실행하면서도
Slurm에는 1 CPU와 4096 MiB만 예약했다. 이는 control plane과 별개로 CPU affinity 제한과
OOM 종료를 일으킬 수 있는 잠재 장애였다.

## 적용한 교정

- protocol v2 lease를 `queued -> offered -> attaching -> active -> releasing -> terminal`
  상태와 accept/activate/close ACK로 관리한다.
- client와 host의 control-plane mutation은 동일한 intent를 bounded retry하며, relay 장애
  동안 host는 새 admission/command를 동결하고 살아 있는 Desktop과 solve를 유지한다.
- client heartbeat는 solver Python/GIL과 독립적인 child process가 소유한다.
- session-host task는 `project_cpus * projects_per_aedt` CPU와
  `project_memory_mb * projects_per_aedt` memory를 예약하고, node capacity도 CPU와 memory의
  작은 쪽으로 계산한다.
- Desktop-global DSO는 host가 Icepak, Maxwell 2D, Maxwell 3D에 대해 한 번만 설치하고
  readback한다. pooled client는 `analyze(cores=...)`를 호출하지 않는다.
- MFT와 IPMSM은 동일한 immutable session profile을 제출한다. 기본 배치는 `family`로
  격리하며, 혼합 canary를 통과한 뒤에만 양쪽을 `shared_if_compatible`로 전환한다.
- host별 PyAEDT log와 session event journal 경로를 DB에 기록한다.
- SQLite WAL/busy timeout과 짧은 placement transaction으로 heartbeat write contention을
  줄인다.
- MFT 동시성 UI는 durable CAS policy를 사용한다. desired 감소는 실행 중 task를 취소하지
  않고 drain하며, 별도 인증된 validation ceiling보다 높은 값은 선택할 수 없다.

## 안전한 재가동 순서

고장 난 구형 풀은 기존 Slurm task를 취소하지 않은 채 disabled로 전환했다. 새 버전은 다음
순서로만 승격한다.

1. 새 DB migration과 전체 unit/regression test
2. MFT 1개 end-to-end attach/solve/project-close ACK
3. MFT 1개 + IPMSM 1개를 `family` 격리 상태로 동시 실행
4. 동일 조합을 `shared_if_compatible` 상태로 실행하고 DSO/project/result identity 검증
5. 10, 50, 이후 단계적 concurrency ramp와 각 단계의 오류율/lease latency/license/memory 확인
6. 검증 ceiling을 통과한 단계까지만 UI에서 허용하고 최종적으로 MFT desired 500까지 승격

500이라는 설정값은 구현 목표이지 현재 시점의 안정성 증거가 아니다. 각 단계에서 새
transport death, close ACK 누락, session recycle, profile mismatch 또는 memory pressure가
발생하면 그 단계에서 자동으로 증설을 멈추고 기존 solve는 drain한다.
