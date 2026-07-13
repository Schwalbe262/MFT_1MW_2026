# ANSYS 라이선스 서버 장애 로그 (관리자 전달용)

## 환경
- License path: 1055@172.16.10.81
- 클라이언트: AEDT 2025.2 (Linux 계산 노드 + Windows 로컬), 동시 세션 수십~수백
- 증상 기간: 2026-07-07 오후 ~ 2026-07-08 (수정 통보 전까지)

## 증상 요약
1. FlexNet -16,10009 'Cannot read data from license server system' / WinSock 10054 reset
   - 대상 feature: elec_solve_maxwell, elec_solve_level2 (간헐, 랜덤)
   - lmstat상 좌석은 충분 (Maxwell 102/550 등) - 수량 고갈이 아니라 데몬 통신 실패
2. 후속 증상: 'Com Engine non-responsive' -> 'Engine terminated unexpectedly'
3. CFD/Icepak 계열은 사실상 전멸 (AnsysIcepak 활성 1/550)

## 원문 에러 샘플 (클러스터 태스크 stderr)


- 로컬 GUI (2026-07-07 14:59): `Cannot read data from license server system... Feature: elec_solve_maxwell License path: 1055@172.16.10.81; FlexNet Licensing error:-16,10009. System Error: 10054 "WinSock: Connection reset by peer"`
- 로컬 GUI (2026-07-07 16:07): 동일 오류, Feature: elec_solve_level2

## 수정 통보(약 6시간 전) 이후 상태
- 신규 태스크들에서 FlexNet 에러 재발 없음 (표본: 6시간+ 실행 태스크 license 에러 0회) -> 데몬 통신은 회복된 것으로 보임
- 단, solve가 여전히 장시간 미완료되는 별도 증상이 있어 노드측 원인(고아 프로세스 등) 자체 조사 중
  (라이선스와 무관 판단 시 별도 보고하지 않음)