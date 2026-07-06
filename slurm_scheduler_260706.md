# slurm_scheduler로 run_simulation_260706.py 돌리기

[slurm_scheduler](https://github.com/Schwalbe262/slurm_scheduler)의 FEA 경로 두 가지에 맞춘 사용법.
공통 전제: 클러스터 계정의 `env_profiles`에 pyaedt conda env가 등록되어 있고
(`config/accounts.yaml`의 `conda:pyaedt2026v1`), 리포는 클러스터의 작업 경로에 클론되어 있음.

스크립트 쪽 대응 (이미 구현됨):

- `--count N`: 랜덤 스윕을 N회 성공 후 **exit 0** — 스케줄러의 exit_code 파일 기반 완료 감지와 호환
  (무한루프 금지). 실패 반복 시 3N회 시도 후 종료.
- `SIMULATION_ID` 환경변수 (dynamic_packed_srun이 주입): 프로젝트 이름을
  `simulation_<SLURM_JOB_ID>_<SIMULATION_ID>`로 지어 공유 파일시스템에서 카운터 파일 락 경합 제거.
- 랜덤 모드는 `keep_project=0` 기본값 → 각 케이스 완료/실패 시 프로젝트 폴더를 재시도 로직으로
  **확실히 삭제** (저장공간 확보). CSV(`simulation_results_260706.csv`)만 남음 (FileLock으로 병렬 안전).
- Linux에서는 자동으로 non-graphical 실행.

## 1. fea_bursty 단건 태스크 (랜덤 스윕 청크)

```bash
curl -sS -X POST "$SCHEDULER_URL/tasks" \
  -F name=mft-260706-sweep \
  -F remote_cwd=__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/MFT_1MW_2026 \
  -F command='python run_simulation_260706.py --count 20' \
  -F required_capability=conda:pyaedt2026v1 \
  -F env_profile=pyaedt2026v1 \
  -F profile=fea_bursty \
  -F cpus=4 \
  -F memory_mb=16384 \
  -F gpus=0
```

- `cpus=4`: 스크립트의 `NUM_CORE=4`와 일치 (해석당 4코어, 다건 병렬은 태스크를 여러 개 제출)
- `memory_mb`: 도면급 모델은 16~32GB 권장 (fea_bursty가 peak로 취급)
- Ansys 라이선스 수 제한은 스케줄러가 관리하지 않음 → 동시 태스크 수를 라이선스 수 이하로 제출할 것

## 2. dynamic_packed_srun (대량 스윕)

packed 잡의 각 시뮬레이션 호출은 `SIMULATION_ID`를 받으므로 케이스당 1회 실행으로 설정:

```bash
curl -sS -X POST "$SCHEDULER_URL/jobs" \
  -F mode=dynamic_packed_srun \
  -F entrypoint=run_simulation_260706.py \
  -F arguments='--count 1' \
  -F simulation_count=200 \
  -F cpus_per_simulation=4 \
  -F ...  # 계정/파티션 옵션은 스케줄러 설정에 따름
```

## 3. 최종 도면 해석 (fixed, 손실+열)

```bash
curl -sS -X POST "$SCHEDULER_URL/tasks" \
  -F name=mft-260706-final \
  -F remote_cwd=__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/MFT_1MW_2026 \
  -F command='python run_simulation_260706.py --fixed --round --thermal' \
  -F required_capability=conda:pyaedt2026v1 \
  -F env_profile=pyaedt2026v1 \
  -F profile=fea_bursty \
  -F cpus=4 \
  -F memory_mb=65536
```

fixed 모드는 `keep_project=1` 기본값이라 프로젝트가 보존됨 — 클러스터에서 공간이 아까우면
`--params` json에 `{"keep_project": 0}` 추가.
