# regression_260707 — MFT 최적설계 캠페인 파이프라인

승인 플랜: `C:\Users\peets\.claude\plans\floofy-stargazing-bird.md` / 관례: `../symmetry_conventions_260706.md`

## 구성

```
campaign/  submit_wave.py     웨이브 제출 (400 태스크 x --count, 태스크별 클론 + golden)
           collect_wave.py    stdout CSV 회수 -> 수렴 필터 -> dataset/train.parquet
           gate1_report.py    대칭 vs 풀 자동 대조 판정 (게이트1)
           quality_report.py  golden 드리프트 / 실패 분류 / 커버리지 (게이트2)
training/  checkpoint_train.py  수집 중 학습곡선 모니터링 (LightGBM 5-fold, 전역+슬라이스)
           train_models.py      본 앙상블 (LGB/XGB/CatBoost/ET x 5-fold, 컨포멀 q90) -> registry/
           tune_optuna.py       하이퍼파라미터 튜닝 (>=4k 데이터에서 1회)
           predictor.py         predict_mu_sigma + DensityGate (외삽 봉쇄)
optimization/ geometry_metrics.py  외곽 박스 부피 (도면 427L 검산 통과)
              nsga2_problem.py     20차원 단위 유전자 + 불확실성 조임 제약 9종
              run_nsga2.py         16 재시작 + warm start + NDS 병합
verify/    scheduler_client.py  검증 태스크 제출/회수 (RESULT_JSON)
           select_candidates.py K=33 선정 (HV활용/제약경계/최대sigma탐사)
           profiles/            standard(캠페인 동일) / fine(풀모델+부력on, 게이트5)
al_driver.py  능동학습 루프 상태기계 (TRAIN->OPTIMIZE->SELECT->SUBMIT->WAIT->INGEST->CHECK)
```

## 운영 순서

1. 게이트1: `python campaign/gate1_report.py --pairs <sym>:<full> ...`
2. 파일럿 400: `python campaign/submit_wave.py --tasks 400 --count 1 --wave 0 --pilot`
   -> `python campaign/collect_wave.py --prefix mft-camp-pilot` -> `python campaign/quality_report.py` (게이트2)
3. 본 웨이브 x5: `submit_wave.py --tasks 400 --count 5 --wave N` (+체크포인트마다 `training/checkpoint_train.py`)
4. 튜닝(1회): `training/tune_optuna.py --all --trials 200` -> 본학습 `training/train_models.py --params best_params.json`
5. AL 루프: `python al_driver.py` (재개 가능; state.json)
6. 최종: fine 프로파일 검증 + final_report.md (게이트5: 메시수렴/해석검산/공차 몬테카를로/재현성)

## 데이터 규약

- 대칭 매트릭스 L 컬럼은 실물의 1/2 -> 학습은 `*_phys` (to_physical에서 x2)
- 손실/B는 CSV에 이미 실물(_phys 보정) 기준으로 기록됨 (`_raw` = 대칭 적분 원값)
- 검증/AL 행은 source, sample_weight(3.0) 태깅
