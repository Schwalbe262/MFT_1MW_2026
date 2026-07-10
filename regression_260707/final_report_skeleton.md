# MFT 1 MW 최종 설계 보고서

이 파일은 구조 참고용이다. 실제 완료 보고서는 `verify/finalize.py`가
`final_report.md`와 `verify/results/final_verification.json`으로 원자적으로 생성한다.

최종 PASS에 필요한 증거:

- strict-full 데이터 수와 surrogate 전 타겟 정확도/불확실성 게이트
- NSGA-II 모델 세대, 데이터 fingerprint, Pareto/후보 선택 기록
- standard FEA 수렴·추출·전력평형·사양 검증과 검증행 재학습
- 사양 통과 후보 중 부피순 full-model fine FEA 결과
- 최소부피 fine PASS 후보의 파라미터, 부피, 손실, Llt, B, 온도, revision

제작공차는 실행·합격조건·최종보고에서 제외한다. 제작 형상은 FEA와 정확히
동일하다고 가정한다.
