# 비침습적 대사이상 선별 연구 — KNHANES 2021–2024

**최신 v20은 v19의 인공 누락 마스크를 교정한 후속 탐색 연구다.** 만 19–39세 4,068명, 같은 원시 비침습 입력 19개와 5개 PSU 역할 분할을 사용했다. WC와 파생 WHtR에 독립된 두 20% 추첨을 OR로 적용한 v19에서는 허리 그룹 마스크율이 이론상 36%, 실측 35.82%였다. v20은 한 추첨을 공유해 실측 19.32%가 됐다. 모델 구조는 v19 그대로다.

| 사전 구분한 조건 · 4소견 macro AP | EPF | MLP | LR |
|---|---:|---:|---:|
| 기본 clean 모델, 그대로 재사용 | 0.346860 | 0.349517 | 0.357270 |
| 교정 paired20 모델의 clean 시험 | 0.348112 | 0.352344 | 0.357964 |
| 교정 paired20 모델의 corrected missing20 시험 | 0.331050 | 0.340484 | 0.342107 |

기본 clean에서 EPF는 같은 입력의 LR보다 낮았다. v19 legacyWCp와 v20 corrected missing20은 **허리 누락 강도가 다르다.** 두 값을 같은 20% 조건의 성능 개선으로 해석하지 않는다. v20의 340개 평가 단위 중 clean 210개는 v19 fit을 읽기 전용으로 재사용했고, paired20 130개(신경망 120개·LR 10개)만 새로 fit했다. 시험에서 본 engineered-paired-anchor EPF 0.362758은 조건 탐색 점수이며 새 기본 모델로 승격하지 않았다.

[**v20 결과와 한계**](CLINICAL_V20_RESULTS.md) · [실행된 집계 노트북](clinical_v20_report.ipynb) · [집계 JSON](clinical_v20_report.json) · [340단위·4,420보기 검증](clinical_v20_verification.json) · [소스 요약](clinical_v20_source_manifest.json) · [교정 마스크 NumPy 코드](models/clinical_v20_missingness.py)

## 공개 예제 실행

공개 모델 7개는 사전 지정한 fold1의 **clean 예제**로 v20에서도 가중치를 바꾸지 않았다. 신경망은 seed42, LR은 seed가 없다. 교정 paired20 새 fit을 성적순으로 공개 예제로 선택하지 않았다. CLI는 로컬에서 한 번 추론하며 혈액검사·혈압 실측값을 입력으로 받지 않는다.

    pip install -r requirements-structures.txt
    python -m models.predict_clinical_v19 --model EPF_basic_clean_native --input example.json
    python -m unittest discover -p "test*.py" -q

EPF_basic_clean_native 외에 MLP_basic_clean_native, EPF_field_only_basic_clean_native, LR_basic_clean, LR_engineered_clean, EPF_regression, EPF_joint가 있다. [공개 v20 마스크 테스트](test_clinical_v20_missingness.py)는 합성 NumPy 입력으로 허리·WHtR의 동일 추첨, 명목 확률, 자연결측 보존과 파생식 일치를 확인한다. 전체 공개 테스트 **21개**가 통과했다. 노트북 재실행에는 numpy, pandas, matplotlib, ipykernel이 필요하다.

340개 단위를 원본 모델에서 13보기씩 재생해 예측·보정·컷오프의 최대 절대오차 **0**을 확인했다. 로컬 전체 테스트 **203개**도 통과했다. 실행 무결성은 임상 타당성이나 독립 외부 검증이 아니다. 기본 clean EPF의 혈압 소견 평균 예측은 실측 유병률보다 높고, 개발 역할에서 정한 90/95% 민감도 컷오프는 clean 시험에서 목표에 못 미쳤다. 이 한계는 마스크 교정만으로 해결되지 않았다. 이 모델은 연구 예제이며 진단이나 실제 검사 수치의 대체물이 아니다.

## 이전 연구와 별도 감사

[v19 원결과·프로파일](CLINICAL_V19_RESULTS.md) · [v19 감사 포함 노트북](clinical_v19_report.ipynb) · [v19 설계](CLINICAL_V19_DESIGN.md) · [v18 공동 학습](CLINICAL_MULTITASK.md) · [v17 단일 셀 수식](MECHANISM.md). v19 노트북에는 소견별·종합위험 보정, 역할·컷오프, legacy 누락 강도, 합성 체격측정, 수치 확장의 **별도 감사**가 포함돼 있으며 원래 44개 사전 대조와 분리된다. 원자료·개인별 예측·분할 인덱스는 공개하지 않는다.
