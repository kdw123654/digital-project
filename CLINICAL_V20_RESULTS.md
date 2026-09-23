# KNHANES 2021–2024 비침습 대사이상 선별 — v20 마스크 교정

19–39세 4,068명(2021: 1,039, 2022: 1,013, 2023: 1,010, 2024: 1,006), 원시 비침습 입력 19개, 대사이상 소견 4개와 그 합집합을 분석했다. 혈액검사 수치와 혈압은 정답이며 입력이 아니다. v19의 5-fold 역할 분리, seed 42/43/44, 모델 구조·선택·보정·threshold 절차를 유지했다. 기존 코호트를 여러 차례 살핀 뒤 수행한 **후속 탐색적 교정**이다.

## 마스크 교정과 비교 가능성

v19의 WC·파생 WHtR은 독립된 두 20% 추첨의 OR이어서 이 그룹의 이론적 누락률이 36%였다. v20에서는 한 추첨을 공유한다. `missing20_0`의 실제 WC 인공 마스크율은 v19 35.82%, v20 19.32%였고, 세 보기 평균은 각각 35.52%, 19.77%였다. 다른 변수의 추첨은 유지했다. Gaussian `fitSD10/25`는 해당 측정치의 **fit 역할 표준편차** 10/25%를 모든 원래 관측값에 더하는 강도이며 사람 10/25%를 고르는 뜻이 아니다.

clean 모델 210개 평가 unit은 v19 원본 모델·선형 계수를 읽기 전용으로 재사용했다. 210개에서 clean 예측, 보정, cutoff가 원본과 정확히 일치했고 16개 clean 모델 정의의 clean 집계도 정확히 일치했다. corrected paired20은 신경망 120개와 LR 10개를 새로 학습·평가했다. 모든 모델은 각 조건에서 같은 사람, 같은 raw19 입력 보기, 소견, 역할, 조사 가중치를 받았다.

| 동일 clean fit 모델 | v19 legacyWCp 20% macro AP | v20 one-draw 20% macro AP | v19 보정 any AP | v20 보정 any AP |
|---|---:|---:|---:|---:|
| EPF@basic_clean_native | 0.292566 | 0.306934 | 0.601622 | 0.625730 |
| MLP@basic_clean_native | 0.308541 | 0.320796 | 0.635246 | 0.653503 |
| LR@basic_clean | 0.316833 | 0.333954 | 0.633983 | 0.659085 |
| EPF_field_only@basic_clean_native | 0.292468 | 0.307462 | 0.604188 | 0.629063 |

위 표는 **같은 clean fit**에 다른 마스크 강도를 넣은 비교다. 이전→수정 missing20의 AP 차이를 같은 강도의 견고성 개선으로 읽을 수 없다. paired20 간 v19→v20 비교에는 fit 입력과 평가 마스크가 함께 바뀌므로 별도로 해석한다.

## 기본 clean 성능과 고정 예측 구간

기본 clean macro AP는 EPF 0.346860, MLP 0.349517, LR 0.357270이다. 이 점추정에서 EPF는 LR보다 높지 않다. 세 seed의 지표 평균이며 앙상블 예측이 아니다. 동일한 소견별 단조 보정과 Fréchet 범위 γ 보정 기회를 EPF·MLP·LR에 적용했다.

| 모델 | 4소견 macro AP ↑ | 4소견 macro Brier ↓ | 보정 any AP ↑ | 보정 any Brier ↓ | 보정 any 평균위험−실제율 |
|---|---:|---:|---:|---:|---:|
| EPF@basic_clean_native | 0.346860 | 0.107605 | 0.681571 | 0.180974 | 0.007773 |
| MLP@basic_clean_native | 0.349517 | 0.107092 | 0.687908 | 0.179341 | 0.008958 |
| LR@basic_clean | 0.357270 | 0.107409 | 0.689132 | 0.180460 | 0.011877 |

| EPF 대조 | 지표 | EPF−비교모델 추정차 | paired PSU 95% 구간 | 유효 반복 |
|---|---|---:|---:|---:|
| primary_EPF_vs_LR | component_macro_ap_difference | -0.010410 | [-0.020556, -0.000433] | 2000 |
| primary_EPF_vs_LR | adjusted_any_ap_difference | -0.007561 | [-0.016479, 0.001632] | 2000 |
| primary_EPF_vs_LR | adjusted_any_brier_improvement | -0.000513 | [-0.002416, 0.001375] | 2000 |
| primary_EPF_vs_MLP | component_macro_ap_difference | -0.002657 | [-0.011400, 0.005573] | 2000 |
| primary_EPF_vs_MLP | adjusted_any_ap_difference | -0.006338 | [-0.014966, 0.002299] | 2000 |
| primary_EPF_vs_MLP | adjusted_any_brier_improvement | -0.001632 | [-0.003190, 0.000016] | 2000 |
| primary_EPF_vs_field_only | component_macro_ap_difference | 0.003624 | [-0.000661, 0.007855] | 2000 |
| primary_EPF_vs_field_only | adjusted_any_ap_difference | -0.000548 | [-0.004770, 0.003515] | 2000 |
| primary_EPF_vs_field_only | adjusted_any_brier_improvement | -0.000057 | [-0.000818, 0.000723] | 2000 |

EPF의 장·spike·plastic field 고유 기여는 이 표만으로 확정되지 않는다. engineered clean anchor 차이는 작은 사후 관찰이고 기존 구간에 0이 포함됐으며, 선택 epoch 0인 unit도 있었다. 이를 기전의 증거로 해석하지 않는다.

## corrected paired20 학습 효과

표는 같은 구조·basis·초기화에서 `paired20 fit − clean fit`의 점추정 차이다. 열 순서는 clean / corrected missing10 / corrected missing20 / corrected missing30 / corrected missing40이다. 양의 AP 차이만 개선 방향이다. clean 조건의 augmentation 대조 구간은 `clinical_v20_report.json`의 44개 대조에 있고, missing10–40 효과는 개별 구간이 없는 기술적 비교다.

| 모델·basis·초기화 | macro AP clean | macro AP 10 | macro AP 20 | macro AP 30 | macro AP 40 | any AP clean | any AP 10 | any AP 20 | any AP 30 | any AP 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EPF@basic_native | 0.00125 | 0.01477 | 0.02412 | 0.02554 | 0.04206 | -0.00256 | 0.01486 | 0.03254 | 0.03162 | 0.06264 |
| EPF@basic_linear_anchor | 0.00075 | 0.00183 | 0.00771 | 0.00832 | 0.00865 | -0.00336 | 0.00314 | 0.00529 | 0.00835 | 0.01625 |
| EPF@engineered_native | -0.00079 | 0.00966 | 0.01903 | 0.02492 | 0.03478 | -0.00066 | 0.01205 | 0.02754 | 0.02704 | 0.05879 |
| EPF@engineered_linear_anchor | 0.00101 | 0.00503 | 0.00839 | 0.00964 | 0.01037 | -0.00257 | 0.00094 | 0.00659 | 0.00970 | 0.01784 |
| MLP@basic_native | 0.00283 | 0.01414 | 0.01969 | 0.02324 | 0.02889 | -0.00043 | 0.00784 | 0.01573 | 0.01521 | 0.03528 |
| MLP@basic_linear_anchor | 0.00081 | 0.00293 | 0.00947 | 0.01082 | 0.01078 | -0.00340 | 0.00163 | 0.00609 | 0.00836 | 0.01614 |
| MLP@engineered_native | -0.00091 | 0.00976 | 0.01489 | 0.02054 | 0.02800 | -0.00136 | 0.00868 | 0.01626 | 0.01742 | 0.04753 |
| MLP@engineered_linear_anchor | 0.00141 | 0.00460 | 0.00870 | 0.00738 | 0.01100 | -0.00244 | 0.00191 | 0.00625 | 0.00856 | 0.01702 |
| LR@basic | 0.00069 | 0.00108 | 0.00815 | 0.00708 | 0.00505 | -0.00373 | 0.00129 | 0.00382 | 0.00561 | 0.01023 |
| LR@engineered | 0.00085 | 0.00436 | 0.00772 | 0.00636 | 0.00881 | -0.00222 | 0.00180 | 0.00561 | 0.00785 | 0.01511 |

같은 10개 paired 대조의 macro/any Brier, 보정 any 평균위험 차이, raw·보정 위험의 90/95% 고정 컷오프 시험 민감도와 의뢰율은 실행 출력이 포함된 노트북의 평탄한 50행 표와 `clinical_v20_report.json`의 `augmentation_effects`에 있다. LR paired C는 v19의 동일 grid에서 clean validation으로 다시 선택했다. 신경망 구성은 v19 fold별 선택을 고정했다. 시험 결과로 조건을 선택하지 않았다.

| 모델 | 시험 보기 | 90% 목표 실제 민감도 | 90% 목표 의뢰율 | 95% 목표 실제 민감도 | 95% 목표 의뢰율 |
|---|---|---:|---:|---:|---:|
| EPF@basic_clean_native | clean | 0.8738 | 0.6628 | 0.9300 | 0.7745 |
| EPF@basic_clean_native | missing20 | 0.9097 | 0.7640 | 0.9605 | 0.8649 |
| MLP@basic_clean_native | clean | 0.8711 | 0.6496 | 0.9272 | 0.7675 |
| MLP@basic_clean_native | missing20 | 0.9035 | 0.7403 | 0.9630 | 0.8682 |
| LR@basic_clean | clean | 0.8732 | 0.6582 | 0.9296 | 0.7707 |
| LR@basic_clean | missing20 | 0.8888 | 0.7167 | 0.9508 | 0.8343 |

90/95%는 별도 threshold 역할의 개발 목표이며 시험 민감도가 해당 목표에 도달했다는 뜻은 아니다. 위 표는 보정 any 위험이다. `clinical_v20_report.json`에는 raw any 위험도 함께 기록했다. 비교모델별 cut은 각자 개발 역할에서 고정했다.

## 비용, 불확실성, 검증 경계

신규 fit은 130개이며 기록된 신경망 120개 fit 시간 합계는 923.3초이다. LR 10개 fit 시간은 `not recorded in LR DONE.json`로 기록되지 않아 합계가 없다. 210개 재사용 clean unit의 과거 `fit_seconds`는 이 신규 비용에 넣지 않았다. 공유 장치 시간은 공정한 모델 속도 순위가 아니다.

v19와 같은 44개 clean 대조·7개 지표에 2,000회 전체 조사틀 PSU paired bootstrap을 적용했다. 이는 **고정 OOF 예측에 조건부인 구간**이다. 재학습 불확실성·다중비교 보정·외부 코호트·독립 확인은 없다. 학습 원본·v19 출처 및 분석·보고 코드의 로컬 manifest는 동결돼 있다. 공개 [소스 해시 요약](clinical_v20_source_manifest.json)은 코드 해시와 원본 manifest SHA-256만 담고 개인 파일명·경로는 제외한다. 별도 전체 원본 모델 재생 검증 상태: `passed` (`clinical_v20_verification.json`).

원자료, 역할 인덱스, 개인별 예측 및 절대 경로는 이 문서와 노트북에 포함하지 않았다.

[공개 집계 JSON](clinical_v20_report.json) · [실행된 집계 노트북](clinical_v20_report.ipynb) · [학습 계보·전체 forward 검증 요약](clinical_v20_verification.json) · [교정 마스크 코드](models/clinical_v20_missingness.py) · [원 v19 연구](CLINICAL_V19_RESULTS.md).
