# KNHANES 2021–2024 비침습 대사이상 선별 — v19

이번 분석은 19–39세 4,068명(2021: 1,039, 2022: 1,013, 2023: 1,010, 2024: 1,006)의 현재 대사이상 네 소견과 하나 이상 발생 위험을 다룬다. 19개 비침습 입력만 사용하며 검사 수치와 혈압은 학습 정답이다. 사전 고정한 기본 조건은 `basic_clean_native`이다. 표의 세 seed는 모델별 성능의 평균이고 앙상블 예측이 아니다.

**사전 고정 주대조:** 기본 EPF의 4소견 macro AP는 0.346860, 동일 basic-clean LR은 0.357270이었다. EPF−LR 차이는 −0.010410, 고정 OOF 예측의 paired 전체 조사 PSU bootstrap 95% 구간은 [−0.020556, −0.000433]이다. 따라서 주 조건에서 EPF의 우위는 관찰되지 않았다. 두 모델 모두 허리둘레·BMI·WHtR을 포함한 동일 원시 19개 비침습 입력을 받았다.

**고정 조건 탐색:** EPF의 linear anchor는 basic-clean에서 native 대비 macro AP +0.008518 [0.001072, 0.016981], engineered-clean에서 +0.012275 [0.004798, 0.020271]이었다. 같은 engineered-clean 비교에서 MLP도 anchor가 +0.009143 [0.000622, 0.018564]였다. 반면 engineered 표현만 native EPF에 추가한 차이는 +0.002615 [−0.003759, 0.010119]였다. 선형 출발이 두 신경망 모두에 도움이 된 관찰이며, EPF 고유의 장 효과나 파생변수 단독 효과로 귀속하지 않는다. engineered-clean-anchor EPF의 AP 0.361751과 대응 LR의 0.359473은 **주대조가 아닌 조건 탐색값**이다. 별도 사후 matched 비교에서 차이 +0.002277, 조건부 95% 구간 [−0.001596, 0.005739]였으며 0을 포함한다. 이 조건을 시험 결과로 새 우승 모델로 선택하거나 확증적 우월성으로 제시하지 않는다.

engineered-clean-anchor의 15개 EPF 단위 중 fold5의 세 seed는 모두 **최초 epoch(0)를 최선으로 선택**했다. 선택된 모델의 clean fit 부분집합에서 평균 선형 skip RMS는 2.5087, 장 readout RMS는 0.1148이었다. 같은 MLP 조건에서도 3/15개가 초기 모델을 선택했고 readout RMS는 0.0763이었다. validation에 따른 초기 모델 선택은 정상 결과이며 강제로 추가 학습하지 않았다. 이 진단은 앵커 이득을 학습된 EPF 장의 고유 우위로 읽지 말아야 한다는 근거이지, 각 경로의 인과적 기여 분해가 아니다.

## 기본 조건의 선별 성능

| 모델 | 4소견 macro AP ↑ | 4소견 macro Brier ↓ | 원 q0 any AP ↑ | 보정 q any AP ↑ | 원 q0 any Brier ↓ | 보정 q any Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|
| EPF@basic_clean_native | 0.346860 | 0.107605 | 0.686578 | 0.681571 | 0.182976 | 0.180974 |
| MLP@basic_clean_native | 0.349517 | 0.107092 | 0.693746 | 0.687908 | 0.180674 | 0.179341 |
| LR@basic_clean | 0.357270 | 0.107409 | 0.692590 | 0.689132 | 0.181841 | 0.180460 |
| EPF_field_only@basic_clean_native | 0.343236 | 0.107842 | 0.687377 | 0.682119 | 0.182785 | 0.180917 |

EPF·MLP·LR·순수장 EPF 모두 동일한 소견별 v18 단조 보정과 γ 보정 기회를 받았다. γ=0은 원래의 `1−∏(1−p)`를 복원하고, 보정 위험은 Fréchet 하한 `max(p)`와 상한 `min(Σp,1)` 안에 있다. γ는 개인 간 원 q0 순위를 보존하지 않을 수 있다. 따라서 any AP 변화는 공통 보정 효과를 포함하며 EPF 구조 자체의 효과로 분리할 수 없다. γ는 4소견 확률을 바꾸지 않아 4소견 AP에 영향을 주지 않는다. 소견별 단조 보정의 양수 기울기는 **같은 fold의 같은 보정기 안에서만** 순위를 유지한다. 기울기가 0이면 동점이 생길 수 있고, OOF에서는 fold마다 보정 계수가 달라 fold 간 순위와 pooled AP가 바뀔 수 있다. 실제 결합 발생률과 원 q0 평균 차이를 전부 독립성 위반 탓으로 돌리지 않는다.

## 기본 EPF의 소견별 시험 집계

| 시험 입력 | 소견 | 실제 가중률 | 평균 예측 | 평균 위험 차이 | AP ↑ | Brier ↓ | NLL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | high_glucose | 0.1288 | 0.1279 | -0.0009 | 0.3479 | 0.0977 | 0.3246 |
| clean | high_blood_pressure | 0.1010 | 0.1198 | 0.0188 | 0.2522 | 0.0847 | 0.2951 |
| clean | high_triglyceride | 0.1866 | 0.1863 | -0.0002 | 0.4272 | 0.1262 | 0.3956 |
| clean | low_hdl | 0.1638 | 0.1675 | 0.0037 | 0.3601 | 0.1218 | 0.3937 |
| missing20 | high_glucose | 0.1288 | 0.1373 | 0.0085 | 0.2840 | 0.1037 | 0.3466 |
| missing20 | high_blood_pressure | 0.1010 | 0.1269 | 0.0259 | 0.2132 | 0.0881 | 0.3095 |
| missing20 | high_triglyceride | 0.1866 | 0.1846 | -0.0020 | 0.3698 | 0.1362 | 0.4296 |
| missing20 | low_hdl | 0.1638 | 0.1839 | 0.0201 | 0.3033 | 0.1298 | 0.4208 |

| 시험 입력 | 종합 위험 | 실제 가중률 | 평균 예측 | 평균 위험 차이 | AP ↑ | Brier ↓ | NLL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | raw_any | 0.3799 | 0.4290 | 0.0491 | 0.6866 | 0.1830 | 0.5467 |
| clean | adjusted_any | 0.3799 | 0.3877 | 0.0078 | 0.6816 | 0.1810 | 0.5413 |
| missing20 | raw_any | 0.3799 | 0.4606 | 0.0806 | 0.6103 | 0.2110 | 0.6103 |
| missing20 | adjusted_any | 0.3799 | 0.4157 | 0.0357 | 0.6016 | 0.2060 | 0.5981 |

기본 EPF의 γ 평균 -0.20217, 범위 [-0.32434, -0.08646]; 경계 γ=-1/0건, γ=+1/0건. calibration 역할 최소 표본 275명, 최소 유효 표본수 220.7, 최소 종합 양성 107명이다. 아래 수치는 fold·seed별 calibration 역할 집계의 평균이며 전체 행을 다시 합친 단일 보정치는 아니다. 계수 경계 접촉 수는 집계 보고서 `clinical_v19_report.json`의 `calibration_role`에 있다. 보정 역할과 시험 역할은 분리했다.

| 보정 역할 위험 | 실제 가중률 | 평균 예측 | 평균 위험 차이 | Brier | NLL |
|---|---:|---:|---:|---:|---:|
| high_glucose | 0.1283 | 0.1283 | -0.0000 | 0.0962 | 0.3202 |
| high_blood_pressure | 0.1198 | 0.1198 | 0.0000 | 0.0940 | 0.3153 |
| high_triglyceride | 0.1866 | 0.1866 | -0.0000 | 0.1283 | 0.4040 |
| low_hdl | 0.1663 | 0.1663 | -0.0000 | 0.1233 | 0.3939 |
| raw_any | 0.3916 | 0.4280 | 0.0364 | 0.1887 | 0.5609 |
| adjusted_any | 0.3916 | 0.3871 | -0.0045 | 0.1863 | 0.5541 |

### 원결과의 별도 보정 해석 감사

동결된 OOF의 **모델 평균**을 원 q0와 γ 보정 q로 비교하면 clean 26개 모델의 종합위험 AP·AUC가 모두 감소했고, 13 view×26모델=338개 평균 비교의 AP도 모두 감소했다. AUC 평균은 `MLP_joint`의 `body_missing` view 한 건만 +0.000103 예외였다. 개별 seed/view에서는 AP 7건, AUC 5건 증가가 있어 모든 seed에 같은 방향이라고 말하지 않는다. LR은 seed가 없어서 seed별 정렬 표에서 같은 값을 세 번 반복한 것이며 독립 실험 3건이 아니다. γ는 네 소견 확률을 변경하지 않으므로 이 종합 위험 순위 변화는 모델 학습의 구조적 판별력 향상으로 귀속되지 않는다.

절댓값 평균 위험 차이의 paired PSU bootstrap 구간은 **계산 가능**하다. 다만 signed gap이 0을 지나는 절댓값 변환 때문에 분포가 비대칭일 수 있고, 일부 밀도 추정에서는 여러 봉우리가 보인다. 동결된 44개 절댓값-gap 대조 구간 중 0을 벗어난 것은 없었다. 위험 보정 해석에는 방향을 가진 signed gap과 proper score인 Brier를 함께 보고, 절댓값-gap만으로 모델 우열을 정하지 않는다. clean 26모델 표, 실제 세 seed 점수·paired 차이, 분포 진단은 [보충 감사 집계](clinical_v19_interpretation_audit.json)와 실행 노트북의 별도 감사 셀에 보존한다. 이 감사는 원 44개 사전 대조를 변경하지 않는다.

**사후 보정 방식 probe:** 원 q0, 원래의 Fréchet 경계 γ, 새 양수기울기 `sigmoid(a·logit(q0)+b)`를 동결된 35개 모델 단위의 clean view에서만 비교했다. 새 신경망 학습·후보 선택·외부 검증은 없다. EPF의 AP는 각각 0.686578/0.681571/0.678965, Brier는 0.182976/0.180974/0.182223였다. MLP·LR에서도 γ의 test Brier·NLL·signed gap이 새 sigmoid보다 좋았다. sigmoid의 합집합 Fréchet 경계 위반 비율은 EPF 14.63%, MLP 9.76%, LR 5.56%였다. 양수 기울기는 fold 안 순위를 보존하지만 fold별 사상이 다른 pooled OOF AP는 변할 수 있다. q0와 새 sigmoid의 threshold 역할 컷오프 운영점은 동일했다. 이는 단조 보정이 항상 성능을 개선한다는 근거가 아니며, 다섯 번째 직접 any head나 더 큰 cross-fit calibration은 **이번에 구현하지 않은 별도 실험**이다. [집계 probe](clinical_v19_union_calibration_probe.json)는 원 v19 주대조와 분리한다.

보정 순서는 **첫째, 네 소견의 점수를 각각 단조 확률로 보정**, **둘째, 그 네 보정 확률에서 종합 위험 q0를 계산한 뒤 γ 또는 사후 probe의 C를 적용**하는 두 단계다. q0는 '모든 보정 전 원확률'이 아니다.

첫 단계만 재계산한 별도 감사에서는 EPF−LR 4소견 macro AP 차이가 raw sigmoid 기준 −0.017104에서 저장된 소견별 보정 후 −0.010410으로 바뀌었다. MLP−LR도 −0.013620에서 −0.007753으로 변했지만 순위 역전은 없다. 양수 기울기의 140개 fold×소견 내부 AP/AUC 검사는 변화 0, 기울기0·새 동점도 0이었다. pooled OOF 순위는 fold별 보정기 때문에 달라질 수 있다. 세 모델 모두 소견별 보정 후 macro Brier·NLL이 **악화**했으므로 한 지표의 gap 감소를 전체 확률 품질 개선으로 부르지 않는다. CPU 재계산과 저장 확률의 최대 차이는 7.77e−7이다. [소견별 보정 감사](clinical_v19_component_calibration_audit.json)는 γ 단계와 분리한 집계다.

## 별도 threshold 역할의 고정 컷오프

| 위험 | 개발 민감도 목표 | 시험 민감도 | 시험 특이도 | 시험 의뢰 비율 |
|---|---:|---:|---:|---:|
| raw_any | 0.9 | 0.8742 | 0.4631 | 0.6651 |
| raw_any | 0.95 | 0.9300 | 0.3176 | 0.7765 |
| adjusted_any | 0.9 | 0.8738 | 0.4666 | 0.6628 |
| adjusted_any | 0.95 | 0.9300 | 0.3208 | 0.7745 |

원 q0와 보정 q는 각기 threshold 역할에서 90%/95% 목표 컷오프를 선택했다. 시험에서 실제 달성한 민감도를 표기했으며 목표 달성을 보장하지 않는다. 동일 민감도 시험 ROC 특이도는 `clinical_v19_report.json`에 별도 기술적 비교로 기록했고 운영 컷오프와 혼동하지 않는다.

별도 역할 감사에서 각 fold 안의 행·PSU 겹침은 0이고 test 사람은 한 번씩 포함됨을 확인했다. 다른 fold와 세 seed 사이에는 같은 사람을 의도적으로 재사용하므로 역할별 수를 더해 독립 표본으로 해석하지 않는다. `wt_itvex/4`라는 공통 재척도는 가중 유병률과 민감도를 바꾸지 않는다. 조사 가중 threshold 역할에서 `risk >= cutoff`를 적용하며 목표 민감도를 충족하는 가장 높은 고유 위험값을 선택한다. 기본 EPF의 저장 컷오프 10개를 정확히 재현했다. **26개 모델 모두** clean test에서 두 목표90/95를 실제로 달성하지 못했다. 이는 개발 목표와 외부 test 성적의 차이이지 컷오프를 test로 다시 고를 근거가 아니다.

BP 소견의 test 실제 가중률은 0.10103, 기본 EPF의 소견별 보정 전 평균은 0.10375, 보정 후 0.11983이었다. calibration 역할의 보정 후 평균도 0.11976으로 test와 역할 유병률이 다르다. 26개 모델의 test BP 평균 예측 범위 0.11969–0.12200은 모두 실제율보다 높았다. 이 편차를 종합 위험 γ나 소견 간 독립성만으로 설명하지 않는다. 결과 표의 최소 종합 양성 107명은 **calibration 역할**의 수이며 해당 fold threshold 역할 양성은 115명이다. [역할·컷오프 감사 집계](clinical_v19_role_audit.json)에 자세한 조건이 있다.

## 여덟 조건과 조건부 효과

| 모델·조건 | clean 4소견 macro AP | clean 보정 any AP | 누락20 4소견 macro AP |
|---|---:|---:|---:|
| EPF@basic_clean_native | 0.346860 | 0.681571 | 0.292566 |
| EPF@basic_clean_linear_anchor | 0.355378 | 0.690841 | 0.314852 |
| EPF@basic_paired20_native | 0.349780 | 0.679047 | 0.328373 |
| EPF@basic_paired20_linear_anchor | 0.355166 | 0.685024 | 0.335173 |
| EPF@engineered_clean_native | 0.349475 | 0.689114 | 0.297664 |
| EPF@engineered_clean_linear_anchor | 0.361751 | 0.693233 | 0.320196 |
| EPF@engineered_paired20_native | 0.350648 | 0.687964 | 0.331991 |
| EPF@engineered_paired20_linear_anchor | 0.361551 | 0.689068 | 0.343088 |
| MLP@basic_clean_native | 0.349517 | 0.687908 | 0.308541 |
| MLP@basic_clean_linear_anchor | 0.355756 | 0.687860 | 0.313496 |
| MLP@basic_paired20_native | 0.352656 | 0.686983 | 0.337056 |
| MLP@basic_paired20_linear_anchor | 0.356361 | 0.683018 | 0.336455 |
| MLP@engineered_clean_native | 0.350439 | 0.687643 | 0.306241 |
| MLP@engineered_clean_linear_anchor | 0.359582 | 0.691298 | 0.318016 |
| MLP@engineered_paired20_native | 0.353182 | 0.688039 | 0.333374 |
| MLP@engineered_paired20_linear_anchor | 0.360462 | 0.687998 | 0.340039 |

56열 basic과 72열 engineered는 같은 원시 19개 입력에서 나온다. paired20은 fit 사람의 clean·missing20_0 두 보기에 가중치를 반씩 부여했다. 모든 validation과 calibration은 clean이며, 모델 선택은 기본 조건의 seed42에서만 수행했다. 여덟 조건에서 같은 선택 구성을 고정했으므로 OFAT 차이와 상호작용은 이 구성에 조건부이며 각 조건의 독립적인 최적 성능은 아니다. 시험 점수로 우승 조건을 고르지 않았다.

**v19 누락 뷰의 역사적 한계:** 표의 `missing20`은 기존 마스크 생성기의 `legacyWCp` 결과다. 허리둘레 WC와 파생 WHtR에 서로 독립된 p=0.20 추첨을 OR로 적용해 이 그룹의 이론적 누락 확률이 **0.36**이었다. `missing20_0`의 실측 WC 그룹 추첨률은 **35.816%**, 초기 관측값의 신규 결측률은 **35.915%**다. 한 번의 허리 추첨을 공유한 교정 view의 대응값 **19.322%/19.474%**는 현재 마스크 강도 감사이며 아직 교정 모델의 성능이 아니다. 따라서 v19과 교정 재학습 결과를 '동일 20% 누락 강도'로 비교하거나 이 표를 v20 성과로 바꾸지 않는다. `BMI_noise25`도 사람의 25%를 선택한 뜻이 아니라 관측 BMI 전부에 fit 역할 BMI 표준편차의 0.25배, 즉 Gaussian σ≈1.0717 kg/m²를 적용한 스트레스다. [누락 생성 감사 집계](clinical_v19_missingness_audit.json)에 검증값이 있다.

체중·키의 상대 변동을 따로 시험한 [체격측정 일관성 probe](clinical_v19_anthropometric_noise_probe.json)는 source-only 체중·키를 고정 합성 CV로 변화시키고 BMI·WHtR을 다시 계산했다. raw19 입력에 체중·키를 새로 넣거나 모델을 재학습하지 않았다. `joint_weightCV25_heightCV3`의 보정 any AP 변화는 EPF −0.01507, MLP −0.01276, LR −0.00669이고 Brier 변화는 각각 +0.00399/+0.00473/+0.00155였다. `weightCV25` 단일 추첨에서 BMI 절대 변화 중앙값 3.87, p95 11.89 kg/m²로 기존 `BMI_noise25` σ≈1.07과 다른 강도다. 이 합성 CV를 실제 자가측정 오류 분포나 현장 성능으로 주장하지 않는다. 노트북에는 고정 컷오프의 실제 민감도·의뢰 비율과 물리식 잔차도 따로 표시한다.

## 로지스틱 회귀의 네 조건

| LR 조건 | clean 4소견 macro AP ↑ | clean macro Brier ↓ | clean 원 q0 any AP ↑ | clean 보정 q any AP ↑ | 누락20 macro AP ↑ |
|---|---:|---:|---:|---:|---:|
| LR@basic_clean | 0.357270 | 0.107409 | 0.692590 | 0.689132 | 0.316833 |
| LR@engineered_clean | 0.359473 | 0.106657 | 0.695652 | 0.691302 | 0.319110 |
| LR@basic_paired20 | 0.357659 | 0.107590 | 0.687480 | 0.683686 | 0.336580 |
| LR@engineered_paired20 | 0.359566 | 0.106864 | 0.691686 | 0.687532 | 0.339695 |

basic(56열)과 engineered(72열)를 각각 clean·paired20 fit에서 비교했다. 네 LR 조건은 각각 clean validation에서 정규화 계수 C를 선택했다. 아래 차이는 engineered−basic이며 같은 test 사람·PSU를 짝지은 고정 예측의 조건부 구간이다. 시험 성적으로 LR 조건을 선택하지 않았다.

| fit 조건 | 비교 지표 | engineered−basic | paired PSU 95% 구간 |
|---|---|---:|---:|
| clean | component_macro_ap_difference | 0.002203 | [-0.003726, 0.007822] |
| clean | adjusted_any_ap_difference | 0.002171 | [-0.004412, 0.008465] |
| paired20 | component_macro_ap_difference | 0.001906 | [-0.003942, 0.007659] |
| paired20 | adjusted_any_ap_difference | 0.003847 | [-0.002292, 0.010115] |

| 주대조 | 지표 | 추정차 | paired PSU 95% 구간 | 유효 반복 |
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

2,000회 전체 조사 틀 PSU bootstrap은 고정 OOF 예측의 paired 조건부 구간이다. 모델 재학습, 다중 비교 보정, 외부 코호트 검증은 포함하지 않는다. 동일한 KNHANES 2021–2024 코호트와 누락 마스크를 v16/v18에서 이미 탐색한 후속 탐색 연구이므로 작은 차이를 확정적 우월성으로 해석할 수 없다. 모든 사전 지정 factor·상호작용 차이와 구간은 `clinical_v19_report.json`에 있다.

## 검사 수치 확장

| 모델 | 검사 수치 | MAE | RMSE | 기준값 주변 MAE | NMSE |
|---|---|---:|---:|---:|---:|
| EPF_regression | 혈당 mg/dL | 6.426 | 12.294 | 5.160 | 0.9127 |
| EPF_regression | 수축기혈압 mmHg | 7.162 | 9.279 | 9.122 | 0.6489 |
| EPF_regression | 이완기혈압 mmHg | 6.036 | 7.820 | 9.958 | 0.7999 |
| EPF_regression | 중성지방 mg/dL | 41.182 | 73.791 | 39.865 | 0.8279 |
| EPF_regression | HDL mg/dL | 9.328 | 11.975 | 9.523 | 0.6834 |
| EPF_joint | 혈당 mg/dL | 6.450 | 12.262 | 5.238 | 0.9081 |
| EPF_joint | 수축기혈압 mmHg | 7.089 | 9.211 | 9.047 | 0.6394 |
| EPF_joint | 이완기혈압 mmHg | 6.035 | 7.822 | 10.015 | 0.8003 |
| EPF_joint | 중성지방 mg/dL | 41.250 | 73.780 | 40.155 | 0.8277 |
| EPF_joint | HDL mg/dL | 9.310 | 11.955 | 9.407 | 0.6811 |
| MLP_regression | 혈당 mg/dL | 6.355 | 12.122 | 5.016 | 0.8874 |
| MLP_regression | 수축기혈압 mmHg | 7.027 | 9.135 | 8.793 | 0.6289 |
| MLP_regression | 이완기혈압 mmHg | 6.012 | 7.785 | 9.964 | 0.7926 |
| MLP_regression | 중성지방 mg/dL | 40.974 | 73.643 | 39.516 | 0.8246 |
| MLP_regression | HDL mg/dL | 9.291 | 11.921 | 9.518 | 0.6772 |
| MLP_joint | 혈당 mg/dL | 6.369 | 12.183 | 5.025 | 0.8964 |
| MLP_joint | 수축기혈압 mmHg | 7.057 | 9.178 | 8.764 | 0.6349 |
| MLP_joint | 이완기혈압 mmHg | 6.011 | 7.788 | 9.927 | 0.7933 |
| MLP_joint | 중성지방 mg/dL | 40.946 | 73.405 | 39.112 | 0.8193 |
| MLP_joint | HDL mg/dL | 9.261 | 11.916 | 9.296 | 0.6767 |
| Ridge@basic_clean | 혈당 mg/dL | 6.364 | 12.182 | 4.798 | 0.8962 |
| Ridge@basic_clean | 수축기혈압 mmHg | 6.937 | 9.060 | 8.812 | 0.6186 |
| Ridge@basic_clean | 이완기혈압 mmHg | 6.006 | 7.756 | 9.963 | 0.7868 |
| Ridge@basic_clean | 중성지방 mg/dL | 41.407 | 74.484 | 40.305 | 0.8435 |
| Ridge@basic_clean | HDL mg/dL | 9.303 | 11.901 | 9.750 | 0.6750 |

회귀 전용은 5개 출력, 공동 학습은 9개 출력으로 기본 분류 모델에서 출력 수와 목표가 모두 바뀌었다. 같은 기본 선택 backbone 구성을 이어받지만 별도로 학습했다. 오차는 원 단위이며 NMSE는 각 수치의 조사 가중 분산으로 나눈 MSE다. 기준값 주변 폭은 v18과 동일하게 혈당±10, 수축기혈압±10, 이완기혈압±5, 중성지방±30, 성별 HDL 기준±5다. 수치 추정은 실제 검사를 대체하지 않는다.

**별도 사후 paired 대조:** MLP 수치 전용−MLP 분류 전용의 4소견 macro AP 차이는 +0.012887 [0.003744, 0.021409]였지만, 보정 any Brier 개선량은 −0.002473 [−0.004438, −0.000467]로 악화 방향이다. EPF 수치 전용−EPF 분류 전용 AP 차이는 +0.005470 [−0.002636, 0.014904]였고 보정 any Brier 개선량은 −0.003530 [−0.005769, −0.001354]였다. MLP 수치 전용−Ridge의 macro NMSE 개선량 +0.001859 [−0.002352, 0.005763]은 0을 포함하고, EPF 수치 전용−Ridge는 −0.010534 [−0.016441, −0.005497]였다. 이 여섯 대조·26구간은 고정 OOF의 사후 탐색이며 다중 비교·전체 재학습 불확실성이 포함되지 않는다. 4출력 분류와 5출력 수치 학습의 연속 정답 정보도 달라 순수 구조 비교가 아니다. [보충 확장 대조](clinical_v19_extension_probe.json)를 원 44개 대조에 합치지 않는다.

## 학습 기록과 재현 경계

| 기본 비교 | 완료 unit | 총 fit 초 | 학습 가능한 파라미터 범위 | 초기 모델 선택 | 실제 업데이트 | 클리핑 비율 |
|---|---:|---:|---:|---:|---:|---:|
| EPF@basic_clean_native | 15 | 235.1 | 4043–16267 | 0 | 8166 | 0.0000 |
| MLP@basic_clean_native | 15 | 16.1 | 4264–4264 | 0 | 4790 | 0.0000 |
| LR@basic_clean | 5 | — | 228–228 | — | — | — |
| EPF_field_only@basic_clean_native | 15 | 203.5 | 3815–16039 | 0 | 8321 | 0.0000 |

`TUNE_EXECUTION.json`에는 공유 GPU에서 튜닝 worker 최대 2개가 기록됐다. 표의 `fit_seconds`는 선택된 unit의 공유 장치 실행시간이며 전체 후보 탐색 비용을 합산하지 않는다. 모델 간 공정한 속도 순위가 아니다.

| 기본 비교 | 선택 epoch 중앙값 | 종료 사유 건수 | event rate | field energy | plastic norm | 선형 skip RMS | 장/MLP readout RMS |
|---|---:|---|---:|---:|---:|---:|---:|
| EPF@basic_clean_native | 26.0 | validation_plateau: 15 | 0.2911 | 0.0295 | 0.0957 | 0.3488 | 2.6548 |
| MLP@basic_clean_native | 25.0 | validation_plateau: 15 | — | — | — | 0.6355 | 1.9317 |
| LR@basic_clean | — | — | — | — | — | — | — |
| EPF_field_only@basic_clean_native | 26.0 | validation_plateau: 15 | 0.3010 | 0.0286 | 0.0771 | 0.0000 | 2.8175 |

분류 모델의 출력은 4개이며 위 수는 해당 출력 구조에서 학습 가능한 파라미터 수다. 코어 활동은 선택된 checkpoint를 clean fit 부분집합에서 측정한 기술적 값이다. 장/MLP readout과 skip RMS도 인과적 기여도가 아니다. 공개 `clinical_v19_report.json`에는 학습 집계만 있으며 unit별 경사 노름·파라미터 변화·학습률 이력은 로컬 연구 산출물에만 보존한다. 원본 학습·분석 소스 manifest는 로컬에 고정했고, 공개 [소스 manifest 요약](clinical_v19_source_manifest.json)에는 코드 해시와 원본 manifest의 SHA-256만 싣는다.

이 결과는 청년층 비침습 선별의 탐색적 연구 결과다. 2021–2024는 동일인의 추적 시계열이 아니다. 원자료·개인별 예측·역할 인덱스는 이 문서와 노트북에 포함하지 않았다.

## 사전 지정 공개 예제의 추론 프로파일

아래 시간은 미리 지정한 fold1 모델 예제 5개를 **공개 합성 입력**으로 측정한 CPU 1스레드·batch1 중앙값이다. 신경망은 seed42, LR은 seed가 없다. 8회 예열 후 40회 계측했다. `raw19→결과 dict`는 입력 인코딩·모델·보정·출력 구성을 포함하고, `인코딩 후 forward`는 모델 계산만 잰다. 파일 로딩·CLI 시작 시간은 포함하지 않는다.

| 공개 예제 | 학습 가능한 파라미터 | raw19→결과 dict 중앙값(ms) | 인코딩 후 forward 중앙값(ms) |
|---|---:|---:|---:|
| EPF basic-clean-native | 16,267 | 1.65350 | 1.12155 |
| 순수장 EPF basic-clean-native | 3,815 | 1.24785 | 0.72635 |
| MLP basic-clean-native | 4,264 | 0.50515 | 0.03220 |
| LR basic-clean | 228 | 0.40715 | 0.00200 |
| LR engineered-clean | 292 | 0.39075 | 0.00200 |

같은 프로파일의 CUDA 측정은 장치별 별도 층에서 수행했고 각 호출 전후 동기화했다. CPU·CUDA를 한 속도 순위로 합치지 않는다. 단건 CUDA 지연은 이 환경에서 오히려 길 수 있다. 본문 표의 파라미터 수는 이 **fold1 공개 예제**에 해당한다. 전체 fold의 EPF 주조건 학습 가능 파라미터 범위는 4,043–16,267개다. 튜닝 8후보×5fold의 fit 초 합계는 EPF 568.7초, 순수장 EPF 572.6초, MLP 44.0초이지만 최대 2개 GPU worker가 장치를 공유한 계산량이며 실제 경과시간·공정한 모델 속도 비교가 아니다. 자세한 장치별 batch1/1024 분포와 경계는 [집계 프로파일](clinical_v19_profile.json)에 있다.

## 검증과 자료 경계

학습 단위 340개를 완료한 뒤, 신경망 4,095개·선형 325개, 합계 **4,420개 test view**의 저장 예측·보정·컷오프를 재계산해 최대 절대오차 0으로 확인했다. 공개 예제 7개도 각 13개 입력 view에서 CPU와 CUDA를 검증했고, 공개 가중치·전처리·코드 해시를 확인했다. 예제 공개는 전체 4,068명 재학습이나 독립 코호트 검증을 뜻하지 않는다. 로컬 테스트 173개와 공개 테스트 18개가 통과했다.

공개 [검증 요약](clinical_v19_verification.json)의 `execution.tuning`은 8후보 탐색을, `execution.evaluation`은 선택 설정 검토 후 **전체 340개 단위 실행**을 구분해 기록한다. 두 단계 모두 공유 GPU worker 최대 2개로 수행했고, 전체 실행 중간에 시험 결과를 조회해 조건을 바꾸지 않았다. 실행시간은 공유 장치의 기록으로만 해석한다.

[전체 집계 JSON](clinical_v19_report.json) · [실행된 집계 노트북](clinical_v19_report.ipynb) · [소견별 보정 감사](clinical_v19_component_calibration_audit.json) · [종합위험 보정 감사](clinical_v19_interpretation_audit.json) · [역할·컷오프 감사](clinical_v19_role_audit.json) · [누락 생성 감사](clinical_v19_missingness_audit.json) · [사후 보정 방식 probe](clinical_v19_union_calibration_probe.json) · [체격측정 합성 probe](clinical_v19_anthropometric_noise_probe.json) · [사후 수치확장 대조](clinical_v19_extension_probe.json) · [학습·공개 검증 요약](clinical_v19_verification.json) · [소스 manifest 요약](clinical_v19_source_manifest.json) · [별도 사후 matched 비교](clinical_v19_posthoc_comparison.json). 공개본에는 원자료·개인 파일명·절대경로를 싣지 않는다.
