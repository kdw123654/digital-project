# Clinical v22: 입력·목적·전이 탐색 결과

상태: complete. 2021–2024년 국민건강영양조사 19–39세 4,068명에서 PSU 분리 5폴드 고정 OOF 예측을 평가했다. 전체는 `wt_itvex`, 영양 참여자 3,784명 별도 도메인은 `wt_tot`로 평가했다. 연구는 현재 소견의 횡단면 선별 탐색이며 임상적 유효성·장래 발병 예측을 입증하지 않는다.

## 사전 지정 대조

지표는 **보정 전 네 소견 macro AP** 차이(왼쪽 − 오른쪽)다. 같은 full 또는 nutrition 대상·가중치 안에서만 대조한다. 두 영역에 동일한 full-frame PSU 재표집 배율을 적용했다. 세 seed는 독립 fit의 점수 변동이며 앙상블이 아니다.

| 대조 | AP 차이 | 개별 95% PSU 구간 | 16대조 동시 95% 근사구간 | 세 seed 차이 SD |
|---|---:|---:|---:|---:|
| full/family_EPF_vs_LR | 0.00131 | [-0.00170, 0.00420] | [-0.00313, 0.00575] | 0.00132 |
| full/family_EPF_vs_MLP | 0.00053 | [-0.00240, 0.00315] | [-0.00359, 0.00465] | 0.00164 |
| full/family_MLP_vs_LR | 0.00078 | [-0.00028, 0.00190] | [-0.00084, 0.00241] | 0.00032 |
| full/input_EPF | -0.00151 | [-0.00921, 0.00532] | [-0.01204, 0.00901] | 0.00129 |
| full/input_LR | -0.00456 | [-0.01142, 0.00190] | [-0.01442, 0.00530] | 0.00000 |
| full/input_MLP | -0.00270 | [-0.00946, 0.00372] | [-0.01243, 0.00704] | 0.00022 |
| full/mode_full10_vs_binary4_EPF | 0.00012 | [-0.00131, 0.00163] | [-0.00208, 0.00231] | 0.00154 |
| full/mode_full10_vs_binary4_MLP | -0.00055 | [-0.00193, 0.00085] | [-0.00262, 0.00152] | 0.00042 |
| full/transfer_vs_scratch_EPF | -0.00092 | [-0.00305, 0.00095] | [-0.00391, 0.00206] | 0.00156 |
| full/transfer_vs_scratch_MLP | -0.00000 | [-0.00095, 0.00095] | [-0.00142, 0.00141] | 0.00082 |
| nutrition/added_vs_base19_EPF | 0.00502 | [-0.00435, 0.01407] | [-0.00868, 0.01873] | 0.00401 |
| nutrition/added_vs_base19_LR | 0.00041 | [-0.00833, 0.00837] | [-0.01183, 0.01265] | 0.00000 |
| nutrition/added_vs_base19_MLP | 0.00237 | [-0.00601, 0.01074] | [-0.01024, 0.01497] | 0.00197 |
| nutrition/added_vs_expanded_EPF | 0.00299 | [-0.00162, 0.00763] | [-0.00387, 0.00985] | 0.00148 |
| nutrition/added_vs_expanded_LR | 0.00257 | [-0.00131, 0.00635] | [-0.00313, 0.00828] | 0.00000 |
| nutrition/added_vs_expanded_MLP | 0.00159 | [-0.00243, 0.00551] | [-0.00427, 0.00745] | 0.00032 |

2,000/2,000개 공통 PSU draw가 유효했다. 동시구간은 16개 사전대조 중 표준편차가 양수인 16개 차이의 중심화 표준화 max|T| 95% 분위수를 공통 적용했다. 고정 예측이 정확히 같은 퇴화 대조는 `[]`이며 점구간 [0,0]은 모델 효과가 없다는 결론이 아니다. 모두 **고정된 OOF 예측에 조건부인 근사 구간**이며 재학습·선택·반복 연구의 변동을 포함하지 않는다. 다중비교 또는 임상 우월성의 보증으로 해석하지 않는다.

## 모든 모델의 clean 시험 점추정

회귀5 raw는 임계값 거리 **순위 점수**여서 raw AP/AUC만 적용한다. Brier/NLL은 calibration 역할에서 적합한 확률에만 적용한다. `state16`의 공동 softmax 분포와 직접 any는 네 주변확률의 독립 결합이나 일반 1차원 Platt 보정과 다르다. AP/BCE/MSE/ANY_BCE는 validation checkpoint 선택 정책이다.

| 모델 | 대상 / 입력 | 목적 | 계열 / 정책 | raw macro AP | 보정 후 macro Brier | 저장 매개변수 | 학습 초 | 추론 ms/1000명 |
|---|---|---|---|---:|---:|---:|---:|---:|
| full_base19_binary4_AP_EPF | full / base19 | binary4 | EPF / AP | 0.36308 | 0.10622 | 4719 | 9.6 | 13.82 |
| full_base19_binary4_AP_LR | full / base19 | binary4 | LR / AP | 0.36125 | 0.10653 | 292 | — | 0.07 |
| full_base19_binary4_AP_MLP | full / base19 | binary4 | MLP / AP | 0.35840 | 0.10644 | 18450 | 1.7 | 1.18 |
| full_base19_binary4_BCE_EPF | full / base19 | binary4 | EPF / BCE | 0.36658 | 0.10603 | 9903 | 10.6 | 28.07 |
| full_base19_binary4_BCE_LR | full / base19 | binary4 | LR / BCE | 0.36832 | 0.10603 | 292 | — | 0.07 |
| full_base19_binary4_BCE_MLP | full / base19 | binary4 | MLP / BCE | 0.36724 | 0.10612 | 22815 | 2.0 | 1.32 |
| full_base19_regression5_MSE_Ridge | full / base19 | regression5 | Ridge / MSE | 0.36857 | 0.10626 | 365 | — | 0.08 |
| full_base19_union1_ANY_BCE_LR | full / base19 | union1 | LR / ANY_BCE | — | — | 73 | — | 0.04 |
| full_expanded_aux9_AP_EPF | full / expanded | aux9 | EPF / AP | 0.36419 | 0.10700 | 15213 | 11.8 | 36.37 |
| full_expanded_aux9_AP_MLP | full / expanded | aux9 | MLP / AP | 0.36024 | 0.10740 | 24016 | 1.9 | 1.50 |
| full_expanded_aux9_BCE_EPF | full / expanded | aux9 | EPF / BCE | 0.36486 | 0.10681 | 15213 | 13.4 | 37.49 |
| full_expanded_aux9_BCE_MLP | full / expanded | aux9 | MLP / BCE | 0.36439 | 0.10686 | 19050 | 3.6 | 1.29 |
| full_expanded_binary4_AP_EPF | full / expanded | binary4 | EPF / AP | 0.36369 | 0.10694 | 11482 | 10.7 | 27.85 |
| full_expanded_binary4_AP_LR | full / expanded | binary4 | LR / AP | 0.35966 | 0.10739 | 460 | — | 0.09 |
| full_expanded_binary4_AP_MLP | full / expanded | binary4 | MLP / AP | 0.35581 | 0.10753 | 22919 | 1.7 | 1.31 |
| full_expanded_binary4_BCE_EPF | full / expanded | binary4 | EPF / BCE | 0.36507 | 0.10674 | 8689 | 10.9 | 20.88 |
| full_expanded_binary4_BCE_LR | full / expanded | binary4 | LR / BCE | 0.36376 | 0.10702 | 460 | — | 0.09 |
| full_expanded_binary4_BCE_MLP | full / expanded | binary4 | MLP / BCE | 0.36454 | 0.10682 | 18017 | 2.3 | 1.16 |
| full_expanded_focal4_AP_EPF | full / expanded | focal4 | EPF / AP | 0.35002 | 0.10741 | 11482 | 11.1 | 27.75 |
| full_expanded_focal4_AP_MLP | full / expanded | focal4 | MLP / AP | 0.35444 | 0.10704 | 22919 | 2.2 | 1.30 |
| full_expanded_focal4_BCE_EPF | full / expanded | focal4 | EPF / BCE | 0.36334 | 0.10706 | 8689 | 9.3 | 20.92 |
| full_expanded_focal4_BCE_MLP | full / expanded | focal4 | MLP / BCE | 0.36318 | 0.10702 | 13114 | 2.0 | 0.98 |
| full_expanded_focal_labelweight4_AP_EPF | full / expanded | focal_labelweight4 | EPF / AP | 0.35100 | 0.10737 | 11482 | 11.2 | 27.80 |
| full_expanded_focal_labelweight4_AP_MLP | full / expanded | focal_labelweight4 | MLP / AP | 0.35154 | 0.10705 | 18017 | 2.0 | 1.15 |
| full_expanded_focal_labelweight4_BCE_EPF | full / expanded | focal_labelweight4 | EPF / BCE | 0.36328 | 0.10706 | 8689 | 9.1 | 20.94 |
| full_expanded_focal_labelweight4_BCE_MLP | full / expanded | focal_labelweight4 | MLP / BCE | 0.36317 | 0.10702 | 13114 | 2.0 | 1.02 |
| full_expanded_full10_ANY_BCE_EPF | full / expanded | full10 | EPF / ANY_BCE | 0.36246 | 0.10709 | 9659 | 11.3 | 21.13 |
| full_expanded_full10_ANY_BCE_MLP | full / expanded | full10 | MLP / ANY_BCE | 0.36342 | 0.10701 | 34194 | 1.8 | 1.85 |
| full_expanded_full10_AP_EPF | full / expanded | full10 | EPF / AP | 0.36290 | 0.10702 | 12530 | 15.8 | 27.55 |
| full_expanded_full10_AP_MLP | full / expanded | full10 | MLP / AP | 0.35813 | 0.10749 | 24236 | 2.1 | 1.50 |
| full_expanded_full10_BCE_EPF | full / expanded | full10 | EPF / BCE | 0.36518 | 0.10677 | 18271 | 15.6 | 42.42 |
| full_expanded_full10_BCE_MLP | full / expanded | full10 | MLP / BCE | 0.36399 | 0.10689 | 24236 | 3.4 | 1.47 |
| full_expanded_labelweight4_AP_EPF | full / expanded | labelweight4 | EPF / AP | 0.36286 | 0.10699 | 11482 | 11.4 | 28.55 |
| full_expanded_labelweight4_AP_MLP | full / expanded | labelweight4 | MLP / AP | 0.35683 | 0.10749 | 22919 | 1.7 | 1.28 |
| full_expanded_labelweight4_BCE_EPF | full / expanded | labelweight4 | EPF / BCE | 0.36373 | 0.10691 | 11482 | 10.6 | 27.69 |
| full_expanded_labelweight4_BCE_MLP | full / expanded | labelweight4 | MLP / BCE | 0.36470 | 0.10683 | 13114 | 3.2 | 0.99 |
| full_expanded_regression5_AP_EPF | full / expanded | regression5 | EPF / AP | 0.36871 | 0.10653 | 11657 | 12.6 | 27.45 |
| full_expanded_regression5_AP_MLP | full / expanded | regression5 | MLP / AP | 0.36712 | 0.10657 | 18223 | 1.7 | 1.08 |
| full_expanded_regression5_MSE_EPF | full / expanded | regression5 | EPF / MSE | 0.37387 | 0.10615 | 11657 | 11.6 | 27.90 |
| full_expanded_regression5_MSE_MLP | full / expanded | regression5 | MLP / MSE | 0.37030 | 0.10636 | 13308 | 2.0 | 0.89 |
| full_expanded_regression5_MSE_Ridge | full / expanded | regression5 | Ridge / MSE | 0.36961 | 0.10642 | 575 | — | 0.11 |
| full_expanded_state16_ANY_BCE_EPF | full / expanded | state16 | EPF / ANY_BCE | 0.36547 | 0.10610 | 15145 | 14.0 | 34.96 |
| full_expanded_state16_ANY_BCE_MLP | full / expanded | state16 | MLP / ANY_BCE | 0.36463 | 0.10611 | 19116 | 2.3 | 1.67 |
| full_expanded_state16_AP_EPF | full / expanded | state16 | EPF / AP | 0.36307 | 0.10637 | 15145 | 16.9 | 35.02 |
| full_expanded_state16_AP_MLP | full / expanded | state16 | MLP / AP | 0.36371 | 0.10631 | 19116 | 2.1 | 1.76 |
| full_expanded_state16_BCE_EPF | full / expanded | state16 | EPF / BCE | 0.36269 | 0.10641 | 12197 | 14.5 | 28.38 |
| full_expanded_state16_BCE_MLP | full / expanded | state16 | MLP / BCE | 0.36451 | 0.10611 | 24172 | 2.1 | 1.88 |
| full_expanded_union1_ANY_BCE_LR | full / expanded | union1 | LR / ANY_BCE | — | — | 115 | — | 0.05 |
| full_expanded_union5_ANY_BCE_EPF | full / expanded | union5 | EPF / ANY_BCE | 0.36214 | 0.10711 | 8850 | 10.4 | 20.87 |
| full_expanded_union5_ANY_BCE_MLP | full / expanded | union5 | MLP / ANY_BCE | 0.36404 | 0.10699 | 23139 | 1.9 | 1.45 |
| full_expanded_union5_AP_EPF | full / expanded | union5 | EPF / AP | 0.35838 | 0.10731 | 6044 | 18.5 | 14.08 |
| full_expanded_union5_AP_MLP | full / expanded | union5 | MLP / AP | 0.35927 | 0.10732 | 23139 | 2.0 | 1.39 |
| full_expanded_union5_BCE_EPF | full / expanded | union5 | EPF / BCE | 0.36423 | 0.10687 | 17270 | 14.2 | 42.16 |
| full_expanded_union5_BCE_MLP | full / expanded | union5 | MLP / BCE | 0.36428 | 0.10685 | 18223 | 2.0 | 1.26 |
| nutrition_base19_binary4_AP_EPF | nutrition / base19 | binary4 | EPF / AP | 0.36877 | 0.10604 | 9903 | 15.1 | 27.15 |
| nutrition_base19_binary4_AP_LR | nutrition / base19 | binary4 | LR / AP | 0.37032 | 0.10625 | 292 | — | 0.06 |
| nutrition_base19_binary4_AP_MLP | nutrition / base19 | binary4 | MLP / AP | 0.35768 | 0.10653 | 18450 | 1.8 | 1.20 |
| nutrition_base19_binary4_BCE_EPF | nutrition / base19 | binary4 | EPF / BCE | 0.37177 | 0.10581 | 12495 | 11.8 | 33.80 |
| nutrition_base19_binary4_BCE_LR | nutrition / base19 | binary4 | LR / BCE | 0.37398 | 0.10578 | 292 | — | 0.06 |
| nutrition_base19_binary4_BCE_MLP | nutrition / base19 | binary4 | MLP / BCE | 0.37310 | 0.10574 | 22815 | 2.3 | 1.36 |
| nutrition_base19_regression5_MSE_Ridge | nutrition / base19 | regression5 | Ridge / MSE | 0.37407 | 0.10600 | 365 | — | 0.07 |
| nutrition_base19_union1_ANY_BCE_LR | nutrition / base19 | union1 | LR / ANY_BCE | — | — | 73 | — | 0.04 |
| nutrition_expanded_binary4_AP_EPF | nutrition / expanded | binary4 | EPF / AP | 0.36914 | 0.10669 | 14276 | 13.2 | 33.78 |
| nutrition_expanded_binary4_AP_LR | nutrition / expanded | binary4 | LR / AP | 0.36329 | 0.10740 | 460 | — | 0.09 |
| nutrition_expanded_binary4_AP_MLP | nutrition / expanded | binary4 | MLP / AP | 0.36171 | 0.10735 | 22919 | 1.8 | 1.25 |
| nutrition_expanded_binary4_BCE_EPF | nutrition / expanded | binary4 | EPF / BCE | 0.37381 | 0.10624 | 8689 | 10.6 | 20.75 |
| nutrition_expanded_binary4_BCE_LR | nutrition / expanded | binary4 | LR / BCE | 0.37181 | 0.10665 | 460 | — | 0.10 |
| nutrition_expanded_binary4_BCE_MLP | nutrition / expanded | binary4 | MLP / BCE | 0.37388 | 0.10637 | 32724 | 1.7 | 1.55 |
| nutrition_expanded_nutrition_binary4_AP_EPF | nutrition / expanded_nutrition | binary4 | EPF / AP | 0.36211 | 0.10699 | 17353 | 14.1 | 40.69 |
| nutrition_expanded_nutrition_binary4_AP_LR | nutrition / expanded_nutrition | binary4 | LR / AP | 0.36496 | 0.10726 | 484 | — | 0.09 |
| nutrition_expanded_nutrition_binary4_AP_MLP | nutrition / expanded_nutrition | binary4 | MLP / AP | 0.36380 | 0.10696 | 33516 | 1.7 | 1.58 |
| nutrition_expanded_nutrition_binary4_BCE_EPF | nutrition / expanded_nutrition | binary4 | EPF / BCE | 0.37680 | 0.10603 | 17353 | 13.2 | 41.09 |
| nutrition_expanded_nutrition_binary4_BCE_LR | nutrition / expanded_nutrition | binary4 | LR / BCE | 0.37439 | 0.10647 | 484 | — | 0.10 |
| nutrition_expanded_nutrition_binary4_BCE_MLP | nutrition / expanded_nutrition | binary4 | MLP / BCE | 0.37547 | 0.10626 | 23558 | 1.4 | 1.25 |
| nutrition_expanded_nutrition_regression5_MSE_Ridge | nutrition / expanded_nutrition | regression5 | Ridge / MSE | 0.37680 | 0.10587 | 605 | — | 0.11 |
| nutrition_expanded_nutrition_union1_ANY_BCE_LR | nutrition / expanded_nutrition | union1 | LR / ANY_BCE | — | — | 121 | — | 0.05 |
| nutrition_expanded_regression5_MSE_Ridge | nutrition / expanded | regression5 | Ridge / MSE | 0.37447 | 0.10620 | 575 | — | 0.11 |
| nutrition_expanded_union1_ANY_BCE_LR | nutrition / expanded | union1 | LR / ANY_BCE | — | — | 115 | — | 0.05 |
| scratch_common_BCE_EPF | full / expanded | development_recipe | EPF / BCE | 0.36437 | 0.10688 | 9103 | 12.3 | 21.48 |
| scratch_common_BCE_MLP | full / expanded | development_recipe | MLP / BCE | 0.36411 | 0.10639 | 13960 | 1.8 | 1.40 |
| transfer_common_BCE_EPF | full / expanded | development_recipe | EPF / BCE | 0.36344 | 0.10701 | 9103 | 12.7 | 21.17 |
| transfer_common_BCE_MLP | full / expanded | development_recipe | MLP / BCE | 0.36410 | 0.10634 | 13960 | 2.9 | 1.44 |

전체 13보기·폴드별·소견별 결과, threshold 역할에서 고른 90/95 목표의 test 실제 민감도·의뢰율, 탐색 대조의 점추정은 동결된 `PUBLIC_REPORT.json`에 있다. 90/95는 개발 목표이며 시험 달성 보장이 아니다. `—`는 해당 지표가 없거나 측정되지 않았음을 뜻한다. 학습 궤적: selected_additional 518, source40 30, tuning 1,040, young_finetune 30. 첫 gradient·실제 update와 선택 epoch는 모델별 집계에 보존한다. 총 예산과 계산량 비교에서 40세 이상 source 사전학습은 추가 자료·연산을 사용하므로 청년 scratch와 동일 총 compute 비교가 아니다.

학습 자원 정책은 사용자 요청에 따라 단계별로 전환됐다. 첫 **484개 완료 단위**는 기존 `auto` 실행(작은 EPF/MLP는 CPU, 큰 EPF는 CUDA)이었고, 이후 새 신경망 단위는 `cuda:0`을 강제했다. 저장된 CPU 재개 1건은 기록된 원래 장치에서 정확히 마쳤다. 제한 단계의 후보 튜닝·선택 모델 학습은 단일 worker, 논리 코어 **11**/affinity **2048**, Idle 우선순위, Torch/BLAS 각 1 thread 정책이었다. 이후 전이 학습에는 2026-09-25 사용자 승인·부모 검토를 거친 자원 변경을 적용했다. 전이 실행 기록의 affinity는 4095 (12개 논리 코어), 우선순위는 NORMAL_PRIORITY_CLASS, Torch/BLAS는 각 1 thread였다. 전이 완료 여부와 실제 학습 장치는 동결된 단위별 기록으로 별도 확인했다. 동결된 TRAINING_DONE별 실제 장치 수는 **cpu 365개, cuda:0 1,253개**다. 위 `학습 초`는 **장치·자원 조건이 섞인 실행 기록**이며 동일 하드웨어 통제 성능 비교가 아니다. 단계별 정책·실행 기록의 SHA를 presentation 계보에 묶었다.

40세 이상 source40의 로지스틱 기준선은 원래 반복 상한 5,000회에서 수렴 실패가 확인되어 별도 수치 amendment로 상한만 20,000회로 늘렸다. solver `lbfgs`, L2, `tol=1e-8`, C 격자와 가중치 정규화, 청년/main 모델의 설정은 유지했다. source baseline 10개를 독립 확인했고 최대 실제 반복은 소견별 10102회, 직접 any 7963회였다. 기존 5,000회를 넘긴 fit은 62건이다. 이 수정은 source 초기 선형 기준선의 수치 수렴 범위에 한정되며, 추가 반복 상한이 source 사전학습 효과의 인과 증거는 아니다.

전이 30쌍의 동결된 학습 출처를 확인한 결과, target–scratch 장치가 다른 쌍은 **5개**, source의 BCE 최선 checkpoint가 epoch 0인 단위는 **0개**다. epoch 0 선택은 source 자료를 학습해서 얻은 효과로 분류하지 않는다. CPU/CUDA 장치 차이와 dropout 난수·수치 경로가 함께 섞여 strict same-execution 짝학습이 아니다. 이 차이는 **source 사전학습 추가, 장치가 달랐던 경우의 실행 차이, 난수, target 미세조정을 포함한 절차 전체** 비교이며 순수 추가자료의 인과 효과라고 주장하지 않는다. source full10→target binary4/state16의 backbone 복사 초기값은 여섯 합성 모델에서 확인했지만 학습 장치·궤적의 동일성을 뜻하지 않는다. 이를 맞추기 위한 새 control 5개는 학습하지 않았다.

| fold | 계열 | seed | 청년 recipe | source 선택 epoch | source 장치 | target 장치 | matched scratch 장치 | target–scratch 장치 |
|---:|---|---:|---|---:|---|---|---|---|
| 1 | EPF | 42 | state16 | 8 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 1 | EPF | 43 | state16 | 8 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 1 | EPF | 44 | state16 | 8 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 2 | EPF | 42 | aux9 | 28 | cuda:0 | cuda:0 | cpu | 다름 |
| 2 | EPF | 43 | aux9 | 75 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 2 | EPF | 44 | aux9 | 37 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 3 | EPF | 42 | binary4 | 8 | cuda:0 | cuda:0 | cpu | 다름 |
| 3 | EPF | 43 | binary4 | 12 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 3 | EPF | 44 | binary4 | 9 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | EPF | 42 | focal_labelweight4 | 42 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | EPF | 43 | focal_labelweight4 | 3 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | EPF | 44 | focal_labelweight4 | 79 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | EPF | 42 | union5 | 16 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | EPF | 43 | union5 | 12 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | EPF | 44 | union5 | 10 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 1 | MLP | 42 | full10 | 18 | cuda:0 | cuda:0 | cpu | 다름 |
| 1 | MLP | 43 | full10 | 11 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 1 | MLP | 44 | full10 | 12 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 2 | MLP | 42 | state16 | 6 | cuda:0 | cuda:0 | cpu | 다름 |
| 2 | MLP | 43 | state16 | 30 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 2 | MLP | 44 | state16 | 7 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 3 | MLP | 42 | binary4 | 12 | cuda:0 | cuda:0 | cpu | 다름 |
| 3 | MLP | 43 | binary4 | 42 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 3 | MLP | 44 | binary4 | 29 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | MLP | 42 | state16 | 10 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | MLP | 43 | state16 | 9 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 4 | MLP | 44 | state16 | 12 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | MLP | 42 | state16 | 26 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | MLP | 43 | state16 | 6 | cuda:0 | cuda:0 | cuda:0 | 같음 |
| 5 | MLP | 44 | state16 | 20 | cuda:0 | cuda:0 | cuda:0 | 같음 |

위 모델표의 `추론 ms/1000명`은 평가기의 **batch256 모델 predict 호출만** 측정한 값이다. 전처리·보정·cutoff·입출력과 단일 요청 지연을 포함하지 않는다.

전이 recipe는 fold마다 활성 head가 다를 수 있다. 전이와 같은 recipe의 scratch 주대조는 모든 fold에서 존재하는 **보정 전 네 소견 확률**로 AP를 계산했다. state16 주변확률과 다른 모드 logit을 한 축에 그대로 섞지 않았다. 일부 fold에만 존재하는 direct any·수치·joint 지표는 부분 fold 평균을 내지 않고 `not_comparable`로 표기했다.

| recipe 모델 | fold별 선택 모드 | 공통 5폴드 비교 불가 지표 |
|---|---|---|
| scratch_common_BCE_EPF | 1:state16, 2:aux9, 3:binary4, 4:focal_labelweight4, 5:union5 | adjusted_any, direct_any_calibrated, direct_any_raw, exploratory/clean/continuous, exploratory/clean/independent_product16_reference, exploratory/clean/joint16, raw_any, raw_any_precal |
| scratch_common_BCE_MLP | 1:full10, 2:state16, 3:binary4, 4:state16, 5:state16 | adjusted_any, direct_any_calibrated, direct_any_raw, exploratory/clean/continuous, exploratory/clean/independent_product16_reference, exploratory/clean/joint16, raw_any, raw_any_precal |
| transfer_common_BCE_EPF | 1:state16, 2:aux9, 3:binary4, 4:focal_labelweight4, 5:union5 | adjusted_any, direct_any_calibrated, direct_any_raw, exploratory/clean/continuous, exploratory/clean/independent_product16_reference, exploratory/clean/joint16, raw_any, raw_any_precal |
| transfer_common_BCE_MLP | 1:full10, 2:state16, 3:binary4, 4:state16, 5:state16 | adjusted_any, direct_any_calibrated, direct_any_raw, exploratory/clean/continuous, exploratory/clean/independent_product16_reference, exploratory/clean/joint16, raw_any, raw_any_precal |

## 보조 수치 예측과 공동 상태 탐색

표준화 MSE는 각 fold의 **fit에서 고정한** 다섯 수치 변환으로 계산하고 fold 조사 가중합에 따라 합친다. RMSE는 seed별 pooled MSE에 제곱근을 적용한 뒤 세 독립 seed의 RMSE를 평균했다. 이는 seed×사람을 합친 하나의 MSE에 제곱근을 적용한 값과 다르다. 원단위 RMSE/MAE는 혈당·수축기/이완기 혈압·TG·HDL별로 따로 보고하며, 서로 다른 단위를 평균해 단일 전체 오차로 부르지 않는다.

| 모델 | 표준화 5수치 MSE | 혈당 RMSE mg/dL | 수축기 RMSE mmHg | 이완기 RMSE mmHg | TG RMSE mg/dL | HDL RMSE mg/dL |
|---|---:|---:|---:|---:|---:|---:|
| full_base19_regression5_MSE_Ridge | 0.72592 | 12.18 | 9.04 | 7.73 | 73.52 | 11.83 |
| full_expanded_aux9_AP_EPF | 0.72569 | 12.14 | 9.04 | 7.72 | 73.69 | 11.84 |
| full_expanded_aux9_AP_MLP | 0.73296 | 12.14 | 9.10 | 7.76 | 74.04 | 11.90 |
| full_expanded_aux9_BCE_EPF | 0.72333 | 12.12 | 9.05 | 7.72 | 73.46 | 11.82 |
| full_expanded_aux9_BCE_MLP | 0.72429 | 12.12 | 9.05 | 7.71 | 73.54 | 11.83 |
| full_expanded_full10_ANY_BCE_EPF | 0.72486 | 12.13 | 9.06 | 7.72 | 73.36 | 11.82 |
| full_expanded_full10_ANY_BCE_MLP | 0.72311 | 12.12 | 9.04 | 7.71 | 73.62 | 11.82 |
| full_expanded_full10_AP_EPF | 0.72464 | 12.13 | 9.05 | 7.72 | 73.68 | 11.83 |
| full_expanded_full10_AP_MLP | 0.72987 | 12.15 | 9.08 | 7.74 | 73.82 | 11.88 |
| full_expanded_full10_BCE_EPF | 0.72331 | 12.12 | 9.05 | 7.72 | 73.37 | 11.82 |
| full_expanded_full10_BCE_MLP | 0.72487 | 12.13 | 9.05 | 7.72 | 73.46 | 11.83 |
| full_expanded_regression5_AP_EPF | 0.72605 | 12.12 | 9.06 | 7.74 | 73.65 | 11.83 |
| full_expanded_regression5_AP_MLP | 0.73012 | 12.12 | 9.11 | 7.77 | 73.52 | 11.86 |
| full_expanded_regression5_MSE_EPF | 0.72332 | 12.11 | 9.04 | 7.72 | 73.49 | 11.82 |
| full_expanded_regression5_MSE_MLP | 0.72405 | 12.11 | 9.05 | 7.72 | 73.53 | 11.82 |
| full_expanded_regression5_MSE_Ridge | 0.72480 | 12.13 | 9.06 | 7.72 | 73.69 | 11.84 |
| nutrition_base19_regression5_MSE_Ridge | 0.71710 | 11.50 | 9.08 | 7.71 | 74.71 | 11.85 |
| nutrition_expanded_nutrition_regression5_MSE_Ridge | 0.71563 | 11.47 | 9.09 | 7.70 | 74.70 | 11.87 |
| nutrition_expanded_regression5_MSE_Ridge | 0.71724 | 11.47 | 9.10 | 7.71 | 74.86 | 11.87 |

`state16`의 joint 지표는 실제 16상태 softmax 분포에 대한 가중 NLL, 16상태 확률 제곱오차 **합** Brier, 최빈 상태 정확도다. 다른 네 소견 모델에서 만든 16상태 독립곱은 별도 **참조분포**이며 학습된 joint로 해석하지 않는다.

| state16 모델 | raw joint NLL | 보정 joint NLL | raw joint Brier | 보정 joint Brier | raw exact state | 보정 exact state |
|---|---:|---:|---:|---:|---:|---:|
| full_expanded_state16_ANY_BCE_EPF | 1.37907 | 1.37819 | 0.53381 | 0.53512 | 0.61643 | 0.61643 |
| full_expanded_state16_ANY_BCE_MLP | 1.37776 | 1.37696 | 0.53370 | 0.53511 | 0.61405 | 0.61405 |
| full_expanded_state16_AP_EPF | 1.37926 | 1.37918 | 0.53409 | 0.53537 | 0.61591 | 0.61591 |
| full_expanded_state16_AP_MLP | 1.37780 | 1.37625 | 0.53478 | 0.53566 | 0.61428 | 0.61428 |
| full_expanded_state16_BCE_EPF | 1.38474 | 1.38273 | 0.53483 | 0.53536 | 0.61316 | 0.61316 |
| full_expanded_state16_BCE_MLP | 1.37991 | 1.37944 | 0.53391 | 0.53545 | 0.61281 | 0.61281 |

## 공개 예시 실행 시간

상태: **measured**. 별도 공개 패키지의 fold1 고정 예시 7개에 대해 합성 raw38을 넣고 CPU 1스레드에서 전처리→모델→보정→threshold까지 측정한 결과다. warmup 10회 후 50회이며 package/model 로딩, CLI 시작과 JSON 파일 읽기는 제외한다. batch256의 1인당 시간은 **batch amortized** 값으로 단일 요청 지연과 다르다. CUDA를 요청한 경우에는 GPU 조건을 별도 행으로 보인다.

| 공개 예시 | 계열 | 기기 | batch | 중앙값 ms/batch | p95 ms/batch | 중앙값 ms/인 amortized | 중앙값 처리 인원/s |
|---|---|---|---:|---:|---:|---:|---:|
| binary4_BCE_EPF | EPF | cpu | 1 | 1.495 | 1.552 | 1.4954 | 668.7 |
| binary4_BCE_EPF | EPF | cpu | 256 | 3.589 | 4.343 | 0.0140 | 71320.1 |
| binary4_BCE_EPF | EPF | cuda | 1 | 4.799 | 7.633 | 4.7991 | 208.4 |
| binary4_BCE_EPF | EPF | cuda | 256 | 4.649 | 5.756 | 0.0182 | 55070.9 |
| binary4_BCE_MLP | MLP | cpu | 1 | 0.579 | 0.640 | 0.5788 | 1727.6 |
| binary4_BCE_MLP | MLP | cpu | 256 | 0.856 | 1.102 | 0.0033 | 298960.6 |
| binary4_BCE_MLP | MLP | cuda | 1 | 0.839 | 1.156 | 0.8394 | 1191.3 |
| binary4_BCE_MLP | MLP | cuda | 256 | 1.044 | 1.168 | 0.0041 | 245187.2 |
| full10_BCE_EPF | EPF | cpu | 1 | 1.948 | 2.215 | 1.9481 | 513.3 |
| full10_BCE_EPF | EPF | cpu | 256 | 12.858 | 18.054 | 0.0502 | 19909.8 |
| full10_BCE_EPF | EPF | cuda | 1 | 5.771 | 6.405 | 5.7713 | 173.3 |
| full10_BCE_EPF | EPF | cuda | 256 | 7.172 | 9.260 | 0.0280 | 35694.1 |
| full10_BCE_MLP | MLP | cpu | 1 | 0.629 | 0.707 | 0.6287 | 1590.6 |
| full10_BCE_MLP | MLP | cpu | 256 | 0.967 | 1.182 | 0.0038 | 264777.4 |
| full10_BCE_MLP | MLP | cuda | 1 | 1.643 | 2.949 | 1.6434 | 608.5 |
| full10_BCE_MLP | MLP | cuda | 256 | 2.250 | 5.583 | 0.0088 | 113770.2 |
| state16_BCE_EPF | EPF | cpu | 1 | 1.936 | 2.244 | 1.9357 | 516.6 |
| state16_BCE_EPF | EPF | cpu | 256 | 12.110 | 13.668 | 0.0473 | 21139.9 |
| state16_BCE_EPF | EPF | cuda | 1 | 7.329 | 8.414 | 7.3295 | 136.4 |
| state16_BCE_EPF | EPF | cuda | 256 | 5.946 | 7.357 | 0.0232 | 43056.7 |
| state16_BCE_MLP | MLP | cpu | 1 | 0.612 | 0.698 | 0.6123 | 1633.2 |
| state16_BCE_MLP | MLP | cpu | 256 | 1.093 | 1.331 | 0.0043 | 234303.5 |
| state16_BCE_MLP | MLP | cuda | 1 | 1.551 | 1.783 | 1.5507 | 644.8 |
| state16_BCE_MLP | MLP | cuda | 256 | 1.774 | 2.441 | 0.0069 | 144270.1 |
| LR_BCE | LR | cpu | 1 | 0.436 | 0.559 | 0.4363 | 2292.0 |
| LR_BCE | LR | cpu | 256 | 0.889 | 0.963 | 0.0035 | 287802.1 |

## 별도 연구 경로

v21 세 seed **확률 평균** 및 의뢰 정책의 집계 상태: **complete**. 기존 v21 enhanced_BCE_EPF의 보정 전 3시드 확률평균 AP는 0.366936, 시드별 AP 평균은 0.366256이다. γ 위험 90% 개발 목표에서 기본 cutoff의 test 민감도/의뢰율은 0.866372/0.637197, PSU 보수 cutoff는 0.904441/0.716940였다. 이 경로는 v22의 새 13 cells와 합산하지 않는다. 확률 평균의 AP는 세 seed AP 평균과 다른 추정량이다. 이 별도 정책의 결과를 v22 main 설정 선택에 사용하지 않았다.

미국 NHANES 공통 5입력 연구의 집계 상태: **completed_aggregate_only**. 이 경로는 원 19/38입력 모델의 직접 외부 검증이 아니다. 미국 결정 지표는 not_defined. 한국·미국 eligibility와 측정 방법이 완전히 같지 않다. 특히 미국 DIQ010=2는 borderline=3을 제외하지만 한국 DE1_dg=0에는 기진단 전당뇨가 포함될 수 있다.

| 미국 별도 모델 | 한국 fold 모델 수 | 미국 raw 4소견 macro AP | 미국 PSU 95% 구간 | 미국 보정 후 macro AP | 추정 가능 상태 |
|---|---:|---:|---:|---:|---|
| EPF_AP | 15 | 0.36729 | [0.30004, 0.46504] | 0.36752 | estimable / estimable |
| EPF_BCE | 15 | 0.37165 | [0.30238, 0.47005] | 0.37093 | estimable / estimable |
| LR_AP | 5 | 0.37926 | [0.30314, 0.48573] | 0.37844 | estimable / estimable |
| LR_BCE | 5 | 0.37422 | [0.30324, 0.47676] | 0.37354 | estimable / estimable |
| MLP_AP | 15 | 0.37318 | [0.30136, 0.47111] | 0.37126 | estimable / estimable |
| MLP_BCE | 15 | 0.37463 | [0.30101, 0.47719] | 0.37260 | estimable / estimable |

## 출처·한계

EPF 내부 4/6단계 상태 반복은 같은 사람의 내부 계산이며 개인 추적 시계열 학습이 아니다. 이번 실험은 로컬 지도학습이다. Jev 실제 가중치, teacher API 호출, 강화학습 RLCD 증류를 실행한 기록이 없다. API 키 부재 상태에서 그런 결과를 주장하지 않는다. 네 소견과 16상태는 미인지 **측정 이상**의 조합이다. 혈당 ≥100 mg/dL와 혈압 ≥130/85 mmHg 등을 확진 당뇨·고혈압 질환 분류로 바꾸어 부르지 않는다. 해석은 입력 확장과 학습 목표·전이의 비교에 한정한다.

훈련 동결 SHA-256: `2a22206751bfbc86b71f1b4b19741711a9454dcb15fcb6e3e566049248129850`. 평가 인덱스 SHA-256: `bdd7c3fe7ec8572bc82a98eb4e729ba48238d1892e28483830c16cb24cf9a51a`. 독립 checkpoint forward 검증 SHA-256: `738e7efcddeeb053c23747c035acde0bdb57b4ba8e702742fc44d9f66a256b5d`. 수치 분석 소스 SHA-256: `b547a132d6cfc0a5abcee16c14ed19ac8749b97c870f1fe76efae84a383d2f44`. 재현용 집계 노트북은 비공개 `report.json`에서 추출한 `PUBLIC_REPORT.json`만 읽는다. 개인별 예측·원자료·PSU 인덱스는 이 문서와 노트북에 없다.
