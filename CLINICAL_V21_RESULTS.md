# KNHANES 비침습 대사이상 선별 v21 결과

19–39세 4,068명(2021: 1,039, 2022: 1,013, 2023: 1,010, 2024: 1,006)을 같은 5개 PSU 외부 fold와 다섯 역할로 평가했다. 네 소견의 정답이 되는 혈액검사·혈압 수치는 입력에 쓰지 않았다. 이전 v16–v20에서 확인한 코호트의 **후속 탐색 결과**다.

## 주 비교: 보정 전 4소견 macro AP

AP 선택 checkpoint 세 seed의 독립 시험 점수를 평균했다. 같은 훈련 궤적의 BCE 선택 checkpoint를 별도 학습 목적함수로 해석하지 않는다. LR_AP와 LR_BCE는 같은 12-C grid에서 validation 정책으로 선택한 기준이다.

| 모델 | 보정 전 macro AP ↑ | 보정 후 macro AP ↑ | 보정 전 macro Brier ↓ | 보정 후 macro Brier ↓ | 보정 전 any AP | 소견 보정 뒤 q0 AP | γ 뒤 any AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| enhanced_AP_EPF | 0.362684 | 0.361521 | 0.105840 | 0.106225 | 0.699006 | 0.696915 | 0.692759 |
| enhanced_AP_MLP | 0.357676 | 0.360665 | 0.106775 | 0.106431 | 0.692596 | 0.693812 | 0.690007 |
| LR_AP | 0.361252 | 0.360251 | 0.106185 | 0.106528 | 0.696281 | 0.693507 | 0.689649 |
| LR_BCE | 0.368316 | 0.365194 | 0.105435 | 0.106030 | 0.700888 | 0.697872 | 0.693592 |

같은 계열 안에서 AP/BCE 정책의 **실제 시험 점추정**을 나란히 두면 다음과 같다. 이 정책들은 validation에서 선택했으며 시험에서 다시 고른 결과가 아니다.

| 계열 | AP 설정·AP checkpoint 보정 전 macro AP | BCE 설정·BCE checkpoint 보정 전 macro AP |
|---|---:|---:|
| EPF | 0.362684 | 0.366256 |
| MLP | 0.357676 | 0.367310 |
| LR | 0.361252 | 0.368316 |

주 AP 정책에서 EPF가 MLP보다 높다는 결과를 모든 MLP 설정에 대한 우위로 넓히지 않는다. `linear_adamw_anchor_BCE`와 `probe_MLP_no_gate_BCE`도 비교 맥락을 위한 탐색적 점추정이며, 이 둘의 별도 paired PSU 구간은 계산하지 않았다.

| 탐색적 모델 | 보정 전 macro AP 점추정 | paired PSU 95% 구간 |
|---|---:|---|
| linear_adamw_anchor_BCE | 0.369046 | 별도 구간 없음 |
| probe_MLP_no_gate_BCE | 0.368967 | 별도 구간 없음 |

**1차 판단은 첫 수치 열**에 둔다. 소견별 보정은 fold 안의 순위를 역전시키지 않지만 기울기 0이나 수치 포화에서는 동률을 만들 수 있다. fold마다 다른 보정식은 전체 OOF의 fold 간 순위를 바꿀 수 있고, γ는 사람 간 q0 순위도 바꿀 수 있다. 네 소견별 AP·Brier·NLL, fold별 결과와 13개 시험 보기는 아래 실행 노트북과 집계 JSON에 있다.

| 대조(왼쪽−오른쪽) | 보정 전 macro AP 차 | paired PSU 95% 구간 | 유효 재표집 | seed 차 SD | seed 차 부호 |
|---|---:|---:|---:|---:|---|
| primary/EPF_vs_MLP | 0.005008 | [0.000559, 0.009631] | 2000 | 0.003372 | positive, positive, positive |
| primary/EPF_vs_LR_AP | 0.001432 | [-0.003952, 0.006336] | 2000 | 0.002505 | negative, positive, positive |
| primary/EPF_vs_LR_BCE | -0.005631 | [-0.010765, -0.001013] | 2000 | 0.002505 | negative, negative, negative |
| secondary/enhanced_vs_plain_EPF | -0.001363 | [-0.003597, 0.000737] | 2000 | 0.001948 | negative, positive, negative |

세 seed의 paired 차이와 범위는 집계 JSON에 보존했다. seed SD는 학습 불확실성의 신뢰구간이 아니다. plain 대비와 AP/BCE 정책 차이 및 재훈련 probe는 주대조와 구분한다. 출력의 선형·장 RMS 비율만으로 기전 기여율을 정하지 않는다.

### 선택 정책과 사전 지정 probe

AP/BCE 정책 대조에서 서로 다른 config의 선택 효과와 **같은 config의 AP/BCE checkpoint 차이**를 이름으로 구별했다. 모델마다 `config_policy`와 checkpoint `policy`를 별도 보존했다. 같은 궤적에서 나온 checkpoint 비교는 독립 목적함수 훈련의 효과가 아니다.

| 보조 대조(왼쪽−오른쪽) | 보정 전 macro AP 차 | paired PSU 95% 구간 | seed 차 SD |
|---|---:|---:|---:|
| secondary/AP_vs_BCE_enhanced_EPF | -0.003572 | [-0.006787, -0.000323] | 0.004256 |
| secondary/same_AP_config_AP_vs_BCE_enhanced_EPF | -0.003315 | [-0.006564, -0.000250] | 0.002723 |
| secondary/same_BCE_config_AP_vs_BCE_enhanced_EPF | -0.001133 | [-0.003025, 0.000796] | 0.000980 |
| secondary/AP_vs_BCE_enhanced_MLP | -0.009634 | [-0.014236, -0.005254] | 0.002255 |
| secondary/same_AP_config_AP_vs_BCE_enhanced_MLP | -0.009465 | [-0.013966, -0.005044] | 0.002608 |
| secondary/same_BCE_config_AP_vs_BCE_enhanced_MLP | -0.009968 | [-0.015045, -0.005202] | 0.001380 |
| secondary/AP_vs_BCE_plain_EPF | -0.002393 | [-0.004835, -0.000115] | 0.003793 |
| secondary/same_AP_config_AP_vs_BCE_plain_EPF | -0.003175 | [-0.005433, -0.000884] | 0.003250 |
| secondary/same_BCE_config_AP_vs_BCE_plain_EPF | -0.003487 | [-0.005911, -0.000976] | 0.004350 |
| secondary/AP_vs_BCE_plain_MLP | -0.011928 | [-0.017708, -0.006526] | 0.004553 |
| secondary/same_AP_config_AP_vs_BCE_plain_MLP | -0.012407 | [-0.017981, -0.007088] | 0.004842 |
| secondary/same_BCE_config_AP_vs_BCE_plain_MLP | -0.012425 | [-0.018101, -0.007129] | 0.003737 |

| 사전 지정 재훈련 probe(기준−probe) | 보정 전 macro AP 차 | paired PSU 95% 구간 | seed 차 SD |
|---|---:|---:|---:|
| preplanned_probe/probe_EPF_no_Preadout | -0.000135 | [-0.000384, 0.000105] | 0.000404 |
| preplanned_probe/probe_EPF_no_plasticity | -0.000028 | [-0.000388, 0.000356] | 0.000098 |
| preplanned_probe/probe_EPF_no_quartic | 0.000417 | [-0.000544, 0.001279] | 0.001173 |

나머지 probe도 기준 모델과의 실제 paired 점추정과 seed 차이 범위를 표시한다. 이 탐색 probe에는 PSU 구간을 계산하지 않았다.

| 탐색 probe(기준−probe) | 보정 전 macro AP 차 | seed paired 차이 범위 | paired PSU 95% 구간 |
|---|---:|---:|---|
| exploratory_probe/probe_EPF_capacity_control | -0.000850 | [-0.002898, 0.000215] | 계산 안 함 |
| exploratory_probe/probe_EPF_capacity_control_BCE | 0.001532 | [-0.000184, 0.004814] | 계산 안 함 |
| exploratory_probe/probe_EPF_field_only | 0.016306 | [0.007811, 0.022454] | 계산 안 함 |
| exploratory_probe/probe_EPF_field_only_BCE | 0.008208 | [0.003643, 0.014056] | 계산 안 함 |
| exploratory_probe/probe_EPF_native_prior | 0.009881 | [0.006139, 0.016273] | 계산 안 함 |
| exploratory_probe/probe_EPF_native_prior_BCE | 0.008876 | [0.006860, 0.012006] | 계산 안 함 |
| exploratory_probe/probe_EPF_native_zero | 0.011882 | [0.010268, 0.014205] | 계산 안 함 |
| exploratory_probe/probe_EPF_native_zero_BCE | 0.016520 | [0.014009, 0.021408] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_Preadout_BCE | -0.000023 | [-0.000098, 0.000032] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_gate | 0.001679 | [-0.000100, 0.004094] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_gate_BCE | 0.001698 | [0.000271, 0.004351] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_plasticity_BCE | 0.001652 | [-0.000011, 0.004862] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_quartic_BCE | -0.000009 | [-0.000039, 0.000040] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_shrink | 0.000133 | [0.000088, 0.000219] | 계산 안 함 |
| exploratory_probe/probe_EPF_no_shrink_BCE | 0.000016 | [0.000006, 0.000028] | 계산 안 함 |
| exploratory_probe/probe_EPF_numeric_embedding | 0.000358 | [-0.000074, 0.000906] | 계산 안 함 |
| exploratory_probe/probe_EPF_numeric_embedding_BCE | 0.000293 | [0.000144, 0.000499] | 계산 안 함 |
| exploratory_probe/probe_EPF_smooth_event | -0.001658 | [-0.002835, -0.000763] | 계산 안 함 |
| exploratory_probe/probe_EPF_smooth_event_BCE | -0.000426 | [-0.000772, 0.000010] | 계산 안 함 |
| exploratory_probe/probe_MLP_native_prior | 0.008904 | [0.008467, 0.009460] | 계산 안 함 |
| exploratory_probe/probe_MLP_native_prior_BCE | 0.008065 | [0.007319, 0.008669] | 계산 안 함 |
| exploratory_probe/probe_MLP_native_zero | 0.017612 | [0.009453, 0.027760] | 계산 안 함 |
| exploratory_probe/probe_MLP_native_zero_BCE | 0.014044 | [0.012945, 0.015101] | 계산 안 함 |
| exploratory_probe/probe_MLP_no_gate | 0.001095 | [-0.002351, 0.003554] | 계산 안 함 |
| exploratory_probe/probe_MLP_no_gate_BCE | -0.001826 | [-0.002681, -0.000867] | 계산 안 함 |
| exploratory_probe/probe_MLP_no_shrink | -0.000019 | [-0.000041, 0.000010] | 계산 안 함 |
| exploratory_probe/probe_MLP_no_shrink_BCE | -0.000003 | [-0.000029, 0.000043] | 계산 안 함 |
| exploratory_probe/probe_MLP_numeric_embedding | 0.001752 | [0.000746, 0.003330] | 계산 안 함 |
| exploratory_probe/probe_MLP_numeric_embedding_BCE | 0.000323 | [0.000112, 0.000450] | 계산 안 함 |

field-only는 선형 경로도 제거하므로 단일 요인 ablation이 아니다.

## 개발 선택 기록

각 단계·계열에 fold당 후보 32개를 같은 seed 42로 적합했다(전체 640개 후보 fit). 한 궤적에서 AP/BCE checkpoint를 모두 보존했다. 아래는 enhanced의 선택 기록이다. plain과 전체 config는 실행 노트북·집계 JSON에 있다.

| fold | 계열 | 선택 정책 | 후보 ID | epoch | validation AP | validation BCE |
|---:|---|---|---:|---:|---:|---:|
| 1 | EPF | AP | 14 | 15 | 0.46173 | 0.30457 |
| 1 | EPF | BCE | 9 | 17 | 0.44973 | 0.30392 |
| 1 | MLP | AP | 27 | 12 | 0.46553 | 0.30403 |
| 1 | MLP | BCE | 22 | 41 | 0.45869 | 0.30316 |
| 2 | EPF | AP | 11 | 41 | 0.39834 | 0.33228 |
| 2 | EPF | BCE | 16 | 13 | 0.37425 | 0.32506 |
| 2 | MLP | AP | 26 | 24 | 0.42648 | 0.33929 |
| 2 | MLP | BCE | 31 | 12 | 0.38344 | 0.32641 |
| 3 | EPF | AP | 18 | 24 | 0.40995 | 0.36720 |
| 3 | EPF | BCE | 0 | 0 | 0.40375 | 0.36582 |
| 3 | MLP | AP | 26 | 6 | 0.40900 | 0.36669 |
| 3 | MLP | BCE | 29 | 2 | 0.40472 | 0.36585 |
| 4 | EPF | AP | 10 | 2 | 0.41230 | 0.31843 |
| 4 | EPF | BCE | 29 | 8 | 0.40727 | 0.31804 |
| 4 | MLP | AP | 0 | 0 | 0.41517 | 0.31850 |
| 4 | MLP | BCE | 24 | 4 | 0.41006 | 0.31816 |
| 5 | EPF | AP | 28 | 6 | 0.39083 | 0.32438 |
| 5 | EPF | BCE | 28 | 8 | 0.38791 | 0.32354 |
| 5 | MLP | AP | 3 | 75 | 0.39590 | 0.32328 |
| 5 | MLP | BCE | 5 | 79 | 0.39307 | 0.32269 |

## 고정 선별 cutoff의 실제 시험 성적

90%/95%는 별도 clean threshold 역할에서 정한 목표다. 다음 수치는 해당 cutoff를 test에 그대로 적용한 **실제 민감도와 의뢰율**이다. 구간은 동일한 고정 예측에 대한 full-frame PSU 재표집이다. 시험 ROC에서 목표 민감도를 보간한 기술 값은 집계 JSON에 별도로 있다.

| 모델 | 위험 | threshold 목표 | 시험 민감도 | 95% 구간 | 시험 의뢰율 | 95% 구간 |
|---|---|---:|---:|---:|---:|---:|
| enhanced_AP_EPF | raw_any_precal | 0.9 | 0.8676 | [0.848612, 0.885503] | 0.6352 | [0.616608, 0.653315] |
| enhanced_AP_EPF | raw_any_precal | 0.95 | 0.9260 | [0.909985, 0.940804] | 0.7553 | [0.738736, 0.771763] |
| enhanced_AP_EPF | raw_any | 0.9 | 0.8693 | [0.849931, 0.887715] | 0.6430 | [0.625096, 0.660739] |
| enhanced_AP_EPF | raw_any | 0.95 | 0.9275 | [0.911851, 0.941960] | 0.7577 | [0.741057, 0.774150] |
| enhanced_AP_EPF | adjusted_any | 0.9 | 0.8663 | [0.845961, 0.885094] | 0.6378 | [0.620005, 0.655578] |
| enhanced_AP_EPF | adjusted_any | 0.95 | 0.9251 | [0.909138, 0.939944] | 0.7559 | [0.739105, 0.772161] |
| enhanced_AP_MLP | raw_any_precal | 0.9 | 0.8665 | [0.847345, 0.885017] | 0.6362 | [0.617613, 0.654663] |
| enhanced_AP_MLP | raw_any_precal | 0.95 | 0.9220 | [0.905430, 0.936943] | 0.7516 | [0.734728, 0.767968] |
| enhanced_AP_MLP | raw_any | 0.9 | 0.8663 | [0.846741, 0.883963] | 0.6418 | [0.623292, 0.659200] |
| enhanced_AP_MLP | raw_any | 0.95 | 0.9276 | [0.911712, 0.941898] | 0.7564 | [0.739901, 0.772403] |
| enhanced_AP_MLP | adjusted_any | 0.9 | 0.8673 | [0.847793, 0.885762] | 0.6410 | [0.622742, 0.658699] |
| enhanced_AP_MLP | adjusted_any | 0.95 | 0.9259 | [0.909998, 0.940151] | 0.7558 | [0.739608, 0.772015] |
| LR_AP | raw_any_precal | 0.9 | 0.8599 | [0.839754, 0.879046] | 0.6260 | [0.607930, 0.644041] |
| LR_AP | raw_any_precal | 0.95 | 0.9279 | [0.912130, 0.942087] | 0.7575 | [0.739766, 0.774386] |
| LR_AP | raw_any | 0.9 | 0.8699 | [0.850159, 0.888247] | 0.6471 | [0.629356, 0.664619] |
| LR_AP | raw_any | 0.95 | 0.9253 | [0.909466, 0.940084] | 0.7557 | [0.738416, 0.772374] |
| LR_AP | adjusted_any | 0.9 | 0.8637 | [0.843885, 0.882547] | 0.6352 | [0.617253, 0.653370] |
| LR_AP | adjusted_any | 0.95 | 0.9242 | [0.907774, 0.939037] | 0.7560 | [0.738170, 0.773009] |
| LR_BCE | raw_any_precal | 0.9 | 0.8670 | [0.846994, 0.885364] | 0.6369 | [0.618394, 0.655408] |
| LR_BCE | raw_any_precal | 0.95 | 0.9266 | [0.910763, 0.940829] | 0.7582 | [0.741236, 0.774757] |
| LR_BCE | raw_any | 0.9 | 0.8708 | [0.850121, 0.889366] | 0.6507 | [0.633244, 0.668818] |
| LR_BCE | raw_any | 0.95 | 0.9244 | [0.908723, 0.939295] | 0.7546 | [0.737764, 0.771301] |
| LR_BCE | adjusted_any | 0.9 | 0.8661 | [0.846064, 0.884729] | 0.6427 | [0.624720, 0.660326] |
| LR_BCE | adjusted_any | 0.95 | 0.9219 | [0.905629, 0.937168] | 0.7539 | [0.736738, 0.770911] |

## 보정 표본과 유병률 차이

| 모델 | calibration 최소 유효 표본 수 | γ 평균 | γ 하한/상한 끝점 unit 수 | test−calibration 가중 any 유병률 |
|---|---:|---:|---:|---:|
| enhanced_AP_EPF | 220.7 | -0.2048 | 0 / 0 | -0.0117 |
| enhanced_AP_MLP | 220.7 | -0.2104 | 0 / 0 | -0.0117 |
| LR_AP | 220.7 | -0.2089 | 0 / 0 | -0.0117 |
| LR_BCE | 220.7 | -0.2052 | 0 / 0 | -0.0117 |

소견별 보정 계수의 경계 도달 수와 calibration 역할의 보정 전후 Brier/NLL은 집계 JSON·노트북에 있다. 표본이 적거나 보정 계수가 경계에 붙은 경우의 불안정성과 역할 간 유병률 차이를 함께 해석한다.

## 학습 및 상태 진단

| 모델 | 저장 파라미터 평균 | 실제 최적화 파라미터 평균 | 선택 epoch 평균 | P 투영 크기 평균 | gate 평균 | 선형 RMS 평균 | 장/잔차 RMS 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| enhanced_AP_EPF | 9903.0 | 9844.6 | 12.4 | 0.00424 | 0.10512 | 2.49954 | 0.17202 |
| enhanced_AP_MLP | 18450.4 | 18392.0 | 21.8 | — | 0.11207 | 2.50096 | 0.24628 |

linear ratio 0에서 가중치 일부가 동결되더라도 저장된 모델 용량이 사라지는 것은 아니다. native_zero/native_prior는 LR 계수를 주입하지 않으나 공통 규제 λ가 fit/validation의 LR_BCE 선택 C에 의존한다. native_prior의 유병률 정보도 fit에서만 얻었다.

## 해석과 재현 범위

주대조 및 사전 지정 일부 probe의 구간은 전체 조사 틀 PSU 2,000회 paired 재표집이다. 이는 고정 OOF 예측 조건의 근사 구간이며 재학습·모델 선택의 전체 변동과 다중비교 보정은 포함하지 않는다. 동일 코호트를 반복 탐색했으므로 독립 확증이나 신규 시계열 검증으로 읽지 않는다. 탐색 probe에서 관찰한 차이도 기전의 임상 기여를 확정하지 않는다.

학습·원자료 출처, 수치 분석, 문서 생성의 SHA-256 manifest를 분리했다. 개인별 예측·역할 인덱스·원자료는 이 문서와 노트북에 포함하지 않는다.
