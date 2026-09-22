> 이전 v15 전문가 결합 실험의 보존 문서다. 현재 단일 셀은 [최신 비교](COMPARISON.md)와 [수식](MECHANISM.md)을 따른다.

# 모델 비교와 해석

2021–2024년 4,068명: 2021년 1,039명, 2022년 1,013명, 2023년 1,010명, 2024년 1,006명. 원래 입력에는 자연 결측이 남아 있으므로 표의 `clean`은 완전 관측이나 정상인을 뜻하지 않는다.

## 과제와 입력

공복혈당 ≥100, 수축기혈압 ≥130 또는 이완기혈압 ≥85, 중성지방 ≥150, HDL 남 <40/여 <50의 네 소견을 동시에 예측한다. 단위는 혈압 mmHg, 검사값 mg/dL이다. 임상 진단과 구별한다. 기존 진단·약물 사용 배제, 12시간 이상 공복, target 검사와 설계변수 유효 등 기존 strict cohort를 유지했다.

19개 비침습 입력을 학습 자료에서만 결측 대체·인코딩한다. 72개 기저에는 hinge와 성별×신체 계측 항이 포함되어 **LR_basis도 원래 입력에 대해서는 완전 선형 모델이 아니다.** 검사 수치·혈압·정답 소견은 입력에서 제외한다.

출력은 4개 주변 확률이다. 16개 조합 확률은 이 확률의 곱으로 구성한 후 단일 온도로 보정한다. 합계 1의 유효한 분포지만 별도 16-class 분류기나 소견 간 상관을 직접 학습한 joint model은 아니다.

## 구조와 학습법

| 모델 | 유지한 계산 | 이번 개선 | 학습 방식 |
|---|---|---|---|
| Vortex_alternating | 복소 진폭·위상, Cayley unitary, GP, winding 진단 | 별도 smooth circulation/frustration proxy, 원래 기저+중복 제거된 파동 특징 | 매 라운드 수렴한 LR readout을 새로 적합한 뒤 wave encoder를 1 epoch 역전파; 최대 12라운드, 검증으로 선택. 고정 LR anchor 없음. round0 선택은 고정 초기 파동 상태임. |
| Physical_regularized | FREE v0.1 quartic 매질, BAOAB, label 없는 국소 적응 | 구동32+감쇠32단계, rank6 추가 특징, 약한 방향 증폭 제한 | 매질은 label/역전파 없이 적응; readout은 weighted LR |
| Portia_regularized | 32 sparse LIF 뉴런, E/I, STDP, 항상성 | 유한 pulse, label로 조절한 학습 신호, rank8 추가 특징·공통 스케일 | fit 행에서 국소 가소성 4회; readout weighted LR. 추론 시 label 미사용 |
| LR_tuned | 같은72기저 | C=.01/.03/.1/1 검증 선택 | 소견별 weighted LR |

Vortex의 smooth proxy는 정수 winding의 미분이나 위상 불변량이 아니다. Portia는 설계 문서 기반 후보이며 실측 거미 커넥톰 복원이 아니다. 물리 시뮬레이션의 시간은 참가자의 건강 시계열이 아니다. `three` 모델에도 logistic readout·stacker가 있으므로 순수 신경망 대 모든 LR 계산의 비교라고 쓰지 않는다.

## 동일 조건과 선택 절차

1. 기존 5개 outer PSU 분할을 고정했다. 모든 연도를 사용하고 각 참가자는 자신과 같은 PSU를 학습하지 않은 모델의 outer test에 한 번 포함된다. 별도의 outer-fit/validation/calibration 역할도 서로 PSU가 겹치지 않는다.
2. v14에서 outer-fit 안에 PSU 3-fold를 만들어 기저 전처리·파동·물리·STDP·readout을 매번 새로 학습했다. 그 inner held-out 예측으로만 meta model을 적합한다. v15는 이 체크포인트와 같은 분할을 재사용하며 base model을 다시 튜닝하지 않았다.
3. clean와 첫 번째 누락 view를 학습에 사용한다. 같은 사람을 두 명으로 세지 않는다. 검증은 clean AP 0.5 + 3개 누락 AP 평균 0.5다. 누락 요청률20%, age/sex 보존, 허리둘레/WHtR는 함께 누락한다.
4. 각 expert set에 15개 meta 설정을 적합하고 fullfit/inner-bag 두 추론 경로를 검증에서 비교한다. 고정5개, 동적10개 설정이며 three/assisted/LR에 같은 후보 틀을 제공했다. 동적 후보군과 전체 선택 절차를 따로 보고한다.
5. 선택을 고정한 뒤 원래 calibration 역할의 clean+누락으로 온도 하나를 적합한다. 그 후 outer test를 계산한다. AP는 소견별 조사 가중 AP의 단순 평균이다.

**제한:** base hyperparameter와 meta 설정은 같은 개발 검증을 재사용했다. 완전히 중첩된 hyperparameter 탐색은 아니다. 모든 데이터는 과거 실험에서 이미 검토했으므로 개발 과정 전체가 탐색적이다. 모델별 계산량도 같지 않다. LR 한 개에는 전문가 간 불일치가 존재하지 않으므로 LR state 후보는 완전성만으로 구분된다.

## 최신 v15 결과 — 후보군 전체 공개

| 모델 | 원래 입력 AP | 인공 누락 AP |
|---|---|---|
| three_static | 0.3707 | 0.3527 |
| three_dynamic | 0.3709 | 0.3534 |
| three_selected | 0.3697 | 0.3528 |
| assisted_static | 0.3710 | 0.3517 |
| assisted_dynamic | 0.3722 | 0.3526 |
| assisted_selected | 0.3719 | 0.3533 |
| LR_static | 0.3734 | 0.3524 |
| LR_dynamic | 0.3728 | 0.3533 |
| LR_selected | 0.3734 | 0.3530 |

`static`과 `dynamic`은 각각 고정/조건부 후보군에서 validation으로 선택한 절차다. `selected`는 둘 중 validation이 선택한 절차이며, outer test에서 `dynamic`이 높다고 `selected`를 바꾸지 않는다. `assisted`에는 별도 LR 전문가가 포함된다.

세 모델 dynamic은 static 대비 +0.000256 / +0.000715였다. 그러나 전체 선택 절차 `three_selected`는 static보다 원래 입력 AP가 낮았다. 조건을 늘리면 항상 좋아진다는 가설은 지지되지 않는다.

## 고정 예측의 조건부 95% 차이 구간

| 비교 | 원래 입력 | 인공 누락 |
|---|---|---|
| three_dynamic − three_static | [-0.0025, +0.0030] | [-0.0028, +0.0038] |
| three_dynamic − LR_selected | [-0.0065, +0.0012] | [-0.0037, +0.0041] |
| three_dynamic − LR_v13 | [-0.0034, +0.0073] | [+0.0009, +0.0112] |
| three_selected − three_static | [-0.0042, +0.0020] | [-0.0032, +0.0030] |
| three_selected − LR_selected | [-0.0083, +0.0008] | [-0.0041, +0.0037] |
| three_selected − LR_v13 | [-0.0047, +0.0064] | [-0.0004, +0.0104] |
| assisted_selected − three_static | [-0.0018, +0.0045] | [-0.0027, +0.0035] |
| assisted_selected − LR_selected | [-0.0062, +0.0019] | [-0.0033, +0.0036] |
| assisted_selected − LR_v13 | [-0.0027, +0.0085] | [+0.0005, +0.0114] |

전체 표본설계 frame의 PSU를 200회 재표집한 paired bootstrap이다. 학습·선택 변동과 다중 비교 보정은 포함하지 않는다. `LR_v13`은 기본 튜닝 LR(0.3692/0.3471), `LR_selected`는 같은 메타 선택을 제공한 강한 대조군이다. 기본 LR만 이긴 점을 전체 LR 우월성으로 해석하지 않는다.

## fold별 최종 선택

| fold | 후보 | 선택 | 설정 | 경로 |
|---|---|---|---|---|
| 1 | three_selected | context_stack | {"C": 0.01, "mode": "profile"} | inner_bag |
| 1 | assisted_selected | context_stack | {"C": 0.01, "mode": "profile"} | inner_bag |
| 1 | LR_selected | context_stack | {"C": 0.01, "mode": "profile"} | inner_bag |
| 2 | three_selected | state_stack | {"C": 0.1, "shrink_people": 500} | fullfit |
| 2 | assisted_selected | stack | {"C": 0.1} | fullfit |
| 2 | LR_selected | context_stack | {"C": 0.1, "mode": "quality"} | fullfit |
| 3 | three_selected | context_stack | {"C": 0.01, "mode": "quality"} | fullfit |
| 3 | assisted_selected | context_stack | {"C": 0.01, "mode": "quality"} | inner_bag |
| 3 | LR_selected | context_stack | {"C": 0.01, "mode": "quality"} | inner_bag |
| 4 | three_selected | stack | {"C": 0.1} | fullfit |
| 4 | assisted_selected | stack | {"C": 0.1} | fullfit |
| 4 | LR_selected | stack | {"C": 0.1} | fullfit |
| 5 | three_selected | context_mixture | {"regularization": 0.1, "mode": "quality"} | inner_bag |
| 5 | assisted_selected | context_mixture | {"regularization": 0.1, "mode": "quality"} | inner_bag |
| 5 | LR_selected | mean | {} | inner_bag |

네 상태 결합은 fold2의 three에서만 선택됐다. 그 모델의 16개 소견×상태 셀은 각각 523–1,965명의 고유 참가자를 포함했고 모두 최소 양·음성 조건을 충족했다. 한 사람이 서로 다른 입력 view에서 다른 상태에 들어갈 수 있으므로 셀 인원 합은 전체 사람 수와 같지 않다. 이 인원은 학습 support이며 시험 성능이 아니다.

## 개선 이력

| 단계 | 모델 | 원래 입력 AP | 인공 누락 AP |
|---|---|---|---|
| v10 | Vortex_original_code | 0.3171 | 0.2994 |
| v10 | FREE_v01_physical_readout | 0.3593 | 0.3359 |
| v10 | Portia_STDP_proposal | 0.3487 | 0.3221 |
| v12 | LR_tuned | 0.3663 | 0.3441 |
| v12 | Vortex_improved | 0.3540 | 0.3329 |
| v12 | Physical_improved | 0.3559 | 0.3336 |
| v12 | Portia_improved | 0.3558 | 0.3315 |
| v13 | LR_tuned | 0.3692 | 0.3471 |
| v13 | Vortex_alternating | 0.3615 | 0.3418 |
| v13 | Physical_regularized | 0.3663 | 0.3451 |
| v13 | Portia_regularized | 0.3651 | 0.3432 |

여러 요소를 함께 바꾼 단계별 결과다. 향상을 특정 물리·생물학적 메커니즘 하나의 인과 효과로 귀속하지 않는다. v14 고정 세 모델 결합은0.3707/0.3527, LR 메타 대조군은0.3707/0.3509였다. v15는 LR에도 평균/조건부 후보를 추가해 대조군을 강화했다.

## 실제 실행 비용

| 체크포인트 | base 실행 수 | p50 ms | p95 ms |
|---|---|---|---|
| fold1 three_selected | 9 | 14.693 | 18.268 |
| fold2 three_selected | 3 | 5.344 | 7.159 |
| fold1 LR_selected | 3 | 1.382 | 2.053 |

CPU 1스레드, 가상 단건 입력, 10회 워밍업 후100회 측정. 입력 검증·전처리·전문가 전체·결합·온도 보정·출력 dict를 포함하며 로딩/JSON 직렬화/HTTP는 제외한다. 체크포인트별 경로가 다르므로 하나의 범용 지연시간으로 해석하지 않는다. LR이 더 빠르다. 이전 버전의 전처리 제외 지연시간과 직접 비교하지 않는다.

## 검증과 활용 결정

전체 로컬 테스트106개 통과. 최신 selected 세 종류×5fold의 모든 outer test/4view 재로딩 예측은 저장 OOF와 최대 절대차0이었다. 공개용 fold1/2 가중치도 전체4,068명×원래/누락 입력에서 원본과 최대차0이었다. batch 크기·행 순서를 바꾸는 경우 float32 파동 계산의 미세한 반올림 차이는 허용한다.

현재 정확도·속도·복잡성을 함께 보면 LR 대조군을 기본 선택으로 유지한다. 세 모델 결합은 추가 검증할 연구 후보다. 새로운 자료에서의 검증, 같은 계산량·용량 대조, 전체 재학습 ablation은 미완료다. 선행연구 원문과 주장 가능한 범위는 [RELATED_WORK.md](LEGACY_V15_RELATED_WORK.md)에 있다.
