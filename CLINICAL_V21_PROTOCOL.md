# v21 비침습 대사이상 선별: 학습 병목과 EPF 기전 후속 평가

이 문서는 KNHANES 2021–2024의 19–39세 4,068명에 대해 시험 예측을 분석하기 전에 고정한 설계다. v17–v20 코드·분할·결과는 변경하지 않는다. 이미 여러 차례 확인한 동일 코호트의 **후속 탐색**이며, 독립 임상 확증이나 새로운 시계열 검증이 아니다. 실제 시험 결과는 별도 결과 문서에만 기록한다.

## 자료와 역할

- 네 연도, 동일 원시 비침습 입력 19개와 fit 역할에서만 맞춘 engineered 72열 인코딩을 사용한다. 허리둘레, BMI, WHtR은 EPF·MLP·LR에 동일하게 제공한다. 혈액검사와 혈압 수치는 네 이상 소견의 정답일 뿐 입력이 아니다.
- 기존 PSU 단위 5개 외부 fold와 각 fold의 fit/validation/calibration/threshold/test 역할을 그대로 사용한다. 코호트 행·정답·역할 인덱스와 원자료 해시를 확인한다. 전처리, 초기화, 수치 embedding knot는 해당 fold의 fit에만 맞춘다.
- 후보 탐색과 epoch 선택은 fit 및 clean validation에 한정한다. 전체 선택과 재훈련 checkpoint를 동결한 다음에만 calibration·threshold·test 역할을 열고, test는 한 번 최종 평가한다. test 성적을 바탕으로 설정을 다시 고르지 않는다.

## 선형 기준과 후보 예산

각 fold에서 LR은 동일 72열과 같은 fit 정보·조사 가중치를 사용한다. 양의 12개 C 값 `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300`을 전부 적합한다. clean validation의 보정 전 가중 4소견 macro AP 최대를 `LR_AP`, 가중 BCE 최소를 `LR_BCE`로 각각 선택한다. 동률 규칙은 코드에 고정한다. `LR_BCE` 계수는 신경망의 선형 시작점이기도 하다. 동일 BCE 및 L2 정의의 PyTorch double/full-batch LBFGS 선형모델이 sklearn LR과 목적함수·검증 예측·gradient 수렴 면에서 합치하는지 먼저 검증한다. 서로 다른 solver의 계수 완전 동일성은 요구하지 않는다. AdamW 선형 대조는 별도 최적화 방식으로 분리해 보고한다.

EPF와 MLP에 plain 32개, enhanced 32개 후보씩 동일한 규모의 탐색 예산을 준다. 각 후보는 fold별 seed 42로 fit/validation만 사용하여 훈련한다(총 5×2단계×2계열×32 = 640 후보 fit). plain EPF는 이전 상태식, P readout 없음, gate 없음이며 linear ratio `0, 0.01, 0.1, 1`과 구조·학습률·weight decay의 2수준 조합이다. enhanced **EPF**에는 P 특징 4개와 gate를 더한다. enhanced **MLP**는 P 상태 없이 gate/shrink 대조를 적용한다. enhanced 후보에는 ratio `0, 0.1`, shrink `0, 0.001`과 동일 2수준 조합을 준다. 값은 이번 실험의 **설계값**이며 최적성 주장이 아니다.

두 단계 모두 EPF는 하나의 `(z, a, P)` 상태와 복소 장·발화·반대칭 가소성 피드백을 유지한다. 별도 전문가의 출력 결합을 사용하지 않는다. MLP는 동일한 선택·학습 예산의 대조군이다. 한 훈련 궤적에서 검증 AP 최고와 BCE 최저 checkpoint/epoch를 각각 저장하며, 이를 서로 독립된 학습 목적함수 실험으로 부르지 않는다. 각 fold·단계·계열에서 AP 및 BCE 정책의 최적 후보를 validation으로만 고르고, 선택된 config의 합집합을 seed 43/44에도 재훈련한다. 선택된 AP config와 BCE config 각각에서 **두 checkpoint를 모두** 평가한다. 같은 config의 AP/BCE checkpoint 대조와 config 자체의 선택 정책 대조를 구분한다. seed 42 결과와 43/44 결과는 독립 모델 세 개로 평가하며 예측 앙상블로 합치지 않는다.

## 읽기 출력과 손실의 코드 계약

| 모델 | 비선형 경로 | 최종 P 읽기 | 네 소견 gate | residual shrink |
|---|---|---|---|---|
| LR_AP / LR_BCE | 없음; 동일 72열의 선형 로짓 4개 | 없음 | 없음 | 없음 |
| MLP plain / enhanced | 72열에서 MLP residual | 없음 | plain 1 / enhanced 학습, 초기 0.1 | plain 0 / enhanced 후보 0 또는 0.001 |
| EPF plain / enhanced | 하나의 복소 상태 `z`, 활동 trace `a`, 반대칭 가소성 `P` | plain 없음 / enhanced 4개 bilinear 특징 | plain 1 / enhanced 학습, 초기 0.1 | plain 0 / enhanced 후보 0 또는 0.001 |

EPF enhanced의 마지막 상태에서 `h=[Re(z), Im(z), |z|², a, energy]`에 `f_r=û_rᵀ P v̂_r` (`r=1,…,4`)을 붙인다. `u_r`,`v_r`는 **별도로 학습하는 실수 벡터**를 각각 정규화한 것이며 같은 벡터로 묶지 않는다(학습 중 불일치를 강제하는 제약은 없다). 이 값은 P의 최종 읽기 특징이고 별도 전문가 출력이 아니다. `h`와 `f`의 평균·표준편차는 그 fold의 clean fit에서만 정하고 표준편차 하한 0.02를 적용한다. 네 로짓은 `linear_c(x)+s_c·readout_c(h,f)`이며 enhanced의 `s_c=sigmoid(g_c)`, `g_c` 초기값은 `log(0.1/0.9)`다. plain은 `s_c=1`이고 P 특징은 readout에 넣지 않는다. MLP도 같은 선형 경로와 gate 규칙을 쓰지만 `(z,a,P)` 상태나 P 읽기는 없다.

fit 전역 조사 가중치를 `w_i=원가중치_i / fit평균가중치`로 **한 번** 정규화하여 `Σ_fit w_i=N_fit`로 둔다. mini-batch `B`의 훈련 목적은 `mean_{i∈B}[w_i·mean_{c=1..4} BCEWithLogits(logit_ic,y_ic)] + ||W_linear||²_F/(2·C_BCE·N_fit·4) + λ·mean_{i∈B}[w_i·mean_c (s_c·readout_ic)²]`이다. `C_BCE`는 해당 fold의 fit/validation에서 선택한 LR_BCE 값이며 L2는 선형 **가중치만** 벌점화하고 bias는 제외한다. 선형 경로가 없는 field-only에서는 이 L2가 0이다. shrink는 gate를 통과한 residual에 적용한다. 매 batch 가중치 합으로 다시 나누지 않는다. 비선형 학습은 AdamW로 수행하되, 선형 경로는 `base_lr×linear_lr_ratio`와 AdamW weight decay 0인 별도 그룹이다. 비선형 행렬에만 후보 weight decay를 적용하고 gate·bias·정규화·동역학 scalar에는 weight decay 0을 쓴다. 이 AdamW 설정과 위 명시적 선형 L2는 다른 항이다. LR 자체는 별도 sklearn/LBFGS 기준이며 AdamW 선형 대조는 구분한다.

## 사전 대조와 조건부 probe

1차 지표는 **보정 전, 네 소견의 조사 가중 macro AP**다. 주대조는 `enhanced_AP_EPF − enhanced_AP_MLP`, `enhanced_AP_EPF − LR_AP`, `enhanced_AP_EPF − LR_BCE`다. plain 대비와 AP/BCE 선택 정책 대비는 별도 표로 보고한다. `plain_AP_EPF`는 필수 비교다. 13개 clean/누락/측정오차 보기는 같은 test 사람에 적용하고, clean이 주 조건이다.

enhanced AP 선택 config를 fold별로 고정한 뒤 세 seed에서 EPF의 no-P-readout, 같은 2Dr 규모의 반대칭 bilinear 용량 대조, no-gate, no-shrink, no-plasticity, no-quartic, field-only와 **β=5 smooth event를 학습·추론에 동일 적용한 대조**, numeric piecewise learned embedding(수치 6개, 각 E=4), native-zero/native-prior 초기화를 검사한다. numeric knot는 fit 분포의 분위수 `.05, .275, .5, .725, .95`에서 만들고 중복 knot 처리와 양끝 선형 외삽을 사용한다. 이는 이번 설계값이며 선행 PLE를 그대로 재현했다는 뜻이 아니다. 가능한 MLP gate/shrink/numeric/native 대조도 같은 조건에서 수행한다. shrink=0과 같은 설정의 재사용은 새 fit으로 세지 않는다. field-only는 선형 경로 제거도 포함하므로 순수 단일 요인 효과가 아니다. gate-off, shrink=0, prior bias readout 등 복합 변경의 해석 역시 조건부로 제한한다. `native_zero`는 공통 선형 W=0,b=0, `native_prior`는 W=0과 fit 유병률 기반 b로 정의한다. 두 native 대조는 LR 계수를 주입하지 않지만 공통 규제 λ는 fit/validation으로 고른 `LR_BCE`의 C에 의존하므로 완전히 LR 독립인 조건은 아니다.

## 보정, 선별, 불확실성

선택 동결 후 모든 계열에 동일한 v19 `RiskCalibration` 절차를 clean calibration 역할에서 적용한다. 보고에는 (1) 보정 전 네 소견 확률과 그 독립 합집합, (2) 소견별 보정 후 확률과 독립 합집합 `q0`, (3) Fréchet 범위의 공통 γ 조정 위험을 구분한다. γ는 calibration에서만 적합한다. γ 조정은 사람 간 위험 순위를 바꿀 수 있고 fold별 소견 보정도 pooled OOF 순위를 바꿀 수 있으므로 보정 후 any AP 변화를 EPF 기전 효과로 돌리지 않는다. 보정 전후의 AP·Brier·NLL·각 소견 지표를 clean OOF 전체와 fold별로 보고한다.

`raw_any_precal`, `raw_any`, `adjusted_any` 세 위험 각각에 대해 별도 clean threshold 역할에서 90%/95% 민감도 **목표** cutoff를 정한다. 이를 test에 고정 적용하여 실제 민감도·의뢰율을 계산한다. 시험 ROC에서 목표 민감도를 보간한 기술적 수치는 운영 cutoff 결과와 구분한다. 역할별 수와 유효 표본 수, 보정 계수 경계, γ의 분포와 끝점, 유병률 차이도 기록한다.

주대조와 미리 명시한 소수의 기전 대조에는 동일한 full-frame 조사 PSU 2,000회 재표집으로 paired 차이의 95% 백분위 구간을 계산한다. 이는 **고정 OOF 예측에 조건부인 구간**이며 학습·선택의 전체 불확실성 또는 다중 비교 보정을 포함하지 않는다. 같은 seed 간 paired 차이 세 개의 평균·표준편차·부호·범위도 별도로 보인다. 학습 불확실성을 이 세 seed나 PSU 구간 하나로 대체하지 않는다. 나머지 probe 및 강도 sweep는 탐색적 기술 결과다.

훈련과 활동에는 후보·선택 epoch, 가중치/gradient 갱신, P 투영 크기와 gate, 선형·장 출력의 평균/중심 표준편차/RMS, 계산 시간 및 실패를 기록한다. 저장된 전체 파라미터 수와 실제 최적화한 학습 가능 파라미터 수를 분리한다. linear ratio 0에서 선형 가중치 288개와 bias 4개, 총 **선형 매개변수 292개**가 동결되어도 저장된 모델의 구조·용량이 사라지는 것은 아니다. 출력 RMS 비율을 성능 기여율로 해석하지 않는다. 기전 기여는 같은 비교 조건에서 **재훈련한 ablation의 시험 성능 차이**로만 조심스럽게 해석한다.

원자료·역할 인덱스·개인별 예측은 로컬 private 산출물에만 둔다. 학습·원자료 SHA-256은 `SOURCES.json`, 수치 분석 코드는 `ANALYSIS_SOURCES.json`, 문서·노트북 생성 코드는 `PRESENTATION_SOURCES.json`에 각각 기록한다. 도표 수정 때문에 동결된 수치 분석 소스가 재작성되지 않게 한다.

## 구성요소의 문헌상 동기와 이번 설계의 범위

[Gorishniy et al. (NeurIPS 2022)](https://papers.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)는 수치 입력의 구간별 선형 embedding을 다루고 MLP에도 수치 embedding이 도움이 될 수 있음을 보였다. [Miconi et al. (ICML 2018)](https://proceedings.mlr.press/v80/miconi18a.html)는 Hebbian 가소성 매개변수를 역전파로 학습할 수 있음을 보였다. [Neftci et al. (2019)](https://arxiv.org/abs/1901.09948)는 이산 발화의 학습을 위한 surrogate gradient 접근을 정리했다. 이는 구성요소를 검사할 근거이지, 이번 반대칭 P·4개 투영·gate 초기값 0.1·fit 분위수 knot·규제 선택의 최적성이나 신규성, KNHANES 우위를 입증하는 근거가 아니다. EPF의 내부 step은 한 사람의 **횡단면 입력에 대한 상태 계산**이며 실제 장기 추적 시간이나 종단 관측을 뜻하지 않는다.
