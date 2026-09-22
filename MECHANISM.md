# 하나의 새 결합 메커니즘: 발화가 전파를 바꾸는 복소장 셀

사용자의 수정 방향에 따라 v16의 독립 전문가 결합을 중단하고, `EventPlasticField`라는 **단일 상태 갱신 연산자**를 구현했다. 이것은 우리 구현의 제안 셀이며 학술적 최초 여부를 확정한 표현은 아니다.

## 공유 상태와 갱신식

상태는 복소장 z, 발화 흔적 a, 빠르게 변하는 결합 P다. 각기 다른 신경망의 출력이 아니라 같은 셀의 내부 상태다. 전파가 발화를 만들고, 그 발화가 다음 전파를 바꾸는 순환을 닫는다.

아래 t는 현재 실험에서 **셀 내부의 6회 갱신**을 뜻한다. 건강에서는 한 사람의 입력 벡터, 시계열에서는 동일한 64개 관측 이력 벡터를 매 갱신에 넣는다. 각 forward마다 상태를 초기화한다. 실제 시간의 64개 관측을 순차적으로 셀에 흘려보내거나 창 사이에 상태를 유지하는 실험은 아니다.

```
H_t       = A + i P_t                         (A는 학습되는 Hermitian 행렬)
U_t       = (I + i dt H_t)^(-1)(I - i dt H_t)
z_drive   = rho U_t z_t + B x_t
z_phase   = z_drive exp(-i dt g |z_drive|^2)
theta_t   = theta_base (1 + h a_t)
s_t       = 1[|z_drive|^2 >= theta_t]
z_(t+1)   = z_phase sqrt(1 - r theta_t s_t / |z_drive|^2)
a_(t+1)   = lambda a_t + (1-lambda) s_t
P_(t+1)   = bounded_odd(delta P_t + eta(a_t s_t^T - s_t a_t^T))
```

실제 구현은 작은 intensity에서 surrogate gradient가 폭증하지 않도록 reset 분모를 `max(intensity, theta.detach())`로 둔다. 발화가 실제로 발생한 경우의 forward 값은 위 식과 같다. 발화는 이진값이며, 학습 시 그 경계를 surrogate gradient로 미분한다.

빠른 결합 P는 입력으로부터 계산되는 단기 상태다. 추론에 정답을 넣지 않으며 다른 사람과 공유해 업데이트하지 않는다. 학습되는 A·입력 투영·속도·문턱·가소성 계수·출력층은 하나의 목표함수로 함께 최적화한다.

출력은 같은 장의 실수부·허수부·intensity·활동 흔적·리셋 전 intensity를 하나의 비선형 decoder가 읽는다. 전문가별 예측이나 투표·확률 평균은 없다. 공통 선형 초기화 경로는 비교 모델에도 동일하게 주며 끝까지 학습 가능하다. 셀의 비선형 보정은 원래 입력을 직접 받는 MLP 우회 경로를 갖지 않는다.

## 설계상 확인할 수 있는 성질

- P가 실수 반대칭이고 갱신의 포화 함수가 odd이면, 다음 P도 반대칭이다. 따라서 iP는 Hermitian이다.
- H가 Hermitian이면 Cayley 전파 U는 정확한 수학에서는 unitary다. CUDA/CPU 테스트에서 수치 오차 범위의 보존을 확인한다.
- 이 보존은 **선형 전파 단계의 norm**에 대한 것이다. 구동·감쇠·발화 리셋을 포함한 전체 셀의 에너지 보존이나 생물학적 타당성을 주장하지 않는다.
- 발화가 결합을 바꾸므로 같은 기본 A에서도 입력의 상태에 따라 다음 전파 연산자가 달라진다. 이것이 세 출력을 나중에 섞는 구조와의 구체적인 차이다.

## 기존 병목의 수정

1. MLP에는64개 이력을 제공하고 다른 모델에는 압축 상태만 제공했던 차이를 없앴다. 새 시계열 비교는 모든 모델이 같은64개 관측값에 접근한다. GRU 대조군도 전체 window hidden state의 attention을 사용한다.
2. 사후 PCA16과 선형 출력층을 제거했다. 입력 투영·상태 표현·비선형 decoder를 과제에 맞춰 함께 학습한다.
3. 발화할 때 amplitude를 일정 비율로 잘라내던 초기 구현을 바꿨다. threshold에 해당하는 intensity를 차감하고, 리셋 전 intensity도 보존해 신호 소실을 줄였다.
4. 같은 학습 자료의 LR/Ridge로 공통 선형 경로를 초기화했다. MLP·GRU·제안 셀의 초기 예측이 같으며, 고정된 LR fallback은 아니다.
5. 마지막 epoch가 최고인데도 수렴으로 해석하지 않는다. 학습률 감소·validation plateau를 기록한다. 시계열의 scheduler/patience 단위를 절대 MSE1e-5로 통일했다. budget limit이면 그대로 표시한다.
6. CUDA `linalg.solve`의 반복적인 CPU 동기화를 `solve_ex`와 묶음 오류 검사로 줄였다. 전파식·gradient의 일치는 별도 검증했다. 초기 두 전이에서는 P가 정확히0인 성질로 같은 기본 전파 행렬을 재사용한다.

## 비교 방식

- 건강: 기존 v16의 동일4개년·5개 PSU fold·3 seed·별도 보정/threshold 역할. Field, 파라미터를 맞춘 residual MLP, 셀의 가소성 항 제거, quartic 항 제거를 비교한다.
- 시계열: 같은64개 이력, 공통 초기 선형 예측, 비선형 decoder, 파라미터 규모를 맞춘 MLP와 GRU+attention. 동일 셀의 작동 항을 제거한 대조군도 사용한다.
- 첫 네 지연 복원 과제는 공통 선형 초기화로 대부분 해결된다. 이 점수로 셀 자체의 장기 기억 능력을 주장하지 않는다. 새 비교의 핵심은 비선형 변환과 작동 항의 기여다.
- 이전 v17 시간 학습은 test 생성 전에 scheduler 단위 문제를 발견해 개발 실행으로 보존했다. 수정된 시간 비교는 모든 모델을 다시 같은 초기 조건에서 시작하며 `aligned_runtime` 아래 저장한다.

## 관련 선행 개념

[Resonate-and-fire](https://www.izhikevich.org/publications/resfire.htm)는 진동 상태와 발화를 결합한 선행 모델이다. [Differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a.html)는 가소성까지 gradient로 학습하는 선행 접근이다. [Surrogate gradient](https://arxiv.org/abs/1901.09948)는 발화의 학습 경로와 관련된다. 이러한 개념 자체를 새로 발명했다고 주장하지 않는다. 우리의 구체적 설계는 **발화 시점의 반대칭 가소성 상태를 Hermitian 전파 연산자에 넣고, 동일한 장의 구동·quartic 흐름·리셋을 함께 학습하는 셀**이다.

코드: `models/event_plastic_field.py`, `models/event_field_runtime.py`. 대조군: `models/event_field_baselines.py`. 재현: `experiment_temporal.py`. [비교 결과](COMPARISON.md).
