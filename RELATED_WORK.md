# 단일 Event-Plastic Field와 선행 개념

2026-09-23. 현재 제안은 **같은 복소장 안에서 발화가 반대칭 결합 상태를 바꾸고, 그 결합이 다음 Hermitian 전파에 들어가는 셀**이다. 별도 전문가 출력의 결합은 v15/v16의 이전 설계다. [현재 수식](MECHANISM.md), [실행 결과](COMPARISON.md), [이전 v15 문헌 비교](LEGACY_V15_RELATED_WORK.md).

## 가까운 선행연구와의 관계

| 원 논문 | 이미 제안된 개념 | 현재 구현의 구체적인 관계 |
|---|---|---|
| Izhikevich, **Resonate-and-Fire Neurons**, Neural Networks 14, 883–894 (2001). [저자 원문](https://www.izhikevich.org/publications/resfire.htm) | 진동하는 내부 상태와 발화를 결합하는 신경 모형 | 진동+발화 자체는 새 개념이 아니다. EPF에서는 여러 복소 성분의 전파를 Cayley 연산자로 계산하고, 발화 흔적이 다음 전파의 반대칭 가소성 상태를 바꾼다. 원 논문의 신경생리 특성을 재현했다고 주장하지 않는다. |
| Miconi, Stanley & Clune, **Differentiable plasticity: training plastic neural networks with backpropagation**, ICML (2018). [공식 논문](https://proceedings.mlr.press/v80/miconi18a.html) | 가소성 계수와 연결 가중치까지 gradient로 학습하는 접근 | 입력에 따라 변하는 빠른 연결과 학습 가능한 가소성 계수를 사용한다. EPF의 빠른 상태는 실수 반대칭 행렬 P이며 iP를 Hermitian 전파에 넣는다. 학습 가능한 가소성 자체의 최초 제안은 아니다. |
| Neftci, Mostafa & Zenke, **Surrogate Gradient Learning in Spiking Neural Networks** (2019). [저자 공개본](https://arxiv.org/abs/1901.09948) | 불연속 발화의 학습을 위해 역전파에서 미분 근사를 사용하는 접근 | forward는 이진 발화, backward는 근사 미분이다. 해당 기법 자체는 선행 개념이며 구현한 근사 미분의 폭이 최적이라는 주장은 없다. |
| Arjovsky, Shah & Bengio, **Unitary Evolution Recurrent Neural Networks**, ICML (2016). [공식 논문](https://proceedings.mlr.press/v48/arjovsky16.html) | unitary 전파를 이용한 recurrent 계산 | EPF는 Hermitian H의 Cayley 변환으로 선형 단계의 norm을 보존한다. 감쇠·입력 구동·발화 리셋까지 포함한 전체 셀의 에너지 보존이나 장기 기억을 증명하지 않는다. |

이 표는 구조·개념 비교다. 위 논문 모델을 모두 재현한 성능 순위표가 아니다. 체계적인 전수 문헌 검색을 완료한 것도 아니므로, 동일하거나 유사한 결합의 선행 사례가 없다고 확정하지 않는다.

## 제안한 구현을 어떻게 한 메커니즘으로 구분하는가

하나의 상태 (z, a, P)와 하나의 학습 목표를 사용한다. z의 전파와 quartic 흐름이 발화를 만들고, 발화 및 과거 흔적의 순서 차이가 a sᵀ−s aᵀ를 통해 P를 갱신한다. 다음 H=A+iP가 바뀌므로 같은 셀의 다음 전파가 달라진다. 개별 전문가의 확률·logit를 평균하거나 gate로 섞지 않는다.

P의 반대칭성, iP의 Hermitian 성질, Cayley 선형 전파의 unitary 성질은 수식과 테스트로 확인한다. 이것은 전체 시스템의 안정성·학술적 최초성·일반적 성능 우월성의 증명이 아니다. 원래 Vortex/FREE/Portia의 모든 구조를 그대로 보존한 재현물도 아니다. 이들의 일부 원리를 하나의 새 갱신식으로 개조한 제안 구현이다.

복소 차원 16, 내부 갱신 6회, decoder 폭 64, 가소성 포화 크기 0.5 등은 이번 실험의 고정 설계값이다. 선행 논문이 이 숫자의 최적성이나 의학적 타당성을 뒷받침한다는 뜻은 아니다. 비교 모델의 입력 접근과 초기 선형 예측을 맞췄으며, 항 제거와 MLP·GRU+attention을 같은 명시된 예산에서 비교했다.

## 현재 근거가 허용하는 주장

단일 셀은 실제로 학습되었고, 같은 64개 이력 입력을 이용하는 합성 비선형 과제 1종에서 MLP·GRU+attention보다 낮은 평균 오차를 냈다. 반대칭 가소성·quartic 항을 제거한 대조군보다도 평균 오차가 낮았지만 모든 seed에서 이기지는 않았다. 발화 자체 제거는 아직 비교하지 않았다.

KNHANES 선별 우월성, 실제 의료 시계열 예후, 창 사이 장기 기억, 모든 연산 항의 필수성, 더 빠른 실행, 일반적인 표현력 우월성은 입증되지 않았다. 세 아이디어가 등장한다는 사실보다 **전파와 가소성이 하나의 상태 갱신에서 서로를 바꾸는 구현 및 그 조건별 효용**을 제안의 구체적인 특징으로 삼는다.
