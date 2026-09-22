> 이전 v15 전문가 결합 실험의 보존 문서다. 현재 단일 셀은 [최신 비교](COMPARISON.md)와 [수식](MECHANISM.md)을 따른다.

# 선행연구와 우리 모델의 차이

확인일: 2026-09-22. 원 논문·저자 공개본·공식 학회 페이지를 대상으로 확인했다. 체계적 문헌고찰이나 신규성에 대한 완전한 검색은 아니다.

**상태에 따라 전문가를 나누고 결합한다는 발상 자체는 기존 연구에 있다.** 우리 연구의 기여 후보는 복소 파동·물리 저장소·스파이킹 모델을 동일 정형 자료에서 유지하면서, 입력 결측과 소견별 예측 불일치에 따라 결합하고 그 효용을 대조군과 검증하는 구체적인 설계다. 현재 결과만으로 새로운 알고리즘의 최초 제안이나 LR 대비 우월성을 주장하지 않는다.

## 가장 가까운 결합·선택 연구

| 선행연구 | 기존에 제안한 부분 | 우리 구현과 겹치는 부분 | 구별되는 구현·검증 범위 |
|---|---|---|---|
| Jacobs et al., **Adaptive Mixtures of Local Experts**, Neural Computation 3:79–87, 1991. [원문](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) | 입력에 따라 gate가 전문가의 기여를 결정 | 입력 조건부 전문가 결합 | 우리는 같은 19개 정형 입력을 받는 서로 다른 메커니즘의 모델을 별도로 학습하고 OOF 출력에서 결합한다. 단일 end-to-end MoE의 최초 제안이 아니다. |
| Wolpert, **Stacked Generalization**, Neural Networks 5:241–259, 1992. [원문 공개본](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf), [DOI](https://doi.org/10.1016/S0893-6080(05)80023-1) | 학습 부분집합으로 나머지 자료를 예측한 값을 상위 학습기의 입력으로 이용 | inner OOF 예측으로 meta model 학습 | PSU를 분리한 3-fold OOF와 조사 가중치를 적용한다. OOF stacking 자체의 신규성을 주장하지 않는다. |
| Cruz et al., **META-DES**, Pattern Recognition 48:1925–1935, 2015. [저자 공개본](https://arxiv.org/abs/1810.01270), [DOI](https://doi.org/10.1016/j.patcog.2014.12.003) | 여러 meta-feature로 입력별 분류기 적합성을 추정하고 동적으로 선택 | 예측값·입력 상태에 따라 결합 방식 변화 | 우리는 이웃 기반 적합성 추정을 재현하지 않았다. 결측 여부와 label별 expert-logit 표준편차로 네 상태를 정의하고, 전역·국소 stacker를 축소 결합한다. arXiv 등록은 2018년이지만 저널 출판은 2015년이다. |
| Novosad, Carano & Krishnan, **A task-conditional mixture-of-experts model for missing modality segmentation**, MICCAI 2024. [공식 논문](https://papers.miccai.org/miccai-2024/paper/2509_paper.pdf), [DOI](https://doi.org/10.1007/978-3-031-72114-4_4) | MRI 모달리티의 가용성 코드를 softmax gate에 넣어 convolution experts를 결합하고 self-distillation 사용 | 결측 상태로 결합 조건을 정의 | 우리는 정형 자료의 개별 변수 결측을 다루고 모델 출력에서 결합한다. convolution MoE·self-distillation은 구현하지 않았다. **결측 상태별 결합에 선행연구가 없다는 주장을 직접 제한하는 연구다.** |
| Sun & Sheng, **QA-MoE**, IJCAI 2026, pp.6886–6894. [공식 페이지](https://www.ijcai.org/proceedings/2026/766), [원문](https://www.ijcai.org/proceedings/2026/0766.pdf) | 임상 다중모달 입력의 evidence 기반 품질 추정, 부분집합 선택, ternary routing을 분리 | 입력 품질에 따라 결합을 바꾸는 목적 | 우리는 Dirichlet evidence나 모달리티 부분집합 선택을 쓰지 않는다. 단순 expert disagreement는 보정된 epistemic uncertainty가 아니다. 모든 전문가를 계산한 후 결합하므로 sparse routing의 계산 절감도 주장하지 않는다. |

이 표는 **개념·구조 비교**다. META-DES, MICCAI 모델, QA-MoE를 동일 자료에서 재현해 성능을 이겼다는 실험은 없다. 논문마다 데이터·목표·지표가 달라 발표 점수를 우리 macro AP와 직접 순위 비교하지 않았다.

## 세 전문가의 메커니즘에 대한 선행연구

| 관련 원리와 원 논문 | 우리 모델에서 유지한 것 | 이번 구현의 범위 |
|---|---|---|
| 복소수·unitary 전파: Arjovsky, Shah & Bengio, **Unitary Evolution Recurrent Neural Networks**, ICML 2016. [공식 논문](https://proceedings.mlr.press/v48/arjovsky16.html) | Vortex의 복소 진폭·위상, Cayley 전파와 GP 비선형 변환 | 원 논문의 장기 시계열 RNN 재현이 아니다. 정적 입력에 유한 단계 파동을 적용하고, 별도 smooth phase proxy와 교대 readout 학습을 쓴다. 복소수/unitary라는 이유만으로 새로운 원리라고 주장하지 않는다. |
| 물리 저장소: Nakajima et al., **Information processing via physical soft body**, Scientific Reports 5:10487, 2015. [원 논문](https://www.nature.com/articles/srep10487) | 동역학 응답을 계산 특징으로 이용하는 관점 | 우리는 실제 연성체 장치를 사용하지 않는다. 기존 FREE v0.1 quartic 매질/BAOAB의 소프트웨어 시뮬레이션으로 구동·감쇠 특징을 만들고, 추가 특징만 정규화한다. 최신 FREE v0.2 전체 시스템의 평가도 아니다. |
| 조절 STDP·eligibility trace: Florian, **Reinforcement Learning Through Modulation of Spike-Timing-Dependent Synaptic Plasticity**, Neural Computation 19:1468–1502, 2007. [저자 원문](https://florian.io/papers/2007_Florian_Modulated_STDP.pdf) | Portia 후보의 LIF 발화, E/I 연결, 국소 가소성·조절 신호 | 우리의 조절 신호는 학습 자료의 label에서 온다. 원 논문의 reward 기반 RL, 실제 거미 커넥톰, 생물학적 검증을 재현했다는 뜻이 아니다. |

## 우리가 실제로 추가한 설계

`statewise_hybrid.py`의 소견 j에 대한 상태는 다음과 같다.

1. 입력 완전성: age·sex를 제외한 입력 중 하나라도 결측인지 판정한다. `clean` 평가에도 원자료의 자연 결측은 남아 있다.
2. 예측 불일치: 세 전문가의 j번째 logit 표준편차가 meta 학습 자료의 조사 가중 75백분위수를 초과하는지 판정한다.
3. 두 조건의 조합으로 `complete_agree`, `complete_disagree`, `missing_agree`, `missing_disagree`를 만든다. 정답 소견이나 혈액 수치로 상태를 정하지 않는다.
4. 전역 stacker와 상태별 stacker의 logit을 `alpha = n / (n + tau)`로 결합한다. `n`은 해당 상태의 **중복 제거된 사람 수**다. 같은 사람의 원래 입력과 누락 입력을 별도 사람으로 세지 않는다. 양성·음성이 각각 20명 미만이면 그 상태에서는 전역 모델을 쓴다.

75백분위수, 최소 양·음성 20명, shrinkage tau 후보 100/500, 입력군 구성은 **이번 연구의 설계 선택**이다. 위 논문이 이 숫자의 최적성이나 의학적 타당성을 입증한 것은 아니다. tau는 개발 검증 자료에서 선택했으며, 나머지 고정 기준의 민감도 분석은 하지 않았다.

별도로 입력 품질/프로필과 expert logit의 상호작용, softmax 혼합도 비교했다. 이들은 각각 `conditional_hybrid.py`의 `ContextualLogitStack`, `ContextualMixture`다. stacker 계수는 음수도 가능하므로 ‘전문가의 기여 비율’로 읽으면 안 된다. softmax 모델의 gate만 비음수·합계 1의 혼합 비율이다.

세 전문가의 출력층과 결합층에도 로지스틱 학습이 쓰인다. `three`는 **별도의 LR 전문가를 넣지 않았다**는 뜻이며, 모델 내부의 모든 선형·로지스틱 계산을 제거했다는 뜻은 아니다.

## 결과로 뒷받침되는 범위

- 동적 결합 후보군은 세 모델의 고정 결합 대비 macro AP가 원래 입력에서 0.3707→0.3709, 인공 누락에서 0.3527→0.3534였다. 차이의 조건부 95% 구간은 두 조건 모두 0을 포함한다.
- 같은 탐색 절차의 LR 대조군은 0.3734 / 0.3530이었다. 따라서 상태 결합의 확정적 효과나 LR 대비 우월성은 입증되지 않았다.
- 명시적인 네 상태 stacker는 세 모델의 최종 선택에서 5개 outer fold 중 **fold2 한 번** 선택됐다. 다른 fold는 연속 context 모델 또는 고정 결합을 선택했다. 전체 결과를 네 상태 모델 하나의 성능이라고 쓰면 안 된다.
- 이 자료는 여러 차례 개선에 재사용됐다. 고정 OOF 예측의 PSU bootstrap은 전체 개발·선택 과정의 불확실성을 보장하지 않는다.

현재 적절한 기술은 **“서로 다른 메커니즘을 유지한 정형 자료 전문가의 상태 조건부 결합과 대조 실험”**이다. 추가로 필요한 근거는 새로운 자료에서의 검증, 고정 결합 대비 안정된 효과, 상태 기준 민감도, 동일 용량·계산량 및 재학습 ablation이다. 이 항목들을 완료했다고 주장하지 않는다.
