# 상태 조건부 모델 결합 비교 — KNHANES 2021–2024

**세 모델의 동적 결합은 누락 조건에서 소폭 개선됐지만, 동일하게 보강한 LR보다 우수하다고 결론 내릴 수 없다.**

[전체 실험 비교](COMPARISON.md) · [선행연구와 차이](RELATED_WORK.md) · [실행 결과가 포함된 노트북](comparison.ipynb)

만 19–39세 4,068명, 같은 5-fold PSU 분할에서 Vortex·물리 저장소·Portia 후보를 비교했다. 네 개 동시 이상 소견을 예측하는 **다중 라벨** 과제다. 당뇨·고혈압 진단 분류가 아니다.

| 모델 | 원래 입력 AP | 인공 누락 AP |
|---|---|---|
| three_static | 0.3707 | 0.3527 |
| three_dynamic | 0.3709 | 0.3534 |
| three_selected | 0.3697 | 0.3528 |
| assisted_selected | 0.3719 | 0.3533 |
| LR_selected | 0.3734 | 0.3530 |

`static`은 고정 결합 후보 중 개발 검증으로 선택, `dynamic`은 조건부 결합 후보 중 선택, `selected`는 둘을 합친 후보에서 선택한 절차다. 최종 선택 절차와 동적 후보군을 따로 보고하며 시험 점수가 높은 쪽으로 교체하지 않았다.
`assisted`는 별도의 LR 전문가도 포함한다. `LR_selected`에는 같은 메타 학습·입력 조건부 보정·선택 기회를 줬다.

현재 활용 기준은 **LR 대조군 유지**, 세 모델 결합은 누락 강건성 연구 후보로 두는 것이다. 재사용 자료의 탐색적 결과이며 신규 자료 검증이나 우월성 입증은 아니다.

## 무엇을 만들었나

- Vortex: 복소 파동·Cayley/GP를 유지하고 smooth phase proxy 및 교대 readout/encoder 학습 적용.
- 물리 저장소: quartic 매질·BAOAB·국소 적응·구동/감쇠를 유지하고 추가 특징의 분산과 중복 제어.
- Portia 후보: LIF/EI·STDP·항상성을 유지하고 회로 특징의 차원·정규화 조정.
- 결합: OOF 점수 기반 고정 결합, 입력 조건 상호작용, softmax gate, 결측×소견별 예측 불일치의 네 상태 stacker 비교.

상태별 결합은 MoE·동적 앙상블·결측 모달리티 연구에 선행 사례가 있다. 우리 설계의 구체적인 차이와 아직 검증하지 못한 부분은 [선행연구 비교](RELATED_WORK.md)에 원 논문 링크와 함께 적었다.

## 실행

```bash
pip install -r requirements-structures.txt
python -m models.predict_hybrid --input example.json
python -m models.predict_hybrid --fold 2 --input example.json
python -m models.predict_hybrid --category LR_selected --input example.json
python -m unittest test_models.py test_improved_models.py test_hybrid_models.py -v
```

실제 v15 선택 모델의 fold1 세 모델 결합·LR 대조군과 fold2 네 상태 결합을 JSON/NPZ로 제공한다. 예제 입력은 가상 사례다. API 키나 외부 추론 호출이 필요 없다. 원자료·개인별 예측·분할 인덱스는 공개하지 않았다.

표의 점수는 5개 fold 전체의 OOF 성능이며, 제공한 단일 fold 가중치의 점수가 아니다. 전체 표본 재학습 모델이나 임상 운영 모델로 해석하지 않는다.
모든 전문가를 계산하므로 단일 신경망의 한 번 전파나 LR보다 빠른 모델이라는 주장은 하지 않는다. 현재 CPU 단건 실행은 결합 경로에 따라 약 5–15ms, LR 대조군 약 1.4ms였다([측정 범위](COMPARISON.md)).

입력은 `example.json`과 `models/predict.py`의 FEATURES를 따른다. `WHtR=허리둘레(cm)/신장(cm)`, 결측은 null. 원시 조사 특수 코드를 그대로 넣지 않는다. `living_alone`은 1인 가구 대리변수이며 실제 자취 여부를 직접 측정한 값이 아니다. 새 모델에는 선별 컷오프를 정하지 않았다.

## 파일 안내

- `models/*_generalizing.py`, `vortex_alternating.py`: 최신 세 전문가의 구조·학습 코드.
- `models/conditional_hybrid.py`, `statewise_hybrid.py`, `hybrid_ensembles.py`: 결합 모델의 학습·추론 코드.
- `models/predict_hybrid.py`, `hybrid_checkpoints/`: 최신 실행 경로·가중치·원본 일치 검증·속도 측정.
- `comparison.json`, `COMPARISON.md`, `RELATED_WORK.md`, `comparison.ipynb`: 집계·해석·선행연구.
- `predict_improved.py`와 `improved_checkpoints/`는 v12, 기존 `predict.py`의 NN/LR/sparse 가중치는 v5/v7/v8 예제다. 최신 결합 모델과 구분한다.

노트북은 내장 집계를 재계산·시각화하며 모든 코드 셀 결과가 저장돼 있다. 원자료 없이 실행되며 `numpy pandas matplotlib ipykernel`이 필요하다. **원자료에서 전체 모델을 다시 학습하는 노트북은 아니다.**
