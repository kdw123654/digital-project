# 비침습적 대사이상 선별을 위한 단일 EPF — KNHANES 2021–2024

**최신 v18은 원래 주제인 청년층 대사이상 조기 선별 안에서 분류·수치 공동 학습을 비교한다.** 하나의 EPF 상태에서 네 이상 소견 확률과 다섯 검사 수치를 학습한다. 검사 수치는 학습 정답이며 입력은 19개 비침습 변수만 사용한다.

[비교 결과와 조건](CLINICAL_MULTITASK.md) · [실행된 노트북](clinical_comparison.ipynb) · [단일 셀 수식](MECHANISM.md) · [선행연구](RELATED_WORK.md)

EPF 공동 학습의 macro AP는 분류 전용 대비 +0.000891, LR 대비 -0.000203였다. 수치 macro NMSE의 개선량(수치 전용−공동)은 -0.000013였다. 수치 오차 감소와 선별 개선은 별개로 판단한다. 이번 비교에서는 공동 학습의 전반적인 개선이 확인되지 않아, LR 기준 모델을 유지하고 EPF 각 방식은 연구 후보로 남긴다.

| 모델 | macro AP ↑ | Brier ↓ |
|---|---:|---:|
| EPF 분류 전용 | 0.359851 | 0.106547 |
| EPF 수치 전용 | 0.364633 | 0.106373 |
| EPF 공동 학습 | 0.360742 | 0.106575 |
| MLP 공동 학습 | 0.361951 | 0.106463 |
| 로지스틱 회귀 | 0.360946 | 0.106811 |

4,068명, 5개 PSU fold × 3 seed. 같은 fit/validation/calibration/threshold/test 역할을 사용했다. MLP도 세 학습 방식을 모두 비교했다. 데이터 재탐색에 따른 탐색적 결과이며, 작은 점추정 차이를 임상 우월성으로 확정하지 않는다.

```bash
pip install -r requirements-structures.txt
python -m models.predict_clinical --model EPF_joint --input example.json
python -m unittest test_models test_improved_models test_hybrid_models test_event_field test_clinical_models -v
```

모델 가중치는 fold1/seed42의 연구용 예제다. 로컬에서 한 번 실행하며 API 키나 외부 추론 호출이 없다. 표의 점수는 전체 OOF 집계이며 공개 예제 하나의 점수가 아니다. 수치 추정은 실제 검사값이 아니며, 개발 민감도 목표90/95는 시험 보장이 아니다.

최신 파일은 `models/clinical_multitask.py`, `clinical_training.py`, `predict_clinical.py`, `clinical_checkpoints/`, `clinical_comparison.json`, `clinical_comparison.ipynb`, `CLINICAL_MULTITASK.md`다. 노트북은 집계 재계산용이며 `numpy pandas matplotlib ipykernel`이 필요하다. 원자료·개인별 예측·분할 인덱스는 공개하지 않는다.

이전 `COMPARISON.md`, `comparison.ipynb/json`, `predict_event_field.py`, `experiment_temporal.py`는 v17 역사적 실험이다. 센서·설비·생체신호 과제로 현재 범위를 확장하지 않는다. hybrid 파일은 v15, improved 파일은 v12, 초기 predict 모델은 v5/v7/v8이다. `living_alone`은 1인 가구 대리변수다.
