# 발화로 전파를 갱신하는 단일 복소장 셀 — EPF

**Event-Plastic Field는 하나의 상태 갱신 셀이다. 전파가 발화를 만들고, 발화가 연결 상태를 바꿔 다음 전파를 바꾼다.**

Vortex·물리·Portia 모델을 따로 호출하던 결합을 바꿨다. 공유 복소장 z, 활동 흔적 a, 반대칭 결합 P를 하나의 목표함수로 학습한다. 별도 전문가의 예측·평균·gate는 없다. [수식과 구조](MECHANISM.md) · [비교 결과](COMPARISON.md) · [선행연구](RELATED_WORK.md) · [실행된 노트북](comparison.ipynb)

| 확인한 조건 | EPF | 대조군 | 해석 |
|---|---:|---:|---|
| 합성 비선형 이력 변환 NMSE ↓, 3 seed | 0.00009755 | MLP 0.00079754 / GRU+attention 0.00037789 | 이번 과제·예산에서 오차 감소 |
| KNHANES 2021–2024 macro AP ↑ | 0.36161 | MLP 0.36334 / LR 0.36476 | 선별 우월성 확인 안 됨 |
| 합성 과제 선택 trial 평균 학습 시간 | 124.6초 | MLP 11.3초 / GRU+attention 60.3초 | 더 큰 계산 비용 |

모든 신경망의 초기 예측을 동일하게 맞추고, 파라미터 수·입력 접근·학습 후보·정지 조건을 명시했다. 시계열은 같은 64개 이력을 이용한 다항식 과제 1종이다. EPF는 이를 받아 6회 내부 갱신하며, 창 간 상태 유지나 장기 기억을 검증한 것은 아니다. GRU 한 seed는 최대 320 epoch에 도달했다. 학술적 최초 제안이나 일반적 우월성을 확정하지 않는다.

## 실행

```bash
pip install -r requirements-structures.txt
python -m models.predict_event_field --input example.json
python experiment_temporal.py --seed 42
python -m unittest test_models test_improved_models test_hybrid_models test_event_field -v
```

합성 실험은 `--seed`를 생략하면 42/43/44를 모두 실행한다. 학습 출력은 `runs/`에 저장된다. 건강 추론은 로컬 NPZ 가중치를 읽으며 API 키나 외부 호출이 없다. 입력은 `example.json`의 19개 비침습 변수이며 결측은 null, WHtR은 허리둘레/신장이다.

공개 건강 모델은 fold1/seed42, 2,205명 fit으로 학습한 연구용 가중치다. 표는 4,068명·5개 PSU fold·3 seed의 OOF 집계다. 네 동시 이상 소견의 다중 라벨 예측이며 확정 진단·미래 발병 예측이 아니다. 종합 선별 컷오프의 개발 민감도 목표 95%는 시험 보장이 아니다. `living_alone`은 1인 가구 대리변수다.

## 파일

- `models/event_plastic_field.py`: 새 단일 셀. `event_field_runtime.py`: 동일 수식의 CUDA 실행 최적화.
- `models/event_field_baselines.py`: 별도로 학습·평가한 MLP와 GRU+attention 대조군.
- `models/predict_event_field.py`, `models/event_field_checkpoint/`: 건강 추론·NPZ 가중치·검증·해시.
- `experiment_temporal.py`: 완료한 합성 비교의 독립 재현 스크립트.
- `comparison.json`, `comparison.ipynb`, `COMPARISON.md`: 집계·실행된 시각화·조건과 한계. 노트북은 집계 재계산용이며 `numpy pandas matplotlib ipykernel`이 필요하다.

이전 `predict_hybrid.py`와 hybrid 모델은 **v15 역사적 실험**이다. `predict_improved.py`는 v12, 기존 `predict.py`의 모델은 v5/v7/v8 예제다. 최신 제안 셀의 실행 경로와 구분한다. 이전 비교 문서는 `LEGACY_V15_COMPARISON.md`에 보존했다. 원자료·개인별 예측·분할 인덱스·pickle/PT 체크포인트는 공개하지 않았다.
