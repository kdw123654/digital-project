# 4개년 모델 비교와 모델 파일

**2021–2024년4,068명 전원을 사용한5-fold 비교**다. 각 사람은 자신을 학습하지 않은 모델로 정확히 한 번 평가했다.

[비교 조건·구조·학습법·결과](COMPARISON.md) · [실행 결과가 저장된 노트북](comparison.ipynb)

공통 신경망 틀의 대조 실험과 **모델별 학습법을 유지한 비교**를 분리했다.
Vortex는 원 코드의 복소수/파동/위상 구조를 호출하고, FREE는v0.1 물리 계산+분류 출력층,
포르티아는 설계 문서에 따른 새 LIF/STDP 후보로 구분했다. 원리 일부를 옮긴 작은 모델을 원형 전체라고 부르지 않는다.

| 대표 모델 | 정상 macro AP | 누락 macro AP |
|---|---:|---:|
| LR_basis |0.3663|0.3441|
| 공통 틀의 희소 어댑터 |0.3672|0.3448|
| 공통 틀의 라벨 이완 어댑터 |0.3672|0.3448|
| Vortex 원 코드 적용 |0.3171|0.2994|
| FREE v0.1 물리 저장소+분류 출력층 |0.3593|0.3359|
| 새 포르티아 STDP 설계 후보 |0.3487|0.3221|

이번4개년 결과에서는 **LR을 실용적인 기준 선택**으로 둔다. 수정 후보의 작은 개선은 원형 전체의 우월성을 뜻하지 않는다.
정적 자료·제한된 설정·이미 확인한 데이터이므로 모델 계열 전체의 우열이나 새 연도 성능을 확정하지 않는다.

## 모델 코드

- `models/native_candidates.py`, `train_vortex.py`: 서로 다른 구조/학습법.
- `models/native_sources/`: 이번에 호출한 Vortex와 FREEv0.1 원형 파일.
- `models/candidate_structures.py`, `adaptive_structures.py`, `baselines.py`: 공통 대조 모델들.
- `comparison.json`: 집계 결과, 전체 연도 포함 여부, fold별 역할, 하이퍼파라미터 선정.

## 기존 동결 모델 실행 예제

아래 가중치는 기존v5/v7/v8의 실행 예제다. 최신4개년 OOF 모델이나 전체4개년 최종 재학습 모델로 간주하면 안 된다.
출처·seed는 `models/config.json`에 기록했다. 비교 점수는 OOF 예측에서 구했으며 이 실행 예제의 점수가 아니다.

```bash
pip install -r requirements.txt
python models/predict.py --model nn --input example.json
python models/predict.py --model lr --input example.json

pip install -r requirements-structures.txt
python models/predict.py --model sparse --input example.json
python -m unittest test_models.py -v
```

API 키·외부 추론은 없다. 입력 예시는 가상 사례다. `WHtR=허리둘레(cm)/신장(cm)`.
정리된 변수명/범주는 `models/predict.py`에 있으며 모르는 값은 null로 둔다. 원시자료의 특수 결측 코드를 그대로 넣지 않는다.
연구용 최소 실행기이며 원래 API의 모든 교차 입력 검증·검토 게이트를 포함하지 않는다. sparse 예제에는 선별 컷오프가 없다.
노트북 재실행에는 `numpy pandas matplotlib ipykernel`이 필요하며 집계값을 다시 표시한다.
원자료·개인별 분할·개인별 예측은 공개하지 않았다.
