# 모델 비교와 실행 파일

**현재 NN 유지 · LR 기준 모델 유지 · 희소 구조를 후속 후보로 선택**했다.

[비교 결과와 실패 원인·수정 내용](COMPARISON.md) · [실행 결과가 보이는 노트북](comparison.ipynb)

| 포함 모델 | 용도 | 상태 |
|---|---|---|
| `nn` | 기존 그룹 NN, 네 소견·종합 위험 | 현재 모델, 765 parameters |
| `lr` | 확장 기저 로지스틱4개 | 1차 비교 기준 모델, 292 parameters |
| `sparse` | 입력 누락에 맞춘 희소 상태 갱신 | 2차 실험 후보, 서비스 교체 미적용 |

기존 `nn`은2차 증강 NN이 아니며 `lr`도2차 증강 LR이 아니다. 제공 가중치의 실험명·seed는 `models/config.json`에 명시했다.
각 표는 세 seed 평균이고 제공 체크포인트는 지정된 한 seed이므로 수치가 완전히 같지는 않다.

## 실행

```bash
pip install -r requirements.txt
python models/predict.py --model nn --input example.json
python models/predict.py --model lr --input example.json

# 희소 후보/구조 코드 실행에는 torch가 추가로 필요
pip install -r requirements-structures.txt
python models/predict.py --model sparse --input example.json
python -m unittest test_models.py -v
```

API 키와 외부 추론 호출은 없다. 예시는 가상 사례다. `WHtR`에는 허리둘레(cm)/신장(cm)을 입력한다.
나이19–39세, 성별1/2는 필수이며 모르는 나머지 값은 생략/null로 전달한다. 현재 구현은 연구용이다.
원시조사의 특수 결측 코드8/9/88/99를 직접 입력하지 않는다. `example.json`은 정리된 변수 값이다.
정확한 변수 순서·범주값은 `models/predict.py`의 `FEATURES`, `CATEGORIES`, `EXTRA_CAT`에 있다.
이식한 실행기는 연구용 최소 인터페이스이며 원래 API의 모든 입력 상호 일관성 검사·검토 게이트를 포함하지 않는다.
`sparse`는 확률만 출력하며 별도 선별 컷오프를 선정하지 않았다.

## 파일

- `models/predict.py`: NN/LR/희소 후보의 실제 추론.
- `models/*.npz`, `config.json`: 가중치·전처리 통계·확률 보정. pickle을 사용하지 않는다.
- `models/candidate_structures.py`: 1차 신경망 구조.
- `models/adaptive_structures.py`: 2차 수정 구조.
- `models/baselines.py`: 비교한 LR/RF/HGB 정의.
- `COMPARISON.md`, `comparison.json`, `comparison.ipynb`: 비교 조건·지표·저장된 출력.

공개 파일은 비교 내용과 모델 실행 자료로 한정했다. 전체 학습 재현에는 별도 확보한 KNHANES 원자료와 원 연구의 역할 분할이 필요하다.
노트북은 내장 집계값을 다시 표시하며 학습 재실행을 하지 않는다(`numpy pandas matplotlib ipykernel` 필요).
현재 모델의 개별 소견 기준은 혈당≥100, 혈압≥130/85, TG≥150, HDL 남<40/여<50이다. 진단된 당뇨·고혈압 라벨이 아니다.
