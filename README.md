# KNHANES 2021–2024 비침습 대사이상 선별 연구

v21은 만 19–39세 4,068명의 원시 비침습 입력 19개를 동일한 fit 전용 engineered 72열로 변환해 네 소견을 평가했다. 같은 5개 PSU 외부 fold와 fit/validation/calibration/threshold/test 역할을 유지했고, 신경망은 seed 42/43/44의 독립 모델이다(LR은 seed 없음). EPF는 하나의 `(z,a,P)` 상태에 gate와 최종 P 읽기 4개를 더해 검사했다. 이미 v16–v20에서 살핀 코호트의 후속 탐색이며 진단 도구가 아니다.

## 선택 정책과 시험 결과

아래는 **validation에서** AP 또는 BCE 기준으로 설정·checkpoint를 선택한 뒤, clean **test에서** 측정한 보정 전 조사 가중 4소견 macro AP다. 신경망의 AP/BCE checkpoint는 각 후보의 동일 BCE 학습 궤적에서 고른다. LR_AP/LR_BCE는 동일 12-C grid에서 validation 정책별 C를 선택한다.

| 계열 | AP 선택 정책 | BCE 선택 정책 |
|---|---:|---:|
| enhanced EPF | 0.362684 | 0.366256 |
| enhanced MLP | 0.357676 | 0.367310 |
| LR | 0.361252 | 0.368316 |

사전 주대조 `EPF_AP−MLP_AP`는 +0.005008, paired PSU 95% 구간 [0.000559, 0.009631]이었다. `EPF_AP−LR_BCE`는 −0.005631 [−0.010765, −0.001013]이다. AP 정책에서의 EPF−MLP 결과를 모든 MLP 조건에 대한 우위로 확대하지 않는다. P 읽기·가소성·사차항 제거의 사전 대조 구간은 모두 0을 포함했다. 현재 성능과 크기를 함께 보면 **기본 연구 기준은 LR_BCE**다. 구간은 고정 OOF 예측 조건의 근사치이며 재학습·선택 불확실성, 다중비교 보정, 독립 외부 검증을 포함하지 않는다.

## 공개 예제와 실행

공개 package는 사전 지정한 fold1의 5개 예제 상태만 담으며 전체 코호트 재적합 모델이 아니다. 아래 입력은 합성값이다.

```bash
pip install -r requirements-structures.txt
python -m models.clinical_v21.predict --model LR_BCE --input example_clinical_v21.json --device cpu
python -m unittest discover -p "test*.py" -q
```

fold1 합성 입력 한 명, CPU 1 thread 중앙 추론 시간은 예열 10회 뒤 50회 측정값이다. 전처리·보정·cutoff 적용을 포함하고 모델 로딩은 제외했다. 배치와 GPU 결과 및 측정 조건은 [전체 runtime JSON](clinical_v21_public_runtime.json)에 있다.

| 공개 예제 | 저장 매개변수 | CPU 1인 중앙값 |
|---|---:|---:|
| enhanced_AP_EPF | 4,719 | 1.274 ms |
| enhanced_AP_MLP | 27,180 | 0.366 ms |
| LR_BCE | 292 | 0.248 ms |

훈련 궤적 963개와 정확한 checkpoint 재사용 21건을 동결했다. 독립 원본 재생은 고유 평가 730단위 × 13보기 = 9,490보기에서 최대 절대오차 0이었고, 공개 테스트 26개가 통과했다. 실행 검증은 임상적 타당성의 증명이 아니다.

[v21 결과·한계](CLINICAL_V21_RESULTS.md) · [사전 프로토콜](CLINICAL_V21_PROTOCOL.md) · [실행 노트북](clinical_v21_report.ipynb) · [54모델 전체 집계 JSON](clinical_v21_report.json) · [전체 재생 검증](clinical_v21_verification.json) · [runtime 상세](clinical_v21_public_runtime.json) · [소스 요약](clinical_v21_source_manifest.json)

이전 연구: [v20 결과와 한계](CLINICAL_V20_RESULTS.md) · [v20 노트북](clinical_v20_report.ipynb) · [v20 집계](clinical_v20_report.json) · [v20 검증](clinical_v20_verification.json) · [v20 소스 요약](clinical_v20_source_manifest.json) · [v20 마스크 코드](models/clinical_v20_missingness.py).

[v19 원결과·프로파일](CLINICAL_V19_RESULTS.md) · [v19 감사 노트북](clinical_v19_report.ipynb) · [v19 설계](CLINICAL_V19_DESIGN.md) · [v18 공동 학습](CLINICAL_MULTITASK.md) · [v17 단일 셀 수식](MECHANISM.md). 원자료·개인별 예측·역할 인덱스는 공개하지 않는다.
