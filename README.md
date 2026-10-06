# EPF 신경망 기반 청년층 대사이상 유형 분류

국민건강영양조사 **2020–2024년, 20–39세, 공통 4,508명**의 15개 비침습 입력으로 다섯 대사소견군을 분류합니다. 교수님 피드백에 따라 **EPF 구조·연도 단위 검증·EPF 자체 SHAP**을 중심으로 갱신했습니다. [my_model.ipynb](my_model.ipynb)가 메인 노트북이며 모든 코드 셀에 실행 출력이 있습니다.

## 결과

| 모델 | 평균 시험연도 macro AP | 전체 OOF macro AP |
|---|---:|---:|
| EPF(jEPF) | 0.3571 | 0.3481 |
| XGBoost | 0.3544 | 0.3438 |
| 공동분포 MLP(jMLP) | 0.3516 | 0.3436 |
| LightGBM | 0.3516 | 0.3406 |
| 직접 MLP | 0.3509 | 0.3395 |
| Logistic regression | 0.3503 | 0.3358 |
| 공동분포 선형 평균(jLinear) | 0.3485 | 0.3417 |
| Random forest | 0.3470 | 0.3347 |

EPF가 가장 높은 점추정이지만 **EPF−LGBM +0.0055의 95% 조건부 구간은 [−0.0056, +0.0151]**로 우수성이 확인되지는 않았습니다. 동일 공동분포 출력 jMLP 대비는 +0.0055, 미보정 구간 [+0.0002, +0.0104]이며 보조 결과입니다. 고정 예측 PSU 구간은 재학습 변동을 포함하지 않습니다.

한 해 전체를 시험용으로 남기고 나머지 네 해 안에서 설정 선택 후 네 해 전체로 재학습하는 평가를 5번 반복했습니다. 모델당 후보8개, 최종seed42–44(LR은 한 적합), 총490회 학습·110개 최종 적합입니다. 직접 분류기와 공동분포 모델의 학습 정답 차이도 보고했습니다.

최대 확률로 한 군을 선택하면 혈압·혈당 단독군의 재현율은0%입니다. AP의 상대 순위 결과를 임상 선별 효과로 바꾸어 쓰지 않습니다. EPF 자체969명 확률 SHAP에서 허리둘레·WHtR·연령의 기여가 높았고, 생활습관 관련 차이를 분석했습니다. 설명값은 원인·예방효과가 아닙니다.

## 논문·모델·재현 자료

| 자료 | 내용 |
|---|---|
| [메인 노트북](my_model.ipynb) | 입력/정답 구분, EPF 구조, 연도별 평가, 혼동행렬, EPF SHAP; 실행 출력 포함 |
| [심사용 Word](paper/manuscript_review.docx) / [출판용 Word](paper/manuscript_publication.docx) | 5쪽 수정 원고; R 그림5개와 표2개를 참고문헌 뒤에 배치 |
| [본문 원본](paper/manuscript.md) | 서론·방법·결과·논의·참고문헌 |
| [연구 정의](paper/01_study_definition.md) / [비교 결과](paper/02_model_comparison.md) | 4,508명·15입력·5군·LOYO와 수치의 해석 범위 |
| [EPF 모델](metabolic/event_field_v23.py) / [실험 실행기](metabolic/epf_year_v32.py) | 현재 학습에 필요한 의존성16개만 포함 |
| [재현 지침](paper/04_reproduction.md) | 공개 집계 재생성과 이용 등록한 원자료의 로컬 재학습 |
| [완료·변경 기록](paper/results/v32/COMPLETION.json) / [변경 설명](paper/results/v32/AMENDMENTS.md) | 수치 해시·환경·사후 해석 점검·패키지 변경 |

29개 원자료 선택 열과 실제15개 예측 입력을 구분했습니다. 검사값·ID·진단·복약 문항은 예측 입력에 없습니다. 기진단 없음과 공복·검사·비임신·영양조사 조건을 적용한 공통 표본이며 동료의 기존6,588명 분석과 점수를 혼합하지 않습니다.

```bash
python -m pip install -r paper/requirements-results.txt
python paper/render_results.py
Rscript --vanilla paper/figures.R paper/results/v32 paper/figures
```

원자료·개인별 예측·행 단위 설명값·내부 체크포인트는 공개하지 않았습니다. 이전 공동 연구의 노트북과 원고는 [수정 전 main](https://github.com/kdw123654/digital-project/tree/b2893dfc039fc824d02dea98c51211985c279313)에 보존되어 있습니다. 기존 PDF·발표 HTML 및 `paper/model_comparison.ipynb`는 이전 분석 자료이며 현재 수치를 대신하지 않습니다.
