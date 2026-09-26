# v28 모델과 학습 핵심

실제 v28 실험의 **binary / joint × Linear / MLP / EPF / EPF_noP / EPF_step1** 10개 칸을 구성하는 코드다. 모델 라이브러리 6개를 옮기고, 원본의 모델 구성·로지스틱 앵커 선택·학습·확률 계산 함수와 상수를 그대로 추출했다. 추출한 함수 본문과 데코레이터는 유지했다. 원본 파일 해시, 추출 AST 해시, 줄바꿈 정리와 import 경로 변경은 [EXPORT_MANIFEST.json](EXPORT_MANIFEST.json)에 기록했다.

**훈련된 체크포인트는 포함하지 않는다.** 아래 예제는 무작위 합성 114열 입력과 합성 앵커로 초기화·forward만 실행한다. 학습된 성능이나 임상적 판단을 보여 주는 예제가 아니다.

저장소 루트에서 실행한다.

```bash
python -m pip install -r v28/requirements.txt
python -m models.clinical_v28.synthetic_demo
python test_clinical_v28.py
```

코어는 NumPy, SciPy, PyTorch, scikit-learn을 사용한다. 집계 노트북의 추가 의존성도 위 requirements에 포함되어 있다. 학습 기본값은 원 실험의 300 epoch 상한·patience 30·seed 5개·설정 4개다. 데모는 학습하지 않으며 테스트의 학습 검사는 합성 자료로 2 epoch만 실행한다.

## API와 준비된 입력

`from models.clinical_v28 import experiment as fx`로 불러온다.

| 함수 | 역할 |
|---|---|
| `fx.build(cell, seed, inputs, device)` | 전달한 fit 앵커와 fit 입력으로 모델 초기화 |
| `fx.logistic_anchor(item, labels_fit, labels_val)` | fit에서 로지스틱 회귀를 적합하고 validation 가중 BCE로 C 선택 |
| `fx.train_unit(inputs, cell, lr, wd, seed, device)` | 원본 학습 루프; 역할별 예측과 `meta`·`state_dict` 반환 |
| `fx.predict(design, model, x, sex, target, transform, device)` | 네 소견 확률, 16상태 확률, 행별 BCE/NLL 반환 |

`inputs`는 다음 계약의 dict다. `item`은 아래 속성을 가진 객체이며, 예제는 `SimpleNamespace`를 사용한다. fit·validation·calibration의 분리와 전처리 적합 범위는 호출자가 보장해야 한다. 원 연구의 114열 **순서와 인코딩 의미**를 유지해야 하며, 열 개수만 맞춘 임의 자료를 연구 입력으로 간주할 수 없다.

| 항목 | 계약 |
|---|---|
| `item.fit_x`, `item.fit_weights`, `item.fit_n` | fit 입력 `(n,114)`, 양의 가중치 `(n,)`, 행 수 |
| `item.fit_z`, `item.target_transform` | fit 표준화 표적 `(n,5)`, fit에서 적합한 `JointTargetTransform` |
| `item.validation_x`, `item.validation_weights` | 로지스틱 앵커 선택에 필요한 validation 입력·가중치 |
| `inputs.baseline` | joint Ridge 앵커 `coef (5,114)`, `bias (5,)`, `residual_cov (5,5)` |
| `inputs.logit_anchor` | binary 앵커 `(coef (4,114), bias (4,), C_by_flag)` 튜플 |
| `inputs.labels_fit` | 네 소견의 fit 이진 표적 `(n,4)` |
| `inputs.roles[role]` | `validation`, `calibration` 각각 `x (n,114)`, `z (n,5)`, `y (n,4)`, `w (n,)`, `sex (n,)` |

`JointTargetTransform`의 원시 표적 순서는 glucose·SBP·DBP·TG·HDL이고 변환 좌표는 log glucose·log DBP·log pulse pressure·log TG·log HDL이다. `sex` 코드는 1 또는 2다. 앵커는 fit 자료로 적합하고 validation으로 선택한 값을 전달한다. 합성 데모의 무작위 앵커는 이 연구용 적합을 대신하지 않는다.

`predict`는 점수 계산을 위해 표적을 요구한다. 정답 없이 forward만 계산할 때는 binary의 `model(x)`에 sigmoid를 적용하거나, joint의 `model.forward_parts(x)` 결과를 제공된 확률 함수로 변환한다. 별도의 raw38 생산 추론 CLI는 제공하지 않는다.

## 공개 범위

이 패키지는 준비된 배열에서 원 모델과 학습 루프를 실행하는 소스 공개다. 원자료, 개인별 예측, 역할 인덱스, fit 전처리 산출물, 기존 v24 Ridge 앵커, 실제 학습 체크포인트는 제외했다. 코호트 생성·114열 전처리·전체 1,000회 실행 관리·원 연구의 선택 동결·PSU 재표집 평가를 복원하는 패키지는 아니다. 집계 결과와 연구 한계는 [v28 보고서](../../v28/REPORT.md)에 있다.

공개 테스트는 이 패키지만 사용한다. 원 워크스페이스와의 AST·수치 동일성 대조는 별도의 로컬 감사이며 원자료 재학습이나 임상 검증을 뜻하지 않는다.
