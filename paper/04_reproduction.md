# 공개 집계와 로컬 재학습

## 완료된 공개 결과

`my_model.ipynb`의 모든 코드 셀은 집계만 읽어 실행했다. `paper/render_results.py`는 원자료 없이 주요 표를 출력하고 이미지 경로를 점검한다. 논문 그림은R 기본 그래픽으로 만들었다.

```bash
python -m pip install -r paper/requirements-results.txt
python paper/render_results.py
Rscript --vanilla paper/figures.R paper/results/v32 paper/figures
```

## 이용 등록한 원자료로 재학습

본인이 질병관리청에서 이용 등록한2020–2024 SAV를 `data/raw/knhanes/` 아래에 둔다. 원래 변수명과 HN20_all.sav 등 연도 식별 파일명을 유지한다. 원자료는Git에 포함하지 않는다. 완료된 실험은Python3.13·CPU학습(torch1 thread, tree2 threads)에서 수행했고 설명 계산은RTX4060도 사용했다. 의존성 버전은 `requirements-training.txt`와 `COMPLETION.json`에 있다. 외부API키는 사용하지 않는다.

```bash
python -m pip install -r paper/requirements-training.txt
python -m metabolic.epf_year_v32 prepare
python -m metabolic.epf_year_v32 context
python scripts/run_epf_year_v32.py
python -m metabolic.epf_year_v32 evaluate
```

최초prepare는 같은 엄격 코호트를 만들기 위한v31 데이터 준비만 호출한다. 실제v31 모델학습은 수행하지 않는다. 실행기는현재Python과4개병렬worker를 사용하고 작업별로그/선택동결/체크포인트/DONE을 `artifacts/epf_year_v32_202024/private`에 저장한다. 완료작업은재사용한다. 같은패널이라도 코드·자료가달라지면 기존private폴더를 별도보존하고 새실험폴더에서 실행한다.

EPF설명은 각연도마다 한 번 실행하고 집계한다.

```bash
python -m metabolic.epf_shap_v32 explain --year 2020
python -m metabolic.epf_shap_v32 explain --year 2021
python -m metabolic.epf_shap_v32 explain --year 2022
python -m metabolic.epf_shap_v32 explain --year 2023
python -m metabolic.epf_shap_v32 explain --year 2024
python -m metabolic.epf_shap_v32 aggregate
python -m metabolic.epf_marginal_shap_v32
python -m pytest tests -q
```

공개REPORT는이번에완료한수치의고정본이다. 재실행결과는로컬artifacts에생성되며 공개REPORT를 자동덮어쓰지 않는다. 연도별보류검증은내부검증이고공동분포군은연속측정값으로추가감독을받는다. 고정모델PSU구간은학습seed/탐색전체의불확실성을포함하지않는다.
