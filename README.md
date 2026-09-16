# 2030 청년층 미인지 대사이상 조기 선별 머신러닝 모델
> **Early Screening of Unaware Metabolic Risk in Young Adults (Ages 19–39) Using Non-Invasive Predictors: Machine Learning Analysis based on KNHANES (2021–2024)**

---

## 📌 1. 연구 배경 및 목적
- **연구 배경**: 최근 2030 청년층에서 대사이상(당뇨, 고혈압, 이상지질혈증) 발생 위험이 급증하고 있으나, 청년층은 무증상과 낮은 검진 수검률로 인해 본인의 위험 상태를 인지하지 못하는 비율(미인지율)이 높습니다.
- **연구 목적**: 고가의 침습적(Invasive) 혈액 검사 없이, 자가 계측(허리둘레, BMI, WHtR) 및 생활습관 설문(음주, 흡연, 신체활동) 등 **비침습적(Non-invasive) 변수만을 활용하여 미인지 대사이상 고위험군을 조기 선별(Early Screening)**하는 머신러닝 예측 모델을 구축합니다.

---

## 📊 2. 데이터 및 연구 대상자
- **데이터 소스**: 질병관리청 국민건강영양조사(KNHANES) 최신 4개년도(2021~2024) 원시자료
- **연구 대상자**: 만 19세 ~ 39세 청년 중 기진단 치료자(고혈압, 당뇨, 이상지질혈증 약물 복용자)를 제외한 미인지 대상자

---

## 🎯 3. 목표변수(Target) 및 설명변수(Features)
### 목표변수 (Target: 0 또는 1)
다음 기준 중 1개 이상 만족 시 미인지 고위험군(Target = 1):
1. **공복혈당**: `HE_glu >= 100 mg/dL`
2. **혈압**: `HE_sbp >= 130 mmHg` 또는 `HE_dbp >= 85 mmHg`
3. **중성지방**: `HE_TG >= 150 mg/dL`
4. **HDL 콜레스테롤**: `HE_HDL_st2 < 40 mg/dL(남)` 또는 `< 50 mg/dL(여)`

### 설명변수 (Non-invasive Features)
- **인구사회학적 요인**: 연령(`age`), 성별(`sex`), 소득수준(`incm`), 교육수준(`edu`)
- **신체 계측**: 체질량지수(`HE_BMI`), 허리둘레(`HE_wc`), 허리둘레-키 비율(`WHtR = HE_wc / HE_ht`)
- **건강 행태**: 현재 흡연(`sm_presnt`), 월간 음주(`dr_month`), 유산소 신체활동(`pa_aerobic`)

---

## 🤖 4. 머신러닝 모델 및 성능
- **비교 모델**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **예측 성능**: LightGBM 기준 ROC-AUC 약 **0.783** 달성
