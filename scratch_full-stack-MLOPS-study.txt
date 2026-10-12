# 🚀 Zero-to-Hero MLOps Full-Stack Bootcamp
> **"단순한 모델 학습을 넘어, 백엔드·프론트엔드·클라우드·자동화까지 아우르는 프로덕션 MLOps 엔지니어로의 도약"**

본 레포지토리는 Python 기반으로 **프론트엔드(UI), 백엔드(FastAPI), 머신러닝/딥러닝(Scikit-Learn/PyTorch), MLOps 파이프라인(MLflow, DVC, Docker, GitHub Actions, K8s, Drift Monitoring)** 전 과정을 바닥부터 실무 수준까지 단계별로 정복하는 스터디 레포지토리입니다.

---

## 🗺️ MLOps Full-Stack 아키텍처 큰 그림

```
[ 사용자 브라우저 / 프론트엔드 (Streamlit / React) ]
                         │
                         ▼  (HTTP REST API / JSON)
        [ 백엔드 서빙 서버 (FastAPI + Pydantic) ]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[ 실시간 추론 엔진 (Inference) ]    [ 배치/비동기 큐 (Worker) ]
(PyTorch / ONNX Runtime)
        │
        ▼ (메트릭 / 예측 로그 수집)
[ 모니터링 시스템 (Evidently AI + Prometheus + Grafana) ]
        │ (성능 저하 감지 시 트리거)
        ▼
[ MLOps 오케스트레이션 & 자동 재학습 파이프라인 (Prefect / Airflow / GitHub Actions) ]
        │
        ├─ 데이터 버전 관리 (DVC)
        ├─ 실험 추적 & 모델 레지스트리 (MLflow)
        └─ 컨테이너 이미지 패키징 (Docker & CI/CD)
```

---

## 📚 전체 커리큘럼 & 학습 진도표 (Roadmap)

### 🟢 Phase 1: MLOps 엔지니어링 기초 (기반 다지기)
- [x] **[제1강] [MLOps 환경 세팅 및 Git/GitHub 실무 워크플로우](01_environment_and_git/README.md)**
  - *MLOps 풀스택 개요, VS Code 최적화, Python 가상환경 격리, Git 실무, 환경 진단 스크립트(`check_env.py`)*
- [x] **[제2강] [파이썬 기초 바닥부터 완전 정복 (Core Fundamentals)](02_python_basics/README.md)**
  - *변수, 4대 자료형, 조건문/반복문, List/Dict/Tuple/Set, 컴프리헨션, 함수, 파일 입출력(CSV)*
- [x] **[제3강] [모던 파이썬 & 객체지향 설계(OOP) 및 Pydantic 데이터 검증](03_modern_python_and_pydantic/README.md)**
  - *Clean Code, Type Hinting(`typing`), Pydantic v2 모델 설계, dataclass, Garbage In 차단*
- [ ] **[제4강] 터미널과 쉘 스크립트 실무 (PowerShell, Bash, WSL2)**
  - *리눅스 필수 명령어, 자동화 쉘 스크립트, 환경 변수 관리*

### 🟡 Phase 2: 머신러닝 & 딥러닝 핵심 파이프라인
- [ ] **[제5강] 프로덕션 데이터 파이프라인 (Pandas & Polars, 데이터 품질 검증)**
- [ ] **[제6강] Scikit-Learn 머신러닝 실전 파이프라인 (Feature Engineering & 재현성)**
- [ ] **[제7강] PyTorch 딥러닝 코어 (Dataset, DataLoader, 모듈화 설계, GPU 가속)**

### 🟠 Phase 3: 백엔드 API & 모델 서빙 (Serving & Backend)
- [ ] **[제8강] FastAPI 기반 초고속 REST API 구축 (비동기 처리, 라우팅, 에러 핸들링)**
- [ ] **[제9강] 머신러닝 모델 서빙 아키텍처 (인메모리 캐싱, 배치 추론, 동시성 처리)**
- [ ] **[제10강] Docker 컨테이너화 (Dockerfile 작성, 멀티스테이지 빌드, 이미지 경량화)**

### 🔵 Phase 4: 프론트엔드 & 인터랙티브 대시보드 (Full-Stack UI)
- [ ] **[제11강] Streamlit & Gradio를 활용한 고속 인터랙티브 AI 데모 UI 구축**
- [ ] **[제12강] 모던 웹 프론트엔드 연동 기초 (HTML/CSS/JS 및 Next.js REST API 통신)**
- [ ] **[제13강] 실시간 모델 제어 및 예측 결과 인터랙티브 시각화 대시보드**

### 🟣 Phase 5: MLOps 핵심 생태계 (실험 추적 & 버전 관리)
- [ ] **[제14강] MLflow를 활용한 실험 추적(Tracking) 및 파라미터/메트릭 로깅**
- [ ] **[제15강] DVC(Data Version Control)를 활용한 대용량 데이터 및 모델 가중치 버전 관리**
- [ ] **[제16강] MLflow Model Registry를 통한 모델 승인/스테이징/프로덕션 배포 수명주기**

### 🟤 Phase 6: 파이프라인 자동화 & CI/CD
- [ ] **[제17강] GitHub Actions를 이용한 CI/CD 자동화 (Ruff, pytest, Docker 자동 빌드)**
- [ ] **[제18강] 파이프라인 오케스트레이션 (Prefect / Apache Airflow로 파이프라인 스케줄링)**

### 🔴 Phase 7: 프로덕션 배포 & 관측성 (Final Boss)
- [ ] **[제19강] 클라우드 배포 기초 & Kubernetes(K8s) 모델 서빙 배포**
- [ ] **[제20강] 모델 성능 저하 & 데이터 드리프트 실시간 모니터링 (Evidently AI, Prometheus)**
- [ ] **[제21강] 최종 실전 풀스택 캡스톤 프로젝트: 실시간 예측 및 자동 재학습 파이프라인 완성**


---

## ⚡ 스터디 시작하기 (Quick Start)

### 1. 가상환경 생성 (Windows PowerShell 기준)
```powershell
# Python 3.10+ 권장
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\Activate.ps1
```
> 만약 PowerShell에서 스크립트 실행 오류(`Execution_Policies`)가 발생하면:
> ```powershell
> Set-ExecutionPolicy Unrestricted -Scope Process
> .venv\Scripts\Activate.ps1
> ```

### 2. 제1강 환경 검증 실행
```powershell
cd 01_environment_and_git
pip install -r requirements.txt
python scripts/check_env.py
```

---

## 📝 스터디 규칙 & 깃허브 커밋 컨벤션

| 태그 | 설명 | 예시 |
| :--- | :--- | :--- |
| `feat:` | 새로운 실습 코드나 기능 추가 | `feat: add check_env.py hardware diagnostics` |
| `docs:` | 강의 문서나 마크다운 정리 | `docs: complete lesson 01 notes and assignment` |
| `fix:` | 실습 코드 오류 수정 | `fix: resolve cuda detection bug in check_env.py` |
| `refactor:` | 코드 리팩토링 및 클린 코드화 | `refactor: modularize system info extraction` |
