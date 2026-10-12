#  Prain Backend (흩어진 협업을 하나로)

> **Prain**은 Discord, GitHub, Figma, Notion 등 여러 협업 도구를 한 화면에서 연동하고 통합 제어하는 올인원 워크스페이스 플랫폼입니다.  
> 본 레포지토리는 고성능 비동기 처리를 지원하는 **FastAPI** 기반의 백엔드 서버입니다.

---

## 기술 스택 및 채택 이유 (Tech Stack & Architecture)

### 1. 웹 프레임워크 & 런타임 (Server & Core)
| 기술 / 라이브러리 | 역할 | 도입 이유 및 배경 |
| :--- | :--- | :--- |
| **FastAPI** | 비동기 웹 프레임워크 | 유저 대시보드 로딩 시 GitHub, Figma, Notion 등 여러 외부 API를 **지연 없이 병렬(`async/await`)로 프록시 호출**하기 위해 채택했습니다. Swagger UI가 자동 생성되어 프론트엔드와의 협업 효율이 높습니다. |
| **Uvicorn (ASGI)** | 비동기 웹 서버 | FastAPI의 비동기 이벤트 루프(`asyncio`)를 안정적이고 빠르게 구동시키는 고성능 ASGI 서버입니다. |
| **Pydantic** | 데이터 유효성 검증 | 클라이언트 요청 Body/Query의 타입을 자동 검증하여 규격에 맞지 않는 데이터 유입을 사전에 차단(422 Validation Error)합니다. |

---

### 2. 보안 및 인증 계층 (Auth & Security)
외부 협업 도구의 민감한 권한(Access Token)을 백엔드가 안전하게 대리 관리하는 것이 핵심입니다.

* **`cryptography` (Fernet / AES-256):**
  * **외부 토큰 금고(Token Vault) 구축:** 유저가 연동한 GitHub, Discord, Figma의 Access Token을 데이터베이스에 평문으로 저장하지 않고 **AES-256 방식으로 양방향 암호화**합니다. 외부 API를 대리 호출할 때만 메모리상에서 복호화하여 보안 사고를 원천 방지합니다.
* **`python-jose` (JWT):**
  * **Prain 로그인 세션 유지:** 사용자가 로그인할 때 무상태(Stateless) JWT 토큰을 발급하여 확장성 높은 유저 인증 상태를 관리합니다.
* **`passlib[bcrypt]`:**
  * **자체 회원가입 비밀번호 해싱:** 이메일 가입 유저의 비밀번호를 안전한 단방향 해시 알고리즘(`bcrypt`)으로 암호화하여 저장합니다.

---

### 3. 데이터베이스 & ORM (Database)
* **`SQLAlchemy` (Async):**
  * 유저 정보(`User`), 연동 툴 정보(`UserIntegration`), 패널 레이아웃 상태(`WorkspaceLayout`) 간의 복잡한 1:N 관계를 객체 지향적으로 다루기 위한 비동기 ORM입니다.
* **`Alembic`:**
  * DB 스키마 형상 관리 도구로, 추후 새로운 연동 툴이 추가되어 DB 모델이 변경되더라도 기존 데이터를 보존하며 안전하게 마이그레이션합니다.

---

### 4. 외부 도구 연동 & AI (Integrations & AI)
* **`httpx`:** Python 표준 비동기 HTTP 클라이언트로, 전용 SDK가 없는 Figma REST API 등의 호출을 논블로킹(Non-blocking) 방식으로 처리합니다.
* **`PyGithub` & `notion-client`:** GitHub(PR, 이슈, 커밋) 및 Notion(페이지, 블록)의 복잡한 API를 간결하게 핸들링하는 공식 SDK입니다.
* **`openai`:** 회의록 보드 메모 및 디스코드 대화 내용을 바탕으로 핵심 액션 아이템과 요약본을 자동 생성합니다.

---

## 프로젝트 폴더 구조 (Project Architecture)

관심사 분리(SoC)를 위해 계층형(Layered) 아키텍처로 구성되어 있으며, 팀원 간 동시 작업 시 Git 충돌을 최소화하도록 설계되었습니다.

```text
prain-backend/
├── app/
│   ├── api/                      # [Presentation Layer] HTTP 요청/응답 라우터
│   │   ├── v1/
│   │   │   ├── auth.py           # 회원가입, 이메일 로그인, Discord/GitHub OAuth
│   │   │   ├── integrations.py   # Figma, Notion, GitHub 연동 및 데이터 프록시
│   │   │   ├── workspace.py      # 대시보드 화면 분할 레이아웃 저장/조회
│   │   │   ├── ai.py             # 회의록 자동 요약 및 AI 질의응답
│   │   │   └── notes.py          # 회의록 & 메모 CRUD
│   │   └── router.py             # v1 API 엔드포인트 통합 관리
│   │
│   ├── core/                     # [Infrastructure Layer] 전역 인프라 및 환경설정
│   │   ├── config.py             # .env 환경 변수 로드 (Pydantic Settings)
│   │   ├── database.py           # 비동기 SQLAlchemy 세션/엔진 설정
│   │   └── security.py           # JWT 발급, 패스워드 해싱, OAuth 토큰 암호화 유틸
│   │
│   ├── models/                   # [Domain Layer] DB 테이블 스키마 (SQLAlchemy Base)
│   │   ├── user.py               # 유저 기본 정보 테이블
│   │   ├── integration.py        # 연동된 툴(Figma, Discord 등) 및 암호화 토큰
│   │   ├── workspace.py          # 사용자별 그리드/분할 패널 레이아웃 상태
│   │   └── note.py               # 회의록 데이터 모델
│   │
│   ├── schemas/                  # [DTO Layer] 데이터 검증 및 직렬화 스키마 (Pydantic)
│   │   ├── user.py               # 회원가입/로그인 입출력 DTO
│   │   ├── workspace.py          # 패널 레이아웃 DTO
│   │   └── ai.py                 # AI 프롬프트 요청/응답 DTO
│   │
│   ├── services/                 # [Business Logic Layer] 비즈니스 로직 및 SDK 통신
│   │   ├── github_service.py     # GitHub API (이슈, PR, 커밋 조회)
│   │   ├── figma_service.py      # Figma API (프로젝트/파일 데이터 연동)
│   │   └── ai_service.py         # OpenAI LLM 회의록 요약 로직
│   │
│   └── main.py                   # FastAPI 인스턴스 초기화, CORS 및 미들웨어 설정
│
├── alembic/                      # DB 스키마 마이그레이션 이력 관리
├── .env                          # 환경 변수 (비밀키, DB 접속 정보 - Git 제외)
├── requirements.txt              # 파이썬 의존성 패키지 목록
└── README.md                     # 프로젝트 문서

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt


# .env 양식
PROJECT_NAME="Prain Backend"
API_V1_STR="/api/v1"
SECRET_KEY="YOUR_SUPER_SECRET_JWT_KEY"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
TOKEN_ENCRYPTION_KEY="FERNET_ENCRYPTION_KEY"
DATABASE_URL="sqlite+aiosqlite:///./prain.db"

# OAuth Keys
DISCORD_CLIENT_ID="YOUR_DISCORD_CLIENT_ID"
DISCORD_CLIENT_SECRET="YOUR_DISCORD_CLIENT_SECRET"
GITHUB_CLIENT_ID="YOUR_GITHUB_CLIENT_ID"
GITHUB_CLIENT_SECRET="YOUR_GITHUB_CLIENT_SECRET"

# 루트 폴더(Socket) 위치에서 실행 시
uvicorn app.main:app --app-dir backend --reload