# TASKS.md
# MediAssist AI - 태스크 분해서

---

## 태스크 ID 규칙

```
[Phase]-[Category]-[Number]

Phase:
  P1 = Foundation
  P2 = ML Models
  P3 = RAG System
  P4 = Integration
  P5 = Deployment
  
Category:
  ENV = Environment
  DB  = Database
  API = API Development
  ML  = Machine Learning
  RAG = RAG System
  UI  = User Interface
  SEC = Security
  TST = Testing
  OPS = Operations
```

---

## Phase 1: Foundation (Week 1-4)

### P1-ENV-001: 프로젝트 저장소 초기화
```yaml
priority: P0
estimate: 4h
assignee: DevOps Engineer

tasks:
  - [ ] GitHub 저장소 생성
  - [ ] Branch protection 규칙 설정
  - [ ] 프로젝트 디렉토리 구조 생성
  - [ ] README.md, CONTRIBUTING.md 작성

output:
  - "GitHub repository"
  - "Project structure"
```

### P1-ENV-002: Docker 개발 환경 구성
```yaml
priority: P0
estimate: 8h
depends_on: [P1-ENV-001]

tasks:
  - [ ] docker-compose.yml 작성
  - [ ] Dockerfile 작성 (각 서비스)
  - [ ] .env.example 작성
  - [ ] Makefile 명령어 추가

output:
  - "docker-compose.yml"
  - "Dockerfiles"
```

### P1-ENV-003: CI/CD 파이프라인 구축
```yaml
priority: P0
estimate: 8h
depends_on: [P1-ENV-002]

tasks:
  - [ ] GitHub Actions 워크플로우 (lint, test, build)
  - [ ] 배포 워크플로우 (staging, prod)
  - [ ] 시크릿 설정

output:
  - ".github/workflows/*.yml"
```

### P1-DB-001: 데이터베이스 스키마 설계
```yaml
priority: P0
estimate: 8h

tasks:
  - [ ] ERD 다이어그램 작성
  - [ ] DDL 스크립트 작성
  - [ ] Alembic 마이그레이션 설정

output:
  - "docs/erd.png"
  - "migrations/"
```

### P1-API-001: FastAPI 프로젝트 구조 설정
```yaml
priority: P0
estimate: 8h

tasks:
  - [ ] 서비스별 프로젝트 구조
  - [ ] Config, Dependencies 설정
  - [ ] 공통 스키마 정의
  - [ ] 에러 핸들러

output:
  - "services/{auth,patient,analysis,report}/"
```

---

## Phase 2: ML Models (Week 5-10)

### P2-ML-001: 학습 데이터 준비
```yaml
priority: P0
estimate: 40h

tasks:
  - [ ] MIMIC-III 데이터 접근
  - [ ] 데이터 전처리 파이프라인
  - [ ] Train/Val/Test 분할
  - [ ] DVC 버전 관리

output:
  - "data/processed/*.parquet"
```

### P2-ML-002: Medical NER 모델 구현
```yaml
priority: P0
estimate: 24h
depends_on: [P2-ML-001]

tasks:
  - [ ] BioBERT 모델 로드
  - [ ] NER 태그 스키마 정의
  - [ ] 학습 파이프라인
  - [ ] 평가 메트릭

output:
  - "models/symptom_classifier/ner.py"
  - "models/symptom_classifier/weights/ner_model.pt"
```

### P2-ML-003: 증상 분류 모델 구현
```yaml
priority: P0
estimate: 32h
depends_on: [P2-ML-002]

tasks:
  - [ ] Feature Engineering
  - [ ] XGBoost 모델
  - [ ] Random Forest 모델
  - [ ] Ensemble 구현
  - [ ] 하이퍼파라미터 튜닝
  - [ ] SHAP 설명 가능성

output:
  - "models/symptom_classifier/classifier.py"
  - "models/symptom_classifier/weights/"

targets:
  accuracy: ">= 85%"
  f1_macro: ">= 0.80"
```

### P2-ML-004: 이미지 분석 모델 구현
```yaml
priority: P0
estimate: 40h

tasks:
  - [ ] CheXpert 데이터셋 준비
  - [ ] DenseNet121 모델 구현
  - [ ] Multi-label 분류 헤드
  - [ ] Grad-CAM 시각화
  - [ ] 모델 평가

output:
  - "models/image_analyzer/"

targets:
  auc_roc: ">= 0.90"
  inference_time: "< 500ms"
```

### P2-ML-005: 위험도 예측 모델 구현
```yaml
priority: P1
estimate: 24h
depends_on: [P2-ML-003]

tasks:
  - [ ] 위험 요인 정의
  - [ ] Neural Network 모델
  - [ ] 캘리브레이션

output:
  - "models/risk_predictor/"
```

### P2-ML-006: ML 서비스 API 구현
```yaml
priority: P0
estimate: 16h
depends_on: [P2-ML-003, P2-ML-004, P2-ML-005]

tasks:
  - [ ] 모델 로딩 서비스
  - [ ] 추론 API 엔드포인트
  - [ ] Celery 비동기 처리
  - [ ] 단위 테스트

output:
  - "services/analysis/services/ml_service.py"
```

---

## Phase 3: RAG System (Week 11-14)

### P3-RAG-001: 문서 수집 파이프라인
```yaml
priority: P0
estimate: 24h

tasks:
  - [ ] PubMed API 연동
  - [ ] PDF/HTML 파서
  - [ ] 메타데이터 추출
  - [ ] 문서 정제

output:
  - "rag/ingestion/"
```

### P3-RAG-002: 임베딩 파이프라인
```yaml
priority: P0
estimate: 16h
depends_on: [P3-RAG-001]

tasks:
  - [ ] 청크 분할 전략
  - [ ] BioBERT 임베딩
  - [ ] ChromaDB 구축

output:
  - "rag/embedding/"
  - "rag/vectordb/"
```

### P3-RAG-003: 하이브리드 검색 구현
```yaml
priority: P0
estimate: 16h
depends_on: [P3-RAG-002]

tasks:
  - [ ] Dense Retrieval
  - [ ] BM25 Sparse Retrieval
  - [ ] Reciprocal Rank Fusion

output:
  - "rag/retrieval/"
```

### P3-RAG-004: Reranker 구현
```yaml
priority: P0
estimate: 8h
depends_on: [P3-RAG-003]

tasks:
  - [ ] Cross-Encoder 모델
  - [ ] Reranking 파이프라인

output:
  - "rag/reranking/"
```

### P3-RAG-005: LLM 통합
```yaml
priority: P0
estimate: 16h
depends_on: [P3-RAG-004]

tasks:
  - [ ] LLM 클라이언트 (GPT-4)
  - [ ] 프롬프트 템플릿
  - [ ] 답변 생성기
  - [ ] 인용 파싱

output:
  - "rag/generation/"
```

### P3-RAG-006: RAG 서비스 API
```yaml
priority: P0
estimate: 16h
depends_on: [P3-RAG-005]

tasks:
  - [ ] RAG 서비스 클래스
  - [ ] 쿼리 API
  - [ ] 캐싱

output:
  - "services/analysis/services/rag_service.py"
```

---

## Phase 4: Integration & UI (Week 15-18)

### P4-UI-001: React 프로젝트 설정
```yaml
priority: P0
estimate: 8h

tasks:
  - [ ] Vite + TypeScript 설정
  - [ ] TailwindCSS 설정
  - [ ] 디렉토리 구조

output:
  - "frontend/"
```

### P4-UI-002: 공통 컴포넌트
```yaml
priority: P0
estimate: 16h
depends_on: [P4-UI-001]

tasks:
  - [ ] Button, Input, Card, Modal
  - [ ] Table, Form 컴포넌트
  - [ ] Loading, Alert 컴포넌트

output:
  - "frontend/src/components/ui/"
```

### P4-UI-003: 인증 페이지
```yaml
priority: P0
estimate: 8h

tasks:
  - [ ] 로그인/회원가입 페이지
  - [ ] Auth Store
  - [ ] Protected Route

output:
  - "frontend/src/pages/auth/"
```

### P4-UI-004: 대시보드
```yaml
priority: P0
estimate: 16h

tasks:
  - [ ] 레이아웃, 네비게이션
  - [ ] 통계 카드
  - [ ] 최근 분석 목록

output:
  - "frontend/src/pages/Dashboard.tsx"
```

### P4-UI-005: 증상 입력 폼
```yaml
priority: P0
estimate: 16h

tasks:
  - [ ] 증상 입력 폼
  - [ ] 자동완성
  - [ ] 기저질환 선택
  - [ ] 유효성 검사

output:
  - "frontend/src/pages/analysis/SymptomInput.tsx"
```

### P4-UI-006: 이미지 업로드
```yaml
priority: P0
estimate: 8h

tasks:
  - [ ] 드래그 앤 드롭
  - [ ] 미리보기
  - [ ] 업로드 진행률

output:
  - "frontend/src/pages/analysis/ImageUpload.tsx"
```

### P4-UI-007: 분석 결과 화면
```yaml
priority: P0
estimate: 24h

tasks:
  - [ ] 결과 요약 카드
  - [ ] 질병 예측 목록
  - [ ] 확률 차트
  - [ ] 위험도 게이지
  - [ ] 히트맵 오버레이
  - [ ] 문헌 검색 결과

output:
  - "frontend/src/pages/analysis/AnalysisResult.tsx"
```

### P4-INT-001: Frontend-Backend 통합
```yaml
priority: P0
estimate: 16h

tasks:
  - [ ] API 클라이언트
  - [ ] 에러 핸들링
  - [ ] 통합 테스트

output:
  - "frontend/src/services/api/"
```

---

## Phase 5: Deployment (Week 19-24)

### P5-OPS-001: Kubernetes 클러스터
```yaml
priority: P0
estimate: 24h

tasks:
  - [ ] 클러스터 프로비저닝
  - [ ] Deployment 매니페스트
  - [ ] Service, Ingress
  - [ ] HPA 설정

output:
  - "k8s/"
```

### P5-OPS-002: 모니터링 구축
```yaml
priority: P0
estimate: 16h

tasks:
  - [ ] Prometheus 배포
  - [ ] Grafana 대시보드
  - [ ] 알림 규칙

output:
  - "k8s/monitoring/"
```

### P5-OPS-003: 로깅 시스템
```yaml
priority: P1
estimate: 16h

tasks:
  - [ ] 로그 수집기
  - [ ] 로그 대시보드

output:
  - "k8s/logging/"
```

### P5-OPS-004: 보안 감사
```yaml
priority: P0
estimate: 16h

tasks:
  - [ ] 취약점 스캔
  - [ ] 침투 테스트
  - [ ] 취약점 수정

output:
  - "docs/security-audit-report.pdf"
```

---

## 태스크 요약

| Phase | 태스크 수 | 예상 시간 |
|-------|----------|-----------|
| P1: Foundation | 5 | 36h |
| P2: ML Models | 6 | 176h |
| P3: RAG System | 6 | 96h |
| P4: Integration | 9 | 112h |
| P5: Deployment | 4 | 72h |
| **Total** | **30** | **492h** |

---

**END OF TASKS**
