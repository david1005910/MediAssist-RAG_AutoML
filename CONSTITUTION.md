# CONSTITUTION.md
# MediAssist AI - 의료 진단 보조 시스템

## 프로젝트 헌법 (Project Constitution)

---

## 1. 프로젝트 정체성

### 1.1 프로젝트명
**MediAssist AI** - Medical RAG + ML 진단 보조 시스템

### 1.2 미션
의료진이 환자 증상을 분석하고, 관련 의학 문헌을 검색하여, 
근거 기반 진단 의사결정을 지원하는 AI 시스템을 구축한다.

### 1.3 비전
- 진단 정확도 향상
- 희귀 질환 조기 발견
- 의료진 업무 효율 개선
- 근거 기반 의료 문화 확산

---

## 2. 핵심 원칙 (Core Principles)

### 2.1 의료 안전 우선 (Medical Safety First)
```
✅ 모든 출력에 "이 정보는 참고용이며 최종 진단은 의사가 결정합니다" 명시
✅ 불확실한 경우 명시적으로 표현
✅ Red flag 증상은 즉시 경고
✅ 모든 제안에 근거 문헌 인용
❌ 절대 최종 진단 또는 처방 결정 금지
```

### 2.2 개인정보 보호 (Privacy Protection)
```
✅ 환자 데이터 익명화 필수
✅ AES-256 암호화 저장
✅ TLS 1.3 전송 암호화
✅ 모든 접근 감사 로그 기록
✅ 최소 권한 원칙 적용
❌ 환자 식별 정보 로깅 금지
```

### 2.3 투명성 (Transparency)
```
✅ AI 결정에 대한 설명 제공 (Explainable AI)
✅ 신뢰도 점수 명시
✅ 참고 문헌 출처 표시
✅ 모델 한계점 고지
```

### 2.4 정확성 (Accuracy)
```
✅ 증상 분류 정확도 ≥ 85%
✅ 문헌 검색 적합도 ≥ 90%
✅ 지속적인 모델 평가 및 개선
✅ 의료진 피드백 반영
```

---

## 3. 기술 원칙 (Technical Principles)

### 3.1 아키텍처 원칙
```yaml
architecture:
  style: "Microservices"
  communication: "REST API + Message Queue"
  database: "Database per Service"
  deployment: "Container-based (Docker/K8s)"
```

### 3.2 코드 원칙
```yaml
code_standards:
  language: "Python 3.11+"
  style_guide: "PEP 8 + Black formatter"
  type_hints: "Required (mypy strict)"
  documentation: "Docstring required for all public APIs"
  testing: "Unit test coverage ≥ 80%"
```

### 3.3 보안 원칙
```yaml
security:
  authentication: "JWT + OAuth 2.0"
  authorization: "RBAC (Role-Based Access Control)"
  encryption:
    at_rest: "AES-256"
    in_transit: "TLS 1.3"
  secrets: "Environment variables / Vault"
  audit: "All access logged with timestamp"
```

### 3.4 ML 원칙
```yaml
ml_standards:
  reproducibility: "Random seed fixed, version controlled"
  validation: "Cross-validation required"
  explainability: "SHAP/Grad-CAM for interpretability"
  monitoring: "Accuracy drift detection"
  bias_check: "Fairness metrics evaluated"
```

---

## 4. 규정 준수 (Compliance)

### 4.1 법적 규정
| 규정 | 적용 | 요구사항 |
|------|------|----------|
| 의료기기법 | 필수 | 의료기기 소프트웨어 허가 |
| 개인정보보호법 | 필수 | 민감정보(건강정보) 처리 기준 |
| 의료법 | 필수 | 진단 보조 도구 책임 한계 |
| HIPAA (해외) | 선택 | 미국 수출 시 준수 |

### 4.2 윤리 규정
```
✅ AI 윤리 가이드라인 준수
✅ 편향(Bias) 최소화 노력
✅ 인간 의사의 최종 판단권 보장
✅ 환자 동의 기반 데이터 활용
```

---

## 5. 팀 구조 및 역할

### 5.1 역할 정의
```yaml
roles:
  product_owner:
    responsibility: "요구사항 정의, 우선순위 결정"
    
  tech_lead:
    responsibility: "기술 의사결정, 아키텍처 설계"
    
  ml_engineer:
    responsibility: "ML 모델 개발, 학습, 평가"
    
  backend_developer:
    responsibility: "API 개발, DB 설계, 서비스 구현"
    
  frontend_developer:
    responsibility: "UI/UX 구현, 사용자 경험 최적화"
    
  devops_engineer:
    responsibility: "CI/CD, 인프라, 모니터링"
    
  medical_advisor:
    responsibility: "의료 자문, 요구사항 검증"
    
  qa_engineer:
    responsibility: "테스트 전략, 품질 보증"
```

### 5.2 의사결정 매트릭스
```
┌────────────────┬──────────────┬──────────────┬──────────────┐
│ 결정 영역       │ 결정자       │ 자문         │ 통보         │
├────────────────┼──────────────┼──────────────┼──────────────┤
│ 제품 기능       │ PO          │ Medical Adv. │ Tech Lead    │
│ 기술 아키텍처   │ Tech Lead   │ DevOps       │ PO           │
│ ML 모델 선택   │ ML Engineer │ Tech Lead    │ PO           │
│ 보안 정책      │ Tech Lead   │ DevOps       │ All          │
│ 릴리스         │ PO          │ QA           │ All          │
└────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6. 품질 기준 (Quality Standards)

### 6.1 코드 품질
```yaml
quality_gates:
  - name: "Lint Check"
    tool: "ruff + black"
    threshold: "0 errors"
    
  - name: "Type Check"
    tool: "mypy --strict"
    threshold: "0 errors"
    
  - name: "Unit Test"
    tool: "pytest"
    threshold: "coverage ≥ 80%"
    
  - name: "Security Scan"
    tool: "bandit + safety"
    threshold: "0 high/critical"
    
  - name: "Dependency Check"
    tool: "pip-audit"
    threshold: "0 vulnerabilities"
```

### 6.2 ML 모델 품질
```yaml
ml_quality_gates:
  symptom_classifier:
    accuracy: "≥ 85%"
    f1_score: "≥ 0.80"
    inference_time: "< 100ms"
    
  image_analyzer:
    auc_roc: "≥ 0.90"
    sensitivity: "≥ 0.85"
    specificity: "≥ 0.80"
    inference_time: "< 500ms"
    
  rag_system:
    relevance_score: "≥ 0.70"
    response_time: "< 3s"
```

### 6.3 서비스 품질
```yaml
sla:
  availability: "99.9%"
  response_time_p99: "< 3s"
  error_rate: "< 1%"
  rto: "< 1 hour"
  rpo: "< 15 minutes"
```

---

## 7. 커뮤니케이션 규칙

### 7.1 코드 리뷰
```
✅ 모든 PR은 최소 1명 이상의 리뷰 필수
✅ ML 코드는 ML Engineer 리뷰 필수
✅ 보안 관련 코드는 Tech Lead 리뷰 필수
✅ 리뷰 요청 후 24시간 내 응답
```

### 7.2 문서화
```
✅ 모든 API는 OpenAPI Spec 문서화
✅ ML 모델은 Model Card 작성
✅ 아키텍처 변경은 ADR 작성
✅ README 최신 상태 유지
```

### 7.3 버전 관리
```yaml
versioning:
  format: "Semantic Versioning (MAJOR.MINOR.PATCH)"
  branching: "Git Flow"
  commit_message: "Conventional Commits"
```

---

## 8. 면책 조항 (Disclaimer)

### 8.1 시스템 면책
```
본 시스템은 의료 진단 보조 도구로서, 다음 사항을 명시합니다:

1. 본 시스템의 모든 분석 결과는 참고 정보이며, 
   최종 진단 및 치료 결정은 반드시 자격을 갖춘 의료 전문가가 수행해야 합니다.

2. 본 시스템은 의료 행위를 대체하지 않으며, 
   응급 상황에서는 즉시 의료 기관을 방문해야 합니다.

3. AI 모델의 예측은 통계적 확률에 기반하며, 
   100% 정확성을 보장하지 않습니다.

4. 사용자는 본 시스템 사용 전 이용약관 및 
   개인정보처리방침에 동의해야 합니다.
```

---

## 9. 문서 버전

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|----------|
| 1.0 | 2025-01-11 | AI Team | 초기 작성 |

---

**END OF CONSTITUTION**
