# SPECIFICATION.md
# MediAssist AI - 기술 명세서

---

## 1. 시스템 아키텍처

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MediAssist AI Architecture                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            CLIENT LAYER                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │   │
│  │  │  Web App    │  │  Mobile     │  │  EMR Plugin │                      │   │
│  │  │  (React)    │  │  (Phase 2)  │  │  (Phase 2)  │                      │   │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘                      │   │
│  └─────────┼────────────────────────────────────────────────────────────────┘   │
│            │ HTTPS                                                              │
│            ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           API GATEWAY                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │   │
│  │  │  • Rate Limiting (100 req/min per user)                           │  │   │
│  │  │  • Authentication (JWT validation)                                │  │   │
│  │  │  • Request Routing                                                │  │   │
│  │  │  • SSL Termination                                                │  │   │
│  │  └───────────────────────────────────────────────────────────────────┘  │   │
│  └─────────┬────────────────────────────────────────────────────────────────┘   │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         APPLICATION LAYER                                │   │
│  │                                                                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │   │
│  │  │    Auth     │ │   Patient   │ │  Analysis   │ │   Report    │       │   │
│  │  │   Service   │ │   Service   │ │   Service   │ │   Service   │       │   │
│  │  │  Port:8001  │ │  Port:8002  │ │  Port:8003  │ │  Port:8004  │       │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │   │
│  │         │               │               │               │              │   │
│  └─────────┼───────────────┼───────────────┼───────────────┼──────────────┘   │
│            │               │               │               │                    │
│            ▼               ▼               ▼               ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            ML/AI LAYER                                   │   │
│  │                                                                          │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │   │
│  │  │  Symptom ML   │  │   Image ML    │  │    Risk ML    │               │   │
│  │  │  (XGBoost +   │  │  (DenseNet +  │  │   (Neural     │               │   │
│  │  │   BioBERT)    │  │   Grad-CAM)   │  │    Network)   │               │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘               │   │
│  │                              │                                          │   │
│  │                              ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │                        RAG SYSTEM                                  │ │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │ │   │
│  │  │  │  Embedding  │ │  Vector DB  │ │  Reranker   │ │    LLM      │  │ │   │
│  │  │  │  (BioBERT)  │ │  (Chroma)   │ │(CrossEncoder)│ │  (GPT-4)   │  │ │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            DATA LAYER                                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │   │
│  │  │ PostgreSQL  │ │    Redis    │ │   ChromaDB  │ │    MinIO    │       │   │
│  │  │  (Primary)  │ │   (Cache)   │ │  (Vectors)  │ │  (Objects)  │       │   │
│  │  │  Port:5432  │ │  Port:6379  │ │  Port:8005  │ │  Port:9000  │       │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 서비스 정의

| 서비스 | 역할 | 기술 스택 | Port |
|--------|------|-----------|------|
| **API Gateway** | 라우팅, 인증, Rate Limit | Kong / Nginx | 8000 |
| **Auth Service** | 인증, 인가, 사용자 관리 | FastAPI | 8001 |
| **Patient Service** | 환자 정보 CRUD | FastAPI | 8002 |
| **Analysis Service** | ML 분석 오케스트레이션 | FastAPI + Celery | 8003 |
| **Report Service** | 리포트 생성, 관리 | FastAPI | 8004 |
| **ML Workers** | ML 모델 추론 | Celery Workers | - |

---

## 2. 기술 스택

### 2.1 Backend

```yaml
framework:
  name: "FastAPI"
  version: "0.109.0"
  features:
    - "Async/Await support"
    - "Automatic OpenAPI docs"
    - "Pydantic validation"
    - "Dependency injection"

language:
  name: "Python"
  version: "3.11+"
  
orm:
  name: "SQLAlchemy"
  version: "2.0+"
  mode: "Async"
  
task_queue:
  name: "Celery"
  version: "5.3+"
  broker: "Redis"
  
cache:
  name: "Redis"
  version: "7.x"
  use_cases:
    - "Session storage"
    - "API response cache"
    - "Rate limiting"
    - "Celery broker/backend"
```

### 2.2 Frontend

```yaml
framework:
  name: "React"
  version: "18.2+"
  
language:
  name: "TypeScript"
  version: "5.x"
  
styling:
  name: "TailwindCSS"
  version: "3.4+"
  
state_management:
  name: "Zustand"
  version: "4.x"
  
data_fetching:
  name: "TanStack Query"
  version: "5.x"
  
form:
  name: "React Hook Form + Zod"
  
charts:
  name: "Chart.js + react-chartjs-2"
```

### 2.3 ML/AI

```yaml
deep_learning:
  framework: "PyTorch"
  version: "2.1+"
  
classical_ml:
  library: "scikit-learn"
  version: "1.4+"
  boosting: "XGBoost"
  
nlp:
  library: "transformers"
  version: "4.36+"
  models:
    - "dmis-lab/biobert-v1.1"
    - "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
rag:
  framework: "LangChain"
  version: "0.1+"
  vector_db: "ChromaDB"
  llm: "OpenAI GPT-4"
  
image:
  library: "torchvision"
  models:
    - "DenseNet121"
    - "ResNet50"
```

### 2.4 Database

```yaml
primary:
  name: "PostgreSQL"
  version: "15+"
  features:
    - "JSONB for flexible data"
    - "Full-text search"
    - "Partitioning"
  
vector:
  name: "ChromaDB"
  version: "0.4+"
  embedding_dim: 768
  index: "HNSW"
  
cache:
  name: "Redis"
  version: "7.x"
  
object_storage:
  name: "MinIO"
  compatibility: "S3"
```

### 2.5 Infrastructure

```yaml
containerization:
  runtime: "Docker"
  orchestration: "Kubernetes"
  
ci_cd:
  platform: "GitHub Actions"
  stages:
    - "lint"
    - "test"
    - "build"
    - "deploy"
  
monitoring:
  metrics: "Prometheus"
  visualization: "Grafana"
  logging: "ELK Stack"
  tracing: "Jaeger"
  
cloud:
  primary: "AWS / GCP / Azure"
  services:
    - "Managed Kubernetes"
    - "Managed PostgreSQL"
    - "Object Storage"
    - "CDN"
```

---

## 3. 데이터 모델

### 3.1 ERD

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Entity Relationship Diagram                         │
│                                                                                 │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐     │
│  │      Users      │        │    Patients     │        │    Analyses     │     │
│  ├─────────────────┤        ├─────────────────┤        ├─────────────────┤     │
│  │ PK: id (UUID)   │───┐    │ PK: id (UUID)   │───┐    │ PK: id (UUID)   │     │
│  │ email           │   │    │ anonymous_id    │   │    │ FK: patient_id  │◀────┤
│  │ password_hash   │   │    │ FK: created_by  │◀──┘    │ FK: created_by  │◀────┘
│  │ name            │   │    │ age             │   ┌───▶│ analysis_type   │     │
│  │ role            │   │    │ gender          │   │    │ status          │     │
│  │ department      │   │    │ medical_history │   │    │ input_data      │     │
│  │ is_active       │   │    │ created_at      │   │    │ created_at      │     │
│  │ created_at      │   │    └─────────────────┘   │    │ completed_at    │     │
│  └─────────────────┘   │                          │    └────────┬────────┘     │
│                        │                          │             │              │
│                        │                          │             │              │
│  ┌─────────────────┐   │    ┌─────────────────┐   │    ┌────────┴────────┐     │
│  │    Symptoms     │   │    │  MedicalImages  │   │    │   MLResults     │     │
│  ├─────────────────┤   │    ├─────────────────┤   │    ├─────────────────┤     │
│  │ PK: id (UUID)   │   │    │ PK: id (UUID)   │   │    │ PK: id (UUID)   │     │
│  │ FK: analysis_id │◀──┼────│ FK: analysis_id │◀──┼────│ FK: analysis_id │     │
│  │ symptom_code    │   │    │ image_type      │   │    │ model_name      │     │
│  │ symptom_name    │   │    │ file_path       │   │    │ predictions     │     │
│  │ severity        │   │    │ file_size       │   │    │ confidence      │     │
│  │ duration_days   │   │    │ analysis_result │   │    │ processing_time │     │
│  │ description     │   │    │ uploaded_at     │   │    │ created_at      │     │
│  └─────────────────┘   │    └─────────────────┘   │    └─────────────────┘     │
│                        │                          │                            │
│  ┌─────────────────┐   │    ┌─────────────────┐   │    ┌─────────────────┐     │
│  │   RAGResults    │   │    │     Reports     │   │    │   AuditLogs     │     │
│  ├─────────────────┤   │    ├─────────────────┤   │    ├─────────────────┤     │
│  │ PK: id (UUID)   │   │    │ PK: id (UUID)   │   │    │ PK: id (UUID)   │     │
│  │ FK: analysis_id │◀──┴────│ FK: analysis_id │◀──┴────│ FK: user_id     │     │
│  │ query           │        │ content         │        │ action          │     │
│  │ retrieved_docs  │        │ format          │        │ resource        │     │
│  │ generated_text  │        │ file_path       │        │ ip_address      │     │
│  │ sources         │        │ created_at      │        │ user_agent      │     │
│  │ relevance_score │        └─────────────────┘        │ created_at      │     │
│  └─────────────────┘                                   └─────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 테이블 스키마

#### Users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'doctor', 'nurse', 'researcher')),
    department VARCHAR(100),
    license_number VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_department ON users(department);
```

#### Analyses

```sql
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE SET NULL,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    analysis_type VARCHAR(50) NOT NULL CHECK (
        analysis_type IN ('symptom', 'image', 'combined', 'rag_only')
    ),
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')
    ),
    input_data JSONB NOT NULL,
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_completion CHECK (
        (status = 'completed' AND completed_at IS NOT NULL) OR
        (status != 'completed')
    )
);

CREATE INDEX idx_analyses_patient ON analyses(patient_id);
CREATE INDEX idx_analyses_created_by ON analyses(created_by);
CREATE INDEX idx_analyses_status ON analyses(status);
CREATE INDEX idx_analyses_type ON analyses(analysis_type);
CREATE INDEX idx_analyses_created_at ON analyses(created_at DESC);
```

#### MLResults

```sql
CREATE TABLE ml_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    predictions JSONB NOT NULL,
    confidence_scores JSONB,
    feature_importance JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- predictions JSONB 예시:
-- {
--   "top_predictions": [
--     {"disease": "급성 상기도 감염", "icd_code": "J06.9", "probability": 0.85},
--     {"disease": "인플루엔자", "icd_code": "J11", "probability": 0.12}
--   ],
--   "risk_level": "low",
--   "risk_score": 25.5
-- }

CREATE INDEX idx_ml_results_analysis ON ml_results(analysis_id);
CREATE INDEX idx_ml_results_model ON ml_results(model_name);
```

---

## 4. API 명세

### 4.1 API 개요

```yaml
base_url: "https://api.mediassist.ai/v1"
authentication: "Bearer JWT"
content_type: "application/json"
rate_limit: "100 requests/minute per user"
```

### 4.2 Authentication API

#### POST /auth/login

```yaml
description: "사용자 로그인"
request:
  body:
    type: object
    required: [email, password]
    properties:
      email:
        type: string
        format: email
        example: "doctor@hospital.com"
      password:
        type: string
        minLength: 8
        example: "SecurePass123!"
        
response:
  200:
    body:
      access_token: "eyJhbGciOiJIUzI1NiIs..."
      refresh_token: "eyJhbGciOiJIUzI1NiIs..."
      token_type: "bearer"
      expires_in: 3600
      user:
        id: "uuid"
        email: "doctor@hospital.com"
        name: "김의사"
        role: "doctor"
  401:
    body:
      error: "invalid_credentials"
      message: "이메일 또는 비밀번호가 올바르지 않습니다."
```

#### POST /auth/refresh

```yaml
description: "토큰 갱신"
request:
  body:
    refresh_token: "eyJhbGciOiJIUzI1NiIs..."
response:
  200:
    body:
      access_token: "eyJhbGciOiJIUzI1NiIs..."
      expires_in: 3600
```

### 4.3 Analysis API

#### POST /analysis/symptoms

```yaml
description: "증상 분석 요청"
authentication: required
request:
  body:
    type: object
    required: [symptoms]
    properties:
      patient_id:
        type: string
        format: uuid
        description: "환자 ID (선택)"
      symptoms:
        type: array
        minItems: 1
        items:
          type: object
          required: [name]
          properties:
            name:
              type: string
              example: "두통"
            severity:
              type: integer
              minimum: 1
              maximum: 10
              example: 7
            duration_days:
              type: integer
              minimum: 0
              example: 3
            description:
              type: string
              example: "전두부 박동성 두통"
      patient_info:
        type: object
        properties:
          age:
            type: integer
            example: 45
          gender:
            type: string
            enum: [male, female, other]
          medical_history:
            type: array
            items:
              type: string
            example: ["고혈압", "당뇨"]
      options:
        type: object
        properties:
          include_rag:
            type: boolean
            default: true
          include_risk_assessment:
            type: boolean
            default: true
            
response:
  202:
    description: "분석 요청 수락"
    body:
      analysis_id: "123e4567-e89b-12d3-a456-426614174000"
      status: "processing"
      estimated_time_seconds: 30
      
  400:
    body:
      error: "validation_error"
      details:
        - field: "symptoms"
          message: "최소 1개 이상의 증상이 필요합니다."
```

#### GET /analysis/{analysis_id}

```yaml
description: "분석 결과 조회"
authentication: required
parameters:
  - name: analysis_id
    in: path
    required: true
    type: string
    format: uuid
    
response:
  200:
    body:
      id: "123e4567-e89b-12d3-a456-426614174000"
      status: "completed"
      created_at: "2025-01-11T10:30:00Z"
      completed_at: "2025-01-11T10:30:25Z"
      processing_time_ms: 25000
      
      ml_results:
        symptom_analysis:
          predictions:
            - disease: "급성 상기도 감염"
              icd_code: "J06.9"
              probability: 0.72
              confidence: "high"
            - disease: "인플루엔자"
              icd_code: "J11"
              probability: 0.15
              confidence: "medium"
          risk_assessment:
            score: 35.5
            level: "low"
            factors:
              - "발열 지속"
              - "기침"
              
      rag_results:
        query: "급성 상기도 감염 진단 및 치료"
        summary: "급성 상기도 감염은 바이러스에 의한..."
        recommendations:
          - "대증 치료 권장"
          - "충분한 휴식과 수분 섭취"
          - "고열 지속 시 해열제 투여"
        sources:
          - title: "2024 호흡기 감염 치료 가이드라인"
            authors: "대한감염학회"
            year: 2024
            relevance: 0.92
            url: "https://..."
            
      disclaimer: "이 분석 결과는 참고용이며, 최종 진단은 의사가 결정해야 합니다."
```

#### POST /analysis/image

```yaml
description: "의료 이미지 분석 요청"
authentication: required
request:
  content_type: "multipart/form-data"
  body:
    patient_id:
      type: string
      format: uuid
    image:
      type: file
      format: binary
      accept: ["image/png", "image/jpeg", "application/dicom"]
      maxSize: "50MB"
    image_type:
      type: string
      enum: [chest_xray, ct_head, ct_chest, ct_abdomen]
      required: true
    notes:
      type: string
      maxLength: 1000
      
response:
  202:
    body:
      analysis_id: "uuid"
      status: "processing"
      estimated_time_seconds: 60
```

### 4.4 Reports API

#### POST /reports/generate

```yaml
description: "진단 보조 리포트 생성"
authentication: required
request:
  body:
    analysis_id:
      type: string
      format: uuid
      required: true
    format:
      type: string
      enum: [pdf, html, json]
      default: pdf
    template:
      type: string
      enum: [standard, detailed, summary]
      default: standard
    include_sections:
      type: array
      items:
        type: string
        enum: [symptoms, predictions, risk, literature, recommendations]
      default: ["symptoms", "predictions", "risk", "literature", "recommendations"]
      
response:
  200:
    body:
      report_id: "uuid"
      download_url: "https://storage.mediassist.ai/reports/xxx.pdf"
      expires_at: "2025-01-12T10:30:00Z"
      file_size_bytes: 125000
```

---

## 5. ML 모델 명세

### 5.1 증상 분류 모델

```yaml
model_name: "SymptomClassifier"
version: "1.0.0"

architecture:
  type: "Ensemble"
  components:
    - name: "BioBERT NER"
      purpose: "증상 엔티티 추출"
      model: "dmis-lab/biobert-v1.1"
      input: "자연어 증상 텍스트"
      output: "추출된 증상 엔티티 목록"
      
    - name: "XGBoost Classifier"
      purpose: "질병 분류"
      features:
        - "BioBERT 임베딩 (768 dim)"
        - "환자 메타데이터 (나이, 성별)"
        - "증상 수치 특징 (심각도, 기간)"
      output: "질병별 확률 분포"
      
    - name: "Random Forest Classifier"
      purpose: "앙상블 멤버"
      output: "질병별 확률 분포"

ensemble:
  method: "Soft Voting"
  weights: [0.6, 0.4]  # XGBoost, RF

training:
  dataset: "MIMIC-III + Custom Korean Medical Data"
  train_size: 50000
  val_size: 10000
  test_size: 10000
  
performance:
  accuracy: ">= 85%"
  f1_macro: ">= 0.80"
  inference_time: "< 100ms"

output_schema:
  predictions:
    type: array
    items:
      disease: string
      icd_code: string
      probability: float
      confidence: "high | medium | low"
```

### 5.2 이미지 분석 모델

```yaml
model_name: "ChestXrayAnalyzer"
version: "1.0.0"

architecture:
  backbone: "DenseNet121"
  pretrained: "CheXpert"
  heads:
    - name: "MultiLabel Classification"
      classes: ["정상", "폐렴", "결핵", "폐암 의심", "심비대", "기흉", "폐부종"]
      activation: "Sigmoid"
  visualization: "Grad-CAM"

preprocessing:
  input_size: [224, 224]
  normalization: "ImageNet mean/std"
  augmentation:
    - "Random Horizontal Flip"
    - "Random Rotation (±15°)"
    - "Color Jitter"

training:
  dataset: "CheXpert + NIH ChestX-ray14"
  train_size: 200000
  val_size: 20000
  optimizer: "AdamW"
  learning_rate: 1e-4
  epochs: 50
  
performance:
  auc_roc: ">= 0.90"
  sensitivity: ">= 0.85"
  specificity: ">= 0.80"
  inference_time: "< 500ms"

output_schema:
  findings:
    type: array
    items:
      condition: string
      probability: float
      confidence: string
      location: string (optional)
  heatmap_url: string
  image_quality:
    resolution: [int, int]
    brightness: float
    contrast: float
    is_acceptable: boolean
```

### 5.3 RAG 시스템

```yaml
system_name: "MedicalRAG"
version: "1.0.0"

embedding:
  model: "dmis-lab/biobert-v1.1"
  dimension: 768
  
vector_database:
  name: "ChromaDB"
  index_type: "HNSW"
  distance_metric: "cosine"
  
retrieval:
  method: "Hybrid"
  components:
    - name: "Dense Retrieval"
      top_k: 20
    - name: "BM25 (Sparse)"
      top_k: 20
  fusion: "Reciprocal Rank Fusion"
  
reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5

generation:
  llm: "GPT-4"
  temperature: 0.3
  max_tokens: 1000
  prompt_template: "Medical QA with Citations"

document_sources:
  - name: "PubMed"
    count: "100,000+"
    update_frequency: "weekly"
  - name: "Clinical Guidelines"
    count: "500+"
    update_frequency: "monthly"
  - name: "Medical Textbooks"
    count: "50+"
    update_frequency: "yearly"

performance:
  relevance_score: ">= 0.70"
  response_time: "< 5 seconds"
```

---

## 6. 보안 명세

### 6.1 인증/인가

```yaml
authentication:
  method: "JWT"
  algorithm: "RS256"
  access_token_expiry: "1 hour"
  refresh_token_expiry: "7 days"
  
authorization:
  method: "RBAC"
  roles:
    admin:
      permissions: ["*"]
    doctor:
      permissions:
        - "analysis:create"
        - "analysis:read"
        - "patient:read"
        - "patient:create"
        - "report:create"
        - "report:read"
    nurse:
      permissions:
        - "analysis:read"
        - "patient:read"
        - "report:read"
    researcher:
      permissions:
        - "analysis:read:anonymized"
        - "report:read:anonymized"
```

### 6.2 암호화

```yaml
encryption:
  at_rest:
    algorithm: "AES-256-GCM"
    key_management: "AWS KMS / HashiCorp Vault"
    
  in_transit:
    protocol: "TLS 1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      
  password_hashing:
    algorithm: "bcrypt"
    cost_factor: 12
    
  pii_encryption:
    fields: ["patient_name", "patient_id", "contact_info"]
    algorithm: "AES-256-GCM"
```

### 6.3 감사 로그

```yaml
audit_logging:
  events:
    - "user.login"
    - "user.logout"
    - "analysis.create"
    - "analysis.read"
    - "patient.create"
    - "patient.read"
    - "report.generate"
    - "report.download"
    
  fields:
    - "timestamp"
    - "user_id"
    - "action"
    - "resource"
    - "resource_id"
    - "ip_address"
    - "user_agent"
    - "result"
    
  retention: "7 years"
  storage: "Encrypted S3 + Elasticsearch"
```

---

## 7. 비기능 요구사항

### 7.1 성능

| 항목 | 요구사항 | 측정 방법 |
|------|----------|-----------|
| 증상 분석 응답 | < 3초 | p99 latency |
| 이미지 분석 응답 | < 10초 | p99 latency |
| RAG 검색 응답 | < 5초 | p99 latency |
| 리포트 생성 | < 15초 | p99 latency |
| 동시 사용자 | 100+ | Load test |
| API 처리량 | 1000 req/min | Prometheus |

### 7.2 가용성

| 항목 | 요구사항 |
|------|----------|
| 서비스 가용률 | 99.9% |
| RTO (복구 시간) | < 1시간 |
| RPO (복구 시점) | < 15분 |
| 계획 유지보수 | 월 1회, 04:00-06:00 |

### 7.3 확장성

```yaml
scaling:
  horizontal:
    api_services: "Auto-scaling (2-10 pods)"
    ml_workers: "Auto-scaling (2-5 pods)"
    
  vertical:
    database: "Read replicas"
    vector_db: "Sharding"
    
  data:
    documents: "1M+ supported"
    analyses: "10M+ per year"
```

---

**END OF SPECIFICATION**
