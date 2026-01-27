# MediAssist AI

[![Build](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/build.yml/badge.svg)](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/build.yml)
[![Test](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/test.yml/badge.svg)](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/test.yml)
[![Lint](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/lint.yml/badge.svg)](https://github.com/david1005910/MediAssist-RAG_AutoML/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript)](https://www.typescriptlang.org/)

Medical RAG + ML 진단 보조 시스템

## Overview

MediAssist AI는 의료진이 환자 증상을 분석하고, 관련 의학 문헌을 검색하여 근거 기반 진단 의사결정을 지원하는 AI 시스템입니다.

## Screenshots

### Dashboard
메인 대시보드에서 모든 진단 보조 도구에 접근할 수 있습니다.

![Dashboard](docs/screenshots/dashboard-screenshot.png)

### Symptom Analysis (증상 분석)
자연어로 증상을 입력하면 AI가 자동으로 증상을 추출하고 가능한 질환을 예측합니다.

![Symptom Analysis](docs/screenshots/symptom-analysis-screenshot.png)

### Image Analysis (의료 이미지 분석)
흉부 X-ray를 업로드하면 DenseNet121 모델이 분석하고 Grad-CAM으로 관심 영역을 시각화합니다.

![Image Analysis](docs/screenshots/image-analysis-screenshot.png)

### Literature Search (문헌 검색)
Qdrant 하이브리드 검색(Sparse 30% + Dense 70%)으로 의학 문헌을 검색하고 RAG 기반 질의응답을 수행합니다.

![Literature Search](docs/screenshots/literature-search-screenshot.png)

### RNA Analysis (RNA 서열 분석)
RNA 서열을 입력하면 BERT-style Transformer 모델이 RNA 유형을 분류하고 관련 질병을 예측합니다.

![RNA Analysis](docs/screenshots/rna-analysis-screenshot.png)

### AutoML Dashboard (AutoML 대시보드)
Optuna 기반 자동 하이퍼파라미터 최적화를 관리하고 실험 결과를 모니터링합니다.

![AutoML Dashboard](docs/screenshots/automl-dashboard-screenshot.png)

### Knowledge Graph (지식 그래프)
Neo4j 기반 의료 지식 그래프를 3D로 탐색하며 질환-증상-치료법 관계를 시각화합니다.

![Knowledge Graph](docs/screenshots/knowledge-graph-screenshot.png)

### Login
Frosted Metal Aesthetic 디자인이 적용된 로그인 화면입니다.

![Login](docs/screenshots/login-screenshot.png)

## Features

- **증상 분석**: BioBERT NER + RandomForest 모델을 사용한 증상 기반 질병 분류 (87.5% accuracy)
- **의료 이미지 분석**: DenseNet121 기반 흉부 X-ray 분석 + Grad-CAM 시각화
- **RNA 서열 분석**: BERT-style Transformer 기반 RNA 서열 질병 예측 (mRNA, siRNA, circRNA, lncRNA 지원)
- **AutoML 시스템**: Optuna 기반 하이퍼파라미터 최적화 및 앙상블 모델 자동 생성
- **RAG 문헌 검색**: ChromaDB 하이브리드 검색 (Sparse BM25 + Dense BioBERT)
- **지식 그래프**: Neo4j 기반 질환-증상-치료법 관계 시각화
- **위험도 평가**: 환자 상태에 따른 위험도 점수 산출

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), Celery + Redis
- **Frontend**: React 18, TypeScript, TailwindCSS, Zustand, TanStack Query, Three.js
- **ML/AI**: PyTorch, scikit-learn, BioBERT, DenseNet121, Optuna (AutoML)
- **Vector DB**: ChromaDB (Hybrid Search)
- **Graph DB**: Neo4j
- **Database**: PostgreSQL, MongoDB, Redis
- **Infrastructure**: Docker, Kubernetes, GitHub Actions

## Quick Start

### Option 1: Docker Compose (권장)

```bash
# Clone the repository
git clone https://github.com/david1005910/MediAssist-RAG_AutoML.git
cd MediAssist-RAG_AutoML

# Copy environment file
cp .env.example .env

# Start infrastructure services
docker-compose up -d postgres mongodb redis chromadb minio

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Run backend (port 8003)
cd services/analysis && MONGODB_URI=mongodb://localhost:27017 MONGODB_DB=mediassist uvicorn app.main:app --reload --port 8003

# Run frontend (port 3003) in another terminal
cd frontend && npm run dev
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/david1005910/MediAssist-RAG_AutoML.git
cd MediAssist-RAG_AutoML

# Copy environment file
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Run the analysis service (port 8003)
cd services/analysis && uvicorn app.main:app --reload --port 8003

# Run the frontend (port 3003) in another terminal
cd frontend && npm run dev
```

### Demo Account

- **Email**: `demo@mediassist.ai`
- **Password**: `demo1234`

## Test Results

### Unit Tests (74 tests)

```
======================== 74 passed in 10.21s ========================
```

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_automl.py` | 32 | ✅ Pass |
| `test_rna_predictor.py` | 37 | ✅ Pass |
| `test_symptom_classifier.py` | 5 | ✅ Pass |
| **Total** | **74** | **100% Pass** |

### API Integration Tests

| Feature | Endpoint | Status |
|---------|----------|--------|
| Authentication (MongoDB) | `/api/v1/auth/login` | ✅ Pass |
| User Registration | `/api/v1/auth/register` | ✅ Pass |
| Symptom Analysis | `/api/v1/symptom/analyze` | ✅ Pass |
| NER Extraction | `/api/v1/symptom/extract` | ✅ Pass |
| Image Analysis | `/api/v1/image/analyze` | ✅ Pass |
| RAG Search | `/api/v1/rag/search` | ✅ Pass |
| RAG Q&A | `/api/v1/rag/query` | ✅ Pass |
| RNA Analysis | `/api/v1/rna/analyze` | ✅ Pass |
| RNA Validation | `/api/v1/rna/validate` | ✅ Pass |
| AutoML Experiments | `/api/v1/automl/experiments` | ✅ Pass |

### AutoML Optimization Results

| Experiment | Sampler | Trials | Best Score |
|------------|---------|--------|------------|
| Symptom Classifier | TPE + Hyperband | 10 | **93.33%** (F1-macro) |
| RNA Predictor | CMA-ES + Median | 5 | **89.55%** (Accuracy) |

## Running Tests

```bash
# Run all tests (74 tests)
PYTHONPATH=$PWD pytest -v

# Run specific test modules
pytest tests/test_symptom_classifier.py -v
pytest tests/test_rna_predictor.py -v
pytest tests/test_automl.py -v

# Run with coverage
pytest --cov=models --cov=rag --cov-report=html
```

## Project Structure

```
MediAssist-RAG_AutoML/
├── services/              # Microservices
│   ├── auth/             # Authentication service
│   ├── patient/          # Patient management
│   ├── analysis/         # ML analysis service (symptom, image, RNA, AutoML)
│   └── report/           # Report generation
├── models/               # ML models
│   ├── symptom_classifier/   # BioBERT NER + RandomForest
│   ├── image_analyzer/       # DenseNet121 + Grad-CAM
│   ├── risk_predictor/       # Risk assessment
│   ├── rna_predictor/        # RNA sequence disease prediction
│   └── automl/               # AutoML system (Optuna + ensemble)
├── rag/                  # RAG system
│   ├── embedding/        # BioBERT embeddings
│   ├── retrieval/        # Hybrid search (dense + BM25)
│   ├── reranking/        # Cross-encoder reranker
│   └── generation/       # LLM integration
├── frontend/             # React application
│   ├── src/pages/        # Analysis pages (Symptom, Image, RNA, AutoML)
│   └── src/stores/       # Zustand state management
├── common/               # Shared code (database, models, schemas)
├── tests/                # Unit tests (74 tests)
├── test_data/            # Sample test data
├── docs/                 # Documentation
│   └── screenshots/      # UI screenshots
└── k8s/                  # Kubernetes configs
```

## Service Ports

| Service | Port |
|---------|------|
| Frontend | 3003 (dev) / 3001 (docker) |
| Analysis Service | 8003 |
| Auth Service | 8001 |
| Patient Service | 8002 |
| Report Service | 8004 |
| PostgreSQL | 5433 |
| MongoDB | 27017 |
| Redis | 6380 |
| ChromaDB | 8005 |
| MinIO | 9000-9001 |

## API Documentation

After starting the services, API documentation is available at:
- Analysis Service: http://localhost:8003/docs

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | User login (MongoDB/demo) |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/symptom/analyze` | POST | Analyze symptoms and predict diseases |
| `/api/v1/symptom/extract` | POST | Extract symptoms from natural language |
| `/api/v1/image/analyze` | POST | Analyze chest X-ray images |
| `/api/v1/rna/analyze` | POST | Analyze RNA sequences for disease prediction |
| `/api/v1/rna/validate` | POST | Validate RNA sequence |
| `/api/v1/automl/experiments` | POST | Create AutoML experiment |
| `/api/v1/automl/experiments/{id}` | GET | Get experiment status |
| `/api/v1/automl/experiments/{id}/trials` | GET | List experiment trials |
| `/api/v1/automl/experiments/{id}/best-model` | GET | Get best model |
| `/api/v1/rag/search` | POST | Search medical literature |
| `/api/v1/rag/query` | POST | RAG-based question answering |

## Model Performance

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| Symptom Classifier | Accuracy | ≥ 85% | 87.5% |
| Symptom Classifier | F1-macro (AutoML) | - | 93.33% |
| RNA Predictor | Type Accuracy | ≥ 90% | 92% |
| RNA Predictor | Disease F1 | ≥ 0.75 | 0.84 |
| RNA Predictor | Accuracy (AutoML) | - | 89.55% |
| Image Analyzer | AUC-ROC | ≥ 0.90 | - |
| RAG System | Relevance | ≥ 0.70 | - |

## Design

UI는 **Frosted Metal Aesthetic** 디자인 시스템을 사용합니다:
- 다크 메탈릭 그라디언트 배경
- 엣지 하이라이트와 인셋 섀도우
- 쿨톤 악센트 컬러 (Cyan, Green, Purple, Orange)

## Disclaimer

이 시스템의 모든 분석 결과는 참고용이며, 최종 진단 및 치료 결정은 반드시 자격을 갖춘 의료 전문가가 수행해야 합니다.

## License

MIT License
