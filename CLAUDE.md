# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MediAssist AI is a medical diagnosis support system combining ML-based symptom analysis with RAG-based medical literature search. The system provides diagnostic suggestions with evidence-based references while explicitly maintaining that final diagnosis must be made by qualified medical professionals.

## Tech Stack

**Backend:** Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), Celery + Redis
**Frontend:** React 18, TypeScript, TailwindCSS, Zustand, TanStack Query
**ML/AI:** PyTorch, XGBoost, BioBERT (dmis-lab/biobert-v1.1), LangChain, ChromaDB
**Database:** PostgreSQL 15+, Redis 7.x, ChromaDB, MinIO
**Infrastructure:** Docker, Kubernetes, GitHub Actions

## Project Structure

```
mediassist-ai/
├── services/           # Microservices (auth, patient, analysis, report)
│   └── {service}/
│       ├── app/        # FastAPI application
│       │   ├── routers/
│       │   ├── schemas/
│       │   ├── crud/
│       │   └── services/
│       ├── tests/
│       └── Dockerfile
├── models/             # ML models
│   ├── symptom_classifier/   # BioBERT NER + XGBoost/RF ensemble
│   ├── image_analyzer/       # DenseNet121 for chest X-ray
│   └── risk_predictor/       # Neural network for risk scoring
├── rag/                # RAG system
│   ├── ingestion/      # Document collection (PubMed, guidelines)
│   ├── embedding/      # BioBERT embedding pipeline
│   ├── retrieval/      # Hybrid search (dense + BM25)
│   ├── reranking/      # Cross-encoder reranker
│   └── generation/     # LLM integration (GPT-4)
├── frontend/           # React application
├── common/             # Shared code (database, models, schemas)
├── k8s/                # Kubernetes manifests
└── docker-compose.yml
```

## Commands

```bash
# Development environment
docker-compose up -d              # Start all services
docker-compose logs -f            # View logs

# Backend (per service)
cd services/analysis
pip install -r requirements.txt
uvicorn app.main:app --reload     # Run service
pytest tests/                     # Run tests
pytest tests/test_file.py -k test_name  # Run single test

# Frontend
cd frontend
npm install
npm run dev                       # Development server
npm run build                     # Production build
npm run test                      # Run tests

# Code quality
ruff check .                      # Linting
black .                           # Formatting
mypy --strict .                   # Type checking
bandit -r .                       # Security scan
```

## Architecture

### Service Communication
- **API Gateway (Kong/Nginx):** Rate limiting (100 req/min per user), JWT validation, routing
- **Services communicate via REST API and Redis message queue**
- **Background tasks processed via Celery workers**

### ML Pipeline Flow
1. **Symptom Analysis:** Text → BioBERT NER → Feature extraction → XGBoost/RF ensemble → Disease predictions
2. **Image Analysis:** X-ray → DenseNet121 → Multi-label classification + Grad-CAM heatmap
3. **Risk Assessment:** Combined features → Neural network → Risk score

### RAG Pipeline Flow
1. **Ingestion:** PubMed/Guidelines → PDF/HTML parser → Text chunking (500 chars, 50 overlap)
2. **Retrieval:** Query → BioBERT embedding → ChromaDB (HNSW) + BM25 → Reciprocal Rank Fusion
3. **Reranking:** Cross-encoder (ms-marco-MiniLM-L-6-v2) → Top 5 documents
4. **Generation:** Context + Prompt → GPT-4 → Answer with citations

## Code Standards

- **Type hints required** (mypy strict mode)
- **PEP 8 + Black formatter**
- **Test coverage ≥ 80%**
- **All public APIs require docstrings**
- **Commit messages:** Conventional Commits format
- **Branching:** Git Flow

## ML Model Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| Symptom Classifier | Accuracy | ≥ 85% |
| Symptom Classifier | F1-macro | ≥ 0.80 |
| Image Analyzer | AUC-ROC | ≥ 0.90 |
| RAG System | Relevance | ≥ 0.70 |

## Medical Safety Requirements

All outputs must include disclaimer: "이 정보는 참고용이며 최종 진단은 의사가 결정합니다" (This information is for reference only; final diagnosis must be made by a physician).

- Never provide final diagnosis or prescriptions
- Always cite sources for medical information
- Red flag symptoms must trigger immediate warnings
- Explicitly state uncertainty when confidence is low

## Security

- **Authentication:** JWT (RS256), OAuth 2.0
- **Authorization:** RBAC (admin, doctor, nurse, researcher roles)
- **Encryption:** AES-256-GCM at rest, TLS 1.3 in transit
- **Patient data must be anonymized; PII fields encrypted**
- **All access logged with audit trail (7-year retention)**
